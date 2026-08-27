//! Write-ahead log (master plan Layer 2, *Write-ahead log*). JSONL, one
//! record per line — a crash can only ever corrupt the *last* line; every
//! earlier line is independently parseable. Vector payloads are
//! base64-encoded raw LE `f32` bytes inside a JSON string field (avoids
//! float-text formatting cost and bloat); payload fields stay native
//! inline JSON so an operator can `grep`/`jq` them directly.

use base64::Engine as _;
use chrono::{DateTime, Utc};
use crc32c::crc32c;
use mara_proto::{DocId, ExtraPayload, Lsn, PayloadRow, PrincipalId, RowId, SessionId, TxnId};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub const WAL_FORMAT_VERSION: u8 = 1;
const CHECKSUM_PLACEHOLDER: &str = "00000000";

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WalOp {
    Insert,
    Update,
    Delete,
    CreateCollection,
    DropCollection,
    AlterSchema,
    Reindex,
}

/// Shape shared by `payload` (the new state) and `undo` (the before-image
/// needed to reverse this record — `None`/all-`None` for `insert`, full
/// prior vector+payload for `update`/`delete`).
#[derive(Clone, PartialEq, Debug, Default, Serialize, Deserialize)]
pub struct WalPayload {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub vector_b64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub fields: Option<PayloadRow>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub extra: Option<ExtraPayload>,
}

impl WalPayload {
    pub fn vector(&self) -> Result<Option<Vec<f32>>, WalError> {
        self.vector_b64.as_deref().map(decode_vector).transpose()
    }
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct WalActor {
    pub id: PrincipalId,
    pub name: String,
}

/// One WAL line. `undo` is what makes a later `undo`/`revert_to` possible
/// without a special-cased inverse-chaining mechanism (see the master
/// plan's *Undo/revert* section) — reversing this record is always just
/// "apply `undo` as a new forward-appended record".
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WalRecord {
    pub v: u8,
    pub lsn: Lsn,
    pub ts: DateTime<Utc>,
    pub txn: TxnId,
    pub actor: WalActor,
    pub session: SessionId,
    pub coll: String,
    pub op: WalOp,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub row_id: Option<RowId>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub doc_id: Option<DocId>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub doc_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub chunk_ord: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub payload: Option<WalPayload>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub undo: Option<WalPayload>,
    /// Set on a compensating record produced by `undo`/`revert_to`,
    /// pointing at the transaction it reverses. `None` for an ordinary
    /// forward write.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub caused_by: Option<TxnId>,
    /// `(i, n)`: record `i` of `n` in this transaction. Lets replay
    /// distinguish a complete document transaction from one torn by a
    /// crash — a transaction whose last record never arrives is discarded
    /// whole, never half-applied.
    pub txn_seq: (u32, u32),
    /// Op-specific parameters for control records (`create_collection`,
    /// `alter_schema`, `reindex`, ...) that carry no row data.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub params: Option<serde_json::Value>,
    pub checksum: String,
}

/// A human-readable description of one transaction's WAL records — used
/// both when a mutation first commits and when the `TxnIndex` is
/// reconstructed from the WAL during recovery, so `history`/`undo` show the
/// same summary whether or not the process ever restarted in between.
pub fn summarize_txn(records: &[WalRecord]) -> String {
    let Some(first) = records.first() else {
        return "empty transaction".into();
    };
    let n = records.len();
    let coll = &first.coll;
    let doc_key = records.iter().find_map(|r| r.doc_key.clone());
    let all_insert = records.iter().all(|r| r.op == WalOp::Insert);
    let all_delete = records.iter().all(|r| r.op == WalOp::Delete);

    match (&doc_key, all_insert, all_delete) {
        (Some(dk), true, _) => format!("put_document {dk:?} ({n} chunks)"),
        (Some(dk), _, true) => format!("delete_document {dk:?} ({n} chunks)"),
        (Some(dk), false, false) => format!("replace_document {dk:?} ({n} records)"),
        (None, _, true) if n == 1 => format!("delete {:?} in {coll:?}", first.key.clone().unwrap_or_default()),
        (None, _, true) => format!("delete ({n} rows) in {coll:?}"),
        (None, _, _) if n == 1 => format!("put {:?} in {coll:?}", first.key.clone().unwrap_or_default()),
        (None, _, _) => format!("put_batch ({n} rows) in {coll:?}"),
    }
}

fn compute_checksum(record: &WalRecord) -> String {
    let mut for_hash = record.clone();
    for_hash.checksum = CHECKSUM_PLACEHOLDER.to_string();
    let bytes = serde_json::to_vec(&for_hash).expect("WalRecord always serializes");
    format!("{:08x}", crc32c(&bytes))
}

/// Whether `record.checksum` is what `compute_checksum` would produce for
/// its own fields — the same trust check `replay_all` applies to every
/// line read from a local WAL file, exposed for `Collection::
/// apply_replicated_lines`, which applies the same standard to a line
/// received from a replication leader: no more trusted than one this
/// process wrote itself.
pub(crate) fn verify_checksum(record: &WalRecord) -> bool {
    compute_checksum(record) == record.checksum
}

pub fn encode_vector(v: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

pub fn decode_vector(s: &str) -> Result<Vec<f32>, WalError> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(s)
        .map_err(|e| WalError::Corrupt(e.to_string()))?;
    if bytes.len() % 4 != 0 {
        return Err(WalError::Corrupt("vector_b64 byte length is not a multiple of 4".into()));
    }
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

#[derive(Debug, thiserror::Error)]
pub enum WalError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("corrupt WAL data: {0}")]
    Corrupt(String),
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum FsyncPolicy {
    Always,
    Interval(Duration),
    Never,
}

fn segment_path(dir: &Path, id: u32) -> PathBuf {
    dir.join(format!("segment-{id:010}.wal"))
}

fn segment_id_from_name(name: &str) -> Option<u32> {
    name.strip_prefix("segment-").and_then(|s| s.strip_suffix(".wal")).and_then(|s| s.parse().ok())
}

/// Segment ids present in `dir`, ascending.
pub fn list_segment_ids(dir: &Path) -> io::Result<Vec<u32>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut ids: Vec<u32> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| segment_id_from_name(&e.file_name().to_string_lossy()))
        .collect();
    ids.sort_unstable();
    Ok(ids)
}

struct WriterState {
    segment_id: u32,
    file: BufWriter<File>,
    offset: u32,
    last_fsync: Instant,
}

/// Appends WAL records, one segment file at a time, assigning each its LSN
/// (`segment_id`, current byte offset) as it's written. A single
/// `parking_lot::Mutex` around the writer state is this crate's stand-in
/// for the "single writer-actor task per collection" in the master plan —
/// see `Collection`'s doc comment for why that's an acceptable substitute
/// at this build stage.
pub struct WalWriter {
    dir: PathBuf,
    segment_size_bytes: u32,
    fsync: FsyncPolicy,
    state: parking_lot::Mutex<WriterState>,
}

impl WalWriter {
    /// Opens (creating if needed) the WAL directory, resuming append at the
    /// end of the highest-numbered existing segment. Callers that just ran
    /// crash recovery must truncate that segment to its last valid length
    /// *before* calling this — `WalWriter` trusts the file's current length
    /// completely; it does not itself detect or repair a torn tail.
    pub fn open(dir: impl Into<PathBuf>, segment_size_bytes: u32, fsync: FsyncPolicy) -> io::Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;
        let ids = list_segment_ids(&dir)?;
        let segment_id = ids.last().copied().unwrap_or(0);
        let path = segment_path(&dir, segment_id);
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let offset = file.metadata()?.len() as u32;
        Ok(WalWriter {
            dir,
            segment_size_bytes,
            fsync,
            state: parking_lot::Mutex::new(WriterState {
                segment_id,
                file: BufWriter::new(file),
                offset,
                last_fsync: Instant::now(),
            }),
        })
    }

    /// Appends every record in `records` as one physically-grouped write,
    /// assigning each an LSN in order, then applies `self.fsync`'s policy
    /// once for the whole batch — the amortization a `put_batch`/
    /// `put_document` relies on to cost one fsync, not one per row.
    /// Records are mutated in place: `lsn` and `checksum` are overwritten
    /// with the values actually assigned/computed here.
    pub fn append_batch(&self, records: &mut [WalRecord]) -> io::Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let mut guard = self.state.lock();
        for record in records.iter_mut() {
            if guard.offset >= self.segment_size_bytes && guard.offset > 0 {
                Self::rotate(&self.dir, &mut guard)?;
            }
            record.lsn = Lsn::new(guard.segment_id, guard.offset);
            record.checksum = compute_checksum(record);
            let line = serde_json::to_string(record).expect("WalRecord always serializes");
            guard.file.write_all(line.as_bytes())?;
            guard.file.write_all(b"\n")?;
            guard.offset += line.len() as u32 + 1;
        }
        match self.fsync {
            FsyncPolicy::Always => Self::flush_and_sync(&mut guard)?,
            FsyncPolicy::Interval(d) => {
                if guard.last_fsync.elapsed() >= d {
                    Self::flush_and_sync(&mut guard)?;
                }
            }
            FsyncPolicy::Never => guard.file.flush()?,
        }
        Ok(())
    }

    fn flush_and_sync(guard: &mut WriterState) -> io::Result<()> {
        guard.file.flush()?;
        guard.file.get_ref().sync_all()?;
        guard.last_fsync = Instant::now();
        Ok(())
    }

    fn rotate(dir: &Path, guard: &mut WriterState) -> io::Result<()> {
        Self::flush_and_sync(guard)?;
        guard.segment_id += 1;
        guard.offset = 0;
        let path = segment_path(dir, guard.segment_id);
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        guard.file = BufWriter::new(file);
        Ok(())
    }

    /// Appends records whose `lsn`/`checksum` are already fixed — assigned
    /// by a replication leader, never by this writer — the follower-side
    /// counterpart to `append_batch`, which mints fresh LSNs. Each record
    /// must land at exactly the position its own `lsn` specifies; a gap (a
    /// record this writer isn't currently positioned to accept — usually a
    /// sign of a missed or reordered batch upstream) is a hard error
    /// rather than silently skipped or overwritten, since either would
    /// corrupt the log's ordering guarantee. Always fsyncs before
    /// returning — a replicated write is, by construction, this
    /// follower's own durability boundary for that data, so it can't defer
    /// to whatever `FsyncPolicy` this writer was opened with.
    pub fn append_replicated_batch(&self, records: &[WalRecord]) -> io::Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let mut guard = self.state.lock();
        for record in records {
            let seg_id = record.lsn.segment_id();
            let want_offset = record.lsn.byte_offset();
            if seg_id != guard.segment_id {
                Self::switch_to_segment(&self.dir, &mut guard, seg_id)?;
            }
            if guard.offset != want_offset {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "replication gap: record lsn {} expects to land at offset {want_offset} in segment {seg_id}, but this WAL is currently at offset {}",
                        record.lsn, guard.offset
                    ),
                ));
            }
            let line = serde_json::to_string(record).expect("WalRecord always serializes");
            guard.file.write_all(line.as_bytes())?;
            guard.file.write_all(b"\n")?;
            guard.offset += line.len() as u32 + 1;
        }
        Self::flush_and_sync(&mut guard)
    }

    /// Repositions the writer onto `segment_id`'s file — opening (creating
    /// if needed) at append position and trusting the file's actual
    /// current length as the resume offset, the same way `WalWriter::open`
    /// trusts it for the writer's very first segment. Used only by
    /// `append_replicated_batch`, whose target segment is dictated by
    /// incoming LSNs rather than this writer's own size-threshold
    /// `rotate`.
    fn switch_to_segment(dir: &Path, guard: &mut WriterState, segment_id: u32) -> io::Result<()> {
        guard.file.flush()?;
        let path = segment_path(dir, segment_id);
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let offset = file.metadata()?.len() as u32;
        guard.segment_id = segment_id;
        guard.offset = offset;
        guard.file = BufWriter::new(file);
        Ok(())
    }

    pub fn current_lsn(&self) -> Lsn {
        let guard = self.state.lock();
        Lsn::new(guard.segment_id, guard.offset)
    }

    /// The directory this writer appends to — what `undo`/`revert_to` scan
    /// (via `replay_all`) to recover a past transaction's original records.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

pub struct WalReplayResult {
    pub records: Vec<WalRecord>,
    /// The id of the last segment touched during replay.
    pub last_segment_id: u32,
    /// The byte length that segment must be truncated to before any new
    /// `WalWriter` resumes appending — the offset of the last clean
    /// transaction boundary. Anything past this in the file is either a
    /// torn line or an incomplete transaction from an in-flight crash.
    pub last_segment_valid_len: u32,
}

/// Replays every segment in `dir`, validating each line's checksum and
/// buffering multi-record transactions until their final `txn_seq` record
/// arrives. Stops at the first corrupt line or incomplete trailing
/// transaction — by construction, since a crash can only ever corrupt the
/// *last* line, everything read before that point is trustworthy.
/// `after_lsn` filters which records are returned for re-application (a
/// snapshot's `last_applied_lsn`), but every line is still read and
/// checksum-verified regardless, since truncation safety depends on the
/// whole file, not just the unapplied tail.
pub fn replay_all(dir: &Path, after_lsn: Option<Lsn>) -> io::Result<WalReplayResult> {
    let segment_ids = list_segment_ids(dir)?;
    let mut records = Vec::new();
    let mut last_segment_id = 0u32;
    let mut last_segment_valid_len = 0u32;

    'segments: for &seg_id in &segment_ids {
        last_segment_id = seg_id;
        let path = segment_path(dir, seg_id);
        let file = File::open(&path)?;
        let mut reader = BufReader::new(file);
        let mut offset: u32 = 0;
        let mut valid_len: u32 = 0;
        let mut pending: Vec<WalRecord> = Vec::new();

        loop {
            let mut line = String::new();
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                break;
            }
            let trimmed = line.trim_end_matches('\n');
            let parsed: Result<WalRecord, _> = serde_json::from_str(trimmed);
            let record = match parsed {
                Ok(r) => r,
                Err(_) => {
                    last_segment_valid_len = valid_len;
                    break 'segments;
                }
            };
            let stored_checksum = record.checksum.clone();
            if compute_checksum(&record) != stored_checksum {
                last_segment_valid_len = valid_len;
                break 'segments;
            }

            offset += bytes_read as u32;
            let (i, n) = record.txn_seq;
            pending.push(record);
            if i >= n {
                for r in pending.drain(..) {
                    if after_lsn.is_none_or(|a| r.lsn > a) {
                        records.push(r);
                    }
                }
                valid_len = offset;
            }
        }
        // A non-empty `pending` here means the segment (and therefore the
        // whole WAL) ended mid-transaction — the exact torn-tail crash
        // scenario. Those buffered records are discarded; `valid_len`
        // never advanced past the transaction boundary before them.
        last_segment_valid_len = valid_len;
    }

    Ok(WalReplayResult {
        records,
        last_segment_id,
        last_segment_valid_len,
    })
}

/// Truncates the given segment (if it exists) to `valid_len` — the
/// recovery step that must run before a `WalWriter` resumes appending to a
/// WAL that had a torn tail.
pub fn truncate_segment(dir: &Path, segment_id: u32, valid_len: u32) -> io::Result<()> {
    let path = segment_path(dir, segment_id);
    if !path.exists() {
        return Ok(());
    }
    let file = OpenOptions::new().write(true).open(&path)?;
    file.set_len(valid_len as u64)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn actor() -> WalActor {
        WalActor {
            id: PrincipalId("p_test".into()),
            name: "tester".into(),
        }
    }

    fn base_record(txn: TxnId, seq: (u32, u32)) -> WalRecord {
        WalRecord {
            v: WAL_FORMAT_VERSION,
            lsn: Lsn::ZERO,
            ts: Utc::now(),
            txn,
            actor: actor(),
            session: SessionId("s".into()),
            coll: "docs".into(),
            op: WalOp::Insert,
            row_id: Some(RowId(1)),
            key: Some("a".into()),
            doc_id: None,
            doc_key: None,
            chunk_ord: None,
            payload: Some(WalPayload {
                vector_b64: Some(encode_vector(&[1.0, 2.0, 3.0])),
                text: None,
                fields: None,
                extra: None,
            }),
            undo: None,
            caused_by: None,
            txn_seq: seq,
            params: None,
            checksum: String::new(),
        }
    }

    #[test]
    fn vector_encoding_round_trips_exactly() {
        let v = vec![1.0f32, -2.5, 0.0, f32::MIN_POSITIVE, 12345.678];
        let encoded = encode_vector(&v);
        let decoded = decode_vector(&encoded).unwrap();
        assert_eq!(v, decoded);
    }

    #[test]
    fn append_then_replay_round_trips_a_single_record_txn() {
        let dir = tempfile::tempdir().unwrap();
        let writer = WalWriter::open(dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
        let txn = TxnId::new();
        let mut records = vec![base_record(txn, (1, 1))];
        writer.append_batch(&mut records).unwrap();

        let result = replay_all(dir.path(), None).unwrap();
        assert_eq!(result.records.len(), 1);
        assert_eq!(result.records[0].txn, txn);
        assert_eq!(result.records[0].lsn, records[0].lsn);
        assert_eq!(
            result.records[0].payload.as_ref().unwrap().vector().unwrap(),
            Some(vec![1.0, 2.0, 3.0])
        );
    }

    #[test]
    fn multi_record_txn_is_only_visible_once_complete() {
        let dir = tempfile::tempdir().unwrap();
        let writer = WalWriter::open(dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
        let txn = TxnId::new();
        let mut records: Vec<WalRecord> = (1..=5).map(|i| base_record(txn, (i, 5))).collect();
        writer.append_batch(&mut records).unwrap();

        let result = replay_all(dir.path(), None).unwrap();
        assert_eq!(result.records.len(), 5, "all 5 records of the complete txn must replay");
        assert!(result.records.iter().all(|r| r.txn == txn));
    }

    #[test]
    fn lsns_are_monotonically_increasing_within_a_segment() {
        let dir = tempfile::tempdir().unwrap();
        let writer = WalWriter::open(dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
        let mut records: Vec<WalRecord> = (0..10).map(|_| base_record(TxnId::new(), (1, 1))).collect();
        writer.append_batch(&mut records).unwrap();
        for w in records.windows(2) {
            assert!(w[0].lsn < w[1].lsn);
        }
    }

    #[test]
    fn segment_rotates_once_the_size_threshold_is_crossed() {
        let dir = tempfile::tempdir().unwrap();
        // A tiny segment size forces a rotation almost immediately.
        let writer = WalWriter::open(dir.path(), 200, FsyncPolicy::Always).unwrap();
        let mut records: Vec<WalRecord> = (0..20).map(|_| base_record(TxnId::new(), (1, 1))).collect();
        writer.append_batch(&mut records).unwrap();

        let ids = list_segment_ids(dir.path()).unwrap();
        assert!(ids.len() > 1, "expected more than one segment after crossing the size threshold");

        let result = replay_all(dir.path(), None).unwrap();
        assert_eq!(result.records.len(), 20, "every record must still replay across segment boundaries");
    }

    #[test]
    fn resuming_a_writer_after_reopen_continues_lsns_forward() {
        let dir = tempfile::tempdir().unwrap();
        let first_lsn = {
            let writer = WalWriter::open(dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
            let mut records = vec![base_record(TxnId::new(), (1, 1))];
            writer.append_batch(&mut records).unwrap();
            records[0].lsn
        };
        let writer2 = WalWriter::open(dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
        let mut records2 = vec![base_record(TxnId::new(), (1, 1))];
        writer2.append_batch(&mut records2).unwrap();
        assert!(records2[0].lsn > first_lsn, "a reopened writer must never reuse or rewind an LSN");
    }

    #[test]
    fn corrupt_last_line_is_detected_and_earlier_lines_still_replay() {
        let dir = tempfile::tempdir().unwrap();
        let writer = WalWriter::open(dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
        let mut records: Vec<WalRecord> = (0..3).map(|_| base_record(TxnId::new(), (1, 1))).collect();
        writer.append_batch(&mut records).unwrap();
        drop(writer);

        // Simulate a crash mid-line: append a truncated, non-JSON tail.
        let seg_path = segment_path(dir.path(), 0);
        {
            let mut f = OpenOptions::new().append(true).open(&seg_path).unwrap();
            f.write_all(b"{\"v\":1,\"lsn\":\"garbage-torn-line").unwrap();
        }

        let result = replay_all(dir.path(), None).unwrap();
        assert_eq!(result.records.len(), 3, "the 3 clean records before the torn line must still replay");
        assert_eq!(result.last_segment_id, 0);

        let full_len = fs::metadata(&seg_path).unwrap().len();
        assert!(
            (result.last_segment_valid_len as u64) < full_len,
            "valid_len must stop before the torn tail"
        );

        truncate_segment(dir.path(), result.last_segment_id, result.last_segment_valid_len).unwrap();
        let after_truncate = fs::metadata(&seg_path).unwrap().len();
        assert_eq!(after_truncate, result.last_segment_valid_len as u64);

        // A writer resuming after truncation must append cleanly, and a
        // fresh replay must show exactly 4 records (3 original + 1 new),
        // never a half-parsed torn one.
        let writer2 = WalWriter::open(dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
        let mut more = vec![base_record(TxnId::new(), (1, 1))];
        writer2.append_batch(&mut more).unwrap();
        let result2 = replay_all(dir.path(), None).unwrap();
        assert_eq!(result2.records.len(), 4);
    }

    #[test]
    fn incomplete_trailing_transaction_is_discarded_whole() {
        let dir = tempfile::tempdir().unwrap();
        let writer = WalWriter::open(dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();

        // A complete single-record txn, then a 5-record txn that only gets
        // 3 of its records written before the simulated crash.
        let mut complete = vec![base_record(TxnId::new(), (1, 1))];
        writer.append_batch(&mut complete).unwrap();

        let torn_txn = TxnId::new();
        let mut torn: Vec<WalRecord> = (1..=3).map(|i| base_record(torn_txn, (i, 5))).collect();
        writer.append_batch(&mut torn).unwrap();
        drop(writer);

        let result = replay_all(dir.path(), None).unwrap();
        assert_eq!(result.records.len(), 1, "only the complete txn may replay");
        assert_eq!(result.records[0].txn, complete[0].txn);
        assert!(result.records.iter().all(|r| r.txn != torn_txn), "no record of the torn txn may leak through");

        let seg_path = segment_path(dir.path(), 0);
        let full_len = fs::metadata(&seg_path).unwrap().len();
        assert!((result.last_segment_valid_len as u64) < full_len);

        truncate_segment(dir.path(), result.last_segment_id, result.last_segment_valid_len).unwrap();
        let writer2 = WalWriter::open(dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
        // The writer must resume exactly at the truncation point, not
        // after the discarded torn bytes.
        assert_eq!(writer2.current_lsn().byte_offset(), result.last_segment_valid_len);
    }

    #[test]
    fn after_lsn_filters_out_already_applied_records_but_still_validates_the_whole_file() {
        let dir = tempfile::tempdir().unwrap();
        let writer = WalWriter::open(dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
        let mut records: Vec<WalRecord> = (0..4).map(|_| base_record(TxnId::new(), (1, 1))).collect();
        writer.append_batch(&mut records).unwrap();

        let cutoff = records[1].lsn;
        let result = replay_all(dir.path(), Some(cutoff)).unwrap();
        assert_eq!(result.records.len(), 2, "only records strictly after the cutoff LSN should be returned");
        assert!(result.records.iter().all(|r| r.lsn > cutoff));
    }

    #[test]
    fn checksum_mismatch_from_a_tampered_line_is_treated_as_a_torn_tail() {
        let dir = tempfile::tempdir().unwrap();
        let writer = WalWriter::open(dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
        let mut records: Vec<WalRecord> = (0..2).map(|_| base_record(TxnId::new(), (1, 1))).collect();
        writer.append_batch(&mut records).unwrap();
        drop(writer);

        let seg_path = segment_path(dir.path(), 0);
        let contents = fs::read_to_string(&seg_path).unwrap();
        // Flip a byte inside the *second* line's key field, same length so
        // the tamper is otherwise invisible — targeting the second line
        // specifically (both records share the same key, so a global
        // find-and-replace would ambiguously hit either).
        let mut lines: Vec<String> = contents.lines().map(String::from).collect();
        assert_eq!(lines.len(), 2);
        lines[1] = lines[1].replacen("\"key\":\"a\"", "\"key\":\"b\"", 1);
        let tampered = lines.join("\n") + "\n";
        fs::write(&seg_path, tampered).unwrap();

        let result = replay_all(dir.path(), None).unwrap();
        assert_eq!(result.records.len(), 1, "the tampered record must fail checksum verification and be dropped");
    }

    #[test]
    fn replicated_batch_lands_at_the_leaders_lsn_and_replays_identically() {
        // Simulate a "leader": records with real, distinct LSNs assigned by
        // a normal writer.
        let leader_dir = tempfile::tempdir().unwrap();
        let leader = WalWriter::open(leader_dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
        let mut records: Vec<WalRecord> = (0..5).map(|_| base_record(TxnId::new(), (1, 1))).collect();
        leader.append_batch(&mut records).unwrap();

        // A brand-new follower WAL, positioned at LSN zero, accepts them
        // verbatim — no fresh LSNs minted, no gap.
        let follower_dir = tempfile::tempdir().unwrap();
        let follower = WalWriter::open(follower_dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
        follower.append_replicated_batch(&records).unwrap();

        let replayed = replay_all(follower_dir.path(), None).unwrap();
        assert_eq!(replayed.records.len(), 5);
        for (original, got) in records.iter().zip(&replayed.records) {
            assert_eq!(original.lsn, got.lsn, "a replicated record must keep the leader's exact LSN");
            assert_eq!(original.txn, got.txn);
        }
        assert_eq!(follower.current_lsn(), leader.current_lsn(), "follower and leader must end up at the same LSN after replicating the same records");
    }

    #[test]
    fn replicated_batch_spanning_a_segment_rotation_lands_in_the_right_files() {
        let leader_dir = tempfile::tempdir().unwrap();
        // A tiny segment size forces the leader to rotate mid-batch.
        let leader = WalWriter::open(leader_dir.path(), 200, FsyncPolicy::Always).unwrap();
        let mut records: Vec<WalRecord> = (0..20).map(|_| base_record(TxnId::new(), (1, 1))).collect();
        leader.append_batch(&mut records).unwrap();
        assert!(list_segment_ids(leader_dir.path()).unwrap().len() > 1, "test setup must actually cross a segment boundary");

        let follower_dir = tempfile::tempdir().unwrap();
        // The follower's own configured segment size is irrelevant — it
        // must land records in whichever segment the leader's LSN names,
        // never re-derive rotation from its own threshold.
        let follower = WalWriter::open(follower_dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
        follower.append_replicated_batch(&records).unwrap();

        assert_eq!(
            list_segment_ids(follower_dir.path()).unwrap(),
            list_segment_ids(leader_dir.path()).unwrap(),
            "the follower must reproduce the leader's exact segment layout"
        );
        let replayed = replay_all(follower_dir.path(), None).unwrap();
        assert_eq!(replayed.records.len(), 20);
    }

    #[test]
    fn a_replicated_batch_that_skips_ahead_of_the_writers_position_is_a_hard_error() {
        let leader_dir = tempfile::tempdir().unwrap();
        let leader = WalWriter::open(leader_dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
        let mut records: Vec<WalRecord> = (0..3).map(|_| base_record(TxnId::new(), (1, 1))).collect();
        leader.append_batch(&mut records).unwrap();

        let follower_dir = tempfile::tempdir().unwrap();
        let follower = WalWriter::open(follower_dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
        // Deliberately skip the first record — the follower is asked to
        // apply record[1] while still positioned at offset 0.
        let err = follower.append_replicated_batch(&records[1..]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn a_replicated_batch_resuming_after_reconnect_continues_from_the_right_offset() {
        let leader_dir = tempfile::tempdir().unwrap();
        let leader = WalWriter::open(leader_dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
        let mut first_half: Vec<WalRecord> = (0..3).map(|_| base_record(TxnId::new(), (1, 1))).collect();
        leader.append_batch(&mut first_half).unwrap();
        let mut second_half: Vec<WalRecord> = (0..3).map(|_| base_record(TxnId::new(), (1, 1))).collect();
        leader.append_batch(&mut second_half).unwrap();

        let follower_dir = tempfile::tempdir().unwrap();
        {
            // First "session": the follower only receives the first half
            // before its connection drops.
            let follower = WalWriter::open(follower_dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
            follower.append_replicated_batch(&first_half).unwrap();
        }
        // Reconnect: a *new* `WalWriter` (mirroring a fresh process/task),
        // reopened against the same directory, must resume exactly where
        // the last one left off and accept the rest cleanly.
        let follower2 = WalWriter::open(follower_dir.path(), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
        // `current_lsn()` is the *next-append* position, which is exactly
        // where the leader's own writer was when it minted `second_half`'s
        // first LSN (nothing else was written to the leader in between).
        assert_eq!(follower2.current_lsn(), second_half[0].lsn, "reopening must resume at the position right after the last replicated record, not restart");
        follower2.append_replicated_batch(&second_half).unwrap();

        let replayed = replay_all(follower_dir.path(), None).unwrap();
        assert_eq!(replayed.records.len(), 6);
    }
}
