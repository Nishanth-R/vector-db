//! On-disk snapshot / checkpoint (master plan Layer 2, *On-disk snapshot*).
//! Binary (`bincode`), atomic-rename write, checksummed (`crc32c`), 3
//! generations retained. Contains rows, doc registry, and the `TxnIndex` —
//! `Collection::open`'s "startup reconciliation" is: load the newest
//! checksummed-valid snapshot, then replay only the WAL tail after its
//! `last_applied_lsn`, instead of replaying the whole WAL from empty.
//!
//! Text/JSON was rejected for this file specifically, per the plan: nobody
//! greps a snapshot, and float-as-JSON-text is a real size/speed
//! regression for thousands of raw vectors — unlike the WAL, where
//! grep-ability was worth paying for.
//!
//! **Scope note**: this covers the snapshot file itself and startup
//! reconciliation. Two adjacent pieces named in the same build step are
//! deliberately *not* implemented here: WAL segment deletion by
//! `wal.retention_days` (undo/history can target any transaction still in
//! `TxnIndex`, bounded only by `txn_history_limit` — deleting WAL segments
//! by a separate, uncoordinated age policy risks breaking `undo` for a
//! transaction the index still thinks is targetable, so it's safer to
//! leave WAL segments accumulating than to delete one live undo depends
//! on), and blob-file compaction (this architecture has no separate
//! `payload-blob.dat`/`text-blob.dat` yet — `extra` and chunk `text` are
//! still embedded directly in each row, snapshotted along with it, so
//! there is no blob file to compact).

use crate::collection::{CollectionInner, RowRecord};
use crate::document::DocEntry;
use mara_proto::{DocId, ExtraPayload, Lsn, PayloadRow, RowId, TxnEntry};
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotRow {
    row_id: RowId,
    key: String,
    vector: Vec<f32>,
    fields: PayloadRow,
    extra: Option<ExtraPayload>,
    text: Option<String>,
    doc_id: Option<DocId>,
    chunk_ord: Option<u32>,
    /// Whether this row was live (vs. tombstoned-but-retained) at
    /// checkpoint time — `guard.live` isn't derivable from anything else
    /// in the snapshot, so it's carried explicitly per row.
    live: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct CollectionSnapshot {
    last_applied_lsn: Lsn,
    next_row_id: u64,
    next_doc_id: u64,
    rows: Vec<SnapshotRow>,
    docs: Vec<DocEntry>,
    /// In chronological order — replaying `TxnIndex::record` in this order
    /// reconstructs `last_txn` and respects the retention limit exactly as
    /// live operation would have.
    txn_entries: Vec<TxnEntry>,
}

impl CollectionSnapshot {
    pub(crate) fn capture(inner: &CollectionInner) -> Self {
        let rows = inner
            .rows
            .iter()
            .map(|(&row_id, rec)| SnapshotRow {
                row_id,
                key: rec.key.to_string(),
                vector: rec.vector.to_vec(),
                fields: (*rec.fields).clone(),
                extra: rec.extra.as_ref().map(|e| (**e).clone()),
                text: rec.text.as_ref().map(|t| t.to_string()),
                doc_id: rec.doc_id,
                chunk_ord: rec.chunk_ord,
                live: inner.live.contains(row_id.to_bitmap_index()),
            })
            .collect();
        let docs = inner.docs.values().cloned().collect();
        let txn_entries = inner.txn_index.iter_chronological().cloned().collect();
        CollectionSnapshot {
            last_applied_lsn: inner.last_applied_lsn,
            next_row_id: inner.next_row_id,
            next_doc_id: inner.next_doc_id,
            rows,
            docs,
            txn_entries,
        }
    }

    pub(crate) fn last_applied_lsn(&self) -> Lsn {
        self.last_applied_lsn
    }

    /// Restores `self` into `inner`, which must be freshly constructed
    /// (empty rows/docs/txn_index) — this does not merge, it seeds. The
    /// columnar payload store is rebuilt from each row's `fields` via
    /// `set_row` (derived state, not snapshotted directly), matching the
    /// same "storage is the source of truth, indexes are rebuildable"
    /// principle used for the vector/BM25 indexes.
    pub(crate) fn restore_into(self, inner: &mut CollectionInner) {
        inner.last_applied_lsn = self.last_applied_lsn;
        inner.next_row_id = self.next_row_id;
        inner.next_doc_id = self.next_doc_id;

        let mut rows = HashMap::with_capacity(self.rows.len());
        let mut key_to_row = HashMap::with_capacity(self.rows.len());
        let mut live = RoaringBitmap::new();
        for r in self.rows {
            let key_arc: Arc<str> = Arc::from(r.key.as_str());
            if r.live {
                live.insert(r.row_id.to_bitmap_index());
                key_to_row.insert(key_arc.clone(), r.row_id);
                // A tombstoned row is retained (no compaction) but was
                // already `clear_row`'d out of the columnar payload index
                // at delete time — only a live row's fields belong in it.
                inner
                    .payload
                    .set_row(r.row_id, &r.fields)
                    .expect("a snapshot's own row fields must satisfy the schema that produced them");
            }
            rows.insert(
                r.row_id,
                RowRecord {
                    key: key_arc,
                    vector: r.vector.into(),
                    fields: Arc::new(r.fields),
                    extra: r.extra.map(Arc::new),
                    text: r.text.map(|t| Arc::from(t.as_str())),
                    doc_id: r.doc_id,
                    chunk_ord: r.chunk_ord,
                },
            );
        }
        inner.rows = rows;
        inner.key_to_row = key_to_row;
        inner.live = live;

        let mut doc_key_to_id = HashMap::with_capacity(self.docs.len());
        let mut docs = HashMap::with_capacity(self.docs.len());
        for entry in self.docs {
            doc_key_to_id.insert(Arc::from(entry.doc_key.as_str()), entry.doc_id);
            docs.insert(entry.doc_id, entry);
        }
        inner.docs = docs;
        inner.doc_key_to_id = doc_key_to_id;

        for entry in self.txn_entries {
            inner.txn_index.record(entry);
        }
    }
}

fn checkpoint_path(dir: &Path, generation: u64) -> PathBuf {
    dir.join(format!("snapshot-{generation:010}.mdb"))
}

fn list_generations(dir: &Path) -> io::Result<Vec<u64>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut gens: Vec<u64> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy().into_owned();
            name.strip_prefix("snapshot-").and_then(|s| s.strip_suffix(".mdb")).and_then(|s| s.parse().ok())
        })
        .collect();
    gens.sort_unstable();
    Ok(gens)
}

fn bincode_config() -> bincode::config::Configuration {
    bincode::config::standard()
}

/// Writes a new checkpoint generation: encode → temp file → fsync (durable
/// contents) → atomic rename → fsync the containing directory (durable
/// rename) — the "atomic-rename + double-fsync write" the plan calls for.
/// A `[u32 crc32c][bincode payload]` framing lets `read_snapshot` detect a
/// torn/corrupt file (e.g. from a crash mid-write of a *previous*
/// generation, before its rename ever landed — the temp file, not this
/// one) and fall back to an older generation instead of trusting garbage.
pub(crate) fn write_checkpoint(dir: &Path, snapshot: &CollectionSnapshot) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let payload = bincode::serde::encode_to_vec(snapshot, bincode_config()).expect("CollectionSnapshot always serializes");
    let crc = crc32c::crc32c(&payload);

    let generation = list_generations(dir)?.last().map(|g| g + 1).unwrap_or(0);
    let tmp_path = dir.join(format!("snapshot-{generation:010}.mdb.tmp"));
    {
        let mut f = File::create(&tmp_path)?;
        f.write_all(&crc.to_le_bytes())?;
        f.write_all(&payload)?;
        f.flush()?;
        f.sync_all()?;
    }
    fs::rename(&tmp_path, checkpoint_path(dir, generation))?;
    File::open(dir)?.sync_all()?;

    retain_generations(dir, 3)
}

fn retain_generations(dir: &Path, keep: usize) -> io::Result<()> {
    let gens = list_generations(dir)?;
    if gens.len() <= keep {
        return Ok(());
    }
    for generation in &gens[..gens.len() - keep] {
        let _ = fs::remove_file(checkpoint_path(dir, *generation));
    }
    Ok(())
}

fn read_checkpoint(path: &Path) -> io::Result<Option<CollectionSnapshot>> {
    let mut f = OpenOptions::new().read(true).open(path)?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    if bytes.len() < 4 {
        return Ok(None);
    }
    let stored_crc = u32::from_le_bytes(bytes[0..4].try_into().expect("checked len >= 4"));
    let payload = &bytes[4..];
    if crc32c::crc32c(payload) != stored_crc {
        return Ok(None);
    }
    match bincode::serde::decode_from_slice::<CollectionSnapshot, _>(payload, bincode_config()) {
        Ok((snapshot, _)) => Ok(Some(snapshot)),
        Err(_) => Ok(None),
    }
}

/// The newest checksummed-valid snapshot in `dir`, falling back to older
/// generations if the newest is corrupt (e.g. truncated by a crash between
/// the rename and its directory fsync becoming visible after all — belt
/// and suspenders on top of the write path's own atomicity).
pub(crate) fn load_latest(dir: &Path) -> io::Result<Option<CollectionSnapshot>> {
    let mut gens = list_generations(dir)?;
    gens.reverse();
    for generation in gens {
        if let Some(snap) = read_checkpoint(&checkpoint_path(dir, generation))? {
            return Ok(Some(snap));
        }
    }
    Ok(None)
}
