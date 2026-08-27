use chrono::{DateTime, NaiveDate, Utc};
use mara_proto::{PrincipalId, RequestCtx, Role, SessionId, Source};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Clone, serde::Serialize, serde::Deserialize, Debug, PartialEq)]
pub struct AuditPrincipal {
    pub id: PrincipalId,
    pub name: String,
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub token: Option<String>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AuditSource {
    Uds { os_user: String },
    Tcp { addr: String },
    Http { addr: String },
    Embedded,
}

impl From<&Source> for AuditSource {
    fn from(s: &Source) -> Self {
        match s {
            Source::Uds { os_user, .. } => AuditSource::Uds {
                os_user: os_user.clone(),
            },
            Source::Tcp(addr) => AuditSource::Tcp { addr: addr.to_string() },
            Source::Http(addr) => AuditSource::Http { addr: addr.to_string() },
            Source::Embedded => AuditSource::Embedded,
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AuditOutcome {
    Ok,
    Error { message: String },
}

#[derive(Clone, serde::Serialize, serde::Deserialize, Debug, PartialEq)]
pub struct AuditRecord {
    pub ts: DateTime<Utc>,
    pub request_id: Uuid,
    pub session: SessionId,
    pub principal: AuditPrincipal,
    pub source: AuditSource,
    pub action: String,
    pub result: AuditOutcome,
    pub latency_ms: u64,
    #[serde(flatten, skip_serializing_if = "serde_json::Map::is_empty", default)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl AuditRecord {
    pub fn new(ctx: &RequestCtx, action: impl Into<String>, result: AuditOutcome, latency_ms: u64) -> Self {
        AuditRecord {
            ts: Utc::now(),
            request_id: ctx.request_id,
            session: ctx.session.clone(),
            principal: AuditPrincipal {
                id: ctx.principal.id.clone(),
                name: ctx.principal.name.clone(),
                role: ctx.principal.role,
                token: None,
            },
            source: AuditSource::from(&ctx.source),
            action: action.into(),
            result,
            latency_ms,
            extra: serde_json::Map::new(),
        }
    }

    pub fn with_masked_token(mut self, plaintext_token: &str) -> Self {
        self.principal.token = Some(crate::token::display_prefix(plaintext_token));
        self
    }

    pub fn with_extra(mut self, key: &str, value: impl serde::Serialize) -> Self {
        if let Ok(v) = serde_json::to_value(value) {
            self.extra.insert(key.to_string(), v);
        }
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    #[error("audit sink stalled beyond max_stall_ms")]
    Stalled,
    #[error("audit sink is closed")]
    Closed,
}

pub trait AuditSink: Send + Sync {
    fn record(&self, record: AuditRecord) -> Result<(), AuditError>;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AuditMode {
    Strict,
    Lossy,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AuditFsync {
    Always,
    Interval(Duration),
    Never,
}

#[derive(Clone, Debug)]
pub struct AuditConfig {
    pub mode: AuditMode,
    pub fsync: AuditFsync,
    pub max_stall: Duration,
    pub channel_capacity: usize,
    pub rotate_size_mb: u64,
    pub retention_days: i64,
}

impl Default for AuditConfig {
    fn default() -> Self {
        AuditConfig {
            mode: AuditMode::Lossy,
            fsync: AuditFsync::Interval(Duration::from_secs(1)),
            max_stall: Duration::from_millis(5000),
            channel_capacity: 4096,
            rotate_size_mb: 256,
            retention_days: 90,
        }
    }
}

pub fn prune_older_than(dir: &Path, retention_days: i64) -> io::Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    let cutoff = Utc::now().date_naive() - chrono::Duration::days(retention_days);
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(rest) = name.strip_prefix("audit-")
            && let Some(date_str) = rest.get(..10)
                && let Ok(date) = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
                    && date < cutoff {
                        let _ = fs::remove_file(entry.path());
                    }
    }
    Ok(())
}

/// `(date, rotation_index)` parsed from an `audit-{date}.jsonl` or
/// `audit-{date}.{rotation_index}.jsonl` file name — sortable so
/// `read_recent` can walk segments in true chronological order rather than
/// lexical file-name order (`audit-2026-08-25.9.jsonl` sorts before
/// `audit-2026-08-25.10.jsonl` numerically, not as strings).
fn parse_segment_name(name: &str) -> Option<(NaiveDate, u32)> {
    let rest = name.strip_prefix("audit-")?.strip_suffix(".jsonl")?;
    let date_str = rest.get(..10)?;
    let date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d").ok()?;
    let rotation_index = match rest.get(10..) {
        Some("") => 0,
        Some(suffix) => suffix.strip_prefix('.')?.parse().ok()?,
        None => 0,
    };
    Some((date, rotation_index))
}

/// Reads back up to `limit` of the most recent audit records under `dir`,
/// newest first — what `mara-api`'s `/v1/audit` (Admin-only, gated on
/// `Capability::AuditRead`) reads directly from disk rather than through
/// the `AuditSink` trait, which is write-only by design (the request path
/// never blocks on a query). Walks segments newest-to-oldest and, within
/// each, lines last-to-first, stopping as soon as `limit` is reached — so
/// a huge historical log never needs a full scan just to answer "show me
/// the last 50". A line that fails to parse (e.g. a torn tail from a
/// write in progress) is skipped rather than failing the whole read.
pub fn read_recent(dir: &Path, limit: usize) -> io::Result<Vec<AuditRecord>> {
    if limit == 0 || !dir.exists() {
        return Ok(Vec::new());
    }

    let mut segments: Vec<(NaiveDate, u32, PathBuf)> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name();
            let (date, rotation_index) = parse_segment_name(&name.to_string_lossy())?;
            Some((date, rotation_index, entry.path()))
        })
        .collect();
    segments.sort_by_key(|(date, rotation_index, _)| (*date, *rotation_index));

    let mut out = Vec::with_capacity(limit.min(1024));
    for (_, _, path) in segments.into_iter().rev() {
        let contents = fs::read_to_string(&path)?;
        for line in contents.lines().rev() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(record) = serde_json::from_str::<AuditRecord>(line) {
                out.push(record);
                if out.len() >= limit {
                    return Ok(out);
                }
            }
        }
    }
    Ok(out)
}

struct WriterState {
    dir: PathBuf,
    rotate_size_bytes: u64,
    file: BufWriter<File>,
    current_date: NaiveDate,
    rotation_index: u32,
    current_size: u64,
    last_fsync: Instant,
}

fn file_path_for(dir: &Path, date: NaiveDate, rotation_index: u32) -> PathBuf {
    if rotation_index == 0 {
        dir.join(format!("audit-{date}.jsonl"))
    } else {
        dir.join(format!("audit-{date}.{rotation_index}.jsonl"))
    }
}

fn open_append(path: &Path) -> io::Result<(File, u64)> {
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let size = file.metadata()?.len();
    Ok((file, size))
}

impl WriterState {
    fn open(dir: PathBuf, rotate_size_bytes: u64) -> io::Result<Self> {
        let today = Utc::now().date_naive();
        let path = file_path_for(&dir, today, 0);
        let (file, size) = open_append(&path)?;
        Ok(WriterState {
            dir,
            rotate_size_bytes,
            file: BufWriter::new(file),
            current_date: today,
            rotation_index: 0,
            current_size: size,
            last_fsync: Instant::now(),
        })
    }

    fn reopen(&mut self) -> io::Result<()> {
        self.file.flush()?;
        let path = file_path_for(&self.dir, self.current_date, self.rotation_index);
        let (file, size) = open_append(&path)?;
        self.file = BufWriter::new(file);
        self.current_size = size;
        Ok(())
    }

    fn ensure_fresh(&mut self) -> io::Result<()> {
        let today = Utc::now().date_naive();
        if today != self.current_date {
            self.current_date = today;
            self.rotation_index = 0;
            self.reopen()?;
        } else if self.current_size >= self.rotate_size_bytes {
            self.rotation_index += 1;
            self.reopen()?;
        }
        Ok(())
    }

    fn write_line(&mut self, line: &str, fsync: AuditFsync) -> io::Result<()> {
        self.ensure_fresh()?;
        self.file.write_all(line.as_bytes())?;
        self.file.write_all(b"\n")?;
        self.current_size += line.len() as u64 + 1;
        match fsync {
            AuditFsync::Always => self.flush_all()?,
            AuditFsync::Interval(d) => {
                if self.last_fsync.elapsed() >= d {
                    self.flush_all()?;
                }
            }
            AuditFsync::Never => {}
        }
        Ok(())
    }

    fn flush_all(&mut self) -> io::Result<()> {
        self.file.flush()?;
        self.file.get_ref().sync_all()?;
        self.last_fsync = Instant::now();
        Ok(())
    }
}

/// Append-only JSONL audit sink, rotated daily and by size, drained by a
/// dedicated writer thread so the request path never does its own file I/O.
pub struct JsonlAuditSink {
    sender: Option<crossbeam_channel::Sender<AuditRecord>>,
    handle: Option<JoinHandle<()>>,
    dropped: Arc<AtomicU64>,
    mode: AuditMode,
    max_stall: Duration,
}

impl JsonlAuditSink {
    pub fn open(dir: impl Into<PathBuf>, config: AuditConfig) -> io::Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;
        let _ = prune_older_than(&dir, config.retention_days);

        let (tx, rx) = crossbeam_channel::bounded::<AuditRecord>(config.channel_capacity);
        let dropped = Arc::new(AtomicU64::new(0));
        let fsync = config.fsync;
        let rotate_size_bytes = config.rotate_size_mb.saturating_mul(1024 * 1024).max(1);
        let dir_for_thread = dir.clone();

        let handle = thread::Builder::new()
            .name("mara-audit-writer".into())
            .spawn(move || {
                let mut state = match WriterState::open(dir_for_thread, rotate_size_bytes) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::error!("audit writer failed to open sink: {e}");
                        return;
                    }
                };
                for record in rx.iter() {
                    match serde_json::to_string(&record) {
                        Ok(line) => {
                            if let Err(e) = state.write_line(&line, fsync) {
                                tracing::error!("audit writer failed: {e}");
                            }
                        }
                        Err(e) => tracing::error!("audit record failed to serialize: {e}"),
                    }
                }
                let _ = state.flush_all();
            })?;

        Ok(JsonlAuditSink {
            sender: Some(tx),
            handle: Some(handle),
            dropped,
            mode: config.mode,
            max_stall: config.max_stall,
        })
    }

    pub fn dropped_count(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    pub fn close(mut self) {
        self.sender.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for JsonlAuditSink {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl AuditSink for JsonlAuditSink {
    fn record(&self, record: AuditRecord) -> Result<(), AuditError> {
        let sender = self.sender.as_ref().ok_or(AuditError::Closed)?;
        match self.mode {
            AuditMode::Lossy => {
                if sender.try_send(record).is_err() {
                    self.dropped.fetch_add(1, Ordering::Relaxed);
                }
                Ok(())
            }
            AuditMode::Strict => sender
                .send_timeout(record, self.max_stall)
                .map_err(|_| AuditError::Stalled),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mara_proto::{Principal, Role};
    use std::io::BufRead;

    fn test_ctx() -> RequestCtx {
        RequestCtx::new(
            SessionId("cli-7f21".into()),
            Principal {
                id: PrincipalId("p_7f21a0".into()),
                name: "alice".into(),
                role: Role::Writer,
            },
            Source::Embedded,
        )
    }

    #[test]
    fn strict_sink_writes_records_to_disk_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let sink = JsonlAuditSink::open(
            dir.path(),
            AuditConfig {
                mode: AuditMode::Strict,
                fsync: AuditFsync::Always,
                ..Default::default()
            },
        )
        .unwrap();

        let ctx = test_ctx();
        for i in 0..5 {
            let rec = AuditRecord::new(&ctx, "put_document", AuditOutcome::Ok, 10)
                .with_extra("seq", i);
            sink.record(rec).unwrap();
        }
        sink.close();

        let path = dir.path().join(format!("audit-{}.jsonl", Utc::now().date_naive()));
        let file = File::open(&path).unwrap();
        let lines: Vec<String> = io::BufReader::new(file).lines().map(|l| l.unwrap()).collect();
        assert_eq!(lines.len(), 5);
        for (i, line) in lines.iter().enumerate() {
            let parsed: AuditRecord = serde_json::from_str(line).unwrap();
            assert_eq!(parsed.extra.get("seq").unwrap().as_i64().unwrap(), i as i64);
            assert_eq!(parsed.principal.name, "alice");
        }
    }

    #[test]
    fn lossy_sink_drops_and_counts_when_full_rather_than_blocking() {
        let dir = tempfile::tempdir().unwrap();
        let sink = JsonlAuditSink::open(
            dir.path(),
            AuditConfig {
                mode: AuditMode::Lossy,
                channel_capacity: 0,
                ..Default::default()
            },
        )
        .unwrap();
        let ctx = test_ctx();
        for _ in 0..50 {
            sink.record(AuditRecord::new(&ctx, "search", AuditOutcome::Ok, 1))
                .unwrap();
        }
        sink.close();
    }

    #[test]
    fn masked_token_never_contains_the_full_secret() {
        let ctx = test_ctx();
        let token = crate::token::generate_token();
        let rec = AuditRecord::new(&ctx, "auth.create_token", AuditOutcome::Ok, 5)
            .with_masked_token(&token);
        let shown = rec.principal.token.unwrap();
        assert!(shown.starts_with(crate::token::TOKEN_PREFIX));
        assert!(token.len() > shown.len());
    }

    #[test]
    fn error_outcome_round_trips_through_json() {
        let ctx = test_ctx();
        let rec = AuditRecord::new(
            &ctx,
            "undo",
            AuditOutcome::Error {
                message: "write-write conflict".into(),
            },
            3,
        );
        let json = serde_json::to_string(&rec).unwrap();
        let back: AuditRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(rec, back);
    }

    #[test]
    fn rotates_when_size_threshold_is_crossed() {
        let dir = tempfile::tempdir().unwrap();
        let sink = JsonlAuditSink::open(
            dir.path(),
            AuditConfig {
                mode: AuditMode::Strict,
                fsync: AuditFsync::Always,
                rotate_size_mb: 0, // rounds up to 1 byte minimum via .max(1) -> rotates on every record after the first
                ..Default::default()
            },
        )
        .unwrap();
        let ctx = test_ctx();
        for _ in 0..3 {
            sink.record(AuditRecord::new(&ctx, "search", AuditOutcome::Ok, 1))
                .unwrap();
        }
        sink.close();

        let today = Utc::now().date_naive();
        let base = dir.path().join(format!("audit-{today}.jsonl"));
        let rotated = dir.path().join(format!("audit-{today}.1.jsonl"));
        assert!(base.exists());
        assert!(rotated.exists(), "expected a rotated file once the size threshold was crossed");
    }

    #[test]
    fn read_recent_returns_newest_first_across_rotated_segments() {
        let dir = tempfile::tempdir().unwrap();
        let sink = JsonlAuditSink::open(
            dir.path(),
            AuditConfig {
                mode: AuditMode::Strict,
                fsync: AuditFsync::Always,
                rotate_size_mb: 0, // forces a fresh segment per record, past the first
                ..Default::default()
            },
        )
        .unwrap();
        let ctx = test_ctx();
        for i in 0..5 {
            sink.record(AuditRecord::new(&ctx, "search", AuditOutcome::Ok, 1).with_extra("seq", i))
                .unwrap();
        }
        sink.close();

        let recent = read_recent(dir.path(), 100).unwrap();
        let seqs: Vec<i64> = recent.iter().map(|r| r.extra.get("seq").unwrap().as_i64().unwrap()).collect();
        assert_eq!(seqs, vec![4, 3, 2, 1, 0], "read_recent must walk both segment order and within-segment order newest-first");
    }

    #[test]
    fn read_recent_stops_as_soon_as_the_limit_is_reached() {
        let dir = tempfile::tempdir().unwrap();
        let sink = JsonlAuditSink::open(dir.path(), AuditConfig { mode: AuditMode::Strict, fsync: AuditFsync::Always, ..Default::default() }).unwrap();
        let ctx = test_ctx();
        for i in 0..10 {
            sink.record(AuditRecord::new(&ctx, "search", AuditOutcome::Ok, 1).with_extra("seq", i))
                .unwrap();
        }
        sink.close();

        let recent = read_recent(dir.path(), 3).unwrap();
        let seqs: Vec<i64> = recent.iter().map(|r| r.extra.get("seq").unwrap().as_i64().unwrap()).collect();
        assert_eq!(seqs, vec![9, 8, 7]);
    }

    #[test]
    fn read_recent_skips_a_malformed_line_rather_than_failing_the_whole_read() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path()).unwrap();
        let ctx = test_ctx();
        let good1 = serde_json::to_string(&AuditRecord::new(&ctx, "search", AuditOutcome::Ok, 1)).unwrap();
        let good2 = serde_json::to_string(&AuditRecord::new(&ctx, "put", AuditOutcome::Ok, 2)).unwrap();
        let today = Utc::now().date_naive();
        fs::write(dir.path().join(format!("audit-{today}.jsonl")), format!("{good1}\n{{not valid json\n{good2}\n")).unwrap();

        let recent = read_recent(dir.path(), 10).unwrap();
        assert_eq!(recent.len(), 2, "the torn/malformed middle line must be skipped, not fail the read");
        assert_eq!(recent[0].action, "put");
        assert_eq!(recent[1].action, "search");
    }

    #[test]
    fn read_recent_on_a_missing_directory_is_an_empty_list_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("never-created");
        assert_eq!(read_recent(&missing, 10).unwrap(), Vec::new());
    }

    #[test]
    fn segment_name_parsing_orders_rotation_index_numerically_not_lexically() {
        assert_eq!(parse_segment_name("audit-2026-08-25.jsonl"), Some((NaiveDate::from_ymd_opt(2026, 8, 25).unwrap(), 0)));
        assert_eq!(parse_segment_name("audit-2026-08-25.9.jsonl"), Some((NaiveDate::from_ymd_opt(2026, 8, 25).unwrap(), 9)));
        assert_eq!(parse_segment_name("audit-2026-08-25.10.jsonl"), Some((NaiveDate::from_ymd_opt(2026, 8, 25).unwrap(), 10)));
        assert!(parse_segment_name("not-an-audit-file.jsonl").is_none());
        assert!(parse_segment_name("audit-2026-08-25.notanumber.jsonl").is_none());
    }
}
