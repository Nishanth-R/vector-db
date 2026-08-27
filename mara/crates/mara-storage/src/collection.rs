use crate::change::{ChangeBatch, ChangeSubscriber};
use crate::error::{StorageError, StorageResult};
use crate::payload::{FilterMask, PayloadSchema, PayloadStore};
use crate::wal::{self, FsyncPolicy, WalActor, WalOp, WalPayload, WalRecord, WalWriter, WAL_FORMAT_VERSION};
use chrono::Utc;
use mara_proto::{DistanceMetric, DocId, ExtraPayload, Filter, Lsn, PayloadRow, RequestCtx, Row, RowId, TxnId};
use parking_lot::RwLock;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// One row to write, at the storage layer's level of abstraction —
/// embedding and chunking have already happened upstream (`mara-embed`,
/// `mara-chunker` run per-request, ahead of the writer, per the master
/// plan) so this is always a concrete vector, never raw text.
pub struct PutInput {
    pub key: String,
    pub vector: Vec<f32>,
    pub fields: PayloadRow,
    pub extra: Option<ExtraPayload>,
}

/// One row's stored state, keyed by `RowId`. Deliberately not yet the typed
/// columnar payload store from the master plan's Layer 2 — that lands with
/// the payload schema + filter compiler; this naive per-row representation
/// is what the columnar store gets correctness-tested against.
pub(crate) struct RowRecord {
    pub(crate) key: Arc<str>,
    pub(crate) vector: Arc<[f32]>,
    pub(crate) fields: Arc<PayloadRow>,
    pub(crate) extra: Option<Arc<ExtraPayload>>,
    /// A chunk's source text (or, for a plain row, whatever text the caller
    /// associated with it for BM25/display purposes). Deliberately its own
    /// slot rather than a payload field: it's present on every document
    /// chunk unconditionally, so folding it into the schema-validated
    /// `fields` map would force every document-bearing collection to
    /// declare a `text` field just to satisfy strict-schema validation —
    /// and it's exactly why the on-disk design later gives it a separate
    /// `text-blob.dat` rather than sharing `payload-blob.dat`.
    pub(crate) text: Option<Arc<str>>,
    pub(crate) doc_id: Option<DocId>,
    pub(crate) chunk_ord: Option<u32>,
}

pub(crate) struct CollectionInner {
    /// The highest LSN applied to this state so far — updated by
    /// `recovery::apply_txn_group` on every apply, live or replayed, which
    /// is what makes it safe for `checkpoint()` to read directly rather
    /// than deriving it from the WAL writer's next-append-position cursor
    /// (which is off by one transaction: it points *past* the last
    /// applied record, not *at* it).
    pub(crate) last_applied_lsn: Lsn,
    pub(crate) next_row_id: u64,
    pub(crate) key_to_row: HashMap<Arc<str>, RowId>,
    pub(crate) rows: HashMap<RowId, RowRecord>,
    /// Currently-live rows. A deleted row is dropped from `key_to_row` (so
    /// a re-inserted key allocates a fresh `RowId` rather than resurrecting
    /// the old one) but its `RowRecord` is left in `rows` — matching the
    /// master plan's "no row-id compaction" stance: gaps are permanent,
    /// `RowId`s are never reused.
    pub(crate) live: RoaringBitmap,
    pub(crate) next_doc_id: u64,
    pub(crate) doc_key_to_id: HashMap<Arc<str>, DocId>,
    pub(crate) docs: HashMap<DocId, crate::document::DocEntry>,
    pub(crate) payload: PayloadStore,
    pub(crate) txn_index: crate::txn_index::TxnIndex,
}

/// `undo.txn_history_limit`'s default — retains the last 10,000
/// transactions per collection before the oldest are forgotten (no longer
/// targetable by `undo --txn`).
pub const DEFAULT_TXN_HISTORY_LIMIT: usize = 10_000;

/// A single collection. Guarded by one `RwLock` — a stand-in, for now, for
/// the "single writer-actor task per collection" design in the master
/// plan's storage layer; the externally-visible behavior (writers
/// serialize against each other) is the same. Step 8 (the WAL) is where
/// that lock's scope narrows to specifically the append-and-apply step,
/// once there's a WAL append to actually serialize.
pub struct Collection {
    pub name: String,
    pub dim: usize,
    pub metric: DistanceMetric,
    pub(crate) inner: RwLock<CollectionInner>,
    subscribers: RwLock<Vec<Arc<dyn ChangeSubscriber>>>,
    /// `None` for a pure in-memory collection (tests, or anything that
    /// hasn't opted into durability). `Some` collections append-and-fsync
    /// every mutation to the WAL *before* applying it in memory — see
    /// `append_wal` — so a WAL append failure aborts the whole call with no
    /// partial in-memory state.
    pub(crate) wal: Option<WalWriter>,
}

impl Collection {
    /// A collection with no declared payload schema and no WAL — every
    /// field is accepted and stored (in the naive per-row map) but none of
    /// them are columnar-indexed, so `compile_filter` will reject any
    /// filter on them as an unknown field. Convenient for tests and for
    /// row-only use that never filters; anything that needs
    /// `compile_filter` should use [`Collection::with_schema`]; anything
    /// that needs durability should use [`Collection::open`].
    pub fn new(name: impl Into<String>, dim: usize, metric: DistanceMetric) -> Self {
        Self::with_schema(name, dim, metric, PayloadSchema::empty(), false)
    }

    pub fn with_schema(name: impl Into<String>, dim: usize, metric: DistanceMetric, schema: PayloadSchema, strict: bool) -> Self {
        Collection {
            name: name.into(),
            dim,
            metric,
            inner: RwLock::new(CollectionInner {
                last_applied_lsn: Lsn::ZERO,
                next_row_id: 0,
                key_to_row: HashMap::new(),
                rows: HashMap::new(),
                live: RoaringBitmap::new(),
                next_doc_id: 0,
                doc_key_to_id: HashMap::new(),
                docs: HashMap::new(),
                payload: PayloadStore::new(schema, strict),
                txn_index: crate::txn_index::TxnIndex::new(DEFAULT_TXN_HISTORY_LIMIT),
            }),
            subscribers: RwLock::new(Vec::new()),
            wal: None,
        }
    }

    /// Opens a durable collection rooted at `coll_dir` (matching the master
    /// plan's `collections/<name>/` layout: `coll_dir/wal/segment-*.wal`,
    /// `coll_dir/snapshot-*.mdb`). Startup reconciliation: load the newest
    /// checksummed-valid snapshot (if any), then replay only the WAL tail
    /// after its `last_applied_lsn` — or, with no snapshot, replay the
    /// whole WAL from empty. Either way this also truncates a torn tail
    /// left by a prior crash before resuming appends. This *is* crash
    /// recovery — there's no separate "recovery mode"; opening a
    /// collection always does it.
    pub fn open(
        name: impl Into<String>,
        dim: usize,
        metric: DistanceMetric,
        schema: PayloadSchema,
        strict: bool,
        coll_dir: impl Into<PathBuf>,
        segment_size_bytes: u32,
        fsync: FsyncPolicy,
    ) -> std::io::Result<Self> {
        let coll_dir = coll_dir.into();
        let wal_dir = coll_dir.join("wal");

        let snapshot = crate::snapshot::load_latest(&coll_dir)?;
        let after_lsn = snapshot.as_ref().map(|s| s.last_applied_lsn());
        let replay = wal::replay_all(&wal_dir, after_lsn)?;
        wal::truncate_segment(&wal_dir, replay.last_segment_id, replay.last_segment_valid_len)?;

        let coll = Self::with_schema(name, dim, metric, schema, strict);
        {
            let mut guard = coll.inner.write();
            if let Some(snapshot) = snapshot {
                snapshot.restore_into(&mut guard);
            }
            crate::recovery::apply_replayed_records(&mut guard, &replay.records)
                .expect("WAL records that passed checksum verification must apply cleanly");
        }

        let writer = WalWriter::open(&wal_dir, segment_size_bytes, fsync)?;
        Ok(Collection { wal: Some(writer), ..coll })
    }

    /// Writes a new checkpoint generation capturing exactly the state
    /// visible right now, tagged with the WAL position it's consistent
    /// with. A no-op for a non-durable (WAL-less) collection — there's
    /// nothing to reconcile a snapshot against. Retains the 3 most recent
    /// generations; older ones are removed once the new one is durable.
    pub fn checkpoint(&self) -> std::io::Result<()> {
        let Some(writer) = &self.wal else {
            return Ok(());
        };
        let coll_dir = writer
            .dir()
            .parent()
            .expect("the WAL directory (coll_dir/wal) always has a parent")
            .to_path_buf();

        // `last_applied_lsn` is tracked as part of `CollectionInner` itself
        // (bumped by `recovery::apply_txn_group` on every apply) rather
        // than read from the WAL writer's append cursor — that cursor
        // points *past* the last applied record, one transaction ahead of
        // what's actually reflected here, which would wrongly make the
        // next real write look already-captured to a later replay.
        let guard = self.inner.read();
        let snapshot = crate::snapshot::CollectionSnapshot::capture(&guard);
        drop(guard);

        crate::snapshot::write_checkpoint(&coll_dir, &snapshot)
    }

    /// Builds and appends WAL records for one transaction. Mutates each
    /// record's `lsn`/`checksum` in place (via `WalWriter`) and returns
    /// early on I/O failure — callers must not have mutated any in-memory
    /// state yet when they call this, so a failure here leaves storage
    /// exactly as it was before the call.
    pub(crate) fn append_wal(&self, records: &mut [WalRecord]) -> StorageResult<()> {
        if let Some(writer) = &self.wal {
            writer.append_batch(records).map_err(|e| StorageError::Wal(e.to_string()))?;
        }
        Ok(())
    }

    /// This collection's highest applied LSN, or `None` if it has never
    /// applied a single record. Deliberately not a bare `Lsn` defaulting to
    /// `Lsn::ZERO` for "nothing yet": `Lsn::ZERO` is a genuinely valid LSN
    /// — the very first record any collection's WAL ever writes lands
    /// exactly there — so collapsing "nothing applied" and "the first
    /// record is the latest one applied" onto the same sentinel value
    /// would make a fresh replication follower's very first catch-up
    /// request skip that first record. What a replication follower reports
    /// (as its resume cursor) and what a leader's [`Collection::
    /// wal_lines_after`] takes to decide where an incoming follower's
    /// WAL-tail read should start — `None` means "send everything,
    /// including the first record ever written."
    pub fn last_applied_lsn(&self) -> Option<Lsn> {
        let writer = self.wal.as_ref()?;
        if writer.current_lsn() == Lsn::ZERO {
            // The writer's next-append position is still at the very
            // start — nothing has ever been durably appended.
            return None;
        }
        Some(self.inner.read().last_applied_lsn)
    }

    /// This collection's payload schema, as `(field name, type)` pairs —
    /// what a replication leader sends a fresh follower so it can
    /// `create_collection` a matching local copy before any WAL streaming
    /// begins.
    pub fn schema_fields(&self) -> Vec<(String, crate::payload::FieldType)> {
        self.inner.read().payload.schema().iter().cloned().collect()
    }

    /// Leader side of replication: every WAL record strictly after
    /// `after_lsn` (`None` meaning "from the very first record"),
    /// re-serialized as raw JSONL lines ready to hand to a follower's
    /// [`Collection::apply_replicated_lines`], paired with the resume
    /// cursor to pass as `after_lsn` on the *next* call — the last
    /// returned record's own LSN, or `after_lsn` unchanged when there was
    /// nothing new. Returning this here (rather than making the caller
    /// derive it, e.g. from `last_applied_lsn()` taken *after* this call)
    /// is what keeps a poll loop race-free: this collection can keep
    /// accepting new local writes between one poll and the next, and a
    /// cursor derived from "whatever's newest right now" instead of "the
    /// last record actually included in *this* batch" would silently skip
    /// whatever landed in that window. Reuses `wal::replay_all`'s already
    /// checksum-verified read path rather than a second, parallel
    /// raw-tailing mechanism. A non-durable collection (`self.wal` is
    /// `None`) never has anything to stream.
    pub fn wal_lines_after(&self, after_lsn: Option<Lsn>) -> StorageResult<(Vec<String>, Option<Lsn>)> {
        let Some(writer) = &self.wal else {
            return Ok((Vec::new(), after_lsn));
        };
        let replay = wal::replay_all(writer.dir(), after_lsn).map_err(|e| StorageError::Wal(e.to_string()))?;
        let new_cursor = replay.records.last().map(|r| r.lsn).or(after_lsn);
        let lines = replay.records.iter().map(|r| serde_json::to_string(r).expect("WalRecord always serializes")).collect();
        Ok((lines, new_cursor))
    }

    /// Follower side of replication: applies WAL lines received verbatim
    /// from a leader. Parses and checksum-verifies each one (the same
    /// trust boundary `replay_all` enforces reading a local file — a
    /// replicated line is no more trusted than one this process wrote
    /// itself), appends them to this collection's own local WAL preserving
    /// the leader's exact LSNs (never minting fresh ones — see
    /// `WalWriter::append_replicated_batch`), applies them to in-memory
    /// state via the same `apply_replayed_records` startup recovery uses,
    /// and notifies this collection's own subscribers (`LiveIndex`, BM25)
    /// exactly as a live local write would: a follower's derived indexes
    /// stay current on their own local trigger, never by receiving an
    /// index shipped over the wire.
    pub fn apply_replicated_lines(&self, lines: &[String]) -> StorageResult<()> {
        if lines.is_empty() {
            return Ok(());
        }
        let mut records = Vec::with_capacity(lines.len());
        for line in lines {
            let record: WalRecord = serde_json::from_str(line).map_err(|e| StorageError::Wal(format!("malformed replicated WAL line: {e}")))?;
            if !wal::verify_checksum(&record) {
                return Err(StorageError::Wal(format!("replicated WAL line for lsn {} failed checksum verification", record.lsn)));
            }
            records.push(record);
        }

        if let Some(writer) = &self.wal {
            writer.append_replicated_batch(&records).map_err(|e| StorageError::Wal(e.to_string()))?;
        }

        let mut guard = self.inner.write();
        crate::recovery::apply_replayed_records(&mut guard, &records)?;
        drop(guard);

        // Records from more than one leader-side transaction can arrive in
        // the same batch; each still gets its own `ChangeBatch` (grouped
        // by `txn`, mirroring how `apply_replayed_records` itself groups
        // by consecutive `txn` runs) so a subscriber sees the same
        // transaction boundaries a live write would have produced.
        // `change_events_from_records` is called per-group, not once over
        // the whole batch then sliced — it `filter_map`s out control
        // records with no `row_id`, so a slice of the whole-batch output
        // wouldn't stay index-aligned with `records[i..j]` once any
        // earlier group contained one.
        let mut i = 0;
        while i < records.len() {
            let txn_id = records[i].txn;
            let mut j = i + 1;
            while j < records.len() && records[j].txn == txn_id {
                j += 1;
            }
            self.notify(ChangeBatch {
                txn_id,
                coll: self.name.clone(),
                events: crate::recovery::change_events_from_records(&records[i..j]),
            });
            i = j;
        }
        Ok(())
    }

    /// Compiles a `Filter` against this collection's live rows.
    /// `Filter::DocIn` is resolved here, against the doc registry;
    /// everything else is delegated to the columnar payload store.
    /// Tombstones are excluded unconditionally — the returned mask is
    /// always `(filter_result) ANDNOT dead_rows`, even when `filter` names
    /// no fields at all (an `And(vec![])`/trivial filter still excludes
    /// dead rows).
    pub fn compile_filter(&self, filter: &Filter) -> StorageResult<FilterMask> {
        let guard = self.inner.read();
        let allowed = Self::eval_filter(filter, &guard)?;
        Ok(FilterMask {
            estimated_cardinality: allowed.len(),
            allowed,
        })
    }

    fn eval_filter(filter: &Filter, guard: &CollectionInner) -> StorageResult<RoaringBitmap> {
        match filter {
            Filter::DocIn { doc_keys } => {
                let mut mask = RoaringBitmap::new();
                for key in doc_keys {
                    if let Some(doc_id) = guard.doc_key_to_id.get(key.as_str())
                        && let Some(entry) = guard.docs.get(doc_id) {
                            mask |= &entry.rows;
                        }
                }
                Ok(mask & &guard.live)
            }
            Filter::And(fs) => {
                let mut acc = guard.live.clone();
                for f in fs {
                    acc &= Self::eval_filter(f, guard)?;
                }
                Ok(acc)
            }
            Filter::Or(fs) => {
                let mut acc = RoaringBitmap::new();
                for f in fs {
                    acc |= Self::eval_filter(f, guard)?;
                }
                Ok(acc & &guard.live)
            }
            Filter::Not(f) => Ok(&guard.live - Self::eval_filter(f, guard)?),
            other => Ok(guard.payload.compile(other, &guard.live)? & &guard.live),
        }
    }

    pub fn subscribe(&self, subscriber: Arc<dyn ChangeSubscriber>) {
        self.subscribers.write().push(subscriber);
    }

    pub(crate) fn notify(&self, batch: ChangeBatch) {
        for sub in self.subscribers.read().iter() {
            sub.on_change(&batch);
        }
    }

    pub(crate) fn check_dim(&self, vector: &[f32]) -> StorageResult<()> {
        if vector.len() != self.dim {
            return Err(StorageError::DimensionMismatch {
                coll: self.name.clone(),
                expected: self.dim,
                got: vector.len(),
            });
        }
        Ok(())
    }

    fn check_dims(&self, items: &[PutInput]) -> StorageResult<()> {
        for item in items {
            self.check_dim(&item.vector)?;
        }
        Ok(())
    }

    pub(crate) fn to_row(row_id: RowId, rec: &RowRecord) -> Row {
        Row {
            id: row_id,
            key: rec.key.to_string(),
            vector: Some(rec.vector.to_vec()),
            fields: (*rec.fields).clone(),
            extra: rec.extra.as_ref().map(|e| (**e).clone()),
            text: rec.text.as_ref().map(|t| t.to_string()),
            doc_id: rec.doc_id,
            chunk_ord: rec.chunk_ord,
        }
    }

    /// Upsert by key, one write-lock acquisition and one `ChangeBatch` for
    /// the whole batch regardless of size — the same amortization the WAL
    /// fsync relies on. A fresh key allocates a new `RowId` (WAL op
    /// `insert`); an existing live key updates that same `RowId` in place
    /// (WAL op `update`).
    ///
    /// Row-id assignment and "before" images are decided in a read-only
    /// pass first, then written to the WAL (if any), and only *then*
    /// applied in memory — a WAL append failure aborts the call with no
    /// partial in-memory state, per the master plan's durability ordering.
    pub fn put_batch(&self, ctx: &RequestCtx, items: Vec<PutInput>) -> StorageResult<Vec<RowId>> {
        self.check_dims(&items)?;
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let txn_id = TxnId::new();
        let mut guard = self.inner.write();
        // Validate every item's fields against the schema before deciding
        // or mutating anything, so a mid-batch validation failure can't
        // leave the columnar payload store half-updated relative to the
        // rows map.
        for item in &items {
            guard.payload.validate_row(&item.fields)?;
        }

        struct Planned {
            row_id: RowId,
            is_new: bool,
            old: Option<(Arc<[f32]>, Arc<PayloadRow>, Option<Arc<ExtraPayload>>, Option<Arc<str>>, Option<DocId>, Option<u32>)>,
        }
        let mut planned = Vec::with_capacity(items.len());
        let mut next_id_counter = guard.next_row_id;
        for item in &items {
            if let Some(&existing) = guard.key_to_row.get(item.key.as_str()) {
                let old = guard
                    .rows
                    .get(&existing)
                    .map(|r| (r.vector.clone(), r.fields.clone(), r.extra.clone(), r.text.clone(), r.doc_id, r.chunk_ord));
                planned.push(Planned {
                    row_id: existing,
                    is_new: false,
                    old,
                });
            } else {
                let row_id = RowId(next_id_counter);
                next_id_counter += 1;
                planned.push(Planned {
                    row_id,
                    is_new: true,
                    old: None,
                });
            }
        }

        let n = items.len() as u32;
        let mut wal_records: Vec<WalRecord> = items
            .iter()
            .zip(&planned)
            .enumerate()
            .map(|(i, (item, p))| WalRecord {
                v: WAL_FORMAT_VERSION,
                lsn: Lsn::ZERO,
                ts: Utc::now(),
                txn: txn_id,
                actor: WalActor {
                    id: ctx.principal.id.clone(),
                    name: ctx.principal.name.clone(),
                },
                session: ctx.session.clone(),
                coll: self.name.clone(),
                op: if p.is_new { WalOp::Insert } else { WalOp::Update },
                row_id: Some(p.row_id),
                key: Some(item.key.clone()),
                // A row-level put never changes document association —
                // preserve it verbatim on update (`None` for a fresh
                // insert, which can never already belong to a document).
                doc_id: p.old.as_ref().and_then(|(_, _, _, _, d, _)| *d),
                doc_key: None,
                chunk_ord: p.old.as_ref().and_then(|(_, _, _, _, _, c)| *c),
                payload: Some(WalPayload {
                    vector_b64: Some(wal::encode_vector(&item.vector)),
                    text: p.old.as_ref().and_then(|(_, _, _, t, _, _)| t.as_ref()).map(|t| t.to_string()),
                    fields: Some(item.fields.clone()),
                    extra: item.extra.clone(),
                }),
                undo: p.old.as_ref().map(|(v, f, e, t, _, _)| WalPayload {
                    vector_b64: Some(wal::encode_vector(v)),
                    text: t.as_ref().map(|s| s.to_string()),
                    fields: Some((**f).clone()),
                    extra: e.as_ref().map(|e| (**e).clone()),
                }),
                caused_by: None,
                txn_seq: (i as u32 + 1, n),
                params: None,
                checksum: String::new(),
            })
            .collect();
        self.append_wal(&mut wal_records)?;

        let row_ids: Vec<RowId> = planned.iter().map(|p| p.row_id).collect();
        let events = crate::recovery::change_events_from_records(&wal_records);
        crate::recovery::apply_replayed_records(&mut guard, &wal_records).expect("just-appended, well-formed records must apply cleanly");
        drop(guard);
        self.notify(ChangeBatch {
            txn_id,
            coll: self.name.clone(),
            events,
        });
        Ok(row_ids)
    }

    pub fn put(
        &self,
        ctx: &RequestCtx,
        key: &str,
        vector: Vec<f32>,
        fields: PayloadRow,
        extra: Option<ExtraPayload>,
    ) -> StorageResult<RowId> {
        let ids = self.put_batch(
            ctx,
            vec![PutInput {
                key: key.to_string(),
                vector,
                fields,
                extra,
            }],
        )?;
        Ok(ids[0])
    }

    pub fn delete(&self, ctx: &RequestCtx, key: &str) -> StorageResult<()> {
        let txn_id = TxnId::new();
        let mut guard = self.inner.write();
        let row_id = *guard.key_to_row.get(key).ok_or_else(|| StorageError::KeyNotFound(key.to_string()))?;
        let rec = guard.rows.get(&row_id).expect("key_to_row and rows must stay in sync");

        let mut wal_records = vec![WalRecord {
            v: WAL_FORMAT_VERSION,
            lsn: Lsn::ZERO,
            ts: Utc::now(),
            txn: txn_id,
            actor: WalActor {
                id: ctx.principal.id.clone(),
                name: ctx.principal.name.clone(),
            },
            session: ctx.session.clone(),
            coll: self.name.clone(),
            op: WalOp::Delete,
            row_id: Some(row_id),
            key: Some(key.to_string()),
            doc_id: rec.doc_id,
            doc_key: None,
            chunk_ord: rec.chunk_ord,
            payload: None,
            undo: Some(WalPayload {
                vector_b64: Some(wal::encode_vector(&rec.vector)),
                text: rec.text.as_ref().map(|t| t.to_string()),
                fields: Some((*rec.fields).clone()),
                extra: rec.extra.as_ref().map(|e| (**e).clone()),
            }),
            caused_by: None,
            txn_seq: (1, 1),
            params: None,
            checksum: String::new(),
        }];
        self.append_wal(&mut wal_records)?;

        let events = crate::recovery::change_events_from_records(&wal_records);
        crate::recovery::apply_replayed_records(&mut guard, &wal_records).expect("just-appended, well-formed records must apply cleanly");
        drop(guard);
        self.notify(ChangeBatch {
            txn_id,
            coll: self.name.clone(),
            events,
        });
        Ok(())
    }

    pub fn get_by_key(&self, key: &str) -> Option<Row> {
        let guard = self.inner.read();
        let row_id = *guard.key_to_row.get(key)?;
        guard.rows.get(&row_id).map(|rec| Self::to_row(row_id, rec))
    }

    /// Rows in ascending `RowId` order, strictly after `after`. The basis
    /// for full/incremental index rebuilds and reservoir-sampled training
    /// data (the index layer does its own sampling over this cursor).
    pub fn scan(&self, after: Option<RowId>, limit: usize) -> Vec<Row> {
        let guard = self.inner.read();
        let after_idx = after.map(RowId::to_bitmap_index);
        guard
            .live
            .iter()
            .filter(|&idx| after_idx.is_none_or(|a| idx > a))
            .take(limit)
            .filter_map(|idx| {
                let row_id = RowId::from_bitmap_index(idx);
                guard.rows.get(&row_id).map(|rec| Self::to_row(row_id, rec))
            })
            .collect()
    }

    pub fn fetch_vectors(&self, ids: &[RowId]) -> Vec<Option<Vec<f32>>> {
        let guard = self.inner.read();
        ids.iter()
            .map(|id| guard.rows.get(id).map(|r| r.vector.to_vec()))
            .collect()
    }

    /// Full `Row`s by id, in `ids`' order — what a candidate-set index
    /// (posting lists, ANN graphs) reranks and reports from, once it has
    /// `RowId`s rather than keys. Unlike `fetch_vectors`, this is
    /// liveness-checked: `guard.rows` keeps a deleted row's record around
    /// for undo, so a `RowId` a candidate set collected before a
    /// since-then delete comes back `None` here rather than silently
    /// resurfacing stale data an index hasn't been rebuilt to drop yet.
    pub fn rows_by_id(&self, ids: &[RowId]) -> Vec<Option<Row>> {
        let guard = self.inner.read();
        ids.iter()
            .map(|id| {
                if !guard.live.contains(id.to_bitmap_index()) {
                    return None;
                }
                guard.rows.get(id).map(|rec| Self::to_row(*id, rec))
            })
            .collect()
    }

    pub fn active_count(&self) -> u64 {
        self.inner.read().live.len()
    }

    pub fn alter_schema(&self, _ctx: &RequestCtx, change: crate::api::SchemaChange) -> StorageResult<()> {
        let mut guard = self.inner.write();
        match change {
            crate::api::SchemaChange::AddField { name, field_type } => guard.payload.add_field(name, field_type)?,
            crate::api::SchemaChange::DropField { name } => guard.payload.drop_field(&name)?,
        }
        Ok(())
    }

    /// The row's naive ground-truth fields (not reconstructed from the
    /// columnar store) — what `get_by_key`/`scan` already return, exposed
    /// standalone for callers that only need the payload, not the vector.
    pub fn payload_of(&self, row_id: RowId) -> Option<PayloadRow> {
        self.inner.read().rows.get(&row_id).map(|r| (*r.fields).clone())
    }

    pub fn payload_batch(&self, ids: &[RowId]) -> Vec<Option<PayloadRow>> {
        let guard = self.inner.read();
        ids.iter().map(|id| guard.rows.get(id).map(|r| (*r.fields).clone())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mara_proto::{Principal, Role, SessionId, Source};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    fn ctx() -> RequestCtx {
        RequestCtx::new(
            SessionId("test-session".into()),
            Principal {
                id: mara_proto::PrincipalId("p_test".into()),
                name: "tester".into(),
                role: Role::Writer,
            },
            Source::Embedded,
        )
    }

    fn coll() -> Collection {
        Collection::new("docs", 3, DistanceMetric::Cosine)
    }

    fn payload_row(field: &str, value: &str) -> PayloadRow {
        let mut m = PayloadRow::new();
        m.insert(field.to_string(), mara_proto::PayloadValue::Keyword(value.to_string()));
        m
    }

    #[test]
    fn put_then_get_round_trips() {
        let c = coll();
        let row_id = c
            .put(&ctx(), "a", vec![1.0, 2.0, 3.0], payload_row("tag", "x"), None)
            .unwrap();
        let row = c.get_by_key("a").unwrap();
        assert_eq!(row.id, row_id);
        assert_eq!(row.vector, Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(c.active_count(), 1);
    }

    #[test]
    fn put_with_wrong_dim_is_rejected() {
        let c = coll();
        let err = c.put(&ctx(), "a", vec![1.0, 2.0], PayloadRow::new(), None).unwrap_err();
        assert!(matches!(err, StorageError::DimensionMismatch { .. }));
        assert_eq!(c.active_count(), 0);
    }

    #[test]
    fn re_putting_an_existing_key_updates_the_same_row_id() {
        let c = coll();
        let first = c.put(&ctx(), "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        let second = c.put(&ctx(), "a", vec![0.0, 1.0, 0.0], PayloadRow::new(), None).unwrap();
        assert_eq!(first, second, "an upsert on an existing key must reuse its RowId");
        assert_eq!(c.active_count(), 1);
        let row = c.get_by_key("a").unwrap();
        assert_eq!(row.vector, Some(vec![0.0, 1.0, 0.0]));
    }

    #[test]
    fn delete_then_reinsert_gets_a_fresh_never_reused_row_id() {
        let c = coll();
        let first = c.put(&ctx(), "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        c.delete(&ctx(), "a").unwrap();
        assert!(c.get_by_key("a").is_none());
        assert_eq!(c.active_count(), 0);

        let second = c.put(&ctx(), "a", vec![0.0, 0.0, 1.0], PayloadRow::new(), None).unwrap();
        assert_ne!(first, second, "RowId must never be reused after a delete");
        assert_eq!(c.active_count(), 1);
    }

    #[test]
    fn delete_of_missing_key_is_an_error() {
        let c = coll();
        assert_eq!(
            c.delete(&ctx(), "missing").unwrap_err(),
            StorageError::KeyNotFound("missing".into())
        );
    }

    #[test]
    fn scan_is_ordered_and_respects_the_cursor() {
        let c = coll();
        for k in ["a", "b", "c", "d"] {
            c.put(&ctx(), k, vec![0.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        }
        let page1 = c.scan(None, 2);
        assert_eq!(page1.len(), 2);
        assert!(page1[0].id < page1[1].id);

        let page2 = c.scan(Some(page1[1].id), 10);
        assert_eq!(page2.len(), 2);
        assert!(page2[0].id > page1[1].id);
    }

    #[test]
    fn scan_skips_deleted_rows() {
        let c = coll();
        c.put(&ctx(), "a", vec![0.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        c.put(&ctx(), "b", vec![0.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        c.delete(&ctx(), "a").unwrap();
        let rows = c.scan(None, 10);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].key, "b");
    }

    #[test]
    fn fetch_vectors_returns_none_for_missing_ids() {
        let c = coll();
        let id = c.put(&ctx(), "a", vec![1.0, 2.0, 3.0], PayloadRow::new(), None).unwrap();
        let missing = RowId(999);
        let result = c.fetch_vectors(&[id, missing]);
        assert_eq!(result[0], Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(result[1], None);
    }

    #[test]
    fn rows_by_id_returns_full_rows_in_the_requested_order() {
        let c = coll();
        let id_a = c.put(&ctx(), "a", vec![1.0, 2.0, 3.0], PayloadRow::new(), None).unwrap();
        let id_b = c.put(&ctx(), "b", vec![4.0, 5.0, 6.0], PayloadRow::new(), None).unwrap();
        let rows = c.rows_by_id(&[id_b, id_a]);
        assert_eq!(rows[0].as_ref().unwrap().key, "b");
        assert_eq!(rows[1].as_ref().unwrap().key, "a");
    }

    #[test]
    fn rows_by_id_hides_a_deleted_row_even_though_its_record_survives_for_undo() {
        let c = coll();
        let id = c.put(&ctx(), "a", vec![1.0, 2.0, 3.0], PayloadRow::new(), None).unwrap();
        c.delete(&ctx(), "a").unwrap();
        let rows = c.rows_by_id(&[id]);
        assert_eq!(rows[0], None, "a candidate set built before the delete must not resurface stale data");
    }

    struct CountingSubscriber {
        batches: AtomicUsize,
        last_event_count: Mutex<usize>,
    }

    impl ChangeSubscriber for CountingSubscriber {
        fn on_change(&self, batch: &ChangeBatch) {
            self.batches.fetch_add(1, Ordering::SeqCst);
            *self.last_event_count.lock().unwrap() = batch.events.len();
        }
    }

    #[test]
    fn put_batch_notifies_subscribers_exactly_once_per_call() {
        let c = coll();
        let sub = Arc::new(CountingSubscriber {
            batches: AtomicUsize::new(0),
            last_event_count: Mutex::new(0),
        });
        c.subscribe(sub.clone());

        let items: Vec<PutInput> = (0..50)
            .map(|i| PutInput {
                key: format!("k{i}"),
                vector: vec![i as f32, 0.0, 0.0],
                fields: PayloadRow::new(),
                extra: None,
            })
            .collect();
        c.put_batch(&ctx(), items).unwrap();

        assert_eq!(sub.batches.load(Ordering::SeqCst), 1, "one ChangeBatch per put_batch call");
        assert_eq!(*sub.last_event_count.lock().unwrap(), 50);
    }

    struct TextCapturingSubscriber {
        texts: Mutex<Vec<Option<String>>>,
    }

    impl ChangeSubscriber for TextCapturingSubscriber {
        fn on_change(&self, batch: &ChangeBatch) {
            let mut texts = self.texts.lock().unwrap();
            for e in &batch.events {
                let t = match e {
                    crate::change::ChangeEvent::Insert { text, .. } => text.clone(),
                    crate::change::ChangeEvent::Update { text, .. } => text.clone(),
                    crate::change::ChangeEvent::Delete { text, .. } => text.clone(),
                };
                texts.push(t.map(|t| t.to_string()));
            }
        }
    }

    #[test]
    fn change_events_carry_the_rows_text_for_a_document_insert_and_delete() {
        let c = coll();
        let sub = Arc::new(TextCapturingSubscriber { texts: Mutex::new(Vec::new()) });
        c.subscribe(sub.clone());

        c.put_document(
            &ctx(),
            crate::document::PutDocumentInput {
                doc_key: "onboarding.md".into(),
                chunks: vec![crate::document::ChunkInput {
                    text: "hello world chunk text".into(),
                    vector: vec![1.0, 0.0, 0.0],
                }],
                doc_payload: PayloadRow::new(),
                chunk_spec: mara_proto::ChunkSpec::default(),
                source: None,
                embedding_model: mara_proto::ModelFingerprint {
                    model_id: "test".into(),
                    revision: None,
                    dim: 3,
                },
            },
        )
        .unwrap();

        {
            let texts = sub.texts.lock().unwrap();
            assert_eq!(texts.len(), 1);
            assert_eq!(texts[0].as_deref(), Some("hello world chunk text"));
        }

        c.delete_document(&ctx(), "onboarding.md").unwrap();
        let texts = sub.texts.lock().unwrap();
        assert_eq!(texts.len(), 2, "the delete's ChangeEvent must also have arrived");
        assert_eq!(texts[1].as_deref(), Some("hello world chunk text"), "a delete's ChangeEvent must carry the text the row was indexed under, from the undo payload");
    }

    #[test]
    fn empty_put_batch_does_not_notify() {
        let c = coll();
        let sub = Arc::new(CountingSubscriber {
            batches: AtomicUsize::new(0),
            last_event_count: Mutex::new(0),
        });
        c.subscribe(sub.clone());
        let ids = c.put_batch(&ctx(), vec![]).unwrap();
        assert!(ids.is_empty());
        assert_eq!(sub.batches.load(Ordering::SeqCst), 0);
    }
}
