//! Undo/revert (master plan Layer 2, *Undo/revert — the core mechanism*).
//!
//! Two naive designs were rejected in the plan: pure replay-to-earlier-LSN
//! breaks the moment new writes follow an undo, and pure
//! per-record-inverse-chaining can't answer "revert to an arbitrary point
//! in time". The actual mechanism: apply the target transaction's stored
//! `undo` images in reverse LSN order, as **brand-new, forward-appended**
//! WAL records. LSNs only ever increase; nothing is ever rewound or
//! truncated. That's what makes undo crash-safe (a crash mid-undo leaves it
//! partially applied; replay resumes normally — see `recovery.rs`) and
//! replication-safe (a follower streaming raw WAL replays undo like any
//! other write) *by construction*, and it's why undo-of-an-undo needs no
//! special case: a compensating transaction is an ordinary transaction.

use crate::change::ChangeBatch;
use crate::collection::Collection;
use crate::error::{StorageError, StorageResult};
use crate::wal::{self, WalActor, WalOp, WalRecord, WAL_FORMAT_VERSION};
use chrono::Utc;
use mara_proto::{Lsn, RequestCtx, SessionId, TxnEntry, TxnId, TxnSummary};

/// Builds the compensating record for `original`: the WAL-level inverse of
/// exactly one record. `Insert` inverts to `Delete` (with `undo` set to
/// what was inserted, so this compensating delete can itself be undone);
/// `Delete` inverts to `Insert` (using the stored `undo` image as the new
/// `payload`); `Update` inverts to `Update` (swap `payload`/`undo`).
/// `row_id`/`key`/`doc_id`/`doc_key`/`chunk_ord`/`params` all carry over
/// unchanged — this reverses what happened to that row, not its identity.
fn invert_record(original: &WalRecord, new_txn: TxnId, actor: &WalActor, session: &SessionId, seq: (u32, u32), caused_by: TxnId) -> WalRecord {
    let (op, payload, undo) = match original.op {
        WalOp::Insert => (WalOp::Delete, None, original.payload.clone()),
        WalOp::Delete => (WalOp::Insert, original.undo.clone(), None),
        WalOp::Update => (WalOp::Update, original.undo.clone(), original.payload.clone()),
        other => (other, original.payload.clone(), original.undo.clone()),
    };
    WalRecord {
        v: WAL_FORMAT_VERSION,
        lsn: Lsn::ZERO,
        ts: Utc::now(),
        txn: new_txn,
        actor: actor.clone(),
        session: session.clone(),
        coll: original.coll.clone(),
        op,
        row_id: original.row_id,
        key: original.key.clone(),
        doc_id: original.doc_id,
        doc_key: original.doc_key.clone(),
        chunk_ord: original.chunk_ord,
        payload,
        undo,
        caused_by: Some(caused_by),
        txn_seq: seq,
        params: original.params.clone(),
        checksum: String::new(),
    }
}

impl Collection {
    /// Reverses transaction `target_txn`, canonical and unambiguous — see
    /// the master plan's *Undo targeting*: the unit of undo is the
    /// transaction, never a positional count, which is a concurrency
    /// hazard on a multi-client server. Refuses (unless `force`) if any of
    /// the target's rows were touched by a later transaction, reporting
    /// exactly which ones so the caller can `undo` that transaction first
    /// instead of destroying it.
    pub fn undo(&self, ctx: &RequestCtx, target_txn: TxnId, force: bool) -> StorageResult<TxnSummary> {
        let wal_dir = self
            .wal
            .as_ref()
            .ok_or_else(|| StorageError::InvalidArgument("undo requires a durable (WAL-backed) collection".into()))?
            .dir()
            .to_path_buf();

        let target_entry = {
            let guard = self.inner.read();
            let entry = guard
                .txn_index
                .get(target_txn)
                .cloned()
                .ok_or_else(|| StorageError::InvalidArgument(format!("no such transaction {target_txn} (it may have been trimmed by undo.txn_history_limit, or never existed)")))?;
            if let Some(undone_by) = entry.undone_by {
                return Err(StorageError::InvalidArgument(format!(
                    "transaction {target_txn} was already undone by {undone_by}; undo is idempotent-by-inspection, not double-appliable"
                )));
            }
            if !force {
                let conflicts = guard.txn_index.conflicts_for(&entry);
                if !conflicts.is_empty() {
                    return Err(StorageError::UndoConflict { txn: target_txn, conflicts });
                }
            }
            entry
        };

        // Reconstruct the target transaction's own records from the WAL —
        // the `TxnIndex` only keeps a summary; the row-level `undo` images
        // needed to actually reverse it live only in the WAL itself.
        let replay = wal::replay_all(&wal_dir, None).map_err(|e| StorageError::Wal(e.to_string()))?;
        let mut target_records: Vec<WalRecord> = replay.records.into_iter().filter(|r| r.txn == target_txn).collect();
        if target_records.is_empty() {
            return Err(StorageError::InvalidArgument(format!(
                "transaction {target_txn} has no records left in the WAL (likely rotated out by wal.retention_days)"
            )));
        }
        target_records.sort_by_key(|r| r.lsn);
        target_records.reverse(); // "reverse LSN order", per the master plan

        let new_txn = TxnId::new();
        let n = target_records.len() as u32;
        let actor = WalActor {
            id: ctx.principal.id.clone(),
            name: ctx.principal.name.clone(),
        };
        let mut compensating: Vec<WalRecord> = target_records
            .iter()
            .enumerate()
            .map(|(i, r)| invert_record(r, new_txn, &actor, &ctx.session, (i as u32 + 1, n), target_txn))
            .collect();

        self.append_wal(&mut compensating)?;

        let mut guard = self.inner.write();
        let events = crate::recovery::change_events_from_records(&compensating);
        // `apply_replayed_records` also records `new_txn`'s own `TxnEntry`
        // and marks `target_txn.undone_by = Some(new_txn)` — see
        // `recovery::apply_txn_group`'s `caused_by` handling. Nothing
        // undo-specific needs to happen here beyond that.
        crate::recovery::apply_replayed_records(&mut guard, &compensating)
            .expect("compensating records built from a valid original transaction must apply cleanly");
        drop(guard);

        self.notify(ChangeBatch {
            txn_id: new_txn,
            coll: self.name.clone(),
            events,
        });

        Ok(TxnSummary {
            txn_id: target_txn,
            op_summary: target_entry.op_summary,
            rows_reversed: n as u64,
        })
    }

    /// Recent transactions, most-recent first — the basis for `mara
    /// history` and for a caller inspecting what an `undo` is about to
    /// reverse before running it.
    pub fn history(&self, limit: usize) -> Vec<TxnEntry> {
        let guard = self.inner.read();
        guard.txn_index.iter_chronological().rev().take(limit).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::PayloadSchema;
    use crate::{ChunkInput, FsyncPolicy, PutDocumentInput};
    use mara_proto::{ChunkSpec, DistanceMetric, ModelFingerprint, PayloadRow, Principal, PrincipalId, Role, Source};

    fn ctx() -> RequestCtx {
        RequestCtx::new(
            SessionId("s".into()),
            Principal {
                id: PrincipalId("p".into()),
                name: "t".into(),
                role: Role::Writer,
            },
            Source::Embedded,
        )
    }

    fn ctx_as(name: &str) -> RequestCtx {
        RequestCtx::new(
            SessionId(format!("session-{name}")),
            Principal {
                id: PrincipalId(format!("p_{name}")),
                name: name.to_string(),
                role: Role::Writer,
            },
            Source::Embedded,
        )
    }

    fn open(dir: &std::path::Path) -> Collection {
        Collection::open("docs", 3, DistanceMetric::Cosine, PayloadSchema::empty(), false, dir, 128 * 1024 * 1024, FsyncPolicy::Always).unwrap()
    }

    fn model() -> ModelFingerprint {
        ModelFingerprint {
            model_id: "m".into(),
            revision: None,
            dim: 3,
        }
    }

    #[test]
    fn undo_of_an_insert_removes_the_row() {
        let dir = tempfile::tempdir().unwrap();
        let c = open(dir.path());
        let ctx = ctx();
        c.put(&ctx, "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();

        let target = c.history(10)[0].txn_id;
        assert!(c.get_by_key("a").is_some());

        let summary = c.undo(&ctx, target, false).unwrap();
        assert_eq!(summary.rows_reversed, 1);
        assert!(c.get_by_key("a").is_none(), "undoing an insert must remove the row");
        assert_eq!(c.active_count(), 0);
    }

    #[test]
    fn undo_of_a_delete_resurrects_the_row_with_the_same_row_id() {
        let dir = tempfile::tempdir().unwrap();
        let c = open(dir.path());
        let ctx = ctx();
        let id = c.put(&ctx, "a", vec![1.0, 2.0, 3.0], PayloadRow::new(), None).unwrap();
        c.delete(&ctx, "a").unwrap();
        assert!(c.get_by_key("a").is_none());

        let delete_txn = c.history(10)[0].txn_id;
        c.undo(&ctx, delete_txn, false).unwrap();

        let row = c.get_by_key("a").expect("undoing a delete must resurrect the row");
        assert_eq!(row.id, id, "resurrection must reuse the exact original RowId");
        assert_eq!(row.vector, Some(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn undo_of_an_update_restores_the_prior_value() {
        let dir = tempfile::tempdir().unwrap();
        let c = open(dir.path());
        let ctx = ctx();
        c.put(&ctx, "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        c.put(&ctx, "a", vec![0.0, 1.0, 0.0], PayloadRow::new(), None).unwrap();
        assert_eq!(c.get_by_key("a").unwrap().vector, Some(vec![0.0, 1.0, 0.0]));

        let update_txn = c.history(10)[0].txn_id;
        c.undo(&ctx, update_txn, false).unwrap();
        assert_eq!(c.get_by_key("a").unwrap().vector, Some(vec![1.0, 0.0, 0.0]));
    }

    #[test]
    fn undo_of_undo_restores_the_undone_state() {
        let dir = tempfile::tempdir().unwrap();
        let c = open(dir.path());
        let ctx = ctx();
        c.put(&ctx, "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        let insert_txn = c.history(10)[0].txn_id;

        c.undo(&ctx, insert_txn, false).unwrap();
        assert!(c.get_by_key("a").is_none());
        let undo_txn = c.history(10)[0].txn_id;

        c.undo(&ctx, undo_txn, false).unwrap();
        assert!(c.get_by_key("a").is_some(), "undo-of-undo must need no special case and just work");
    }

    #[test]
    fn undo_is_rejected_once_already_undone() {
        let dir = tempfile::tempdir().unwrap();
        let c = open(dir.path());
        let ctx = ctx();
        c.put(&ctx, "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        let target = c.history(10)[0].txn_id;
        c.undo(&ctx, target, false).unwrap();

        let err = c.undo(&ctx, target, false).unwrap_err();
        assert!(matches!(err, StorageError::InvalidArgument(_)));
    }

    #[test]
    fn undo_document_removes_every_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let c = open(dir.path());
        let ctx = ctx();
        c.put_document(
            &ctx,
            PutDocumentInput {
                doc_key: "handbook.md".into(),
                chunks: vec![
                    ChunkInput { text: "c0".into(), vector: vec![1.0, 0.0, 0.0] },
                    ChunkInput { text: "c1".into(), vector: vec![0.0, 1.0, 0.0] },
                    ChunkInput { text: "c2".into(), vector: vec![0.0, 0.0, 1.0] },
                ],
                doc_payload: PayloadRow::new(),
                chunk_spec: ChunkSpec::default(),
                source: None,
                embedding_model: model(),
            },
        )
        .unwrap();
        assert_eq!(c.active_count(), 3);

        let target = c.history(10)[0].txn_id;
        let summary = c.undo(&ctx, target, false).unwrap();
        assert_eq!(summary.rows_reversed, 3);
        assert_eq!(c.active_count(), 0);
        assert!(c.get_document("handbook.md").is_none());
    }

    #[test]
    fn undo_of_replace_document_restores_the_prior_version_and_content() {
        let dir = tempfile::tempdir().unwrap();
        let c = open(dir.path());
        let ctx = ctx();
        let doc_id = c
            .put_document(
                &ctx,
                PutDocumentInput {
                    doc_key: "doc.md".into(),
                    chunks: vec![ChunkInput { text: "v1".into(), vector: vec![1.0, 0.0, 0.0] }],
                    doc_payload: PayloadRow::new(),
                    chunk_spec: ChunkSpec::default(),
                    source: None,
                    embedding_model: model(),
                },
            )
            .unwrap();

        c.replace_document(
            &ctx,
            "doc.md",
            vec![
                ChunkInput { text: "v2a".into(), vector: vec![0.0, 1.0, 0.0] },
                ChunkInput { text: "v2b".into(), vector: vec![0.0, 0.0, 1.0] },
            ],
            PayloadRow::new(),
            ChunkSpec::default(),
            None,
            model(),
        )
        .unwrap();
        assert_eq!(c.get_document("doc.md").unwrap().version, 2);
        assert_eq!(c.active_count(), 2);

        let replace_txn = c.history(10)[0].txn_id;
        c.undo(&ctx, replace_txn, false).unwrap();

        let entry = c.get_document("doc.md").expect("undo of replace must leave the document present, not deleted");
        assert_eq!(entry.doc_id, doc_id);
        assert_eq!(entry.version, 1, "version must revert to its pre-replace value");
        let chunks = c.list_chunks(doc_id).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text.as_deref(), Some("v1"));
    }

    #[test]
    fn undo_of_delete_document_restores_metadata_and_all_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let c = open(dir.path());
        let ctx = ctx();
        let doc_id = c
            .put_document(
                &ctx,
                PutDocumentInput {
                    doc_key: "doc.md".into(),
                    chunks: vec![
                        ChunkInput { text: "a".into(), vector: vec![1.0, 0.0, 0.0] },
                        ChunkInput { text: "b".into(), vector: vec![0.0, 1.0, 0.0] },
                    ],
                    doc_payload: PayloadRow::new(),
                    chunk_spec: ChunkSpec::default(),
                    source: Some("origin.md".into()),
                    embedding_model: model(),
                },
            )
            .unwrap();
        c.delete_document(&ctx, "doc.md").unwrap();
        assert!(c.get_document("doc.md").is_none());

        let delete_txn = c.history(10)[0].txn_id;
        c.undo(&ctx, delete_txn, false).unwrap();

        let entry = c.get_document("doc.md").expect("undo of delete_document must restore the document");
        assert_eq!(entry.doc_id, doc_id);
        assert_eq!(entry.version, 1);
        assert_eq!(entry.source.as_deref(), Some("origin.md"));
        let chunks = c.list_chunks(doc_id).unwrap();
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn concurrent_writers_undo_only_touches_its_own_transaction() {
        // The multi-client hazard the review flagged: client A's undo must
        // never destroy client B's independent write.
        let dir = tempfile::tempdir().unwrap();
        let c = open(dir.path());
        let alice = ctx_as("alice");
        let bob = ctx_as("bob");

        c.put(&alice, "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        c.put(&bob, "b", vec![0.0, 1.0, 0.0], PayloadRow::new(), None).unwrap();

        let alice_txn = c.history(10).iter().find(|e| e.principal.0 == "p_alice").unwrap().txn_id;
        c.undo(&alice, alice_txn, false).unwrap();

        assert!(c.get_by_key("a").is_none(), "alice's own row must be gone");
        assert!(c.get_by_key("b").is_some(), "bob's independent write must be completely untouched");
    }

    #[test]
    fn undo_refuses_on_write_write_conflict_without_force() {
        let dir = tempfile::tempdir().unwrap();
        let c = open(dir.path());
        let ctx = ctx();
        c.put(&ctx, "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        let insert_txn = c.history(10)[0].txn_id;

        // A later transaction touches the same row.
        c.put(&ctx, "a", vec![0.0, 1.0, 0.0], PayloadRow::new(), None).unwrap();

        let err = c.undo(&ctx, insert_txn, false).unwrap_err();
        match err {
            StorageError::UndoConflict { txn, conflicts } => {
                assert_eq!(txn, insert_txn);
                assert_eq!(conflicts.len(), 1);
                assert_eq!(conflicts[0].row_count, 1);
            }
            other => panic!("expected UndoConflict, got {other:?}"),
        }
        // Nothing changed — the conflicting update's value must still stand.
        assert_eq!(c.get_by_key("a").unwrap().vector, Some(vec![0.0, 1.0, 0.0]));
    }

    #[test]
    fn undo_with_force_overrides_a_conflict() {
        let dir = tempfile::tempdir().unwrap();
        let c = open(dir.path());
        let ctx = ctx();
        c.put(&ctx, "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        let insert_txn = c.history(10)[0].txn_id;
        c.put(&ctx, "a", vec![0.0, 1.0, 0.0], PayloadRow::new(), None).unwrap();

        // Force reverses the original insert anyway — since the row's
        // *current* value is what an insert's compensating delete removes
        // regardless of which transaction most recently set it, forcing
        // this undo deletes row "a" entirely.
        c.undo(&ctx, insert_txn, true).unwrap();
        assert!(c.get_by_key("a").is_none());
    }

    #[test]
    fn history_is_most_recent_first_and_shows_undone_by() {
        let dir = tempfile::tempdir().unwrap();
        let c = open(dir.path());
        let ctx = ctx();
        c.put(&ctx, "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        c.put(&ctx, "b", vec![0.0, 1.0, 0.0], PayloadRow::new(), None).unwrap();
        let b_txn = c.history(10)[0].txn_id;
        c.undo(&ctx, b_txn, false).unwrap();

        let history = c.history(10);
        assert_eq!(history.len(), 3, "insert a, insert b, undo of b");
        // The newest entry is the undo transaction itself — a different
        // txn id from the one it reversed.
        assert_ne!(history[0].txn_id, b_txn, "most recent entry must be the new undo txn, not the target");
        let b_entry = history.iter().find(|e| e.txn_id == b_txn).unwrap();
        assert_eq!(b_entry.undone_by, Some(history[0].txn_id));
    }

    #[test]
    fn undo_survives_a_restart_before_and_after() {
        let dir = tempfile::tempdir().unwrap();
        let target = {
            let c = open(dir.path());
            let ctx = ctx();
            c.put(&ctx, "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
            c.history(10)[0].txn_id
        };

        // Reopen (simulating a restart) before undoing.
        let c2 = open(dir.path());
        assert!(c2.get_by_key("a").is_some());
        c2.undo(&ctx(), target, false).unwrap();
        assert!(c2.get_by_key("a").is_none());
        drop(c2);

        // Reopen again (simulating a second restart) — the undo itself
        // must have survived too.
        let c3 = open(dir.path());
        assert!(c3.get_by_key("a").is_none());
        let entry = c3.history(10).iter().find(|e| e.txn_id == target).cloned();
        assert!(entry.unwrap().undone_by.is_some(), "undone_by must survive a restart, reconstructed from the WAL");
    }
}
