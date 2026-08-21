//! Snapshot/checkpoint + startup reconciliation (master plan build step 9):
//! a checkpoint must let recovery skip straight to reconstructing from the
//! snapshot plus only the WAL tail written after it, a corrupt newest
//! snapshot must fall back to an older valid generation rather than
//! failing recovery outright, and undo/history must keep working
//! correctly across a checkpoint (their state — `TxnIndex` — round-trips
//! through the snapshot too, not just row/document data).

use mara_proto::{ChunkSpec, DistanceMetric, ModelFingerprint, PayloadRow, Principal, PrincipalId, RequestCtx, Role, SessionId, Source};
use mara_storage::{ChunkInput, Collection, FsyncPolicy, PayloadSchema, PutDocumentInput};
use std::fs;
use std::path::Path;

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

fn model() -> ModelFingerprint {
    ModelFingerprint {
        model_id: "m".into(),
        revision: None,
        dim: 3,
    }
}

fn open(dir: &Path) -> Collection {
    Collection::open("docs", 3, DistanceMetric::Cosine, PayloadSchema::empty(), false, dir, 128 * 1024 * 1024, FsyncPolicy::Always).unwrap()
}

#[test]
fn checkpoint_then_reopen_reconstructs_state_from_the_snapshot_alone() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = ctx();
    let doc_id = {
        let c = open(dir.path());
        c.put(&ctx, "a", vec![1.0, 2.0, 3.0], PayloadRow::new(), None).unwrap();
        let doc_id = c
            .put_document(
                &ctx,
                PutDocumentInput {
                    doc_key: "handbook.md".into(),
                    chunks: vec![ChunkInput { text: "c0".into(), vector: vec![0.1, 0.2, 0.3] }],
                    doc_payload: PayloadRow::new(),
                    chunk_spec: ChunkSpec::default(),
                    source: None,
                    embedding_model: model(),
                },
            )
            .unwrap();
        c.checkpoint().unwrap();
        doc_id
    };

    // Every WAL segment gone (simulating "the snapshot alone must be
    // enough") proves reconstruction didn't secretly depend on replay.
    let wal_dir = dir.path().join("wal");
    for entry in fs::read_dir(&wal_dir).unwrap() {
        fs::remove_file(entry.unwrap().path()).unwrap();
    }

    let c2 = open(dir.path());
    assert_eq!(c2.active_count(), 2);
    assert_eq!(c2.get_by_key("a").unwrap().vector, Some(vec![1.0, 2.0, 3.0]));
    let entry = c2.get_document("handbook.md").unwrap();
    assert_eq!(entry.doc_id, doc_id);
    assert_eq!(c2.list_chunks(doc_id).unwrap().len(), 1);
}

#[test]
fn writes_after_a_checkpoint_still_replay_correctly_from_the_wal_tail() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = ctx();
    {
        let c = open(dir.path());
        c.put(&ctx, "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        c.checkpoint().unwrap();
        // Written *after* the checkpoint — must come from WAL replay, not
        // the (older) snapshot.
        c.put(&ctx, "b", vec![0.0, 1.0, 0.0], PayloadRow::new(), None).unwrap();
    }

    let c2 = open(dir.path());
    assert_eq!(c2.active_count(), 2);
    assert!(c2.get_by_key("a").is_some());
    assert!(c2.get_by_key("b").is_some(), "post-checkpoint write must survive via WAL-tail replay");
}

#[test]
fn a_corrupt_newest_snapshot_falls_back_to_an_older_valid_generation() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = ctx();
    {
        let c = open(dir.path());
        c.put(&ctx, "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        c.checkpoint().unwrap(); // generation 0, has "a"
        c.put(&ctx, "b", vec![0.0, 1.0, 0.0], PayloadRow::new(), None).unwrap();
        c.checkpoint().unwrap(); // generation 1, has "a" and "b"
    }

    // Corrupt the newest snapshot file directly.
    let mut entries: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("snapshot-") && e.file_name().to_string_lossy().ends_with(".mdb"))
        .collect();
    entries.sort_by_key(|e| e.file_name());
    let newest = entries.last().unwrap().path();
    fs::write(&newest, b"not a valid snapshot at all").unwrap();

    // Recovery must fall back to generation 0 rather than fail outright —
    // WAL replay on top fills in "b" from the WAL tail after gen 0's LSN.
    let c2 = open(dir.path());
    assert!(c2.get_by_key("a").is_some());
    assert!(c2.get_by_key("b").is_some(), "the WAL tail after the older valid snapshot must still supply what the corrupt newest one would have");
}

#[test]
fn only_three_snapshot_generations_are_retained() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = ctx();
    let c = open(dir.path());
    for i in 0..5 {
        c.put(&ctx, &format!("k{i}"), vec![i as f32, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        c.checkpoint().unwrap();
    }
    let count = fs::read_dir(dir.path())
        .unwrap()
        .filter(|e| {
            let name = e.as_ref().unwrap().file_name();
            let name = name.to_string_lossy();
            name.starts_with("snapshot-") && name.ends_with(".mdb")
        })
        .count();
    assert_eq!(count, 3, "only the 3 most recent checkpoint generations should survive");
}

#[test]
fn undo_history_survives_a_checkpoint_and_still_detects_conflicts() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = ctx();
    let insert_txn = {
        let c = open(dir.path());
        c.put(&ctx, "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        let txn = c.history(10)[0].txn_id;
        c.checkpoint().unwrap();
        txn
    };

    let c2 = open(dir.path());
    // history must show the pre-checkpoint transaction, reconstructed from
    // the snapshot's own txn_entries, not just from WAL replay.
    assert!(c2.history(10).iter().any(|e| e.txn_id == insert_txn));

    c2.undo(&ctx, insert_txn, false).unwrap();
    assert!(c2.get_by_key("a").is_none());

    // Conflict detection must also still work post-checkpoint: undo the
    // same txn again should now be rejected as already-undone.
    let err = c2.undo(&ctx, insert_txn, false).unwrap_err();
    assert!(matches!(err, mara_storage::StorageError::InvalidArgument(_)));
}
