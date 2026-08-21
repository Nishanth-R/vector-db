//! End-to-end crash recovery through `Collection::open`: a torn WAL tail —
//! whether a corrupt/truncated line or an incomplete multi-record document
//! transaction — must never surface as a half-applied document, and
//! everything durably committed before the tear must survive exactly.
//! This is the master plan's explicit build-step-8 acceptance test.

use mara_proto::{
    ChunkSpec, DistanceMetric, ModelFingerprint, PayloadRow, Principal, PrincipalId, RequestCtx, Role, SessionId, Source, TxnId,
};
use mara_storage::wal::{self, WalActor, WalOp, WalPayload, WalRecord, WAL_FORMAT_VERSION};
use mara_storage::{ChunkInput, Collection, FsyncPolicy, PayloadSchema, PutDocumentInput, WalWriter};
use std::fs::OpenOptions;
use std::io::Write as _;
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
        model_id: "test-model".into(),
        revision: None,
        dim: 3,
    }
}

fn open(dir: &Path) -> Collection {
    Collection::open("docs", 3, DistanceMetric::Cosine, PayloadSchema::empty(), false, dir, 128 * 1024 * 1024, FsyncPolicy::Always).unwrap()
}

#[test]
fn clean_reopen_reconstructs_rows_and_documents_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let (id_a, doc_id) = {
        let c = open(dir.path());
        let id_a = c.put(&ctx(), "a", vec![1.0, 2.0, 3.0], PayloadRow::new(), None).unwrap();
        let doc_id = c
            .put_document(
                &ctx(),
                PutDocumentInput {
                    doc_key: "handbook.md".into(),
                    chunks: vec![
                        ChunkInput { text: "chunk 0".into(), vector: vec![0.1, 0.2, 0.3] },
                        ChunkInput { text: "chunk 1".into(), vector: vec![0.4, 0.5, 0.6] },
                    ],
                    doc_payload: PayloadRow::new(),
                    chunk_spec: ChunkSpec::default(),
                    source: None,
                    embedding_model: model(),
                },
            )
            .unwrap();
        (id_a, doc_id)
    };

    let c2 = open(dir.path());
    assert_eq!(c2.active_count(), 3, "1 plain row + 2 chunks");

    let row_a = c2.get_by_key("a").unwrap();
    assert_eq!(row_a.id, id_a);
    assert_eq!(row_a.vector, Some(vec![1.0, 2.0, 3.0]));

    let entry = c2.get_document("handbook.md").unwrap();
    assert_eq!(entry.doc_id, doc_id);
    assert_eq!(entry.version, 1);
    assert_eq!(entry.chunk_count, 2);

    let chunks = c2.list_chunks(doc_id).unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].text.as_deref(), Some("chunk 0"));
    assert_eq!(chunks[1].text.as_deref(), Some("chunk 1"));
    assert_eq!(chunks[0].chunk_ord, Some(0));
    assert_eq!(chunks[1].chunk_ord, Some(1));
}

/// Directly appends a torn (incomplete) multi-record transaction to the WAL,
/// bypassing `Collection` entirely — simulating a process crash after
/// fsyncing only some of a `put_document`'s records.
fn append_torn_document_txn(coll_dir: &Path, doc_key: &str, op: WalOp, total_chunks: u32, chunks_actually_written: u32) {
    // Matches `Collection::open`'s layout: WAL segments live under
    // `coll_dir/wal/`, not `coll_dir` directly.
    let writer = WalWriter::open(coll_dir.join("wal"), 128 * 1024 * 1024, FsyncPolicy::Always).unwrap();
    let txn = TxnId::new();
    let mut records: Vec<WalRecord> = (0..chunks_actually_written)
        .map(|ord| WalRecord {
            v: WAL_FORMAT_VERSION,
            lsn: mara_proto::Lsn::ZERO,
            ts: chrono::Utc::now(),
            txn,
            actor: WalActor {
                id: PrincipalId("p".into()),
                name: "t".into(),
            },
            session: SessionId("s".into()),
            coll: "docs".into(),
            op,
            row_id: Some(mara_proto::RowId(1000 + ord as u64)),
            key: Some(format!("{doc_key}#{ord}")),
            doc_id: Some(mara_proto::DocId(999)),
            doc_key: Some(doc_key.to_string()),
            chunk_ord: Some(ord),
            payload: Some(WalPayload {
                vector_b64: Some(wal::encode_vector(&[0.1, 0.2, 0.3])),
                text: Some(format!("torn chunk {ord}")),
                fields: Some(PayloadRow::new()),
                extra: None,
            }),
            undo: None,
            caused_by: None,
            txn_seq: (ord + 1, total_chunks),
            params: None,
            checksum: String::new(),
        })
        .collect();
    writer.append_batch(&mut records).unwrap();
}

#[test]
fn crash_mid_document_transaction_discards_it_whole_but_keeps_prior_state() {
    let dir = tempfile::tempdir().unwrap();
    {
        let c = open(dir.path());
        c.put(&ctx(), "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        c.put_document(
            &ctx(),
            PutDocumentInput {
                doc_key: "first.md".into(),
                chunks: vec![ChunkInput { text: "t0".into(), vector: vec![0.1, 0.0, 0.0] }],
                doc_payload: PayloadRow::new(),
                chunk_spec: ChunkSpec::default(),
                source: None,
                embedding_model: model(),
            },
        )
        .unwrap();
    }
    // A 5-chunk document transaction that only got 3 of its 5 records
    // durably written before the (simulated) crash.
    append_torn_document_txn(dir.path(), "second.md", WalOp::Insert, 5, 3);

    let c2 = open(dir.path());
    assert_eq!(
        c2.active_count(),
        2,
        "1 plain row + 1 chunk from the complete document; the torn document must not appear at all, not even partially"
    );
    assert!(c2.get_document("second.md").is_none(), "a torn document transaction must not create a partial doc entry");
    assert!(c2.get_by_key("second.md#0").is_none(), "not even the first chunk of a torn multi-record txn may survive");
    assert!(c2.get_by_key("a").is_some(), "state committed before the tear must be untouched");
    assert!(c2.get_document("first.md").is_some());

    // The collection must also be fully usable after recovery — new writes
    // land correctly and don't collide with the discarded torn bytes.
    let new_id = c2
        .put(&ctx(), "second.md#0", vec![0.5, 0.5, 0.5], PayloadRow::new(), None)
        .unwrap();
    assert_eq!(c2.active_count(), 3);
    let c3 = open(dir.path());
    assert_eq!(c3.get_by_key("second.md#0").unwrap().id, new_id);
}

#[test]
fn crash_mid_line_is_recovered_identically_to_crash_mid_transaction() {
    let dir = tempfile::tempdir().unwrap();
    {
        let c = open(dir.path());
        c.put(&ctx(), "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
        c.put(&ctx(), "b", vec![0.0, 1.0, 0.0], PayloadRow::new(), None).unwrap();
    }
    // Simulate a crash mid-line: a truncated, non-JSON tail appended
    // directly to the segment file (the OS wrote some bytes of the next
    // record before the process died).
    let seg_path = dir.path().join("wal").join("segment-0000000000.wal");
    {
        let mut f = OpenOptions::new().append(true).open(&seg_path).unwrap();
        f.write_all(b"{\"v\":1,\"lsn\":\"garbage-torn").unwrap();
    }

    let c2 = open(dir.path());
    assert_eq!(c2.active_count(), 2, "both clean records must survive; the torn tail must be dropped silently and safely");
    assert!(c2.get_by_key("a").is_some());
    assert!(c2.get_by_key("b").is_some());

    // Resuming must not reintroduce or choke on the torn bytes.
    c2.put(&ctx(), "c", vec![0.0, 0.0, 1.0], PayloadRow::new(), None).unwrap();
    assert_eq!(c2.active_count(), 3);
}

#[test]
fn replace_document_survives_a_crash_between_the_delete_half_and_the_insert_half() {
    let dir = tempfile::tempdir().unwrap();
    {
        let c = open(dir.path());
        c.put_document(
            &ctx(),
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
    }
    // A replace_document's delete-old + insert-new is one txn of 1+1 = 2
    // records; simulate a crash after only the delete half landed.
    append_torn_document_txn(dir.path(), "doc.md", WalOp::Delete, 2, 1);

    let c2 = open(dir.path());
    // The torn replace must not apply at all — the document must still be
    // at its pre-replace state (v1), not deleted and not partially updated.
    let entry = c2.get_document("doc.md").expect("a torn replace must leave the document exactly as it was");
    assert_eq!(entry.version, 1);
    let chunks = c2.list_chunks(entry.doc_id).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text.as_deref(), Some("v1"));
}
