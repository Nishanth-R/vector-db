//! The document/chunk layer (master plan Layer 2, *Document/chunk model*).
//! A row is a chunk; chunks belong to a document, and the document is the
//! unit of every lifecycle operation — `put_document`/`replace_document`/
//! `delete_document` are each exactly one transaction, one `ChangeBatch`,
//! so readers never see a half-swapped mix of old and new chunks.

use crate::change::ChangeBatch;
use crate::collection::{Collection, RowRecord};
use crate::error::{StorageError, StorageResult};
use crate::wal::{self, WalActor, WalOp, WalPayload, WalRecord, WAL_FORMAT_VERSION};
use chrono::Utc;
use mara_proto::{ChunkSpec, DocId, Lsn, ModelFingerprint, PayloadRow, RequestCtx, Row, RowId, TxnId};
use roaring::RoaringBitmap;
use std::sync::Arc;

/// One already-chunked, already-embedded piece of a document. Chunking
/// (`mara-chunker`) and embedding (`mara-embed`) both run upstream of
/// storage, per the master plan, so by the time a document reaches here
/// there's no text-to-vector work left to do — only text is kept (for BM25
/// and future re-embedding), alongside the vector that was already
/// computed from it.
#[derive(Clone, Debug)]
pub struct ChunkInput {
    pub text: String,
    pub vector: Vec<f32>,
}

pub struct PutDocumentInput {
    pub doc_key: String,
    pub chunks: Vec<ChunkInput>,
    pub doc_payload: PayloadRow,
    pub chunk_spec: ChunkSpec,
    pub source: Option<String>,
    pub embedding_model: ModelFingerprint,
}

/// The doc registry's row. `rows` is the live chunk `RowId`s (a bitmap
/// rather than a `Vec` because most of what's done with it — membership
/// test, cascade delete, `filter: doc_id IN [...]` — is a bitmap op
/// elsewhere in the design too).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DocEntry {
    pub doc_id: DocId,
    pub doc_key: String,
    pub version: u32,
    pub rows: RoaringBitmap,
    pub chunk_count: u32,
    pub source: Option<String>,
    pub chunk_spec: ChunkSpec,
    pub chunker_version: u32,
    pub embedding_model: ModelFingerprint,
    pub created_txn: TxnId,
    pub updated_txn: TxnId,
    pub doc_payload: PayloadRow,
}

/// A chunk's external key: `{doc_key}#{chunk_ord}`, matching the master
/// plan's WAL example (`"onboarding.md#12"`).
fn chunk_key(doc_key: &str, chunk_ord: u32) -> String {
    format!("{doc_key}#{chunk_ord}")
}

/// `DocEntry`'s metadata, projected into a WAL record's `params` field.
/// Written redundantly on every chunk record of a `put_document`/
/// `replace_document` transaction (simple to apply during replay — no
/// "which record carries it" bookkeeping — at the cost of a little
/// per-chunk WAL bytes, acceptable at this project's stated scale). Row
/// data itself (vector/fields/text) already carries `doc_id`/`chunk_ord`
/// on every record; `doc_payload` is recoverable from any chunk's own
/// `fields`, since every chunk's fields *are* the doc payload.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct WalDocMeta {
    pub(crate) version: u32,
    pub(crate) source: Option<String>,
    pub(crate) chunk_spec: ChunkSpec,
    pub(crate) chunker_version: u32,
    pub(crate) embedding_model: ModelFingerprint,
}

impl WalDocMeta {
    pub(crate) fn to_params(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("WalDocMeta always serializes")
    }
}

impl Collection {
    /// Inserts a brand-new document: allocates a `DocId`, inserts every
    /// chunk under **one** `txn_id` with **one** `ChangeBatch`. Either the
    /// whole document lands or none of it does.
    pub fn put_document(&self, ctx: &RequestCtx, input: PutDocumentInput) -> StorageResult<DocId> {
        if input.chunks.is_empty() {
            return Err(StorageError::InvalidArgument(
                "put_document requires at least one chunk".into(),
            ));
        }
        for chunk in &input.chunks {
            self.check_dim(&chunk.vector)?;
        }

        let txn_id = TxnId::new();
        let mut guard = self.inner.write();
        if guard.doc_key_to_id.contains_key(input.doc_key.as_str()) {
            return Err(StorageError::InvalidArgument(format!(
                "document {:?} already exists; use replace_document",
                input.doc_key
            )));
        }

        let doc_id = DocId(guard.next_doc_id);
        guard.next_doc_id += 1;

        let prepared = insert_chunks(&mut guard.next_row_id, doc_id, &input.doc_key, &input.chunks, &input.doc_payload);
        // Validate every chunk's fields against the schema before deciding
        // or mutating anything — same reasoning as put_batch: a mid-document
        // failure must never leave some chunks indexed and others not.
        for (_, record) in &prepared {
            guard.payload.validate_row(&record.fields)?;
        }

        let n = prepared.len() as u32;
        let params = WalDocMeta {
            version: 1,
            source: input.source.clone(),
            chunk_spec: input.chunk_spec.clone(),
            chunker_version: 1,
            embedding_model: input.embedding_model.clone(),
        }
        .to_params();
        let mut wal_records: Vec<WalRecord> = prepared
            .iter()
            .enumerate()
            .map(|(idx, (row_id, record))| WalRecord {
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
                op: WalOp::Insert,
                row_id: Some(*row_id),
                key: Some(record.key.to_string()),
                doc_id: Some(doc_id),
                doc_key: Some(input.doc_key.clone()),
                chunk_ord: record.chunk_ord,
                payload: Some(WalPayload {
                    vector_b64: Some(wal::encode_vector(&record.vector)),
                    text: Some(input.chunks[idx].text.clone()),
                    fields: Some((*record.fields).clone()),
                    extra: None,
                }),
                undo: None,
                caused_by: None,
                txn_seq: (idx as u32 + 1, n),
                params: Some(params.clone()),
                checksum: String::new(),
            })
            .collect();
        self.append_wal(&mut wal_records)?;

        let events = crate::recovery::change_events_from_records(&wal_records);
        crate::recovery::apply_replayed_records(&mut guard, &wal_records).expect("just-appended, well-formed records must apply cleanly");
        drop(guard);

        self.notify(ChangeBatch {
            txn_id,
            coll: self.name.clone(),
            events,
        });
        Ok(doc_id)
    }

    /// Replaces a document's chunks wholesale, in one transaction: every
    /// row in the old `rows` bitmap is tombstoned, the new chunks are
    /// inserted, `version` bumps, and both derived indexes see one
    /// coherent `ChangeBatch` — readers see version N or N+1, never a
    /// half-swapped mix. The `DocId` and `doc_key` are unchanged; only the
    /// chunk content and `version` move.
    #[allow(clippy::too_many_arguments)]
    pub fn replace_document(
        &self,
        ctx: &RequestCtx,
        doc_key: &str,
        chunks: Vec<ChunkInput>,
        doc_payload: PayloadRow,
        chunk_spec: ChunkSpec,
        source: Option<String>,
        embedding_model: ModelFingerprint,
    ) -> StorageResult<DocId> {
        if chunks.is_empty() {
            return Err(StorageError::InvalidArgument(
                "replace_document requires at least one chunk".into(),
            ));
        }
        for chunk in &chunks {
            self.check_dim(&chunk.vector)?;
        }

        let txn_id = TxnId::new();
        let mut guard = self.inner.write();
        let doc_id = *guard
            .doc_key_to_id
            .get(doc_key)
            .ok_or_else(|| StorageError::DocumentNotFound(doc_key.to_string()))?;

        let old_entry = guard.docs.get(&doc_id).expect("doc registry out of sync with doc_key_to_id").clone();

        let prepared = insert_chunks(&mut guard.next_row_id, doc_id, doc_key, &chunks, &doc_payload);
        for (_, record) in &prepared {
            guard.payload.validate_row(&record.fields)?;
        }

        // Before-images for every old row, read out now (before any
        // mutation) so the WAL's `undo` field can carry them.
        let old_rows: Vec<_> = old_entry
            .rows
            .iter()
            .filter_map(|idx| {
                let row_id = RowId::from_bitmap_index(idx);
                guard.rows.get(&row_id).map(|r| (row_id, r.key.clone(), r.vector.clone(), r.fields.clone(), r.extra.clone(), r.text.clone(), r.chunk_ord))
            })
            .collect();

        let new_version = old_entry.version + 1;
        let n = (old_rows.len() + prepared.len()) as u32;
        let params = WalDocMeta {
            version: new_version,
            source: source.clone(),
            chunk_spec: chunk_spec.clone(),
            chunker_version: old_entry.chunker_version,
            embedding_model: embedding_model.clone(),
        }
        .to_params();
        // The delete-half records carry the *old* (pre-replace) doc
        // metadata rather than `params` above. Forward replay never reads
        // `params` off a `Delete` record (only `Insert`/`Update` touch doc
        // metadata — see `recovery::apply_upsert`), so this is inert for
        // normal replay; it exists so `undo` can recover the exact
        // pre-replace `DocEntry` state from the original transaction's own
        // WAL records, with no separate metadata history to maintain.
        let old_params = WalDocMeta {
            version: old_entry.version,
            source: old_entry.source.clone(),
            chunk_spec: old_entry.chunk_spec.clone(),
            chunker_version: old_entry.chunker_version,
            embedding_model: old_entry.embedding_model.clone(),
        }
        .to_params();

        let mut wal_records: Vec<WalRecord> = Vec::with_capacity(n as usize);
        let mut seq = 0u32;
        for (row_id, key_arc, vector, fields, extra, text, chunk_ord) in &old_rows {
            seq += 1;
            wal_records.push(WalRecord {
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
                row_id: Some(*row_id),
                key: Some(key_arc.to_string()),
                doc_id: Some(doc_id),
                doc_key: Some(doc_key.to_string()),
                chunk_ord: *chunk_ord,
                payload: None,
                undo: Some(WalPayload {
                    vector_b64: Some(wal::encode_vector(vector)),
                    text: text.as_ref().map(|t| t.to_string()),
                    fields: Some((**fields).clone()),
                    extra: extra.as_ref().map(|e| (**e).clone()),
                }),
                caused_by: None,
                txn_seq: (seq, n),
                params: Some(old_params.clone()),
                checksum: String::new(),
            });
        }
        for (idx, (row_id, record)) in prepared.iter().enumerate() {
            seq += 1;
            wal_records.push(WalRecord {
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
                op: WalOp::Insert,
                row_id: Some(*row_id),
                key: Some(record.key.to_string()),
                doc_id: Some(doc_id),
                doc_key: Some(doc_key.to_string()),
                chunk_ord: record.chunk_ord,
                payload: Some(WalPayload {
                    vector_b64: Some(wal::encode_vector(&record.vector)),
                    text: Some(chunks[idx].text.clone()),
                    fields: Some((*record.fields).clone()),
                    extra: None,
                }),
                undo: None,
                caused_by: None,
                txn_seq: (seq, n),
                params: Some(params.clone()),
                checksum: String::new(),
            });
        }
        self.append_wal(&mut wal_records)?;

        let events = crate::recovery::change_events_from_records(&wal_records);
        crate::recovery::apply_replayed_records(&mut guard, &wal_records).expect("just-appended, well-formed records must apply cleanly");
        drop(guard);

        self.notify(ChangeBatch {
            txn_id,
            coll: self.name.clone(),
            events,
        });
        Ok(doc_id)
    }

    /// One transaction, cascading to every chunk.
    pub fn delete_document(&self, ctx: &RequestCtx, doc_key: &str) -> StorageResult<()> {
        let txn_id = TxnId::new();
        let mut guard = self.inner.write();
        let doc_id = *guard
            .doc_key_to_id
            .get(doc_key)
            .ok_or_else(|| StorageError::DocumentNotFound(doc_key.to_string()))?;
        let entry = guard.docs.get(&doc_id).expect("doc registry out of sync with doc_key_to_id").clone();

        let old_rows: Vec<_> = entry
            .rows
            .iter()
            .filter_map(|idx| {
                let row_id = RowId::from_bitmap_index(idx);
                guard.rows.get(&row_id).map(|r| (row_id, r.key.clone(), r.vector.clone(), r.fields.clone(), r.extra.clone(), r.text.clone(), r.chunk_ord))
            })
            .collect();
        let n = old_rows.len() as u32;
        // Carried purely for `undo`'s benefit — see the identical note in
        // `replace_document` on why a `Delete` record's `params` is inert
        // for forward replay but is exactly what a compensating re-insert
        // needs to restore the pre-delete `DocEntry`.
        let params = WalDocMeta {
            version: entry.version,
            source: entry.source.clone(),
            chunk_spec: entry.chunk_spec.clone(),
            chunker_version: entry.chunker_version,
            embedding_model: entry.embedding_model.clone(),
        }
        .to_params();

        let mut wal_records: Vec<WalRecord> = old_rows
            .iter()
            .enumerate()
            .map(|(i, (row_id, key_arc, vector, fields, extra, text, chunk_ord))| WalRecord {
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
                row_id: Some(*row_id),
                key: Some(key_arc.to_string()),
                doc_id: Some(doc_id),
                doc_key: Some(doc_key.to_string()),
                chunk_ord: *chunk_ord,
                payload: None,
                undo: Some(WalPayload {
                    vector_b64: Some(wal::encode_vector(vector)),
                    text: text.as_ref().map(|t| t.to_string()),
                    fields: Some((**fields).clone()),
                    extra: extra.as_ref().map(|e| (**e).clone()),
                }),
                caused_by: None,
                txn_seq: (i as u32 + 1, n),
                params: Some(params.clone()),
                checksum: String::new(),
            })
            .collect();
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

    pub fn get_document(&self, doc_key: &str) -> Option<DocEntry> {
        let guard = self.inner.read();
        let doc_id = guard.doc_key_to_id.get(doc_key)?;
        guard.docs.get(doc_id).cloned()
    }

    /// A document's live chunks, ordered by `chunk_ord`.
    pub fn list_chunks(&self, doc_id: DocId) -> StorageResult<Vec<Row>> {
        let guard = self.inner.read();
        let entry = guard
            .docs
            .get(&doc_id)
            .ok_or_else(|| StorageError::DocumentNotFound(doc_id.to_string()))?;
        let mut rows: Vec<Row> = entry
            .rows
            .iter()
            .filter_map(|idx| {
                let row_id = RowId::from_bitmap_index(idx);
                guard.rows.get(&row_id).map(|rec| Collection::to_row(row_id, rec))
            })
            .collect();
        rows.sort_by_key(|r| r.chunk_ord.unwrap_or(u32::MAX));
        Ok(rows)
    }

    /// All documents currently registered, ordered by `doc_id`. Unfiltered
    /// for now — `Filter`/`FilterMask` support arrives with the payload
    /// store and filter compiler.
    pub fn list_documents(&self) -> Vec<DocEntry> {
        let guard = self.inner.read();
        let mut docs: Vec<DocEntry> = guard.docs.values().cloned().collect();
        docs.sort_by_key(|d| d.doc_id);
        docs
    }
}

/// Builds `(RowId, RowRecord)` for each chunk — pure data, no
/// `CollectionInner` access. Callers only read these to build WAL records;
/// the actual in-memory mutation happens uniformly afterward, via
/// `recovery::apply_replayed_records` on the WAL records those produced
/// (real `Lsn`s attached), so there's exactly one code path that turns a
/// decided write into `CollectionInner` state — live or replayed.
fn insert_chunks(next_row_id: &mut u64, doc_id: DocId, doc_key: &str, chunks: &[ChunkInput], doc_payload: &PayloadRow) -> Vec<(RowId, RowRecord)> {
    chunks
        .iter()
        .enumerate()
        .map(|(ord, chunk)| {
            let chunk_ord = ord as u32;
            let key_arc: Arc<str> = Arc::from(chunk_key(doc_key, chunk_ord).as_str());
            let fields = Arc::new(doc_payload.clone());
            let text: Arc<str> = Arc::from(chunk.text.as_str());
            let vector: Arc<[f32]> = chunk.vector.clone().into();

            let row_id = RowId(*next_row_id);
            *next_row_id += 1;

            let record = RowRecord {
                key: key_arc,
                vector,
                fields,
                extra: None,
                text: Some(text),
                doc_id: Some(doc_id),
                chunk_ord: Some(chunk_ord),
            };
            (row_id, record)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mara_proto::{DistanceMetric, Principal, Role, SessionId, Source};

    fn ctx() -> RequestCtx {
        RequestCtx::new(
            SessionId("s".into()),
            Principal {
                id: mara_proto::PrincipalId("p".into()),
                name: "t".into(),
                role: Role::Writer,
            },
            Source::Embedded,
        )
    }

    fn model() -> ModelFingerprint {
        ModelFingerprint {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".into(),
            revision: None,
            dim: 3,
        }
    }

    fn chunks(n: usize) -> Vec<ChunkInput> {
        (0..n)
            .map(|i| ChunkInput {
                text: format!("chunk {i}"),
                vector: vec![i as f32, 0.0, 0.0],
            })
            .collect()
    }

    fn put_input(doc_key: &str, n: usize) -> PutDocumentInput {
        PutDocumentInput {
            doc_key: doc_key.to_string(),
            chunks: chunks(n),
            doc_payload: PayloadRow::new(),
            chunk_spec: ChunkSpec::default(),
            source: Some("handbook.md".into()),
            embedding_model: model(),
        }
    }

    #[test]
    fn put_document_inserts_every_chunk_under_one_doc() {
        let c = Collection::new("docs", 3, DistanceMetric::Cosine);
        let doc_id = c.put_document(&ctx(), put_input("onboarding.md", 5)).unwrap();

        assert_eq!(c.active_count(), 5);
        let entry = c.get_document("onboarding.md").unwrap();
        assert_eq!(entry.doc_id, doc_id);
        assert_eq!(entry.version, 1);
        assert_eq!(entry.chunk_count, 5);

        let chunks = c.list_chunks(doc_id).unwrap();
        assert_eq!(chunks.len(), 5);
        for (i, row) in chunks.iter().enumerate() {
            assert_eq!(row.chunk_ord, Some(i as u32));
            assert_eq!(row.doc_id, Some(doc_id));
            assert_eq!(row.key, format!("onboarding.md#{i}"));
        }
    }

    #[test]
    fn put_document_twice_with_same_key_is_rejected() {
        let c = Collection::new("docs", 3, DistanceMetric::Cosine);
        c.put_document(&ctx(), put_input("onboarding.md", 2)).unwrap();
        let err = c.put_document(&ctx(), put_input("onboarding.md", 2)).unwrap_err();
        assert!(matches!(err, StorageError::InvalidArgument(_)));
    }

    #[test]
    fn empty_document_is_rejected() {
        let c = Collection::new("docs", 3, DistanceMetric::Cosine);
        let mut input = put_input("empty.md", 0);
        input.chunks = vec![];
        assert!(c.put_document(&ctx(), input).is_err());
    }

    #[test]
    fn replace_document_swaps_every_chunk_and_bumps_version() {
        let c = Collection::new("docs", 3, DistanceMetric::Cosine);
        let doc_id = c.put_document(&ctx(), put_input("onboarding.md", 3)).unwrap();
        assert_eq!(c.active_count(), 3);

        let new_id = c
            .replace_document(
                &ctx(),
                "onboarding.md",
                chunks(5),
                PayloadRow::new(),
                ChunkSpec::default(),
                Some("handbook.md".into()),
                model(),
            )
            .unwrap();

        assert_eq!(new_id, doc_id, "replace must keep the same DocId");
        assert_eq!(c.active_count(), 5, "old chunks must be fully tombstoned, new ones fully live");
        let entry = c.get_document("onboarding.md").unwrap();
        assert_eq!(entry.version, 2);
        assert_eq!(entry.chunk_count, 5);

        // The old chunk keys (0..3) point at fresh RowIds now, not the old
        // ones — no accidental resurrection of the pre-replace rows.
        let chunks_after = c.list_chunks(doc_id).unwrap();
        assert_eq!(chunks_after.len(), 5);
        assert!(chunks_after.iter().all(|r| r.doc_id == Some(doc_id)));
    }

    #[test]
    fn replace_document_never_shows_a_torn_state_via_row_ids() {
        // Regression guard for the atomicity claim: every surviving row
        // after a replace belongs to the *new* generation, and old row ids
        // are gone from both the key index and the live set together.
        let c = Collection::new("docs", 3, DistanceMetric::Cosine);
        c.put_document(&ctx(), put_input("onboarding.md", 4)).unwrap();
        let old_chunks = c.get_document("onboarding.md").unwrap().rows;

        c.replace_document(
            &ctx(),
            "onboarding.md",
            chunks(4),
            PayloadRow::new(),
            ChunkSpec::default(),
            None,
            model(),
        )
        .unwrap();

        let new_entry = c.get_document("onboarding.md").unwrap();
        for idx in old_chunks.iter() {
            let old_row_id = RowId::from_bitmap_index(idx);
            assert!(
                !new_entry.rows.contains(idx),
                "a pre-replace RowId must not appear in the post-replace document"
            );
            assert!(
                c.fetch_vectors(&[old_row_id])[0].is_some(),
                "the old row's data is retained (no compaction), just no longer live/attached"
            );
        }
    }

    #[test]
    fn replace_of_unknown_document_errors() {
        let c = Collection::new("docs", 3, DistanceMetric::Cosine);
        let err = c
            .replace_document(&ctx(), "missing.md", chunks(1), PayloadRow::new(), ChunkSpec::default(), None, model())
            .unwrap_err();
        assert_eq!(err, StorageError::DocumentNotFound("missing.md".into()));
    }

    #[test]
    fn delete_document_cascades_to_every_chunk() {
        let c = Collection::new("docs", 3, DistanceMetric::Cosine);
        c.put_document(&ctx(), put_input("onboarding.md", 4)).unwrap();
        assert_eq!(c.active_count(), 4);

        c.delete_document(&ctx(), "onboarding.md").unwrap();
        assert_eq!(c.active_count(), 0);
        assert!(c.get_document("onboarding.md").is_none());
        assert!(c.get_by_key("onboarding.md#0").is_none());
    }

    #[test]
    fn delete_of_unknown_document_errors() {
        let c = Collection::new("docs", 3, DistanceMetric::Cosine);
        assert_eq!(
            c.delete_document(&ctx(), "missing.md").unwrap_err(),
            StorageError::DocumentNotFound("missing.md".into())
        );
    }

    #[test]
    fn list_documents_is_ordered_by_doc_id() {
        let c = Collection::new("docs", 3, DistanceMetric::Cosine);
        c.put_document(&ctx(), put_input("b.md", 1)).unwrap();
        c.put_document(&ctx(), put_input("a.md", 1)).unwrap();
        let docs = c.list_documents();
        assert_eq!(docs.len(), 2);
        assert!(docs[0].doc_id < docs[1].doc_id);
        assert_eq!(docs[0].doc_key, "b.md");
    }

    #[test]
    fn doc_payload_is_inherited_by_every_chunk() {
        let c = Collection::new("docs", 3, DistanceMetric::Cosine);
        let mut input = put_input("onboarding.md", 2);
        input
            .doc_payload
            .insert("title".to_string(), mara_proto::PayloadValue::Keyword("Onboarding".into()));
        let doc_id = c.put_document(&ctx(), input).unwrap();

        for row in c.list_chunks(doc_id).unwrap() {
            assert_eq!(
                row.fields.get("title"),
                Some(&mara_proto::PayloadValue::Keyword("Onboarding".into()))
            );
            assert!(row.text.is_some(), "chunk text is stored in its own slot, not the payload fields");
        }
    }

    struct RecordingSubscriber {
        batches: parking_lot::Mutex<Vec<ChangeBatch>>,
    }

    impl crate::change::ChangeSubscriber for RecordingSubscriber {
        fn on_change(&self, batch: &ChangeBatch) {
            self.batches.lock().push(batch.clone());
        }
    }

    #[test]
    fn replace_document_delivers_one_coherent_batch() {
        let c = Collection::new("docs", 3, DistanceMetric::Cosine);
        c.put_document(&ctx(), put_input("onboarding.md", 3)).unwrap();

        let sub = Arc::new(RecordingSubscriber {
            batches: parking_lot::Mutex::new(Vec::new()),
        });
        c.subscribe(sub.clone());

        c.replace_document(&ctx(), "onboarding.md", chunks(6), PayloadRow::new(), ChunkSpec::default(), None, model())
            .unwrap();

        let batches = sub.batches.lock();
        assert_eq!(batches.len(), 1, "replace_document must deliver exactly one ChangeBatch");
        assert_eq!(batches[0].events.len(), 3 + 6, "3 deletes + 6 inserts in one batch");
    }
}
