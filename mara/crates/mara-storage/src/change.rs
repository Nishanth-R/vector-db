//! The commit feed derived indexes (the vector index, BM25, replication)
//! subscribe to. Events are delivered batched per transaction — a
//! 400-chunk `replace_document` costs subscribers one delta-buffer append
//! and one bitmap update, not 400 of each — and carry `Arc`-shared data so
//! fan-out to multiple subscribers is a cheap clone, not a copy.

use mara_proto::{DocId, ExtraPayload, PayloadRow, RowId, TxnId};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum ChangeEvent {
    Insert {
        row_id: RowId,
        key: Arc<str>,
        vector: Arc<[f32]>,
        fields: Arc<PayloadRow>,
        extra: Option<Arc<ExtraPayload>>,
        doc_id: Option<DocId>,
        /// The row's source text, if any — `None` for a plain row never
        /// inserted through the document API. BM25's subscriber is the
        /// reason this is here: it has no other way to learn what to
        /// tokenize without a redundant per-row storage lookup.
        text: Option<Arc<str>>,
    },
    Update {
        row_id: RowId,
        key: Arc<str>,
        vector: Arc<[f32]>,
        fields: Arc<PayloadRow>,
        extra: Option<Arc<ExtraPayload>>,
        doc_id: Option<DocId>,
        text: Option<Arc<str>>,
    },
    Delete {
        row_id: RowId,
        doc_id: Option<DocId>,
        /// The text the deleted row was indexed under — a BM25
        /// subscriber's `remove` must tokenize the *same* text `insert`
        /// did to be an exact inverse (see `mara_index_bm25::Bm25Index`),
        /// and by the time a delete reaches subscribers the row is
        /// already gone from storage, so there's no other way to recover
        /// it here.
        text: Option<Arc<str>>,
    },
}

impl ChangeEvent {
    pub fn row_id(&self) -> RowId {
        match self {
            ChangeEvent::Insert { row_id, .. }
            | ChangeEvent::Update { row_id, .. }
            | ChangeEvent::Delete { row_id, .. } => *row_id,
        }
    }

    pub fn doc_id(&self) -> Option<DocId> {
        match self {
            ChangeEvent::Insert { doc_id, .. }
            | ChangeEvent::Update { doc_id, .. }
            | ChangeEvent::Delete { doc_id, .. } => *doc_id,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChangeBatch {
    pub txn_id: TxnId,
    pub coll: String,
    pub events: Vec<ChangeEvent>,
}

/// Implemented by the vector index, the BM25 index, and (later) the
/// replication stream. `on_change` runs synchronously, in-process, inside
/// the writer's commit path — subscribers that need to do real work off
/// that path (e.g. a background `LiveIndex` rebuild) queue it themselves
/// rather than blocking the caller.
pub trait ChangeSubscriber: Send + Sync {
    fn on_change(&self, batch: &ChangeBatch);
}
