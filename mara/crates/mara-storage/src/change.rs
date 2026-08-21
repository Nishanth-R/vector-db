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
    },
    Update {
        row_id: RowId,
        key: Arc<str>,
        vector: Arc<[f32]>,
        fields: Arc<PayloadRow>,
        extra: Option<Arc<ExtraPayload>>,
        doc_id: Option<DocId>,
    },
    Delete {
        row_id: RowId,
        doc_id: Option<DocId>,
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
