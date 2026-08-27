//! `max_chunks_per_doc` grouping, shared by every `VectorIndex`
//! implementation (`Flat`, `Ivf`, `IvfPq`) rather than copy-pasted per
//! index — the policy is index-agnostic: cap how many hits from the same
//! document survive, in score order, once results are already ranked.

use mara_proto::DocId;
use std::collections::HashMap;

pub(crate) trait HasDocId {
    fn doc_id(&self) -> Option<DocId>;
}

impl HasDocId for mara_proto::Row {
    fn doc_id(&self) -> Option<DocId> {
        self.doc_id
    }
}

impl HasDocId for &mara_proto::Row {
    fn doc_id(&self) -> Option<DocId> {
        self.doc_id
    }
}

impl HasDocId for crate::hit::SearchHit {
    fn doc_id(&self) -> Option<DocId> {
        self.doc_id
    }
}

/// Without this, a single verbose document reliably monopolizes the
/// top-k. `None` disables grouping entirely.
pub(crate) fn apply_doc_grouping<T: HasDocId>(mut ranked: Vec<(f32, T)>, k: usize, max_chunks_per_doc: Option<u32>) -> Vec<(f32, T)> {
    if let Some(max) = max_chunks_per_doc {
        let mut per_doc: HashMap<DocId, u32> = HashMap::new();
        ranked.retain(|(_, row)| match row.doc_id() {
            Some(doc_id) => {
                let count = per_doc.entry(doc_id).or_insert(0);
                let keep = *count < max;
                if keep {
                    *count += 1;
                }
                keep
            }
            None => true,
        });
    }
    ranked.truncate(k);
    ranked
}
