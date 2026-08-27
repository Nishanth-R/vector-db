use crate::error::IndexResult;
use crate::hit::SearchResult;
use crate::params::SearchParams;
use mara_storage::{FilterMask, StorageApi};

/// Implemented uniformly by `Flat`, `Ivf`/`IvfPq`, and later `Lsh`. Takes
/// `StorageApi` directly (reconciled from two independently-drafted
/// interfaces into one, per the master plan) rather than a parallel
/// `VectorSource` trait — storage is the single source of truth, and the
/// index layer has no reason to abstract over it further.
pub trait VectorIndex: Send + Sync {
    fn search(
        &self,
        storage: &dyn StorageApi,
        coll: &str,
        query: &[f32],
        k: usize,
        filter: Option<&FilterMask>,
        params: &SearchParams,
    ) -> IndexResult<SearchResult>;
}
