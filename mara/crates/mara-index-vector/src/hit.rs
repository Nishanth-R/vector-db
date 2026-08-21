use mara_proto::{DistanceMetric, DocId, RowId};

/// One search result. `score` is **always** higher-is-better, in the
/// collection's metric convention — for `L2`, that means the raw distance
/// has already been negated; callers never see a "lower is better" value.
/// `exact` distinguishes a rerank against the true vector (`true`, always
/// the case for `FlatIndex`) from a raw PQ/LSH approximation (`false`,
/// only once approximate indexes exist).
#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    pub id: RowId,
    pub doc_id: Option<DocId>,
    pub chunk_ord: Option<u32>,
    pub score: f32,
    pub metric: DistanceMetric,
    pub exact: bool,
}
