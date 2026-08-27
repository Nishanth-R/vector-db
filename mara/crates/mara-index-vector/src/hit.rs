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

/// A `VectorIndex::search` call's full outcome (master plan Layer 4,
/// *Filtered search*): `hits`, plus whether the filtered-ANN/plain-ANN
/// escalation hit its `filter_nprobe_max`/`nlist` cap before reaching a
/// full `k` post-filter candidates. `truncated_by_filter` is the plan's
/// explicit alternative to a silent short return — a caller can tell the
/// difference between "there genuinely aren't `k` matching rows" and
/// "there might be, but the escalation budget ran out first."
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub hits: Vec<SearchHit>,
    pub truncated_by_filter: bool,
}
