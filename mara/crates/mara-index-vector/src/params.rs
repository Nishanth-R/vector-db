/// Search-time knobs shared by every `VectorIndex` implementation. Not
/// every field means something to every index — `nprobe`/`rerank_k` are
/// meaningless to `FlatIndex`, which is already exact and scans
/// everything, and `beam_width` only matters when a coarse quantizer is
/// actually hierarchical — but one shared struct beats a parallel params
/// type per index, since a caller switching indexes shouldn't have to
/// switch call shapes too.
#[derive(Clone, Debug)]
pub struct SearchParams {
    /// Caps how many chunks from the same document can appear in one
    /// result set — without this, a single verbose document reliably
    /// monopolizes the top-k. `None` disables grouping entirely.
    pub max_chunks_per_doc: Option<u32>,
    /// `IvfIndex`/`IvfPqIndex`: how many coarse leaf clusters to scan in
    /// total. Clamped to the index's actual leaf count — `nprobe` equal
    /// to (or exceeding) that count degenerates to an exhaustive scan
    /// across every posting list, which is exactly `FlatIndex`'s result
    /// set (see `ivf.rs`'s `probing_every_cluster_matches_flat_index_exactly`).
    pub nprobe: usize,
    /// `IvfIndex`/`IvfPqIndex` with a *hierarchical* coarse quantizer
    /// only (see `crate::coarse` and `IvfParams::hierarchical_threshold`):
    /// how many top-level parent clusters beam search keeps alive before
    /// expanding to their children, rather than just the single nearest
    /// parent — the guard against the classic k-means-tree boundary
    /// problem where the true nearest leaf's parent isn't top-ranked.
    /// Ignored by a flat coarse quantizer, which has no parent level.
    pub beam_width: usize,
    /// `IvfPqIndex` only: how many ADC-shortlisted candidates get
    /// exact-reranked against their real stored vectors. `None` resolves
    /// to the master plan's dynamic default, `max(200, 20*k)`, computed
    /// per search since it depends on `k`; `Some(0)` disables reranking
    /// entirely for a max-speed/lower-accuracy mode (`SearchHit::exact`
    /// is `false` in that case — see `ivf_pq.rs`'s `adc_score_estimate`).
    /// `IvfIndex`/`IvfPqIndex` also reuse this as the post-filter
    /// candidate-count target for the filtered-search escalation below.
    pub rerank_k: Option<usize>,
    /// Master plan *Filtered search*, regime 1: a filter mask this small
    /// or smaller skips ANN routing entirely — `IvfIndex`/`IvfPqIndex`
    /// fall back to `FlatIndex`'s own exact-scan-over-the-mask behavior,
    /// and even `FlatIndex` uses it to fetch just the allowed rows by id
    /// instead of scanning the whole collection. 100% recall by
    /// construction, and faster than ANN once the candidate set is this
    /// small.
    pub filter_exact_threshold: usize,
    /// Master plan *Filtered search*: the `|allowed| / active_count`
    /// selectivity boundary between regime 2 (**filtered ANN**, mask
    /// pushed into coarse routing from the start) and regime 3 (**plain
    /// ANN, then mask**, which only escalates if masking the normal
    /// unfiltered shortlist leaves fewer than `k` survivors). Irrelevant
    /// once `filter_exact_threshold` already routed to regime 1.
    pub filter_selectivity_high: f64,
    /// Caps how far `IvfIndex`/`IvfPqIndex` escalate `nprobe` while
    /// hunting for a full post-filter candidate set (regimes 2 and 3).
    /// `None` resolves to the master plan's default, `4 * nprobe`.
    /// Hitting this cap (or the coarse quantizer's total leaf count,
    /// whichever is smaller) before reaching the target sets
    /// `SearchResult::truncated_by_filter` rather than silently returning
    /// short.
    pub filter_nprobe_max: Option<usize>,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            max_chunks_per_doc: Some(3),
            nprobe: 8,
            beam_width: 8,
            rerank_k: None,
            filter_exact_threshold: 20_000,
            filter_selectivity_high: 0.6,
            filter_nprobe_max: None,
        }
    }
}
