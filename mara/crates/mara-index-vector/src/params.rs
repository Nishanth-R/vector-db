/// Search-time knobs shared by every `VectorIndex` implementation.
/// Deliberately minimal for now — `nprobe`/`rerank_k`/`beam_width` and
/// friends join this struct when IVF-PQ lands (they don't mean anything to
/// `FlatIndex`, which is already exact).
#[derive(Clone, Debug)]
pub struct SearchParams {
    /// Caps how many chunks from the same document can appear in one
    /// result set — without this, a single verbose document reliably
    /// monopolizes the top-k. `None` disables grouping entirely.
    pub max_chunks_per_doc: Option<u32>,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            max_chunks_per_doc: Some(3),
        }
    }
}
