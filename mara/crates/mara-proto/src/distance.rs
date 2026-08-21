use serde::{Deserialize, Serialize};

/// Distance metric a collection searches under. `SearchHit::score` is always
/// reported higher-is-better in this metric's convention, regardless of
/// whether the underlying index internally computes something else (e.g.
/// cosine is handled as L2-on-the-unit-sphere internally by the vector
/// index, but the score returned to any caller always comes from an exact
/// rerank in true cosine terms).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    L2,
    DotProduct,
}
