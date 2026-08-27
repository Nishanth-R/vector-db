use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Distance metric a collection searches under. `SearchHit::score` is always
/// reported higher-is-better in this metric's convention, regardless of
/// whether the underlying index internally computes something else (e.g.
/// cosine is handled as L2-on-the-unit-sphere internally by the vector
/// index, but the score returned to any caller always comes from an exact
/// rerank in true cosine terms).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    /// Cosine similarity.
    Cosine,
    /// Euclidean (L2) distance.
    L2,
    /// Raw dot product.
    DotProduct,
}
