use serde::{Deserialize, Serialize};

/// Identifies which embedding model produced a collection's (and every
/// document's) vectors. Recorded on a collection at creation and on every
/// `DocEntry`; a request whose configured model doesn't match a
/// collection's fingerprint is rejected rather than silently accepted —
/// cross-model vectors are geometrically meaningless, and this is the
/// single easiest way to produce a database that returns confident
/// nonsense.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct ModelFingerprint {
    /// Identifier of the embedding model.
    pub model_id: String,
    /// Optional model revision or version string.
    pub revision: Option<String>,
    /// Output vector dimensionality.
    pub dim: usize,
}
