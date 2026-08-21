//! Local text-to-vector embeddings (master plan Layer 3, *Embedding model
//! selection*): a `fastembed`-backed wrapper, a canonical model registry
//! with dimension auto-detection, and `ModelFingerprint` construction.
//! Model selection is a config string, validated at load time rather than
//! failing on first insert.

pub mod backend;
pub mod error;
pub mod testing;

pub use backend::EmbeddingBackend;
pub use error::{EmbedError, EmbedResult};
pub use testing::DeterministicTestBackend;

#[cfg(feature = "fastembed-backend")]
mod fastembed_backend;
#[cfg(feature = "fastembed-backend")]
mod registry;

#[cfg(feature = "fastembed-backend")]
pub use fastembed_backend::FastEmbedBackend;
#[cfg(feature = "fastembed-backend")]
pub use registry::{known_model_ids, resolve_model};
