//! Vector search / indexing (master plan Layer 4): `Flat` (exact), `Ivf`
//! (coarse-quantized), and `IvfPq` (coarse + product-quantized, with
//! exact rerank) so far — OPQ, hierarchical coarse quantization + beam
//! search, and `Lsh` join them as the corresponding build steps land.
//! `Flat` is not a fallback bolted on last; it's the recall oracle every
//! approximate index gets validated against, and the default at this
//! project's stated scale.

mod coarse;
pub mod error;
pub mod flat;
mod grouping;
pub mod hit;
pub mod index_trait;
pub mod ivf;
pub mod ivf_pq;
mod kmeans;
pub mod live;
pub mod lsh;
pub mod math;
pub mod opq;
pub mod params;
pub mod pq;
mod scoring;

pub use error::{IndexError, IndexResult};
pub use flat::FlatIndex;
pub use hit::{SearchHit, SearchResult};
pub use index_trait::VectorIndex;
pub use ivf::{IvfIndex, IvfParams};
pub use ivf_pq::IvfPqIndex;
pub use live::{Builder, LiveIndex, LiveIndexParams};
pub use lsh::{LshError, LshIndex, LshParams};
pub use opq::{train_opq, OpqParams, OpqRotation};
pub use params::SearchParams;
pub use pq::{AdcTable, PqCodebook, PqError, PqParams};
