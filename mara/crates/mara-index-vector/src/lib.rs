//! Vector search / indexing (master plan Layer 4): `Flat` for now — `IvfPq`
//! (with OPQ preprocessing and beam-search cluster routing) and `Lsh` join
//! it once the corresponding build steps land. `Flat` is not a fallback
//! bolted on last; it's the recall oracle every approximate index gets
//! validated against, and the default at this project's stated scale.

pub mod error;
pub mod flat;
pub mod hit;
pub mod index_trait;
pub mod math;
pub mod params;

pub use error::{IndexError, IndexResult};
pub use flat::FlatIndex;
pub use hit::SearchHit;
pub use index_trait::VectorIndex;
pub use params::SearchParams;
