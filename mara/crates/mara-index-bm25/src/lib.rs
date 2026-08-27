//! BM25 lexical search (master plan Layer 5, *BM25*): a small,
//! purpose-built in-memory inverted index behind a `LexicalIndex` trait
//! symmetric with `mara-index-vector`'s `VectorIndex` — same
//! `ChangeBatch`-driven live-update pattern, same own-its-own-snapshot
//! rule. `Filter::TextMatch` resolves here, not in `mara-storage`'s
//! payload store (see `mara_storage::payload::FilterError::TextMatchUnsupported`).

mod index;
mod tokenize;

pub use index::{Bm25Index, Bm25Params, Bm25Snapshot, LexicalHit, LexicalIndex};
pub use tokenize::{tokenize, TokenizeConfig};
