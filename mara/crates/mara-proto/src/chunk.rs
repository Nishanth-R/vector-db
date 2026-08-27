use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// How a document's raw text is split into chunks. Persisted on the
/// `DocEntry` alongside a `chunker_version` so a re-ingest reproduces the
/// same boundaries.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ChunkStrategy {
    /// Split on a fixed number of characters.
    Characters,
    /// Split using a named tokenizer's token boundaries.
    Tokens {
        /// Name of the tokenizer used to determine token boundaries.
        tokenizer: String,
    },
    /// Split on Markdown structure.
    Markdown {
        /// Whether heading boundaries force a new chunk.
        respect_headings: bool,
    },
    /// Split on sentence boundaries.
    Sentences,
}

impl Default for ChunkStrategy {
    fn default() -> Self {
        ChunkStrategy::Markdown {
            respect_headings: true,
        }
    }
}

/// Full configuration for splitting a document into chunks.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, ToSchema)]
pub struct ChunkSpec {
    /// The splitting strategy to apply.
    pub strategy: ChunkStrategy,
    /// Maximum number of tokens allowed per chunk.
    pub max_tokens: usize,
    /// Number of tokens shared between consecutive chunks.
    pub overlap_tokens: usize,
    /// Whether to trim leading/trailing whitespace from each chunk.
    pub trim: bool,
}

impl Default for ChunkSpec {
    fn default() -> Self {
        ChunkSpec {
            strategy: ChunkStrategy::default(),
            max_tokens: 512,
            overlap_tokens: 64,
            trim: true,
        }
    }
}
