use serde::{Deserialize, Serialize};

/// How a document's raw text is split into chunks. Persisted on the
/// `DocEntry` alongside a `chunker_version` so a re-ingest reproduces the
/// same boundaries.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkStrategy {
    Characters,
    Tokens { tokenizer: String },
    Markdown { respect_headings: bool },
    Sentences,
}

impl Default for ChunkStrategy {
    fn default() -> Self {
        ChunkStrategy::Markdown {
            respect_headings: true,
        }
    }
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct ChunkSpec {
    pub strategy: ChunkStrategy,
    pub max_tokens: usize,
    pub overlap_tokens: usize,
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
