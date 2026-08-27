//! Splits a document's raw text into chunks per a `ChunkSpec` (master plan
//! Layer 3, *mara-chunker*), wrapping `text-splitter`. Runs upstream of
//! embedding — `mara-daemon`'s `PutDocument` handling calls this, then
//! hands the resulting strings to `mara-embed`, then `mara-storage`'s
//! `put_document`. Chunking here is pure text-in, text-out and has no
//! knowledge of vectors, storage, or the wire protocol.
//!
//! `ChunkSpec::strategy` names four variants, but `text-splitter` backs
//! only two distinct splitting behaviors: character/token-counted plain
//! text (`TextSplitter`) and markdown-structure-aware text
//! (`MarkdownSplitter`). `Characters` and `Sentences` both map to the
//! former — the crate has no separate sentence-only mode; its default
//! semantic-boundary search (paragraph -> sentence -> word -> char)
//! already applies regardless of how chunk size is counted, so `Sentences`
//! is documented here as a scope limitation of the underlying crate, not
//! implemented as a distinct code path. `Markdown{respect_headings}` maps
//! to `MarkdownSplitter` for both `true` and `false`: the crate has no
//! lever to disable heading-awareness specifically while keeping the rest
//! of its markdown structure handling (code fences, lists, etc), so both
//! values currently produce identical output — the field is accepted (and
//! round-trips on `DocEntry`) for forward compatibility if a future
//! `text-splitter` version exposes finer control.

use mara_proto::{ChunkSpec, ChunkStrategy};
use text_splitter::{ChunkConfig, MarkdownSplitter, TextSplitter};

#[derive(Debug, thiserror::Error)]
pub enum ChunkError {
    #[error("invalid chunk_spec: {0}")]
    InvalidChunkSpec(String),
    #[error("failed to load tokenizer {tokenizer:?}: {reason}")]
    TokenizerLoad { tokenizer: String, reason: String },
}

pub type ChunkResult<T> = Result<T, ChunkError>;

fn char_config(spec: &ChunkSpec) -> ChunkResult<ChunkConfig<text_splitter::Characters>> {
    ChunkConfig::new(spec.max_tokens)
        .with_overlap(spec.overlap_tokens)
        .map_err(|e| ChunkError::InvalidChunkSpec(e.to_string()))
        .map(|c| c.with_trim(spec.trim))
}

/// Splits `text` per `spec`. Chunk boundaries are a pure function of
/// `(text, spec)` — the same document re-ingested with the same
/// `ChunkSpec` reproduces byte-identical chunks, which is what lets a
/// `DocEntry`'s persisted `chunk_spec` serve as a real re-ingest contract.
pub fn chunk_text(text: &str, spec: &ChunkSpec) -> ChunkResult<Vec<String>> {
    match &spec.strategy {
        ChunkStrategy::Characters | ChunkStrategy::Sentences => {
            let splitter = TextSplitter::new(char_config(spec)?);
            Ok(splitter.chunks(text).map(str::to_string).collect())
        }
        ChunkStrategy::Markdown { .. } => {
            let splitter = MarkdownSplitter::new(char_config(spec)?);
            Ok(splitter.chunks(text).map(str::to_string).collect())
        }
        ChunkStrategy::Tokens { tokenizer } => {
            let tok = tokenizers::Tokenizer::from_pretrained(tokenizer, None).map_err(|e| ChunkError::TokenizerLoad {
                tokenizer: tokenizer.clone(),
                reason: e.to_string(),
            })?;
            let config = ChunkConfig::new(spec.max_tokens)
                .with_sizer(tok)
                .with_overlap(spec.overlap_tokens)
                .map_err(|e| ChunkError::InvalidChunkSpec(e.to_string()))?
                .with_trim(spec.trim);
            let splitter = TextSplitter::new(config);
            Ok(splitter.chunks(text).map(str::to_string).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(strategy: ChunkStrategy, max_tokens: usize, overlap_tokens: usize) -> ChunkSpec {
        ChunkSpec {
            strategy,
            max_tokens,
            overlap_tokens,
            trim: true,
        }
    }

    #[test]
    fn characters_strategy_splits_long_text_into_multiple_chunks() {
        let text = "hello world. ".repeat(50);
        let chunks = chunk_text(&text, &spec(ChunkStrategy::Characters, 100, 10)).unwrap();
        assert!(chunks.len() > 1, "expected multiple chunks, got {}", chunks.len());
        for c in &chunks {
            assert!(c.len() <= 100, "chunk exceeded max_tokens: {} chars", c.len());
        }
    }

    #[test]
    fn sentences_strategy_behaves_like_characters_by_documented_design() {
        let text = "hello world. ".repeat(50);
        let a = chunk_text(&text, &spec(ChunkStrategy::Characters, 100, 10)).unwrap();
        let b = chunk_text(&text, &spec(ChunkStrategy::Sentences, 100, 10)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn markdown_strategy_keeps_a_short_document_with_headings_as_one_chunk() {
        let text = "# Title\n\nA short paragraph under the heading.";
        let chunks = chunk_text(text, &spec(ChunkStrategy::Markdown { respect_headings: true }, 512, 64)).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("Title"));
    }

    #[test]
    fn markdown_strategy_splits_a_long_document_across_headings() {
        let mut text = String::new();
        for i in 0..30 {
            text.push_str(&format!("## Section {i}\n\nSome body text for section {i} that takes up a bit of room.\n\n"));
        }
        let chunks = chunk_text(&text, &spec(ChunkStrategy::Markdown { respect_headings: true }, 100, 10)).unwrap();
        assert!(chunks.len() > 1);
    }

    #[test]
    fn empty_text_produces_no_chunks() {
        let chunks = chunk_text("", &spec(ChunkStrategy::Characters, 100, 10)).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn overlap_greater_than_or_equal_to_max_tokens_is_a_clear_error() {
        let err = chunk_text("some text", &spec(ChunkStrategy::Characters, 10, 20)).unwrap_err();
        assert!(matches!(err, ChunkError::InvalidChunkSpec(_)));
    }

    #[test]
    fn same_input_and_spec_always_produce_identical_chunks() {
        let text = "# Doc\n\n".to_string() + &"paragraph text here. ".repeat(80);
        let s = spec(ChunkStrategy::Markdown { respect_headings: true }, 150, 20);
        let a = chunk_text(&text, &s).unwrap();
        let b = chunk_text(&text, &s).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    #[ignore = "downloads a real tokenizer.json from the network on first run"]
    fn tokens_strategy_loads_a_real_tokenizer_and_splits_by_token_count() {
        let text = "hello world. ".repeat(50);
        let chunks = chunk_text(
            &text,
            &spec(
                ChunkStrategy::Tokens {
                    tokenizer: "bert-base-cased".to_string(),
                },
                50,
                5,
            ),
        )
        .unwrap();
        assert!(chunks.len() > 1);
    }

    #[test]
    #[ignore = "from_pretrained hits the network even to discover an id doesn't exist"]
    fn unknown_tokenizer_id_is_a_clear_error_not_a_panic() {
        let err = chunk_text(
            "some text",
            &spec(
                ChunkStrategy::Tokens {
                    tokenizer: "this-model-definitely-does-not-exist/nope".to_string(),
                },
                50,
                5,
            ),
        )
        .unwrap_err();
        assert!(matches!(err, ChunkError::TokenizerLoad { .. }));
    }
}
