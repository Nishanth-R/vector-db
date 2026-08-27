//! BM25 tokenization (master plan Layer 5, *BM25*): UAX #29 word
//! segmentation via `unicode-segmentation`, lowercasing, a compile-time
//! embedded English stopword set via `phf` (no NLTK-style runtime
//! download), and optional Snowball stemming via `rust-stemmers` — off by
//! default, since `insert`/`remove`/`search` must all tokenize identically
//! for the index to stay internally consistent, and stemming is a
//! meaningful recall/precision trade a deployment should opt into
//! deliberately rather than inherit silently.

use rust_stemmers::{Algorithm, Stemmer};
use unicode_segmentation::UnicodeSegmentation;

/// A standard, Lucene-style English stop list — common function words
/// that carry no discriminating power in term-frequency scoring and would
/// otherwise dominate every posting list's document frequency.
static STOPWORDS: phf::Set<&'static str> = phf::phf_set! {
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "for", "if", "in", "into", "is", "it", "its", "no", "not",
    "of", "on", "or", "such", "that", "the", "their", "then",
    "there", "these", "they", "this", "to", "was", "will", "with",
};

#[derive(Clone, Copy, Debug, Default)]
pub struct TokenizeConfig {
    pub stemming: bool,
}

/// Splits `text` into indexable terms: unicode word boundaries, lowercased,
/// stopwords dropped, and — only when `config.stemming` is set — reduced
/// to a Snowball English stem. Deterministic and total: every call with
/// the same `(text, config)` produces the same terms, which is what lets
/// `Bm25Index::remove` be an exact inverse of `insert` without the index
/// itself needing to remember what it tokenized to.
pub fn tokenize(text: &str, config: &TokenizeConfig) -> Vec<String> {
    let words = text.unicode_words().map(|w| w.to_lowercase()).filter(|w| !STOPWORDS.contains(w.as_str()));
    if config.stemming {
        let stemmer = Stemmer::create(Algorithm::English);
        words.map(|w| stemmer.stem(&w).into_owned()).collect()
    } else {
        words.collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_unicode_word_boundaries_and_lowercases() {
        let terms = tokenize("The Quick-Brown Fox!", &TokenizeConfig::default());
        assert_eq!(terms, vec!["quick", "brown", "fox"], "\"the\" is a stopword and gets dropped");
    }

    #[test]
    fn stopwords_are_dropped() {
        let terms = tokenize("this is a test of the stopword list", &TokenizeConfig::default());
        assert_eq!(terms, vec!["test", "stopword", "list"]);
    }

    #[test]
    fn stemming_is_off_by_default() {
        let terms = tokenize("running runs jumps", &TokenizeConfig::default());
        assert_eq!(terms, vec!["running", "runs", "jumps"]);
    }

    #[test]
    fn stemming_reduces_related_forms_to_the_same_term_when_enabled() {
        let terms = tokenize("running runs jumps", &TokenizeConfig { stemming: true });
        assert_eq!(terms[0], terms[1], "\"running\" and \"runs\" must stem identically for BM25 to treat them as the same term");
        assert_ne!(terms[0], terms[2]);
    }

    #[test]
    fn empty_text_produces_no_terms() {
        assert!(tokenize("", &TokenizeConfig::default()).is_empty());
        assert!(tokenize("   ", &TokenizeConfig::default()).is_empty());
    }

    #[test]
    fn same_input_always_tokenizes_identically() {
        let cfg = TokenizeConfig { stemming: true };
        let text = "The quick brown fox jumps over the lazy dog, running fast.";
        assert_eq!(tokenize(text, &cfg), tokenize(text, &cfg));
    }
}
