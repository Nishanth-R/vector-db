//! `Bm25Index`: a small, purpose-built in-memory inverted index (master
//! plan Layer 5, *BM25*) — deliberately not `tantivy`, whose independent
//! commit/segment lifecycle would create a second durability mechanism
//! alongside the WAL. `insert`/`remove` are the live update path (the same
//! `ChangeBatch`-driven pattern `IvfIndex`'s sibling `LiveIndex` will use);
//! `search` never mutates.

use crate::tokenize::{tokenize, TokenizeConfig};
use mara_proto::RowId;
use mara_storage::FilterMask;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bm25Params {
    pub k1: f32,
    pub b: f32,
    /// Off by default — see `crate::tokenize`'s module doc.
    pub stemming: bool,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Bm25Params { k1: 1.2, b: 0.75, stemming: false }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LexicalHit {
    pub id: RowId,
    /// Raw BM25 score — unbounded, higher is more relevant. Not on the
    /// same scale as a vector index's cosine/L2/dot score; `mara-fusion`
    /// (the next build step) is what reconciles the two.
    pub score: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct PostingList {
    doc_ids: roaring::RoaringBitmap,
    /// A `HashMap` rather than an array literally parallel to `doc_ids`'
    /// iteration order: unlike `IvfIndex`'s posting lists (built once from
    /// a full scan and never mutated), BM25's are live — `insert`/`remove`
    /// need O(1) amortized updates, which a positionally-parallel array
    /// can't offer without an O(n) shift on every non-append change.
    term_freq: HashMap<RowId, u32>,
}

/// `Bm25Index::snapshot`'s serializable output — see the master plan's
/// "same own-its-own-snapshot rule" as `VectorIndex`. A distinct type from
/// `Bm25Index` itself (even though the field shapes mirror each other)
/// so a caller can't mistake "clone the live index" for "durably persist
/// it" — actual file-level persistence (bincode framing, atomic rename,
/// generation retention, matching `mara-storage::snapshot`'s conventions)
/// is daemon-integration plumbing that lands with the layers that need
/// it, same as `IvfIndex`/`IvfPqIndex` having no disk persistence yet
/// either.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bm25Snapshot {
    k1: f32,
    b: f32,
    stemming: bool,
    postings: HashMap<String, PostingList>,
    doc_len: HashMap<RowId, u32>,
    n: u64,
    total_len: u64,
}

pub trait LexicalIndex: Send + Sync {
    fn insert(&mut self, id: RowId, text: &str);
    fn remove(&mut self, id: RowId, text: &str);
    fn search(&self, query: &str, k: usize, filter: Option<&FilterMask>) -> Vec<LexicalHit>;
    fn snapshot(&self) -> Bm25Snapshot;
    fn restore(&mut self, snapshot: Bm25Snapshot);
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bm25Index {
    params: Bm25Params,
    postings: HashMap<String, PostingList>,
    doc_len: HashMap<RowId, u32>,
    n: u64,
    total_len: u64,
}

impl Bm25Index {
    pub fn new(params: Bm25Params) -> Self {
        Bm25Index {
            params,
            postings: HashMap::new(),
            doc_len: HashMap::new(),
            n: 0,
            total_len: 0,
        }
    }

    fn tokenize_config(&self) -> TokenizeConfig {
        TokenizeConfig { stemming: self.params.stemming }
    }

    pub fn doc_count(&self) -> u64 {
        self.n
    }

    /// Resolves `mara_proto::Filter::TextMatch { terms, .. }` into a
    /// bitmap of matching rows — the building block a higher layer (the
    /// daemon's filter-tree evaluator, wired in the next build step) uses
    /// when it encounters a `TextMatch` node while composing a larger
    /// `And`/`Or`/`Not` predicate. `mara-storage::compile_filter` can't
    /// resolve this itself (it has no BM25 index to ask, by design — see
    /// `FilterError::TextMatchUnsupported`), and this index has no
    /// business parsing `Filter` trees itself, so the split sits exactly
    /// at "give me a bitmap for this one leaf." AND semantics across
    /// `terms` (a row must contain all of them) — a filter should be
    /// precise, unlike ranked `search`'s recall-oriented OR-across-terms.
    pub fn resolve_text_match(&self, terms: &[String]) -> roaring::RoaringBitmap {
        let tokenized: Vec<String> = terms.iter().flat_map(|t| tokenize(t, &self.tokenize_config())).collect();
        if tokenized.is_empty() {
            return roaring::RoaringBitmap::new();
        }
        let mut result: Option<roaring::RoaringBitmap> = None;
        for term in &tokenized {
            let ids = self.postings.get(term).map(|p| p.doc_ids.clone()).unwrap_or_default();
            result = Some(match result {
                Some(acc) => acc & ids,
                None => ids,
            });
        }
        result.unwrap_or_default()
    }

    fn term_counts(&self, text: &str) -> (Vec<String>, HashMap<String, u32>) {
        let tokens = tokenize(text, &self.tokenize_config());
        let mut counts: HashMap<String, u32> = HashMap::new();
        for t in &tokens {
            *counts.entry(t.clone()).or_insert(0) += 1;
        }
        (tokens, counts)
    }

    /// Indexes `text` under `id`. `id` must not already be indexed —
    /// re-indexing a changed document is `remove` then `insert`
    /// (`replace_document`'s own transaction already gives the caller
    /// both the old and new text to do exactly that), not an implicit
    /// update.
    pub fn insert(&mut self, id: RowId, text: &str) {
        let (tokens, counts) = self.term_counts(text);
        self.doc_len.insert(id, tokens.len() as u32);
        self.total_len += tokens.len() as u64;
        self.n += 1;

        for (term, freq) in counts {
            let posting = self.postings.entry(term).or_default();
            posting.doc_ids.insert(id.to_bitmap_index());
            posting.term_freq.insert(id, freq);
        }
    }

    /// The exact inverse of `insert(id, text)` — `text` must be the same
    /// text `insert` was called with, so tokenization reproduces the same
    /// terms and counts. After this call, every posting list, per-term
    /// document frequency, `N`, and `avgdl` input is back to exactly what
    /// it was before the matching `insert` — including dropping a term's
    /// entire `postings` entry once its last document is removed, so no
    /// empty-but-present posting list lingers as an observable difference
    /// from "this term was never indexed."
    pub fn remove(&mut self, id: RowId, text: &str) {
        let (tokens, counts) = self.term_counts(text);
        self.doc_len.remove(&id);
        self.total_len -= tokens.len() as u64;
        self.n -= 1;

        for term in counts.keys() {
            if let Some(posting) = self.postings.get_mut(term) {
                posting.doc_ids.remove(id.to_bitmap_index());
                posting.term_freq.remove(&id);
                if posting.doc_ids.is_empty() {
                    self.postings.remove(term);
                }
            }
        }
    }

    /// Lucene-style smoothed IDF × standard BM25 TF term, `k1`/`b` from
    /// `params`. `filter` intersects into posting-list traversal directly
    /// — the same pre-filtering principle `mara-index-vector`'s *Filtered
    /// search* applies, not a post-hoc mask over an unfiltered result.
    /// Query terms are deduplicated (a repeated term contributes its IDF
    /// once, not `k` times for `k` repetitions) since BM25's TF term
    /// already captures how often a term appears in the *document*; how
    /// often it appears in the *query* isn't part of the standard formula.
    pub fn search(&self, query: &str, k: usize, filter: Option<&FilterMask>) -> Vec<LexicalHit> {
        if k == 0 || self.n == 0 {
            return Vec::new();
        }
        let terms = tokenize(query, &self.tokenize_config());
        let avgdl = (self.total_len as f32 / self.n as f32).max(1.0);

        let mut scores: HashMap<RowId, f32> = HashMap::new();
        let mut seen: HashSet<&str> = HashSet::new();
        for term in &terms {
            if !seen.insert(term.as_str()) {
                continue;
            }
            let Some(posting) = self.postings.get(term) else { continue };
            let df = posting.doc_ids.len() as f32;
            let idf = (1.0 + (self.n as f32 - df + 0.5) / (df + 0.5)).ln();

            let mut candidates = posting.doc_ids.clone();
            if let Some(f) = filter {
                candidates &= &f.allowed;
            }

            for bits in candidates.iter() {
                let id = RowId::from_bitmap_index(bits);
                let tf = *posting.term_freq.get(&id).expect("doc_ids and term_freq must stay in sync") as f32;
                let doclen = *self.doc_len.get(&id).expect("every indexed row has a doc_len") as f32;
                let denom = tf + self.params.k1 * (1.0 - self.params.b + self.params.b * doclen / avgdl);
                let contribution = idf * (tf * (self.params.k1 + 1.0)) / denom;
                *scores.entry(id).or_insert(0.0) += contribution;
            }
        }

        let mut ranked: Vec<LexicalHit> = scores.into_iter().map(|(id, score)| LexicalHit { id, score }).collect();
        ranked.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        ranked.truncate(k);
        ranked
    }

    pub fn snapshot(&self) -> Bm25Snapshot {
        Bm25Snapshot {
            k1: self.params.k1,
            b: self.params.b,
            stemming: self.params.stemming,
            postings: self.postings.clone(),
            doc_len: self.doc_len.clone(),
            n: self.n,
            total_len: self.total_len,
        }
    }

    pub fn restore(&mut self, snapshot: Bm25Snapshot) {
        self.params = Bm25Params {
            k1: snapshot.k1,
            b: snapshot.b,
            stemming: snapshot.stemming,
        };
        self.postings = snapshot.postings;
        self.doc_len = snapshot.doc_len;
        self.n = snapshot.n;
        self.total_len = snapshot.total_len;
    }
}

impl LexicalIndex for Bm25Index {
    fn insert(&mut self, id: RowId, text: &str) {
        Bm25Index::insert(self, id, text)
    }

    fn remove(&mut self, id: RowId, text: &str) {
        Bm25Index::remove(self, id, text)
    }

    fn search(&self, query: &str, k: usize, filter: Option<&FilterMask>) -> Vec<LexicalHit> {
        Bm25Index::search(self, query, k, filter)
    }

    fn snapshot(&self) -> Bm25Snapshot {
        Bm25Index::snapshot(self)
    }

    fn restore(&mut self, snapshot: Bm25Snapshot) {
        Bm25Index::restore(self, snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use roaring::RoaringBitmap;

    fn id(n: u64) -> RowId {
        RowId(n)
    }

    #[test]
    fn insert_then_search_finds_the_document() {
        let mut idx = Bm25Index::new(Bm25Params::default());
        idx.insert(id(1), "the quick brown fox jumps over the lazy dog");
        let hits = idx.search("quick fox", 5, None);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, id(1));
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn higher_term_frequency_scores_higher() {
        let mut idx = Bm25Index::new(Bm25Params::default());
        idx.insert(id(1), "cat cat cat cat cat dog");
        idx.insert(id(2), "cat dog dog dog dog dog");
        let hits = idx.search("cat", 5, None);
        assert_eq!(hits[0].id, id(1), "doc 1 mentions \"cat\" far more often and must rank first");
    }

    #[test]
    fn rare_terms_contribute_more_than_common_terms() {
        let mut idx = Bm25Index::new(Bm25Params::default());
        // "common" appears in every doc; "rare" appears in only one.
        for i in 0..10 {
            idx.insert(id(i), "common word filler text here");
        }
        idx.insert(id(100), "common rare word");
        let hits = idx.search("common rare", 20, None);
        // doc 100 matches both a common and a rare term; it should score
        // above docs that only match the ubiquitous "common".
        let top = &hits[0];
        assert_eq!(top.id, id(100), "the doc matching the rare term must outrank docs matching only the common one");
    }

    #[test]
    fn filter_mask_excludes_non_matching_rows() {
        let mut idx = Bm25Index::new(Bm25Params::default());
        idx.insert(id(1), "hello world");
        idx.insert(id(2), "hello there");
        let mut allowed = RoaringBitmap::new();
        allowed.insert(id(1).to_bitmap_index());
        let mask = FilterMask { allowed, estimated_cardinality: 1 };
        let hits = idx.search("hello", 5, Some(&mask));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, id(1));
    }

    #[test]
    fn insert_then_remove_restores_the_exact_prior_state() {
        let mut idx = Bm25Index::new(Bm25Params::default());
        idx.insert(id(1), "some baseline document with several words");
        let before = idx.clone();

        idx.insert(id(2), "the quick brown fox jumps over the lazy dog");
        idx.remove(id(2), "the quick brown fox jumps over the lazy dog");

        assert_eq!(idx, before, "insert followed by remove of the same (id, text) must restore bit-identical internal state");
    }

    #[test]
    fn removing_a_terms_only_document_drops_the_term_entirely_not_just_its_membership() {
        let mut idx = Bm25Index::new(Bm25Params::default());
        let empty = idx.clone();
        idx.insert(id(1), "unique-term-nowhere-else");
        idx.remove(id(1), "unique-term-nowhere-else");
        assert_eq!(idx, empty, "a term with no remaining documents must not leave a stale empty posting-list entry behind");
        assert!(idx.search("unique-term-nowhere-else", 5, None).is_empty());
    }

    #[test]
    fn resolve_text_match_requires_every_term_present() {
        let mut idx = Bm25Index::new(Bm25Params::default());
        idx.insert(id(1), "red apple pie");
        idx.insert(id(2), "red velvet cake");
        idx.insert(id(3), "apple pie recipe");

        let both = idx.resolve_text_match(&["apple".into(), "pie".into()]);
        assert_eq!(both.len(), 2, "docs 1 and 3 both contain \"apple\" and \"pie\"");
        assert!(both.contains(id(1).to_bitmap_index()));
        assert!(both.contains(id(3).to_bitmap_index()));
        assert!(!both.contains(id(2).to_bitmap_index()));
    }

    #[test]
    fn resolve_text_match_with_an_unknown_term_matches_nothing() {
        let mut idx = Bm25Index::new(Bm25Params::default());
        idx.insert(id(1), "apple pie");
        let hits = idx.resolve_text_match(&["apple".into(), "xyzzy".into()]);
        assert!(hits.is_empty());
    }

    #[test]
    fn snapshot_restore_round_trips() {
        let mut idx = Bm25Index::new(Bm25Params { k1: 1.5, b: 0.8, stemming: true });
        idx.insert(id(1), "the quick brown fox");
        idx.insert(id(2), "the lazy dog sleeps");

        let snap = idx.snapshot();
        let mut restored = Bm25Index::new(Bm25Params::default());
        restored.restore(snap);

        assert_eq!(idx, restored);
        assert_eq!(idx.search("fox", 5, None), restored.search("fox", 5, None));
    }

    #[test]
    fn k_zero_returns_empty() {
        let mut idx = Bm25Index::new(Bm25Params::default());
        idx.insert(id(1), "some text");
        assert!(idx.search("text", 0, None).is_empty());
    }

    #[test]
    fn empty_index_returns_empty() {
        let idx = Bm25Index::new(Bm25Params::default());
        assert!(idx.search("anything", 5, None).is_empty());
    }

    #[test]
    fn unknown_query_terms_return_no_hits() {
        let mut idx = Bm25Index::new(Bm25Params::default());
        idx.insert(id(1), "hello world");
        assert!(idx.search("xyzzy plugh", 5, None).is_empty());
    }

    #[test]
    fn repeated_query_terms_are_deduplicated() {
        let mut idx = Bm25Index::new(Bm25Params::default());
        idx.insert(id(1), "hello world");
        let once = idx.search("hello", 5, None);
        let repeated = idx.search("hello hello hello", 5, None);
        assert_eq!(once, repeated, "repeating a query term must not multiply its score contribution");
    }

    proptest! {
        #[test]
        fn insert_remove_is_always_an_exact_inverse(
            texts in prop::collection::vec("[a-z ]{0,40}", 1..6),
        ) {
            let mut idx = Bm25Index::new(Bm25Params::default());
            let before = idx.clone();
            let ids: Vec<RowId> = (0..texts.len() as u64).map(id).collect();

            for (i, t) in ids.iter().zip(&texts) {
                idx.insert(*i, t);
            }
            for (i, t) in ids.iter().zip(&texts).rev() {
                idx.remove(*i, t);
            }

            prop_assert_eq!(idx, before);
        }
    }
}
