//! A deterministic, offline `EmbeddingBackend` for tests — anywhere in the
//! workspace that needs to exercise the embed-then-store-then-search path
//! without a network connection, an ONNX runtime, or a multi-hundred-MB
//! model download. Vectors are `blake3`-derived from the input text: stable
//! across runs (so tests are reproducible) and distinct for distinct texts,
//! but carry **no semantic meaning whatsoever** — never use this outside
//! tests.

use crate::backend::EmbeddingBackend;
use crate::error::EmbedResult;
use mara_proto::ModelFingerprint;

pub struct DeterministicTestBackend {
    fingerprint: ModelFingerprint,
}

impl DeterministicTestBackend {
    pub fn new(dim: usize) -> Self {
        DeterministicTestBackend {
            fingerprint: ModelFingerprint {
                model_id: "mara-embed/deterministic-test-backend".into(),
                revision: None,
                dim,
            },
        }
    }

    fn embed_one(&self, text: &str) -> Vec<f32> {
        let dim = self.fingerprint.dim;
        let mut out = Vec::with_capacity(dim);
        let mut counter: u32 = 0;
        while out.len() < dim {
            let mut hasher = blake3::Hasher::new();
            hasher.update(text.as_bytes());
            hasher.update(&counter.to_le_bytes());
            let hash = hasher.finalize();
            for chunk in hash.as_bytes().chunks_exact(4) {
                if out.len() == dim {
                    break;
                }
                let bits = u32::from_le_bytes(chunk.try_into().unwrap());
                // Map to [-1, 1) so vectors resemble typical (roughly
                // unit-scale) embedding output rather than raw byte noise.
                out.push((bits as f32 / u32::MAX as f32) * 2.0 - 1.0);
            }
            counter += 1;
        }
        let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut out {
                *v /= norm;
            }
        }
        out
    }
}

impl EmbeddingBackend for DeterministicTestBackend {
    fn fingerprint(&self) -> &ModelFingerprint {
        &self.fingerprint
    }

    fn embed_batch(&self, texts: &[String]) -> EmbedResult<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_one(t)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_text_always_embeds_to_the_same_vector() {
        let b = DeterministicTestBackend::new(16);
        let a = b.embed_batch(&["hello world".to_string()]).unwrap();
        let c = b.embed_batch(&["hello world".to_string()]).unwrap();
        assert_eq!(a, c);
    }

    #[test]
    fn different_text_embeds_differently() {
        let b = DeterministicTestBackend::new(16);
        let a = b.embed_batch(&["hello".to_string()]).unwrap();
        let c = b.embed_batch(&["world".to_string()]).unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn vectors_have_the_configured_dimension_and_unit_norm() {
        let b = DeterministicTestBackend::new(384);
        let out = b.embed_batch(&["a".to_string(), "b".to_string()]).unwrap();
        for v in &out {
            assert_eq!(v.len(), 384);
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "expected unit norm, got {norm}");
        }
    }

    #[test]
    fn embed_batched_matches_embed_batch_regardless_of_chunk_size() {
        let b = DeterministicTestBackend::new(8);
        let texts: Vec<String> = (0..10).map(|i| format!("text-{i}")).collect();
        let whole = b.embed_batch(&texts).unwrap();
        let chunked = b.embed_batched(&texts, 3).unwrap();
        assert_eq!(whole, chunked);
    }
}
