use crate::backend::EmbeddingBackend;
use crate::error::{EmbedError, EmbedResult};
use crate::registry::{resolve_model, variant_for};
use fastembed::{TextEmbedding, TextInitOptions};
use mara_proto::ModelFingerprint;
use parking_lot::Mutex;
use std::path::Path;

/// The production `EmbeddingBackend`: a local ONNX Runtime session via
/// `fastembed`. Wrapped in a `Mutex` because `fastembed::TextEmbedding`'s
/// `embed` takes `&mut self` (it owns the ONNX Runtime session), while
/// `EmbeddingBackend` needs `&self` so it can be shared as
/// `Arc<dyn EmbeddingBackend>` across concurrent requests.
pub struct FastEmbedBackend {
    inner: Mutex<TextEmbedding>,
    fingerprint: ModelFingerprint,
}

impl FastEmbedBackend {
    /// Loads (downloading into `cache_dir` on first use, per the master
    /// plan's `[embedding] cache_dir`) the model named by `model_id`. This
    /// is the slow first-run path autostart's readiness spinner names
    /// explicitly — everything after the first load reads from
    /// `cache_dir`.
    pub fn load(model_id: &str, cache_dir: &Path, show_download_progress: bool) -> EmbedResult<Self> {
        let fingerprint = resolve_model(model_id)?;
        let variant = variant_for(model_id).expect("resolve_model already validated this id resolves to a variant");
        let options = TextInitOptions::new(variant)
            .with_cache_dir(cache_dir.to_path_buf())
            .with_show_download_progress(show_download_progress);
        let inner = TextEmbedding::try_new(options).map_err(|e| EmbedError::Backend(e.to_string()))?;
        Ok(FastEmbedBackend {
            inner: Mutex::new(inner),
            fingerprint,
        })
    }
}

impl EmbeddingBackend for FastEmbedBackend {
    fn fingerprint(&self) -> &ModelFingerprint {
        &self.fingerprint
    }

    fn embed_batch(&self, texts: &[String]) -> EmbedResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut guard = self.inner.lock();
        let out = guard.embed(texts, None).map_err(|e| EmbedError::Backend(e.to_string()))?;
        if let Some(v) = out.first() {
            if v.len() != self.fingerprint.dim {
                return Err(EmbedError::DimMismatch {
                    model_id: self.fingerprint.model_id.clone(),
                    expected: self.fingerprint.dim,
                    got: v.len(),
                });
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Actually loads a model — downloads it on first run. Not run by
    /// default (no network / no multi-hundred-MB download in ordinary
    /// `cargo test`); run explicitly with
    /// `cargo test -p mara-embed -- --ignored` when you want to exercise
    /// the real backend end to end.
    #[test]
    #[ignore = "downloads a real ONNX model from the network on first run"]
    fn loads_and_embeds_a_real_model() {
        let dir = tempfile::tempdir().unwrap();
        let backend = FastEmbedBackend::load("sentence-transformers/all-MiniLM-L6-v2", dir.path(), false).unwrap();
        assert_eq!(backend.fingerprint().dim, 384);

        let out = backend
            .embed_batch(&["hello world".to_string(), "a second sentence".to_string()])
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 384);
    }
}
