use crate::error::EmbedResult;
use mara_proto::ModelFingerprint;

pub trait EmbeddingBackend: Send + Sync {
    fn fingerprint(&self) -> &ModelFingerprint;
    fn embed_batch(&self, texts: &[String]) -> EmbedResult<Vec<Vec<f32>>>;

    fn embed_batched(&self, texts: &[String], batch_size: usize) -> EmbedResult<Vec<Vec<f32>>> {
        let batch_size = batch_size.max(1);
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(batch_size) {
            out.extend(self.embed_batch(chunk)?);
        }
        Ok(out)
    }
}
