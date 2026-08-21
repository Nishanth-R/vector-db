#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("unknown embedding model {model:?}; supported models: {known}")]
    UnknownModel { model: String, known: String },
    #[error("custom local ONNX models ({0:?}) aren't supported yet — pass a supported model id instead")]
    CustomModelUnsupported(String),
    #[error("model {model_id:?} produced a {got}-dim vector, expected {expected} — this is an mara-embed/fastembed bug, not a config error")]
    DimMismatch { model_id: String, expected: usize, got: usize },
    #[error("embedding backend error: {0}")]
    Backend(String),
}

pub type EmbedResult<T> = Result<T, EmbedError>;
