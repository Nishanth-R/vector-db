#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("query vector has dimension {got}, index expects {expected}")]
    DimMismatch { expected: usize, got: usize },
    #[error(transparent)]
    Storage(#[from] mara_storage::StorageError),
}

pub type IndexResult<T> = Result<T, IndexError>;
