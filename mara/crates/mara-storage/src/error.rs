use mara_proto::RowId;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StorageError {
    #[error("collection {0:?} not found")]
    CollectionNotFound(String),
    #[error("collection {0:?} already exists")]
    CollectionAlreadyExists(String),
    #[error("key {0:?} not found")]
    KeyNotFound(String),
    #[error("row {0:?} not found")]
    RowNotFound(RowId),
    #[error("document {0:?} not found")]
    DocumentNotFound(String),
    #[error("vector has dimension {got}, collection {coll:?} expects {expected}")]
    DimensionMismatch {
        coll: String,
        expected: usize,
        got: usize,
    },
    #[error("{0}")]
    InvalidArgument(String),
    #[error(transparent)]
    Filter(#[from] crate::payload::FilterError),
    #[error("WAL error: {0}")]
    Wal(String),
    #[error("cannot undo txn {txn}: {} of its rows were touched by a later transaction", conflicts.len())]
    UndoConflict { txn: mara_proto::TxnId, conflicts: Vec<mara_proto::Conflict> },
}

pub type StorageResult<T> = Result<T, StorageError>;
