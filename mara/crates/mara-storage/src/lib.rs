//! Storage engine: WAL, snapshot, documents, payload store + filters,
//! `TxnIndex`, undo/revert (`Layer 2` of the master plan). Built up in
//! stages — this module tree grows alongside the build order rather than
//! being declared all at once against unbuilt functionality.

pub mod api;
pub mod change;
pub mod collection;
pub mod document;
pub mod error;
pub mod payload;
mod recovery;
mod snapshot;
mod txn_index;
mod undo;
pub mod wal;

pub use api::{CollectionInfo, SchemaChange, Storage, StorageApi};
pub use change::{ChangeBatch, ChangeEvent, ChangeSubscriber};
pub use collection::{Collection, PutInput};
pub use document::{ChunkInput, DocEntry, PutDocumentInput};
pub use error::{StorageError, StorageResult};
pub use payload::{FieldType, FilterError, FilterMask, PayloadSchema, PayloadStore};
pub use wal::{summarize_txn, FsyncPolicy, WalActor, WalError, WalOp, WalPayload, WalRecord, WalReplayResult, WalWriter};
