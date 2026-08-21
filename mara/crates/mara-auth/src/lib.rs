//! Identity, auth, audit, and the capability table (`Layer 1` of the master
//! plan). Placed early in the build order because `StorageApi` and the WAL
//! record format both depend on `RequestCtx` from `mara-proto`.

pub mod audit;
pub mod capability;
pub mod principal;
pub mod store;
pub mod token;

pub use audit::{
    prune_older_than, AuditConfig, AuditError, AuditFsync, AuditMode, AuditOutcome,
    AuditPrincipal, AuditRecord, AuditSink, AuditSource, JsonlAuditSink,
};
pub use capability::{can_use_undo_scope, require, Capability};
pub use principal::PrincipalRecord;
pub use store::{TokenStore, TokenStoreError};
