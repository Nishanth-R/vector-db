use crate::ids::{DocId, PrincipalId, SessionId, TxnId};
use crate::lsn::Lsn;
use chrono::{DateTime, Utc};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

/// Scope for `UndoTarget::LastN` — deliberately narrow by default. A global
/// positional count is the wrong addressing scheme for a concurrent server
/// (client A's `undo(1)` could otherwise destroy client B's write), so
/// `LastN` defaults to `Session` and `Global` requires `Admin`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UndoScope {
    /// Only the calling session's own transactions.
    Session,
    /// Any transaction against the target collection.
    Collection,
    /// Any transaction cluster-wide; requires `Admin`.
    Global,
}

/// A point in history to revert to, resolved down to one or more `TxnId`s before applying.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RevertPoint {
    /// Revert everything after this timestamp.
    Timestamp(DateTime<Utc>),
    /// Revert everything after this log sequence number.
    Lsn(Lsn),
    /// Revert everything after this transaction.
    Txn(TxnId),
}

/// The unit of undo is the transaction, not a positional count — see
/// `UndoScope`. `Txn` is canonical and unambiguous; the others are
/// convenience sugar resolved down to one or more `TxnId`s before storage
/// applies anything.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UndoTarget {
    /// Revert exactly one transaction.
    Txn(TxnId),
    /// Revert the most recent version(s) of a specific document.
    Document {
        /// Key of the document to revert.
        doc_key: String,
        /// Number of most recent versions to revert.
        versions: u32,
    },
    /// Revert the last `n` transactions within a scope.
    LastN {
        /// Number of most recent transactions to revert.
        n: u32,
        /// What set of transactions `n` counts over.
        scope: UndoScope,
    },
    /// Revert everything after a given point in history.
    ToPoint(RevertPoint),
}

/// One entry in the `TxnIndex` — enough to render `mara history` and to
/// drive write-write conflict detection without re-reading the WAL.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct TxnEntry {
    /// Unique identifier of the transaction.
    pub txn_id: TxnId,
    /// Session the transaction was made under.
    pub session: SessionId,
    /// Principal that made the transaction.
    pub principal: PrincipalId,
    /// When the transaction committed.
    pub ts: DateTime<Utc>,
    /// The range of LSNs the transaction's WAL entries span.
    pub lsn_range: (Lsn, Lsn),
    /// Collection the transaction wrote to.
    pub coll: String,
    /// Short human-readable summary of the operation.
    pub op_summary: String,
    /// Rows touched by this transaction.
    pub affected_rows: RoaringBitmap,
    /// Documents touched by this transaction.
    pub doc_ids: Vec<DocId>,
    /// The transaction that undid this one, if any.
    pub undone_by: Option<TxnId>,
}

/// A single row-level conflict reported when an undo target's rows were
/// touched by a later transaction (write-write conflict detection).
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Conflict {
    /// The later transaction that touched the same rows.
    pub conflicting_txn: TxnId,
    /// When the conflicting transaction committed.
    pub ts: DateTime<Utc>,
    /// Principal that made the conflicting transaction.
    pub principal: PrincipalId,
    /// Short human-readable summary of the conflicting operation.
    pub op_summary: String,
    /// Number of rows in common with the undo target.
    pub row_count: u64,
}

/// Short summary of one transaction reversed (or attempted) by an undo,
/// returned to the caller in `Response::UndoSummary`.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct TxnSummary {
    /// Identifier of the transaction that was reversed.
    pub txn_id: TxnId,
    /// Short human-readable summary of the original operation.
    pub op_summary: String,
    /// Number of rows actually reverted.
    pub rows_reversed: u64,
}
