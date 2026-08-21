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
    Session,
    Collection,
    Global,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RevertPoint {
    Timestamp(DateTime<Utc>),
    Lsn(Lsn),
    Txn(TxnId),
}

/// The unit of undo is the transaction, not a positional count — see
/// `UndoScope`. `Txn` is canonical and unambiguous; the others are
/// convenience sugar resolved down to one or more `TxnId`s before storage
/// applies anything.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UndoTarget {
    Txn(TxnId),
    Document { doc_key: String, versions: u32 },
    LastN { n: u32, scope: UndoScope },
    ToPoint(RevertPoint),
}

/// One entry in the `TxnIndex` — enough to render `mara history` and to
/// drive write-write conflict detection without re-reading the WAL.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct TxnEntry {
    pub txn_id: TxnId,
    pub session: SessionId,
    pub principal: PrincipalId,
    pub ts: DateTime<Utc>,
    pub lsn_range: (Lsn, Lsn),
    pub coll: String,
    pub op_summary: String,
    pub affected_rows: RoaringBitmap,
    pub doc_ids: Vec<DocId>,
    pub undone_by: Option<TxnId>,
}

/// A single row-level conflict reported when an undo target's rows were
/// touched by a later transaction (write-write conflict detection).
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Conflict {
    pub conflicting_txn: TxnId,
    pub ts: DateTime<Utc>,
    pub principal: PrincipalId,
    pub op_summary: String,
    pub row_count: u64,
}

/// Short summary of one transaction reversed (or attempted) by an undo,
/// returned to the caller in `Response::UndoSummary`.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct TxnSummary {
    pub txn_id: TxnId,
    pub op_summary: String,
    pub rows_reversed: u64,
}
