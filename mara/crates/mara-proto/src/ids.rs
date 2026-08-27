use serde::{Deserialize, Serialize};
use std::fmt;
use utoipa::ToSchema;
use uuid::Uuid;

/// Dense, monotonic, per-collection row identifier. Never reused, even after
/// a delete — that's what lets tombstone bitmaps and `last_txn` conflict
/// detection stay correct without a compaction pass.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize, ToSchema)]
pub struct RowId(pub u64);

impl fmt::Display for RowId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl RowId {
    /// `RoaringBitmap` (used throughout storage and the index layers for
    /// tombstones, per-cluster, and per-document row sets) only indexes
    /// `u32`. That holds at this project's stated scale (at most a few
    /// million rows per collection, vs. `u32::MAX` ≈ 4.29 billion) — this
    /// panics rather than silently wrapping if that assumption is ever
    /// violated, since a silent wrap would corrupt every bitmap-backed
    /// index in a way that's very hard to notice after the fact.
    pub fn to_bitmap_index(self) -> u32 {
        u32::try_from(self.0)
            .expect("RowId exceeded u32::MAX — RoaringBitmap-based indexes need a redesign at this scale")
    }

    /// Reconstructs a `RowId` from a `RoaringBitmap` index value.
    pub fn from_bitmap_index(v: u32) -> Self {
        RowId(v as u64)
    }
}

/// Dense, monotonic, per-collection document identifier. Never reused.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize, ToSchema)]
pub struct DocId(pub u64);

impl fmt::Display for DocId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Identifies a transaction — the unit of undo. Wraps a `Uuid` rather than a
/// counter so transaction ids are safe to mint independently on leader and
/// follower without coordination.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct TxnId(pub Uuid);

impl fmt::Display for TxnId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TxnId {
    /// Mints a fresh, random transaction id.
    pub fn new() -> Self {
        TxnId(Uuid::new_v4())
    }
}

impl Default for TxnId {
    fn default() -> Self {
        Self::new()
    }
}

/// A connection-scoped session id (e.g. `"cli-7f21"`), used to scope
/// session-local `undo --n` and to attribute audit/WAL entries.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct SessionId(pub String);

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Stable identifier for a principal (e.g. `"p_7f21a0"`). Distinct from the
/// principal's display `name`, which is mutable.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct PrincipalId(pub String);

impl fmt::Display for PrincipalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
