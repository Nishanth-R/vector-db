use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;

/// A Postgres-style log sequence number: a `(segment_id, byte_offset)` pair
/// packed into a `u64`. Not a record counter — comparisons are `>`/`<` only,
/// and `lsn + 1` must never be assumed to be "the next record".
///
/// Packing `segment_id` into the high 32 bits and `byte_offset` into the low
/// 32 bits keeps the type trivially `Ord` (segment first, then offset within
/// it) while staying a single machine word. This caps `wal.segment_size_mb`
/// at 4 GiB and total addressable WAL at 2^32 segments — validated at config
/// load in `mara-storage`, not here.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct Lsn(u64);

impl Lsn {
    /// The smallest possible LSN, at segment 0 offset 0.
    pub const ZERO: Lsn = Lsn(0);

    /// Packs a segment id and byte offset into an `Lsn`.
    pub fn new(segment_id: u32, byte_offset: u32) -> Self {
        Lsn(((segment_id as u64) << 32) | byte_offset as u64)
    }

    /// The WAL segment number component.
    pub fn segment_id(&self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// The byte offset within the segment.
    pub fn byte_offset(&self) -> u32 {
        (self.0 & 0xFFFF_FFFF) as u32
    }

    /// The raw packed `u64` representation.
    pub fn as_u64(&self) -> u64 {
        self.0
    }

    /// Reconstructs an `Lsn` from its raw packed `u64` representation.
    pub fn from_u64(v: u64) -> Self {
        Lsn(v)
    }
}

impl fmt::Display for Lsn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:x}/{:X}", self.segment_id(), self.byte_offset())
    }
}

/// Errors parsing an `Lsn` from its `"segment/offset"` string form.
#[derive(Debug, thiserror::Error)]
pub enum LsnParseError {
    /// The string wasn't in `"segment/offset"` form at all.
    #[error("invalid LSN format {0:?}, expected \"segment/offset\" hex")]
    BadFormat(String),
    /// The segment part wasn't valid hex.
    #[error("invalid LSN segment hex {0:?}")]
    BadSegment(String),
    /// The offset part wasn't valid hex.
    #[error("invalid LSN offset hex {0:?}")]
    BadOffset(String),
}

impl FromStr for Lsn {
    type Err = LsnParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (seg, off) = s
            .split_once('/')
            .ok_or_else(|| LsnParseError::BadFormat(s.to_string()))?;
        let segment_id = u32::from_str_radix(seg, 16)
            .map_err(|_| LsnParseError::BadSegment(seg.to_string()))?;
        let byte_offset = u32::from_str_radix(off, 16)
            .map_err(|_| LsnParseError::BadOffset(off.to_string()))?;
        Ok(Lsn::new(segment_id, byte_offset))
    }
}

// Serialized as the "segment/offset" string form everywhere (WAL JSONL,
// audit JSONL, CLI, protocol) — never as a bare integer — so an operator can
// `grep`/`jq` an LSN and get the same shape shown in `mara history`.
impl Serialize for Lsn {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Lsn {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Lsn::from_str(&s).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_matches_postgres_style() {
        let lsn = Lsn::new(0, 0x1A2B3C);
        assert_eq!(lsn.to_string(), "0/1A2B3C");
    }

    #[test]
    fn round_trips_through_string() {
        let lsn = Lsn::new(7, 0xDEADBEEF);
        let s = lsn.to_string();
        let parsed: Lsn = s.parse().unwrap();
        assert_eq!(lsn, parsed);
    }

    #[test]
    fn round_trips_through_serde_json() {
        let lsn = Lsn::new(1, 42);
        let json = serde_json::to_string(&lsn).unwrap();
        assert_eq!(json, "\"1/2A\"");
        let back: Lsn = serde_json::from_str(&json).unwrap();
        assert_eq!(lsn, back);
    }

    #[test]
    fn ordering_is_segment_major() {
        let a = Lsn::new(0, u32::MAX);
        let b = Lsn::new(1, 0);
        assert!(a < b, "segment boundary must order above any offset in the prior segment");
    }

    #[test]
    fn rejects_malformed_strings() {
        assert!("not-an-lsn".parse::<Lsn>().is_err());
        assert!("zz/10".parse::<Lsn>().is_err());
        assert!("0/zz".parse::<Lsn>().is_err());
    }
}
