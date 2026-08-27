use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use utoipa::ToSchema;

/// A single payload field's value, as carried on the wire and in the WAL
/// record's `fields` object. `mara-storage`'s payload store compiles these
/// into typed dictionary-encoded columns on apply; this type is the common
/// input shape every client speaks, not the storage representation.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum PayloadValue {
    /// A single exact-match string, filterable but not searched.
    Keyword(String),
    /// A list of exact-match strings.
    KeywordList(Vec<String>),
    /// A signed 64-bit integer.
    I64(i64),
    /// A 64-bit floating point number.
    F64(f64),
    /// Epoch milliseconds.
    DateTime(i64),
    /// A boolean.
    Bool(bool),
    /// BM25-searchable; not stored columnar — see the payload schema table.
    Text(String),
}

/// The typed, schema-checked fields of one row or document. Keyed by field
/// name; a field absent from the map is a first-class "not present" state,
/// distinct from any sentinel value.
pub type PayloadRow = BTreeMap<String, PayloadValue>;

/// Un-indexed free-form JSON, written out-of-line to `payload-blob.dat` and
/// retrievable but never filterable.
pub type ExtraPayload = serde_json::Value;
