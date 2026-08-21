use serde::{Deserialize, Serialize};

/// A typed scalar value usable in a [`Filter`] predicate or a payload field.
/// `DateTime` is epoch-milliseconds, matching the payload store's `i64`
/// representation for `datetime` fields.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scalar {
    I64(i64),
    F64(f64),
    Str(String),
    Bool(bool),
    DateTime(i64),
}

/// Filter AST shared by the CLI parser, the HTTP API, and native clients, so
/// all three speak one predicate language. Compiled inside `mara-storage`
/// against the payload indexes into a `FilterMask` — the vector and lexical
/// indexes never see this type or any JSON, only the compiled bitmap.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Filter {
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
    Eq {
        field: String,
        value: Scalar,
    },
    In {
        field: String,
        values: Vec<Scalar>,
    },
    Range {
        field: String,
        gte: Option<Scalar>,
        lte: Option<Scalar>,
        gt: Option<Scalar>,
        lt: Option<Scalar>,
    },
    Exists {
        field: String,
    },
    /// Sugar over the implicit `doc_id` field.
    DocIn {
        doc_keys: Vec<String>,
    },
    /// Resolved via BM25 posting lists, not the columnar payload store.
    TextMatch {
        field: String,
        terms: Vec<String>,
    },
}
