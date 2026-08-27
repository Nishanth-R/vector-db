use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// A typed scalar value usable in a [`Filter`] predicate or a payload field.
/// `DateTime` is epoch-milliseconds, matching the payload store's `i64`
/// representation for `datetime` fields.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum Scalar {
    /// A signed 64-bit integer.
    I64(i64),
    /// A 64-bit floating point number.
    F64(f64),
    /// A UTF-8 string.
    Str(String),
    /// A boolean.
    Bool(bool),
    /// A timestamp, in epoch milliseconds.
    DateTime(i64),
}

/// Filter AST shared by the CLI parser, the HTTP API, and native clients, so
/// all three speak one predicate language. Compiled inside `mara-storage`
/// against the payload indexes into a `FilterMask` — the vector and lexical
/// indexes never see this type or any JSON, only the compiled bitmap.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum Filter {
    // `no_recursion` breaks utoipa's schema-collection cycle on this
    // self-referential type (`Filter` containing `Filter`) — omitting it
    // is a compile-time-silent, runtime stack overflow inside
    // `ApiDoc::openapi()`, not a type error, so don't drop it while
    // refactoring this enum. https://github.com/juhaku/utoipa/issues/1134
    /// Matches when all sub-filters match.
    #[schema(no_recursion)]
    And(Vec<Filter>),
    /// Matches when any sub-filter matches.
    #[schema(no_recursion)]
    Or(Vec<Filter>),
    /// Matches when the inner filter does not match.
    #[schema(no_recursion)]
    Not(Box<Filter>),
    /// Matches when the field equals a value exactly.
    Eq {
        /// Payload field to compare.
        field: String,
        /// Value the field must equal.
        value: Scalar,
    },
    /// Matches when the field's value is one of a set.
    In {
        /// Payload field to compare.
        field: String,
        /// Set of values the field may match.
        values: Vec<Scalar>,
    },
    /// Matches when the field falls within (optional) bounds.
    Range {
        /// Payload field to compare.
        field: String,
        /// Inclusive lower bound, if any.
        gte: Option<Scalar>,
        /// Inclusive upper bound, if any.
        lte: Option<Scalar>,
        /// Exclusive lower bound, if any.
        gt: Option<Scalar>,
        /// Exclusive upper bound, if any.
        lt: Option<Scalar>,
    },
    /// Matches when the field is present on the document.
    Exists {
        /// Payload field that must be present.
        field: String,
    },
    /// Sugar over the implicit `doc_id` field.
    DocIn {
        /// Document keys to match against.
        doc_keys: Vec<String>,
    },
    /// Resolved via BM25 posting lists, not the columnar payload store.
    TextMatch {
        /// Text field to search.
        field: String,
        /// Terms that must match in the field.
        terms: Vec<String>,
    },
}
