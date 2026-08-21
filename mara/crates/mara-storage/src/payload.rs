//! The typed payload schema, columnar dictionary-encoded store, and filter
//! compiler (master plan Layer 2, *Payload store*). Replaces the
//! free-form-JSON-per-row design the review flagged: every filterable
//! field is a primitive in a dense column or a bitmap, so `compile_filter`
//! never touches heap-allocated JSON and produces nothing but a
//! `RoaringBitmap` — the vector and lexical indexes never see a `Filter` or
//! any JSON, only that bitmap.
//!
//! The row-level ground truth (`RowRecord::fields`, a plain
//! `PayloadRow`) is left in place rather than replaced outright — it's
//! exactly what the filter compiler is property-tested against as a naive
//! linear-scan oracle (see the `proptest` module below), and it's what
//! `get_by_key`/`scan`/`list_chunks` already return.

use mara_proto::{Filter, PayloadRow, PayloadValue, RowId, Scalar};
use roaring::RoaringBitmap;
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    Keyword,
    KeywordList,
    I64,
    F64,
    DateTime,
    Bool,
    /// BM25-searchable; not stored columnar here — see the payload schema
    /// table in the master plan. Presence is still tracked so `Exists`
    /// works, but the text itself lives only in the row's naive fields map
    /// until `mara-index-bm25` exists.
    Text,
}

impl FieldType {
    fn matches(&self, value: &PayloadValue) -> bool {
        matches!(
            (self, value),
            (FieldType::Keyword, PayloadValue::Keyword(_))
                | (FieldType::KeywordList, PayloadValue::KeywordList(_))
                | (FieldType::I64, PayloadValue::I64(_))
                | (FieldType::F64, PayloadValue::F64(_))
                | (FieldType::DateTime, PayloadValue::DateTime(_))
                | (FieldType::Bool, PayloadValue::Bool(_))
                | (FieldType::Text, PayloadValue::Text(_))
        )
    }
}

/// A collection's declared payload fields, in insertion order. Extensible
/// via `alter_schema` (`add_field` appends; `drop_field` removes — changing
/// a field's type requires a collection rebuild, not supported here).
#[derive(Clone, Default, Debug)]
pub struct PayloadSchema {
    fields: Vec<(String, FieldType)>,
}

impl PayloadSchema {
    pub fn empty() -> Self {
        PayloadSchema { fields: Vec::new() }
    }

    pub fn builder() -> PayloadSchemaBuilder {
        PayloadSchemaBuilder { fields: Vec::new() }
    }

    pub fn field_type(&self, name: &str) -> Option<FieldType> {
        self.fields.iter().find(|(n, _)| n == name).map(|(_, t)| *t)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.fields.iter().any(|(n, _)| n == name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &(String, FieldType)> {
        self.fields.iter()
    }
}

pub struct PayloadSchemaBuilder {
    fields: Vec<(String, FieldType)>,
}

impl PayloadSchemaBuilder {
    pub fn field(mut self, name: impl Into<String>, ty: FieldType) -> Self {
        self.fields.push((name.into(), ty));
        self
    }

    pub fn build(self) -> PayloadSchema {
        PayloadSchema { fields: self.fields }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum FilterError {
    #[error("unknown payload field {0:?} — not declared in the collection's schema")]
    UnknownField(String),
    #[error("field {field:?} is type {actual}, this filter needs {expected}")]
    TypeMismatch {
        field: String,
        expected: &'static str,
        actual: &'static str,
    },
    #[error("TextMatch filtering requires the BM25 index, not available yet")]
    TextMatchUnsupported,
    #[error("field {0:?} already exists in the schema")]
    FieldAlreadyExists(String),
}

pub type FilterResult<T> = Result<T, FilterError>;

/// The compiled result of a `Filter`: a bitmap over live `RowId`s (as
/// `u32` bitmap indices — see `RowId::to_bitmap_index`) plus its
/// cardinality, cheap since `RoaringBitmap::len()` is O(1). This is what
/// reaches the vector and lexical indexes; they never see the `Filter` AST
/// or any JSON.
#[derive(Clone, Debug)]
pub struct FilterMask {
    pub allowed: RoaringBitmap,
    pub estimated_cardinality: u64,
}

impl FilterMask {
    pub fn contains(&self, row_id: RowId) -> bool {
        self.allowed.contains(row_id.to_bitmap_index())
    }
}

fn scalar_type_name(s: &Scalar) -> &'static str {
    match s {
        Scalar::I64(_) => "i64",
        Scalar::F64(_) => "f64",
        Scalar::Str(_) => "str",
        Scalar::Bool(_) => "bool",
        Scalar::DateTime(_) => "datetime",
    }
}

fn value_type_name(v: &PayloadValue) -> &'static str {
    match v {
        PayloadValue::Keyword(_) => "keyword",
        PayloadValue::KeywordList(_) => "keyword[]",
        PayloadValue::I64(_) => "i64",
        PayloadValue::F64(_) => "f64",
        PayloadValue::DateTime(_) => "datetime",
        PayloadValue::Bool(_) => "bool",
        PayloadValue::Text(_) => "text",
    }
}

/// Maps a non-negative-preserving-order `f64` to a `u64` whose unsigned
/// ordering matches the float's total ordering (the standard
/// flip-sign-then-maybe-invert trick), so range queries over `f64` payload
/// values can reuse the same `BTreeMap::range` machinery as `i64`. NaN
/// payload values are not supported (undefined ordering); nothing in the
/// write path rejects them explicitly, but filtering behavior against a
/// NaN is unspecified.
fn f64_sort_key(f: f64) -> u64 {
    let bits = f.to_bits();
    if (bits >> 63) == 1 {
        !bits
    } else {
        bits | (1u64 << 63)
    }
}

struct Dictionary {
    code_to_value: Vec<String>,
    value_to_code: HashMap<String, u32>,
}

impl Dictionary {
    fn new() -> Self {
        Dictionary {
            code_to_value: Vec::new(),
            value_to_code: HashMap::new(),
        }
    }

    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&code) = self.value_to_code.get(s) {
            return code;
        }
        let code = self.code_to_value.len() as u32;
        self.code_to_value.push(s.to_string());
        self.value_to_code.insert(s.to_string(), code);
        code
    }

    fn lookup(&self, s: &str) -> Option<u32> {
        self.value_to_code.get(s).copied()
    }
}

enum Column {
    Keyword {
        dict: Dictionary,
        values: HashMap<RowId, u32>,
        index: HashMap<u32, RoaringBitmap>,
        presence: RoaringBitmap,
    },
    KeywordList {
        dict: Dictionary,
        values: HashMap<RowId, Vec<u32>>,
        index: HashMap<u32, RoaringBitmap>,
        presence: RoaringBitmap,
    },
    I64 {
        values: HashMap<RowId, i64>,
        sorted: BTreeMap<i64, RoaringBitmap>,
        presence: RoaringBitmap,
    },
    F64 {
        values: HashMap<RowId, f64>,
        sorted: BTreeMap<u64, RoaringBitmap>,
        presence: RoaringBitmap,
    },
    DateTime {
        values: HashMap<RowId, i64>,
        sorted: BTreeMap<i64, RoaringBitmap>,
        presence: RoaringBitmap,
    },
    Bool {
        set: RoaringBitmap,
        presence: RoaringBitmap,
    },
    Text {
        presence: RoaringBitmap,
    },
}

impl Column {
    fn new(ty: FieldType) -> Self {
        match ty {
            FieldType::Keyword => Column::Keyword {
                dict: Dictionary::new(),
                values: HashMap::new(),
                index: HashMap::new(),
                presence: RoaringBitmap::new(),
            },
            FieldType::KeywordList => Column::KeywordList {
                dict: Dictionary::new(),
                values: HashMap::new(),
                index: HashMap::new(),
                presence: RoaringBitmap::new(),
            },
            FieldType::I64 => Column::I64 {
                values: HashMap::new(),
                sorted: BTreeMap::new(),
                presence: RoaringBitmap::new(),
            },
            FieldType::F64 => Column::F64 {
                values: HashMap::new(),
                sorted: BTreeMap::new(),
                presence: RoaringBitmap::new(),
            },
            FieldType::DateTime => Column::DateTime {
                values: HashMap::new(),
                sorted: BTreeMap::new(),
                presence: RoaringBitmap::new(),
            },
            FieldType::Bool => Column::Bool {
                set: RoaringBitmap::new(),
                presence: RoaringBitmap::new(),
            },
            FieldType::Text => Column::Text {
                presence: RoaringBitmap::new(),
            },
        }
    }

    fn type_name(&self) -> &'static str {
        match self {
            Column::Keyword { .. } => "keyword",
            Column::KeywordList { .. } => "keyword[]",
            Column::I64 { .. } => "i64",
            Column::F64 { .. } => "f64",
            Column::DateTime { .. } => "datetime",
            Column::Bool { .. } => "bool",
            Column::Text { .. } => "text",
        }
    }

    fn presence(&self) -> &RoaringBitmap {
        match self {
            Column::Keyword { presence, .. }
            | Column::KeywordList { presence, .. }
            | Column::I64 { presence, .. }
            | Column::F64 { presence, .. }
            | Column::DateTime { presence, .. }
            | Column::Bool { presence, .. }
            | Column::Text { presence, .. } => presence,
        }
    }

    /// Removes any value this row previously held for this column —
    /// idempotent, safe to call whether or not the row had a value.
    fn clear(&mut self, row_id: RowId) {
        let idx = row_id.to_bitmap_index();
        match self {
            Column::Keyword { values, index, presence, .. } => {
                if let Some(code) = values.remove(&row_id)
                    && let Some(bm) = index.get_mut(&code) {
                        bm.remove(idx);
                        if bm.is_empty() {
                            index.remove(&code);
                        }
                    }
                presence.remove(idx);
            }
            Column::KeywordList { values, index, presence, .. } => {
                if let Some(codes) = values.remove(&row_id) {
                    for code in codes {
                        if let Some(bm) = index.get_mut(&code) {
                            bm.remove(idx);
                            if bm.is_empty() {
                                index.remove(&code);
                            }
                        }
                    }
                }
                presence.remove(idx);
            }
            Column::I64 { values, sorted, presence } => {
                if let Some(v) = values.remove(&row_id)
                    && let Some(bm) = sorted.get_mut(&v) {
                        bm.remove(idx);
                        if bm.is_empty() {
                            sorted.remove(&v);
                        }
                    }
                presence.remove(idx);
            }
            Column::F64 { values, sorted, presence } => {
                if let Some(v) = values.remove(&row_id) {
                    let key = f64_sort_key(v);
                    if let Some(bm) = sorted.get_mut(&key) {
                        bm.remove(idx);
                        if bm.is_empty() {
                            sorted.remove(&key);
                        }
                    }
                }
                presence.remove(idx);
            }
            Column::DateTime { values, sorted, presence } => {
                if let Some(v) = values.remove(&row_id)
                    && let Some(bm) = sorted.get_mut(&v) {
                        bm.remove(idx);
                        if bm.is_empty() {
                            sorted.remove(&v);
                        }
                    }
                presence.remove(idx);
            }
            Column::Bool { set, presence } => {
                set.remove(idx);
                presence.remove(idx);
            }
            Column::Text { presence } => {
                presence.remove(idx);
            }
        }
    }

    /// Sets this row's value, first clearing any prior value so `set` is
    /// safe to call on both a fresh row and an in-place update.
    fn set(&mut self, field: &str, row_id: RowId, value: &PayloadValue) -> FilterResult<()> {
        self.clear(row_id);
        let idx = row_id.to_bitmap_index();
        match (&mut *self, value) {
            (Column::Keyword { dict, values, index, presence }, PayloadValue::Keyword(s)) => {
                let code = dict.intern(s);
                values.insert(row_id, code);
                index.entry(code).or_default().insert(idx);
                presence.insert(idx);
            }
            (Column::KeywordList { dict, values, index, presence }, PayloadValue::KeywordList(list)) => {
                let codes: Vec<u32> = list.iter().map(|s| dict.intern(s)).collect();
                for &code in &codes {
                    index.entry(code).or_default().insert(idx);
                }
                values.insert(row_id, codes);
                presence.insert(idx);
            }
            (Column::I64 { values, sorted, presence }, PayloadValue::I64(v)) => {
                values.insert(row_id, *v);
                sorted.entry(*v).or_default().insert(idx);
                presence.insert(idx);
            }
            (Column::F64 { values, sorted, presence }, PayloadValue::F64(v)) => {
                values.insert(row_id, *v);
                sorted.entry(f64_sort_key(*v)).or_default().insert(idx);
                presence.insert(idx);
            }
            (Column::DateTime { values, sorted, presence }, PayloadValue::DateTime(v)) => {
                values.insert(row_id, *v);
                sorted.entry(*v).or_default().insert(idx);
                presence.insert(idx);
            }
            (Column::Bool { set, presence }, PayloadValue::Bool(v)) => {
                if *v {
                    set.insert(idx);
                }
                presence.insert(idx);
            }
            (Column::Text { presence }, PayloadValue::Text(_)) => {
                presence.insert(idx);
            }
            (col, other) => {
                return Err(FilterError::TypeMismatch {
                    field: field.to_string(),
                    expected: col.type_name(),
                    actual: value_type_name(other),
                });
            }
        }
        Ok(())
    }
}

/// The columnar payload store for one collection: a typed schema plus one
/// `Column` per declared field. `strict` gates *write*-time validation
/// (reject a payload naming a field the schema doesn't declare); it's
/// orthogonal to filtering, which always rejects an unknown field outright
/// — there's no column to consult.
pub struct PayloadStore {
    schema: PayloadSchema,
    columns: HashMap<String, Column>,
    strict: bool,
}

impl PayloadStore {
    pub fn new(schema: PayloadSchema, strict: bool) -> Self {
        let columns = schema.iter().map(|(name, ty)| (name.clone(), Column::new(*ty))).collect();
        PayloadStore { schema, columns, strict }
    }

    pub fn schema(&self) -> &PayloadSchema {
        &self.schema
    }

    /// Checks a row's fields against the schema without mutating anything —
    /// the pre-pass callers use to validate an entire batch (or an entire
    /// document's chunks) before mutating *any* row, so a mid-batch
    /// validation failure never leaves the columnar store half-updated.
    pub fn validate_row(&self, fields: &PayloadRow) -> FilterResult<()> {
        if self.strict
            && let Some(unknown) = fields.keys().find(|k| !self.schema.contains(k)) {
                return Err(FilterError::UnknownField(unknown.clone()));
            }
        for (name, ty) in self.schema.iter() {
            if let Some(value) = fields.get(name)
                && !ty.matches(value) {
                    let column = self.columns.get(name).expect("schema and columns must stay in sync");
                    return Err(FilterError::TypeMismatch {
                        field: name.clone(),
                        expected: column.type_name(),
                        actual: value_type_name(value),
                    });
                }
        }
        Ok(())
    }

    /// Indexes a row's fields into the columnar store. Callers that need
    /// batch atomicity must call [`PayloadStore::validate_row`] on every
    /// row in the batch first — by the time `set_row` runs, validation is
    /// assumed to already have passed, so a failure here would indicate a
    /// logic bug rather than bad input.
    pub fn set_row(&mut self, row_id: RowId, fields: &PayloadRow) -> FilterResult<()> {
        self.validate_row(fields)?;
        for (name, _ty) in self.schema.iter() {
            let column = self.columns.get_mut(name).expect("schema and columns must stay in sync");
            match fields.get(name) {
                Some(value) => column.set(name, row_id, value)?,
                None => column.clear(row_id),
            }
        }
        Ok(())
    }

    pub fn clear_row(&mut self, row_id: RowId) {
        for column in self.columns.values_mut() {
            column.clear(row_id);
        }
    }

    pub fn add_field(&mut self, name: impl Into<String>, ty: FieldType) -> FilterResult<()> {
        let name = name.into();
        if self.schema.contains(&name) {
            return Err(FilterError::FieldAlreadyExists(name));
        }
        self.columns.insert(name.clone(), Column::new(ty));
        self.schema.fields.push((name, ty));
        Ok(())
    }

    pub fn drop_field(&mut self, name: &str) -> FilterResult<()> {
        if !self.schema.contains(name) {
            return Err(FilterError::UnknownField(name.to_string()));
        }
        self.schema.fields.retain(|(n, _)| n != name);
        self.columns.remove(name);
        Ok(())
    }

    fn eq_mask(&self, field: &str, value: &Scalar) -> FilterResult<RoaringBitmap> {
        let column = self.columns.get(field).ok_or_else(|| FilterError::UnknownField(field.to_string()))?;
        match (column, value) {
            (Column::Keyword { dict, index, .. }, Scalar::Str(s)) => {
                Ok(dict.lookup(s).and_then(|code| index.get(&code).cloned()).unwrap_or_default())
            }
            (Column::KeywordList { dict, index, .. }, Scalar::Str(s)) => {
                Ok(dict.lookup(s).and_then(|code| index.get(&code).cloned()).unwrap_or_default())
            }
            (Column::I64 { sorted, .. }, Scalar::I64(v)) => Ok(sorted.get(v).cloned().unwrap_or_default()),
            (Column::DateTime { sorted, .. }, Scalar::DateTime(v)) => Ok(sorted.get(v).cloned().unwrap_or_default()),
            (Column::F64 { sorted, .. }, Scalar::F64(v)) => {
                Ok(sorted.get(&f64_sort_key(*v)).cloned().unwrap_or_default())
            }
            (Column::Bool { set, presence }, Scalar::Bool(v)) => {
                Ok(if *v { set.clone() } else { presence - set })
            }
            (col, scalar) => Err(FilterError::TypeMismatch {
                field: field.to_string(),
                expected: col.type_name(),
                actual: scalar_type_name(scalar),
            }),
        }
    }

    fn range_mask(
        &self,
        field: &str,
        gte: Option<&Scalar>,
        lte: Option<&Scalar>,
        gt: Option<&Scalar>,
        lt: Option<&Scalar>,
    ) -> FilterResult<RoaringBitmap> {
        let column = self.columns.get(field).ok_or_else(|| FilterError::UnknownField(field.to_string()))?;
        use std::ops::Bound;

        fn bounds<K: Ord + Copy>(
            gte: Option<K>,
            lte: Option<K>,
            gt: Option<K>,
            lt: Option<K>,
        ) -> (Bound<K>, Bound<K>) {
            let lower = match (gte, gt) {
                (Some(g), _) => Bound::Included(g),
                (None, Some(g)) => Bound::Excluded(g),
                (None, None) => Bound::Unbounded,
            };
            let upper = match (lte, lt) {
                (Some(l), _) => Bound::Included(l),
                (None, Some(l)) => Bound::Excluded(l),
                (None, None) => Bound::Unbounded,
            };
            (lower, upper)
        }

        fn extract_i64(s: Option<&Scalar>) -> Option<i64> {
            s.and_then(|s| if let Scalar::I64(v) = s { Some(*v) } else { None })
        }
        fn extract_dt(s: Option<&Scalar>) -> Option<i64> {
            s.and_then(|s| if let Scalar::DateTime(v) = s { Some(*v) } else { None })
        }
        fn extract_f64(s: Option<&Scalar>) -> Option<f64> {
            s.and_then(|s| if let Scalar::F64(v) = s { Some(*v) } else { None })
        }

        match column {
            Column::I64 { sorted, .. } => {
                let (lo, hi) = bounds(extract_i64(gte), extract_i64(lte), extract_i64(gt), extract_i64(lt));
                Ok(sorted.range((lo, hi)).fold(RoaringBitmap::new(), |mut acc, (_, bm)| {
                    acc |= bm;
                    acc
                }))
            }
            Column::DateTime { sorted, .. } => {
                let (lo, hi) = bounds(extract_dt(gte), extract_dt(lte), extract_dt(gt), extract_dt(lt));
                Ok(sorted.range((lo, hi)).fold(RoaringBitmap::new(), |mut acc, (_, bm)| {
                    acc |= bm;
                    acc
                }))
            }
            Column::F64 { sorted, .. } => {
                let (lo, hi) = bounds(
                    extract_f64(gte).map(f64_sort_key),
                    extract_f64(lte).map(f64_sort_key),
                    extract_f64(gt).map(f64_sort_key),
                    extract_f64(lt).map(f64_sort_key),
                );
                Ok(sorted.range((lo, hi)).fold(RoaringBitmap::new(), |mut acc, (_, bm)| {
                    acc |= bm;
                    acc
                }))
            }
            other => Err(FilterError::TypeMismatch {
                field: field.to_string(),
                expected: other.type_name(),
                actual: "range bound (i64/f64/datetime)",
            }),
        }
    }

    fn exists_mask(&self, field: &str) -> FilterResult<RoaringBitmap> {
        self.columns
            .get(field)
            .map(|c| c.presence().clone())
            .ok_or_else(|| FilterError::UnknownField(field.to_string()))
    }

    /// Compiles everything except `Filter::DocIn`, which needs the doc
    /// registry (owned by `Collection`, not `PayloadStore`) — see
    /// `Collection::compile_filter`, which handles `DocIn` itself and
    /// delegates every other variant here.
    pub fn compile(&self, filter: &Filter, universe: &RoaringBitmap) -> FilterResult<RoaringBitmap> {
        match filter {
            Filter::And(fs) => fs.iter().try_fold(universe.clone(), |acc, f| Ok(acc & self.compile(f, universe)?)),
            Filter::Or(fs) => fs.iter().try_fold(RoaringBitmap::new(), |acc, f| Ok(acc | self.compile(f, universe)?)),
            Filter::Not(f) => Ok(universe - self.compile(f, universe)?),
            Filter::Eq { field, value } => self.eq_mask(field, value),
            Filter::In { field, values } => values.iter().try_fold(RoaringBitmap::new(), |acc, v| Ok(acc | self.eq_mask(field, v)?)),
            Filter::Range { field, gte, lte, gt, lt } => {
                self.range_mask(field, gte.as_ref(), lte.as_ref(), gt.as_ref(), lt.as_ref())
            }
            Filter::Exists { field } => self.exists_mask(field),
            Filter::DocIn { .. } => {
                unreachable!("Filter::DocIn is resolved by Collection::compile_filter, not PayloadStore::compile")
            }
            Filter::TextMatch { .. } => Err(FilterError::TextMatchUnsupported),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> PayloadStore {
        PayloadStore::new(
            PayloadSchema::builder()
                .field("tags", FieldType::KeywordList)
                .field("title", FieldType::Keyword)
                .field("published_at", FieldType::DateTime)
                .field("score", FieldType::F64)
                .field("count", FieldType::I64)
                .field("internal", FieldType::Bool)
                .field("body", FieldType::Text)
                .build(),
            true,
        )
    }

    fn row(fields: &[(&str, PayloadValue)]) -> PayloadRow {
        fields.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn eq_on_keyword_finds_exact_matches_only() {
        let mut s = store();
        s.set_row(RowId(0), &row(&[("title", PayloadValue::Keyword("a".into()))])).unwrap();
        s.set_row(RowId(1), &row(&[("title", PayloadValue::Keyword("b".into()))])).unwrap();
        let universe: RoaringBitmap = [0, 1].into_iter().collect();

        let mask = s
            .compile(
                &Filter::Eq {
                    field: "title".into(),
                    value: Scalar::Str("a".into()),
                },
                &universe,
            )
            .unwrap();
        assert_eq!(mask, [0].into_iter().collect());
    }

    #[test]
    fn keyword_list_membership_matches_any_element() {
        let mut s = store();
        s.set_row(
            RowId(0),
            &row(&[("tags", PayloadValue::KeywordList(vec!["rag".into(), "hr".into()]))]),
        )
        .unwrap();
        s.set_row(RowId(1), &row(&[("tags", PayloadValue::KeywordList(vec!["eng".into()]))]))
            .unwrap();
        let universe: RoaringBitmap = [0, 1].into_iter().collect();

        let mask = s
            .compile(
                &Filter::Eq {
                    field: "tags".into(),
                    value: Scalar::Str("rag".into()),
                },
                &universe,
            )
            .unwrap();
        assert_eq!(mask, [0].into_iter().collect());
    }

    #[test]
    fn range_on_datetime_respects_bounds() {
        let mut s = store();
        for (i, ts) in [100i64, 200, 300].into_iter().enumerate() {
            s.set_row(RowId(i as u64), &row(&[("published_at", PayloadValue::DateTime(ts))]))
                .unwrap();
        }
        let universe: RoaringBitmap = [0, 1, 2].into_iter().collect();

        let mask = s
            .compile(
                &Filter::Range {
                    field: "published_at".into(),
                    gte: Some(Scalar::DateTime(150)),
                    lte: None,
                    gt: None,
                    lt: Some(Scalar::DateTime(300)),
                },
                &universe,
            )
            .unwrap();
        assert_eq!(mask, [1].into_iter().collect(), "only 200 is within [150, 300)");
    }

    #[test]
    fn f64_range_survives_negative_and_positive_values() {
        let mut s = store();
        for (i, v) in [-5.5f64, -0.1, 0.0, 3.2].into_iter().enumerate() {
            s.set_row(RowId(i as u64), &row(&[("score", PayloadValue::F64(v))])).unwrap();
        }
        let universe: RoaringBitmap = (0..4).collect();
        let mask = s
            .compile(
                &Filter::Range {
                    field: "score".into(),
                    gte: Some(Scalar::F64(-1.0)),
                    lte: Some(Scalar::F64(3.2)),
                    gt: None,
                    lt: None,
                },
                &universe,
            )
            .unwrap();
        assert_eq!(mask, [1u32, 2, 3].into_iter().collect());
    }

    #[test]
    fn bool_false_is_presence_minus_true_set() {
        let mut s = store();
        s.set_row(RowId(0), &row(&[("internal", PayloadValue::Bool(true))])).unwrap();
        s.set_row(RowId(1), &row(&[("internal", PayloadValue::Bool(false))])).unwrap();
        // Row 2 never sets `internal` at all — absent, not false.
        let universe: RoaringBitmap = [0, 1, 2].into_iter().collect();

        let false_mask = s
            .compile(
                &Filter::Eq {
                    field: "internal".into(),
                    value: Scalar::Bool(false),
                },
                &universe,
            )
            .unwrap();
        assert_eq!(false_mask, [1].into_iter().collect());

        let exists_mask = s.compile(&Filter::Exists { field: "internal".into() }, &universe).unwrap();
        assert_eq!(exists_mask, [0, 1].into_iter().collect());
    }

    #[test]
    fn re_setting_a_row_clears_its_old_index_entry() {
        let mut s = store();
        s.set_row(RowId(0), &row(&[("title", PayloadValue::Keyword("old".into()))])).unwrap();
        s.set_row(RowId(0), &row(&[("title", PayloadValue::Keyword("new".into()))])).unwrap();
        let universe: RoaringBitmap = [0].into_iter().collect();

        let old_mask = s
            .compile(
                &Filter::Eq {
                    field: "title".into(),
                    value: Scalar::Str("old".into()),
                },
                &universe,
            )
            .unwrap();
        assert!(old_mask.is_empty(), "stale index entry must be gone after re-set");

        let new_mask = s
            .compile(
                &Filter::Eq {
                    field: "title".into(),
                    value: Scalar::Str("new".into()),
                },
                &universe,
            )
            .unwrap();
        assert_eq!(new_mask, [0].into_iter().collect());
    }

    #[test]
    fn clear_row_removes_it_from_every_column() {
        let mut s = store();
        s.set_row(
            RowId(0),
            &row(&[
                ("title", PayloadValue::Keyword("a".into())),
                ("count", PayloadValue::I64(5)),
                ("internal", PayloadValue::Bool(true)),
            ]),
        )
        .unwrap();
        s.clear_row(RowId(0));

        let universe: RoaringBitmap = [0].into_iter().collect();
        assert!(s.compile(&Filter::Exists { field: "title".into() }, &universe).unwrap().is_empty());
        assert!(s.compile(&Filter::Exists { field: "count".into() }, &universe).unwrap().is_empty());
        assert!(s.compile(&Filter::Exists { field: "internal".into() }, &universe).unwrap().is_empty());
    }

    #[test]
    fn unknown_field_is_a_clear_error_not_an_empty_result() {
        let s = store();
        let universe = RoaringBitmap::new();
        let err = s
            .compile(
                &Filter::Eq {
                    field: "nope".into(),
                    value: Scalar::Bool(true),
                },
                &universe,
            )
            .unwrap_err();
        assert_eq!(err, FilterError::UnknownField("nope".into()));
    }

    #[test]
    fn type_mismatch_is_rejected_at_write_time() {
        let mut s = store();
        let err = s
            .set_row(RowId(0), &row(&[("count", PayloadValue::Keyword("not-a-number".into()))]))
            .unwrap_err();
        assert!(matches!(err, FilterError::TypeMismatch { .. }));
    }

    #[test]
    fn strict_schema_rejects_undeclared_fields() {
        let mut s = store();
        let err = s.set_row(RowId(0), &row(&[("ghost", PayloadValue::Bool(true))])).unwrap_err();
        assert_eq!(err, FilterError::UnknownField("ghost".into()));
    }

    #[test]
    fn non_strict_schema_tolerates_undeclared_fields_by_ignoring_them() {
        let mut s = PayloadStore::new(PayloadSchema::builder().field("title", FieldType::Keyword).build(), false);
        s.set_row(RowId(0), &row(&[("title", PayloadValue::Keyword("a".into())), ("ghost", PayloadValue::Bool(true))]))
            .unwrap();
        let universe: RoaringBitmap = [0].into_iter().collect();
        assert_eq!(
            s.compile(
                &Filter::Eq {
                    field: "title".into(),
                    value: Scalar::Str("a".into())
                },
                &universe
            )
            .unwrap(),
            [0].into_iter().collect()
        );
    }

    #[test]
    fn text_match_is_explicitly_unsupported_not_silently_wrong() {
        let s = store();
        let universe = RoaringBitmap::new();
        let err = s
            .compile(
                &Filter::TextMatch {
                    field: "body".into(),
                    terms: vec!["hello".into()],
                },
                &universe,
            )
            .unwrap_err();
        assert_eq!(err, FilterError::TextMatchUnsupported);
    }

    #[test]
    fn and_or_not_compose_correctly() {
        let mut s = store();
        s.set_row(
            RowId(0),
            &row(&[("title", PayloadValue::Keyword("a".into())), ("count", PayloadValue::I64(1))]),
        )
        .unwrap();
        s.set_row(
            RowId(1),
            &row(&[("title", PayloadValue::Keyword("a".into())), ("count", PayloadValue::I64(2))]),
        )
        .unwrap();
        s.set_row(
            RowId(2),
            &row(&[("title", PayloadValue::Keyword("b".into())), ("count", PayloadValue::I64(1))]),
        )
        .unwrap();
        let universe: RoaringBitmap = [0, 1, 2].into_iter().collect();

        let and_mask = s
            .compile(
                &Filter::And(vec![
                    Filter::Eq { field: "title".into(), value: Scalar::Str("a".into()) },
                    Filter::Eq { field: "count".into(), value: Scalar::I64(1) },
                ]),
                &universe,
            )
            .unwrap();
        assert_eq!(and_mask, [0].into_iter().collect());

        let or_mask = s
            .compile(
                &Filter::Or(vec![
                    Filter::Eq { field: "title".into(), value: Scalar::Str("b".into()) },
                    Filter::Eq { field: "count".into(), value: Scalar::I64(2) },
                ]),
                &universe,
            )
            .unwrap();
        assert_eq!(or_mask, [1, 2].into_iter().collect());

        let not_mask = s
            .compile(
                &Filter::Not(Box::new(Filter::Eq {
                    field: "title".into(),
                    value: Scalar::Str("a".into()),
                })),
                &universe,
            )
            .unwrap();
        assert_eq!(not_mask, [2].into_iter().collect());
    }

    #[test]
    fn add_field_then_drop_field_round_trips() {
        let mut s = PayloadStore::new(PayloadSchema::empty(), true);
        assert!(s.add_field("size", FieldType::I64).is_ok());
        assert_eq!(s.add_field("size", FieldType::I64).unwrap_err(), FilterError::FieldAlreadyExists("size".into()));

        s.set_row(RowId(0), &row(&[("size", PayloadValue::I64(42))])).unwrap();
        let universe: RoaringBitmap = [0].into_iter().collect();
        assert_eq!(
            s.compile(&Filter::Eq { field: "size".into(), value: Scalar::I64(42) }, &universe).unwrap(),
            [0].into_iter().collect()
        );

        s.drop_field("size").unwrap();
        assert_eq!(
            s.compile(&Filter::Eq { field: "size".into(), value: Scalar::I64(42) }, &universe)
                .unwrap_err(),
            FilterError::UnknownField("size".into())
        );
    }
}
