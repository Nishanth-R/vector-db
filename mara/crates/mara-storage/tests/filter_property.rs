//! Property test: `compile_filter`'s compiled `RoaringBitmap` must always
//! agree with a naive linear scan over the same rows' ground-truth payload
//! fields — the check the master plan calls for explicitly in the payload
//! store's build step. The compiled path and the oracle are implemented
//! completely independently (bitmap/dictionary machinery vs. a
//! straightforward per-row boolean evaluator) so a bug in one is unlikely
//! to be masked by the same bug in the other.

use mara_proto::{DistanceMetric, Filter, PayloadRow, PayloadValue, Principal, PrincipalId, RequestCtx, Role, Scalar, SessionId, Source};
use mara_storage::{Collection, FieldType, PayloadSchema};
use proptest::prelude::*;
use roaring::RoaringBitmap;

fn schema() -> PayloadSchema {
    PayloadSchema::builder()
        .field("tag", FieldType::Keyword)
        .field("score", FieldType::I64)
        .field("active", FieldType::Bool)
        .build()
}

fn ctx() -> RequestCtx {
    RequestCtx::new(
        SessionId("prop".into()),
        Principal {
            id: PrincipalId("p_prop".into()),
            name: "prop".into(),
            role: Role::Writer,
        },
        Source::Embedded,
    )
}

fn row_strategy() -> impl Strategy<Value = PayloadRow> {
    (
        prop::option::of("[a-c]"),
        prop::option::of(-3i64..3i64),
        prop::option::of(any::<bool>()),
    )
        .prop_map(|(tag, score, active)| {
            let mut row = PayloadRow::new();
            if let Some(t) = tag {
                row.insert("tag".into(), PayloadValue::Keyword(t));
            }
            if let Some(s) = score {
                row.insert("score".into(), PayloadValue::I64(s));
            }
            if let Some(a) = active {
                row.insert("active".into(), PayloadValue::Bool(a));
            }
            row
        })
}

fn leaf_filter_strategy() -> impl Strategy<Value = Filter> {
    prop_oneof![
        "[a-c]".prop_map(|t| Filter::Eq {
            field: "tag".into(),
            value: Scalar::Str(t)
        }),
        prop::collection::vec("[a-c]", 1..3).prop_map(|ts| Filter::In {
            field: "tag".into(),
            values: ts.into_iter().map(Scalar::Str).collect()
        }),
        (-3i64..3i64).prop_map(|v| Filter::Eq {
            field: "score".into(),
            value: Scalar::I64(v)
        }),
        (-3i64..3i64, -3i64..3i64).prop_map(|(a, b)| {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            Filter::Range {
                field: "score".into(),
                gte: Some(Scalar::I64(lo)),
                lte: Some(Scalar::I64(hi)),
                gt: None,
                lt: None,
            }
        }),
        any::<bool>().prop_map(|b| Filter::Eq {
            field: "active".into(),
            value: Scalar::Bool(b)
        }),
        prop_oneof![Just("tag"), Just("score"), Just("active")].prop_map(|f| Filter::Exists { field: f.into() }),
    ]
}

fn filter_strategy() -> impl Strategy<Value = Filter> {
    leaf_filter_strategy().prop_recursive(4, 32, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 1..4).prop_map(Filter::And),
            prop::collection::vec(inner.clone(), 1..4).prop_map(Filter::Or),
            inner.prop_map(|f| Filter::Not(Box::new(f))),
        ]
    })
}

fn scalar_matches(value: &PayloadValue, scalar: &Scalar) -> bool {
    match (value, scalar) {
        (PayloadValue::Keyword(s), Scalar::Str(t)) => s == t,
        (PayloadValue::I64(v), Scalar::I64(t)) => v == t,
        (PayloadValue::Bool(v), Scalar::Bool(t)) => v == t,
        _ => false,
    }
}

fn range_matches(value: &PayloadValue, gte: &Option<Scalar>, lte: &Option<Scalar>, gt: &Option<Scalar>, lt: &Option<Scalar>) -> bool {
    let v = match value {
        PayloadValue::I64(v) => *v,
        _ => return false,
    };
    if let Some(Scalar::I64(b)) = gte
        && v < *b {
            return false;
        }
    if let Some(Scalar::I64(b)) = lte
        && v > *b {
            return false;
        }
    if let Some(Scalar::I64(b)) = gt
        && v <= *b {
            return false;
        }
    if let Some(Scalar::I64(b)) = lt
        && v >= *b {
            return false;
        }
    true
}

/// The oracle: evaluates `filter` against one row's fields directly,
/// with no bitmaps, no dictionaries, no columnar anything.
fn eval_row(filter: &Filter, fields: &PayloadRow) -> bool {
    match filter {
        Filter::And(fs) => fs.iter().all(|f| eval_row(f, fields)),
        Filter::Or(fs) => fs.iter().any(|f| eval_row(f, fields)),
        Filter::Not(f) => !eval_row(f, fields),
        Filter::Eq { field, value } => fields.get(field).is_some_and(|v| scalar_matches(v, value)),
        Filter::In { field, values } => fields
            .get(field)
            .is_some_and(|v| values.iter().any(|sv| scalar_matches(v, sv))),
        Filter::Range { field, gte, lte, gt, lt } => fields.get(field).is_some_and(|v| range_matches(v, gte, lte, gt, lt)),
        Filter::Exists { field } => fields.contains_key(field),
        Filter::DocIn { .. } | Filter::TextMatch { .. } => false,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn compiled_filter_matches_naive_linear_scan(
        rows in prop::collection::vec(row_strategy(), 0..30),
        filter in filter_strategy(),
    ) {
        let c = Collection::with_schema("prop", 1, DistanceMetric::Cosine, schema(), true);
        let ctx = ctx();
        let mut universe = Vec::new();
        for (i, fields) in rows.into_iter().enumerate() {
            let key = format!("k{i}");
            let row_id = c.put(&ctx, &key, vec![0.0], fields.clone(), None).unwrap();
            universe.push((row_id, fields));
        }

        let compiled = c.compile_filter(&filter).unwrap();

        let mut oracle = RoaringBitmap::new();
        for (row_id, fields) in &universe {
            if eval_row(&filter, fields) {
                oracle.insert(row_id.to_bitmap_index());
            }
        }

        prop_assert_eq!(compiled.allowed, oracle);
    }

    #[test]
    fn deletes_are_excluded_from_every_filter_result_including_the_trivially_true_one(
        rows in prop::collection::vec(row_strategy(), 1..20),
        delete_every_other in any::<bool>(),
    ) {
        let c = Collection::with_schema("prop-del", 1, DistanceMetric::Cosine, schema(), true);
        let ctx = ctx();
        let mut ids = Vec::new();
        for (i, fields) in rows.into_iter().enumerate() {
            let key = format!("k{i}");
            ids.push(c.put(&ctx, &key, vec![0.0], fields, None).unwrap());
        }
        // A tautology: true for every row regardless of whether "tag" is
        // set, so the only way a row can be absent from its result is
        // tombstone exclusion, not the filter's own semantics.
        let tautology = Filter::Or(vec![
            Filter::Exists { field: "tag".into() },
            Filter::Not(Box::new(Filter::Exists { field: "tag".into() })),
        ]);
        if delete_every_other {
            for (i, id) in ids.iter().enumerate() {
                if i % 2 == 0 {
                    c.delete(&ctx, &format!("k{i}")).unwrap();
                    let mask = c.compile_filter(&tautology).unwrap();
                    prop_assert!(!mask.contains(*id), "a deleted row must never satisfy any filter, including a tautology it matched before deletion");
                }
            }
        }
    }
}
