//! Recall/latency comparison against the `Flat` oracle (master plan step
//! 17: "benchmark against the Flat oracle via criterion to confirm [OPQ]
//! earns its keep on the configured embedding model"). Recall isn't
//! something criterion's timer measures directly, so it's computed once
//! up front and printed via `eprintln!` (outside any `b.iter` closure, so
//! it never gets mistaken for a timed sample); the criterion groups below
//! then measure what recall alone can't show — how much `nprobe`/OPQ's
//! rotation actually cost in search latency for whatever recall they buy.
//!
//! Run with `cargo bench -p mara-index-vector`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mara_index_vector::{FlatIndex, IvfParams, IvfPqIndex, OpqParams, PqParams, SearchParams, VectorIndex};
use mara_proto::{DistanceMetric, PayloadRow, Principal, PrincipalId, RequestCtx, Role, SessionId, Source};
use mara_storage::{Storage, StorageApi};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;

const DIM: usize = 64;
const N_PER_CLUSTER: usize = 400;
const N_CLUSTERS: usize = 20;
const N_QUERIES: usize = 30;
const K: usize = 10;

fn ctx() -> RequestCtx {
    RequestCtx::new(
        SessionId("bench".into()),
        Principal {
            id: PrincipalId("bench".into()),
            name: "bench".into(),
            role: Role::Writer,
        },
        Source::Embedded,
    )
}

/// Synthetic, embedding-shaped data: `N_CLUSTERS` well-separated Gaussian
/// blobs in `DIM`-dimensional space, roughly what sentence-transformer
/// output looks like at a coarse level (clustered by topic, isotropic
/// noise within a topic) — good enough to compare recall/latency shapes
/// without needing a real model or network access in a benchmark run.
fn synthetic_dataset(seed: u64) -> (Storage, Vec<Vec<f32>>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let storage = Storage::new();
    storage
        .create_collection(&ctx(), "bench", DIM, DistanceMetric::L2, mara_storage::PayloadSchema::empty())
        .unwrap();

    let mut queries = Vec::with_capacity(N_QUERIES);
    let mut row = 0usize;
    for c in 0..N_CLUSTERS {
        let center: Vec<f32> = (0..DIM).map(|d| if d % N_CLUSTERS == c { 20.0 } else { 0.0 }).collect();
        for _ in 0..N_PER_CLUSTER {
            let v: Vec<f32> = center.iter().map(|x| x + rng.gen_range(-1.0..1.0)).collect();
            storage.put(&ctx(), "bench", &format!("r{row}"), v.clone(), PayloadRow::new(), None).unwrap();
            row += 1;
        }
        if queries.len() < N_QUERIES {
            queries.push(center.iter().map(|x| x + rng.gen_range(-0.5..0.5)).collect());
        }
    }
    (storage, queries)
}

fn recall_at_k(hits: &[mara_index_vector::SearchHit], oracle: &[mara_index_vector::SearchHit]) -> f64 {
    let oracle_ids: HashSet<_> = oracle.iter().map(|h| h.id).collect();
    let found = hits.iter().filter(|h| oracle_ids.contains(&h.id)).count();
    found as f64 / oracle.len().max(1) as f64
}

fn bench_recall_report(_c: &mut Criterion) {
    let (storage, queries) = synthetic_dataset(1);
    let flat = FlatIndex::new(DIM, DistanceMetric::L2);
    let ivf_params = IvfParams {
        n_clusters: 64,
        kmeans_iters: 15,
        seed: 2,
        ..IvfParams::default()
    };
    let pq_params = PqParams {
        m: 16,
        ksub: 64,
        kmeans_iters: 15,
        seed: 3,
    };
    let plain = IvfPqIndex::build(&storage, "bench", DIM, DistanceMetric::L2, &ivf_params, &pq_params).unwrap();
    let opq = IvfPqIndex::build_with_opq(
        &storage,
        "bench",
        DIM,
        DistanceMetric::L2,
        &ivf_params,
        &OpqParams { pq: pq_params, iterations: 15 },
    )
    .unwrap();

    let search_params = SearchParams {
        max_chunks_per_doc: None,
        nprobe: 8,
        rerank_k: None,
        ..SearchParams::default()
    };

    let mut plain_recall = 0.0;
    let mut opq_recall = 0.0;
    for q in &queries {
        let oracle = flat.search(&storage, "bench", q, K, None, &search_params).unwrap().hits;
        let plain_hits = plain.search(&storage, "bench", q, K, None, &search_params).unwrap().hits;
        let opq_hits = opq.search(&storage, "bench", q, K, None, &search_params).unwrap().hits;
        plain_recall += recall_at_k(&plain_hits, &oracle);
        opq_recall += recall_at_k(&opq_hits, &oracle);
    }
    plain_recall /= queries.len() as f64;
    opq_recall /= queries.len() as f64;

    eprintln!("\n=== recall@{K} vs. Flat oracle (nprobe={}, {N_QUERIES} queries) ===", search_params.nprobe);
    eprintln!("IvfPq (no OPQ): {:.1}%", plain_recall * 100.0);
    eprintln!("IvfPq (OPQ):    {:.1}%", opq_recall * 100.0);
    eprintln!("=== see the groups below for the latency each costs ===\n");
}

fn bench_search_latency(c: &mut Criterion) {
    let (storage, queries) = synthetic_dataset(1);
    let flat = FlatIndex::new(DIM, DistanceMetric::L2);
    let ivf_params = IvfParams {
        n_clusters: 64,
        kmeans_iters: 15,
        seed: 2,
        ..IvfParams::default()
    };
    let pq_params = PqParams {
        m: 16,
        ksub: 64,
        kmeans_iters: 15,
        seed: 3,
    };
    let plain = IvfPqIndex::build(&storage, "bench", DIM, DistanceMetric::L2, &ivf_params, &pq_params).unwrap();
    let opq = IvfPqIndex::build_with_opq(
        &storage,
        "bench",
        DIM,
        DistanceMetric::L2,
        &ivf_params,
        &OpqParams { pq: pq_params, iterations: 15 },
    )
    .unwrap();
    let search_params = SearchParams {
        max_chunks_per_doc: None,
        nprobe: 8,
        rerank_k: None,
        ..SearchParams::default()
    };

    let mut group = c.benchmark_group("search_latency");
    group.bench_function("flat", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(flat.search(&storage, "bench", q, K, None, &search_params).unwrap());
            }
        })
    });
    group.bench_function("ivf_pq_no_opq", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(plain.search(&storage, "bench", q, K, None, &search_params).unwrap());
            }
        })
    });
    group.bench_function("ivf_pq_opq", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(opq.search(&storage, "bench", q, K, None, &search_params).unwrap());
            }
        })
    });
    group.finish();
}

fn bench_opq_training_cost(c: &mut Criterion) {
    let (storage, _queries) = synthetic_dataset(1);
    let ivf_params = IvfParams {
        n_clusters: 64,
        kmeans_iters: 15,
        seed: 2,
        ..IvfParams::default()
    };
    let pq_params = PqParams {
        m: 16,
        ksub: 64,
        kmeans_iters: 15,
        seed: 3,
    };

    let mut group = c.benchmark_group("build_cost");
    group.sample_size(10);
    group.bench_function("ivf_pq_no_opq", |b| {
        b.iter(|| black_box(IvfPqIndex::build(&storage, "bench", DIM, DistanceMetric::L2, &ivf_params, &pq_params).unwrap()))
    });
    group.bench_function("ivf_pq_opq", |b| {
        b.iter(|| {
            black_box(
                IvfPqIndex::build_with_opq(
                    &storage,
                    "bench",
                    DIM,
                    DistanceMetric::L2,
                    &ivf_params,
                    &OpqParams { pq: pq_params.clone(), iterations: 15 },
                )
                .unwrap(),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, bench_recall_report, bench_search_latency, bench_opq_training_cost);
criterion_main!(benches);
