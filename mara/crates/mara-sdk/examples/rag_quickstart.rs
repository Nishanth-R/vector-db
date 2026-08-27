//! A minimal, runnable RAG walkthrough: spin up a `mara-daemon` against a
//! throwaway temp directory, ingest a few short documents via
//! `MaraStore::add_document` (server-side chunk → embed → commit), then
//! ask it a question via `MaraStore::query` (hybrid BM25 + vector search,
//! fused server-side) and print what comes back.
//!
//! `cargo run --example rag_quickstart -p mara-sdk`
//!
//! First run downloads the configured embedding model
//! (`sentence-transformers/all-MiniLM-L6-v2`) if it isn't already cached
//! under the temp data dir — expect a short pause the first time.

use mara_daemon::Config;
use mara_proto::PayloadRow;
use mara_sdk::MaraStore;
use std::time::Duration;

const DOCS: &[(&str, &str)] = &[
    (
        "rust-ownership",
        "Rust's ownership system tracks, at compile time, exactly one owner for every value. \
         When the owner goes out of scope, the value is dropped automatically — there is no \
         garbage collector and no manual free. Borrowing lets other code read or, exclusively, \
         mutate a value without taking ownership of it.",
    ),
    (
        "vector-search",
        "Approximate nearest neighbor search trades a small amount of recall for a large \
         speedup over brute-force scanning. Inverted-file indexes coarsely cluster vectors, \
         then only scan the clusters nearest a query; product quantization further compresses \
         each vector into a short code so millions of candidates fit in memory.",
    ),
    (
        "bm25",
        "BM25 is a bag-of-words ranking function used by search engines to estimate how relevant \
         a document is to a query. It scores based on term frequency, inverse document frequency, \
         and document length normalization, without needing any notion of vector embeddings.",
    ),
];

async fn wait_for_socket(path: &std::path::Path) {
    for _ in 0..600 {
        if path.exists() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("daemon socket never appeared at {path:?} — check the printed daemon log output above");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let config_path = dir.path().join("mara.toml");
    std::fs::write(
        &config_path,
        format!(
            "[server]\ndata_dir = {:?}\nunix_socket = \"mara.sock\"\n[embedding]\nmodel = \"sentence-transformers/all-MiniLM-L6-v2\"\n",
            dir.path()
        ),
    )?;
    let config = Config::load(&config_path)?;
    let socket_path = config.unix_socket_path();

    println!("booting a mara daemon at {} (first run downloads the embedding model)...", dir.path().display());
    tokio::spawn(async move {
        if let Err(e) = mara_daemon::run(config).await {
            eprintln!("daemon exited: {e}");
        }
    });
    wait_for_socket(&socket_path).await;

    let store = MaraStore::connect_unix(&socket_path, "rag-quickstart", "docs")?;
    store.create_collection(384, mara_sdk::DistanceMetric::Cosine, vec![]).await?;

    println!("ingesting {} documents...", DOCS.len());
    for (doc_key, text) in DOCS {
        let summary = store.add_document(doc_key, text, None, PayloadRow::new(), None).await?;
        println!("  {doc_key:?} -> {} chunk(s), doc_id={}", summary.chunk_count, summary.doc_id);
    }

    let question = "How does BM25 rank documents?";
    println!("\nquery: {question:?}\n");
    let outcome = store.query(question, 3).await?;
    for (rank, hit) in outcome.hits.iter().enumerate() {
        let snippet: String = hit.row.text.as_deref().unwrap_or("").chars().take(120).collect();
        println!("  #{}: doc_id={:?} score={:.4} exact={}\n      {snippet}...", rank + 1, hit.row.doc_id, hit.score, hit.exact);
    }

    Ok(())
}
