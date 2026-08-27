//! `MaraStore` against a real `mara-daemon`, over a real Unix socket —
//! mirrors `mara-client`'s own e2e test, one layer up. Skips
//! `add_document`/`query` (both need a configured embedding model, which
//! means a real ONNX model on the first run — slow and not worth this
//! crate's own fast test suite); `examples/rag_quickstart.rs` is where
//! those get a real, illustrative exercise instead.

use mara_daemon::Config;
use mara_proto::{DistanceMetric, PayloadRow, SearchMode, WireIndexKind, WireSearchParams};
use mara_sdk::MaraStore;
use std::time::Duration;

fn minimal_config(data_dir: &std::path::Path) -> Config {
    let config_path = data_dir.join("mara.toml");
    std::fs::write(&config_path, format!("[server]\ndata_dir = {data_dir:?}\nunix_socket = \"mara.sock\"\n")).unwrap();
    Config::load(&config_path).unwrap()
}

async fn wait_for_socket(path: &std::path::Path) {
    for _ in 0..200 {
        if path.exists() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("daemon socket never appeared at {path:?}");
}

#[tokio::test]
async fn store_scoped_to_one_collection_round_trips_put_get_delete() {
    let dir = tempfile::tempdir().unwrap();
    let config = minimal_config(dir.path());
    let socket_path = config.unix_socket_path();
    tokio::spawn(async move {
        mara_daemon::run(config).await.unwrap();
    });
    wait_for_socket(&socket_path).await;

    let store = MaraStore::connect_unix(&socket_path, "sdk-e2e", "docs").unwrap();
    assert_eq!(store.collection(), "docs");
    store.create_collection(3, DistanceMetric::Cosine, vec![]).await.unwrap();

    let row = store.put("a", vec![1.0, 2.0, 3.0], PayloadRow::new(), None).await.unwrap();
    assert_eq!(row.key, "a");

    let got = store.get("a").await.unwrap().expect("row should exist");
    assert_eq!(got.id, row.id);

    store.delete("a").await.unwrap();
    assert!(store.get("a").await.unwrap().is_none());
}

#[tokio::test]
async fn search_and_reindex_work_through_the_scoped_store() {
    let dir = tempfile::tempdir().unwrap();
    let config = minimal_config(dir.path());
    let socket_path = config.unix_socket_path();
    tokio::spawn(async move {
        mara_daemon::run(config).await.unwrap();
    });
    wait_for_socket(&socket_path).await;

    let store = MaraStore::connect_unix(&socket_path, "sdk-e2e", "docs").unwrap();
    store.create_collection(2, DistanceMetric::L2, vec![]).await.unwrap();
    for i in 0..10u32 {
        store.put(&format!("k{i}"), vec![i as f32, 0.0], PayloadRow::new(), None).await.unwrap();
    }

    let outcome = store
        .search(None, Some(vec![5.2, 0.0]), SearchMode::VectorOnly, 1, None, WireSearchParams::default())
        .await
        .unwrap();
    assert_eq!(outcome.hits[0].row.key, "k5");

    let summary = store.reindex(WireIndexKind::Flat).await.unwrap();
    assert_eq!(summary.coll, "docs");
    assert_eq!(summary.row_count, 10);
}

#[tokio::test]
async fn from_client_shares_a_pool_with_another_store_on_the_same_connection() {
    let dir = tempfile::tempdir().unwrap();
    let config = minimal_config(dir.path());
    let socket_path = config.unix_socket_path();
    tokio::spawn(async move {
        mara_daemon::run(config).await.unwrap();
    });
    wait_for_socket(&socket_path).await;

    let client = mara_client::MaraClient::connect_unix(&socket_path, "sdk-e2e").unwrap();
    let docs = MaraStore::from_client(client.clone(), "docs");
    docs.create_collection(2, DistanceMetric::L2, vec![]).await.unwrap();
    docs.put("a", vec![1.0, 0.0], PayloadRow::new(), None).await.unwrap();

    // A second store cloned from the exact same `MaraClient` — same
    // underlying connection pool, different collection.
    let notes = MaraStore::from_client(client, "notes");
    notes.create_collection(2, DistanceMetric::L2, vec![]).await.unwrap();
    notes.put("n1", vec![0.0, 1.0], PayloadRow::new(), None).await.unwrap();

    assert!(docs.get("a").await.unwrap().is_some());
    assert!(notes.get("n1").await.unwrap().is_some());
    assert!(docs.get("n1").await.unwrap().is_none(), "the two collections must stay isolated");
}
