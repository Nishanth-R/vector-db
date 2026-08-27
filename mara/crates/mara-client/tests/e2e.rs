//! `mara-client` against a real `mara-daemon`, over a real Unix socket —
//! the pooled-client counterpart to `mara-daemon`'s own hand-rolled-client
//! e2e test.

use mara_client::MaraClient;
use mara_daemon::Config;
use mara_proto::{DistanceMetric, PayloadRow, PayloadValue, SearchMode, WireFieldType, WireIndexKind, WireSearchParams};
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
async fn pooled_client_round_trips_put_get_delete() {
    let dir = tempfile::tempdir().unwrap();
    let config = minimal_config(dir.path());
    let socket_path = config.unix_socket_path();
    tokio::spawn(async move {
        mara_daemon::run(config).await.unwrap();
    });
    wait_for_socket(&socket_path).await;

    let client = MaraClient::connect_unix(&socket_path, "e2e-client").unwrap();
    client
        .create_collection("docs", 3, DistanceMetric::Cosine, vec![("title".into(), WireFieldType::Keyword)])
        .await
        .unwrap();

    let mut fields = PayloadRow::new();
    fields.insert("title".into(), PayloadValue::Keyword("hello".into()));
    let row = client.put("docs", "a", vec![1.0, 2.0, 3.0], fields.clone(), None).await.unwrap();
    assert_eq!(row.key, "a");
    assert_eq!(row.fields, fields);

    let got = client.get_by_key("docs", "a").await.unwrap().expect("row should exist");
    assert_eq!(got.id, row.id);

    client.delete("docs", "a").await.unwrap();
    assert!(client.get_by_key("docs", "a").await.unwrap().is_none());
}

#[tokio::test]
async fn concurrent_calls_reuse_the_pool_without_stomping_each_other() {
    let dir = tempfile::tempdir().unwrap();
    let config = minimal_config(dir.path());
    let socket_path = config.unix_socket_path();
    tokio::spawn(async move {
        mara_daemon::run(config).await.unwrap();
    });
    wait_for_socket(&socket_path).await;

    let client = std::sync::Arc::new(MaraClient::connect_unix(&socket_path, "e2e-client").unwrap());
    client.create_collection("docs", 2, DistanceMetric::L2, vec![]).await.unwrap();

    let mut handles = Vec::new();
    for i in 0..20 {
        let client = client.clone();
        handles.push(tokio::spawn(async move {
            client
                .put(
                    "docs",
                    &format!("k{i}"),
                    vec![i as f32, 0.0],
                    PayloadRow::new(),
                    None,
                )
                .await
                .unwrap();
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    for i in 0..20 {
        let row = client.get_by_key("docs", &format!("k{i}")).await.unwrap();
        assert_eq!(row.unwrap().vector, Some(vec![i as f32, 0.0]));
    }
}

#[tokio::test]
async fn creating_the_same_collection_twice_surfaces_as_a_server_error() {
    let dir = tempfile::tempdir().unwrap();
    let config = minimal_config(dir.path());
    let socket_path = config.unix_socket_path();
    tokio::spawn(async move {
        mara_daemon::run(config).await.unwrap();
    });
    wait_for_socket(&socket_path).await;

    let client = MaraClient::connect_unix(&socket_path, "e2e-client").unwrap();
    client.create_collection("docs", 2, DistanceMetric::L2, vec![]).await.unwrap();
    let err = client.create_collection("docs", 2, DistanceMetric::L2, vec![]).await.unwrap_err();
    match err {
        mara_client::ClientError::Server { code, .. } => assert_eq!(code, "already_exists"),
        other => panic!("expected ClientError::Server, got {other:?}"),
    }
}

#[tokio::test]
async fn search_finds_the_true_nearest_neighbor_by_explicit_vector() {
    let dir = tempfile::tempdir().unwrap();
    let config = minimal_config(dir.path());
    let socket_path = config.unix_socket_path();
    tokio::spawn(async move {
        mara_daemon::run(config).await.unwrap();
    });
    wait_for_socket(&socket_path).await;

    let client = MaraClient::connect_unix(&socket_path, "e2e-client").unwrap();
    client.create_collection("docs", 2, DistanceMetric::L2, vec![]).await.unwrap();
    for i in 0..10u32 {
        client.put("docs", &format!("k{i}"), vec![i as f32, 0.0], PayloadRow::new(), None).await.unwrap();
    }

    let outcome = client
        .search("docs", None, Some(vec![5.2, 0.0]), SearchMode::VectorOnly, 1, None, WireSearchParams::default())
        .await
        .unwrap();
    assert_eq!(outcome.hits.len(), 1);
    assert_eq!(outcome.hits[0].row.key, "k5");
    assert!(!outcome.truncated_by_filter);
}

#[tokio::test]
async fn reindex_builds_the_requested_index_kind_and_reports_the_row_count() {
    let dir = tempfile::tempdir().unwrap();
    let config = minimal_config(dir.path());
    let socket_path = config.unix_socket_path();
    tokio::spawn(async move {
        mara_daemon::run(config).await.unwrap();
    });
    wait_for_socket(&socket_path).await;

    let client = MaraClient::connect_unix(&socket_path, "e2e-client").unwrap();
    client.create_collection("docs", 2, DistanceMetric::L2, vec![]).await.unwrap();
    for i in 0..5u32 {
        client.put("docs", &format!("k{i}"), vec![i as f32, 0.0], PayloadRow::new(), None).await.unwrap();
    }

    let summary = client.reindex("docs", WireIndexKind::Flat).await.unwrap();
    assert_eq!(summary.coll, "docs");
    assert_eq!(summary.index_kind, WireIndexKind::Flat);
    assert_eq!(summary.row_count, 5);
}
