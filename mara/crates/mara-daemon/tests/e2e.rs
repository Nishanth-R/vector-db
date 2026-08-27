//! End-to-end over a real Unix socket: `Hello` -> `CreateCollection` ->
//! `Put` -> `GetByKey` -> `Delete` -> `GetByKey` (now missing). The master
//! plan's explicit step-11 acceptance criterion. No `mara-client` exists
//! yet (that's step 12), so this drives the wire protocol with a small
//! hand-rolled client using the same `MaraCodec` the daemon itself uses.

use futures_util::{SinkExt, StreamExt};
use mara_daemon::Config;
use mara_proto::{decode_response_body, encode_request, MaraCodec, PayloadRow, Request, Response, SessionId, WireFieldType};
use std::time::Duration;
use tokio::net::UnixStream;
use tokio_util::codec::Framed;
use uuid::Uuid;

fn minimal_config(data_dir: &std::path::Path) -> Config {
    let toml_str = format!(
        "[server]\ndata_dir = {:?}\nunix_socket = \"mara.sock\"\n",
        data_dir
    );
    let config_path = data_dir.join("mara.toml");
    std::fs::write(&config_path, toml_str).unwrap();
    Config::load(&config_path).unwrap()
}

struct TestClient {
    framed: Framed<UnixStream, MaraCodec>,
}

impl TestClient {
    async fn connect(socket_path: &std::path::Path) -> Self {
        // The daemon's listener starts asynchronously; poll briefly rather
        // than assuming it's already bound the instant the spawned task
        // begins running.
        let stream = loop {
            match UnixStream::connect(socket_path).await {
                Ok(s) => break s,
                Err(_) => tokio::time::sleep(Duration::from_millis(20)).await,
            }
        };
        TestClient {
            framed: Framed::new(stream, MaraCodec),
        }
    }

    async fn call(&mut self, req: Request) -> Response {
        let request_id = Uuid::new_v4();
        let bytes = encode_request(request_id, &req).unwrap();
        self.framed.send(bytes).await.unwrap();
        let frame = self.framed.next().await.expect("connection closed unexpectedly").unwrap();
        assert_eq!(frame.request_id, request_id, "response must correlate to its request");
        decode_response_body(&frame.body).unwrap()
    }

    async fn hello(&mut self) -> Response {
        self.call(Request::Hello {
            client_name: "e2e-test".into(),
            session_id: SessionId("e2e-session".into()),
            auth_token: None,
        })
        .await
    }
}

#[tokio::test]
async fn hello_put_get_delete_round_trip_over_a_real_unix_socket() {
    let dir = tempfile::tempdir().unwrap();
    let config = minimal_config(dir.path());
    let socket_path = config.unix_socket_path();

    tokio::spawn(async move {
        mara_daemon::run(config).await.unwrap();
    });

    let mut client = TestClient::connect(&socket_path).await;

    let ack = client.hello().await;
    assert!(matches!(ack, Response::HelloAck { .. }), "expected HelloAck, got {ack:?}");

    let created = client
        .call(Request::CreateCollection {
            name: "docs".into(),
            dim: 3,
            metric: mara_proto::DistanceMetric::Cosine,
            schema: vec![("title".into(), WireFieldType::Keyword)],
        })
        .await;
    assert!(matches!(created, Response::Ok), "expected Ok, got {created:?}");

    let mut fields = PayloadRow::new();
    fields.insert("title".into(), mara_proto::PayloadValue::Keyword("hello".into()));
    let put = client
        .call(Request::Put {
            coll: "docs".into(),
            key: "a".into(),
            text: None,
            vector: Some(vec![1.0, 2.0, 3.0]),
            fields: fields.clone(),
            extra: None,
        })
        .await;
    let Response::Row(row) = put else { panic!("expected Row, got {put:?}") };
    assert_eq!(row.key, "a");
    assert_eq!(row.vector, Some(vec![1.0, 2.0, 3.0]));
    assert_eq!(row.fields, fields);

    let got = client
        .call(Request::GetByKey {
            coll: "docs".into(),
            key: "a".into(),
        })
        .await;
    let Response::Row(row2) = got else { panic!("expected Row, got {got:?}") };
    assert_eq!(row2.id, row.id);

    let deleted = client
        .call(Request::Delete {
            coll: "docs".into(),
            key: "a".into(),
        })
        .await;
    assert!(matches!(deleted, Response::Ok), "expected Ok, got {deleted:?}");

    let missing = client
        .call(Request::GetByKey {
            coll: "docs".into(),
            key: "a".into(),
        })
        .await;
    match missing {
        Response::Error { code, .. } => assert_eq!(code, "not_found"),
        other => panic!("expected Error(not_found) after delete, got {other:?}"),
    }
}

#[tokio::test]
async fn a_second_connection_can_use_the_collection_the_first_created() {
    // Exercises the semaphore-bounded accept loop handling more than one
    // live connection, and that collection state is genuinely shared
    // across connections (via the one in-process `Storage`), not
    // per-connection.
    let dir = tempfile::tempdir().unwrap();
    let config = minimal_config(dir.path());
    let socket_path = config.unix_socket_path();

    tokio::spawn(async move {
        mara_daemon::run(config).await.unwrap();
    });

    let mut client1 = TestClient::connect(&socket_path).await;
    client1.hello().await;
    client1
        .call(Request::CreateCollection {
            name: "shared".into(),
            dim: 2,
            metric: mara_proto::DistanceMetric::L2,
            schema: vec![],
        })
        .await;
    client1
        .call(Request::Put {
            coll: "shared".into(),
            key: "x".into(),
            text: None,
            vector: Some(vec![1.0, 1.0]),
            fields: PayloadRow::new(),
            extra: None,
        })
        .await;

    let mut client2 = TestClient::connect(&socket_path).await;
    client2.hello().await;
    let got = client2
        .call(Request::GetByKey {
            coll: "shared".into(),
            key: "x".into(),
        })
        .await;
    assert!(matches!(got, Response::Row(_)), "a second connection must see the first connection's committed write");
}

#[tokio::test]
async fn data_survives_a_daemon_restart_once_the_collection_is_recreated() {
    // `Storage::open` gives each created collection a real WAL — but the
    // *registry* of which collections exist isn't itself persisted yet
    // (a documented v0 limitation), so recovery requires re-issuing
    // `CreateCollection` with the same parameters; `Collection::open` then
    // replays the prior data underneath it.
    let dir = tempfile::tempdir().unwrap();
    let config = minimal_config(dir.path());
    let socket_path = config.unix_socket_path();

    let handle = tokio::spawn({
        let config = config.clone();
        async move { mara_daemon::run(config).await }
    });

    {
        let mut client = TestClient::connect(&socket_path).await;
        client.hello().await;
        client
            .call(Request::CreateCollection {
                name: "docs".into(),
                dim: 2,
                metric: mara_proto::DistanceMetric::Cosine,
                schema: vec![],
            })
            .await;
        client
            .call(Request::Put {
                coll: "docs".into(),
                key: "a".into(),
                text: None,
                vector: Some(vec![0.5, 0.5]),
                fields: PayloadRow::new(),
                extra: None,
            })
            .await;
    }

    handle.abort();
    let _ = handle.await;
    // The socket file from the killed daemon is still on disk; the new
    // daemon's `serve_unix` removes and rebinds it.

    tokio::spawn(async move {
        mara_daemon::run(config).await.unwrap();
    });

    let mut client2 = TestClient::connect(&socket_path).await;
    client2.hello().await;
    client2
        .call(Request::CreateCollection {
            name: "docs".into(),
            dim: 2,
            metric: mara_proto::DistanceMetric::Cosine,
            schema: vec![],
        })
        .await;
    let got = client2
        .call(Request::GetByKey {
            coll: "docs".into(),
            key: "a".into(),
        })
        .await;
    let Response::Row(row) = got else {
        panic!("expected the row to survive the restart, got {got:?}")
    };
    assert_eq!(row.vector, Some(vec![0.5, 0.5]));
}

#[tokio::test]
async fn a_request_before_hello_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let config = minimal_config(dir.path());
    let socket_path = config.unix_socket_path();

    tokio::spawn(async move {
        mara_daemon::run(config).await.unwrap();
    });

    let mut client = TestClient::connect(&socket_path).await;
    // Skip Hello — send GetByKey as the very first frame.
    let response = client
        .call(Request::GetByKey {
            coll: "docs".into(),
            key: "a".into(),
        })
        .await;
    assert!(matches!(response, Response::Error { .. }), "a non-Hello first frame must be rejected, got {response:?}");
}
