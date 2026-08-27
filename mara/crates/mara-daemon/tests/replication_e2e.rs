//! Leader-follower replication over two real `mara-daemon` processes (well
//! — two real `mara_daemon::run` instances, each with its own `data_dir`
//! and UDS socket, talking to each other over a real TCP replication
//! connection) — the end-to-end counterpart to `mara-storage/tests/
//! replication.rs`'s storage-layer-only coverage.

use futures_util::{SinkExt, StreamExt};
use mara_auth::TokenStore;
use mara_daemon::Config;
use mara_proto::{decode_response_body, encode_request, DistanceMetric, MaraCodec, PayloadRow, Request, Response, Role, SessionId};
use std::time::Duration;
use tokio::net::UnixStream;
use tokio_util::codec::Framed;
use uuid::Uuid;

/// A free TCP port, obtained by binding to port 0 and immediately
/// releasing it — a small, standard race (another process could grab it
/// before `leader::serve` binds moments later) that's overwhelmingly
/// reliable in practice for sequential test setup, and simplest given
/// `leader::serve` binds internally and doesn't hand its bound port back.
fn free_tcp_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn leader_config(dir: &std::path::Path, replication_port: u16) -> Config {
    let toml_str = format!(
        "[server]\ndata_dir = {dir:?}\nunix_socket = \"mara.sock\"\n[auth]\nenabled = true\n[replication]\nrole = \"leader\"\nlisten = \"127.0.0.1:{replication_port}\"\n"
    );
    let config_path = dir.join("mara.toml");
    std::fs::write(&config_path, toml_str).unwrap();
    Config::load(&config_path).unwrap()
}

fn follower_config(dir: &std::path::Path, replication_port: u16, token: &str) -> Config {
    let toml_str = format!(
        "[server]\ndata_dir = {dir:?}\nunix_socket = \"mara.sock\"\n[replication]\nrole = \"follower\"\nleader_addr = \"127.0.0.1:{replication_port}\"\nauth_token = \"{token}\"\npoll_interval_ms = 20\n"
    );
    let config_path = dir.join("mara.toml");
    std::fs::write(&config_path, toml_str).unwrap();
    Config::load(&config_path).unwrap()
}

/// Mints a `Role::Replica` token directly into `dir`'s token store,
/// *before* the leader daemon boots — `TokenStore` loads once at startup
/// and keeps it in memory, so a token minted after boot would never be
/// seen by the running process.
fn mint_replica_token(dir: &std::path::Path) -> String {
    let store = TokenStore::load_or_create(dir.join("auth/tokens.toml")).unwrap();
    let (_record, token) = store.create_token("follower-1", Role::Replica).unwrap();
    token
}

struct TestClient {
    framed: Framed<UnixStream, MaraCodec>,
}

impl TestClient {
    async fn connect(socket_path: &std::path::Path) -> Self {
        let stream = loop {
            match UnixStream::connect(socket_path).await {
                Ok(s) => break s,
                Err(_) => tokio::time::sleep(Duration::from_millis(20)).await,
            }
        };
        TestClient { framed: Framed::new(stream, MaraCodec) }
    }

    async fn call(&mut self, req: Request) -> Response {
        let request_id = Uuid::new_v4();
        let bytes = encode_request(request_id, &req).unwrap();
        self.framed.send(bytes).await.unwrap();
        let frame = self.framed.next().await.expect("connection closed unexpectedly").unwrap();
        decode_response_body(&frame.body).unwrap()
    }

    async fn hello(&mut self) -> Response {
        self.call(Request::Hello { client_name: "e2e-test".into(), session_id: SessionId("e2e-session".into()), auth_token: None }).await
    }
}

/// Polls `client` with `GetByKey{coll, key}` until it returns a `Row`, or
/// panics after a generous timeout — replication is async, so "eventually
/// visible" is the correct thing to assert, never "visible on the very
/// next call".
async fn eventually_get(client: &mut TestClient, coll: &str, key: &str) -> mara_proto::Row {
    for _ in 0..200 {
        if let Response::Row(row) = client.call(Request::GetByKey { coll: coll.into(), key: key.into() }).await {
            return row;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("row {key:?} in {coll:?} never became visible within the timeout");
}

/// The `not_found` counterpart to `eventually_get` — polls until a
/// previously-visible row is gone.
async fn eventually_not_found(client: &mut TestClient, coll: &str, key: &str) {
    for _ in 0..200 {
        if let Response::Error { code, .. } = client.call(Request::GetByKey { coll: coll.into(), key: key.into() }).await
            && code == "not_found"
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("row {key:?} in {coll:?} never disappeared within the timeout");
}

#[tokio::test]
async fn a_row_written_on_the_leader_is_eventually_visible_on_the_follower() {
    let leader_dir = tempfile::tempdir().unwrap();
    let follower_dir = tempfile::tempdir().unwrap();
    let port = free_tcp_port();

    let token = mint_replica_token(leader_dir.path());
    let leader_cfg = leader_config(leader_dir.path(), port);
    let leader_socket = leader_cfg.unix_socket_path();
    tokio::spawn(async move {
        mara_daemon::run(leader_cfg).await.unwrap();
    });

    let follower_cfg = follower_config(follower_dir.path(), port, &token);
    let follower_socket = follower_cfg.unix_socket_path();
    tokio::spawn(async move {
        mara_daemon::run(follower_cfg).await.unwrap();
    });

    let mut leader = TestClient::connect(&leader_socket).await;
    assert!(matches!(leader.hello().await, Response::HelloAck { .. }));
    let created = leader.call(Request::CreateCollection { name: "docs".into(), dim: 3, metric: DistanceMetric::Cosine, schema: vec![] }).await;
    assert!(matches!(created, Response::Ok), "expected Ok, got {created:?}");
    let put = leader
        .call(Request::Put { coll: "docs".into(), key: "a".into(), text: None, vector: Some(vec![1.0, 2.0, 3.0]), fields: PayloadRow::new(), extra: None })
        .await;
    assert!(matches!(put, Response::Row(_)), "expected Row, got {put:?}");

    let mut follower = TestClient::connect(&follower_socket).await;
    assert!(matches!(follower.hello().await, Response::HelloAck { .. }));
    let row = eventually_get(&mut follower, "docs", "a").await;
    assert_eq!(row.vector, Some(vec![1.0, 2.0, 3.0]));
}

#[tokio::test]
async fn a_delete_on_the_leader_eventually_removes_the_row_on_the_follower() {
    let leader_dir = tempfile::tempdir().unwrap();
    let follower_dir = tempfile::tempdir().unwrap();
    let port = free_tcp_port();

    let token = mint_replica_token(leader_dir.path());
    let leader_cfg = leader_config(leader_dir.path(), port);
    let leader_socket = leader_cfg.unix_socket_path();
    tokio::spawn(async move {
        mara_daemon::run(leader_cfg).await.unwrap();
    });
    let follower_cfg = follower_config(follower_dir.path(), port, &token);
    let follower_socket = follower_cfg.unix_socket_path();
    tokio::spawn(async move {
        mara_daemon::run(follower_cfg).await.unwrap();
    });

    let mut leader = TestClient::connect(&leader_socket).await;
    leader.hello().await;
    leader.call(Request::CreateCollection { name: "docs".into(), dim: 2, metric: DistanceMetric::L2, schema: vec![] }).await;
    leader.call(Request::Put { coll: "docs".into(), key: "a".into(), text: None, vector: Some(vec![1.0, 0.0]), fields: PayloadRow::new(), extra: None }).await;

    let mut follower = TestClient::connect(&follower_socket).await;
    follower.hello().await;
    eventually_get(&mut follower, "docs", "a").await;

    leader.call(Request::Delete { coll: "docs".into(), key: "a".into() }).await;
    eventually_not_found(&mut follower, "docs", "a").await;
}

#[tokio::test]
async fn a_write_sent_directly_to_the_follower_is_rejected_as_not_the_leader() {
    let leader_dir = tempfile::tempdir().unwrap();
    let follower_dir = tempfile::tempdir().unwrap();
    let port = free_tcp_port();

    let token = mint_replica_token(leader_dir.path());
    let leader_cfg = leader_config(leader_dir.path(), port);
    tokio::spawn(async move {
        mara_daemon::run(leader_cfg).await.unwrap();
    });
    let follower_cfg = follower_config(follower_dir.path(), port, &token);
    let follower_socket = follower_cfg.unix_socket_path();
    tokio::spawn(async move {
        mara_daemon::run(follower_cfg).await.unwrap();
    });

    let mut follower = TestClient::connect(&follower_socket).await;
    follower.hello().await;
    let resp = follower.call(Request::CreateCollection { name: "docs".into(), dim: 2, metric: DistanceMetric::L2, schema: vec![] }).await;
    match resp {
        Response::Error { code, .. } => assert_eq!(code, "not_the_leader"),
        other => panic!("expected a not_the_leader error, got {other:?}"),
    }
}

#[tokio::test]
async fn a_collection_created_after_the_follower_connects_is_discovered_and_replicated() {
    let leader_dir = tempfile::tempdir().unwrap();
    let follower_dir = tempfile::tempdir().unwrap();
    let port = free_tcp_port();

    let token = mint_replica_token(leader_dir.path());
    let leader_cfg = leader_config(leader_dir.path(), port);
    let leader_socket = leader_cfg.unix_socket_path();
    tokio::spawn(async move {
        mara_daemon::run(leader_cfg).await.unwrap();
    });
    let follower_cfg = follower_config(follower_dir.path(), port, &token);
    let follower_socket = follower_cfg.unix_socket_path();
    tokio::spawn(async move {
        mara_daemon::run(follower_cfg).await.unwrap();
    });

    // Let the follower connect and send its (empty) ReplicaHello before
    // the leader creates anything, so this genuinely exercises
    // `ReplicaNewCollection` discovery rather than the initial
    // `ReplicaWelcome` collection list.
    tokio::time::sleep(Duration::from_millis(150)).await;

    let mut leader = TestClient::connect(&leader_socket).await;
    leader.hello().await;
    leader.call(Request::CreateCollection { name: "late".into(), dim: 2, metric: DistanceMetric::L2, schema: vec![] }).await;
    leader.call(Request::Put { coll: "late".into(), key: "x".into(), text: None, vector: Some(vec![9.0, 9.0]), fields: PayloadRow::new(), extra: None }).await;

    let mut follower = TestClient::connect(&follower_socket).await;
    follower.hello().await;
    let row = eventually_get(&mut follower, "late", "x").await;
    assert_eq!(row.vector, Some(vec![9.0, 9.0]));
}

