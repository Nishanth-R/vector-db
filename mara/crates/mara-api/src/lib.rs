//! `mara-api`: an axum HTTP frontend over `mara-daemon`'s `Engine`, the
//! REST-facing peer of `marad`'s native UDS/TCP protocol (master plan
//! Layer 5, *`mara-api` (axum)*). A standalone process against its own
//! `data_dir` — like `marad`, not embedded inside it — booted through the
//! exact same `mara_daemon::boot` composition root, so the two frontends
//! can never disagree about how storage, the audit sink, the token store,
//! or the embedder come up. `Authorization: Bearer <token>` resolves to
//! the same `RequestCtx` `Hello`'s `auth_token` does; every write/read
//! endpoint dispatches through the same `Arc<dyn Engine>`, so there is
//! exactly one authorization table and one audit path regardless of
//! transport.

mod auth;
mod dto;
mod error;
mod handlers;
mod openapi;

pub use openapi::ApiDoc;

use axum::routing::{get, post, put};
use axum::Router;
use mara_daemon::{BootError, Config, DaemonShared};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone)]
pub struct ApiState {
    pub shared: Arc<DaemonShared>,
    pub audit_dir: PathBuf,
}

pub fn router(state: ApiState) -> Router {
    Router::new()
        .route("/v1/health", get(handlers::health))
        .route("/v1/audit", get(handlers::audit))
        .route("/v1/collections", post(handlers::create_collection))
        .route("/v1/collections/{coll}/rows/batch", post(handlers::put_batch))
        .route(
            "/v1/collections/{coll}/rows/{key}",
            put(handlers::put_row).get(handlers::get_row).delete(handlers::delete_row),
        )
        .route("/v1/collections/{coll}/documents", post(handlers::put_document))
        .route("/v1/collections/{coll}/search", post(handlers::search))
        .route("/v1/collections/{coll}/reindex", post(handlers::reindex))
        .route("/v1/openapi.json", get(openapi_json))
        .with_state(state)
}

async fn openapi_json() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::to_value(ApiDoc::openapi()).expect("ApiDoc always serializes"))
}

use utoipa::OpenApi as _;

/// Boots a fresh `data_dir` via `mara_daemon::boot` — the same composition
/// root `marad`/`mara serve` use — then serves HTTP on `config.http.listen`
/// until the listener errors or the process is killed. Errors immediately
/// if `[http] listen` was left unset; unlike `server.tcp_listen` (an
/// optional extra transport alongside UDS, which `mara-daemon::run`
/// already handles skipping), an *HTTP-only* process with nothing to bind
/// to has no reason to exist.
pub async fn run(config: Config) -> Result<(), BootError> {
    let Some(listen) = config.http.listen.clone() else {
        return Err("mara-api requires `[http] listen` to be set in the config file".into());
    };

    let (shared, _data_dir_lock) = mara_daemon::boot(&config).await?;
    let state = ApiState { shared, audit_dir: config.audit_dir() };
    let app = router(state);

    let listener = tokio::net::TcpListener::bind(&listen).await?;
    tracing::info!(%listen, "mara-api listening");
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>()).await?;
    Ok(())
}
