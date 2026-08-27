//! Resolving one HTTP request's identity — the stateless-per-request
//! analogue of `mara-daemon::listener`'s `resolve_hello`, which does the
//! same job once per persistent socket connection via a `Hello` frame.
//! HTTP has no such handshake, so every request carries (or omits) its own
//! `Authorization: Bearer <token>` header and gets a fresh `RequestCtx`,
//! scoped to a session id minted for just that request — `mara-api` never
//! offers session-scoped `undo --n` continuity across requests the way a
//! long-lived UDS/TCP connection does.
//!
//! Mirrors `serve_tcp`'s rule, not `serve_unix`'s: HTTP always requires a
//! token when `auth.enabled`, with no `allow_local_unauthenticated`
//! exemption (that flag is UDS-only — a loopback HTTP client is still a
//! network client, not a trusted local peer identified by `SO_PEERCRED`).

use crate::error::error_response;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response as AxumResponse};
use mara_daemon::DaemonShared;
use mara_proto::{Principal, PrincipalId, RequestCtx, Role, SessionId, Source};
use std::net::SocketAddr;

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("auth_token required but missing")]
    TokenRequired,
    #[error("auth_token did not resolve to an enabled principal")]
    InvalidToken,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> AxumResponse {
        error_response("auth_failed", self.to_string())
    }
}

fn synthetic_principal(addr: SocketAddr) -> Principal {
    Principal {
        id: PrincipalId(format!("http:{addr}")),
        name: format!("http:{addr}"),
        role: Role::Admin,
    }
}

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers.get(axum::http::header::AUTHORIZATION)?.to_str().ok()?.strip_prefix("Bearer ")
}

pub fn resolve_ctx(shared: &DaemonShared, headers: &HeaderMap, addr: SocketAddr) -> Result<RequestCtx, AuthError> {
    let source = Source::Http(addr);
    let principal = if !shared.auth_enabled {
        synthetic_principal(addr)
    } else if let Some(token) = bearer_token(headers) {
        let store = shared.token_store.as_ref().ok_or(AuthError::InvalidToken)?;
        store.authenticate(token).ok_or(AuthError::InvalidToken)?
    } else {
        return Err(AuthError::TokenRequired);
    };

    let session = SessionId(format!("http-{}", uuid::Uuid::new_v4()));
    Ok(RequestCtx::new(session, principal, source))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mara_auth::TokenStore;

    fn addr() -> SocketAddr {
        "127.0.0.1:54321".parse().unwrap()
    }

    fn headers_with_bearer(token: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(axum::http::header::AUTHORIZATION, format!("Bearer {token}").parse().unwrap());
        h
    }

    fn shared_with_auth_disabled() -> DaemonShared {
        DaemonShared {
            engine: std::sync::Arc::new(NoopEngine),
            token_store: None,
            auth_enabled: false,
            allow_local_unauthenticated: true,
            storage: None,
        }
    }

    struct NoopEngine;
    #[async_trait::async_trait]
    impl mara_daemon::Engine for NoopEngine {
        async fn handle(&self, _ctx: &RequestCtx, _req: mara_proto::Request) -> mara_proto::Response {
            mara_proto::Response::Ok
        }
    }

    #[test]
    fn auth_disabled_resolves_a_synthetic_admin_principal() {
        let shared = shared_with_auth_disabled();
        let ctx = resolve_ctx(&shared, &HeaderMap::new(), addr()).unwrap();
        assert_eq!(ctx.principal.role, Role::Admin);
        assert_eq!(ctx.principal.id, PrincipalId("http:127.0.0.1:54321".into()));
        assert!(matches!(ctx.source, Source::Http(_)));
    }

    #[test]
    fn auth_enabled_without_a_bearer_header_is_token_required() {
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::load_or_create(dir.path().join("tokens.toml")).unwrap();
        let shared = DaemonShared {
            engine: std::sync::Arc::new(NoopEngine),
            token_store: Some(std::sync::Arc::new(store)),
            auth_enabled: true,
            allow_local_unauthenticated: true,
            storage: None,
        };
        let err = resolve_ctx(&shared, &HeaderMap::new(), addr()).unwrap_err();
        assert!(matches!(err, AuthError::TokenRequired), "unlike UDS, HTTP gets no allow_local_unauthenticated exemption");
    }

    #[test]
    fn auth_enabled_with_a_valid_token_resolves_the_real_principal() {
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::load_or_create(dir.path().join("tokens.toml")).unwrap();
        let (record, token) = store.create_token("alice", Role::Writer).unwrap();
        let shared = DaemonShared {
            engine: std::sync::Arc::new(NoopEngine),
            token_store: Some(std::sync::Arc::new(store)),
            auth_enabled: true,
            allow_local_unauthenticated: true,
            storage: None,
        };
        let ctx = resolve_ctx(&shared, &headers_with_bearer(&token), addr()).unwrap();
        assert_eq!(ctx.principal.id, record.id);
        assert_eq!(ctx.principal.role, Role::Writer);
    }

    #[test]
    fn auth_enabled_with_a_bogus_token_is_invalid_token() {
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::load_or_create(dir.path().join("tokens.toml")).unwrap();
        let shared = DaemonShared {
            engine: std::sync::Arc::new(NoopEngine),
            token_store: Some(std::sync::Arc::new(store)),
            auth_enabled: true,
            allow_local_unauthenticated: true,
            storage: None,
        };
        let err = resolve_ctx(&shared, &headers_with_bearer("mara_pat_not-a-real-token"), addr()).unwrap_err();
        assert!(matches!(err, AuthError::InvalidToken));
    }

    #[test]
    fn every_call_mints_a_distinct_session_id() {
        let shared = shared_with_auth_disabled();
        let a = resolve_ctx(&shared, &HeaderMap::new(), addr()).unwrap();
        let b = resolve_ctx(&shared, &HeaderMap::new(), addr()).unwrap();
        assert_ne!(a.session, b.session, "HTTP requests are stateless — each gets its own session, never a shared/reused one");
    }
}
