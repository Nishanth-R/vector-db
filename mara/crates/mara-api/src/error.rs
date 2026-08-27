//! Maps `mara_proto::Response::Error`'s `code` vocabulary (defined by
//! `mara-daemon::engine`'s handful of `fn ..._err`/`forbidden`/
//! `invalid_argument` helpers — there is no shared enum to match
//! exhaustively against, so this is intentionally a `match` over string
//! codes with a conservative fallback) onto HTTP status codes.

use crate::dto::ErrorBody;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response as AxumResponse};
use axum::Json;

pub fn error_response(code: &str, message: String) -> AxumResponse {
    let status = match code {
        "not_found" => StatusCode::NOT_FOUND,
        "already_exists" => StatusCode::CONFLICT,
        "dimension_mismatch" | "invalid_argument" | "invalid_chunk_spec" => StatusCode::BAD_REQUEST,
        "auth_failed" => StatusCode::UNAUTHORIZED,
        "forbidden" => StatusCode::FORBIDDEN,
        "embedding_not_configured" => StatusCode::SERVICE_UNAVAILABLE,
        "embedding_error" | "index_error" | "storage_error" => StatusCode::INTERNAL_SERVER_ERROR,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, Json(ErrorBody { code: code.to_string(), message })).into_response()
}

/// A `Response` variant a given handler never expects to see back from
/// `Engine::handle` for the `Request` it just sent (e.g. `Search` handled
/// by anything other than `SearchResults`/`Error`) — defensive, since
/// `Response` is one shared enum across every request kind and nothing at
/// the type level rules this out for an individual handler.
pub fn unexpected_response(resp: mara_proto::Response) -> AxumResponse {
    tracing::error!(?resp, "engine returned a Response variant this handler doesn't know how to render");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorBody {
            code: "unexpected_response".into(),
            message: "the engine returned a response shape this endpoint doesn't expect".into(),
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    async fn status_of(code: &str) -> StatusCode {
        error_response(code, "x".into()).status()
    }

    #[tokio::test]
    async fn every_known_code_maps_to_the_expected_status() {
        assert_eq!(status_of("not_found").await, StatusCode::NOT_FOUND);
        assert_eq!(status_of("already_exists").await, StatusCode::CONFLICT);
        assert_eq!(status_of("dimension_mismatch").await, StatusCode::BAD_REQUEST);
        assert_eq!(status_of("invalid_argument").await, StatusCode::BAD_REQUEST);
        assert_eq!(status_of("invalid_chunk_spec").await, StatusCode::BAD_REQUEST);
        assert_eq!(status_of("auth_failed").await, StatusCode::UNAUTHORIZED, "a bearer-auth failure must be 401, not fall through to the generic 500");
        assert_eq!(status_of("forbidden").await, StatusCode::FORBIDDEN);
        assert_eq!(status_of("embedding_not_configured").await, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(status_of("embedding_error").await, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(status_of("index_error").await, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(status_of("storage_error").await, StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn an_unrecognized_code_falls_back_to_500_rather_than_panicking() {
        assert_eq!(status_of("some_future_code_this_match_hasnt_seen_yet").await, StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn the_body_carries_the_original_code_and_message_verbatim() {
        let resp = error_response("not_found", "no row with that key".into());
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["code"], "not_found");
        assert_eq!(parsed["message"], "no row with that key");
    }
}
