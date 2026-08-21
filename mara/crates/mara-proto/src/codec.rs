//! Length-prefixed framing over `bincode` payloads: `[version][kind]
//! [request_id][u32 body_len][body]`. Deliberately not gRPC — nothing here
//! is tied to `tokio`; `mara-daemon`/`mara-client` wrap these pure functions
//! in an async codec once the socket layer exists.

use crate::error::{ProtoError, ProtoResult};
use crate::request::Request;
use crate::response::Response;
use uuid::Uuid;

pub const PROTOCOL_VERSION: u8 = 1;

/// `HEADER_LEN = version(1) + kind(1) + request_id(16) + body_len(4)`.
pub const HEADER_LEN: usize = 1 + 1 + 16 + 4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FrameKind {
    Request = 0,
    Response = 1,
}

impl FrameKind {
    fn from_byte(b: u8) -> ProtoResult<Self> {
        match b {
            0 => Ok(FrameKind::Request),
            1 => Ok(FrameKind::Response),
            other => Err(ProtoError::InvalidFrameKind(other)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecodedFrame {
    pub version: u8,
    pub kind: FrameKind,
    pub request_id: Uuid,
    pub body: Vec<u8>,
}

fn bincode_config() -> bincode::config::Configuration {
    bincode::config::standard()
}

fn encode_frame(kind: FrameKind, request_id: Uuid, body: &[u8]) -> ProtoResult<Vec<u8>> {
    let mut out = Vec::with_capacity(HEADER_LEN + body.len());
    out.push(PROTOCOL_VERSION);
    out.push(kind as u8);
    out.extend_from_slice(request_id.as_bytes());
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(body);
    Ok(out)
}

pub fn encode_request(request_id: Uuid, req: &Request) -> ProtoResult<Vec<u8>> {
    let body = bincode::serde::encode_to_vec(req, bincode_config())?;
    encode_frame(FrameKind::Request, request_id, &body)
}

pub fn encode_response(request_id: Uuid, resp: &Response) -> ProtoResult<Vec<u8>> {
    let body = bincode::serde::encode_to_vec(resp, bincode_config())?;
    encode_frame(FrameKind::Response, request_id, &body)
}

/// Attempts to decode exactly one frame from the front of `buf`. Returns
/// `Ok(None)` if `buf` doesn't yet hold a complete frame — the caller (a
/// future `tokio_util::codec::Decoder` impl) should read more bytes and
/// retry rather than treating this as an error. Never consumes a partial
/// frame's bytes.
pub fn try_decode_frame(buf: &[u8]) -> ProtoResult<Option<(DecodedFrame, usize)>> {
    if buf.len() < HEADER_LEN {
        return Ok(None);
    }
    let version = buf[0];
    if version != PROTOCOL_VERSION {
        return Err(ProtoError::UnsupportedVersion(version, PROTOCOL_VERSION));
    }
    let kind = FrameKind::from_byte(buf[1])?;
    let request_id = Uuid::from_bytes(buf[2..18].try_into().expect("18-2=16 bytes"));
    let body_len = u32::from_be_bytes(buf[18..22].try_into().expect("4 bytes"));
    let total = HEADER_LEN + body_len as usize;
    if buf.len() < total {
        return Ok(None);
    }
    let body = buf[HEADER_LEN..total].to_vec();
    Ok(Some((
        DecodedFrame {
            version,
            kind,
            request_id,
            body,
        },
        total,
    )))
}

pub fn decode_request_body(body: &[u8]) -> ProtoResult<Request> {
    let (req, _) = bincode::serde::decode_from_slice(body, bincode_config())?;
    Ok(req)
}

pub fn decode_response_body(body: &[u8]) -> ProtoResult<Response> {
    let (resp, _) = bincode::serde::decode_from_slice(body, bincode_config())?;
    Ok(resp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::SessionId;

    #[test]
    fn request_round_trips_through_a_frame() {
        let req = Request::GetByKey {
            coll: "handbook".into(),
            key: "onboarding.md#12".into(),
        };
        let request_id = Uuid::new_v4();
        let bytes = encode_request(request_id, &req).unwrap();

        let (frame, consumed) = try_decode_frame(&bytes).unwrap().unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(frame.kind, FrameKind::Request);
        assert_eq!(frame.request_id, request_id);

        let decoded = decode_request_body(&frame.body).unwrap();
        assert_eq!(decoded, req);
    }

    #[test]
    fn incomplete_buffer_returns_none_not_error() {
        let req = Request::Hello {
            client_name: "cli".into(),
            session_id: SessionId("cli-7f21".into()),
            auth_token: None,
        };
        let bytes = encode_request(Uuid::new_v4(), &req).unwrap();
        // Header present but body truncated.
        assert!(try_decode_frame(&bytes[..HEADER_LEN]).unwrap().is_none());
        assert!(try_decode_frame(&bytes[..bytes.len() - 1])
            .unwrap()
            .is_none());
        // Too short even for a header.
        assert!(try_decode_frame(&bytes[..5]).unwrap().is_none());
    }

    #[test]
    fn two_frames_back_to_back_decode_independently() {
        let req_id = Uuid::new_v4();
        let resp_id = Uuid::new_v4();
        let mut buf = encode_request(
            req_id,
            &Request::GetByKey {
                coll: "c".into(),
                key: "k".into(),
            },
        )
        .unwrap();
        buf.extend(encode_response(resp_id, &Response::Ok).unwrap());

        let (frame1, consumed1) = try_decode_frame(&buf).unwrap().unwrap();
        assert_eq!(frame1.kind, FrameKind::Request);
        let (frame2, consumed2) = try_decode_frame(&buf[consumed1..]).unwrap().unwrap();
        assert_eq!(frame2.kind, FrameKind::Response);
        assert_eq!(consumed1 + consumed2, buf.len());
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes =
            encode_request(Uuid::new_v4(), &Request::Delete { coll: "c".into(), key: "k".into() })
                .unwrap();
        bytes[0] = 99;
        let err = try_decode_frame(&bytes).unwrap_err();
        assert!(matches!(err, ProtoError::UnsupportedVersion(99, PROTOCOL_VERSION)));
    }
}
