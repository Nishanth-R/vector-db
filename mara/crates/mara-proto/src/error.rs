/// Errors from wire encoding/decoding. Kept separate from `StorageError` and
/// friends (defined in the crates that own those failure domains) — this is
/// only the framing/codec failure surface.
#[derive(Debug, thiserror::Error)]
pub enum ProtoError {
    #[error("frame too short: need at least {need} bytes, have {have}")]
    FrameTooShort { need: usize, have: usize },
    #[error("unsupported protocol version {0}, expected {1}")]
    UnsupportedVersion(u8, u8),
    #[error("invalid frame kind byte {0}")]
    InvalidFrameKind(u8),
    #[error("declared body length {declared} exceeds buffer ({have} bytes available)")]
    BodyLengthMismatch { declared: u32, have: usize },
    #[error("bincode encode failed: {0}")]
    Encode(#[from] bincode::error::EncodeError),
    #[error("bincode decode failed: {0}")]
    Decode(#[from] bincode::error::DecodeError),
}

pub type ProtoResult<T> = Result<T, ProtoError>;
