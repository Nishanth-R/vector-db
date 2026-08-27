/// Errors from wire encoding/decoding. Kept separate from `StorageError` and
/// friends (defined in the crates that own those failure domains) — this is
/// only the framing/codec failure surface.
#[derive(Debug, thiserror::Error)]
pub enum ProtoError {
    /// The buffer doesn't yet hold enough bytes for a complete frame header.
    #[error("frame too short: need at least {need} bytes, have {have}")]
    FrameTooShort {
        /// Minimum number of bytes required.
        need: usize,
        /// Number of bytes actually available.
        have: usize,
    },
    /// The frame's protocol version byte doesn't match `PROTOCOL_VERSION`.
    #[error("unsupported protocol version {0}, expected {1}")]
    UnsupportedVersion(u8, u8),
    /// The frame's kind byte isn't a recognized `FrameKind`.
    #[error("invalid frame kind byte {0}")]
    InvalidFrameKind(u8),
    /// The declared body length doesn't fit in the available buffer.
    #[error("declared body length {declared} exceeds buffer ({have} bytes available)")]
    BodyLengthMismatch {
        /// Body length declared in the frame header.
        declared: u32,
        /// Number of bytes actually available.
        have: usize,
    },
    /// Bincode failed to serialize a request or response body.
    #[error("bincode encode failed: {0}")]
    Encode(#[from] bincode::error::EncodeError),
    /// Bincode failed to deserialize a request or response body.
    #[error("bincode decode failed: {0}")]
    Decode(#[from] bincode::error::DecodeError),
}

/// Convenience alias for results that fail with `ProtoError`.
pub type ProtoResult<T> = Result<T, ProtoError>;
