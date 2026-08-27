/// Errors that can occur while using `MaraClient` or its lower-level connection types.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    /// An underlying I/O error from the transport.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// The connection was closed before a response arrived.
    #[error("connection closed unexpectedly")]
    ConnectionClosed,
    /// The response's request id didn't match the request that was sent.
    #[error("response did not correlate to the request that was sent")]
    Correlation,
    /// A protocol-level framing or (de)serialization error.
    #[error("protocol error: {0}")]
    Proto(#[from] mara_proto::ProtoError),
    /// An async codec decode/encode error.
    #[error("codec error: {0}")]
    Codec(#[from] mara_proto::CodecError),
    /// The connection pool failed to check out a connection.
    #[error("connection pool error: {0}")]
    Pool(String),
    /// The daemon returned `Response::Error`.
    #[error("server error [{code}]: {message}")]
    Server {
        /// Stable, machine-readable error code from the daemon.
        code: String,
        /// Human-readable description of the failure.
        message: String,
    },
    /// The daemon returned a response of an unexpected shape for the request that was sent.
    #[error("unexpected response shape: {0}")]
    Unexpected(String),
}
