//! `tokio_util::codec::{Decoder, Encoder}` glue over the synchronous
//! framing in `codec.rs`, so a `Framed<UnixStream, _>` /
//! `Framed<TcpStream, _>` can speak this protocol directly. The framing
//! logic itself stays in `codec.rs`, testable without an async runtime;
//! this module only adapts it to `BytesMut` and wires up `advance()`.

use crate::codec::{try_decode_frame, DecodedFrame};
use crate::error::ProtoError;
use bytes::{Buf, BytesMut};
use tokio_util::codec::{Decoder, Encoder};

/// Errors that can occur while decoding or encoding frames on an async stream.
#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    /// A protocol-level framing or (de)serialization error.
    #[error(transparent)]
    Proto(#[from] ProtoError),
    /// An underlying I/O error from the transport.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Decodes `DecodedFrame`s; encodes pre-framed bytes (from
/// `codec::encode_request`/`encode_response`) verbatim. One codec serves
/// both directions of a connection — the frame's own `kind` byte already
/// distinguishes a request from a response.
#[derive(Default)]
pub struct MaraCodec;

impl Decoder for MaraCodec {
    type Item = DecodedFrame;
    type Error = CodecError;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        match try_decode_frame(src)? {
            Some((frame, consumed)) => {
                src.advance(consumed);
                Ok(Some(frame))
            }
            None => Ok(None),
        }
    }
}

impl Encoder<Vec<u8>> for MaraCodec {
    type Error = CodecError;

    fn encode(&mut self, item: Vec<u8>, dst: &mut BytesMut) -> Result<(), Self::Error> {
        dst.extend_from_slice(&item);
        Ok(())
    }
}
