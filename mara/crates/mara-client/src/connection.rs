use crate::error::ClientError;
use futures_util::{SinkExt, StreamExt};
use mara_proto::{decode_response_body, encode_request, MaraCodec, Request, Response};
use tokio::net::UnixStream;
use tokio_util::codec::Framed;
use uuid::Uuid;

/// One request/response round trip over an established connection —
/// pooled connections are always past `Hello` (see `ConnectionManager`),
/// so every call here is an ordinary application request.
pub struct Connection {
    pub(crate) framed: Framed<UnixStream, MaraCodec>,
}

impl Connection {
    /// Sends one request and awaits its correlated response.
    pub async fn call(&mut self, req: Request) -> Result<Response, ClientError> {
        let request_id = Uuid::new_v4();
        let bytes = encode_request(request_id, &req)?;
        self.framed.send(bytes).await.map_err(|_| ClientError::ConnectionClosed)?;
        let frame = self.framed.next().await.ok_or(ClientError::ConnectionClosed)?.map_err(|_| ClientError::ConnectionClosed)?;
        if frame.request_id != request_id {
            return Err(ClientError::Correlation);
        }
        Ok(decode_response_body(&frame.body)?)
    }
}
