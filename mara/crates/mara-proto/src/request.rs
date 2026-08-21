use crate::ids::SessionId;
use crate::payload::{ExtraPayload, PayloadRow};
use serde::{Deserialize, Serialize};

/// One row to insert in a `PutBatch`.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct PutItem {
    pub key: String,
    pub text: Option<String>,
    pub vector: Option<Vec<f32>>,
    pub fields: PayloadRow,
    pub extra: Option<ExtraPayload>,
}

/// The wire protocol's request vocabulary. Deliberately minimal for now —
/// `Hello`/`Put`/`PutBatch`/`GetByKey`/`Delete` — matching what storage
/// (steps 0-10) actually needs; document, search, undo, and admin request
/// kinds are added alongside the daemon (`Engine::handle`) and the index
/// layers that serve them.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Request {
    Hello {
        client_name: String,
        session_id: SessionId,
        auth_token: Option<String>,
    },
    Put {
        coll: String,
        key: String,
        text: Option<String>,
        vector: Option<Vec<f32>>,
        fields: PayloadRow,
        extra: Option<ExtraPayload>,
    },
    PutBatch {
        coll: String,
        items: Vec<PutItem>,
    },
    GetByKey {
        coll: String,
        key: String,
    },
    Delete {
        coll: String,
        key: String,
    },
}
