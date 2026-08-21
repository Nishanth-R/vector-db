use chrono::{DateTime, Utc};
use mara_proto::{PrincipalId, Role};
use serde::{Deserialize, Serialize};

/// The full persisted principal record — richer than `mara_proto::Principal`
/// (which is only the resolved-identity view carried on a `RequestCtx`).
/// This is `TokenStore`'s row: it also carries the token hash and
/// lifecycle metadata, none of which belong on every request context.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct PrincipalRecord {
    pub id: PrincipalId,
    pub name: String,
    pub role: Role,
    #[serde(with = "hash_hex")]
    pub token_hash: [u8; 32],
    pub created_at: DateTime<Utc>,
    pub last_seen: Option<DateTime<Utc>>,
    pub disabled: bool,
}

impl PrincipalRecord {
    pub fn to_context_view(&self) -> mara_proto::Principal {
        mara_proto::Principal {
            id: self.id.clone(),
            name: self.name.clone(),
            role: self.role,
        }
    }
}

mod hash_hex {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        hex.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let s = String::deserialize(d)?;
        if s.len() != 64 {
            return Err(serde::de::Error::custom(format!(
                "expected 64 hex chars for a blake3 hash, got {}",
                s.len()
            )));
        }
        let mut out = [0u8; 32];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).map_err(serde::de::Error::custom)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::{generate_token, hash_token};

    #[test]
    fn token_hash_round_trips_through_toml() {
        let record = PrincipalRecord {
            id: PrincipalId("p_7f21a0".into()),
            name: "alice".into(),
            role: Role::Writer,
            token_hash: hash_token(&generate_token()),
            created_at: Utc::now(),
            last_seen: None,
            disabled: false,
        };
        let toml_str = toml::to_string(&record).unwrap();
        let back: PrincipalRecord = toml::from_str(&toml_str).unwrap();
        assert_eq!(record, back);
    }
}
