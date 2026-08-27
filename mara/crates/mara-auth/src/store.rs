use crate::principal::PrincipalRecord;
use crate::token::{generate_token, hash_token, verify_token};
use chrono::Utc;
use mara_proto::{PrincipalId, Role};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum TokenStoreError {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse tokens file {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("failed to serialize tokens file: {0}")]
    Serialize(#[from] toml::ser::Error),
    #[error("no principal {0:?}")]
    NotFound(String),
}

/// On-disk shape of `auth/tokens.toml`: an array of tables rather than a map
/// keyed by `PrincipalId`, since TOML's map-key serialization is awkward for
/// non-string-native key types. In-memory lookups still go through a
/// `HashMap`.
#[derive(Default, serde::Serialize, serde::Deserialize)]
struct TokensFile {
    #[serde(default)]
    principals: Vec<PrincipalRecord>,
}

/// `data_dir/auth/tokens.toml`, mode `0600`, holding only token hashes — the
/// plaintext token is returned exactly once, at creation, and never
/// persisted.
pub struct TokenStore {
    path: PathBuf,
    principals: RwLock<HashMap<PrincipalId, PrincipalRecord>>,
}

impl TokenStore {
    pub fn load_or_create(path: impl Into<PathBuf>) -> Result<Self, TokenStoreError> {
        let path = path.into();
        let principals = if path.exists() {
            let contents = fs::read_to_string(&path).map_err(|source| TokenStoreError::Io {
                path: path.clone(),
                source,
            })?;
            let file: TokensFile = toml::from_str(&contents).map_err(|source| TokenStoreError::Parse {
                path: path.clone(),
                source,
            })?;
            file.principals.into_iter().map(|p| (p.id.clone(), p)).collect()
        } else {
            HashMap::new()
        };
        let store = TokenStore {
            path,
            principals: RwLock::new(principals),
        };
        store.save()?;
        Ok(store)
    }

    fn save(&self) -> Result<(), TokenStoreError> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|source| TokenStoreError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let mut principals: Vec<PrincipalRecord> = self.principals.read().values().cloned().collect();
        principals.sort_by_key(|a| a.created_at);
        let toml_str = toml::to_string_pretty(&TokensFile { principals })?;
        fs::write(&self.path, &toml_str).map_err(|source| TokenStoreError::Io {
            path: self.path.clone(),
            source,
        })?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&self.path, fs::Permissions::from_mode(0o600)).map_err(|source| {
                TokenStoreError::Io {
                    path: self.path.clone(),
                    source,
                }
            })?;
        }
        Ok(())
    }

    /// Mints a new token for `name`/`role`, persists its hash, and returns
    /// `(record, plaintext_token)`. The caller must display the plaintext
    /// immediately — this is the only time it's ever available.
    pub fn create_token(&self, name: &str, role: Role) -> Result<(PrincipalRecord, String), TokenStoreError> {
        let token = generate_token();
        let record = PrincipalRecord {
            id: PrincipalId(format!("p_{}", &Uuid::new_v4().simple().to_string()[..6])),
            name: name.to_string(),
            role,
            token_hash: hash_token(&token),
            created_at: Utc::now(),
            last_seen: None,
            disabled: false,
        };
        self.principals.write().insert(record.id.clone(), record.clone());
        self.save()?;
        Ok((record, token))
    }

    pub fn list(&self) -> Vec<PrincipalRecord> {
        let mut v: Vec<_> = self.principals.read().values().cloned().collect();
        v.sort_by_key(|a| a.created_at);
        v
    }

    /// Permanently removes a principal — its token stops authenticating
    /// immediately (subject to `auth.revalidate_secs` for already-open
    /// pooled connections; see the open risks in the master plan).
    pub fn revoke(&self, id: &PrincipalId) -> Result<(), TokenStoreError> {
        let removed = self.principals.write().remove(id);
        if removed.is_none() {
            return Err(TokenStoreError::NotFound(id.0.clone()));
        }
        self.save()
    }

    /// Soft-disables a principal — the record (and audit trail) is kept,
    /// but it can no longer authenticate.
    pub fn disable(&self, id: &PrincipalId) -> Result<(), TokenStoreError> {
        {
            let mut guard = self.principals.write();
            let record = guard
                .get_mut(id)
                .ok_or_else(|| TokenStoreError::NotFound(id.0.clone()))?;
            record.disabled = true;
        }
        self.save()
    }

    /// Resolves a plaintext token to the lightweight context view used to
    /// build a `RequestCtx`, bumping `last_seen`. `None` if no enabled
    /// principal's hash matches.
    pub fn authenticate(&self, token: &str) -> Option<mara_proto::Principal> {
        let view = {
            let mut guard = self.principals.write();
            let hit = guard
                .values_mut()
                .find(|p| !p.disabled && verify_token(token, &p.token_hash))?;
            hit.last_seen = Some(Utc::now());
            hit.to_context_view()
        };
        let _ = self.save(); // best-effort persistence of last_seen; auth already succeeded
        Some(view)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_list_and_persist_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.toml");
        let store = TokenStore::load_or_create(&path).unwrap();
        let (record, token) = store.create_token("rag-app", Role::Writer).unwrap();
        assert!(token.starts_with(crate::token::TOKEN_PREFIX));
        assert_eq!(store.list().len(), 1);

        // Reload from disk into a fresh store — the hash must survive, the
        // plaintext token must authenticate against it.
        let reloaded = TokenStore::load_or_create(&path).unwrap();
        assert_eq!(reloaded.list().len(), 1);
        let resolved = reloaded.authenticate(&token).expect("token should authenticate");
        assert_eq!(resolved.id, record.id);
        assert_eq!(resolved.role, Role::Writer);
    }

    #[test]
    fn file_permissions_are_owner_only_on_unix() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("tokens.toml");
            let _store = TokenStore::load_or_create(&path).unwrap();
            let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
    }

    #[test]
    fn disabled_principal_cannot_authenticate() {
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::load_or_create(dir.path().join("tokens.toml")).unwrap();
        let (record, token) = store.create_token("alice", Role::Reader).unwrap();
        store.disable(&record.id).unwrap();
        assert!(store.authenticate(&token).is_none());
    }

    #[test]
    fn revoked_principal_is_gone() {
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::load_or_create(dir.path().join("tokens.toml")).unwrap();
        let (record, token) = store.create_token("alice", Role::Reader).unwrap();
        store.revoke(&record.id).unwrap();
        assert!(store.authenticate(&token).is_none());
        assert!(store.list().is_empty());
    }

    #[test]
    fn wrong_token_never_authenticates() {
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::load_or_create(dir.path().join("tokens.toml")).unwrap();
        store.create_token("alice", Role::Reader).unwrap();
        assert!(store.authenticate("mara_pat_not-a-real-token").is_none());
    }
}
