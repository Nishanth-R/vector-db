//! The authorization surface, deliberately expressed as one readable table
//! rather than scattered checks throughout the engine, so it can be audited
//! by reading a single function.

use mara_proto::{Role, UndoScope};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Capability {
    // Reader baseline
    Search,
    Get,
    ListCollections,
    ListDocuments,
    Status,
    // Writer
    Put,
    PutBatch,
    PutDocument,
    Undo,
    // Admin
    Config,
    Checkpoint,
    Reindex,
    CreateCollection,
    DropCollection,
    AlterSchema,
    TokenManagement,
    ReplicationAdmin,
    // Replica
    ReplicaHello,
    WalStream,
}

/// `Role -> Capability` authorization table. One `match` arm per
/// `(role, capability)` pair that's allowed; anything not listed here falls
/// through to the `_ => false` at the bottom and is denied.
pub fn require(role: Role, cap: Capability) -> bool {
    use Capability::*;
    use Role::*;
    match (role, cap) {
        (Reader | Writer | Admin, Search) => true,
        (Reader | Writer | Admin, Get) => true,
        (Reader | Writer | Admin, ListCollections) => true,
        (Reader | Writer | Admin, ListDocuments) => true,
        (Reader | Writer | Admin, Status) => true,

        (Writer | Admin, Put) => true,
        (Writer | Admin, PutBatch) => true,
        (Writer | Admin, PutDocument) => true,
        (Writer | Admin, Undo) => true,

        (Admin, Config) => true,
        (Admin, Checkpoint) => true,
        (Admin, Reindex) => true,
        (Admin, CreateCollection) => true,
        (Admin, DropCollection) => true,
        (Admin, AlterSchema) => true,
        (Admin, TokenManagement) => true,
        (Admin, ReplicationAdmin) => true,

        (Replica, ReplicaHello) => true,
        (Replica, WalStream) => true,

        _ => false,
    }
}

/// `undo --n` scope gating: `Session`/`Collection` are available to any
/// writer, `Global` requires `Admin`. Kept separate from [`require`] because
/// it's not a fixed capability but a per-call parameter of `Undo` itself —
/// folding it into the same match would need a third dimension.
pub fn can_use_undo_scope(role: Role, scope: UndoScope) -> bool {
    match scope {
        UndoScope::Session | UndoScope::Collection => require(role, Capability::Undo),
        UndoScope::Global => role == Role::Admin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mara_proto::Role::*;

    #[test]
    fn reader_cannot_write() {
        assert!(!require(Reader, Capability::Put));
        assert!(!require(Reader, Capability::Undo));
    }

    #[test]
    fn writer_has_reader_capabilities_too() {
        assert!(require(Writer, Capability::Search));
        assert!(require(Writer, Capability::Put));
        assert!(!require(Writer, Capability::Config));
    }

    #[test]
    fn admin_has_everything_but_replica_only_caps() {
        assert!(require(Admin, Capability::Put));
        assert!(require(Admin, Capability::Config));
        assert!(!require(Admin, Capability::ReplicaHello));
    }

    #[test]
    fn replica_is_isolated_to_its_own_two_capabilities() {
        assert!(require(Replica, Capability::ReplicaHello));
        assert!(require(Replica, Capability::WalStream));
        assert!(!require(Replica, Capability::Get));
        assert!(!require(Replica, Capability::Search));
    }

    #[test]
    fn global_undo_scope_requires_admin() {
        assert!(!can_use_undo_scope(Writer, UndoScope::Global));
        assert!(can_use_undo_scope(Admin, UndoScope::Global));
        assert!(can_use_undo_scope(Writer, UndoScope::Session));
    }
}
