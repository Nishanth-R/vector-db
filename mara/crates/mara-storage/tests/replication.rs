//! `Collection::wal_lines_after` (leader) + `Collection::apply_replicated_lines`
//! (follower) against two real, durable `Collection`s — the storage-layer
//! primitives `mara-daemon`'s replication task is built on. Verifies state
//! reconstruction, subscriber notification, and that a follower's own
//! local WAL survives a reopen exactly like any other durable collection's
//! does (see `crash_recovery.rs`).

use mara_proto::{DistanceMetric, PayloadRow, Principal, RequestCtx, Role, SessionId, Source};
use mara_storage::{ChangeBatch, ChangeSubscriber, Collection, FsyncPolicy, PayloadSchema, PutInput};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

fn ctx() -> RequestCtx {
    RequestCtx::new(
        SessionId("test-session".into()),
        Principal {
            id: mara_proto::PrincipalId("p_test".into()),
            name: "tester".into(),
            role: Role::Writer,
        },
        Source::Embedded,
    )
}

fn durable_coll(dir: &std::path::Path) -> Collection {
    Collection::open("docs", 3, DistanceMetric::Cosine, PayloadSchema::empty(), false, dir, 128 * 1024 * 1024, FsyncPolicy::Always).unwrap()
}

#[test]
fn wal_lines_after_none_covers_every_record_on_a_fresh_leader() {
    let leader_dir = tempfile::tempdir().unwrap();
    let leader = durable_coll(leader_dir.path());
    for k in ["a", "b", "c"] {
        leader.put(&ctx(), k, vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
    }

    let (lines, cursor) = leader.wal_lines_after(None).unwrap();
    assert_eq!(lines.len(), 3);
    assert_eq!(cursor, leader.last_applied_lsn());
}

#[test]
fn follower_applying_leader_lines_reconstructs_identical_state() {
    let leader_dir = tempfile::tempdir().unwrap();
    let leader = durable_coll(leader_dir.path());
    leader.put(&ctx(), "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
    leader.put(&ctx(), "b", vec![0.0, 1.0, 0.0], PayloadRow::new(), None).unwrap();
    leader.delete(&ctx(), "a").unwrap();
    leader.put(&ctx(), "c", vec![0.0, 0.0, 1.0], PayloadRow::new(), None).unwrap();

    let follower_dir = tempfile::tempdir().unwrap();
    let follower = durable_coll(follower_dir.path());
    let (lines, _) = leader.wal_lines_after(None).unwrap();
    follower.apply_replicated_lines(&lines).unwrap();

    assert!(follower.get_by_key("a").is_none(), "the delete must have replayed on the follower too");
    assert_eq!(follower.get_by_key("b").unwrap().vector, Some(vec![0.0, 1.0, 0.0]));
    assert_eq!(follower.get_by_key("c").unwrap().vector, Some(vec![0.0, 0.0, 1.0]));
    assert_eq!(follower.active_count(), leader.active_count());
    assert_eq!(follower.last_applied_lsn(), leader.last_applied_lsn());
}

#[test]
fn follower_catches_up_incrementally_across_multiple_streaming_batches() {
    let leader_dir = tempfile::tempdir().unwrap();
    let leader = durable_coll(leader_dir.path());
    let follower_dir = tempfile::tempdir().unwrap();
    let follower = durable_coll(follower_dir.path());

    leader.put(&ctx(), "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
    let (batch1, _) = leader.wal_lines_after(follower.last_applied_lsn()).unwrap();
    follower.apply_replicated_lines(&batch1).unwrap();

    leader.put(&ctx(), "b", vec![0.0, 1.0, 0.0], PayloadRow::new(), None).unwrap();
    // The follower asks for everything after what it has *already*
    // applied — exactly what a real `ReplicaHello{known_lsn}` resume does.
    let (batch2, _) = leader.wal_lines_after(follower.last_applied_lsn()).unwrap();
    assert_eq!(batch2.len(), 1, "the second batch must only contain the new record, not a and b both");
    follower.apply_replicated_lines(&batch2).unwrap();

    assert!(follower.get_by_key("a").is_some());
    assert!(follower.get_by_key("b").is_some());
    assert_eq!(follower.last_applied_lsn(), leader.last_applied_lsn());
}

#[test]
fn wal_lines_after_returns_the_unchanged_cursor_when_there_is_nothing_new() {
    let leader_dir = tempfile::tempdir().unwrap();
    let leader = durable_coll(leader_dir.path());
    leader.put(&ctx(), "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
    let (_, cursor) = leader.wal_lines_after(None).unwrap();

    let (lines, cursor2) = leader.wal_lines_after(cursor).unwrap();
    assert!(lines.is_empty());
    assert_eq!(cursor2, cursor, "polling again with no new writes must return the same cursor, not None or something advanced");
}

#[test]
fn apply_replicated_lines_rejects_a_tampered_checksum_and_applies_nothing() {
    let leader_dir = tempfile::tempdir().unwrap();
    let leader = durable_coll(leader_dir.path());
    leader.put(&ctx(), "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
    let (mut lines, _) = leader.wal_lines_after(None).unwrap();
    lines[0] = lines[0].replacen("\"key\":\"a\"", "\"key\":\"tampered\"", 1);

    let follower_dir = tempfile::tempdir().unwrap();
    let follower = durable_coll(follower_dir.path());
    let err = follower.apply_replicated_lines(&lines).unwrap_err();
    assert!(err.to_string().contains("checksum"), "expected a checksum error, got {err}");
    assert_eq!(follower.active_count(), 0, "a rejected batch must leave the follower untouched");
    assert_eq!(follower.last_applied_lsn(), None);
}

struct CountingSubscriber {
    batches: AtomicUsize,
    event_counts: Mutex<Vec<usize>>,
}

impl ChangeSubscriber for CountingSubscriber {
    fn on_change(&self, batch: &ChangeBatch) {
        self.batches.fetch_add(1, Ordering::SeqCst);
        self.event_counts.lock().unwrap().push(batch.events.len());
    }
}

#[test]
fn apply_replicated_lines_notifies_subscribers_grouped_by_transaction() {
    let leader_dir = tempfile::tempdir().unwrap();
    let leader = durable_coll(leader_dir.path());
    // Two single-row transactions (two separate `put` calls) followed by
    // one 2-row transaction (`put_batch`).
    leader.put(&ctx(), "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();
    leader.put(&ctx(), "b", vec![0.0, 1.0, 0.0], PayloadRow::new(), None).unwrap();
    leader
        .put_batch(
            &ctx(),
            vec![
                PutInput { key: "c".into(), vector: vec![0.0, 0.0, 1.0], fields: PayloadRow::new(), extra: None },
                PutInput { key: "d".into(), vector: vec![1.0, 1.0, 0.0], fields: PayloadRow::new(), extra: None },
            ],
        )
        .unwrap();

    let follower_dir = tempfile::tempdir().unwrap();
    let follower = durable_coll(follower_dir.path());
    let sub = std::sync::Arc::new(CountingSubscriber { batches: AtomicUsize::new(0), event_counts: Mutex::new(Vec::new()) });
    follower.subscribe(sub.clone());

    let (lines, _) = leader.wal_lines_after(None).unwrap();
    follower.apply_replicated_lines(&lines).unwrap();

    assert_eq!(sub.batches.load(Ordering::SeqCst), 3, "3 leader-side transactions must produce 3 separate ChangeBatch notifications, not 1 merged one");
    assert_eq!(*sub.event_counts.lock().unwrap(), vec![1, 1, 2]);
}

#[test]
fn a_replicating_follower_survives_a_reopen_and_keeps_accepting_new_batches() {
    let leader_dir = tempfile::tempdir().unwrap();
    let leader = durable_coll(leader_dir.path());
    leader.put(&ctx(), "a", vec![1.0, 0.0, 0.0], PayloadRow::new(), None).unwrap();

    let follower_dir = tempfile::tempdir().unwrap();
    {
        let follower = durable_coll(follower_dir.path());
        let (lines, _) = leader.wal_lines_after(None).unwrap();
        follower.apply_replicated_lines(&lines).unwrap();
    }
    // Simulate a process restart: reopen against the same directory —
    // ordinary crash-recovery startup, no replication-specific code path.
    let follower = durable_coll(follower_dir.path());
    assert!(follower.get_by_key("a").is_some(), "the replicated row must survive the reopen");
    assert_eq!(follower.last_applied_lsn(), leader.last_applied_lsn());

    leader.put(&ctx(), "b", vec![0.0, 1.0, 0.0], PayloadRow::new(), None).unwrap();
    let (more, _) = leader.wal_lines_after(follower.last_applied_lsn()).unwrap();
    follower.apply_replicated_lines(&more).unwrap();
    assert!(follower.get_by_key("b").is_some());
}
