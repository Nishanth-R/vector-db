//! Applies already-validated WAL records (checksum-verified, torn tails
//! already excluded by `wal::replay_all`) to reconstruct a collection's
//! in-memory state. Used by both `Collection::open` (startup recovery) and
//! `Collection::undo`/`revert_to` (step 10) for applying compensating
//! records the same way a fresh boot would.
//!
//! Distinct from the normal write path (`Collection::put_batch` and
//! friends): those *decide* what happens and mint fresh `RowId`s; this
//! module only *replays* decisions that were already made and durably
//! recorded, using exactly the `RowId`s the WAL specifies.

use crate::change::ChangeEvent;
use crate::collection::{CollectionInner, RowRecord};
use crate::document::{DocEntry, WalDocMeta};
use crate::error::{StorageError, StorageResult};
use crate::wal::{self, WalOp, WalRecord};
use mara_proto::{DocId, PayloadRow, TxnEntry};
use roaring::RoaringBitmap;
use std::collections::HashSet;
use std::sync::Arc;

/// Derives the `ChangeEvent`s a batch of (already WAL-durable)
/// insert/update/delete records represents — the single conversion every
/// mutation method uses to notify subscribers, so "what actually got
/// applied" and "what subscribers are told happened" can never drift apart.
/// Control-op records (schema/collection lifecycle) produce no event.
pub(crate) fn change_events_from_records(records: &[WalRecord]) -> Vec<ChangeEvent> {
    records
        .iter()
        .filter_map(|r| {
            let row_id = r.row_id?;
            match r.op {
                WalOp::Insert | WalOp::Update => {
                    let p = r.payload.as_ref()?;
                    let vector: Arc<[f32]> = p.vector().ok()??.into();
                    let fields = Arc::new(p.fields.clone().unwrap_or_default());
                    let extra = p.extra.clone().map(Arc::new);
                    let key: Arc<str> = Arc::from(r.key.clone()?.as_str());
                    Some(if r.op == WalOp::Insert {
                        ChangeEvent::Insert { row_id, key, vector, fields, extra, doc_id: r.doc_id }
                    } else {
                        ChangeEvent::Update { row_id, key, vector, fields, extra, doc_id: r.doc_id }
                    })
                }
                WalOp::Delete => Some(ChangeEvent::Delete { row_id, doc_id: r.doc_id }),
                _ => None,
            }
        })
        .collect()
}

/// Applies a flat, transaction-ordered record list (as returned by
/// `wal::replay_all`) to `inner`. Records from the same transaction are
/// always contiguous in that list (a single `WalWriter` never interleaves
/// transactions), so this groups by consecutive `txn` runs and applies
/// each group as a unit — needed so a `replace_document`'s "delete all old
/// rows, insert all new ones" resolves to the *new* document state rather
/// than transiently deleting the doc registry entry mid-transaction.
pub(crate) fn apply_replayed_records(inner: &mut CollectionInner, records: &[WalRecord]) -> StorageResult<()> {
    let mut i = 0;
    while i < records.len() {
        let txn = records[i].txn;
        let mut j = i + 1;
        while j < records.len() && records[j].txn == txn {
            j += 1;
        }
        apply_txn_group(inner, &records[i..j])?;
        i = j;
    }
    Ok(())
}

fn apply_txn_group(inner: &mut CollectionInner, group: &[WalRecord]) -> StorageResult<()> {
    let mut touched_docs: HashSet<DocId> = HashSet::new();
    let mut affected_rows = RoaringBitmap::new();
    for record in group {
        apply_single_record(inner, record)?;
        if let Some(doc_id) = record.doc_id {
            touched_docs.insert(doc_id);
        }
        if let Some(row_id) = record.row_id {
            affected_rows.insert(row_id.to_bitmap_index());
        }
        inner.last_applied_lsn = inner.last_applied_lsn.max(record.lsn);
    }
    // A document touched by this transaction that ends up with zero live
    // rows was deleted (either `delete_document`, or the delete half of a
    // `replace_document` whose insert half never landed — which can't
    // happen for a *complete* transaction, but the empty-rows check is the
    // correct signal either way).
    for &doc_id in &touched_docs {
        if let Some(entry) = inner.docs.get(&doc_id)
            && entry.rows.is_empty() {
                let doc_key = entry.doc_key.clone();
                inner.docs.remove(&doc_id);
                inner.doc_key_to_id.remove(doc_key.as_str());
            }
    }

    // Reconstruct this transaction's `TxnIndex` entry the same way it was
    // recorded live (see `wal::summarize_txn`), so `history`/`undo` behave
    // identically whether or not the process restarted since the write.
    let first = group.first().expect("a txn group is never empty");
    let mut doc_ids: Vec<DocId> = touched_docs.into_iter().collect();
    doc_ids.sort();
    let lsn_range = (
        group.iter().map(|r| r.lsn).min().expect("non-empty group"),
        group.iter().map(|r| r.lsn).max().expect("non-empty group"),
    );
    inner.txn_index.record(TxnEntry {
        txn_id: first.txn,
        session: first.session.clone(),
        principal: first.actor.id.clone(),
        ts: first.ts,
        lsn_range,
        coll: first.coll.clone(),
        op_summary: wal::summarize_txn(group),
        affected_rows,
        doc_ids,
        undone_by: None,
    });
    // A compensating (undo) transaction always carries `caused_by` on
    // every one of its records, pointing at the transaction it reverses.
    if let Some(caused_by) = first.caused_by {
        inner.txn_index.mark_undone(caused_by, first.txn);
    }
    Ok(())
}

fn apply_single_record(inner: &mut CollectionInner, record: &WalRecord) -> StorageResult<()> {
    match record.op {
        WalOp::Insert | WalOp::Update => apply_upsert(inner, record),
        WalOp::Delete => apply_delete(inner, record),
        WalOp::CreateCollection | WalOp::DropCollection | WalOp::AlterSchema | WalOp::Reindex => {
            // Collection lifecycle ops are handled at the `Storage`
            // registry level (outside any single `Collection`); `Reindex`
            // carries no storage-state change at all, by the governing
            // principle that derived indexes are never part of storage's
            // own durable state. Nothing to apply here for any of them.
            Ok(())
        }
    }
}

fn apply_upsert(inner: &mut CollectionInner, record: &WalRecord) -> StorageResult<()> {
    let row_id = record
        .row_id
        .ok_or_else(|| StorageError::InvalidArgument("WAL insert/update record missing row_id".into()))?;
    let key = record
        .key
        .clone()
        .ok_or_else(|| StorageError::InvalidArgument("WAL insert/update record missing key".into()))?;
    let payload = record
        .payload
        .as_ref()
        .ok_or_else(|| StorageError::InvalidArgument("WAL insert/update record missing payload".into()))?;
    let vector = payload
        .vector()
        .map_err(|e| StorageError::InvalidArgument(e.to_string()))?
        .ok_or_else(|| StorageError::InvalidArgument("WAL insert/update record missing vector".into()))?;
    let fields: PayloadRow = payload.fields.clone().unwrap_or_default();

    inner.next_row_id = inner.next_row_id.max(row_id.0 + 1);
    let key_arc: Arc<str> = Arc::from(key.as_str());
    inner.key_to_row.insert(key_arc.clone(), row_id);
    inner.rows.insert(
        row_id,
        RowRecord {
            key: key_arc,
            vector: vector.into(),
            fields: Arc::new(fields.clone()),
            extra: payload.extra.clone().map(Arc::new),
            text: payload.text.clone().map(|t| Arc::from(t.as_str())),
            doc_id: record.doc_id,
            chunk_ord: record.chunk_ord,
        },
    );
    inner.live.insert(row_id.to_bitmap_index());
    inner.payload.set_row(row_id, &fields)?;

    if let Some(doc_id) = record.doc_id {
        inner.next_doc_id = inner.next_doc_id.max(doc_id.0 + 1);
        // `params` (doc-level metadata) is only ever set by
        // put_document/replace_document/delete_document. A row-level
        // put/put_batch that happens to touch an existing chunk row (rare —
        // chunks are normally only mutated via the document API) preserves
        // that row's `doc_id`/`chunk_ord` but must not touch doc metadata
        // it never carried in the first place.
        if record.params.is_some() {
            apply_doc_meta(inner, doc_id, record, &fields)?;
        }
        if let Some(entry) = inner.docs.get_mut(&doc_id) {
            entry.rows.insert(row_id.to_bitmap_index());
            entry.chunk_count = entry.rows.len() as u32;
        }
    }
    Ok(())
}

fn apply_delete(inner: &mut CollectionInner, record: &WalRecord) -> StorageResult<()> {
    let row_id = record
        .row_id
        .ok_or_else(|| StorageError::InvalidArgument("WAL delete record missing row_id".into()))?;
    inner.live.remove(row_id.to_bitmap_index());
    inner.payload.clear_row(row_id);
    if let Some(rec) = inner.rows.get(&row_id) {
        inner.key_to_row.remove(&rec.key);
    }
    if let Some(doc_id) = record.doc_id
        && let Some(entry) = inner.docs.get_mut(&doc_id) {
            entry.rows.remove(row_id.to_bitmap_index());
            entry.chunk_count = entry.rows.len() as u32;
        }
    Ok(())
}

fn apply_doc_meta(inner: &mut CollectionInner, doc_id: DocId, record: &WalRecord, doc_payload: &PayloadRow) -> StorageResult<()> {
    let Some(params) = &record.params else {
        return Ok(());
    };
    let meta: WalDocMeta =
        serde_json::from_value(params.clone()).map_err(|e| StorageError::InvalidArgument(format!("bad doc meta in WAL record: {e}")))?;
    let doc_key = record
        .doc_key
        .clone()
        .ok_or_else(|| StorageError::InvalidArgument("WAL doc-chunk record missing doc_key".into()))?;

    let entry = inner.docs.entry(doc_id).or_insert_with(|| DocEntry {
        doc_id,
        doc_key: doc_key.clone(),
        version: meta.version,
        rows: RoaringBitmap::new(),
        chunk_count: 0,
        source: meta.source.clone(),
        chunk_spec: meta.chunk_spec.clone(),
        chunker_version: meta.chunker_version,
        embedding_model: meta.embedding_model.clone(),
        created_txn: record.txn,
        updated_txn: record.txn,
        doc_payload: doc_payload.clone(),
    });
    entry.version = meta.version;
    entry.source = meta.source;
    entry.chunk_spec = meta.chunk_spec;
    entry.chunker_version = meta.chunker_version;
    entry.embedding_model = meta.embedding_model;
    entry.updated_txn = record.txn;
    entry.doc_payload = doc_payload.clone();
    inner.doc_key_to_id.insert(Arc::from(doc_key.as_str()), doc_id);
    Ok(())
}
