//! The `TxnIndex` (master plan Layer 2, *Undo targeting*): the unit of
//! undo is the transaction, not a positional count. Retains the last
//! `history_limit` transactions and a dense `last_txn` map for write-write
//! conflict detection — undoing transaction *T* is conflict-free iff
//! `last_txn[r] == T` for every row *T* touched.

use mara_proto::{Conflict, RowId, TxnEntry, TxnId};
use std::collections::HashMap;

pub(crate) struct TxnIndex {
    entries: HashMap<TxnId, TxnEntry>,
    /// Insertion order, oldest first — drives retention trimming and
    /// chronological scans (`history`, `LastN`).
    order: Vec<TxnId>,
    last_txn: HashMap<RowId, TxnId>,
    history_limit: usize,
}

impl TxnIndex {
    pub(crate) fn new(history_limit: usize) -> Self {
        TxnIndex {
            entries: HashMap::new(),
            order: Vec::new(),
            last_txn: HashMap::new(),
            history_limit: history_limit.max(1),
        }
    }

    /// Records a newly-committed transaction and updates `last_txn` for
    /// every row it touched. Trims the oldest entry once `history_limit`
    /// is exceeded — trimmed transactions can no longer be targeted by
    /// `undo --txn`, matching `undo.txn_history_limit`'s documented effect.
    pub(crate) fn record(&mut self, entry: TxnEntry) {
        for idx in entry.affected_rows.iter() {
            self.last_txn.insert(RowId::from_bitmap_index(idx), entry.txn_id);
        }
        self.order.push(entry.txn_id);
        self.entries.insert(entry.txn_id, entry);
        while self.order.len() > self.history_limit {
            let oldest = self.order.remove(0);
            self.entries.remove(&oldest);
        }
    }

    pub(crate) fn get(&self, txn_id: TxnId) -> Option<&TxnEntry> {
        self.entries.get(&txn_id)
    }

    pub(crate) fn mark_undone(&mut self, txn_id: TxnId, undone_by: TxnId) {
        if let Some(e) = self.entries.get_mut(&txn_id) {
            e.undone_by = Some(undone_by);
        }
    }

    /// Every later transaction that has since touched one of `entry`'s
    /// rows — the write-write conflict set `undo --txn` refuses against
    /// unless `--force` is given.
    pub(crate) fn conflicts_for(&self, entry: &TxnEntry) -> Vec<Conflict> {
        let mut row_counts: HashMap<TxnId, u64> = HashMap::new();
        for idx in entry.affected_rows.iter() {
            let row_id = RowId::from_bitmap_index(idx);
            if let Some(&last) = self.last_txn.get(&row_id)
                && last != entry.txn_id {
                    *row_counts.entry(last).or_insert(0) += 1;
                }
        }
        let mut conflicts: Vec<Conflict> = row_counts
            .into_iter()
            .filter_map(|(txn_id, row_count)| {
                self.entries.get(&txn_id).map(|e| Conflict {
                    conflicting_txn: txn_id,
                    ts: e.ts,
                    principal: e.principal.clone(),
                    op_summary: e.op_summary.clone(),
                    row_count,
                })
            })
            .collect();
        conflicts.sort_by_key(|c| c.ts);
        conflicts
    }

    /// Entries in chronological order (oldest first).
    pub(crate) fn iter_chronological(&self) -> impl DoubleEndedIterator<Item = &TxnEntry> {
        self.order.iter().filter_map(|id| self.entries.get(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use mara_proto::{PrincipalId, SessionId};
    use roaring::RoaringBitmap;

    fn entry(txn_id: TxnId, rows: &[u32]) -> TxnEntry {
        TxnEntry {
            txn_id,
            session: SessionId("s".into()),
            principal: PrincipalId("p".into()),
            ts: Utc::now(),
            lsn_range: (mara_proto::Lsn::ZERO, mara_proto::Lsn::ZERO),
            coll: "docs".into(),
            op_summary: "test".into(),
            affected_rows: rows.iter().copied().collect(),
            doc_ids: vec![],
            undone_by: None,
        }
    }

    #[test]
    fn conflict_free_undo_when_no_later_txn_touched_the_rows() {
        let mut idx = TxnIndex::new(100);
        let a = TxnId::new();
        idx.record(entry(a, &[1, 2, 3]));
        assert!(idx.conflicts_for(idx.get(a).unwrap()).is_empty());
    }

    #[test]
    fn detects_conflict_when_a_later_txn_touches_the_same_row() {
        let mut idx = TxnIndex::new(100);
        let a = TxnId::new();
        let b = TxnId::new();
        idx.record(entry(a, &[1, 2, 3]));
        idx.record(entry(b, &[2]));

        let conflicts = idx.conflicts_for(idx.get(a).unwrap());
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].conflicting_txn, b);
        assert_eq!(conflicts[0].row_count, 1);
    }

    #[test]
    fn retention_limit_forgets_the_oldest_transactions() {
        let mut idx = TxnIndex::new(2);
        let a = TxnId::new();
        let b = TxnId::new();
        let c = TxnId::new();
        idx.record(entry(a, &[1]));
        idx.record(entry(b, &[2]));
        idx.record(entry(c, &[3]));
        assert!(idx.get(a).is_none(), "oldest entry must be trimmed once the limit is exceeded");
        assert!(idx.get(b).is_some());
        assert!(idx.get(c).is_some());
    }

    #[test]
    fn mark_undone_is_visible_on_the_original_entry() {
        let mut idx = TxnIndex::new(100);
        let a = TxnId::new();
        let undo_txn = TxnId::new();
        idx.record(entry(a, &[1]));
        idx.mark_undone(a, undo_txn);
        assert_eq!(idx.get(a).unwrap().undone_by, Some(undo_txn));
    }

    #[test]
    fn chronological_order_matches_insertion() {
        let mut idx = TxnIndex::new(100);
        let a = TxnId::new();
        let b = TxnId::new();
        idx.record(entry(a, &[1]));
        idx.record(entry(b, &[2]));
        let order: Vec<TxnId> = idx.iter_chronological().map(|e| e.txn_id).collect();
        assert_eq!(order, vec![a, b]);
    }

    #[test]
    fn empty_affected_rows_never_conflicts() {
        let mut idx = TxnIndex::new(100);
        let a = TxnId::new();
        idx.record(entry(a, &[]));
        assert!(idx.conflicts_for(idx.get(a).unwrap()).is_empty());
        let _ = RoaringBitmap::new(); // sanity: affected_rows accepts an empty bitmap
    }
}
