//! Inclusion witness storage (WP §11.6; erratum 177): hold, look up,
//! and regenerate witnesses for finalized transactions.

use std::collections::BTreeMap;

use nerv_core::types::{LegIndex, TxId};
use nerv_state::block::ShardBlock;
use nerv_witness::regenerate;
use nerv_witness::InclusionWitness;

use crate::prove::ProvedTx;

/// The wallet's witness store: (txid, leg) → witness.
#[derive(Clone, Debug, Default)]
pub struct WitnessStore {
    witnesses: BTreeMap<([u8; 32], u8), InclusionWitness>,
    /// ProvedTx by txid for regeneration context.
    transactions: BTreeMap<[u8; 32], ProvedTx>,
}

impl WitnessStore {
    pub fn new() -> WitnessStore {
        WitnessStore::default()
    }

    pub fn len(&self) -> usize {
        self.witnesses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.witnesses.is_empty()
    }

    /// Record a proved transaction for future witness regeneration.
    pub fn record_transaction(&mut self, proved: &ProvedTx) {
        self.transactions.insert(*proved.txid.as_bytes(), proved.clone());
    }

    /// Store a witness for a specific leg.
    pub fn store(&mut self, txid: &TxId, leg: LegIndex, witness: InclusionWitness) {
        self.witnesses.insert((*txid.as_bytes(), leg.as_u8()), witness);
    }

    /// Look up a witness.
    pub fn get(&self, txid: &TxId, leg: LegIndex) -> Option<&InclusionWitness> {
        self.witnesses.get(&(*txid.as_bytes(), leg.as_u8()))
    }

    /// Remove a witness (after spending, for example).
    pub fn remove(&mut self, txid: &TxId, leg: LegIndex) -> Option<InclusionWitness> {
        self.witnesses.remove(&(*txid.as_bytes(), leg.as_u8()))
    }

    /// All witnesses for a txid (the cross-shard receipt, §11.6).
    pub fn get_all_for_txid(&self, txid: &TxId) -> Vec<&InclusionWitness> {
        self.witnesses
            .range((*txid.as_bytes(), 0)..(*txid.as_bytes(), 255))
            .map(|(_, w)| w)
            .collect()
    }

    /// Regenerate a witness from archival data (erratum 177).
    pub fn regenerate_from_block(
        &mut self,
        block: &ShardBlock,
        txid: &TxId,
        leg: LegIndex,
    ) -> Result<InclusionWitness, nerv_witness::WitnessError> {
        let witness = regenerate(block, txid, leg)?;
        self.store(txid, leg, witness.clone());
        Ok(witness)
    }

    /// All stored witnesses.
    pub fn all(&self) -> impl Iterator<Item = (&([u8; 32], u8), &InclusionWitness)> {
        self.witnesses.iter()
    }

    /// Forget everything about a txid (after full spending).
    pub fn forget_txid(&mut self, txid: &TxId) {
        let keys: Vec<([u8; 32], u8)> = self
            .witnesses
            .range((*txid.as_bytes(), 0)..(*txid.as_bytes(), 255))
            .map(|(k, _)| *k)
            .collect();
        for k in keys {
            self.witnesses.remove(&k);
        }
        self.transactions.remove(txid.as_bytes());
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use nerv_core::hash::Hash256;
    use nerv_witness::GPath;

    fn witness(seed: u64, leg: u8) -> InclusionWitness {
        InclusionWitness {
            txid: TxId::from_hash(Hash256::from_bytes([seed as u8; 32])),
            leg: LegIndex::from_u8(leg),
            leaf_index: seed,
            siblings: vec![Hash256::from_bytes([seed as u8; 32]); 14],
            shard: nerv_core::types::ShardSet::genesis().ids()[7],
            height: seed,
            interval: 86_400,
            header_hash: Hash256::from_bytes([(seed + 1) as u8; 32]),
            g_path: None,
        }
    }

    #[test]
    fn store_lookup_remove() {
        let mut ws = WitnessStore::new();
        assert!(ws.is_empty());

        let txid = TxId::from_hash(Hash256::from_bytes([1u8; 32]));
        let leg = LegIndex::from_u8(0);
        let w = witness(1, 0);
        ws.store(&txid, leg, w);
        assert_eq!(ws.len(), 1);
        assert!(ws.get(&txid, leg).is_some());
        assert!(ws.get(&txid, LegIndex::from_u8(1)).is_none());

        let removed = ws.remove(&txid, leg);
        assert!(removed.is_some());
        assert!(ws.is_empty());
    }

    #[test]
    fn cross_shard_receipt() {
        let mut ws = WitnessStore::new();
        let txid = TxId::from_hash(Hash256::from_bytes([2u8; 32]));
        let w1 = witness(2, 0);
        let w2 = witness(3, 1);
        ws.store(&txid, LegIndex::from_u8(0), w1);
        ws.store(&txid, LegIndex::from_u8(1), w2);
        assert_eq!(ws.len(), 2);
        let receipt = ws.get_all_for_txid(&txid);
        assert_eq!(receipt.len(), 2);
        assert_eq!(receipt[0].leg, LegIndex::from_u8(0));
        assert_eq!(receipt[1].leg, LegIndex::from_u8(1));

        ws.forget_txid(&txid);
        assert!(ws.is_empty());
    }

    #[test]
    fn different_txids_dont_interfere() {
        let mut ws = WitnessStore::new();
        let t1 = TxId::from_hash(Hash256::from_bytes([3u8; 32]));
        let t2 = TxId::from_hash(Hash256::from_bytes([4u8; 32]));
        ws.store(&t1, LegIndex::from_u8(0), witness(3, 0));
        ws.store(&t2, LegIndex::from_u8(0), witness(4, 0));
        assert_eq!(ws.len(), 2);
        assert_eq!(ws.get_all_for_txid(&t1).len(), 1);
        assert_eq!(ws.get_all_for_txid(&t2).len(), 1);
        ws.forget_txid(&t1);
        assert_eq!(ws.len(), 1);
        assert!(ws.get(&t2, LegIndex::from_u8(0)).is_some());
    }
}

