//! Threshold-Encrypted Mempool — No Plaintext Until Block Execution.
//!
//! In NERV v2.0, the mempool stores transactions **fully encrypted**
//! under the DKG collective public key. No single node can read any
//! transaction. Only during the block production ceremony do ≥t
//! validators perform threshold decryption to reveal the batch.
//!
//! # Privacy Guarantee
//!
//! ```text
//! Client → encrypt(tx, DKG_pk) → Mempool → Block Producer
//!                                                    ↓
//!                                        Threshold Decryption Ceremony
//!                                                    ↓
//!                                        Execute decrypted batch
//!                                        Zeroize plaintext immediately
//! ```
//!
//! # Fee Market
//!
//! Each encrypted transaction includes an **unencrypted** fee field
//! (the only metadata visible) for priority ordering. The fee does
//! not leak transaction content — it merely expresses the sender's
//! willingness to pay for inclusion.

use crate::{
    ONE_NERV, BATCH_MAX_SIZE,
    BlockHeight, TxHash,
    NervError, N0ervResult,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Instant;

// ─── Mempool Entry ───────────────────────────────────────────────────────

/// An entry in the threshold-encrypted mempool.
///
/// The encrypted_payload is opaque — no node can read it.
/// The fee field is intentionally unencrypted for the fee market.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MempoolEntry {
    /// Hash of the encrypted payload (not the plaintext tx).
    pub tx_hash: TxHash,

    /// The encrypted transaction payload (ML-KEM + ChaCha20-Poly1305).
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub encrypted_payload: Vec<u8>,

    /// Fee in nano-NERV (unencrypted, for priority ordering).
    /// This is the ONLY visible metadata — it does not reveal
    /// sender, receiver, amount, or transaction type.
    pub fee_nano: u64,

    /// Gas limit requested (unencrypted, for capacity planning).
    pub gas_limit: u64,

    /// Size of the encrypted payload in bytes.
    pub size_bytes: usize,

    /// Arrival timestamp (Unix epoch millis).
    pub arrival_ts: u64,

    /// DKG session ID that produced the encryption key.
    pub dkg_session_id: [u8; 32],

    /// Whether this entry has been selected for a block.
    pub selected: bool,
}

impl MempoolEntry {
    /// Create a new mempool entry.
    pub fn new(
        tx_hash: TxHash,
        encrypted_payload: Vec<u8>,
        fee_nano: u64,
        gas_limit: u64,
        dkg_session_id: [u8; 32],
    ) -> Self {
        let size_bytes = encrypted_payload.len();
        Self {
            tx_hash,
            encrypted_payload,
            fee_nano,
            gas_limit,
            size_bytes,
            arrival_ts: current_epoch_millis(),
            dkg_session_id,
            selected: false,
        }
    }

    /// Compute the fee-per-gas ratio (for priority ordering).
    pub fn fee_per_gas(&self) -> f64 {
        if self.gas_limit == 0 {
            return 0.0;
        }
        self.fee_nano as f64 / self.gas_limit as f64
    }

    /// Effective priority score (higher = better).
    ///
    /// Combines fee-per-gas with time-in-mempool penalty
    /// to prevent starvation of lower-fee transactions.
    pub fn priority_score(&self, current_ts: u64) -> f64 {
        let fee_score = self.fee_per_gas();
        // Small time bonus: increases by 0.1% per second in mempool
        let time_in_pool = current_ts.saturating_sub(self.arrival_ts) / 1000;
        let time_bonus = 1.0 + (time_in_pool as f64 * 0.001);
        fee_score * time_bonus
    }

    /// Mark as selected for a block.
    pub fn select(&mut self) {
        self.selected = true;
    }

    /// Check if the DKG session matches the current session.
    pub fn matches_dkg_session(&self, session_id: &[u8; 32]) -> bool {
        &self.dkg_session_id == session_id
    }
}

// ─── Mempool Priority ────────────────────────────────────────────────────

/// Priority metric for mempool ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MempoolPriority {
    /// Order by fee-per-gas (highest first).
    FeePerGas,
    /// Order by arrival time (FIFO).
    ArrivalTime,
    /// Order by total fee (highest first).
    TotalFee,
}

// ─── Encrypted Mempool ───────────────────────────────────────────────────

/// The threshold-encrypted mempool.
///
/// All transactions are stored encrypted. Ordering is based on
/// the unencrypted fee field. The mempool supports:
/// - Fee-priority ordering
/// - Size limits and eviction
/// - Batch selection for block production
/// - DKG session filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedMempool {
    /// Entries indexed by tx_hash.
    entries: HashMap<[u728; 32], MempoolEntry>,

    /// Fee-ordered index: (fee_nano, tx_hash) for priority queue.
    fee_index: BTreeMap<(u64, [u8; 32]), ()>,

    /// Maximum number of entries.
    max_entries: usize,

    /// Maximum total size in bytes.
    max_total_size: usize,

    /// Default batch size for block production.
    default_batch_size: usize,

    /// Current total size in bytes.
    current_total_size: usize,

    /// Total entries added (lifetime counter).
    total_added: u64,

    /// Total entries evicted (lifetime counter).
    total_evicted: u64,
}

impl EncryptedMempool {
    /// Create a new encrypted mempool.
    pub fn new(max_entries: usize, default_batch_size: usize) -> Self {
        Self {
            entries: HashMap::new(),
            fee_index: BTreeMap::new(),
            max_entries,
            max_total_size: 500 * 1024 * 1024, // 500 MB
            default_batch_size,
            current_total_size: 0,
            total_added: 0,
            total_evicted: 0,
        }
    }

    /// Add an encrypted transaction to the mempool.
    pub fn add(&mut self, entry: MempoolEntry) -> NervResult<()> {
        // Check capacity
        if self.entries.len() >= self.max_entries {
            // Evict lowest-fee entries to make room
            self.evict_lowest_fee(1);
        }

        if self.current_total_size + entry.size_bytes > self.max_total_size {
            // Evict until we have room
            let needed = entry.size_bytes;
            let excess = (self.current_total_size + needed)
                .saturating_sub(self.max_total_size);
            // Evict entries totaling at least 'excess' bytes
            self.evict_by_size(excess);
        }

        // Check for duplicate
        let key = *entry.tx_hash.as_bytes();
        if self.entries.contains_key(&key) {
            return Err(NervError::Network(format!(
                "duplicate tx {} in mempool", entry.tx_hash
            )));
        }

        // Add to fee index
        let fee_key = (entry.fee_nano, key);
        self.fee_index.insert(fee_key, ());

        // Add to entries
        self.current_total_size += entry.size_bytes;
        self.entries.insert(key, entry);
        self.total_added += 1;

        Ok(())
    }

    /// Remove a transaction from the mempool.
    pub fn remove(&mut self, tx_hash: &TxHash) -> Option<MempoolEntry> {
        let key = *tx_hash.as_bytes();
        if let Some(entry) = self.entries.remove(&key) {
            let fee_key = (entry.fee_nano, key);
            self.fee_index.remove(&fee_key);
            self.current_total_size = self.current_total_size
                .saturating_sub(entry.size_bytes);
            Some(entry)
        } else {
            None
        }
    }

    /// Get an entry by tx_hash.
    pub fn get(&self, tx_hash: &TxHash) -> Option<&MempoolEntry> {
        self.entries.get(tx_hash.as_bytes())
    }

    /// Check if a transaction is in the mempool.
    pub fn contains(&self, tx_hash: &TxHash) -> bool {
        self.entries.contains_key(tx_hash.as_bytes())
    }

    /// Number of entries in the mempool.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the mempool is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total size in bytes.
    pub fn total_size(&self) -> usize {
        self.current_total_size
    }

    /// Select a batch of transactions for block production.
    ///
    /// Returns the highest-fee entries that:
    /// - Match the current DKG session
    /// - Have not already been selected
    /// - Fit within the gas limit
    pub fn select_batch(
        &mut self,
        batch_size: usize,
        gas_limit: u64,
        dkg_session_id: &[u8; 32],
    ) -> Vec<MempoolEntry> {
        let batch_size = if batch_size == 0 {
            self.default_batch_size
        } else {
            batch_size
        };

        let current_ts = current_epoch_millis();
        let mut batch = Vec::with_capacity(batch_size);
        let mut total_gas = 0u64;

        // Iterate from highest fee to lowest (BTreeMap is sorted)
        for ((fee, tx_key), _) in self.fee_index.iter().rev() {
            if batch.len() >= batch_size {
                break;
            }

            if let Some(entry) = self.entries.get_mut(tx_key) {
                // Skip if already selected
                if entry.selected {
                    continue;
                }

                // Skip if wrong DKG session
                if !entry.matches_dkg_session(dkg_session_id) {
                    continue;
                }

                // Check gas limit
                if total_gas + entry.gas_limit > gas_limit {
                    continue;
                }

                total_gas += entry.gas_limit;
                entry.select();
                batch.push(entry.clone());
            }
        }

        batch
    }

    /// Mark entries as processed (remove from mempool).
    pub fn mark_processed(&mut self, tx_hashes: &[TxHash]) {
        for tx_hash in tx_hashes {
            self.remove(tx_hash);
        }
    }

    /// Evict the lowest-fee entries.
    fn evict_lowest_fee(&mut self, count: usize) {
        let keys_to_evict: Vec<[u8; 32]> = self.fee_index.iter()
            .take(count)
            .map(|((_, k), _)| *k)
            .collect();

        for key in keys_to_evict {
            if let Some(entry) = self.entries.remove(&key) {
                let fee_key = (entry.fee_nano, key);
                self.fee_index.remove(&fee_key);
                self.current_total_size = self.current_total_size
                    .saturating_sub(entry.size_bytes);
                self.total_evicted += 1;
            }
        }
    }

    /// Evict entries totaling at least `needed_bytes` bytes.
    fn evict_by_size(&mut self, needed_bytes: usize) {
        let mut freed = 0usize;
        let keys_to_evict: Vec<[u8; 32]> = self.fee_index.iter()
            .take_while(|_| freed < needed_bytes)
            .map(|((_, k), _)| *k)
            .collect();

        for key in keys_to_evict {
            if let Some(entry) = self.entries.remove(&key) {
                let fee_key = (entry.fee_nano, key);
                self.fee_index.remove(&fee_key);
                freed += entry.size_bytes;
                self.current_total_size = self.current_total_size
                    .saturating_sub(entry.size_bytes);
                self.total_evicted += 1;
            }
        }
    }

    /// Get statistics about the mempool.
    pub fn stats(&self) -> MempoolStats {
        let fees: Vec<u64> = self.entries.values().map(|e| e.fee_nano).collect();
        let avg_fee = if fees.is_empty() {
            0.0
        } else {
            fees.iter().sum::<u64>() as f64 / fees.len() as f64
        };

        MempoolStats {
            entry_count: self.entries.len(),
            total_size_bytes: self.current_total_size,
            min_fee_nano: fees.iter().copied().min().unwrap_or(0),
            max_fee_nano: fees.iter().copied().max().unwrap_or(0),
            avg_fee_nano: avg_fee,
            total_added: self.total_added,
            total_evicted: self.total_evicted,
        }
    }

    /// Clear all entries (for testing or network reset).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.fee_index.clear();
        self.current_total_size = 0;
    }
}

// ─── Mempool Statistics ──────────────────────────────────────────────────

/// Statistics about the mempool state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MempoolStats {
    /// Number of entries.
    pub entry_count: usize,

    /// Total size in bytes.
    pub total_size_bytes: usize,

    /// Minimum fee in the pool (nano-NERV).
    pub min_fee_nano: u64,

    /// Maximum fee in the pool (nano-NERV).
    pub max_fee_nano: u64,

    /// Average fee (nano-NERV).
    pub avg_fee_nano: f64,

    /// Lifetime total entries added.
    pub total_added: u64,

    /// Lifetime total entries evicted.
    pub total_evicted: u64,
}

// ─── Helpers ─────────────────────────────────────────────────────────────

fn current_epoch_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(fee: u64, id: u8) -> MempoolEntry {
        MempoolEntry::new(
            TxHash::from_bytes([id; 32]),
            vec![0u8; 100], // Encrypted payload
            fee,
            100_000, // Gas limit
            [0u8; 32], // DKG session
        )
    }

    #[test]
    fn test_mempool_entry_creation() {
        let entry = make_entry(100 * ONE_NERV, 1);
        assert_eq!(entry.fee_nano, 100 * ONE_NERV);
        assert!(!entry.selected);
        assert!(entry.fee_per_gas() > 0.0);
    }

    #[test]
    fn test_mempool_entry_priority() {
        let entry = make_entry(100 * ONE_NERV, 1);
        let score = entry.priority_score(entry.arrival_ts);
        assert!(score > 0.0);
    }

    #[test]
    fn test_mempool_add() {
        let mut mempool = EncryptedMempool::new(1000, 100);
        let entry = make_entry(100 * ONE_NERV, 1);
        assert!(mempool.add(entry).is_ok());
        assert_eq!(mempool.len(), 1);
    }

    #[test]
    fn test_mempool_duplicate() {
        let mut mempool = EncryptedMempool::new(1000, 100);
        let entry = make_entry(100 * ONE_NERV, 1);
        mempool.add(entry.clone()).unwrap();
        // Same tx_hash → duplicate
        assert!(mempool.add(entry).is_err());
    }

    #[test]
    fn test_mempool_remove() {
        let mut mempool = EncryptedMempool::new(1000, 100);
        let entry = make_entry(100 * ONE_NERV, 1);
        let tx_hash = entry.tx_hash;
        mempool.add(entry).unwrap();
        let removed = mempool.remove(&tx_hash);
        assert!(removed.is_some());
        assert_eq!(mempool.len(), 0);
    }

    #[test]
    fn test_mempool_contains() {
        let mut mempool = EncryptedMempool::new(1000, 100);
        let entry = make_entry(100 * ONE_NERV, 1);
        let tx_hash = entry.tx_hash;
        mempool.add(entry).unwrap();
        assert!(mempool.contains(&tx_hash));
    }

    #[test]
    fn test_mempool_select_batch() {
        let mut mempool = EncryptedMempool::new(1000, 100);

        // Add entries with different fees
        for i in 1..=10u8 {
            let fee = (10 - i as u64 + 1) * ONE_NERV; // 10, 9, 8, ... 1
            let entry = MempoolEntry::new(
                TxHash::from_bytes([i; 32]),
                vec![0u8; 100],
                fee,
                100_000,
                [0u8; 32],
            );
            mempool.add(entry).unwrap();
        }

        // Select batch of 5
        let batch = mempool.select_batch(5, 1_000_000_000, &[0u8; 32]);
        assert_eq!(batch.len(), 5);

        // Highest-fee entries should be selected first
        // Fee 10 was entry with id=1, fee 9 was id=2, etc.
        assert!(batch[0].fee_nano >= batch[1].fee_nano);
    }

    #[test]
    fn test_mempool_select_batch_gas_limit() {
        let mut mempool = EncryptedMempool::new(1000, 100);

        for i in 1..=5u8 {
            let mut entry = make_entry(100 * ONE_NERV, i);
            // Override gas to make some too expensive
            let gas = if i <= 3 { 100_000 } else { 1_000_000 };
            entry.gas_limit = gas;
            mempool.add(entry).unwrap();
        }

        // Gas limit allows only 3 entries
        let batch = mempool.select_batch(10, 350_000, &[0u8; 32]);
        assert!(batch.len() <= 3);
    }

    #[test]
    fn test_mempool_dkg_session_filter() {
        let mut mempool = EncryptedMempool::new(1000, 100);

        // Add entry with session A
        let mut entry_a = make_entry(100 * ONE_NERV, 1);
        entry_a.dkg_session_id = [1u8; 32];
        mempool.add(entry_a).unwrap();

        // Add entry with session B
        let mut entry_b = make_entry(200 * ONE_NERV, 2);
        entry_b.dkg_session_id = [2u8; 32];
        mempool.add(entry_b).unwrap();

        // Select batch with session A
        let batch = mempool.select_batch(10, 1_000_000_000, &[1u8; 32]);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].dkg_session_id, [1u8; 32]);
    }

    #[test]
    fn test_mempool_eviction() {
        let mut mempool = EncryptedMempool::new(5, 100); // Max 5 entries

        for i in 1..=10u8 {
            let fee = (i as u64) * ONE_NERV;
            let entry = make_entry(fee, i);
            mempool.add(entry).unwrap();
        }

        // After adding 10 to a max-5 pool, some should have been evicted
        assert!(mempool.len() <= 5);
        assert!(mempool.total_evicted > 0);
    }

    #[test]
    fn test_mempool_stats() {
        let mut mempool = EncryptedMempool::new(1000, 100);
        mempool.add(make_entry(100 * ONE_NERV, 1)).unwrap();
        mempool.add(make_entry(200 * ONE_NERV, 2)).unwrap();

        let stats = mempool.stats();
        assert_eq!(stats.entry_count, 2);
        assert!(stats.total_size_bytes > 0);
        assert!(stats.avg_fee_nano > 0.0);
    }

    #[test]
    fn test_mempool_mark_processed() {
        let mut mempool = EncryptedMempool::new(1000, 100);
        let entry1 = make_entry(100 * ONE_NERV, 1);
        let entry2 = make_entry(200 * ONE_NERV, 2);
        let h1 = entry1.tx_hash;
        let h2 = entry2.tx_hash;

        mempool.add(entry1).unwrap();
        mempool.add(entry2).unwrap();
        assert_eq!(mempool.len(), 2);

        mempool.mark_processed(&[h1, h2]);
        assert_eq!(mempool.len(), 0);
    }

    #[test]
    fn test_mempool_clear() {
        let mut mempool = EncryptedMempool::new(1000, 100);
        mempool.add(make_entry(100 * ONE_NERV, 1)).unwrap();
        mempool.clear();
        assert_eq!(mempool.len(), 0);
        assert_eq!(mempool.total_size(), 0);
    }

    #[test]
    fn test_mempool_entry_fee_per_gas() {
        let entry = MempoolEntry::new(
            TxHash::from_bytes([1u8; 32]),
            vec![0u8; 100],
            1000,
            500,
            [0u8; 32],
        );
        assert!((entry.fee_per_gas() - 2.0).abs() < 1e-10);
    }
}
