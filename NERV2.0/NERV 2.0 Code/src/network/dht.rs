//! Distributed Hash Table — Kademlia DHT for NERV.
//!
//! The DHT serves three purposes in NERV:
//!
//! 1. **Peer Discovery**: Find and connect to other nodes
//! 2. **Shard Data Location**: Find which peers store data for a shard
//! 3. **VDW Storage**: Distributed storage of Verifiable Delay Witnesses
//!
//! # Key Namespaces
//!
//! ```text
//! /nerv/peer/<peer_id>       → Peer multiaddrs
//! /nerv/shard/<shard_id>     → Shard metadata + replica locations
//! /nerv/vdw/<tx_hash>        → VDW data
//! /nerv/weights/<epoch>      → Network weight hash
//! /nerv/embedding/<shard_id> → Current embedding root
//! ```

use crate::{
    ShardId, BlockHeight, Epoch, TxHash, EmbeddingRoot,
    EMBEDDING_BYTES,
    NervError, NervResult,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// ─── DHT Key ─────────────────────────────────────────────────────────────

/// A typed key in the DHT namespace.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DhtKey {
    /// The namespace prefix.
    pub namespace: DhtNamespace,

    /// The key bytes (hash of the identifier).
    pub key_bytes: Vec<u8>,

    /// Expiration timestamp (Unix epoch millis, 0 = never expires).
    pub expires_at: u64,
}

/// DHT key namespaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DhtNamespace {
    /// Peer records (peer_id → multiaddrs).
    Peer,
    /// Shard metadata (shard_id → metadata).
    Shard,
    /// VDW data (tx_hash → VDW bytes).
    Vdw,
    /// Network weights (epoch → weight hash).
    Weights,
    /// Embedding roots (shard_id → root hash).
    Embedding,
}

impl DhtNamespace {
    /// Get the namespace prefix string.
    pub fn prefix(&self) -> &'static str {
        match self {
            Self::Peer => "/nerv/peer/",
            Self::Shard => "/nerv/shard/",
            Self::Vdw => "/nerv/vdw/",
            Self::Weights => "/nerv/weights/",
            Self::Embedding => "/nerv/embedding/",
        }
    }
}

impl std::fmt::Display for DhtNamespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.prefix())
    }
}

impl DhtKey {
    /// Create a peer key.
    pub fn peer(peer_id: &[u8]) -> Self {
        Self {
            namespace: DhtNamespace::Peer,
            key_bytes: peer_id.to_vec(),
            expires_at: 0,
        }
    }

    /// Create a shard key.
    pub fn shard(shard_id: ShardId) -> Self {
        Self {
            namespace: DhtNamespace::Shard,
            key_bytes: shard_id.0.to_le_bytes().to_vec(),
            expires_at: 0,
        }
    }

    /// Create a VDW key.
    pub fn vdw(tx_hash: &TxHash) -> Self {
        Self {
            namespace: DhtNamespace::Vdw,
            key_bytes: tx_hash.as_bytes().to_vec(),
            expires_at: current_epoch_millis() + (5 * 365 * 24 * 3600 * 1000), // 5 years
        }
    }

    /// Create a weights key.
    pub fn weights(epoch: Epoch) -> Self {
        Self {
            namespace: DhtNamespace::Weights,
            key_bytes: epoch.0.to_le_bytes().to_vec(),
            expires_at: 0,
        }
    }

    /// Create an embedding root key.
    pub fn embedding(shard_id: ShardId) -> Self {
        Self {
            namespace: DhtNamespace::Embedding,
            key_bytes: shard_id.0.to_le_bytes().to_vec(),
            expires_at: 0,
        }
    }

    /// Create the full wire-format key (namespace + hash).
    pub fn wire_key(&self) -> Vec<u8> {
        let mut key = self.namespace.prefix().as_bytes().to_vec();
        key.extend_from_slice(&self.key_bytes);
        key
    }

    /// Check if this key has expired.
    pub fn is_expired(&self) -> bool {
        if self.expires_at == 0 {
            return false;
        }
        current_epoch_millis() > self.expires_at
    }

    /// With a custom expiration.
    pub fn with_expiration(mut self, expires_at: u64) -> Self {
        self.expires_at = expires_at;
        self
    }
}

// ─── DHT Record ──────────────────────────────────────────────────────────

/// A record stored in the DHT.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DhtRecord {
    /// The key.
    pub key: DhtKey,

    /// The value bytes.
    pub value: Vec<u8>,

    /// Publisher's peer ID.
    pub publisher: Vec<u8>,

    /// Time of publication (Unix epoch millis).
    pub published_at: u64,

    /// BLAKE3 hash of the value for integrity.
    pub value_hash: [u8; 32],
}

impl DhtRecord {
    /// Create a new DHT record.
    pub fn new(key: DhtKey, value: Vec<u8>, publisher: Vec<u8>) -> Self {
        let value_hash = blake3::hash(&value).into();
        Self {
            key,
            value,
            publisher,
            published_at: current_epoch_millis(),
            value_hash,
        }
    }

    /// Verify the record's integrity.
    pub fn verify_integrity(&self) -> bool {
        let computed: [u8; 32] = blake3::hash(&self.value).into();
        computed == self.value_hash
    }

    /// Size of the value in bytes.
    pub fn size(&self) -> usize {
        self.value.len()
    }
}

// ─── DHT Service ─────────────────────────────────────────────────────────

/// The DHT service providing key-value storage and lookup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DhtService {
    /// Local record cache (wire_key → record).
    records: HashMap<Vec<u8>, DhtRecord>,

    /// Provider records (which peers provide which keys).
    providers: HashMap<Vec<u8>, Vec<Vec<u8>>>,

    /// Kademlia replication factor.
    replication_factor: usize,

    /// Maximum record size (bytes).
    max_record_size: usize,

    /// Total records stored.
    total_stored: u64,

    /// Total lookups performed.
    total_lookups: u64,
}

impl DhtService {
    /// Create a new DHT service.
    pub fn new(replication_factor: usize) -> Self {
        Self {
            records: HashMap::new(),
            providers: HashMap::new(),
            replication_factor,
            max_record_size: 1024 * 1024, // 1 MB
            total_stored: 0,
            total_lookups: 0,
        }
    }

    /// Put a record into the DHT.
    pub fn put(&mut self, record: DhtRecord) -> NervResult<()> {
        if record.key.is_expired() {
            return Err(NervError::Network("cannot store expired record".into()));
        }

        if record.value.len() > self.max_record_size {
            return Err(NervError::Network(format!(
                "record too large: {} bytes (max {})",
                record.value.len(), self.max_record_size
            )));
        }

        let wire_key = record.key.wire_key();

        // Check TTL of existing record
        if let Some(existing) = self.records.get(&wire_key) {
            if existing.published_at > record.published_at {
                // Newer record already exists
                return Ok(());
            }
        }

        self.records.insert(wire_key, record);
        self.total_stored += 1;
        Ok(())
    }

    /// Get a record from the DHT.
    pub fn get(&mut self, key: &DhtKey) -> Option<&DhtRecord> {
        self.total_lookups += 1;
        let wire_key = key.wire_key();
        let record = self.records.get(&wire_key)?;

        if record.key.is_expired() {
            None
        } else {
            Some(record)
        }
    }

    /// Remove a record.
    pub fn remove(&mut self, key: &DhtKey) -> Option<DhtRecord> {
        let wire_key = key.wire_key();
        self.records.remove(&wire_key)
    }

    /// Register a provider for a key.
    pub fn add_provider(&mut self, key: &DhtKey, peer_id: Vec<u8>) {
        let wire_key = key.wire_key();
        let providers = self.providers.entry(wire_key).or_default();
        if !providers.contains(&peer_id) {
            providers.push(peer_id);
        }
    }

    /// Get providers for a key.
    pub fn get_providers(&mut self, key: &DhtKey) -> Vec<Vec<u8>> {
        self.total_lookups += 1;
        let wire_key = key.wire_key();
        self.providers.get(&wire_key).cloned().unwrap_or_default()
    }

    /// Store a VDW in the DHT.
    pub fn store_vdw(
        &mut self,
        tx_hash: &TxHash,
        vdw_data: Vec<u8>,
        publisher: Vec<u8>,
    ) -> NervResult<()> {
        let key = DhtKey::vdw(tx_hash);
        let record = DhtRecord::new(key, vdw_data, publisher);
        self.put(record)
    }

    /// Retrieve a VDW from the DHT.
    pub fn get_vdw(&mut self, tx_hash: &TxHash) -> Option<Vec<u8>> {
        let key = DhtKey::vdw(tx_hash);
        self.get(&key).map(|r| r.value.clone())
    }

    /// Store shard metadata.
    pub fn store_shard_metadata(
        &mut self,
        shard_id: ShardId,
        metadata: Vec<u8>,
        publisher: Vec<u8>,
    ) -> NervResult<()> {
        let key = DhtKey::shard(shard_id);
        let record = DhtRecord::new(key, metadata, publisher);
        self.put(record)
    }

    /// Get shard metadata.
    pub fn get_shard_metadata(&mut self, shard_id: ShardId) -> Option<Vec<u8>> {
        let key = DhtKey::shard(shard_id);
        self.get(&key).map(|r| r.value.clone())
    }

    /// Store the current embedding root for a shard.
    pub fn store_embedding_root(
        &mut self,
        shard_id: ShardId,
        root: &EmbeddingRoot,
        publisher: Vec<u8>,
    ) -> NervResult<()> {
        let key = DhtKey::embedding(shard_id);
        let record = DhtRecord::new(key, root.as_bytes().to_vec(), publisher);
        self.put(record)
    }

    /// Get the embedding root for a shard.
    pub fn get_embedding_root(&mut self, shard_id: ShardId) -> Option<EmbeddingRoot> {
        let key = DhtKey::embedding(shard_id);
        let record = self.get(&key)?;
        if record.value.len() == 32 {
            let mut bytes = [0u8; 32];
            bytes.copy_from_slice(&record.value);
            Some(EmbeddingRoot::from_bytes(bytes))
        } else {
            None
        }
    }

    /// Store network weights for an epoch.
    pub fn store_weights(
        &mut self,
        epoch: Epoch,
        weight_hash: [u8; 32],
        publisher: Vec<u8>,
    ) -> NervResult<()> {
        let key = DhtKey::weights(epoch);
        let record = DhtRecord::new(key, weight_hash.to_vec(), publisher);
        self.put(record)
    }

    /// Get network weights for an epoch.
    pub fn get_weights(&mut self, epoch: Epoch) -> Option<[u8; 32]> {
        let key = DhtKey::weights(epoch);
        let record = self.get(&key)?;
        if record.value.len() == 32 {
            let mut bytes = [0u8; 32];
            bytes.copy_from_slice(&record.value);
            Some(bytes)
        } else {
            None
        }
    }

    /// Number of records stored.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Number of provider records.
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }

    /// Purge expired records.
    pub fn purge_expired(&mut self) -> usize {
        let expired_keys: Vec<Vec<u8>> = self.records.iter()
            .filter(|(_, r)| r.key.is_expired())
            .map(|(k, _)| k.clone())
            .collect();

        let count = expired_keys.len();
        for key in expired_keys {
            self.records.remove(&key);
            self.providers.remove(&key);
        }
        count
    }

    /// Replication factor.
    pub fn replication_factor(&self) -> usize {
        self.replication_factor
    }
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

    #[test]
    fn test_dht_key_peer() {
        let key = DhtKey::peer(&[1u8; 32]);
        assert_eq!(key.namespace, DhtNamespace::Peer);
        assert!(!key.is_expired());
        assert!(key.wire_key().starts_with(b"/nerv/peer/"));
    }

    #[test]
    fn test_dht_key_shard() {
        let key = DhtKey::shard(ShardId::new(42));
        assert_eq!(key.namespace, DhtNamespace::Shard);
        assert!(key.wire_key().starts_with(b"/nerv/shard/"));
    }

    #[test]
    fn test_dht_key_vdw() {
        let tx_hash = TxHash::from_bytes([1u8; 32]);
        let key = DhtKey::vdw(&tx_hash);
        assert_eq!(key.namespace, DhtNamespace::Vdw);
        assert!(!key.is_expired()); // 5-year TTL
    }

    #[test]
    fn test_dht_record_integrity() {
        let key = DhtKey::shard(ShardId::new(1));
        let record = DhtRecord::new(key, b"test data".to_vec(), vec![0u8; 32]);
        assert!(record.verify_integrity());
    }

    #[test]
    fn test_dht_record_tampered() {
        let key = DhtKey::shard(ShardId::new(1));
        let mut record = DhtRecord::new(key, b"test data".to_vec(), vec![0u8; 32]);
        record.value[0] ^= 0xFF; // Tamper
        assert!(!record.verify_integrity());
    }

    #[test]
    fn test_dht_service_put_get() {
        let mut dht = DhtService::new(20);
        let key = DhtKey::shard(ShardId::new(1));
        let record = DhtRecord::new(key.clone(), b"shard data".to_vec(), vec![0u8; 32]);

        dht.put(record).unwrap();
        let retrieved = dht.get(&key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value, b"shard data");
    }

    #[test]
    fn test_dht_service_overwrite() {
        let mut dht = DhtService::new(20);
        let key = DhtKey::shard(ShardId::new(1));

        let r1 = DhtRecord::new(key.clone(), b"old".to_vec(), vec![0u8; 32]);
        dht.put(r1).unwrap();

        let mut r2 = DhtRecord::new(key.clone(), b"new".to_vec(), vec![0u8; 32]);
        r2.published_at += 1000; // Newer
        dht.put(r2).unwrap();

        assert_eq!(dht.get(&key).unwrap().value, b"new");
    }

    #[test]
    fn test_dht_service_remove() {
        let mut dht = DhtService::new(20);
        let key = DhtKey::shard(ShardId::new(1));
        let record = DhtRecord::new(key.clone(), b"data".to_vec(), vec![0u8; 32]);
        dht.put(record).unwrap();
        dht.remove(&key);
        assert!(dht.get(&key).is_none());
    }

    #[test]
    fn test_dht_service_providers() {
        let mut dht = DhtService::new(20);
        let key = DhtKey::shard(ShardId::new(1));
        dht.add_provider(&key, vec![1u8; 32]);
        dht.add_provider(&key, vec![2u8; 32]);
        assert_eq!(dht.get_providers(&key).len(), 2);
    }

    #[test]
    fn test_dht_service_vdw() {
        let mut dht = DhtService::new(20);
        let tx_hash = TxHash::from_bytes([1u8; 32]);
        dht.store_vdw(&tx_hash, b"vdw data".to_vec(), vec![0u8; 32]).unwrap();
        assert_eq!(dht.get_vdw(&tx_hash), Some(b"vdw data".to_vec()));
    }

    #[test]
    fn test_dht_service_embedding_root() {
        let mut dht = DhtService::new(20);
        let root = EmbeddingRoot::from_bytes([42u8; 32]);
        dht.store_embedding_root(ShardId::new(1), &root, vec![0u8; 32]).unwrap();
        assert_eq!(dht.get_embedding_root(ShardId::new(1)), Some(root));
    }

    #[test]
    fn test_dht_service_weights() {
        let mut dht = DhtService::new(20);
        let weight_hash = [99u8; 32];
        dht.store_weights(Epoch::from(5), weight_hash, vec![0u8; 32]).unwrap();
        assert_eq!(dht.get_weights(Epoch::from(5)), Some(weight_hash));
    }

    #[test]
    fn test_dht_namespace_prefix() {
        assert_eq!(DhtNamespace::Peer.prefix(), "/nerv/peer/");
        assert_eq!(DhtNamespace::Vdw.prefix(), "/nerv/vdw/");
    }
}
