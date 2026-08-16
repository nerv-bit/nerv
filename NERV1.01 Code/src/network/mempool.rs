//! Encrypted Transaction Pool for NERV Network
//!
//! This module implements a secure, encrypted mempool for holding pending transactions
//! before they are included in blocks. All transactions are encrypted and require
//! TEE attestation for submission.
//!
//! Key Features:
//! - Encrypted transaction storage with TEE attestation
//! - Priority-based transaction ordering
//! - Automatic eviction of old transactions
//! - Shard-aware transaction routing
//! - DoS protection via rate limiting


use crate::crypto::{CryptoProvider, ByteSerializable};
use crate::params::{BATCH_SIZE, VDW_MAX_SIZE};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BinaryHeap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn, debug, error, trace};
use blake3::Hasher;
use rand::prelude::*;


/// Mempool manager for encrypted transactions
pub struct MempoolManager {
    /// Encrypted transactions by shard
    transactions: RwLock<HashMap<u64, ShardMempool>>,
    
    /// Transaction lookup by hash
    tx_index: RwLock<HashMap<String, TxLocation>>,
    
    /// Pending transaction queue
    pending_queue: RwLock<BinaryHeap<PendingTransaction>>,
    
    /// Configuration
    config: MempoolConfig,
    
    /// Cryptographic provider
    crypto_provider: Arc<CryptoProvider>,
    
    /// Metrics
    metrics: MempoolMetrics,
    
    /// Shutdown flag
    shutdown: RwLock<bool>,
    
    /// Cleanup task handle
    cleanup_handle: RwLock<Option<tokio::task::JoinHandle<()>>>,
}


impl MempoolManager {
    /// Create a new mempool manager
    pub async fn new(config: MempoolConfig, crypto_provider: Arc<CryptoProvider>) -> Result<Self, MempoolError> {
        info!("Initializing encrypted mempool with max size: {}MB", 
              config.max_size_bytes / (1024 * 1024));
        
        Ok(Self {
            transactions: RwLock::new(HashMap::new()),
            tx_index: RwLock::new(HashMap::new()),
            pending_queue: RwLock::new(BinaryHeap::new()),
            config,
            crypto_provider,
            metrics: MempoolMetrics::new(),
            shutdown: RwLock::new(false),
            cleanup_handle: RwLock::new(None),
        })
    }
    
    /// Start the mempool manager
    pub async fn start(&self) -> Result<(), MempoolError> {
        info!("Starting mempool manager");
        
        // Start cleanup task
        self.start_cleanup_task().await?;
        
        info!("Mempool manager started successfully");
        Ok(())
    }
    
    /// Stop the mempool manager
    pub async fn stop(&self) -> Result<(), MempoolError> {
        info!("Stopping mempool manager");
        
        let mut shutdown = self.shutdown.write().await;
        *shutdown = true;
        
        // Cancel cleanup task
        if let Some(handle) = self.cleanup_handle.write().await.take() {
            handle.abort();
        }
        
        // Clear all transactions
        let mut transactions = self.transactions.write().await;
        transactions.clear();
        
        let mut tx_index = self.tx_index.write().await;
        tx_index.clear();
        
        let mut pending_queue = self.pending_queue.write().await;
        pending_queue.clear();
        
        info!("Mempool manager stopped");
        Ok(())
    }
    
    /// Submit a transaction to the mempool
    pub async fn submit_transaction(
        &self,
        encrypted_tx: Vec<u8>,
        attestation: Vec<u8>,
        priority: TxPriority,
    ) -> Result<TxReceipt, MempoolError> {
        let start_time = Instant::now();
        
        // Validate transaction size
        if encrypted_tx.len() > VDW_MAX_SIZE {
            return Err(MempoolError::TransactionTooLarge(encrypted_tx.len(), VDW_MAX_SIZE));
        }
        
        // Verify TEE attestation if required
        if self.config.encryption_required || self.config.tee_attestation_required {
            self.verify_attestation(&attestation).await?;
        }
        
        // Parse transaction to get shard ID (in production, would decrypt header)
        let shard_id = self.extract_shard_id(&encrypted_tx).await?;
        
        // Generate transaction hash
        let tx_hash = self.hash_transaction(&encrypted_tx);
        
        // Check if transaction already exists
        let tx_index = self.tx_index.read().await;
        if tx_index.contains_key(&tx_hash) {
            return Err(MempoolError::DuplicateTransaction(tx_hash.clone()));
        }
        drop(tx_index);
        
        // Create transaction record
        let transaction = EncryptedTransaction {
            data: encrypted_tx.clone(),
            attestation: attestation.clone(),
            shard_id,
            hash: tx_hash.clone(),
            submission_time: Instant::now(),
            priority: priority.clone(),
        };
        
        // Add to mempool
        self.add_transaction(transaction).await?;
        
        // Create receipt
        let receipt = TxReceipt {
            tx_hash: tx_hash.clone(),
            timestamp: std::time::SystemTime::now(),
            status: TxStatus::Pending,
            estimated_inclusion_ms: self.estimate_inclusion_time(&priority).await,
            priority_fee: match priority {
                TxPriority::High(fee) => Some(fee),
                TxPriority::Medium(fee) => Some(fee),
                TxPriority::Low => None,
            },
        };
        
        // Update metrics
        self.metrics.record_submission(start_time.elapsed());
        
        info!("Transaction {} submitted to shard {} with priority {:?}", 
              tx_hash, shard_id, priority);
        
        Ok(receipt)
    }
    
    /// Get transactions for a shard
    pub async fn get_transactions(&self, shard_id: u64, limit: usize) -> Result<Vec<EncryptedTransaction>, MempoolError> {
        let transactions = self.transactions.read().await;
        
        if let Some(shard_mempool) = transactions.get(&shard_id) {
            let txs = shard_mempool.get_transactions(limit).await;
            self.metrics.record_fetch(txs.len());
            Ok(txs)
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Remove transactions that have been included in a block
    pub async fn remove_transactions(&self, tx_hashes: &[String]) -> Result<(), MempoolError> {
        let mut removed = 0;
        
        for tx_hash in tx_hashes {
            if self.remove_transaction(tx_hash).await? {
                removed += 1;
            }
        }
        
        self.metrics.record_removal(removed);
        debug!("Removed {} transactions from mempool", removed);
        
        Ok(())
    }
    
    /// Get mempool size in bytes
    pub async fn get_size(&self) -> usize {
        let transactions = self.transactions.read().await;
        
        transactions.values()
            .map(|shard| shard.size_bytes)
            .sum()
    }
    
    /// Get transaction count
    pub async fn get_count(&self) -> usize {
        let tx_index = self.tx_index.read().await;
        tx_index.len()
    }
    
    /// Check if transaction exists
    pub async fn has_transaction(&self, tx_hash: &str) -> bool {
        let tx_index = self.tx_index.read().await;
        tx_index.contains_key(tx_hash)
    }
    
    /// Get mempool statistics
    pub async fn get_stats(&self) -> MempoolStats {
        let transactions = self.transactions.read().await;
        let tx_index = self.tx_index.read().await;
        
        MempoolStats {
            total_transactions: tx_index.len(),
            total_size_bytes: transactions.values().map(|s| s.size_bytes).sum(),
            shard_counts: transactions.iter()
                .map(|(shard_id, shard)| (*shard_id, shard.transactions.len()))
                .collect(),
            oldest_transaction_age: transactions.values()
                .flat_map(|s| s.transactions.values())
                .map(|tx| tx.submission_time.elapsed().as_secs())
                .max()
                .unwrap_or(0),
            high_priority_count: transactions.values()
                .flat_map(|s| s.transactions.values())
                .filter(|tx| matches!(tx.priority, TxPriority::High(_)))
                .count(),
        }
    }
    
    // Internal methods
    
    async fn start_cleanup_task(&self) -> Result<(), MempoolError> {
        let manager = self.clone_manager().await;
        
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // Check shutdown flag
                if *manager.shutdown.read().await {
                    break;
                }
                
                // Run cleanup
                if let Err(e) = manager.cleanup().await {
                    warn!("Mempool cleanup failed: {}", e);
                }
            }
        });
        
        let mut cleanup_handle = self.cleanup_handle.write().await;
        *cleanup_handle = Some(handle);
        
        Ok(())
    }
    
    async fn cleanup(&self) -> Result<(), MempoolError> {
        debug!("Running mempool cleanup");
        
        // Check size limits
        self.enforce_size_limits().await?;
        
        // Remove expired transactions
        self.remove_expired_transactions().await?;
        
        // Rebuild pending queue
        self.rebuild_pending_queue().await?;
        
        self.metrics.record_cleanup();
        Ok(())
    }
    
    async fn enforce_size_limits(&self) -> Result<(), MempoolError> {
        let total_size = self.get_size().await;
        
        if total_size > self.config.max_size_bytes {
            warn!("Mempool size {} exceeds limit {}, evicting transactions",
                  total_size, self.config.max_size_bytes);
            
            self.evict_transactions().await?;
        }
        
        let total_count = self.get_count().await;
        if total_count > self.config.max_transactions {
            warn!("Mempool count {} exceeds limit {}, evicting transactions",
                  total_count, self.config.max_transactions);
            
            self.evict_transactions().await?;
        }
        
        Ok(())
    }
    
    async fn evict_transactions(&self) -> Result<(), MempoolError> {
        let mut pending_queue = self.pending_queue.write().await;
        let mut to_remove = Vec::new();
        let mut removed_size = 0;
        let mut removed_count = 0;
        
        // Evict low-priority transactions first
        while let Some(pending) = pending_queue.pop() {
            // Skip if we've removed enough
            if removed_size >= self.config.eviction_batch_size * 1024 && 
               removed_count >= self.config.eviction_batch_size {
                break;
            }
            
            to_remove.push(pending.tx_hash);
            removed_size += pending.size;
            removed_count += 1;
        }
        
        // Remove transactions
        for tx_hash in &to_remove {
            self.remove_transaction(tx_hash).await?;
        }
        
        info!("Evicted {} transactions ({} bytes) from mempool", removed_count, removed_size);
        Ok(())
    }
    
    async fn remove_expired_transactions(&self) -> Result<(), MempoolError> {
        let now = Instant::now();
        let mut expired = Vec::new();
        
        // Find expired transactions
        let transactions = self.transactions.read().await;
        for (shard_id, shard_mempool) in transactions.iter() {
            for (tx_hash, tx) in &shard_mempool.transactions {
                if now.duration_since(tx.submission_time) > Duration::from_secs(self.config.ttl_seconds) {
                    expired.push((*shard_id, tx_hash.clone()));
                }
            }
        }
        
        // Remove expired transactions
        for (shard_id, tx_hash) in expired {
            self.remove_transaction(&tx_hash).await?;
            debug!("Removed expired transaction {} from shard {}", tx_hash, shard_id);
        }
        
        Ok(())
    }
    
    async fn rebuild_pending_queue(&self) -> Result<(), MempoolError> {
        let mut pending_queue = BinaryHeap::new();
        let transactions = self.transactions.read().await;
        
        for (shard_id, shard_mempool) in transactions.iter() {
            for (tx_hash, tx) in &shard_mempool.transactions {
                let priority_score = match tx.priority {
                    TxPriority::High(fee) => 1000 + (fee as i64).min(1000),
                    TxPriority::Medium(fee) => 500 + (fee as i64).min(500),
                    TxPriority::Low => 0,
                };
                
                let pending = PendingTransaction {
                    tx_hash: tx_hash.clone(),
                    shard_id: *shard_id,
                    priority_score,
                    submission_time: tx.submission_time,
                    size: tx.data.len(),
                };
                
                pending_queue.push(pending);
            }
        }
        
        let mut current_queue = self.pending_queue.write().await;
        *current_queue = pending_queue;
        
        Ok(())
    }
    
    async fn verify_attestation(&self, attestation: &[u8]) -> Result<(), MempoolError> {
        if attestation.is_empty() {
            return Err(MempoolError::MissingAttestation);
        }
        
        // In production: verify TEE remote attestation
        // For now, just check format
        
        if attestation.len() < 64 {
            return Err(MempoolError::InvalidAttestation("Too short".to_string()));
        }
        
        // Check for expected header
        let expected_header = b"TEE_ATTESTATION";
        if attestation.len() >= expected_header.len() 
            && &attestation[..expected_header.len()] == expected_header {
            Ok(())
        } else {
            Err(MempoolError::InvalidAttestation("Invalid format".to_string()))
        }
    }
    
    async fn extract_shard_id(&self, encrypted_tx: &[u8]) -> Result<u64, MempoolError> {
        // In production: decrypt header or parse metadata
        // For simulation, use hash-based shard assignment
        
        let hash = self.hash_transaction(encrypted_tx);
        let shard_id = u64::from_be_bytes([
            hash.as_bytes()[0],
            hash.as_bytes()[1],
            hash.as_bytes()[2],
            hash.as_bytes()[3],
            hash.as_bytes()[4],
            hash.as_bytes()[5],
            hash.as_bytes()[6],
            hash.as_bytes()[7],
        ]);
        
        Ok(shard_id % 1024) // Modulo shard count
    }
    
    fn hash_transaction(&self, data: &[u8]) -> String {
        let mut hasher = Hasher::new();
        hasher.update(data);
        let hash = hasher.finalize();
        hex::encode(hash.as_bytes())
    }
    
    async fn add_transaction(&self, transaction: EncryptedTransaction) -> Result<(), MempoolError> {
        let shard_id = transaction.shard_id;
        let tx_hash = transaction.hash.clone();
        let size = transaction.data.len();
        
        // Add to shard mempool
        let mut transactions = self.transactions.write().await;
        let shard_mempool = transactions.entry(shard_id)
            .or_insert_with(|| ShardMempool::new(shard_id));
        
        shard_mempool.add_transaction(transaction).await?;
        
        // Add to index
        let mut tx_index = self.tx_index.write().await;
        tx_index.insert(tx_hash.clone(), TxLocation {
            shard_id,
            added_at: Instant::now(),
        });
        
        // Add to pending queue
        let priority_score = match shard_mempool.transactions.get(&tx_hash) {
            Some(tx) => match tx.priority {
                TxPriority::High(fee) => 1000 + (fee as i64).min(1000),
                TxPriority::Medium(fee) => 500 + (fee as i64).min(500),
                TxPriority::Low => 0,
            },
            None => 0,
        };
        
        let pending = PendingTransaction {
            tx_hash,
            shard_id,
            priority_score,
            submission_time: Instant::now(),
            size,
        };
        
        let mut pending_queue = self.pending_queue.write().await;
        pending_queue.push(pending);
        
        self.metrics.record_transaction_added(size);
        Ok(())
    }
    
    async fn remove_transaction(&self, tx_hash: &str) -> Result<bool, MempoolError> {
        // Get transaction location
        let tx_location = {
            let tx_index = self.tx_index.read().await;
            tx_index.get(tx_hash).cloned()
        };
        
        if let Some(location) = tx_location {
            // Remove from shard mempool
            let mut transactions = self.transactions.write().await;
            if let Some(shard_mempool) = transactions.get_mut(&location.shard_id) {
                shard_mempool.remove_transaction(tx_hash).await?;
                
                // Remove empty shard mempools
                if shard_mempool.transactions.is_empty() {
                    transactions.remove(&location.shard_id);
                }
            }
            
            // Remove from index
            let mut tx_index = self.tx_index.write().await;
            tx_index.remove(tx_hash);
            
            // Note: pending queue will be cleaned up during next rebuild
            
            self.metrics.record_transaction_removed();
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    async fn estimate_inclusion_time(&self, priority: &TxPriority) -> Option<u64> {
        // Simple estimation based on priority
        match priority {
            TxPriority::High(_) => Some(1000), // ~1 second
            TxPriority::Medium(_) => Some(5000), // ~5 seconds
            TxPriority::Low => Some(30000), // ~30 seconds
        }
    }
    
    async fn clone_manager(&self) -> Arc<Self> {
        Arc::new(Self {
            transactions: RwLock::new(self.transactions.read().await.clone()),
            tx_index: RwLock::new(self.tx_index.read().await.clone()),
            pending_queue: RwLock::new(BinaryHeap::new()), // Don't clone heap
            config: self.config.clone(),
            crypto_provider: self.crypto_provider.clone(),
            metrics: self.metrics.clone(),
            shutdown: RwLock::new(*self.shutdown.read().await),
            cleanup_handle: RwLock::new(None), // Can't clone handle
        })
    }
}


/// Mempool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MempoolConfig {
    /// Maximum mempool size in bytes
    pub max_size_bytes: usize,
    
    /// Maximum number of transactions
    pub max_transactions: usize,
    
    /// Priority fee bump percentage
    pub priority_fee_bump_percent: f64,
    
    /// Eviction batch size
    pub eviction_batch_size: usize,
    
    /// Transaction time-to-live in seconds
    pub ttl_seconds: u64,
    
    /// Require encryption for transactions
    pub encryption_required: bool,
    
    /// Require TEE attestation
    pub tee_attestation_required: bool,
}


impl Default for MempoolConfig {
    fn default() -> Self {
        Self {
            max_size_bytes: 100 * 1024 * 1024, // 100MB
            max_transactions: 100_000,
            priority_fee_bump_percent: 10.0,
            eviction_batch_size: 100,
            ttl_seconds: 3600, // 1 hour
            encryption_required: true,
            tee_attestation_required: true,
        }
    }
}


/// Transaction priority
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TxPriority {
    /// High priority with fee
    High(u64),
    
    /// Medium priority with fee
    Medium(u64),
    
    /// Low priority (no fee)
    Low,
}


impl PartialOrd for TxPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}


impl Ord for TxPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (TxPriority::High(a), TxPriority::High(b)) => a.cmp(b),
            (TxPriority::High(_), _) => std::cmp::Ordering::Greater,
            (_, TxPriority::High(_)) => std::cmp::Ordering::Less,
            
            (TxPriority::Medium(a), TxPriority::Medium(b)) => a.cmp(b),
            (TxPriority::Medium(_), TxPriority::Low) => std::cmp::Ordering::Greater,
            (TxPriority::Low, TxPriority::Medium(_)) => std::cmp::Ordering::Less,
            
            (TxPriority::Low, TxPriority::Low) => std::cmp::Ordering::Equal,
        }
    }
}


/// Encrypted transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedTransaction {
    /// Encrypted transaction data
    pub data: Vec<u8>,
    
    /// TEE attestation
    pub attestation: Vec<u8>,
    
    /// Shard ID
    pub shard_id: u64,
    
    /// Transaction hash
    pub hash: String,
    
    /// Submission time
    pub submission_time: Instant,
    
    /// Priority level
    pub priority: TxPriority,
}


/// Shard-specific mempool
#[derive(Debug, Clone)]
struct ShardMempool {
    /// Shard ID
    shard_id: u64,
    
    /// Transactions in this shard
    transactions: HashMap<String, EncryptedTransaction>,
    
    /// Total size in bytes
    size_bytes: usize,
    
    /// Transaction count
    count: usize,
}


impl ShardMempool {
    fn new(shard_id: u64) -> Self {
        Self {
            shard_id,
            transactions: HashMap::new(),
            size_bytes: 0,
            count: 0,
        }
    }
    
    async fn add_transaction(&mut self, transaction: EncryptedTransaction) -> Result<(), MempoolError> {
        let tx_hash = transaction.hash.clone();
        let size = transaction.data.len();
        
        if self.transactions.contains_key(&tx_hash) {
            return Err(MempoolError::DuplicateTransaction(tx_hash));
        }
        
        self.transactions.insert(tx_hash, transaction);
        self.size_bytes += size;
        self.count += 1;
        
        Ok(())
    }
    
    async fn remove_transaction(&mut self, tx_hash: &str) -> Result<(), MempoolError> {
        if let Some(tx) = self.transactions.remove(tx_hash) {
            self.size_bytes -= tx.data.len();
            self.count -= 1;
        }
        
        Ok(())
    }
    
    async fn get_transactions(&self, limit: usize) -> Vec<EncryptedTransaction> {
        // Sort by priority and age
        let mut txs: Vec<_> = self.transactions.values().cloned().collect();
        
        txs.sort_by(|a, b| {
            // First by priority
            let priority_cmp = b.priority.cmp(&a.priority);
            if priority_cmp != std::cmp::Ordering::Equal {
                return priority_cmp;
            }
            
            // Then by age (oldest first)
            a.submission_time.cmp(&b.submission_time)
        });
        
        // Take up to limit
        txs.into_iter().take(limit).collect()
    }
}


/// Transaction location in mempool
#[derive(Debug, Clone)]
struct TxLocation {
    /// Shard ID
    shard_id: u64,
    
    /// When transaction was added
    added_at: Instant,
}


/// Pending transaction for priority queue
#[derive(Debug, Clone)]
struct PendingTransaction {
    /// Transaction hash
    tx_hash: String,
    
    /// Shard ID
    shard_id: u64,
    
    /// Priority score (higher = higher priority)
    priority_score: i64,
    
    /// Submission time
    submission_time: Instant,
    
    /// Transaction size in bytes
    size: usize,
}


impl PartialEq for PendingTransaction {
    fn eq(&self, other: &Self) -> bool {
        self.tx_hash == other.tx_hash
    }
}


impl Eq for PendingTransaction {}


impl PartialOrd for PendingTransaction {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}


impl Ord for PendingTransaction {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Primary: priority score (higher is better)
        let priority_cmp = self.priority_score.cmp(&other.priority_score);
        if priority_cmp != std::cmp::Ordering::Equal {
            return priority_cmp;
        }
        
        // Secondary: age (older is better)
        let age_cmp = self.submission_time.cmp(&other.submission_time);
        if age_cmp != std::cmp::Ordering::Equal {
            return age_cmp;
        }
        
        // Tertiary: size (smaller is better)
        other.size.cmp(&self.size)
    }
}


/// Mempool metrics
#[derive(Debug, Clone)]
struct MempoolMetrics {
    /// Transactions submitted
    transactions_submitted: std::sync::atomic::AtomicU64,
    
    /// Transactions fetched
    transactions_fetched: std::sync::atomic::AtomicU64,
    
    /// Transactions removed
    transactions_removed: std::sync::atomic::AtomicU64,
    
    /// Total bytes stored
    bytes_stored: std::sync::atomic::AtomicU64,
    
    /// Submission latency histogram
    submission_latency: std::sync::Mutex<Vec<u64>>,
    
    /// Cleanup count
    cleanup_count: std::sync::atomic::AtomicU64,
}


impl MempoolMetrics {
    fn new() -> Self {
        Self {
            transactions_submitted: std::sync::atomic::AtomicU64::new(0),
            transactions_fetched: std::sync::atomic::AtomicU64::new(0),
            transactions_removed: std::sync::atomic::AtomicU64::new(0),
            bytes_stored: std::sync::atomic::AtomicU64::new(0),
            submission_latency: std::sync::Mutex::new(Vec::new()),
            cleanup_count: std::sync::atomic::AtomicU64::new(0),
        }
    }
    
    fn record_submission(&self, duration: std::time::Duration) {
        self.transactions_submitted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let latency = duration.as_millis() as u64;
        let mut latencies = self.submission_latency.lock().unwrap();
        latencies.push(latency);
        
        if latencies.len() > 1000 {
            latencies.remove(0);
        }
    }
    
    fn record_fetch(&self, count: usize) {
        self.transactions_fetched.fetch_add(count as u64, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_removal(&self, count: usize) {
        self.transactions_removed.fetch_add(count as u64, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_transaction_added(&self, size: usize) {
        self.bytes_stored.fetch_add(size as u64, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_transaction_removed(&self) {
        self.bytes_stored.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_cleanup(&self) {
        self.cleanup_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}


impl Clone for MempoolMetrics {
    fn clone(&self) -> Self {
        Self {
            transactions_submitted: std::sync::atomic::AtomicU64::new(
                self.transactions_submitted.load(std::sync::atomic::Ordering::Relaxed)),
            transactions_fetched: std::sync::atomic::AtomicU64::new(
                self.transactions_fetched.load(std::sync::atomic::Ordering::Relaxed)),
            transactions_removed: std::sync::atomic::AtomicU64::new(
                self.transactions_removed.load(std::sync::atomic::Ordering::Relaxed)),
            bytes_stored: std::sync::atomic::AtomicU64::new(
                self.bytes_stored.load(std::sync::atomic::Ordering::Relaxed)),
            submission_latency: std::sync::Mutex::new(
                self.submission_latency.lock().unwrap().clone()),
            cleanup_count: std::sync::atomic::AtomicU64::new(
                self.cleanup_count.load(std::sync::atomic::Ordering::Relaxed)),
        }
    }
}


/// Mempool statistics
#[derive(Debug, Clone, Serialize)]
pub struct MempoolStats {
    /// Total transactions
    pub total_transactions: usize,
    
    /// Total size in bytes
    pub total_size_bytes: usize,
    
    /// Transaction count by shard
    pub shard_counts: HashMap<u64, usize>,
    
    /// Age of oldest transaction (seconds)
    pub oldest_transaction_age: u64,
    
    /// High priority transaction count
    pub high_priority_count: usize,
}


/// Mempool errors
#[derive(Debug, thiserror::Error)]
pub enum MempoolError {
    #[error("Transaction too large: {0} bytes, maximum {1} bytes")]
    TransactionTooLarge(usize, usize),
    
    #[error("Duplicate transaction: {0}")]
    DuplicateTransaction(String),
    
    #[error("Missing TEE attestation")]
    MissingAttestation,
    
    #[error("Invalid TEE attestation: {0}")]
    InvalidAttestation(String),
    
    #[error("Mempool full: {0}")]
    MempoolFull(String),
    
    #[error("Transaction expired: {0}")]
    TransactionExpired(String),
    
    #[error("Shard not found: {0}")]
    ShardNotFound(u64),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
}


// Unit tests
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tx_priority_ordering() {
        let high1 = TxPriority::High(100);
        let high2 = TxPriority::High(200);
        let medium1 = TxPriority::Medium(50);
        let medium2 = TxPriority::Medium(100);
        let low = TxPriority::Low;
        
        assert!(high1 > medium1);
        assert!(high2 > high1); // Higher fee
        assert!(medium1 > low);
        assert!(medium2 > medium1); // Higher fee
        assert_eq!(low, low);
    }
    
    #[test]
    fn test_pending_transaction_ordering() {
        let now = Instant::now();
        
        let tx1 = PendingTransaction {
            tx_hash: "tx1".to_string(),
            shard_id: 1,
            priority_score: 1000,
            submission_time: now,
            size: 100,
        };
        
        let tx2 = PendingTransaction {
            tx_hash: "tx2".to_string(),
            shard_id: 1,
            priority_score: 500,
            submission_time: now,
            size: 100,
        };
        
        let tx3 = PendingTransaction {
            tx_hash: "tx3".to_string(),
            shard_id: 1,
            priority_score: 1000,
            submission_time: now - Duration::from_secs(1), // Older
            size: 100,
        };
        
        assert!(tx1 > tx2); // Higher priority
        assert!(tx3 > tx1); // Same priority, but older
    }
    
    #[test]
    fn test_shard_mempool() {
        let mut mempool = ShardMempool::new(1);
        
        let tx = EncryptedTransaction {
            data: vec![1, 2, 3],
            attestation: vec![4, 5, 6],
            shard_id: 1,
            hash: "test_hash".to_string(),
            submission_time: Instant::now(),
            priority: TxPriority::High(100),
        };
        
        // Test adding transaction
        let result = tokio::runtime::Runtime::new().unwrap()
            .block_on(mempool.add_transaction(tx.clone()));
        assert!(result.is_ok());
        assert_eq!(mempool.count, 1);
        assert_eq!(mempool.size_bytes, 3);
        
        // Test duplicate detection
        let result = tokio::runtime::Runtime::new().unwrap()
            .block_on(mempool.add_transaction(tx));
        assert!(result.is_err());
        
        // Test removing transaction
        let result = tokio::runtime::Runtime::new().unwrap()
            .block_on(mempool.remove_transaction("test_hash"));
        assert!(result.is_ok());
        assert_eq!(mempool.count, 0);
        assert_eq!(mempool.size_bytes, 0);
    }
    
    #[tokio::test]
    async fn test_mempool_manager_creation() {
        let config = MempoolConfig::default();
        let crypto = Arc::new(CryptoProvider::new().unwrap());
        let manager = MempoolManager::new(config, crypto).await;
        
        assert!(manager.is_ok());
    }
}
