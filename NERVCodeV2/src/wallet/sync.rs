// src/sync.rs
// Epic 4: Light-Client Synchronization and Delta Fetching
// Complete implementation with efficient sync and minimal data footprint

use crate::balance::BalanceTracker;
use crate::keys::Account;
use crate::types::{Output, EmbeddingRoot, SyncState};
use nerv::embedding::circuit::recursive::RecursiveVerifier;
use nerv::privacy::tee::TEEAttestation;
use nerv::network::{LightNodeRPC, NetworkConfig};
use serde::{Serialize, Deserialize};
use thiserror::Error;
use tokio::sync::{RwLock, mpsc};
use std::sync::Arc;
use std::time::Duration;
use futures::{StreamExt, stream::BoxStream};

// Superior UI Flow Description:
// 1. Sync screen: Beautiful gradient progress ring with percentage
// 2. Real-time updates: Smooth animation for each synced block
// 3. Network status: Visual indicator of connection quality
// 4. Offline mode: Elegant offline state with retry button
// 5. Progress details: Expandable view with sync statistics
// 6. Background sync: Subtle notification when sync completes

#[derive(Error, Debug)]
pub enum SyncError {
    #[error("Network connection failed")]
    NetworkError,
    #[error("Proof verification failed")]
    ProofFailed,
    #[error("RPC call failed")]
    RpcFailed,
    #[error("Sync timeout")]
    Timeout,
    #[error("Invalid chain state")]
    InvalidState,
    #[error("Storage error")]
    StorageError,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SyncProgress {
    pub current_height: u64,
    pub target_height: u64,
    pub percent_complete: f64,
    pub downloaded_bytes: u64,
    pub remaining_bytes: u64,
    pub estimated_time_remaining: Duration,
    pub speed_bytes_per_sec: f64,
    pub active_connections: u32,
    pub state: SyncState,
}

pub struct LightClient {
    verifier: Arc<RecursiveVerifier>,
    rpc_client: Arc<LightNodeRPC>,
    state: Arc<RwLock<SyncState>>,
    progress: Arc<RwLock<SyncProgress>>,
    is_syncing: Arc<RwLock<bool>>,
    network_config: NetworkConfig,
}

impl LightClient {
    /// Create new light client with beautiful initialization
    /// UI: Client initialization with loading animation
    pub fn new(network_config: NetworkConfig) -> Result<Self, SyncError> {
        let verifier = RecursiveVerifier::new()
            .map_err(|_| SyncError::ProofFailed)?;
        
        let rpc_client = LightNodeRPC::new(&network_config)
            .map_err(|_| SyncError::NetworkError)?;
        
        let state = SyncState {
            current_height: 0,
            verified_height: 0,
            embedding_roots: Vec::new(),
            last_sync: chrono::Utc::now(),
            sync_mode: SyncMode::Initial,
        };
        
        let progress = SyncProgress {
            current_height: 0,
            target_height: 0,
            percent_complete: 0.0,
            downloaded_bytes: 0,
            remaining_bytes: 0,
            estimated_time_remaining: Duration::from_secs(0),
            speed_bytes_per_sec: 0.0,
            active_connections: 0,
            state: state.clone(),
        };
        
        Ok(Self {
            verifier: Arc::new(verifier),
            rpc_client: Arc::new(rpc_client),
            state: Arc::new(RwLock::new(state)),
            progress: Arc::new(RwLock::new(progress)),
            is_syncing: Arc::new(RwLock::new(false)),
            network_config,
        })
    }
    
    /// Perform full initial sync with beautiful progress UI
    /// UI: Full-screen sync with detailed progress animation
    pub async fn synchronize(
        &self,
        accounts: &[Account],
        balance_tracker: Arc<BalanceTracker>,
    ) -> Result<SyncProgress, SyncError> {
        // Check if already syncing
        {
            let mut is_syncing = self.is_syncing.write().await;
            if *is_syncing {
                return Ok(self.progress.read().await.clone());
            }
            *is_syncing = true;
        }
        
        // Update progress
        {
            let mut progress = self.progress.write().await;
            progress.state.sync_mode = SyncMode::Initial;
            progress.percent_complete = 0.0;
        }
        
        // Get current chain state from network
        let chain_state = self.fetch_chain_state().await?;
        
        // Update target height
        {
            let mut progress = self.progress.write().await;
            progress.target_height = chain_state.height;
        }
        
        // Verify recursive proof of entire chain
        self.verify_chain_proof(&chain_state).await?;
        
        // Sync recent outputs (last 10,000 blocks or from last sync)
        let start_height = {
            let state = self.state.read().await;
            state.verified_height.saturating_sub(10000)
        };
        
        // Sync in parallel batches for performance
        let batch_size = 100;
        let total_batches = (chain_state.height - start_height) / batch_size + 1;
        
        for batch_num in 0..total_batches {
            let batch_start = start_height + batch_num * batch_size;
            let batch_end = std::cmp::min(batch_start + batch_size, chain_state.height);
            
            // Update progress
            {
                let mut progress = self.progress.write().await;
                progress.current_height = batch_end;
                progress.percent_complete = batch_num as f64 / total_batches as f64 * 100.0;
                
                // Estimate remaining time
                if batch_num > 0 {
                    let elapsed = progress.estimated_time_remaining;
                    let remaining_batches = total_batches - batch_num;
                    progress.estimated_time_remaining = Duration::from_secs_f64(
                        elapsed.as_secs_f64() * remaining_batches as f64 / batch_num as f64
                    );
                }
            }
            
            // Fetch and process batch
            self.sync_batch(batch_start, batch_end, accounts, &balance_tracker).await?;
            
            // Update state
            {
                let mut state = self.state.write().await;
                state.current_height = batch_end;
                state.verified_height = batch_end;
                state.last_sync = chrono::Utc::now();
            }
        }
        
        // Finalize sync
        {
            let mut progress = self.progress.write().await;
            progress.current_height = chain_state.height;
            progress.percent_complete = 100.0;
            progress.estimated_time_remaining = Duration::from_secs(0);
            
            let mut state = self.state.write().await;
            state.sync_mode = SyncMode::Complete;
        }
        
        // Reset syncing flag
        *self.is_syncing.write().await = false;
        
        Ok(self.progress.read().await.clone())
    }
    
    /// Sync a batch of blocks
    async fn sync_batch(
        &self,
        start_height: u64,
        end_height: u64,
        accounts: &[Account],
        balance_tracker: &Arc<BalanceTracker>,
    ) -> Result<(), SyncError> {
        // Fetch batch data from network
        let batch_data = self.fetch_batch(start_height, end_height).await?;
        
        // Verify batch proof
        self.verify_batch_proof(&batch_data).await?;
        
        // Extract outputs from batch
        let outputs = self.extract_outputs(&batch_data, accounts).await?;
        
        // Update downloaded bytes in progress
        {
            let mut progress = self.progress.write().await;
            progress.downloaded_bytes += batch_data.size_bytes;
        }
        
        // Scan outputs for our notes
        if !outputs.is_empty() {
            balance_tracker.scan_outputs(&outputs, &self.get_account_keys(accounts), end_height).await
                .map_err(|_| SyncError::InvalidState)?;
        }
        
        // Store embedding roots
        self.store_embedding_roots(&batch_data.embedding_roots).await?;
        
        Ok(())
    }
    
    /// Fetch current chain state from network
    async fn fetch_chain_state(&self) -> Result<ChainState, SyncError> {
        let (height, final_proof, final_hash) = self.rpc_client.get_chain_head()
            .await
            .map_err(|_| SyncError::RpcFailed)?;
        
        Ok(ChainState {
            height,
            final_proof,
            final_hash,
            timestamp: chrono::Utc::now(),
        })
    }
    
    /// Verify recursive proof of chain state
    async fn verify_chain_proof(&self, chain_state: &ChainState) -> Result<(), SyncError> {
        let verifier = self.verifier.clone();
        
        tokio::task::spawn_blocking(move || {
            verifier.verify_final(&chain_state.final_proof, chain_state.final_hash)
        })
        .await
        .map_err(|_| SyncError::ProofFailed)?
        .map_err(|_| SyncError::ProofFailed)
    }
    
    /// Fetch batch of blocks from network
    async fn fetch_batch(&self, start: u64, end: u64) -> Result<BatchData, SyncError> {
        self.rpc_client.get_batch(start, end)
            .await
            .map_err(|_| SyncError::RpcFailed)
    }
    
    /// Verify batch proof
    async fn verify_batch_proof(&self, batch_data: &BatchData) -> Result<(), SyncError> {
        let verifier = self.verifier.clone();
        let proof = batch_data.proof.clone();
        let roots = batch_data.embedding_roots.clone();
        
        tokio::task::spawn_blocking(move || {
            verifier.verify_batch(&proof, &roots)
        })
        .await
        .map_err(|_| SyncError::ProofFailed)?
        .map_err(|_| SyncError::ProofFailed)
    }
    
    /// Extract outputs relevant to our accounts using bloom filter
    async fn extract_outputs(
        &self,
        batch_data: &BatchData,
        accounts: &[Account],
    ) -> Result<Vec<Output>, SyncError> {
        // Build bloom filter of our public keys
        let bloom_filter = self.build_bloom_filter(accounts);
        
        // Filter outputs server-side using bloom filter
        self.rpc_client.filter_outputs(&bloom_filter, batch_data.batch_id)
            .await
            .map_err(|_| SyncError::RpcFailed)
    }
    
    /// Build bloom filter for our account keys
    fn build_bloom_filter(&self, accounts: &[Account]) -> BloomFilter {
        let mut filter = BloomFilter::new(10000, 0.01); // 1% false positive rate
        
        for account in accounts {
            for key in &account.keys {
                filter.insert(&key.enc_pk);
                filter.insert(&key.commitment);
            }
        }
        
        filter
    }
    
    /// Get all account keys as references
    fn get_account_keys(&self, accounts: &[Account]) -> Vec<&crate::keys::AccountKeys> {
        accounts.iter()
            .flat_map(|account| account.keys.iter())
            .collect()
    }
    
    /// Store embedding roots for offline verification
    async fn store_embedding_roots(&self, roots: &[EmbeddingRoot]) -> Result<(), SyncError> {
        let mut state = self.state.write().await;
        state.embedding_roots.extend_from_slice(roots);
        
        // Keep only recent roots (last 1000)
        if state.embedding_roots.len() > 1000 {
            state.embedding_roots = state.embedding_roots[state.embedding_roots.len() - 1000..].to_vec();
        }
        
        Ok(())
    }
    
    /// Get current sync progress
    /// UI: Progress ring with animated gradient
    pub async fn get_progress(&self) -> SyncProgress {
        self.progress.read().await.clone()
    }
    
    /// Check if fully synced
    /// UI: Green checkmark with subtle pulse animation
    pub async fn is_synced(&self) -> bool {
        let progress = self.progress.read().await;
        progress.current_height >= progress.target_height && progress.percent_complete >= 99.9
    }
    
    /// Get sync state
    pub async fn get_state(&self) -> SyncState {
        self.state.read().await.clone()
    }
    
    /// Start background sync
    /// UI: Subtle notification when background sync starts
    pub async fn start_background_sync(
        &self,
        accounts: &[Account],
        balance_tracker: Arc<BalanceTracker>,
    ) -> mpsc::Receiver<SyncProgress> {
        let (tx, rx) = mpsc::channel(100);
        let self_clone = self.clone();
        let accounts_clone = accounts.to_vec();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                if !self_clone.is_syncing.read().await {
                    if let Ok(progress) = self_clone.synchronize(&accounts_clone, balance_tracker.clone()).await {
                        let _ = tx.send(progress).await;
                    }
                }
            }
        });
        
        rx
    }
    
    /// Perform quick sync (last 100 blocks)
    /// UI: Quick sync with shimmer animation
    pub async fn quick_sync(
        &self,
        accounts: &[Account],
        balance_tracker: Arc<BalanceTracker>,
    ) -> Result<SyncProgress, SyncError> {
        let state = self.state.read().await;
        let start_height = state.current_height.saturating_sub(100);
        
        drop(state);
        self.sync_batch(start_height, start_height + 100, accounts, &balance_tracker).await?;
        
        Ok(self.progress.read().await.clone())
    }
    
    /// Export sync state for backup
    pub async fn export_state(&self) -> Result<Vec<u8>, SyncError> {
        let state = self.state.read().await.clone();
        bincode::serialize(&state)
            .map_err(|_| SyncError::StorageError)
    }
    
    /// Import sync state from backup
    pub async fn import_state(&self, data: &[u8]) -> Result<(), SyncError> {
        let state: SyncState = bincode::deserialize(data)
            .map_err(|_| SyncError::InvalidState)?;
        
        *self.state.write().await = state;
        Ok(())
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ChainState {
    pub height: u64,
    pub final_proof: Vec<u8>,
    pub final_hash: [u8; 32],
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BatchData {
    pub batch_id: [u8; 32],
    pub start_height: u64,
    pub end_height: u64,
    pub embedding_roots: Vec<EmbeddingRoot>,
    pub proof: Vec<u8>,
    pub size_bytes: u64,
}

#[derive(Clone)]
pub struct BloomFilter {
    bits: Vec<u8>,
    hash_count: usize,
}

impl BloomFilter {
    fn new(capacity: usize, error_rate: f64) -> Self {
        let bits_count = (-(capacity as f64 * error_rate.ln()) / (2.0f64.ln().powi(2))).ceil() as usize;
        let hash_count = ((bits_count as f64 / capacity as f64) * 2.0f64.ln()).ceil() as usize;
        
        Self {
            bits: vec![0; (bits_count + 7) / 8],
            hash_count,
        }
    }
    
    fn insert(&mut self, item: &[u8]) {
        for i in 0..self.hash_count {
            let hash = self.hash(item, i);
            let bit_index = hash % (self.bits.len() * 8);
            self.bits[bit_index / 8] |= 1 << (bit_index % 8);
        }
    }
    
    fn hash(&self, item: &[u8], seed: usize) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        seed.hash(&mut hasher);
        hasher.finish() as usize
    }
}
