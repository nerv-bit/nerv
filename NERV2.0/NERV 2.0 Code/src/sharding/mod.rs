//! Dynamic Neural Sharding (Lightweight).
//!
//! A secondary NWO Perceptron predicts shard load based on TPS, mempool
//! size, inter-shard queue depth, and gas utilization. When predicted
//! load exceeds 90%, deterministic embedding bisection splits the shard.
//! Merges occur automatically below threshold.
//!
//! Submodules:
//! - `nwo_predictor` — NWO Perceptron for load prediction
//! - `bisection` — Embedding bisection algorithm
//! - `erasure` — Reed-Solomon erasure coding (k=5, m=2)
//!
//! Dynamic Neural Sharding — Shards that live, breathe, split, and merge like cells.
//!
//! NERV v2.0 uses a dedicated NWO Perceptron to predict shard load and
//! trigger deterministic embedding bisections (splits) or merges. Combined
//! with Reed-Solomon erasure coding (k=5, m=2), this provides:
//!
//! - **Infinite horizontal scalability**: No theoretical ceiling on TPS
//! - **Sub-4-second splits**: Deterministic bisection of 512-byte embeddings
//! - **Automatic merges**: Idle shards consolidate without governance
//! - **Fault tolerance**: 2 of 7 replicas can fail with zero downtime
//!
//! # V2.0 Changes vs V1.01
//!
//! | Component | V1.01 | V2.0 |
//! |-----------|-------|------|
//! | Load predictor | 1.1 MB LSTM (weekly FL) | NWO Perceptron (per-block Adam) |
//! | Split trigger | Overload prob > 0.92 | Overload prob > 0.90 (configurable) |
//! | Embedding dim | ℝ⁵¹² (64-dim × 8 bytes) | ℝ⁶⁴ (512 bytes total) |
//! | Re-execution | Last 500 txs in TEEs | Last 500 txs (pure crypto) |
//! | Merge threshold | <10 TPS for 10 min | Configurable |
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐
//! │  Load Metrics   │ ──→ NWO Predictor ──→ overload probability
//! └─────────────────┘         │
//!                             ▼
//!                    ┌─────────────────┐
//!                    │  Split / Merge  │
//!                    │  Decision       │
//!                    └────────┬────────┘
//!                             │
//!              ┌──────────────┼──────────────┐
//!              ▼                             ▼
//!     ┌─────────────────┐          ┌─────────────────┐
//!     │  Bisection       │          │  Reverse         │
//!     │  (split)         │          │  Bisection (merge)│
//!     └────────┬────────┘          └────────┬────────┘
//!              │                             │
//!              ▼                             ▼
//!     ┌─────────────────┐          ┌─────────────────┐
//!     │  Erasure Encode  │          │  Erasure Decode  │
//!     │  (k=5, m=2)      │          │  (k=5, m=2)      │
//!     └─────────────────┘          └─────────────────┘
//! ```

pub mod nwo_predictor;
pub mod bisection;
pub mod erasure;

// ─── Re-exports ─────────────────────────────────────────────────────────

pub use nwo_predictor::{NwoPredictor, LoadFeatures, LoadMetrics, OverloadPrediction};
pub use bisection::{
    BisectionResult, MergeResult, SplitProposal, MergeProposal,
    SplitStateMachine, MergeStateMachine, SplitState, MergeState,
};
pub use erasure::{ErasureEncoder, ErasureDecoder, ShardChunk, ErasureConfig};

use crate::{
    EMBEDDING_DIM, EMBEDDING_BYTES, ERASURE_K, ERASURE_M,
    ShardId, BlockHeight, EmbeddingRoot, ValidatorId, StakeAmount,
    VotingWeight, ReputationScore, Epoch,
    NervError, NervResult,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// ─── Shard Lifecycle State ───────────────────────────────────────────────

/// The lifecycle state of a shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShardLifecycle {
    /// Shard is actively processing transactions.
    Active,
    /// Shard is in the process of splitting into two children.
    Splitting,
    /// Shard is in the process of merging with its sibling.
    Merging,
    /// Shard has been split and is no longer active (replaced by children).
    Split,
    /// Shard has been merged and is no longer active (absorbed into parent).
    Merged,
    /// Shard is temporarily offline (waiting for data recovery).
    Offline,
}

impl std::fmt::Display for ShardLifecycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Splitting => write!(f, "splitting"),
            Self::Merging => write!(f, "merging"),
            Self::Split => write!(f, "split"),
            Self::Merged => write!(f, "merged"),
            Self::Offline => write!(f, "offline"),
        }
    }
}

// ─── Shard Metadata ──────────────────────────────────────────────────────

/// Persistent metadata about a shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMetadata {
    /// Unique shard identifier.
    pub shard_id: ShardId,

    /// Current lifecycle state.
    pub lifecycle: ShardLifecycle,

    /// Block height at which this shard was created.
    pub created_at_height: BlockHeight,

    /// Block height of the most recent state update.
    pub last_updated_height: BlockHeight,

    /// Current canonical embedding root (32-byte BLAKE3 hash).
    pub embedding_root: EmbeddingRoot,

    /// Parent shard ID (None for genesis shard).
    pub parent_shard_id: Option<ShardId>,

    /// Left child shard ID (set after split).
    pub left_child: Option<ShardId>,

    /// Right child shard ID (set after split).
    pub right_child: Option<ShardId>,

    /// Sibling shard ID (for merge detection).
    pub sibling_shard_id: Option<ShardId>,

    /// Split seed used for bisection (if this shard was created by a split).
    pub split_seed: Option<[u8; 32]>,

    /// Current TPS (transactions per second, exponentially weighted moving average).
    pub current_tps: f64,

    /// Number of active validators assigned to this shard.
    pub validator_count: usize,

    /// Number of accounts (estimated, for load balancing).
    pub estimated_accounts: u64,

    /// Cross-shard transaction ratio (0.0 = all intra-shard, 1.0 = all cross-shard).
    pub cross_shard_ratio: f64,

    /// Erasure-coded replica locations (node IDs storing this shard's data).
    pub replica_nodes: Vec<ValidatorId>,

    /// Timestamp of the last split/merge event (Unix epoch millis).
    pub last_reconfiguration_ms: u64,

    /// Number of splits this shard has undergone (depth in the shard tree).
    pub split_depth: u32,
}

impl ShardMetadata {
    /// Create metadata for the genesis shard.
    pub fn genesis() -> Self {
        Self {
            shard_id: ShardId::GENESIS,
            lifecycle: ShardLifecycle::Active,
            created_at_height: BlockHeight::GENESIS,
            last_updated_height: BlockHeight::GENESIS,
            embedding_root: EmbeddingRoot::GENESIS,
            parent_shard_id: None,
            left_child: None,
            right_child: None,
            sibling_shard_id: None,
            split_seed: None,
            current_tps: 0.0,
            validator_count: 0,
            estimated_accounts: 0,
            cross_shard_ratio: 0.0,
            replica_nodes: Vec::new(),
            last_reconfiguration_ms: 0,
            split_depth: 0,
        }
    }

    /// Create a new shard with the given ID and parent.
    pub fn new(
        shard_id: ShardId,
        parent_shard_id: Option<ShardId>,
        created_at_height: BlockHeight,
        embedding_root: EmbeddingRoot,
        split_depth: u32,
    ) -> Self {
        Self {
            shard_id,
            lifecycle: ShardLifecycle::Active,
            created_at_height,
            last_updated_height: created_at_height,
            embedding_root,
            parent_shard_id,
            left_child: None,
            right_child: None,
            sibling_shard_id: None,
            split_seed: None,
            current_tps: 0.0,
            validator_count: 0,
            estimated_accounts: 0,
            cross_shard_ratio: 0.0,
            replica_nodes: Vec::new(),
            last_reconfiguration_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            split_depth,
        }
    }

    /// Check if this shard can be split (must be Active).
    pub fn can_split(&self) -> bool {
        self.lifecycle == ShardLifecycle::Active
    }

    /// Check if this shard can be merged (must be Active with a sibling).
    pub fn can_merge(&self) -> bool {
        self.lifecycle == ShardLifecycle::Active && self.sibling_shard_id.is_some()
    }

    /// Update TPS with exponential moving average.
    pub fn update_tps_ema(&mut self, new_tps: f64, alpha: f64) {
        self.current_tps = alpha * new_tps + (1.0 - alpha) * self.current_tps;
    }
}

// ─── Shard Runtime Metrics ───────────────────────────────────────────────

/// Real-time metrics collected for a shard (used by the NWO predictor).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShardMetrics {
    /// Current TPS.
    pub tps: f64,

    /// Mempool size (number of pending encrypted transactions).
    pub mempool_size: usize,

    /// Inter-shard message queue depth.
    pub inter_shard_queue_depth: usize,

    /// Gas utilization fraction (0.0–1.0).
    pub gas_utilization: f64,

    /// Average block execution time in milliseconds.
    pub avg_block_exec_ms: f64,

    /// P95 finality latency in milliseconds.
    pub p95_finality_ms: f64,

    /// Cross-shard transaction ratio (0.0–1.0).
    pub cross_shard_ratio: f64,
}

// ─── Shard Event ─────────────────────────────────────────────────────────

/// Events emitted by the sharding subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardEvent {
    /// A shard split has been proposed.
    SplitProposed {
        /// The shard being split.
        shard_id: ShardId,
        /// The proposing validator.
        proposer: ValidatorId,
        /// The block height at proposal.
        height: BlockHeight,
    },

    /// A shard split has been approved and will execute.
    SplitApproved {
        /// The parent shard.
        parent_shard_id: ShardId,
        /// The left child shard.
        left_child: ShardId,
        /// The right child shard.
        right_child: ShardId,
        /// The block height of approval.
        height: BlockHeight,
    },

    /// A shard split has completed successfully.
    SplitCompleted {
        /// The parent shard (now marked as Split).
        parent_shard_id: ShardId,
        /// The left child (now Active).
        left_child: ShardId,
        /// The right child (now Active).
        right_child: ShardId,
        /// Duration of the split in milliseconds.
        duration_ms: u64,
    },

    /// A shard split has failed.
    SplitFailed {
        /// The shard that failed to split.
        shard_id: ShardId,
        /// Reason for failure.
        reason: String,
    },

    /// A shard merge has been proposed.
    MergeProposed {
        /// Left sibling.
        left_shard_id: ShardId,
        /// Right sibling.
        right_shard_id: ShardId,
    },

    /// A shard merge has completed.
    MergeCompleted {
        /// The new parent shard.
        parent_shard_id: ShardId,
        /// The former left child (now Merged).
        left_child: ShardId,
        /// The former right child (now Merged).
        right_child: ShardId,
        /// Duration of the merge in milliseconds.
        duration_ms: u64,
    },

    /// A shard has gone offline.
    ShardOffline {
        shard_id: ShardId,
        reason: String,
    },

    /// A shard has recovered from offline state.
    ShardRecovered {
        shard_id: ShardId,
    },
}

// ─── Shard Registry ──────────────────────────────────────────────────────

/// The global registry of all shards in the network.
///
/// Tracks active, splitting, merging, and historical shards.
/// This is the authoritative source of shard topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardRegistry {
    /// All known shards by ID.
    shards: HashMap<u64, ShardMetadata>,

    /// Currently active shard IDs (for fast lookup).
    active_shards: Vec<ShardId>,

    /// Monotonically increasing counter for generating new shard IDs.
    next_shard_id: u64,

    /// Configuration parameters.
    config: ShardRegistryConfig,
}

/// Configuration for the shard registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardRegistryConfig {
    /// Minimum number of active shards.
    pub min_shards: usize,

    /// Maximum number of active shards (0 = unlimited).
    pub max_shards: usize,

    /// Split threshold (overload probability above this triggers split).
    pub split_threshold: f64,

    /// Merge threshold (load below this for sustained period triggers merge).
    pub merge_threshold: f64,

    /// Minimum TPS below which merge is considered.
    pub merge_tps_threshold: f64,

    /// Duration (in seconds) of low TPS before merge triggers.
    pub merge_idle_duration_secs: u64,

    /// Number of recent transactions to re-execute after split/merge.
    pub reexecution_window: usize,

    /// Reed-Solomon k parameter.
    pub erasure_k: usize,

    /// Reed-Solomon m parameter.
    pub erasure_m: usize,
}

impl Default for ShardRegistryConfig {
    fn default() -> Self {
        Self {
            min_shards: 1,
            max_shards: 0,
            split_threshold: 0.90,
            merge_threshold: 0.20,
            merge_tps_threshold: 10.0,
            merge_idle_duration_secs: 600, // 10 minutes
            reexecution_window: 500,
            erasure_k: ERASURE_K,
            erasure_m: ERASURE_M,
        }
    }
}

impl ShardRegistry {
    /// Create a new shard registry with the genesis shard.
    pub fn new(config: ShardRegistryConfig) -> Self {
        let genesis = ShardMetadata::genesis();
        let mut shards = HashMap::new();
        shards.insert(genesis.shard_id.0, genesis);

        Self {
            shards,
            active_shards: vec![ShardId::GENESIS],
            next_shard_id: 1,
            config,
        }
    }

    /// Get a shard by ID.
    pub fn get(&self, shard_id: &ShardId) -> Option<&ShardMetadata> {
        self.shards.get(&shard_id.0)
    }

    /// Get a mutable reference to a shard by ID.
    pub fn get_mut(&mut self, shard_id: &ShardId) -> Option<&mut ShardMetadata> {
        self.shards.get_mut(&shard_id.0)
    }

    /// List all active shard IDs.
    pub fn active_shards(&self) -> &[ShardId] {
        &self.active_shards
    }

    /// Get the number of active shards.
    pub fn active_count(&self) -> usize {
        self.active_shards.len()
    }

    /// Check if a split is allowed (below max_shards limit).
    pub fn can_split(&self) -> bool {
        if self.config.max_shards > 0 && self.active_count() >= self.config.max_shards {
            return false;
        }
        true
    }

    /// Generate a new unique shard ID.
    pub fn allocate_shard_id(&mut self) -> ShardId {
        let id = ShardId::new(self.next_shard_id);
        self.next_shard_id += 1;
        id
    }

    /// Register a new shard (e.g., after a split creates children).
    pub fn register_shard(&mut self, metadata: ShardMetadata) -> NervResult<()> {
        if self.shards.contains_key(&metadata.shard_id.0) {
            return Err(NervError::Sharding(format!(
                "shard {} already exists", metadata.shard_id
            )));
        }
        self.shards.insert(metadata.shard_id.0, metadata.clone());
        if metadata.lifecycle == ShardLifecycle::Active {
            self.active_shards.push(metadata.shard_id);
        }
        Ok(())
    }

    /// Mark a shard as split (replaced by its two children).
    pub fn mark_split(
        &mut self,
        parent_id: ShardId,
        left_child_id: ShardId,
        right_child_id: ShardId,
    ) -> NervResult<()> {
        let parent = self.shards.get_mut(&parent_id.0)
            .ok_or_else(|| NervError::Sharding(format!("shard {} not found", parent_id)))?;

        parent.lifecycle = ShardLifecycle::Split;
        parent.left_child = Some(left_child_id);
        parent.right_child = Some(right_child_id);

        // Remove from active list
        self.active_shards.retain(|id| id != &parent_id);

        Ok(())
    }

    /// Mark two sibling shards as merged (replaced by parent).
    pub fn mark_merged(
        &mut self,
        left_id: ShardId,
        right_id: ShardId,
        parent_id: ShardId,
    ) -> NervResult<()> {
        for id in [&left_id, &right_id] {
            if let Some(shard) = self.shards.get_mut(&id.0) {
                shard.lifecycle = ShardLifecycle::Merged;
            }
            self.active_shards.retain(|active| active != id);
        }

        // Activate the parent
        if let Some(parent) = self.shards.get_mut(&parent_id.0) {
            parent.lifecycle = ShardLifecycle::Active;
            self.active_shards.push(parent_id);
        }

        Ok(())
    }

    /// Find the shard responsible for a given account key.
    ///
    /// Uses a deterministic hash-based routing: the account is assigned
    /// to the shard whose ID is closest to hash(account_key) in the
    /// shard ID space.
    pub fn find_shard_for_account(&self, account_key: &[u8]) -> Option<ShardId> {
        if self.active_shards.is_empty() {
            return None;
        }

        let hash = blake3::hash(account_key);
        let hash_u64 = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap_or([0u8; 8]));

        // Find the active shard with the closest ID
        let mut best = self.active_shards[0];
        let mut best_dist = u64::MAX;

        for &shard_id in &self.active_shards {
            let dist = if hash_u64 > shard_id.0 {
                hash_u64 - shard_id.0
            } else {
                shard_id.0 - hash_u64
            };
            if dist < best_dist {
                best_dist = dist;
                best = shard_id;
            }
        }

        Some(best)
    }

    /// Get all shards with a given lifecycle.
    pub fn shards_by_lifecycle(&self, lifecycle: ShardLifecycle) -> Vec<&ShardMetadata> {
        self.shards.values()
            .filter(|s| s.lifecycle == lifecycle)
            .collect()
    }

    /// Get the total TPS across all active shards.
    pub fn total_tps(&self) -> f64 {
        self.active_shards.iter()
            .filter_map(|id| self.shards.get(&id.0))
            .map(|s| s.current_tps)
            .sum()
    }

    /// Compute the total voting weight of all validators across all active shards.
    pub fn total_validator_weight(&self) -> usize {
        self.active_shards.iter()
            .filter_map(|id| self.shards.get(&id.0))
            .map(|s| s.validator_count)
            .sum()
    }
}

// ─── Shard Reconfiguration Coordinator ───────────────────────────────────

/// Coordinates shard split and merge operations.
///
/// This is the top-level orchestrator that:
/// 1. Receives load metrics from each shard
/// 2. Runs the NWO predictor
/// 3. Initiates split/merge proposals
/// 4. Tracks proposal approval
/// 5. Executes bisections
/// 6. Updates the shard registry
pub struct ShardCoordinator {
    /// The shard registry.
    pub registry: ShardRegistry,

    /// The NWO load predictor.
    pub predictor: NwoPredictor,

    /// Active split state machines (shard_id → state).
    pub active_splits: HashMap<u64, SplitStateMachine>,

    /// Active merge state machines (left_shard_id → state).
    pub active_merges: HashMap<u64, MergeStateMachine>,

    /// Pending events to be consumed by the networking layer.
    pub pending_events: Vec<ShardEvent>,
}

impl ShardCoordinator {
    /// Create a new shard coordinator.
    pub fn new(registry_config: ShardRegistryConfig, predictor: NwoPredictor) -> Self {
        Self {
            registry: ShardRegistry::new(registry_config),
            predictor,
            active_splits: HashMap::new(),
            active_merges: HashMap::new(),
            pending_events: Vec::new(),
        }
    }

    /// Evaluate a shard for potential split based on current metrics.
    ///
    /// Returns true if a split proposal was initiated.
    pub fn evaluate_for_split(
        &mut self,
        shard_id: ShardId,
        metrics: &ShardMetrics,
        height: BlockHeight,
        proposer: ValidatorId,
    ) -> bool {
        // Check preconditions
        if !self.registry.can_split() {
            return false;
        }

        let shard = match self.registry.get(&shard_id) {
            Some(s) if s.can_split() => s.clone(),
            _ => return false,
        };

        // Run the NWO predictor
        let prediction = self.predictor.predict_overload(metrics);

        if prediction.overload_probability > self.registry.config.split_threshold {
            // Initiate a split proposal
            let proposal = SplitProposal::new(
                shard_id,
                height,
                proposer,
                prediction.overload_probability,
            );

            let sm = SplitStateMachine::new(proposal);
            self.active_splits.insert(shard_id.0, sm);

            self.pending_events.push(ShardEvent::SplitProposed {
                shard_id,
                proposer,
                height,
            });

            return true;
        }

        false
    }

    /// Evaluate two sibling shards for potential merge.
    pub fn evaluate_for_merge(
        &mut self,
        left_id: ShardId,
        right_id: ShardId,
        left_metrics: &ShardMetrics,
        right_metrics: &ShardMetrics,
    ) -> bool {
        let left = match self.registry.get(&left_id) {
            Some(s) if s.can_merge() => s,
            _ => return false,
        };

        // Check if both shards are below merge threshold
        let combined_tps = left_metrics.tps + right_metrics.tps;
        if combined_tps > self.registry.config.merge_tps_threshold * 2.0 {
            return false;
        }

        // Check if both are siblings
        if left.sibling_shard_id != Some(right_id) {
            return false;
        }

        let proposal = MergeProposal::new(left_id, right_id, combined_tps);
        let ms = MergeStateMachine::new(proposal);
        self.active_merges.insert(left_id.0, ms);

        self.pending_events.push(ShardEvent::MergeProposed {
            left_shard_id: left_id,
            right_shard_id: right_id,
        });

        true
    }

    /// Drain pending events.
    pub fn drain_events(&mut self) -> Vec<ShardEvent> {
        std::mem::take(&mut self.pending_events)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_metadata_genesis() {
        let genesis = ShardMetadata::genesis();
        assert_eq!(genesis.shard_id, ShardId::GENESIS);
        assert_eq!(genesis.lifecycle, ShardLifecycle::Active);
        assert!(genesis.can_split());
        assert!(!genesis.can_merge()); // No sibling
    }

    #[test]
    fn test_shard_registry_new() {
        let registry = ShardRegistry::new(ShardRegistryConfig::default());
        assert_eq!(registry.active_count(), 1);
        assert!(registry.get(&ShardId::GENESIS).is_some());
    }

    #[test]
    fn test_shard_registry_allocate_id() {
        let mut registry = ShardRegistry::new(ShardRegistryConfig::default());
        let id1 = registry.allocate_shard_id();
        let id2 = registry.allocate_shard_id();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_shard_registry_register() {
        let mut registry = ShardRegistry::new(ShardRegistryConfig::default());
        let new_id = registry.allocate_shard_id();
        let metadata = ShardMetadata::new(
            new_id,
            Some(ShardId::GENESIS),
            BlockHeight::from(100),
            EmbeddingRoot::GENESIS,
            1,
        );
        assert!(registry.register_shard(metadata).is_ok());
        assert_eq!(registry.active_count(), 2);
    }

    #[test]
    fn test_shard_registry_find_shard_for_account() {
        let registry = ShardRegistry::new(ShardRegistryConfig::default());
        let shard = registry.find_shard_for_account(b"test_account");
        assert!(shard.is_some());
        assert_eq!(shard.unwrap(), ShardId::GENESIS);
    }

    #[test]
    fn test_shard_lifecycle_display() {
        assert_eq!(ShardLifecycle::Active.to_string(), "active");
        assert_eq!(ShardLifecycle::Splitting.to_string(), "splitting");
        assert_eq!(ShardLifecycle::Merged.to_string(), "merged");
    }

    #[test]
    fn test_shard_metadata_update_tps() {
        let mut meta = ShardMetadata::genesis();
        meta.current_tps = 100.0;
        meta.update_tps_ema(200.0, 0.3);
        // EMA: 0.3 * 200 + 0.7 * 100 = 60 + 70 = 130
        assert!((meta.current_tps - 130.0).abs() < 1e-6);
    }

    #[test]
    fn test_shard_registry_mark_split() {
        let mut registry = ShardRegistry::new(ShardRegistryConfig::default());
        let left = registry.allocate_shard_id();
        let right = registry.allocate_shard_id();

        // Register children
        registry.register_shard(ShardMetadata::new(
            left, Some(ShardId::GENESIS), BlockHeight::from(1), EmbeddingRoot::GENESIS, 1,
        )).unwrap();
        registry.register_shard(ShardMetadata::new(
            right, Some(ShardId::GENESIS), BlockHeight::from(1), EmbeddingRoot::GENESIS, 1,
        )).unwrap();

        assert!(registry.mark_split(ShardId::GENESIS, left, right).is_ok());

        let parent = registry.get(&ShardId::GENESIS).unwrap();
        assert_eq!(parent.lifecycle, ShardLifecycle::Split);
        assert_eq!(parent.left_child, Some(left));
        assert_eq!(parent.right_child, Some(right));
        // Genesis should no longer be in active list
        assert!(!registry.active_shards().contains(&ShardId::GENESIS));
    }

    #[test]
    fn test_shard_registry_total_tps() {
        let mut registry = ShardRegistry::new(ShardRegistryConfig::default());
        registry.get_mut(&ShardId::GENESIS).unwrap().current_tps = 500.0;
        assert!((registry.total_tps() - 500.0).abs() < 1e-6);
    }
}

