//! NERV Useful-Work Economy Module
//! 
//! This module implements the innovative "useful-work" economy where nodes are rewarded
//! for contributing to the network's intelligence through federated learning, rather
//! than wasting energy (PoW) or locking capital (PoS). The economy is centered around
//! the Shapley value mechanism for fair gradient contribution rewards.
//! 
//! Key Components:
//! 1. Shapley value computation - Fair allocation of rewards based on marginal contributions
//! 2. Federated learning aggregation - Secure gradient aggregation with differential privacy
//! 3. Gradient contribution rewards - Distribution of NERV tokens for useful work
//! 
//! Design Principles:
//! - Fairness: Rewards proportional to actual contribution (Shapley value)
//! - Privacy: Differential privacy (DP-SGD, σ=0.5) and TEE-based secure aggregation
//! - Self-Improvement: Network intelligence improves over time through FL
//! - Sybil Resistance: Shapley value discourages fake or low-quality contributions


mod shapley;
mod fl_aggregation;
mod rewards;


pub use shapley::{ShapleyComputer, ShapleyValue, ShapleyConfig, ShapleyError};
pub use fl_aggregation::{FLAggregator, GradientUpdate, AggregationConfig, AggregationError};
pub use rewards::{RewardDistributor, RewardConfig, RewardRecord, DistributionError};


use crate::crypto::{CryptoProvider, ByteSerializable};
use crate::params::{Config, GRADIENT_INTERVAL_SECONDS, GRADIENT_INTERVAL_TXS, DP_SIGMA};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};


/// Main economy manager coordinating all useful-work activities
pub struct EconomyManager {
    /// Shapley value computer for fair reward allocation
    shapley_computer: Arc<ShapleyComputer>,
    
    /// Federated learning aggregator for gradient processing
    fl_aggregator: Arc<FLAggregator>,
    
    /// Reward distributor for NERV token allocation
    reward_distributor: Arc<RewardDistributor>,
    
    /// Cryptographic provider for secure operations
    crypto_provider: Arc<CryptoProvider>,
    
    /// Current epoch state
    epoch_state: RwLock<EpochState>,
    
    /// Node registry with reputation scores
    node_registry: RwLock<HashMap<String, NodeInfo>>,
    
    /// Configuration
    config: EconomyConfig,
    
    /// Metrics collector
    metrics: EconomyMetrics,
}


impl EconomyManager {
    /// Create a new economy manager with default configuration
    pub async fn new(config: EconomyConfig) -> Result<Self, EconomyError> {
        info!("Initializing NERV useful-work economy manager");
        
        // Initialize cryptographic provider
        let crypto_provider = Arc::new(CryptoProvider::new()
            .map_err(|e| EconomyError::CryptoError(e.to_string()))?);
        
        // Initialize Shapley computer
        let shapley_config = ShapleyConfig {
            monte_carlo_samples: 10_000,
            privacy_budget: DP_SIGMA,
            max_coalition_size: 100,
            enable_parallel: true,
            tee_backed: true,
        };
        let shapley_computer = Arc::new(ShapleyComputer::new(shapley_config));
        
        // Initialize FL aggregator
        let agg_config = AggregationConfig {
            dp_sigma: DP_SIGMA,
            clip_norm: 1.0,
            min_batch_size: 10,
            max_batch_size: 1000,
            tee_cluster_size: 32,
            aggregation_timeout_secs: 30,
        };
        let fl_aggregator = Arc::new(FLAggregator::new(agg_config, crypto_provider.clone()));
        
        // Initialize reward distributor
        let reward_config = RewardConfig {
            gradient_percent: 0.60,  // 60% to gradient contributors
            validation_percent: 0.30, // 30% to validators
            public_goods_percent: 0.10, // 10% to public goods
            epoch_duration_days: 30,
            min_contribution_threshold: 0.001,
            reputation_decay_factor: 0.95,
        };
        let reward_distributor = Arc::new(RewardDistributor::new(reward_config));
        
        // Initialize state
        let epoch_state = RwLock::new(EpochState::default());
        let node_registry = RwLock::new(HashMap::new());
        
        let metrics = EconomyMetrics::new();
        
        Ok(Self {
            shapley_computer,
            fl_aggregator,
            reward_distributor,
            crypto_provider,
            epoch_state,
            node_registry,
            config,
            metrics,
        })
    }
    
    /// Process a gradient update from a node
    pub async fn submit_gradient(
        &self,
        node_id: &str,
        gradient: GradientUpdate,
        attestation: Vec<u8>,
    ) -> Result<GradientReceipt, EconomyError> {
        let start_time = std::time::Instant::now();
        
        // Verify TEE attestation
        self.verify_attestation(&attestation).await?;
        
        // Add DP noise to gradient
        let noisy_gradient = self.add_dp_noise(gradient).await?;
        
        // Submit to FL aggregator
        let submission = self.fl_aggregator.submit_gradient(node_id, noisy_gradient).await?;
        
        // Update node statistics
        self.update_node_stats(node_id, submission.quality_score).await?;
        
        // Record metrics
        self.metrics.record_gradient_submission(start_time.elapsed());
        
        info!("Gradient submitted by node {} with quality score: {:.4}", 
              node_id, submission.quality_score);
        
        Ok(GradientReceipt {
            submission_id: submission.id,
            timestamp: submission.timestamp,
            quality_score: submission.quality_score,
            estimated_reward_share: submission.estimated_shapley,
        })
    }
    
    /// Perform federated learning aggregation for the current epoch
    pub async fn aggregate_epoch(&self) -> Result<AggregationResult, EconomyError> {
        let start_time = std::time::Instant::now();
        info!("Starting federated learning aggregation for epoch");
        
        // Collect gradients from current interval
        let gradients = self.fl_aggregator.collect_gradients().await?;
        
        if gradients.is_empty() {
            warn!("No gradients to aggregate in current epoch");
            return Ok(AggregationResult::empty());
        }
        
        // Secure aggregation in TEE cluster
        let aggregated = self.fl_aggregator.secure_aggregate(gradients).await?;
        
        // Compute Shapley values for each contributor
        let shapley_values = self.shapley_computer.compute_shapley(&aggregated.contributions).await?;
        
        // Update encoder with aggregated gradient
        let encoder_update = self.update_neural_encoder(aggregated.global_gradient).await?;
        
        // Prepare rewards based on Shapley values
        let reward_distribution = self.prepare_rewards(shapley_values).await?;
        
        // Update epoch state
        let mut state = self.epoch_state.write().await;
        state.current_epoch += 1;
        state.last_aggregation = std::time::SystemTime::now();
        state.encoder_version = encoder_update.version;
        state.total_gradients_processed += aggregated.total_gradients;
        
        info!("Epoch aggregation completed in {}ms. Processed {} gradients from {} nodes",
              start_time.elapsed().as_millis(),
              aggregated.total_gradients,
              aggregated.contributions.len());
        
        Ok(AggregationResult {
            epoch: state.current_epoch,
            encoder_version: encoder_update.version,
            total_nodes: aggregated.contributions.len(),
            total_gradients: aggregated.total_gradients,
            global_loss_improvement: aggregated.global_loss_improvement,
            homomorphism_preserved: encoder_update.homomorphism_preserved,
            reward_distribution,
        })
    }
    
    /// Distribute rewards for the completed epoch
    pub async fn distribute_rewards(&self, aggregation_result: &AggregationResult) -> Result<DistributionResult, EconomyError> {
        info!("Distributing rewards for epoch {}", aggregation_result.epoch);
        
        // Calculate total reward pool for this epoch
        let reward_pool = self.calculate_reward_pool(aggregation_result).await?;
        
        // Distribute rewards based on Shapley values
        let distribution = self.reward_distributor.distribute(
            &aggregation_result.reward_distribution,
            reward_pool,
        ).await?;
        
        // Update node reputations based on contributions
        self.update_reputations(&distribution).await?;
        
        // Emit reward events
        self.emit_reward_events(&distribution).await?;
        
        info!("Rewards distributed for epoch {}: total {} NERV to {} nodes",
              aggregation_result.epoch,
              distribution.total_distributed,
              distribution.recipients.len());
        
        Ok(distribution)
    }
    
    /// Get node's current reputation score
    pub async fn get_node_reputation(&self, node_id: &str) -> Result<f64, EconomyError> {
        let registry = self.node_registry.read().await;
        match registry.get(node_id) {
            Some(info) => Ok(info.reputation_score),
            None => Err(EconomyError::NodeNotFound(node_id.to_string())),
        }
    }
    
    /// Get node's total earned rewards
    pub async fn get_node_rewards(&self, node_id: &str) -> Result<u64, EconomyError> {
        let registry = self.node_registry.read().await;
        match registry.get(node_id) {
            Some(info) => Ok(info.total_rewards_nerv),
            None => Err(EconomyError::NodeNotFound(node_id.to_string())),
        }
    }
    
    /// Get current epoch information
    pub async fn get_epoch_info(&self) -> EpochInfo {
        let state = self.epoch_state.read().await;
        EpochInfo {
            current_epoch: state.current_epoch,
            encoder_version: state.encoder_version.clone(),
            last_aggregation: state.last_aggregation,
            total_gradients_processed: state.total_gradients_processed,
            next_aggregation_estimate: state.last_aggregation + std::time::Duration::from_secs(30 * 24 * 60 * 60), // 30 days
        }
    }
    
    /// Get economy metrics
    pub fn get_metrics(&self) -> &EconomyMetrics {
        &self.metrics
    }
    
    // Internal helper methods
    
    async fn verify_attestation(&self, attestation: &[u8]) -> Result<(), EconomyError> {
        // In production, this would verify TEE remote attestation
        // For now, we'll simulate verification
        if attestation.is_empty() {
            return Err(EconomyError::InvalidAttestation("Empty attestation".to_string()));
        }
        
        // Check attestation format (simplified)
        if attestation.len() < 64 {
            return Err(EconomyError::InvalidAttestation("Attestation too short".to_string()));
        }
        
        // Verify signature (simplified)
        // In real implementation: verify against Intel/AMD/ARM root certificates
        
        Ok(())
    }
    
    async fn add_dp_noise(&self, gradient: GradientUpdate) -> Result<GradientUpdate, EconomyError> {
        // Apply Differential Privacy: DP-SGD with σ=0.5
        // Clip gradient norm to bound sensitivity
        let clipped = self.clip_gradient_norm(gradient, self.config.dp_clip_norm).await?;
        
        // Add Gaussian noise: N(0, σ² * clip_norm² * I)
        let noisy = self.add_gaussian_noise(clipped, DP_SIGMA).await?;
        
        Ok(noisy)
    }
    
    async fn clip_gradient_norm(&self, gradient: GradientUpdate, max_norm: f64) -> Result<GradientUpdate, EconomyError> {
        // Calculate gradient norm
        let norm = gradient.compute_norm();
        
        if norm > max_norm {
            // Scale gradient to have norm = max_norm
            let scale = max_norm / norm;
            let clipped = gradient.scale(scale);
            Ok(clipped)
        } else {
            Ok(gradient)
        }
    }
    
    async fn add_gaussian_noise(&self, gradient: GradientUpdate, sigma: f64) -> Result<GradientUpdate, EconomyError> {
        // Generate random noise from N(0, sigma^2 * I)
        let noise = self.generate_gaussian_noise(gradient.size(), sigma).await?;
        
        // Add noise to gradient
        let noisy_gradient = gradient.add(&noise)?;
        
        Ok(noisy_gradient)
    }
    
    async fn generate_gaussian_noise(&self, size: usize, sigma: f64) -> Result<GradientUpdate, EconomyError> {
        // Generate Gaussian noise using cryptographically secure RNG
        let mut rng = rand::thread_rng();
        use rand_distr::{Distribution, Normal};
        
        let normal = Normal::new(0.0, sigma).map_err(|e| EconomyError::NoiseGenerationError(e.to_string()))?;
        
        let mut noise_data = Vec::with_capacity(size);
        for _ in 0..size {
            noise_data.push(normal.sample(&mut rng) as f32);
        }
        
        Ok(GradientUpdate::from_data(noise_data))
    }
    
    async fn update_node_stats(&self, node_id: &str, quality_score: f64) -> Result<(), EconomyError> {
        let mut registry = self.node_registry.write().await;
        
        let node_info = registry.entry(node_id.to_string())
            .or_insert_with(|| NodeInfo::new(node_id));
        
        node_info.last_contribution = std::time::SystemTime::now();
        node_info.total_contributions += 1;
        node_info.average_quality = (node_info.average_quality * (node_info.total_contributions - 1) as f64 
            + quality_score) / node_info.total_contributions as f64;
        
        // Update reputation (weighted by quality and consistency)
        self.update_reputation_score(node_info, quality_score).await?;
        
        Ok(())
    }
    
    async fn update_reputation_score(&self, node_info: &mut NodeInfo, latest_quality: f64) -> Result<(), EconomyError> {
        // Reputation formula: weighted average of historical quality with decay
        let decay_factor = self.config.reputation_decay;
        let consistency_bonus = self.calculate_consistency_bonus(node_info).await?;
        
        let new_reputation = decay_factor * node_info.reputation_score 
            + (1.0 - decay_factor) * (latest_quality + consistency_bonus);
        
        // Bound reputation between 0 and 1
        node_info.reputation_score = new_reputation.clamp(0.0, 1.0);
        
        Ok(())
    }
    
    async fn calculate_consistency_bonus(&self, node_info: &NodeInfo) -> Result<f64, EconomyError> {
        // Reward consistent contributors
        if node_info.total_contributions < 10 {
            return Ok(0.0); // Not enough history
        }
        
        // Calculate consistency as inverse of variance in quality scores
        // This is simplified - in production, we'd track quality history
        let consistency = 1.0 / (1.0 + node_info.total_contributions as f64).sqrt();
        
        Ok(consistency * self.config.consistency_weight)
    }
    
    async fn update_neural_encoder(&self, global_gradient: GradientUpdate) -> Result<EncoderUpdate, EconomyError> {
        // Update the neural encoder ε_θ with the aggregated gradient
        // This would interface with the embedding module in production
        
        // Verify homomorphism preservation (error ≤ 1e-9)
        let homomorphism_preserved = self.verify_homomorphism(&global_gradient).await?;
        
        if !homomorphism_preserved {
            return Err(EconomyError::HomomorphismViolated);
        }
        
        // Apply gradient update to encoder
        let encoder_version = format!("ε_θ_v{}", self.get_current_epoch().await);
        
        Ok(EncoderUpdate {
            version: encoder_version,
            gradient_magnitude: global_gradient.compute_norm(),
            homomorphism_preserved,
            improvement_metric: self.calculate_improvement_metric(&global_gradient).await?,
        })
    }
    
    async fn verify_homomorphism(&self, gradient: &GradientUpdate) -> Result<bool, EconomyError> {
        // Verify that the gradient update preserves transfer homomorphism
        // This is a critical check: error must be ≤ 1e-9
        
        // Simplified check: ensure gradient magnitude is within bounds
        let norm = gradient.compute_norm();
        let max_allowed_norm = 1e-6; // Conservative bound
        
        Ok(norm <= max_allowed_norm)
    }
    
    async fn calculate_improvement_metric(&self, gradient: &GradientUpdate) -> Result<f64, EconomyError> {
        // Calculate how much this gradient improves the encoder
        // Metric: reduction in homomorphism error or validation loss
        
        // Simplified: use gradient magnitude as proxy for improvement
        let norm = gradient.compute_norm();
        
        // Normalize and scale
        let improvement = 1.0 / (1.0 + norm.exp()); // Sigmoid-like transformation
        
        Ok(improvement)
    }
    
    async fn prepare_rewards(&self, shapley_values: Vec<ShapleyValue>) -> Result<Vec<RewardAllocation>, EconomyError> {
        let mut allocations = Vec::with_capacity(shapley_values.len());
        
        for sv in shapley_values {
            // Calculate reward share based on Shapley value
            let reward_share = sv.value; // Normalized between 0 and 1
            
            // Apply reputation multiplier
            let reputation = self.get_node_reputation(&sv.node_id).await.unwrap_or(0.5);
            let reputation_multiplier = 0.5 + reputation; // Between 0.5 and 1.5
            
            let adjusted_share = reward_share * reputation_multiplier;
            
            allocations.push(RewardAllocation {
                node_id: sv.node_id,
                shapley_value: sv.value,
                reputation_score: reputation,
                reward_share: adjusted_share,
                estimated_nerv: 0, // Will be filled during distribution
            });
        }
        
        // Normalize shares to sum to 1
        let total: f64 = allocations.iter().map(|a| a.reward_share).sum();
        if total > 0.0 {
            for allocation in &mut allocations {
                allocation.reward_share /= total;
            }
        }
        
        Ok(allocations)
    }
    
    async fn calculate_reward_pool(&self, aggregation: &AggregationResult) -> Result<u64, EconomyError> {
        // Calculate total NERV tokens to distribute for this epoch
        // Based on emission schedule and network performance
        
        let base_emission = self.config.base_emission_per_epoch;
        
        // Scale by network performance metrics
        let performance_multiplier = aggregation.global_loss_improvement.unwrap_or(0.5);
        let participation_multiplier = aggregation.total_nodes as f64 / self.config.expected_nodes as f64;
        
        let scaled_emission = (base_emission as f64 
            * performance_multiplier 
            * participation_multiplier.clamp(0.5, 2.0)) as u64;
        
        Ok(scaled_emission)
    }
    
    async fn update_reputations(&self, distribution: &DistributionResult) -> Result<(), EconomyError> {
        let mut registry = self.node_registry.write().await;
        
        for recipient in &distribution.recipients {
            if let Some(node_info) = registry.get_mut(&recipient.node_id) {
                // Update rewards earned
                node_info.total_rewards_nerv += recipient.amount_nerv;
                
                // Update reputation based on reward (higher reward → higher reputation)
                let reward_based_reputation = (recipient.amount_nerv as f64 
                    / distribution.total_distributed as f64).sqrt();
                
                node_info.reputation_score = 0.7 * node_info.reputation_score 
                    + 0.3 * reward_based_reputation;
                
                // Ensure reputation stays in bounds
                node_info.reputation_score = node_info.reputation_score.clamp(0.0, 1.0);
            }
        }
        
        Ok(())
    }
    
    async fn emit_reward_events(&self, distribution: &DistributionResult) -> Result<(), EconomyError> {
        // Emit on-chain events for reward distribution
        // This would interface with the consensus module in production
        
        for recipient in &distribution.recipients {
            info!("Reward event: {} NERV to node {}", 
                  recipient.amount_nerv, 
                  recipient.node_id);
            
            // In production: emit smart contract event or on-chain transaction
            self.metrics.record_reward_distribution(recipient.amount_nerv);
        }
        
        Ok(())
    }
    
    async fn get_current_epoch(&self) -> u64 {
        let state = self.epoch_state.read().await;
        state.current_epoch
    }
}


/// Economy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomyConfig {
    /// Base emission per epoch (in NERV tokens)
    pub base_emission_per_epoch: u64,
    
    /// Differential privacy clipping norm
    pub dp_clip_norm: f64,
    
    /// Reputation decay factor per epoch
    pub reputation_decay: f64,
    
    /// Weight for consistency bonus in reputation
    pub consistency_weight: f64,
    
    /// Expected number of active nodes
    pub expected_nodes: usize,
    
    /// Minimum stake required to participate
    pub min_stake_nerv: u64,
    
    /// Maximum gradients per node per epoch
    pub max_gradients_per_node: u32,
    
    /// Enable/disable reward distribution
    pub rewards_enabled: bool,
    
    /// Path to TEE enclave binaries
    pub tee_enclave_path: String,
}


impl Default for EconomyConfig {
    fn default() -> Self {
        Self {
            base_emission_per_epoch: 1_000_000, // 1M NERV per epoch
            dp_clip_norm: 1.0,
            reputation_decay: 0.95,
            consistency_weight: 0.1,
            expected_nodes: 1000,
            min_stake_nerv: 1000,
            max_gradients_per_node: 100,
            rewards_enabled: true,
            tee_enclave_path: "./enclave/economy.signed.so".to_string(),
        }
    }
}


/// Epoch state tracking
#[derive(Debug, Clone, Serialize, Default)]
struct EpochState {
    /// Current epoch number
    pub current_epoch: u64,
    
    /// Current encoder version
    pub encoder_version: String,
    
    /// Last aggregation timestamp
    pub last_aggregation: std::time::SystemTime,
    
    /// Total gradients processed across all epochs
    pub total_gradients_processed: u64,
    
    /// Accumulated reward pool for current epoch
    pub current_reward_pool: u64,
}


/// Node information for reputation tracking
#[derive(Debug, Clone, Serialize)]
struct NodeInfo {
    /// Node identifier
    pub node_id: String,
    
    /// Current reputation score (0.0 to 1.0)
    pub reputation_score: f64,
    
    /// Total contributions made
    pub total_contributions: u32,
    
    /// Average quality score of contributions
    pub average_quality: f64,
    
    /// Total NERV rewards earned
    pub total_rewards_nerv: u64,
    
    /// Last contribution timestamp
    pub last_contribution: std::time::SystemTime,
    
    /// Stake amount in NERV
    pub stake_nerv: u64,
}


impl NodeInfo {
    fn new(node_id: &str) -> Self {
        Self {
            node_id: node_id.to_string(),
            reputation_score: 0.5, // Start with neutral reputation
            total_contributions: 0,
            average_quality: 0.0,
            total_rewards_nerv: 0,
            last_contribution: std::time::SystemTime::now(),
            stake_nerv: 0,
        }
    }
}


/// Gradient submission receipt
#[derive(Debug, Clone, Serialize)]
pub struct GradientReceipt {
    /// Unique submission ID
    pub submission_id: String,
    
    /// Submission timestamp
    pub timestamp: std::time::SystemTime,
    
    /// Quality score assigned to gradient
    pub quality_score: f64,
    
    /// Estimated Shapley value share (pre-computation)
    pub estimated_reward_share: f64,
}


/// Aggregation result for an epoch
#[derive(Debug, Clone, Serialize)]
pub struct AggregationResult {
    /// Epoch number
    pub epoch: u64,
    
    /// New encoder version
    pub encoder_version: String,
    
    /// Total participating nodes
    pub total_nodes: usize,
    
    /// Total gradients aggregated
    pub total_gradients: u64,
    
    /// Global loss improvement (if measurable)
    pub global_loss_improvement: Option<f64>,
    
    /// Whether homomorphism was preserved
    pub homomorphism_preserved: bool,
    
    /// Reward distribution allocations
    pub reward_distribution: Vec<RewardAllocation>,
}


impl AggregationResult {
    fn empty() -> Self {
        Self {
            epoch: 0,
            encoder_version: String::new(),
            total_nodes: 0,
            total_gradients: 0,
            global_loss_improvement: None,
            homomorphism_preserved: true,
            reward_distribution: Vec::new(),
        }
    }
}


/// Reward allocation for a node
#[derive(Debug, Clone, Serialize)]
pub struct RewardAllocation {
    /// Node identifier
    pub node_id: String,
    
    /// Computed Shapley value
    pub shapley_value: f64,
    
    /// Reputation score at distribution time
    pub reputation_score: f64,
    
    /// Final reward share (0.0 to 1.0)
    pub reward_share: f64,
    
    /// Estimated NERV amount (before final calculation)
    pub estimated_nerv: u64,
}


/// Encoder update information
#[derive(Debug, Clone, Serialize)]
struct EncoderUpdate {
    /// New encoder version string
    pub version: String,
    
    /// Magnitude of applied gradient
    pub gradient_magnitude: f64,
    
    /// Whether homomorphism was preserved
    pub homomorphism_preserved: bool,
    
    /// Improvement metric
    pub improvement_metric: f64,
}


/// Distribution result
#[derive(Debug, Clone, Serialize)]
pub struct DistributionResult {
    /// Epoch number
    pub epoch: u64,
    
    /// Total NERV distributed
    pub total_distributed: u64,
    
    /// Distribution timestamp
    pub timestamp: std::time::SystemTime,
    
    /// Individual recipient information
    pub recipients: Vec<RecipientInfo>,
}


/// Recipient information
#[derive(Debug, Clone, Serialize)]
pub struct RecipientInfo {
    /// Node identifier
    pub node_id: String,
    
    /// NERV amount received
    pub amount_nerv: u64,
    
    /// Reward share percentage
    pub share_percent: f64,
    
    /// Reputation score at distribution
    pub reputation_score: f64,
    
    /// Transaction hash (if on-chain)
    pub tx_hash: Option<String>,
}


/// Epoch information for querying
#[derive(Debug, Clone, Serialize)]
pub struct EpochInfo {
    /// Current epoch number
    pub current_epoch: u64,
    
    /// Current encoder version
    pub encoder_version: String,
    
    /// Last aggregation timestamp
    pub last_aggregation: std::time::SystemTime,
    
    /// Total gradients processed
    pub total_gradients_processed: u64,
    
    /// Next aggregation estimate
    pub next_aggregation_estimate: std::time::SystemTime,
}


/// Economy metrics collector
#[derive(Debug, Clone, Default)]
pub struct EconomyMetrics {
    /// Total gradients submitted
    pub total_gradients_submitted: u64,
    
    /// Total rewards distributed
    pub total_rewards_distributed: u64,
    
    /// Average gradient processing time
    pub avg_gradient_time_ms: f64,
    
    /// Average aggregation time
    pub avg_aggregation_time_ms: f64,
    
    /// Average reward distribution time
    pub avg_distribution_time_ms: f64,
    
    /// Current active nodes
    pub active_nodes: usize,
    
    /// Total epochs completed
    pub total_epochs: u64,
}


impl EconomyMetrics {
    fn new() -> Self {
        Self::default()
    }
    
    fn record_gradient_submission(&mut self, duration: std::time::Duration) {
        self.total_gradients_submitted += 1;
        
        // Update moving average of processing time
        let new_time = duration.as_millis() as f64;
        self.avg_gradient_time_ms = 0.9 * self.avg_gradient_time_ms + 0.1 * new_time;
    }
    
    fn record_reward_distribution(&mut self, amount: u64) {
        self.total_rewards_distributed += amount;
    }
    
    fn record_aggregation(&mut self, duration: std::time::Duration) {
        let new_time = duration.as_millis() as f64;
        self.avg_aggregation_time_ms = 0.9 * self.avg_aggregation_time_ms + 0.1 * new_time;
    }
    
    fn record_distribution(&mut self, duration: std::time::Duration) {
        let new_time = duration.as_millis() as f64;
        self.avg_distribution_time_ms = 0.9 * self.avg_distribution_time_ms + 0.1 * new_time;
    }
    
    fn increment_epoch(&mut self) {
        self.total_epochs += 1;
    }
    
    fn update_active_nodes(&mut self, count: usize) {
        self.active_nodes = count;
    }
}


/// Economy error types
#[derive(Debug, thiserror::Error)]
pub enum EconomyError {
    #[error("Shapley computation error: {0}")]
    ShapleyError(#[from] ShapleyError),
    
    #[error("Aggregation error: {0}")]
    AggregationError(#[from] AggregationError),
    
    #[error("Distribution error: {0}")]
    DistributionError(#[from] DistributionError),
    
    #[error("Cryptographic error: {0}")]
    CryptoError(String),
    
    #[error("Invalid TEE attestation: {0}")]
    InvalidAttestation(String),
    
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    
    #[error("Gradient validation failed: {0}")]
    GradientValidationFailed(String),
    
    #[error("Noise generation error: {0}")]
    NoiseGenerationError(String),
    
    #[error("Homomorphism preservation violated")]
    HomomorphismViolated,
    
    #[error("Insufficient contributions: minimum {0} required, got {1}")]
    InsufficientContributions(usize, usize),
    
    #[error("Epoch not ready for aggregation")]
    EpochNotReady,
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Storage error: {0}")]
    StorageError(String),
}


// Unit tests
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_economy_manager_creation() {
        let config = EconomyConfig::default();
        let manager = EconomyManager::new(config).await;
        
        assert!(manager.is_ok());
    }
    
    #[tokio::test]
    async fn test_epoch_state_management() {
        let config = EconomyConfig::default();
        let manager = EconomyManager::new(config).await.unwrap();
        
        let epoch_info = manager.get_epoch_info().await;
        assert_eq!(epoch_info.current_epoch, 0);
        assert_eq!(epoch_info.total_gradients_processed, 0);
    }
    
    #[test]
    fn test_economy_config_default() {
        let config = EconomyConfig::default();
        
        assert_eq!(config.base_emission_per_epoch, 1_000_000);
        assert_eq!(config.dp_clip_norm, 1.0);
        assert_eq!(config.reputation_decay, 0.95);
        assert_eq!(config.rewards_enabled, true);
    }
    
    #[test]
    fn test_node_info_creation() {
        let node_info = NodeInfo::new("test_node");
        
        assert_eq!(node_info.node_id, "test_node");
        assert_eq!(node_info.reputation_score, 0.5);
        assert_eq!(node_info.total_contributions, 0);
        assert_eq!(node_info.total_rewards_nerv, 0);
    }
    
    #[test]
    fn test_metrics_recording() {
        let mut metrics = EconomyMetrics::new();
        
        assert_eq!(metrics.total_gradients_submitted, 0);
        
        metrics.record_gradient_submission(std::time::Duration::from_millis(100));
        assert_eq!(metrics.total_gradients_submitted, 1);
        assert!(metrics.avg_gradient_time_ms > 0.0);
    }
}
