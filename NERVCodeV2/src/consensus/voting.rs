//! Weighted quorum voting system
//! 
//! This module implements NERV's weighted quorum voting where each validator's
//! voting power is determined by: stake × reputation
//! 
//! Features:
//! - Dynamic validator set with stake-based weighting
//! - Reputation system for honest behavior
//! - Slashing for misbehavior
//! - View change protocol for liveness
//! - 3-second finality with 33% adversary tolerance

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Validator information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Validator {
    /// Validator address
    pub address: [u8; 20],
    
    /// Staked amount (wei)
    pub stake: u128,
    
    /// Reputation score (0-1)
    pub reputation: f64,
    
    /// Voting power (stake × reputation)
    pub voting_power: u128,
    
    /// Commission rate (0-1)
    pub commission_rate: f64,
    
    /// Last active block
    pub last_active_block: u64,
    
    /// Total blocks produced
    pub blocks_produced: u64,
    
    /// Slashing history
    pub slashing_events: Vec<SlashingEvent>,
    
    /// Jail status (if jailed, cannot vote)
    pub jailed_until: Option<u64>,
    
    /// Public key for signatures
    pub public_key: Vec<u8>,
    
    /// Network address
    pub network_address: String,

}

impl Validator {
    /// Create a new validator
    pub fn new(address: [u8; 20], stake: u128, public_key: Vec<u8>) -> Self {
        let reputation = 0.5; // Starting reputation
        
        Self {
            address,
            stake,
            reputation,
            voting_power: Self::calculate_voting_power(stake, reputation),
            commission_rate: 0.1, // 10% default commission
            last_active_block: 0,
            blocks_produced: 0,
            slashing_events: Vec::new(),
            jailed_until: None,
            public_key,
            network_address: String::new(),
        }
    }
    
    /// Calculate voting power
    fn calculate_voting_power(stake: u128, reputation: f64) -> u128 {
        (stake as f64 * reputation) as u128
    }
    
    /// Update voting power based on current stake and reputation
    pub fn update_voting_power(&mut self) {
        self.voting_power = Self::calculate_voting_power(self.stake, self.reputation);
    }
    
    /// Check if validator is active (not jailed and has stake)
    pub fn is_active(&self, current_block: u64) -> bool {
        !self.is_jailed(current_block) && self.stake > 0
    }
    
    /// Check if validator is jailed
    pub fn is_jailed(&self, current_block: u64) -> bool {
        if let Some(jail_end) = self.jailed_until {
            current_block < jail_end
        } else {
            false
        }
    }
    
    /// Apply slashing penalty
    pub fn slash(&mut self, percentage: f64, reason: SlashingReason, block_height: u64) {
        let penalty = (self.stake as f64 * percentage) as u128;
        self.stake = self.stake.saturating_sub(penalty);
        
        // Reduce reputation
        self.reputation *= 0.5; // 50% reputation penalty
        
        // Record slashing event
        self.slashing_events.push(SlashingEvent {
            block_height,
            amount: penalty,
            reason,
            remaining_stake: self.stake,
        });
        
        // Update voting power
        self.update_voting_power();
    }
    
    /// Jail validator
    pub fn jail(&mut self, duration_blocks: u64, current_block: u64) {
        self.jailed_until = Some(current_block + duration_blocks);
    }
    
    /// Unjail validator
    pub fn unjail(&mut self) {
        self.jailed_until = None;
    }
    
    /// Update reputation based on performance
    pub fn update_reputation(&mut self, performance: ValidatorPerformance) {
        let mut new_reputation = self.reputation;
        
        // Reward for good performance
        if performance.blocks_produced > 0 {
            let production_rate = performance.blocks_produced as f64 / 
                                 performance.expected_blocks as f64;
            new_reputation += 0.01 * production_rate.min(1.0); // Up to 1% increase
        }
        
        if performance.vote_accuracy > 0.9 {
            new_reputation += 0.005; // 0.5% increase for high accuracy
        }
        
        // Penalize for poor performance
        if performance.latency_ms > 1000.0 {
            new_reputation *= 0.95; // 5% penalty for high latency
        }
        
        if performance.slashing_events > 0 {
            new_reputation *= 0.8f64.powi(performance.slashing_events as i32);
        }
        
        // Apply reputation decay
        new_reputation *= performance.reputation_decay;
        
        // Clamp to valid range
        self.reputation = new_reputation.clamp(0.0, 1.0);
        self.update_voting_power();
    }
}

/// Slashing event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashingEvent {
    pub block_height: u64,
    pub amount: u128,
    pub reason: SlashingReason,
    pub remaining_stake: u128,
}

/// Slashing reasons
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SlashingReason {
    DoubleSign,
    Unavailability,
    InvalidBlock,
    Censorship,
    GovernanceViolation,
    SecurityBreach,
}

/// Validator performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorPerformance {
    pub uptime: f64,
    pub vote_accuracy: f64,
    pub votes_participated: u64,
    pub blocks_produced: u64,
    pub expected_blocks: u64,
    pub latency_ms: f64,
    pub slashing_events: u64,
    pub double_sign_events: u64,
    pub reputation_decay: f64,
}

/// Vote structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    /// Block hash being voted on
    pub block_hash: [u8; 32],
    
    /// View number
    pub view: u64,
    
    /// Voter address
    pub voter: [u8; 20],
    
    /// Signature
    pub signature: Vec<u8>,
    
    /// Timestamp
    pub timestamp: u64,
    
    /// Vote type (pre-vote, pre-commit, etc.)
    pub vote_type: VoteType,
}

/// Vote type
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VoteType {
    PreVote,
    PreCommit,
    Timeout,
    ViewChange,
}

/// Vote result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoteResult {
    Accepted,
    InvalidSignature,
    InvalidValidator,
    NoVotingPower,
    AlreadyVoted,
    QuorumReached,
    ViewChanged,
}

/// Weighted quorum manager
#[derive(Debug, Clone)]
pub struct WeightedQuorum {
    /// Validator set (protected by RwLock for concurrent access)
    validators: Arc<RwLock<Vec<Validator>>>,
    
    /// Validator index by address
    validator_index: Arc<RwLock<HashMap<[u8; 20], usize>>>,
    
    /// Total voting power
    total_voting_power: Arc<RwLock<u128>>,
    
    /// Quorum threshold (2/3 + 1)
    quorum_threshold: Arc<RwLock<u128>>,
    
    /// My validator address (if I'm a validator)
    my_address: Option<[u8; 20]>,
    
    /// Epoch length in blocks
    epoch_length: u64,
    
    /// Minimum stake to become validator
    min_stake: u128,

    /// Current encoder weight hash (consensus-critical)
    current_encoder_hash: Arc<RwLock<[u8; 32]>>,
    
    /// Pending encoder update proposal (hash + supporting power)
    pending_encoder_update: Arc<RwLock<Option<([u8; 32], u128)>>>,
    
    /// Encoder update quorum threshold (stricter than BFT - 80%)
    const ENCODER_UPDATE_QUORUM: f64 = 0.8;  // 80% of total voting power
}

impl WeightedQuorum {
    /// Create a new weighted quorum
    pub fn new() -> Self {
        Self {
            validators: Arc::new(RwLock::new(Vec::new())),
            validator_index: Arc::new(RwLock::new(HashMap::new())),
            total_voting_power: Arc::new(RwLock::new(0)),
            quorum_threshold: Arc::new(RwLock::new(0)),
            my_address: None,
            epoch_length: 1000,
            min_stake: 1000 * 10u128.pow(18), // 1000 NERV
            current_encoder_hash: Arc::new(RwLock::new([0u8; 32])),
        pending_encoder_update: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Initialize validator set
    pub async fn initialize(&mut self) -> Result<()> {
        // Load validators from state
        // In production: load from blockchain state
        let mut validators = self.validators.write().await;
        let mut index = self.validator_index.write().await;
        
        // Clear existing data
        validators.clear();
        index.clear();
        
        // Add genesis validators (simplified)
        // In production: load from genesis file
        
        // Recalculate total voting power
        self.recalculate_powers().await?;
        
        Ok(())
    }
    
    /// Add a new validator
    pub async fn add_validator(&mut self, validator: Validator) -> Result<()> {
        let mut validators = self.validators.write().await;
        let mut index = self.validator_index.write().await;
        
        // Check minimum stake
        if validator.stake < self.min_stake {
            return Err(ConsensusError::ValidatorError(
                format!("Insufficient stake: {} < {}", validator.stake, self.min_stake)
            ).into());
        }
        
        // Check if validator already exists
        if index.contains_key(&validator.address) {
            return Err(ConsensusError::ValidatorError("Validator already exists".to_string()).into());
        }
        
        // Add to collections
        let idx = validators.len();
        validators.push(validator);
        index.insert(validators[idx].address, idx);
        
        // Recalculate powers
        self.recalculate_powers().await?;
        
        Ok(())
    }
    
    /// Remove a validator
    pub async fn remove_validator(&mut self, address: &[u8; 20]) -> Result<()> {
        let mut validators = self.validators.write().await;
        let mut index = self.validator_index.write().await;
        
        // Find validator index
        let idx = *index.get(address)
            .ok_or_else(|| ConsensusError::ValidatorError("Validator not found".to_string()))?;
        
        // Remove from collections
        validators.remove(idx);
        index.remove(address);
        
        // Rebuild index
        index.clear();
        for (i, validator) in validators.iter().enumerate() {
            index.insert(validator.address, i);
        }
        
        // Recalculate powers
        self.recalculate_powers().await?;
        
        Ok(())
    }
    
    /// Update validator stake
    pub async fn update_stake(&mut self, address: &[u8; 20], new_stake: u128) -> Result<()> {
        let mut validators = self.validators.write().await;
        
        // Find validator
        let idx = *self.validator_index.read().await.get(address)
            .ok_or_else(|| ConsensusError::ValidatorError("Validator not found".to_string()))?;
        
        // Update stake
        validators[idx].stake = new_stake;
        validators[idx].update_voting_power();
        
        // Recalculate powers
        self.recalculate_powers().await?;
        
        Ok(())
    }
    
    /// Update validator reputation
    pub async fn update_reputation(&mut self, address: &[u8; 20], performance: ValidatorPerformance) -> Result<()> {
        let mut validators = self.validators.write().await;
        
        // Find validator
        let idx = *self.validator_index.read().await.get(address)
            .ok_or_else(|| ConsensusError::ValidatorError("Validator not found".to_string()))?;
        
        // Update reputation
        validators[idx].update_reputation(performance);
        
        // Recalculate powers
        self.recalculate_powers().await?;
        
        Ok(())
    }
    
    /// Slash a validator
    pub async fn slash_validator(
        &mut self,
        address: &[u8; 20],
        percentage: f64,
        reason: SlashingReason,
        block_height: u64,
        jail_duration: Option<u64>,
    ) -> Result<u128> {
        let mut validators = self.validators.write().await;
        
        // Find validator
        let idx = *self.validator_index.read().await.get(address)
            .ok_or_else(|| ConsensusError::ValidatorError("Validator not found".to_string()))?;
        
        // Apply slashing
        let old_stake = validators[idx].stake;
        validators[idx].slash(percentage, reason, block_height);
        let penalty = old_stake - validators[idx].stake;
        
        // Jail if specified
        if let Some(duration) = jail_duration {
            validators[idx].jail(duration, block_height);
        }
        
        // Recalculate powers
        self.recalculate_powers().await?;
        
        Ok(penalty)
    }
    
    /// Get validator by address
    pub async fn get_validator(&self, address: &[u8; 20]) -> Result<Option<Validator>> {
        let validators = self.validators.read().await;
        let index = self.validator_index.read().await;
        
        if let Some(&idx) = index.get(address) {
            Ok(Some(validators[idx].clone()))
        } else {
            Ok(None)
        }
    }
    
    /// Get all validators
    pub async fn validators(&self) -> Result<Vec<Validator>> {
        let validators = self.validators.read().await;
        Ok(validators.clone())
    }
    
    /// Get active validators (not jailed and with stake)
    pub async fn active_validators(&self, current_block: u64) -> Result<Vec<Validator>> {
        let validators = self.validators.read().await;
        
        Ok(validators
            .iter()
            .filter(|v| v.is_active(current_block))
            .cloned()
            .collect())
    }
    
    /// Get top validators by voting power
    pub async fn top_validators(&self, count: usize) -> Result<Vec<Validator>> {
        let mut validators = self.validators().await?;
        
        // Sort by voting power (descending)
        validators.sort_by(|a, b| b.voting_power.cmp(&a.voting_power));
        
        // Take top N
        validators.truncate(count);
        
        Ok(validators)
    }
    
    /// Get voting power for a validator
    pub async fn voting_power(&self, address: &[u8; 20]) -> Result<u128> {
        if let Some(validator) = self.get_validator(address).await? {
            Ok(validator.voting_power)
        } else {
            Ok(0)
        }
    }
    
    /// Get total voting power
    pub async fn total_voting_power(&self) -> Result<u128> {
        let total = self.total_voting_power.read().await;
        Ok(*total)
    }
    
    /// Get quorum threshold (2/3 + 1)
    pub async fn quorum_threshold(&self) -> Result<u128> {
        let threshold = self.quorum_threshold.read().await;
        Ok(*threshold)
    }
    
    /// Check if address is a validator
    pub async fn is_validator(&self, address: &[u8; 20]) -> Result<bool> {
        let index = self.validator_index.read().await;
        Ok(index.contains_key(address))
    }
    
    /// Check if I am a validator
    pub async fn is_me_validator(&self) -> bool {
        if let Some(my_address) = self.my_address {
            self.is_validator(&my_address).await.unwrap_or(false)
        } else {
            false
        }
    }
    
    /// Get my validator address
    pub async fn my_address(&self) -> Result<[u8; 20]> {
        self.my_address
            .ok_or_else(|| ConsensusError::ValidatorError("Not a validator".to_string()).into())
    }
    
    /// Get number of validators
    pub async fn len(&self) -> usize {
        let validators = self.validators.read().await;
        validators.len()
    }
    
    /// Check if validator set is empty
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
    
    /// Get validator set hash (for state verification)
    pub async fn hash(&self) -> Result<[u8; 32]> {
        let validators = self.validators.read().await;
        
        let mut hasher = blake3::Hasher::new();
        
        for validator in validators.iter() {
            hasher.update(&validator.address);
            hasher.update(&validator.stake.to_be_bytes());
            hasher.update(&validator.reputation.to_be_bytes());
        }
        
        Ok(*hasher.finalize().as_bytes())
    }
    
    /// Update validator set with new list
    pub async fn update(&mut self, new_validators: Vec<Validator>) -> Result<()> {
        let mut validators = self.validators.write().await;
        let mut index = self.validator_index.write().await;
        
        // Replace validators
        *validators = new_validators;
        
        // Rebuild index
        index.clear();
        for (i, validator) in validators.iter().enumerate() {
            index.insert(validator.address, i);
        }
        
        // Recalculate powers
        self.recalculate_powers().await?;
        
        Ok(())
    }
    
    /// Recalculate total voting power and quorum threshold
    async fn recalculate_powers(&self) -> Result<()> {
        let validators = self.validators.read().await;
        
        // Calculate total voting power
        let total: u128 = validators.iter()
            .map(|v| v.voting_power)
            .sum();
        
        // Calculate quorum threshold (2/3 + 1)
        let threshold = (total * 2 / 3) + 1;
        
        // Update atomic values
        *self.total_voting_power.write().await = total;
        *self.quorum_threshold.write().await = threshold;
        
        Ok(())
    }
    
    /// Check if we have quorum for a set of votes
    pub async fn check_quorum(&self, votes: &[Vote], current_block: u64) -> Result<bool> {
        let mut total_voting_power = 0;
        let mut seen_validators = std::collections::HashSet::new();
        
        for vote in votes {
            // Prevent double counting
            if seen_validators.contains(&vote.voter) {
                continue;
            }
            
            // Check if voter is an active validator
            if let Some(validator) = self.get_validator(&vote.voter).await? {
                if validator.is_active(current_block) {
                    total_voting_power += validator.voting_power;
                    seen_validators.insert(vote.voter);
                }
            }
        }
        
        // Check against quorum threshold
        let threshold = self.quorum_threshold().await?;
        Ok(total_voting_power >= threshold)
    }
    
    /// Get proposer for a view (round-robin)
    pub async fn get_proposer(&self, view: u64) -> Result<Option<Validator>> {
        let validators = self.active_validators(view).await?; // Using view as block number for simplicity
        
        if validators.is_empty() {
            return Ok(None);
        }
        
        let proposer_index = (view as usize) % validators.len();
        Ok(validators.get(proposer_index).cloned())
    }

    /// Propose an encoder update (called by EncoderUpdater when new gradients are applied)
/// 
/// This accumulates supporting voting power for a new encoder weight hash.
/// When quorum (80% of total voting power) is reached, the update is queued for next epoch.
pub async fn propose_encoder_update(
    &mut self,
    new_hash: [u8; 32],
    proposer_power: u128,
) -> Result<()> {
    let total_power = self.total_voting_power().await?;
    let mut pending = self.pending_encoder_update.write().await;
    
    match *pending {
        Some((current_hash, current_support)) => {
            if current_hash == new_hash {
                // Same hash, add support
                let new_support = current_support + proposer_power;
                *pending = Some((new_hash, new_support));
                
                tracing::info!(
                    "Encoder update support increased: {} / {} (need: {})",
                    new_support,
                    total_power,
                    (total_power as f64 * Self::ENCODER_UPDATE_QUORUM) as u128
                );
                
                // Check if we've reached quorum immediately
                if new_support as f64 >= total_power as f64 * Self::ENCODER_UPDATE_QUORUM {
                    tracing::info!("Encoder update reached quorum immediately!");
                }
            } else {
                // Different hash - reset with new proposal (first-come-first-served per epoch)
                tracing::warn!("Encoder update conflict: {:x?} vs {:x?}, resetting", 
                    current_hash, new_hash);
                *pending = Some((new_hash, proposer_power));
            }
        }
        None => {
            // First proposal for this epoch
            *pending = Some((new_hash, proposer_power));
            tracing::info!("New encoder update proposed: {:x?} with {} power", 
                new_hash, proposer_power);
        }
    }
    
    Ok(())
}

/// Get current encoder hash (used by validators to verify predictions match current model)
pub async fn get_current_encoder_hash(&self) -> Result<[u8; 32]> {
    let hash = self.current_encoder_hash.read().await;
    Ok(*hash)
}

/// Set current encoder hash (only called at epoch transition when quorum is reached)
pub async fn set_current_encoder_hash(&mut self, new_hash: [u8; 32]) -> Result<()> {
    let mut hash = self.current_encoder_hash.write().await;
    *hash = new_hash;
    tracing::info!("Encoder hash updated to: {:x?}", new_hash);
    Ok(())
}

/// Check and apply pending encoder update at epoch transition
/// Returns true if encoder was updated
async fn check_encoder_update_at_epoch(&mut self, total_power: u128) -> Result<bool> {
    let mut pending = self.pending_encoder_update.write().await;
    
    if let Some((new_hash, supporting_power)) = *pending {
        let quorum_threshold = (total_power as f64 * Self::ENCODER_UPDATE_QUORUM) as u128;
        
        if supporting_power >= quorum_threshold {
            // Apply update
            self.set_current_encoder_hash(new_hash).await?;
            tracing::info!(
                "Encoder update applied at epoch transition: {:x?} with {}/{} support",
                new_hash, supporting_power, total_power
            );
            
            // Clear pending
            *pending = None;
            return Ok(true);
        } else {
            tracing::warn!(
                "Encoder update proposal insufficient support: {}/{} (need {}) - discarding",
                supporting_power, total_power, quorum_threshold
            );
            // Clear pending - didn't reach quorum
            *pending = None;
        }
    }
    
    Ok(false)
}
    
    /// Process epoch transition (update validator set based on performance)
    pub async fn process_epoch(&mut self, current_block: u64) -> Result<()> {
        if current_block % self.epoch_length != 0 {
            return Ok(());
        }
        
        tracing::info!("Processing epoch transition at block {}", current_block);

        // Check and apply any pending encoder updates first
    let total_power = self.total_voting_power().await?;
    if self.check_encoder_update_at_epoch(total_power).await? {
        tracing::info!("Encoder was updated during epoch transition");
    }
    
    let mut validators = self.validators.write().await;
        
       
        
        // Update each validator based on epoch performance
        for validator in validators.iter_mut() {
            // Unjail if jail time has passed
            if let Some(jail_end) = validator.jailed_until {
                if current_block >= jail_end {
                    validator.unjail();
                }
            }
            
            // Apply reputation decay
            validator.reputation *= 0.99; // 1% decay per epoch
            validator.reputation = validator.reputation.clamp(0.0, 1.0);
            
            // Update voting power
            validator.update_voting_power();
        }
        
        // Drop validators with too low stake or reputation
        validators.retain(|v| v.stake >= self.min_stake && v.reputation > 0.1);
        
        // Sort by voting power
        validators.sort_by(|a, b| b.voting_power.cmp(&a.voting_power));
        
        // Rebuild index
        let mut index = self.validator_index.write().await;
        index.clear();
        for (i, validator) in validators.iter().enumerate() {
            index.insert(validator.address, i);
        }
        
        // Recalculate powers
        drop(validators); // Release write lock
        self.recalculate_powers().await?;
        
        tracing::info!("Epoch transition completed, {} validators active", self.len().await);
        
        Ok(())
    }
}

/// Consensus errors related to voting
#[derive(Debug, thiserror::Error)]
pub enum ConsensusError {
    #[error("Validator error: {0}")]
    ValidatorError(String),
    
    #[error("Insufficient voting power")]
    InsufficientVotingPower,
    
    #[error("Quorum not reached")]
    QuorumNotReached,
    
    #[error("Invalid vote signature")]
    InvalidVoteSignature,
    
    #[error("Validator jailed")]
    ValidatorJailed,
    
    #[error("Validator not found")]
    ValidatorNotFound,
    
    #[error("Invalid validator set")]
    InvalidValidatorSet,

    #[error("Encoder update rejected: {0}")]
    EncoderUpdateRejected(String),
    
}
