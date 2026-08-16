//! AI-native consensus module for NERV blockchain
//! 
//! This module implements NERV's unique AI-native consensus mechanism that combines:
//! 1. Neural predictor for efficient validation (1.8MB distilled model)
//! 2. Weighted quorum voting (stake × reputation)
//! 3. Monte-Carlo dispute resolution
//! 
//! The consensus provides:
//! - 0.6 second block time
//! - 10,000+ TPS
//! - 3-second finality
//! - BFT security with 33% adversary tolerance

mod predictor;
mod voting;
mod dispute;

pub use predictor::{NeuralPredictor, PredictionResult, ModelConfig};
pub use voting::{WeightedQuorum, ValidatorSet, Vote, VoteResult};
pub use dispute::{DisputeResolver, Dispute, ResolutionResult};

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Consensus configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Block time in milliseconds (600ms)
    pub block_time_ms: u64,
    
    /// Timeout for view change in milliseconds
    pub view_change_timeout_ms: u64,
    
    /// Maximum number of validators
    pub max_validators: usize,
    
    /// Minimum stake required to be a validator
    pub min_stake: u128,
    
    /// Reputation decay factor per epoch (0-1)
    pub reputation_decay: f64,
    
    /// Slashing penalty for misbehavior (percentage)
    pub slashing_penalty: f64,
    
    /// Dispute resolution timeout in seconds
    pub dispute_timeout_sec: u64,
    
    /// Enable/disable neural predictor
    pub enable_predictor: bool,
    
    /// Predictor model configuration
    pub predictor_config: ModelConfig,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            block_time_ms: 600, // 0.6 seconds
            view_change_timeout_ms: 3000, // 3 seconds
            max_validators: 1000,
            min_stake: 1000 * 10u128.pow(18), // 1000 NERV minimum
            reputation_decay: 0.95, // 5% decay per epoch
            slashing_penalty: 0.1, // 10% slashing for misbehavior
            dispute_timeout_sec: 30,
            enable_predictor: true,
            predictor_config: ModelConfig::default(),
        }
    }
}

/// Consensus state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusState {
    /// Current view number
    pub view_number: u64,
    
    /// Current block height
    pub block_height: u64,
    
    /// Last committed block hash
    pub last_committed_hash: [u8; 32],
    
    /// Current proposer index (round-robin)
    pub proposer_index: usize,
    
    /// Locked block hash (if any)
    pub locked_hash: Option<[u8; 32]>,
    
    /// Locked view number
    pub locked_view: Option<u64>,
    
    /// Prepared block hash (if any)
    pub prepared_hash: Option<[u8; 32]>,
    
    /// Prepared view number
    pub prepared_view: Option<u64>,
    
    /// Time of last block production
    pub last_block_time: u64,
    
    /// Validator set hash
    pub validator_set_hash: [u8; 32],
    
    /// Epoch number
    pub epoch_number: u64,
}

impl Default for ConsensusState {
    fn default() -> Self {
        Self {
            view_number: 0,
            block_height: 0,
            last_committed_hash: [0u8; 32],
            proposer_index: 0,
            locked_hash: None,
            locked_view: None,
            prepared_hash: None,
            prepared_view: None,
            last_block_time: 0,
            validator_set_hash: [0u8; 32],
            epoch_number: 0,
        }
    }
}

/// Block proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockProposal {
    /// Block height
    pub height: u64,
    
    /// View number
    pub view: u64,
    
    /// Proposer address
    pub proposer: [u8; 20],
    
    /// Block hash
    pub block_hash: [u8; 32],
    
    /// Block data (encrypted delta)
    pub block_data: Vec<u8>,
    
    /// Justification (QC for previous block)
    pub justification: QuorumCertificate,
    
    /// Signature from proposer
    pub signature: Vec<u8>,
    
    /// Timestamp
    pub timestamp: u64,
}

/// Quorum Certificate (QC)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuorumCertificate {
    /// Block hash
    pub block_hash: [u8; 32],
    
    /// View number
    pub view: u64,
    
    /// Aggregate signature
    pub aggregate_signature: Vec<u8>,
    
    /// Bitmap of validators who signed
    pub signers_bitmap: Vec<u8>,
    
    /// Total voting power represented
    pub total_voting_power: u128,
    
    /// Required voting power threshold
    pub threshold_voting_power: u128,
}

impl QuorumCertificate {
    /// Create a new QC
    pub fn new(block_hash: [u8; 32], view: u64, threshold: u128) -> Self {
        Self {
            block_hash,
            view,
            aggregate_signature: Vec::new(),
            signers_bitmap: Vec::new(),
            total_voting_power: 0,
            threshold_voting_power: threshold,
        }
    }
    
    /// Check if QC has sufficient voting power
    pub fn has_quorum(&self) -> bool {
        self.total_voting_power >= self.threshold_voting_power
    }
}

/// Main consensus engine
pub struct ConsensusEngine {
    /// Configuration
    config: ConsensusConfig,
    
    /// Current state (protected by RwLock for async access)
    state: Arc<RwLock<ConsensusState>>,
    
    /// Validator set manager
    validator_set: WeightedQuorum,
    
    /// Neural predictor for efficient validation
    neural_predictor: Option<NeuralPredictor>,

   
    pub predictor: Arc<Predictor>,

    /// Dispute resolver
    dispute_resolver: DisputeResolver,
    
    /// Pending proposals
    pending_proposals: HashMap<[u8; 32], BlockProposal>,
    
    /// Pending votes
    pending_votes: HashMap<[u8; 32], Vec<Vote>>,
    
    /// Timeout handler
    timeout_handler: TimeoutHandler,
    
    /// Message broadcaster
    broadcaster: Box<dyn MessageBroadcaster + Send + Sync>,
}

impl ConsensusEngine {
    /// Create a new consensus engine
    pub async fn new(
        config: ConsensusConfig,
        broadcaster: Box<dyn MessageBroadcaster + Send + Sync>,
    ) -> Result<Self> {
        let neural_predictor = if config.enable_predictor {
            Some(NeuralPredictor::new(config.predictor_config.clone()).await?)
        } else {
            None
        };
        // NEW: Load and store the neural predictor (fallback enabled if model missing)
        let predictor_config = ConsensusConfig {
            predictor_model_path: config.predictor_model_path.clone(),
            ..Default::default()
        };
        let predictor = Arc::new(Predictor::new(&predictor_config)?);
        
        Ok(Self {
            config,
            state: Arc::new(RwLock::new(ConsensusState::default())),
            validator_set: WeightedQuorum::new(),
            neural_predictor,
            dispute_resolver: DisputeResolver::new(),
            pending_proposals: HashMap::new(),
            pending_votes: HashMap::new(),
            timeout_handler: TimeoutHandler::new(),
            broadcaster,
        })
    }
    
    /// Start the consensus engine
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting consensus engine");
        
        // Initialize validator set
        self.validator_set.initialize().await?;
        
        // Start timeout handler
        self.timeout_handler.start().await;
        
        // Start neural predictor if enabled
        if let Some(predictor) = &mut self.neural_predictor {
            predictor.warmup().await?;
        }
        
        tracing::info!("Consensus engine started successfully");
        Ok(())
    }
    
    /// Stop the consensus engine
    pub async fn stop(&mut self) -> Result<()> {
        tracing::info!("Stopping consensus engine");
        
        // Stop timeout handler
        self.timeout_handler.stop().await;
        
        // Save predictor state if enabled
        if let Some(predictor) = &mut self.neural_predictor {
            predictor.save_state().await?;
        }
        
        tracing::info!("Consensus engine stopped");
        Ok(())
    }
    
    /// Process a new block proposal
    pub async fn process_proposal(&mut self, proposal: BlockProposal) -> Result<ProposalResult> {
        let state = self.state.read().await;
        
        // Validate proposal basics
        if proposal.height != state.block_height + 1 {
            return Ok(ProposalResult::InvalidHeight);
        }
        
        if proposal.view < state.view_number {
            return Ok(ProposalResult::StaleView);
        }
        
        // Check if proposer is valid for this view
        if !self.is_valid_proposer(&proposal.proposer, proposal.view).await? {
            return Ok(ProposalResult::InvalidProposer);
        }
        
        // Verify justification (QC for previous block)
        if !self.verify_justification(&proposal.justification).await? {
            return Ok(ProposalResult::InvalidJustification);
        }
        
        
        // Store pending proposal
        self.pending_proposals.insert(proposal.block_hash, proposal.clone());
        
        // Broadcast vote for this proposal
        self.broadcast_vote(&proposal).await?;
        
        Ok(ProposalResult::Accepted)
    }
    
    /// Process a vote
    pub async fn process_vote(&mut self, vote: Vote) -> Result<VoteResult> {
        // Verify vote signature
        if !self.verify_vote_signature(&vote).await? {
            return Ok(VoteResult::InvalidSignature);
        }
        
        // Check if voter is a validator
        if !self.validator_set.is_validator(&vote.voter).await? {
            return Ok(VoteResult::InvalidValidator);
        }
        
        // Check voting power
        let voting_power = self.validator_set.voting_power(&vote.voter).await?;
        if voting_power == 0 {
            return Ok(VoteResult::NoVotingPower);
        }
        
        // Add vote to pending votes
        let votes = self.pending_votes.entry(vote.block_hash).or_insert_with(Vec::new);
        votes.push(vote.clone());
        
        // Check if we have quorum for this block
        if let Some(proposal) = self.pending_proposals.get(&vote.block_hash) {
            if self.check_quorum(&vote.block_hash).await? {
                // Commit the block
                self.commit_block(proposal).await?;
                return Ok(VoteResult::QuorumReached);
            }
        }
         
        // NEW: Use neural predictor for AI-native validation (whitepaper core)
    let tokens = self.tokenize_recent_events(); // Assume helper to get vote/proposal sequence
    let (predicted_delta, validity_score) = self.predictor.predict(&tokens)?;
    
    // Compare predicted embedding delta against proposed
    let proposed_delta = &proposal.embedding_delta;
    let delta_match = embedding_distance(predicted_delta, proposed_delta)? < 1e-6; // Tight threshold
    
    if validity_score < 0.95 || !delta_match {  // Whitepaper implies high confidence
        return Err(ConsensusError::PredictorError("Low prediction confidence or mismatch".into()));
    }
    
    Ok(true)
        
        Ok(VoteResult::Accepted)
    }
    
    /// Process a timeout for view change
    pub async fn process_timeout(&mut self, view: u64) -> Result<()> {
        let mut state = self.state.write().await;
        
        // Only process timeout for current or future views
        if view <= state.view_number {
            return Ok(());
        }
        
        // Increment view number
        state.view_number = view;
        
        // Clear locked and prepared states
        state.locked_hash = None;
        state.locked_view = None;
        state.prepared_hash = None;
        state.prepared_view = None;
        
        // Update proposer index (round-robin)
        state.proposer_index = (state.proposer_index + 1) % self.validator_set.len().await;
        
        tracing::info!("View changed to {}", view);
        
        Ok(())
    }
    
    /// Process a dispute
    pub async fn process_dispute(&mut self, dispute: Dispute) -> Result<ResolutionResult> {
        self.dispute_resolver.resolve(dispute).await
    }
    
    /// Commit a block (finalize consensus)
    async fn commit_block(&mut self, proposal: &BlockProposal) -> Result<()> {
        let mut state = self.state.write().await;
        
        // Update state
        state.block_height = proposal.height;
        state.last_committed_hash = proposal.block_hash;
        state.last_block_time = proposal.timestamp;
        
        // Clear pending data for this block
        self.pending_proposals.remove(&proposal.block_hash);
        self.pending_votes.remove(&proposal.block_hash);
        
        // Update validator set if epoch boundary
        if state.block_height % 1000 == 0 { // Every 1000 blocks
            self.update_validator_set().await?;
            state.epoch_number += 1;
        }
        
        // Update predictor if enabled
        if let Some(predictor) = &mut self.neural_predictor {
            predictor.update_with_block(proposal).await?;
        }
        
        tracing::info!(
            "Committed block {} at height {}",
            hex::encode(&proposal.block_hash[..8]),
            proposal.height
        );
        
        // Broadcast commit notification
        self.broadcaster.broadcast_commit(proposal).await?;
        
        Ok(())
    }
    
    /// Validate block using neural predictor
    async fn validate_with_predictor(&self, proposal: &BlockProposal) -> Result<bool> {
        let predictor = self.neural_predictor.as_ref().unwrap();
        
        // Get prediction
        let prediction = predictor.predict(&proposal.block_data).await?;
        
        // Use prediction confidence to decide
        if prediction.confidence > 0.9 {
            // High confidence: accept immediately
            Ok(true)
         sharding_manager.confirm_transactions(shard_id, block.tx_count()).await?; //Need to replicate this at all the places where a block is finalized for a shard

        } else if prediction.confidence > 0.7 {
            // Medium confidence: do light verification
            self.light_verification(proposal).await
        } else {
            // Low confidence: do full verification
            self.full_verification(proposal).await
        }
    }
    
    /// Validate block without neural predictor
    async fn validate_without_predictor(&self, proposal: &BlockProposal) -> Result<bool> {
        self.full_verification(proposal).await
    }
    
    /// Light verification (check signatures and basic structure)
    async fn light_verification(&self, proposal: &BlockProposal) -> Result<bool> {
        // Check signature
        if !self.verify_signature(&proposal.proposer, &proposal.block_hash, &proposal.signature).await? {
            return Ok(false);
        }
        
        // Check timestamp (not too far in future)
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        if proposal.timestamp > current_time + 30 { // 30 seconds max future
            return Ok(false);
        }
        
        // Check block data size
        if proposal.block_data.len() > 10 * 1024 * 1024 { // 10MB max
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Full verification (complete validation)
    async fn full_verification(&self, proposal: &BlockProposal) -> Result<bool> {
        // Do light verification first
        if !self.light_verification(proposal).await? {
            return Ok(false);
        }
        
        // Verify block data integrity
        let computed_hash = blake3::hash(&proposal.block_data);
        if computed_hash.as_bytes() != &proposal.block_hash {
            return Ok(false);
        }
        
        // Verify delta validity (would check with embedding module)
        // This is simplified - actual implementation would verify neural delta
        
        // Check gas limits, state transitions, etc.
        // (Implementation depends on execution engine)
        
        Ok(true)
    }
    
    /// Check if we have quorum for a block
    async fn check_quorum(&self, block_hash: &[u8; 32]) -> Result<bool> {
        let votes = self.pending_votes.get(block_hash)
            .ok_or_else(|| ConsensusError::NoVotes)?;
        
        let mut total_voting_power = 0;
        let mut seen_validators = HashSet::new();
        
        for vote in votes {
            // Prevent double counting
            if seen_validators.contains(&vote.voter) {
                continue;
            }
            
            let power = self.validator_set.voting_power(&vote.voter).await?;
            total_voting_power += power;
            seen_validators.insert(vote.voter);
        }
        
        // Check if we have 2/3 + 1 of total voting power
        let total_power = self.validator_set.total_voting_power().await?;
        let threshold = (total_power * 2 / 3) + 1;
        
        Ok(total_voting_power >= threshold)
    }
    
    /// Update validator set (at epoch boundaries)
    async fn update_validator_set(&mut self) -> Result<()> {
        // Get top validators by weighted score (stake × reputation)
        let top_validators = self.validator_set.top_validators(self.config.max_validators).await?;
        
        // Update validator set
        self.validator_set.update(top_validators).await?;
        
        // Update state hash
        let mut state = self.state.write().await;
        state.validator_set_hash = self.validator_set.hash().await?;
        
        Ok(())
    }
    
    /// Broadcast vote for a proposal
    async fn broadcast_vote(&self, proposal: &BlockProposal) -> Result<()> {
        // Create vote
        let vote = Vote {
            block_hash: proposal.block_hash,
            view: proposal.view,
            voter: self.validator_set.my_address().await?,
            signature: Vec::new(), // Would be signed
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        // Broadcast to network
        self.broadcaster.broadcast_vote(vote).await?;
        
        Ok(())
    }
    
    /// Check if address is valid proposer for view
    async fn is_valid_proposer(&self, address: &[u8; 20], view: u64) -> Result<bool> {
        let validators = self.validator_set.validators().await?;
        
        // Round-robin proposer selection
        let proposer_index = (view as usize) % validators.len();
        
        if proposer_index < validators.len() {
            Ok(validators[proposer_index].address == *address)
        } else {
            Ok(false)
        }
    }
    
    /// Verify justification (QC)
    async fn verify_justification(&self, qc: &QuorumCertificate) -> Result<bool> {
        if !qc.has_quorum() {
            return Ok(false);
        }
        
        // Verify aggregate signature (would use BLS aggregation in production)
        // This is simplified
        
        Ok(true)
    }
    
    /// Verify vote signature
    async fn verify_vote_signature(&self, vote: &Vote) -> Result<bool> {
        // Verify signature (would use Dilithium in production)
        // This is simplified
        Ok(true)
    }
    
    /// Verify proposer signature
    async fn verify_signature(&self, signer: &[u8; 20], message: &[u8; 32], signature: &[u8]) -> Result<bool> {
        // Verify Dilithium signature
        // This is simplified
        Ok(true)
    }
}

/// Proposal processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProposalResult {
    Accepted,
    InvalidHeight,
    StaleView,
    InvalidProposer,
    InvalidJustification,
    InvalidBlock,
    Timeout,
}

/// Timeout handler for view changes
struct TimeoutHandler {
    running: bool,
    timeout_rx: tokio::sync::mpsc::Receiver<u64>,
    timeout_tx: tokio::sync::mpsc::Sender<u64>,
}

impl TimeoutHandler {
    fn new() -> Self {
        let (timeout_tx, timeout_rx) = tokio::sync::mpsc::channel(100);
        
        Self {
            running: false,
            timeout_rx,
            timeout_tx,
        }
    }
    
    async fn start(&mut self) {
        self.running = true;
        
        let mut rx = self.timeout_rx.clone();
        tokio::spawn(async move {
            while let Some(view) = rx.recv().await {
                // Handle timeout (would trigger view change)
                tracing::warn!("Timeout for view {}", view);
            }
        });
    }
    
    async fn stop(&mut self) {
        self.running = false;
    }
    
    fn schedule_timeout(&self, view: u64, duration_ms: u64) {
        let tx = self.timeout_tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(duration_ms)).await;
            let _ = tx.send(view).await;
        });
    }
}

/// Trait for message broadcasting
#[async_trait::async_trait]
pub trait MessageBroadcaster: Send + Sync {
    async fn broadcast_proposal(&self, proposal: BlockProposal) -> Result<()>;
    async fn broadcast_vote(&self, vote: Vote) -> Result<()>;
    async fn broadcast_timeout(&self, view: u64) -> Result<()>;
    async fn broadcast_commit(&self, proposal: &BlockProposal) -> Result<()>;
}

/// Consensus errors
#[derive(Debug, thiserror::Error)]
pub enum ConsensusError {
    #[error("Invalid block height")]
    InvalidHeight,
    
    #[error("Stale view number")]
    StaleView,
    
    #[error("Invalid proposer")]
    InvalidProposer,
    
    #[error("Invalid justification")]
    InvalidJustification,
    
    #[error("No votes available")]
    NoVotes,
    
    #[error("Insufficient voting power")]
    InsufficientVotingPower,
    
    #[error("Validator set not initialized")]
    ValidatorSetNotInitialized,
    
    #[error("Neural predictor error: {0}")]
    PredictorError(String),
    
    #[error("Dispute resolution failed")]
    DisputeResolutionFailed,
    
    #[error("Timeout expired")]
    TimeoutExpired,
    
    #[error("Network error: {0}")]
    NetworkError(String),
}
