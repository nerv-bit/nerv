//! Monte-Carlo dispute resolution system
//! 
//! This module implements NERV's dispute resolution mechanism that uses
//! Monte-Carlo simulations to resolve conflicts between validators.
//! 
//! Features:
//! - Random sampling of validators for dispute committees
//! - Cryptographic sortition for committee selection
//! - Game-theoretic incentives for honest behavior
//! - Slashing for provable misbehavior
//! - Fast resolution (<30 seconds) with high confidence

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Dispute type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DisputeType {
    /// Invalid block proposal
    InvalidBlock,
    
    /// Double signing (equivocation)
    DoubleSign,
    
    /// Censorship (withholding votes)
    Censorship,
    
    /// Invalid state transition
    InvalidState,
    
    /// Data availability failure
    DataUnavailable,
    
    /// Incorrect computation
    ComputationError,
    
    /// Governance violation
    GovernanceViolation,
}

/// Dispute evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisputeEvidence {
    /// Evidence type
    pub evidence_type: EvidenceType,
    
    /// Raw evidence data
    pub data: Vec<u8>,
    
    /// Cryptographic proof
    pub proof: Vec<u8>,
    
    /// Witness signatures (if any)
    pub witness_signatures: Vec<Vec<u8>>,
    
    /// Timestamp
    pub timestamp: u64,
    
    /// Block height in question
    pub block_height: u64,
    
    /// Block hash in question
    pub block_hash: [u8; 32],
}

/// Evidence type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Merkle proof
    MerkleProof,
    
    /// Cryptographic signature
    Signature,
    
    /// Zero-knowledge proof
    ZkProof,
    
    /// Statistical proof
    Statistical,
    
    /// Witness testimony
    Witness,
    
    /// Data availability proof
    DataAvailability,
}

/// Dispute structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dispute {
    /// Unique dispute ID
    pub dispute_id: u64,
    
    /// Accused validator address
    pub accused_validator: [u8; 20],
    
    /// Accuser address
    pub accuser: [u8; 20],
    
    /// Dispute type
    pub dispute_type: DisputeType,
    
    /// Evidence
    pub evidence: DisputeEvidence,
    
    /// Stake deposited by accuser
    pub accuser_stake: u128,
    
    /// Stake deposited by accused (auto-slashable)
    pub accused_stake: u128,
    
    /// Resolution status
    pub status: DisputeStatus,
    
    /// Committee members
    pub committee: Option<Vec<[u8; 20]>>,
    
    /// Committee votes
    pub votes: Option<HashMap<[u8; 20], Vote>>,
    
    /// Resolution result
    pub result: Option<ResolutionResult>,
    
    /// Timestamp when dispute was created
    pub created_at: u64,
    
    /// Timeout for resolution (seconds)
    pub timeout_sec: u64,
}

/// Dispute status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisputeStatus {
    /// Dispute created, waiting for committee selection
    Pending,
    
    /// Committee selected, waiting for votes
    Active,
    
    /// Votes collected, resolving
    Resolving,
    
    /// Dispute resolved
    Resolved,
    
    /// Dispute timeout expired
    Timeout,
    
    /// Dispute cancelled
    Cancelled,
}

/// Committee vote
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    /// Voter address
    pub voter: [u8; 20],
    
    /// Vote (true = guilty, false = innocent)
    pub vote: bool,
    
    /// Confidence level (0-1)
    pub confidence: f64,
    
    /// Justification
    pub justification: String,
    
    /// Signature
    pub signature: Vec<u8>,
    
    /// Timestamp
    pub timestamp: u64,
}

/// Resolution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionResult {
    /// Whether accused was found guilty
    pub guilty: bool,
    
    /// Slashing percentage (0-1) if guilty
    pub slashing_percentage: f64,
    
    /// Jail duration in blocks (if guilty)
    pub jail_duration: Option<u64>,
    
    /// Reward for accuser (if accusation was correct)
    pub accuser_reward: u128,
    
    /// Penalty for false accusation (if accusation was wrong)
    pub accuser_penalty: u128,
    
    /// Committee rewards
    pub committee_rewards: HashMap<[u8; 20], u128>,
    
    /// Final decision hash
    pub decision_hash: [u8; 32],
    
    /// Resolution timestamp
    pub resolved_at: u64,
    
    /// Number of votes for guilty
    pub guilty_votes: usize,
    
    /// Number of votes for innocent
    pub innocent_votes: usize,
    
    /// Total voting power in committee
    pub total_voting_power: u128,
}

/// Dispute resolver
pub struct DisputeResolver {
    /// Active disputes
    active_disputes: Arc<RwLock<HashMap<u64, Dispute>>>,
    
    /// Resolved disputes
    resolved_disputes: Arc<RwLock<HashMap<u64, Dispute>>>,
    
    /// Committee size for disputes
    committee_size: usize,
    
    /// Minimum stake to participate in committee
    min_committee_stake: u128,
    
    /// Timeout for dispute resolution (seconds)
    resolution_timeout: u64,
    
    /// Slashing percentage for guilty verdict
    slashing_percentage: f64,
    
    /// Reward percentage for accuser
    accuser_reward_percentage: f64,
    
    /// Committee reward percentage
    committee_reward_percentage: f64,
    
    /// Message channel for dispute notifications
    notification_tx: mpsc::Sender<DisputeNotification>,
    notification_rx: mpsc::Receiver<DisputeNotification>,
    
    /// RNG for committee selection
    rng: Arc<RwLock<StdRng>>,
}

impl DisputeResolver {
    /// Create a new dispute resolver
    pub fn new() -> Self {
        let (notification_tx, notification_rx) = mpsc::channel(100);
        
        Self {
            active_disputes: Arc::new(RwLock::new(HashMap::new())),
            resolved_disputes: Arc::new(RwLock::new(HashMap::new())),
            committee_size: 21, // 21 members for statistical significance
            min_committee_stake: 1000 * 10u128.pow(18), // 1000 NERV
            resolution_timeout: 30, // 30 seconds
            slashing_percentage: 0.1, // 10% slashing
            accuser_reward_percentage: 0.1, // 10% of slashed amount
            committee_reward_percentage: 0.1, // 10% of slashed amount
            notification_tx,
            notification_rx,
            rng: Arc::new(RwLock::new(StdRng::from_entropy())),
        }
    }
    
    /// Start dispute resolution process
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting dispute resolver");
        
        // Start notification handler
        let mut rx = self.notification_rx.clone();
        let active_disputes = self.active_disputes.clone();
        let resolved_disputes = self.resolved_disputes.clone();
        
        tokio::spawn(async move {
            while let Some(notification) = rx.recv().await {
                match notification {
                    DisputeNotification::NewDispute(dispute_id) => {
                        tracing::info!("New dispute created: {}", dispute_id);
                    }
                    DisputeNotification::VoteReceived(dispute_id, voter) => {
                        tracing::debug!("Vote received for dispute {} from {}", dispute_id, hex::encode(&voter[..4]));
                    }
                    DisputeNotification::DisputeResolved(dispute_id, result) => {
                        tracing::info!("Dispute {} resolved: guilty = {}", dispute_id, result.guilty);
                        
                        // Move from active to resolved
                        let mut active = active_disputes.write().await;
                        let mut resolved = resolved_disputes.write().await;
                        
                        if let Some(dispute) = active.remove(&dispute_id) {
                            resolved.insert(dispute_id, dispute);
                        }
                    }
                    DisputeNotification::Timeout(dispute_id) => {
                        tracing::warn!("Dispute {} timeout", dispute_id);
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Create a new dispute
    pub async fn create_dispute(
        &mut self,
        accused_validator: [u8; 20],
        accuser: [u8; 20],
        dispute_type: DisputeType,
        evidence: DisputeEvidence,
        accuser_stake: u128,
    ) -> Result<u64> {
        // Generate dispute ID
        let dispute_id = self.generate_dispute_id().await;
        
        // Create dispute
        let dispute = Dispute {
            dispute_id,
            accused_validator,
            accuser,
            dispute_type,
            evidence,
            accuser_stake,
            accused_stake: 0, // Will be set when accused responds
            status: DisputeStatus::Pending,
            committee: None,
            votes: None,
            result: None,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            timeout_sec: self.resolution_timeout,
        };
        
        // Store dispute
        self.active_disputes.write().await.insert(dispute_id, dispute);
        
        // Notify
        self.notification_tx.send(DisputeNotification::NewDispute(dispute_id)).await
            .map_err(|e| ConsensusError::DisputeError(format!("Failed to send notification: {}", e)))?;
        
        // Start committee selection
        self.select_committee(dispute_id).await?;
        
        Ok(dispute_id)
    }
    
    /// Resolve a dispute using Monte-Carlo simulation
    pub async fn resolve(&mut self, dispute: Dispute) -> Result<ResolutionResult> {
        let dispute_id = dispute.dispute_id;
        
        tracing::info!("Resolving dispute {} with Monte-Carlo simulation", dispute_id);
        
        // Check if dispute is already resolved
        if dispute.status == DisputeStatus::Resolved {
            if let Some(result) = dispute.result {
                return Ok(result);
            }
        }
        
        // Update status
        let mut disputes = self.active_disputes.write().await;
        if let Some(active_dispute) = disputes.get_mut(&dispute_id) {
            active_dispute.status = DisputeStatus::Resolving;
        }
        
        // Run Monte-Carlo simulation to determine truth
        let simulation_result = self.run_monte_carlo_simulation(&dispute).await?;
        
        // Collect and verify votes
        let votes = self.collect_votes(&dispute).await?;
        
        // Determine verdict based on votes and simulation
        let verdict = self.determine_verdict(&votes, simulation_result).await?;
        
        // Calculate rewards and penalties
        let result = self.calculate_resolution(&dispute, &verdict, &votes).await?;
        
        // Update dispute with result
        if let Some(active_dispute) = disputes.get_mut(&dispute_id) {
            active_dispute.result = Some(result.clone());
            active_dispute.status = DisputeStatus::Resolved;
        }
        
        // Notify
        self.notification_tx.send(DisputeNotification::DisputeResolved(dispute_id, result.clone())).await
            .map_err(|e| ConsensusError::DisputeError(format!("Failed to send notification: {}", e)))?;
        
        Ok(result)
    }
    
    /// Submit a vote for a dispute
    pub async fn submit_vote(
        &mut self,
        dispute_id: u64,
        voter: [u8; 20],
        vote: bool,
        confidence: f64,
        justification: String,
        signature: Vec<u8>,
    ) -> Result<()> {
        let mut disputes = self.active_disputes.write().await;
        
        let dispute = disputes.get_mut(&dispute_id)
            .ok_or_else(|| ConsensusError::DisputeError("Dispute not found".to_string()))?;
        
        // Check if dispute is active
        if dispute.status != DisputeStatus::Active {
            return Err(ConsensusError::DisputeError("Dispute not active".to_string()).into());
        }
        
        // Check if voter is in committee
        let committee = dispute.committee.as_ref()
            .ok_or_else(|| ConsensusError::DisputeError("Committee not selected".to_string()))?;
        
        if !committee.contains(&voter) {
            return Err(ConsensusError::DisputeError("Voter not in committee".to_string()).into());
        }
        
        // Check if already voted
        if let Some(votes) = &dispute.votes {
            if votes.contains_key(&voter) {
                return Err(ConsensusError::DisputeError("Already voted".to_string()).into());
            }
        }
        
        // Create vote
        let vote_struct = Vote {
            voter,
            vote,
            confidence: confidence.clamp(0.0, 1.0),
            justification,
            signature,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        // Add vote
        let votes = dispute.votes.get_or_insert_with(HashMap::new);
        votes.insert(voter, vote_struct.clone());
        
        // Check if we have all votes
        if votes.len() == committee.len() {
            dispute.status = DisputeStatus::Resolving;
        }
        
        // Notify
        self.notification_tx.send(DisputeNotification::VoteReceived(dispute_id, voter)).await
            .map_err(|e| ConsensusError::DisputeError(format!("Failed to send notification: {}", e)))?;
        
        Ok(())
    }
    
    /// Get dispute by ID
    pub async fn get_dispute(&self, dispute_id: u64) -> Result<Option<Dispute>> {
        let disputes = self.active_disputes.read().await;
        Ok(disputes.get(&dispute_id).cloned())
    }
    
    /// Get active disputes
    pub async fn get_active_disputes(&self) -> Result<Vec<Dispute>> {
        let disputes = self.active_disputes.read().await;
        Ok(disputes.values().cloned().collect())
    }
    
    /// Get resolved disputes
    pub async fn get_resolved_disputes(&self) -> Result<Vec<Dispute>> {
        let disputes = self.resolved_disputes.read().await;
        Ok(disputes.values().cloned().collect())
    }
    
    /// Check timeout for disputes
    pub async fn check_timeouts(&mut self) -> Result<Vec<u64>> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let mut timed_out = Vec::new();
        let mut disputes = self.active_disputes.write().await;
        
        for (dispute_id, dispute) in disputes.iter_mut() {
            if dispute.status != DisputeStatus::Resolved &&
               current_time > dispute.created_at + dispute.timeout_sec {
                
                dispute.status = DisputeStatus::Timeout;
                timed_out.push(*dispute_id);
                
                // Notify
                self.notification_tx.send(DisputeNotification::Timeout(*dispute_id)).await
                    .map_err(|e| ConsensusError::DisputeError(format!("Failed to send notification: {}", e)))?;
            }
        }
        
        Ok(timed_out)
    }
    
    // Private methods
    
    async fn generate_dispute_id(&self) -> u64 {
        let mut rng = self.rng.write().await;
        rng.gen()
    }
    
    async fn select_committee(&mut self, dispute_id: u64) -> Result<()> {
        // In production: select committee from validator set using cryptographic sortition
        // This is simplified
        
        let mut disputes = self.active_disputes.write().await;
        let dispute = disputes.get_mut(&dispute_id)
            .ok_or_else(|| ConsensusError::DisputeError("Dispute not found".to_string()))?;
        
        // Generate random committee (simplified)
        let mut rng = self.rng.write().await;
        let mut committee = Vec::new();
        
        for _ in 0..self.committee_size {
            let mut member = [0u8; 20];
            rng.fill(&mut member);
            committee.push(member);
        }
        
        dispute.committee = Some(committee);
        dispute.status = DisputeStatus::Active;
        
        Ok(())
    }
    
    async fn run_monte_carlo_simulation(&self, dispute: &Dispute) -> Result<MonteCarloResult> {
        tracing::info!("Running Monte-Carlo simulation for dispute {}", dispute.dispute_id);
        
        let mut rng = self.rng.write().await;
        
        // Number of simulations
        let num_simulations = 1000;
        let mut guilty_count = 0;
        let mut innocent_count = 0;
        let mut confidence_scores = Vec::new();
        
        for _ in 0..num_simulations {
            // Simulate random sampling of evidence
            let evidence_strength = self.evaluate_evidence(&dispute.evidence).await?;
            
            // Add random noise (simulating imperfect information)
            let noise: f64 = rng.gen_range(-0.2..0.2);
            let adjusted_strength = (evidence_strength + noise).clamp(0.0, 1.0);
            
            // Determine verdict for this simulation
            if adjusted_strength > 0.7 {
                guilty_count += 1;
            } else if adjusted_strength < 0.3 {
                innocent_count += 1;
            }
            // Middle range is uncertain
            
            confidence_scores.push(adjusted_strength);
        }
        
        // Calculate statistics
        let guilty_ratio = guilty_count as f64 / num_simulations as f64;
        let innocent_ratio = innocent_count as f64 / num_simulations as f64;
        let uncertain_ratio = 1.0 - guilty_ratio - innocent_ratio;
        
        let mean_confidence: f64 = confidence_scores.iter().sum::<f64>() / confidence_scores.len() as f64;
        let variance: f64 = confidence_scores.iter()
            .map(|&x| (x - mean_confidence).powi(2))
            .sum::<f64>() / confidence_scores.len() as f64;
        
        Ok(MonteCarloResult {
            guilty_ratio,
            innocent_ratio,
            uncertain_ratio,
            mean_confidence,
            confidence_variance: variance,
            simulation_count: num_simulations,
        })
    }
    
    async fn evaluate_evidence(&self, evidence: &DisputeEvidence) -> Result<f64> {
        // Evaluate evidence strength (0-1)
        // This is simplified - in production would verify cryptographic proofs
        
        match evidence.evidence_type {
            EvidenceType::MerkleProof => {
                // Merkle proofs are strong evidence
                Ok(0.8)
            }
            EvidenceType::Signature => {
                // Cryptographic signatures are very strong
                Ok(0.9)
            }
            EvidenceType::ZkProof => {
                // Zero-knowledge proofs are strongest
                Ok(1.0)
            }
            EvidenceType::Statistical => {
                // Statistical evidence is weaker
                Ok(0.6)
            }
            EvidenceType::Witness => {
                // Witness testimony is moderate
                Ok(0.5)
            }
            EvidenceType::DataAvailability => {
                // Data availability proofs are strong
                Ok(0.85)
            }
        }
    }
    
    async fn collect_votes(&self, dispute: &Dispute) -> Result<Vec<Vote>> {
        let votes = dispute.votes.as_ref()
            .ok_or_else(|| ConsensusError::DisputeError("No votes collected".to_string()))?;
        
        Ok(votes.values().cloned().collect())
    }
    
    async fn determine_verdict(
        &self,
        votes: &[Vote],
        simulation: MonteCarloResult,
    ) -> Result<Verdict> {
        // Count votes
        let mut guilty_votes = 0;
        let mut innocent_votes = 0;
        let mut total_confidence = 0.0;
        
        for vote in votes {
            if vote.vote {
                guilty_votes += 1;
            } else {
                innocent_votes += 1;
            }
            total_confidence += vote.confidence;
        }
        
        let avg_confidence = total_confidence / votes.len() as f64;
        
        // Combine committee votes with Monte-Carlo simulation
        let vote_ratio = if votes.is_empty() {
            0.5
        } else {
            guilty_votes as f64 / votes.len() as f64
        };
        
        // Weighted combination: 70% committee votes, 30% simulation
        let combined_guilt = 0.7 * vote_ratio + 0.3 * simulation.guilty_ratio;
        
        // Determine verdict
        if combined_guilt > 0.66 { // 2/3 threshold
            Ok(Verdict::Guilty(combined_guilt))
        } else if combined_guilt < 0.33 {
            Ok(Verdict::Innocent(1.0 - combined_guilt))
        } else {
            Ok(Verdict::Uncertain(combined_guilt))
        }
    }
    
    async fn calculate_resolution(
        &self,
        dispute: &Dispute,
        verdict: &Verdict,
        votes: &[Vote],
    ) -> Result<ResolutionResult> {
        let (guilty, confidence) = match verdict {
            Verdict::Guilty(c) => (true, *c),
            Verdict::Innocent(c) => (false, *c),
            Verdict::Uncertain(c) => {
                // For uncertain cases, favor accused (presumption of innocence)
                (false, *c)
            }
        };
        
        // Calculate slashing if guilty
        let slashing_percentage = if guilty {
            self.slashing_percentage * confidence.min(1.0)
        } else {
            0.0
        };
        
        // Calculate rewards
        let accuser_reward = if guilty {
            // Accuser gets percentage of slashed amount
            (dispute.accused_stake as f64 * slashing_percentage * self.accuser_reward_percentage) as u128
        } else {
            0
        };
        
        let accuser_penalty = if !guilty {
            // False accusation penalty
            dispute.accuser_stake / 10 // 10% penalty
        } else {
            0
        };
        
        // Calculate committee rewards
        let mut committee_rewards = HashMap::new();
        let total_committee_reward = if guilty {
            (dispute.accused_stake as f64 * slashing_percentage * self.committee_reward_percentage) as u128
        } else {
            0
        };
        
        if total_committee_reward > 0 && !votes.is_empty() {
            let reward_per_voter = total_committee_reward / votes.len() as u128;
            for vote in votes {
                // Reward voters who voted with majority
                if vote.vote == guilty {
                    committee_rewards.insert(vote.voter, reward_per_voter);
                }
            }
        }
        
        // Create result
        let result = ResolutionResult {
            guilty,
            slashing_percentage,
            jail_duration: if guilty { Some(1000) } else { None }, // 1000 blocks if guilty
            accuser_reward,
            accuser_penalty,
            committee_rewards,
            decision_hash: self.calculate_decision_hash(dispute, verdict).await?,
            resolved_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            guilty_votes: votes.iter().filter(|v| v.vote).count(),
            innocent_votes: votes.iter().filter(|v| !v.vote).count(),
            total_voting_power: votes.len() as u128 * 100, // Simplified
        };
        
        Ok(result)
    }
    
    async fn calculate_decision_hash(&self, dispute: &Dispute, verdict: &Verdict) -> Result<[u8; 32]> {
        let mut hasher = blake3::Hasher::new();
        
        hasher.update(&dispute.dispute_id.to_be_bytes());
        hasher.update(&dispute.accused_validator);
        hasher.update(&dispute.accuser);
        
        match verdict {
            Verdict::Guilty(c) => {
                hasher.update(b"guilty");
                hasher.update(&c.to_be_bytes());
            }
            Verdict::Innocent(c) => {
                hasher.update(b"innocent");
                hasher.update(&c.to_be_bytes());
            }
            Verdict::Uncertain(c) => {
                hasher.update(b"uncertain");
                hasher.update(&c.to_be_bytes());
            }
        }
        
        Ok(*hasher.finalize().as_bytes())
    }
}

/// Monte-Carlo simulation result
#[derive(Debug, Clone)]
struct MonteCarloResult {
    guilty_ratio: f64,
    innocent_ratio: f64,
    uncertain_ratio: f64,
    mean_confidence: f64,
    confidence_variance: f64,
    simulation_count: usize,
}

/// Verdict with confidence
#[derive(Debug, Clone)]
enum Verdict {
    Guilty(f64),    // Confidence level
    Innocent(f64),  // Confidence level
    Uncertain(f64), // Probability of guilt
}

/// Dispute notification
#[derive(Debug)]
enum DisputeNotification {
    NewDispute(u64),
    VoteReceived(u64, [u8; 20]),
    DisputeResolved(u64, ResolutionResult),
    Timeout(u64),
}

/// Dispute-related errors
#[derive(Debug, thiserror::Error)]
pub enum ConsensusError {
    #[error("Dispute error: {0}")]
    DisputeError(String),
    
    #[error("Invalid evidence")]
    InvalidEvidence,
    
    #[error("Committee selection failed")]
    CommitteeSelectionFailed,
    
    #[error("Vote verification failed")]
    VoteVerificationFailed,
    
    #[error("Resolution timeout")]
    ResolutionTimeout,
    
    #[error("Insufficient stake")]
    InsufficientStake,
    
    #[error("Double dispute detected")]
    DoubleDispute,
}
