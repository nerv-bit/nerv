//! Reward calculation and distribution
//! Implements NERV's useful-work economy with Shapley-value incentives
//! 
//! Reward Distribution (per block):
//! 1. 60% - Gradient Contributions (useful work)
//! 2. 30% - Validation & Consensus (stake × reputation)
//! 3. 10% - Public Goods Funding (community governance)
//! 
//! All rewards are distributed proportionally based on verifiable contributions.


use crate::Result;
use crate::params::{
    REWARD_GRADIENT_PERCENT, REWARD_VALIDATION_PERCENT, REWARD_PUBLIC_GOODS_PERCENT,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;


/// Gradient reward for useful-work contributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientReward {
    /// Contributor address
    pub address: [u8; 20],
    
    /// Shapley value contribution (0-1 normalized)
    pub shapley_value: f64,
    
    /// Gradient quality score (0-1)
    pub quality_score: f64,
    
    /// Differential privacy epsilon used
    pub dp_epsilon: f64,
    
    /// Timestamp of contribution
    pub timestamp: u64,
    
    /// Gradient vector hash (for deduplication)
    pub gradient_hash: [u8; 32],
    
    /// Model identifier
    pub model_id: u64,
    
    /// Contribution round
    pub round: u64,
}


/// Validation reward for honest consensus participation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReward {
    /// Validator address
    pub address: [u8; 20],
    
    /// Staked amount (wei)
    pub stake: u128,
    
    /// Reputation score (0-1)
    pub reputation_score: f64,
    
    /// Uptime percentage (0-1)
    pub uptime: f64,
    
    /// Number of votes participated in
    pub votes_count: u64,
    
    /// Slashing history adjustment (negative for penalties)
    pub slashing_adjustment: i64,
    
    /// Block production count
    pub blocks_produced: u64,
    
    /// Latency score (0-1, higher is better)
    pub latency_score: f64,
}


impl ValidationReward {
    /// Calculate weight for reward distribution
    pub fn weight(&self) -> f64 {
        // Weight = stake × reputation × uptime × latency
        // Normalize stake to avoid huge numbers
        let normalized_stake = (self.stake as f64).ln_1p(); // log(1 + stake) for diminishing returns
        
        let mut weight = normalized_stake * 
                        self.reputation_score * 
                        self.uptime * 
                        self.latency_score;
        
        // Apply slashing adjustment (reduce weight for penalties)
        if self.slashing_adjustment < 0 {
            let penalty_factor = 1.0 - (self.slashing_adjustment.abs() as f64 / 100.0).min(0.5);
            weight *= penalty_factor;
        }
        
        weight.max(0.0)
    }
}


/// Public goods funding proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicGoodsProposal {
    /// Proposal ID
    pub proposal_id: u64,
    
    /// Recipient address
    pub recipient: [u8; 20],
    
    /// Amount requested (wei)
    pub requested_amount: u128,
    
    /// Amount actually funded
    pub funded_amount: u128,
    
    /// Description of the public good
    pub description: String,
    
    /// Community vote score (0-1)
    pub vote_score: f64,
    
    /// Technical review score (0-1)
    pub technical_score: f64,
    
    /// Impact multiplier (1-5)
    pub impact_multiplier: u8,
    
    /// Category
    pub category: ProposalCategory,
    
    /// Duration (months)
    pub duration_months: u64,
    
    /// KYC/verification status
    pub verified: bool,
    
    /// Completion milestones
    pub milestones: Vec<Milestone>,
}


/// Proposal category for public goods funding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalCategory {
    Research,
    Development,
    Education,
    Community,
    Security,
    Infrastructure,
    Privacy,
    Interoperability,
    Other,
}


impl ProposalCategory {
    /// Get priority multiplier for category
    pub fn priority_multiplier(&self) -> f64 {
        match self {
            Self::Security | Self::Privacy => 1.5,
            Self::Research | Self::Development => 1.3,
            Self::Infrastructure => 1.2,
            Self::Education | Self::Community => 1.1,
            Self::Interoperability => 1.1,
            Self::Other => 1.0,
        }
    }
}


/// Milestone for public goods proposals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Milestone {
    /// Milestone number
    pub number: u8,
    
    /// Description
    pub description: String,
    
    /// Completion criteria
    pub criteria: String,
    
    /// Funding percentage released on completion
    pub funding_percentage: u8,
    
    /// Deadline timestamp
    pub deadline: u64,
    
    /// Completion status
    pub completed: bool,
    
    /// Completion proof (hash of deliverables)
    pub completion_proof: Option<[u8; 32]>,
}


/// Reward calculator
#[derive(Debug, Clone)]
pub struct RewardCalculator {
    /// Gradient reward percentage (60%)
    gradient_percent: f64,
    
    /// Validation reward percentage (30%)
    validation_percent: f64,
    
    /// Public goods percentage (10%)
    public_goods_percent: f64,
    
    /// Minimum Shapley value threshold
    min_shapley_threshold: f64,
    
    /// Maximum reward per gradient contribution
    max_gradient_reward: u128,
    
    /// Reputation decay rate per epoch
    reputation_decay_rate: f64,
    
    /// Quality score weight in Shapley calculation
    quality_weight: f64,
    
    /// Privacy bonus multiplier
    privacy_bonus_multiplier: f64,
}


impl Default for RewardCalculator {
    fn default() -> Self {
        Self {
            gradient_percent: REWARD_GRADIENT_PERCENT,
            validation_percent: REWARD_VALIDATION_PERCENT,
            public_goods_percent: REWARD_PUBLIC_GOODS_PERCENT,
            min_shapley_threshold: 0.01, // 1% minimum contribution
            max_gradient_reward: 100_000 * 10u128.pow(18), // 100k NERV max per contribution
            reputation_decay_rate: 0.95, // 5% decay per epoch
            quality_weight: 0.3,
            privacy_bonus_multiplier: 1.2,
        }
    }
}


impl RewardCalculator {
    /// Create a new reward calculator with custom parameters
    pub fn new(
        gradient_percent: f64,
        validation_percent: f64,
        public_goods_percent: f64,
    ) -> Result<Self> {
        // Validate percentages sum to 100%
        let total = gradient_percent + validation_percent + public_goods_percent;
        if (total - 1.0).abs() > 0.001 {
            return Err(RewardError::InvalidDistribution(
                gradient_percent, validation_percent, public_goods_percent
            ).into());
        }
        
        Ok(Self {
            gradient_percent,
            validation_percent,
            public_goods_percent,
            ..Default::default()
        })
    }
    
    /// Calculate gradient rewards for a round
    pub fn calculate_gradient_rewards(
        &self,
        contributions: &[GradientReward],
        total_reward_pool: u128,
    ) -> Vec<([u8; 20], u128)> {
        if contributions.is_empty() {
            return Vec::new();
        }
        
        // Filter out contributions below threshold
        let valid_contributions: Vec<_> = contributions
            .iter()
            .filter(|c| c.shapley_value >= self.min_shapley_threshold)
            .collect();
        
        if valid_contributions.is_empty() {
            return Vec::new();
        }
        
        // Calculate adjusted Shapley values with quality and privacy bonuses
        let mut adjusted_values = Vec::new();
        let mut total_adjusted = 0.0;
        
        for contribution in &valid_contributions {
            let mut adjusted = contribution.shapley_value;
            
            // Apply quality multiplier
            adjusted *= 1.0 + (contribution.quality_score - 0.5) * self.quality_weight;
            
            // Apply privacy bonus for strong differential privacy
            if contribution.dp_epsilon <= 1.0 {
                adjusted *= self.privacy_bonus_multiplier;
            }
            
            adjusted_values.push(adjusted);
            total_adjusted += adjusted;
        }
        
        // Normalize and distribute rewards
        let mut rewards = Vec::new();
        
        for (i, contribution) in valid_contributions.iter().enumerate() {
            let share = adjusted_values[i] / total_adjusted;
            let mut reward = (total_reward_pool as f64 * share) as u128;
            
            // Cap individual reward
            reward = reward.min(self.max_gradient_reward);
            
            rewards.push((contribution.address, reward));
        }
        
        rewards
    }
    
    /// Calculate validation rewards for an epoch
    pub fn calculate_validation_rewards(
        &self,
        validators: &[ValidationReward],
        total_reward_pool: u128,
    ) -> Vec<([u8; 20], u128)> {
        if validators.is_empty() {
            return Vec::new();
        }
        
        // Calculate weights for all validators
        let weights: Vec<f64> = validators.iter()
            .map(|v| v.weight())
            .collect();
        
        let total_weight: f64 = weights.iter().sum();
        
        if total_weight == 0.0 {
            return Vec::new();
        }
        
        // Distribute rewards proportionally to weights
        let mut rewards = Vec::new();
        
        for (i, validator) in validators.iter().enumerate() {
            let share = weights[i] / total_weight;
            let reward = (total_reward_pool as f64 * share) as u128;
            
            rewards.push((validator.address, reward));
        }
        
        rewards
    }
    
    /// Select public goods proposals for funding
    pub fn select_public_goods_proposals(
        &self,
        proposals: &[PublicGoodsProposal],
        total_funding_pool: u128,
        min_vote_threshold: f64,
    ) -> Vec<(u64, u128)> {
        if proposals.is_empty() {
            return Vec::new();
        }
        
        // Filter and score proposals
        let mut scored_proposals: Vec<(f64, &PublicGoodsProposal)> = proposals
            .iter()
            .filter(|p| p.vote_score >= min_vote_threshold && p.verified)
            .map(|p| {
                // Calculate composite score
                let mut score = p.vote_score * 0.6 + p.technical_score * 0.4;
                score *= p.category.priority_multiplier();
                score *= p.impact_multiplier as f64 / 3.0; // Normalize impact multiplier
                (score, p)
            })
            .collect();
        
        // Sort by score (descending)
        scored_proposals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        
        // Select proposals until funding pool is exhausted
        let mut selected = Vec::new();
        let mut remaining_pool = total_funding_pool;
        
        for (score, proposal) in scored_proposals {
            if remaining_pool == 0 {
                break;
            }
            
            let max_funding = (proposal.requested_amount as f64 * score.min(1.0)) as u128;
            let funding = max_funding.min(remaining_pool);
            
            if funding > 0 {
                selected.push((proposal.proposal_id, funding));
                remaining_pool -= funding;
            }
        }
        
        selected
    }
    
    /// Calculate Shapley values for gradient contributions
    pub fn calculate_shapley_values(
        &self,
        gradients: &[Vec<f32>],
        baseline_loss: f64,
        final_loss: f64,
        quality_scores: &[f64],
        dp_epsilons: &[f64],
    ) -> Vec<f64> {
        let n = gradients.len();
        if n == 0 {
            return Vec::new();
        }
        
        // Monte Carlo approximation of Shapley values
        let samples = 1000.min(1 << n.min(10)); // Cap samples for large n
        
        let mut shapley_values = vec![0.0; n];
        
        for _ in 0..samples {
            // Generate random permutation
            let mut permutation: Vec<usize> = (0..n).collect();
            for i in 0..n {
                let j = i + (rand::random::<usize>() % (n - i));
                permutation.swap(i, j);
            }
            
            // Calculate marginal contributions
            let mut current_loss = baseline_loss;
            let mut coalition = Vec::new();
            
            for &contributor_idx in &permutation {
                // Add contributor to coalition
                coalition.push(contributor_idx);
                
                // Estimate loss with this coalition
                // In production: actually compute with secure MPC
                let coalition_size = coalition.len();
                let total_improvement = baseline_loss - final_loss;
                
                // Simplified: assume linear contribution
                let marginal_improvement = total_improvement * 
                    (1.0 / coalition_size as f64 - 
                     1.0 / (coalition_size - 1) as f64).abs();
                
                // Adjust for quality and privacy
                let quality_adjustment = 1.0 + (quality_scores[contributor_idx] - 0.5) * self.quality_weight;
                let privacy_adjustment = if dp_epsilons[contributor_idx] <= 1.0 {
                    self.privacy_bonus_multiplier
                } else {
                    1.0
                };
                
                let adjusted_improvement = marginal_improvement * quality_adjustment * privacy_adjustment;
                
                shapley_values[contributor_idx] += adjusted_improvement;
                current_loss -= adjusted_improvement;
            }
        }
        
        // Average and normalize
        for value in &mut shapley_values {
            *value /= samples as f64;
        }
        
        // Ensure sum equals total improvement (within rounding)
        let total_shapley: f64 = shapley_values.iter().sum();
        let total_improvement = baseline_loss - final_loss;
        
        if total_shapley > 0.0 && (total_shapley - total_improvement).abs() > 0.01 {
            let scaling_factor = total_improvement / total_shapley;
            for value in &mut shapley_values {
                *value *= scaling_factor;
            }
        }
        
        shapley_values
    }
    
    /// Update validator reputation scores
    pub fn update_reputation_scores(
        &self,
        validators: &mut [ValidationReward],
        performance_metrics: &HashMap<[u8; 20], ValidatorPerformance>,
    ) {
        for validator in validators {
            if let Some(performance) = performance_metrics.get(&validator.address) {
                // Calculate new reputation
                let mut new_reputation = validator.reputation_score;
                
                // Apply performance bonuses
                if performance.blocks_produced > 0 {
                    let production_rate = performance.blocks_produced as f64 / 
                                         performance.expected_blocks as f64;
                    new_reputation *= 0.5 + 0.5 * production_rate.min(1.0);
                }
                
                if performance.vote_accuracy > 0.9 {
                    new_reputation *= 1.05; // 5% bonus for high accuracy
                }
                
                // Apply penalties for poor performance
                if performance.latency_ms > 1000.0 {
                    new_reputation *= 0.95; // 5% penalty for high latency
                }
                
                if performance.slashing_events > 0 {
                    new_reputation *= 0.8f64.powi(performance.slashing_events as i32);
                }
                
                // Apply decay
                new_reputation *= self.reputation_decay_rate;
                
                // Clamp to valid range
                validator.reputation_score = new_reputation.clamp(0.0, 1.0);
                
                // Update other metrics
                validator.uptime = performance.uptime;
                validator.votes_count = performance.votes_participated;
                validator.blocks_produced = performance.blocks_produced;
                validator.latency_score = (1000.0 / (performance.latency_ms + 100.0)).clamp(0.0, 1.0);
            }
        }
    }
}


/// Validator performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorPerformance {
    /// Uptime percentage (0-1)
    pub uptime: f64,
    
    /// Vote accuracy (0-1)
    pub vote_accuracy: f64,
    
    /// Votes participated in
    pub votes_participated: u64,
    
    /// Blocks produced
    pub blocks_produced: u64,
    
    /// Expected blocks to produce
    pub expected_blocks: u64,
    
    /// Average latency in milliseconds
    pub latency_ms: f64,
    
    /// Slashing events count
    pub slashing_events: u64,
    
    /// Double-signing events
    pub double_sign_events: u64,
}


/// Reward distribution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardDistributionResult {
    /// Total reward pool
    pub total_pool: u128,
    
    /// Gradient rewards
    pub gradient_rewards: Vec<([u8; 20], u128)>,
    
    /// Validation rewards
    pub validation_rewards: Vec<([u8; 20], u128)>,
    
    /// Public goods funding
    pub public_goods_funding: Vec<(u64, u128)>,
    
    /// Remaining unallocated rewards
    pub remaining_rewards: u128,
    
    /// Distribution timestamp
    pub timestamp: u64,
}


/// Reward errors
#[derive(Debug, thiserror::Error)]
pub enum RewardError {
    #[error("Invalid reward distribution: gradient={0}, validation={1}, public_goods={2}")]
    InvalidDistribution(f64, f64, f64),
    
    #[error("No valid contributions")]
    NoValidContributions,
    
    #[error("Shapley value calculation failed")]
    ShapleyCalculationFailed,
    
    #[error("Reward pool exhausted")]
    PoolExhausted,
    
    #[error("Invalid validator data")]
    InvalidValidatorData,
    
    #[error("Proposal verification failed")]
    ProposalVerificationFailed,
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gradient_reward_calculation() {
        let calculator = RewardCalculator::default();
        
        let contributions = vec![
            GradientReward {
                address: [1u8; 20],
                shapley_value: 0.6,
                quality_score: 0.9,
                dp_epsilon: 0.5,
                timestamp: 0,
                gradient_hash: [0u8; 32],
                model_id: 1,
                round: 1,
            },
            GradientReward {
                address: [2u8; 20],
                shapley_value: 0.4,
                quality_score: 0.7,
                dp_epsilon: 3.0,
                timestamp: 0,
                gradient_hash: [0u8; 32],
                model_id: 1,
                round: 1,
            },
        ];
        
        let total_pool = 1_000_000 * 10u128.pow(18); // 1M NERV
        
        let rewards = calculator.calculate_gradient_rewards(&contributions, total_pool);
        
        assert_eq!(rewards.len(), 2);
        
        // First contributor should get more due to higher Shapley value and privacy bonus
        let (addr1, reward1) = rewards[0];
        let (addr2, reward2) = rewards[1];
        
        assert_eq!(addr1, [1u8; 20]);
        assert_eq!(addr2, [2u8; 20]);
        assert!(reward1 > reward2); // Higher Shapley + privacy bonus
        
        // Sum should not exceed total pool
        let total_reward: u128 = rewards.iter().map(|(_, r)| r).sum();
        assert!(total_reward <= total_pool);
    }
    
    #[test]
    fn test_validation_reward_calculation() {
        let calculator = RewardCalculator::default();
        
        let validators = vec![
            ValidationReward {
                address: [1u8; 20],
                stake: 10_000 * 10u128.pow(18), // 10k NERV
                reputation_score: 0.95,
                uptime: 0.99,
                votes_count: 1000,
                slashing_adjustment: 0,
                blocks_produced: 100,
                latency_score: 0.9,
            },
            ValidationReward {
                address: [2u8; 20],
                stake: 5_000 * 10u128.pow(18), // 5k NERV
                reputation_score: 0.90,
                uptime: 0.98,
                votes_count: 500,
                slashing_adjustment: -20, // Was slashed
                blocks_produced: 50,
                latency_score: 0.8,
            },
        ];
        
        let total_pool = 500_000 * 10u128.pow(18); // 500k NERV
        
        let rewards = calculator.calculate_validation_rewards(&validators, total_pool);
        
        assert_eq!(rewards.len(), 2);
        
        // First validator should get more due to higher stake and no slashing
        let (addr1, reward1) = rewards[0];
        let (addr2, reward2) = rewards[1];
        
        assert_eq!(addr1, [1u8; 20]);
        assert_eq!(addr2, [2u8; 20]);
        assert!(reward1 > reward2);
        
        // Weights should be proportional
        let weight1 = validators[0].weight();
        let weight2 = validators[1].weight();
        let expected_ratio = weight1 / weight2;
        let actual_ratio = reward1 as f64 / reward2 as f64;
        
        assert!((expected_ratio - actual_ratio).abs() < 0.1); // Within 10% tolerance
    }
    
    #[test]
    fn test_shapley_calculation() {
        let calculator = RewardCalculator::default();
        
        // Mock gradients
        let gradients = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];
        
        let quality_scores = vec![0.9, 0.7, 0.8];
        let dp_epsilons = vec![0.5, 3.0, 0.8];
        
        let baseline_loss = 10.0;
        let final_loss = 5.0;
        
        let shapley_values = calculator.calculate_shapley_values(
            &gradients,
            baseline_loss,
            final_loss,
            &quality_scores,
            &dp_epsilons,
        );
        
        assert_eq!(shapley_values.len(), 3);
        
        // All values should be positive
        for &value in &shapley_values {
            assert!(value >= 0.0);
        }
        
        // Sum should approximate total improvement
        let total_shapley: f64 = shapley_values.iter().sum();
        let total_improvement = baseline_loss - final_loss;
        
        assert!((total_shapley - total_improvement).abs() < 0.1);
        
        // Contributor 0 should have higher value due to better quality and privacy
        assert!(shapley_values[0] > shapley_values[1]);
    }
    
    #[test]
    fn test_proposal_selection() {
        let calculator = RewardCalculator::default();
        
        let proposals = vec![
            PublicGoodsProposal {
                proposal_id: 1,
                recipient: [1u8; 20],
                requested_amount: 100_000 * 10u128.pow(18),
                funded_amount: 0,
                description: "High impact research".to_string(),
                vote_score: 0.9,
                technical_score: 0.8,
                impact_multiplier: 5,
                category: ProposalCategory::Research,
                duration_months: 12,
                verified: true,
                milestones: Vec::new(),
            },
            PublicGoodsProposal {
                proposal_id: 2,
                recipient: [2u8; 20],
                requested_amount: 50_000 * 10u128.pow(18),
                funded_amount: 0,
                description: "Community education".to_string(),
                vote_score: 0.7,
                technical_score: 0.6,
                impact_multiplier: 3,
                category: ProposalCategory::Education,
                duration_months: 6,
                verified: true,
                milestones: Vec::new(),
            },
        ];
        
        let total_pool = 120_000 * 10u128.pow(18); // 120k NERV
        
        let selected = calculator.select_public_goods_proposals(
            &proposals,
            total_pool,
            0.5, // Minimum vote threshold
        );
        
        // Should select both proposals since pool is sufficient
        assert_eq!(selected.len(), 2);
        
        let total_funded: u128 = selected.iter().map(|(_, amount)| amount).sum();
        assert_eq!(total_funded, total_pool); // All funds allocated
        
        // First proposal should get more funding due to higher score
        assert!(selected[0].1 > selected[1].1);
    }
}
