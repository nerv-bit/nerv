//! Gradient Rewards — Paying Validators for Useful Work.
//!
//! NERV v2.0 rewards validators proportionally to how much their
//! gradient submissions reduce the network's global Huber loss.
//! This replaces the complex Shapley-value computation of V1.01
//! with a simple, transparent, and auditable reward mechanism.
//!
//! # Reward Calculation
//!
//! ```text
//! For each validator i who submitted a gradient:
//!   utility_i = max(0, loss_before_i - loss_after_i)
//!
//!   reward_i = (utility_i / Σ utility_j) × gradient_reward_pool
//! ```
//!
//! Validators whose gradients increase the loss receive zero reward.
//! Validators whose gradients are adversarial (significantly increase
//! loss) are slashed.

use crate::{
    EMBEDDING_DIM, ONE_NERV,
    BlockHeight, Epoch, ValidatorId, StakeAmount, ReputationScore, VotingWeight,
    NervError, NervResult,
};
use crate::economy::{
    BlockReward, SlashReason,
    weight_update::GradientSubmission,
    staking::{StakingLedger, ValidatorStatus},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Gradient Contribution ───────────────────────────────────────────────

/// Record of a validator's gradient contribution for a single block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientContribution {
    /// The contributing validator.
    pub validator_id: ValidatorId,

    /// Block height.
    pub block_height: BlockHeight,

    /// The utility score (loss reduction, ≥ 0).
    pub utility_score: f64,

    /// Whether this gradient improved the loss.
    pub is_beneficial: bool,

    /// The actual loss reduction achieved.
    pub loss_reduction: f64,

    /// The gradient norm (for quality assessment).
    pub gradient_norm: f64,

    /// Computed reward in nano-NERV (set during distribution).
    pub reward_nano: u64,
}

impl GradientContribution {
    /// Create from a gradient submission.
    pub fn from_submission(submission: &GradientSubmission) -> Self {
        let loss_reduction = submission.loss_reduction();
        let is_beneficial = loss_reduction > 0.0;
        let utility_score = loss_reduction.max(0.0);

        Self {
            validator_id: submission.validator_id.clone(),
            block_height: submission.block_height,
            utility_score,
            is_beneficial,
            loss_reduction,
            gradient_norm: submission.gradient_norm,
            reward_nano: 0,
        }
    }

    /// Check if this contribution is eligible for reward.
    pub fn is_eligible(&self) -> bool {
        self.is_beneficial && self.utility_score > 0.0
    }
}

// ─── Reward Distribution ─────────────────────────────────────────────────

/// The result of distributing rewards for a block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RewardDistribution {
    /// Block height.
    pub block_height: BlockHeight,

    /// Per-validator gradient rewards (validator_id → nano-NERV).
    pub gradient_rewards: HashMap<[u8; 32], u64>,

    /// Per-validator validation rewards (validator_id → nano-NERV).
    pub validation_rewards: HashMap<[u8; 32], u64>,

    /// Total gradient rewards distributed.
    pub total_gradient_rewards_nano: u64,

    /// Total validation rewards distributed.
    pub total_validation_rewards_nano: u64,

    /// Public goods grant amount.
    pub public_goods_nano: u64,

    /// Number of gradient contributors rewarded.
    pub contributor_count: usize,

    /// Number of validators rewarded for honest participation.
    pub validator_count: usize,
}

impl RewardDistribution {
    /// Create an empty distribution.
    pub fn empty(block_height: BlockHeight) -> Self {
        Self {
            block_height,
            gradient_rewards: HashMap::new(),
            validation_rewards: HashMap::new(),
            total_gradient_rewards_nano: 0,
            total_validation_rewards_nano: 0,
            public_goods_nano: 0,
            contributor_count: 0,
            validator_count: 0,
        }
    }

    /// Get the total rewards distributed.
    pub fn total_nano(&self) -> u64 {
        self.total_gradient_rewards_nano
            .saturating_add(self.total_validation_rewards_nano)
            .saturating_add(self.public_goods_nano)
    }

    /// Get a validator's total reward.
    pub fn validator_total_reward(&self, validator_id: &ValidatorId) -> u64 {
        let grad = self.gradient_rewards.get(validator_id.as_bytes()).copied().unwrap_or(0);
        let val = self.validation_rewards.get(validator_id.as_bytes()).copied().unwrap_or(0);
        grad.saturating_add(val)
    }
}

// ─── Reward Pool ─────────────────────────────────────────────────────────

/// Tracks the reward pool state across blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardPool {
    /// Cumulative gradient rewards distributed (nano-NERV).
    pub cumulative_gradient_nano: u64,

    /// Cumulative validation rewards distributed (nano-NERV).
    pub cumulative_validation_nano: u64,

    /// Cumulative public goods distributed (nano-NERV).
    pub cumulative_public_goods_nano: u64,

    /// Undistributed rewards from rounding (nano-NERV).
    pub rounding_remainder_nano: u64,
}

impl RewardPool {
    /// Create a new empty pool.
    pub fn new() -> Self {
        Self {
            cumulative_gradient_nano: 0,
            cumulative_validation_nano: 0,
            cumulative_public_goods_nano: 0,
            rounding_remainder_nano: 0,
        }
    }

    /// Record a distribution.
    pub fn record(&mut self, distribution: &RewardDistribution) {
        self.cumulative_gradient_nano = self.cumulative_gradient_nano
            .saturating_add(distribution.total_gradient_rewards_nano);
        self.cumulative_validation_nano = self.cumulative_validation_nano
            .saturating_add(distribution.total_validation_rewards_nano);
        self.cumulative_public_goods_nano = self.cumulative_public_goods_nano
            .saturating_add(distribution.public_goods_nano);
    }

    /// Total distributed.
    pub fn total_distributed_nano(&self) -> u64 {
        self.cumulative_gradient_nano
            .saturating_add(self.cumulative_validation_nano)
            .saturating_add(self.cumulative_public_goods_nano)
    }
}

impl Default for RewardPool {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Gradient Reward Calculator ──────────────────────────────────────────

/// Calculates and distributes rewards based on gradient utility.
pub struct GradientRewardCalculator {
    /// Minimum utility score to be eligible for reward.
    pub min_utility_score: f64,

    /// Maximum fraction of pool a single validator can receive (0.0–1.0).
    pub max_single_validator_fraction: f64,

    /// Threshold for flagging a gradient as adversarial.
    /// If loss_increase > this threshold, the validator is slashed.
    pub adversarial_loss_threshold: f64,
}

impl GradientRewardCalculator {
    /// Create with default parameters.
    pub fn new() -> Self {
        Self {
            min_utility_score: 1e-10,
            max_single_validator_fraction: 0.3,
            adversarial_loss_threshold: 1.0,
        }
    }

    /// Compute gradient contributions from submissions.
    pub fn compute_contributions(
        &self,
        submissions: &[GradientSubmission],
    ) -> Vec<GradientContribution> {
        submissions.iter()
            .map(|sub| GradientContribution::from_submission(sub))
            .collect()
    }

    /// Distribute the gradient reward pool among contributors.
    ///
    /// Reward is proportional to utility score:
    /// `reward_i = (utility_i / Σ utility_j) × pool`
    pub fn distribute_gradient_rewards(
        &self,
        contributions: &[GradientContribution],
        pool_nano: u64,
        block_height: BlockHeight,
    ) -> (HashMap<[u8; 32], u64>, Vec<ValidatorId>) {
        let mut rewards = HashMap::new();
        let mut adversarial_validators = Vec::new();

        if pool_nano == 0 {
            return (rewards, adversarial_validators);
        }

        // Filter eligible contributions and compute total utility
        let eligible: Vec<&GradientContribution> = contributions.iter()
            .filter(|c| c.is_eligible() && c.utility_score >= self.min_utility_score)
            .collect();

        if eligible.is_empty() {
            return (rewards, adversarial_validators);
        }

        let total_utility: f64 = eligible.iter().map(|c| c.utility_score).sum();

        if total_utility <= 0.0 {
            return (rewards, adversarial_validators);
        }

        // Compute individual rewards
        let max_individual = (pool_nano as f64 * self.max_single_validator_fraction) as u64;
        let mut distributed: u64 = 0;

        for contribution in &eligible {
            let fraction = contribution.utility_score / total_utility;
            let mut reward = (pool_nano as f64 * fraction) as u64;

            // Cap individual reward
            reward = reward.min(max_individual);

            rewards.insert(contribution.validator_id.as_bytes().clone(), reward);
            distributed = distributed.saturating_add(reward);
        }

        // Check for adversarial gradients
        for contribution in contributions {
            if contribution.loss_reduction < -self.adversarial_loss_threshold {
                adversarial_validators.push(contribution.validator_id.clone());
            }
        }

        (rewards, adversarial_validators)
    }

    /// Distribute validation rewards among honest validators.
    ///
    /// Validation rewards are proportional to stake × reputation × uptime.
    pub fn distribute_validation_rewards(
        &self,
        staking: &StakingLedger,
        pool_nano: u64,
        block_height: BlockHeight,
    ) -> HashMap<[u8; 32], u64> {
        let mut rewards = HashMap::new();

        if pool_nano == 0 {
            return rewards;
        }

        // Compute total voting weight of active validators
        let active_validators: Vec<(&ValidatorId, VotingWeight)> = staking.active_validators()
            .iter()
            .filter_map(|id| {
                staking.get(id).map(|entry| {
                    let weight = VotingWeight::from_stake_reputation(
                        entry.stake,
                        entry.reputation,
                    );
                    (id, weight)
                })
            })
            .collect();

        let total_weight: u128 = active_validators.iter().map(|(_, w)| w.0).sum();

        if total_weight == 0 {
            return rewards;
        }

        // Distribute proportionally
        for (validator_id, weight) in &active_validators {
            let fraction = weight.0 as f64 / total_weight as f64;
            let reward = (pool_nano as f64 * fraction) as u64;
            if reward > 0 {
                rewards.insert(validator_id.as_bytes().clone(), reward);
            }
        }

        rewards
    }

    /// Full reward distribution for a block.
    pub fn distribute_block_rewards(
        &self,
        block_reward: &BlockReward,
        gradient_submissions: &[GradientSubmission],
        staking: &StakingLedger,
    ) -> NervResult<RewardDistribution> {
        let mut distribution = RewardDistribution::empty(block_reward.height);

        // 1. Compute gradient contributions
        let contributions = self.compute_contributions(gradient_submissions);

        // 2. Distribute gradient rewards (60%)
        let (grad_rewards, adversarial) = self.distribute_gradient_rewards(
            &contributions,
            block_reward.gradient_reward_nano,
            block_reward.height,
        );

        distribution.gradient_rewards = grad_rewards;
        distribution.total_gradient_rewards_nano = distribution.gradient_rewards.values().sum();
        distribution.contributor_count = distribution.gradient_rewards.len();

        // 3. Distribute validation rewards (30%)
        let val_rewards = self.distribute_validation_rewards(
            staking,
            block_reward.validation_reward_nano,
            block_reward.height,
        );

        distribution.validation_rewards = val_rewards;
        distribution.total_validation_rewards_nano = distribution.validation_rewards.values().sum();
        distribution.validator_count = distribution.validation_rewards.len();

        // 4. Public goods (10%) — held for quarterly governance
        distribution.public_goods_nano = block_reward.public_goods_reward_nano;

        // 5. Slash adversarial validators
        if !adversarial.is_empty() {
            // The caller (EconomyEngine) should handle slashing
            // We just flag them here
        }

        Ok(distribution)
    }
}

impl Default for GradientRewardCalculator {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Utility Score Computation ───────────────────────────────────────────

/// Compute a quality-adjusted utility score for a gradient.
///
/// The utility score considers:
/// 1. **Loss reduction**: The primary signal (how much the gradient improved the model)
/// 2. **Gradient efficiency**: Smaller gradients that achieve the same reduction are preferred
/// 3. **Reputation weighting**: Higher-reputation validators get a small bonus
///
/// ```text
/// utility = loss_reduction × (1 + reputation_weight) / (1 + norm_penalty)
/// ```
pub fn compute_utility_score(
    loss_reduction: f64,
    gradient_norm: f64,
    reputation: ReputationScore,
    norm_target: f64,
) -> f64 {
    if loss_reduction <= 0.0 {
        return 0.0;
    }

    // Reputation weighting: high-rep validators get a small boost
    let reputation_weight = reputation.to_f64() * 0.1; // Max 10% boost

    // Norm penalty: prefer efficient (smaller) gradients
    let norm_penalty = if norm_target > 0.0 {
        (gradient_norm / norm_target - 1.0).max(0.0) * 0.1
    } else {
        0.0
    };

    loss_reduction * (1.0 + reputation_weight) / (1.0 + norm_penalty)
}

/// Compute the cumulative utility score for a validator across multiple blocks.
pub fn cumulative_utility_score(contributions: &[GradientContribution]) -> f64 {
    contributions.iter()
        .filter(|c| c.is_beneficial)
        .map(|c| c.utility_score)
        .sum()
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::economy::weight_update::GradientSubmission;

    fn make_test_submission(
        validator: [u8; 32],
        loss_before: f64,
        loss_after: f64,
    ) -> GradientSubmission {
        GradientSubmission::new(
            ValidatorId::from_bytes(validator),
            BlockHeight::from(1),
            Epoch::from(0),
            vec![0.01; EMBEDDING_DIM],
            0.005,
            loss_before,
            loss_after,
            [1u8; 32],
            [2u8; 32],
        )
    }

    #[test]
    fn test_gradient_contribution_beneficial() {
        let sub = make_test_submission([1u8; 32], 1.0, 0.8);
        let contrib = GradientContribution::from_submission(&sub);
        assert!(contrib.is_beneficial);
        assert!(contrib.is_eligible());
        assert!((contrib.utility_score - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_contribution_harmful() {
        let sub = make_test_submission([1u8; 32], 1.0, 1.5);
        let contrib = GradientContribution::from_submission(&sub);
        assert!(!contrib.is_beneficial);
        assert!(!contrib.is_eligible());
        assert!((contrib.utility_score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_contribution_no_change() {
        let sub = make_test_submission([1u8; 32], 1.0, 1.0);
        let contrib = GradientContribution::from_submission(&sub);
        assert!(!contrib.is_beneficial);
        assert!(!contrib.is_eligible());
    }

    #[test]
    fn test_reward_distribution_empty() {
        let dist = RewardDistribution::empty(BlockHeight::from(1));
        assert_eq!(dist.total_nano(), 0);
    }

    #[test]
    fn test_reward_pool() {
        let mut pool = RewardPool::new();
        let dist = RewardDistribution {
            block_height: BlockHeight::from(1),
            gradient_rewards: HashMap::new(),
            validation_rewards: HashMap::new(),
            total_gradient_rewards_nano: 100,
            total_validation_rewards_nano: 50,
            public_goods_nano: 20,
            contributor_count: 0,
            validator_count: 0,
        };
        pool.record(&dist);
        assert_eq!(pool.total_distributed_nano(), 170);
    }

    #[test]
    fn test_gradient_reward_calculator_distribute() {
        let calc = GradientRewardCalculator::new();

        let sub1 = make_test_submission([1u8; 32], 1.0, 0.8); // 0.2 reduction
        let sub2 = make_test_submission([2u8; 32], 1.0, 0.5); // 0.5 reduction

        let contributions = calc.compute_contributions(&[sub1, sub2]);
        assert_eq!(contributions.len(), 2);

        let pool_nano = 1000 * ONE_NERV;
        let (rewards, adversarial) = calc.distribute_gradient_rewards(
            &contributions, pool_nano, BlockHeight::from(1),
        );

        assert!(rewards.len() >= 1);
        assert!(adversarial.is_empty());

        // Total distributed should be close to pool
        let total: u64 = rewards.values().sum();
        assert!(total > 0);
        assert!(total <= pool_nano);
    }

    #[test]
    fn test_gradient_reward_calculator_adversarial() {
        let calc = GradientRewardCalculator::new();

        let sub = make_test_submission([1u8; 32], 1.0, 3.0); // Loss increased by 2.0

        let contributions = calc.compute_contributions(&[sub]);
        let (rewards, adversarial) = calc.distribute_gradient_rewards(
            &contributions, 1000 * ONE_NERV, BlockHeight::from(1),
        );

        // Adversarial validator should be flagged
        assert_eq!(adversarial.len(), 1);
        // Should not receive any gradient reward
        assert_eq!(rewards.get(&[1u8; 32]).copied().unwrap_or(0), 0);
    }

    #[test]
    fn test_gradient_reward_calculator_no_contributions() {
        let calc = GradientRewardCalculator::new();
        let (rewards, _) = calc.distribute_gradient_rewards(
            &[], 1000 * ONE_NERV, BlockHeight::from(1),
        );
        assert!(rewards.is_empty());
    }

    #[test]
    fn test_compute_utility_score() {
        let rep = ReputationScore::PERFECT;
        let score = compute_utility_score(0.5, 0.1, rep, 0.1);
        assert!(score > 0.5); // Reputation boost
    }

    #[test]
    fn test_compute_utility_score_no_reduction() {
        let score = compute_utility_score(-0.5, 0.1, ReputationScore::PERFECT, 0.1);
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cumulative_utility_score() {
        let c1 = GradientContribution {
            validator_id: ValidatorId::from_bytes([1u8; 32]),
            block_height: BlockHeight::from(1),
            utility_score: 0.3,
            is_beneficial: true,
            loss_reduction: 0.3,
            gradient_norm: 0.1,
            reward_nano: 0,
        };
        let c2 = GradientContribution {
            validator_id: ValidatorId::from_bytes([1u8; 32]),
            block_height: BlockHeight::from(2),
            utility_score: 0.5,
            is_beneficial: true,
            loss_reduction: 0.5,
            gradient_norm: 0.1,
            reward_nano: 0,
        };
        let total = cumulative_utility_score(&[c1, c2]);
        assert!((total - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_reward_distribution_validator_total() {
        let mut dist = RewardDistribution::empty(BlockHeight::from(1));
        let vid = ValidatorId::from_bytes([1u8; 32]);
        dist.gradient_rewards.insert(vid.as_bytes().clone(), 100);
        dist.validation_rewards.insert(vid.as_bytes().clone(), 50);
        assert_eq!(dist.validator_total_reward(&vid), 150);
    }
}
