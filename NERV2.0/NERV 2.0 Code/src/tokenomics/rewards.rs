//! Reward Records and History.
//!
//! Tracks per-validator reward distributions across blocks,
//! providing cumulative accounting and historical queries.

use crate::{
    ONE_NERV,
    BlockHeight, Epoch, ValidatorId, StakeAmount,
    NervError, NervResult,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Reward Type ─────────────────────────────────────────────────────────

/// The type of reward earned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RewardType {
    /// Reward for submitting a gradient that reduced Huber loss (60% of block).
    GradientContribution,

    /// Reward for honest consensus participation (30% of block).
    ValidationParticipation,

    /// Portion of transaction fees distributed to the block producer.
    TransactionFee,

    /// Reward from a public goods grant (10% of block, via governance).
    PublicGoodsGrant,

    /// Tail emission reward (after useful-work pool exhaustion).
    TailEmission,
}

impl RewardType {
    /// Get a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::GradientContribution => "gradient",
            Self::ValidationParticipation => "validation",
            Self::TransactionFee => "tx_fee",
            Self::PublicGoodsGrant => "public_goods",
            Self::TailEmission => "tail_emission",
        }
    }

    /// Is this a consensus-level reward?
    pub fn is_consensus_reward(&self) -> bool {
        matches!(self, Self::GradientContribution | Self::ValidationParticipation)
    }
}

impl std::fmt::Display for RewardType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ─── Reward Record ───────────────────────────────────────────────────────

/// A single reward distribution record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RewardRecord {
    /// The validator who received the reward.
    pub validator_id: ValidatorId,

    /// Block height of the reward.
    pub block_height: BlockHeight,

    /// Epoch of the reward.
    pub epoch: Epoch,

    /// Type of reward.
    pub reward_type: RewardType,

    /// Amount in nano-NERV.
    pub amount_nano: u64,

    /// Timestamp (Unix epoch millis).
    pub timestamp_ms: u64,
}

impl RewardRecord {
    /// Create a new reward record.
    pub fn new(
        validator_id: ValidatorId,
        block_height: BlockHeight,
        epoch: Epoch,
        reward_type: RewardType,
        amount_nano: u64,
    ) -> Self {
        Self {
            validator_id,
            block_height,
            epoch,
            reward_type,
            amount_nano,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Convert amount to whole NERV.
    pub fn amount_nerv(&self) -> u64 {
        self.amount_nano / ONE_NERV
    }
}

// ─── Reward History ──────────────────────────────────────────────────────

/// Cumulative reward history across all blocks.
///
/// Optimized for common queries:
/// - Total rewards per validator
/// - Total rewards per type
/// - Rewards in a specific epoch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardHistory {
    /// Per-validator cumulative rewards (validator_key → total_nano).
    per_validator: HashMap<[u8; 32], u64>,

    /// Per-type cumulative rewards.
    per_type: HashMap<String, u64>,

    /// Per-epoch cumulative rewards (epoch → total_nano).
    per_epoch: HashMap<u64, u64>,

    /// Total rewards distributed (nano-NERV).
    total_distributed_nano: u64,

    /// Number of reward records.
    record_count: u64,

    /// First block height with a reward.
    first_reward_height: Option<BlockHeight>,

    /// Last block height with a reward.
    last_reward_height: Option<BlockHeight>,
}

impl RewardHistory {
    /// Create a new empty reward history.
    pub fn new() -> Self {
        Self {
            per_validator: HashMap::new(),
            per_type: HashMap::new(),
            per_epoch: HashMap::new(),
            total_distributed_nano: 0,
            record_count: 0,
            first_reward_height: None,
            last_reward_height: None,
        }
    }

    /// Record a reward distribution.
    pub fn record(&mut self, reward: &RewardRecord) {
        // Per-validator
        let key = reward.validator_id.as_bytes().clone();
        let current = self.per_validator.get(&key).copied().unwrap_or(0);
        self.per_validator.insert(key, current.saturating_add(reward.amount_nano));

        // Per-type
        let type_key = reward.reward_type.name().to_string();
        let current = self.per_type.get(&type_key).copied().unwrap_or(0);
        self.per_type.insert(type_key, current.saturating_add(reward.amount_nano));

        // Per-epoch
        let epoch_key = reward.epoch.0;
        let current = self.per_epoch.get(&epoch_key).copied().unwrap_or(0);
        self.per_epoch.insert(epoch_key, current.saturating_add(reward.amount_nano));

        // Totals
        self.total_distributed_nano = self.total_distributed_nano
            .saturating_add(reward.amount_nano);
        self.record_count += 1;

        // Height tracking
        match self.first_reward_height {
            None => self.first_reward_height = Some(reward.block_height),
            Some(h) if reward.block_height < h => self.first_reward_height = Some(reward.block_height),
            _ => {}
        }
        self.last_reward_height = Some(reward.block_height);
    }

    /// Record multiple rewards at once.
    pub fn record_batch(&mut self, rewards: &[RewardRecord]) {
        for reward in rewards {
            self.record(reward);
        }
    }

    /// Get total rewards for a validator.
    pub fn validator_total(&self, validator_id: &ValidatorId) -> u64 {
        self.per_validator.get(validator_id.as_bytes()).copied().unwrap_or(0)
    }

    /// Get total rewards for a type.
    pub fn type_total(&self, reward_type: RewardType) -> u64 {
        self.per_type.get(reward_type.name()).copied().unwrap_or(0)
    }

    /// Get total rewards for an epoch.
    pub fn epoch_total(&self, epoch: Epoch) -> u64 {
        self.per_epoch.get(&epoch.0).copied().unwrap_or(0)
    }

    /// Total rewards distributed.
    pub fn total_distributed(&self) -> u64 {
        self.total_distributed_nano
    }

    /// Number of reward records.
    pub fn record_count(&self) -> u64 {
        self.record_count
    }

    /// Average reward per record.
    pub fn average_reward(&self) -> f64 {
        if self.record_count == 0 {
            return 0.0;
        }
        self.total_distributed_nano as f64 / self.record_count as f64
    }

    /// Number of unique validators who received rewards.
    pub fn unique_recipients(&self) -> usize {
        self.per_validator.len()
    }

    /// Get the top N validators by total rewards.
    pub fn top_recipients(&self, n: usize) -> Vec<(ValidatorId, u64)> {
        let mut entries: Vec<_> = self.per_validator.iter()
            .filter_map(|(key, &amount)| {
                // Reconstruct ValidatorId from bytes
                let mut id_bytes = [0u8; 32];
                id_bytes.copy_from_slice(key);
                Some((ValidatorId::from_bytes(id_bytes), amount))
            })
            .collect();

        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.into_iter().take(n).collect()
    }

    /// Reward distribution by type (for reporting).
    pub fn distribution_by_type(&self) -> Vec<(RewardType, u64, f64)> {
        let types = [
            RewardType::GradientContribution,
            RewardType::ValidationParticipation,
            RewardType::TransactionFee,
            RewardType::PublicGoodsGrant,
            RewardType::TailEmission,
        ];

        types.iter().map(|&rt| {
            let total = self.type_total(rt);
            let fraction = if self.total_distributed_nano > 0 {
                total as f64 / self.total_distributed_nano as f64
            } else {
                0.0
            };
            (rt, total, fraction)
        }).collect()
    }
}

impl Default for RewardHistory {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Fee Distribution ────────────────────────────────────────────────────

/// Transaction fee distribution logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeDistribution {
    /// Fraction of fees burned (0.0–1.0).
    pub burn_fraction: f64,

    /// Fraction of fees to block producer.
    pub producer_fraction: f64,

    /// Fraction of fees to validators.
    pub validator_fraction: f64,
}

impl FeeDistribution {
    /// Default NERV fee distribution: 50% burn, 30% producer, 20% validators.
    pub fn nerv_default() -> Self {
        Self {
            burn_fraction: 0.50,
            producer_fraction: 0.30,
            validator_fraction: 0.20,
        }
    }

    /// Distribute a fee pool.
    ///
    /// Returns (burned, producer_share, validator_share).
    pub fn distribute(&self, total_fees_nano: u64) -> (u64, u64, u64) {
        let burned = (total_fees_nano as f64 * self.burn_fraction) as u64;
        let producer = (total_fees_nano as f64 * self.producer_fraction) as u64;
        let validator = total_fees_nano
            .saturating_sub(burned)
            .saturating_sub(producer);

        (burned, producer, validator)
    }

    /// Validate the distribution fractions.
    pub fn validate(&self) -> bool {
        let total = self.burn_fraction + self.producer_fraction + self.validator_fraction;
        (total - 1.0).abs() < 1e-6
    }
}

impl Default for FeeDistribution {
    fn default() -> Self {
        Self::nerv_default()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reward_type_names() {
        assert_eq!(RewardType::GradientContribution.name(), "gradient");
        assert_eq!(RewardType::ValidationParticipation.name(), "validation");
    }

    #[test]
    fn test_reward_record_creation() {
        let record = RewardRecord::new(
            ValidatorId::from_bytes([1u8; 32]),
            BlockHeight::from(100),
            Epoch::from(0),
            RewardType::GradientContribution,
            1000 * ONE_NERV,
        );
        assert_eq!(record.amount_nerv(), 1000);
    }

    #[test]
    fn test_reward_history_basic() {
        let mut history = RewardHistory::new();
        let record = RewardRecord::new(
            ValidatorId::from_bytes([1u8; 32]),
            BlockHeight::from(1),
            Epoch::from(0),
            RewardType::GradientContribution,
            100 * ONE_NERV,
        );
        history.record(&record);

        assert_eq!(history.total_distributed(), 100 * ONE_NERV);
        assert_eq!(history.record_count(), 1);
        assert_eq!(history.unique_recipients(), 1);
    }

    #[test]
    fn test_reward_history_per_validator() {
        let mut history = RewardHistory::new();
        let vid = ValidatorId::from_bytes([1u8; 32]);

        history.record(&RewardRecord::new(
            vid.clone(), BlockHeight::from(1), Epoch::from(0),
            RewardType::GradientContribution, 50 * ONE_NERV,
        ));
        history.record(&RewardRecord::new(
            vid.clone(), BlockHeight::from(2), Epoch::from(0),
            RewardType::ValidationParticipation, 30 * ONE_NERV,
        ));

        assert_eq!(history.validator_total(&vid), 80 * ONE_NERV);
    }

    #[test]
    fn test_reward_history_per_type() {
        let mut history = RewardHistory::new();
        history.record(&RewardRecord::new(
            ValidatorId::from_bytes([1u8; 32]), BlockHeight::from(1), Epoch::from(0),
            RewardType::GradientContribution, 60 * ONE_NERV,
        ));
        history.record(&RewardRecord::new(
            ValidatorId::from_bytes([2u8; 32]), BlockHeight::from(1), Epoch::from(0),
            RewardType::GradientContribution, 40 * ONE_NERV,
        ));
        history.record(&RewardRecord::new(
            ValidatorId::from_bytesG       let vid = ValidatorId::from_bytes([1u8; 32]);    // Fix syntax error below
            ValidatorId::from_bytes([3u8; 32]), BlockHeight::from(1), Epoch::from(0),
            RewardType::ValidationParticipation, 30 * ONE_NERV,
        ));

        assert_eq!(history.type_total(RewardType::GradientContribution), 100 * ONE_NERV);
        assert_eq!(history.type_total(RewardType::ValidationParticipation), 30 * ONE_NERV);
    }

    #[test]
    fn test_reward_history_per_epoch() {
        let mut history = RewardHistory::new();
        history.record(&RewardRecord::new(
            ValidatorId::from_bytes([1u8; 32]), BlockHeight::from(1), Epoch::from(0),
            RewardType::GradientContribution, 100 * ONE_NERV,
        ));
        history.record(&RewardRecord::new(
            ValidatorId::from_bytes([1u8; 32]), BlockHeight::from(1), Epoch::from(1),
            RewardType::GradientContribution, 200 * ONE_NERV,
        ));

        assert_eq!(history.epoch_total(Epoch::from(0)), 100 * ONE_NERV);
        assert_eq!(history.epoch_total(Epoch::from(1)), 200 * ONE_NERV);
    }

    #[test]
    fn test_reward_history_batch() {
        let mut history = RewardHistory::new();
        let rewards = vec![
            RewardRecord::new(
                ValidatorId::from_bytes([1u8; 32]), BlockHeight::from(1), Epoch::from(0),
                RewardType::GradientContribution, 50 * ONE_NERV,
            ),
            RewardRecord::new(
                ValidatorId::from_bytes([2u8; 32]), BlockHeight::from(1), Epoch::from(0),
                RewardType::GradientContribution, 50 * ONE_NERV,
            ),
        ];
        history.record_batch(&rewards);
        assert_eq!(history.record_count(), 2);
    }

    #[test]
    fn test_reward_history_top_recipients() {
        let mut history = RewardHistory::new();
        history.record(&RewardRecord::new(
            ValidatorId::from_bytes([1u8; 32]), BlockHeight::from(1), Epoch::from(0),
            RewardType::GradientContribution, 300 * ONE_NERV,
        ));
        history.record(&RewardRecord::new(
            ValidatorId::from_bytes([2u8; 32]), BlockHeight::from(1), Epoch::from(0),
            RewardType::GradientContribution, 100 * ONE_NERV,
        ));

        let top = history.top_recipients(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].1, 300 * ONE_NERV);
    }

    #[test]
    fn test_fee_distribution() {
        let dist = FeeDistribution::nerv_default();
        assert!(dist.validate());

        let (burned, producer, validators) = dist.distribute(1000 * ONE_NERV);
        assert_eq!(burned, 500 * ONE_NERV);
        assert_eq!(producer, 300 * ONE_NERV);
        assert_eq!(validators, 200 * ONE_NERV);
    }

    #[test]
    fn test_reward_history_distribution_by_type() {
        let mut history = RewardHistory::new();
        history.record(&RewardRecord::new(
            ValidatorId::from_bytes([1u8; 32]), BlockHeight::from(1), Epoch::from(0),
            RewardType::GradientContribution, 60 * ONE_NERV,
        ));
        history.record(&RewardRecord::new(
            ValidatorId::from_bytes([1u8; 32]), BlockHeight::from(1), Epoch::from(0),
            RewardType::ValidationParticipation, 30 * ONE_NERV,
        ));

        let dist = history.distribution_by_type();
        assert_eq!(dist.len(), 5);

        // Gradient should be 60/90 ≈ 66.7%
        let (gradient_type, gradient_total, gradient_frac) = dist[0];
        assert_eq!(gradient_type, RewardType::GradientContribution);
        assert_eq!(gradient_total, 60 * ONE_NERV);
        assert!((gradient_frac - 60.0 / 90.0).abs() < 1e-6);
    }

    #[test]
    fn test_reward_history_average() {
        let mut history = RewardHistory::new();
        history.record(&RewardRecord::new(
            ValidatorId::from_bytes([1u8; 32]), BlockHeight::from(1), Epoch::from(0),
            RewardType::GradientContribution, 100 * ONE_NERV,
        ));
        history.record(&RewardRecord::new(
            ValidatorId::from_bytes([2u8; 32]), BlockHeight::from(1), Epoch::from(0),
            RewardType::GradientContribution, 200 * ONE_NERV,
        ));
        assert!((history.average_reward() - 150.0 * ONE_NERV as f64).abs() < 1e-6);
    }
}
