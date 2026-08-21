//! Real-Time Self-Evolution — Continuous Backprop.
//!
//! V2.0 replaces 30-day Federated Learning with per-block Adam optimizer
//! updates. Validators compute gradients of the Huber Loss with respect
//! to the Perceptron weights and submit them. Rewards are proportional
//! to how much a gradient reduces the global loss.
//!
//! Submodules:
//! - `weight_update` — Validates submitted gradient updates from validators
//! - `gradient_rewards` — Rewards nodes for gradients that reduce global Huber loss
//! - `staking` — Base staking logic
//!
//! Real-Time Self-Evolution — The Useful-Work Economy.
//!
//! NERV v2.0 replaces wasteful PoW and capital-hoarding PoS with a
//! **useful-work economy** that pays validators for training the
//! network's own intelligence via per-block Adam backpropagation.
//!
//! # V2.0 Changes vs V1.01
//!
//! | Component | V1.01 | V2.0 |
//! |-----------|-------|------|
//! | Learning | 30-day Federated Learning rounds | Per-block Adam/Huber backprop |
//! | Rewards | Shapley-value contributions | Loss-reduction-based rewards |
//! | Aggregation | Secure SMPC in TEEs | Direct submission + consensus |
//! | Complexity | High (FL orchestration, Shapley) | Low (standard Adam optimizer) |
//! | Update freq | Every 30 days | Every block (~400ms) |
//!
//! # Block Reward Distribution
//!
//! | Category | Fraction | Description |
//! |----------|----------|-------------|
//! | Gradient contribution | 60% | Validators whose gradients reduce Huber loss |
//! | Honest validation | 30% | Stake × reputation × uptime |
//! | Public goods | 10% | Quarterly on-chain governance vote |
//!
//! # How It Works
//!
//! ```text
//! Every block:
//! 1. Each validator computes Huber loss: L = Huber(ŷ, y)
//!    where ŷ = W·S_t + b (prediction) and y = actual embedding root
//!
//! 2. Each validator backpropagates: ∂L/∂W, ∂L/∂b
//!
//! 3. Each validator applies Adam: W_new = Adam(W, ∂L/∂W)
//!
//! 4. Validators broadcast: hash(W_new) + Dilithium signature
//!
//! 5. If ≥67% agree on hash(W_new):
//!    → W is updated on-chain (network "gets smarter")
//!    → Contributors rewarded proportionally to loss reduction
//!    → Adversarial gradients (increase loss) are slashed
//! ```

pub mod weight_update;
pub mod gradient_rewards;
pub mod staking;

// ─── Re-exports ─────────────────────────────────────────────────────────

pub use weight_update::{
    GradientSubmission, WeightUpdateProposal, WeightUpdateResult,
    GradientAggregation,
};
pub use gradient_rewards::{
    GradientContribution, RewardDistribution, RewardPool,
    GradientRewardCalculator,
};
pub use staking::{
    StakingLedger, StakeEntry, ValidatorStatus, UnbondingEntry,
};

use crate::{
    EMBEDDING_DIM, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, ADAM_LEARNING_RATE, HUBER_DELTA,
    ONE_NERV, TOTAL_SUPPLY_NANO,
    BlockHeight, Epoch, ValidatorId, StakeAmount, ReputationScore, VotingWeight,
    NervError, NervResult,
};
use crate::config::EconomyConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Block Reward ────────────────────────────────────────────────────────

/// A block's reward allocation across the three categories.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockReward {
    /// Total reward for this block (nano-NERV).
    pub total_nano: u64,

    /// Reward for gradient contributors (60%).
    pub gradient_reward_nano: u64,

    /// Reward for honest validators (30%).
    pub validation_reward_nano: u64,

    /// Reward for public goods grants (10%).
    pub public_goods_reward_nano: u64,

    /// Block height this reward is for.
    pub height: BlockHeight,

    /// Epoch at this block.
    pub epoch: Epoch,
}

impl BlockReward {
    /// Gradient reward fraction (60%).
    pub const GRADIENT_FRACTION_BPS: u64 = 6000;

    /// Validation reward fraction (30%).
    pub const VALIDATION_FRACTION_BPS: u64 = 3000;

    /// Public goods fraction (10%).
    pub const PUBLIC_GOODS_FRACTION_BPS: u64 = 1000;

    /// Basis points denominator.
    const BPS_DENOM: u64 = 10000;

    /// Compute block reward allocation from a total.
    pub fn from_total(total_nano: u64, height: BlockHeight, epoch: Epoch) -> Self {
        let gradient = (total_nano * Self::GRADIENT_FRACTION_BPS) / Self::BPS_DENOM;
        let validation = (total_nano * Self::VALIDATION_FRACTION_BPS) / Self::BPS_DENOM;
        let public_goods = total_nano
            .saturating_sub(gradient)
            .saturating_sub(validation);

        Self {
            total_nano,
            gradient_reward_nano: gradient,
            validation_reward_nano: validation,
            public_goods_reward_nano: public_goods,
            height,
            epoch,
        }
    }

    /// Verify the allocation sums correctly.
    pub fn verify_allocation(&self) -> bool {
        self.gradient_reward_nano + self.validation_reward_nano + self.public_goods_reward_nano
            == self.total_nano
    }
}

// ─── Emission Schedule ───────────────────────────────────────────────────

/// Emission schedule for NERV tokens.
///
/// Total supply: 10B NERV (10^19 nano-NERV)
/// Useful-work pool: 80% = 8B NERV
/// Emission decays over time with a tail emission of 0.5%/yr forever.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmissionSchedule {
    /// Total supply in nano-NERV.
    pub total_supply_nano: u64,

    /// Useful-work pool in nano-NERV (80% of total).
    pub useful_work_pool_nano: u64,

    /// Already emitted nano-NERV.
    pub emitted_nano: u64,

    /// Initial emission rate (nano-NERV per block at genesis).
    pub initial_emission_per_block: u64,

    /// Current emission rate (nano-NERV per block).
    pub current_emission_per_block: u64,

    /// Decay factor per epoch (e.g., 0.95 = 5% reduction per epoch).
    pub epoch_decay_factor: f64,

    /// Tail emission rate (nano-NERV per block,永不归零).
    pub tail_emission_per_block: u64,

    /// Blocks per epoch (for emission recalculation).
    pub blocks_per_epoch: u64,

    /// Current epoch.
    pub current_epoch: Epoch,
}

impl EmissionSchedule {
    /// Create the genesis emission schedule.
    pub fn genesis() -> Self {
        let total = TOTAL_SUPPLY_NANO;
        let useful_work_pool = (total as u128 * 8000 / 10000) as u64;

        // Initial emission: ~100 NERV per block (100 * 10^9 nano)
        // At 400ms blocks: 100 * 86400 / 0.4 = 21.6M NERV/day
        let initial_emission = 100 * ONE_NERV;

        Self {
            total_supply_nano: total,
            useful_work_pool_nano: useful_work_pool,
            emitted_nano: 0,
            initial_emission_per_block: initial_emission,
            current_emission_per_block: initial_emission,
            epoch_decay_factor: 0.95,
            tail_emission_per_block: ONE_NERV / 10, // 0.1 NERV per block tail
            blocks_per_epoch: 6_480_000, // ~30 days at 400ms blocks
            current_epoch: Epoch::GENESIS,
        }
    }

    /// Compute the emission for the next block.
    pub fn next_block_emission(&mut self) -> u64 {
        if self.emitted_nano >= self.useful_work_pool_nano {
            // Past the useful-work pool: only tail emission
            return self.tail_emission_per_block;
        }

        let emission = self.current_emission_per_block;
        self.emitted_nano = self.emitted_nano.saturating_add(emission);

        // Don't emit more than the pool
        if self.emitted_nano > self.useful_work_pool_nano {
            let excess = self.emitted_nano - self.useful_work_pool_nano;
            self.emitted_nano = self.useful_work_pool_nano;
            return emission.saturating_sub(excess);
        }

        emission
    }

    /// Advance to a new epoch and decay the emission rate.
    pub fn advance_epoch(&mut self) {
        self.current_epoch = self.current_epoch.next();
        let decayed = (self.current_emission_per_block as f64 * self.epoch_decay_factor) as u64;
        self.current_emission_per_block = decayed.max(self.tail_emission_per_block);
    }

    /// Check if a block triggers an epoch transition.
    pub fn should_advance_epoch(&self, block_height: BlockHeight) -> bool {
        block_height.0 > 0 && block_height.0 % self.blocks_per_epoch == 0
    }

    /// Remaining tokens in the useful-work pool.
    pub fn remaining_pool(&self) -> u64 {
        self.useful_work_pool_nano.saturating_sub(self.emitted_nano)
    }

    /// Fraction of pool already emitted (0.0–1.0).
    pub fn emitted_fraction(&self) -> f64 {
        if self.useful_work_pool_nano == 0 {
            return 0.0;
        }
        self.emitted_nano as f64 / self.useful_work_pool_nano as f64
    }
}

impl Default for EmissionSchedule {
    fn default() -> Self {
        Self::genesis()
    }
}

// ─── Economy State ───────────────────────────────────────────────────────

/// The global state of the useful-work economy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomyState {
    /// The staking ledger.
    pub staking: StakingLedger,

    /// The emission schedule.
    pub emission: EmissionSchedule,

    /// Current Huber loss of the network's embedding model.
    pub current_huber_loss: f64,

    /// Huber loss at the previous block (for delta computation).
    pub previous_huber_loss: f64,

    /// Total gradient rewards distributed (nano-NERV).
    pub total_gradient_rewards_distributed: u64,

    /// Total validation rewards distributed (nano-NERV).
    pub total_validation_rewards_distributed: u64,

    /// Total slashing applied (nano-NERV).
    pub total_slashed: u64,

    /// Number of blocks processed.
    pub blocks_processed: u64,

    /// Current block height.
    pub current_height: BlockHeight,

    /// Current epoch.
    pub current_epoch: Epoch,

    /// Configuration.
    pub config: EconomyConfig,
}

impl EconomyState {
    /// Create the genesis economy state.
    pub fn genesis(config: EconomyConfig) -> Self {
        Self {
            staking: StakingLedger::new(),
            emission: EmissionSchedule::genesis(),
            current_huber_loss: 0.0,
            previous_huber_loss: 0.0,
            total_gradient_rewards_distributed: 0,
            total_validation_rewards_distributed: 0,
            total_slashed: 0,
            blocks_processed: 0,
            current_height: BlockHeight::GENESIS,
            current_epoch: Epoch::GENESIS,
            config,
        }
    }

    /// Compute the change in Huber loss (negative = improvement).
    pub fn loss_delta(&self) -> f64 {
        self.current_huber_loss - self.previous_huber_loss
    }

    /// Check if the network improved (loss decreased).
    pub fn is_improving(&self) -> bool {
        self.current_huber_loss < self.previous_huber_loss
    }

    /// Update the Huber loss after a block.
    pub fn update_loss(&mut self, new_loss: f64) {
        self.previous_huber_loss = self.current_huber_loss;
        self.current_huber_loss = new_loss;
    }

    /// Advance to the next block.
    pub fn advance_block(&mut self) -> BlockReward {
        self.current_height = self.current_height.next();
        self.blocks_processed += 1;

        // Check epoch transition
        if self.emission.should_advance_epoch(self.current_height) {
            self.emission.advance_epoch();
            self.current_epoch = self.current_epoch.next();
        }

        // Compute emission
        let emission = self.emission.next_block_emission();
        BlockReward::from_total(emission, self.current_height, self.current_epoch)
    }
}

// ─── Slashing Reasons ────────────────────────────────────────────────────

/// Reasons for slashing a validator's stake.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SlashReason {
    /// Signed two different blocks at the same height (equivocation).
    DoubleSign,

    /// Submitted a gradient that was demonstrably adversarial
    /// (significantly increased loss or had excessive norm).
    AdversarialGradient,

    /// Failed to participate in threshold decryption ceremony.
    DecryptionFailure,

    /// Dropped Sphinx relay packets (for relay operators).
    RelayDropout,

    /// Proposed an invalid weight update.
    InvalidWeightUpdate,

    /// Lost a dispute resolution.
    DisputeLost,

    /// General downtime / liveness failure.
    LivenessFailure,
}

impl SlashReason {
    /// Get the slashing rate as a fraction of stake (basis points).
    pub fn slash_rate_bps(&self) -> u64 {
        match self {
            Self::DoubleSign => 500,          // 5%
            Self::AdversarialGradient => 100, // 1%
            Self::DecryptionFailure => 50,    // 0.5%
            Self::RelayDropout => 50,         // 0.5%
            Self::InvalidWeightUpdate => 200, // 2%
            Self::DisputeLost => 500,         // 5%
            Self::LivenessFailure => 10,      // 0.1%
        }
    }

    /// Whether this slash also reduces reputation.
    pub fn affects_reputation(&self) -> bool {
        true // All slashes affect reputation in V2.0
    }

    /// Get a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::DoubleSign => "double-signed at same height",
            Self::AdversarialGradient => "submitted adversarial gradient",
            Self::DecryptionFailure => "failed decryption ceremony",
            Self::RelayDropout => "dropped relay packets",
            Self::InvalidWeightUpdate => "proposed invalid weight update",
            Self::DisputeLost => "lost dispute resolution",
            Self::LivenessFailure => "liveness failure",
        }
    }
}

impl std::fmt::Display for SlashReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_reward_allocation() {
        let reward = BlockReward::from_total(
            1000 * ONE_NERV,
            BlockHeight::from(1),
            Epoch::GENESIS,
        );
        // 60% gradient, 30% validation, 10% public goods
        assert_eq!(reward.gradient_reward_nano, 600 * ONE_NERV);
        assert_eq!(reward.validation_reward_nano, 300 * ONE_NERV);
        assert_eq!(reward.public_goods_reward_nano, 100 * ONE_NERV);
        assert!(reward.verify_allocation());
    }

    #[test]
    fn test_emission_schedule_genesis() {
        let schedule = EmissionSchedule::genesis();
        assert_eq!(schedule.emitted_nano, 0);
        assert!(schedule.current_emission_per_block > 0);
    }

    #[test]
    fn test_emission_schedule_decay() {
        let mut schedule = EmissionSchedule::genesis();
        let initial = schedule.current_emission_per_block;
        schedule.advance_epoch();
        // After one epoch, emission should decay
        assert!(schedule.current_emission_per_block < initial);
        // But not below tail emission
        assert!(schedule.current_emission_per_block >= schedule.tail_emission_per_block);
    }

    #[test]
    fn test_emission_schedule_remaining() {
        let schedule = EmissionSchedule::genesis();
        assert!(schedule.remaining_pool() > 0);
        assert!((schedule.emitted_fraction()).abs() < 1e-10);
    }

    #[test]
    fn test_economy_state_genesis() {
        let state = EconomyState::genesis(EconomyConfig::default());
        assert_eq!(state.current_height, BlockHeight::GENESIS);
        assert_eq!(state.blocks_processed, 0);
    }

    #[test]
    fn test_economy_state_loss_improvement() {
        let mut state = EconomyState::genesis(EconomyConfig::default());
        state.current_huber_loss = 1.0;
        state.previous_huber_loss = 1.0;
        assert!(!state.is_improving());

        state.update_loss(0.5);
        assert!(state.is_improving());
        assert!((state.loss_delta() - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_slash_reason_rates() {
        assert_eq!(SlashReason::DoubleSign.slash_rate_bps(), 500);
        assert_eq!(SlashReason::AdversarialGradient.slash_rate_bps(), 100);
        assert!(SlashReason::DoubleSign.affects_reputation());
    }

    #[test]
    fn test_slash_reason_display() {
        assert_eq!(SlashReason::DoubleSign.to_string(), "double-signed at same height");
    }

    #[test]
    fn test_emission_next_block() {
        let mut schedule = EmissionSchedule::genesis();
        let e1 = schedule.next_block_emission();
        let e2 = schedule.next_block_emission();
        assert_eq!(e1, e2); // Same within an epoch
        assert!(e1 > 0);
    }
}

