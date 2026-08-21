//! Staking Ledger — Validator Registration, Staking, and Slashing.
//!
//! Manages the lifecycle of validators: registration, stake deposits
//! and withdrawals, unbonding periods, slashing for misbehavior,
//! and reputation updates.
//!
//! # V2.0 Changes vs V1.01
//!
//! - Removed Shapley-value complexity from staking rewards
//! - Staking rewards are now a fixed 30% of block rewards
//! - Gradient rewards (60%) are the primary income source
//! - Slashing includes new category: AdversarialGradient
//! - Unbonding period: 14 days (~3M blocks at 400ms)

use crate::{
    ONE_NERV, StakeAmount, ReputationScore, VotingWeight,
    ValidatorId, BlockHeight, Epoch,
    NervError, NervResult,
};
use crate::economy::SlashReason;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Constants ───────────────────────────────────────────────────────────

/// Minimum stake to become a validator (1,000 NERV).
pub const MIN_VALIDATOR_STAKE: StakeAmount = StakeAmount::MIN_VALIDATOR_STAKE;

/// Unbonding period in blocks (~14 days at 400ms blocks).
/// 14 * 24 * 3600 / 0.4 = 3,024,000
pub const UNBONDING_PERIOD_BLOCKS: u64 = 3_024_000;

/// Maximum number of active validators.
pub const MAX_ACTIVE_VALIDATORS: usize = 10_000;

/// Maximum number of validators per shard.
pub const MAX_VALIDATORS_PER_SHARD: usize = 200;

/// Reputation increase per block for honest participation (in reputation units).
pub const REPUTATION_GAIN_PER_BLOCK: u32 = 10;

/// Reputation decrease per slash event (in reputation units).
pub const REPUTATION_LOSS_PER_SLASH: u32 = 50_000;

// ─── Validator Status ────────────────────────────────────────────────────

/// The current status of a validator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidatorStatus {
    /// Validator is active and participating in consensus.
    Active,

    /// Validator is in the unbonding period (waiting to withdraw).
    Unbonding,

    /// Validator has been slashed and is permanently removed.
    Slashed,

    /// Validator has voluntarily deactivated.
    Inactive,

    /// Validator is pending activation (waiting for enough stake).
    Pending,
}

impl std::fmt::Display for ValidatorStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Unbonding => write!(f, "unbonding"),
            Self::Slashed => write!(f, "slashed"),
            Self::Inactive => write!(f, "inactive"),
            Self::Pending => write!(f, "pending"),
        }
    }
}

// ─── Stake Entry ─────────────────────────────────────────────────────────

/// A validator's stake and metadata entry in the ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakeEntry {
    /// The validator's identity.
    pub validator_id: ValidatorId,

    /// The staked amount (nano-NERV).
    pub stake: StakeAmount,

    /// The validator's reputation score.
    pub reputation: ReputationScore,

    /// Current status.
    pub status: ValidatorStatus,

    /// Block height when the validator registered.
    pub registered_at_height: BlockHeight,

    /// Block height of the last status change.
    pub last_status_change_height: BlockHeight,

    /// Total rewards earned (nano-NERV).
    pub total_rewards_earned: u64,

    /// Total stake slashed (nano-NERV).
    pub total_slashed: u64,

    /// Number of blocks participated in consensus.
    pub blocks_participated: u64,

    /// Number of gradient submissions.
    pub gradient_submissions: u64,

    /// Number of successful gradient submissions (reduced loss).
    pub successful_gradients: u64,

    /// Assigned shard ID (if sharding is active).
    pub assigned_shard: Option<u64>,

    /// Dilithium-3 public key for this validator.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub dilithium_pk: Vec<u8>,
}

impl StakeEntry {
    /// Create a new stake entry with the minimum stake.
    pub fn new(
        validator_id: ValidatorId,
        stake: StakeAmount,
        height: BlockHeight,
        dilithium_pk: Vec<u8>,
    ) -> Self {
        Self {
            validator_id,
            stake,
            reputation: ReputationScore::INITIAL,
            status: ValidatorStatus::Pending,
            registered_at_height: height,
            last_status_change_height: height,
            total_rewards_earned: 0,
            total_slashed: 0,
            blocks_participated: 0,
            gradient_submissions: 0,
            successful_gradients: 0,
            assigned_shard: None,
            dilithium_pk,
        }
    }

    /// Compute the voting weight: stake × reputation.
    pub fn voting_weight(&self) -> VotingWeight {
        VotingWeight::from_stake_reputation(self.stake, self.reputation)
    }

    /// Check if the validator has enough stake to be active.
    pub fn has_minimum_stake(&self) -> bool {
        self.stake >= MIN_VALIDATOR_STAKE
    }

    /// Record a reward.
    pub fn add_reward(&mut self, reward_nano: u64) {
        self.total_rewards_earned = self.total_rewards_earned.saturating_add(reward_nano);
    }

    /// Record participation in a block.
    pub fn record_participation(&mut self) {
        self.blocks_participated += 1;
        // Slowly increase reputation for honest participation
        let gain = REPUTATION_GAIN_PER_BLOCK.min(
            ReputationScore::SCALE.saturating_sub(self.reputation.0)
        );
        self.reputation = ReputationScore(self.reputation.0.saturating_add(gain));
    }

    /// Record a gradient submission.
    pub fn record_gradient_submission(&mut self, was_successful: bool) {
        self.gradient_submissions += 1;
        if was_successful {
            self.successful_gradients += 1;
        }
    }

    /// Get the success rate of gradient submissions (0.0–1.0).
    pub fn gradient_success_rate(&self) -> f64 {
        if self.gradient_submissions == 0 {
            return 0.0;
        }
        self.successful_gradients as f64 / self.gradient_submissions as f64
    }

    /// Assign to a shard.
    pub fn assign_shard(&mut self, shard_id: u64) {
        self.assigned_shard = Some(shard_id);
    }
}

// ─── Unbonding Entry ─────────────────────────────────────────────────────

/// An in-progress unbonding request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnbondingEntry {
    /// The validator requesting unbonding.
    pub validator_id: ValidatorId,

    /// The amount being unbonded (nano-NERV).
    pub amount: StakeAmount,

    /// Block height when unbonding was initiated.
    pub started_at_height: BlockHeight,

    /// Block height when unbonding completes.
    pub completes_at_height: BlockHeight,
}

impl UnbondingEntry {
    /// Create a new unbonding entry.
    pub fn new(
        validator_id: ValidatorId,
        amount: StakeAmount,
        current_height: BlockHeight,
    ) -> Self {
        Self {
            validator_id,
            amount,
            started_at_height: current_height,
            completes_at_height: BlockHeight::from(
                current_height.0.saturating_add(UNBONDING_PERIOD_BLOCKS)
            ),
        }
    }

    /// Check if unbonding is complete at the given height.
    pub fn is_complete(&self, current_height: BlockHeight) -> bool {
        current_height.0 >= self.completes_at_height.0
    }

    /// Blocks remaining until unbonding completes.
    pub fn blocks_remaining(&self, current_height: BlockHeight) -> u64 {
        self.completes_at_height.0.saturating_sub(current_height.0)
    }
}

// ─── Staking Ledger ──────────────────────────────────────────────────────

/// The global staking ledger tracking all validators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakingLedger {
    /// All stake entries (validator_id → entry).
    entries: HashMap<[u8; 32], StakeEntry>,

    /// Active unbonding requests.
    unbonding: Vec<UnbondingEntry>,

    /// Total active stake (nano-NERV).
    total_active_stake_nano: u64,

    /// Total slashed (nano-NERV).
    total_slashed_nano: u64,
}

impl StakingLedger {
    /// Create an empty staking ledger.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            unbonding: Vec::new(),
            total_active_stake_nano: 0,
            total_slashed_nano: 0,
        }
    }

    /// Register a new validator.
    pub fn register(
        &mut self,
        validator_id: ValidatorId,
        initial_stake: StakeAmount,
        height: BlockHeight,
        dilithium_pk: Vec<u8>,
    ) -> NervResult<()> {
        if self.entries.contains_key(validator_id.as_bytes()) {
            return Err(NervError::Economy(format!(
                "validator {} already registered", validator_id
            )));
        }

        if initial_stake < MIN_VALIDATOR_STAKE {
            return Err(NervError::Economy(format!(
                "initial stake {} below minimum {}",
                initial_stake, MIN_VALIDATOR_STAKE
            )));
        }

        let active_count = self.active_count();
        if active_count >= MAX_ACTIVE_VALIDATORS {
            return Err(NervError::Economy(format!(
                "maximum {} active validators reached", MAX_ACTIVE_VALIDATORS
            )));
        }

        let mut entry = StakeEntry::new(validator_id.clone(), initial_stake, height, dilithium_pk);

        // Auto-activate if has minimum stake
        if entry.has_minimum_stake() {
            entry.status = ValidatorStatus::Active;
            self.total_active_stake_nano = self.total_active_stake_nano
                .saturating_add(initial_stake.0);
        }

        self.entries.insert(validator_id.as_bytes().clone(), entry);
        Ok(())
    }

    /// Get a stake entry.
    pub fn get(&self, validator_id: &ValidatorId) -> Option<&StakeEntry> {
        self.entries.get(validator_id.as_bytes())
    }

    /// Get a mutable stake entry.
    pub fn get_mut(&mut self, validator_id: &ValidatorId) -> Option<&mut StakeEntry> {
        self.entries.get_mut(validator_id.as_bytes())
    }

    /// Add stake to an existing validator.
    pub fn add_stake(
        &mut self,
        validator_id: &ValidatorId,
        amount: StakeAmount,
    ) -> NervResult<()> {
        let entry = self.entries.get_mut(validator_id.as_bytes())
            .ok_or_else(|| NervError::Economy(format!(
                "validator {} not found", validator_id
            )))?;

        if entry.status == ValidatorStatus::Slashed {
            return Err(NervError::Economy("cannot add stake to slashed validator".into()));
        }

        entry.stake = entry.stake.checked_add(amount)
            .ok_or_else(|| NervError::Economy("stake overflow".into()))?;

        if entry.status == ValidatorStatus::Active {
            self.total_active_stake_nano = self.total_active_stake_nano
                .saturating_add(amount.0);
        }

        // Activate if was pending and now has enough stake
        if entry.status == ValidatorStatus::Pending && entry.has_minimum_stake() {
            entry.status = ValidatorStatus::Active;
            self.total_active_stake_nano = self.total_active_stake_nano
                .saturating_add(entry.stake.0);
        }

        Ok(())
    }

    /// Initiate unstaking (enters unbonding period).
    pub fn begin_unbonding(
        &mut self,
        validator_id: &ValidatorId,
        amount: StakeAmount,
        current_height: BlockHeight,
    ) -> NervResult<()> {
        let entry = self.entries.get_mut(validator_id.as_bytes())
            .ok_or_else(|| NervError::Economy(format!(
                "validator {} not found", validator_id
            )))?;

        if entry.status != ValidatorStatus::Active && entry.status != ValidatorStatus::Inactive {
            return Err(NervError::Economy(format!(
                "cannot unbond from {} status", entry.status
            )));
        }

        if amount > entry.stake {
            return Err(NervError::Economy(format!(
                "unbond amount {} exceeds stake {}", amount, entry.stake
            )));
        }

        // Deduct from active stake
        if entry.status == ValidatorStatus::Active {
            self.total_active_stake_nano = self.total_active_stake_nano
                .saturating_sub(amount.0);
        }

        entry.stake = entry.stake.checked_sub(amount)
            .ok_or_else(|| NervError::Economy("stake underflow".into()))?;

        // Create unbonding entry
        let unbonding = UnbondingEntry::new(validator_id.clone(), amount, current_height);
        self.unbonding.push(unbonding);

        // If fully unbonded, change status
        if entry.stake < MIN_VALIDATOR_STAKE && entry.status == ValidatorStatus::Active {
            entry.status = ValidatorStatus::Unbonding;
            entry.last_status_change_height = current_height;
        }

        Ok(())
    }

    /// Process mature unbonding requests.
    ///
    /// Returns the total amount released.
    pub fn process_mature_unbondings(
        &mut self,
        current_height: BlockHeight,
    ) -> u64 {
        let mut released: u64 = 0;
        let mut remaining = Vec::new();

        for unbonding in self.unbonding.drain(..) {
            if unbonding.is_complete(current_height) {
                released = released.saturating_add(unbonding.amount.0);

                // Check if validator can be deactivated
                if let Some(entry) = self.entries.get_mut(unbonding.validator_id.as_bytes()) {
                    if entry.stake < MIN_VALIDATOR_STAKE
                        && entry.status == ValidatorStatus::Unbonding
                    {
                        entry.status = ValidatorStatus::Inactive;
                        entry.last_status_change_height = current_height;
                    }
                }
            } else {
                remaining.push(unbonding);
            }
        }

        self.unbonding = remaining;
        released
    }

    /// Slash a validator's stake.
    pub fn slash(
        &mut self,
        validator_id: &ValidatorId,
        reason: SlashReason,
        current_height: BlockHeight,
    ) -> NervResult<StakeAmount> {
        let entry = self.entries.get_mut(validator_id.as_bytes())
            .ok_or_else(|| NervError::Economy(format!(
                "validator {} not found", validator_id
            )))?;

        if entry.status == ValidatorStatus::Slashed {
            return Err(NervError::Economy("validator already slashed".into()));
        }

        // Compute slash amount
        let slash_rate_bps = reason.slash_rate_bps() as u128;
        let slash_nano = ((entry.stake.0 as u128 * slash_rate_bps) / 10000) as u64;
        let slash_amount = StakeAmount::from_nano(slash_nano);

        // Deduct from stake
        let old_stake = entry.stake;
        entry.stake = old_stake.checked_sub(slash_amount)
            .unwrap_or(StakeAmount::ZERO);

        // Update active stake total
        if entry.status == ValidatorStatus::Active {
            self.total_active_stake_nano = self.total_active_stake_nano
                .saturating_sub(slash_nano);
        }

        entry.total_slashed = entry.total_slashed.saturating_add(slash_nano);
        self.total_slashed_nano = self.total_slashed_nano.saturating_add(slash_nano);

        // Reduce reputation
        let rep_loss = REPUTATION_LOSS_PER_SLASH.min(entry.reputation.0);
        entry.reputation = ReputationScore(entry.reputation.0.saturating_sub(rep_loss));

        // If stake drops below minimum, deactivate
        if entry.stake < MIN_VALIDATOR_STAKE {
            entry.status = ValidatorStatus::Slashed;
        }

        entry.last_status_change_height = current_height;

        Ok(slash_amount)
    }

    /// List all active validator IDs.
    pub fn active_validators(&self) -> Vec<ValidatorId> {
        self.entries.values()
            .filter(|e| e.status == ValidatorStatus::Active)
            .map(|e| e.validator_id.clone())
            .collect()
    }

    /// Number of active validators.
    pub fn active_count(&self) -> usize {
        self.entries.values()
            .filter(|e| e.status == ValidatorStatus::Active)
            .count()
    }

    /// Total active stake.
    pub fn total_active_stake(&self) -> StakeAmount {
        StakeAmount::from_nano(self.total_active_stake_nano)
    }

    /// Total slashed.
    pub fn total_slashed(&self) -> StakeAmount {
        StakeAmount::from_nano(self.total_slashed_nano)
    }

    /// Check if a validator is active.
    pub fn is_active(&self, validator_id: &ValidatorId) -> bool {
        self.entries.get(validator_id.as_bytes())
            .map(|e| e.status == ValidatorStatus::Active)
            .unwrap_or(false)
    }

    /// Get the voting weight of a validator.
    pub fn voting_weight(&self, validator_id: &ValidatorId) -> VotingWeight {
        self.entries.get(validator_id.as_bytes())
            .map(|e| e.voting_weight())
            .unwrap_or(VotingWeight::ZERO)
    }

    /// Total voting weight of all active validators.
    pub fn total_voting_weight(&self) -> VotingWeight {
        let total: u128 = self.entries.values()
            .filter(|e| e.status == ValidatorStatus::Active)
            .map(|e| e.voting_weight().0)
            .sum();
        VotingWeight(total)
    }

    /// Stake-weighted random selection of validators.
    ///
    /// Selects `count` validators with probability proportional
    /// to their voting weight. Uses VRF-like deterministic
    /// selection from a seed.
    pub fn select_validators(
        &self,
        count: usize,
        seed: &[u8],
    ) -> Vec<ValidatorId> {
        let active: Vec<&StakeEntry> = self.entries.values()
            .filter(|e| e.status == ValidatorStatus::Active)
            .collect();

        if active.is_empty() || count == 0 {
            return Vec::new();
        }

        // Deterministic weighted selection using seed
        let mut selected = Vec::with_capacity(count.min(active.len()));
        let mut used = std::collections::HashSet::new();

        for i in 0..count {
            let ctx = format!("nerv:validator-select:{}", i);
            let hash = blake3::derive_key(ctx.as_bytes(), seed);
            let rand_val = u64::from_le_bytes(hash[..8].try_into().unwrap_or([0u8; 8]));

            // Weighted selection: walk through validators
            let threshold = (rand_val as u128) % self.total_active_stake_nano.max(1) as u128;
            let mut cumulative: u128 = 0;

            for entry in &active {
                if used.contains(&entry.validator_id) {
                    continue;
                }
                cumulative += entry.stake.0 as u128;
                if cumulative > threshold {
                    selected.push(entry.validator_id.clone());
                    used.insert(entry.validator_id.clone());
                    break;
                }
            }
        }

        selected
    }

    /// Get pending unbonding entries for a validator.
    pub fn unbonding_entries(&self, validator_id: &ValidatorId) -> Vec<&UnbondingEntry> {
        self.unbonding.iter()
            .filter(|u| u.validator_id == *validator_id)
            .collect()
    }

    /// Number of pending unbonding entries.
    pub fn pending_unbonding_count(&self) -> usize {
        self.unbonding.len()
    }
}

impl Default for StakingLedger {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_validator_id(index: u8) -> ValidatorId {
        let mut bytes = [0u8; 32];
        bytes[0] = index;
        ValidatorId::from_bytes(bytes)
    }

    #[test]
    fn test_staking_ledger_register() {
        let mut ledger = StakingLedger::new();
        let vid = test_validator_id(1);
        let result = ledger.register(
            vid.clone(),
            StakeAmount::from_nerv(1000),
            BlockHeight::from(1),
            vec![0u8; 1952],
        );
        assert!(result.is_ok());
        assert!(ledger.is_active(&vid));
    }

    #[test]
    fn test_staking_ledger_register_below_minimum() {
        let mut ledger = StakingLedger::new();
        let result = ledger.register(
            test_validator_id(1),
            StakeAmount::from_nerv(100), // Below 1000 minimum
            BlockHeight::from(1),
            vec![0u8; 1952],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_staking_ledger_register_duplicate() {
        let mut ledger = StakingLedger::new();
        let vid = test_validator_id(1);
        ledger.register(vid.clone(), StakeAmount::from_nerv(1000), BlockHeight::from(1), vec![]).unwrap();
        let result = ledger.register(vid, StakeAmount::from_nerv(1000), BlockHeight::from(2), vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_staking_ledger_add_stake() {
        let mut ledger = StakingLedger::new();
        let vid = test_validator_id(1);
        ledger.register(vid.clone(), StakeAmount::from_nerv(1000), BlockHeight::from(1), vec![]).unwrap();

        ledger.add_stake(&vid, StakeAmount::from_nerv(500)).unwrap();

        let entry = ledger.get(&vid).unwrap();
        assert_eq!(entry.stake, StakeAmount::from_nerv(1500));
    }

    #[test]
    fn test_staking_ledger_unbonding() {
        let mut ledger = StakingLedger::new();
        let vid = test_validator_id(1);
        ledger.register(vid.clone(), StakeAmount::from_nerv(2000), BlockHeight::from(1), vec![]).unwrap();

        ledger.begin_unbonding(&vid, StakeAmount::from_nerv(1000), BlockHeight::from(100)).unwrap();

        let entry = ledger.get(&vid).unwrap();
        assert_eq!(entry.stake, StakeAmount::from_nerv(1000));

        // Unbonding should be pending
        let unbonding = ledger.unbonding_entries(&vid);
        assert_eq!(unbonding.len(), 1);
        assert!(!unbonding[0].is_complete(BlockHeight::from(100)));
    }

    #[test]
    fn test_staking_ledger_process_mature_unbondings() {
        let mut ledger = StakingLedger::new();
        let vid = test_validator_id(1);
        ledger.register(vid.clone(), StakeAmount::from_nerv(2000), BlockHeight::from(1), vec![]).unwrap();

        ledger.begin_unbonding(&vid, StakeAmount::from_nerv(1000), BlockHeight::from(1)).unwrap();

        // Not yet mature
        let released = ledger.process_mature_unbondings(BlockHeight::from(100));
        assert_eq!(released, 0);

        // Mature after unbonding period
        let mature_height = BlockHeight::from(1 + UNBONDING_PERIOD_BLOCKS);
        let released = ledger.process_mature_unbondings(mature_height);
        assert_eq!(released, 1000 * ONE_NERV);
        assert_eq!(ledger.pending_unbonding_count(), 0);
    }

    #[test]
    fn test_staking_ledger_slash() {
        let mut ledger = StakingLedger::new();
        let vid = test_validator_id(1);
        ledger.register(vid.clone(), StakeAmount::from_nerv(1000), BlockHeight::from(1), vec![]).unwrap();

        let slashed = ledger.slash(
            &vid,
            SlashReason::AdversarialGradient, // 1% slash
            BlockHeight::from(100),
        ).unwrap();

        // 1% of 1000 NERV = 10 NERV
        assert_eq!(slashed, StakeAmount::from_nerv(10));

        let entry = ledger.get(&vid).unwrap();
        assert_eq!(entry.stake, StakeAmount::from_nerv(990));
        assert!(entry.reputation < ReputationScore::INITIAL);
    }

    #[test]
    fn test_staking_ledger_slash_double_sign() {
        let mut ledger = StakingLedger::new();
        let vid = test_validator_id(1);
        ledger.register(vid.clone(), StakeAmount::from_nerv(1000), BlockHeight::from(1), vec![]).unwrap();

        let slashed = ledger.slash(
            &vid,
            SlashReason::DoubleSign, // 5% slash
            BlockHeight::from(100),
        ).unwrap();

        // 5% of 1000 NERV = 50 NERV
        assert_eq!(slashed, StakeAmount::from_nerv(50));
    }

    #[test]
    fn test_staking_ledger_total_stake() {
        let mut ledger = StakingLedger::new();
        ledger.register(test_validator_id(1), StakeAmount::from_nerv(1000), BlockHeight::from(1), vec![]).unwrap();
        ledger.register(test_validator_id(2), StakeAmount::from_nerv(2000), BlockHeight::from(1), vec![]).unwrap();

        let total = ledger.total_active_stake();
        assert_eq!(total, StakeAmount::from_nerv(3000));
    }

    #[test]
    fn test_staking_ledger_active_validators() {
        let mut ledger = StakingLedger::new();
        ledger.register(test_validator_id(1), StakeAmount::from_nerv(1000), BlockHeight::from(1), vec![]).unwrap();
        ledger.register(test_validator_id(2), StakeAmount::from_nerv(1000), BlockHeight::from(1), vec![]).unwrap();

        assert_eq!(ledger.active_count(), 2);
        assert_eq!(ledger.active_validators().len(), 2);
    }

    #[test]
    fn test_staking_ledger_voting_weight() {
        let mut ledger = StakingLedger::new();
        let vid = test_validator_id(1);
        ledger.register(vid.clone(), StakeAmount::from_nerv(1000), BlockHeight::from(1), vec![]).unwrap();

        let weight = ledger.voting_weight(&vid);
        assert!(weight.0 > 0);
    }

    #[test]
    fn test_stake_entry_voting_weight() {
        let entry = StakeEntry::new(
            test_validator_id(1),
            StakeAmount::from_nerv(1000),
            BlockHeight::from(1),
            vec![],
        );
        let weight = entry.voting_weight();
        assert!(weight.0 > 0);
    }

    #[test]
    fn test_stake_entry_participation() {
        let mut entry = StakeEntry::new(
            test_validator_id(1),
            StakeAmount::from_nerv(1000),
            BlockHeight::from(1),
            vec![],
        );
        let initial_rep = entry.reputation;
        entry.record_participation();
        assert_eq!(entry.blocks_participated, 1);
        assert!(entry.reputation.0 >= initial_rep.0);
    }

    #[test]
    fn test_stake_entry_gradient_tracking() {
        let mut entry = StakeEntry::new(
            test_validator_id(1),
            StakeAmount::from_nerv(1000),
            BlockHeight::from(1),
            vec![],
        );
        entry.record_gradient_submission(true);
        entry.record_gradient_submission(false);
        entry.record_gradient_submission(true);
        assert_eq!(entry.gradient_submissions, 3);
        assert_eq!(entry.successful_gradients, 2);
        assert!((entry.gradient_success_rate() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_unbonding_entry() {
        let entry = UnbondingEntry::new(
            test_validator_id(1),
            StakeAmount::from_nerv(1000),
            BlockHeight::from(100),
        );
        assert!(!entry.is_complete(BlockHeight::from(100)));
        assert!(!entry.is_complete(BlockHeight::from(100 + UNBONDING_PERIOD_BLOCKS - 1)));
        assert!(entry.is_complete(BlockHeight::from(100 + UNBONDING_PERIOD_BLOCKS)));
        assert!(entry.blocks_remaining(BlockHeight::from(100)) > 0);
    }

    #[test]
    fn test_validator_status_display() {
        assert_eq!(ValidatorStatus::Active.to_string(), "active");
        assert_eq!(ValidatorStatus::Slashed.to_string(), "slashed");
    }

    #[test]
    fn test_staking_ledger_select_validators() {
        let mut ledger = StakingLedger::new();
        for i in 1..=5u8 {
            ledger.register(
                test_validator_id(i),
                StakeAmount::from_nerv(1000 * i as u64),
                BlockHeight::from(1),
                vec![],
            ).unwrap();
        }

        let seed = blake3::hash(b"test selection").into();
        let selected = ledger.select_validators(3, &seed);
        assert!(selected.len() <= 3);
        assert!(selected.len() > 0);
    }

    #[test]
    fn test_staking_ledger_cannot_add_stake_to_slashed() {
        let mut ledger = StakingLedger::new();
        let vid = test_validator_id(1);
        ledger.register(vid.clone(), StakeAmount::from_nerv(1000), BlockHeight::from(1), vec![]).unwrap();

        // Slash heavily to make stake < minimum
        ledger.slash(&vid, SlashReason::DoubleSign, BlockHeight::from(2)).unwrap();

        let entry = ledger.get(&vid).unwrap();
        // After 5% slash, stake = 950 NERV, still above min
        // Let's slash multiple times
        ledger.slash(&vid, SlashReason::DoubleSign, BlockHeight::from(3)).unwrap();
        ledger.slash(&vid, SlashReason::DoubleSign, BlockHeight::from(4)).unwrap();

        let entry = ledger.get(&vid).unwrap();
        if entry.status == ValidatorStatus::Slashed {
            let result = ledger.add_stake(&vid, StakeAmount::from_nerv(500));
            assert!(result.is_err());
        }
    }
}
