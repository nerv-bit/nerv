//! Vesting Schedules — Cliff + Linear Release.
//!
//! NERV uses two vesting patterns:
//!
//! | Category | Pattern | Duration | Cliff |
//! |----------|---------|----------|-------|
//! | Visionary | Linear | 6 months | None |
//! | Code & Research | Linear + Cliff | 1 year | 3 months |
//! | Early Donors | Linear + Cliff | 1 year | 6 months |
//!
//! During the cliff period, NO tokens are released. After the
//! cliff, tokens vest linearly until the full amount is released.

use crate::{
    BlockHeight,
    NervError, NervResult,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Vesting Schedule ────────────────────────────────────────────────────

/// A vesting schedule with optional cliff and linear release.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VestingSchedule {
    /// Block height when vesting begins.
    pub start_height: BlockHeight,

    /// Total vesting duration in blocks (including cliff).
    pub duration_blocks: u64,

    /// Cliff duration in blocks (0 = no cliff).
    pub cliff_blocks: u(0),
}

impl VestingSchedule {
    /// Create a linear vesting schedule (no cliff).
    pub fn linear(start_height: BlockHeight, duration_blocks: u64, cliff_blocks: u64) -> Self {
        Self {
            start_height,
            duration_blocks,
            cliff_blocks: cliff_blocks.min(duration_blocks),
        }
    }

    /// Create a linear vesting schedule with a cliff.
    pub fn linear_with_cliff(
        start_height: BlockHeight,
        duration_blocks: u64,
        cliff_blocks: u64,
    ) -> Self {
        Self::linear(start_height, duration_blocks, cliff_blocks)
    }

    /// Create an immediate vesting schedule (no lockup).
    pub fn immediate(start_height: BlockHeight) -> Self {
        Self {
            start_height,
            duration_blocks: 0,
            cliff_blocks: 0,
        }
    }

    /// Create a no-vesting schedule (tokens locked forever).
    pub fn locked(start_height: BlockHeight) -> Self {
        Self {
            start_height,
            duration_blocks: u64::MAX,
            cliff_blocks: 0,
        }
    }

    /// Compute the vested amount at a given block height.
    pub fn vested_amount(&self, total_amount: u64, height: BlockHeight) -> u64 {
        if total_amount == 0 {
            return 0;
        }

        if self.duration_blocks == 0 {
            return if height.0 >= self.start_height.0 { total_amount } else { 0 };
        }

        if height.0 < self.start_height.0 {
            return 0;
        }

        let elapsed = height.0 - self.start_height.0;

        if elapsed < self.cliff_blocks {
            return 0;
        }

        if elapsed >= self.duration_blocks {
            return total_amount;
        }

        let vesting_elapsed = elapsed - self.cliff_blocks;
        let vesting_duration = self.duration_blocks.saturating_sub(self.cliff_blocks);

        if vesting_duration == 0 {
            return total_amount;
        }

        let vested = (total_amount as u128 * vesting_elapsed as u128
            / vesting_duration as u128) as u64;
        vested.min(total_amount)
    }

    /// Compute the locked (unvested) amount.
    pub fn locked_amount(&self, total_amount: u64, height: BlockHeight) -> u64 {
        total_amount.saturating_sub(self.vested_amount(total_amount, height))
    }

    /// Check if vesting has fully completed.
    pub fn is_complete(&self, height: BlockHeight) -> bool {
        if self.duration_blocks == 0 {
            return height.0 >= self.start_height.0;
        }
        let end = self.start_height.0.saturating_add(self.duration_blocks);
        height.0 >= end
    }

    /// Check if we're in the cliff period.
    pub fn is_in_cliff(&self, height: BlockHeight) -> bool {
        if self.cliff_blocks == 0 || height.0 < self.start_height.0 {
            return false;
        }
        let elapsed = height.0 - self.start_height.0;
        elapsed < self.cliff_blocks
    }

    /// Check if vesting has started.
    pub fn has_started(&self, height: BlockHeight) -> bool {
        height.0 >= self.start_height.0
    }

    /// Block height when the cliff ends.
    pub fn cliff_end_height(&self) -> BlockHeight {
        BlockHeight::from(self.start_height.0.saturating_add(self.cliff_blocks))
    }

    /// Block height when vesting fully completes.
    pub fn completion_height(&self) -> BlockHeight {
        BlockHeight::from(self.start_height.0.saturating_add(self.duration_blocks))
    }

    /// Blocks remaining until full vesting.
    pub fn blocks_remaining(&self, height: BlockHeight) -> u64 {
        self.completion_height().0.saturating_sub(height.0)
    }

    /// Vesting progress as a fraction (0.0–1.0).
    pub fn progress(&self, height: BlockHeight) -> f64 {
        if self.duration_blocks == 0 {
            return if height.0 >= self.start_height.0 { 1.0 } else { 0.0 };
        }
        let vested_frac = self.vested_amount(1_000_000_000, height);
        vested_frac as f64 / 1_000_000_000.0
    }
}

// ─── Vesting Entry ───────────────────────────────────────────────────────

/// A vesting entry for a specific beneficiary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VestingEntry {
    /// Unique identifier.
    pub id: u64,

    /// The beneficiary.
    pub beneficiary: String,

    /// The allocation category.
    pub category: String,

    /// Total amount to vest (nano-NERV).
    pub total_amount_nano: u64,

    /// Amount already claimed/released (nano-NERV).
    pub claimed_nano: u64,

    /// The vesting schedule.
    pub schedule: VestingSchedule,
}

impl VestingEntry {
    /// Create a new vesting entry.
    pub fn new(
        id: u64,
        beneficiary: String,
        category: String,
        total_amount_nano: u64,
        schedule: VestingSchedule,
    ) -> Self {
        Self {
            id, beneficiary, category, total_amount_nano, claimed_nano: 0, schedule,
        }
    }

    /// Compute the vested but unclaimed amount.
    pub fn releasable(&self, height: BlockHeight) -> u64 {
        let vested = self.schedule.vested_amount(self.total_amount_nano, height);
        vested.saturating_sub(self.claimed_nano)
    }

    /// Compute the total locked amount.
    pub fn locked(&self, height: BlockHeight) -> u64 {
        self.total_amount_nano
            .saturating_sub(self.schedule.vested_amount(self.total_amount_nano, height))
    }

    /// Claim the releasable amount. Returns amount claimed.
    pub fn claim(&mut self, height: BlockHeight) -> u64 {
        let amount = self.releasable(height);
        self.claimed_nano = self.claimed_nano.saturating_add(amount);
        amount
    }

    /// Check if fully claimed.
    pub fn is_fully_claimed(&self, height: BlockHeight) -> bool {
        self.schedule.is_complete(height) && self.claimed_nano >= self.total_amount_nano
    }
}

// ─── Vesting Registry ────────────────────────────────────────────────────

/// Registry of all vesting entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VestingRegistry {
    /// Entries by ID.
    entries: HashMap<u64, VestingEntry>,

    /// Entry IDs by beneficiary.
    by_beneficiary: HashMap<String, Vec<u64>>,

    /// Next entry ID.
    next_id: u64,
}

impl VestingRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            by_beneficiary: HashMap::new(),
            next_id: 1,
        }
    }

    /// Add a vesting entry.
    pub fn add_entry(&mut self, entry: VestingEntry) -> NervResult<()> {
        if self.entries.contains_key(&entry.id) {
            return Err(NervError::Tokenomics(format!(
                "vesting entry {} already exists", entry.id
            )));
        }
        let beneficiary = entry.beneficiary.clone();
        let id = entry.id;
        self.entries.insert(id, entry);
        self.by_beneficiary.entry(beneficiary).or_default().push(id);
        if id >= self.next_id {
            self.next_id = id + 1;
        }
        Ok(())
    }

    /// Create and add a new entry. Returns the entry ID.
    pub fn create_entry(
        &mut self,
        beneficiary: String,
        category: String,
        total_amount_nano: u64,
        schedule: VestingSchedule,
    ) -> NervResult<u64> {
        let id = self.next_id;
        self.next_id += 1;
        let entry = VestingEntry::new(id, beneficiary, category, total_amount_nano, schedule);
        self.add_entry(entry)?;
        Ok(id)
    }

    /// Get an entry by ID.
    pub fn get(&self, id: u64) -> Option<&VestingEntry> {
        self.entries.get(&id)
    }

    /// Get a mutable entry by ID.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut VestingEntry> {
        self.entries.get_mut(&id)
    }

    /// Get all entries for a beneficiary.
    pub fn entries_for_beneficiary(&self, beneficiary: &str) -> Vec<&VestingEntry> {
        self.by_beneficiary.get(beneficiary)
            .map(|ids| ids.iter().filter_map(|id| self.entries.get(id)).collect())
            .unwrap_or_default()
    }

    /// Total vested amount across all entries at a given height.
    pub fn total_vested_at(&self, height: BlockHeight) -> u64 {
        self.entries.values()
            .map(|e| e.schedule.vested_amount(e.total_amount_nano, height))
            .sum()
    }

    /// Total claimed across all entries.
    pub fn total_claimed(&self) -> u64 {
        self.entries.values().map(|e| e.claimed_nano).sum()
    }

    /// Total locked across all entries.
    pub fn total_locked_at(&self, height: BlockHeight) -> u64 {
        self.entries.values()
            .map(|e| e.locked(height))
            .sum()
    }

    /// Claim all releasable for an entry. Returns amount claimed.
    pub fn claim(&mut self, id: u64, height: BlockHeight) -> u64 {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.claim(height)
        } else {
            0
        }
    }

    /// Claim all releasable for a beneficiary. Returns total claimed.
    pub fn claim_for_beneficiary(&mut self, beneficiary: &str, height: BlockHeight) -> u64 {
        let ids: Vec<u64> = self.by_beneficiary.get(beneficiary)
            .cloned()
            .unwrap_or_default();

        let mut total = 0;
        for id in ids {
            total += self.claim(id, height);
        }
        total
    }

    /// Number of vesting entries.
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Number of entries still vesting (not complete).
    pub fn active_count(&self, height: BlockHeight) -> usize {
        self.entries.values()
            .filter(|e| !e.schedule.is_complete(height))
            .count()
    }
}

impl Default for VestingRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vesting_schedule_linear_before_start() {
        let schedule = VestingSchedule::linear(BlockHeight::from(100), 1000, 0);
        assert_eq!(schedule.vested_amount(1000, BlockHeight::from(0)), 0);
    }

    #[test]
    fn test_vesting_schedule_linear_at_start() {
        let schedule = VestingSchedule::linear(BlockHeight::from(100), 1000, 0);
        assert_eq!(schedule.vested_amount(1000, BlockHeight::from(100)), 0);
    }

    #[test]
    fn test_vesting_schedule_linear_halfway() {
        let schedule = VestingSchedule::linear(BlockHeight::from(0), 1000, 0);
        assert_eq!(schedule.vested_amount(1000, BlockHeight::from(500)), 500);
    }

    #[test]
    fn test_vesting_schedule_linear_complete() {
        let schedule = VestingSchedule::linear(BlockHeight::from(0), 1000, 0);
        assert_eq!(schedule.vested_amount(1000, BlockHeight::from(1000)), 1000);
        assert!(schedule.is_complete(BlockHeight::from(1000)));
    }

    #[test]
    fn test_vesting_schedule_with_cliff_during_cliff() {
        let schedule = VestingSchedule::linear_with_cliff(BlockHeight::from(0), 1000, 200);
        assert_eq!(schedule.vested_amount(1000, BlockHeight::from(100)), 0);
        assert!(schedule.is_in_cliff(BlockHeight::from(100)));
    }

    #[test]
    fn test_vesting_schedule_with_cliff_after_cliff() {
        let schedule = VestingSchedule::linear_with_cliff(BlockHeight::from(0), 1000, 200);
        // After cliff at height 600: elapsed=600, after_cliff=400, duration=800
        assert_eq!(schedule.vested_amount(1000, BlockHeight::from(600)), 500);
    }

    #[test]
    fn test_vesting_schedule_immediate() {
        let schedule = VestingSchedule::immediate(BlockHeight::from(0));
        assert_eq!(schedule.vested_amount(1000, BlockHeight::from(0)), 1000);
    }

    #[test]
    fn test_vesting_schedule_locked() {
        let schedule = VestingSchedule::locked(BlockHeight::from(0));
        assert_eq!(schedule.vested_amount(1000, BlockHeight::from(1000)), 0);
    }

    #[test]
    fn test_vesting_schedule_progress() {
        let schedule = VestingSchedule::linear(BlockHeight::from(0), 1000, 0);
        assert!((schedule.progress(BlockHeight::from(500)) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_vesting_entry_claim() {
        let mut entry = VestingEntry::new(
            1, "alice".into(), "visionary".into(), 1000,
            VestingSchedule::linear(BlockHeight::from(0), 1000, 0),
        );
        assert_eq!(entry.claim(BlockHeight::from(500)), 500);
        assert_eq!(entry.claim(BlockHeight::from(500)), 0); // Already claimed
        assert_eq!(entry.claim(BlockHeight::from(800)), 300);
    }

    #[test]
    fn test_vesting_registry_create() {
        let mut registry = VestingRegistry::new();
        let id = registry.create_entry(
            "alice".into(), "visionary".into(), 1000,
            VestingSchedule::linear(BlockHeight::from(0), 1000, 0),
        ).unwrap();
        assert_eq!(id, 1);
        assert_eq!(registry.count(), 1);
    }

    #[test]
    fn test_vesting_registry_by_beneficiary() {
        let mut registry = VestingRegistry::new();
        registry.create_entry(
            "alice".into(), "visionary".into(), 1000,
            VestingSchedule::linear(BlockHeight::from(0), 1000, 0),
        ).unwrap();
        registry.create_entry(
            "alice".into(), "research".into(), 500,
            VestingSchedule::linear(BlockHeight::from(0), 500, 0),
        ).unwrap();
        assert_eq!(registry.entries_for_beneficiary("alice").len(), 2);
    }

    #[test]
    fn test_vesting_registry_total_vested() {
        let mut registry = VestingRegistry::new();
        registry.create_entry(
            "alice".into(), "visionary".into(), 1000,
            VestingSchedule::linear(BlockHeight::from(0), 1000, 0),
        ).unwrap();
        registry.create_entry(
            "bob".into(), "donor".into(), 2000,
            VestingSchedule::linear(BlockHeight::from(0), 2000, 0),
        ).unwrap();
       =       let total = registry.total_vested_at(BlockHeight::from(500));
        assert_eq!(total, 1000); // 500 + 500
    }

    #[test]
    fn test_vesting_registry_claim() {
        let mut registry = VestingRegistry::new();
        registry.create_entry(
            "alice".into(), "visionary".into(), 1000,
            VestingSchedule::linear(BlockHeight::from(0), 1000, 0),
        ).unwrap();
        assert_eq!(registry.claim(1, BlockHeight::from(500)), 500);
    }

    #[test]
    fn test_vesting_schedule_cliff_end_and_completion() {
        let schedule = VestingSchedule::linear_with_cliff(BlockHeight::from(100), 1000, 200);
        assert_eq!(schedule.cliff_end_height(), BlockHeight::from(300));
        assert_eq!(schedule.completion_height(), BlockHeight::from(1100));
    }

    #[test]
    fn test_vesting_schedule_blocks_remaining() {
        let schedule = VestingSchedule::linear(BlockHeight::from(100), 1000, 0);
        assert_eq!(schedule.blocks_remaining(BlockHeight::from(100)), 1000);
        assert_eq!(schedule.blocks_remaining(BlockHeight::from(600)), 500);
    }

    #[test]
    fn test_vesting_registry_claim_for_beneficiary() {
        let mut registry = VestingRegistry::new();
        registry.create_entry(
            "alice".into(), "visionary".into(), 1000,
            VestingSchedule::linear(BlockHeight::from(0), 1000, 0),
        ).unwrap();
        registry.create_entry(
            "alice".into(), "research".into(), 500,
            VestingSchedule::linear(BlockHeight::from(0), 500, 0),
        ).unwrap();

        let claimed = registry.claim_for_beneficiary("alice", BlockHeight::from(250));
        // Entry 1: 250/1000 * 1000 = 250; Entry 2: 250/500 * 500 = 250
        assert_eq@       assert_eq!(claimed, 500);
    }
}
