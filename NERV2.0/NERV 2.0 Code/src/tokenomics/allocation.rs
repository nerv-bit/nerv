//! Genesis Allocation Management.
//!
//! Tracks the five genesis allocation categories and verifies
//! that they sum to the total supply. Each allocation may have
//! an associated vesting schedule.

use crate::{
    TOTAL_SUPPLY_NANO, ONE_NERV,
    BlockHeight,
    NervError, NervResult,
};
use crate::tokenomics::vesting::VestingSchedule;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Allocation Category ─────────────────────────────────────────────────

/// The five genesis allocation categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AllocationCategory {
    /// Useful-Work Pool (80%) — emitted over time via gradient rewards.
    UsefulWorkPool,

    /// Visionary (5%) — originator allocation, 6-month linear vest.
    Visionary,

    /// Code & Research (5%) — core contributors, 1-year vest with 3-month cliff.
    CodeResearch,

    /// Early Donors (5%) — hardware/audit capital, 1-year vest with 6-month cliff.
    EarlyDonors,

    /// Treasury (5%) — community-governed grants and protocol upgrades.
    Treasury,
}

impl AllocationCategory {
    /// Get the human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::UsefulWorkPool => "Useful-Work Pool",
            Self::Visionary => "Visionary",
            Self::CodeResearch => "Code & Research",
            Self::EarlyDonors => "Early Donors",
            Self::Treasury => "Treasury",
        }
    }

    /// Get the default basis points (fraction of total supply).
    pub fn default_bps(&self) -> u64 {
        match self {
            Self::UsefulWorkPool => 8000,
            Self::Visionary => 500,
            Self::CodeResearch => 500,
            Self::EarlyDonors => 500,
            Self::Treasury => 500,
        }
    }

    /// Get the default vesting schedule for this category.
    pub fn default_vesting(&self) -> Option<VestingSchedule> {
        match self {
            Self::UsefulWorkPool => None, // Emitted over time, not vested
            Self::Visionary => Some(VestingSchedule::linear(
                BlockHeight::GENESIS,
                1_296_000, // ~6 months at 400ms blocks
                0,
            )),
            Self::CodeResearch => Some(VestingSchedule::linear_with_cliff(
                BlockHeight::GENESIS,
                2_592_000, // ~1 year
                648_000,   // ~3 months cliff
            )),
            Self::EarlyDonors => Some(VestingSchedule::linear_with_cliff(
                BlockHeight::GENESIS,
                2_592_000, // ~1 year
                1_296_000, // ~6 months cliff
            )),
            Self::Treasury => None, // Community-governed, no auto vesting
        }
    }

    /// All categories in order.
    pub fn all() -> &'static [AllocationCategory] {
        &[
            Self::UsefulWorkPool,
            Self::Visionary,
            Self::CodeResearch,
            Self::EarlyDonors,
            Self::Treasury,
        ]
    }
}

impl std::fmt::Display for AllocationCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ─── Allocation Entry ────────────────────────────────────────────────────

/// A single genesis allocation entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationEntry {
    /// The allocation category.
    pub category: AllocationCategory,

    /// Total allocated amount (nano-NERV).
    pub amount_nano: u64,

    /// Amount already released/emitted (nano-NERV).
    pub released_nano: u64,

    /// Optional vesting schedule.
    pub vesting: Option<VestingSchedule>,

    /// Recipient addresses (if applicable).
    /// Empty for Useful-Work Pool (distributed to validators).
    pub recipients: HashMap<String, u64>, // address → amount_nano
}

impl AllocationEntry {
    /// Create a new allocation entry.
    pub fn new(
        category: AllocationCategory,
        amount_nano: u64,
        vesting: Option<VestingSchedule>,
    ) -> Self {
        Self {
            category,
            amount_nano,
            released_nano: 0,
            vesting,
            recipients: HashMap::new(),
        }
    }

    /// Create with default vesting for the category.
    pub fn with_default_vesting(category: AllocationCategory, amount_nano: u64) -> Self {
        let vesting = category.default_vesting();
        Self::new(category, amount_nano, vesting)
    }

    /// Compute the locked (unreleased) amount at a given height.
    pub fn locked_at(&self, height: BlockHeight) -> u64 {
        match &self.vesting {
            Some(schedule) => {
                let vested = schedule.vested_amount(self.amount_nano, height);
                self.amount_nano.saturating_sub(vested)
            }
            None => {
                // No vesting: locked = total - released (for Useful-Work Pool)
                self.amount_nano.saturating_sub(self.released_nano)
            }
        }
    }

    /// Compute the vested/releasable amount at a given height.
    pub fn vested_at(&self, height: BlockHeight) -> u64 {
        match &self.vesting {
            Some(schedule) => schedule.vested_amount(self.amount_nano, height),
            None => self.released_nano,
        }
    }

    /// Record a release (emission or vesting release).
    pub fn release(&mut self, amount: u64) -> NervResult<()> {
        let new_released = self.released_nano.checked_add(amount)
            .ok_or_else(|| NervError::Tokenomics("release overflow".into()))?;

        if new_released > self.amount_nano {
            return Err(NervError::Tokenomics(format!(
                "release {} exceeds allocation {} for {}",
                new_released, self.amount_nano, self.category
            )));
        }

        self.released_nano = new_released;
        Ok(())
    }

    /// Add a recipient to this allocation.
    pub fn add_recipient(&mut self, address: String, amount: u64) -> NervResult<()> {
        let current = self.recipients.get(&address).copied().unwrap_or(0);
        let new_total = current.checked_add(amount)
            .ok_or_else(|| NervError::Tokenomics("recipient amount overflow".into()))?;

        let total_recipients: u64 = self.recipients.values().sum();
        if total_recipients.saturating_sub(current).saturating_add(new_total) > self.amount_nano {
            return Err(NervError::Tokenomics(
                "total recipients exceed allocation amount".into()
            ));
        }

        self.recipients.insert(address, new_total);
        Ok(())
    }

    /// Fraction released (0.0–1.0).
    pub fn released_fraction(&self) -> f64 {
        if self.amount_nano == 0 {
            return 0.0;
        }
        self.released_nano as f64 / self.amount_nano as f64
    }
}

// ─── Allocation Registry ─────────────────────────────────────────────────

/// Registry of all genesis allocations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationRegistry {
    /// Allocations by category.
    allocations: HashMap<String, AllocationEntry>,

    /// Total supply (nano-NERV).
    total_supply_nano: u64,
}

impl AllocationRegistry {
    /// Create a new empty registry.
    pub fn new(total_supply_nano: u64) -> Self {
        Self {
            allocations: HashMap::new(),
            total_supply_nano,
        }
    }

    /// Add an allocation entry.
    pub fn add_allocation(&mut self, entry: AllocationEntry) -> NervResult<()> {
        let key = entry.category.name().to_string();
        if self.allocations.contains_key(&key) {
            return Err(NervError::Tokenomics(format!(
                "duplicate allocation for {}", entry.category
            )));
        }
        self.allocations.insert(key, entry);
        Ok(())
    }

    /// Get the total amount for a category.
    pub fn category_amount(&self, category: AllocationCategory) -> u64 {
        self.allocations.get(category.name())
            .map(|e| e.amount_nano)
            .unwrap_or(0)
    }

    /// Get the released amount for a category.
    pub fn category_released(&self, category: AllocationCategory) -> u64 {
        self.allocations.get(category.name())
            .map(|e| e.released_nano)
            .unwrap_or(0)
    }

    /// Get a mutable reference to an allocation entry.
    pub fn get_mut(&mut self, category: AllocationCategory) -> Option<&mut AllocationEntry> {
        self.allocations.get_mut(category.name())
    }

    /// Get an allocation entry.
    pub fn get(&self, category: AllocationCategory) -> Option<&AllocationEntry> {
        self.allocations.get(category.name())
    }

    /// Total allocated across all categories.
    pub fn total_allocated(&self) -> u64 {
        self.allocations.values().map(|e| e.amount_nano).sum()
    }

    /// Total released across all categories.
    pub fn total_released(&self) -> u64 {
        self.allocations.values().map(|e| e.released_nano).sum()
    }

    /// Total locked at a given height.
    pub fn total_locked_at(&self, height: BlockHeight) -> u64 {
        self.allocations.values().map(|e| e.locked_at(height)).sum()
    }

    /// Total vested at a given height.
    pub fn total_vested_at(&self, height: BlockHeight) -> u64 {
        self.allocations.values().map(|e| e.vested_at(height)).sum()
    }

    /// Verify that allocations sum to the expected total.
    pub fn verify_total(&self, expected: u64) -> NervResult<()> {
        let actual = self.total_allocated();
        if actual != expected {
            return Err(NervError::Tokenomics(format!(
                "allocations sum to {} but expected {}", actual, expected
            )));
        }
        Ok(())
    }

    /// Release amount from a category (for emission).
    pub fn release_from(&mut self, category: AllocationCategory, amount: u64) -> NervResult<()> {
        let entry = self.allocations.get_mut(category.name())
            .ok_or_else(|| NervError::Tokenomics(format!(
                "no allocation for {}", category
            )))?;
        entry.release(amount)
    }

    /// Number of allocation categories.
    pub fn count(&self) -> usize {
        self.allocations.len()
    }

    /// List all categories with their amounts.
    pub fn summary(&self) -> Vec<(AllocationCategory, u64, u64)> {
        AllocationCategory::all().iter().map(|&cat| {
            let amount = self.category_amount(cat);
            let released = self.category_released(cat);
            (cat, amount, released)
        }).collect()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_category_names() {
        assert_eq!(AllocationCategory::UsefulWorkPool.name(), "Useful-Work Pool");
        assert_eq!(AllocationCategory::Visionary.name(), "Visionary");
    }

    #[test]
    fn test_allocation_category_bps() {
        let total: u64 = AllocationCategory::all().iter()
            .map(|c| c.default_bps())
            .sum();
        assert_eq!(total, 10000); // Must sum to 100%
    }

    #[test]
    fn test_allocation_entry_new() {
        let entry = AllocationEntry::new(
            AllocationCategory::Visionary,
            500 * ONE_NERV,
            None,
        );
        assert_eq!(entry.amount_nano, 500 * ONE_NERV);
        assert_eq!(entry.released_nano, 0);
    }

    #[test]
    fn test_allocation_entry_release() {
        let mut entry = AllocationEntry::new(
            AllocationCategory::UsefulWorkPool,
            1000 * ONE_NERV,
            None,
        );
        entry.release(100 * ONE_NERV).unwrap();
        assert_eq!(entry.released_nano, 100 * ONE_NERV);
        assert!((entry.released_fraction() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_allocation_entry_release_overflow() {
        let mut entry = AllocationEntry::new(
            AllocationCategory::UsefulWorkPool,
            100 * ONE_NERV,
            None,
        );
        assert!(entry.release(200 * ONE_NERV).is_err());
    }

    #[test]
    fn test_allocation_entry_vesting() {
        let schedule = VestingSchedule::linear_with_cliff(
            BlockHeight::GENESIS,
            1000,
            100,
        );
        let entry = AllocationEntry::new(
            AllocationCategory::CodeResearch,
            1000 * ONE_NERV,
            Some(schedule),
        );

        // Before cliff: nothing vested
        assert_eq!(entry.vested_at(BlockHeight::from(50)), 0);
        assert_eq!(entry.locked_at(BlockHeight::from(50)), 1000 * ONE_NERV);

        // After cliff + some vesting
        let vested = entry.vested_at(BlockHeight::from(500));
        assert!(vested > 0);
        assert!(vested < 1000 * ONE_NERV);
    }

    #[test]
    fn test_allocation_entry_add_recipient() {
        let mut entry = AllocationEntry::new(
            AllocationCategory::Visionary,
            500 * ONE_NERV,
            None,
        );
        entry.add_recipient("nerv1origin".into(), 500 * ONE_NERV).unwrap();
        assert_eq!(entry.recipients.len(), 1);
    }

    #[test]
    fn test_allocation_registry_basic() {
        let mut registry = AllocationRegistry::new(TOTAL_SUPPLY_NANO);

        registry.add_allocation(AllocationEntry::new(
            AllocationCategory::UsefulWorkPool, 8000, None,
        )).unwrap();
        registry.add_allocation(AllocationEntry::new(
            AllocationCategory::Visionary, 500, None,
        )).unwrap();

        assert_eq!(registry.count(), 2);
        assert_eq!(registry.total_allocated(), 8500);
    }

    #[test]
    fn test_allocation_registry_duplicate() {
        let mut registry = AllocationRegistry::new(TOTAL_SUPPLY_NANO);
        registry.add_allocation(AllocationEntry::new(
            AllocationCategory::Visionary, 500, None,
        )).unwrap();
        assert!(registry.add_allocation(AllocationEntry::new(
            AllocationCategory::Visionary, 500, None,
        )).is_err());
    }

    #[test]
    fn test_allocation_registry_verify() {
        let mut registry = AllocationRegistry::new(10000);
        for cat in AllocationCategory::all() {
            let amount = (10000u128 * cat.default_bps() as u128 / 10000) as u64;
            registry.add_allocation(AllocationEntry::new(*cat, amount, None)).unwrap();
        }
        assert!(registry.verify_total(10000).is_ok());
    }

    #[test]
    fn test_allocation_registry_summary() {
        let mut registry = AllocationRegistry::new(10000);
        for cat in AllocationCategory::all() {
            let amount = (10000u128 * cat.default_bps() as u128 / 10000) as u64;
            registry.add_allocation(AllocationEntry::new(*cat, amount, None)).unwrap();
        }
        let summary = registry.summary();
        assert_eq!(summary.len(), 5);
    }
}
