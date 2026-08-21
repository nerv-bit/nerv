//! Tokenomics & Genesis Allocation.
//!
//! Fixed supply of 10 billion NERV. No perpetual inflation.
//! - 80% Useful-Work Pool (gradient rewards)
//! - 5% Visionary (6-month vesting)
//! - 5% Code & Research (1-year vesting, 3-month cliff)
//! - 5% Early Donors (1-year vesting, 6-month cliff)
//! - 5% Treasury (community-governed)
//!
//! Submodules:
//! - `allocation` — Genesis allocation management
//! - `emission` — Emission schedule management
//! - `rewards` — Reward calculation & distribution
//! - `vesting` — Vesting logic
//!
//! Tokenomics — Genesis Allocation, Emission, Rewards & Vesting.
//!
//! NERV features a **fixed supply** of 10,000,000,000 (10 Billion) NERV
//! with no perpetual inflation. Tokens are distributed at genesis to
//! align long-term network incentives, then emitted over time through
//! the useful-work economy.
//!
//! # Genesis Allocation
//!
//! | Category | Fraction | Tokens | Vesting |
//! |----------|----------|--------|---------|
//! | Useful-Work Pool | 80% | 8.0B NERV | Emitted over time via gradient rewards |
//! | Visionary | 5% | 0.5B NERV | 6-month linear vest |
//! | Code & Research | 5% | 0.5B NERV | 1-year linear, 3-month cliff |
//! | Early Donors | 5% | 0.5B NERV | 1-year linear, 6-month cliff |
//! | Treasury | 5% | 0.5B NERV | Community-governed grants |
//!
//! # Emission Schedule
//!
//! ```text
//! Blocks 0 – ~2.5 years:  Decaying emission (95% per epoch, ~30 days)
//! After useful-work pool exhausted: Tail emission 0.5%/yr forever
//! ```
//!
//! # Key Invariants
//!
//! - Σ(genesis allocations) = total supply
//! - Σ(emitted) ≤ useful-work pool (before tail)
//! - Σ(vested at height h) ≤ allocated amount
//! - No pre-mine, no hidden allocations

pub mod allocation;
pub mod emission;
pub mod rewards;
pub mod vesting;

// ─── Re-exports ─────────────────────────────────────────────────────────

pub use allocation::{
    AllocationCategory, AllocationEntry, AllocationRegistry,
};
pub use emission::{
    EmissionCurve, EmissionState,
};
pub use rewards::{
    RewardType, RewardRecord, RewardHistory,
};
pub use vesting::{
    VestingSchedule, VestingEntry, VestingRegistry,
};

use crate::{
    TOTAL_SUPPLY_NERV, TOTAL_SUPPLY_NANO, ONE_NERV, NERV_DECIMALS,
    BlockHeight, Epoch, ValidatorId,
    NervError, NervResult,
};
use crate::config::TokenomicsConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Token Information ───────────────────────────────────────────────────

/// Static information about the NERV token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenInfo {
    /// Token name.
    pub name: String,

    /// Token symbol.
    pub symbol: String,

    /// Total supply in whole NERV.
    pub total_supply_nerv: u64,

    /// Total supply in nano-NERV (base unit).
    pub total_supply_nano: u64,

    /// Number of decimal places (9 for nano-NERV).
    pub decimals: u8,

    /// One NERV in the base unit.
    pub one_nerv_nano: u64,
}

impl TokenInfo {
    /// NERV token information.
    pub const NERV: Self = Self {
        name: String::new(),
        symbol: String::new(),
        total_supply_nerv: TOTAL_SUPPLY_NERV,
        total_supply_nano: TOTAL_SUPPLY_NANO,
        decimals: NERV_DECIMALS,
        one_nerv_nano: ONE_NERV,
    };

    /// Create with full strings (cannot be const).
    pub fn nerv() -> Self {
        Self {
            name: "NERV".into(),
            symbol: "NERV".into(),
            total_supply_nerv: TOTAL_SUPPLY_NERV,
            total_supply_nano: TOTAL_SUPPLY_NANO,
            decimals: NERV_DECIMALS,
            one_nerv_nano: ONE_NERV,
        }
    }

    /// Convert NERV to nano-NERV.
    #[inline]
    pub fn to_nano(&self, nerv: u64) -> u64 {
        nerv * self.one_nerv_nano
    }

    /// Convert nano-NERV to NERV (truncating fractional part).
    #[inline]
    pub fn to_nerv(&self, nano: u64) -> u64 {
        nano / self.one_nerv_nano
    }

    /// Format nano-NERV as a decimal string.
    pub fn format_nano(&self, nano: u64) -> String {
        let whole = nano / self.one_nerv_nano;
        let frac = nano % self.one_nerv_nano;
        format!("{}.{:09}", whole, frac)
    }

    /// Parse a decimal string to nano-NERV.
    pub fn parse_nano(&self, s: &str) -> NervResult<u64> {
        let parts: Vec<&str> = s.split('.').collect();
        let whole: u64 = parts.first()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let frac: u64 = if parts.len() > 1 {
            let frac_str = parts[1];
            let padded = format!("{:0<9}", &frac_str[..frac_str.len().min(9)]);
            padded.parse().unwrap_or(0)
        } else {
            0
        };

        let result = whole
            .checked_mul(self.one_nerv_nano)
            .and_then(|v| v.checked_add(frac));

        result.ok_or_else(|| NervError::Tokenomics("amount overflow".into()))
    }
}

impl Default for TokenInfo {
    fn default() -> Self {
        Self::nerv()
    }
}

// ─── Tokenomics Engine ───────────────────────────────────────────────────

/// The top-level tokenomics engine that coordinates allocation,
/// emission, vesting, and rewards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenomicsEngine {
    /// Token information.
    pub token: TokenInfo,

    /// Genesis allocation registry.
    pub allocations: AllocationRegistry,

    /// Emission state.
    pub emission: EmissionState,

    /// Vesting registry.
    pub vesting: VestingRegistry,

    /// Reward history.
    pub rewards: RewardHistory,

    /// Configuration.
    pub config: TokenomicsConfig,

    /// Whether genesis has been initialized.
    pub genesis_initialized: bool,
}

impl TokenomicsEngine {
    /// Create a new tokenomics engine with the given configuration.
    pub fn new(config: TokenomicsConfig) -> Self {
        Self {
            token: TokenInfo::nerv(),
            allocations: AllocationRegistry::new(config.total_supply_nano),
            emission: EmissionState::new(config.total_supply_nano),
            vesting: VestingRegistry::new(),
            rewards: RewardHistory::new(),
            config,
            genesis_initialized: false,
        }
    }

    /// Initialize genesis allocations.
    ///
    /// This creates the five allocation categories and sets up
    /// vesting schedules. Can only be called once.
    pub fn initialize_genesis(&mut self) -> NervResult<()> {
        if self.genesis_initialized {
            return Err(NervError::Tokenomics("genesis already initialized".into()));
        }

        let total = self.config.total_supply_nano;

        // Compute allocation amounts
        let useful_work = (total as u128 * self.config.useful_work_pool_bps as u128 / 10000) as u64;
        let visionary = (total as u128 * self.config.visionary_bps as u128 / 10000) as u64;
        let code_research = (total as u128 * self.config.code_research_bps as u128 / 10000) as u64;
        let early_donors = (total as u128 * self.config.early_donors_bps as u128 / 10000) as u64;
        // Treasury gets the remainder to ensure exact total
        let treasury = total
            .saturating_sub(useful_work)
            .saturating_sub(visionary)
            .saturating_sub(code_research)
            .saturating_sub(early_donors);

        // Register allocations
        self.allocations.add_allocation(AllocationEntry::new(
            AllocationCategory::UsefulWorkPool,
            useful_work,
            None, // No vesting — emitted over time
        ))?;

        self.allocations.add_allocation(AllocationEntry::new(
            AllocationCategory::Visionary,
            visionary,
            Some(VestingSchedule::linear(
                BlockHeight::GENESIS,
                self.config.visionary_vesting_blocks,
                0, // No cliff for visionary
            )),
        ))?;

        self.allocations.add_allocation(AllocationEntry::new(
            AllocationCategory::CodeResearch,
            code_research,
            Some(VestingSchedule::linear_with_cliff(
                BlockHeight::GENESIS,
                self.config.code_research_vesting_blocks,
                self.config.code_research_cliff_blocks,
            )),
        ))?;

        self.allocations.add_allocation(AllocationEntry::new(
            AllocationCategory::EarlyDonors,
            early_donors,
            Some(VestingSchedule::linear_with_cliff(
                BlockHeight::GENESIS,
                self.config.early_donors_vesting_blocks,
                self.config.early_donors_cliff_blocks,
            )),
        ))?;

        self.allocations.add_allocation(AllocationEntry::new(
            AllocationCategory::Treasury,
            treasury,
            None, // Community-governed, no automatic vesting
        ))?;

        // Verify allocations sum to total
        self.allocations.verify_total(total)?;

        self.genesis_initialized = true;
        Ok(())
    }

    /// Get the total circulating supply at a given block height.
    ///
    /// Circulating = emitted + vested (excluding locked tokens).
    pub fn circulating_supply(&self, height: BlockHeight) -> u64 {
        let emitted = self.emission.total_emitted_nano;
        let vested = self.vesting.total_vested_at(height);
        emitted.saturating_add(vested)
    }

    /// Get the total locked supply (unvested + unemitted).
    pub fn locked_supply(&self, height: BlockHeight) -> u64 {
        self.token.total_supply_nano
            .saturating_sub(self.circulating_supply(height))
    }

    /// Check invariants.
    pub fn verify_invariants(&self) -> NervResult<()> {
        // Total allocations must equal total supply
        self.allocations.verify_total(self.config.total_supply_nano)?;

        // Emitted must not exceed useful-work pool
        let useful_work = self.allocations.category_amount(AllocationCategory::UsefulWorkPool);
        if self.emission.total_emitted_nano > useful_work {
            return Err(NervError::Tokenomics(format!(
                "emitted {} exceeds useful-work pool {}",
                self.emission.total_emitted_nano, useful_work
            )));
        }

        Ok(())
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_info_nerv() {
        let info = TokenInfo::nerv();
        assert_eq!(info.total_supply_nerv, 10_000_000_000);
        assert_eq!(info.decimals, 9);
        assert_eq!(info.to_nano(1), ONE_NERV);
        assert_eq!(info.to_nerv(ONE_NERV), 1);
    }

    #[test]
    fn test_token_info_format() {
        let info = TokenInfo::nerv();
        let formatted = info.format_nano(1_500_000_000); // 1.5 NERV
        assert_eq!(formatted, "1.500000000");
    }

    #[test]
    fn test_token_info_parse() {
        let info = TokenInfo::nerv();
        let parsed = info.parse_nano("1.5").unwrap();
        assert_eq!(parsed, 1_500_000_000);
    }

    #[test]
    fn test_tokenomics_engine_genesis() {
        let config = TokenomicsConfig::default();
        let mut engine = TokenomicsEngine::new(config);
        assert!(engine.initialize_genesis().is_ok());
        assert!(engine.genesis_initialized);
    }

    #[test]
    fn test_tokenomics_engine_double_genesis() {
        let config = TokenomicsConfig::default();
        let mut engine = TokenomicsEngine::new(config);
        engine.initialize_genesis().unwrap();
        assert!(engine.initialize_genesis().is_err());
    }

    #[test]
    fn test_tokenomics_engine_invariants() {
        let config = TokenomicsConfig::default();
        let mut engine = TokenomicsEngine::new(config);
        engine.initialize_genesis().unwrap();
        assert!(engine.verify_invariants().is_ok());
    }

    #[test]
    fn test_tokenomics_allocation_categories() {
        let config = TokenomicsConfig::default();
        let mut engine = TokenomicsEngine::new(config);
        engine.initialize_genesis().unwrap();

        // All five categories should be present
        assert!(engine.allocations.category_amount(AllocationCategory::UsefulWorkPool) > 0);
        assert!(engine.allocations.category_amount(AllocationCategory::Visionary) > 0);
        assert!(engine.allocations.category_amount(AllocationCategory::CodeResearch) > 0);
        assert!(engine.allocations.category_amount(AllocationCategory::EarlyDonors) > 0);
        assert!(engine.allocations.category_amount(AllocationCategory::Treasury) > 0);
    }
}
