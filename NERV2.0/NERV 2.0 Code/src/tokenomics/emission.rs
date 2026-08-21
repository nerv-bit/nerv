//! Emission Schedule Management.
//!
//! NERV's emission follows a decaying schedule with a permanent
//! tail emission, ensuring the network always has incentives
//! for useful work while respecting the fixed total supply.
//!
//! # Schedule
//!
//! ```text
//! Phase 1 (Decaying):
//!   emission_per_block = initial * decay^(epoch)
//!   Decays by 5% per epoch (~30 days)
//!
//! Phase 2 (Tail):
//!   emission_per_block = tail_rate (0.5% of remaining pool / year)
//!   Continues forever, ensuring perpetual security
//! ```
//!
//! The useful-work pool (80% of total supply) is emitted over
//! approximately 10 years, after which tail emission sustains
//! the network indefinitely at ~0.5%/yr.

use crate::{
    TOTAL_SUPPLY_NANO, ONE_NERV,
    BlockHeight, Epoch,
    NervError, NervResult,
};
use serde::{Deserialize, Serialize};

// ─── Emission Constants ──────────────────────────────────────────────────

/// Blocks per epoch (~30 days at 400ms blocks).
/// 30 * 24 * 3600 / 0.4 = 6,480,000
pub const BLOCKS_PER_EPOCH: u64 = 6_480_000;

/// Initial emission per block at genesis (100 NERV).
pub const INITIAL_EMISSION_PER_BLOCK_NANO: u64 = 100 * ONE_NERV;

/// Decay factor per epoch (0.95 = 5% reduction).
pub const EPOCH_DECAY_FACTOR: f64 = 0.95;

/// Tail emission per block (0.1 NERV).
/// At 0.4s blocks: 0.1 * 86400 / 0.4 = 21,600 NERV/day
/// ≈ 7.88M NERV/year ≈ 0.08%/yr of 10B supply
pub const TAIL_EMISSION_PER_BLOCK_NANO: u64 = ONE_NERV / 10;

/// Maximum emission per block (sanity limit).
pub const MAX_EMISSION_PER_BLOCK_NANO: u64 = 1000 * ONE_NERV;

// ─── Emission Curve ──────────────────────────────────────────────────────

/// Defines the mathematical emission curve.
///
/// The curve has two phases:
/// 1. **Decaying**: `E(h) = initial * decay^(h / blocks_per_epoch)`
/// 2. **Tail**: `E(h) = tail_rate` (once the decaying phase
///    reaches the tail rate)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmissionCurve {
    /// Initial emission per block (nano-NERV).
    pub initial_emission_nano: u64,

    /// Decay factor per epoch (e.g., 0.95).
    pub epoch_decay: f64,

    /// Tail emission per block (nano-NERV).
    pub tail_emission_nano: u64,

    /// Blocks per epoch.
    pub blocks_per_epoch: u64,

    /// Height at which tail emission begins (computed).
    pub tail_start_height: BlockHeight,
}

impl EmissionCurve {
    /// Create the default NERV emission curve.
    pub fn nerv_default() -> Self {
        // Compute when decaying emission reaches tail rate:
        // initial * decay^n = tail
        // n = log(tail / initial) / log(decay)
        let ratio = TAIL_EMISSION_PER_BLOCK_NANO as f64 / INITIAL_EMISSION_PER_BLOCK_NANO as f64;
        let epochs_to_tail = if ratio > 0.0 && EPOCH_DECAY_FACTOR > 0.0 {
            (ratio.ln() / EPOCH_DECAY_FACTOR.ln()).abs().ceil() as u64
        } else {
            100 // Fallback: ~8 years
        };

        Self {
            initial_emission_nano: INITIAL_EMISSION_PER_BLOCK_NANO,
            epoch_decay: EPOCH_DECAY_FACTOR,
            tail_emission_nano: TAIL_EMISSION_PER_BLOCK_NANO,
            blocks_per_epoch: BLOCKS_PER_EPOCH,
            tail_start_height: BlockHeight::from(epochs_to_tail * BLOCKS_PER_EPOCH),
        }
    }

    /// Compute the emission for a given block height.
    pub fn emission_at(&self, height: BlockHeight) -> u64 {
        if height.0 == 0 {
            return 0; // Genesis block has no emission
        }

        if height.0 >= self.tail_start_height.0 {
            return self.tail_emission_nano;
        }

        // Decaying phase
        let epoch = height.0 / self.blocks_per_epoch;
        let decayed = self.initial_emission_nano as f64
            * self.epoch_decay.powi(epoch as i32);

        let emission = decayed as u64;
        emission.max(self.tail_emission_nano).min(MAX_EMISSION_PER_BLOCK_NANO)
    }

    /// Compute the epoch for a given height.
    pub fn epoch_at(&self, height: BlockHeight) -> Epoch {
        Epoch::from(height.0 / self.blocks_per_epoch)
    }

    /// Check if we're in the tail emission phase.
    pub fn is_tail_phase(&self, height: BlockHeight) -> bool {
        height.0 >= self.tail_start_height.0
    }

    /// Estimate total emission between two heights.
    pub fn estimate_emission_range(
        &self,
        from: BlockHeight,
        to: BlockHeight,
    ) -> u64 {
        if to.0 <= from.0 {
            return 0;
        }

        // Simple estimation: use the emission rate at the midpoint
        let mid = BlockHeight::from((from.0 + to.0) / 2);
        let blocks = to.0 - from.0;
        let rate = self.emission_at(mid);

        (rate as u128 * blocks as u128 / 1).min(u64::MAX as u128) as u64
    }

    /// Compute the current emission rate as a fraction of total supply per year.
    pub fn annual_rate(&self, height: BlockHeight, total_supply_nano: u64) -> f64 {
        if total_supply_nano == 0 {
            return 0.0;
        }
        let emission_per_block = self.emission_at(height) as f64;
        let blocks_per_year = 365.25 * 24.0 * 3600.0 / 0.4; // ~78.9M blocks/year
        let annual_emission = emission_per_block * blocks_per_year;
        annual_emission / total_supply_nano as f64
    }
}

impl Default for EmissionCurve {
    fn default() -> Self {
        Self::nerv_default()
    }
}

// ─── Emission State ──────────────────────────────────────────────────────

/// Runtime state of the emission schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmissionState {
    /// The emission curve.
    pub curve: EmissionCurve,

    /// Total emitted so far (nano-NERV).
    pub total_emitted_nano: u64,

    /// Maximum amount to emit (useful-work pool, nano-NERV).
    pub max_emission_nano: u64,

    /// Current block height.
    pub current_height: BlockHeight,

    /// Current epoch.
    pub current_epoch: Epoch,

    /// Emission for the current block (nano-NERV).
    pub current_block_emission_nano: u64,

    /// Whether tail emission has been reached.
    pub in_tail_phase: bool,

    /// Total blocks that have emitted.
    pub emitting_blocks: u64,
}

impl EmissionState {
    /// Create a new emission state.
    pub fn new(max_emission_nano: u64) -> Self {
        let curve = EmissionCurve::nerv_default();
        Self {
            curve,
            total_emitted_nano: 0,
            max_emission_nano,
            current_height: BlockHeight::GENESIS,
            current_epoch: Epoch::GENESIS,
            current_block_emission_nano: 0,
            in_tail_phase: false,
            emitting_blocks: 0,
        }
    }

    /// Create with a custom curve.
    pub fn with_curve(max_emission_nano: u64, curve: EmissionCurve) -> Self {
        Self {
            curve,
            total_emitted_nano: 0,
            max_emission_nano,
            current_height: BlockHeight::GENESIS,
            current_epoch: Epoch::GENESIS,
            current_block_emission_nano: 0,
            in_tail_phase: false,
            emitting_blocks: 0,
        }
    }

    /// Advance to the next block and compute emission.
    pub fn advance_block(&mut self) -> u64 {
        self.current_height = self.current_height.next();
        self.current_epoch = self.curve.epoch_at(self.current_height);

        // Check if we've exhausted the useful-work pool
        if self.total_emitted_nano >= self.max_emission_nano {
            // Tail emission only
            self.current_block_emission_nano = self.curve.tail_emission_nano;
            self.in_tail_phase = true;
        } else {
            self.current_block_emission_nano = self.curve.emission_at(self.current_height);
            self.in_tail_phase = self.curve.is_tail_phase(self.current_height);

            // Don't emit more than the remaining pool
            let remaining = self.max_emission_nano.saturating_sub(self.total_emitted_nano);
            if self.current_block_emission_nano > remaining {
                self.current_block_emission_nano = remaining;
            }
        }

        self.total_emitted_nano = self.total_emitted_nano
            .saturating_add(self.current_block_emission_nano);
        self.emitting_blocks += 1;

        self.current_block_emission_nano
    }

    /// Remaining tokens in the useful-work pool.
    pub fn remaining_pool(&self) -> u64 {
        self.max_emission_nano.saturating_sub(self.total_emitted_nano)
    }

    /// Fraction of pool emitted (0.0–1.0).
    pub fn emitted_fraction(&self) -> f64 {
        if self.max_emission_nano == 0 {
            return 0.0;
        }
        self.total_emitted_nano as f64 / self.max_emission_nano as f64
    }

    /// Current annual emission rate as fraction of total supply.
    pub fn current_annual_rate(&self, total_supply_nano: u64) -> f64 {
        self.curve.annual_rate(self.current_height, total_supply_nano)
    }

    /// Estimated blocks until the useful-work pool is exhausted.
    pub fn estimated_blocks_until_exhausted(&self) -> Option<u64> {
        let remaining = self.remaining_pool();
        if remaining == 0 {
            return Some(0);
        }
        let current_rate = self.current_block_emission_nano;
        if current_rate == 0 {
            return None;
        }
        Some(remaining / current_rate)
    }

    /// Reset state (for testing).
    pub fn reset(&mut self) {
        self.total_emitted_nano = 0;
        self.current_height = BlockHeight::GENESIS;
        self.current_epoch = Epoch::GENESIS;
        self.current_block_emission_nano = 0;
        self.in_tail_phase = false;
        self.emitting_blocks = 0;
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emission_curve_default() {
        let curve = EmissionCurve::nerv_default();
        assert_eq!(curve.initial_emission_nano, INITIAL_EMISSION_PER_BLOCK_NANO);
        assert_eq!(curve.tail_emission_nano, TAIL_EMISSION_PER_BLOCK_NANO);
        assert!((curve.epoch_decay - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_emission_curve_genesis() {
        let curve = EmissionCurve::nerv_default();
        assert_eq!(curve.emission_at(BlockHeight::GENESIS), 0);
    }

    #[test]
    fn test_emission_curve_first_block() {
        let curve = EmissionCurve::nerv_default();
        let emission = curve.emission_at(BlockHeight::from(1));
        assert_eq!(emission, INITIAL_EMISSION_PER_BLOCK_NANO);
    }

    #[test]
    fn test_emission_curve_decay() {
        let curve = EmissionCurve::nerv_default();
        let e1 = curve.emission_at(BlockHeight::from(1));
        let e_after_epoch = curve.emission_at(BlockHeight::from(BLOCKS_PER_EPOCH + 1));
        // After one epoch, emission should be less
        assert!(e_after_epoch < e1);
        // But still above tail
        assert!(e_after_epoch >= TAIL_EMISSION_PER_BLOCK_NANO);
    }

    #[test]
    fn test_emission_curve_tail_phase() {
        let curve = EmissionCurve::nerv_default();
        let tail_height = curve.tail_start_height;
        assert!(curve.is_tail_phase(tail_height));
        assert!(!curve.is_tail_phase(BlockHeight::from(1)));

        let emission = curve.emission_at(tail_height);
        assert_eq!(emission, TAIL_EMISSION_PER_BLOCK_NANO);
    }

    #[test]
    fn test_emission_curve_epoch() {
        let curve = EmissionCurve::nerv_default();
        assert_eq!(curve.epoch_at(BlockHeight::from(0)), Epoch::from(0));
        assert_eq!(curve.epoch_at(BlockHeight::from(BLOCKS_PER_EPOCH)), Epoch::from(1));
    }

    #[test]
    fn test_emission_curve_annual_rate() {
        let curve = EmissionCurve::nerv_default();
        let rate = curve.annual_rate(BlockHeight::from(1), TOTAL_SUPPLY_NANO);
        assert!(rate > 0.0);
        assert!(rate < 1.0); // Should be a small fraction
    }

    #[test]
    fn test_emission_state_advance() {
        let max_pool = 8000 * ONE_NERV; // 8000 NERV pool
        let mut state = EmissionState::new(max_pool);

        let e1 = state.advance_block();
        assert!(e1 > 0);
        assert_eq!(state.total_emitted_nano, e1);
        assert_eq!(state.emitting_blocks, 1);
    }

    #[test]
    fn test_emission_state_multiple_blocks() {
        let max_pool = 8000 * ONE_NERV;
        let mut state = EmissionState::new(max_pool);

        let mut total = 0u64;
        for _ in 0..100 {
            total = total.saturating_add(state.advance_block());
        }

        assert_eq!(state.total_emitted_nano, total);
        assert_eq!(state.emitting_blocks, 100);
        assert!(state.total_emitted_nano > 0);
    }

    #[test]
    fn test_emission_state_pool_exhaustion() {
        let max_pool = 500 * ONE_NERV; // Small pool for testing
        let mut state = EmissionState::new(max_pool);

        // Advance until pool is exhausted or many blocks
        for _ in 0..10_000 {
            state.advance_block();
            if state.remaining_pool() == 0 {
                break;
            }
        }

        // Total emitted should not exceed the pool
        assert!(state.total_emitted_nano <= max_pool);
    }

    #[test]
    fn test_emission_state_remaining() {
        let max_pool = 10000 * ONE_NERV;
        let state = EmissionState::new(max_pool);
        assert_eq!(state.remaining_pool(), max_pool);
    }

    #[test]
    fn test_emission_state_fraction() {
        let max_pool = 10000 * ONE_NERV;
        let mut state = EmissionState::new(max_pool);
        state.advance_block();
        assert!(state.emitted_fraction() > 0.0);
        assert!(state.emitted_fraction() < 1.0);
    }

    #[test]
    fn test_emission_state_reset() {
        let max_pool = 10000 * ONE_NERV;
        let mut state = EmissionState::new(max_pool);
        state.advance_block();
        assert!(state.total_emitted_nano > 0);
        state.reset();
        assert_eq!(state.total_emitted_nano, 0);
    }

    #[test]
    fn test_emission_curve_estimate_range() {
        let curve = EmissionCurve::nerv_default();
        let estimate = curve.estimate_emission_range(
            BlockHeight::from(1),
            BlockHeight::from(1001),
        );
        assert!(estimate > 0);
    }

    #[test]
    fn test_emission_state_estimated_blocks_until_exhausted() {
        let max_pool = 10000 * ONE_NERV;
        let mut state = EmissionState::new(max_pool);
        state.advance_block();
        let estimate = state.estimated_blocks_until_exhausted();
        assert!(estimate.is_some());
        assert!(estimate.unwrap() > 0);
    }
}
