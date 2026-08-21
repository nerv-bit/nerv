//! Unit tests for Tokenomics, Emission, and Vesting.
//!
//! Validates the NERV V2.0 economic model. Ensures that the fixed 10 Billion
//! NERV supply is perfectly allocated at genesis across the 5 categories,
//! that cliff and linear vesting schedules calculate locked/liquid tokens
//! accurately, and that the block emission curve transitions correctly
//! from the decaying phase to the 0.5%/yr tail emission.

use nerv::tokenomics::{
    TokenomicsEngine, AllocationCategory, AllocationRegistry, 
    EmissionCurve, EmissionState, VestingSchedule,
};
use nerv::config::TokenomicsConfig;
use nerv::{BlockHeight, Epoch, ONE_NERV, TOTAL_SUPPLY_NERV, TOTAL_SUPPLY_NANO};

// ─── Genesis Allocation Tests ────────────────────────────────────────────

#[test]
fn test_genesis_allocation_sums_to_total_supply() {
    let config = TokenomicsConfig::default();
    let mut engine = TokenomicsEngine::new(config);
    
    engine.initialize_genesis().unwrap();
    
    // The sum of all allocations must exactly equal the 10B NERV supply
    let total_allocated = engine.allocations.total_allocated();
    assert_eq!(
        total_allocated, 
        TOTAL_SUPPLY_NANO, 
        "Genesis allocations must perfectly sum to the total supply"
    );
    
    // Verify individual category amounts (in nano-NERV)
    assert_eq!(engine.allocations.category_amount(AllocationCategory::UsefulWorkPool), 8_000_000_000 * ONE_NERV);
    assert_eq!(engine.allocations.category_amount(AllocationCategory::Visionary), 500_000_000 * ONE_NERV);
    assert_eq!(engine.allocations.category_amount(AllocationCategory::CodeResearch), 500_000_000 * ONE_NERV);
    assert_eq!(engine.allocations.category_amount(AllocationCategory::EarlyDonors), 500_000_000 * ONE_NERV);
    assert_eq!(engine.allocations.category_amount(AllocationCategory::Treasury), 500_000_000 * ONE_NERV);
}

#[test]
fn test_genesis_double_initialization_fails() {
    let config = TokenomicsConfig::default();
    let mut engine = TokenomicsEngine::new(config);
    
    engine.initialize_genesis().unwrap();
    
    // Second initialization must fail to prevent inflation
    let result = engine.initialize_genesis();
    assert!(result.is_err(), "Double genesis initialization must be rejected");
}

#[test]
fn test_genesis_invariants_hold() {
    let config = TokenomicsConfig::default();
    let mut engine = TokenomicsEngine::new(config);
    engine.initialize_genesis().unwrap();
    
    // verify_invariants checks that allocations sum to total and emitted <= useful-work pool
    assert!(engine.verify_invariants().is_ok(), "Tokenomics invariants must hold post-genesis");
}

// ─── Vesting Schedule Tests ──────────────────────────────────────────────

#[test]
fn test_vesting_linear_with_cliff() {
    // Code & Research: 1-year vest, 3-month cliff
    // 1 year = 2,592,000 blocks. 3 months = 648,000 blocks.
    let schedule = VestingSchedule::linear_with_cliff(
        BlockHeight::GENESIS,
        2_592_000,
        648_000,
    );
    
    let total_amount = 1_000_000 * ONE_NERV;
    
    // 1. Before cliff (e.g., 2 months in) -> 0 vested
    let vested_before_cliff = schedule.vested_amount(total_amount, BlockHeight::from(400_000));
    assert_eq!(vested_before_cliff, 0, "No tokens should vest before the cliff");
    
    // 2. At exactly the cliff -> 0 vested (cliff means nothing vests *until* after this point)
    let vested_at_cliff = schedule.vested_amount(total_amount, BlockHeight::from(648_000));
    assert_eq!(vested_at_cliff, 0, "No tokens should vest exactly at the cliff");
    
    // 3. Midway through vesting (e.g., 6 months in -> ~3 months after cliff)
    // Should have vested roughly 25% of the total amount
    let vested_midway = schedule.vested_amount(total_amount, BlockHeight::from(1_296_000));
    assert!(vested_midway > 0, "Tokens should be vesting after the cliff");
    let expected_midway = total_amount / 4; // 25%
    let diff = if vested_midway > expected_midway { vested_midway - expected_midway } else { expected_midway - vested_midway };
    assert!(diff <= ONE_NERV, "Midway vesting should be approximately 25%"); // Allow 1 NERV diff for rounding
    
    // 4. At full duration -> 100% vested
    let vested_end = schedule.vested_amount(total_amount, BlockHeight::from(2_592_000));
    assert_eq!(vested_end, total_amount, "100% of tokens should be vested at the end");
    
    // 5. After full duration -> remains 100% vested
    let vested_after = schedule.vested_amount(total_amount, BlockHeight::from(3_000_000));
    assert_eq!(vested_after, total_amount, "Vested amount cannot exceed total");
}

#[test]
fn test_vesting_linear_no_cliff() {
    // Visionary: 6-month linear vest, no cliff
    let schedule = VestingSchedule::linear(
        BlockHeight::GENESIS,
        1_296_000,
        0,
    );
    
    let total_amount = 500_000 * ONE_NERV;
    
    // At 50% duration, 50% should be vested
    let vested_50 = schedule.vested_amount(total_amount, BlockHeight::from(648_000));
    assert_eq!(vested_50, total_amount / 2, "50% should be vested midway without a cliff");
}

// ─── Emission Curve & Tail Emission Tests ────────────────────────────────

#[test]
fn test_emission_curve_genesis_block() {
    let curve = EmissionCurve::nerv_default();
    
    // Genesis block (height 0) produces no emission
    assert_eq!(curve.emission_at(BlockHeight::GENESIS), 0, "Genesis block must emit 0");
}

#[test]
fn test_emission_curve_initial_block() {
    let curve = EmissionCurve::nerv_default();
    
    // Block 1 emits the initial reward: 100 NERV
    let emission = curve.emission_at(BlockHeight::from(1));
    assert_eq!(emission, 100 * ONE_NERV, "Initial emission must be 100 NERV");
}

#[test]
fn test_emission_curve_epoch_decay() {
    let curve = EmissionCurve::nerv_default();
    
    // Emission at Block 1 (Epoch 0)
    let e1 = curve.emission_at(BlockHeight::from(1));
    
    // Emission after 1 full epoch (Block 6,480,001 -> Epoch 1)
    let e_after_epoch = curve.emission_at(BlockHeight::from(nerv::tokenomics::emission::BLOCKS_PER_EPOCH + 1));
    
    // Must decay by 5% (0.95 multiplier)
    let expected_decay = (e1 as f64 * 0.95).round() as u64;
    assert_eq!(
        e_after_epoch, expected_decay,
        "Emission must decay by 5% per epoch"
    );
    
    // Must still be above tail emission
    assert!(e_after_epoch >= nerv::tokenomics::emission::TAIL_EMISSION_PER_BLOCK_NANO);
}

#[test]
fn test_emission_curve_tail_phase_transition() {
    let curve = EmissionCurve::nerv_default();
    
    // At the tail start height, emission must equal the tail rate (0.1 NERV)
    let tail_height = curve.tail_start_height;
    let emission_at_tail = curve.emission_at(tail_height);
    
    assert_eq!(
        emission_at_tail, 
        nerv::tokenomics::emission::TAIL_EMISSION_PER_BLOCK_NANO,
        "Emission must hit the tail rate exactly at tail_start_height"
    );
    
    // After tail start, it must remain at the tail rate forever
    let emission_far_future = curve.emission_at(BlockHeight::from(tail_height.0 + 10_000_000));
    assert_eq!(
        emission_far_future, 
        nerv::tokenomics::emission::TAIL_EMISSION_PER_BLOCK_NANO,
        "Emission must remain at tail rate indefinitely"
    );
}

#[test]
fn test_emission_state_pool_exhaustion() {
    // Use a small pool to simulate exhaustion quickly
    let max_pool = 500 * ONE_NERV;
    let mut state = EmissionState::new(max_pool);
    
    let mut total_emitted = 0u64;
    
    // Advance blocks until pool is exhausted
    for _ in 0..10_000 {
        let e = state.advance_block();
        total_emitted = total_emitted.saturating_add(e);
        
        if state.remaining_pool() == 0 {
            break;
        }
    }
    
    // Total emitted must not exceed the max pool
    assert!(total_emitted <= max_pool, "Total emitted must not exceed the useful-work pool");
    assert_eq!(state.total_emitted_nano, max_pool, "State must track exact exhaustion");
    
    // Advancing another block should only emit the tail emission
    let tail_emission = state.advance_block();
    assert_eq!(
        tail_emission, 
        nerv::tokenomics::emission::TAIL_EMISSION_PER_BLOCK_NANO,
        "Must emit tail emission after pool exhaustion"
    );
    assert!(state.in_tail_phase, "State must mark in_tail_phase as true");
}

#[test]
fn test_emission_state_epoch_tracking() {
    let mut state = EmissionState::new(1_000_000 * ONE_NERV);
    
    // Advance to block 1 (Epoch 0)
    state.advance_block();
    assert_eq!(state.current_epoch, Epoch::from(0));
    
    // Advance past the first epoch boundary
    for _ in 0..nerv::tokenomics::emission::BLOCKS_PER_EPOCH {
        state.advance_block();
    }
    
    // Should now be in Epoch 1
    assert_eq!(
        state.current_epoch, 
        Epoch::from(1), 
        "State must correctly track epoch transitions"
    );
}
