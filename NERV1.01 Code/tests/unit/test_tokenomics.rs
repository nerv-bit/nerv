// tests/unit/test_tokenomics.rs
// ============================================================================
// TOKENOMICS MODULE UNIT TESTS
// ============================================================================

use nerv_bit::tokenomics::{
    TokenomicsEngine, TokenomicsConfig,
    GenesisAllocation, AllocationCategory,
    VestingManager, VestingType, VestingSchedule,
    EmissionSchedule,
    RewardCalculator, GradientReward, ValidationReward, PublicGoodsProposal, ProposalCategory,
};
use nerv_bit::params::{
    TOTAL_SUPPLY, WEI_PER_NERV,
    EMISSION_YEAR_1_2_PERCENT, EMISSION_YEAR_3_5_PERCENT,
    EMISSION_YEAR_6_10_PERCENT, TAIL_EMISSION_PERCENT,
    REWARD_GRADIENT_PERCENT, REWARD_VALIDATION_PERCENT, REWARD_PUBLIC_GOODS_PERCENT,
};
use std::collections::HashMap;

const GENESIS_TIME: u64 = 1735689600; // Fixed genesis timestamp
const BLOCKS_PER_YEAR: u64 = 365 * 24 * 60 * 6; // ~6 blocks/min ≈ 3.15M blocks/year

#[test]
fn test_tokenomics_config_defaults() {
    let config = TokenomicsConfig::default();

    assert_eq!(config.fee_burn_percent, 0.20);
    assert_eq!(config.fee_validator_percent, 0.80);
    assert!(config.enable_deflationary_burn);
    assert_eq!(config.public_goods_vote_threshold, 0.5);
    assert_eq!(config.max_proposal_funding_percent, 0.1);
}

#[test]
fn test_genesis_allocation_totals() {
    let total_supply = TOTAL_SUPPLY * WEI_PER_NERV;
    let manager = GenesisAllocation::new(total_supply, GENESIS_TIME);

    // Sum of all category percentages should be 100%
    let sum_percent = GENESIS_USEFUL_WORK_PERCENT +
                      GENESIS_CODE_CONTRIB_PERCENT +
                      GENESIS_AUDIT_BOUNTY_PERCENT +
                      GENESIS_RESEARCH_PERCENT +
                      GENESIS_EARLY_DONOR_PERCENT +
                      GENESIS_TREASURY_PERCENT +
                      GENESIS_VISIONARY_PERCENT;

    assert!((sum_percent - 1.0).abs() < 1e-9);

    // Useful work gets 60%
    let useful_work_amount = (total_supply as f64 * GENESIS_USEFUL_WORK_PERCENT as f64) as u128;
    assert_eq!(manager.useful_work_reserve, useful_work_amount);
}

#[test]
fn test_genesis_allocation_vesting_and_claiming() {
    let total_supply = 10_000_000_000 * WEI_PER_NERV;
    let mut manager = GenesisAllocation::new(total_supply, GENESIS_TIME);

    let recipient = [1u8; 20];
    let amount = 500_000_000 * WEI_PER_NERV; // 5% category

    let id = manager.add_allocation(
        recipient,
        amount,
        AllocationCategory::Treasury,
        "Treasury allocation".to_string(),
        Some(4 * 365 * 24 * 60 * 60), // 4-year linear vesting
    ).unwrap();

    // Immediate claim (no vesting yet)
    let claimable = manager.claim_vested(id, GENESIS_TIME).unwrap();
    assert_eq!(claimable, 0);

    // After 1 year (25% vested)
    let one_year = GENESIS_TIME + 365 * 24 * 60 * 60;
    let claimable = manager.claim_vested(id, one_year).unwrap();
    assert_eq!(claimable, amount / 4);

    // Claim again immediately (no additional)
    let claimable = manager.claim_vested(id, one_year).unwrap();
    assert_eq!(claimable, 0);

    // After full vesting
    let full = GENESIS_TIME + 4 * 365 * 24 * 60 * 60;
    let claimable = manager.claim_vested(id, full).unwrap();
    assert_eq!(claimable, amount * 3 / 4); // Remaining 75%

    let allocation = manager.get_allocation(id).unwrap();
    assert_eq!(allocation.claimed_amount, amount);
}

#[test]
fn test_vesting_schedule_linear() {
    let mut manager = VestingManager::new();
    let owner = [2u8; 20];
    let total = 1_000_000 * WEI_PER_NERV;
    let start = GENESIS_TIME;

    let id = manager.create_schedule(
        owner,
        total,
        start,
        VestingType::Linear,
        Some(4 * 365 * 24 * 60 * 60), // 4 years
        false,
        None,
        None,
    ).unwrap();

    // 2 years in
    let mid = start + 2 * 365 * 24 * 60 * 60;
    let claimable = manager.claim_vested(id, mid).unwrap();
    assert_eq!(claimable, total / 2);

    let schedule = manager.get_schedule(id).unwrap();
    assert_eq!(schedule.vested_amount, total / 2);
}

#[test]
fn test_vesting_schedule_cliff() {
    let mut manager = VestingManager::new();
    let owner = [3u8; 20];
    let total = 1_000_000 * WEI_PER_NERV;
    let start = GENESIS_TIME;
    let cliff = 365 * 24 * 60 * 60; // 1 year cliff
    let duration = 4 * 365 * 24 * 60 * 60;

    let id = manager.create_schedule(
        owner,
        total,
        start,
        VestingType::CliffLinear { cliff_duration: cliff, total_duration: duration },
        None,
        false,
        None,
        None,
    ).unwrap();

    // Before cliff
    let before = start + cliff - 1;
    let claimable = manager.claim_vested(id, before).unwrap();
    assert_eq!(claimable, 0);

    // Just after cliff (0% additional yet)
    let after_cliff = start + cliff;
    let claimable = manager.claim_vested(id, after_cliff).unwrap();
    assert_eq!(claimable, 0);

    // 1 year after cliff (25% of total vested)
    let post = start + cliff + 365 * 24 * 60 * 60;
    let claimable = manager.claim_vested(id, post).unwrap();
    assert_eq!(claimable, total / 4);
}

#[test]
fn test_emission_schedule_block_rewards() {
    let schedule = EmissionSchedule::default();

    // Year 1 block reward
    let block_0_reward = schedule.block_reward(0);
    let expected_year1 = (TOTAL_SUPPLY as f64 * EMISSION_YEAR_1_2_PERCENT as f64 / BLOCKS_PER_YEAR as f64) as u128 * WEI_PER_NERV;
    assert!((block_0_reward - expected_year1).abs() < WEI_PER_NERV); // Tolerance for rounding

    // Tail emission
    let tail_block = BLOCKS_PER_YEAR * 10;
    let tail_reward = schedule.block_reward(tail_block);
    let expected_tail = (TOTAL_SUPPLY as f64 * TAIL_EMISSION_PERCENT as f64 / BLOCKS_PER_YEAR as f64) as u128 * WEI_PER_NERV;
    assert!((tail_reward - expected_tail).abs() < WEI_PER_NERV);
}

#[test]
fn test_reward_calculator_gradient_distribution() {
    let mut calculator = RewardCalculator::new();

    let mut rewards = vec![
        GradientReward {
            address: [1u8; 20],
            shapley_value: 0.6,
            quality_score: 0.9,
            dp_epsilon: 3.0,
            timestamp: GENESIS_TIME,
            gradient_hash: [0u8; 32],
        },
        GradientReward {
            address: [2u8; 20],
            shapley_value: 0.4,
            quality_score: 0.8,
            dp_epsilon: 3.0,
            timestamp: GENESIS_TIME,
            gradient_hash: [1u8; 32],
        },
    ];

    let block_reward = 100_000 * WEI_PER_NERV;
    let gradient_pool = (block_reward as f64 * REWARD_GRADIENT_PERCENT as f64) as u128;

    let distributed = calculator.distribute_gradient_rewards(&mut rewards, gradient_pool);

    assert_eq!(distributed.len(), 2);
    let total: u128 = distributed.iter().map(|(_, amt)| *amt).sum();
    assert_eq!(total, gradient_pool);

    // Higher shapley + quality gets more
    assert!(distributed[0].1 > distributed[1].1);
}

#[test]
fn test_reward_calculator_public_goods_selection() {
    let mut calculator = RewardCalculator::new();

    let proposals = vec![
        PublicGoodsProposal {
            proposal_id: 1,
            recipient: [1u8; 20],
            requested_amount: 100_000 * WEI_PER_NERV,
            funded_amount: 0,
            description: "Security audit".to_string(),
            vote_score: 0.9,
            technical_score: 0.95,
            impact_multiplier: 5,
            category: ProposalCategory::Security,
            duration_months: 3,
            verified: true,
            milestones: vec![],
        },
        PublicGoodsProposal {
            proposal_id: 2,
            recipient: [2u8; 20],
            requested_amount: 50_000 * WEI_PER_NERV,
            funded_amount: 0,
            description: "Education".to_string(),
            vote_score: 0.7,
            technical_score: 0.6,
            impact_multiplier: 3,
            category: ProposalCategory::Education,
            duration_months: 6,
            verified: true,
            milestones: vec![],
        },
    ];

    let pg_pool = 120_000 * WEI_PER_NERV;

    let selected = calculator.select_public_goods_proposals(&proposals, pg_pool, 0.5);

    assert_eq!(selected.len(), 2);
    let total_funded: u128 = selected.iter().map(|(_, amt)| *amt).sum();
    assert!(total_funded <= pg_pool);

    // Higher scoring proposal gets more
    assert!(selected.iter().find(|(p, _)| p.proposal_id == 1).unwrap().1 >
            selected.iter().find(|(p, _)| p.proposal_id == 2).unwrap().1);
}

#[test]
fn test_tokenomics_engine_fee_processing() {
    let config = TokenomicsConfig::default();
    let mut engine = TokenomicsEngine::new(config, GENESIS_TIME);

    let fee = 1000 * WEI_PER_NERV;
    let result = engine.process_fee(fee);

    assert_eq!(result.total_fee, fee);
    assert_eq!(result.burned, (fee as f64 * 0.2) as u128);
    assert_eq!(result.to_validators, (fee as f64 * 0.8) as u128);
    assert_eq!(engine.total_burned, result.burned);
}

#[test]
fn test_tokenomics_engine_block_reward_distribution() {
    let config = TokenomicsConfig::default();
    let mut engine = TokenomicsEngine::new(config, GENESIS_TIME);

    // Simulate a block at height 0
    let reward = engine.calculate_block_reward(0);
    let distribution = engine.distribute_block_reward(reward);

    assert_eq!(distribution.gradient_pool + distribution.validation_pool + distribution.public_goods_pool, reward);

    // Percentages match params
    assert_eq!(distribution.gradient_pool, (reward as f64 * REWARD_GRADIENT_PERCENT as f64) as u128);
    assert_eq!(distribution.validation_pool, (reward as f64 * REWARD_VALIDATION_PERCENT as f64) as u128);
    assert_eq!(distribution.public_goods_pool, (reward as f64 * REWARD_PUBLIC_GOODS_PERCENT as f64) as u128);
}