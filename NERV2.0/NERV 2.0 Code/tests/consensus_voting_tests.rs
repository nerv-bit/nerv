//! Unit tests for Consensus, Voting Weight, and Quorum.
//!
//! Validates the NERV V2.0 AI-native consensus mechanism. Ensures that
//! the combined voting weight (Stake × Reputation) is calculated accurately
//! using safe 128-bit arithmetic, that the 67% quorum threshold strictly
//! passes or fails, and that validator reputation scores correctly bound
//! between 0.0 and 1.0 during slashing and reward cycles.

use nerv::{
    StakeAmount, ReputationScore, VotingWeight, ValidatorId,
    check_quorum, compute_voting_weight,
    CONSENSUS_QUORUM_NUMERATOR, CONSENSUS_QUORUM_DENOMINATOR,
    NervError, ONE_NERV,
};

// ─── Voting Weight Calculation Tests ─────────────────────────────────────

#[test]
fn test_compute_voting_weight_exact() {
    // 1,000 NERV stake, Perfect (1.0) reputation
    let stake = StakeAmount::from_nerv(1000);
    let rep = ReputationScore::PERFECT;
    
    let weight = compute_voting_weight(stake, rep);
    
    // Weight = stake_nano * rep_scaled / SCALE
    // 1000 * 1e9 * 1e6 / 1e6 = 1e12
    assert_eq!(weight.0, 1000 * ONE_NERV as u128, "Voting weight must be exact for perfect reputation");
}

#[test]
fn test_compute_voting_weight_fractional_reputation() {
    // 2,000 NERV stake, 0.5 reputation
    let stake = StakeAmount::from_nerv(2000);
    let rep = ReputationScore::INITIAL; // 500,000 / 1,000,000 = 0.5
    
    let weight = compute_voting_weight(stake, rep);
    
    // 2000 * 1e9 * 5e5 / 1e6 = 1e12
    assert_eq!(
        weight.0, 
        (2000 * ONE_NERV as u128) / 2, 
        "Voting weight must accurately reflect fractional reputation"
    );
}

#[test]
fn test_voting_weight_checked_addition() {
    let w1 = VotingWeight(100);
    let w2 = VotingWeight(200);
    
    let sum = w1.checked_add(w2).unwrap();
    assert_eq!(sum.0, 300, "Checked addition must sum correctly");
    
    // Test overflow
    let max_weight = VotingWeight(u128::MAX);
    assert!(max_weight.checked_add(w1).is_none(), "Addition must fail on u128 overflow");
}

#[test]
fn test_voting_weight_saturating_addition() {
    let w1 = VotingWeight(u128::MAX - 100);
    let w2 = VotingWeight(200);
    
    let sum = w1.saturating_add(w2);
    assert_eq!(sum.0, u128::MAX, "Saturating addition must cap at u128::MAX");
}

// ─── Quorum Threshold Tests ──────────────────────────────────────────────

#[test]
fn test_quorum_success_exact_threshold() {
    // 67% exactly
    let total = VotingWeight(100);
    let achieved = VotingWeight(67);
    
    let result = check_quorum(achieved, total);
    assert!(result.is_ok(), "Exact 67% quorum must succeed");
}

#[test]
fn test_quorum_success_above_threshold() {
    let total = VotingWeight(1000);
    let achieved = VotingWeight(680); // 68%
    
    let result = check_quorum(achieved, total);
    assert!(result.is_ok(), "68% quorum must succeed");
}

#[test]
fn test_quorum_failure_one_short() {
    let total = VotingWeight(100);
    let achieved = VotingWeight(66); // 66%
    
    let result = check_quorum(achieved, total);
    
    assert!(
        matches!(result, Err(NervError::QuorumNotReached { achieved, required }) if achieved == 66.0 && required == 67.0),
        "66% quorum must fail with exact percentages"
    );
}

#[test]
fn test_quorum_failure_zero_total_weight() {
    let total = VotingWeight::ZERO;
    let achieved = VotingWeight::ZERO;
    
    let result = check_quorum(achieved, total);
    
    assert!(
        matches!(result, Err(NervError::Consensus(_))),
        "Zero total weight must result in a consensus error, not divide by zero"
    );
}

#[test]
fn test_quorum_large_numbers_no_overflow() {
    // Simulate a high-stake network
    let total = VotingWeight(u128::MAX / 2);
    let achieved = VotingWeight(u128::MAX / 3); // ~66.6%
    
    let result = check_quorum(achieved, total);
    assert!(result.is_err(), "Must not panic on large u128 multiplications");
}

// ─── Validator Reputation Tests ──────────────────────────────────────────

#[test]
fn test_reputation_score_initialization() {
    let initial = ReputationScore::INITIAL;
    assert_eq!(initial.to_f64(), 0.5, "Initial reputation must be 0.5");
    
    let perfect = ReputationScore::PERFECT;
    assert_eq!(perfect.to_f64(), 1.0, "Perfect reputation must be 1.0");
    
    let zero = ReputationScore::ZERO;
    assert_eq!(zero.to_f64(), 0.0, "Zero reputation must be 0.0");
}

#[test]
fn test_reputation_score_saturating_sub() {
    let mut rep = ReputationScore::from_f64(0.3);
    
    // Subtract more than available
    let slashed = rep.saturating_sub(ReputationScore::from_f64(0.5));
    assert_eq!(slashed.to_f64(), 0.0, "Reputation must floor at 0.0");
    
    // Verify original wasn't mutated
    assert_eq!(rep.to_f64(), 0.3, "saturating_sub must not mutate original");
    
    // Perform actual mutation
    rep = rep.saturating_sub(ReputationScore::from_f64(0.1));
    assert!((rep.to_f64() - 0.2).abs() < 1e-6, "Reputation must subtract correctly");
}

#[test]
fn test_reputation_score_saturating_add() {
    let mut rep = ReputationScore::from_f64(0.8);
    
    // Add more than perfect
    let rewarded = rep.saturating_add(ReputationScore::from_f64(0.5));
    assert_eq!(rewarded.to_f64(), 1.0, "Reputation must cap at 1.0");
    
    // Verify original wasn't mutated
    assert_eq!(rep.to_f64(), 0.8, "saturating_add must not mutate original");
    
    // Perform actual mutation
    rep = rep.saturating_add(ReputationScore::from_f64(0.1));
    assert!((rep.to_f64() - 0.9).abs() < 1e-6, "Reputation must add correctly");
}

// ─── Validator ID Tests ──────────────────────────────────────────────────

#[test]
fn test_validator_id_creation() {
    let pk_bytes = [42u8; 1952]; // Mock Dilithium-3 PK
    let val_id = ValidatorId::from_dilithium_pk(&pk_bytes);
    
    // Must be exactly 32 bytes (BLAKE3 hash)
    assert_eq!(val_id.as_bytes().len(), 32, "Validator ID must be 32 bytes");
    
    // Must be deterministic
    let val_id_2 = ValidatorId::from_dilithium_pk(&pk_bytes);
    assert_eq!(val_id, val_id_2, "Validator ID must be deterministic for same PK");
    
    // Different PK must yield different ID
    let pk_bytes_2 = [43u8; 1952];
    let val_id_3 = ValidatorId::from_dilithium_pk(&pk_bytes_2);
    assert_ne!(val_id, val_id_3, "Different PKs must yield different Validator IDs");
}
