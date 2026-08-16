Rust// tests/unit/test_consensus.rs
// ============================================================================
// CONSENSUS MODULE UNIT TESTS
// ============================================================================

use nerv_bit::consensus::{
    predictor::{Predictor, ConsensusConfig as PredictorConfigPart},
    voting::{Validator, ValidatorSet, SlashingReason, EncoderUpdateVote},
    dispute::{DisputeResolver, Dispute, DisputeType, DisputeEvidence, EvidenceType, DisputeStatus, ResolutionResult},
    ConsensusConfig, ConsensusState,
};
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::path::PathBuf;
use tempfile::tempdir;

#[test]
fn test_consensus_config_defaults() {
    let config = ConsensusConfig::default();

    assert_eq!(config.block_time_ms, 600);
    assert_eq!(config.view_change_timeout_ms, 3000);
    assert_eq!(config.max_validators, 1000);
    assert_eq!(config.min_stake, 1000 * 10u128.pow(18));
    assert!((config.reputation_decay - 0.95).abs() < 1e-9);
    assert!((config.slashing_penalty - 0.1).abs() < 1e-9);
    assert_eq!(config.dispute_timeout_sec, 30);
    assert!(config.enable_predictor);
}

#[test]
fn test_consensus_state_defaults() {
    let state = ConsensusState::default();

    assert_eq!(state.view_number, 0);
    assert_eq!(state.block_height, 0);
    assert_eq!(state.last_committed_hash, [0u8; 32]);
    assert_eq!(state.proposer_index, 0);
    assert!(state.locked_hash.is_none());
    assert!(state.prepared_hash.is_none());
}

#[test]
fn test_predictor_fallback_mode() {
    // Use a non-existent model path to force fallback
    let mut config = ConsensusConfig::default();
    config.predictor_model_path = PathBuf::from("nonexistent_model.pt");

    let predictor = Predictor::new(&config).unwrap();
    assert!(predictor.fallback);

    let tokens = vec![1u16, 2, 3, 4, 5];
    let (embedding, score) = predictor.predict(&tokens).unwrap();

    assert_eq!(embedding.0.len(), 512);
    // In fallback: even length → 0.95, odd → 0.85
    assert!((score - 0.85).abs() < 1e-6);

    assert_eq!(predictor.parameter_count(), 0);
    assert!((predictor.size_mb() - 0.0).abs() < 1e-6);
}

#[test]
fn test_predictor_input_validation() {
    let mut config = ConsensusConfig::default();
    config.predictor_model_path = PathBuf::from("nonexistent.pt");

    let predictor = Predictor::new(&config).unwrap();

    // Empty input
    assert!(predictor.predict(&[]).is_err());

    // Too long
    let too_long = vec![0u16; 600];
    assert!(predictor.predict(&too_long).is_err());

    // Valid length
    let valid = vec![0u16; 512];
    assert!(predictor.predict(&valid).is_ok());
}

#[test]
fn test_validator_creation_and_voting_power() {
    let address = [1u8; 20];
    let stake = 1_000_000_000_000_000_000u128; // 1e18
    let pubkey = vec![0u8; 32];

    let mut validator = Validator::new(address, stake, pubkey);

    assert_eq!(validator.stake, stake);
    assert!((validator.reputation - 0.5).abs() < 1e-9);
    assert_eq!(validator.voting_power, (stake as f64 * 0.5) as u128);
    assert!(validator.is_active(100));

    // Update reputation
    validator.reputation = 0.9;
    validator.update_voting_power();
    assert_eq!(validator.voting_power, (stake as f64 * 0.9) as u128);
}

#[test]
fn test_validator_slashing_and_jailing() {
    let address = [1u8; 20];
    let stake = 1_000u128;
    let mut validator = Validator::new(address, stake, vec![]);

    // Slash 10%
    validator.slash(0.1, SlashingReason::DoubleSign, 100);

    assert_eq!(validator.stake, 900);
    assert!((validator.reputation - 0.25).abs() < 1e-9); // 0.5 * 0.5
    assert_eq!(validator.slashing_events.len(), 1);

    // Jail for 1000 blocks
    validator.jail(1000, 100);
    assert!(validator.is_jailed(500));
    assert!(!validator.is_jailed(1100));

    validator.unjail();
    assert!(!validator.is_jailed(500));
}

#[tokio::test]
async fn test_validator_set_basic_operations() {
    let mut set = ValidatorSet::new(1000 * 10u128.pow(18), 1000);

    let val1 = Validator::new([1u8; 20], 2000 * 10u128.pow(18), vec![0u8; 32]);
    let val2 = Validator::new([2u8; 20], 3000 * 10u128.pow(18), vec![0u8; 32]);

    set.add_validator(val1.clone()).await.unwrap();
    set.add_validator(val2.clone()).await.unwrap();

    assert_eq!(set.len().await, 2);

    let total_power = set.total_voting_power().await.unwrap();
    assert_eq!(total_power, (2000 * 10u128.pow(18) as f64 * 0.5) as u128 + (3000 * 10u128.pow(18) as f64 * 0.5) as u128);

    // Check quorum (2/3)
    let quorum = set.has_quorum(total_power / 2).await;
    assert!(!quorum); // half power
    let quorum = set.has_quorum((total_power * 2 / 3) + 1).await;
    assert!(quorum);
}

#[tokio::test]
async fn test_validator_set_epoch_processing() {
    let mut set = ValidatorSet::new(100 * 10u128.pow(18), 100);

    let mut val1 = Validator::new([1u8; 20], 1000 * 10u128.pow(18), vec![]);
    val1.reputation = 0.95;
    set.add_validator(val1).await.unwrap();

    let mut val2 = Validator::new([2u8; 20], 200 * 10u128.pow(18), vec![]);
    val2.reputation = 0.05; // Low reputation
    set.add_validator(val2).await.unwrap();

    // Trigger epoch (block multiple of epoch_length, assume 100)
    set.process_epoch(100).await.unwrap();

    // val2 should be dropped due to reputation < 0.1
    assert_eq!(set.len().await, 1);

    let remaining = set.validators().await.unwrap();
    assert_eq!(remaining[0].address, [1u8; 20]);
    assert!(remaining[0].reputation < 0.95); // Decay applied
}

#[tokio::test]
async fn test_validator_set_encoder_update_quorum() {
    let mut set = ValidatorSet::new(100 * 10u128.pow(18), 100);

    // 3 validators with equal stake → equal power
    for i in 1..=3 {
        let mut val = Validator::new([i; 20], 1000 * 10u128.pow(18), vec![]);
        val.reputation = 1.0;
        val.update_voting_power();
        set.add_validator(val).await.unwrap();
    }

    let total_power = set.total_voting_power().await.unwrap();
    let new_hash = [42u8; 32];

    // 2 validators vote yes → 2/3 power
    set.vote_encoder_update([1u8; 20], new_hash, true).await.unwrap();
    set.vote_encoder_update([2u8; 20], new_hash, true).await.unwrap();

    // Process epoch → should apply update
    set.process_epoch(100).await.unwrap();
    let current = set.current_encoder_hash().await.unwrap();
    assert_eq!(current, new_hash);

    // Insufficient (only 1 vote)
    let mut set2 = set.clone();
    set2.propose_encoder_update([99u8; 32]).await.unwrap();
    set2.vote_encoder_update([1u8; 20], [99u8; 32], true).await.unwrap();
    set2.process_epoch(200).await.unwrap();
    let current2 = set2.current_encoder_hash().await.unwrap();
    assert_ne!(current2, [99u8; 32]);
}

#[test]
fn test_dispute_resolver_committee_selection() {
    let mut rng = thread_rng();
    let mut validators = HashSet::new();
    for i in 0..100 {
        validators.insert([i as u8; 20]);
    }

    let resolver = DisputeResolver::new(30, 0.1, 0.5); // committee size 30

    let committee = resolver.select_committee(&validators, 12345, &mut rng);
    assert_eq!(committee.len(), 30);
    // Deterministic for same seed
    let committee2 = resolver.select_committee(&validators, 12345, &mut rng);
    assert_eq!(committee, committee2);
}

#[tokio::test]
async fn test_dispute_resolution_guilty_case() {
    let resolver = DisputeResolver::new(10, 0.2, 0.6); // small committee for test

    let accused = [1u8; 20];
    let accuser = [2u8; 20];

    let evidence = DisputeEvidence {
        evidence_type: EvidenceType::Signature,
        data: vec![0u8; 64],
        proof: vec![0u8; 96],
        witness_signatures: vec![],
        timestamp: 1234567890,
        block_height: 100,
        block_hash: [0u8; 32],
    };

    let mut dispute = Dispute {
        dispute_id: 1,
        accused_validator: accused,
        accuser,
        dispute_type: DisputeType::DoubleSign,
        evidence,
        accuser_stake: 1000,
        accused_stake: 10000,
        status: DisputeStatus::Pending,
        committee: None,
        votes: None,
        result: None,
        created_at: 1234567890,
        timeout_sec: 30,
    };

    // Select committee
    let mut validators = HashSet::new();
    for i in 0..20 {
        validators.insert([i as u8; 20]);
    }
    let committee = resolver.select_committee(&validators, dispute.dispute_id, &mut thread_rng());
    dispute.committee = Some(committee.clone());

    // Simulate 8/10 guilty votes (80% > 66% threshold implied)
    let mut votes = HashMap::new();
    for (i, voter) in committee.iter().enumerate() {
        let guilty = i < 8;
        votes.insert(*voter, resolver::Vote {
            voter: *voter,
            vote: guilty,
            confidence: if guilty { 0.95 } else { 0.9 },
            justification: "test".into(),
            signature: vec![], // skipped verification
        });
    }
    dispute.votes = Some(votes);

    let result = resolver.resolve_dispute(&dispute).await.unwrap();

    assert!(result.guilty);
    assert!((result.slashing_percentage - 0.2).abs() < 1e-9); // 20% slashing
    assert!(result.accuser_reward > 0);
    assert_eq!(result.accuser_penalty, 0);
    assert!(!result.committee_rewards.is_empty());
}

#[tokio::test]
async fn test_dispute_resolution_innocent_case() {
    let resolver = DisputeResolver::new(10, 0.1, 0.5);

    let accused = [1u8; 20];
    let accuser = [2u8; 20];

    let mut dispute = Dispute {
        dispute_id: 2,
        accused_validator: accused,
        accuser,
        dispute_type: DisputeType::InvalidBlock,
        evidence: DisputeEvidence {
            evidence_type: EvidenceType::Statistical,
            data: vec![],
            proof: vec![],
            witness_signatures: vec![],
            timestamp: 1234567890,
            block_height: 200,
            block_hash: [0u8; 32],
        },
        accuser_stake: 5000,
        accused_stake: 10000,
        status: DisputeStatus::Pending,
        committee: None,
        votes: None,
        result: None,
        created_at: 1234567890,
        timeout_sec: 30,
    };

    // Small committee
    let committee: Vec<[u8; 20]> = (0..10).map(|i| [i as u8; 20]).collect();
    dispute.committee = Some(committee.clone());

    // 7 innocent votes, 3 guilty → innocent majority
    let mut votes = HashMap::new();
    for (i, voter) in committee.iter().enumerate() {
        let guilty = i >= 7; // only last 3 vote guilty
        votes.insert(*voter, resolver::Vote {
            voter: *voter,
            vote: guilty,
            confidence: 0.9,
            justification: "test".into(),
            signature: vec![],
        });
    }
    dispute.votes = Some(votes);

    let result = resolver.resolve_dispute(&dispute).await.unwrap();

    assert!(!result.guilty);
    assert_eq!(result.slashing_percentage, 0.0);
    assert_eq!(result.accuser_reward, 0);
    assert!(result.accuser_penalty > 0); // 10% penalty on false accusation
}