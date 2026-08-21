//! End-to-End Integration Tests for NERV V2.0 Node Lifecycle.
//!
//! Validates the complete block production and finality pipeline:
//! 1. Threshold-encrypt a batch of private transactions (Mempool).
//! 2. Validators perform threshold decryption ceremony.
//! 3. NWO Perceptron computes homomorphic state delta and updates embedding.
//! 4. Adam optimizer & Huber loss compute useful-work gradients.
//! 5. Validators vote on the new embedding root (67% quorum).
//! 6. Block finality achieved and VDW generated.

use nerv::embedding::fixed_point::{EmbeddingVector, FixedPoint};
use nerv::embedding::homomorphism::EmbeddingDelta;
use nerv::embedding::perceptron::{NwoPerceptron, PerceptronConfig};
use nerv::privacy::dkg::{
    DkgScalar, DkgPublicKey, DkgSecretShare, Polynomial, 
    FeldmanCommitment, ThresholdCiphertext,
};
use nerv::privacy::threshold_dec::{PartialDecryption, DecryptionCeremony};
use nerv::privacy::vdw::Vdw;
use nerv::consensus::voting::{check_quorum, compute_voting_weight}; // Assuming module path based on standard structure
use nerv::{
    ValidatorId, StakeAmount, ReputationScore, VotingWeight,
    BlockHeight, EmbeddingRoot, TxHash, EMBEDDING_DIM, HUBER_DELTA,
    ONE_NERV,
};
use bls12_381::Scalar as BlsScalar;
use std::sync::Arc;

// ─── E2E Test Setup Helpers ──────────────────────────────────────────────

/// Simulates a DKG setup for `n` validators with threshold `t`.
fn setup_dkg(n: usize, t: usize) -> (DkgPublicKey, Vec<DkgSecretShare>) {
    let mut rng = rand::thread_rng();
    let mut coefficients = Vec::new();
    for _ in 0..t {
        coefficients.push(DkgScalar::random());
    }
    
    let poly = Polynomial { coefficients };
    
    // Generate commitments and public key
    let commitments: Vec<FeldmanCommitment> = poly.coefficients.iter()
        .map(FeldmanCommitment::commit)
        .collect();
    
    let mut pk_point = bls12_381::G1Projective::identity();
    for c in &commitments {
        let point = bls12_381::G1Affine::from_compressed(&c.point.clone().try_into().unwrap()).unwrap();
        pk_point += bls12_381::G1Projective::from(point);
    }
    let pk_bytes = bls12_381::G1Affine::from(pk_point).to_compressed();
    
    let dkg_pk = DkgPublicKey {
        point: pk_bytes.as_ref().to_vec(),
        hash: nerv::utils::blake3_hash(pk_bytes.as_ref()),
        session_id: [0u8; 32],
        threshold: t,
        num_participants: n,
    };
    
    // Generate shares for n participants
    let mut shares = Vec::new();
    for i in 1..=n {
        let index = i as u32;
        let share_val = poly.evaluate(&DkgScalar::from_bls_scalar(&BlsScalar::from(index as u64)));
        shares.push(DkgSecretShare::new(
            ValidatorId::from_bytes([index as u8; 32]),
            index,
            share_val,
            [0u8; 32],
        ));
    }
    
    (dkg_pk, shares)
}

// ─── E2E Node Lifecycle Test ─────────────────────────────────────────────

#[tokio::test]
async fn test_e2e_block_production_and_finality() {
    // 1. Setup Network State
    let n_validators = 5;
    let threshold = 3; // 60% threshold for DKG
    let (dkg_pk, validator_shares) = setup_dkg(n_validators, threshold);
    
    let mut config = PerceptronConfig::default();
    config.learning_rate = 0.01;
    let mut perceptron = NwoPerceptron::new(config);
    
    // Initial canonical state (e_t)
    let initial_state_data = [100i64; EMBEDDING_DIM];
    let current_state = EmbeddingVector::new(initial_state_data);
    
    // 2. Mempool Phase: Threshold encrypt a batch of private transactions
    let tx_payload = b"batch_of_10000_private_transactions_delta_data";
    let ciphertext = ThresholdCiphertext::encrypt(tx_payload, &dkg_pk, [0u8; 32]).unwrap();
    
    // 3. Decryption Ceremony: `t` validators decrypt the batch
    let mut ceremony = DecryptionCeremony::new(ciphertext.clone(), threshold);
    for share in validator_shares.iter().take(threshold) {
        let partial = PartialDecryption::compute(&ciphertext, share, share.participant_id.clone());
        ceremony.add_partial(partial).unwrap();
    }
    
    let decrypted_payload = ceremony.combine().unwrap();
    assert_eq!(decrypted_payload, tx_payload.to_vec(), "DKG decryption must yield original payload");
    
    // 4. State Transition Phase: Compute homomorphic delta
    // Simulate extracting delta from the decrypted payload
    let delta_data = [10i64; EMBEDDING_DIM];
    let batch_delta = EmbeddingDelta::new(delta_data);
    
    // Apply delta to state: e_{t+1} = e_t + delta
    let new_state = current_state.add(&batch_delta.to_vector());
    
    // 5. Self-Evolution Phase: Adam backprop on Huber Loss
    // Target is the actual new_state, prediction is perceptron's output
    let pred_state = perceptron.forward(&new_state);
    let loss = perceptron.compute_batch_loss(&new_state, &pred_state);
    
    // Ensure loss is computed (useful-work)
    assert!(loss.to_f64() >= 0.0, "Huber loss must be non-negative");
    
    // Compute gradients and update weights
    let gradients = perceptron.compute_gradients(&new_state, &pred_state);
    let prev_weights = perceptron.weights.clone();
    perceptron.adam_update(&gradients);
    
    // Verify weights evolved
    let mut weights_changed = false;
    for i in 0..EMBEDDING_DIM {
        if perceptron.weights[i] != prev_weights[i] {
            weights_changed = true;
            break;
        }
    }
    assert!(weights_changed, "NWO Perceptron weights must evolve per-block via Adam");
    
    // 6. Consensus Phase: Validators vote on the new embedding root
    let new_root = EmbeddingRoot::from_bytes(nerv::utils::blake3_hash(&new_state.to_bytes()));
    
    // Simulate 4 out of 5 validators voting (80% > 67% quorum)
    let mut total_weight = VotingWeight::ZERO;
    let mut achieved_weight = VotingWeight::ZERO;
    
    for i in 0..n_validators {
        let stake = StakeAmount::from_nerv(10_000);
        let rep = ReputationScore::PERFECT;
        let weight = compute_voting_weight(stake, rep);
        
        total_weight = total_weight.saturating_add(weight);
        if i < 4 { // First 4 validators vote YES
            achieved_weight = achieved_weight.saturating_add(weight);
        }
    }
    
    let quorum_result = check_quorum(achieved_weight, total_weight);
    assert!(quorum_result.is_ok(), "Consensus must be reached with 80% voting power");
    
    // 7. Finality & VDW Generation
    let val_id = ValidatorId::from_bytes([1u8; 32]);
    let dilithium_sig = vec![0u8; 3293]; // Mocked for structural test
    let dkg_sig = vec![0u8; 48];         // Mocked for structural test
    
    let vdw = Vdw::generate(
        TxHash::from_bytes([1u8; 32]),
        0, // shard_id
        1, // lattice_height
        vec![0u8; 400], // delta_proof
        new_root,
        dkg_sig,
        dilithium_sig,
    );
    
    assert_eq!(vdw.lattice_height, 1, "VDW must be generated for block 1");
    assert!(vdw.is_valid_size(), "VDW must be within valid size limits");
}

#[tokio::test]
async fn test_e2e_consensus_failure_blocks_finality() {
    // 1. Setup Network State
    let (dkg_pk, _validator_shares) = setup_dkg(5, 3);
    let perceptron = NwoPerceptron::new(PerceptronConfig::default());
    let current_state = EmbeddingVector::new([100i64; EMBEDDING_DIM]);
    
    // 2. State Transition (Simulated)
    let batch_delta = EmbeddingDelta::new([10i64; EMBEDDING_DIM]);
    let new_state = current_state.add(&batch_delta.to_vector());
    let new_root = EmbeddingRoot::from_bytes(nerv::utils::blake3_hash(&new_state.to_bytes()));
    
    // 3. Consensus Phase: Only 3 out of 5 validators vote (60% < 67% quorum)
    let mut total_weight = VotingWeight::ZERO;
    let mut achieved_weight = VotingWeight::ZERO;
    
    for i in 0..5 {
        let stake = StakeAmount::from_nerv(10_000);
        let rep = ReputationScore::PERFECT;
        let weight = compute_voting_weight(stake, rep);
        
        total_weight = total_weight.saturating_add(weight);
        if i < 3 { // First 3 validators vote YES
            achieved_weight = achieved_weight.saturating_add(weight);
        }
    }
    
    let quorum_result = check_quorum(achieved_weight, total_weight);
    
    // 4. Verify finality is NOT reached
    assert!(
        quorum_result.is_err(),
        "Consensus must fail if quorum (67%) is not reached"
    );
    
    // In a real node, the block would be rejected here, and no VDW would be generated.
}


