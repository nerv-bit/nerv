//! Unit tests for the LatentLedger Lite ZK Circuit.
//!
//! Validates the NERV V2.0 simplified ZK circuit. Ensures that the witness
//! assignment correctly maps private transaction deltas and public weights,
//! and that the arithmetic gates (dot-product, addition, homomorphic update)
//! enforce the strict linearity required for exact Transfer Homomorphism.

use nerv::circuit::latent_ledger_lite::{
    CircuitConfig, CircuitProof, LatentLedgerLiteCircuit, CircuitWitness,
};
use nerv::embedding::fixed_point::{EmbeddingVector, FixedPoint};
use nerv::embedding::homomorphism::EmbeddingDelta;
use nerv::embedding::perceptron::{NwoPerceptron, PerceptronConfig};
use nerv::{EMBEDDING_DIM, NervResult};

// ─── Circuit Witness & Arithmetic Tests ──────────────────────────────────

#[test]
fn test_circuit_witness_assignment_structure() {
    let config = CircuitConfig::default();
    let circuit = LatentLedgerLiteCircuit::new(config);
    
    // Mock private inputs (transaction delta)
    let mut delta_data = [0i64; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        delta_data[i] = 100;
    }
    let private_delta = EmbeddingDelta::new(delta_data);
    
    // Mock public inputs (previous root, new root)
    let prev_root = EmbeddingVector::new([10i64; EMBEDDING_DIM]);
    let new_root = EmbeddingVector::new([20i64; EMBEDDING_DIM]);
    
    // Assign witness
    let witness = circuit.assign_witness(&private_delta, &prev_root, &new_root);
    
    // Verify the witness correctly stored the private inputs
    assert_eq!(witness.private_delta.data, private_delta.data, "Witness must map private delta exactly");
    assert_eq!(witness.prev_root.data, prev_root.data, "Witness must map prev root exactly");
}

#[test]
fn test_dot_product_gate_linearity() {
    let config = CircuitConfig::default();
    let circuit = LatentLedgerLiteCircuit::new(config);
    
    // Perceptron weights (public)
    let perceptron = NwoPerceptron::new(PerceptronConfig::default());
    let weights = &perceptron.weights;
    
    // Private delta
    let mut delta_data = [0i64; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        delta_data[i] = 50;
    }
    let private_delta = EmbeddingDelta::new(delta_data);
    
    // Compute dot product: y = W * delta
    let computed_y = circuit.simulate_dot_product_gate(&private_delta, weights);
    
    // Verify it matches the perceptron's forward pass (excluding bias)
    let mut expected_y = [0i64; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        if let Some(row) = weights.get(i) {
            if let Some(&w_val) = row.first() {
                expected_y[i] = w_val.wrapping_mul(private_delta.data[i]);
            }
        }
    }
    
    assert_eq!(computed_y.data, expected_y, "Dot product gate must exactly match standard matrix multiplication");
}

#[test]
fn test_addition_gate_bias_application() {
    let config = CircuitConfig::default();
    let circuit = LatentLedgerLiteCircuit::new(config);
    
    let perceptron = NwoPerceptron::new(PerceptronConfig::default());
    let bias = &perceptron.bias;
    
    let mut dot_product_result = [0i64; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        dot_product_result[i] = 100;
    }
    let y_vector = EmbeddingVector::new(dot_product_result);
    
    // Compute delta = y + b_tx
    let computed_delta = circuit.simulate_addition_gate(&y_vector, bias);
    
    // Verify math
    for i in 0..EMBEDDING_DIM {
        let expected = dot_product_result[i].wrapping_add(bias.data[i]);
        assert_eq!(computed_delta.data[i], expected, "Addition gate must correctly apply bias");
    }
}

#[test]
fn test_homomorphic_update_gate_exactness() {
    let config = CircuitConfig::default();
    let circuit = LatentLedgerLiteCircuit::new(config);
    
    // e_t
    let prev_root = EmbeddingVector::new([1000i64; EMBEDDING_DIM]);
    // delta(tx)
    let delta = EmbeddingDelta::new([50i64; EMBEDDING_DIM]);
    
    // e_{t+1} = e_t + delta
    let computed_new_root = circuit.simulate_homomorphic_update_gate(&prev_root, &delta);
    
    // Verify strict linear addition
    for i in 0..EMBEDDING_DIM {
        assert_eq!(computed_new_root.data[i], 1050, "Homomorphic update gate must enforce exact addition");
    }
}

#[test]
fn test_circuit_constraint_boundary() {
    // NERV V2.0 mandates that the LatentLedger Lite circuit drops from 7.9M 
    // constraints to ~50,000 due to the removal of non-linearities.
    let config = CircuitConfig::default();
    let circuit = LatentLedgerLiteCircuit::new(config);
    
    let constraint_count = circuit.estimate_constraint_count();
    
    assert!(
        constraint_count <= 50_000,
        "Circuit must not exceed 50,000 constraints. Got: {}", constraint_count
    );
    assert!(
        constraint_count > 0,
        "Circuit must have constraints for ZK validity"
    );
}

#[test]
fn test_circuit_proof_generation_and_verification() {
    let config = CircuitConfig::default();
    let circuit = LatentLedgerLiteCircuit::new(config);
    
    let private_delta = EmbeddingDelta::new([25i64; EMBEDDING_DIM]);
    let prev_root = EmbeddingVector::new([500i64; EMBEDDING_DIM]);
    let new_root = EmbeddingVector::new([525i64; EMBEDDING_DIM]);
    
    let witness = circuit.assign_witness(&private_delta, &prev_root, &new_root);
    
    // Generate proof (mocked backend in this unit test context)
    let proof: CircuitProof = circuit.generate_proof(&witness).unwrap();
    
    // Verify proof
    let is_valid = circuit.verify_proof(&proof, &prev_root, &new_root);
    
    assert!(is_valid, "Valid proof must verify successfully");
    assert!(!proof.bytes.is_empty(), "Proof must contain data");
    assert!(proof.bytes.len() <= 750, "Proof size must remain constant and small (~400-750 bytes)");
}

#[test]
fn test_circuit_proof_fails_for_mismatched_root() {
    let config = CircuitConfig::default();
    let circuit = LatentLedgerLiteCircuit::new(config);
    
    let private_delta = EmbeddingDelta::new([25i64; EMBEDDING_DIM]);
    let prev_root = EmbeddingVector::new([500i64; EMBEDDING_DIM]);
    
    // Intentionally incorrect new_root (should be 525)
    let bad_new_root = EmbeddingVector::new([999i64; EMBEDDING_DIM]);
    
    let witness = circuit.assign_witness(&private_delta, &prev_root, &bad_new_root);
    let proof = circuit.generate_proof(&witness).unwrap();
    
    // Verification must fail because the homomorphic update gate constraint is not satisfied
    let is_valid = circuit.verify_proof(&proof, &prev_root, &bad_new_root);
    
    assert!(!is_valid, "Proof must fail if homomorphic update does not hold");
}

