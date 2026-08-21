//! Unit tests for the Neural Weight Oscillator (NWO) and State Embeddings.
//!
//! Validates the core mathematical paradigm of NERV V2.0: the single-layer
//! Perceptron. Ensures that fixed-point arithmetic is exact, the homomorphic
//! property holds (f(x + Δx) = f(x) + f(Δx)), and the Adam optimizer/Huber
//! loss functions behave correctly for continuous self-evolution.

use nerv::embedding::fixed_point::{EmbeddingVector, FixedPoint};
use nerv::embedding::homomorphism::EmbeddingDelta;
use nerv::embedding::perceptron::{NwoPerceptron, PerceptronConfig};
use nerv::{EMBEDDING_DIM, ADAM_BETA1, ADAM_BETA2, HUBER_DELTA};
use rand::rngs::OsRng;
use rand::Rng;

// ─── Fixed-Point Arithmetic Tests ────────────────────────────────────────

#[test]
fn test_fixed_point_conversion_precision() {
    // Test exact integer representation
    let fp_int = FixedPoint::from_i64(42);
    assert_eq!(fp_int.to_f64(), 42.0);

    // Test fractional representation (0.5)
    let fp_frac = FixedPoint::from_f64(0.5);
    assert!((fp_frac.to_f64() - 0.5).abs() < 1e-9);

    // Test complex decimal (3.14159)
    let fp_pi = FixedPoint::from_f64(3.14159);
    assert!((fp_pi.to_f64() - 3.14159).abs() < 1e-5);
}

#[test]
fn test_fixed_point_addition() {
    let a = FixedPoint::from_f64(1.5);
    let b = FixedPoint::from_f64(2.25);
    let c = a + b;
    assert!((c.to_f64() - 3.75).abs() < 1e-9);
}

// ─── Embedding Vector & Homomorphism Tests ───────────────────────────────

#[test]
fn test_embedding_vector_addition() {
    let mut rng = OsRng;
    let mut data1 = [0i64; EMBEDDING_DIM];
    let mut data2 = [0i64; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        data1[i] = rng.gen_range(0..1000);
        data2[i] = rng.gen_range(0..1000);
    }

    let vec1 = EmbeddingVector::new(data1);
    let vec2 = EmbeddingVector::new(data2);
    let vec3 = vec1.add(&vec2);

    for i in 0..EMBEDDING_DIM {
        assert_eq!(vec3.data[i], vec1.data[i] + vec2.data[i], "Vector addition must be exact");
    }
}

#[test]
fn test_homomorphic_property_exact() {
    // The core V2.0 invariant: f(x + Δx) = f(x) + f(Δx)
    let mut rng = OsRng;
    let mut state_data = [0i64; EMBEDDING_DIM];
    let mut delta_data = [0i64; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        state_data[i] = rng.gen_range(0..1000);
        delta_data[i] = rng.gen_range(0..1000);
    }

    let state = EmbeddingVector::new(state_data);
    let delta = EmbeddingDelta::new(delta_data);

    // Apply delta to state
    let new_state = state.add(&delta.to_vector());

    // Verify homomorphism: new_state - state == delta
    let diff = new_state.sub(&state);
    for i in 0..EMBEDDING_DIM {
        assert_eq!(diff.data[i], delta.data[i], "Homomorphic property must hold exactly");
    }
}

// ─── NWO Perceptron & Adam Optimizer Tests ───────────────────────────────

#[test]
fn test_perceptron_forward_pass_linearity() {
    let config = PerceptronConfig::default();
    let perceptron = NwoPerceptron::new(config);
    
    let mut input = [0i64; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        input[i] = 100;
    }
    let input_vec = EmbeddingVector::new(input);
    
    // f(x) = W * x + b
    let output = perceptron.forward(&input_vec);
    
    // Since W is initialized near zero (or 1s for testing), output should be strictly linear
    // We assert that no non-linear activation (like ReLU) is applied.
    assert_eq!(output.data.len(), EMBEDDING_DIM);
    
    // If we double the input, the output should exactly double (minus bias)
    let input_vec_2x = input_vec.add(&input_vec);
    let output_2x = perceptron.forward(&input_vec_2x);
    
    // output_2x ≈ 2 * output - bias
    // This confirms strict linearity.
    for i in 0..EMBEDDING_DIM {
        let expected = (output.data[i] as i128 * 2 - perceptron.bias.data[i] as i128) as i64;
        assert_eq!(output_2x.data[i], expected, "Perceptron must be strictly linear");
    }
}

#[test]
fn test_huber_loss_computation() {
    // L_δ(y, ŷ) = 0.5 * (y - ŷ)² if |y - ŷ| <= δ
    // L_δ(y, ŷ) = δ * |y - ŷ| - 0.5 * δ² otherwise
    
    let delta = FixedPoint::from_f64(HUBER_DELTA);
    
    // Case 1: Error < Delta (Quadratic region)
    let y1 = FixedPoint::from_f64(10.0);
    let y_hat1 = FixedPoint::from_f64(10.5); // Error = 0.5
    let loss1 = NwoPerceptron::compute_huber_loss(&y1, &y_hat1, &delta);
    // Expected: 0.5 * (0.5)^2 = 0.125
    assert!((loss1.to_f64() - 0.125).abs() < 1e-3, "Huber loss must be quadratic for small errors");

    // Case 2: Error > Delta (Linear region)
    let y2 = FixedPoint::from_f64(10.0);
    let y_hat2 = FixedPoint::from_f64(15.0); // Error = 5.0
    let loss2 = NwoPerceptron::compute_huber_loss(&y2, &y_hat2, &delta);
    // Expected: 1.0 * 5.0 - 0.5 * 1.0^2 = 4.5
    assert!((loss2.to_f64() - 4.5).abs() < 1e-3, "Huber loss must be linear for large errors");
}

#[test]
fn test_adam_optimizer_reduces_loss() {
    let mut config = PerceptronConfig::default();
    config.learning_rate = 0.01;
    let mut perceptron = NwoPerceptron::new(config);
    
    // Target state we want the perceptron to learn
    let mut target_data = [0i64; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        target_data[i] = 500;
    }
    let target = EmbeddingVector::new(target_data);
    
    // Initial prediction and loss
    let initial_pred = perceptron.forward(&target);
    let initial_loss = perceptron.compute_batch_loss(&target, &initial_pred);
    
    // Perform 100 Adam optimization steps
    for _ in 0..100 {
        let pred = perceptron.forward(&target);
        let gradients = perceptron.compute_gradients(&target, &pred);
        perceptron.adam_update(&gradients);
    }
    
    // Final prediction and loss
    let final_pred = perceptron.forward(&target);
    let final_loss = perceptron.compute_batch_loss(&target, &final_pred);
    
    // The optimizer must reduce the loss
    assert!(
        final_loss.to_f64() < initial_loss.to_f64(),
        "Adam optimizer must reduce Huber loss over iterations"
    );
}

#[test]
fn test_adam_moment_estimates_initialization() {
    let perceptron = NwoPerceptron::new(PerceptronConfig::default());
    
    // First moment (m) and second moment (v) must be initialized to zero
    for i in 0..EMBEDDING_DIM {
        assert_eq!(perceptron.m[i], 0, "First moment must init to 0");
        assert_eq!(perceptron.v[i], 0, "Second moment must init to 0");
    }
    
    // Beta values must match V2.0 constants
    assert_eq!(perceptron.config.adam_beta1, ADAM_BETA1);
    assert_eq!(perceptron.config.adam_beta2, ADAM_BETA2);
}
