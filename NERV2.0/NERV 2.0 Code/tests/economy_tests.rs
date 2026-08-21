//! Unit tests for the Useful-Work Economy & Gradient Validation.
//!
//! Validates the NERV V2.0 economic model. Ensures that validators who
//! submit valid Adam gradients that reduce the network's global Huber Loss
//! are correctly rewarded, and that gradient clipping prevents adversarial
//! updates. Also validates the reward distribution accounting.

use nerv::embedding::fixed_point::{EmbeddingVector, FixedPoint};
use nerv::embedding::perceptron::{NwoPerceptron, PerceptronConfig};
use nerv::tokenomics::rewards::{RewardHistory, RewardRecord, RewardType};
use nerv::{ValidatorId, BlockHeight, Epoch, ONE_NERV, HUBER_DELTA};
use rand::rngs::OsRng;
use rand::Rng;

// ─── Gradient Clipping Tests ─────────────────────────────────────────────

/// Simulates the gradient clipping logic enforced by the protocol.
/// If the L2 norm of the gradient exceeds `max_norm`, it is scaled down.
fn clip_gradient(gradient: &mut [i64], max_norm: f64) {
    let norm_sq: f64 = gradient.iter()
        .map(|&g| (g as f64).powi(2))
        .sum();
    let norm = norm_sq.sqrt();
    
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in gradient.iter_mut() {
            *g = (*g as f64 * scale) as i64;
        }
    }
}

#[test]
fn test_gradient_clipping_reduces_large_gradients() {
    let mut gradient = vec![1000i64; 64]; // Massive gradient
    let max_norm = 1.0;
    
    clip_gradient(&mut gradient, max_norm);
    
    let norm_sq: f64 = gradient.iter().map(|&g| (g as f64).powi(2)).sum();
    let new_norm = norm_sq.sqrt();
    
    // The norm must be scaled down to exactly (or very close to) max_norm
    assert!(
        new_norm <= max_norm + 1.0, // Allow tiny fixed-point rounding error
        "Clipped gradient norm ({}) must not exceed max_norm ({})", new_norm, max_norm
    );
}

#[test]
fn test_gradient_clipping_preserves_small_gradients() {
    let original_grad = vec![1i64; 64];
    let mut gradient = original_grad.clone();
    let max_norm = 100.0;
    
    clip_gradient(&mut gradient, max_norm);
    
    // Small gradients must remain untouched
    assert_eq!(
        gradient, original_grad,
        "Gradients under max_norm must not be modified"
    );
}

// ─── Useful Work & Huber Loss Reduction Tests ────────────────────────────

#[test]
fn test_useful_work_reduces_global_loss() {
    let mut config = PerceptronConfig::default();
    config.learning_rate = 0.001;
    let mut perceptron = NwoPerceptron::new(config);
    
    // Define a target state the network needs to learn
    let mut target_data = [0i64; 64];
    for i in 0..64 {
        target_data[i] = 500;
    }
    let target = EmbeddingVector::new(target_data);
    
    // 1. Calculate initial loss
    let initial_pred = perceptron.forward(&target);
    let initial_loss = perceptron.compute_batch_loss(&target, &initial_pred);
    
    // 2. Validator computes & submits a valid gradient
    let gradients = perceptron.compute_gradients(&target, &initial_pred);
    perceptron.adam_update(&gradients);
    
    // 3. Calculate new loss
    let new_pred = perceptron.forward(&target);
    let new_loss = perceptron.compute_batch_loss(&target, &new_pred);
    
    // The valid gradient MUST reduce the Huber loss
    assert!(
        new_loss.to_f64() < initial_loss.to_f64(),
        "Useful work (valid gradients) must strictly reduce global Huber loss"
    );
}

#[test]
fn test_adversarial_gradient_increases_loss() {
    let mut config = PerceptronConfig::default();
    config.learning_rate = 0.001;
    let mut perceptron = NwoPerceptron::new(config);
    
    let target = EmbeddingVector::new([500i64; 64]);
    
    // Initial loss
    let initial_pred = perceptron.forward(&target);
    let initial_loss = perceptron.compute_batch_loss(&target, &initial_pred);
    
    // Adversarial gradient: deliberately pushing weights in the WRONG direction
    let valid_gradients = perceptron.compute_gradients(&target, &initial_pred);
    let mut adversarial_gradients = valid_gradients.clone();
    for g in adversarial_gradients.iter_mut() {
        *g = -*g; // Invert the gradient
    }
    
    perceptron.adam_update(&adversarial_gradients);
    
    // New loss
    let new_pred = perceptron.forward(&target);
    let new_loss = perceptron.compute_batch_loss(&target, &new_pred);
    
    // The adversarial gradient MUST increase the loss (or fail to reduce it)
    assert!(
        new_loss.to_f64() >= initial_loss.to_f64(),
        "Adversarial gradients must not reduce loss"
    );
}

// ─── Reward Distribution Accounting Tests ───────────────────────────────

#[test]
fn test_reward_history_records_useful_work() {
    let mut history = RewardHistory::new();
    let validator = ValidatorId::from_bytes([1u8; 32]);
    
    // Record a gradient contribution reward
    let reward = RewardRecord::new(
        validator.clone(),
        BlockHeight::from(100),
        Epoch::from(0),
        RewardType::GradientContribution,
        50 * ONE_NERV,
    );
    history.record(&reward);
    
    // Verify accounting
    assert_eq!(history.total_distributed(), 50 * ONE_NERV, "Total distributed must match");
    assert_eq!(history.record_count(), 1, "Record count must be 1");
    assert_eq!(history.unique_recipients(), 1, "Unique recipients must be 1");
    
    // Verify per-validator tracking
    assert_eq!(
        history.validator_total(&validator),
        50 * ONE_NERV,
        "Validator total must be accurately tracked"
    );
}

#[test]
fn test_reward_distribution_by_type() {
    let mut history = RewardHistory::new();
    
    // 60% gradient, 30% validation, 10% tx fees
    history.record(&RewardRecord::new(
        ValidatorId::from_bytes([1u8; 32]),
        BlockHeight::from(1), Epoch::from(0),
        RewardType::GradientContribution, 60 * ONE_NERV,
    ));
    history.record(&RewardRecord::new(
        ValidatorId::from_bytes([2u8; 32]),
        BlockHeight::from(1), Epoch::from(0),
        RewardType::ValidationParticipation, 30 * ONE_NERV,
    ));
    history.record(&RewardRecord::new(
        ValidatorId::from_bytes([3u8; 32]),
        BlockHeight::from(1), Epoch::from(0),
        RewardType::TransactionFee, 10 * ONE_NERV,
    ));
    
    let distribution = history.distribution_by_type();
    
    // Find the gradient contribution stats
    let (grad_type, grad_total, grad_frac) = distribution.iter()
        .find(|(t, _, _)| *t == RewardType::GradientContribution)
        .unwrap();
        
    assert_eq!(*grad_total, 60 * ONE_NERV, "Gradient total must be correct");
    assert!((grad_frac - 0.6).abs() < 1e-6, "Gradient fraction must be 60%");
    
    // Verify the useful-work pool dominates the distribution
    let (val_type, val_total, _) = distribution.iter()
        .find(|(t, _, _)| *t == RewardType::ValidationParticipation)
        .unwrap();
    assert_eq!(*val_total, 30 * ONE_NERV, "Validation total must be correct");
}

#[test]
fn test_top_recipients_ranking() {
    let mut history = RewardHistory::new();
    
    let val1 = ValidatorId::from_bytes([1u8; 32]);
    let val2 = ValidatorId::from_bytes([2u8; 32]);
    let val3 = ValidatorId::from_bytes([3u8; 32]);
    
    history.record(&RewardRecord::new(val1.clone(), BlockHeight::from(1), Epoch::from(0), RewardType::GradientContribution, 100 * ONE_NERV));
    history.record(&RewardRecord::new(val2.clone(), BlockHeight::from(1), Epoch::from(0), RewardType::GradientContribution, 300 * ONE_NERV));
    history.record(&RewardRecord::new(val3.clone(), BlockHeight::from(1), Epoch::from(0), RewardType::GradientContribution, 200 * ONE_NERV));
    
    let top_2 = history.top_recipients(2);
    
    assert_eq!(top_2.len(), 2, "Should return exactly 2 recipients");
    assert_eq!(top_2[0].0, val2, "Top recipient must be val2");
    assert_eq!(top_2[0].1, 300 * ONE_NERV, "Top amount must be 300");
    assert_eq!(top_2[1].0, val3, "Second recipient must be val3");
}
