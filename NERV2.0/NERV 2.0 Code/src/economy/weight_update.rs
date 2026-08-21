//! Weight Update Validation and Aggregation.
//!
//! This module handles the core of NERV's self-evolution:
//! validating, aggregating, and applying gradient updates to the
//! network's NWO Perceptron weights via the Adam optimizer.
//!
//! # Flow (Per Block)
//!
//! ```text
//! 1. Each validator computes:
//!    - Prediction: ŷ = W · S_t + b
//!    - Huber loss: L = Huber(ŷ, y)
//!    - Gradient: ∂L/∂W, ∂L/∂b
//!    - Proposed weights: W_new = Adam(W, ∂L/∂W)
//!
//! 2. Each validator broadcasts:
//!    - GradientSubmission { gradient, loss_before, loss_after, hash(W_new), sig }
//!
//! 3. Block producer aggregates:
//!    - Validates each submission (norm bounds, DP noise, signature)
//!    - Computes consensus: ≥67% agree on hash(W_new)?
//!    - If yes: apply the majority weight update
//!    - If no: keep current weights (safety first)
//! ```

use crate::{
    EMBEDDING_DIM, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, ADAM_LEARNING_RATE, HUBER_DELTA,
    ShardId, BlockHeight, Epoch, ValidatorId, StakeAmount, ReputationScore, VotingWeight,
    NervError, NervResult,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Gradient Submission ─────────────────────────────────────────────────

/// A validator's submitted gradient for a single block.
///
/// This is the core "useful work" that validators get paid for.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientSubmission {
    /// The submitting validator.
    pub validator_id: ValidatorId,

    /// Block height this gradient applies to.
    pub block_height: BlockHeight,

    /// Epoch at submission.
    pub epoch: Epoch,

    /// Gradient of Huber loss w.r.t. weights (∂L/∂W).
    /// Length = EMBEDDING_DIM (64).
    pub weight_gradient: Vec<f64>,

    /// Gradient of Huber loss w.r.t. bias (∂L/∂b).
    pub bias_gradient: f64,

    /// Huber loss BEFORE applying this gradient.
    pub loss_before: f64,

    /// Huber loss AFTER applying this gradient (predicted).
    pub loss_after: f64,

    /// BLAKE3 hash of the weights BEFORE update.
    pub weight_hash_before: [u8; 32],

    /// BLAKE3 hash of the proposed weights AFTER update.
    pub weight_hash_proposed: [u8; 32],

    /// L2 norm of the gradient vector.
    pub gradient_norm: f64,

    /// Dilithium-3 signature over the submission.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub signature: Vec<u8>,

    /// Timestamp (Unix epoch millis).
    pub timestamp_ms: u64,
}

impl GradientSubmission {
    /// Create a new gradient submission.
    pub fn new(
        validator_id: ValidatorId,
        block_height: BlockHeight,
        epoch: Epoch,
        weight_gradient: Vec<f64>,
        bias_gradient: f64,
        loss_before: f64,
        loss_after: f64,
        weight_hash_before: [u8; 32],
        weight_hash_proposed: [u8; 32],
    ) -> Self {
        let gradient_norm = compute_gradient_norm(&weight_gradient, bias_gradient);

        Self {
            validator_id,
            block_height,
            epoch,
            weight_gradient,
            bias_gradient,
            loss_before,
            loss_after,
            weight_hash_before,
            weight_hash_proposed,
            gradient_norm,
            signature: Vec::new(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Compute the loss reduction achieved by this gradient.
    ///
    /// Positive = improvement, negative = degradation.
    #[inline]
    pub fn loss_reduction(&self) -> f64 {
        self.loss_before - self.loss_after
    }

    /// Check if this gradient improves the loss.
    #[inline]
    pub fn is_improving(&self) -> bool {
        self.loss_after < self.loss_before
    }

    /// Compute a unique hash for this submission.
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.validator_id.as_bytes());
        hasher.update(&self.block_height.0.to_le_bytes());
        hasher.update(&self.epoch.0.to_le_bytes());
        hasher.update(&self.gradient_norm.to_le_bytes());
        hasher.update(&self.loss_before.to_le_bytes());
        hasher.update(&self.loss_after.to_le_bytes());
        hasher.update(&self.weight_hash_before);
        hasher.update(&self.weight_hash_proposed);
        hasher.finalize().into()
    }

    /// Validate the gradient submission.
    ///
    /// Checks:
    /// 1. Gradient vector has correct dimensionality
    /// 2. Gradient norm is within acceptable bounds
    /// 3. Loss values are non-negative
    /// 4. Weight hashes are non-zero
    pub fn validate(&self, max_gradient_norm: f64) -> NervResult<()> {
        // Check gradient dimensionality
        if self.weight_gradient.len() != EMBEDDING_DIM {
            return Err(NervError::Economy(format!(
                "gradient dimension mismatch: expected {}, got {}",
                EMBEDDING_DIM, self.weight_gradient.len()
            )));
        }

        // Check gradient norm
        if self.gradient_norm > max_gradient_norm {
            return Err(NervError::Economy(format!(
                "gradient norm {} exceeds maximum {}",
                self.gradient_norm, max_gradient_norm
            )));
        }

        // Check gradient norm consistency
        let computed_norm = compute_gradient_norm(&self.weight_gradient, self.bias_gradient);
        if (computed_norm - self.gradient_norm).abs() > 1e-6 {
            return Err(NervError::Economy(
                "gradient norm mismatch with actual gradient".into()
            ));
        }

        // Check loss values are finite and non-negative
        if !self.loss_before.is_finite() || self.loss_before < 0.0 {
            return Err(NervError::Economy("loss_before is invalid".into()));
        }
        if !self.loss_after.is_finite() || self.loss_after < 0.0 {
            return Err(NervError::Economy("loss_after is invalid".into()));
        }

        // Check weight hashes are non-zero
        if self.weight_hash_before == [0u8; 32] {
            return Err(NervError::Economy("weight_hash_before is zero".into()));
        }
        if self.weight_hash_proposed == [0u8; 32] {
            return Err(NervError::Economy("weight_hash_proposed is zero".into()));
        }

        Ok(())
    }
}

// ─── Weight Update Proposal ──────────────────────────────────────────────

/// A proposal for the network's new weights, aggregating
/// multiple validators' gradient submissions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightUpdateProposal {
    /// Block height of the proposal.
    pub block_height: BlockHeight,

    /// Epoch of the proposal.
    pub epoch: Epoch,

    /// Hash of the current (pre-update) weights.
    pub weight_hash_before: [u8; 32],

    /// Hash of the proposed new weights.
    pub weight_hash_proposed: [u8; 32],

    /// The aggregated weight gradient.
    pub aggregated_weight_gradient: Vec<f64>,

    /// The aggregated bias gradient.
    pub aggregated_bias_gradient: f64,

    /// Number of validators who submitted gradients.
    pub contributor_count: usize,

    /// Voting weight supporting this proposal.
    pub supporting_weight: u128,

    /// Total voting weight of all active validators.
    pub total_weight: u128,

    /// Average Huber loss before the update.
    pub avg_loss_before: f64,

    /// Predicted average Huber loss after the update.
    pub avg_loss_after: f64,

    /// Whether the ≥67% quorum was reached.
    pub quorum_reached: bool,

    /// Timestamp.
    pub timestamp_ms: u64,
}

impl WeightUpdateProposal {
    /// Check if the quorum was reached.
    pub fn is_approved(&self) -> bool {
        self.quorum_reached
    }

    /// Compute the average loss reduction.
    pub fn avg_loss_reduction(&self) -> f64 {
        self.avg_loss_before - self.avg_loss_after
    }

    /// Get the approval percentage.
    pub fn approval_percentage(&self) -> f64 {
        if self.total_weight == 0 {
            return 0.0;
        }
        (self.supporting_weight as f64 / self.total_weight as f64) * 100.0
    }
}

// ─── Weight Update Result ────────────────────────────────────────────────

/// Result of applying a weight update.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WeightUpdateResult {
    /// Weight update was successfully applied.
    Applied {
        /// Hash of the new weights.
        new_weight_hash: [u8; 32],
        /// Loss reduction achieved.
        loss_reduction: f64,
        /// Number of contributors.
        contributor_count: usize,
    },

    /// Weight update was rejected (quorum not reached).
    RejectedQuorum {
        /// Approval percentage.
        approval_pct: f64,
    },

    /// Weight update was rejected (would increase loss).
    RejectedLossIncrease {
        /// Predicted loss increase.
        loss_increase: f64,
    },

    /// Weight update was rejected (validation failure).
    RejectedInvalid {
        /// Reason.
        reason: String,
    },

    /// No submissions received; weights unchanged.
    NoSubmissions,
}

impl WeightUpdateResult {
    /// Check if the update was applied.
    pub fn is_applied(&self) -> bool {
        matches!(self, Self::Applied { .. })
    }
}

// ─── Gradient Aggregation ────────────────────────────────────────────────

/// Aggregates gradient submissions from multiple validators.
pub struct GradientAggregation {
    /// Collected submissions.
    submissions: Vec<GradientSubmission>,

    /// Maximum gradient norm for clipping.
    max_gradient_norm: f64,

    /// DP noise sigma (0.0 = disabled).
    dp_noise_sigma: f64,
}

impl GradientAggregation {
    /// Create a new aggregation context.
    pub fn new(max_gradient_norm: f64, dp_noise_sigma: f64) -> Self {
        Self {
            submissions: Vec::new(),
            max_gradient_norm,
            dp_noise_sigma,
        }
    }

    /// Add a gradient submission.
    pub fn add_submission(&mut self, submission: GradientSubmission) -> NervResult<()> {
        submission.validate(self.max_gradient_norm)?;
        self.submissions.push(submission);
        Ok(())
    }

    /// Number of submissions collected.
    pub fn count(&self) -> usize {
        self.submissions.len()
    }

    /// Aggregate all submissions by averaging the gradients.
    ///
    /// In V2.0 (no TEEs/SMPC), we simply average the gradients
    /// from all validators. This is the standard FedAvg approach.
    ///
    /// Future: Could weight by stake, reputation, or loss reduction.
    pub fn aggregate(&self) -> NervResult<(Vec<f64>, f64)> {
        if self.submissions.is_empty() {
            return Err(NervError::Economy("no submissions to aggregate".into()));
        }

        let n = self.submissions.len() as f64;

        // Average weight gradients
        let mut avg_weight_grad = vec![0.0; EMBEDDING_DIM];
        for sub in &self.submissions {
            for (i, &g) in sub.weight_gradient.iter().enumerate() {
                if i < EMBEDDING_DIM {
                    avg_weight_grad[i] += g;
                }
            }
        }
        for g in &mut avg_weight_grad {
            *g /= n;
        }

        // Average bias gradient
        let avg_bias_grad: f64 = self.submissions.iter()
            .map(|s| s.bias_gradient)
            .sum::<f64>() / n;

        // Gradient clipping
        let norm = compute_gradient_norm(&avg_weight_grad, avg_bias_grad);
        if norm > self.max_gradient_norm {
            let clip = self.max_gradient_norm / norm;
            for g in &mut avg_weight_grad {
                *g *= clip;
            }
            let clipped_bias = avg_bias_grad * clip;
            return Ok((avg_weight_grad, clipped_bias));
        }

        Ok((avg_weight_grad, avg_bias_grad))
    }

    /// Compute the consensus weight hash (majority vote).
    ///
    /// Returns the weight hash with the most votes, along with
    /// the vote count and total submissions.
    pub fn consensus_weight_hash(&self) -> Option<([u8; 32], usize)> {
        if self.submissions.is_empty() {
            return None;
        }

        let mut vote_counts: HashMap<[u8; 32], usize> = HashMap::new();
        for sub in &self.submissions {
            *vote_counts.entry(sub.weight_hash_proposed).or_insert(0) += 1;
        }

        vote_counts.into_iter().max_by_key(|&(_, count)| count)
    }

    /// Get submissions that voted for a specific weight hash.
    pub fn submissions_for_hash(&self, weight_hash: &[u8; 32]) -> Vec<&GradientSubmission> {
        self.submissions.iter()
            .filter(|s| &s.weight_hash_proposed == weight_hash)
            .collect()
    }

    /// Compute average loss before and after.
    pub fn average_losses(&self) -> (f64, f64) {
        if self.submissions.is_empty() {
            return (0.0, 0.0);
        }
        let n = self.submissions.len() as f64;
        let avg_before = self.submissions.iter().map(|s| s.loss_before).sum::<f64>() / n;
        let avg_after = self.submissions.iter().map(|s| s.loss_after).sum::<f64>() / n;
        (avg_before, avg_after)
    }

    /// Compute the weighted consensus using voting weights.
    ///
    /// Returns the weight hash with the most weighted support,
    /// the supporting weight, and the total weight.
    pub fn weighted_consensus(
        &self,
        weights: &HashMap<ValidatorId, VotingWeight>,
    ) -> Option<([u8; 32], u128, u128)> {
        if self.submissions.is_empty() {
            return None;
        }

        let mut hash_weights: HashMap<[u8; 32], u128> = HashMap::new();
        let mut total_weight: u128 = 0;

        for sub in &self.submissions {
            let w = weights.get(&sub.validator_id)
                .map(|vw| vw.0)
                .unwrap_or(0);
            *hash_weights.entry(sub.weight_hash_proposed).or_insert(0) += w;
            total_weight += w;
        }

        let (best_hash, best_weight) = hash_weights.into_iter()
            .max_by_key(|&(_, w)| w)
            .unwrap_or(([0u8; 32], 0));

        Some((best_hash, best_weight, total_weight))
    }
}

// ─── Adam Optimizer (Economy-Level) ──────────────────────────────────────

/// Adam optimizer state for the network-level weight updates.
///
/// This is the economy-level Adam that aggregates gradients from
/// all validators and applies the network-wide weight update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkAdam {
    /// First moment for weights (m_W).
    pub m_weights: Vec<f64>,

    /// Second moment for weights (v_W).
    pub v_weights: Vec<f64>,

    /// First moment for bias (m_b).
    pub m_bias: f64,

    /// Second moment for bias (v_b).
    pub v_bias: f64,

    /// Step counter.
    pub step: u64,

    /// β₁.
    pub beta1: f64,

    /// β₂.
    pub beta2: f64,

    /// ε.
    pub epsilon: f64,

    /// Learning rate α.
    pub learning_rate: f64,
}

impl NetworkAdam {
    /// Create a new Adam optimizer.
    pub fn new() -> Self {
        Self {
            m_weights: vec![0.0; EMBEDDING_DIM],
            v_weights: vec![0.0; EMBEDDING_DIM],
            m_bias: 0.0,
            v_bias: 0.0,
            step: 0,
            beta1: ADAM_BETA1,
            beta2: ADAM_BETA2,
            epsilon: ADAM_EPSILON,
            learning_rate: ADAM_LEARNING_RATE,
        }
    }

    /// Apply the Adam update rule to compute new weights.
    ///
    /// Given the current weights and a gradient, returns the updated weights.
    pub fn update(
        &mut self,
        current_weights: &[f64],
        current_bias: f64,
        weight_gradient: &[f64],
        bias_gradient: f64,
    ) -> NervResult<(Vec<f64>, f64)> {
        if current_weights.len() != EMBEDDING_DIM {
            return Err(NervError::Economy(format!(
                "weight vector size mismatch: expected {}, got {}",
                EMBEDDING_DIM, current_weights.len()
            )));
        }
        if weight_gradient.len() != EMBEDDING_DIM {
            return Err(NervError::Economy(format!(
                "gradient vector size mismatch: expected {}, got {}",
                EMBEDDING_DIM, weight_gradient.len()
            )));
        }

        self.step += 1;
        let step = self.step;

        // Update weight moments and compute new weights
        let mut new_weights = vec![0.0; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            // First moment: m = β₁ * m + (1 - β₁) * g
            self.m_weights[i] = self.beta1 * self.m_weights[i]
                + (1.0 - self.beta1) * weight_gradient[i];

            // Second moment: v = β₂ * v + (1 - β₂) * g²
            self.v_weights[i] = self.beta2 * self.v_weights[i]
                + (1.0 - self.beta2) * weight_gradient[i] * weight_gradient[i];

            // Bias correction
            let m_hat = self.m_weights[i] / (1.0 - self.beta1.powi(step as i32));
            let v_hat = self.v_weights[i] / (1.0 - self.beta2.powi(step as i32));

            // Update: W = W - α * m_hat / (√v_hat + ε)
            new_weights[i] = current_weights[i]
                - self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }

        // Update bias moments
        self.m_bias = self.beta1 * self.m_bias + (1.0 - self.beta1) * bias_gradient;
        self.v_bias = self.beta2 * self.v_bias + (1.0 - self.beta2) * bias_gradient * bias_gradient;

        let m_hat_bias = self.m_bias / (1.0 - self.beta1.powi(step as i32));
        let v_hat_bias = self.v_bias / (1.0 - self.beta2.powi(step as i32));
        let new_bias = current_bias
            - self.learning_rate * m_hat_bias / (v_hat_bias.sqrt() + self.epsilon);

        Ok((new_weights, new_bias))
    }

    /// Compute the BLAKE3 hash of a weight vector.
    pub fn compute_weight_hash(weights: &[f64], bias: f64) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        for w in weights {
            hasher.update(&w.to_le_bytes());
        }
        hasher.update(&bias.to_le_bytes());
        hasher.finalize().into()
    }

    /// Get the step count.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Reset the optimizer state (e.g., after detecting divergence).
    pub fn reset(&mut self) {
        self.m_weights = vec![0.0; EMBEDDING_DIM];
        self.v_weights = vec![0.0; EMBEDDING_DIM];
        self.m_bias = 0.0;
        self.v_bias = 0.0;
        self.step = 0;
    }

    /// Check if the optimizer has diverged (moments exploding).
    pub fn is_diverged(&self, max_moment: f64) -> bool {
        if self.m_bias.abs() > max_moment || self.v_bias.abs() > max_moment {
            return true;
        }
        for i in 0..EMBEDDING_DIM.min(self.m_weights.len()) {
            if self.m_weights[i].abs() > max_moment || self.v_weights[i].abs() > max_moment {
                return true;
            }
        }
        false
    }
}

impl Default for NetworkAdam {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Helper Functions ────────────────────────────────────────────────────

/// Compute the L2 norm of a gradient (weight gradient + bias gradient).
pub fn compute_gradient_norm(weight_gradient: &[f64], bias_gradient: f64) -> f64 {
    let mut sum_sq = bias_gradient * bias_gradient;
    for &g in weight_gradient {
        sum_sq += g * g;
    }
    sum_sq.sqrt()
}

/// Compute the Huber loss.
#[inline]
pub fn huber_loss(prediction: f64, target: f64, delta: f64) -> f64 {
    let residual = target - prediction;
    let abs_r = residual.abs();
    if abs_r <= delta {
        0.5 * residual * residual
    } else {
        delta * abs_r - 0.5 * delta * delta
    }
}

/// Compute the Huber loss derivative w.r.t. the prediction.
#[inline]
pub fn huber_loss_derivative(prediction: f64, target: f64, delta: f64) -> f64 {
    let residual = target - prediction;
    if residual.abs() <= delta {
        -residual
    } else {
        -delta * residual.signum()
    }
}

/// Compute the gradient of Huber loss w.r.t. weights for a linear model.
///
/// For a linear model ŷ = W · x + b:
/// ∂L/∂W_i = ∂L/∂ŷ · x_i
/// ∂L/∂b = ∂L/∂ŷ
pub fn compute_linear_gradient(
    weights: &[f64],
    bias: f64,
    features: &[f64],
    target: f64,
    delta: f64,
) -> NervResult<(Vec<f64>, f64)> {
    if weights.len() != EMBEDDING_DIM {
        return Err(NervError::Economy(format!(
            "weight vector size mismatch: expected {}, got {}",
            EMBEDDING_DIM, weights.len()
        )));
    }
    if features.len() != EMBEDDING_DIM {
        return Err(NervError::Economy(format!(
            "feature vector size mismatch: expected {}, got {}",
            EMBEDDING_DIM, features.len()
        )));
    }

    // Forward pass: ŷ = W · x + b
    let mut prediction = bias;
    for i in 0..EMBEDDING_DIM {
        prediction += weights[i] * features[i];
    }

    // ∂L/∂ŷ
    let dloss_dpred = huber_loss_derivative(prediction, target, delta);

    // ∂L/∂W_i = ∂L/∂ŷ · x_i
    let mut weight_gradient = vec![0.0; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        weight_gradient[i] = dloss_dpred * features[i];
    }

    // ∂L/∂b = ∂L/∂ŷ
    let bias_gradient = dloss_dpred;

    Ok((weight_gradient, bias_gradient))
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_gradient(value: f64) -> Vec<f64> {
        vec![value; EMBEDDING_DIM]
    }

    #[test]
    fn test_gradient_submission_creation() {
        let sub = GradientSubmission::new(
            ValidatorId::from_bytes([1u8; 32]),
            BlockHeight::from(100),
            Epoch::from(0),
            make_test_gradient(0.01),
            0.005,
            1.0,
            0.8,
            [1u8; 32],
            [2u8; 32],
        );
        assert!(sub.is_improving());
        assert!((sub.loss_reduction() - 0.2).abs() < 1e-10);
        assert!(sub.gradient_norm > 0.0);
    }

    #[test]
    fn test_gradient_submission_validate_ok() {
        let sub = GradientSubmission::new(
            ValidatorId::from_bytes([1u8; 32]),
            BlockHeight::from(1),
            Epoch::from(0),
            make_test_gradient(0.01),
            0.005,
            1.0,
            0.8,
            [1u8; 32],
            [2u8; 32],
        );
        assert!(sub.validate(10.0).is_ok());
    }

    #[test]
    fn test_gradient_submission_validate_bad_dimension() {
        let mut sub = GradientSubmission::new(
            ValidatorId::from_bytes([1u8; 32]),
            BlockHeight::from(1),
            Epoch::from(0),
            make_test_gradient(0.01),
            0.005,
            1.0,
            0.8,
            [1u8; 32],
            [2u8; 32],
        );
        sub.weight_gradient = vec![0.01; 10]; // Wrong dimension
        assert!(sub.validate(10.0).is_err());
    }

    #[test]
    fn test_gradient_submission_validate_norm_exceeded() {
        let sub = GradientSubmission::new(
            ValidatorId::from_bytes([1u8; 32]),
            BlockHeight::from(1),
            Epoch::from(0),
            make_test_gradient(100.0), // Huge gradient
            50.0,
            1.0,
            0.8,
            [1u8; 32],
            [2u8; 32],
        );
        assert!(sub.validate(1.0).is_err()); // Norm exceeds max
    }

    #[test]
    fn test_gradient_submission_validate_zero_hash() {
        let mut sub = GradientSubmission::new(
            ValidatorId::from_bytes([1u8; 32]),
            BlockHeight::from(1),
            Epoch::from(0),
            make_test_gradient(0.01),
            0.005,
            1.0,
            0.8,
            [1u8; 32],
            [2u8; 32],
        );
        sub.weight_hash_before = [0u8; 32];
        assert!(sub.validate(10.0).is_err());
    }

    #[test]
    fn test_gradient_aggregation_basic() {
        let mut agg = GradientAggregation::new(10.0, 0.0);

        let sub1 = GradientSubmission::new(
            ValidatorId::from_bytes([1u8; 32]),
            BlockHeight::from(1), Epoch::from(0),
            make_test_gradient(0.01), 0.005, 1.0, 0.8,
            [1u8; 32], [3u8; 32],
        );
        let sub2 = GradientSubmission::new(
            ValidatorId::from_bytes([2u8; 32]),
            BlockHeight::from(1), Epoch::from(0),
            make_test_gradient(0.02), 0.010, 1.0, 0.7,
            [1u8; 32], [3u8; 32],
        );

        agg.add_submission(sub1).unwrap();
        agg.add_submission(sub2).unwrap();
        assert_eq!(agg.count(), 2);

        let (avg_w, avg_b) = agg.aggregate().unwrap();
        // Average of 0.01 and 0.02 = 0.015
        for &g in &avg_w {
            assert!((g - 0.015).abs() < 1e-10);
        }
        // Average of 0.005 and 0.010 = 0.0075
        assert!((avg_b - 0.0075).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_aggregation_consensus() {
        let mut agg = GradientAggregation::new(10.0, 0.0);

        // Three validators, two agree on hash A, one on hash B
        let sub1 = GradientSubmission::new(
            ValidatorId::from_bytes([1u8; 32]),
            BlockHeight::from(1), Epoch::from(0),
            make_test_gradient(0.01), 0.005, 1.0, 0.8,
            [1u8; 32], [3u8; 32], // Hash A
        );
        let sub2 = GradientSubmission::new(
            ValidatorId::from_bytes([2u8; 32]),
            BlockHeight::from(1), Epoch::from(0),
            make_test_gradient(0.01), 0.005, 1.0, 0.8,
            [1u8; 32], [3u8; 32], // Hash A
        );
        let sub3 = GradientSubmission::new(
            ValidatorId::from_bytes([3u8; 32]),
            BlockHeight::from(1), Epoch::from(0),
            make_test_gradient(0.01), 0.005, 1.0, 0.8,
            [1u8; 32], [4u8; 32], // Hash B
        );

        agg.add_submission(sub1).unwrap();
        agg.add_submission(sub2).unwrap();
        agg.add_submission(sub3).unwrap();

        let (consensus_hash, votes) = agg.consensus_weight_hash().unwrap();
        assert_eq!(consensus_hash, [3u8; 32]); // Hash A wins (2 vs 1)
        assert_eq!(votes, 2);
    }

    #[test]
    fn test_gradient_aggregation_average_losses() {
        let mut agg = GradientAggregation::new(10.0, 0.0);

        let sub1 = GradientSubmission::new(
            ValidatorId::from_bytes([1u8; 32]),
            BlockHeight::from(1), Epoch::from(0),
            make_test_gradient(0.01), 0.005, 1.0, 0.8,
            [1u8; 32], [2u8; 32],
        );
        let sub2 = GradientSubmission::new(
            ValidatorId::from_bytes([2u8; 32]),
            BlockHeight::from(1), Epoch::from(0),
            make_test_gradient(0.01), 0.005, 3.0, 2.0,
            [1u8; 32], [2u8; 32],
        );

        agg.add_submission(sub1).unwrap();
        agg.add_submission(sub2).unwrap();

        let (avg_before, avg_after) = agg.average_losses();
        assert!((avg_before - 2.0).abs() < 1e-10);
        assert!((avg_after - 1.4).abs() < 1e-10);
    }

    #[test]
    fn test_network_adam_update() {
        let mut adam = NetworkAdam::new();

        let weights = vec![0.1; EMBEDDING_DIM];
        let bias = 0.05;
        let gradient = vec![0.01; EMBEDDING_DIM];
        let bias_grad = 0.005;

        let (new_weights, new_bias) = adam.update(&weights, bias, &gradient, bias_grad).unwrap();

        // Weights should have changed
        assert_ne!(weights, new_weights);
        assert!((bias - new_bias).abs() > 1e-10);
        assert_eq!(adam.step(), 1);
    }

    #[test]
    fn test_network_adam_convergence() {
        let mut adam = NetworkAdam::new();
        let mut weights = vec![0.0; EMBEDDING_DIM];
        let mut bias = 0.0;

        // Simulate training: gradient always pushes towards target
        for _ in 0..100 {
            let gradient = vec![-0.1; EMBEDDING_DIM];
            let bias_grad = -0.05;
            let (new_w, new_b) = adam.update(&weights, bias, &gradient, bias_grad).unwrap();
            weights = new_w;
            bias = new_b;
        }

        // After 100 steps with consistent negative gradients,
        // weights should have increased
        assert!(weights[0] > 0.0);
        assert!(bias > 0.0);
    }

    #[test]
    fn test_network_adam_weight_hash() {
        let w = vec![1.0; EMBEDDING_DIM];
        let h1 = NetworkAdam::compute_weight_hash(&w, 0.5);
        let h2 = NetworkAdam::compute_weight_hash(&w, 0.5);
        assert_eq!(h1, h2);

        let w2 = vec![2.0; EMBEDDING_DIM];
        let h3 = NetworkAdam::compute_weight_hash(&w2, 0.5);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_network_adam_reset() {
        let mut adam = NetworkAdam::new();
        let _ = adam.update(&vec![0.0; EMBEDDING_DIM], 0.0, &vec![0.1; EMBEDDING_DIM], 0.05);
        assert_eq!(adam.step(), 1);
        adam.reset();
        assert_eq!(adam.step(), 0);
    }

    #[test]
    fn test_network_adam_diverged() {
        let mut adam = NetworkAdam::new();
        // Manually set large moments
        adam.m_bias = 1e10;
        assert!(adam.is_diverged(1e6));
    }

    #[test]
    fn test_compute_gradient_norm() {
        let g = vec![3.0, 4.0];
        let norm = compute_gradient_norm(&g, 0.0);
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_huber_loss() {
        // Quadratic region
        let l1 = huber_loss(0.5, 0.7, 1.0);
        assert!((l1 - 0.5 * 0.04).abs() < 1e-10);

        // Linear region
        let l2 = huber_loss(0.0, 5.0, 1.0);
        assert!((l2 - (1.0 * 5.0 - 0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_compute_linear_gradient() {
        let weights = vec![1.0; EMBEDDING_DIM];
        let bias = 0.0;
        let features = vec![0.5; EMBEDDING_DIM];
        let target = 0.0; // Want prediction to be 0

        let (w_grad, b_grad) = compute_linear_gradient(
            &weights, bias, &features, target, 1.0,
        ).unwrap();

        // Prediction = 1.0 * 0.5 * 64 + 0 = 32.0
        // target - prediction = 0 - 32 = -32
        // |residual| = 32 > delta=1.0, so in linear region
        // dL/dŷ = -delta * sign(-32) = -1.0 * (-1) = 1.0
        // dL/dW_i = 1.0 * 0.5 = 0.5
        // dL/db = 1.0
        for &g in &w_grad {
            assert!((g - 0.5).abs() < 1e-10);
        }
        assert!((b_grad - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weight_update_result_applied() {
        let result = WeightUpdateResult::Applied {
            new_weight_hash: [1u8; 32],
            loss_reduction: 0.1,
            contributor_count: 10,
        };
        assert!(result.is_applied());
    }

    #[test]
    fn test_weight_update_result_rejected() {
        let result = WeightUpdateResult::RejectedQuorum { approval_pct: 50.0 };
        assert!(!result.is_applied());
    }

    #[test]
    fn test_weight_update_proposal() {
        let proposal = WeightUpdateProposal {
            block_height: BlockHeight::from(100),
            epoch: Epoch::from(1),
            weight_hash_before: [1u8; 32],
            weight_hash_proposed: [2u8; 32],
            aggregated_weight_gradient: make_test_gradient(0.01),
            aggregated_bias_gradient: 0.005,
            contributor_count: 5,
            supporting_weight: 700,
            total_weight: 1000,
            avg_loss_before: 1.0,
            avg_loss_after: 0.8,
            quorum_reached: true,
            timestamp_ms: 0,
        };
        assert!(proposal.is_approved());
        assert!((proposal.avg_loss_reduction() - 0.2).abs() < 1e-10);
        assert!((proposal.approval_percentage() - 70.0).abs() < 1e-10);
    }
}
