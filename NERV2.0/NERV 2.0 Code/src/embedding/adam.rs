//! Adam Optimizer — Per-block continuous weight updates.
//!
//! The Adam (Adaptive Moment Estimation) optimizer adjusts the NWO
//! Perceptron's weights `W` and bias `b` every block based on the
//! gradient of the Huber loss.
//!
//! # Algorithm
//!
//! For each weight parameter θ:
//!
//! ```text
//! m_t = β₁ · m_{t-1} + (1 - β₁) · g_t           (first moment)
//! v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²           (second moment)
//! m̂_t = m_t / (1 - β₁^t)                          (bias-corrected first moment)
//! v̂_t = v_t / (1 - β₂^t)                          (bias-corrected second moment)
//! θ_{t+1} = θ_t - α · m̂_t / (√v̂_t + ε)           (parameter update)
//! ```
//!
//! # Parameters (from V2.0 spec)
//!
//! | Parameter | Value | Purpose |
//! |-----------|-------|---------|
//! | α         | 0.001 | Learning rate |
//! | β₁        | 0.9   | First moment decay |
//! | β₂        | 0.999 | Second moment decay |
//! | ε         | 10⁻⁸  | Numerical stability |
//!
//! # Gradient Clipping
//!
//! Gradients are clipped to `max_norm` before applying Adam to prevent
//! adversarial validators from submitting extreme gradient updates that
//! would destabilize the network's weights.

use crate::{EMBEDDING_DIM, NervError, NervResult};
use crate::embedding::fixed_point::FixedPoint64;
use crate::embedding::perceptron::NwoWeights;
use serde::{Deserialize, Serialize};

// ─── Adam Configuration ──────────────────────────────────────────────────

/// Configuration for the Adam optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamConfig {
    /// Learning rate α.
    pub learning_rate: FixedPoint64,

    /// First moment decay β₁ (default: 0.9).
    pub beta1: FixedPoint64,

    /// Second moment decay β₂ (default: 0.999).
    pub beta2: FixedPoint64,

    /// Numerical stability constant ε (default: 10⁻⁸).
    pub epsilon: FixedPoint64,

    /// Maximum gradient norm for clipping (prevents adversarial updates).
    pub max_gradient_norm: FixedPoint64,

    /// Minimum second moment before applying update (safety threshold).
    pub min_second_moment: FixedPoint64,
}

impl AdamConfig {
    /// Create with the V2.0 default parameters.
    pub fn default_v2() -> Self {
        Self {
            learning_rate: FixedPoint64::from_f64(0.001),
            beta1: FixedPoint64::from_f64(0.9),
            beta2: FixedPoint64::from_f64(0.999),
            epsilon: FixedPoint64::from_f64(1e-8),
            max_gradient_norm: FixedPoint64::ONE,
            min_second_moment: FixedPoint64::from_f64(1e-10),
        }
    }

    /// Create with custom learning rate.
    pub fn with_lr(lr: f64) -> Self {
        let mut cfg = Self::default_v2();
        cfg.learning_rate = FixedPoint64::from_f64(lr);
        cfg
    }

    /// Validate the configuration.
    pub fn validate(&self) -> NervResult<()> {
        if self.learning_rate.is_zero() || self.learning_rate.is_negative() {
            return Err(NervError::Other("learning rate must be positive".into()));
        }
        if self.beta1.is_negative() || self.beta1.to_f64() >= 1.0 {
            return Err(NervError::Other("beta1 must be in [0, 1)".into()));
        }
        if self.beta2.is_negative() || self.beta2.to_f64() >= 1.0 {
            return Err(NervError::Other("beta2 must be in [0, 1)".into()));
        }
        if self.epsilon.is_zero() || self.epsilon.is_negative() {
            return Err(NervError::Other("epsilon must be positive".into()));
        }
        if self.max_gradient_norm.is_zero() || self.max_gradient_norm.is_negative() {
            return Err(NervError::Other("max_gradient_norm must be positive".into()));
        }
        Ok(())
    }
}

impl std::default::Default for AdamConfig {
    fn default() -> Self {
        Self::default_v2()
    }
}

// ─── Adam Gradients ──────────────────────────────────────────────────────

/// Gradients of the loss with respect to the Perceptron parameters.
///
/// Computed via backpropagation through the linear model:
/// ```text
/// ∂L/∂W[i][j] = ∂L/∂ŷ[i] × x[j]       (weight gradients)
/// ∂L/∂b[i]    = ∂L/∂ŷ[i]               (bias gradients)
/// ```
///
/// Where `∂L/∂ŷ[i]` comes from the Huber loss gradient.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamGradients {
    /// Gradients for the weight matrix (same layout as NwoWeights::weights).
    pub weight_grads: Vec<FixedPoint64>,

    /// Gradients for the bias vector.
    pub bias_grads: [FixedPoint64; EMBEDDING_DIM],

    /// The original loss value (for logging and reward computation).
    pub loss: FixedPoint64,
}

impl AdamGradients {
    /// Compute the L2 norm of all gradients (for clipping and reward calculation).
    pub fn norm(&self) -> FixedPoint64 {
        let mut sum = FixedPoint64::ZERO;
        for g in &self.weight_grads {
            sum += (*g) * (*g);
        }
        for g in &self.bias_grads {
            sum += (*g) * (*g);
        }
        sum.sqrt().unwrap_or(FixedPoint64::ZERO)
    }

    /// Clip gradients in-place to the given maximum norm.
    ///
    /// If `||g|| > max_norm`, scales all gradients by `max_norm / ||g||`.
    pub fn clip(&mut self, max_norm: FixedPoint64) {
        let norm = self.norm();
        if norm > max_norm && !norm.is_zero() {
            if let Some(scale) = max_norm.div(norm) {
                for g in &mut self.weight_grads {
                    *g = (*g) * scale;
                }
                for g in &mut self.bias_grads {
                    *g = (*g) * scale;
                }
            }
        }
    }

    /// Zero gradients.
    pub fn zero(weight_count: usize) -> Self {
        Self {
            weight_grads: vec![FixedPoint64::ZERO; weight_count],
            bias_grads: [FixedPoint64::ZERO; EMBEDDING_DIM],
            loss: FixedPoint64::ZERO,
        }
    }

    /// Check if all gradients are zero.
    pub fn is_zero(&self) -> bool {
        self.weight_grads.iter().all(|g| g.is_zero())
            && self.bias_grads.iter().all(|g| g.is_zero())
    }

    /// Compute gradients from Huber loss gradient and input features.
    ///
    /// Given:
    /// - `dl_dy`: ∂L/∂ŷ ∈ ℝ⁶⁴ (from Huber loss)
    /// - `input`: x ∈ ℝ^N (the transaction feature vector)
    ///
    /// The weight gradients are:
    /// ∂L/∂W[i][j] = ∂L/∂ŷ[i] × x[j]
    ///
    /// The bias gradients are:
    /// ∂L/∂b[i] = ∂L/∂ŷ[i]
    pub fn from_loss_gradient_and_input(
        dl_dy: &[FixedPoint64; EMBEDDING_DIM],
        input: &[FixedPoint64],
        loss: FixedPoint64,
    ) -> Self {
        let input_dim = input.len();
        let mut weight_grads = vec![FixedPoint64::ZERO; EMBEDDING_DIM * input_dim];

        for i in 0..EMBEDDING_DIM {
            for j in 0..input_dim {
                weight_grads[i * input_dim + j] = dl_dy[i] * input[j];
            }
        }

        Self {
            weight_grads,
            bias_grads: *dl_dy,
            loss,
        }
    }
}

// ─── Adam State ──────────────────────────────────────────────────────────

/// State of the Adam optimizer: first and second moment estimates
/// for each weight and bias parameter.
///
/// This state is persisted in RocksDB and evolves across blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamState {
    /// First moment estimates m for weights (same layout as NwoWeights::weights).
    pub weight_m: Vec<FixedPoint64>,

    /// Second moment estimates v for weights.
    pub weight_v: Vec<FixedPoint64>,

    /// First moment estimates m for bias.
    pub bias_m: [FixedPoint64; EMBEDDING_DIM],

    /// Second moment estimates v for bias.
    pub bias_v: [FixedPoint64; EMBEDDING_DIM],

    /// Current step count (incremented on each update).
    pub step: u64,

    /// The Adam configuration.
    pub config: AdamConfig,
}

impl AdamState {
    /// Create a new Adam state initialized to zero.
    pub fn new(weight_count: usize, config: AdamConfig) -> Self {
        Self {
            weight_m: vec![FixedPoint64::ZERO; weight_count],
            weight_v: vec![FixedPoint64::ZERO; weight_count],
            bias_m: [FixedPoint64::ZERO; EMBEDDING_DIM],
            bias_v: [FixedPoint64::ZERO; EMBEDDING_DIM],
            step: 0,
            config,
        }
    }

    /// Create with default V2.0 configuration.
    pub fn default_v2(weight_count: usize) -> Self {
        Self::new(weight_count, AdamConfig::default_v2())
    }

    /// Get the current step.
    #[inline]
    pub fn step(&self) -> u64 {
        self.step
    }

    // ── Single Parameter Update ──────────────────────────────────────

    /// Update a single parameter using Adam.
    ///
    /// Given gradient `g`, current moments `m` and `v`, and config:
    ///
    /// ```text
    /// m_new = β₁ * m + (1 - β₁) * g
    /// v_new = β₂ * v + (1 - β₂) * g²
    /// m̂ = m_new / (1 - β₁^t)
    /// v̂ = v_new / (1 - β₂^t)
    /// θ_new = θ - α * m̂ / (√v̂ + ε)
    /// ```
    ///
    /// Returns `(m_new, v_new, θ_new)`.
    #[inline]
    fn update_param(
        &self,
        theta: FixedPoint64,
        m: FixedPoint64,
        v: FixedPoint64,
        gradient: FixedPoint64,
    ) -> (FixedPoint64, FixedPoint64, FixedPoint64) {
        let one = FixedPoint64::ONE;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let lr = self.config.learning_rate;
        let eps = self.config.epsilon;

        // Update biased first moment: m = β₁ * m + (1 - β₁) * g
        let m_new = beta1 * m + (one - beta1) * gradient;

        // Update biased second moment: v = β₂ * v + (1 - β₂) * g²
        let g_sq = gradient * gradient;
        let v_new = beta2 * v + (one - beta2) * g_sq;

        // Bias correction
        // β₁^t and β₂^t — computed as f64 for numerical stability
        let t = self.step as f64;
        let beta1_f64 = self.config.beta1.to_f64();
        let beta2_f64 = self.config.beta2.to_f64();
        let beta1_pow_t = beta1_f64.powf(t);
        let beta2_pow_t = beta2_f64.powf(t);

        let one_minus_beta1_pow_t = 1.0 - beta1_pow_t;
        let one_minus_beta2_pow_t = 1.0 - beta2_pow_t;

        // Avoid division by zero for very early steps
        let bias_corr1 = if one_minus_beta1_pow_t > 1e-10 {
            FixedPoint64::from_f64(one_minus_beta1_pow_t)
        } else {
            FixedPoint64::from_f64(1e-10)
        };
        let bias_corr2 = if one_minus_beta2_pow_t > 1e-10 {
            FixedPoint64::from_f64(one_minus_beta2_pow_t)
        } else {
            FixedPoint64::from_f64(1e-10)
        };

        // Bias-corrected moments
        let m_hat = m_new.div(bias_corr1).unwrap_or(m_new);
        let v_hat = v_new.div(bias_corr2).unwrap_or(v_new);

        // Parameter update: θ_new = θ - α * m̂ / (√v̂ + ε)
        let v_hat_sqrt = v_hat.sqrt().unwrap_or(FixedPoint64::ZERO);
        let denominator = v_hat_sqrt + eps;
        let update = if let Some(ratio) = m_hat.div(denominator) {
            lr * ratio
        } else {
            FixedPoint64::ZERO
        };

        let theta_new = theta - update;

        (m_new, v_new, theta_new)
    }

    // ── Full Weight Update ───────────────────────────────────────────

    /// Apply an Adam update step to the NWO weights.
    ///
    /// Given the computed gradients, this method:
    /// 1. Clips gradients to `max_gradient_norm`
    /// 2. Updates first and second moment estimates
    /// 3. Applies bias correction
    /// 4. Updates weights and biases
    /// 5. Increments the step counter
    ///
    /// Returns the updated weights and the loss value.
    pub fn step(
        &mut self,
        weights: &NwoWeights,
        mut gradients: AdamGradients,
    ) -> NervResult<NwoWeights> {
        // 1. Clip gradients
        gradients.clip(self.config.max_gradient_norm);

        // 2. Validate dimensions
        if gradients.weight_grads.len() != weights.weight_count() {
            return Err(NervError::Other(format!(
                "gradient dimension mismatch: expected {}, got {}",
                weights.weight_count(),
                gradients.weight_grads.len()
            )));
        }

        // 3. Increment step
        self.step = self.step.saturating_add(1);

        // 4. Update each weight
        let mut new_weights = weights.clone();
        for (idx, &grad) in gradients.weight_grads.iter().enumerate() {
            let old_w = new_weights.weights_flat()[idx];
            let old_m = self.weight_m[idx];
            let old_v = self.weight_v[idx];

            let (m_new, v_new, w_new) = self.update_param(old_w, old_m, old_v, grad);

            new_weights.weights_flat_mut()[idx] = w_new;
            self.weight_m[idx] = m_new;
            self.weight_v[idx] = v_new;
        }

        // 5. Update each bias
        for i in 0..EMBEDDING_DIM {
            let old_b = weights.bias_at(i).unwrap();
            let old_m = self.bias_m[i];
            let old_v = self.bias_v[i];
            let grad = gradients.bias_grads[i];

            let (m_new, v_new, b_new) = self.update_param(old_b, old_m, old_v, grad);

            new_weights.bias_mut()[i] = b_new;
            self.bias_m[i] = m_new;
            self.bias_v[i] = v_new;
        }

        // 6. Recompute weight hash
        // (NwoWeights doesn't expose a public method for this,
        //  but it will be recomputed on next compute_hash() call)

        Ok(new_weights)
    }

    /// Serialize the Adam state to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Step count
        buf.extend_from_slice(&self.step.to_le_bytes());
        // Weight moments
        for m in &self.weight_m {
            buf.extend_from_slice(&m.to_le_bytes());
        }
        for v in &self.weight_v {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // Bias moments
        for m in &self.bias_m {
            buf.extend_from_slice(&m.to_le_bytes());
        }
        for v in &self.bias_v {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    /// Deserialize Adam state from bytes.
    pub fn from_bytes(data: &[u8], weight_count: usize, config: AdamConfig) -> NervResult<Self> {
        let expected = 8 + weight_count * 8 * 2 + EMBEDDING_DIM * 8 * 2;
        if data.len() < expected {
            return Err(NervError::Serialization(
                format!("Adam state data too short: expected {expected}, got {}", data.len())
            ));
        }

        let step = u64::from_le_bytes(data[..8].try_into().unwrap());
        let mut offset = 8;

        let mut weight_m = Vec::with_capacity(weight_count);
        for _ in 0..weight_count {
            let mut le = [0u8; 8];
            le.copy_from_slice(&data[offset..offset + 8]);
            weight_m.push(FixedPoint64::from_le_bytes(le));
            offset += 8;
        }

        let mut weight_v = Vec::with_capacity(weight_count);
        for _ in 0..weight_count {
            let mut le = [0u8; 8];
            le.copy_from_slice(&data[offset..offset + 8]);
            weight_v.push(FixedPoint64::from_le_bytes(le));
            offset += 8;
        }

        let mut bias_m = [FixedPoint64::ZERO; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            let mut le = [0u8; 8];
            le.copy_from_slice(&data[offset..offset + 8]);
            bias_m[i] = FixedPoint64::from_le_bytes(le);
            offset += 8;
        }

        let mut bias_v = [FixedPoint64::ZERO; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            let mut le = [0u8; 8];
            le.copy_from_slice(&data[offset..offset + 8]);
            bias_v[i] = FixedPoint64::from_le_bytes(le);
            offset += 8;
        }

        Ok(Self {
            weight_m,
            weight_v,
            bias_m,
            bias_v,
            step,
            config,
        })
    }

    /// Reset the optimizer state (e.g., after loading new weights from consensus).
    pub fn reset(&mut self) {
        for m in &mut self.weight_m {
            *m = FixedPoint64::ZERO;
        }
        for v in &mut self.weight_v {
            *v = FixedPoint64::ZERO;
        }
        for m in &mut self.bias_m {
            *m = FixedPoint64::ZERO;
        }
        for v in &mut self.bias_v {
            *v = FixedPoint64::ZERO;
        }
        self.step = 0;
    }

    /// Compute the effective learning rate for each parameter at the current step.
    ///
    /// Returns the maximum effective LR across all parameters (for diagnostics).
    pub fn max_effective_lr(&self) -> FixedPoint64 {
        if self.step == 0 {
            return self.config.learning_rate;
        }
        let t = self.step as f64;
        let beta1_f64 = self.config.beta1.to_f64();
        let beta2_f64 = self.config.beta2.to_f64();
        let beta1_pow_t = beta1_f64.powf(t);
        let beta2_pow_t = beta2_f64.powf(t);

        let bias_corr1 = 1.0 - beta1_pow_t;
        let bias_corr2 = 1.0 - beta2_pow_t;

        if bias_corr1 < 1e-10 || bias_corr2 < 1e-10 {
            return self.config.learning_rate;
        }

        // Effective LR = α * √(1 - β₂^t) / (1 - β₁^t)
        let eff_lr = self.config.learning_rate.to_f64()
            * bias_corr2.sqrt()
            / bias_corr1;

        FixedPoint64::from_f64(eff_lr)
    }

    /// Get the total number of parameters being optimized.
    pub fn param_count(&self) -> usize {
        self.weight_m.len() + EMBEDDING_DIM
    }

    /// Get a summary of the optimizer state for diagnostics.
    pub fn summary(&self) -> AdamSummary {
        let weight_m_norm = {
            let mut sum = FixedPoint64::ZERO;
            for m in &self.weight_m {
                sum += (*m) * (*m);
            }
            sum.sqrt().unwrap_or(FixedPoint64::ZERO)
        };

        let weight_v_norm = {
            let mut sum = FixedPoint64::ZERO;
            for v in &self.weight_v {
                sum += (*v) * (*v);
            }
            sum.sqrt().unwrap_or(FixedPoint64::ZERO)
        };

        AdamSummary {
            step: self.step,
            weight_m_norm,
            weight_v_norm,
            effective_lr: self.max_effective_lr(),
            learning_rate: self.config.learning_rate,
            beta1: self.config.beta1,
            beta2: self.config.beta2,
        }
    }
}

/// Diagnostic summary of the Adam optimizer state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamSummary {
    /// Current step count.
    pub step: u64,
    /// L2 norm of weight first moments.
    pub weight_m_norm: FixedPoint64,
    /// L2 norm of weight second moments.
    pub weight_v_norm: FixedPoint64,
    /// Maximum effective learning rate.
    pub effective_lr: FixedPoint64,
    /// Configured learning rate.
    pub learning_rate: FixedPoint64,
    /// Configured β₁.
    pub beta1: FixedPoint64,
    /// Configured β₂.
    pub beta2: FixedPoint64,
}

impl std::fmt::Display for AdamSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Adam(step={}, lr={:.6}, eff_lr={:.6}, β1={:.4}, β2={:.6}, |m|={:.6}, |v|={:.6})",
            self.step,
            self.learning_rate.to_f64(),
            self.effective_lr.to_f64(),
            self.beta1.to_f64(),
            self.beta2.to_f64(),
            self.weight_m_norm.to_f64(),
            self.weight_v_norm.to_f64(),
        )
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::DEFAULT_INPUT_DIM;

    fn make_test_weights() -> NwoWeights {
        NwoWeights::new_constant(DEFAULT_INPUT_DIM, 0.01)
    }

    fn make_test_state() -> AdamState {
        let weight_count = EMBEDDING_DIM * DEFAULT_INPUT_DIM;
        AdamState::default_v2(weight_count)
    }

    #[test]
    fn test_adam_config_default() {
        let cfg = AdamConfig::default_v2();
        assert!((cfg.learning_rate.to_f64() - 0.001).abs() < 1e-8);
        assert!((cfg.beta1.to_f64() - 0.9).abs() < 1e-8);
        assert!((cfg.beta2.to_f64() - 0.999).abs() < 1e-8);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_adam_config_validate() {
        let mut cfg = AdamConfig::default_v2();
        assert!(cfg.validate().is_ok());

        cfg.learning_rate = FixedPoint64::ZERO;
        assert!(cfg.validate().is_err());

        cfg.learning_rate = FixedPoint64::from_f64(-0.001);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_adam_gradients_zero() {
        let grads = AdamGradients::zero(100);
        assert!(grads.is_zero());
        assert!(grads.norm().is_zero());
    }

    #[test]
    fn test_adam_gradients_norm() {
        let mut grads = AdamGradients::zero(2);
        grads.weight_grads[0] = FixedPoint64::from_int(3);
        grads.weight_grads[1] = FixedPoint64::from_int(4);
        // norm = sqrt(3² + 4²) = 5
        let norm = grads.norm();
        assert!((norm.to_f64() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_adam_gradients_clip() {
        let mut grads = AdamGradients::zero(2);
        grads.weight_grads[0] = FixedPoint64::from_int(10);
        grads.weight_grads[1] = FixedPoint64::from_int(10);
        // norm = sqrt(200) ≈ 14.14, clip to 1.0
        grads.clip(FixedPoint64::ONE);
        let norm_after = grads.norm();
        assert!((norm_after.to_f64() - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_adam_gradients_clip_below_threshold() {
        let mut grads = AdamGradients::zero(2);
        grads.weight_grads[0] = FixedPoint64::from_f64(0.1);
        grads.weight_grads[1] = FixedPoint64::from_f64(0.1);
        // norm ≈ 0.141 < 1.0 → no clipping
        let norm_before = grads.norm();
        grads.clip(FixedPoint64::ONE);
        let norm_after = grads.norm();
        assert!((norm_before.to_f64() - norm_after.to_f64()).abs() < 1e-6);
    }

    #[test]
    fn test_adam_gradients_from_loss_gradient() {
        let dl_dy = [FixedPoint64::from_f64(0.5); EMBEDDING_DIM];
        let input = vec![FixedPoint64::from_f64(0.3); DEFAULT_INPUT_DIM];
        let loss = FixedPoint64::from_f64(1.0);

        let grads = AdamGradients::from_loss_gradient_and_input(
            &dl_dy, &input, loss,
        );

        // Weight gradient for W[0][0] = 0.5 * 0.3 = 0.15
        let wg00 = grads.weight_grads[0];
        assert!((wg00.to_f64() - 0.15).abs() < 1e-4);

        // Bias gradient for b[0] = 0.5
        let bg0 = grads.bias_grads[0];
        assert!((bg0.to_f64() - 0.5).abs() < 1e-6);

        // Loss should be preserved
        assert!((grads.loss.to_f64() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_adam_state_new() {
        let state = make_test_state();
        assert_eq!(state.step(), 0);
        assert_eq!(state.param_count(), EMBEDDING_DIM * DEFAULT_INPUT_DIM + EMBEDDING_DIM);
    }

    #[test]
    fn test_adam_state_step_single_update() {
        let weights = make_test_weights();
        let mut state = make_test_state();

        // Create a small gradient
        let mut grads = AdamGradients::zero(weights.weight_count());
        for g in grads.weight_grads.iter_mut() {
            *g = FixedPoint64::from_f64(0.01);
        }
        for g in grads.bias_grads.iter_mut() {
            *g = FixedPoint64::from_f64(0.01);
        }
        grads.loss = FixedPoint64::ONE;

        let new_weights = state.step(&weights, grads).unwrap();

        // Step should have incremented
        assert_eq!(state.step(), 1);

        // Weights should have changed
        let old_w = weights.weight(0, 0).unwrap();
        let new_w = new_weights.weight(0, 0).unwrap();
        assert_ne!(old_w, new_w);

        // The weight should have decreased (gradient is positive → weight decreases)
        assert!(new_w.to_f64() < old_w.to_f64());
    }

    #[test]
    fn test_adam_state_multiple_steps() {
        let weights = make_test_weights();
        let mut state = make_test_state();

        for step in 0..5 {
            let mut grads = AdamGradients::zero(weights.weight_count());
            for g in grads.weight_grads.iter_mut() {
                *g = FixedPoint64::from_f64(0.01);
            }
            grads.loss = FixedPoint64::from_f64(1.0 / (step as f64 + 1.0));

            let _new_weights = state.step(&weights, grads.clone()).unwrap();
            assert_eq!(state.step(), (step + 1) as u64);
        }
    }

    #[test]
    fn test_adam_state_zero_gradient_no_change() {
        let weights = make_test_weights();
        let mut state = make_test_state();

        let grads = AdamGradients::zero(weights.weight_count());
        let new_weights = state.step(&weights, grads).unwrap();

        // With zero gradients, weights should not change
        // (moments will still be zero, so update is zero)
        for i in 0..weights.weight_count() {
            let old_w = weights.weights_flat()[i];
            let new_w = new_weights.weights_flat()[i];
            // Due to floating-point representation, check approximate equality
            assert!((old_w.to_f64() - new_w.to_f64()).abs() < 1e-12);
        }
    }

    #[test]
    fn test_adam_state_reset() {
        let mut state = make_test_state();
        state.step = 42;
        state.weight_m[0] = FixedPoint64::ONE;
        state.weight_v[0] = FixedPoint64::ONE;

        state.reset();

        assert_eq!(state.step, 0);
        assert!(state.weight_m[0].is_zero());
        assert!(state.weight_v[0].is_zero());
    }

    #[test]
    fn test_adam_state_serialization_roundtrip() {
        let state = make_test_state();
        let bytes = state.to_bytes();
        let weight_count = EMBEDDING_DIM * DEFAULT_INPUT_DIM;
        let recovered = AdamState::from_bytes(
            &bytes, weight_count, AdamConfig::default_v2(),
        ).unwrap();

        assert_eq!(state.step, recovered.step);
        assert_eq!(state.weight_m.len(), recovered.weight_m.len());
        assert_eq!(state.weight_v.len(), recovered.weight_v.len());
        for i in 0..state.weight_m.len() {
            assert_eq!(state.weight_m[i], recovered.weight_m[i]);
            assert_eq!(state.weight_v[i], recovered.weight_v[i]);
        }
    }

    #[test]
    fn test_adam_effective_lr() {
        let state = make_test_state();
        // At step 0, effective LR should be approximately the base LR
        let eff_lr = state.max_effective_lr();
        assert!((eff_lr.to_f64() - 0.001).abs() < 0.001);
    }

    #[test]
    fn test_adam_summary() {
        let state = make_test_state();
        let summary = state.summary();
        assert_eq!(summary.step, 0);
        let display = format!("{summary}");
        assert!(display.contains("Adam"));
        assert!(display.contains("step=0"));
    }

    #[test]
    fn test_adam_with_large_gradient_clipping() {
        let weights = make_test_weights();
        let mut state = make_test_state();

        let mut grads = AdamGradients::zero(weights.weight_count());
        // Set an adversarially large gradient
        for g in grads.weight_grads.iter_mut() {
            *g = FixedPoint64::from_int(1000);
        }
        grads.loss = FixedPoint64::from_int(1000000);

        // This should be clipped to max_gradient_norm = 1.0
        let new_weights = state.step(&weights, grads).unwrap();

        // The weight change should be modest despite the huge gradient
        let old_w = weights.weight(0, 0).unwrap().to_f64();
        let new_w = new_weights.weight(0, 0).unwrap().to_f64();
        let change = (old_w - new_w).abs();
        // Change should be much less than if we used the unclipped gradient
        assert!(change < 1.0, "weight change {change} too large after clipping");
    }

    #[test]
    fn test_adam_bias_correction_warms_up() {
        // At early steps, bias correction increases the effective learning rate.
        // At step 1 with β₁=0.9: correction = 1/(1-0.9) = 10×
        // At step 100: correction ≈ 1.0 (converged)

        let weights = make_test_weights();
        let mut state_early = make_test_state();
        let mut state_late = make_test_state();
        state_late.step = 100;

        let mut grads = AdamGradients::zero(weights.weight_count());
        grads.weight_grads[0] = FixedPoint64::from_f64(0.1);
        grads.bias_grads[0] = FixedPoint64::from_f64(0.1);
        grads.loss = FixedPoint64::ONE;

        let new_w_early = state_early.step(&weights, grads.clone()).unwrap();
        let new_w_late = state_late.step(&weights, grads).unwrap();

        // Early step should produce a larger weight change (bias correction)
        let change_early = (weights.weight(0, 0).unwrap().to_f64()
            - new_w_early.weight(0, 0).unwrap().to_f64()).abs();
        let change_late = (weights.weight(0, 0).unwrap().to_f64()
            - new_w_late.weight(0, 0).unwrap().to_f64()).abs();

        // Early change should be larger (bias correction amplifies)
        assert!(
            change_early > change_late * 2.0,
            "early change {change_early} should be significantly larger than late change {change_late} due to bias correction"
        );
    }
}
