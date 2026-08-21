//! Huber Loss — Robust loss function for the NWO Perceptron.
//!
//! The Huber loss is quadratic for small residuals and linear for large
//! residuals, making it robust to outliers (MEV spikes, exchange inflows,
//! anomalous transactions) while remaining differentiable everywhere.
//!
//! # Definition
//!
//! For a single residual `r = y - ŷ`:
//!
//! ```text
//! L_δ(r) = { 0.5 * r²        if |r| ≤ δ
//!           { δ * |r| - 0.5δ²  if |r| > δ
//! ```
//!
//! For the full embedding vector, we compute the element-wise loss
//! and sum over all 64 dimensions.
//!
//! # Gradient
//!
//! The gradient of the Huber loss with respect to the prediction ŷ:
//!
//! ```text
//! ∂L/∂ŷ = { -(y - ŷ)      if |y - ŷ| ≤ δ
//!          { -δ * sign(y - ŷ)  if |y - ŷ| > δ
//! ```
//!
//! # Why Huber Instead of MSE?
//!
//! Blockchain state dynamics contain outliers: massive exchange inflows,
//! MEV extraction bursts, and flash-loan cascades. Mean Squared Error
//! would cause the Perceptron weights to violently overreact to these
//! spikes, destabilizing the embedding. Huber loss limits the gradient
//! magnitude to ±δ, ensuring controlled weight updates.

use crate::EMBEDDING_DIM;
use crate::embedding::fixed_point::{FixedPoint64, EmbeddingVector};
use serde::{Deserialize, Serialize};

// ─── Huber Loss ──────────────────────────────────────────────────────────

/// Computes Huber loss between the target (actual) embedding and the
/// predicted (model) embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuberLoss {
    /// The Huber δ threshold — transitions from quadratic to linear.
    /// Default: 1.0 (from the V2.0 specification).
    pub delta: FixedPoint64,

    /// Precomputed: 0.5 * δ² (used in the linear regime).
    half_delta_sq: FixedPoint64,
}

impl HuberLoss {
    /// Create a Huber loss with the given δ threshold.
    pub fn new(delta: FixedPoint64) -> Self {
        let half_delta_sq = (delta * delta).div_int(2).unwrap_or(FixedPoint64::ZERO);
        Self {
            delta,
            half_delta_sq,
        }
    }

    /// Create with the default δ = 1.0.
    pub fn default_delta() -> Self {
        Self::new(FixedPoint64::ONE)
    }

    /// Create with δ from an f64 value.
    pub fn from_f64(delta: f64) -> Self {
        Self::new(FixedPoint64::from_f64(delta))
    }

    // ── Single-Element Loss ──────────────────────────────────────────

    /// Compute the Huber loss for a single residual `r = y - ŷ`.
    ///
    /// ```text
    /// L_δ(r) = { 0.5 * r²        if |r| ≤ δ
    ///           { δ * |r| - 0.5δ²  if |r| > δ
    /// ```
    #[inline]
    pub fn loss_single(&self, residual: FixedPoint64) -> FixedPoint64 {
        let abs_r = residual.abs();
        if abs_r <= self.delta {
            // Quadratic regime: 0.5 * r²
            let r_sq = residual * residual;
            r_sq.div_int(2).unwrap_or(FixedPoint64::ZERO)
        } else {
            // Linear regime: δ * |r| - 0.5 * δ²
            let linear = self.delta * abs_r;
            linear - self.half_delta_sq
        }
    }

    /// Compute the gradient of the loss with respect to the prediction ŷ.
    ///
    /// ```text
    /// ∂L/∂ŷ = { -(y - ŷ) = -r           if |r| ≤ δ
    ///          { -δ * sign(y - ŷ) = -δ*sign(r)  if |r| > δ
    /// ```
    #[inline]
    pub fn gradient_single(&self, residual: FixedPoint64) -> FixedPoint64 {
        let abs_r = residual.abs();
        if abs_r <= self.delta {
            // Quadratic regime: gradient = -r
            residual.neg()
        } else {
            // Linear regime: gradient = -δ * sign(r)
            if residual.is_positive() {
                self.delta.neg()
            } else if residual.is_negative() {
                self.delta
            } else {
                FixedPoint64::ZERO
            }
        }
    }

    // ── Full Embedding Loss ──────────────────────────────────────────

    /// Compute the total Huber loss between target and predicted embeddings.
    ///
    /// Returns the sum of element-wise losses over all 64 dimensions:
    ///
    /// ```text
    /// L_total = Σ_{i=0}^{63} L_δ(y[i] - ŷ[i])
    /// ```
    pub fn loss(&self, target: &EmbeddingVector, predicted: &EmbeddingVector) -> FixedPoint64 {
        let mut total = FixedPoint64::ZERO;
        for i in 0..EMBEDDING_DIM {
            let residual = target.get(i).unwrap().sub(predicted.get(i).unwrap());
            total += self.loss_single(residual);
        }
        total
    }

    /// Compute the per-element Huber loss vector.
    ///
    /// Returns a 64-dimensional vector where each element is the
    /// Huber loss for that dimension's residual.
    pub fn loss_per_element(
        &self,
        target: &EmbeddingVector,
        predicted: &EmbeddingVector,
    ) -> EmbeddingVector {
        let mut losses = [FixedPoint64::ZERO; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            let residual = target.get(i).unwrap().sub(predicted.get(i).unwrap());
            losses[i] = self.loss_single(residual);
        }
        EmbeddingVector::from_array(losses)
    }

    /// Compute the gradient vector ∂L/∂ŷ.
    ///
    /// This is the gradient of the total loss with respect to each
    /// element of the predicted embedding. It will be used to compute
    /// the gradients with respect to the Perceptron weights via the
    /// chain rule.
    pub fn gradient(&self, target: &EmbeddingVector, predicted: &EmbeddingVector) -> EmbeddingVector {
        let mut grads = [FixedPoint64::ZERO; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            let residual = target.get(i).unwrap().sub(predicted.get(i).unwrap());
            grads[i] = self.gradient_single(residual);
        }
        EmbeddingVector::from_array(grads)
    }

    /// Compute both loss and gradient in a single pass (more efficient).
    pub fn loss_and_gradient(
        &self,
        target: &EmbeddingVector,
        predicted: &EmbeddingVector,
    ) -> (FixedPoint64, EmbeddingVector) {
        let mut total_loss = FixedPoint64::ZERO;
        let mut grads = [FixedPoint64::ZERO; EMBEDDING_DIM];

        for i in 0..EMBEDDING_DIM {
            let residual = target.get(i).unwrap().sub(predicted.get(i).unwrap());
            total_loss += self.loss_single(residual);
            grads[i] = self.gradient_single(residual);
        }

        (total_loss, EmbeddingVector::from_array(grads))
    }

    /// Compute the residual vector: `r = target - predicted`.
    pub fn residual(&self, target: &EmbeddingVector, predicted: &EmbeddingVector) -> EmbeddingVector {
        target.sub(predicted)
    }

    /// Count how many dimensions are in the quadratic vs linear regime.
    pub fn regime_counts(
        &self,
        target: &EmbeddingVector,
        predicted: &EmbeddingVector,
    ) -> (usize, usize) {
        let mut quadratic = 0;
        let mut linear = 0;
        for i in 0..EMBEDDING_DIM {
            let residual = target.get(i).unwrap().sub(predicted.get(i).unwrap());
            if residual.abs() <= self.delta {
                quadratic += 1;
            } else {
                linear += 1;
            }
        }
        (quadratic, linear)
    }

    /// Get the δ threshold.
    #[inline]
    pub fn delta(&self) -> FixedPoint64 {
        self.delta
    }

    /// Get 0.5 * δ².
    #[inline]
    pub fn half_delta_sq(&self) -> FixedPoint64 {
        self.half_delta_sq
    }
}

impl std::default::Default for HuberLoss {
    fn default() -> Self {
        Self::default_delta()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huber_loss_quadratic_regime() {
        let hl = HuberLoss::default_delta(); // δ = 1.0
        // Residual r = 0.5 < δ → quadratic: 0.5 * 0.5² = 0.125
        let r = FixedPoint64::from_f64(0.5);
        let loss = hl.loss_single(r);
        assert!((loss.to_f64() - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_huber_loss_linear_regime() {
        let hl = HuberLoss::default_delta(); // δ = 1.0
        // Residual r = 2.0 > δ → linear: δ * |r| - 0.5*δ² = 1*2 - 0.5 = 1.5
        let r = FixedPoint64::from_int(2);
        let loss = hl.loss_single(r);
        assert!((loss.to_f64() - 1.5).abs() < 1e-4);
    }

    #[test]
    fn test_huber_loss_at_boundary() {
        let hl = HuberLoss::default_delta(); // δ = 1.0
        // At |r| = δ exactly: quadratic = 0.5 * 1² = 0.5, linear = 1*1 - 0.5 = 0.5
        let r = FixedPoint64::ONE;
        let loss = hl.loss_single(r);
        assert!((loss.to_f64() - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_huber_loss_zero_residual() {
        let hl = HuberLoss::default_delta();
        let loss = hl.loss_single(FixedPoint64::ZERO);
        assert!(loss.is_zero());
    }

    #[test]
    fn test_huber_gradient_quadratic() {
        let hl = HuberLoss::default_delta();
        // r = 0.5 → gradient = -0.5
        let r = FixedPoint64::from_f64(0.5);
        let grad = hl.gradient_single(r);
        assert!((grad.to_f64() + 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_huber_gradient_linear_positive() {
        let hl = HuberLoss::default_delta();
        // r = 2.0 > δ → gradient = -δ = -1.0
        let r = FixedPoint64::from_int(2);
        let grad = hl.gradient_single(r);
        assert!((grad.to_f64() + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_huber_gradient_linear_negative() {
        let hl = HuberLoss::default_delta();
        // r = -2.0 < -δ → gradient = +δ = +1.0
        let r = FixedPoint64::from_int(-2);
        let grad = hl.gradient_single(r);
        assert!((grad.to_f64() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_huber_full_loss() {
        let hl = HuberLoss::default_delta();
        let target = EmbeddingVector::splat(FixedPoint64::from_int(2));
        let predicted = EmbeddingVector::splat(FixedPoint64::from_int(1));
        // Residual = 1.0 = δ → loss per element = 0.5
        // Total = 64 * 0.5 = 32
        let loss = hl.loss(&target, &predicted);
        assert!((loss.to_f64() - 32.0).abs() < 0.1);
    }

    #[test]
    fn test_huber_loss_and_gradient_consistency() {
        let hl = HuberLoss::default_delta();
        let target = EmbeddingVector::splat(FixedPoint64::from_f64(1.5));
        let predicted = EmbeddingVector::splat(FixedPoint64::from_f64(0.5));

        let (loss, grad) = hl.loss_and_gradient(&target, &predicted);
        let loss_separate = hl.loss(&target, &predicted);
        let grad_separate = hl.gradient(&target, &predicted);

        // Should match
        assert!((loss.to_f64() - loss_separate.to_f64()).abs() < 1e-6);
        for i in 0..EMBEDDING_DIM {
            let g1 = grad.get(i).unwrap().to_f64();
           let g2 = grad_separate.get(i).unwrap().to_f64();
            assert!((g1 - g2).abs() < 1e-6);
        }
    }

    #[test]
    fn test_huber_regime_counts() {
        let hl = HuberLoss::default_delta();
        let target = EmbeddingVector::splat(FixedPoint64::from_int(2));
        let predicted = EmbeddingVector::ZERO;
        // |residual| = 2 > δ → all in linear regime
        let (quad, linear) = hl.regime_counts(&target, &predicted);
        assert_eq!(quad, 0);
        assert_eq!(linear, EMBEDDING_DIM);
    }
}
