//! NWO Perceptron for Shard Load Prediction.
//!
//! Replaces the 1.1 MB LSTM from V1.01 with a lightweight single-layer
//! Perceptron that predicts shard overload probability. The NWO Perceptron
//! is continuously updated via the Adam optimizer and Huber Loss, enabling
//! the sharding model to improve its predictions on every block.
//!
//! # Architecture
//!
//! ```text
//! Input Features (6-dim):          NWO Perceptron:          Output:
//! ┌──────────────────┐            ┌──────────┐
//! │ current_tps      │──┐        │ W (6×1)  │
//! │ mempool_util     │──┤        │ b (1)    │
//! │ inter_shard_q    │──┼──→  x·W + b  ──→  sigmoid  ──→  P(overload)
//! │ gas_util         │──┤        └──────────┘
//! │ block_exec_time  │──┤
//! │ cross_shard_ratio│──┘
//! └──────────────────┘
//! ```
//!
//! # Training
//!
//! Every block, the predictor:
//! 1. Computes predicted overload probability from current features
//! 2. Observes actual overload (1.0 if TPS exceeded capacity, 0.0 otherwise)
//! 3. Computes Huber Loss between prediction and observation
//! 4. Backpropagates through the single linear layer
//! 5. Updates weights via Adam optimizer
//!
//! Because the model is linear, backpropagation is exact and trivial:
//! ∂L/∂W = ∂L/∂ŷ · x  and  ∂L/∂b = ∂L/∂ŷ

use crate::{
    EMBEDDING_DIM, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, ADAM_LEARNING_RATE, HUBER_DELTA,
    NervError, NervResult,
};
use crate::config::ShardingConfig;
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ─── Constants ───────────────────────────────────────────────────────────

/// Number of input features for the sharding NWO predictor.
pub const PREDICTOR_FEATURES: usize = 6;

/// Sigmoid steepness for overload probability output.
const SIGMOID_STEEPNESS: f64 = 10.0;

/// Feature normalization constants (approximate maximums for scaling to [0,1]).
const MAX_TPS: f64 = 2_000_000.0;         // 2M TPS peak burst
const MAX_MEMPOOL: f64 = 500_000.0;        // 500K encrypted txs
const_MAX_INTER_SHARD_Q: f64 = 100_000.0;  // 100K queued messages
const MAX_BLOCK_EXEC_MS: f64 = 1000.0;     // 1 second target

// ─── Load Features ───────────────────────────────────────────────────────

/// Normalized load features for the NWO predictor.
///
/// All values are in [0.0, 1.0] range (normalized from raw metrics).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadFeatures {
    /// Current TPS normalized to maximum capacity.
    pub current_tps: f64,

    /// Mempool utilization fraction.
    pub mempool_utilization: f64,

    /// Inter-shard message queue depth normalized.
    pub inter_shard_queue_depth: f64,

    /// Gas utilization fraction.
    pub gas_utilization: f64,

    /// Average block execution time normalized.
    pub block_execution_time: f64,

    /// Cross-shard transaction ratio.
    pub cross_shard_ratio: f64,
}

impl LoadFeatures {
    /// All-zero features (idle shard).
    pub const ZERO: Self = Self {
        current_tps: 0.0,
        mempool_utilization: 0.0,
        inter_shard_queue_depth: 0.0,
        gas_utilization: 0.0,
        block_execution_time: 0.0,
        cross_shard_ratio: 0.0,
    };

    /// All-max features (fully loaded shard).
    pub const MAX: Self = Self {
        current_tps: 1.0,
        mempool_utilization: 1.0,
        inter_shard_queue_depth: 1.0,
        gas_utilization: 1.0,
        block_execution_time: 1.0,
        cross_shard_ratio: 1.0,
    };

    /// Convert to a fixed-size array for matrix operations.
    #[inline]
    pub fn to_array(&self) -> [f64; PREDICTOR_FEATURES] {
        [
            self.current_tps,
            self.mempool_utilization,
            self.inter_shard_queue_depth,
            self.gas_utilization,
            self.block_execution_time,
            self.cross_shard_ratio,
        ]
    }

    /// Convert from a fixed-size array.
    #[inline]
    pub fn from_array(arr: &[f64; PREDICTOR_FEATURES]) -> Self {
        Self {
            current_tps: arr[0],
            mempool_utilization: arr[1],
            inter_shard_queue_depth: arr[2],
            gas_utilization: arr[3],
            block_execution_time: arr[4],
            cross_shard_ratio: arr[5],
        }
    }

    /// Clamp all features to [0.0, 1.0].
    pub fn clamp(&self) -> Self {
        Self {
            current_tps: self.current_tps.clamp(0.0, 1.0),
            mempool_utilization: self.mempool_utilization.clamp(0.0, 1.0),
            inter_shard_queue_depth: self.inter_shard_queue_depth.clamp(0.0, 1.0),
            gas_utilization: self.gas_utilization.clamp(0.0, 1.0),
            block_execution_time: self.block_execution_time.clamp(0.0, 1.0),
            cross_shard_ratio: self.cross_shard_ratio.clamp(0.0, 1.0),
        }
    }

    /// L2 norm of the feature vector.
    pub fn norm(&self) -> f64 {
        let arr = self.to_array();
        arr.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

// ─── Raw Load Metrics ────────────────────────────────────────────────────

/// Raw (un-normalized) load metrics collected from a shard.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoadMetrics {
    /// Current transactions per second.
    pub current_tps: f64,

    /// Mempool size (number of encrypted transactions).
    pub mempool_size: usize,

    /// Inter-shard message queue depth.
    pub inter_shard_queue_depth: usize,

    /// Gas utilization fraction (already 0.0–1.0).
    pub gas_utilization: f64,

    /// Average block execution time in milliseconds.
    pub avg_block_exec_ms: f64,

    /// Cross-shard transaction ratio (0.0–1.0).
    pub cross_shard_ratio: f64,

    /// Maximum TPS capacity for this shard.
    pub max_tps_capacity: f64,
}

impl LoadMetrics {
    /// Normalize raw metrics into [0.0, 1.0] features for the predictor.
    pub fn normalize(&self) -> LoadFeatures {
        let max_tps = if self.max_tps_capacity > 0.0 {
            self.max_tps_capacity
        } else {
            MAX_TPS
        };

        LoadFeatures {
            current_tps: (self.current_tps / max_tps).clamp(0.0, 1.0),
            mempool_utilization: (self.mempool_size as f64 / MAX_MEMPOOL).clamp(0.0, 1.0),
            inter_shard_queue_depth: (self.inter_shard_queue_depth as f64 / MAX_INTER_SHARD_Q)
                .clamp(0.0, 1.0),
            gas_utilization: self.gas_utilization.clamp(0.0, 1.0),
            block_execution_time: (self.avg_block_exec_ms / MAX_BLOCK_EXEC_MS).clamp(0.0, 1.0),
            cross_shard_ratio: self.cross_shard_ratio.clamp(0.0, 1.0),
        }
    }

    /// Determine if the shard was actually overloaded (ground truth for training).
    ///
    /// A shard is overloaded if:
    /// - TPS exceeds 90% of capacity, OR
    /// - Mempool is >80% full, OR
    /// - Gas utilization >95%
    pub fn is_overloaded(&self) -> bool {
        let max_tps = if self.max_tps_capacity > 0.0 {
            self.max_tps_capacity
        } else {
            MAX_TPS
        };

        self.current_tps > 0.90 * max_tps
            || self.mempool_size as f64 > 0.80 * MAX_MEMPOOL
            || self.gas_utilization > 0.95
    }

    /// Ground truth overload value for training (soft label).
    ///
    /// Returns a value in [0.0, 1.0] indicating overload severity:
    /// - 1.0 = severely overloaded
    /// - 0.5 = moderately loaded
    /// - 0.0 = idle
    pub fn overload_label(&self) -> f64 {
        let max_tps = if self.max_tps_capacity > 0.0 {
            self.max_tps_capacity
        } else {
            MAX_TPS
        };

        let tps_load = (self.current_tps / max_tps).clamp(0.0, 1.0);
        let mempool_load = (self.mempool_size as f64 / MAX_MEMPOOL).clamp(0.0, 1.0);
        let gas_load = self.gas_utilization.clamp(0.0, 1.0);

        // Weighted combination: TPS is the primary signal
        let combined = 0.5 * tps_load + 0.25 * mempool_load + 0.25 * gas_load;

        // Apply sigmoid-like mapping to create sharper decision boundary
        // centered at 0.75 load: overload = 1 / (1 + exp(-10 * (combined - 0.75)))
        1.0 / (1.0 + (-SIGMOID_STEEPNESS * (combined - 0.75)).exp())
    }
}

// ─── Overload Prediction ─────────────────────────────────────────────────

/// Output of the NWO predictor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverloadPrediction {
    /// Predicted probability of overload in the next prediction horizon (0.0–1.0).
    pub overload_probability: f64,

    /// Raw linear output before sigmoid (for debugging/analysis).
    pub raw_output: f64,

    /// The input features used for this prediction.
    pub features: LoadFeatures,

    /// Huber loss from the most recent training step (if available).
    pub last_loss: Option<f64>,

    /// Prediction timestamp (Unix epoch millis).
    pub timestamp_ms: u64,
}

impl OverloadPrediction {
    /// Returns true if the overload probability exceeds the given threshold.
    #[inline]
    pub fn exceeds_threshold(&self, threshold: f64) -> bool {
        self.overload_probability > threshold
    }

    /// Returns the overload probability.
    #[inline]
    pub fn probability(&self) -> f64 {
        self.overload_probability
    }
}

// ─── Adam Optimizer State ────────────────────────────────────────────────

/// Adam optimizer state for the NWO Perceptron weights.
///
/// Maintains first moment (m) and second moment (v) estimates
/// for each weight, enabling adaptive learning rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamState {
    /// First moment estimates for weights (same shape as W).
    pub m_weights: [f64; PREDICTOR_FEATURES],

    /// Second moment estimates for weights.
    pub v_weights: [f64; PREDICTOR_FEATURES],

    /// First moment estimate for bias.
    pub m_bias: f64,

    /// Second moment estimate for bias.
    pub v_bias: f64,

    /// Step counter (incremented on each update).
    pub step: u64,

    /// β₁ (exponential decay rate for first moment).
    pub beta1: f64,

    /// β₂ (exponential decay rate for second moment).
    pub beta2: f64,

    /// ε (numerical stability constant).
    pub epsilon: f64,

    /// Learning rate α.
    pub learning_rate: f64,
}

impl AdamState {
    /// Create a new Adam state with default hyperparameters.
    pub fn new() -> Self {
        Self {
            m_weights: [0.0; PREDICTOR_FEATURES],
            v_weights: [0.0; PREDICTOR_FEATURES],
            m_bias: 0.0,
            v_bias: 0.0,
            step: 0,
            beta1: ADAM_BETA1,
            beta2: ADAM_BETA2,
            epsilon: ADAM_EPSILON,
            learning_rate: ADAM_LEARNING_RATE,
        }
    }

    /// Create with custom hyperparameters.
    pub fn with_params(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        Self {
            m_weights: [0.0; PREDICTOR_FEATURES],
            v_weights: [0.0; PREDICTOR_FEATURES],
            m_bias: 0.0,
            v_bias: 0.0,
            step: 0,
            beta1,
            beta2,
            epsilon,
            learning_rate,
        }
    }

    /// Reset the optimizer state (e.g., after a shard split creates new weights).
    pub fn reset(&mut self) {
        self.m_weights = [0.0; PREDICTOR_FEATURES];
        self.v_weights = [0.0; PREDICTOR_FEATURES];
        self.m_bias = 0.0;
        self.v_bias = 0.0;
        self.step = 0;
    }

    /// Update a single weight using Adam.
    ///
    /// Returns the updated weight value.
    pub fn update_weight(
        &mut self,
        weight: f64,
        gradient: f64,
        idx: usize,
    ) -> f64 {
        if idx >= PREDICTOR_FEATURES {
            return weight;
        }

        // m = β₁ * m + (1 - β₁) * g
        self.m_weights[idx] = self.beta1 * self.m_weights[idx]
            + (1.0 - self.beta1) * gradient;

        // v = β₂ * v + (1 - β₂) * g²
        self.v_weights[idx] = self.beta2 * self.v_weights[idx]
            + (1.0 - self.beta2) * gradient * gradient;

        // Bias correction
        let m_hat = self.m_weights[idx] / (1.0 - self.beta1.powi(self.step as i32 + 1));
        let v_hat = self.v_weights[idx] / (1.0 - self.beta2.powi(self.step as i32 + 1));

        // Weight update: W = W - α * m_hat / (√v_hat + ε)
        weight - self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon)
    }

    /// Update the bias using Adam.
    ///
    /// Returns the updated bias value.
    pub fn update_bias(&mut self, bias: f64, gradient: f64) -> f64 {
        self.m_bias = self.beta1 * self.m_bias + (1.0 - self.beta1) * gradient;
        self.v_bias = self.beta2 * self.v_bias + (1.0 - self.beta2) * gradient * gradient;

        let m_hat = self.m_bias / (1.0 - self.beta1.powi(self.step as i32 + 1));
        let v_hat = self.v_bias / (1.0 - self.beta2.powi(self.step as i32 + 1));

        bias - self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon)
    }

    /// Increment the step counter.
    pub fn advance_step(&mut self) {
        self.step += 1;
    }
}

impl Default for AdamState {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Huber Loss ──────────────────────────────────────────────────────────

/// Compute the Huber loss.
///
/// ```text
/// L_δ(y, ŷ) = {
///   0.5 * (y - ŷ)²           if |y - ŷ| ≤ δ
///   δ * |y - ŷ| - 0.5 * δ²   if |y - ŷ| > δ
/// }
/// ```
///
/// Huber loss is quadratic for small residuals (like MSE) and linear
/// for large residuals (like MAE). This makes it robust to outliers
/// such as MEV spikes or massive exchange inflows.
#[inline]
pub fn huber_loss(prediction: f64, target: f64, delta: f64) -> f64 {
    let residual = target - prediction;
    let abs_residual = residual.abs();

    if abs_residual <= delta {
        0.5 * residual * residual
    } else {
        delta * abs_residual - 0.5 * delta * delta
    }
}

/// Compute the derivative of Huber loss with respect to the prediction.
///
/// ```text
/// dL/dŷ = {
///   -(y - ŷ)           if |y - ŷ| ≤ δ
///   -δ * sign(y - ŷ)   if |y - ŷ| > δ
/// }
/// ```
#[inline]
pub fn huber_loss_derivative(prediction: f64, target: f64, delta: f64) -> f64 {
    let residual = target - prediction;

    if residual.abs() <= delta {
        -residual
    } else {
        -delta * residual.signum()
    }
}

// ─── Sigmoid ─────────────────────────────────────────────────────────────

/// Numerically stable sigmoid function.
///
/// σ(x) = 1 / (1 + exp(-x))
///
/// Uses the stable formulation to avoid overflow:
/// - For x >= 0: 1 / (1 + exp(-x))
/// - For x < 0: exp(x) / (1 + exp(x))
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x)).
#[inline]
pub fn sigmoid_derivative(sigmoid_output: f64) -> f64 {
    sigmoid_output * (1.0 - sigmoid_output)
}

// ─── NWO Predictor ───────────────────────────────────────────────────────

/// The Neural Weight Oscillator (NWO) Perceptron for shard load prediction.
///
/// A single-layer linear model: output = W · x + b, followed by sigmoid
/// to produce an overload probability in [0, 1].
///
/// The model is continuously trained via Adam optimizer and Huber Loss
/// on every block, adapting to changing network conditions in real-time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NwoPredictor {
    /// Weight vector W (PREDICTOR_FEATURES dimensions).
    pub weights: [f64; PREDICTOR_FEATURES],

    /// Bias term b.
    pub bias: f64,

    /// Adam optimizer state.
    pub adam: AdamState,

    /// Huber loss δ parameter.
    pub huber_delta: f64,

    /// Maximum gradient norm for clipping (prevents adversarial updates).
    pub max_gradient_norm: f64,

    /// Number of training steps completed.
    pub train_steps: u64,

    /// Running average of loss (for monitoring).
    pub avg_loss: f64,

    /// Sigmoid steepness multiplier (scales raw output before sigmoid).
    pub sigmoid_scale: f64,
}

impl NwoPredictor {
    /// Create a new NWO predictor with default initialization.
    ///
    /// Weights are initialized with small random-like values using
    /// Xavier-like initialization scaled for 6 inputs.
    pub fn new() -> Self {
        // Xavier-like initialization: scale = 1/sqrt(n_in)
        let scale = 1.0 / (PREDICTOR_FEATURES as f64).sqrt();

        // Deterministic initial weights (in production, these would be randomized)
        // The NWO learns quickly via Adam, so initialization doesn't need to be perfect
        let weights = [
            scale * 0.8,  // TPS is the strongest signal
            scale * 0.6,  // Mempool utilization
            scale * 0.4,  // Inter-shard queue
            scale * 0.5,  // Gas utilization
            scale * 0.3,  // Block execution time
            scale * 0.1,  // Cross-shard ratio (weakest signal)
        ];

        Self {
            weights,
            bias: -scale * 2.0, // Negative bias: default to low overload probability
            adam: AdamState::new(),
            huber_delta: HUBER_DELTA,
            max_gradient_norm: 1.0,
            train_steps: 0,
            avg_loss: 0.0,
            sigmoid_scale: SIGMOID_STEEPNESS,
        }
    }

    /// Create with specific configuration parameters.
    pub fn with_config(config: &ShardingConfig) -> Self {
        let mut predictor = Self::new();
        predictor.huber_delta = HUBER_DELTA;
        predictor.sigmoid_scale = SIGMOID_STEEPNESS;
        predictor
    }

    /// Create a predictor initialized to a specific split threshold.
    ///
    /// This configures the weights so that the sigmoid crosses 0.5
    /// at the desired threshold, giving a good initial decision boundary.
    pub fn with_split_threshold(split_threshold: f64) -> Self {
        let mut predictor = Self::new();
        // Inverse sigmoid at threshold: log(p / (1-p)) / scale
        if split_threshold > 0.0 && split_threshold < 1.0 {
            let inv_sigmoid = (split_threshold / (1.0 - split_threshold)).ln();
            // We want: when features are all at split_threshold, raw_output ≈ inv_sigmoid
            // Since features sum up to roughly 6 * split_threshold, and weights are roughly equal:
            // avg_weight * 6 * split_threshold + bias ≈ inv_sigmoid
            // We set avg_weight and bias to satisfy this
            let avg_weight = inv_sigmoid / (6.0 * split_threshold + 1.0);
            predictor.bias = avg_weight;
            predictor.weights = [avg_weight; PREDICTOR_FEATURES];
        }
        predictor
    }

    /// Forward pass: compute raw linear output W · x + b.
    #[inline]
    pub fn forward_raw(&self, features: &LoadFeatures) -> f64 {
        let x = features.to_array();
        let mut output = self.bias;
        for i in 0..PREDICTOR_FEATURES {
            output += self.weights[i] * x[i];
        }
        output
    }

    /// Forward pass: compute overload probability.
    ///
    /// Returns σ(scale * (W · x + b)) where σ is the sigmoid function.
    pub fn forward(&self, features: &LoadFeatures) -> f64 {
        let raw = self.forward_raw(features);
        sigmoid(self.sigmoid_scale * raw)
    }

    /// Predict overload probability from raw metrics.
    ///
    /// This is the primary API for the sharding subsystem.
    pub fn predict_overload(&self, metrics: &crate::sharding::ShardMetrics) -> OverloadPrediction {
        // Convert ShardMetrics to LoadMetrics
        let load_metrics = LoadMetrics {
            current_tps: metrics.tps,
            mempool_size: metrics.mempool_size,
            inter_shard_queue_depth: metrics.inter_shard_queue_depth,
            gas_utilization: metrics.gas_utilization,
            avg_block_exec_ms: metrics.avg_block_exec_ms,
            cross_shard_ratio: metrics.cross_shard_ratio,
            max_tps_capacity: MAX_TPS,
        };

        let features = load_metrics.normalize().clamp();
        let raw_output = self.forward_raw(&features);
        let overload_probability = sigmoid(self.sigmoid_scale * raw_output);

        OverloadPrediction {
            overload_probability,
            raw_output,
            features,
            last_loss: if self.train_steps > 0 {
                Some(self.avg_loss)
            } else {
                None
            },
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Predict overload from already-normalized features.
    pub fn predict_from_features(&self, features: &LoadFeatures) -> OverloadPrediction {
        let features = features.clamp();
        let raw_output = self.forward_raw(&features);
        let overload_probability = sigmoid(self.sigmoid_scale * raw_output);

        OverloadPrediction {
            overload_probability,
            raw_output,
            features,
            last_loss: if self.train_steps > 0 {
                Some(self.avg_loss)
            } else {
                None
            },
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Train the predictor on a single observation.
    ///
    /// # Arguments
    ///
    /// * `features` - The normalized input features
    /// * `target` - The ground truth overload value (0.0 or 1.0, or soft label)
    ///
    /// # Returns
    ///
    /// The Huber loss for this training step.
    ///
    /// # Backpropagation (for a single linear layer + sigmoid)
    ///
    /// ```text
    /// ŷ = σ(scale * (W·x + b))       (prediction)
    /// L = Huber(ŷ, y)                (loss)
    /// dL/dŷ = Huber'(ŷ, y)           (loss derivative)
    /// dŷ/dz = σ'(z) * scale          (sigmoid derivative × scale)
    /// dL/dW_i = dL/dŷ * dŷ/dz * x_i (weight gradient)
    /// dL/db = dL/dŷ * dŷ/dz          (bias gradient)
    /// ```
    pub fn train_step(&mut self, features: &LoadFeatures, target: f64) -> f64 {
        let features = features.clamp();
        let x = features.to_array();

        // Forward pass
        let raw = self.forward_raw(&features);
        let scaled_raw = self.sigmoid_scale * raw;
        let prediction = sigmoid(scaled_raw);

        // Loss
        let loss = huber_loss(prediction, target, self.huber_delta);

        // Backpropagation
        let dloss_dpred = huber_loss_derivative(prediction, target, self.huber_delta);
        let dpred_draw = sigmoid_derivative(prediction) * self.sigmoid_scale;

        // Gradient of loss w.r.t. raw output
        let dloss_draw = dloss_dpred * dpred_draw;

        // Compute weight gradients: dL/dW_i = dL/draw * x_i
        let mut weight_grads = [0.0; PREDICTOR_FEATURES];
        for i in 0..PREDICTOR_FEATURES {
            weight_grads[i] = dloss_draw * x[i];
        }
        let bias_grad = dloss_draw;

        // Gradient clipping
        let mut grad_norm_sq: f64 = bias_grad * bias_grad;
        for &g in &weight_grads {
            grad_norm_sq += g * g;
        }
        let grad_norm = grad_norm_sq.sqrt();

        let clip_factor = if grad_norm > self.max_gradient_norm {
            self.max_gradient_norm / grad_norm
        } else {
            1.0
        };

        // Adam update for each weight
        for i in 0..PREDICTOR_FEATURES {
            let clipped_grad = weight_grads[i] * clip_factor;
            self.weights[i] = self.adam.update_weight(self.weights[i], clipped_grad, i);
        }

        // Adam update for bias
        let clipped_bias_grad = bias_grad * clip_factor;
        self.bias = self.adam.update_bias(self.bias, clipped_bias_grad);

        // Advance Adam step
        self.adam.advance_step();
        self.train_steps += 1;

        // Update running average loss
        let alpha = 0.01; // Slow EMA for loss tracking
        self.avg_loss = alpha * loss + (1.0 - alpha) * self.avg_loss;

        loss
    }

    /// Train on raw metrics (convenience wrapper).
    ///
    /// Normalizes the metrics and uses `overload_label()` as the target.
    pub fn train_on_metrics(&mut self, metrics: &LoadMetrics) -> f64 {
        let features = metrics.normalize();
        let target = metrics.overload_label();
        self.train_step(&features, target)
    }

    /// Get the current weight vector.
    pub fn get_weights(&self) -> &[f64; PREDICTOR_FEATURES] {
        &self.weights
    }

    /// Get the current bias.
    pub fn get_bias(&self) -> f64 {
        self.bias
    }

    /// Get the average loss.
    pub fn average_loss(&self) -> f64 {
        self.avg_loss
    }

    /// Reset the predictor (e.g., for a new shard after split).
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Compute a BLAKE3 hash of the current weights for consensus.
    pub fn weight_hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        for &w in &self.weights {
            hasher.update(&w.to_le_bytes());
        }
        hasher.update(&self.bias.to_le_bytes());
        hasher.finalize().into()
    }

    /// Serialize the predictor state to bytes (for persistence).
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> NervResult<Self> {
        bincode::deserialize(data)
            .map_err(|e| NervError::Sharding(format!("NWO predictor deserialize: {e}")))
    }
}

impl Default for NwoPredictor {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
        assert!((sigmoid(1.0) - 0.7310585786).abs() < 1e-9);
    }

    #[test]
    fn test_sigmoid_derivative() {
        let s = sigmoid(0.0);
        let ds = sigmoid_derivative(s);
        assert!((ds - 0.25).abs() < 1e-10); // σ'(0) = 0.25
    }

    #[test]
    fn test_huber_loss_quadratic_region() {
        // |y - ŷ| ≤ δ: quadratic
        let loss = huber_loss(0.5, 0.7, 1.0);
        let expected = 0.5 * (0.2 * 0.2); // 0.5 * 0.04 = 0.02
        assert!((loss - expected).abs() < 1e-10);
    }

    #[test]
    fn test_huber_loss_linear_region() {
        // |y - ŷ| > δ: linear
        let loss = huber_loss(0.0, 5.0, 1.0);
        let expected = 1.0 * 5.0 - 0.5 * 1.0; // 5.0 - 0.5 = 4.5
        assert!((loss - expected).abs() < 1e-10);
    }

    #[test]
    fn test_huber_loss_derivative() {
        // In quadratic region
        let deriv = huber_loss_derivative(0.5, 0.7, 1.0);
        assert!((deriv - (-0.2)).abs() < 1e-10);

        // In linear region (positive residual)
        let deriv = huber_loss_derivative(0.0, 5.0, 1.0);
        assert!((deriv - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_load_features_clamp() {
        let features = LoadFeatures {
            current_tps: 1.5,
            mempool_utilization: -0.1,
            inter_shard_queue_depth: 0.5,
            gas_utilization: 0.8,
            block_execution_time: 2.0,
            cross_shard_ratio: 0.3,
        };
        let clamped = features.clamp();
        assert!((clamped.current_tps - 1.0).abs() < 1e-10);
        assert!((clamped.mempool_utilization - 0.0).abs() < 1e-10);
        assert!((clamped.block_execution_time - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_load_metrics_normalize() {
        let metrics = LoadMetrics {
            current_tps: 500_000.0,
            mempool_size: 250_000,
            inter_shard_queue_depth: 50_000,
            gas_utilization: 0.75,
            avg_block_exec_ms: 500.0,
            cross_shard_ratio: 0.3,
            max_tps_capacity: 1_000_000.0,
        };
        let features = metrics.normalize();
        assert!((features.current_tps - 0.5).abs() < 1e-6);
        assert!((features.mempool_utilization - 0.5).abs() < 1e-6);
        assert!((features.gas_utilization - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_load_metrics_overload_label() {
        // Idle shard
        let idle = LoadMetrics::default();
        assert!(idle.overload_label() < 0.1);

        // Heavily loaded shard
        let heavy = LoadMetrics {
            current_tps: 1_800_000.0,
            mempool_size: 450_000,
            gas_utilization: 0.98,
            max_tps_capacity: 2_000_000.0,
            ..Default::default()
        };
        assert!(heavy.overload_label() > 0.5);
    }

    #[test]
    fn test_nwo_predictor_forward() {
        let predictor = NwoPredictor::new();

        // Zero features should give low overload probability
        let zero_pred = predictor.forward(&LoadFeatures::ZERO);
        assert!(zero_pred < 0.5);

        // Max features should give high overload probability
        let max_pred = predictor.forward(&LoadFeatures::MAX);
        assert!(max_pred > 0.5);
    }

    #[test]
    fn test_nwo_predictor_train_convergence() {
        let mut predictor = NwoPredictor::new();

        // Train on overloaded examples (should increase overload prediction)
        let overloaded_features = LoadFeatures {
            current_tps: 0.95,
            mempool_utilization: 0.9,
            gas_utilization: 0.95,
            ..LoadFeatures::ZERO
        };

        let idle_features = LoadFeatures {
            current_tps: 0.1,
            mempool_utilization: 0.05,
            gas_utilization: 0.1,
            ..LoadFeatures::ZERO
        };

        let initial_overload_pred = predictor.forward(&overloaded_features);

        // Train for many steps
        for _ in 0..100 {
            predictor.train_step(&overloaded_features, 1.0);
            predictor.train_step(&idle_features, 0.0);
        }

        let final_overload_pred = predictor.forward(&overloaded_features);
        let final_idle_pred = predictor.forward(&idle_features);

        // After training, the predictor should better distinguish
        assert!(final_overload_pred > final_idle_pred);
    }

    #[test]
    fn test_adam_state_update() {
        let mut adam = AdamState::new();

        let w0 = 1.0;
        let w1 = adam.update_weight(w0, -0.5, 0);
        adam.advance_step();

        // After one step, the weight should have changed
        assert_ne!(w0, w1);
    }

    #[test]
    fn test_nwo_predictor_weight_hash() {
        let p1 = NwoPredictor::new();
        let p2 = NwoPredictor::new();
        // Same initialization → same hash
        assert_eq!(p1.weight_hash(), p2.weight_hash());
    }

    #[test]
    fn test_nwo_predictor_serialization() {
        let predictor = NwoPredictor::new();
        let bytes = predictor.to_bytes();
        let restored = NwoPredictor::from_bytes(&bytes).unwrap();
        assert_eq!(predictor.weights, restored.weights);
        assert!((predictor.bias - restored.bias).abs() < 1e-10);
    }

    #[test]
    fn test_overload_prediction_threshold() {
        let pred = OverloadPrediction {
            overload_probability: 0.95,
            raw_output: 3.0,
            features: LoadFeatures::ZERO,
            last_loss: None,
            timestamp_ms: 0,
        };
        assert!(pred.exceeds_threshold(0.90));
        assert!(!pred.exceeds_threshold(0.99));
    }
}
