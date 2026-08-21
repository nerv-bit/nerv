//! Neural State Embeddings — NWO Paradigm.
//!
//! Implements the single-layer Perceptron (`f(x) = W·x + b`),
//! the Adam optimizer (β₁=0.9, β₂=0.999), Huber loss calculation,
//! and exact linear homomorphic delta application (`e_{t+1} = e_t + δ(tx)`).
//!
//! Key innovation: Because the Perceptron is strictly linear, the Transfer
//! Homomorphism is exact with 0 error — no approximation, no epoch rollback.
//!
//! Submodules:
//! - `perceptron` — Single-layer linear model (replaces 24-layer Transformer)
//! - `adam` — Adam optimizer for continuous per-block weight updates
//! - `huber_loss` — Huber loss (robust to MEV/state outliers)
//! - `homomorphism` — Exact linear delta application
//! - `fixed_point` — Fixed-point 32.32 arithmetic for embedding vectors
//!
//! Neural State Embeddings — NWO Paradigm.
//!
//! The **Neural Weight Oscillator (NWO)** replaces the 24-layer Transformer
//! of V1.01 with a single-layer Perceptron: `f(x) = W·x + b`.
//!
//! Because this model is **strictly linear**, the Transfer Homomorphism is
//! **exact** (0 error, guaranteed by mathematics):
//!
//! ```text
//! f(x + Δx) = W·(x + Δx) + b = (W·x + b) + W·Δx = f(x) + f(Δx)
//! ```
//!
//! This eliminates:
//! - Pre-training (no federated learning needed)
//! - Approximation error bounds (no epoch rollback)
//! - ZK-ML complexity (7.9M → 50K constraints)
//!
//! The network **self-evolves every block** via the Adam optimizer and
//! Huber Loss, continuously adapting its weights to shifting transaction
//! distributions.

pub mod fixed_point;
pub mod perceptron;
pub mod huber_loss;
pub mod adam;
pub mod homomorphism;

// Re-export primary types
pub use fixed_point::{FixedPoint64, EmbeddingVector};
pub use perceptron::{NwoPerceptron, NwoWeights, TransactionFeatures, TransactionType};
pub use huber_loss::HuberLoss;
pub use adam::{AdamState, AdamConfig, AdamGradients};
pub use homomorphism::{EmbeddingDelta, apply_delta, aggregate_batch_deltas};

use crate::{EMBEDDING_DIM, EMBEDDING_BYTES};

/// Default input feature dimension for the NWO Perceptron.
///
/// The weight matrix W has shape `(EMBEDDING_DIM, INPUT_DIM)` = `(64, 128)`,
/// yielding 8,192 weight parameters. At 8 bytes each, this is ~64 KB —
/// easily fitting in L1 cache for sub-microsecond inference.
pub const DEFAULT_INPUT_DIM: usize = 128;

/// Total number of weight parameters: 64 × 128 = 8,192.
pub const DEFAULT_WEIGHT_COUNT: usize = EMBEDDING_DIM * DEFAULT_INPUT_DIM;

/// Total byte size of the weight matrix: 8,192 × 8 = 65,536 bytes (~64 KB).
pub const DEFAULT_WEIGHT_BYTES: usize = DEFAULT_WEIGHT_COUNT * 8;

/// Total byte size of the bias vector: 64 × 8 = 512 bytes.
pub const BIAS_BYTES: usize = EMBEDDING_DIM * 8;
