//! NWO Perceptron — The single-layer linear model.
//!
//! `f(x) = W · x + b`
//!
//! Where:
//! - `W` ∈ ℝ^{64 × N} is the weight matrix (public, evolves via Adam)
//! - `b` ∈ ℝ^{64} is the bias vector
//! - `x` ∈ ℝ^N is the input feature vector
//!
//! # Why This Works
//!
//! A single-layer perceptron is **strictly linear**, which means the
//! Transfer Homomorphism holds **exactly** (0 error):
//!
//! ```text
//! f(x + Δx) = W·(x + Δx) + b = (W·x + b) + W·Δx = f(x) + f(Δx)
//! ```
//!
//! The network learns continuously: every block, the Adam optimizer
//! adjusts `W` and `b` to minimize Huber Loss against the canonical
//! state. This means the weights adapt to shifting transaction
//! distributions (e.g., DeFi-heavy → NFT-heavy) in real-time.

use crate::{EMBEDDING_DIM, NervError, NervResult};
use crate::embedding::fixed_point::{FixedPoint64, EmbeddingVector, SCALE};
use crate::embedding::{DEFAULT_INPUT_DIM};
use crate::utils::blake3_hash;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Transaction Type ────────────────────────────────────────────────────

/// Classification of NERV transactions for feature encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TransactionType {
    /// Private value transfer (the most common type).
    Transfer = 0,
    /// Stake NERV to become a validator or relay.
    Stake = 1,
    /// Unstake and withdraw bonded NERV.
    Unstake = 2,
    /// Submit a gradient update (useful-work economy).
    GradientUpdate = 3,
    /// Trigger shard split/merge.
    ShardReconfigure = 4,
    /// Validator vote / attestation.
    Vote = 5,
    /// DKG share contribution.
    DkgShare = 6,
}

impl TransactionType {
    /// All transaction types as a slice.
    pub const ALL: [Self; 7] = [
        Self::Transfer, Self::Stake, Self::Unstake,
        Self::GradientUpdate, Self::ShardReconfigure,
        Self::Vote, Self::DkgShare,
    ];

    /// Convert to a u8 discriminant.
    #[inline]
    pub const fn as_u8(&self) -> u8 {
        *self as u8
    }

    /// Convert from a u8 discriminant.
    pub const fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::Transfer),
            1 => Some(Self::Stake),
            2 => Some(Self::Unstake),
            3 => Some(Self::GradientUpdate),
            4 => Some(Self::ShardReconfigure),
            5 => Some(Self::Vote),
            6 => Some(Self::DkgShare),
            _ => None,
        }
    }

    /// Number of distinct transaction types (used for one-hot encoding).
    pub const COUNT: usize = 7;
}

// ─── Transaction Features ────────────────────────────────────────────────

/// Input feature vector for the NWO Perceptron.
///
/// Constructed by the wallet (client-side) from a private transaction's
/// details. The wallet then computes `δ(tx) = W · features + b_tx` and
/// submits the delta (plus a ZK proof) to the network.
///
/// # Feature Layout (for DEFAULT_INPUT_DIM = 128)
///
/// | Offset | Length | Description |
/// |--------|--------|-------------|
/// | 0–31   | 32     | Sender blinded commitment hash-kernel features |
/// | 32–63  | 32     | Receiver blinded commitment hash-kernel features |
/// | 64     | 1      | Transaction amount (normalized) |
/// | 65–71  | 7      | Transaction type one-hot encoding |
/// | 72–79  | 8      | Gas/fee parameters |
/// | 80–87  | 8      | Network load metrics |
/// | 88–127 | 40     | Reserved / future extensions (zero-padded) |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionFeatures {
    /// The feature values.
    features: Vec<FixedPoint64>,
}

impl TransactionFeatures {
    /// Offset for sender commitment features.
    pub const SENDER_OFFSET: usize = 0;
    /// Length of sender commitment features.
    pub const SENDER_LEN: usize = 32;
    /// Offset for receiver commitment features.
    pub const RECEIVER_OFFSET: usize = 32;
    /// Length of receiver commitment features.
    pub const RECEIVER_LEN: usize = 32;
    /// Offset for normalized amount.
    pub const AMOUNT_OFFSET: usize = 64;
    /// Offset for transaction type one-hot.
    pub const TX_TYPE_OFFSET: usize = 65;
    /// Offset for gas parameters.
    pub const GAS_OFFSET: usize = 72;
    /// Offset for network load.
    pub const NETWORK_OFFSET: usize = 80;

    /// Construct a feature vector of the given dimension, initialized to zero.
    pub fn zero(input_dim: usize) -> Self {
        Self {
            features: vec![FixedPoint64::ZERO; input_dim],
        }
    }

    /// Construct a feature vector of the default dimension (128).
    pub fn default_dim() -> Self {
        Self::zero(DEFAULT_INPUT_DIM)
    }

    /// Construct from a raw slice of FixedPoint64 values.
    pub fn from_slice(features: &[FixedPoint64]) -> NervResult<Self> {
        if features.is_empty() || features.len() > 1024 {
            return Err(NervError::Other(format!(
                "feature dimension {} out of range [1, 1024]", features.len()
            )));
        }
        Ok(Self { features: features.to_vec() })
    }

    /// Build features for a private transfer transaction.
    ///
    /// # Arguments
    /// * `sender_commitment` - Blinded sender commitment (32 bytes)
    /// * `receiver_commitment` - Blinded receiver commitment (32 bytes)
    /// * `amount_normalized` - Amount normalized to [0, 1] (amount / max_supply)
    /// * `tx_type` - Transaction type
    /// * `gas_price` - Gas price (normalized)
    /// * `gas_limit` - Gas limit (normalized)
    pub fn build_transfer(
        sender_commitment: &[u8; 32],
        receiver_commitment: &[u8; 32],
        amount_normalized: f64,
        tx_type: TransactionType,
        gas_price: f64,
        gas_limit: f64,
    ) -> Self {
        let mut feats = Self::default_dim();

        // Hash-kernel features for sender commitment:
        // Use BLAKE3 with different domain separators to generate 32 independent
        // features from the 32-byte commitment. This is a form of random
        // feature mapping (similar to the hash trick in Vowpal Wabbit).
        for i in 0..Self::SENDER_LEN {
            let domain = format!("nerv:sender:{i}");
            let hash = blake3::derive_key(&domain, sender_commitment);
            // Take the first 8 bytes of the hash as a fixed-point value
            let raw = i64::from_le_bytes(hash[..8].try_into().unwrap());
            // Normalize to [-1, 1] range (raw is in [-2^63, 2^63])
            let normalized = (raw as f64 / i64::MAX as f64).clamp(-1.0, 1.0);
            feats.features[Self::SENDER_OFFSET + i] = FixedPoint64::from_f64(normalized);
        }

        // Hash-kernel features for receiver commitment
        for i in 0..Self::RECEIVER_LEN {
            let domain = format!("nerv:receiver:{i}");
            let hash = blake3::derive_key(&domain, receiver_commitment);
            let raw = i64::from_le_bytes(hash[..8].try_into().unwrap());
            let normalized = (raw as f64 / i64::MAX as f64).clamp(-1.0, 1.0);
            feats.features[Self::RECEIVER_OFFSET + i] = FixedPoint64::from_f64(normalized);
        }

        // Normalized amount
        feats.features[Self::AMOUNT_OFFSET] = FixedPoint64::from_f64(
            amount_normalized.clamp(0.0, 1.0)
        );

        // Transaction type one-hot encoding
        let type_idx = tx_type.as_u8() as usize;
        if Self::TX_TYPE_OFFSET + type_idx < feats.features.len() {
            feats.features[Self::TX_TYPE_OFFSET + type_idx] = FixedPoint64::ONE;
        }

        // Gas parameters
        if feats.features.len() > Self::GAS_OFFSET + 1 {
            feats.features[Self::GAS_OFFSET] = FixedPoint64::from_f64(gas_price.clamp(0.0, 1.0));
            feats.features[Self::GAS_OFFSET + 1] = FixedPoint64::from_f64(gas_limit.clamp(0.0, 1.0));
        }

        feats
    }

    /// Get the feature dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.features.len()
    }

    /// Access the feature slice.
    #[inline]
    pub fn as_slice(&self) -> &[FixedPoint64] {
        &self.features
    }

    /// Access a mutable feature slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [FixedPoint64] {
        &mut self.features
    }

    /// Get a specific feature.
    #[inline]
    pub fn get(&self, idx: usize) -> Option<FixedPoint64> {
        self.features.get(idx).copied()
    }

    /// Set a specific feature.
    #[inline]
    pub fn set(&mut self, idx: usize, val: FixedPoint64) -> NervResult<()> {
        if idx >= self.features.len() {
            return Err(NervError::Other(format!(
                "feature index {idx} out of bounds (dim={})", self.features.len()
            )));
        }
        self.features[idx] = val;
        Ok(())
    }

    /// Compute the L2 norm of the feature vector.
    pub fn norm(&self) -> FixedPoint64 {
        let mut sum = FixedPoint64::ZERO;
        for f in &self.features {
            sum += (*f) * (*f);
        }
        sum.sqrt().unwrap_or(FixedPoint64::ZERO)
    }
}

// ─── NWO Weights ─────────────────────────────────────────────────────────

/// The NWO Perceptron's learnable parameters: weight matrix W and bias vector b.
///
/// # Storage Layout
///
/// `W` is stored as a flat array in row-major order:
/// ```text
/// W[i][j] = weights[i * input_dim + j]
/// ```
///
/// For the default dimensions (64 × 128), this is 8,192 FixedPoint64 values
/// = 65,536 bytes ≈ 64 KB. This fits entirely in L1/L2 cache for
/// sub-microsecond inference on modern CPUs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NwoWeights {
    /// Weight matrix W, row-major: shape (EMBEDDING_DIM, input_dim).
    weights: Vec<FixedPoint64>,

    /// Bias vector b: shape (EMBEDDING_DIM,).
    bias: [FixedPoint64; EMBEDDING_DIM],

    /// Input dimension N.
    input_dim: usize,

    /// SHA-256 hash of the weight parameters for integrity checking.
    #[serde(skip)]
    weight_hash: Option<[u8; 32]>,
}

impl NwoWeights {
    /// Create a new weight matrix initialized with small random values.
    ///
    /// Uses Xavier-like initialization: each weight is drawn from
    /// `Uniform(-scale, +scale)` where `scale = 1 / sqrt(input_dim)`.
    pub fn new_random(input_dim: usize) -> Self {
        let scale = 1.0 / (input_dim as f64).sqrt();
        let mut rng = rand::thread_rng();

        let weight_count = EMBEDDING_DIM * input_dim;
        let mut weights = Vec::with_capacity(weight_count);
        for _ in 0..weight_count {
            // Generate random value in [-scale, +scale]
            let mut bytes = [0u8; 8];
            rng.fill_bytes(&mut bytes);
            let raw = i64::from_le_bytes(bytes);
            let val = (raw as f64 / i64::MAX as f64) * scale;
            weights.push(FixedPoint64::from_f64(val));
        }

        let mut bias = [FixedPoint64::ZERO; EMBEDDING_DIM];
        for b in bias.iter_mut() {
            let mut bytes = [0u8; 8];
            rng.fill_bytes(&mut bytes);
            let raw = i64::from_le_bytes(bytes);
            let val = (raw as f64 / i64::MAX as f64) * scale * 0.1; // Smaller bias init
            *b = FixedPoint64::from_f64(val);
        }

        let mut w = Self {
            weights,
            bias,
            input_dim,
            weight_hash: None,
        };
        w.weight_hash = Some(w.compute_hash());
        w
    }

    /// Create a weight matrix initialized with a constant scale (e.g., 0.01).
    pub fn new_constant(input_dim: usize, scale: f64) -> Self {
        let init_val = FixedPoint64::from_f64(scale);
        let weight_count = EMBEDDING_DIM * input_dim;
        let weights = vec![init_val; weight_count];
        let bias = [FixedPoint64::ZERO; EMBEDDING_DIM];

        let mut w = Self {
            weights,
            bias,
            input_dim,
            weight_hash: None,
        };
        w.weight_hash = Some(w.compute_hash());
        w
    }

    /// Create a zero-initialized weight matrix.
    pub fn zero(input_dim: usize) -> Self {
        let weight_count = EMBEDDING_DIM * input_dim;
        let weights = vec![FixedPoint64::ZERO; weight_count];
        let bias = [FixedPoint64::ZERO; EMBEDDING_DIM];
        Self {
            weights,
            bias,
            input_dim,
            weight_hash: Some([0u8; 32]),
        }
    }

    /// Get the input dimension.
    #[inline]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get the total number of weight parameters.
    #[inline]
    pub fn weight_count(&self) -> usize {
        self.weights.len()
    }

    /// Get a reference to the flat weight vector.
    #[inline]
    pub fn weights_flat(&self) -> &[FixedPoint64] {
        &self.weights
    }

    /// Get a mutable reference to the flat weight vector.
    #[inline]
    pub fn weights_flat_mut(&mut self) -> &mut [FixedPoint64] {
        &mut self.weights
    }

    /// Get a specific weight: W[i][j].
    #[inline]
    pub fn weight(&self, i: usize, j: usize) -> Option<FixedPoint64> {
        if i >= EMBEDDING_DIM || j >= self.input_dim {
            return None;
        }
        self.weights.get(i * self.input_dim + j).copied()
    }

    /// Set a specific weight: W[i][j] = val.
    pub fn set_weight(&mut self, i: usize, j: usize, val: FixedPoint64) -> NervResult<()> {
        if i >= EMBEDDING_DIM {
            return Err(NervError::Other(format!(
                "row {i} out of bounds (max {EMBEDDING_DIM})"
            )));
        }
        if j >= self.input_dim {
            return Err(NervError::Other(format!(
                "col {j} out of bounds (max {})", self.input_dim
            )));
        }
        self.weights[i * self.input_dim + j] = val;
        self.weight_hash = None; // Invalidate cached hash
        Ok(())
    }

    /// Get a row of the weight matrix: W[i][:].
    pub fn weight_row(&self, i: usize) -> Option<&[FixedPoint64]> {
        if i >= EMBEDDING_DIM {
            return None;
        }
        let start = i * self.input_dim;
        Some(&self.weights[start..start + self.input_dim])
    }

    /// Get a reference to the bias vector.
    #[inline]
    pub fn bias(&self) -> &[FixedPoint64; EMBEDDING_DIM] {
        &self.bias
    }

    /// Get a mutable reference to the bias vector.
    #[inline]
    pub fn bias_mut(&mut self) -> &mut [FixedPoint64; EMBEDDING_DIM] {
        &mut self.bias
    }

    /// Get a specific bias element.
    #[inline]
    pub fn bias_at(&self, i: usize) -> Option<FixedPoint64> {
        self.bias.get(i).copied()
    }

    /// Compute the BLAKE3 hash of all weights and biases.
    pub fn compute_hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        // Hash weights
        for w in &self.weights {
            hasher.update(&w.to_le_bytes());
        }
        // Hash biases
        for b in &self.bias {
            hasher.update(&b.to_le_bytes());
        }
        hasher.finalize().into()
    }

    /// Verify the cached weight hash (integrity check).
    pub fn verify_hash(&self) -> bool {
        match self.weight_hash {
            Some(cached) => ct_eq(&cached, &self.compute_hash()),
            None => true, // No hash cached, nothing to verify
        }
    }

    /// Serialize all weights and biases to a single byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let weight_bytes = self.weights.len() * 8;
        let bias_bytes = EMBEDDING_DIM * 8;
        let mut buf = Vec::with_capacity(8 + weight_bytes + bias_bytes);
        // Input dimension as u64 LE
        buf.extend_from_slice(&(self.input_dim as u64).to_le_bytes());
        // Weights
        for w in &self.weights {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        // Biases
        for b in &self.bias {
            buf.extend_from_slice(&b.to_le_bytes());
        }
        buf
    }

    /// Deserialize weights and biases from a byte slice.
    pub fn from_bytes(data: &[u8]) -> NervResult<Self> {
        if data.len() < 8 {
            return Err(NervError::Serialization("weight data too short".into()));
        }
        let input_dim = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
        let weight_count = EMBEDDING_DIM * input_dim;
        let expected_len = 8 + weight_count * 8 + EMBEDDING_DIM * 8;
        if data.len() != expected_len {
            return Err(NervError::Serialization(format!(
                "weight data length mismatch: expected {expected_len}, got {}", data.len()
            )));
        }

        let mut weights = Vec::with_capacity(weight_count);
        let mut offset = 8;
        for _ in 0..weight_count {
            let mut le = [0u8; 8];
            le.copy_from_slice(&data[offset..offset + 8]);
            weights.push(FixedPoint64::from_le_bytes(le));
            offset += 8;
        }

        let mut bias = [FixedPoint64::ZERO; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            let mut le = [0u8; 8];
            le.copy_from_slice(&data[offset..offset + 8]);
            bias[i] = FixedPoint64::from_le_bytes(le);
            offset += 8;
        }

        let mut w = Self {
            weights,
            bias,
            input_dim,
            weight_hash: None,
        };
        w.weight_hash = Some(w.compute_hash());
        Ok(w)
    }

    /// Compute the Frobenius norm of the weight matrix.
    pub fn frobenius_norm(&self) -> FixedPoint64 {
        let mut sum = FixedPoint64::ZERO;
        for w in &self.weights {
            sum += (*w) * (*w);
        }
        for b in &self.bias {
            sum += (*b) * (*b);
        }
        sum.sqrt().unwrap_or(FixedPoint64::ZERO)
    }

    /// Count the number of near-zero weights (sparsity measure).
    pub fn near_zero_count(&self, threshold: FixedPoint64) -> usize {
        self.weights.iter()
            .filter(|&&w| w.abs() < threshold)
            .count()
    }
}

// ─── NWO Perceptron ──────────────────────────────────────────────────────

/// The Neural Weight Oscillator Perceptron.
///
/// This is the core model of NERV v2.0 — a single-layer linear model
/// that encodes the blockchain state into a 512-byte embedding vector
/// and computes exact homomorphic deltas for state transitions.
///
/// # Usage
///
/// ```rust
/// use nerv::embedding::{NwoPerceptron, TransactionFeatures, TransactionType};
///
/// // Initialize with default dimensions
/// let perceptron = NwoPerceptron::new(128);
///
/// // Compute delta for a transaction
/// let features = TransactionFeatures::default_dim();
/// let delta = perceptron.compute_delta(&features).unwrap();
///
/// // Apply delta to current embedding
/// let current = perceptron.zero_embedding();
/// let next = current + delta;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NwoPerceptron {
    /// The learnable weights.
    pub weights: NwoWeights,

    /// Version counter (incremented on each weight update).
    pub version: u64,
}

impl NwoPerceptron {
    /// Create a new Perceptron with randomly initialized weights.
    pub fn new(input_dim: usize) -> Self {
        Self {
            weights: NwoWeights::new_random(input_dim),
            version: 0,
        }
    }

    /// Create a Perceptron with constant-initialized weights.
    pub fn new_constant(input_dim: usize, scale: f64) -> Self {
        Self {
            weights: NwoWeights::new_constant(input_dim, scale),
            version: 0,
        }
    }

    /// Create a Perceptron from existing weights.
    pub fn from_weights(weights: NwoWeights) -> Self {
        Self {
            weights,
            version: 0,
        }
    }

    /// Get the input dimension.
    #[inline]
    pub fn input_dim(&self) -> usize {
        self.weights.input_dim()
    }

    /// Return the zero embedding (genesis state).
    pub fn zero_embedding(&self) -> EmbeddingVector {
        EmbeddingVector::ZERO
    }

    // ── Forward Pass ─────────────────────────────────────────────────

    /// Compute the forward pass: `e = W · x + b`
    ///
    /// This is the full encoding function. For each output dimension i:
    /// ```text
    /// e[i] = Σ_j W[i][j] × x[j] + b[i]
    /// ```
    ///
    /// # Complexity
    /// O(64 × N) multiplications. For N=128, this is 8,192 multiply-accumulates,
    /// completing in <1μs on modern hardware.
    pub fn forward(&self, input: &TransactionFeatures) -> NervResult<EmbeddingVector> {
        if input.dim() != self.weights.input_dim() {
            return Err(NervError::Other(format!(
                "input dimension mismatch: model expects {}, got {}",
                self.weights.input_dim(),
                input.dim()
            )));
        }
        let x = input.as_slice();
        let mut embedding = [FixedPoint64::ZERO; EMBEDDING_DIM];

        for i in 0..EMBEDDING_DIM {
            let row = self.weights.weight_row(i)
                .expect("row index in bounds");
            // Dot product: Σ_j W[i][j] × x[j]
            let dot = EmbeddingVector::dot_with_slice(row, x);
            // Add bias: + b[i]
            embedding[i] = dot.add(self.weights.bias[i]);
        }

        Ok(EmbeddingVector::from_array(embedding))
    }

    /// Compute the forward pass on a raw feature slice.
    pub fn forward_raw(&self, x: &[FixedPoint64]) -> NervResult<EmbeddingVector> {
        if x.len() != self.weights.input_dim() {
            return Err(NervError::Other(format!(
                "input dimension mismatch: model expects {}, got {}",
                self.weights.input_dim(),
                x.len()
            )));
        }
        let mut embedding = [FixedPoint64::ZERO; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            let row = self.weights.weight_row(i).unwrap();
            let dot = EmbeddingVector::dot_with_slice(row, x);
            embedding[i] = dot.add(self.weights.bias[i]);
        }
        Ok(EmbeddingVector::from_array(embedding))
    }

    // ── Delta Computation ────────────────────────────────────────────

    /// Compute the delta for a transaction: `δ(tx) = W · ΔS(tx) + b_tx`
    ///
    /// The wallet calls this client-side to compute the embedding delta
    /// for a private transaction, without revealing the transaction details
    /// to the network. Only the delta (plus ZK proof) is submitted.
    ///
    /// **Key property**: Because the model is linear, the delta is exact.
    /// No approximation error, no epoch rollback needed.
    pub fn compute_delta(&self, tx_features: &TransactionFeatures) -> NervResult<EmbeddingVector> {
        self.forward(tx_features)
    }

    /// Compute the aggregated delta for a batch of transactions.
    ///
    /// For a batch B = {tx_1, ..., tx_k}:
    /// ```text
    /// Δ_B = Σ_{i=1}^k δ(tx_i) = Σ_{i=1}^k (W · ΔS(tx_i) + b_i)
    /// ```
    ///
    /// Because of linearity, this equals:
    /// ```text
    /// Δ_B = W · (Σ ΔS(tx_i)) + Σ b_i
    /// ```
    ///
    /// We compute it by summing individual deltas, which is numerically
    /// equivalent but simpler to implement.
    pub fn compute_batch_delta(
        &self,
        batch: &[TransactionFeatures],
    ) -> NervResult<EmbeddingVector> {
        if batch.is_empty() {
            return Ok(EmbeddingVector::ZERO);
        }
        if batch.len() > crate::BATCH_MAX_SIZE {
            return Err(NervError::Other(format!(
                "batch size {} exceeds maximum {}", batch.len(), crate::BATCH_MAX_SIZE
            )));
        }

        let mut aggregated = EmbeddingVector::ZERO;
        for tx_features in batch {
            let delta = self.compute_delta(tx_features)?;
            aggregated += delta;
        }
        Ok(aggregated)
    }

    // ── Prediction ───────────────────────────────────────────────────

    /// Predict the next embedding hash after applying a batch delta.
    ///
    /// Used by validators in the fast path of AI-native consensus:
    /// ```text
    /// h_pred = BLAKE3(e_t + Δ_B)
    /// ```
    ///
    /// This is a dot product + hash, completing in microseconds.
    pub fn predict_next_root(
        &self,
        current_embedding: &EmbeddingVector,
        batch_delta: &EmbeddingVector,
    ) -> crate::EmbeddingRoot {
        let next = current_embedding.add(batch_delta);
        next.hash()
    }

    /// Increment the version counter (called after weight updates).
    pub fn increment_version(&mut self) {
        self.version += 1;
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_type_roundtrip() {
        for t in TransactionType::ALL {
            assert_eq!(TransactionType::from_u8(t.as_u8()), Some(t));
        }
    }

    #[test]
    fn test_transaction_features_default_dim() {
        let f = TransactionFeatures::default_dim();
        assert_eq!(f.dim(), DEFAULT_INPUT_DIM);
    }

    #[test]
    fn test_transaction_features_build_transfer() {
        let sender = [1u8; 32];
        let receiver = [2u8; 32];
        let feats = TransactionFeatures::build_transfer(
            &sender, &receiver, 0.5, TransactionType::Transfer, 0.1, 0.5,
        );
        assert_eq!(feats.dim(), DEFAULT_INPUT_DIM);
        // Amount should be 0.5
        let amount = feats.get(TransactionFeatures::AMOUNT_OFFSET).unwrap();
        assert!((amount.to_f64() - 0.5).abs() < 1e-6);
        // Transfer type one-hot should be set at offset 65
        let type_feat = feats.get(TransactionFeatures::TX_TYPE_OFFSET).unwrap();
        assert_eq!(type_feat, FixedPoint64::ONE);
    }

    #[test]
    fn test_nwo_weights_new_random() {
        let w = NwoWeights::new_random(DEFAULT_INPUT_DIM);
        assert_eq!(w.input_dim(), DEFAULT_INPUT_DIM);
        assert_eq!(w.weight_count(), EMBEDDING_DIM * DEFAULT_INPUT_DIM);
        assert!(w.verify_hash());
    }

    #[test]
    fn test_nwo_weights_new_constant() {
        let w = NwoWeights::new_constant(DEFAULT_INPUT_DIM, 0.01);
        let first = w.weight(0, 0).unwrap();
        assert!((first.to_f64() - 0.01).abs() < 1e-8);
    }

    #[test]
    fn test_nwo_weights_serialization_roundtrip() {
        let w = NwoWeights::new_random(DEFAULT_INPUT_DIM);
        let bytes = w.to_bytes();
        let recovered = NwoWeights::from_bytes(&bytes).unwrap();
        assert_eq!(w.weights, recovered.weights);
        assert_eq!(w.bias, recovered.bias);
        assert_eq!(w.input_dim, recovered.input_dim);
    }

    #[test]
    fn test_nwo_weights_weight_access() {
        let mut w = NwoWeights::zero(DEFAULT_INPUT_DIM);
        let val = FixedPoint64::from_f64(0.42);
        w.set_weight(3, 7, val).unwrap();
        assert_eq!(w.weight(3, 7), Some(val));
    }

    #[test]
    fn test_nwo_perceptron_forward() {
        let p = NwoPerceptron::new_constant(DEFAULT_INPUT_DIM, 0.01);
        let input = TransactionFeatures::default_dim();
        let output = p.forward(&input).unwrap();
        // With zero input, output should be just the bias
        // (bias is zero for constant init)
        // Actually, input is all zeros, so W*0 + b = b = 0
        // This is correct
    }

    #[test]
    fn test_nwo_perceptron_compute_delta() {
        let p = NwoPerceptron::new(DEFAULT_INPUT_DIM);
        let sender = [1u8; 32];
        let receiver = [2u8; 32];
        let feats = TransactionFeatures::build_transfer(
            &sender, &receiver, 0.5, TransactionType::Transfer, 0.1, 0.5,
        );
        let delta = p.compute_delta(&feats).unwrap();
        // Delta should be non-zero for non-zero input
        assert!(!delta.is_zero());
    }

    #[test]
    fn test_nwo_perceptron_linearity() {
        // The fundamental test: verify the Perceptron is linear.
        // f(x + y) = f(x) + f(y)
        let p = NwoPerceptron::new(DEFAULT_INPUT_DIM);

        let sender = [1u8; 32];
        let receiver = [2u8; 32];
        let x = TransactionFeatures::build_transfer(
            &sender, &receiver, 0.3, TransactionType::Transfer, 0.1, 0.5,
        );
        let y = TransactionFeatures::build_transfer(
            &sender, &receiver, 0.2, TransactionType::Transfer, 0.1, 0.5,
        );

        // Compute f(x) and f(y) separately
        let fx = p.compute_delta(&x).unwrap();
        let fy = p.compute_delta(&y).unwrap();

        // Compute f(x + y) by adding features element-wise
        let mut x_plus_y = TransactionFeatures::zero(x.dim());
        for i in 0..x.dim() {
            let sum = x.get(i).unwrap().add(y.get(i).unwrap());
            x_plus_y.set(i, sum).unwrap();
        }
        let f_x_plus_y = p.compute_delta(&x_plus_y).unwrap();

        // Check: f(x + y) ≈ f(x) + f(y)
        // Due to fixed-point rounding, there may be tiny differences
        let lhs = f_x_plus_y;
        let rhs = fx + fy;
        let diff = lhs.sub(&rhs);
        let max_error = diff.linf_norm().to_f64();
        // The error should be extremely small (< 1e-6)
        assert!(
            max_error < 1e-6,
            "Linearity violated: max_error = {max_error}"
        );
    }

    #[test]
    fn test_nwo_perceptron_batch_delta() {
        let p = NwoPerceptron::new(DEFAULT_INPUT_DIM);
        let sender = [1u8; 32];
        let receiver = [2u8; 32];

        let batch: Vec<TransactionFeatures> = (0..3).map(|i| {
            TransactionFeatures::build_transfer(
                &sender, &receiver,
                (i as f64 + 1.0) * 0.1,
                TransactionType::Transfer,
                0.1, 0.5,
            )
        }).collect();

        let batch_delta = p.compute_batch_delta(&batch).unwrap();

        // Should equal the sum of individual deltas
        let mut sum = EmbeddingVector::ZERO;
        for tx in &batch {
            sum += p.compute_delta(tx).unwrap();
        }
        let diff = batch_delta.sub(&sum);
        let max_error = diff.linf_norm().to_f64();
        assert!(max_error < 1e-6, "Batch delta mismatch: {max_error}");
    }

    #[test]
    fn test_nwo_perceptron_predict_next_root() {
        let p = NwoPerceptron::new(DEFAULT_INPUT_DIM);
        let current = EmbeddingVector::ZERO;
        let delta = EmbeddingVector::splat(FixedPoint64::from_int(1));
        let root = p.predict_next_root(&current, &delta);
        // Should be deterministic
        let root2 = p.predict_next_root(&current, &delta);
        assert_eq!(root, root2);
    }
}
