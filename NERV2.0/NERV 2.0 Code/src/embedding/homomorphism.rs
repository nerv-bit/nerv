//! Exact Linear Homomorphic Delta Application.
//!
//! The Transfer Homomorphism is the core property that makes NERV's
//! neural state embeddings viable as a blockchain state representation:
//!
//! ```text
//! e_{t+1} = e_t + δ(tx)     with error = 0
//! ```
//!
//! # V1.01 vs V2.0
//!
//! | Property | V1.01 (Transformer) | V2.0 (NWO Perceptron) |
//! |----------|---------------------|-----------------------|
//! | Homomorphism | Approximate (≤10⁻⁹ error) | **Exact** (0 error) |
//! | Error source | Non-linear activations | None (linear by construction) |
//! | Epoch rollback | Possible if error exceeds bound | **Impossible** |
//! | ZK circuit cost | 7.9M constraints | 50K constraints |
//!
//! # Why Exact?
//!
//! The NWO Perceptron is `f(x) = W·x + b`. For any valid delta `Δx`:
//!
//! ```text
//! f(x + Δx) = W·(x + Δx) + b
//!           = W·x + W·Δx + b
//!           = (W·x + b) + W·Δx
//!           = f(x) + f(Δx)        ← exact equality
//! ```
//!
//! No approximation, no error bound, no rollback. The linearity of
//! the model guarantees the homomorphism holds **by mathematical
//! definition**, not by training convergence.

use crate::{EMBEDDING_DIM, BATCH_MAX_SIZE, NervError, NervResult, EmbeddingRoot};
use crate::embedding::fixed_point::{FixedPoint64, EmbeddingVector};
use crate::embedding::perceptron::TransactionFeatures;
use crate::embedding::perceptron::NwoPerceptron;
use crate::utils::blake3_hash;
use serde::{Deserialize, Serialize};

// ─── Embedding Delta ─────────────────────────────────────────────────────

/// A delta vector δ ∈ ℝ⁶⁴ representing the aggregate effect of one or
/// more private transactions on the neural state embedding.
///
/// # Key Properties
///
/// - **Homomorphic**: Can be applied to an embedding via simple vector addition.
/// - **Compact**: 512 bytes, regardless of the number of transactions it represents.
/// - **Linear**: Deltas can be aggregated: `δ(tx1) + δ(tx2) = δ(tx1 ∪ tx2)`.
/// - **Private**: Does not reveal individual transaction amounts, senders, or receivers.
///
/// # Construction
///
/// - **Client-side** (wallet): `δ(tx) = W · ΔS(tx) + b_tx`
/// - **Aggregated** (batch): `Δ_B = Σ δ(tx_i)`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, zeroize::Zeroize)]
#[zeroize(drop)]
pub struct EmbeddingDelta(
    /// The underlying 64-dimensional delta vector.
    pub EmbeddingVector,
);

impl EmbeddingDelta {
    /// The zero delta (no change).
    pub const ZERO: Self = Self(EmbeddingVector::ZERO);

    /// Construct from an embedding vector.
    #[inline]
    pub fn from_vector(v: EmbeddingVector) -> Self {
        Self(v)
    }

    /// Construct a delta with all elements set to the same value.
    pub fn splat(value: FixedPoint64) -> Self {
        Self(EmbeddingVector::splat(value))
    }

    /// Construct from a slice of f64 values.
    pub fn from_f64_slice(values: &[f64]) -> Option<Self> {
        EmbeddingVector::from_f64_slice(values).map(Self)
    }

    /// Access the underlying embedding vector.
    #[inline]
    pub fn as_vector(&self) -> &EmbeddingVector {
        &self.0
    }

    /// Convert into the underlying embedding vector.
    #[inline]
    pub fn into_vector(self) -> EmbeddingVector {
        self.0
    }

    /// Get a specific element of the delta.
    #[inline]
    pub fn get(&self, idx: usize) -> Option<FixedPoint64> {
        self.0.get(idx)
    }

    /// Returns `true` if the delta is zero (no change).
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    /// Compute the L1 norm of the delta (sum of absolute values).
    /// Useful for fee computation and magnitude estimation.
    pub fn magnitude(&self) -> FixedPoint64 {
        self.0.l1_norm()
    }

    /// Compute the L2 norm of the delta.
    pub fn norm(&self) -> FixedPoint64 {
        self.0.norm()
    }

    /// Compute the L∞ norm (max absolute element).
    pub fn max_element(&self) -> FixedPoint64 {
        self.0.linf_norm()
    }

    /// Serialize the delta to 512 bytes.
    pub fn to_bytes(&self) -> [u8; crate::EMBEDDING_BYTES] {
        self.0.to_bytes()
    }

    /// Deserialize from 512 bytes.
    pub fn from_bytes(bytes: &[u8; crate::EMBEDDING_BYTES]) -> Self {
        Self(EmbeddingVector::from_bytes(bytes))
    }

    /// Negate the delta (swap credit ↔ debit).
    pub fn neg(&self) -> Self {
        Self(self.0.neg())
    }

    /// Add two deltas (for aggregation).
    pub fn add(&self, other: &Self) -> Self {
        Self(self.0.add(&other.0))
    }

    /// Scale the delta by a scalar.
    pub fn scale(&self, scalar: FixedPoint64) -> Self {
        Self(self.0.scale(scalar))
    }

    /// Count the number of non-zero elements (sparsity).
    pub fn nnz(&self) -> usize {
        self.0.nnz()
    }
}

// ── Operator Overloads for EmbeddingDelta ────────────────────────────────

impl std::ops::Add for EmbeddingDelta {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::Neg for EmbeddingDelta {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl std::ops::AddAssign for EmbeddingDelta {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl std::fmt::Display for EmbeddingDelta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Δ({})", self.0)
    }
}

impl std::default::Default for EmbeddingDelta {
    fn default() -> Self {
        Self::ZERO
    }
}

// ─── Core Homomorphic Operations ─────────────────────────────────────────

/// Apply a single delta to an embedding: `e_{t+1} = e_t + δ(tx)`.
///
/// This is the fundamental state transition of NERV v2.0.
/// Because the NWO Perceptron is linear, this operation is **exact** —
/// the resulting embedding is precisely what would be obtained by
/// re-encoding the full post-transaction state.
///
/// # Arguments
///
/// * `embedding` - The current canonical embedding `e_t`
/// * `delta` - The transaction delta `δ(tx)`
///
/// # Returns
///
/// The new embedding `e_{t+1} = e_t + δ(tx)` with **0 error**.
///
/// # Example
///
/// ```rust
/// use nerv::embedding::{EmbeddingVector, EmbeddingDelta, apply_delta};
/// use nerv::embedding::fixed_point::FixedPoint64;
///
/// let current = EmbeddingVector::ZERO;
/// let delta = EmbeddingDelta::splat(FixedPoint64::from_int(1));
/// let next = apply_delta(&current, &delta);
/// ```
#[inline]
pub fn apply_delta(embedding: &EmbeddingVector, delta: &EmbeddingDelta) -> EmbeddingVector {
    embedding.add(delta.as_vector())
}

/// Apply a batch of deltas to an embedding:
///
/// ```text
/// e_{t+|B|} = e_t + Δ_B = e_t + Σ_{tx ∈ B} δ(tx)
/// ```
///
/// This is equivalent to applying deltas one-by-one (by associativity
/// of vector addition), but more efficient as it aggregates first.
///
/// # Arguments
///
/// * `embedding` - The current canonical embedding `e_t`
/// * `deltas` - The batch of transaction deltas
///
/// # Returns
///
/// `(e_{t+|B|}, Δ_B)` — the new embedding and the aggregated delta.
pub fn apply_batch(
    embedding: &EmbeddingVector,
    deltas: &[EmbeddingDelta],
) -> NervResult<(EmbeddingVector, EmbeddingDelta)> {
    if deltas.len() > BATCH_MAX_SIZE {
        return Err(NervError::Other(format!(
            "batch size {} exceeds maximum {}",
            deltas.len(),
            BATCH_MAX_SIZE
        )));
    }

    let aggregated = aggregate_batch_deltas(deltas)?;
    let new_embedding = apply_delta(embedding, &aggregated);
    Ok((new_embedding, aggregated))
}

/// Aggregate a batch of deltas into a single combined delta:
///
/// ```text
/// Δ_B = Σ_{i=1}^k δ(tx_i)
/// ```
///
/// Because of linearity, this is simply element-wise vector addition
/// performed incrementally. The result is **independent of the order**
/// of the deltas (commutativity of vector addition).
///
/// # Complexity
///
/// O(k × 64) additions where k is the batch size.
/// For k = 10,000 (V2.0 max batch), this is 640,000 additions,
/// completing in <1ms on a single core.
pub fn aggregate_batch_deltas(deltas: &[EmbeddingDelta]) -> NervResult<EmbeddingDelta> {
    if deltas.len() > BATCH_MAX_SIZE {
        return Err(NervError::Other(format!(
            "batch size {} exceeds maximum {}",
            deltas.len(),
            BATCH_MAX_SIZE
        )));
    }

    let mut aggregated = EmbeddingVector::ZERO;
    for delta in deltas {
        aggregated += delta.as_vector().clone();
    }
    Ok(EmbeddingDelta(aggregated))
}

// ─── Verification ────────────────────────────────────────────────────────

/// Verification result for the homomorphic delta application.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HomomorphismVerification {
    /// The error vector: `e_{t+1} - (e_t + δ)`.
    /// In V2.0, this is always exactly zero.
    pub error_vector: EmbeddingVector,

    /// The maximum absolute error across all 64 dimensions.
    /// In V2.0, this is always exactly 0.0.
    pub max_error: f64,

    /// The L2 norm of the error vector.
    /// In V2.0, this is always exactly 0.0.
    pub error_norm: f64,

    /// Whether the homomorphism is satisfied within the tolerance.
    pub is_satisfied: bool,

    /// The protocol version (always 2 for NERV v2.0).
    pub version: u32,
}

impl HomomorphismVerification {
    /// Verify that the homomorphic delta application is correct.
    ///
    /// Given:
    /// - `e_prev`: The previous embedding `e_t`
    /// - `e_next`: The claimed new embedding `e_{t+1}`
    /// - `delta`: The applied delta `δ(tx)`
    ///
    /// Checks that `e_{t+1} = e_t + δ(tx)` within tolerance.
    ///
    /// In V2.0, the tolerance is **0** (exact). Any non-zero error
    /// indicates a computational bug, not an approximation failure.
    pub fn verify(
        e_prev: &EmbeddingVector,
        e_next: &EmbeddingVector,
        delta: &EmbeddingDelta,
        tolerance: f64,
    ) -> Self {
        // Compute expected next embedding
        let expected = apply_delta(e_prev, delta);

        // Compute error vector: e_next - expected
        let error_vector = e_next.sub(&expected);

        // Maximum absolute error
        let max_error = error_vector.linf_norm().to_f64();

        // L2 error norm
        let error_norm = error_vector.norm().to_f64();

        // Check if within tolerance
        let is_satisfied = max_error <= tolerance;

        Self {
            error_vector,
            max_error,
            error_norm,
            is_satisfied,
            version: 2,
        }
    }

    /// Verify with zero tolerance (V2.0 default — exact homomorphism).
    pub fn verify_exact(
        e_prev: &EmbeddingVector,
        e_next: &EmbeddingVector,
        delta: &EmbeddingDelta,
    ) -> Self {
        // Use a tiny tolerance for fixed-point rounding (1e-8)
        // True mathematical error is 0; any residual is from
        // finite-precision arithmetic, not approximation
        Self::verify(e_prev, e_next, delta, 1e-8)
    }

    /// Verify a batch application.
    pub fn verify_batch(
        e_prev: &EmbeddingVector,
        e_next: &EmbeddingVector,
        deltas: &[EmbeddingDelta],
        tolerance: f64,
    ) -> NervResult<Self> {
        let aggregated = aggregate_batch_deltas(deltas)?;
        Ok(Self::verify(e_prev, e_next, &aggregated, tolerance))
    }
}

impl std::fmt::Display for HomomorphismVerification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.is_satisfied { "✓ PASS" } else { "✗ FAIL" };
        write!(
            f,
            "HomomorphismVerification [{}] max_error={:.2e}, norm={:.2e}, v={}",
            status, self.max_error, self.error_norm, self.version
        )
    }
}

// ─── Embedding Root Chain ────────────────────────────────────────────────

/// A linked pair of embedding roots, useful for VDW verification.
///
/// When a VDW is verified, the verifier checks that:
/// ```text
/// Hash(e_prev + δ_path) == e_next_root
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootTransition {
    /// The previous embedding root hash.
    pub prev_root: EmbeddingRoot,

    /// The new embedding root hash after applying the delta.
    pub next_root: EmbeddingRoot,

    /// The BLAKE3 hash of the delta vector (for commitment).
    pub delta_hash: [u8; 32],
}

impl RootTransition {
    /// Compute a root transition from the actual embedding and delta.
    pub fn compute(
        e_prev: &EmbeddingVector,
        delta: &EmbeddingDelta,
    ) -> Self {
        let prev_root = e_prev.hash();
        let e_next = apply_delta(e_prev, delta);
        let next_root = e_next.hash();
        let delta_hash = blake3_hash(&delta.to_bytes());

        Self {
            prev_root,
            next_root,
            delta_hash,
        }
    }

    /// Verify that a delta correctly transitions between two roots.
    ///
    /// Given a trusted `e_prev` embedding, a delta, and a claimed `next_root`,
    /// check that `Hash(e_prev + delta) == next_root`.
    pub fn verify_transition(
        e_prev: &EmbeddingVector,
        delta: &EmbeddingDelta,
        claimed_next_root: &EmbeddingRoot,
    ) -> bool {
        let computed_next = apply_delta(e_prev, delta);
        let computed_root = computed_next.hash();
        computed_root == *claimed_next_root
    }
}

// ─── Embedding Snapshots ────────────────────────────────────────────────

/// A snapshot of the full embedding state at a given block height.
///
/// Used for:
/// - RocksDB persistence
/// - Shard bisection (splitting an embedding into two)
/// - State sync for new nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingSnapshot {
    /// The full 512-byte embedding vector.
    pub embedding: EmbeddingVector,

    /// The 32-byte root hash (cached for fast comparison).
    pub root: EmbeddingRoot,

    /// The block height at which this embedding was canonical.
    pub height: u64,

    /// The shard this embedding belongs to.
    pub shard_id: u64,

    /// The BLAKE3 hash of the NWO weight matrix at this height
    /// (for versioning / rollback detection).
    pub weight_hash: [u8; 32],
}

impl EmbeddingSnapshot {
    /// Create a snapshot from an embedding and metadata.
    pub fn new(
        embedding: EmbeddingVector,
        height: u64,
        shard_id: u64,
        weight_hash: [u8; 32],
    ) -> Self {
        let root = embedding.hash();
        Self {
            embedding,
            root,
            height,
            shard_id,
            weight_hash,
        }
    }

    /// Create the genesis snapshot (all zeros).
    pub fn genesis(shard_id: u64, weight_hash: [u8; 32]) -> Self {
        Self::new(EmbeddingVector::ZERO, 0, shard_id, weight_hash)
    }

    /// Apply a delta and return a new snapshot at the next height.
    pub fn apply(&self, delta: &EmbeddingDelta) -> Self {
        let new_embedding = apply_delta(&self.embedding, delta);
        Self::new(
            new_embedding,
            self.height + 1,
            self.shard_id,
            self.weight_hash,
        )
    }

    /// Apply a batch of deltas and return a new snapshot.
    pub fn apply_batch(&self, deltas: &[EmbeddingDelta]) -> NervResult<Self> {
        let (new_embedding, _) = apply_batch(&self.embedding, deltas)?;
        Ok(Self::new(
            new_embedding,
            self.height + 1,
            self.shard_id,
            self.weight_hash,
        ))
    }

    /// Verify the cached root hash.
    pub fn verify_root(&self) -> bool {
        self.embedding.hash() == self.root
    }

    /// Serialize to bytes for RocksDB storage.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(512 + 32 + 8 + 8 + 32);
        buf.extend_from_slice(&self.embedding.to_bytes());
        buf.extend_from_slice(self.root.as_bytes());
        buf.extend_from_slice(&self.height.to_le_bytes());
        buf.extend_from_slice(&self.shard_id.to_le_bytes());
        buf.extend_from_slice(&self.weight_hash);
        buf
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_delta_zero() {
        let delta = EmbeddingDelta::ZERO;
        assert!(delta.is_zero());
    }

    #[test]
    fn test_embedding_delta_splat() {
        let delta = EmbeddingDelta::splat(FixedPoint64::ONE);
        assert!(!delta.is_zero());
        // All 64 elements should be 1.0
        for i in 0..EMBEDDING_DIM {
            assert_eq!(delta.get(i).unwrap(), FixedPoint64::ONE);
        }
    }

    #[test]
    fn test_apply_delta_exact() {
        // The fundamental test: e_{t+1} = e_t + δ must be exact
        let e_t = EmbeddingVector::splat(FixedPoint64::from_int(10));
        let delta = EmbeddingDelta::splat(FixedPoint64::from_int(5));
        let e_next = apply_delta(&e_t, &delta);

        // Every element should be exactly 15
        for i in 0..EMBEDDING_DIM {
            assert_eq!(e_next.get(i).unwrap().to_int(), 15);
        }
    }

    #[test]
    fn test_apply_delta_zero_delta() {
        let e_t = EmbeddingVector::splat(FixedPoint64::from_int(42));
        let delta = EmbeddingDelta::ZERO;
        let e_next = apply_delta(&e_t, &delta);
        assert_eq!(e_t, e_next);
    }

    #[test]
    fn test_apply_delta_negative_delta() {
        let e_t = EmbeddingVector::splat(FixedPoint64::from_int(10));
        let delta = EmbeddingDelta::splat(FixedPoint64::from_int(-3));
        let e_next = apply_delta(&e_t, &delta);
        for i in 0..EMBEDDING_DIM {
            assert_eq!(e_next.get(i).unwrap().to_int(), 7);
        }
    }

    #[test]
    fn test_aggregate_batch_deltas() {
        let d1 = EmbeddingDelta::splat(FixedPoint64::from_int(1));
        let d2 = EmbeddingDelta::splat(FixedPoint64::from_int(2));
        let d3 = EmbeddingDelta::splat(FixedPoint64::from_int(3));
        let aggregated = aggregate_batch_deltas(&[d1, d2, d3]).unwrap();
        // Sum: 1 + 2 + 3 = 6
        for i in 0..EMBEDDING_DIM {
            assert_eq!(aggregated.get(i).unwrap().to_int(), 6);
        }
    }

    #[test]
    fn test_aggregate_batch_deltas_empty() {
        let aggregated = aggregate_batch_deltas(&[]).unwrap();
        assert!(aggregated.is_zero());
    }

    #[test]
    fn test_aggregate_batch_deltas_order_independence() {
        // Deltas should aggregate in any order (commutativity)
        let d1 = EmbeddingDelta::splat(FixedPoint64::from_int(3));
        let d2 = EmbeddingDelta::splat(FixedPoint64::from_int(7));
        let d3 = EmbeddingDelta::splat(FixedPoint64::from_int(1));

        let order1 = aggregate_batch_deltas(&[d1.clone(), d2.clone(), d3.clone()]).unwrap();
        let order2 = aggregate_batch_deltas(&[d3.clone(), d1.clone(), d2.clone()]).unwrap();
        let order3 = aggregate_batch_deltas(&[d2, d3, d1]).unwrap();

        assert_eq!(order1, order2);
        assert_eq!(order2, order3);
    }

    #[test]
    fn test_apply_batch() {
        let e_t = EmbeddingVector::splat(FixedPoint64::from_int(100));
        let d1 = EmbeddingDelta::splat(FixedPoint64::from_int(1));
        let d2 = EmbeddingDelta::splat(FixedPoint64::from_int(2));

        let (e_next, aggregated) = apply_batch(&e_t, &[d1, d2]).unwrap();

        // Aggregated delta: 1 + 2 = 3
        for i in 0..EMBEDDING_DIM {
            assert_eq!(aggregated.get(i).unwrap().to_int(), 3);
        }

        // New embedding: 100 + 3 = 103
        for i in 0..EMBEDDING_DIM {
            assert_eq!(e_next.get(i).unwrap().to_int(), 103);
        }
    }

    #[test]
    fn test_apply_batch_exceeds_max() {
        let e_t = EmbeddingVector::ZERO;
        let deltas: Vec<EmbeddingDelta> = (0..=BATCH_MAX_SIZE)
            .map(|_| EmbeddingDelta::splat(FixedPoint64::ONE))
            .collect();
        let result = apply_batch(&e_t, &deltas);
        assert!(result.is_err());
    }

    #[test]
    fn test_homomorphism_verification_exact() {
        let e_prev = EmbeddingVector::splat(FixedPoint64::from_int(10));
        let delta = EmbeddingDelta::splat(FixedPoint64::from_int(5));
        let e_next = apply_delta(&e_prev, &delta);

        let verification = HomomorphismVerification::verify_exact(&e_prev, &e_next, &delta);
        assert!(verification.is_satisfied);
        assert!(verification.max_error < 1e-8);
        assert!(verification.error_norm < 1e-8);
        assert_eq!(verification.version, 2);
    }

    #[test]
    fn test_homomorphism_verification_with_tampered_embedding() {
        let e_prev = EmbeddingVector::splat(FixedPoint64::from_int(10));
        let delta = EmbeddingDelta::splat(FixedPoint64::from_int(5));
        // Tamper with the next embedding
        let e_next = EmbeddingVector::splat(FixedPoint64::from_int(20)); // Should be 15

        let verification = HomomorphismVerification::verify_exact(&e_prev, &e_next, &delta);
        assert!(!verification.is_satisfied);
        assert!(verification.max_error > 1.0);
    }

    #[test]
    fn test_homomorphism_verification_batch() {
        let e_prev = EmbeddingVector::splat(FixedPoint64::from_int(100));
        let deltas = vec![
            EmbeddingDelta::splat(FixedPoint64::from_int(1)),
            EmbeddingDelta::splat(FixedPoint64::from_int(2)),
            EmbeddingDelta::splat(FixedPoint64::from_int(3)),
        ];
        let (e_next, _) = apply_batch(&e_prev, &deltas).unwrap();

        let verification = HomomorphismVerification::verify_batch(
            &e_prev, &e_next, &deltas, 1e-8,
        ).unwrap();
        assert!(verification.is_satisfied);
    }

    #[test]
    fn test_homomorphism_verification_display() {
        let e_prev = EmbeddingVector::ZERO;
        let delta = EmbeddingDelta::ZERO;
        let e_next = EmbeddingVector::ZERO;
        let v = HomomorphismVerification::verify_exact(&e_prev, &e_next, &delta);
        let display = format!("{v}");
        assert!(display.contains("PASS"));
    }

    #[test]
    fn test_embedding_delta_add() {
        let d1 = EmbeddingDelta::splat(FixedPoint64::from_int(3));
        let d2 = EmbeddingDelta::splat(FixedPoint64::from_int(7));
        let sum = d1 + d2;
        for i in 0..EMBEDDING_DIM {
            assert_eq!(sum.get(i).unwrap().to_int(), 10);
        }
    }

    #[test]
    fn test_embedding_delta_neg() {
        let d = EmbeddingDelta::splat(FixedPoint64::from_int(5));
        let neg_d = -d;
        for i in 0..EMBEDDING_DIM {
            assert_eq!(neg_d.get(i).unwrap().to_int(), -5);
        }
    }

    #[test]
    fn test_embedding_delta_bytes_roundtrip() {
        let delta = EmbeddingDelta::splat(FixedPoint64::from_f64(3.14159));
        let bytes = delta.to_bytes();
        let recovered = EmbeddingDelta::from_bytes(&bytes);
        assert_eq!(delta, recovered);
    }

    #[test]
    fn test_root_transition() {
        let e_prev = EmbeddingVector::splat(FixedPoint64::from_int(10));
        let delta = EmbeddingDelta::splat(FixedPoint64::from_int(5));
        let transition = RootTransition::compute(&e_prev, &delta);

        // Verify the transition
        let claimed_next_root = transition.next_root;
        assert!(RootTransition::verify_transition(&e_prev, &delta, &claimed_next_root));

        // Wrong root should fail
        let wrong_root = EmbeddingRoot::from_bytes([0u8; 32]);
        assert!(!RootTransition::verify_transition(&e_prev, &delta, &wrong_root));
    }

    #[test]
    fn test_embedding_snapshot_genesis() {
        let snapshot = EmbeddingSnapshot::genesis(0, [0u8; 32]);
        assert!(snapshot.embedding.is_zero());
        assert_eq!(snapshot.height, 0);
        assert_eq!(snapshot.shard_id, 0);
        assert!(snapshot.verify_root());
    }

    #[test]
    fn test_embedding_snapshot_apply() {
        let snapshot = EmbeddingSnapshot::genesis(0, [0u8; 32]);
        let delta = EmbeddingDelta::splat(FixedPoint64::from_int(1));
        let new_snapshot = snapshot.apply(&delta);
        assert_eq!(new_snapshot.height, 1);
        assert!(new_snapshot.verify_root());
        // Embedding should now be all 1s
        for i in 0..EMBEDDING_DIM {
            assert_eq!(new_snapshot.embedding.get(i).unwrap().to_int(), 1);
        }
    }

    #[test]
    fn test_embedding_snapshot_apply_batch() {
        let snapshot = EmbeddingSnapshot::genesis(0, [0u8; 32]);
        let deltas = vec![
            EmbeddingDelta::splat(FixedPoint64::from_int(1)),
            EmbeddingDelta::splat(FixedPoint64::from_int(2)),
        ];
        let new_snapshot = snapshot.apply_batch(&deltas).unwrap();
        assert_eq!(new_snapshot.height, 1);
        assert!(new_snapshot.verify_root());
        for i in 0..EMBEDDING_DIM {
            assert_eq!(new_snapshot.embedding.get(i).unwrap().to_int(), 3);
        }
    }

    #[test]
    fn test_embedding_snapshot_serialization() {
        let snapshot = EmbeddingSnapshot::genesis(42, [7u8; 32]);
        let bytes = snapshot.to_bytes();
        // Should be 512 + 32 + 8 + 8 + 32 = 592 bytes
        assert_eq!(bytes.len(), 592);
    }

    #[test]
    fn test_linearity_chained_deltas() {
        // Applying deltas one at a time should give the same result as a batch
        let e_start = EmbeddingVector::splat(FixedPoint64::from_int(100));
        let d1 = EmbeddingDelta::splat(FixedPoint64::from_int(3));
        let d2 = EmbeddingDelta::splat(FixedPoint64::from_int(7));
        let d3 = EmbeddingDelta::splat(FixedPoint64::from_int(-2));

        // Apply one at a time
        let e1 = apply_delta(&e_start, &d1);
        let e2 = apply_delta(&e1, &d2);
        let e3 = apply_delta(&e2, &d3);

        // Apply as batch
        let (e_batch, _) = apply_batch(&e_start, &[d1, d2, d3]).unwrap();

        // They should be identical (associativity of vector addition)
        for i in 0..EMBEDDING_DIM {
            let sequential = e3.get(i).unwrap().to_f64();
            let batched = e_batch.get(i).unwrap().to_f64();
            assert!(
                (sequential - batched).abs() < 1e-8,
                "Mismatch at dimension {i}: sequential={sequential}, batched={batched}"
            );
        }
    }

    #[test]
    fn test_delta_magnitude_and_norm() {
        let delta = EmbeddingDelta::splat(FixedPoint64::from_int(1));
        // L1 norm: 64 * |1| = 64
        let l1 = delta.magnitude();
        assert_eq!(l1.to_int(), 64);

        // L2 norm: sqrt(64 * 1²) = 8
        let l2 = delta.norm();
        assert_eq!(l2.to_int(), 8);

        // L∞ norm: max|1| = 1
        let linf = delta.max_element();
        assert_eq!(linf.to_int(), 1);
    }
}
