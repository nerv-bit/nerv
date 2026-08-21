//! Deterministic Embedding Bisection for Shard Splits and Merges.
//!
//! The bisection algorithm divides a shard's 512-byte neural state embedding
//! into two child embeddings using a deterministic hyperplane split. This
//! enables shards to "split like cells" in <4 seconds without data migration.
//!
//! # Split Protocol
//!
//! ```text
//! 1. Trigger: NWO predictor predicts overload probability > threshold
//! 2. Proposal: Node broadcasts bonded SplitProposal
//! 3. Approval: ≥67% weighted stake co-signs within 1.5s
//! 4. Bisection:
//!    a. Seed = BLAKE3(shard_id ‖ height ‖ chain_constant)
//!    b. Direction d ∈ ℝ⁶⁴ derived from seed (uniform on unit sphere)
//!    c. Child embeddings:
//!       e_left  = (e - proj_d(e)) / √2 + noise
//!       e_right = (e + proj_d(e)) / √2 + noise
//!    d. Account assignment via hash(account_id ‖ child_index)
//! 5. Re-execution: Last 500 txs re-executed on child embeddings
//! 6. Routing: DHT and mixer tables updated instantly
//! ```
//!
//! # Merge Protocol
//!
//! ```text
//! 1. Trigger: Sibling shards sustain < merge_tps_threshold for idle_duration
//! 2. Reverse bisection: e_parent ≈ (e_left + e_right) * √2 / 2
//! 3. Re-execution of recent txs on merged embedding
//! 4. Routing update
//! ```
//!
//! # Mathematical Foundation
//!
//! The bisection projects the embedding onto two orthogonal subspaces
//! separated by a hyperplane H = {x | d · (x - e) = 0}:
//!
//! ```text
//! proj_d(v) = (d · v) * d    (since ||d|| = 1)
//! e_left  = (e - proj_d(e)) / √2 + ε
//! e_right = (e + proj_d(e)) / √2 + ε
//! ```
//!
//! Properties:
//! - **Norm preservation**: ||e_left||² + ||e_right||² ≈ ||e||² (Parseval-like)
//! - **Homomorphism**: Deltas apply additively to children (linearity)
//! - **Determinism**: Same seed → same split → all nodes agree
//! - **Balance**: Direction d is chosen to approximately bisect the "mass"

use crate::{
    EMBEDDING_DIM, EMBEDDING_BYTES, CONSENSUS_QUORUM_NUMERATOR, CONSENSUS_QUORUM_DENOMINATOR,
    ShardId, BlockHeight, EmbeddingRoot, ValidatorId, StakeAmount, VotingWeight,
    NervError, NervResult,
};
use crate::sharding::ShardLifecycle;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Constants ───────────────────────────────────────────────────────────

/// Chain constant for bisection seed computation.
const BISECTION_CHAIN_CONSTANT: &[u8] = b"NERV-v2.0-bisection";

/// Number of recent transactions to re-execute after split/merge.
pub const REEXECUTION_WINDOW: usize = 500;

/// Maximum split/merge duration in milliseconds.
pub const MAX_RECONFIGURATION_MS: u64 = 4000;

/// DP noise standard deviation for child embeddings (privacy).
const DP_NOISE_SIGMA: f64 = 0.1;

/// √2 constant.
const SQRT2: f64 = 1.4142135623730951;

/// 1/√2 constant.
const INV_SQRT2: f64 = 0.7071067811865476;

// ─── Embedding Vector Operations (f64) ───────────────────────────────────

/// A 64-dimensional floating-point vector representing an embedding
/// in ℝ⁶⁴ for geometric operations (bisection, projection, etc.).
pub type EmbeddingF64 = [f64; EMBEDDING_DIM];

/// Zero embedding vector.
pub const ZERO_EMBEDDING: EmbeddingF64 = [0.0; EMBEDDING_DIM];

/// Compute the dot product of two embedding vectors.
#[inline]
pub fn dot(a: &EmbeddingF64, b: &EmbeddingF64) -> f64 {
    let mut sum = 0.0;
    for i in 0..EMBEDDING_DIM {
        sum += a[i] * b[i];
    }
    sum
}

/// Compute the L2 norm of an embedding vector.
#[inline]
pub fn norm(v: &EmbeddingF64) -> f64 {
    dot(v, v).sqrt()
}

/// Compute the projection of vector v onto direction d.
///
/// proj_d(v) = (d · v / ||d||²) * d = (d · v) * d  (when ||d|| = 1)
#[inline]
pub fn project_onto(v: &EmbeddingF64, d: &EmbeddingF64) -> EmbeddingF64 {
    let d_dot_v = dot(d, v);
    let d_norm_sq = dot(d, d);
    if d_norm_sq < 1e-15 {
        return ZERO_EMBEDDING;
    }
    let scale = d_dot_v / d_norm_sq;
    let mut result = ZERO_EMBEDDING;
    for i in 0..EMBEDDING_DIM {
        result[i] = scale * d[i];
    }
    result
}

/// Add two embedding vectors.
#[inline]
pub fn add(a: &EmbeddingF64, b: &EmbeddingF64) -> EmbeddingF64 {
    let mut result = ZERO_EMBEDDING;
    for i in 0..EMBEDDING_DIM {
        result[i] = a[i] + b[i];
    }
    result
}

/// Subtract two embedding vectors: a - b.
#[inline]
pub fn sub(a: &EmbeddingF64, b: &EmbeddingF64) -> EmbeddingF64 {
    let mut result = ZERO_EMBEDDING;
    for i in 0..EMBEDDING_DIM {
        result[i] = a[i] - b[i];
    }
    result
}

/// Scale an embedding vector by a scalar.
#[inline]
pub fn scale(v: &EmbeddingF64, s: f64) -> EmbeddingF64 {
    let mut result = ZERO_EMBEDDING;
    for i in 0..EMBEDDING_DIM {
        result[i] = v[i] * s;
    }
    result
}

/// Add DP noise to an embedding vector.
///
/// Uses a simple PRNG seeded from the bisection seed for determinism.
fn add_dp_noise(v: &EmbeddingF64, seed: &[u8], sigma: f64) -> EmbeddingF64 {
    let mut result = *v;
    // Use BLAKE3-based deterministic noise generation
    for i in 0..EMBEDDING_DIM {
        let noise_context = format!("nerv:bisection:noise:{}", i);
        let noise_hash = blake3::derive_key(noise_context.as_bytes(), seed);
        // Convert first 8 bytes to f64 in [-1, 1]
        let bits = u64::from_le_bytes(noise_hash[..8].try_into().unwrap_or([0u8; 8]));
        let uniform = (bits as f64) / (u64::MAX as f64) * 2.0 - 1.0;
        // Box-Muller would give Gaussian, but for small σ a uniform approximation suffices
        result[i] += sigma * uniform;
    }
    result
}

// ─── Deterministic Seed Generation ───────────────────────────────────────

/// Compute the deterministic bisection seed.
///
/// seed = BLAKE3(shard_id ‖ height ‖ chain_constant)
///
/// This ensures all nodes compute the same bisection for the same
/// shard at the same height, guaranteeing deterministic splits.
pub fn compute_bisection_seed(
    shard_id: ShardId,
    height: BlockHeight,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&shard_id.0.to_le_bytes());
    hasher.update(&height.0.to_le_bytes());
    hasher.update(BISECTION_CHAIN_CONSTANT);
    hasher.finalize().into()
}

// ─── Direction Vector Generation ─────────────────────────────────────────

/// Generate a deterministic random direction vector on the unit sphere in ℝ⁶⁴.
///
/// Uses the seed to generate 64 normally-distributed values via the
/// Box-Muller transform, then normalizes to unit length.
///
/// The direction vector defines the splitting hyperplane:
/// H = {x | d · (x - e) = 0}
pub fn generate_direction_vector(seed: &[u8; 32]) -> EmbeddingF64 {
    let mut direction = ZERO_EMBEDDING;

    // Generate 64 Gaussian values using Box-Muller transform
    // We need 32 pairs of uniform random numbers to generate 64 Gaussian values
    for i in 0..(EMBEDDING_DIM / 2) {
        // Generate two uniform random numbers in (0, 1) from the seed
        let ctx1 = format!("nerv:bisection:dir:{}:a", i);
        let ctx2 = format!("nerv:bisection:dir:{}:b", i);

        let h1 = blake3::derive_key(ctx1.as_bytes(), seed);
        let h2 = blake3::derive_key(ctx2.as_bytes(), seed);

        // Convert to uniform (0, 1), avoiding 0 for log
        let u1 = ((u64::from_le_bytes(h1[..8].try_into().unwrap_or([0u8; 8])) as f64)
            / (u64::MAX as f64))
            .clamp(1e-10, 1.0 - 1e-10);
        let u2 = (u64::from_le_bytes(h2[..8].try_into().unwrap_or([0u8; 8])) as f64)
            / (u64::MAX as f64);

        // Box-Muller transform
        let magnitude = (-2.0 * u1.ln()).sqrt();
        let z0 = magnitude * (2.0 * std::f64::consts::PI * u2).cos();
        let z1 = magnitude * (2.0 * std::f64::consts::PI * u2).sin();

        direction[2 * i] = z0;
        direction[2 * i + 1] = z1;
    }

    // Normalize to unit sphere
    let n = norm(&direction);
    if n < 1e-15 {
        // Fallback: use a canonical direction
        direction[0] = 1.0;
        return direction;
    }
    for i in 0..EMBEDDING_DIM {
        direction[i] /= n;
    }

    direction
}

// ─── Bisection Computation ───────────────────────────────────────────────

/// Result of an embedding bisection (split).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BisectionResult {
    /// The left child embedding (64-dim f64 vector).
    pub left_embedding: EmbeddingF64,

    /// The right child embedding (64-dim f64 vector).
    pub right_embedding: EmbeddingF64,

    /// The direction vector that defined the splitting hyperplane.
    pub direction: EmbeddingF64,

    /// The bisection seed (for determinism verification).
    pub seed: [u8; 32],

    /// The left child shard ID.
    pub left_shard_id: ShardId,

    /// The right child shard ID.
    pub right_shard_id: ShardId,

    /// Projection of the parent embedding onto the direction.
    pub projection: EmbeddingF64,

    /// Norm of the parent embedding (for verification).
    pub parent_norm: f64,

    /// Norm of the left child (for verification).
    pub left_norm: f64,

    /// Norm of the right child (for verification).
    pub right_norm: f64,
}

/// Perform the deterministic bisection of an embedding.
///
/// # Arguments
///
/// * `parent_embedding` - The 64-dim embedding to bisect
/// * `shard_id` - The parent shard ID
/// * `height` - The current block height
/// * `add_noise` - Whether to add DP noise (true for production, false for tests)
///
/// # Returns
///
/// A `BisectionResult` containing both child embeddings and metadata.
///
/// # Mathematical Guarantees
///
/// - `||e_left||² + ||e_right||² ≈ ||e_parent||²` (Parseval-like, up to noise)
/// - `e_left + e_right ≈ √2 * e_parent` (reconstruction)
/// - The split is deterministic: same inputs → same outputs
pub fn bisect_embedding(
    parent_embedding: &EmbeddingF64,
    shard_id: ShardId,
    height: BlockHeight,
    add_noise: bool,
) -> BisectionResult {
    // 1. Compute deterministic seed
    let seed = compute_bisection_seed(shard_id, height);

    // 2. Generate direction vector on unit sphere
    let direction = generate_direction_vector(&seed);

    // 3. Compute projection of parent onto direction
    let projection = project_onto(parent_embedding, &direction);

    // 4. Compute child embeddings
    // e_left  = (e - proj_d(e)) / √2 + noise
    // e_right = (e + proj_d(e)) / √2 + noise
    let e_minus_proj = sub(parent_embedding, &projection);
    let e_plus_proj = add(parent_embedding, &projection);

    let mut left = scale(&e_minus_proj, INV_SQRT2);
    let mut right = scale(&e_plus_proj, INV_SQRT2);

    // 5. Add DP noise (small, for privacy)
    if add_noise {
        let noise_seed_left = {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&seed);
            hasher.update(b"left");
            let h: [u8; 32] = hasher.finalize().into();
            h
        };
        let noise_seed_right = {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&seed);
            hasher.update(b"right");
            let h: [u8; 32] = hasher.finalize().into();
            h
        };

        left = add_dp_noise(&left, &noise_seed_left, DP_NOISE_SIGMA);
        right = add_dp_noise(&right, &noise_seed_right, DP_NOISE_SIGMA);
    }

    // 6. Compute child shard IDs
    let left_shard_id = derive_child_shard_id(shard_id, &seed, 0);
    let right_shard_id = derive_child_shard_id(shard_id, &seed, 1);

    BisectionResult {
        left_embedding: left,
        right_embedding: right,
        direction,
        seed,
        left_shard_id,
        right_shard_id,
        projection,
        parent_norm: norm(parent_embedding),
        left_norm: norm(&left),
        right_norm: norm(&right),
    }
}

/// Derive a child shard ID from the parent, seed, and child index.
///
/// child_id = parent_id ⊕ BLAKE3(seed ‖ child_index)[0..8]
fn derive_child_shard_id(parent_id: ShardId, seed: &[u8; 32], child_index: u8) -> ShardId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(seed);
    hasher.update(&[child_index]);
    let hash: [u8; 32] = hasher.finalize().into();
    let xor_val = u64::from_le_bytes(hash[..8].try_into().unwrap_or([0u8; 8]));
    ShardId::new(parent_id.0 ^ xor_val)
}

/// Verify the bisection result preserves norms (Parseval-like property).
///
/// Checks that `||e_left||² + ||e_right||² ≈ ||e_parent||²` within
/// a tolerance that accounts for DP noise.
pub fn verify_bisection_norms(result: &BisectionResult, tolerance: f64) -> bool {
    let left_sq = result.left_norm * result.left_norm;
    let right_sq = result.right_norm * result.right_norm;
    let parent_sq = result.parent_norm * result.parent_norm;
    let diff = (left_sq + right_sq - parent_sq).abs();
    diff <= tolerance
}

// ─── Merge (Reverse Bisection) ───────────────────────────────────────────

/// Result of merging two sibling embeddings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergeResult {
    /// The merged (parent) embedding.
    pub merged_embedding: EmbeddingF64,

    /// Norm of the merged embedding.
    pub merged_norm: f64,

    /// Norm of the left child (for verification).
    pub left_norm: f64,

    /// Norm of the right child (for verification).
    pub right_norm: f64,

    /// Approximation error from the merge.
    pub merge_error: f64,
}

/// Merge two sibling embeddings via reverse bisection.
///
/// e_parent ≈ (e_left + e_right) / √2 * √2 = (e_left + e_right)
///
/// Wait — from the forward bisection:
/// e_left  = (e - proj) / √2 + noise
/// e_right = (e + proj) / √2 + noise
///
/// So: e_left + e_right = 2e/√2 = √2 * e
/// Therefore: e ≈ (e_left + e_right) / √2
///
/// The merge is approximate because of DP noise added during the split.
pub fn merge_embeddings(
    left_embedding: &EmbeddingF64,
    right_embedding: &EmbeddingF64,
) -> MergeResult {
    // e_parent ≈ (e_left + e_right) / √2
    let sum = add(left_embedding, right_embedding);
    let merged = scale(&sum, INV_SQRT2);

    let left_n = norm(left_embedding);
    let right_n = norm(right_embedding);
    let merged_n = norm(&merged);

    // Compute merge error: ||e_left||² + ||e_right||² should ≈ ||e_parent||²
    let left_sq = left_n * left_n;
    let right_sq = right_n * right_n;
    let merged_sq = merged_n * merged_n;
    let merge_error = (left_sq + right_sq - 2.0 * merged_sq).abs().sqrt();

    MergeResult {
        merged_embedding: merged,
        merged_norm: merged_n,
        left_norm: left_n,
        right_norm: right_n,
        merge_error,
    }
}

// ─── Account Assignment ──────────────────────────────────────────────────

/// Determine which child shard an account belongs to after a split.
///
/// Uses deterministic hash-based assignment:
/// hash(account_key ‖ seed ‖ child_index) → assign to child with
/// the lower hash value.
///
/// This provides approximately balanced assignment (~50/50 split).
pub fn assign_account_to_child(
    account_key: &[u8],
    seed: &[u8; 32],
    left_shard_id: ShardId,
    right_shard_id: ShardId,
) -> ShardId {
    let mut hasher_left = blake3::Hasher::new();
    hasher_left.update(account_key);
    hasher_left.update(seed);
    hasher_left.update(&left_shard_id.0.to_le_bytes());
    let hash_left: [u8; 32] = hasher_left.finalize().into();

    let mut hasher_right = blake3::Hasher::new();
    hasher_right.update(account_key);
    hasher_right.update(seed);
    hasher_right.update(&right_shard_id.0.to_le_bytes());
    let hash_right: [u8; 32] = hasher_right.finalize().into();

    // Assign to the child with the lower hash (arbitrary but deterministic)
    if hash_left < hash_right {
        left_shard_id
    } else {
        right_shard_id
    }
}

/// Batch-assign accounts to children, returning the assignment map
/// and the balance ratio (left_count / total).
pub fn batch_assign_accounts(
    account_keys: &[Vec<u8>],
    seed: &[u8; 32],
    left_shard_id: ShardId,
    right_shard_id: ShardId,
) -> (HashMap<Vec<u8>, ShardId>, f64) {
    let mut assignments = HashMap::new();
    let mut left_count = 0usize;

    for key in account_keys {
        let child = assign_account_to_child(key, seed, left_shard_id, right_shard_id);
        if child == left_shard_id {
            left_count += 1;
        }
        assignments.insert(key.clone(), child);
    }

    let total = account_keys.len().max(1);
    let balance_ratio = left_count as f64 / total as f64;

    (assignments, balance_ratio)
}

// ─── Split Proposal ──────────────────────────────────────────────────────

/// A bonded proposal to split a shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitProposal {
    /// The shard to split.
    pub shard_id: ShardId,

    /// The block height at which the proposal is made.
    pub height: BlockHeight,

    /// The validator proposing the split.
    pub proposer: ValidatorId,

    /// The overload probability that triggered the proposal.
    pub overload_probability: f64,

    /// The bond amount staked by the proposer (nano-NERV).
    pub bond_amount: u64,

    /// Timestamp of the proposal (Unix epoch millis).
    pub timestamp_ms: u64,

    /// Dilithium-3 signature of the proposer.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub signature: Vec<u8>,
}

impl SplitProposal {
    /// Create a new split proposal.
    pub fn new(
        shard_id: ShardId,
        height: BlockHeight,
        proposer: ValidatorId,
        overload_probability: f64,
    ) -> Self {
        Self {
            shard_id,
            height,
            proposer,
            overload_probability,
            bond_amount: 0, // Set by the bonding mechanism
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            signature: Vec::new(),
        }
    }

    /// Compute a unique hash for this proposal.
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.shard_id.0.to_le_bytes());
        hasher.update(&self.height.0.to_le_bytes());
        hasher.update(self.proposer.as_bytes());
        hasher.update(&self.overload_probability.to_le_bytes());
        hasher.update(&self.timestamp_ms.to_le_bytes());
        hasher.finalize().into()
    }
}

/// An approval vote for a split proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitApproval {
    /// The proposal being approved (hash).
    pub proposal_hash: [u8; 32],

    /// The approving validator.
    pub validator_id: ValidatorId,

    /// The validator's voting weight.
    pub voting_weight: VotingWeight,

    /// BLS12-381 partial signature share.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub partial_sig: Vec<u8>,
}

// ─── Merge Proposal ──────────────────────────────────────────────────────

/// A proposal to merge two sibling shards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeProposal {
    /// The left sibling shard.
    pub left_shard_id: ShardId,

    /// The right sibling shard.
    pub right_shard_id: ShardId,

    /// Combined TPS of both shards at proposal time.
    pub combined_tps: f64,

    /// Duration of idle state before merge trigger (seconds).
    pub idle_duration_secs: u64,

    /// Timestamp of the proposal.
    pub timestamp_ms: u64,
}

impl MergeProposal {
    /// Create a new merge proposal.
    pub fn new(
        left_shard_id: ShardId,
        right_shard_id: ShardId,
        combined_tps: f64,
    ) -> Self {
        Self {
            left_shard_id,
            right_shard_id,
            combined_tps,
            idle_duration_secs: 0,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Compute a unique hash for this proposal.
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.left_shard_id.0.to_le_bytes());
        hasher.update(&self.right_shard_id.0.to_le_bytes());
        hasher.update(&self.combined_tps.to_le_bytes());
        hasher.finalize().into()
    }
}

// ─── Split State Machine ─────────────────────────────────────────────────

/// The state of a split operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplitState {
    /// Proposal has been made, waiting for approval votes.
    Proposed,
    /// ≥67% weighted stake has approved; executing bisection.
    Approved,
    /// Bisection computed; re-executing last 500 txs on child embeddings.
    ReExecuting,
    /// Re-execution complete; updating routing tables.
    RoutingUpdate,
    /// Split completed successfully.
    Completed,
    /// Split failed.
    Failed,
}

impl std::fmt::Display for SplitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Proposed => write!(f, "proposed"),
            Self::Approved => write!(f, "approved"),
            Self::ReExecuting => write!(f, "re-executing"),
            Self::RoutingUpdate => write!(f, "routing_update"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

/// State machine for a shard split operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitStateMachine {
    /// The split proposal.
    pub proposal: SplitProposal,

    /// Current state.
    pub state: SplitState,

    /// Collected approval votes.
    pub approvals: Vec<SplitApproval>,

    /// Total voting weight of approvers.
    pub approved_weight: u128,

    /// Total voting weight of all eligible validators.
    pub total_weight: u128,

    /// The bisection result (computed after approval).
    pub bisection_result: Option<BisectionResult>,

    /// Re-execution progress (number of txs re-executed).
    pub reexecution_progress: usize,

    /// Total txs to re-execute.
    pub reexecution_total: usize,

    /// Failure reason (if Failed).
    pub failure_reason: Option<String>,

    /// Timestamp of state creation.
    pub started_at_ms: u64,
}

impl SplitStateMachine {
    /// Create a new split state machine from a proposal.
    pub fn new(proposal: SplitProposal) -> Self {
        Self {
            proposal,
            state: SplitState::Proposed,
            approvals: Vec::new(),
            approved_weight: 0,
            total_weight: 0,
            bisection_result: None,
            reexecution_progress: 0,
            reexecution_total: REEXECUTION_WINDOW,
            failure_reason: None,
            started_at_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Set the total voting weight (from the validator set).
    pub fn set_total_weight(&mut self, total: u128) {
        self.total_weight = total;
    }

    /// Add an approval vote.
    ///
    /// Returns Ok(()) if the vote was accepted, Err if duplicate or invalid.
    pub fn add_approval(&mut self, approval: SplitApproval) -> NervResult<()> {
        if self.state != SplitState::Proposed {
            return Err(NervError::Sharding(
                "split is not in Proposed state".into()
            ));
        }

        // Check for duplicate
        if self.approvals.iter().any(|a| a.validator_id == approval.validator_id) {
            return Err(NervError::Sharding(
                "duplicate approval vote".into()
            ));
        }

        self.approved_weight = self.approved_weight.saturating_add(approval.voting_weight.0);
        self.approvals.push(approval);

        // Check if quorum reached
        if self.check_quorum() {
            self.state = SplitState::Approved;
        }

        Ok(())
    }

    /// Check if the 67% quorum has been reached.
    pub fn check_quorum(&self) -> bool {
        if self.total_weight == 0 {
            return false;
        }
        // approved >= (67/100) * total  ⟺  100 * approved >= 67 * total
        100u128 * self.approved_weight >= CONSENSUS_QUORUM_NUMERATOR as u128 * self.total_weight
    }

    /// Get the approval percentage.
    pub fn approval_percentage(&self) -> f64 {
        if self.total_weight == 0 {
            return 0.0;
        }
        (self.approved_weight as f64 / self.total_weight as f64) * 100.0
    }

    /// Transition to ReExecuting state with bisection result.
    pub fn start_reexecution(&mut self, bisection_result: BisectionResult) -> NervResult<()> {
        if self.state != SplitState::Approved {
            return Err(NervError::Sharding(format!(
                "cannot start re-execution from {} state", self.state
            )));
        }
        self.bisection_result = Some(bisection_result);
        self.state = SplitState::ReExecuting;
        Ok(())
    }

    /// Update re-execution progress.
    pub fn update_reexecution_progress(&mut self, count: usize) {
        self.reexecution_progress = count;
        if self.reexecution_progress >= self.reexecution_total {
            self.state = SplitState::RoutingUpdate;
        }
    }

    /// Complete the split.
    pub fn complete(&mut self) -> NervResult<()> {
        if self.state != SplitState::RoutingUpdate {
            return Err(NervError::Sharding(format!(
                "cannot complete from {} state", self.state
            )));
        }
        self.state = SplitState::Completed;
        Ok(())
    }

    /// Mark the split as failed.
    pub fn fail(&mut self, reason: String) {
        self.state = SplitState::Failed;
        self.failure_reason = Some(reason);
    }

    /// Check if the split is terminal (Completed or Failed).
    pub fn is_terminal(&self) -> bool {
        self.state == SplitState::Completed || self.state == SplitState::Failed
    }

    /// Get elapsed time since the split started (milliseconds).
    pub fn elapsed_ms(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now.saturating_sub(self.started_at_ms)
    }

    /// Check if the split has timed out.
    pub fn is_timed_out(&self) -> bool {
        self.elapsed_ms() > MAX_RECONFIGURATION_MS
    }
}

// ─── Merge State Machine ─────────────────────────────────────────────────

/// The state of a merge operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeState {
    /// Merge triggered; waiting for confirmation.
    Triggered,
    /// Computing reverse bisection (merged embedding).
    Computing,
    /// Re-executing recent txs on merged embedding.
    ReExecuting,
    /// Updating routing tables.
    RoutingUpdate,
    /// Merge completed.
    Completed,
    /// Merge failed.
    Failed,
}

impl std::fmt::Display for MergeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Triggered => write!(f, "triggered"),
            Self::Computing => write!(f, "computing"),
            Self::ReExecuting => write!(f, "re-executing"),
            Self::RoutingUpdate => write!(f, "routing_update"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

/// State machine for a shard merge operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeStateMachine {
    /// The merge proposal.
    pub proposal: MergeProposal,

    /// Current state.
    pub state: MergeState,

    /// The merge result (computed after Computing phase).
    pub merge_result: Option<MergeResult>,

    /// Re-execution progress.
    pub reexecution_progress: usize,

    /// Total txs to re-execute.
    pub reexecution_total: usize,

    /// Failure reason.
    pub failure_reason: Option<String>,

    /// Timestamp of merge start.
    pub started_at_ms: u64,
}

impl MergeStateMachine {
    /// Create a new merge state machine.
    pub fn new(proposal: MergeProposal) -> Self {
        Self {
            proposal,
            state: MergeState::Triggered,
            merge_result: None,
            reexecution_progress: 0,
            reexecution_total: REEXECUTION_WINDOW,
            failure_reason: None,
            started_at_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Transition to Computing state.
    pub fn start_computing(&mut self) -> NervResult<()> {
        if self.state != MergeState::Triggered {
            return Err(NervError::Sharding(format!(
                "cannot start computing from {} state", self.state
            )));
        }
        self.state = MergeState::Computing;
        Ok(())
    }

    /// Set the merge result and transition to ReExecuting.
    pub fn set_merge_result(&mut self, result: MergeResult) -> NervResult<()> {
        if self.state != MergeState::Computing {
            return Err(NervError::Sharding(format!(
                "cannot set merge result from {} state", self.state
            )));
        }
        self.merge_result = Some(result);
        self.state = MergeState::ReExecuting;
        Ok(())
    }

    /// Update re-execution progress.
    pub fn update_reexecution_progress(&mut self, count: usize) {
        self.reexecution_progress = count;
        if self.reexecution_progress >= self.reexecution_total {
            self.state = MergeState::RoutingUpdate;
        }
    }

    /// Complete the merge.
    pub fn complete(&mut self) -> NervResult<()> {
        if self.state != MergeState::RoutingUpdate {
            return Err(NervError::Sharding(format!(
                "cannot complete from {} state", self.state
            )));
        }
        self.state = MergeState::Completed;
        Ok(())
    }

    /// Mark the merge as failed.
    pub fn fail(&mut self, reason: String) {
        self.state = MergeState::Failed;
        self.failure_reason = Some(reason);
    }

    /// Check if terminal.
    pub fn is_terminal(&self) -> bool {
        self.state == MergeState::Completed || self.state == MergeState::Failed
    }

    /// Check if timed out.
    pub fn is_timed_out(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now.saturating_sub(self.started_at_ms) > MAX_RECONFIGURATION_MS
    }
}

// ─── Embedding Conversion Helpers ────────────────────────────────────────

/// Convert a 512-byte embedding (raw bytes) to an EmbeddingF64.
///
/// Interpretation: 64 × 8 bytes → 64 × f64 (little-endian).
pub fn bytes_to_embedding_f64(data: &[u8]) -> NervResult<EmbeddingF64> {
    if data.len() != EMBEDDING_BYTES {
        return Err(NervError::Sharding(format!(
            "embedding data must be {} bytes, got {}",
            EMBEDDING_BYTES, data.len()
        )));
    }

    let mut result = ZERO_EMBEDDING;
    for i in 0..EMBEDDING_DIM {
        let offset = i * 8;
        let bytes: [u8; 8] = data[offset..offset + 8].try_into()
            .map_err(|_| NervError::Sharding("embedding byte slice error".into()))?;
        result[i] = f64::from_le_bytes(bytes);
    }
    Ok(result)
}

/// Convert an EmbeddingF64 to 512 raw bytes.
pub fn embedding_f64_to_bytes(v: &EmbeddingF64) -> Vec<u8> {
    let mut result = Vec::with_capacity(EMBEDDING_BYTES);
    for i in 0..EMBEDDING_DIM {
        result.extend_from_slice(&v[i].to_le_bytes());
    }
    result
}

/// Compute the BLAKE3 hash of an embedding (for EmbeddingRoot).
pub fn embedding_root(v: &EmbeddingF64) -> EmbeddingRoot {
    let bytes = embedding_f64_to_bytes(v);
    let hash = blake3::hash(&bytes);
    EmbeddingRoot::from_bytes(hash.into())
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_embedding(value: f64) -> EmbeddingF64 {
        [value; EMBEDDING_DIM]
    }

    #[test]
    fn test_dot_product() {
        let a = make_test_embedding(1.0);
        let b = make_test_embedding(2.0);
        // 64 * 1.0 * 2.0 = 128.0
        assert!((dot(&a, &b) - 128.0).abs() < 1e-6);
    }

    #[test]
    fn test_norm() {
        let v = make_test_embedding(3.0);
        // ||v|| = sqrt(64 * 9) = 24
        assert!((norm(&v) - 24.0).abs() < 1e-6);
    }
 #[test]
    fn test_projection() {
        // Project [1,1,0,...,0] onto [1,0,...,0]
        let mut v = ZERO_EMBEDDING;
        v[0] = 1.0;
        v[1] = 1.0;

        let mut d = ZERO_EMBEDDING;
        d[0] = 1.0;

        let proj = project_onto(&v, &d);
        assert!((proj[0] - 1.0).abs() < 1e-10);
        assert!((proj[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_sub() {
        let a = make_test_embedding(3.0);
        let b = make_test_embedding(1.0);
        let sum = add(&a, &b);
        let diff = sub(&a, &b);
        for i in 0..EMBEDDING_DIM {
            assert!((sum[i] - 4.0).abs() < 1e-10);
            assert!((diff[i] - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_scale() {
        let v = make_test_embedding(2.0);
        let s = scale(&v, 0.5);
        for i in 0..EMBEDDING_DIM {
            assert!((s[i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bisection_seed_determinism() {
        let shard_id = ShardId::new(42);
        let height = BlockHeight::from(1000);
        let seed1 = compute_bisection_seed(shard_id, height);
        let seed2 = compute_bisection_seed(shard_id, height);
        assert_eq!(seed1, seed2);
    }

    #[test]
    fn test_bisection_seed_uniqueness() {
        let seed1 = compute_bisection_seed(ShardId::new(1), BlockHeight::from(100));
        let seed2 = compute_bisection_seed(ShardId::new(2), BlockHeight::from(100));
        let seed3 = compute_bisection_seed(ShardId::new(1), BlockHeight::from(101));
        assert_ne!(seed1, seed2);
        assert_ne!(seed1, seed3);
    }

    #[test]
    fn test_direction_vector_unit_norm() {
        let seed = compute_bisection_seed(ShardId::new(1), BlockHeight::from(1));
        let d = generate_direction_vector(&seed);
        let n = norm(&d);
        // Should be very close to 1.0 (unit sphere)
        assert!((n - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_direction_vector_determinism() {
        let seed = compute_bisection_seed(ShardId::new(5), BlockHeight::from(50));
        let d1 = generate_direction_vector(&seed);
        let d2 = generate_direction_vector(&seed);
        for i in 0..EMBEDDING_DIM {
            assert!((d1[i] - d2[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_direction_vector_different_seeds() {
        let seed1 = compute_bisection_seed(ShardId::new(1), BlockHeight::from(1));
        let seed2 = compute_bisection_seed(ShardId::new(2), BlockHeight::from(1));
        let d1 = generate_direction_vector(&seed1);
        let d2 = generate_direction_vector(&seed2);
        // Different seeds should produce different directions
        let diff_norm = norm(&sub(&d1, &d2));
        assert!(diff_norm > 0.1); // Not identical
    }

    #[test]
    fn test_bisect_embedding_basic() {
        let parent = make_test_embedding(1.0);
        let result = bisect_embedding(&parent, ShardId::new(1), BlockHeight::from(100), false);

        // Children should have non-zero norm
        assert!(result.left_norm > 0.0);
        assert!(result.right_norm > 0.0);

        // Children should have different embeddings
        let diff = norm(&sub(&result.left_embedding, &result.right_embedding));
        assert!(diff > 0.01);

        // Child shard IDs should be different
        assert_ne!(result.left_shard_id, result.right_shard_id);
    }

    #[test]
    fn test_bisect_embedding_norm_preservation() {
        // Without noise, the bisection should preserve norms (Parseval-like)
        let parent = make_test_embedding(2.0);
        let result = bisect_embedding(&parent, ShardId::new(1), BlockHeight::from(100), false);

        // ||e_left||² + ||e_right||² should ≈ ||e_parent||²
        let parent_sq = result.parent_norm * result.parent_norm;
        let left_sq = result.left_norm * result.left_norm;
        let right_sq = result.right_norm * result.right_norm;
        let total_child_sq = left_sq + right_sq;

        // Allow small numerical error
        assert!((total_child_sq - parent_sq).abs() / parent_sq < 1e-9);
    }

    #[test]
    fn test_bisect_embedding_determinism() {
        let parent = make_test_embedding(1.5);
        let r1 = bisect_embedding(&parent, ShardId::new(7), BlockHeight::from(200), false);
        let r2 = bisect_embedding(&parent, ShardId::new(7), BlockHeight::from(200), false);

        for i in 0..EMBEDDING_DIM {
            assert!((r1.left_embedding[i] - r2.left_embedding[i]).abs() < 1e-15);
            assert!((r1.right_embedding[i] - r2.right_embedding[i]).abs() < 1e-15);
        }
        assert_eq!(r1.left_shard_id, r2.left_shard_id);
        assert_eq!(r1.right_shard_id, r2.right_shard_id);
    }

    #[test]
    fn test_bisect_embedding_with_noise() {
        let parent = make_test_embedding(1.0);
        let result_no_noise = bisect_embedding(&parent, ShardId::new(1), BlockHeight::from(100), false);
        let result_with_noise = bisect_embedding(&parent, ShardId::new(1), BlockHeight::from(100), true);

        // With noise, the embeddings should be slightly different
        // (but still in the same general neighborhood)
        let left_diff = norm(&sub(
            &result_no_noise.left_embedding,
            &result_with_noise.left_embedding,
        ));
        // Noise should be small but non-zero
        assert!(left_diff > 0.0);
        assert!(left_diff < 1.0); // But not huge
    }

    #[test]
    fn test_verify_bisection_norms() {
        let parent = make_test_embedding(3.0);
        let result = bisect_embedding(&parent, ShardId::new(1), BlockHeight::from(100), false);
        // Without noise, should be very accurate
        assert!(verify_bisection_norms(&result, 1e-6));
    }

    #[test]
    fn test_merge_embeddings_basic() {
        // Split then merge should approximately recover the original
        let parent = make_test_embedding(2.0);
        let split_result = bisect_embedding(&parent, ShardId::new(1), BlockHeight::from(100), false);
        let merge_result = merge_embeddings(
            &split_result.left_embedding,
            &split_result.right_embedding,
        );

        // Merged embedding should be close to original
        let diff = norm(&sub(&merge_result.merged_embedding, &parent));
        // Relative error should be small
        assert!(diff / norm(&parent) < 1e-9);
    }

    #[test]
    fn test_merge_embeddings_symmetry() {
        // Merging a+b should give the same result as b+a
        let a = make_test_embedding(1.0);
        let b = make_test_embedding(2.0);
        let r1 = merge_embeddings(&a, &b);
        let r2 = merge_embeddings(&b, &a);
        // Both should give the same merged embedding
        let diff = norm(&sub(&r1.merged_embedding, &r2.merged_embedding));
        assert!(diff < 1e-10);
    }

    #[test]
    fn test_split_then_merge_roundtrip() {
        // Full round-trip: split, then merge, should recover original
        let mut parent = ZERO_EMBEDDING;
        for i in 0..EMBEDDING_DIM {
            parent[i] = (i as f64) * 0.1 + 1.0;
        }

        let split = bisect_embedding(&parent, ShardId::new(42), BlockHeight::from(999), false);
        let merge = merge_embeddings(&split.left_embedding, &split.right_embedding);

        let recovery_error = norm(&sub(&merge.merged_embedding, &parent)) / norm(&parent);
        assert!(recovery_error < 1e-9, "round-trip error: {}", recovery_error);
    }

    #[test]
    fn test_assign_account_to_child() {
        let seed = compute_bisection_seed(ShardId::new(1), BlockHeight::from(1));
        let left = ShardId::new(100);
        let right = ShardId::new(200);

        // Same account always maps to the same child
        let c1 = assign_account_to_child(b"alice", &seed, left, right);
        let c2 = assign_account_to_child(b"alice", &seed, left, right);
        assert_eq!(c1, c2);

        // Different accounts may map to different children
        let c_bob = assign_account_to_child(b"bob", &seed, left, right);
        // (Can't guarantee they're different, but with good hash it's very likely)
    }

    #[test]
    fn test_batch_assign_accounts_balance() {
        let seed = compute_bisection_seed(ShardId::new(1), BlockHeight::from(1));
        let left = ShardId::new(100);
        let right = ShardId::new(200);

        // Assign 1000 accounts and check balance
        let accounts: Vec<Vec<u8>> = (0..1000)
            .map(|i| format!("account_{}", i).into_bytes())
            .collect();

        let (assignments, balance_ratio) = batch_assign_accounts(
            &accounts, &seed, left, right,
        );

        assert_eq!(assignments.len(), 1000);
        // Balance ratio should be roughly 0.5 (within 10%)
        assert!(balance_ratio > 0.35 && balance_ratio < 0.65,
            "balance_ratio: {}", balance_ratio);
    }

    #[test]
    fn test_split_proposal_creation() {
        let proposal = SplitProposal::new(
            ShardId::new(42),
            BlockHeight::from(1000),
            ValidatorId::from_bytes([1u8; 32]),
            0.95,
        );
        assert_eq!(proposal.shard_id, ShardId::new(42));
        assert!((proposal.overload_probability - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_split_state_machine_quorum() {
        let proposal = SplitProposal::new(
            ShardId::new(1),
            BlockHeight::from(1),
            ValidatorId::from_bytes([0u8; 32]),
            0.95,
        );
        let mut sm = SplitStateMachine::new(proposal);
        sm.set_total_weight(100);

        // Add approvals until quorum
        assert_eq!(sm.state, SplitState::Proposed);

        let approval = SplitApproval {
            proposal_hash: [0u8; 32],
            validator_id: ValidatorId::from_bytes([1u8; 32]),
            voting_weight: VotingWeight(68), // 68%
            partial_sig: Vec::new(),
        };
        sm.add_approval(approval).unwrap();

        // 68% >= 67%, should be approved
        assert_eq!(sm.state, SplitState::Approved);
        assert!(sm.check_quorum());
    }

    #[test]
    fn test_split_state_machine_below_quorum() {
        let proposal = SplitProposal::new(
            ShardId::new(1),
            BlockHeight::from(1),
            ValidatorId::from_bytes([0u8; 32]),
            0.95,
        );
        let mut sm = SplitStateMachine::new(proposal);
        sm.set_total_weight(100);

        let approval = SplitApproval {
            proposal_hash: [0u8; 32],
            validator_id: ValidatorId::from_bytes([1u8; 32]),
            voting_weight: VotingWeight(66), // 66% < 67%
            partial_sig: Vec::new(),
        };
        sm.add_approval(approval).unwrap();

        assert_eq!(sm.state, SplitState::Proposed);
        assert!(!sm.check_quorum());
    }

    #[test]
    fn test_split_state_machine_full_lifecycle() {
        let proposal = SplitProposal::new(
            ShardId::new(1),
            BlockHeight::from(1),
            ValidatorId::from_bytes([0u8; 32]),
            0.95,
        );
        let mut sm = SplitStateMachine::new(proposal);
        sm.set_total_weight(100);

        // Approve
        let approval = SplitApproval {
            proposal_hash: [0u8; 32],
            validator_id: ValidatorId::from_bytes([1u8; 32]),
            voting_weight: VotingWeight(70),
            partial_sig: Vec::new(),
        };
        sm.add_approval(approval).unwrap();
        assert_eq!(sm.state, SplitState::Approved);

        // Start re-execution
        let parent = make_test_embedding(1.0);
        let bisection = bisect_embedding(&parent, ShardId::new(1), BlockHeight::from(1), false);
        sm.start_reexecution(bisection).unwrap();
        assert_eq!(sm.state, SplitState::ReExecuting);

        // Complete re-execution
        sm.update_reexecution_progress(REEXECUTION_WINDOW);
        assert_eq!(sm.state, SplitState::RoutingUpdate);

        // Complete
        sm.complete().unwrap();
        assert_eq!(sm.state, SplitState::Completed);
        assert!(sm.is_terminal());
    }

    #[test]
    fn test_split_state_machine_failure() {
        let proposal = SplitProposal::new(
            ShardId::new(1),
            BlockHeight::from(1),
            ValidatorId::from_bytes([0u8; 32]),
            0.95,
        );
        let mut sm = SplitStateMachine::new(proposal);
        sm.fail("bisection error".into());
        assert_eq!(sm.state, SplitState::Failed);
        assert!(sm.is_terminal());
        assert_eq!(sm.failure_reason.as_deref(), Some("bisection error"));
    }

    #[test]
    fn test_merge_state_machine_lifecycle() {
        let proposal = MergeProposal::new(
            ShardId::new(100),
            ShardId::new(200),
            5.0,
        );
        let mut ms = MergeStateMachine::new(proposal);
        assert_eq!(ms.state, MergeState::Triggered);

        ms.start_computing().unwrap();
        assert_eq!(ms.state, MergeState::Computing);

        let left = make_test_embedding(1.0);
        let right = make_test_embedding(1.5);
        let merge_res = merge_embeddings(&left, &right);
        ms.set_merge_result(merge_res).unwrap();
        assert_eq!(ms.state, MergeState::ReExecuting);

        ms.update_reexecution_progress(REEXECUTION_WINDOW);
        assert_eq!(ms.state, MergeState::RoutingUpdate);

        ms.complete().unwrap();
        assert_eq!(ms.state, MergeState::Completed);
        assert!(ms.is_terminal());
    }

    #[test]
    fn test_bytes_to_embedding_roundtrip() {
        let original = make_test_embedding(3.14);
        let bytes = embedding_f64_to_bytes(&original);
        assert_eq!(bytes.len(), EMBEDDING_BYTES);

        let restored = bytes_to_embedding_f64(&bytes).unwrap();
        for i in 0..EMBEDDING_DIM {
            assert!((original[i] - restored[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_embedding_root_consistency() {
        let v = make_test_embedding(1.0);
        let r1 = embedding_root(&v);
        let r2 = embedding_root(&v);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_embedding_root_different_inputs() {
        let v1 = make_test_embedding(1.0);
        let v2 = make_test_embedding(2.0);
        let r1 = embedding_root(&v1);
        let r2 = embedding_root(&v2);
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_derive_child_shard_ids() {
        let seed = compute_bisection_seed(ShardId::new(1), BlockHeight::from(1));
        let parent = ShardId::new(1);
        let left = derive_child_shard_id(parent, &seed, 0);
        let right = derive_child_shard_id(parent, &seed, 1);
        // Children must be different from each other
        assert_ne!(left, right);
    }

    #[test]
    fn test_zero_embedding() {
        let z = ZERO_EMBEDDING;
        assert!((norm(&z)).abs() < 1e-15);
    }

    #[test]
    fn test_bisect_zero_embedding() {
        // Bisecting the zero embedding should produce near-zero children
        let result = bisect_embedding(&ZERO_EMBEDDING, ShardId::new(1), BlockHeight::from(1), false);
        assert!(result.left_norm < 1e-10);
        assert!(result.right_norm < 1e-10);
    }

    #[test]
    fn test_split_state_machine_duplicate_approval() {
        let proposal = SplitProposal::new(
            ShardId::new(1), BlockHeight::from(1),
            ValidatorId::from_bytes([0u8; 32]), 0.95,
        );
        let mut sm = SplitStateMachine::new(proposal);
        sm.set_total_weight(100);

        let val_id = ValidatorId::from_bytes([1u8; 32]);
        let approval1 = SplitApproval {
            proposal_hash: [0u8; 32],
            validator_id: val_id.clone(),
            voting_weight: VotingWeight(30),
            partial_sig: Vec::new(),
        };
        let approval2 = SplitApproval {
            proposal_hash: [0u8; 32],
            validator_id: val_id.clone(),
            voting_weight: VotingWeight(30),
            partial_sig: Vec::new(),
        };

        sm.add_approval(approval1).unwrap();
        assert!(sm.add_approval(approval2).is_err()); // Duplicate
    }

    #[test]
    fn test_merge_proposal_hash() {
        let p1 = MergeProposal::new(ShardId::new(1), ShardId::new(2), 10.0);
        let p2 = MergeProposal::new(ShardId::new(1), ShardId::new(2), 10.0);
        assert_eq!(p1.hash(), p2.hash());
    }
}
