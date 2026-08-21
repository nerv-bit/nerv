//! # NERV v2.0 — The Self-Evolving Blockchain
//!
//! A private, post-quantum, infinitely scalable Layer-1 blockchain
//! powered by the **Neural Weight Oscillator (NWO) Paradigm** and
//! **Pure-Cryptographic Privacy** (Staked PQ-Sphinx Mixnet + Threshold
//! Decryption Mempool).
//!
//! ## Architecture Summary
//!
//! | Component | V1.01 (Old) | V2.0 (Current) |
//! |-----------|-------------|----------------|
//! | State Model | 24-Layer Transformer | Single-Layer Perceptron (NWO) |
//! | ZK Circuit | 7.9M constraints (LatentLedger) | ~50K constraints (LatentLedger Lite) |
//! | Privacy Mixer | 5-Hop TEE (SGX/SEV) | 5-Hop PQ-Sphinx + DKG Mempool |
//! | Self-Evolution | 30-Day Federated Learning | Per-Block Adam/Huber Backprop |
//! | Homomorphism | Approximate (≤10⁻⁹ error) | Exact (0 error, guaranteed by linearity) |
//!
//! ## Core Innovation
//!
//! The NWO Paradigm uses a single-layer Perceptron `f(x) = W·x + b` that is
//! **natively and exactly homomorphic**:
//!
//! ```text
//! f(x + Δx) = W·(x + Δx) + b = (W·x + b) + W·Δx = f(x) + f(Δx)
//! ```
//!
//! This eliminates pre-training, reduces ZK circuit constraints by 99%, and
//! enables the network to learn on every single block via the Adam optimizer
//! and Huber Loss.

#![deny(
    unsafe_code,
    dead_code,
    unused_imports,
    unused_variables,
    unused_mut,
    unreachable_pub,
    missing_docs,
    rustdoc::broken_intra_doc_links
)]
#![warn(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo
)]
#![allow(
    clippy::module_name_repetitions,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

// ─── Module Declarations ────────────────────────────────────────────────

pub mod config;
pub mod utils;

pub mod embedding;
pub mod circuit;
pub mod privacy;
pub mod consensus;
pub mod sharding;
pub mod crypto;
pub mod economy;
pub mod tokenomics;
pub mod network;

// ─── Re-exports ─────────────────────────────────────────────────────────

pub use config::NodeConfig;
pub use utils::balance::{Amount, BalanceCommitment, BalanceDelta, BalanceDirection};

// ─── Core Constants ─────────────────────────────────────────────────────

/// Dimensionality of the neural state embedding vector.
/// e_t ∈ ℝ⁶⁴ — 64 dimensions of 64-bit fixed-point numbers.
pub const EMBEDDING_DIM: usize = 64;

/// Total byte size of the embedding: 64 × 8 = 512 bytes.
pub const EMBEDDING_BYTES: usize = EMBEDDING_DIM * 8;

/// Maximum number of private transactions per batch.
/// V2.0 batches up to 10,000 txs (up from 256 in V1.01 due to linear aggregation).
pub const BATCH_MAX_SIZE: usize = 10_000;

/// Number of Sphinx onion-routing hops.
pub const SPHINX_HOPS: usize = 5;

/// Consensus quorum threshold as a fraction (67% of weighted stake).
pub const CONSENSUS_QUORUM_NUMERATOR: u64 = 67;
pub const CONSENSUS_QUORUM_DENOMINATOR: u64 = 100;

/// Adam optimizer β₁ (exponential decay rate for first moment estimate).
pub const ADAM_BETA1: f64 = 0.9;

/// Adam optimizer β₂ (exponential decay rate for second moment estimate).
pub const ADAM_BETA2: f64 = 0.999;

/// Adam optimizer ε (numerical stability constant).
pub const ADAM_EPSILON: f64 = 1e-8;

/// Huber loss δ threshold — transitions from quadratic to linear loss.
/// Chosen to be robust to MEV spikes and large exchange inflows.
pub const HUBER_DELTA: f64 = 1.0;

/// Adam optimizer default learning rate α.
pub const ADAM_LEARNING_RATE: f64 = 0.001;

/// Target probabilistic finality latency (milliseconds).
pub const TARGET_FINALITY_MS: u64 = 600;

/// Target embedding root prediction latency (microseconds).
/// A dot product on 64 dimensions takes sub-microsecond on modern hardware.
pub const TARGET_PREDICTION_US: u64 = 100;

/// Vote window for consensus agreement (milliseconds).
pub const VOTE_WINDOW_MS: u64 = 800;

/// Monte-Carlo dispute resolution: number of parallel simulations.
pub const DISPUTE_SIMULATIONS: usize = 10_000;

/// Monte-Carlo dispute resolution: number of randomly selected validators.
pub const DISPUTE_VALIDATOR_COUNT: usize = 32;

/// Minimum challenger bond as fraction of stake (0.5%).
pub const CHALLENGER_BOND_MIN_BPS: u64 = 50;

/// Maximum challenger bond as fraction of stake (1%).
pub const CHALLENGER_BOND_MAX_BPS: u64 = 100;

/// Slashing range for losing dispute side (0.5%–5%).
pub const SLASH_MIN_BPS: u64 = 50;
pub const SLASH_MAX_BPS: u64 = 500;

/// Maximum reorg depth before VDW invalidation (10 confirmations ≈ 18s).
pub const MAX_REORG_DEPTH: u32 = 10;

/// Fixed-point integer bits (for 32.32 format in i64).
pub const FP_INTEGER_BITS: u32 = 32;

/// Fixed-point fractional bits (for 32.32 format in i64).
pub const FP_FRACTIONAL_BITS: u32 = 32;

/// Reed-Solomon erasure coding parameters (k=5, m=2 → tolerates 2 failures).
pub const ERASURE_K: usize = 5;
pub const ERASURE_M: usize = 2;

/// Cover traffic base delay μ in milliseconds.
pub const COVER_TRAFFIC_BASE_DELAY_MS: u64 = 100;

/// Cover traffic jitter σ in milliseconds.
pub const COVER_TRAFFIC_JITTER_MS: u64 = 200;

/// Cover traffic ratio range (1×–10× real traffic).
pub const COVER_TRAFFIC_MIN_RATIO: f64 = 1.0;
pub const COVER_TRAFFIC_MAX_RATIO: f64 = 10.0;

/// Sphinx packet fixed size in bytes.
pub const SPHINX_PACKET_SIZE: usize = 1500;

/// VDW maximum size in bytes (V2.0 with Dilithium-3 signatures).
pub const VDW_MAX_BYTES: usize = 3600;

/// VDW average size in bytes.
pub const VDW_AVG_BYTES: usize = 3500;

/// Dilithium-3 public key size in bytes.
pub const DILITHIUM3_PK_BYTES: usize = 1952;

/// Dilithium-3 secret key size in bytes.
pub const DILITHIUM3_SK_BYTES: usize = 4032;

/// Dilithium-3 signature size in bytes.
pub const DILITHIUM3_SIG_BYTES: usize = 3293;

/// ML-KEM-768 public key size in bytes.
pub const ML_KEM768_PK_BYTES: usize = 1184;

/// ML-KEM-768 secret key size in bytes.
pub const ML_KEM768_SK_BYTES: usize = 2400;

/// ML-KEM-768 ciphertext size in bytes.
pub const ML_KEM768_CT_BYTES: usize = 1088;

/// ML-KEM-768 shared secret size in bytes.
pub const ML_KEM768_SS_BYTES: usize = 32;

/// SPHINCS+-SHA256-192f-simple public key size in bytes.
pub const SPHINCS_PK_BYTES: usize = 48;

/// SPHINCS+-SHA256-192f-simple signature size in bytes.
pub const SPHINCS_SIG_BYTES: usize = 29792;

/// BLS12-381 public key size (compressed, 48 bytes).
pub const BLS_PK_BYTES: usize = 48;

/// BLS12-381 signature size (compressed, 48 bytes).
pub const BLS_SIG_BYTES: usize = 48;

/// Threshold for DKG participants (minimum).
pub const DKG_MIN_PARTICIPANTS: usize = 3;

/// NERV total supply: 10,000,000,000 NERV (10 billion).
pub const TOTAL_SUPPLY_NERV: u64 = 10_000_000_000;

/// NERV decimal precision (9 decimal places → nano-NERV).
pub const NERV_DECIMALS: u8 = 9;

/// One NERV in the base unit (nano-NERV).
pub const ONE_NERV: u64 = 1_000_000_000;

/// Total supply in nano-NERV = 10^19.
pub const TOTAL_SUPPLY_NANO: u64 = TOTAL_SUPPLY_NERV * ONE_NERV;

// Compile-time validation
static_assertions::const_assert!(EMBEDDING_DIM * 8 == EMBEDDING_BYTES);
static_assertions::const_assert!(TOTAL_SUPPLY_NANO < u64::MAX / 2); // headroom for arithmetic
static_assertions::const_assert!(SPHINX_HOPS == 5);
static_assertions::const_assert!(CONSENSUS_QUORUM_NUMERATOR < CONSENSUS_QUORUM_DENOMINATOR);
static_assertions::const_assert!(ERASURE_K > 0);
static_assertions::const_assert!(ERASURE_M > 0);

// ─── Core Type Definitions ──────────────────────────────────────────────

/// A 32-byte BLAKE3 transaction hash.
///
/// This uniquely identifies a private transaction on the NERV network.
/// Tx hashes are the only on-chain identifier — no addresses, amounts, or
/// metadata are ever visible.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
    serde::Serialize, serde::Deserialize, zeroize::Zeroize,
)]
#[zeroize(drop)]
pub struct TxHash(
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub [u8; 32],
);

impl TxHash {
    /// The all-zero null hash (used as a sentinel value).
    pub const NULL: Self = Self([0u8; 32]);

    /// Construct from a raw byte array.
    #[inline]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Return the raw byte array.
    #[inline]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Convert to a hexadecimal string.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Construct from a hexadecimal string.
    pub fn from_hex(s: &str) -> Result<Self, NervError> {
        let bytes: [u8; 32] = hex::decode(s)
            .map_err(|e| NervError::Crypto(format!("invalid hex for TxHash: {e}")))?
            .try_into()
            .map_err(|_| NervError::Crypto("TxHash must be exactly 32 bytes".into()))?;
        Ok(Self(bytes))
    }

    /// Returns true if this is the null hash.
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0 == [0u8; 32]
    }
}

impl std::fmt::Display for TxHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl AsRef<[u8]> for TxHash {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// A 32-byte BLAKE3 hash of the canonical embedding root.
///
/// This is the "state root" of a NERV shard — a 32-byte commitment
/// to the full 512-byte neural state embedding.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
    serde::Serialize, serde::Deserialize,
)]
pub struct EmbeddingRoot(
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub [u8; 32],
);

impl EmbeddingRoot {
    /// The genesis (initial) embedding root.
    pub const GENESIS: Self = Self([0u8; 32]);

    /// Construct from a raw byte array.
    #[inline]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Return the raw byte array.
    #[inline]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Convert to a hexadecimal string.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Construct from a hexadecimal string.
    pub fn from_hex(s: &str) -> Result<Self, NervError> {
        let bytes: [u8; 32] = hex::decode(s)
            .map_err(|e| NervError::Crypto(format!("invalid hex for EmbeddingRoot: {e}")))?
            .try_into()
            .map_err(|_| NervError::Crypto("EmbeddingRoot must be exactly 32 bytes".into()))?;
        Ok(Self(bytes))
    }

    /// Returns true if this is the genesis root.
    #[inline]
    pub fn is_genesis(&self) -> bool {
        self.0 == [0u8; 32]
    }
}

impl std::fmt::Display for EmbeddingRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl AsRef<[u8]> for EmbeddingRoot {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Shard identifier.
///
/// Shards in NERV are dynamic — they split like cells under AI-predicted
/// load (NWO Perceptron) and merge when load drops below threshold.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
    serde::Serialize, serde::Deserialize,
)]
pub struct ShardId(pub u64);

impl ShardId {
    /// The root (genesis) shard.
    pub const GENESIS: Self = Self(0);

    /// Construct a shard ID from a u64.
    #[inline]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns true if this is the genesis shard.
    #[inline]
    pub const fn is_genesis(&self) -> bool {
        self.0 == 0
    }
}

impl std::fmt::Display for ShardId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "shard:{}", self.0)
    }
}

impl From<u64> for ShardId {
    #[inline]
    fn from(v: u64) -> Self {
        Self(v)
    }
}

impl From<ShardId> for u64 {
    #[inline]
    fn from(s: ShardId) -> Self {
        s.0
    }
}

/// Block height (monotonically increasing).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
    serde::Serialize, serde::Deserialize,
)]
pub struct BlockHeight(pub u64);

impl BlockHeight {
    /// Genesis block height.
    pub const GENESIS: Self = Self(0);

    /// Increment the block height by one.
    #[inline]
    pub const fn next(&self) -> Self {
        Self(self.0 + 1)
    }

    /// Compute the difference between two heights.
    #[inline]
    pub const fn diff(&self, other: Self) -> u64 {
        if self.0 >= other.0 {
            self.0 - other.0
        } else {
            other.0 - self.0
        }
    }
}

impl std::fmt::Display for BlockHeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "h:{}", self.0)
    }
}

impl From<u64> for BlockHeight {
    #[inline]
    fn from(v: u64) -> Self {
        Self(v)
    }
}

impl From<BlockHeight> for u64 {
    #[inline]
    fn from(h: BlockHeight) -> Self {
        h.0
    }
}

/// Validator identifier — BLAKE3 hash of a Dilithium-3 public key.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash,
    serde::Serialize, serde::Deserialize,
)]
pub struct ValidatorId(
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub [u8; 32],
);

impl ValidatorId {
    /// Construct from a Dilithium-3 public key by hashing it with BLAKE3.
    pub fn from_dilithium_pk(pk: &[u8; DILITHIUM3_PK_BYTES]) -> Self {
        let hash = blake3::hash(pk);
        Self(hash.into())
    }

    /// Construct from a raw 32-byte hash.
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Return the raw byte array.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Convert to a hexadecimal string.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

impl std::fmt::Display for ValidatorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "val:{}", self.to_hex())
    }
}

impl AsRef<[u8]> for ValidatorId {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Stake amount in nano-NERV.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
    serde::Serialize, serde::Deserialize,
)]
pub struct StakeAmount(pub u64);

impl StakeAmount {
    /// Zero stake.
    pub const ZERO: Self = Self(0);

    /// Minimum stake to become a validator (1,000 NERV).
    pub const MIN_VALIDATOR_STAKE: Self = Self(1_000 * ONE_NERV);

    /// Create a stake amount from nano-NERV.
    #[inline]
    pub const fn from_nano(nano: u64) -> Self {
        Self(nano)
    }

    /// Create a stake amount from whole NERV.
    #[inline]
    pub const fn from_nerv(nerv: u64) -> Self {
        Self(nerv * ONE_NERV)
    }

    /// Checked addition.
    #[inline]
    pub const fn checked_add(&self, other: Self) -> Option<Self> {
        match self.0.checked_add(other.0) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }

    /// Checked subtraction.
    #[inline]
    pub const fn checked_sub(&self, other: Self) -> Option<Self> {
        match self.0.checked_sub(other.0) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }

    /// Returns the nano-NERV value.
    #[inline]
    pub const fn as_nano(&self) -> u64 {
        self.0
    }

    /// Convert to f64 NERV (for display / reputation weighting).
    #[inline]
    pub fn as_nerv_f64(&self) -> f64 {
        self.0 as f64 / ONE_NERV as f64
    }
}

impl std::fmt::Display for StakeAmount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let whole = self.0 / ONE_NERV;
        let frac = self.0 % ONE_NERV;
        write!(f, "{whole}.{frac:09}")
    }
}

/// Validator reputation score, stored as a fixed-point u32 where
/// 1,000,000 represents 1.0 (full reputation).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
    serde::Serialize, serde::Deserialize,
)]
pub struct ReputationScore(pub u32);

impl ReputationScore {
    /// Scale factor: 1,000,000 = 1.0.
    pub const SCALE: u32 = 1_000_000;

    /// Perfect reputation (1.0).
    pub const PERFECT: Self = Self(Self::SCALE);

    /// Zero reputation.
    pub const ZERO: Self = Self(0);

    /// Initial reputation for new validators (0.5).
    pub const INITIAL: Self = Self(500_000);

    /// Convert from a floating-point value [0.0, 1.0].
    pub fn from_f64(val: f64) -> Self {
        let clamped = val.clamp(0.0, 1.0);
        Self((clamped * Self::SCALE as f64).round() as u32)
    }

    /// Convert to f64.
    pub fn to_f64(&self) -> f64 {
        self.0 as f64 / Self::SCALE as f64
    }

    /// Checked subtraction with floor at zero.
    #[inline]
    pub const fn saturating_sub(&self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    /// Checked addition with ceiling at PERFECT.
    #[inline]
    pub const fn saturating_add(&self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0).min(Self::SCALE))
    }
}

impl std::fmt::Display for ReputationScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.4}", self.to_f64())
    }
}

/// Epoch number (incremented each time the network's weights W are updated).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
    serde::Serialize, serde::Deserialize,
)]
pub struct Epoch(pub u64);

impl Epoch {
    /// Genesis epoch.
    pub const GENESIS: Self = Self(0);

    /// Advance to the next epoch.
    #[inline]
    pub const fn next(&self) -> Self {
        Self(self.0 + 1)
    }
}

impl std::fmt::Display for Epoch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "epoch:{}", self.0)
    }
}

impl From<u64> for Epoch {
    #[inline]
    fn from(v: u64) -> Self {
        Self(v)
    }
}

/// Combined weight for consensus voting: `stake × reputation`.
///
/// This is the metric used to determine if the 67% quorum is reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct VotingWeight(pub u128);

impl VotingWeight {
    /// Zero weight.
    pub const ZERO: Self = Self(0);

    /// Compute the voting weight from stake and reputation.
    #[inline]
    pub fn from_stake_reputation(stake: StakeAmount, reputation: ReputationScore) -> Self {
        // weight = stake_nano * reputation_scaled / SCALE
        // Using u128 to avoid overflow: stake can be up to ~10^19, reputation up to 10^6
        let weight = (stake.0 as u128) * (reputation.0 as u128) / (ReputationScore::SCALE as u128);
        Self(weight)
    }

    /// Checked addition.
    #[inline]
    pub const fn checked_add(&self, other: Self) -> Option<Self> {
        match self.0.checked_add(other.0) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }

    /// Saturating addition.
    #[inline]
    pub const fn saturating_add(&self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }
}

// ─── Error Type ─────────────────────────────────────────────────────────

/// Comprehensive error type for all NERV operations.
#[derive(Debug, thiserror::Error)]
pub enum NervError {
    /// Configuration loading or validation error.
    #[error("configuration error: {0}")]
    Config(String),

    /// Cryptographic operation failure (key generation, signing, encryption, etc.).
    #[error("cryptographic error: {0}")]
    Crypto(String),

    /// Zero-knowledge circuit error (proof generation, verification, etc.).
    #[error("circuit error: {0}")]
    Circuit(String),

    /// Privacy layer error (Sphinx construction, mixnet, DKG, threshold decryption).
    #[error("privacy error: {0}")]
    Privacy(String),

    /// Consensus error (voting, dispute, finality).
    #[error("consensus error: {0}")]
    Consensus(String),

    /// Sharding error (bisection, erasure coding, load prediction).
    #[error("sharding error: {0}")]
    Sharding(String),

    /// Network / P2P communication error.
    #[error("network error: {0}")]
    Network(String),

    /// Persistent storage error (RocksDB, etc.).
    #[error("storage error: {0}")]
    Storage(String),

    /// Economy / incentive error (gradient validation, reward distribution).
    #[error("economy error: {0}")]
    Economy(String),

    /// Tokenomics error (emission, vesting, allocation).
    #[error("tokenomics error: {0}")]
    Tokenomics(String),

    /// Insufficient balance for a transfer.
    #[error("insufficient balance: required {required} nano-NERV, available {available} nano-NERV")]
    InsufficientBalance {
        /// Amount required.
        required: u64,
        /// Amount available.
        available: u64,
    },

    /// Invalid transaction (malformed, double-spend, etc.).
    #[error("invalid transaction: {0}")]
    InvalidTransaction(String),

    /// Homomorphism error exceeded the tolerable bound.
    /// In V2.0, this should NEVER occur (exact by construction), but we
    /// retain it as a safety invariant check.
    #[error("homomorphism error exceeded bound: observed {observed}, allowed {allowed}")]
    HomomorphismBoundExceeded {
        /// Observed error magnitude.
        observed: f64,
        /// Maximum allowed error (0.0 in V2.0).
        allowed: f64,
    },

    /// Epoch rollback required — weight update produced unacceptable drift.
    #[error("epoch rollback required: Huber loss {loss} exceeds threshold {threshold}")]
    EpochRollback {
        /// Current Huber loss.
        loss: f64,
        /// Maximum allowed loss.
        threshold: f64,
    },

    /// Consensus quorum not reached.
    #[error("quorum not reached: {achieved:.2}% of required {required}%")]
    QuorumNotReached {
        /// Percentage achieved.
        achieved: f64,
        /// Percentage required (67%).
        required: f64,
    },

    /// Dispute resolution failure.
    #[error("dispute resolution failed: {0}")]
    DisputeFailed(String),

    /// Serialization / deserialization error.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A generic catch-all for errors that don't fit a specific category.
    #[error("{0}")]
    Other(String),
}

/// Convenience result alias.
pub type NervResult<T> = Result<T, NervError>;

// ─── Helper: Compute consensus quorum threshold ─────────────────────────

/// Check if a fraction of total voting weight meets the 67% quorum threshold.
///
/// Returns `Ok(())` if `achieved_weight >= 0.67 * total_weight`,
/// `Err(NervError::QuorumNotReached)` otherwise.
#[inline]
pub fn check_quorum(achieved_weight: VotingWeight, total_weight: VotingWeight) -> NervResult<()> {
    if total_weight.0 == 0 {
        return Err(NervError::Consensus("total voting weight is zero".into()));
    }
    // achieved >= (67/100) * total  ⟺  100 * achieved >= 67 * total
    let lhs = 100u128 * achieved_weight.0;
    let rhs = CONSENSUS_QUORUM_NUMERATOR as u128 * total_weight.0;
    if lhs >= rhs {
        Ok(())
    } else {
        let achieved_pct = (achieved_weight.0 as f64 / total_weight.0 as f64) * 100.0;
        let required_pct = (CONSENSUS_QUORUM_NUMERATOR as f64 / CONSENSUS_QUORUM_DENOMINATOR as f64) * 100.0;
        Err(NervError::QuorumNotReached {
            achieved: achieved_pct,
            required: required_pct,
        })
    }
}

/// Compute the per-validator voting weight: `stake × reputation`.
#[inline]
pub fn compute_voting_weight(stake: StakeAmount, reputation: ReputationScore) -> VotingWeight {
    VotingWeight::from_stake_reputation(stake, reputation)
}

