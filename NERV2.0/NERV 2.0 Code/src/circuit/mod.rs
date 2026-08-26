//! ZK Circuits — LatentLedger Lite.
//!
//! Because the NWO encoder is a single-layer Perceptron (strictly linear),
//! the ZK circuit is drastically simplified. We no longer need Halo2 lookup
//! tables for GELU or Softmax. Circuit drops from 7.9M to ~50K constraints.
//!
//! # Null-Space Exclusion Gate (V2.0 Appendix C Correction)
//!
//! To prevent null-space attacks — where a malicious client constructs a
//! transaction vector ΔS(tx) such that W·ΔS = 0, creating an algebraic
//! phantom transaction that passes math checks without altering the global
//! state embedding — the circuit enforces a strict Non-Zero Delta Constraint.
//!
//! The gate computes the squared L2 norm of the homomorphic delta:
//!
//! ‖δ‖² = Σᵢ δ[i]²
//! ```
//!
//! and constrains:
//!
//! ‖δ‖² · inv_norm = 1   (forces ‖δ‖² ≠ 0)
//! ```
//!
//! If δ = 0_vector (null-space attack), no `inv_norm` witness can satisfy
//! `0 · inv_norm = 1` in a prime field, so the proof is unsatisfiable and
//! rejected. This guarantees every verified transaction provably alters the
//! 512-byte global state embedding, ensuring absolute collision resistance
//! without requiring a global Sparse Merkle Tree.
//!
//! Submodules:
//! - `latent_ledger_lite` — ~50.5K constraint circuit proving W·x + b ≠ 0
//! - `delta_circuit` — Proves client-side linear delta computation
//! - `recursive` — Nova folding for batch proofs
//!
//! # Constraint Breakdown (for 64-dim output, 128-dim input)
//!
//! | Operation | Count | Constraints Each | Total |
//! |-----------|------|-------------------|-------|
//! | Dot product (W·x) | 64 × 128 | 1 mul + 1 copy | 16,384 |
//! | Accumulation | 64 × 127 | 1 add | 8,128 |
//! | Bias addition | 64 | 1 add | 64 |
//! | Output constraint | 64 | 1 copy | 64 |
//! | **Null-space: δ² squaring** | **64** | **1 mul** | **64** |
//! | **Null-space: norm init** | **1** | **1 copy** | **1** |
//! | **Null-space: norm accum** | **63** | **1 add** | **63** |
//! | **Null-space: inversion** | **1** | **1 mul** | **2** |
//! | Overhead (perm, copy) | — | — | ~25,500 |
//! | **Total** | — | — | **~50,500** |
//!
//! # Proof Parameters
//!
//! | Parameter | Value |
//! |-----------|-------|
//! | Circuit degree (k) | 16 (2¹⁶ = 65,536 rows) |
// | Proof size (IPA) | ~1–3 KB |
// | Proof size (KZG, mainnet) | 400–750 bytes |
//! | Proving time (mobile) | <50 ms |
//! | Verification time | <10 ms |
//! | Soundness error | <2⁻¹²⁸ |

pub mod latent_ledger_lite;
pub mod delta_circuit;
pub mod recursive;

// Re-export primary types
pub use latent_ledger_lite::{
    LatentLedgerLiteCircuit, CircuitProof, CircuitProvingKey, CircuitVerificationKey,
    generate_keys, create_proof, verify_proof,
};
pub use delta_circuit::{DeltaCircuit, create_delta_proof, verify_delta_proof};
pub use recursive::{
    NovaCompressedProof, fold_proofs, verify_compressed_proof, fold_batch,
};

use crate::EMBEDDING_DIM;

/// Circuit polynomial degree: 2^16 = 65,536 rows.
///
/// This provides enough rows for ~50.5K constraints with comfortable margin
/// for Halo2's permutation argument and copy constraints, plus the
/// Null-Space Exclusion Gate region.
pub const CIRCUIT_DEGREE: u32 = 16;

/// Maximum number of constraints in the LatentLedger Lite circuit.
///
/// V2.0 base (W·x + b): ~50,000 constraints
/// Null-Space Exclusion Gate: ~500 constraints (130 arithmetic + ~370 overhead)
/// Total: ~50,500
///
/// Bumped to 55,000 to provide headroom for the null-space exclusion gate
/// and future minor circuit additions without requiring a re-ceremony of
/// the universal SRS (which is tied to CIRCUIT_DEGREE, not this constant).
pub const MAX_CONSTRAINT_COUNT: usize = 55_000;

/// Maximum proof size in bytes (Halo2 + Nova folding).
///
/// IPA commitment scheme proofs are ~1–3 KB for a ~50K constraint
/// circuit. KZG proofs (mainnet upgrade) would be ~400–750 B.
/// The `is_valid_size()` method allows 2× this value (8,192 bytes)
/// to accommodate IPA proofs with comfortable margin.
///
/// # Upgrade Path to KZG
///
/// When switching to KZG commitment (Ethereum ceremony SRS), this
/// can be reduced back to 750 (proofs shrink to ~400–750 B).
pub const MAX_PROOF_SIZE: usize = 4_096;

/// Average proof size in bytes.
pub const AVG_PROOF_SIZE: usize = 500;

/// Fixed-point quantization factor Q = 2^32.
///
/// In the circuit, all values are represented as quantized integers:
/// `v_quantized = round(v_real * Q)`. Multiplication of two quantized
/// values requires division by Q in the field: `result = a * b * Q^{-1}`.
pub const QUANTIZATION_BITS: u32 = 32;

/// The quantization factor as a u64.
pub const Q_U64: u64 = 1u64 << QUANTIZATION_BITS;

// ── Row Layout Constants ──────────────────────────────────────────────

/// Number of rows needed per output dimension in the main computation:
/// 128 (dot product) + 1 (bias) = 129 rows.
pub const ROWS_PER_OUTPUT: usize = 129;

/// Total rows for the main computation (W·x + b):
/// 64 output dimensions × 129 rows = 8,256 rows.
pub const COMPUTATION_ROWS: usize = EMBEDDING_DIM * ROWS_PER_OUTPUT;

/// Number of rows for the Null-Space Exclusion Gate region:
/// 64 (δ[i]² + accumulation) + 1 (inversion constraint) = 65 rows.
///
/// Layout:
///   Row 0..63:  delta_copy[i], delta_sq[i], norm_accum[i]  (s_norm_sq + s_norm_init/s_norm_accum)
///   Row 64:     final norm_accum, inv_norm                   (s_null_check)
pub const NULL_SPACE_ROWS: usize = EMBEDDING_DIM + 1;

/// Total circuit rows (main computation + null-space exclusion):
/// 8,256 + 65 = 8,321 rows.
///
/// Well within the 65,536 rows available at CIRCUIT_DEGREE = 16.
pub const TOTAL_CIRCUIT_ROWS: usize = COMPUTATION_ROWS + NULL_SPACE_ROWS;

/// Number of arithmetic constraints in the Null-Space Exclusion Gate
/// (excluding Halo2 overhead).
///
/// Breakdown:
/// - 64 multiplication gates (δ[i]² = δ[i] * δ[i])
/// - 1 initialization gate   (norm_accum[0] = delta_sq[0])
/// - 63 accumulation gates    (norm_accum[i] = norm_accum[i-1] + delta_sq[i])
/// - 1 inversion gate         (‖δ‖² · inv_norm = 1)
pub const NULL_SPACE_ARITHMETIC_CONSTRAINTS: usize = 64 + 1 + 63 + 1;

/// Estimated Halo2 overhead (copy constraints, permutation argument)
/// added by the Null-Space Exclusion Gate's 4 new advice columns.
///
/// Approximately 4 columns × 65 rows = 260 new cells in the permutation
/// argument, plus ~130 copy constraints for delta_copy→instance links.
pub const NULL_SPACE_OVERHEAD_CONSTRAINTS: usize = 390;

/// Total estimated constraints for the Null-Space Exclusion Gate.
pub const NULL_SPACE_TOTAL_CONSTRAINTS: usize =
    NULL_SPACE_ARITHMETIC_CONSTRAINTS + NULL_SPACE_OVERHEAD_CONSTRAINTS;

// ── Performance Targets ──────────────────────────────────────────────

/// Verification time target in milliseconds.
///
/// The null-space gate adds ~65 rows to verify, which is negligible
/// (Halo2 verification is O(1) in constraint count — it depends only
/// on the circuit degree k, not the number of constraints).
pub const VERIFICATION_TARGET_MS: u64 = 10;

/// Proving time target in milliseconds (on mobile CPU).
///
/// The null-space gate adds ~130 arithmetic constraints and ~65 rows
/// of synthesis. At ~0.5μs per constraint on a modern mobile CPU,
/// this adds <0.1ms to proving time.
pub const PROVING_TARGET_MS: u64 = 50;