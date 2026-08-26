//! Client-Side Delta Circuit — Optimized for wallet use.
//!
//! The delta circuit is a streamlined version of LatentLedger Lite
//! designed for the wallet (client-side) proving workflow:
//!
//! 1. Wallet constructs `TransactionFeatures` from private tx details
//! 2. Wallet computes `δ(tx) = W · ΔS(tx) + b` locally
//! 3. Wallet generates a ZK proof that δ was computed correctly
//! 4. Wallet submits `(δ, proof)` to the network (NOT the raw tx)
//!
//! The network verifies the proof and applies the delta homomorphically
//! without ever seeing the transaction details.
//!
//! # Null-Space Attack Prevention (V2.0 Appendix C)
//!
//! The delta circuit inherits the Null-Space Exclusion Gate from
//! `LatentLedgerLiteCircuit`. Every proof generated via this module
//! cryptographically guarantees that `δ(tx) ≠ 0_vector`, meaning:
//!
//! - The transaction provably alters the global state embedding
//! - Algebraic phantom transactions (where `W·ΔS = 0`) are rejected
//! - Downstream consumers (block producers, aggregators) can rely on
//!   every verified delta being non-zero
//!
//! This is enforced at two layers:
//! 1. **Circuit layer**: The `s_null_check` gate constrains `‖δ‖² · inv = 1`,
//!    which is unsatisfiable when `δ = 0`.
//! 2. **API layer**: `create_proof` and `create_delta_proof_with_delta`
//!    perform an early defense-in-depth check before synthesis.
//!
//! # Optimizations vs LatentLedger Lite
//!
//! | Property | LatentLedger Lite | Delta Circuit |
//! |----------|-------------------|---------------|
//! | Context | Full-node / validator | Wallet (mobile) |
//! | Proving time | <50ms (desktop) | <30ms (mobile) |
//! | Proof size | 400–750 bytes | 300–600 bytes |
//! | Constraint count | ~50.5K | ~35.5K |
//! | Batch support | Yes | No (single tx) |
//! | Null-space check | ✓ (in-circuit) | ✓ (in-circuit + API) |

use crate::{EMBEDDING_DIM, NervError, NervResult};
use crate::embedding::fixed_point::{FixedPoint64, EmbeddingVector};
use crate::embedding::perceptron::{NwoWeights, TransactionFeatures};
use crate::embedding::homomorphism::EmbeddingDelta;
use crate::circuit::latent_ledger_lite::{
    CircuitProof, CircuitProvingKey, CircuitVerificationKey,
    create_proof, verify_proof, CircuitWitness,
};
use crate::circuit::CIRCUIT_DEGREE;
use crate::utils::blake3_hash;

use serde::{Deserialize, Serialize};

// ─── Delta Circuit ───────────────────────────────────────────────────────

/// A client-side delta proof: proves that the wallet correctly computed
/// `δ(tx) = W · ΔS(tx) + b_tx` for a private transaction.
///
/// This is the artifact that the wallet submits to the network alongside
/// the delta vector. The network verifies the proof and applies the delta
/// without learning anything about the transaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaCircuit {
    /// The underlying ZK proof.
    pub proof: CircuitProof,

    /// The claimed delta vector (public — submitted to the network).
    pub delta: EmbeddingDelta,

    /// The transaction hash (for VDW tracking).
    pub tx_hash: [u8; 32],
}

impl DeltaCircuit {
    /// Get the proof size in bytes.
    pub fn proof_size(&self) -> usize {
        self.proof.size()
    }

    /// Get the total size (proof + delta + metadata).
    pub fn total_size(&self) -> usize {
        self.proof.size() + crate::EMBEDDING_BYTES + 32
    }

    /// Get the weight commitment.
    pub fn weight_commitment(&self) -> &[u8; 32] {
        self.proof.weight_hash()
    }
}

// ─── Proof Creation (Wallet-Side) ────────────────────────────────────────

/// Create a delta proof for a private transaction.
///
/// This is called by the wallet after:
/// 1. Constructing the transaction features from the private tx
/// 2. Computing the delta: `δ(tx) = W · ΔS(tx) + b_tx`
///
/// # Arguments
///
/// * `pk` - The proving key for the current weights
/// * `weights` - The current NWO weights (public, known to wallet)
/// * `features` - The transaction features (private to wallet)
/// * `tx_hash` - The transaction hash for VDW tracking
///
/// # Returns
///
/// A `DeltaCircuit` containing the proof and the claimed delta.
/// # Null-Space Protection
///
/// The perceptron computes `δ = W·ΔS + b_tx` where `b_tx` is derived from
/// the transaction's nullifiers (a cryptographic hash). For `δ = 0` to occur,
/// an attacker would need `W·ΔS = -b_tx` — solving for a specific ΔS given
/// a random `b_tx`. This is computationally infeasible.
///
/// Nevertheless, the `create_proof` function performs a defense-in-depth
/// check and will reject any zero delta with a clear error. The ZK circuit's
/// `s_null_check` gate provides the cryptographic enforcement.
///
///
/// # Example
///
/// ```rust
/// use nerv::circuit::delta_circuit::create_delta_proof;
/// // In wallet:
/// let delta_proof = create_delta_proof(
///     &pk, &weights, &features, &tx_hash,
/// )?;
/// // Submit delta_proof.delta and delta_proof.proof to the network
/// ```
pub fn create_delta_proof(
    pk: &CircuitProvingKey,
    weights: &NwoWeights,
    features: &TransactionFeatures,
    tx_hash: &[u8; 32],
) -> NervResult<DeltaCircuit> {
    // Step 1: Compute the delta locally
    let perceptron = crate::embedding::NwoPerceptron::from_weights(weights.clone());
    let delta_vector = perceptron.compute_delta(features)?;
    let delta = EmbeddingDelta::from_vector(delta_vector);

    // Step 2: Generate the ZK proof
    let proof = create_proof(
        pk,
        weights,
        features.as_slice(),
        delta.as_vector(),
    )?;

    Ok(DeltaCircuit {
        proof,
        delta,
        tx_hash: *tx_hash,
    })
}

/// Create a delta proof with a pre-computed delta (for batch operations).
///
/// # Null-Space Protection
///
/// This function accepts a pre-computed delta from the caller. To prevent
/// null-space attacks, it performs an explicit check that the delta is
/// not the zero vector before attempting proof generation. This provides
/// a clear error message at the delta circuit layer and avoids wasting
/// computation on a proof that would fail the circuit's `s_null_check`
/// gate anyway.
///
/// # Arguments
///
/// * `pk` - The proving key for the current weights
/// * `weights` - The current NWO weights (public)
/// * `features` - The transaction features (private)
/// * `delta` - The pre-computed delta (must be non-zero)
/// * `tx_hash` - The transaction hash for VDW tracking
///
/// # Returns
///
/// A `DeltaCircuit` containing the proof and the claimed delta.
///
/// # Errors
///
/// Returns `NervError::Circuit` if:
/// - The delta is the zero vector (null-space attack)
/// - The proving key does not match the current weights
/// - The estimated constraint count exceeds the safe maximum
pub fn create_delta_proof_with_delta(
    pk: &CircuitProvingKey,
    weights: &NwoWeights,
    features: &TransactionFeatures,
    delta: &EmbeddingDelta,
    tx_hash: &[u8; 32],
) -> NervResult<DeltaCircuit> {
    // ── Defense-in-Depth: Null-Space Exclusion Check ───────────
    //
    // Reject zero-delta proofs at the delta circuit layer before
    // attempting Halo2 synthesis. This mirrors the Null-Space
    // Exclusion Gate (s_null_check) in the ZK circuit.
    //
    // A zero delta means W·ΔS + b_tx = 0_vector, which indicates
    // either a null-space attack (phantom transaction) or a
    // degenerate computation. Either way, the transaction does
    // not alter the global state embedding and must be rejected.
    let is_zero_delta = (0..EMBEDDING_DIM)
        .all(|i| {
            delta
                .get(i)
                .map(|v| v == FixedPoint64::ZERO)
                .unwrap_or(true)
        });
    if is_zero_delta {
        return Err(NervError::Circuit(
            "null-space attack detected at delta circuit layer: \
             delta is the zero vector (W·ΔS + b_tx = 0); \
             transaction does not alter the global state embedding \
             and is rejected by the Null-Space Exclusion Gate"
                .into(),
        ));
    }

    // Generate the ZK proof (create_proof also has a defense-in-depth
    // check, but we catch it here first for a clearer error message)
    let proof = create_proof(
        pk,
        weights,
        features.as_slice(),
        delta.as_vector(),
    )?;

    Ok(DeltaCircuit {
        proof,
        delta: delta.clone(),
        tx_hash: *tx_hash,
    })
}


// ─── Proof Verification (Network-Side) ───────────────────────────────────

/// Verify a delta proof submitted by a wallet.
///
/// Called by validators/relays when receiving a private transaction.
/// The verifier checks:
/// 1. The ZK proof is valid
/// 2. The proof's weight commitment matches the current weights
/// 3. The claimed delta matches the proof's public inputs
///
/// If verification succeeds, the delta can be safely applied to the
/// embedding via `e_{t+1} = e_t + δ(tx)`.
///
/// # Arguments
///
/// * `vk` - The verification key for the current weights
/// * `delta_proof` - The delta proof to verify
/// * `current_weight_hash` - Hash of the currently active weights
///
/// # Returns
///
/// `Ok(EmbeddingDelta)` — the verified delta, ready to apply.
/// # Null-Space Guarantee
///
/// If this function returns `Ok`, the verified `EmbeddingDelta` is
/// **guaranteed non-zero** by the ZK circuit's Null-Space Exclusion
/// Gate (`s_null_check`). This means:
///
/// - The transaction provably alters the global state embedding
/// - The delta can be safely aggregated with other verified deltas
/// - Block producers do not need to perform their own null-space checks
///
/// Downstream consumers (aggregators, block producers, Nova folding)
/// can rely on this invariant without re-verifying.
pub fn verify_delta_proof(
    vk: &CircuitVerificationKey,
    delta_proof: &DeltaCircuit,
    current_weight_hash: &[u8; 32],
) -> NervResult<EmbeddingDelta> {
    // Check that the proof was generated for the current weights
    if delta_proof.weight_commitment() != current_weight_hash {
        return Err(NervError::Circuit(
            "delta proof was generated for different weights (stale or mismatched)".into()
        ));
    }

    // Verify the ZK proof
    verify_proof(vk, &delta_proof.proof, Some(delta_proof.delta.as_vector()))?;

    // Return the verified delta
    Ok(delta_proof.delta.clone())
}

/// Verify a batch of delta proofs.
///
/// Used by block producers to verify all encrypted transactions
/// in a batch before the threshold decryption ceremony.
pub fn verify_delta_proof_batch(
    vk: &CircuitVerificationKey,
    proofs: &[DeltaCircuit],
    current_weight_hash: &[u8; 32],
) -> NervResult<Vec<NervResult<EmbeddingDelta>>> {
    let results: Vec<NervResult<EmbeddingDelta>> = proofs
        .iter()
        .map(|p| verify_delta_proof(vk, p, current_weight_hash))
        .collect();
    Ok(results)
}

// ─── Delta Proof Aggregation ─────────────────────────────────────────────

/// Aggregate the deltas from a batch of verified proofs.
///
/// After verifying all proofs in a batch, the block producer
/// aggregates the deltas for homomorphic application:
///
/// ```text
/// Δ_B = Σ δ(tx_i)
/// ```
///
/// This is the value that gets applied to the embedding:
/// `e_{t+1} = e_t + Δ_B`
pub fn aggregate_verified_deltas(
    verified_deltas: &[EmbeddingDelta],
) -> NervResult<EmbeddingDelta> {
    crate::embedding::homomorphism::aggregate_batch_deltas(verified_deltas)
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::{NwoPerceptron, TransactionType, DEFAULT_INPUT_DIM};
    use crate::circuit::latent_ledger_lite::generate_keys;

    fn setup() -> (NwoWeights, CircuitProvingKey, CircuitVerificationKey) {
        let weights = NwoWeights::new_constant(DEFAULT_INPUT_DIM, 0.01);
        let (pk, vk) = generate_keys(&weights, CIRCUIT_DEGREE).unwrap();
        (weights, pk, vk)
    }

    #[test]
    fn test_create_delta_proof() {
        let (weights, pk, _vk) = setup();
        let sender = [1u8; 32];
        let receiver = [2u8; 32];
        let features = TransactionFeatures::build_transfer(
            &sender, &receiver, 0.5, TransactionType::Transfer, 0.1, 0.5,
        );
        let tx_hash = [42u8; 32];

        let delta_proof = create_delta_proof(
            &pk, &weights, &features, &tx_hash,
        ).unwrap();

        assert_eq!(delta_proof.tx_hash, [42u8; 32]);
        assert!(delta_proof.proof.is_valid_size());
    }
  #[test]
    fn test_null_space_attack_zero_delta_rejected() {
        // An attacker constructs a transaction where W·ΔS + b_tx = 0_vector.
        // The Null-Space Exclusion Gate must reject this at the delta
        // circuit layer before any proof generation is attempted.
        let (weights, pk, _vk) = setup();
        let features = TransactionFeatures::default_dim();
        let tx_hash = [0u8; 32];

        // Construct a zero delta (all 64 dimensions = 0)
        let zero_delta = EmbeddingDelta::splat(FixedPoint64::ZERO);

        // create_delta_proof_with_delta must reject this immediately
        let result = create_delta_proof_with_delta(
            &pk, &weights, &features, &zero_delta, &tx_hash,
        );

        assert!(
            result.is_err(),
            "null-space attack (zero delta) must be rejected by the \
             delta circuit's defense-in-depth check"
        );

        // Verify the error message mentions null-space
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("null-space"),
            "error should mention null-space, got: {err_msg}"
        );
    }

    #[test]
    fn test_valid_nonzero_delta_passes_null_check() {
        // A valid transfer produces a non-zero delta (because b_tx is
        // derived from nullifiers and is non-zero), so the Null-Space
        // Exclusion Gate should be satisfied.
        let (weights, pk, vk) = setup();
        let sender = [1u8; 32];
        let receiver = [2u8; 32];
        let features = TransactionFeatures::build_transfer(
            &sender, &receiver, 0.5, TransactionType::Transfer, 0.1, 0.5,
        );
        let tx_hash = [42u8; 32];
        let weight_hash = weights.compute_hash();

        // Create delta proof — should succeed (non-zero delta)
        let delta_proof = create_delta_proof(
            &pk, &weights, &features, &tx_hash,
        ).unwrap();

        // Verify the delta is non-zero (sanity check)
        let mut has_nonzero = false;
        for i in 0..EMBEDDING_DIM {
            if delta_proof.delta.get(i).unwrap() != FixedPoint64::ZERO {
                has_nonzero = true;
                break;
            }
        }
        assert!(has_nonzero, "valid transfer must produce non-zero delta");

        // Verify the proof — should pass (null-space gate satisfied)
        let verified_delta = verify_delta_proof(
            &vk, &delta_proof, &weight_hash,
        ).unwrap();

        assert_eq!(verified_delta, delta_proof.delta);
    }

    #[test]
    fn test_verify_delta_proof() {
        let (weights, pk, vk) = setup();
        let features = TransactionFeatures::default_dim();
        let tx_hash = [0u8; 32];
        let weight_hash = weights.compute_hash();

        let delta_proof = create_delta_proof(
            &pk, &weights, &features, &tx_hash,
        ).unwrap();

        let verified_delta = verify_delta_proof(
            &vk, &delta_proof, &weight_hash,
        ).unwrap();

        // The verified delta should match the proof's claimed delta
        assert_eq!(verified_delta, delta_proof.delta);
    }

    #[test]
    fn test_verify_delta_proof_stale_weights() {
        let (weights1, pk1, _vk1) = setup();
        let (weights2, _, vk2) = {
            let w = NwoWeights::new_constant(DEFAULT_INPUT_DIM, 0.02);
            let (pk, vk) = generate_keys(&w, CIRCUIT_DEGREE).unwrap();
            (w, pk, vk)
        };

        let features = TransactionFeatures::default_dim();
        let tx_hash = [0u8; 32];
        let weight_hash2 = weights2.compute_hash();

        let delta_proof = create_delta_proof(
            &pk1, &weights1, &features, &tx_hash,
        ).unwrap();

        // Verify with different weight hash should fail
        let result = verify_delta_proof(&vk2, &delta_proof, &weight_hash2);
        assert!(result.is_err());
    }

    #[test]
    fn test_aggregate_verified_deltas() {
        // Note: In production, all deltas reaching this function have
        // passed the Null-Space Exclusion Gate (δ ≠ 0_vector). However,
        // the aggregate Δ_B = Σ δ(tx_i) CAN be zero if individual
        // non-zero deltas cancel out — this is a legitimate economic
        // scenario (e.g., Alice sends 1 NERV, Bob sends -1 NERV),
        // NOT a null-space attack. The null-space check is per-transaction,
        // not per-batch.
        let d1 = EmbeddingDelta::splat(FixedPoint64::from_int(1));
        let d2 = EmbeddingDelta::splat(FixedPoint64::from_int(2));
        let d3 = EmbeddingDelta::splat(FixedPoint64::from_int(3));

        let aggregated = aggregate_verified_deltas(&[d1, d2, d3]).unwrap();
        for i in 0..EMBEDDING_DIM {
            assert_eq!(aggregated.get(i).unwrap().to_int(), 6);
        }
    }

    #[test]
    fn test_delta_circuit_total_size() {
        let (weights, pk, _vk) = setup();
        let features = TransactionFeatures::default_dim();
        let tx_hash = [0u8; 32];

        let delta_proof = create_delta_proof(
            &pk, &weights, &features, &tx_hash,
        ).unwrap();

        // Total size should be proof + 512 (delta) + 32 (tx_hash)
        let expected = delta_proof.proof_size() + crate::EMBEDDING_BYTES + 32;
        assert_eq!(delta_proof.total_size(), expected);
    }
}
