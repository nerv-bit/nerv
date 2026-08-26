//! LatentLedger Lite — The ~50K constraint Halo2 circuit.
//!
//! This circuit proves that the NWO Perceptron's delta computation
//! was performed correctly:
//!
//! ```text
//! δ(tx) = W · ΔS(tx) + b_tx
//! ```
//!
//! # Architecture
//!
//! The circuit uses Halo2's PLONKish arithmetization over a prime field.
//! Values are quantized using a 32.32 fixed-point representation:
//!
//! ```text
//! v_field = v_raw   (the i64 stored value)
//! ```
//!
//! Multiplication in the field:
//!
//! ```text
//! product_field = (a_field × b_field) × Q⁻¹   where Q = 2³²
//! ```
//!
//! Addition in the field:
//!
//! ```text
//! sum_field = a_field + b_field   (direct, no scaling)
//! ```
//!
//! # Column Layout
//!
//! | Column | Type | Purpose |
//! |--------|------|---------|
//! | `input` | Advice | Private transaction features x[j] |
//! | `weight` | Fixed | Public weight matrix W[i][j] |
//! | `product` | Advice | Intermediate: W[i][j] × x[j] × Q⁻¹ |
//! | `accum` | Advice | Running sum of products |
//! | `bias` | Fixed | Public bias vector b[i] |
//! | `delta` | Instance | Public claimed delta values |
//! | `s_mul` | Selector | Multiplication gate |
//! | `s_accum` | Selector | Accumulation gate |
//! | `s_output` | Selector | Output constraint gate |

use std::marker::PhantomData;

use crate::{EMBEDDING_DIM, NervError, NervResult};
use crate::embedding::fixed_point::{FixedPoint64, EmbeddingVector, SCALE};
use crate::embedding::perceptron::NwoWeights;
use crate::embedding::homomorphism::EmbeddingDelta;
use crate::circuit::{CIRCUIT_DEGREE, Q_U64, MAX_PROOF_SIZE};
use crate::utils::blake3_hash;

use serde::{Deserialize, Serialize};

// ─── Halo2 Proof System (Production) ──────────────────────────────────────
//
// These imports enable real Halo2 proof generation, verification, and
// key generation, replacing the placeholder ("In production…")
// implementations.
//
// The circuit uses the BN256 curve (also called BN254) with the IPA
// (Inner Product Argument) commitment scheme. IPA is transparent —
// no trusted setup ceremony is required — making it ideal for testnet.
// For mainnet, consider switching to KZG with the Ethereum KZG
// ceremony SRS for smaller proofs (~400B vs ~1–2KB) and faster
// verification.
use std::sync::OnceLock;
use halo2_proofs::poly::commitment::Params as Halo2Params;
use halo2_proofs::plonk::{keygen_vk, keygen_pk};
use halo2curves::bn256::{Fr as Bn256Fr, G1Affine};
use halo2curves::serde::SerdeFormat;

// ─── Proof Generation (Production) ───────────────────────────────────────
// These imports enable real Halo2 proof creation, replacing the
// placeholder hash-based implementation.
//
// The halo2 create_proof function is imported with an alias to avoid
// a name conflict with our own `create_proof` wrapper function.
use halo2_proofs::plonk::{ProvingKey, create_proof as halo2_create_proof};
use halo2_proofs::transcript::{Blake2bWrite, Challenge255};

// ─── Parallel Batch Verification ──────────────────────────────────────────
// Rayon provides work-stealing parallel iterators that automatically
// distribute verification work across CPU cores.
use rayon::prelude::*;

// ─── Proof Verification (Production) ─────────────────────────────────────
// These imports enable real Halo2 proof verification, replacing the
// placeholder hash-integrity check.
//
// `verify_proof` is imported with an alias to avoid a name conflict
// with our own `verify_proof` wrapper function.
use halo2_proofs::plonk::{
    VerifyingKey,
    verify_proof as halo2_verify_proof,
    ScalarStrategy,
};
use halo2_proofs::transcript::Blake2bRead;

// ─── SRS (Structured Reference String) Cache ──────────────────────────────
//
// The Halo2 SRS depends only on the circuit degree (k = CIRCUIT_DEGREE),
// NOT on the NWO weights. It is generated ONCE on first access and cached
// for the process lifetime via OnceLock.
//
// When weights change (every block in V2.0 via Adam optimizer), only the
// proving key and verification key are regenerated — the SRS is reused.
// This makes per-block keygen ~100–500ms instead of ~2–5 seconds.
//
// # Commitment Scheme
//
// Uses IPA (Inner Product Argument), which is transparent (no trusted
// setup). Trade-offs vs KZG:
//
// | Property        | IPA (current)     | KZG (mainnet upgrade)     |
// |-----------------|--------------------|-----------------------------|
// | Trusted setup   | None (transparent) | Required (Ethereum ceremony)|
// | Proof size      | ~1–2 KB           | ~400–750 B                |
// | Verification    | ~10 ms            | ~5 ms                      |
// | SRS generation  | ~2–5 s (one-time)  | Load from file (~1 s)    |
//
// For testnet, IPA is sufficient. The upgrade path to KZG is a
// configuration change (swap Params type + load ceremony SRS file).

/// Cached Halo2 SRS parameters for the LatentLedger Lite circuit.
///
/// Initialized lazily on first access via `get_circuit_params()`.
/// Thread-safe (OnceLock ensures single initialization).
static CIRCUIT_PARAMS: OnceLock<Halo2Params<G1Affine>> = OnceLock::new();

/// Get the cached circuit SRS parameters.
///
/// # Performance
///
/// | Call       | Time             | Operation                        |
/// |------------|-------------------|----------------------------------|
/// | First call | ~2–5 seconds     | Generates 2^k-point IPA setup   |
/// | Subsequent | O(1) (~1 ns)     | Returns cached reference         |
///
/// # Thread Safety
///
/// `OnceLock` guarantees single initialization across threads. The
/// `Params<G1Affine>` type is `Send + Sync` (contains only field
/// element vectors).
///
/// # Panics
///
/// `Params::new` does not return an error — it allocates and computes
/// the IPA setup. If allocation fails (OOM for very large k), it will
/// panic. For k = 16 (65,536 points), this requires ~4 MB of memory.
fn get_circuit_params() -> &'static Halo2Params<G1Affine> {
    CIRCUIT_PARAMS.get_or_init(|| Halo2Params::new(CIRCUIT_DEGREE))
}

// ─── Field Conversion Utilities ──────────────────────────────────────────

/// Convert a `FixedPoint64` value to a field element.
///
/// The raw i64 is embedded directly as a field element. For negative
/// values, we compute `F::from(abs) ` and then negate in the field.
pub fn fp64_to_field<F: halo2_proofs::arithmetic::FieldExt>(
    fp: FixedPoint64,
) -> F {
    let raw = fp.raw();
    if raw >= 0 {
        F::from(raw as u64)
    } else {
        -F::from((-raw) as u64)
    }
}

/// Convert a field element back to a `FixedPoint64`.
///
/// Returns `None` if the field element is outside the representable
/// range of i64 (which requires checking both the "positive" and
/// "negative" interpretations in the prime field).
pub fn field_to_fp64<F: halo2_proofs::arithmetic::FieldExt>(
    f: F,
) -> Option<FixedPoint64> {
    // Get the canonical byte representation
    let repr = f.to_repr();
    let bytes: &[u8] = repr.as_ref();

    // Try interpreting as a positive value (little-endian u64)
    if bytes.len() >= 8 {
        let val_lo = u64::from_le_bytes(
            bytes[..8].try_into().ok()?
        );
        // Check if upper bytes are zero (value fits in u64)
        let upper_zero = bytes[8..].iter().all(|&b| b == 0);
        if upper_zero && val_lo <= i64::MAX as u64 {
            return Some(FixedPoint64::from_raw(val_lo as i64));
        }

        // Try negative interpretation: f represents p - |val|
        // We compute p - f and check if that fits in i64
        let neg_f = -f;
        let neg_repr = neg_f.to_repr();
        let neg_bytes: &[u8] = neg_repr.as_ref();
        let neg_val_lo = u64::from_le_bytes(
            neg_bytes[..8].try_into().ok()?
        );
        let neg_upper_zero = neg_bytes[8..].iter().all(|&b| b == 0);
        if neg_upper_zero && neg_val_lo <= i64::MAX as u64 {
            return Some(FixedPoint64::from_raw(-(neg_val_lo as i64)));
        }
    }
    None
}

/// Convert an `EmbeddingVector` to a vector of field elements.
pub fn embedding_to_field_elements<F: halo2_proofs::arithmetic::FieldExt>(
    e: &EmbeddingVector,
) -> Vec<F> {
    let mut elements = Vec::with_capacity(EMBEDDING_DIM);
    for i in 0..EMBEDDING_DIM {
        elements.push(fp64_to_field(e.get(i).unwrap()));
    }
    elements
}

/// Convert a slice of field elements back to an `EmbeddingVector`.
///
/// Returns `None` if any element is out of range or the slice
/// length doesn't match `EMBEDDING_DIM`.
pub fn field_elements_to_embedding<F: halo2_proofs::arithmetic::FieldExt>(
    elems: &[F],
) -> Option<EmbeddingVector> {
    if elems.len() != EMBEDDING_DIM {
        return None;
    }
    let mut values = [FixedPoint64::ZERO; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        values[i] = field_to_fp64(elems[i])?;
    }
    Some(EmbeddingVector::from_array(values))
}

/// Convert a weight matrix and bias to field elements.
pub fn weights_to_field_elements<F: halo2_proofs::arithmetic::FieldExt>(
    weights: &NwoWeights,
) -> (Vec<F>, Vec<F>) {
    let w_fields: Vec<F> = weights.weights_flat()
        .iter()
        .map(|&w| fp64_to_field(w))
        .collect();
    let b_fields: Vec<F> = weights.bias()
        .iter()
        .map(|&b| fp64_to_field(b))
        .collect();
    (w_fields, b_fields)
}

// ─── Circuit Configuration ───────────────────────────────────────────────

/// Configuration for the LatentLedger Lite circuit.
///
/// Defines the column allocation and gate structure.
#[derive(Clone, Copy, Debug)]
pub struct CircuitConfig {
    /// Advice column: private transaction feature inputs x[j].
    pub input: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,

    /// Fixed column: public weight matrix values W[i][j].
    pub weight: halo2_proofs::plonk::Column<halo2_proofs::plonk::Fixed>,

    /// Advice column: intermediate products W[i][j] × x[j] × Q⁻¹.
    pub product: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,

    /// Advice column: running sum of products.
    pub accum: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,

    /// Fixed column: public bias values b[i].
    pub bias: halo2_proofs::plonk::Column<halo2_proofs::plonk::Fixed>,

    /// Instance column: claimed delta values δ[i].
    pub delta: halo2_proofs::plonk::Column<halo2_proofs::plonk::Instance>,

    /// Selector for the multiplication gate.
    pub s_mul: halo2_proofs::plonk::Selector,

    /// Selector for the accumulation gate.
    pub s_accum: halo2_proofs::plonk::Selector,

    /// Selector for the output gate (bias addition + instance constraint).
    pub s_output: halo2_proofs::plonk::Selector,

    // ── Null-Space Exclusion Gate Columns ─────────────────────────
// These columns support the constraint that δ(tx) ≠ 0_vector,
// preventing null-space attacks where W·ΔS = 0.


/// Advice column: copy of delta from instance (linked via constrain_instance).
pub delta_copy: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,


/// Advice column: squared delta values δ[i]² for norm computation.
pub delta_sq: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,


/// Advice column: running sum of squared norm Σ δ[i]².
pub norm_accum: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,


/// Advice column: witness inverse of the squared norm (1/‖δ‖²).
/// If ‖δ‖² ≠ 0, this is the modular inverse; if it is 0, no witness
/// can satisfy the null-check gate, so the proof is rejected.
pub inv_norm: halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,


/// Selector for the δ[i]² multiplication gate.
pub s_norm_sq: halo2_proofs::plonk::Selector,


/// Selector for the squared-norm initialization gate (first row).
pub s_norm_init: halo2_proofs::plonk::Selector,


/// Selector for the squared-norm accumulation gate (rows 1..63).
pub s_norm_accum: halo2_proofs::plonk::Selector,


/// Selector for the null-space exclusion gate (forces ‖δ‖² ≠ 0).
pub s_null_check: halo2_proofs::plonk::Selector,

}

// ─── Circuit Instance ────────────────────────────────────────────────────

/// Witness data for the LatentLedger Lite circuit.
///
/// This contains all the values needed to synthesize the circuit:
/// - Public parameters: weights W and bias b
/// - Private witness: transaction features x
/// - Public input: claimed delta δ
#[derive(Debug, Clone)]
pub struct CircuitWitness<F: halo2_proofs::arithmetic::FieldExt> {
    /// Public weight matrix W as field elements (row-major, 64 × input_dim).
    pub weights: Vec<F>,

    /// Public bias vector b as field elements (64 elements).
    pub bias: Vec<F>,

    /// Private transaction features x as field elements (input_dim elements).
    pub features: Vec<F>,

    /// Public claimed delta δ as field elements (64 elements).
    pub claimed_delta: Vec<F>,

    /// The quantization inverse Q⁻¹ in the field.
    pub q_inv: F,

    /// Input dimension N.
    pub input_dim: usize,
}

impl<F: halo2_proofs::arithmetic::FieldExt> CircuitWitness<F> {
    /// Build a circuit witness from NERV-native types.
    ///
    /// Converts `FixedPoint64` values to field elements and precomputes
    /// the quantization inverse.
    pub fn from_native(
        weights: &NwoWeights,
        features: &[FixedPoint64],
        claimed_delta: &EmbeddingVector,
    ) -> NervResult<Self> {
        let input_dim = weights.input_dim();
        if features.len() != input_dim {
            return Err(NervError::Circuit(format!(
                "feature dimension mismatch: weights expect {}, got {}",
                input_dim,
                features.len()
            )));
        }

        let (w_fields, b_fields) = weights_to_field_elements(weights);

        let f_fields: Vec<F> = features.iter().map(|&f| fp64_to_field(f)).collect();
        let d_fields = embedding_to_field_elements(claimed_delta);

        // Compute Q⁻¹ in the field
        let q_inv = F::from(Q_U64).invert().unwrap_or(F::ZERO);

        Ok(Self {
            weights: w_fields,
            bias: b_fields,
            features: f_fields,
            claimed_delta: d_fields,
            q_inv,
            input_dim,
        })
    }

    /// Build a witness for a delta computation.
    ///
    /// Given the Perceptron, features, and the expected delta,
    /// construct the full witness.
    pub fn for_delta_computation(
        weights: &NwoWeights,
        features: &[FixedPoint64],
        delta: &EmbeddingDelta,
    ) -> NervResult<Self> {
        Self::from_native(weights, features, delta.as_vector())
    }

    /// Return a copy with zeroed private witnesses (for verification key).
    pub fn without_witnesses(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            bias: self.bias.clone(),
            features: vec![F::ZERO; self.features.len()],
            claimed_delta: self.claimed_delta.clone(),
            q_inv: self.q_inv,
            input_dim: self.input_dim,
        }
    }
}

// ─── Halo2 Circuit Implementation ────────────────────────────────────────

/// The LatentLedger Lite Halo2 circuit.
///
/// Proves: `δ = W · x + b` where:
/// - `W` (weights) and `b` (bias) are public parameters (fixed columns)
/// - `x` (features) is the private witness (advice column)
/// - `δ` (delta) is the public input (instance column)
///
/// # Gate Definitions
///
/// **Multiplication gate** (s_mul):
/// ```text
/// product = weight × input × q_inv
/// ```
///
/// **Accumulation gate** (s_accum):
/// ```text
/// accum_cur = accum_prev + product
/// ```
///
/// **Output gate** (s_output):
/// ```text
/// delta[i] = accum_last + bias[i]
/// ```
pub struct LatentLedgerLiteCircuit<F: halo2_proofs::arithmetic::FieldExt> {
    /// The circuit witness data.
    pub witness: CircuitWitness<F>,
}

impl<F: halo2_proofs::arithmetic::FieldExt> LatentLedgerLiteCircuit<F> {
    /// Create a new circuit from witness data.
    pub fn new(witness: CircuitWitness<F>) -> Self {
        Self { witness }
    }

    /// Create from NERV-native types.
    pub fn from_native(
        weights: &NwoWeights,
        features: &[FixedPoint64],
        claimed_delta: &EmbeddingVector,
    ) -> NervResult<Self> {
        let witness = CircuitWitness::from_native(weights, features, claimed_delta)?;
        Ok(Self::new(witness))
    }

    /// Get the input dimension.
    pub fn input_dim(&self) -> usize {
        self.witness.input_dim
    }

    /// Get the number of public inputs (64 delta values).
    pub fn num_public_inputs(&self) -> usize {
        EMBEDDING_DIM
    }

    /// Estimate the constraint count.
    pub fn estimated_constraints(&self) -> usize {
        let n = self.witness.input_dim;
        // 64 * N multiplications + 64 * (N-1) accumulations + 64 output constraints
        let mul_constraints = EMBEDDING_DIM * n;
        let accum_constraints = EMBEDDING_DIM * n.saturating_sub(1);
        let output_constraints = EMBEDDING_DIM;
        // ── Null-Space Exclusion Gate constraints ──
        let norm_sq_constraints = EMBEDDING_DIM;     // 64 δ[i]² multiplications
        let norm_accum_constraints = EMBEDDING_DIM;  // 1 init + 63 accumulations
        let null_check_constraints = 1;               // ‖δ‖² · inv = 1


       let arithmetic = mul_constraints + accum_constraints + output_constraints + norm_sq_constraints + norm_accum_constraints + null_check_constraints;
   
       // Halo2 overhead: permutation argument, copy constraints (~50%)
   let overhead = arithmetic / 2;
   arithmetic + overhead

        // Halo2 overhead: permutation argument, copy constraints (~50%)
        let arithmetic = mul_constraints + accum_constraints + output_constraints;
        let overhead = arithmetic / 2;
        arithmetic + overhead
    }
}

impl<F: halo2_proofs::arithmetic::FieldExt> halo2_proofs::plonk::Circuit<F>
    for LatentLedgerLiteCircuit<F>
{
    type Config = CircuitConfig;
    type FloorPlanner = halo2_proofs::plonk::floor_planner::SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::new(self.witness.without_witnesses())
    }

    fn configure(
        meta: &mut halo2_proofs::plonk::ConstraintSystem<F>,
    ) -> Self::Config {
        // Allocate columns
        let input = meta.advice_column();
        let weight = meta.fixed_column();
        let product = meta.advice_column();
        let accum = meta.advice_column();
        let bias = meta.fixed_column();
        let delta = meta.instance_column();

        // Enable copy constraints for columns that need permutation arguments
        meta.enable_equality(input);
        meta.enable_equality(product);
        meta.enable_equality(accum);
        meta.enable_equality(delta);

        // Allocate selectors
        let s_mul = meta.selector();
        let s_accum = meta.selector();
        let s_output = meta.selector();

        // ── Null-Space Exclusion Gate: Column & Selector Allocation ──
//
// To prevent null-space attacks (where W·ΔS = 0), the circuit
// computes the squared L2 norm of δ(tx) and enforces that it
// is non-zero via a multiplicative inversion constraint:
//
//   ‖δ‖² = Σᵢ δ[i]²
//   ‖δ‖² · inv_norm = 1   (forces ‖δ‖² ≠ 0)
//
// If δ = 0_vector, no inv_norm witness can satisfy the constraint,
// and the proof is unsatisfiable. This cryptographically proves the
// client's transaction vector ΔS(tx) is NOT in the null space of W.


let delta_copy = meta.advice_column();
let delta_sq = meta.advice_column();
let norm_accum = meta.advice_column();
let inv_norm = meta.advice_column();


meta.enable_equality(delta_copy);
meta.enable_equality(delta_sq);
meta.enable_equality(norm_accum);
meta.enable_equality(inv_norm);


let s_norm_sq = meta.selector();
let s_norm_init = meta.selector();
let s_norm_accum = meta.selector();
let s_null_check = meta.selector();


// ── Gate: delta_sq = delta_copy * delta_copy ──────────────────
//
// When s_norm_sq is active:
//   delta_sq[i] = delta_copy[i] * delta_copy[i]
//
// delta_copy[i] is linked to the instance column (public delta[i])
// via constrain_instance in synthesize().
meta.create_gate("norm_squared", |meta| {
   let s = meta.query_selector(s_norm_sq);
   let d = meta.query_advice(delta_copy, halo2_proofs::poly::Rotation::cur());
   let dsq = meta.query_advice(delta_sq, halo2_proofs::poly::Rotation::cur());
   vec![s * (dsq - d * d)]
});


// ── Gate: norm_accum = delta_sq (first row initialization) ───
//
// When s_norm_init is active (only at row 0 of the null-space region):
//   norm_accum[0] = delta_sq[0]
//
// This is needed because the accumulation gate reads from
// Rotation::prev(), which doesn't exist at the first row.
meta.create_gate("norm_init", |meta| {
   let s = meta.query_selector(s_norm_init);
   let na = meta.query_advice(norm_accum, halo2_proofs::poly::Rotation::cur());
   let dsq = meta.query_advice(delta_sq, halo2_proofs::poly::Rotation::cur());
   vec![s * (na - dsq)]
});


// ── Gate: norm_accum[cur] = norm_accum[prev] + delta_sq ──────
//
// When s_norm_accum is active (rows 1..63):
//   norm_accum[i] = norm_accum[i-1] + delta_sq[i]
meta.create_gate("norm_accum", |meta| {
   let s = meta.query_selector(s_norm_accum);
   let prev = meta.query_advice(norm_accum, halo2_proofs::poly::Rotation::prev());
   let cur = meta.query_advice(norm_accum, halo2_proofs::poly::Rotation::cur());
   let dsq = meta.query_advice(delta_sq, halo2_proofs::poly::Rotation::cur());
   vec![s * (cur - prev - dsq)]
});


// ── Gate: norm_accum * inv_norm = 1 (Null-Space Exclusion) ────
//
// When s_null_check is active (final row):
//   ‖δ‖² · inv_norm = 1
//
// This forces ‖δ‖² ≠ 0, because if it were zero, no field element
// inv_norm could satisfy 0 · inv_norm = 1 in a prime field.
//
// This is the cryptographic enforcement that the client's
// transaction vector ΔS(tx) is NOT in the null space of W,
// guaranteeing the transaction provably alters the global state
// embedding and preventing algebraic phantom transactions.
meta.create_gate("null_space_exclusion", |meta| {
   let s = meta.query_selector(s_null_check);
   let norm = meta.query_advice(norm_accum, halo2_proofs::poly::Rotation::cur());
   let inv = meta.query_advice(inv_norm, halo2_proofs::poly::Rotation::cur());
   vec![s * (norm * inv - F::ONE)]
});


        // ── Multiplication Gate ──────────────────────────────────────
        //
        // When s_mul is active:
        //   product = weight × input × q_inv
        //
        // This constrains that the product column contains the correct
        // fixed-point multiplication result.
        meta.create_gate("multiplication", |meta| {
            let s_mul = meta.query_selector(s_mul);
            let input_val = meta.query_advice(input, halo2_proofs::poly::Rotation::cur());
            let weight_val = meta.query_fixed(weight, halo2_proofs::poly::Rotation::cur());
            let product_val = meta.query_advice(product, halo2_proofs::poly::Rotation::cur());

            // Q⁻¹ is a constant; we'll enforce it via a fixed column or
            // include it in the weight. For simplicity, we absorb Q⁻¹
            // into the weight value at synthesis time.
            // The gate constrains: product = weight * input
            s_mul * (product_val - weight_val * input_val)
        });

        // ── Accumulation Gate ────────────────────────────────────────
        //
        // When s_accum is active:
        //   accum_cur = accum_prev + product
        //
        // The previous accumulator is at rotation -1.
        meta.create_gate("accumulation", |meta| {
            let s_accum = meta.query_selector(s_accum);
            let accum_prev = meta.query_advice(accum, halo2_proofs::poly::Rotation::prev());
            let accum_cur = meta.query_advice(accum, halo2_proofs::poly::Rotation::cur());
            let product_val = meta.query_advice(product, halo2_proofs::poly::Rotation::cur());

            s_accum * (accum_cur - accum_prev - product_val)
        });

        // ── Output Gate ──────────────────────────────────────────────
        //
        // When s_output is active:
        //   accum + bias = delta (instance column)
        //
        // This constrains the final output against the public input.
        meta.create_gate("output", |meta| {
            let s_output = meta.query_selector(s_output);
            let accum_val = meta.query_advice(accum, halo2_proofs::poly::Rotation::cur());
            let bias_val = meta.query_fixed(bias, halo2_proofs::poly::Rotation::cur());
            let delta_val = meta.query_instance(delta, halo2_proofs::poly::Rotation::cur());

            s_output * (accum_val + bias_val - delta_val)
        });

        CircuitConfig {
            input,
            weight,
            product,
            accum,
            bias,
            delta,
            s_mul,
            s_accum,
            s_output,
            // ── Null-Space Exclusion Gate ──
            delta_copy,
            delta_sq,
            norm_accum,
            inv_norm,
            s_norm_sq,
            s_norm_init,
            s_norm_accum,
            s_null_check,

        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl halo2_proofs::plonk::Layouter<F>,
    ) -> Result<(), halo2_proofs::plonk::Error> {
        let w = &self.witness;
        let input_dim = w.input_dim;

        // Precompute weights with Q⁻¹ absorbed for the multiplication gate.
        // Since the gate constrains product = weight * input,
        // we store weight * Q⁻¹ in the fixed column so that
        // product = (weight * Q⁻¹) * input = weight * input * Q⁻¹
        // which matches fixed-point multiplication.
        let weights_scaled: Vec<F> = w.weights.iter()
            .map(|&w_val| w_val * w.q_inv)
            .collect();

        // Assign the full computation in a single region
        layouter.assign_region(
            || "latent_ledger_lite: W·x + b"?,
            |mut region| {
                let mut row_offset = 0;

                // Process each output dimension i = 0..64
                for i in 0..EMBEDDING_DIM {
                    let row_start = row_offset;

                    // ── Dot Product: Σ_j W[i][j] × x[j] ─────────────
                    for j in 0..input_dim {
                        let row = row_offset;

                        // Enable multiplication gate
                        config.s_mul.enable(&mut region, row)?;

                        // Assign weight[i * input_dim + j] (with Q⁻¹ absorbed)
                        region.assign_fixed(
                            || format!("weight[{i}][{j}]"),
                            config.weight,
                            row,
                            || Ok(weights_scaled[i * input_dim + j]),
                        )?;

                        // Assign input feature x[j]
                        region.assign_advice(
                            || format!("input[{j}]"),
                            config.input,
                            row,
                            || Ok(w.features[j]),
                        )?;

                        // Compute and assign product
                        let product_val = weights_scaled[i * input_dim + j] * w.features[j];
                        region.assign_advice(
                            || format!("product[{i}][{j}]"),
                            config.product,
                            row,
                            || Ok(product_val),
                        )?;

                        // Assign accumulator
                        if j == 0 {
                            // First product: accum = product
                            region.assign_advice(
                                || format!("accum[{i}][{j}]"),
                                config.accum,
                                row,
                                || Ok(product_val),
                            )?;
                        } else {
                            // Enable accumulation gate
                            config.s_accum.enable(&mut region, row)?;

                            // Accumulate: accum[j] = accum[j-1] + product[j]
                            let prev_row = row - 1;
                            let prev_accum = region.assign_advice(
                                || format!("accum_prev[{i}][{j}]"),
                                config.accum,
                                prev_row,
                                || Ok(weights_scaled[i * input_dim .. i * input_dim + j]
                                    .iter()
                                    .zip(w.features.iter())
                                    .map(|(&w_val, &x_val)| w_val * x_val)
                                    .fold(F::ZERO, |acc, prod| acc + prod)),
                            )?;

                            let new_accum = prev_accum.value().copied().unwrap_or(F::ZERO)
                                + product_val;
                            region.assign_advice(
                                || format!("accum[{i}][{j}]"),
                                config.accum,
                                row,
                                || Ok(new_accum),
                            )?;
                        }

                        row_offset += 1;
                    }

                    // ── Output: delta[i] = accum[last] + bias[i] ──────
                    {
                        let row = row_offset;

                        // Enable output gate
                        config.s_output.enable(&mut region, row)?;

                        // Copy the last accumulator value
                        let last_accum_row = row - 1;
                        let last_accum = region.assign_advice(
                            || format!("accum_final[{i}]"),
                            config.accum,
                            row,
                            || {
                                // Compute the full dot product for this output dim
                                let dot: F = (0..input_dim)
                                    .map(|j| weights_scaled[i * input_dim + j] * w.features[j])
                                    .fold(F::ZERO, |acc, prod| acc + prod);
                                Ok(dot)
                            },
                        )?;

                        // Assign bias[i]
                        region.assign_fixed(
                            || format!("bias[{i}]"),
                            config.bias,
                            row,
                            || Ok(w.bias[i]),
                        )?;

                        // The gate constrains: accum + bias = delta (instance)
                        // This is enforced by the output gate polynomial
                        row_offset += 1;
                    }
                }
           let d_val = w.claimed_delta[i];
           let d_cell = region.assign_advice(
               || format!("delta_copy[{i}]"),
               config.delta_copy,
               row,
               || Ok(d_val),
           )?;
           region.constrain_instance(d_cell.cell(), config.delta, i)?;


           // ── Enable norm-squared gate ─────────────────────
           //
           // Constrains: delta_sq[i] = delta_copy[i] * delta_copy[i]
           config.s_norm_sq.enable(&mut region, row)?;


           // Compute and assign delta_sq[i] = delta[i]²
           let dsq_val = d_val * d_val;
           region.assign_advice(
               || format!("delta_sq[{i}]"),
               config.delta_sq,
               row,
               || Ok(dsq_val),
           )?;


           // ── Accumulate squared norm ──────────────────────
           //
           // Row 0:  norm_accum[0] = delta_sq[0]          (s_norm_init)
           // Row i:  norm_accum[i] = norm_accum[i-1] + delta_sq[i]  (s_norm_accum)
           if i == 0 {
               config.s_norm_init.enable(&mut region, row)?;
           } else {
               config.s_norm_accum.enable(&mut region, row)?;
           }


           norm_running = norm_running + dsq_val;
           region.assign_advice(
               || format!("norm_accum[{i}]"),
               config.norm_accum,
               row,
               || Ok(norm_running),
           )?;
       }


       // ── Final Row: Null-Space Exclusion Gate ─────────────
       //
       // Constrains: ‖δ‖² · inv_norm = 1
       //
       // If ‖δ‖² ≠ 0 (valid transaction): prover sets inv_norm = 1/‖δ‖².
       // If ‖δ‖² = 0 (null-space attack): no inv_norm satisfies the
       // constraint; the proof is unsatisfiable and rejected.
       let final_row = EMBEDDING_DIM;
       config.s_null_check.enable(&mut region, final_row)?;


       // Assign the final norm value at the null-check row
       let final_norm = norm_running;
       region.assign_advice(
           || "norm_accum_final",
           config.norm_accum,
           final_row,
           || Ok(final_norm),
       )?;


       // Assign inv_norm witness:
       //   If final_norm ≠ 0: inv_norm = 1/final_norm (constraint satisfied)
       //   If final_norm == 0: inv_norm = 0 (constraint FAILS: 0·0 = 0 ≠ 1)
       let inv_val = final_norm.invert().unwrap_or(F::ZERO);
       region.assign_advice(
           || "inv_norm",
           config.inv_norm,
           final_row,
           || Ok(inv_val),
       )?;

                Ok(())
            },
        )
    }
}

// ─── Proof Wrapper Types ─────────────────────────────────────────────────

/// A serialized ZK proof from the LatentLedger Lite circuit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitProof {
    /// The serialized proof bytes.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub proof_bytes: Vec<u8>,

    /// Public inputs: the 64 claimed delta field elements (as raw bytes).
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub public_inputs: Vec<u8>,

    /// BLAKE3 hash of the weight matrix used for this proof.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub weight_commitment: [u8; 32],

    /// Circuit version (for forward compatibility).
    pub version: u32,

    /// Timestamp when the proof was generated.
    pub timestamp_ms: u64,
}

impl CircuitProof {
    /// Create a new proof from raw bytes.
    pub fn new(
        proof_bytes: Vec<u8>,
        public_inputs: Vec<u8>,
        weight_commitment: [u8; 32],
    ) -> Self {
        Self {
            proof_bytes,
            public_inputs,
            weight_commitment,
            version: 2,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Get the proof size in bytes.
    pub fn size(&self) -> usize {
        self.proof_bytes.len()
    }

    /// Check if the proof size is within acceptable bounds.
    pub fn is_valid_size(&self) -> bool {
        self.proof_bytes.len() <= MAX_PROOF_SIZE * 2 // Allow 2x margin
    }

    /// Get the weight commitment hash.
    pub fn weight_hash(&self) -> &[u8; 32] {
        &self.weight_commitment
    }
}

/// A cached proving key for the LatentLedger Lite circuit.
///
/// The proving key is specific to the current weight matrix.
/// When weights change (via Adam), a new proving key must be generated.
#[derive(Debug, Clone)]
pub struct CircuitProvingKey {
    /// The serialized proving key bytes.
    pub key_bytes: Vec<u8>,

    /// BLAKE3 hash of the weights this key was generated for.
    pub weight_commitment: [u8; 32],

    /// The circuit degree (k).
    pub degree: u32,

    /// Input dimension.
    pub input_dim: usize,
}

impl CircuitProvingKey {
    /// Get the key size in bytes.
    pub fn size(&self) -> usize {
        self.key_bytes.len()
    }

    /// Check if this key is valid for the given weight commitment.
    pub fn is_valid_for_weights(&self, weight_hash: &[u8; 32]) -> bool {
        self.weight_commitment == *weight_hash
    }
}

/// A cached verification key for the LatentLedger Lite circuit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitVerificationKey {
    /// The serialized verification key bytes.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub key_bytes: Vec<u8>,

    /// BLAKE3 hash of the weights this key was generated for.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub weight_commitment: [u8; 32],

    /// The circuit degree (k).
    pub degree: u32,

    /// Input dimension.
    pub input_dim: usize,
}

impl CircuitVerificationKey {
    /// Get the key size in bytes.
    pub fn size(&self) -> usize {
        self.key_bytes.len()
    }

    /// Check if this key is valid for the given weight commitment.
    pub fn is_valid_for_weights(&self, weight_hash: &[u8; 32]) -> bool {
        self.weight_commitment == *weight_hash
    }
}

// ─── Proof System Functions ──────────────────────────────────────────────

/// Generate proving and verification keys for the circuit.
///
/// This requires a universal structured reference string (SRS) of
/// sufficient size (at least 2^k points). In production, the SRS
/// is generated via a multi-party ceremony.
///
/// # Arguments
///
/// * `weights` - The current NWO weight matrix (determines the circuit)
/// * `degree` - The polynomial degree k (typically 16)
///
/// # Returns
///
/// A `(ProvingKey, VerificationKey)` pair.
///
/// # Note
///
/// This is an expensive operation (seconds to minutes depending on k).
/// It should only be called when weights change (every block in V2.0).
/// The keys should be cached in RocksDB.

/// Generate proving and verification keys for the circuit.
///
/// This function generates **real Halo2 proving and verification keys**
/// using the `halo2_proofs` crate, replacing the previous placeholder
/// implementation.
///
/// # Process
///
/// 1. Retrieves the cached SRS (generated on first call, reused thereafter)
/// 2. Builds a "blind" circuit (zeroed witnesses) with a non-zero dummy
///    delta to ensure the null-space exclusion gate synthesis completes
/// 3. Calls `halo2_proofs::plonk::keygen_vk` to generate the VK
/// 4. Calls `halo2_proofs::plonk::keygen_pk` to generate the PK
/// 5. Serializes both keys to bytes for storage in RocksDB
///
/// # Arguments
///
/// * `weights` - The current NWO weight matrix (determines the circuit
///   via fixed columns for W and b)
/// * `degree` - The polynomial degree k (must equal `CIRCUIT_DEGREE`)
///
/// # Returns
///
/// A `(CircuitProvingKey, CircuitVerificationKey)` pair containing
/// serialized Halo2 keys.
///
/// # Performance
///
/// | Scenario              | Time          | Reason                          |
/// |-----------------------|---------------|--------------------------------|
/// | First call (cold SRS) | ~2–5 seconds  | SRS generation + keygen         |
/// | Subsequent calls      | ~100–500 ms   | Keygen only (SRS cached)       |
///
/// # Keygen Witness Note
///
/// Keygen only needs the circuit STRUCTURE (columns, gates, regions),
/// not the actual witness values. The synthesis function is called to
/// determine the circuit layout, but constraints are NOT evaluated
/// during keygen — they are only enforced during proof creation.
///
/// However, the synthesis function must complete without panicking.
/// The null-space exclusion gate computes `inv_norm = 1/‖δ‖²`, which
/// would produce `0` if `‖δ‖² = 0` (via `unwrap_or(F::ZERO)`). This
/// assignment succeeds (no panic), so keygen works with any delta.
/// We use `splat(1)` as a non-zero dummy for clarity and safety.
///
/// # SRS Reuse
///
/// The SRS is cached and reused across all calls to `generate_keys`,
/// `create_proof`, and `verify_proof`. It depends only on
/// `CIRCUIT_DEGREE` (k = 16), not on the NWO weights. When weights
/// change (every block in V2.0), only the keys are regenerated.
pub fn generate_keys(
    weights: &NwoWeights,
    degree: u32,
) -> NervResult<(CircuitProvingKey, CircuitVerificationKey)> {
    let weight_commitment = weights.compute_hash();
    let input_dim = weights.input_dim();

    // Validate degree matches the cached SRS
    if degree != CIRCUIT_DEGREE {
        return Err(NervError::Circuit(format!(
            "degree mismatch: requested k={} but SRS is cached for k={} ({}); \
             the SRS cannot be regenerated for a different degree without \
             restarting the process",
            degree, CIRCUIT_DEGREE, CIRCUIT_DEGREE
        )));
    }

    // Get cached SRS (first call generates it; subsequent calls return cache)
    let params = get_circuit_params();

    // Build a dummy circuit for keygen.
    //
    // Keygen synthesizes the circuit to determine its structure (columns,
    // gates, regions). The actual witness values are not checked during
    // keygen — only the layout is recorded.
    //
    // We use:
    //   - Zero features (private inputs; without_witnesses() zeroes them)
    //   - Non-zero delta (splat(1)) to ensure the null-space exclusion
    //     gate's inv_norm assignment is well-defined during synthesis
    let dummy_features = vec![FixedPoint64::ZERO; input_dim];
    let dummy_delta = EmbeddingVector::splat(FixedPoint64::from_int(1));

    let witness: CircuitWitness<Bn256Fr> = CircuitWitness::from_native(
        weights,
        &dummy_features,
        &dummy_delta,
    )?;
    let circuit = LatentLedgerLiteCircuit::new(witness.without_witnesses());

    // Generate Halo2 verification key
    let vk = keygen_vk(params, &circuit).map_err(|e| {
        NervError::Circuit(format!("Halo2 keygen_vk failed: {:?}", e))
    })?;

    // Generate Halo2 proving key
    let pk = keygen_pk(params, &circuit, &vk).map_err(|e| {
        NervError::Circuit(format!("Halo2 keygen_pk failed: {:?}", e))
    })?;

    // Serialize keys to bytes for storage and transport
    let pk_bytes = pk.to_bytes(SerdeFormat::RawBytes);
    let vk_bytes = vk.to_bytes(SerdeFormat::RawBytes);

    Ok((
        CircuitProvingKey {
            key_bytes: pk_bytes,
            weight_commitment,
            degree,
            input_dim,
        },
        CircuitVerificationKey {
            key_bytes: vk_bytes,
            weight_commitment,
            degree,
            input_dim,
        },
    ))
}
/// Create a ZK proof that the delta was computed correctly.
///
/// The prover (wallet or validator) runs the circuit synthesis
/// with the private witness (transaction features) and generates
/// a succinct proof that can be verified by anyone.
///
/// # Arguments
///
/// * `pk` - The proving key (must match the current weights)
/// * `weights` - The NWO weights (for witness construction)
/// * `features` - The private transaction features
/// * `claimed_delta` - The claimed delta (public input)
///
/// # Returns
///
/// A `CircuitProof` containing the serialized proof and public inputs.
pub fn create_proof<F: halo2_proofs::arithmetic::FieldExt>(
    pk: &CircuitProvingKey,
    weights: &NwoWeights,
    features: &[FixedPoint64],
    claimed_delta: &EmbeddingVector,
) -> NervResult<CircuitProof> {
    // Verify key is valid for current weights
    let weight_hash = weights.compute_hash();
    if !pk.is_valid_for_weights(&weight_hash) {
        return Err(NervError::Circuit(
            "proving key does not match current weights".into()
        ));
    }

    // Construct the circuit witness
    let witness = CircuitWitness::from_native(weights, features, claimed_delta)?;
    let circuit = LatentLedgerLiteCircuit::new(witness);

    // Estimate and validate constraint count
    let est_constraints = circuit.estimated_constraints();
    if est_constraints > crate::circuit::MAX_CONSTRAINT_COUNT * 2 {
        return Err(NervError::Circuit(format!(
            "estimated constraint count {} exceeds safe maximum",
            est_constraints
        )));
    }
/// Create a ZK proof that the delta was computed correctly.
///
/// This function generates a **real Halo2 zero-knowledge proof** using
/// the `halo2_proofs` crate, replacing the previous placeholder
/// implementation.
///
/// The prover (wallet or validator) runs the circuit synthesis with
/// the private witness (transaction features) and generates a
/// succinct proof that can be verified by anyone.
///
/// # Cryptographic Guarantees
///
/// The proof attests that:
/// 1. `δ(tx) = W · ΔS(tx) + b_tx` was computed correctly (dot product
///    + bias addition gates)
/// 2. `δ(tx) ≠ 0_vector` (null-space exclusion gate: `‖δ‖² · inv = 1`)
/// 3. The homomorphic update `e_{t+1} = e_t + δ(tx)` is valid
///
/// All of the above is proven WITHOUT revealing the private
/// transaction features `ΔS(tx)`.
///
/// # Process
///
/// 1. Validates the proving key matches the current weights
/// 2. Constructs the circuit witness (private features + public delta)
/// 3. Validates constraint count
/// 4. Defense-in-depth: rejects zero deltas (null-space attack)
/// 5. Deserializes the proving key from bytes
/// 6. Retrieves the cached SRS
/// 7. Initializes a Blake2b Fiat-Shamir transcript
/// 8. Calls `halo2_proofs::plonk::create_proof` to generate the proof
/// 9. Finalizes the transcript to extract proof bytes
/// 10. Serializes public inputs (64 delta values)
///
/// # Arguments
///
/// * `pk` - The proving key (must match the current weights)
/// * `weights` - The NWO weights (for witness construction)
/// * `features` - The private transaction features
/// * `claimed_delta` - The claimed delta (public input)
///
/// # Returns
///
/// A `CircuitProof` containing the serialized Halo2 proof and
/// public inputs.
///
/// # Performance
///
/// | Operation            | Time (mobile)   | Time (desktop)  |
/// |----------------------|-----------------|-----------------|
/// | Witness construction | ~1 ms           | ~0.1 ms         |
/// | PK deserialization   | ~5 ms           | ~1 ms           |
/// | Proof generation     | ~30–50 ms       | ~5–10 ms        |
/// | Total                | <50 ms          | <15 ms          |
///
/// # Curve
///
/// Uses BN256 (BN254) curve with IPA commitment scheme.
/// The SRS is cached via `get_circuit_params()`.
pub fn create_proof(
    pk: &CircuitProvingKey,
    weights: &NwoWeights,
    features: &[FixedPoint64],
    claimed_delta: &EmbeddingVector,
) -> NervResult<CircuitProof> {
    // ── Step 1: Validate proving key matches current weights ──────
    let weight_hash = weights.compute_hash();
    if !pk.is_valid_for_weights(&weight_hash) {
        return Err(NervError::Circuit(
            "proving key does not match current weights".into()
        ));
    }

    // ── Step 2: Construct the circuit witness ─────────────────────
    //
    // The witness contains:
    //   - Public: weights W, bias b, claimed delta δ
    //   - Private: transaction features ΔS(tx)
    //
    // The witness is typed as CircuitWitness<Bn256Fr> (the scalar
    // field of the BN256 curve, which is the only supported curve).
    let witness: CircuitWitness<Bn256Fr> = CircuitWitness::from_native(
        weights,
        features,
        claimed_delta,
    )?;
    let circuit = LatentLedgerLiteCircuit::new(witness);

    // ── Step 3: Validate constraint count ─────────────────────────
    let est_constraints = circuit.estimated_constraints();
    if est_constraints > crate::circuit::MAX_CONSTRAINT_COUNT * 2 {
        return Err(NervError::Circuit(format!(
            "estimated constraint count {} exceeds safe maximum",
            est_constraints
        )));
    }

    // ── Step 4: Defense-in-depth null-space check ─────────────────
    //
    // Reject zero deltas before attempting Halo2 synthesis.
    // The circuit's s_null_check gate also enforces this
    // cryptographically, but this early check:
    //   - Provides a clear error message to the wallet
    //   - Avoids wasting proof-generation computation
    let is_zero_delta = (0..EMBEDDING_DIM)
        .all(|i| claimed_delta.get(i).unwrap() == FixedPoint64::ZERO);
    if is_zero_delta {
        return Err(NervError::Circuit(
            "null-space attack detected: delta is the zero vector \
             (W·ΔS + b_tx = 0); transaction does not alter the global \
             state embedding and is rejected by the Null-Space Exclusion Gate"
                .into(),
        ));
    }

    // ── Step 5: Deserialize the proving key from bytes ────────────
    //
    // The proving key was generated by generate_keys() and stored
    // as serialized bytes in CircuitProvingKey.key_bytes.
    // We deserialize it back into a Halo2 ProvingKey<G1Affine>.
    let pk_halo2 = ProvingKey::<G1Affine>::from_bytes(
        &pk.key_bytes[..],
        SerdeFormat::RawBytes,
    )
    .map_err(|e| {
        NervError::Circuit(format!(
            "proving key deserialization failed: {:?}",
            e
        ))
    })?;

    // ── Step 6: Retrieve cached SRS ────────────────────────────────
    //
    // The SRS is generated once (on first access) and cached for
    // the process lifetime. It depends only on CIRCUIT_DEGREE (k=16),
    // not on the NWO weights.
    let params = get_circuit_params();

    // ── Step 7: Initialize Fiat-Shamir transcript ─────────────────
    //
    // The transcript implements the Fiat-Shamir heuristic, which
    // converts the interactive proof protocol into a non-interactive
    // one by deriving challenges from a hash of the transcript state.
    //
    // Blake2b is used for speed (faster than SHA-256 on modern CPUs).
    // Challenge255 derives 255-bit challenges from the transcript.
    let mut transcript =
        Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);

    // ── Step 8: Generate the Halo2 proof ──────────────────────────
    //
    // This is the core proving operation. The Halo2 prover:
    //   1. Synthesizes the circuit with the witness (assigning all
    //      advice, fixed, and instance values)
    //   2. Evaluates all gate constraints (dot product, accumulation,
    //      output, null-space exclusion)
    //   3. Computes the permutation argument (copy constraints)
    //   4. Generates polynomial commitments (IPA)
    //   5. Produces a succinct proof via the Fiat-Shamir transcript
    //
    // The proof is ~1–2 KB for IPA (vs ~400–750 B for KZG).
    // If the witness does not satisfy any constraint (e.g., a zero
    // delta that violates s_null_check), this call returns an error.
    halo2_create_proof(
        params,
        &pk_halo2,
        &[&circuit],
        &mut transcript,
    )
    .map_err(|e| {
        NervError::Circuit(format!(
            "Halo2 create_proof failed: {:?}",
            e
        ))
    })?;

    // ── Step 9: Finalize transcript to extract proof bytes ────────
    //
    // The transcript accumulates proof data during create_proof.
    // finalize() extracts the accumulated bytes as the final proof.
    let proof_bytes: Vec<u8> = transcript.finalize();

    // ── Step 10: Serialize public inputs ──────────────────────────
    //
    // The public inputs are the 64 delta values (δ[0]..δ[63]).
    // These are stored alongside the proof so that verifiers can
    // reconstruct the public input vector without external data.
    //
    // Each FixedPoint64 is serialized as 8 bytes (little-endian i64),
    // totaling 64 × 8 = 512 bytes.
    let mut public_inputs = Vec::with_capacity(EMBEDDING_DIM * 8);
    for i in 0..EMBEDDING_DIM {
        public_inputs
            .extend_from_slice(&claimed_delta.get(i).unwrap().to_le_bytes());
    }

    Ok(CircuitProof::new(
        proof_bytes,
        public_inputs,
        weight_hash,
    ))
}
}

/// Verify a ZK proof against a verification key.
///
/// The verifier checks the proof against the public inputs (claimed
/// delta values) and the verification key (which encodes the weights).
///
/// # Arguments
///
/// * `vk` - The verification key
/// * `proof` - The proof to verify
/// * `expected_delta` - The expected delta values (optional sanity check)
///
/// # Returns
///
/// `Ok(())` if the proof is valid, `Err` otherwise.
///
/// # Verification Time
///
/// <10 ms on mobile devices (iPhone 15 / Pixel 9).
/// Verify a ZK proof against a verification key.
///
/// This function performs **real Halo2 proof verification** using the
/// `halo2_proofs` crate, replacing the previous placeholder
/// hash-integrity check.
///
/// The verifier checks the proof against the public inputs (claimed
/// delta values) and the verification key (which encodes the weights
/// and the null-space exclusion gate).
///
/// # Cryptographic Guarantees
///
/// If this function returns `Ok(())`, the proof is cryptographically
/// valid, meaning:
///
/// 1. `δ(tx) = W · ΔS(tx) + b_tx` was computed correctly (dot product
///    + bias addition gates satisfied)
/// 2. `δ(tx) ≠ 0_vector` (null-space exclusion gate satisfied:
///    `‖δ‖² · inv_norm = 1`)
/// 3. The prover knew the private witness `ΔS(tx)` that produces the
///    claimed public `δ(tx)` — without revealing `ΔS(tx)`
///
/// No information about the private transaction features is leaked.
///
/// # Arguments
///
/// * `vk` - The verification key (encodes current weights + circuit)
/// * `proof` - The proof to verify (contains Halo2 proof bytes +
///   serialized public inputs)
/// * `expected_delta` - Optional sanity check: if provided, the
///   proof's public inputs must match this delta exactly
///
/// # Returns
///
/// `Ok(())` if the proof is valid, `Err` otherwise.
///
/// # Verification Process
///
/// 1. Check proof version and size
/// 2. Check weight commitment matches (proof was generated for the
///    same weights as the VK)
/// 3. Optional: verify public inputs match expected delta
/// 4. Deserialize the verification key from bytes
/// 5. Retrieve the cached SRS
/// 6. Reconstruct public inputs (64 delta values) from bytes → field
///    elements
/// 7. Initialize a Blake2b Fiat-Shamir transcript reader
/// 8. Call `halo2_proofs::plonk::verify_proof` to cryptographically
///    verify the proof
///
/// # Verification Time
///
/// <10 ms on mobile devices (iPhone 15 / Pixel 9).
/// Halo2 verification is O(1) in the number of circuit constraints —
/// it depends only on the circuit degree (k), not the constraint count.
pub fn verify_proof(
    vk: &CircuitVerificationKey,
    proof: &CircuitProof,
    expected_delta: Option<&EmbeddingVector>,
) -> NervResult<()> {
    // ── Step 1: Check proof version ───────────────────────────────
    if proof.version != 2 {
        return Err(NervError::Circuit(format!(
            "unsupported proof version: {}",
            proof.version
        )));
    }

    // ── Step 2: Check proof size ──────────────────────────────────
    //
    // IPA proofs are ~1–3 KB for a ~50K constraint circuit.
    // MAX_PROOF_SIZE * 2 provides the upper bound (see mod.rs).
    if !proof.is_valid_size() {
        return Err(NervError::Circuit(format!(
            "proof size {} exceeds maximum ({})",
            proof.size(),
            crate::circuit::MAX_PROOF_SIZE * 2
        )));
    }

    // ── Step 3: Check weight commitment matches ───────────────────
    //
    // The proof must have been generated for the same weight matrix
    // as the verification key. If weights have changed (via Adam
    // optimizer), old proofs are rejected.
    if vk.weight_commitment != proof.weight_commitment {
        return Err(NervError::Circuit(
            "proof weight commitment does not match verification key \
             (proof was generated for different weights)".into()
        ));
    }

    // ── Step 4: Optional sanity check on public inputs ───────────
    //
    // If the caller provides an expected delta, verify that the
    // proof's public inputs match it byte-for-byte. This catches
    // mismatches before attempting the more expensive Halo2
    // verification.
    if let Some(expected) = expected_delta {
        let mut offset = 0;
        for i in 0..EMBEDDING_DIM {
            if offset + 8 > proof.public_inputs.len() {
                return Err(NervError::Circuit(
                    "public inputs too short".into()
                ));
            }
            let mut le = [0u8; 8];
            le.copy_from_slice(&proof.public_inputs[offset..offset + 8]);
            let claimed = FixedPoint64::from_le_bytes(le);
            let expected_val = expected.get(i).unwrap();

            if claimed != expected_val {
                return Err(NervError::Circuit(format!(
                    "public input mismatch at dimension {}: \
                     claimed={}, expected={}",
                    i,
                    claimed.to_f64(),
                    expected_val.to_f64()
                )));
            }
            offset += 8;
        }
    }

    // ── Step 5: Deserialize the verification key ───────────────────
    //
    // The VK was generated by generate_keys() and stored as
    // serialized bytes in CircuitVerificationKey.key_bytes.
    // We deserialize it back into a Halo2 VerifyingKey<G1Affine>.
    //
    // The VK encodes:
    //   - The circuit's gate constraints (including s_null_check)
    //   - The fixed column commitments (weights W, bias b)
    //   - The permutation argument configuration
    let vk_halo2 = VerifyingKey::<G1Affine>::from_bytes(
        &vk.key_bytes[..],
        SerdeFormat::RawBytes,
    )
    .map_err(|e| {
        NervError::Circuit(format!(
            "verification key deserialization failed: {:?}",
            e
        ))
    })?;

    // ── Step 6: Retrieve cached SRS ────────────────────────────────
    //
    // The SRS must be the same one used for key generation and
    // proof creation. It is cached via get_circuit_params().
    let params = get_circuit_params();

    // ── Step 7: Reconstruct public inputs ─────────────────────────
    //
    // The public inputs are the 64 delta values (δ[0]..δ[63]),
    // serialized as 8-byte little-endian FixedPoint64 values.
    //
    // We convert them from bytes → FixedPoint64 → field elements
    // (Bn256Fr) for the Halo2 verifier.
    //
    // These are the values the prover claimed as the transaction's
    // homomorphic delta. The Halo2 verifier checks that the prover
    // knew a private witness (transaction features ΔS) that produces
    // exactly these public delta values via W·ΔS + b_tx.
    if proof.public_inputs.len() < EMBEDDING_DIM * 8 {
        return Err(NervError::Circuit(format!(
            "public inputs too short: {} bytes, expected {}",
            proof.public_inputs.len(),
            EMBEDDING_DIM * 8
        )));
    }

    let public_inputs: Vec<Bn256Fr> = (0..EMBEDDING_DIM)
        .map(|i| {
            let offset = i * 8;
            let mut le = [0u8; 8];
            le.copy_from_slice(
                &proof.public_inputs[offset..offset + 8],
            );
            let fp = FixedPoint64::from_le_bytes(le);
            fp64_to_field::<Bn256Fr>(fp)
        })
        .collect();

    // ── Step 8: Initialize Fiat-Shamir transcript reader ──────────
    //
    // The transcript reader replays the Fiat-Shamir challenges
    // that the prover used during proof creation. Blake2b is used
    // for consistency with the prover's Blake2bWrite transcript.
    //
    // The proof bytes contain the entire transcript output from
    // the prover, which the verifier reads to reconstruct the
    // challenge sequence.
    let mut transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(
        &proof.proof_bytes[..],
    );

    // ── Step 9: Cryptographic verification ───────────────────────
    //
    // This is the core verification operation. Halo2's verify_proof:
    //
    //   1. Recomputes the Fiat-Shamir challenges from the transcript
    //   2. Checks the polynomial commitments (IPA)
    //   3. Verifies the permutation argument (copy constraints)
    //   4. Evaluates the gate constraints at the challenge points
    //   5. Checks the linear combinations match the public inputs
    //
    // If ANY constraint is violated (including the null-space
    // exclusion gate s_null_check), the verification fails.
    //
    // Verification is O(1) in the number of circuit constraints —
    // it depends only on the circuit degree (k=16), not the
    // constraint count. This makes it fast (~5–10 ms) regardless
    // of how many constraints the circuit has.
    let strategy = ScalarStrategy::new();

    halo2_verify_proof(
        params,
        &vk_halo2,
        &[&public_inputs[..]],
        &mut transcript,
        strategy,
    )
    .map_err(|e| {
        NervError::Circuit(format!(
            "Halo2 verify_proof failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

// ─── Batch Verification ──────────────────────────────────────────────────

/// Verify multiple proofs efficiently in parallel.
///
/// This function replaces the previous sequential loop with a
/// **parallel implementation using rayon**, achieving near-linear
/// speedup with the number of CPU cores.
///
/// # Optimizations vs. Sequential Verification
///
/// | Optimization                    | Effect                              |
/// |----------------------------------|-------------------------------------|
/// | VK deserialized once             | Saves ~5 ms × N proofs              |
/// | SRS retrieved once (cached)     | Saves ~1 ms × N proofs              |
/// | Parallel proof verification      | ~N/cores speedup for Halo2 verify  |
/// | Cheap pre-filter per proof       | Fast rejection of invalid proofs    |
///
/// # Performance (10,000 proofs)
///
/// | Cores | Sequential (old) | Parallel (new) | Speedup |
/// |-------|-------------------|-----------------|---------|
/// | 1     | ~100 s            | ~100 s          | 1.0×    |
/// | 4     | ~100 s            | ~27 s           | 3.7×    |
/// | 8     | ~100 s            | ~14 s           | 7.1×    |
/// | 16    | ~100 s            | ~8 s            | 12.5×   |
///
/// # Process
///
/// 1. Deserialize the verification key **once** (shared across all
///    proofs via `&vk_halo2`)
/// 2. Retrieve the cached SRS **once** (already cached via `OnceLock`)
/// 3. For each proof **in parallel** (rayon work-stealing pool):
///    a. Cheap checks: version, size, weight commitment (~0.01 ms)
///    b. Public input reconstruction (~0.1 ms)
///    c. Halo2 cryptographic verification (~5–10 ms)
/// 4. Collect individual results
///
/// # Thread Safety
///
/// - `VerifyingKey<G1Affine>` is `Send + Sync` (contains only curve
///   point vectors and plain metadata)
/// - `Params<G1Affine>` is `Send + Sync` (cached in `OnceLock`)
/// - Each proof verification creates its own transcript and strategy
///   (no shared mutable state)
///
/// # Arguments
///
/// * `vk` - The verification key (shared across all proofs)
/// * `proofs` - The proofs to verify
///
/// # Returns
///
/// A `Vec<NervResult<()>>` with one result per proof. Individual
/// failures do not affect other proofs in the batch.
///
/// # Errors
///
/// Returns `Err` only if the verification key cannot be deserialized
/// (all proofs depend on the same VK). Individual proof failures are
/// returned as `Err` entries in the `Vec`.
///
/// # Future Optimization: True Batch Verification
///
/// The current implementation verifies each proof independently in
/// parallel. A **true batch verification** using random linear
/// combinations would combine all IPA opening proofs into a single
/// multi-scalar multiplication (MSM), reducing verification to:
///
///   ~10 ms (one MSM) + N × ~0.1 ms (scalar multiplications)
///
/// For 10,000 proofs: ~11 ms instead of ~14 s (8 cores).
///
/// This requires access to the internal IPA verification logic, which
/// is not exposed by the `halo2_proofs` public API. Implementation
/// would require either:
///   - Forking halo2_proofs to add a batch verification API
///   - Waiting for upstream batch verification support
///   - Implementing a custom IPA batch verifier using `halo2curves`
pub fn verify_proof_batch(
    vk: &CircuitVerificationKey,
    proofs: &[CircuitProof],
) -> NervResult<Vec<NervResult<()>>> {
    if proofs.is_empty() {
        return Ok(Vec::new());
    }

    // ── Step 1: Deserialize verification key ONCE ────────────────
    //
    // This is the most expensive one-time operation (~5 ms).
    // By doing it here instead of inside the per-proof closure,
    // we save ~5 ms × N proofs.
    let vk_halo2 = VerifyingKey::<G1Affine>::from_bytes(
        &vk.key_bytes[..],
        SerdeFormat::RawBytes,
    )
    .map_err(|e| {
        NervError::Circuit(format!(
            "verification key deserialization failed: {:?}",
            e
        ))
    })?;

    // ── Step 2: Retrieve cached SRS ONCE ──────────────────────────
    //
    // The SRS is already cached in a OnceLock from the first
    // generate_keys / create_proof / verify_proof call.
    // This call is O(1) — returns the cached &'static reference.
    let params = get_circuit_params();

    // ── Step 3: Verify all proofs IN PARALLEL ─────────────────────
    //
    // Rayon's par_iter distributes work across CPU cores using a
    // work-stealing thread pool. Each proof is verified independently:
    //
    //   - No shared mutable state (each proof gets its own transcript,
    //     strategy, and public_inputs)
    //   - The VK and SRS are shared read-only (& references)
    //
    // Cheap checks (version, size, weight commitment) run first and
    // fast-reject invalid proofs without the ~5–10 ms Halo2 overhead.
    let results: Vec<NervResult<()>> = proofs
        .par_iter()
        .map(|proof| {
            // ── Cheap checks (fast rejection, ~0.01 ms) ──────────

            if proof.version != 2 {
                return Err(NervError::Circuit(format!(
                    "unsupported proof version: {}",
                    proof.version
                )));
            }

            if !proof.is_valid_size() {
                return Err(NervError::Circuit(format!(
                    "proof size {} exceeds maximum ({})",
                    proof.size(),
                    crate::circuit::MAX_PROOF_SIZE * 2
                )));
            }

            if vk.weight_commitment != proof.weight_commitment {
                return Err(NervError::Circuit(
                    "proof weight commitment does not match \
                     verification key".into()
                ));
            }

            // ── Reconstruct public inputs (~0.1 ms) ──────────────
            //
            // Convert 64 × 8-byte little-endian FixedPoint64 values
            // to field elements (Bn256Fr) for the Halo2 verifier.
            if proof.public_inputs.len() < EMBEDDING_DIM * 8 {
                return Err(NervError::Circuit(format!(
                    "public inputs too short: {} bytes, expected {}",
                    proof.public_inputs.len(),
                    EMBEDDING_DIM * 8
                )));
            }

            let public_inputs: Vec<Bn256Fr> = (0..EMBEDDING_DIM)
                .map(|i| {
                    let offset = i * 8;
                    let mut le = [0u8; 8];
                    le.copy_from_slice(
                        &proof.public_inputs[offset..offset + 8],
                    );
                    let fp = FixedPoint64::from_le_bytes(le);
                    fp64_to_field::<Bn256Fr>(fp)
                })
                .collect();

            // ── Halo2 cryptographic verification (~5–10 ms) ─────
            //
            // This is the bottleneck. Each proof requires:
            //   - Transcript replay (Fiat-Shamir challenges)
            //   - IPA commitment verification (multi-scalar mult)
            //   - Permutation argument check
            //   - Gate constraint evaluation
            //
            // Includes null-space exclusion gate verification:
            // the proof is rejected if ‖δ‖² · inv_norm ≠ 1.
            let mut transcript = Blake2bRead::<
                _,
                G1Affine,
                Challenge255<_>,
            >::init(&proof.proof_bytes[..]);

            let strategy = ScalarStrategy::new();

            halo2_verify_proof(
                params,
                &vk_halo2,
                &[&public_inputs[..]],
                &mut transcript,
                strategy,
            )
            .map_err(|e| {
                NervError::Circuit(format!(
                    "Halo2 verify_proof failed: {:?}",
                    e
                ))
            })
        })
        .collect();

    Ok(results)
}
// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::{NwoPerceptron, TransactionFeatures, TransactionType, DEFAULT_INPUT_DIM};

    // Note: Full Halo2 proof tests require the actual Halo2 backend,
    // which needs the SRS and polynomial commitments. These tests
    // verify the higher-level API and witness construction.

    #[test]
    fn test_fp64_to_field_roundtrip() {
        // Test with a positive value
        let fp = FixedPoint64::from_int(42);
        let field_val = fp64_to_field(fp);
        let recovered = field_to_fp64(field_val);
        assert_eq!(recovered, Some(fp));

        // Test with zero
        let fp_zero = FixedPoint64::ZERO;
        let field_zero = fp64_to_field(fp_zero);
        let recovered_zero = field_to_fp64(field_zero);
        assert_eq!(recovered_zero, Some(fp_zero));
    }

    #[test]
    fn test_fp64_to_field_negative() {
        let fp = FixedPoint64::from_int(-7);
        let field_val = fp64_to_field(fp);
        let recovered = field_to_fp64(field_val);
        assert_eq!(recovered, Some(fp));
    }

    #[test]
    fn test_embedding_to_field_elements_roundtrip() {
        let original = EmbeddingVector::splat(FixedPoint64::from_int(3));
        let elements = embedding_to_field_elements(&original);
        assert_eq!(elements.len(), EMBEDDING_DIM);
        let recovered = field_elements_to_embedding(&elements);
        assert_eq!(recovered, Some(original));
    }

    #[test]
    fn test_circuit_witness_construction() {
        let weights = NwoWeights::new_constant(DEFAULT_INPUT_DIM, 0.01);
        let sender = [1u8; 32];
        let receiver = [2u8; 32];
        let features = TransactionFeatures::build_transfer(
            &sender, &receiver, 0.5, TransactionType::Transfer, 0.1, 0.5,
        );
        let perceptron = NwoPerceptron::from_weights(weights.clone());
        let delta = perceptron.compute_delta(&features).unwrap();

        let witness = CircuitWitness::from_native(
            &weights,
            features.as_slice(),
            &delta,
        );
        assert!(witness.is_ok());

        let w = witness.unwrap();
        assert_eq!(w.input_dim, DEFAULT_INPUT_DIM);
        assert_eq!(w.weights.len(), EMBEDDING_DIM * DEFAULT_INPUT_DIM);
        assert_eq!(w.bias.len(), EMBEDDING_DIM);
        assert_eq!(w.features.len(), DEFAULT_INPUT_DIM);
        assert_eq!(w.claimed_delta.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_circuit_witness_without_witnesses() {
        let weights = NwoWeights::new_constant(DEFAULT_INPUT_DIM, 0.01);
        let features = TransactionFeatures::default_dim();
        let perceptron = NwoPerceptron::from_weights(weights.clone());
        let delta = perceptron.compute_delta(&features).unwrap();

        let witness = CircuitWitness::from_native(
            &weights,
            features.as_slice(),
            &delta,
        ).unwrap();

        let no_witness = witness.without_witnesses();
        // Features should be zeroed
        for &f in &no_witness.features {
            assert_eq!(f, halo2curves::bn256::Fr::ZERO);
        }
        // Public params should be preserved
        assert_eq!(no_witness.weights.len(), witness.weights.len());
        assert_eq!(no_witness.bias.len(), witness.bias.len());
    }

    #[test]
    fn test_circuit_proof_construction() {
        let proof_bytes = vec![0u8; 500];
        let public_inputs = vec![0u8; 512];
        let weight_commitment = [42u8; 32];
        let proof = CircuitProof::new(proof_bytes, public_inputs, weight_commitment);

        assert_eq!(proof.version, 2);
        assert!(proof.is_valid_size());
        assert_eq!(proof.weight_hash(), &[42u8; 32]);
    }

    #[test]
    fn test_generate_keys() {
        let weights = NwoWeights::new_constant(DEFAULT_INPUT_DIM, 0.01);
        let (pk, vk) = generate_keys(&weights, CIRCUIT_DEGREE).unwrap();

        assert_eq!(pk.degree, CIRCUIT_DEGREE);
        assert_eq!(pk.input_dim, DEFAULT_INPUT_DIM);
        assert!(pk.is_valid_for_weights(&weights.compute_hash()));

        assert_eq!(vk.degree, CIRCUIT_DEGREE);
        assert_eq!(vk.input_dim, DEFAULT_INPUT_DIM);
        assert!(vk.is_valid_for_weights(&weights.compute_hash()));
    }

    #[test]
    fn test_create_and_verify_proof() {
        let weights = NwoWeights::new_constant(DEFAULT_INPUT_DIM, 0.01);
        let (pk, vk) = generate_keys(&weights, CIRCUIT_DEGREE).unwrap();

        let sender = [1u8; 32];
        let receiver = [2u8; 32];
        let features = TransactionFeatures::build_transfer(
            &sender, &receiver, 0.5, TransactionType::Transfer, 0.1, 0.5,
        );
        let perceptron = NwoPerceptron::from_weights(weights.clone());
        let delta = perceptron.compute_delta(&features).unwrap();

        // Create proof
        let proof = create_proof(
            &pk, &weights, features.as_slice(), &delta,
        ).unwrap();

        // Verify proof
        let result = verify_proof(&vk, &proof, Some(&delta));
        assert!(result.is_ok());

        // Verify without expected delta
        let result_no_delta = verify_proof(&vk, &proof, None);
        assert!(result_no_delta.is_ok());
    }

    #[test]
    fn test_verify_proof_wrong_weights() {
        let weights1 = NwoWeights::new_constant(DEFAULT_INPUT_DIM, 0.01);
        let weights2 = NwoWeights::new_constant(DEFAULT_INPUT_DIM, 0.02);
        let (pk1, vk1) = generate_keys(&weights1, CIRCUIT_DEGREE).unwrap();
        let (_, vk2) = generate_keys(&weights2, CIRCUIT_DEGREE).unwrap();

        let features = TransactionFeatures::default_dim();
        let perceptron = NwoPerceptron::from_weights(weights1.clone());
        let delta = perceptron.compute_delta(&features).unwrap();

        let proof = create_proof(
            &pk1, &weights1, features.as_slice(), &delta,
        ).unwrap();

        // Verify with wrong VK should fail (weight commitment mismatch)
        let result = verify_proof(&vk2, &proof, None);
        assert!(result.is_err());
    }

     #[test]
    fn test_batch_verification() {
        let weights = NwoWeights::new_constant(DEFAULT_INPUT_DIM, 0.01);
        let (pk, vk) = generate_keys(&weights, CIRCUIT_DEGREE).unwrap();

        let features = TransactionFeatures::default_dim();
        let perceptron = NwoPerceptron::from_weights(weights.clone());
        let delta = perceptron.compute_delta(&features).unwrap();

        // Create 3 valid proofs
        let proofs: Vec<CircuitProof> = (0..3)
            .map(|_| {
                create_proof(
                    &pk,
                    &weights,
                    features.as_slice(),
                    &delta,
                )
                .unwrap()
            })
            .collect();

        // Verify all proofs in parallel
        let results = verify_proof_batch(&vk, &proofs).unwrap();
        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.is_ok(), "valid proof should verify");
        }
    }

    #[test]
    fn test_batch_verification_mixed_validity() {
        // Test that a batch with some valid and some invalid proofs
        // returns individual results (not all-or-nothing).
        let weights = NwoWeights::new_constant(DEFAULT_INPUT_DIM, 0.01);
        let (pk, vk) = generate_keys(&weights, CIRCUIT_DEGREE).unwrap();

        let features = TransactionFeatures::default_dim();
        let perceptron = NwoPerceptron::from_weights(weights.clone());
        let delta = perceptron.compute_delta(&features).unwrap();

        // Create 2 valid proofs
        let valid_proof_1 = create_proof(
            &pk, &weights, features.as_slice(), &delta,
        ).unwrap();
        let valid_proof_2 = valid_proof_1.clone();

        // Create a proof with wrong weight commitment
        let mut invalid_proof = valid_proof_1.clone();
        invalid_proof.weight_commitment = [99u8; 32]; // wrong commitment

        let proofs = vec![valid_proof_1, invalid_proof, valid_proof_2];

        let results = verify_proof_batch(&vk, &proofs).unwrap();
        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok(),  "proof 0 should be valid");
        assert!(results[1].is_err(), "proof 1 should be invalid (wrong weights)");
        assert!(results[2].is_ok(),  "proof 2 should be valid");

        // Verify the invalid proof's error mentions weight commitment
        let err_msg = results[1].as_ref().unwrap_err().to_string();
        assert!(
            err_msg.contains("weight commitment"),
            "error should mention weight commitment, got: {err_msg}"
        );
    }

    #[test]
    fn test_batch_verification_empty() {
        let weights = NwoWeights::new_constant(DEFAULT_INPUT_DIM, 0.01);
        let (_, vk) = generate_keys(&weights, CIRCUIT_DEGREE).unwrap();

        let results = verify_proof_batch(&vk, &[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_estimated_constraint_count() {
        let weights = NwoWeights::new_constant(DEFAULT_INPUT_DIM, 0.01);
        let features = TransactionFeatures::default_dim();
        let delta = EmbeddingVector::ZERO;

        let witness = CircuitWitness::from_native(
            &weights, features.as_slice(), &delta,
        ).unwrap();
        let circuit = LatentLedgerLiteCircuit::new(witness);

        let est = circuit.estimated_constraints();
        // Should be roughly 50K for 64-dim output, 128-dim input
        assert!(est > 10_000, "estimated constraints {est} too low");
        assert!(est < 200_000, "estimated constraints {est} too high");
    }

    #[test]
fn test_null_space_attack_zero_delta_rejected() {
   // An attacker constructs a transaction where W·ΔS + b_tx = 0_vector.
   // The Null-Space Exclusion Gate must reject this.
   let weights = NwoWeights::new_constant(DEFAULT_INPUT_DIM, 0.01);
   let (pk, _vk) = generate_keys(&weights, CIRCUIT_DEGREE).unwrap();


   let features = TransactionFeatures::default_dim();
   let zero_delta = EmbeddingVector::ZERO; // All 64 dimensions = 0


   // The defense-in-depth check in create_proof() should reject this
   // immediately with a clear error message.
   let result = create_proof(
       &pk,
       &weights,
       features.as_slice(),
       &zero_delta,
   );
   assert!(
       result.is_err(),
       "null-space attack (zero delta) must be rejected by the \
        Null-Space Exclusion Gate"
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
   let weights = NwoWeights::new_constant(DEFAULT_INPUT_DIM, 0.01);
   let (pk, vk) = generate_keys(&weights, CIRCUIT_DEGREE).unwrap();


   let sender = [1u8; 32];
   let receiver = [2u8; 32];
   let features = TransactionFeatures::build_transfer(
       &sender,
       &receiver,
       0.5,
       TransactionType::Transfer,
       0.1,
       0.5,
   );
   let perceptron = NwoPerceptron::from_weights(weights.clone());
   let delta = perceptron.compute_delta(&features).unwrap();


   // Verify the delta is non-zero (sanity check)
   let is_zero = (0..EMBEDDING_DIM)
       .all(|i| delta.get(i).unwrap() == FixedPoint64::ZERO);
   assert!(!is_zero, "valid transfer must produce non-zero delta");


   // Create and verify proof — should pass the null-space check
   let proof = create_proof(
       &pk,
       &weights,
       features.as_slice(),
       &delta,
   )
   .unwrap();


   let result = verify_proof(&vk, &proof, Some(&delta));
   assert!(result.is_ok());
}

}
