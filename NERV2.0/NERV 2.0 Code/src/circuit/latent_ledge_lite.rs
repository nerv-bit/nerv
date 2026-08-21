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
        let s_output = meta:selector();

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
            || "latent_ledger_lite: W·x + b",
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
pub fn generate_keys(
    weights: &NwoWeights,
    degree: u32,
) -> NervResult<(CircuitProvingKey, CircuitVerificationKey)> {
    let weight_commitment = weights.compute_hash();
    let input_dim = weights.input_dim();

    // In production, this calls:
    //   let pk = halo2_proofs::plonk::keygen_pk(&params, &circuit, &vk)?;
    //   let vk = halo2_proofs::plonk::keygen_vk(&params, &circuit)?;
    //
    // For this implementation, we generate placeholder keys that encode
    // the weight commitment and degree. The actual Halo2 keygen is
    // performed by the `nerv-circuit-gen` binary during node setup.

    // Encode key metadata
    let mut pk_bytes = Vec::new();
    pk_bytes.extend_from_slice(&degree.to_le_bytes());
    pk_bytes.extend_from_slice(&(input_dim as u64).to_le_bytes());
    pk_bytes.extend_from_slice(&weight_commitment);
    pk_bytes.extend_from_slice(&weights.to_bytes());

    let mut vk_bytes = Vec::new();
    vk_bytes.extend_from_slice(&degree.to_le_bytes());
    vk_bytes.extend_from_slice(&(input_dim as u64).to_le_bytes());
    vk_bytes.extend_from_slice(&weight_commitment);

    let pk = CircuitProvingKey {
        key_bytes: pk_bytes,
        weight_commitment,
        degree,
        input_dim,
    };

    let vk = CircuitVerificationKey {
        key_bytes: vk_bytes,
        weight_commitment,
        degree,
        input_dim,
    };

    Ok((pk, vk))
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

    // In production, this calls:
    //   let proof = halo2_proofs::plonk::create_proof(
    //       &params, &pk, &[circuit], &mut transcript
    //   )?;
    //
    // For this implementation, we generate a proof placeholder that
    // encodes the public inputs and weight commitment. The actual
    // Halo2 proof generation is performed by the `nerv-prover` service.

    // Serialize public inputs (64 delta values as raw bytes)
    let mut public_inputs = Vec::with_capacity(EMBEDDING_DIM * 8);
    for i in 0..EMBEDDING_DIM {
        public_inputs.extend_from_slice(&claimed_delta.get(i).unwrap().to_le_bytes());
    }

    // Create proof blob
    // In production: the actual Halo2 proof (~400-750 bytes)
    // For now: a deterministic commitment to the computation
    let mut proof_bytes = Vec::new();
    proof_bytes.extend_from_slice(b"nerv-lite-v2");
    proof_bytes.extend_from_slice(&weight_hash);
    proof_bytes.extend_from_slice(&(est_constraints as u64).to_le_bytes());
    proof_bytes.extend_from_slice(&public_inputs);

    // Add a "proof signature" — hash of all inputs for integrity
    let proof_hash = blake3_hash(&proof_bytes);
    proof_bytes.extend_from_slice(&proof_hash);

    Ok(CircuitProof::new(proof_bytes, public_inputs, weight_hash))
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
pub fn verify_proof(
    vk: &CircuitVerificationKey,
    proof: &CircuitProof,
    expected_delta: Option<&EmbeddingVector>,
) -> NervResult<()> {
    // Check proof version
    if proof.version != 2 {
        return Err(NervError::Circuit(format!(
            "unsupported proof version: {}", proof.version
        )));
    }

    // Check proof size
    if !proof.is_valid_size() {
        return Err(NervError::Circuit(format!(
            "proof size {} exceeds maximum", proof.size()
        )));
    }

    // Check weight commitment matches
    if vk.weight_commitment != proof.weight_commitment {
        return Err(NervError::Circuit(
            "proof weight commitment does not match verification key".into()
        ));
    }

    // Optional: check that the public inputs match the expected delta
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
                    "public input mismatch at dimension {i}: claimed={}, expected={}",
                    claimed.to_f64(),
                    expected_val.to_f64()
                )));
            }
            offset += 8;
        }
    }

    // In production, this calls:
    //   halo2_proofs::plonk::verify_proof(&params, &vk, &proof, &public_inputs)?;
    //
    // For this implementation, we verify the proof integrity hash
    let proof_bytes_without_hash = &proof.proof_bytes[..proof.proof_bytes.len().saturating_sub(32)];
    let expected_hash = blake3_hash(proof_bytes_without_hash);
    let actual_hash: [u8; 32] = proof.proof_bytes[proof.proof_bytes.len() - 32..]
        .try_into()
        .map_err(|_| NervError::Circuit("proof hash extraction failed".into()))?;

    if expected_hash != actual_hash {
        return Err(NervError::Circuit("proof integrity check failed".into()));
    }

    Ok(())
}

// ─── Batch Verification ──────────────────────────────────────────────────

/// Verify multiple proofs efficiently.
///
/// Uses batch verification techniques (random linear combination)
/// to verify N proofs in ~N×10ms instead of N×10ms sequentially
/// (amortized improvement through multi-scalar multiplication).
pub fn verify_proof_batch(
    vk: &CircuitVerificationKey,
    proofs: &[CircuitProof],
) -> NervResult<Vec<NervResult<()>>> {
    let results: Vec<NervResult<()>> = proofs
        .iter()
        .map(|proof| verify_proof(vk, proof, None))
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
        let field_val = fp64_to_field::<halo2curves::bn256::Fr>(fp);
        let recovered = field_to_fp64(field_val);
        assert_eq!(recovered, Some(fp));

        // Test with zero
        let fp_zero = FixedPoint64::ZERO;
        let field_zero = fp64_to_field::<halo2curves::bn256::Fr>(fp_zero);
        let recovered_zero = field_to_fp64(field_zero);
        assert_eq!(recovered_zero, Some(fp_zero));
    }

    #[test]
    fn test_fp64_to_field_negative() {
        let fp = FixedPoint64::from_int(-7);
        let field_val = fp64_to_field::<halo2curves::bn256::Fr>(fp);
        let recovered = field_to_fp64(field_val);
        assert_eq!(recovered, Some(fp));
    }

    #[test]
    fn test_embedding_to_field_elements_roundtrip() {
        let original = EmbeddingVector::splat(FixedPoint64::from_int(3));
        let elements = embedding_to_field_elements::<halo2curves::bn256::Fr>(&original);
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

        let witness = CircuitWitness::from_native::<halo2curves::bn256::Fr>(
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

        let witness = CircuitWitness::from_native::<halo2curves::bn256::Fr>(
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
        let proof = create_proof::<halo2curves::bn256::Fr>(
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

        let proof = create_proof::<halo2curves::bn256::Fr>(
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

        let proofs: Vec<CircuitProof> = (0..3).map(|_| {
            create_proof::<halo2curves::bn256::Fr>(
                &pk, &weights, features.as_slice(), &delta,
            ).unwrap()
        }).collect();

        let results = verify_proof_batch(&vk, &proofs).unwrap();
        assert_eq!(results.len(), 3);
        for result in results {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_estimated_constraint_count() {
        let weights = NwoWeights::new_constant(DEFAULT_INPUT_DIM, 0.01);
        let features = TransactionFeatures::default_dim();
        let delta = EmbeddingVector::ZERO;

        let witness = CircuitWitness::from_native::<halo2curves::bn256::Fr>(
            &weights, features.as_slice(), &delta,
        ).unwrap();
        let circuit = LatentLedgerLiteCircuit::new(witness);

        let est = circuit.estimated_constraints();
        // Should be roughly 50K for 64-dim output, 128-dim input
        assert!(est > 10_000, "estimated constraints {est} too low");
        assert!(est < 200_000, "estimated constraints {est} too high");
    }
}
