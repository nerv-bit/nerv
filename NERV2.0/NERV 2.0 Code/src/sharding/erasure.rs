//! Reed-Solomon Erasure Coding for Shard Fault Tolerance.
//!
//! NERV uses Reed-Solomon coding with parameters (k=5, m=2) to erasure-code
//! each shard's embedding data across 7 replicas. Any k=5 replicas suffice
//! for full recovery, tolerating up to m=2 simultaneous failures (~28.6% of
//! replicas, or 40% of the original data nodes).
//!
//! # Parameters
//!
//! | Parameter | Value | Description |
//! |-----------|-------|-------------|
//! | k | 5 | Data shards (minimum for recovery) |
//! | m | 2 | Parity shards |
//! | n = k + m | 7 | Total replicas |
//! | Fault tolerance | 2/7 ≈ 28.6% | Survivable node failures |
//!
//! # Encoding Strategy
//!
//! Each shard's 512-byte embedding is treated as 512 independent
//! symbols (bytes). For each byte position, we apply RS encoding
//! across the k data shards to produce n total symbols:
//!
//! ```text
//! For byte position i (0..512):
//!   data_symbols = [shard_0[i], shard_1[i], ..., shard_{k-1}[i]]
//!   encoded = RS_Encode(data_symbols) → [s_0, s_1, ..., s_{n-1}]
//!   Distribute s_j to replica j
//! ```
//!
//! This byte-wise approach is simple, efficient, and naturally handles
//! the 512-byte embedding size (no need for larger field elements).
//!
//! # GF(256) Implementation
//!
//! We implement GF(256) using the irreducible polynomial
//! p(x) = x⁸ + x⁴ + x³ + x² + 1 (0x11D), which is the same
//! polynomial used in AES (Rijndael). Multiplication and division
//! use precomputed log/antilog (exp) tables for O(1) performance.

use crate::{
    ERASURE_K, ERASURE_M,
    ShardId, EmbeddingRoot, ValidatorId,
    EMBEDDING_BYTES,
    NervError, NervResult,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── GF(256) Constants ───────────────────────────────────────────────────

/// Irreducible polynomial for GF(256): x^8 + x^4 + x^3 + x^2 + 1
/// Represented as 0x11D (the reduction polynomial).
const GF256_POLY: u16 = 0x11D;

/// Generator element for GF(256) — the primitive element α.
/// For the AES polynomial, α = 2 is a generator.
const GF256_GENERATOR: u8 = 2;

// ─── GF(256) Arithmetic ──────────────────────────────────────────────────

/// GF(256) element with full field arithmetic via log/antilog tables.
///
/// The field is constructed as GF(2)[x] / p(x) where
/// p(x) = x⁸ + x⁴ + x³ + x² + 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Gf256(pub u8);

impl Gf256 {
    /// The additive identity (0).
    pub const ZERO: Self = Self(0);

    /// The multiplicative identity (1).
    pub const ONE: Self = Self(1);

    /// Create a GF(256) element from a u8.
    #[inline]
    pub const fn new(val: u8) -> Self {
        Self(val)
    }

    /// Addition in GF(256) is XOR.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }

    /// Subtraction in GF(256) is the same as addition (XOR).
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }

    /// Multiplication using log/antilog tables.
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        if self.0 == 0 || other.0 == 0 {
            return Self::ZERO;
        }
        let log_sum = GF256_LOG[self.0 as usize] as u16 + GF256_LOG[other.0 as usize] as u16;
        Self(GF256_EXP[(log_sum % 255) as usize])
    }

    /// Division using log/antilog tables.
    #[inline]
    pub fn div(self, other: Self) -> Self {
        if self.0 == 0 {
            return Self::ZERO;
        }
        if other.0 == 0 {
            panic!("GF(256) division by zero");
        }
        let log_diff = (GF256_LOG[self.0 as usize] as i16 + 255 - GF256_LOG[other.0 as usize] as i16);
        Self(GF256_EXP[(log_diff.rem_euclid(255)) as usize])
    }

    /// Multiplicative inverse.
    #[inline]
    pub fn inv(self) -> Self {
        if self.0 == 0 {
            panic!("GF(256) inverse of zero");
        }
        Self(GF256_EXP[(255 - GF256_LOG[self.0 as usize] as usize) % 255])
    }

    /// Exponentiation: self^exp.
    pub fn pow(self, mut exp: u32) -> Self {
        if exp == 0 {
            return Self::ONE;
        }
        let mut result = Self::ONE;
        let mut base = self;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(base);
            }
            base = base.mul(base);
            exp >>= 1;
        }
        result
    }

    /// Check if this is the zero element.
    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }
}

impl std::fmt::Display for Gf256 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#04x}", self.0)
    }
}

// ─── Log/Antilog Tables ──────────────────────────────────────────────────

/// Build the GF(256) exponent (antilog) table.
///
/// exp[i] = α^i where α is the generator element.
/// The table has 512 entries (two periods) to avoid modular reduction
/// during multiplication: log(a) + log(b) can be up to 254 + 254 = 508.
const fn build_exp_table() -> [u8; 512] {
    let mut table = [0u8; 512];
    let mut val = 1u8;
    let mut i = 0;
    while i < 255 {
        table[i] = val;
        // Multiply by generator: val * 2 in GF(256)
        // If high bit set, reduce by XOR with polynomial (excluding x^8 term)
        let next = if val & 0x80 != 0 {
            (val << 1) ^ (GF256_POLY as u8)
        } else {
            val << 1
        };
        val = next;
        i += 1;
    }
    table[255] = 1; // α^255 = α^0 = 1
    // Duplicate for wraparound
    let mut i = 256;
    while i < 512 {
        table[i] = table[i - 255];
        i += 1;
    }
    table
}

/// Build the GF(256) logarithm table.
///
/// log[exp[i]] = i for all i in [0, 254].
/// log[0] is undefined (set to 0 as sentinel; callers must check for zero).
const fn build_log_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut val = 1u8;
    let mut i = 0;
    while i < 255 {
        table[val as usize] = i as u8;
        let next = if val & 0x80 != 0 {
            (val << 1) ^ (GF256_POLY as u8)
        } else {
            val << 1
        };
        val = next;
        i += 1;
    }
    table
}

/// Precomputed exponent (antilog) table for GF(256).
static GF256_EXP: [u8; 512] = build_exp_table();

/// Precomputed logarithm table for GF(256).
static GF256_LOG: [u8; 256] = build_log_table();

// ─── Reed-Solomon Polynomial Operations ──────────────────────────────────

/// A polynomial over GF(256) with coefficients in ascending order.
///
/// coeffs[i] is the coefficient of x^i.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GfPoly {
    /// Coefficients in ascending order: [a_0, a_1, ..., a_n]
    pub coeffs: Vec<Gf256>,
}

impl GfPoly {
    /// Create the zero polynomial.
    pub fn zero() -> Self {
        Self { coeffs: vec![] }
    }

    /// Create a constant polynomial.
    pub fn constant(c: Gf256) -> Self {
        Self { coeffs: vec![c] }
    }

    /// Create from a slice of coefficients.
    pub fn from_coeffs(coeffs: &[Gf256]) -> Self {
        let mut c = coeffs.to_vec();
        // Remove trailing zeros
        while c.len() > 1 && c.last() == Some(&Gf256::ZERO) {
            c.pop();
        }
        Self { coeffs: c }
    }

    /// Degree of the polynomial.
    pub fn degree(&self) -> usize {
        if self.coeffs.is_empty() {
            0
        } else {
            self.coeffs.len() - 1
        }
    }

    /// Evaluate the polynomial at a point x using Horner's method.
    pub fn evaluate(&self, x: Gf256) -> Gf256 {
        if self.coeffs.is_empty() {
            return Gf256::ZERO;
        }
        // Horner: a_n * x^n + ... + a_1 * x + a_0
        //       = (...((a_n * x + a_{n-1}) * x + a_{n-2}) * x + ...) + a_0
        let mut result = Gf256::ZERO;
        for coeff in self.coeffs.iter().rev() {
            result = result.mul(x).add(*coeff);
        }
        result
    }

    /// Multiply two polynomials.
    pub fn multiply(&self, other: &GfPoly) -> GfPoly {
        if self.coeffs.is_empty() || other.coeffs.is_empty() {
            return GfPoly::zero();
        }
        let result_len = self.coeffs.len() + other.coeffs.len() - 1;
        let mut result = vec![Gf256::ZERO; result_len];

        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                result[i + j] = result[i + j].add(a.mul(b));
            }
        }

        GfPoly::from_coeffs(&result)
    }

    /// Scale the polynomial by a constant.
    pub fn scale(&self, c: Gf256) -> GfPoly {
        let coeffs: Vec<Gf256> = self.coeffs.iter().map(|&a| a.mul(c)).collect();
        GfPoly::from_coeffs(&coeffs)
    }

    /// Polynomial addition.
    pub fn add_poly(&self, other: &GfPoly) -> GfPoly {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut result = vec![Gf256::ZERO; max_len];

        for (i, &c) in self.coeffs.iter().enumerate() {
            result[i] = result[i].add(c);
        }
        for (i, &c) in other.coeffs.iter().enumerate() {
            result[i] = result[i].add(c);
        }

        GfPoly::from_coeffs(&result)
    }

    /// Multiply by (x - root) — used to build the generator polynomial.
    pub fn mul_by_linear(&self, root: Gf256) -> GfPoly {
        // (x - root) * p(x) = x*p(x) - root*p(x)
        // If p(x) = a_0 + a_1*x + ... + a_n*x^n
        // x*p(x) = a_0*x + a_1*x^2 + ... + a_n*x^{n+1}
        // root*p(x) = root*a_0 + root*a_1*x + ... + root*a_n*x^n
        // Result: [-root*a_0, a_0-root*a_1, a_1-root*a_2, ..., a_{n-1}-root*a_n, a_n]
        let n = self.coeffs.len();
        let mut result = vec![Gf256::ZERO; n + 1];

        for i in 0..n {
            result[i] = result[i].add(self.coeffs[i].mul(root)); // -root * a_i (subtraction = addition in GF(2^8))
        }
        for i in 0..n {
            result[i + 1] = result[i + 1].add(self.coeffs[i]); // a_i (shifted by one position)
        }

        GfPoly::from_coeffs(&result)
    }
}

// ─── Reed-Solomon Encoder ────────────────────────────────────────────────

/// Configuration for Reed-Solomon coding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErasureConfig {
    /// Number of data shards (k).
    pub data_shards: usize,

    /// Number of parity shards (m).
    pub parity_shards: usize,

    /// Total shards (n = k + m).
    pub total_shards: usize,
}

impl ErasureConfig {
    /// Create the default NERV configuration (k=5, m=2).
    pub fn nerv_default() -> Self {
        Self {
            data_shards: ERASURE_K,
            parity_shards: ERASURE_M,
            total_shards: ERASURE_K + ERASURE_M,
        }
    }

    /// Create a custom configuration.
    pub fn new(data_shards: usize, parity_shards: usize) -> NervResult<Self> {
        if data_shards == 0 {
            return Err(NervError::Sharding("data_shards must be > 0".into()));
        }
        if parity_shards == 0 {
            return Err(NervError::Sharding("parity_shards must be > 0".into()));
        }
        let total = data_shards + parity_shards;
        if total > 256 {
            return Err(NervError::Sharding(
                "total_shards cannot exceed 256 (GF(256) limit)".into()
            ));
        }
        Ok(Self {
            data_shards,
            parity_shards,
            total_shards: total,
        })
    }

    /// Validate that the config is consistent.
    pub fn validate(&self) -> NervResult<()> {
        if self.data_shards + self.parity_shards != self.total_shards {
            return Err(NervError::Sharding(
                "data_shards + parity_shards != total_shards".into()
            ));
        }
        if self.total_shards > 256 {
            return Err(NervError::Sharding(
                "total_shards exceeds GF(256) limit".into()
            ));
        }
        Ok(())
    }
}

impl Default for ErasureConfig {
    fn default() -> Self {
        Self::nerv_default()
    }
}

/// Reed-Solomon encoder.
pub struct ErasureEncoder {
    /// Configuration.
    pub config: ErasureConfig,

    /// Evaluation points (distinct elements of GF(256)).
    /// First k points are for data, remaining m are for parity.
    eval_points: Vec<Gf256>,

    /// Generator polynomial g(x) = (x - α^0)(x - α^1)...(x - α^{m-1}).
    generator_poly: GfPoly,
}

impl ErasureEncoder {
    /// Create a new RS encoder with the given configuration.
    pub fn new(config: ErasureConfig) -> NervResult<Self> {
        config.validate()?;

        // Choose evaluation points: 0, 1, 2, ..., n-1
        // (These must be distinct elements of GF(256))
        let eval_points: Vec<Gf256> = (0..config.total_shards)
            .map(|i| Gf256::new(i as u8))
            .collect();

        // Build generator polynomial: g(x) = (x - α^0)(x - α^1)...(x - α^{m-1})
        // where α is the primitive element
        let mut gen_poly = GfPoly::constant(Gf256::ONE);
        for i in 0..config.parity_shards {
            let root = Gf256(GF256_GENERATOR).pow(i as u32);
            gen_poly = gen_poly.mul_by_linear(root);
        }

        Ok(Self {
            config,
            eval_points,
            generator_poly: gen_poly,
        })
    }

    /// Create with NERV default parameters (k=5, m=2).
    pub fn nerv_default() -> Self {
        Self::new(ErasureConfig::nerv_default()).unwrap()
    }

    /// Encode a single byte across shards using polynomial evaluation.
    ///
    /// Given k data values (bytes from k shards at the same position),
    /// compute the unique polynomial of degree < k that interpolates
    /// the data at the first k evaluation points, then evaluate at
    /// all n points to produce the encoded symbols.
    ///
    /// Returns n encoded symbols.
    pub fn encode_symbol(&self, data: &[u8]) -> NervResult<Vec<u8>> {
        if data.len() != self.config.data_shards {
            return Err(NervError::Sharding(format!(
                "expected {} data symbols, got {}",
                self.config.data_shards, data.len()
            )));
        }

        // Interpolate polynomial through the data points
        let data_gf: Vec<Gf256> = data.iter().map(|&b| Gf256::new(b)).collect();
        let poly = self.interpolate(&data_gf)?;

        // Evaluate at all n points
        let encoded: Vec<u8> = self.eval_points.iter()
            .map(|&x| poly.evaluate(x).0)
            .collect();

        Ok(encoded)
    }

    /// Encode a full shard dataset.
    ///
    /// Takes k data shards (each a byte slice of the same length)
    /// and produces n encoded shards (k data + m parity).
    ///
    /// Each output shard has the same length as the input shards.
    pub fn encode(&self, data_shards: &[&[u8]]) -> NervResult<Vec<Vec<u8>>> {
        if data_shards.len() != self.config.data_shards {
            return Err(NervError::Sharding(format!(
                "expected {} data shards, got {}",
                self.config.data_shards, data_shards.len()
            )));
        }

        // Check all shards have the same length
        let shard_len = data_shards.first().map(|s| s.len()).unwrap_or(0);
        for (i, shard) in data_shards.iter().enumerate() {
            if shard.len() != shard_len {
                return Err(NervError::Sharding(format!(
                    "shard {} has length {}, expected {}", i, shard.len(), shard_len
                )));
            }
        }

        // Initialize output shards
        let mut output: Vec<Vec<u8>> = vec![vec![0u8; shard_len]; self.config.total_shards];

        // Encode byte-by-byte
        for byte_pos in 0..shard_len {
            // Gather the k data bytes at this position
            let data_bytes: Vec<u8> = data_shards.iter()
                .map(|shard| shard[byte_pos])
                .collect();

            let encoded = self.encode_symbol(&data_bytes)?;

            // Distribute to output shards
            for (shard_idx, &byte) in encoded.iter().enumerate() {
                output[shard_idx][byte_pos] = byte;
            }
        }

        Ok(output)
    }

    /// Lagrange interpolation: find the polynomial of degree < k
    /// that passes through (x_i, y_i) for i = 0..k-1.
    fn interpolate(&self, values: &[Gf256]) -> NervResult<GfPoly> {
        let k = values.len();
        if k == 0 {
            return Ok(GfPoly::zero());
        }

        let xs: Vec<Gf256> = self.eval_points[..k].to_vec();

        // Lagrange basis polynomial interpolation:
        // L(x) = Σ_i y_i * l_i(x)
        // where l_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)

        let mut result = GfPoly::zero();

        for i in 0..k {
            if values[i].is_zero() {
                continue;
            }

            // Compute the Lagrange basis polynomial l_i(x)
            let mut basis = GfPoly::constant(Gf256::ONE);
            let mut denom = Gf256::ONE;

            for j in 0..k {
                if j == i {
                    continue;
                }
                // Multiply basis by (x - x_j)
                basis = basis.mul_by_linear(xs[j]);
                // Accumulate denominator: (x_i - x_j)
                denom = denom.mul(xs[i].sub(xs[j]));
            }

            // l_i(x) = basis / denom
            if denom.is_zero() {
                return Err(NervError::Sharding(
                    "Lagrange interpolation: duplicate evaluation points".into()
                ));
            }

            let scale = values[i].mul(denom.inv());
            basis = basis.scale(scale);
            result = result.add_poly(&basis);
        }

        Ok(result)
    }

    /// Get the generator polynomial.
    pub fn generator_polynomial(&self) -> &GfPoly {
        &self.generator_poly
    }
}

// ─── Reed-Solomon Decoder ────────────────────────────────────────────────

/// Reed-Solomon decoder (erasure-only).
pub struct ErasureDecoder {
    /// Configuration.
    pub config: ErasureConfig,

    /// Evaluation points (must match the encoder).
    eval_points: Vec<Gf256>,
}

impl ErasureDecoder {
    /// Create a new RS decoder with the given configuration.
    pub fn new(config: ErasureConfig) -> NervResult<Self> {
        config.validate()?;
        let eval_points: Vec<Gf256> = (0..config.total_shards)
            .map(|i| Gf256::new(i as u8))
            .collect();
        Ok(Self { config, eval_points })
    }

    /// Create with NERV default parameters.
    pub fn nerv_default() -> Self {
        Self::new(ErasureConfig::nerv_default()).unwrap()
    }

    /// Decode a single symbol from shards with erasures.
    ///
    /// # Arguments
    ///
    /// * `shards` - Symbol values for each shard (None = erased)
    /// * `erased` - Boolean mask indicating which shards are erased
    ///
    /// # Returns
    ///
    /// The k recovered data symbols.
    pub fn decode_symbol(
        &self,
        shards: &[Option<u8>],
        erased: &[bool],
    ) -> NervResult<Vec<u8>> {
        if shards.len() != self.config.total_shards {
            return Err(NervError::Sharding(format!(
                "expected {} shards, got {}",
                self.config.total_shards, shards.len()
            )));
        }

        // Count available (non-erased) shards
        let available: Vec<usize> = (0..self.config.total_shards)
            .filter(|&i| !erased[i])
            .collect();

        if available.len() < self.config.data_shards {
            return Err(NervError::Sharding(format!(
                "not enough shards for recovery: {} available, {} needed",
                available.len(), self.config.data_shards
            )));
        }

        // Select k available shards
        let selected: Vec<usize> = available.into_iter()
            .take(self.config.data_shards)
            .collect();

        // Gather known (x, y) pairs
        let xs: Vec<Gf256> = selected.iter()
            .map(|&i| self.eval_points[i])
            .collect();
        let ys: Vec<Gf256> = selected.iter()
            .map(|&i| {
                shards[i].map(Gf256::new)
                    .ok_or_else(|| NervError::Sharding("erased shard has no value".into()))
            })
            .collect::<NervResult<Vec<Gf256>>>()?;

        // Lagrange interpolation to recover the polynomial
        let poly = self.interpolate_at(&xs, &ys)?;

        // Evaluate at the first k evaluation points to get data symbols
        let data: Vec<u8> = self.eval_points[..self.config.data_shards].iter()
            .map(|&x| poly.evaluate(x).0)
            .collect();

        Ok(data)
    }

    /// Decode full shard data from shards with erasures.
    ///
    /// # Arguments
    ///
    /// * `shards` - Shard data (None for erased shards)
    /// * `erased` - Boolean mask indicating which shards are erased
    ///
    /// # Returns
    ///
    /// The k recovered data shards.
    pub fn decode(
        &self,
        shards: &[Option<&[u8]>],
        erased: &[bool],
    ) -> NervResult<Vec<Vec<u8>>> {
        if shards.len() != self.config.total_shards {
            return Err(NervError::Sharding(format!(
                "expected {} shards, got {}",
                self.config.total_shards, shards.len()
            )));
        }

        // Determine shard length from available shards
        let shard_len = shards.iter()
            .flatten()
            .map(|s| s.len())
            .next()
            .ok_or_else(|| NervError::Sharding("no available shards".into()))?;

        // Verify all available shards have the same length
        for (i, shard) in shards.iter().enumerate() {
            if let Some(data) = shard {
                if data.len() != shard_len {
                    return Err(NervError::Sharding(format!(
                        "shard {} has length {}, expected {}", i, data.len(), shard_len
                    )));
                }
            }
        }

        // Count available shards
        let available_count = erased.iter().filter(|&&e| !e).count();
        if available_count < self.config.data_shards {
            return Err(NervError::Sharding(format!(
                "not enough shards: {} available, {} needed",
                available_count, self.config.data_shards
            )));
        }

        // Initialize output data shards
        let mut output: Vec<Vec<u8>> = vec![vec![0u8; shard_len]; self.config.data_shards];

        // Decode byte-by-byte
        for byte_pos in 0..shard_len {
            // Gather symbols at this position
            let symbols: Vec<Option<u8>> = (0..self.config.total_shards).map(|i| {
                if erased[i] {
                    None
                } else {
                    shards[i].map(|data| data[byte_pos])
                }
            }).collect();

            let data = self.decode_symbol(&symbols, erased)?;

            for (shard_idx, &byte) in data.iter().enumerate() {
                output[shard_idx][byte_pos] = byte;
            }
        }

        Ok(output)
    }

    /// Lagrange interpolation at arbitrary points.
    fn interpolate_at(&self, xs: &[Gf256], ys: &[Gf256]) -> NervResult<GfPoly> {
        let k = xs.len();
        if k != ys.len() {
            return Err(NervError::Sharding(
                "x and y arrays must have the same length".into()
            ));
        }
        if k == 0 {
            return Ok(GfPoly::zero());
        }

        let mut result = GfPoly::zero();

        for i in 0..k {
            if ys[i].is_zero() {
                continue;
            }

            let mut basis = GfPoly::constant(Gf256::ONE);
            let mut denom = Gf256::ONE;

            for j in 0..k {
                if j == i {
                    continue;
                }
                basis = basis.mul_by_linear(xs[j]);
                denom = denom.mul(xs[i].sub(xs[j]));
            }

            if denom.is_zero() {
                return Err(NervError::Sharding(
                    "interpolation: duplicate x values".into()
                ));
            }

            let scale = ys[i].mul(denom.inv());
            basis = basis.scale(scale);
            result = result.add_poly(&basis);
        }

        Ok(result)
    }
}

// ─── Shard Chunk ─────────────────────────────────────────────────────────

/// An erasure-coded chunk of shard data, stored on a replica node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardChunk {
    /// The shard this chunk belongs to.
    pub shard_id: ShardId,

    /// The replica index (0..n-1).
    pub replica_index: usize,

    /// The chunk data (same length as the original shard data).
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub data: Vec<u8>,

    /// BLAKE3 hash of the data for integrity verification.
    pub data_hash: [u8; 32],

    /// The node storing this replica.
    pub node_id: ValidatorId,
}

impl ShardChunk {
    /// Create a new shard chunk.
    pub fn new(
        shard_id: ShardId,
        replica_index: usize,
        data: Vec<u8>,
        node_id: ValidatorId,
    ) -> Self {
        let data_hash = blake3::hash(&data).into();
        Self {
            shard_id,
            replica_index,
            data,
            data_hash,
            node_id,
        }
    }

    /// Verify the integrity of this chunk.
    pub fn verify_integrity(&self) -> bool {
        let computed: [u8; 32] = blake3::hash(&self.data).into();
        computed == self.data_hash
    }

    /// Size of the chunk data in bytes.
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

// ─── High-Level Encode/Decode Functions ──────────────────────────────────

/// Encode a shard's embedding data into erasure-coded replicas.
///
/// Takes k=5 data replicas and produces n=7 total replicas.
/// This is the primary encoding function used during shard
/// initialization and after embedding updates.
pub fn erasure_encode(
    data_replicas: &[&[u8]],
    config: &ErasureConfig,
) -> NervResult<Vec<Vec<u8>>> {
    let encoder = ErasureEncoder::new(config.clone())?;
    encoder.encode(data_replicas)
}

/// Decode shard data from erasure-coded replicas.
///
/// Takes n shards (some possibly erased) and recovers the k data shards.
/// This is used during shard recovery when some replicas are unavailable.
pub fn erasure_decode(
    shards: &[Option<&[u8]>],
    erased: &[bool],
    config: &ErasureConfig,
) -> NervResult<Vec<Vec<u8>>> {
    let decoder = ErasureDecoder::new(config.clone())?;
    decoder.decode(shards, erased)
}

/// Encode a single shard's embedding across n replica nodes.
///
/// This is the convenience function for the common case of encoding
/// a single shard's 512-byte embedding for fault-tolerant storage.
/// It treats the single shard as k=1 data shard with m=6 parity shards.
///
/// **Note**: For the standard NERV configuration (k=5, m=2), use
/// `erasure_encode` with 5 data replicas instead.
pub fn erasure_encode_single(
    data: &[u8],
    total_replicas: usize,
    parity_shards: usize,
) -> NervResult<Vec<Vec<u8>>> {
    if total_replicas < 2 || parity_shards == 0 {
        return Err(NervError::Sharding("invalid erasure parameters".into()));
    }
    if total_replicas != 1 + parity_shards {
        return Err(NervError::Sharding(
            "total_replicas must equal 1 + parity_shards for single encoding".into()
        ));
    }

    let config = ErasureConfig::new(1, parity_shards)?;
    let encoder = ErasureEncoder::new(config)?;
    encoder.encode(&[data])
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── GF(256) Arithmetic Tests ───────────────────────────────────

    #[test]
    fn test_gf256_add() {
        let a = Gf256::new(0x53);
        let b = Gf256::new(0xCA);
        let sum = a.add(b);
        assert_eq!(sum, Gf256::new(0x53 ^ 0xCA)); // XOR
    }

    #[test]
    fn test_gf256_add_self() {
        // a + a = 0 in GF(2^8)
        let a = Gf256::new(0x42);
        assert_eq!(a.add(a), Gf256::ZERO);
    }

    #[test]
    fn test_gf256_mul_identity() {
        let a = Gf256::new(0x53);
        assert_eq!(a.mul(Gf256::ONE), a);
        assert_eq!(Gf256::ONE.mul(a), a);
    }

    #[test]
    fn test_gf256_mul_zero() {
        let a = Gf256::new(0x53);
        assert_eq!(a.mul(Gf256::ZERO), Gf256::ZERO);
        assert_eq!(Gf256::ZERO.mul(a), Gf256::ZERO);
    }

    #[test]
    fn test_gf256_mul_commutative() {
        let a = Gf256::new(0x53);
        let b = Gf256::new(0xCA);
        assert_eq!(a.mul(b), b.mul(a));
    }

    #[test]
    fn test_gf256_mul_associative() {
        let a = Gf256::new(0x53);
        let b = Gf256::new(0xCA);
        let c = Gf256::new(0x1F);
        assert_eq!(a.mul(b.mul(c)), a.mul(b).mul(c));
    }

    #[test]
    fn test_gf256_div() {
        let a = Gf256::new(0x53);
        let b = Gf256::new(0xCA);
        let product = a.mul(b);
        // product / b should equal a
        assert_eq!(product.div(b), a);
    }

    #[test]
    fn test_gf256_inv() {
        let a = Gf256::new(0x53);
        let inv = a.inv();
        assert_eq!(a.mul(inv), Gf256::ONE);
    }

    #[test]
    fn test_gf256_pow() {
        let a = Gf256::new(0x02);
        // 2^0 = 1
        assert_eq!(a.pow(0), Gf256::ONE);
        // 2^1 = 2
        assert_eq!(a.pow(1), Gf256::new(0x02));
        // 2^8 should reduce by the polynomial
        let a8 = a.pow(8);
        // α^8 = α^4 + α^3 + α^2 + 1 = 0x1D
        assert_eq!(a8, Gf256::new(0x1D));
    }

    #[test]
    fn test_gf256_generator_order() {
        // The generator α=2 should have order 255 in GF(256)*
        let alpha = Gf256::new(GF256_GENERATOR);
        assert_eq!(alpha.pow(255), Gf256::ONE);
        // And no smaller power should give 1
        assert_ne!(alpha.pow(1), Gf256::ONE);
    }

    #[test]
    fn test_gf256_exp_log_consistency() {
        // exp[log[a]] = a for all non-zero a
        for a in 1u8..=255u8 {
            let log_val = GF256_LOG[a as usize];
            let exp_val = GF256_EXP[log_val as usize];
            assert_eq!(exp_val, a, "exp[log[{}]] failed", a);
        }
    }

    // ─── Polynomial Tests ────────────────────────────────────────────

    #[test]
    fn test_gf_poly_evaluate() {
        // p(x) = 3 + 2x + x²
        let poly = GfPoly::from_coeffs(&[
            Gf256::new(3),
            Gf256::new(2),
            Gf256::new(1),
        ]);
        // p(1) = 3 + 2 + 1 = 6 (in GF(256), this is just XOR: 3^2^1 = 0)
        // Wait, addition is XOR in GF(2^8): 3 ^ 2 ^ 1 = 0
        let val = poly.evaluate(Gf256::new(1));
        assert_eq!(val, Gf256::new(3 ^ 2 ^ 1));

        // p(0) = 3
        assert_eq!(poly.evaluate(Gf256::ZERO), Gf256::new(3));
    }

    #[test]
    fn test_gf_poly_multiply() {
        // (1 + x) * (1 + x) = 1 + 2x + x² = 1 + x² (since 2x = 0 in GF(2^8))
        // Wait, 2 is not 0 in GF(256). 1+1 = 0 (XOR), but the coefficient 2 is the field element 2.
        // (1 + x) * (1 + x):
        //   = 1*1 + (1*1+1*1)*x + 1*1*x²
        //   = 1 + (1^1)*x + x²
        //   = 1 + 0*x + x²
        //   = 1 + x²
        let p = GfPoly::from_coeffs(&[Gf256::ONE, Gf256::ONE]);
        let pp = p.multiply(&p);
        // Should be 1 + 0*x + x² = [1, 0, 1]
        assert_eq!(pp.coeffs.len(), 3);
        assert_eq!(pp.coeffs[0], Gf256::ONE);
        assert_eq!(pp.coeffs[1], Gf256::ZERO);
        assert_eq!(pp.coeffs[2], Gf256::ONE);
    }

    #[test]
    fn test_gf_poly_mul_by_linear() {
        // p(x) = 1, multiply by (x - 3)
        let p = GfPoly::constant(Gf256::new(1));
        let result = p.mul_by_linear(Gf256::new(3));
        // Result: -3 + x = 3 + x (since -3 = 3 in GF(2^8))
        assert_eq!(result.coeffs[0], Gf256::new(3));
        assert_eq!(result.coeffs[1], Gf256::ONE);
    }

    // ─── RS Encoding/Decoding Tests ──────────────────────────────────

    #[test]
    fn test_erasure_config_default() {
        let config = ErasureConfig::nerv_default();
        assert_eq!(config.data_shards, 5);
        assert_eq!(config.parity_shards, 2);
        assert_eq!(config.total_shards, 7);
    }

    #[test]
    fn test_erasure_config_validation() {
        let config = ErasureConfig {
            data_shards: 5,
            parity_shards: 2,
            total_shards: 6, // Wrong!
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_rs_encode_basic() {
        let encoder = ErasureEncoder::nerv_default();
        let data: Vec<&[u8]> = vec![
            &[1, 2, 3, 4][..],
            &[5, 6, 7, 8][..],
            &[9, 10, 11, 12][..],
            &[13, 14, 15, 16][..],
            &[17, 18, 19, 20][..],
        ];

        let encoded = encoder.encode(&data).unwrap();
        assert_eq!(encoded.len(), 7); // k + m = 5 + 2

        // First k shards should be the original data
        for i in 0..5 {
            assert_eq!(encoded[i], data[i]);
        }

        // Parity shards should be different
        assert_ne!(encoded[5], encoded[6]);
    }

    #[test]
    fn test_rs_decode_no_erasures() {
        let encoder = ErasureEncoder::nerv_default();
        let data: Vec<&[u8]> = vec![
            &[10, 20, 30][..],
            &[40, 50, 60][..],
            &[70, 80, 90][..],
            &[100, 110, 120][..],
            &[130, 140, 150][..],
        ];

        let encoded = encoder.encode(&data).unwrap();

        // Decode with no erasures
        let decoder = ErasureDecoder::nerv_default();
        let shards: Vec<Option<&[u8]>> = encoded.iter().map(|s| Some(s.as_slice())).collect();
        let erased = vec![false; 7];

        let decoded = decoder.decode(&shards, &erased).unwrap();
        assert_eq!(decoded.len(), 5);
        for i in 0..5 {
            assert_eq!(decoded[i], data[i]);
        }
    }

    #[test]
    fn test_rs_decode_one_erasure() {
        let encoder = ErasureEncoder::nerv_default();
        let data: Vec<&[u8]> = vec![
            &[1, 2, 3][..],
            &[4, 5, 6][..],
            &[7, 8, 9][..],
            &[10, 11, 12][..],
            &[13, 14, 15][..],
        ];

        let encoded = encoder.encode(&data).unwrap();

        // Erase shard 0
        let decoder = ErasureDecoder::nerv_default();
        let shards: Vec<Option<&[u8]>> = encoded.iter().enumerate().map(|(i, s)| {
            if i == 0 { None } else { Some(s.as_slice()) }
        }).collect();
        let erased: Vec<bool> = (0..7).map(|i| i == 0).collect();

        let decoded = decoder.decode(&shards, &erased).unwrap();
        for i in 0..5 {
            assert_eq!(decoded[i], data[i]);
        }
    }

    #[test]
    fn test_rs_decode_two_erasures() {
        let encoder = ErasureEncoder::nerv_default();
        let data: Vec<&[u8]> = vec![
            &[42, 84, 126][..],
            &[168, 210, 252][..],
            &[1, 2, 3][..],
            &[4, 5, 6][..],
            &[7, 8, 9][..],
        ];

        let encoded = encoder.encode(&data).unwrap();

        // Erase shards 0 and 6 (one data + one parity)
        let decoder = ErasureDecoder::nerv_default();
        let shards: Vec<Option<&[u8]>> = encoded.iter().enumerate().map(|(i, s)| {
            if i == 0 || i == 6 { None } else { Some(s.as_slice()) }
        }).collect();
        let erased: Vec<bool> = (0..7).map(|i| i == 0 || i == 6).collect();

        let decoded = decoder.decode(&shards, &erased).unwrap();
        for i in 0..5 {
            assert_eq!(decoded[i], data[i]);
        }
    }

    #[test]
    fn test_rs_decode_two_data_erasures() {
        let encoder = ErasureEncoder::nerv_default();
        let data: Vec<&[u8]> = vec![
            &[100, 200, 55][..],
            &[33, 77, 99][..],
            &[11, 22, 33][..],
            &[44, 55, 66][..],
            &[77, 88, 99][..],
        ];

        let encoded = encoder.encode(&data).unwrap();

        // Erase two data shards (indices 1 and 3)
        let decoder = ErasureDecoder::nerv_default();
        let shards: Vec<Option<&[u8]>> = encoded.iter().enumerate().map(|(i, s)| {
            if i == 1 || i == 3 { None } else { Some(s.as_slice()) }
        }).collect();
        let erased: Vec<bool> = (0..7).map(|i| i == 1 || i == 3).collect();

        let decoded = decoder.decode(&shards, &erased).unwrap();
        for i in 0..5 {
            assert_eq!(decoded[i], data[i]);
        }
    }

    #[test]
    fn test_rs_decode_too_many_erasures() {
        let encoder = ErasureEncoder::nerv_default();
        let data: Vec<&[u8]> = vec![
            &[1, 2, 3][..],
            &[4, 5, 6][..],
            &[7, 8, 9][..],
            &[10, 11, 12][..],
            &[13, 14, 15][..],
        ];

        let encoded = encoder.encode(&data).unwrap();

        // Erase 3 shards (more than m=2)
        let decoder = ErasureDecoder::nerv_default();
        let shards: Vec<Option<&[u8]>> = encoded.iter().enumerate().map(|(i, s)| {
            if i == 0 || i == 1 || i == 5 { None } else { Some(s.as_slice()) }
        }).collect();
        let erased: Vec<bool> = (0..7).map(|i| i == 0 || i == 1 || i == 5).collect();

        assert!(decoder.decode(&shards, &erased).is_err());
    }

    #[test]
    fn test_rs_encode_decode_embedding_sized() {
        // Test with actual embedding-sized data (512 bytes)
        let encoder = ErasureEncoder::nerv_default();

        let mut data_shards = Vec::new();
        for shard_idx in 0..5 {
            let mut shard = vec![0u8; EMBEDDING_BYTES];
            for (i, byte) in shard.iter_mut().enumerate() {
                *byte = ((shard_idx * 100 + i) % 256) as u8;
            }
            data_shards.push(shard);
        }

        let data_refs: Vec<&[u8]> = data_shards.iter().map(|s| s.as_slice()).collect();
        let encoded = encoder.encode(&data_refs).unwrap();

        // Verify all shards are 512 bytes
        for shard in &encoded {
            assert_eq!(shard.len(), EMBEDDING_BYTES);
        }

        // Erase two shards and decode
        let decoder = ErasureDecoder::nerv_default();
        let shards: Vec<Option<&[u8]>> = encoded.iter().enumerate().map(|(i, s)| {
            if i == 2 || i == 6 { None } else { Some(s.as_slice()) }
        }).collect();
        let erased: Vec<bool> = (0..7).map(|i| i == 2 || i == 6).collect();

        let decoded = decoder.decode(&shards, &erased).unwrap();
        for i in 0..5 {
            assert_eq!(decoded[i], data_shards[i]);
        }
    }

    #[test]
    fn test_rs_encode_all_zeros() {
        let encoder = ErasureEncoder::nerv_default();
        let data: Vec<&[u8]> = vec![
            &[0, 0, 0][..],
            &[0, 0, 0][..],
            &[0, 0, 0][..],
            &[0, 0, 0][..],
            &[0, 0, 0][..],
        ];

        let encoded = encoder.encode(&data).unwrap();

        // All encoded shards should be all zeros
        for shard in &encoded {
            assert!(shard.iter().all(|&b| b == 0));
        }
    }

    #[test]
    fn test_shard_chunk_integrity() {
        let chunk = ShardChunk::new(
            ShardId::new(42),
            0,
            vec![1, 2, 3, 4, 5],
            ValidatorId::from_bytes([1u8; 32]),
        );
        assert!(chunk.verify_integrity());
        assert_eq!(chunk.size(), 5);
    }

    #[test]
    fn test_shard_chunk_integrity_corrupted() {
        let mut chunk = ShardChunk::new(
            ShardId::new(42),
            0,
            vec![1, 2, 3, 4, 5],
            ValidatorId::from_bytes([1u8; 32]),
        );
        chunk.data[0] = 99; // Corrupt
        assert!(!chunk.verify_integrity());
    }

    #[test]
    fn test_high_level_erasure_encode_decode() {
        let config = ErasureConfig::nerv_default();
        let data: Vec<&[u8]> = vec![
            &[10, 20][..],
            &[30, 40][..],
            &[50, 60][..],
            &[70, 80][..],
            &[90, 100][..],
        ];

        let encoded = erasure_encode(&data, &config).unwrap();

        let shards: Vec<Option<&[u8]>> = encoded.iter().enumerate().map(|(i, s)| {
            if i == 0 || i == 5 { None } else { Some(s.as_slice()) }
        }).collect();
        let erased: Vec<bool> = (0..7).map(|i| i == 0 || i == 5).collect();

        let decoded = erasure_decode(&shards, &erased, &config).unwrap();
        for i in 0..5 {
            assert_eq!(decoded[i], data[i]);
        }
    }

    #[test]
    fn test_rs_generator_polynomial() {
        let encoder = ErasureEncoder::nerv_default();
        let gen = encoder.generator_polynomial();

        // Generator for (5,2) code should have degree m=2
        assert_eq!(gen.degree(), 2);

        // The roots should be α^0 = 1 and α^1 = 2
        let alpha = Gf256::new(GF256_GENERATOR);
        assert!(gen.evaluate(Gf256::ONE).is_zero(), "g(1) should be 0");
        assert!(gen.evaluate(alpha).is_zero(), "g(α) should be 0");
    }

    #[test]
    fn test_gf256_distributive() {
        let a = Gf256::new(0x53);
        let b = Gf256::new(0xCA);
        let c = Gf256::new(0x1F);
        // a * (b + c) = a * b + a * c
        let lhs = a.mul(b.add(c));
        let rhs = a.mul(b).add(a.mul(c));
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_erasure_config_custom() {
        let config = ErasureConfig::new(10, 4).unwrap();
        assert_eq!(config.data_shards, 10);
        assert_eq!(config.parity_shards, 4);
        assert_eq!(config.total_shards, 14);
    }

    #[test]
    fn test_erasure_config_too_large() {
        // total > 256 should fail
        assert!(ErasureConfig::new(200, 100).is_err());
    }

    #[test]
    fn test_rs_encode_systematic() {
        // RS encoding should be systematic: first k shards are the original data
        let encoder = ErasureEncoder::nerv_default();
        let data: Vec<&[u8]> = vec![
            &[0xAA, 0xBB][..],
            &[0xCC, 0xDD][..],
            &[0xEE, 0xFF][..],
            &[0x11, 0x22][..],
            &[0x33, 0x44][..],
        ];

        let encoded = encoder.encode(&data).unwrap();

        // First 5 shards should be identical to input
        for i in 0..5 {
            assert_eq!(encoded[i], data[i]);
        }
    }
}
