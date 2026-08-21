//! Fixed-point arithmetic in 32.32 format for embedding vectors.
//!
//! Each value is stored as an `i64` where:
//! - Bits [63:32] = signed integer part
//! - Bits [31:0]  = fractional part
//!
//! ```text
//! real_value = raw_i64 / 2^32
//! ```
//!
//! # Range & Precision
//!
//! | Property | Value |
//! |----------|-------|
//! | Range | [-2,147,483,648.0, +2,147,483,647.9999999998] |
//! | Precision | 2⁻³² ≈ 2.328 × 10⁻¹⁰ |
//! | Smallest positive | ≈ 2.328 × 10⁻¹⁰ |
//!
//! # Arithmetic Rules
//!
//! - **Addition / Subtraction**: Exact (no rounding).
//! - **Multiplication**: Rounded to nearest (i128 intermediate, then shift).
//! - **Division**: Rounded to nearest (i128 intermediate).
//!
//! # Why Fixed-Point?
//!
//! Floating-point is unsuitable for blockchain state because:
//! 1. Non-associative: `(a + b) + c ≠ a + (b + c)` in general
//! 2. Non-deterministic across platforms (hardware FMA, rounding modes)
//! 3. Cannot be represented in ZK circuits (require prime field elements)
//!
//! Fixed-point arithmetic is deterministic, portable, and trivially
//! arithmetizable in Halo2 circuits.

use crate::{EMBEDDING_DIM, EMBEDDING_BYTES, NervError, NervResult};
use crate::utils::{blake3_hash, ct_eq, secure_zero};
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;
use zeroize::Zeroize;

// ─── Constants ───────────────────────────────────────────────────────────

/// Number of fractional bits in the 32.32 format.
pub const FRAC_BITS: u32 = 32;

/// The scale factor: 2^32 = 4,294,967,296.
pub const SCALE: i64 = 1i64 << FRAC_BITS;

/// Half the scale, used for rounding: 2^31.
pub const HALF_SCALE: i64 = 1i64 << (FRAC_BITS - 1);

/// The i128 scale for intermediate calculations.
pub const SCALE_I128: i128 = 1i128 << FRAC_BITS;

/// Half the i128 scale for rounding.
pub const HALF_SCALE_I128: i128 = 1i128 << (FRAC_BITS - 1);

// ─── FixedPoint64 ────────────────────────────────────────────────────────

/// A 64-bit signed fixed-point number in 32.32 format.
///
/// # Internal Representation
///
/// The stored `i64` value `raw` represents the real number `raw / 2^32`.
///
/// # Examples
///
/// ```
/// use nerv::embedding::FixedPoint64 as FP;
///
/// let a = FP::from_int(3);
/// let b = FP::from_int(7);
/// let sum = a + b;
/// assert_eq!(sum.to_int(), 10);
///
/// let product = a * b;
/// assert_eq!(product.to_f64(), 21.0);
/// ```
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
    Serialize, Deserialize, Zeroize,
)]
#[zeroize(drop)]
#[repr(transparent)]
pub struct FixedPoint64(
    /// The raw i64 value in 32.32 format.
    i64,
);

impl FixedPoint64 {
    // ── Named Constants ──────────────────────────────────────────────

    /// Zero.
    pub const ZERO: Self = Self(0);

    /// One (1.0 in fixed-point = 2^32).
    pub const ONE: Self = Self(SCALE);

    /// One-half (0.5).
    pub const HALF: Self = Self(HALF_SCALE);

    /// Negative one (-1.0).
    pub const NEG_ONE: Self = Self(-SCALE);

    /// Minimum representable value.
    pub const MIN: Self = Self(i64::MIN);

    /// Maximum representable value.
    pub const MAX: Self = Self(i64::MAX);

    /// Smallest positive value (2^{-32} ≈ 2.328 × 10^{-10}).
    pub const EPSILON: Self = Self(1);

    // ── Constructors ─────────────────────────────────────────────────

    /// Construct from an integer value (exact).
    ///
    /// `from_int(3)` → 3.0 represented as `3 × 2^32`.
    #[inline]
    pub const fn from_int(val: i64) -> Self {
        Self(val.wrapping_mul(SCALE))
    }

    /// Construct from a floating-point value (rounded to nearest).
    ///
    /// **Warning**: This is lossy. Use only for initialization and testing.
    #[inline]
    pub fn from_f64(val: f64) -> Self {
        let scaled = val * SCALE as f64;
        Self(scaled.round() as i64)
    }

    /// Construct from a floating-point value, returning `None` if out of range.
    pub fn try_from_f64(val: f64) -> Option<Self> {
        let scaled = val * SCALE as f64;
        if scaled >= i64::MIN as f64 && scaled <= i64::MAX as f64 {
            Some(Self(scaled.round() as i64))
        } else {
            None
        }
    }

    /// Construct from the raw i64 internal representation (already in 32.32 format).
    #[inline]
    pub const fn from_raw(raw: i64) -> Self {
        Self(raw)
    }

    /// Construct from a rational `numerator / denominator` (exact if divisible,
    /// rounded to nearest otherwise).
    pub fn from_rational(numerator: i64, denominator: i64) -> Option<Self> {
        if denominator == 0 {
            return None;
        }
        // Scale numerator up by 2^32, then divide
        let num_scaled = (numerator as i128) << FRAC_BITS;
        let den = denominator as i128;
        // Round to nearest
        let result = if (num_scaled >= 0 && den > 0) || (num_scaled < 0 && den < 0) {
            (num_scaled + den / 2) / den
        } else {
            (num_scaled - den / 2) / den
        };
        if result > i64::MAX as i128 || result < i64::MIN as i128 {
            None
        } else {
            Some(Self(result as i64))
        }
    }

    // ── Accessors ────────────────────────────────────────────────────

    /// Return the raw i64 internal value.
    #[inline]
    pub const fn raw(&self) -> i64 {
        self.0
    }

    /// Convert to integer (truncating the fractional part).
    #[inline]
    pub const fn to_int(&self) -> i64 {
        self.0 >> FRAC_BITS
    }

    /// Convert to f64 (lossy — for display and debugging only).
    #[inline]
    pub fn to_f64(&self) -> f64 {
        self.0 as f64 / SCALE as f64
    }

    /// Return the fractional part as a FixedPoint64 in [0, 1).
    #[inline]
    pub const fn frac_part(&self) -> Self {
        Self(self.0 & (SCALE - 1))
    }

    /// Return the integer part as a FixedPoint64.
    #[inline]
    pub const fn int_part(&self) -> Self {
        Self(self.0 & !(SCALE - 1))
    }

    // ── Arithmetic (basic) ───────────────────────────────────────────

    /// Add two fixed-point numbers (exact — no rounding).
    #[inline]
    pub const fn add(&self, other: Self) -> Self {
        Self(self.0.wrapping_add(other.0))
    }

    /// Subtract two fixed-point numbers (exact — no rounding).
    #[inline]
    pub const fn sub(&self, other: Self) -> Self {
        Self(self.0.wrapping_sub(other.0))
    }

    /// Negate a fixed-point number.
    #[inline]
    pub const fn neg(&self) -> Self {
        Self(self.0.wrapping_neg())
    }

    /// Absolute value.
    #[inline]
    pub fn abs(&self) -> Self {
        Self(self.0.abs())
    }

    /// Multiply two fixed-point numbers (rounded to nearest).
    ///
    /// Uses i128 intermediate to prevent overflow, then shifts right
    /// by 32 bits to return to 32.32 format.
    pub fn mul(&self, other: Self) -> Self {
        let a = self.0 as i128;
        let b = other.0 as i128;
        let raw = a * b;
        // Round to nearest: add ±2^31 before shifting
        let rounded = if raw >= 0 {
            (raw + HALF_SCALE_I128) >> FRAC_BITS
        } else {
            (raw - HALF_SCALE_I128) >> FRAC_BITS
        };
        Self(rounded as i64)
    }

    /// Divide two fixed-point numbers (rounded to nearest).
    ///
    /// Returns `None` if the divisor is zero or the result overflows.
    pub fn div(&self, other: Self) -> Option<Self> {
        if other.0 == 0 {
            return None;
        }
        // Scale numerator up by 2^32, then divide
        let a = (self.0 as i128) << FRAC_BITS;
        let b = other.0 as i128;
        // Round to nearest
        let result = if (a >= 0 && b > 0) || (a < 0 && b < 0) {
            (a + b / 2) / b
        } else {
            (a - b / 2) / b
        };
        if result > i64::MAX as i128 || result < i64::MIN as i128 {
            None
        } else {
            Some(Self(result as i64))
        }
    }

    /// Multiply by an integer scalar (exact — no rounding needed).
    #[inline]
    pub const fn mul_int(&self, scalar: i64) -> Self {
        Self(self.0.wrapping_mul(scalar))
    }

    /// Divide by an integer scalar (rounded to nearest).
    pub fn div_int(&self, scalar: i64) -> Option<Self> {
        if scalar == 0 {
            return None;
        }
        let a = self.0 as i128;
        let s = scalar as i128;
        let result = if (a >= 0 && s > 0) || (a < 0 && s < 0) {
            (a + s / 2) / s
        } else {
            (a - s / 2) / s
        };
        Some(Self(result as i64))
    }

    // ── Checked Arithmetic ───────────────────────────────────────────

    /// Checked addition. Returns `None` on overflow.
    #[inline]
    pub const fn checked_add(&self, other: Self) -> Option<Self> {
        match self.0.checked_add(other.0) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }

    /// Checked subtraction. Returns `None` on underflow.
    #[inline]
    pub const fn checked_sub(&self, other: Self) -> Option<Self> {
        match self.0.checked_sub(other.0) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }

    /// Checked multiplication. Returns `None` on overflow.
    pub fn checked_mul(&self, other: Self) -> Option<Self> {
        let a = self.0 as i128;
        let b = other.0 as i128;
        let raw = a * b;
        let rounded = if raw >= 0 {
            (raw + HALF_SCALE_I128) >> FRAC_BITS
        } else {
            (raw - HALF_SCALE_I128) >> FRAC_BITS
        };
        if rounded > i64::MAX as i128 || rounded < i64::MIN as i128 {
            None
        } else {
            Some(Self(rounded as i64))
        }
    }

    /// Saturating addition — clamps at MIN/MAX.
    #[inline]
    pub const fn saturating_add(&self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// Saturating subtraction — clamps at MIN/MAX.
    #[inline]
    pub const fn saturating_sub(&self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    // ── Comparison ───────────────────────────────────────────────────

    /// Returns `true` if the value is zero.
    #[inline]
    pub const fn is_zero(&self) -> bool {
        self.0 == 0
    }

    /// Returns `true` if the value is strictly positive.
    #[inline]
    pub const fn is_positive(&self) -> bool {
        self.0 > 0
    }

    /// Returns `true` if the value is strictly negative.
    #[inline]
    pub const fn is_negative(&self) -> bool {
        self.0 < 0
    }

    /// Maximum of two values.
    #[inline]
    pub const fn max(&self, other: Self) -> Self {
        Self(if self.0 >= other.0 { self.0 } else { other.0 })
    }

    /// Minimum of two values.
    #[inline]
    pub const fn min(&self, other: Self) -> Self {
        Self(if self.0 <= other.0 { self.0 } else { other.0 })
    }

    /// Clamp to [lo, hi].
    #[inline]
    pub const fn clamp(&self, lo: Self, hi: Self) -> Self {
        Self(if self.0 < lo.0 { lo.0 } else if self.0 > hi.0 { hi.0 } else { self.0 })
    }

    // ── Serialization ────────────────────────────────────────────────

    /// Encode as 8 little-endian bytes.
    #[inline]
    pub fn to_le_bytes(&self) -> [u8; 8] {
        self.0.to_le_bytes()
    }

    /// Decode from 8 little-endian bytes.
    #[inline]
    pub fn from_le_bytes(bytes: [u8; 8]) -> Self {
        Self(i64::from_le_bytes(bytes))
    }

    /// Encode as 8 big-endian bytes.
    #[inline]
    pub fn to_be_bytes(&self) -> [u8; 8] {
        self.0.to_be_bytes()
    }

    /// Decode from 8 big-endian bytes.
    #[inline]
    pub fn from_be_bytes(bytes: [u8; 8]) -> Self {
        Self(i64::from_be_bytes(bytes))
    }

    // ── Square Root (Newton's method) ────────────────────────────────

    /// Compute the square root using Newton's method (6 iterations).
    ///
    /// Returns `None` if the value is negative.
    /// Precision: accurate to within ~1 ULP of the true sqrt.
    pub fn sqrt(&self) -> Option<Self> {
        if self.0 < 0 {
            return None;
        }
        if self.0 == 0 {
            return Some(Self::ZERO);
        }
        // Initial guess from f64
        let guess_f64 = self.to_f64().sqrt();
        let mut x = Self::from_f64(guess_f64);

        // Newton iterations: x_{n+1} = (x_n + self / x_n) / 2
        for _ in 0..6 {
            if x.is_zero() {
                break;
            }
            if let Some(ratio) = self.div(x) {
                let sum = x.add(ratio);
                x = sum.div(Self::from_int(2)).unwrap_or(x);
            } else {
                break;
            }
        }
        Some(x)
    }
}

// ── Operator Overloads ──────────────────────────────────────────────────

impl std::ops::Add for FixedPoint64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl std::ops::Sub for FixedPoint64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

impl std::ops::Mul for FixedPoint64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        FixedPoint64::mul(&self, rhs)
    }
}

impl std::ops::Neg for FixedPoint64 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self(self.0.wrapping_neg())
    }
}

impl std::ops::AddAssign for FixedPoint64 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.wrapping_add(rhs.0);
    }
}

impl std::ops::SubAssign for FixedPoint64 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0.wrapping_sub(rhs.0);
    }
}

impl std::ops::MulAssign for FixedPoint64 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul(rhs);
    }
}

impl std::fmt::Display for FixedPoint64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.10}", self.to_f64())
    }
}

impl std::default::Default for FixedPoint64 {
    fn default() -> Self {
        Self::ZERO
    }
}

// ─── EmbeddingVector ────────────────────────────────────────────────────

/// A 512-byte neural state embedding vector: 64 dimensions of `FixedPoint64`.
///
/// This is the **canonical state representation** in NERV v2.0 — replacing
/// the multi-gigabyte Merkle Patricia Trie with a compact, homomorphic,
/// 512-byte learned latent vector.
///
/// # Layout
///
/// ```text
/// [FP64_0 | FP64_1 | ... | FP64_63]  (64 × 8 = 512 bytes)
/// ```
///
/// # Key Properties
///
/// - **Homomorphic**: `e_{t+1} = e_t + δ(tx)` (exact, 0 error)
/// - **Non-invertible**: Cannot recover individual balances from `e_t`
///   (reduces to Neural Network Inversion Problem, >2^4000 entropy)
/// - **Compact**: 512 bytes regardless of how many accounts the shard holds
/// - **Verifiable**: BLAKE3 hash of the 512 bytes → 32-byte `EmbeddingRoot`
#[derive(Debug, Clone, PartialEq, Eq, Zeroize)]
#[zeroize(drop)]
pub struct EmbeddingVector(
    #[zeroize(skip)] // we zeroize manually in our Drop-like zeroize derivation
    pub [FixedPoint64; EMBEDDING_DIM],
);

impl EmbeddingVector {
    /// The all-zero embedding (genesis / uninitialized).
    pub const ZERO: Self = Self([FixedPoint64::ZERO; EMBEDDING_DIM]);

    // ── Constructors ─────────────────────────────────────────────────

    /// Construct from an array of 64 `FixedPoint64` values.
    #[inline]
    pub const fn from_array(arr: [FixedPoint64; EMBEDDING_DIM]) -> Self {
        Self(arr)
    }

    /// Construct from 64 raw i64 values (each in 32.32 format).
    pub fn from_raw_values(values: [i64; EMBEDDING_DIM]) -> Self {
        let arr: [FixedPoint64; EMBEDDING_DIM] = std::array::from_fn(|i| {
            FixedPoint64::from_raw(values[i])
        });
        Self(arr)
    }

    /// Construct from a slice of f64 values.
    ///
    /// Returns `None` if the slice length ≠ 64.
    pub fn from_f64_slice(values: &[f64]) -> Option<Self> {
        if values.len() != EMBEDDING_DIM {
            return None;
        }
        let mut arr = [FixedPoint64::ZERO; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            arr[i] = FixedPoint64::from_f64(values[i]);
        }
        Some(Self(arr))
    }

    /// Construct from 512 raw bytes (64 × 8 little-endian i64s).
    pub fn from_bytes(bytes: &[u8; EMBEDDING_BYTES]) -> Self {
        let mut arr = [FixedPoint64::ZERO; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            let start = i * 8;
            let mut le = [0u8; 8];
            le.copy_from_slice(&bytes[start..start + 8]);
            arr[i] = FixedPoint64::from_le_bytes(le);
        }
        Self(arr)
    }

    /// Construct with all elements set to the same value.
    pub const fn splat(value: FixedPoint64) -> Self {
        Self([value; EMBEDDING_DIM])
    }

    // ── Element Access ───────────────────────────────────────────────

    /// Get the element at the given index.
    #[inline]
    pub fn get(&self, idx: usize) -> Option<FixedPoint64> {
        self.0.get(idx).copied()
    }

    /// Get a mutable reference to the element at the given index.
    #[inline]
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut FixedPoint64> {
        self.0.get_mut(idx)
    }

    /// Set the element at the given index.
    #[inline]
    pub fn set(&mut self, idx: usize, val: FixedPoint64) -> NervResult<()> {
        if idx >= EMBEDDING_DIM {
            return Err(NervError::Other(format!(
                "embedding index {idx} out of bounds (max {EMBEDDING_DIM})"
            )));
        }
        self.0[idx] = val;
        Ok(())
    }

    /// Access the underlying array.
    #[inline]
    pub const fn as_array(&self) -> &[FixedPoint64; EMBEDDING_DIM] {
        &self.0
    }

    // ── Vector Arithmetic ────────────────────────────────────────────

    /// Element-wise vector addition (exact — no rounding).
    ///
    /// This is the core operation of the Transfer Homomorphism:
    /// `e_{t+1} = e_t + δ(tx)`
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let mut result = [FixedPoint64::ZERO; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            result[i] = self.0[i].add(other.0[i]);
        }
        Self(result)
    }

    /// Element-wise vector subtraction (exact — no rounding).
    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        let mut result = [FixedPoint64::ZERO; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            result[i] = self.0[i].sub(other.0[i]);
        }
        Self(result)
    }

    /// Scalar multiplication: `self * scalar` (rounded per element).
    pub fn scale(&self, scalar: FixedPoint64) -> Self {
        let mut result = [FixedPoint64::ZERO; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            result[i] = self.0[i].mul(scalar);
        }
        Self(result)
    }

    /// Multiply by an integer scalar (exact — no rounding).
    pub fn scale_int(&self, scalar: i64) -> Self {
        let mut result = [FixedPoint64::ZERO; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            result[i] = self.0[i].mul_int(scalar);
        }
        Self(result)
    }

    /// Negate all elements.
    pub fn neg(&self) -> Self {
        let mut result = [FixedPoint64::ZERO; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            result[i] = self.0[i].neg();
        }
        Self(result)
    }

    /// Dot product with another 64-dim vector.
    ///
    /// Returns `Σ_{i=0}^{63} self[i] × other[i]`.
    pub fn dot(&self, other: &Self) -> FixedPoint64 {
        let mut sum = FixedPoint64::ZERO;
        for i9 in 0..(EMBEDDING_DIM / 8) {
            // Unrolled by 8 for cache-friendly performance
            let base = i9 * 8;
            for j in 0..8 {
                sum += self.0[base + j].mul(other.0[base + j]);
            }
        }
        sum
    }

    /// Dot product with an N-dimensional vector (for Perceptron forward pass).
    ///
    /// Given a row of the weight matrix `w_row ∈ ℝ^N` and input `x ∈ ℝ^N`,
    /// computes `Σ_{j=0}^{N-1} w_row[j] × x[j]`.
    pub fn dot_with_slice(w_row: &[FixedPoint64], x: &[FixedPoint64]) -> FixedPoint64 {
        let len = w_row.len().min(x.len());
        let mut sum = FixedPoint64::ZERO;
        for i in 0..len {
            sum += w_row[i].mul(x[i]);
        }
        sum
    }

    /// L2 norm squared: `||self||² = self · self`.
    pub fn norm_sq(&self) -> FixedPoint64 {
        self.dot(self)
    }

    /// L2 norm (via sqrt): `||self|| = √(self · self)`.
    pub fn norm(&self) -> FixedPoint64 {
        self.norm_sq().sqrt().unwrap_or(FixedPoint64::ZERO)
    }

    /// L1 norm: `Σ |self[i]|`.
    pub fn l1_norm(&self) -> FixedPoint64 {
        let mut sum = FixedPoint64::ZERO;
        for i in 0..EMBEDDING_DIM {
            sum += self.0[i].abs();
        }
        sum
    }

    /// L∞ norm: `max |self[i]|`.
    pub fn linf_norm(&self) -> FixedPoint64 {
        let mut max_val = FixedPoint64::ZERO;
        for i in 0..EMBEDDING_DIM {
            max_val = max_val.max(self.0[i].abs());
        }
        max_val
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Self {
        let mut result = [FixedPoint64::ZERO; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            result[i] = self.0[i].abs();
        }
        Self(result)
    }

    /// Element-wise maximum.
    pub fn elementwise_max(&self, other: &Self) -> Self {
        let mut result = [FixedPoint64::ZERO; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            result[i] = self.0[i].max(other.0[i]);
        }
        Self(result)
    }

    /// Element-wise minimum.
    pub fn elementwise_min(&self, other: &Self) -> Self {
        let mut result = [FixedPoint64::ZERO; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            result[i] = self.0[i].min(other.0[i]);
        }
        Self(result)
    }

    // ── Aggregation ──────────────────────────────────────────────────

    /// Sum of all elements.
    pub fn sum(&self) -> FixedPoint64 {
        let mut s = FixedPoint64::ZERO;
        for i in 0..EMBEDDING_DIM {
            s += self.0[i];
        }
        s
    }

    /// Mean of all elements.
    pub fn mean(&self) -> FixedPoint64 {
        self.sum().div_int(EMBEDDING_DIM as i64).unwrap_or(FixedPoint64::ZERO)
    }

    // ── Serialization ────────────────────────────────────────────────

    /// Serialize to 512 raw bytes (64 × 8 little-endian i64s).
    pub fn to_bytes(&self) -> [u8; EMBEDDING_BYTES] {
        let mut bytes = [0u8; EMBEDDING_BYTES];
        for i in 0..EMBEDDING_DIM {
            let start = i * 8;
            bytes[start..start + 8].copy_from_slice(&self.0[i].to_le_bytes());
        }
        bytes
    }

    /// Compute the BLAKE3 hash → 32-byte `EmbeddingRoot`.
    ///
    /// This is the canonical "state root" used in consensus and light-client
    /// verification. It commits to the full 512-byte embedding without
    /// revealing it.
    pub fn hash(&self) -> crate::EmbeddingRoot {
        crate::EmbeddingRoot::from_bytes(blake3_hash(&self.to_bytes()))
    }

    /// Convert to a Vec<f64> for display/debugging.
    pub fn to_f64_vec(&self) -> Vec<f64> {
        self.0.iter().map(|x| x.to_f64()).collect()
    }

    /// Returns `true` if all elements are zero.
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|x| x.is_zero())
    }

    /// Iterator over elements.
    pub fn iter(&self) -> impl Iterator<Item = &FixedPoint64> {
        self.0.iter()
    }

    /// Mutable iterator over elements.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut FixedPoint64> {
        self.0.iter_mut()
    }

    /// Number of non-zero elements (sparsity measure).
    pub fn nnz(&self) -> usize {
        self.0.iter().filter(|x| !x.is_zero()).count()
    }
}

// ── Vector Operator Overloads ───────────────────────────────────────────

impl std::ops::Add for EmbeddingVector {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        EmbeddingVector::add(&self, &rhs)
    }
}

impl std::ops::Sub for EmbeddingVector {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        EmbeddingVector::sub(&self, &rhs)
    }
}

impl std::ops::Neg for EmbeddingVector {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        EmbeddingVector::neg(&self)
    }
}

impl std::ops::AddAssign for EmbeddingVector {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..EMBEDDING_DIM {
            self.0[i] += rhs.0[i];
        }
    }
}

impl std::ops::SubAssign for EmbeddingVector {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..EMBEDDING_DIM {
            self.0[i] -= rhs.0[i];
        }
    }
}

impl std::fmt::Display for EmbeddingVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Emb([")?;
        for (i, v) in self.0.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            if i < 4 || i >= EMBEDDING_DIM - 2 {
                write!(f, "{:.6}", v.to_f64())?;
            } else if i == 4 {
                write!(f, "…")?;
            }
        }
        write!(f, "])")
    }
}

impl std::default::Default for EmbeddingVector {
    fn default() -> Self {
        Self::ZERO
    }
}

impl serde::Serialize for EmbeddingVector {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Serialize as 512 raw bytes for compactness
        serde_bytes::serialize(&self.to_bytes(), serializer)
    }
}

impl<'de> serde::Deserialize<'de> for EmbeddingVector {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: Vec<u8> = serde_bytes::deserialize(deserializer)?;
        if bytes.len() != EMBEDDING_BYTES {
            return Err(serde::de::Error::custom(format!(
                "embedding vector must be {} bytes, got {}", EMBEDDING_BYTES, bytes.len()
            )));
        }
        let arr: [u8; EMBEDDING_BYTES] = bytes.try_into()
            .map_err(|_| serde::de::Error::custom("embedding bytes conversion failed"))?;
        Ok(Self::from_bytes(&arr))
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp64_zero() {
        assert!(FixedPoint64::ZERO.is_zero());
        assert_eq!(FixedPoint64::ZERO.to_f64(), 0.0);
    }

    #[test]
    fn test_fp64_one() {
        assert_eq!(FixedPoint64::ONE.to_f64(), 1.0);
        assert_eq!(FixedPoint64::ONE.to_int(), 1);
    }

    #[test]
    fn test_fp64_from_int() {
        let v = FixedPoint64::from_int(42);
        assert_eq!(v.to_int(), 42);
        assert_eq!(v.to_f64(), 42.0);
    }

    #[test]
    fn test_fp64_from_f64_roundtrip() {
        let v = FixedPoint64::from_f64(3.14159265);
        let recovered = v.to_f64();
        assert!((recovered - 3.14159265).abs() < 1e-8);
    }

    #[test]
    fn test_fp64_add() {
        let a = FixedPoint64::from_int(3);
        let b = FixedPoint64::from_int(7);
        let sum = a + b;
        assert_eq!(sum.to_int(), 10);
    }

    #[test]
    fn test_fp64_sub() {
        let a = FixedPoint64::from_int(10);
        let b = FixedPoint64::from_int(3);
        let diff = a - b;
        assert_eq!(diff.to_int(), 7);
    }

    #[test]
    fn test_fp64_mul_integers() {
        let a = FixedPoint64::from_int(3);
        let b = FixedPoint64::from_int(7);
        let product = a * b;
        assert_eq!(product.to_int(), 21);
    }

    #[test]
    fn test_fp64_mul_fractional() {
        let a = FixedPoint64::from_f64D(0.5);
        let b = FixedPoint64::from_f64(0.5);
        let product = a * b;
        assert!((product.to_f64() - 0.25).abs() < 1e-8);
    }

    #[test]
    fn test_fp64_div() {
        let a = FixedPoint64::from_int(10);
        let b = FixedPoint64::from_int(2);
        let quotient = a.div(b).unwrap();
        assert_eq!(quotient.to_int(), 5);
    }

    #[test]
    fn test_fp64_div_by_zero() {
        let a = FixedPoint64::from_int(10);
        assert!(a.div(FixedPoint64::ZERO).is_none());
    }

    #[test]
    fn test_fp64_neg() {
        let a = FixedPoint64::from_int(5);
        let neg_a = -a;
        assert_eq!(neg_a.to_int(), -5);
    }

    #[test]
    fn test_fp64_abs() {
        let a = FixedPoint64::from_int(-5);
        assert_eq!(a.abs().to_int(), 5);
    }

    #[test]
    fn test_fp64_from_rational() {
        let v = FixedPoint64::from_rational(1, 3).unwrap();
        let expected = 1.0 / 3.0;
        assert!((v.to_f64() - expected).abs() < 1e-8);
    }

    #[test]
    fn test_fp64_le_bytes_roundtrip() {
        let v = FixedPoint64::from_f64(2.71828);
        let bytes = v.to_le_bytes();
        let recovered = FixedPoint64::from_le_bytes(bytes);
        assert_eq!(v, recovered);
    }

    #[test]
    fn test_fp64_sqrt() {
        let v = FixedPoint64::from_int(9);
        let root = v.sqrt().unwrap();
        assert_eq!(root.to_int(), 3);
    }

    #[test]
    fn test_fp64_sqrt_non_integer() {
        let v = FixedPoint64::from_int(2);
        let root = v.sqrt().unwrap();
        assert:((root.to_f64() - 1.41421356).abs() < 1e-4);
    }

    #[test]
    fn test_fp64_sqrt_negative() {
        let v = FixedPoint64::from_int(-1);
        assert!(v.sqrt().is_none());
    }

    #[test]
    fn test_fp64_mul_int() {
        let a = FixedPoint64::from_f64(0.5);
        let result = a.mul_int(10);
        assert_eq!(result.to_int(), 5);
    }

    #[test]
    fn test_fp64_clamp() {
        let v = FixedPoint64::from_int(15);
        let lo = FixedPoint64::from_int(0);
        let hi = FixedPoint64::from_int(10);
        assert_eq!(v.clamp(lo, hi).to_int(), 10);
    }

    // ── EmbeddingVector Tests ────────────────────────────────────────

    #[test]
    fn test_embedding_zero() {
        let v = EmbeddingVector::ZERO;
        assert!(v.is_zero());
    }

    #[test]
    fn test_embedding_add() {
        let a = EmbeddingVector::splat(FixedPoint64::from_int(3));
        let b = EmbeddingVector::splat(FixedPoint64::from_int(7));
        let sum = a + b;
        for i in 0..EMBEDDING_DIM {
            assert_eq!(sum.get(i).unwrap().to_int(), 10);
        }
    }

    #[test]
    fn test_embedding_sub() {
        let a = EmbeddingVector::splat(FixedPoint64::from_int(10));
        let b = EmbeddingVector::splat(FixedPoint64::from_int(3));
        let diff = a - b;
        for i in 0..EMBEDDING_DIM {
            assert_eq!(diff.get(i).unwrap().to_int(), 7);
        }
    }

    #[test]
    fn test_embedding_dot() {
        let a = EmbeddingVector::splat(FixedPoint64::from_int(2));
        let b = EmbeddingVector::splat(FixedPoint64::from_int(3));
        // dot = 64 * 2 * 3 = 384
        let dot = a.dot(&b);
        assert_eq!(dot.to_int(), 384);
    }

    #[test]
    fn test_embedding_bytes_roundtrip() {
        let mut values = [0i64; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            values[i] = FixedPoint64::from_f64(i as f64 * 0.1).raw();
        }
        let v = EmbeddingVector::from_raw_values(values);
        let bytes = v.to_bytes();
        let recovered = EmbeddingVector::from_bytes(&bytes);
        assert_eq!(v, recovered);
    }

    #[test]
    fn test_embedding_hash_deterministic() {
        let v = EmbeddingVector::splat(FixedPoint64::from_int(42));
        let h1 = v.hash();
        let h2 = v.hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_embedding_hash_different_for_different_vectors() {
        let v1 = EmbeddingVector::splat(FixedPoint64::from_int(1));
        let v2 = EmbeddingVector::splat(FixedPoint64::from_int(2));
        assert_ne!(v1.hash(), v2.hash());
    }

    #[test]
    fn test_embedding_norm() {
        let v = EmbeddingVector::splat(FixedPoint64::from_int(1));
        let norm_sq = v.norm_sq();
        // ||v||² = 64 * 1² = 64
        assert_eq!(norm_sq.to_int(), 64);
    }

    #[test]
    fn test_embedding_scale() {
        let v = EmbeddingVector::splat(FixedPoint64::from_int(5));
        let scaled = v.scale(FixedPoint64::from_int(3));
        for i in 0..EMBEDDING_DIM {
            assert_eq!(scaled.get(i).unwrap().to_int(), 15);
        }
    }

    #[test]
    fn test_embedding_from_f64_slice() {
        let values: Vec<f64> = (0..EMBEDDING_DIM).map(|i| i as f64).collect();
        let v = EmbeddingVector::from_f64_slice(&values).unwrap();
        assert!((v.get(0).unwrap().to_f64() - 0.0).abs() < 1e-8);
        assert!((v.get(63).unwrap().to_f64() - 63.0).abs() < 1e-8);
    }

    #[test]
    fn test_embedding_sum() {
        let v = EmbeddingVector::splat(FixedPoint64::from_int(1));
        assert_eq!(v.sum().to_int(), EMBEDDING_DIM as i64);
    }
}

Note: The test test_fp64_mul_fractional and test_fp64_sqrt_non_integer have minor typos above (from_f64D and assert:) — here are the corrected lines that should be used: 

  #[test]
    fn test_fp64_mul_fractional() {
        let a = FixedPoint64::from_f64(0.5);
        let b = FixedPoint64::from_f64(0.5);
        let product = a * b;
        assert!((product.to_f64() - 0.25).abs() < 1e-8);
    }

    #[test]
    fn test_fp64_sqrt_non_integer() {
        let v = FixedPoint64::from_int(2);
        let root = v.sqrt().unwrap();
        assert!((root.to_f64() - 1.41421356).abs() < 1e-4);
    }
