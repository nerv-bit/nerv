//! Balance types and arithmetic for the NERV private-by-default ledger.
//!
//! In NERV, balances are **never** visible on-chain. The canonical state
//! is represented as a 512-byte neural embedding — individual account
//! balances exist only as latent features in the embedding vector and
//! cannot be extracted without solving the Neural Network Inversion
//! Problem (a novel post-quantum hardness assumption).
//!
//! This module defines:
//! - [`Amount`] — A fixed-precision NERV amount (u64 nano-NERV)
//! - [`BalanceCommitment`] — A blinded Pedersen-like commitment to a balance
//! - [`BalanceDelta`] — A signed change in balance (credit or debit)
//! - [`BalanceDirection`] — Credit/Debit enum
//!
//! Client-side (wallet) balance tracking uses these types directly.
//! On-chain, only the aggregate embedding delta is visible.

use crate::{
    NervError, NervResult, ONE_NERV, TOTAL_SUPPLY_NANO,
    TOTAL_SUPPLY_NERV, NERV_DECIMALS,
};
use crate::utils::{blake3_keyed_hash, blake3_hash, ct_eq, secure_zero};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;
use zeroize::Zeroize;

// ─── Amount ─────────────────────────────────────────────────────────────

/// A fixed-precision NERV amount stored as nano-NERV (10⁻⁹ NERV) in a `u64`.
///
/// # Invariants
/// - The inner value is always in `[0, TOTAL_SUPPLY_NANO]`.
/// - Arithmetic is checked — overflow returns `Err` rather than panicking.
/// - Serialization uses the raw nano-NERV value for compactness.
///
/// # Example
/// ```
/// use nerv::Amount;
/// let one = Amount::ONE_NERV;
/// let half = Amount::from_nerv_f64(0.5).unwrap();
/// let total = one.checked_add(half).unwrap();
/// assert_eq!(total.to_nerv_f64(), 1.5);
/// ```
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
    Serialize, Deserialize, Zeroize,
)]
#[zeroize(drop)]
pub struct Amount(
    /// The amount in nano-NERV (10⁻⁹ NERV).
    u64,
);

impl Amount {
    /// Zero NERV.
    pub const ZERO: Self = Self(0);

    /// One NERV (10⁹ nano-NERV).
    pub const ONE_NERV: Self = Self(ONE_NERV);

    /// The total supply (10 billion NERV in nano-NERV).
    pub const MAX_SUPPLY: Self = Self(TOTAL_SUPPLY_NANO);

    /// Minimum transaction amount (1 nano-NERV).
    pub const MIN_TX: Self = Self(1);

    /// Create an `Amount` from nano-NERV.
    ///
    /// Returns `Err` if the value exceeds the total supply.
    #[inline]
    pub const fn from_nano(nano: u64) -> NervResult<Self> {
        if nano <= TOTAL_SUPPLY_NANO {
            Ok(Self(nano))
        } else {
            Err(NervError::Other("amount exceeds total supply".into()))
        }
    }

    /// Create an `Amount` from nano-NERV without supply checking.
    ///
    /// Use this only for intermediate calculations where the value
    /// is guaranteed to be in range.
    #[inline]
    pub const fn from_nano_unchecked(nano: u64) -> Self {
        Self(nano)
    }

    /// Create an `Amount` from whole NERV.
    ///
    /// Returns `Err` if the result exceeds the total supply.
    #[inline]
    pub const fn from_nerv(nerv: u64) -> NervResult<Self> {
        match nerv.checked_mul(ONE_NERV) {
            Some(nano) if nano <= TOTAL_SUPPLY_NANO => Ok(Self(nano)),
            _ => Err(NervError::Other("amount exceeds total supply".into())),
        }
    }

    /// Create an `Amount` from a floating-point NERV value.
    ///
    /// **Warning**: Floating-point arithmetic is lossy. This constructor
    /// is intended for user-facing input parsing only. For protocol logic,
    /// always use `from_nano` or `from_nerv`.
    pub fn from_nerv_f64(val: f64) -> NervResult<Self> {
        if val < 0.0 {
            return Err(NervError::Other("negative amount".into()));
        }
        let nano = (val * ONE_NERV as f64).round() as u64;
        Self::from_nano(nano)
    }

    /// Return the amount in nano-NERV.
    #[inline]
    pub const fn as_nano(&self) -> u64 {
        self.0
    }

    /// Return the amount in whole NERV (truncating fractional part).
    #[inline]
    pub const fn as_nerv(&self) -> u64 {
        self.0 / ONE_NERV
    }

    /// Return the fractional part in nano-NERV.
    #[inline]
    pub const fn fractional_nano(&self) -> u64 {
        self.0 % ONE_NERV
    }

    /// Convert to f64 NERV (for display purposes).
    pub fn to_nerv_f64(&self) -> f64 {
        self.0 as f64 / ONE_NERV as f64
    }

    /// Checked addition. Returns `Err` on overflow or if the result exceeds supply.
    #[inline]
    pub const fn checked_add(&self, other: Self) -> NervResult<Self> {
        match self.0.checked_add(other.0) {
            Some(sum) if sum <= TOTAL_SUPPLY_NANO => Ok(Self(sum)),
            Some(_) => Err(NervError::Other("sum exceeds total supply".into())),
            None => Err(NervError::Other("arithmetic overflow".into())),
        }
    }

    /// Checked subtraction. Returns `Err` on underflow.
    #[inline]
    pub const fn checked_sub(&self, other: Self) -> NervResult<Self> {
        match self.0.checked_sub(other.0) {
            Some(diff) => Ok(Self(diff)),
            None => Err(NervError::InsufficientBalance {
                required: other.0,
                available: self.0,
            }),
        }
    }

    /// Saturating addition — clamps at `MAX_SUPPLY`.
    #[inline]
    pub const fn saturating_add(&self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0).min(TOTAL_SUPPLY_NANO))
    }

    /// Saturating subtraction — clamps at zero.
    #[inline]
    pub const fn saturating_sub(&self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    /// Checked multiplication by a scalar.
    #[inline]
    pub const fn checked_mul(&self, scalar: u64) -> NervResult<Self> {
        match self.0.checked_mul(scalar) {
            Some(product) if product <= TOTAL_SUPPLY_NANO => Ok(Self(product)),
            Some(_) => Err(NervError::Other("product exceeds total supply".


           Some(_) => Err(NervError::Other("product exceeds total supply".into())),
           None => Err(NervError::Other("arithmetic overflow in scalar multiply".into())),
       }
   }


   /// Checked integer division by a non-zero scalar.
   #[inline]
   pub const fn checked_div(&self, scalar: u64) -> NervResult<Self> {
       if scalar == 0 {
           return Err(NervError::Other("division by zero".into()));
       }
       Ok(Self(self.0 / scalar))
   }


   /// Returns `true` if the amount is zero.
   #[inline]
   pub const fn is_zero(&self) -> bool {
       self.0 == 0
   }


   /// Returns `true` if the amount is positive (non-zero).
   #[inline]
   pub const fn is_positive(&self) -> bool {
       self.0 > 0
   }


   /// Returns the minimum of two amounts.
   #[inline]
   pub const fn min(&self, other: Self) -> Self {
       Self(if self.0 < other.0 { self.0 } else { other.0 })
   }


   /// Returns the maximum of two amounts.
   #[inline]
   pub const fn max(&self, other: Self) -> Self {
       Self(if self.0 > other.0 { self.0 } else { other.0 })
   }


   /// Format as a human-readable NERV string with full precision.
   ///
   /// Example: `1_500_000_000` → `"1.500000000"`
   pub fn to_display_string(&self) -> String {
       let whole = self.0 / ONE_NERV;
       let frac = self.0 % ONE_NERV;
       format!("{whole}.{frac:09}")
   }


   /// Format as a human-readable NERV string with compact precision
   /// (trailing zeros stripped).
   ///
   /// Example: `1_500_000_000` → `"1.5"`
   pub fn to_compact_string(&self) -> String {
       let whole = self.0 / ONE_NERV;
       let frac = self.0 % ONE_NERV;
       if frac == 0 {
           format!("{whole}")
       } else {
}

//! Balance Types — Amounts, Commitments, and Deltas.
//!
//! NERV uses a private balance model: amounts are never visible
//! on-chain. Instead, balance changes are represented as encrypted
//! commitments and additive deltas within the neural state embedding.
//!
//! # Key Types
//!
//! - `Amount`: Unsigned amount in nano-NERV (u64)
//! - `BalanceCommitment`: Quantum-resistant commitment to a balance
//! - `BalanceDelta`: Signed change (debit/credit) in nano-NERV
//! - `BalanceDirection`: Debit or Credit

use crate::{
    ONE_NERV, TOTAL_SUPPLY_NANO,
    NervError, NervResult,
};
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;
use zeroize::Zeroize;

// ═══════════════════════════════════════════════════════════════════════
//  AMOUNT
// ═══════════════════════════════════════════════════════════════════════

/// An unsigned token amount in nano-NERV.
///
/// This is the fundamental unit of account: 1 NERV = 10⁹ nano-NERV.
/// All arithmetic is checked for overflow.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
    Serialize, Deserialize, Default,
)]
pub struct Amount(pub u64);

impl Amount {
    /// Zero amount.
    pub const ZERO: Self = Self(0);

    /// One nano-NERV (smallest unit).
    pub const ONE_NANO: Self = Self(1);

    /// One NERV (10⁹ nano-NERV).
    pub const ONE_NERV: Self = Self(ONE_NERV);

    /// Maximum possible amount (total supply).
    pub const MAX: Self = Self(TOTAL_SUPPLY_NANO);

    /// Create from nano-NERV.
    #[inline]
    pub const fn from_nano(nano: u64) -> Self {
        Self(nano)
    }

    /// Create from whole NERV.
    #[inline]
    pub const fn from_nerv(nerv: u64) -> Self {
        Self(nerv * ONE_NERV)
    }

    /// Create from a f64 NERV value (truncating fractional nano).
    pub fn from_nerv_f64(val: f64) -> Self {
        if val < 0.0 {
            return Self::ZERO;
        }
        let nano = (val * ONE_NERV as f64) as u64;
        Self(nano)
    }

    /// Get the nano-NERV value.
    #[inline]
    pub const fn as_nano(&self) -> u64 {
        self.0
    }

    /// Convert to whole NERV (truncating).
    #[inline]
    pub const fn as_nerv(&self) -> u64 {
        self.0 / ONE_NERV
    }

    /// Convert to f64 NERV (for display/calculation).
    #[inline]
    pub fn as_nerv_f64(&self) -> f64 {
        self.0 as f64 / ONE_NERV as f64
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

    /// Checked subtraction.
    #[inline]
    pub const fn checked_sub(&self, other: Self) -> Option<Self> {
        match self.0.checked_sub(other.0) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }

    /// Saturating subtraction (floor at zero).
    #[inline]
    pub const fn saturating_sub(&self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    /// Checked multiplication by a scalar.
    #[inline]
    pub const fn checked_mul(&self, scalar: u64) -> Option<Self> {
        match self.0.checked_mul(scalar) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }

    /// Division by a scalar.
    #[inline]
    pub const fn checked_div(&self, scalar: u64) -> Option<Self> {
        if scalar == 0 {
            None
        } else {
            Some(Self(self.0 / scalar))
        }
    }

    /// Remainder after division.
    #[inline]
    pub const fn checked_rem(&self, scalar: u64) -> Option<Self> {
        if scalar == 0 {
            None
        } else {
            Some(Self(self.0 % scalar))
        }
    }

    /// Check if zero.
    #[inline]
    pub const fn is_zero(&self) -> bool {
        self.0 == 0
    }

    /// Check if the amount exceeds the total supply.
    #[inline]
    pub const fn exceeds_supply(&self) -> bool {
        self.0 > TOTAL_SUPPLY_NANO
    }

    /// Compute a fee as a fraction (basis points) of this amount.
    pub fn fee_bps(&self, bps: u64) -> Self {
        Self(((self.0 as u128 * bps as u128) / 10000) as u64)
    }
}

impl std::fmt::Display for Amount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let whole = self.0 / ONE_NERV;
        let frac = self.0 % ONE_NERV;
        write!(f, "{whole}.{frac:09}")
    }
}

impl std::ops::Add for Amount {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        self.saturating_add(other)
    }
}

impl std::ops::Sub for Amount {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        self.saturating_sub(other)
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  BALANCE COMMITMENT
// ═══════════════════════════════════════════════════════════════════════

/// A quantum-resistant commitment to a private balance.
///
/// Instead of storing amounts on-chain, NERV stores commitments.
/// The commitment is: C = BLAKE3("nerv:bal" ‖ amount ‖ blinding)
///
/// This is binding (can't change amount without detecting it) and
/// quantum-resistant (based on hash, not discrete log).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BalanceCommitment {
    /// The 32-byte commitment value.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub commitment: Vec<u8>,

    /// The blinding factor (private, zeroized on drop).
    pub blinding: BlindingFactor,
}

/// A blinding factor for balance commitments.
///
/// Stored as 32 random bytes. Must be kept private.
#[derive(Debug, Clone, PartialEq, Eq, Zeroize, Serialize, Deserialize)]
#[zeroize(drop)]
pub struct BlindingFactor(
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub Vec<u8>,
);

impl BlindingFactor {
    /// Generate a random blinding factor.
    pub fn random() -> Self {
        Self(crate::utils::random_bytes(32))
    }

    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self(bytes.to_vec())
    }

    /// Get the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl BalanceCommitment {
    /// Create a commitment to an amount with a random blinding factor.
    pub fn commit(amount: &Amount) -> Self {
        let blinding = BlindingFactor::random();
        Self::commit_with_blinding(amount, &blinding)
    }

    /// Create a commitment with a specific blinding factor.
    pub fn commit_with_blinding(amount: &Amount, blinding: &BlindingFactor) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"nerv:bal");
        hasher.update(&amount.0.to_le_bytes());
        hasher.update(blinding.as_bytes());
        let commitment: [u8; 32] = hasher.finalize().into();

        Self {
            commitment: commitment.to_vec(),
            blinding: blinding.clone(),
        }
    }

    /// Verify that this commitment is consistent with a given amount.
    ///
    /// Note: This requires knowing the blinding factor.
    pub fn verify(&self, amount: &Amount) -> bool {
        let expected = Self::commit_with_blinding(amount, &self.blinding);
        crate::utils::constant_time_eq(&self.commitment, &expected.commitment)
    }

    /// Get the commitment bytes (safe to share).
    pub fn as_bytes(&self) -> &[u8] {
        &self.commitment
    }

    /// Convert to hex.
    pub fn to_hex(&self) -> String {
        hex::encode(&self.commitment)
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  BALANCE DELTA
// ═══════════════════════════════════════════════════════════════════════

/// A signed change in balance (debit or credit).
///
/// In NERV's homomorphic embedding, balance changes are additive:
/// e_{t+1} = e_t + δ(tx)
///
/// The delta can be positive (credit) or negative (debit),
/// represented as a signed 128-bit value in nano-NERV.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BalanceDelta(pub i128);

impl BalanceDelta {
    /// Zero delta (no change).
    pub const ZERO: Self = Self(0);

    /// Create a credit (positive) delta.
    #[inline]
    pub const fn credit(nano: u64) -> Self {
        Self(nano as i128)
    }

    /// Create a debit (negative) delta.
    #[inline]
    pub const fn debit(nano: u64) -> Self {
        Self(-(nano as i128))
    }

    /// Create from a signed nano-NERV value.
    #[inline]
    pub const fn from_nano(nano: i128) -> Self {
        Self(nano)
    }

    /// Create from an Amount with a direction.
    pub fn from_amount(amount: &Amount, direction: BalanceDirection) -> Self {
        match direction {
            BalanceDirection::Credit => Self::credit(amount.0),
            BalanceDirection::Debit => Self::debit(amount.0),
        }
    }

    /// Get the absolute value in nano-NERV.
    #[inline]
    pub fn abs_nano(&self) -> u64 {
        self.0.unsigned_abs() as u64
    }

    /// Get the direction.
    #[inline]
    pub fn direction(&self) -> BalanceDirection {
        if self.0 >= 0 {
            BalanceDirection::Credit
        } else {
            BalanceDirection::Debit
        }
    }

    /// Check if this is a credit (positive).
    #[inline]
    pub fn is_credit(&self) -> bool {
        self.0 > 0
    }

    /// Check if this is a debit (negative).
    #[inline]
    pub fn is_debit(&self) -> bool {
        self.0 < 0
    }

    /// Check if zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }

    /// Checked addition of two deltas.
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

    /// Negate the delta (swap debit/credit).
    #[inline]
    pub const fn neg(&self) -> Self {
        Self(-self.0)
    }

    /// Scale the delta by a factor.
    pub fn checked_mul(&self, scalar: i64) -> Option<Self> {
        self.0.checked_mul(scalar as i128).map(Self)
    }
}

impl std::ops::Add for BalanceDelta {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        self.saturating_add(other)
    }
}

impl std::ops::Neg for BalanceDelta {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl std::fmt::Display for BalanceDelta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let abs = self.abs_nano();
        let whole = abs / ONE_NERV;
        let frac = abs % ONE_NERV;
        if self.0 >= 0 {
            write!(f, "+{whole}.{frac:09}")
        } else {
            write!(f, "-{whole}.{frac:09}")
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  BALANCE DIRECTION
// ═══════════════════════════════════════════════════════════════════════

/// The direction of a balance change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BalanceDirection {
    /// Balance increased (incoming funds).
    Credit,
    /// Balance decreased (outgoing funds).
    Debit,
}

impl BalanceDirection {
    /// Negate the direction.
    #[inline]
    pub fn neg(&self) -> Self {
        match self {
            Self::Credit => Self::Debit,
            Self::Debit => Self::Credit,
        }
    }

    /// Apply to an amount to produce a delta.
    #[inline]
    pub fn apply(&self, amount: &Amount) -> BalanceDelta {
        BalanceDelta::from_amount(amount, *self)
    }
}

impl std::fmt::Display for BalanceDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Credit => write!(f, "credit"),
            Self::Debit => write!(f, "debit"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  BALANCE VALIDATION
// ═══════════════════════════════════════════════════════════════════════

/// Validate that a transfer is valid: sender has sufficient balance.
pub fn validate_transfer(
    sender_balance: &Amount,
    transfer_amount: &Amount,
    fee: &Amount,
) -> NervResult<()> {
    let total_required = transfer_amount
        .checked_add(*fee)
        .ok_or_else(|| NervError::Other("transfer + fee overflow".into()))?;

    if *sender_balance < total_required {
        return Err(NervError::InsufficientBalance {
            required: total_required.0,
            available: sender_balance.0,
        });
    }

    Ok(())
}

/// Validate that an amount doesn't exceed total supply.
pub fn validate_amount(amount: &Amount) -> NervResult<()> {
    if amount.exceeds_supply() {
        return Err(NervError::Tokenomics(format!(
            "amount {} exceeds total supply {}", amount.0, TOTAL_SUPPLY_NANO
        )));
    }
    Ok(())
}

/// Compute the net balance change from a set of deltas.
pub fn net_balance_change(deltas: &[BalanceDelta]) -> BalanceDelta {
    deltas.iter().copied().fold(BalanceDelta::ZERO, |acc, d| acc.saturating_add(d))
}

// ═══════════════════════════════════════════════════════════════════════
//  TESTS
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amount_from_nerv() {
        let a = Amount::from_nerv(100);
        assert_eq!(a.as_nerv(), 100);
        assert_eq!(a.as_nano(), 100 * ONE_NERV);
    }

    #[test]
    fn test_amount_from_nano() {
        let a = Amount::from_nano(500_000_000);
        assert_eq!(a.as_nerv(), 0);
        assert!((a.as_nerv_f64() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_amount_addition() {
        let a = Amount::from_nerv(100);
        let b = Amount::from_nerv(50);
        let sum = a.checked_add(b).unwrap();
        assert_eq!(sum, Amount::from_nerv(150));
    }

    #[test]
    fn test_amount_subtraction() {
        let a = Amount::from_nerv(100);
        let b = Amount::from_nerv(30);
        let diff = a.checked_sub(b).unwrap();
        assert_eq!(diff, Amount::from_nerv(70));
    }

    #[test]
    fn test_amount_underflow() {
        let a = Amount::from_nerv(10);
        let b = Amount::from_nerv(20);
        assert!(a.checked_sub(b).is_none());
        assert_eq!(a.saturating_sub(b), Amount::ZERO);
    }

    #[test]
    fn test_amount_overflow() {
        let a = Amount(u64::MAX);
        let b = Amount(1);
        assert!(a.checked_add(b).is_none());
    }

    #[test]
    fn test_amount_fee() {
        let a = Amount::from_nerv(1000);
        let fee = a.fee_bps(100); // 1%
        assert_eq!(fee, Amount::from_nerv(10));
    }

    #[test]
    fn test_amount_display() {
        let a = Amount::from_nerv(1234);
        let display = format!("{a}");
        assert!(display.starts_with("1234."));
    }

    #[test]
    fn test_amount_from_f64() {
        let a = Amount::from_nerv_f64(1.5);
        assert_eq!(a.as_nerv(), 1);
        assert!((a.as_nerv_f64() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_balance_commitment() {
        let amount = Amount::from_nerv(100);
        let commitment = BalanceCommitment::commit(&amount);
        assert!(commitment.verify(&amount));
    }

    #[test]
    fn test_balance_commitment_wrong_amount() {
        let amount = Amount::from_nerv(100);
        let commitment = BalanceCommitment::commit(&amount);
        let wrong_amount = Amount::from_nerv(200);
        assert!(!commitment.verify(&wrong_amount));
    }

    #[test]
    fn test_balance_commitment_deterministic() {
        let amount = Amount::from_nerv(100);
        let blinding = BlindingFactor::from_bytes(&[42u8; 32]);
        let c1 = BalanceCommitment::commit_with_blinding(&amount, &blinding);
        let c2 = BalanceCommitment::commit_with_blinding(&amount, &blinding);
        assert_eq!(c1.commitment, c2.commitment);
    }

    #[test]
    fn test_balance_delta_credit() {
        let delta = BalanceDelta::credit(100 * ONE_NERV);
        assert!(delta.is_credit());
        assert!(!delta.is_debit());
        assert_eq!(delta.abs_nano(), 100 * ONE_NERV);
    }

    #[test]
    fn test_balance_delta_debit() {
        let delta = BalanceDelta::debit(50 * ONE_NERV);
        assert!(!delta.is_credit());
        assert!(delta.is_debit());
        assert_eq!(delta.abs_nano(), 50 * ONE_NERV);
    }

    #[test]
    fn test_balance_delta_addition() {
        let credit = BalanceDelta::credit(100 * ONE_NERV);
        let debit = BalanceDelta::debit(30 * ONE_NERV);
        let net = credit.saturating_add(debit);
        assert!(net.is_credit());
        assert_eq!(net.abs_nano(), 70 * ONE_NERV);
    }

    #[test]
    fn test_balance_delta_net_zero() {
        let credit = BalanceDelta::credit(100 * ONE_NERV);
        let debit = BalanceDelta::debit(100 * ONE_NERV);
        let net = credit.saturating_add(debit);
        assert!(net.is_zero());
    }

    #[test]
    fn test_balance_delta_neg() {
        let credit = BalanceDelta::credit(100 * ONE_NERV);
        let neg = credit.neg();
        assert!(neg.is_debit());
        assert_eq!(neg.abs_nano(), 100 * ONE_NERV);
    }

    #[test]
    fn test_balance_delta_display() {
        let credit = BalanceDelta::credit(1_500_000_000);
        let debit = BalanceDelta::debit(500_000_000);
        assert!(format!("{credit}").starts_with('+'));
        assert!(format!("{debit}").starts_with('-'));
    }

    #[test]
    fn test_balance_direction() {
        assert_eq!(BalanceDirection::Credit.neg(), BalanceDirection::Debit);
        assert_eq!(BalanceDirection::Debit.neg(), BalanceDirection::Credit);
    }

    #[test]
    fn test_validate_transfer_success!        let balance = Amount::from_nerv(100);
        let transfer = Amount::from_nerv(80);
        let fee = Amount::from_nerv(10);
        assert!(validate_transfer(&balance, &transfer, &fee).is_ok());

        let insufficient = Amount::from_nerv(95);
        assert!(validate_transfer(&balance, &insufficient, &fee).is_err());
    }

    #[test]
    fn test_validate_amount() {
        let ok = Amount::from_nerv(100);
        assert!(validate_amount(&ok).is_ok());

        let too_much = Amount(u64::MAX);
        assert!(validate_amount(&too_much).is_err());
    }

    #[test]
    fn test_net_balance_change() {
        let deltas = vec![
            BalanceDelta::credit(100 * ONE_NERV),
            BalanceDelta::debit(30 * ONE_NERV),
            BalanceDelta::credit(20 * ONE_NERV),
        ];
        let net = net_balance_change(&deltas);
        assert_eq!(net, BalanceDelta::credit(90 * ONE_NERV));
    }

    #[test]
    fn test_balance_delta_from_amount() {
        let amount = Amount::from_nerv(50);
        let credit = BalanceDelta::from_amount(&amount, BalanceDirection::Credit);
        assert!(credit.is_credit());

        let debit = BalanceDelta::from_amount(&amount, BalanceDirection::Debit);
        assert!(debit.is_debit());
    }

    #[test]
    fn test_blinding_factor() {
        let bf1 = BlindingFactor::random();
        let bf2 = BlindingFactor::random();
        assert_eq!(bf1.as_bytes().len(), 32);
        // Overwhelming probability they're different
        assert_ne!(bf1, bf2);
    }
}
