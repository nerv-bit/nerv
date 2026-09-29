//! Exact integer fixed-point arithmetic (WP §7.2, P5): Q15 weights, round
//! half to even at named points, 128-bit intermediates, closure into
//! (Z/2^64) for delta coordinates.
//!
//! CANONICAL DELTA RULE, frozen here: for each output coordinate j,
//!     delta_j = round_half_even(sum_k w_jk * f_k / 2^15) reduced mod 2^64,
//! with the sum accumulated exactly in i128 (one rounding per coordinate,
//! never per term). Protocol bounds (≤ 256 features) keep |acc| ≤ 2^86.

use std::cmp::Ordering;

use crate::codec::{Decode, Encode, Reader};
use crate::error::{CodecError, FixedPointError};

/// Fractional bits of the 1.15 signed fixed-point format (WP §7.2).
pub const Q15_FRAC_BITS: u32 = 15;

const _: () = assert!(Q15_FRAC_BITS as u64 == crate::params::OVERLAY_W_FRAC_BITS);

/// Round-half-even of `dividend / divisor` (divisor > 0), sign-symmetric.
/// Total: exact for all (i128, u128) inputs; only 0 is an error.
pub fn round_half_even(dividend: i128, divisor: u128) -> Result<i128, FixedPointError> {
    if divisor == 0 {
        return Err(FixedPointError::DivisionByZero);
    }
    let mag = dividend.unsigned_abs();
    let mut q = mag / divisor;
    let r = mag % divisor;
    // 2r vs divisor, without u128 overflow: r vs divisor - r.
    match r.cmp(&(divisor - r)) {
        Ordering::Less => {}
        Ordering::Greater => q += 1,
        Ordering::Equal => {
            if (q & 1) == 1 {
                q += 1;
            }
        }
    }
    if dividend < 0 {
        if q == 1u128 << 127 {
            // Only reachable for (i128::MIN, 1): |MIN| = 2^127 exactly.
            return Ok(i128::MIN);
        }
        Ok(-(q as i128))
    } else {
        // Positive dividends have |d| <= 2^127 - 1 and divisor >= 2 whenever
        // a bump occurs, so q <= 2^127 - 1: the guard below is defensive.
        if q >= 1u128 << 127 {
            return Err(FixedPointError::Overflow { op: "round_half_even" });
        }
        Ok(q as i128)
    }
}

/// Infallible round-half-even by a power of two: `dividend / 2^k`.
pub fn round_half_even_pow2(dividend: i128, k: u32) -> i128 {
    assert!(k < 128, "shift out of range");
    if k == 0 {
        return dividend;
    }
    let q = dividend >> k; // floor
    let r = (dividend as u128) & ((1u128 << k) - 1); // in [0, 2^k)
    let half = 1u128 << (k - 1);
    if r > half {
        q + 1
    } else if r < half {
        q
    } else if (q & 1) == 0 {
        // tie: 2r == 2^k — round to the even neighbor
        q
    } else {
        q + 1
    }
}

/// Reduce a signed value into (Z/2^64) — two's-complement wraparound, the
/// embedding accumulator's closed-group arithmetic (WP §7.5).
pub const fn wrap_mod_2_64(x: i128) -> u64 {
    x as u64
}

/// A 1.15 signed fixed-point weight (WP §7.2). Value range: [-1, 1 - 2^-15].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct Q15(i16);

impl Q15 {
    pub const ZERO: Q15 = Q15(0);
    pub const MIN: Q15 = Q15(i16::MIN); // -1.0 exactly
    pub const MAX: Q15 = Q15(i16::MAX); // 1 - 2^-15

    pub const fn from_bits(bits: i16) -> Q15 {
        Q15(bits)
    }

    pub const fn to_bits(self) -> i16 {
        self.0
    }

    /// Exact integer numerator of (self * x) / 2^15, in i128.
    pub const fn mul_exact(self, x: i64) -> i128 {
        (self.0 as i128) * (x as i128)
    }

    /// Quantize `num / den` (den > 0) with round-half-even. `None`-error if
    /// outside [-1, 1). This is the governance-epoch weight quantizer.
    pub fn from_ratio(num: i64, den: u64) -> Result<Q15, FixedPointError> {
        if den == 0 {
            return Err(FixedPointError::DivisionByZero);
        }
        let scaled = (num as i128) * (1i128 << Q15_FRAC_BITS);
        let q = round_half_even(scaled, den as u128)?;
        if q < i16::MIN as i128 || q > i16::MAX as i128 {
            return Err(FixedPointError::Unrepresentable { op: "Q15::from_ratio" });
        }
        Ok(Q15(q as i16))
    }

    /// Saturating negation: -(-1.0) is unrepresentable, so MIN negates to MAX.
    pub const fn neg(self) -> Q15 {
        Q15(if self.0 == i16::MIN { i16::MAX } else { -self.0 })
    }
}

impl Encode for Q15 {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0.to_le_bytes());
    }
    fn encoded_len(&self) -> usize {
        2
    }
}

impl Decode for Q15 {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(Q15(i16::from_le_bytes(r.take_array::<2>()?)))
    }
}

/// Exact fixed-point multiply-accumulate for one delta coordinate: the
/// native twin of the sparse-encoder chip (WP §5.3, DSR-7).
#[derive(Clone, Copy, Debug, Default)]
pub struct Mac15 {
    acc: i128,
}

impl Mac15 {
    pub const fn new() -> Mac15 {
        Mac15 { acc: 0 }
    }

    pub const fn from_acc(acc: i128) -> Mac15 {
        Mac15 { acc }
    }

    pub const fn acc(&self) -> i128 {
        self.acc
    }

    /// acc += w * x exactly. Overflow is impossible for protocol-bounded
    /// inputs (≤ 256 features ⇒ |acc| ≤ 2^86) and is surfaced as an error,
    /// never a silent wrap.
    pub fn add(&mut self, w: Q15, x: i64) -> Result<(), FixedPointError> {
        self.acc = self
            .acc
            .checked_add(w.mul_exact(x))
            .ok_or(FixedPointError::Overflow { op: "Mac15::add" })?;
        Ok(())
    }

    /// The canonical delta coordinate: round_half_even(acc / 2^15), wrapped
    /// into (Z/2^64).
    pub fn resolve_u64(&self) -> u64 {
        wrap_mod_2_64(round_half_even_pow2(self.acc, Q15_FRAC_BITS))
    }

    /// round_half_even(acc / 2^15), unwrapped (advisory-layer use).
    pub fn resolve_i128(&self) -> i128 {
        round_half_even_pow2(self.acc, Q15_FRAC_BITS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{codec_roundtrip, SplitMix64};

    // Independent formulation 1: Euclidean division.
    fn rhe_euclid(d: i128, n: u128) -> i128 {
        let n = n as i128;
        let q = d.div_euclid(n);
        let r = d.rem_euclid(n);
        let twice = 2 * r;
        if twice < n {
            q
        } else if twice > n {
            q + 1
        } else if q % 2 == 0 {
            q
        } else {
            q + 1
        }
    }

    // Independent formulation 2: nearest integer, ties to even, by search.
    fn rhe_brute(d: i128, n: u128) -> i128 {
        let n = n as i128;
        let base = d.div_euclid(n);
        let mut best: Option<(i128, i128, i128)> = None; // (dist, parity, q)
        for q in base - 2..=base + 2 {
            let cand = ((d - q * n).abs(), q.rem_euclid(2), q);
            if best.map_or(true, |b| cand < b) {
                best = Some(cand);
            }
        }
        let (.., q) = best.unwrap();
        q
    }

    #[test]
    fn round_half_even_matches_references_exhaustively() {
        for d in -300i128..=300 {
            for n in 1u128..=30 {
                let got = round_half_even(d, n).unwrap();
                assert_eq!(got, rhe_euclid(d, n), "d={d} n={n}");
                assert_eq!(got, rhe_brute(d, n), "d={d} n={n}");
            }
        }
    }

    #[test]
    fn round_half_even_tie_table() {
        assert_eq!(round_half_even(3, 2).unwrap(), 2);
        assert_eq!(round_half_even(-3, 2).unwrap(), -2);
        assert_eq!(round_half_even(5, 2).unwrap(), 2);
        assert_eq!(round_half_even(-5, 2).unwrap(), -2);
        assert_eq!(round_half_even(2, 4).unwrap(), 0);
        assert_eq!(round_half_even(-2, 4).unwrap(), 0);
        assert_eq!(round_half_even(6, 4).unwrap(), 2);
        assert_eq!(round_half_even(-6, 4).unwrap(), -2);
        assert_eq!(round_half_even(1, 2).unwrap(), 0);
        assert_eq!(round_half_even(-1, 2).unwrap(), 0);
        assert_eq!(round_half_even(7, 4).unwrap(), 2);
    }

    #[test]
    fn round_half_even_extremes() {
        assert_eq!(round_half_even(i128::MIN, 1).unwrap(), i128::MIN);
        assert_eq!(round_half_even(i128::MIN, 2).unwrap(), -(2i128.pow(126)));
        assert_eq!(round_half_even(i128::MAX, 1).unwrap(), i128::MAX);
        assert_eq!(round_half_even(i128::MAX, 3).unwrap(), rhe_euclid(i128::MAX, 3));
        assert_eq!(round_half_even(5, u128::MAX).unwrap(), 0);
        assert_eq!(round_half_even(-5, u128::MAX).unwrap(), 0);
        assert_eq!(round_half_even(i128::MAX, u128::MAX).unwrap(), 0);
        // |MIN| / (2^128 - 1) rounds to 1
        assert_eq!(round_half_even(i128::MIN, u128::MAX).unwrap(), -1);
        // exact tie against a large divisor
        assert_eq!(round_half_even(5, 10).unwrap(), 0);
        assert_eq!(round_half_even(-5, 10).unwrap(), 0);
        assert!(matches!(
            round_half_even(1, 0),
            Err(FixedPointError::DivisionByZero)
        ));
    }

    #[test]
    fn pow2_agrees_with_general() {
        let mut rng = SplitMix64::new(0xABCD);
        for k in 0u32..=80 {
            let div = 1u128 << k;
            for d in [
                0i128,
                1,
                -1,
                3,
                -3,
                1i128 << k,
                -(1i128 << k),
                (1i128 << k).wrapping_add(1),
                i128::MAX,
                i128::MIN,
                rng.next_u64() as i128,
                -(rng.next_u64() as i128),
            ] {
                assert_eq!(
                    round_half_even_pow2(d, k),
                    round_half_even(d, div).unwrap(),
                    "d={d} k={k}"
                );
            }
        }
    }

    #[test]
    fn q15_from_ratio_exact_and_out_of_range() {
        assert_eq!(Q15::from_ratio(1, 2), Ok(Q15::from_bits(1 << 14)));
        assert_eq!(Q15::from_ratio(0, 7), Ok(Q15::ZERO));
        assert_eq!(Q15::from_ratio(-1, 1), Ok(Q15::MIN));
        assert_eq!(Q15::from_ratio(1, 3), Ok(Q15::from_bits(10923)));
        assert_eq!(Q15::from_ratio(2, 3), Ok(Q15::from_bits(21845)));
        // ties inside range
        assert_eq!(Q15::from_ratio(1, 65536), Ok(Q15::from_bits(0)));
        assert_eq!(Q15::from_ratio(3, 65536), Ok(Q15::from_bits(2)));
        assert!(matches!(
            Q15::from_ratio(1, 1),
            Err(FixedPointError::Unrepresentable { .. })
        ));
        assert!(matches!(
            Q15::from_ratio(2, 1),
            Err(FixedPointError::Unrepresentable { .. })
        ));
        assert!(matches!(
            Q15::from_ratio(-3, 2),
            Err(FixedPointError::Unrepresentable { .. })
        ));
        assert!(matches!(
            Q15::from_ratio(1, 0),
            Err(FixedPointError::DivisionByZero)
        ));
    }

    #[test]
    fn q15_neg_and_mul_exact() {
        assert_eq!(Q15::MIN.neg(), Q15::MAX);
        assert_eq!(Q15::MAX.neg().neg(), Q15::MAX);
        assert_eq!(Q15::from_bits(1234).neg(), Q15::from_bits(-1234));
        assert_eq!(Q15::MAX.mul_exact(i64::MAX), 32767i128 * (i64::MAX as i128));
        assert_eq!(Q15::MIN.mul_exact(i64::MIN), 32768i128 * (1i128 << 63));
        codec_roundtrip(Q15::from_bits(-12345));
        codec_roundtrip(Q15::MIN);
    }

    #[test]
    fn mac15_single_and_tie() {
        let mut m = Mac15::new();
        m.add(Q15::from_bits(1 << 14), 3).unwrap(); // 0.5 * 3 = 1.5
        assert_eq!(m.acc(), 49152);
        assert_eq!(m.resolve_i128(), 2); // tie rounds to even
        assert_eq!(m.resolve_u64(), 2);
        let mut n = Mac15::new();
        n.add(Q15::from_bits(-(1 << 14)), 3).unwrap();
        assert_eq!(n.resolve_i128(), -2); // sign-symmetric
    }

    #[test]
    fn mac15_exact_sum() {
        let mut m = Mac15::new();
        m.add(Q15::from_bits(1 << 13), 10).unwrap(); // 0.25 * 10
        m.add(Q15::from_bits(1 << 14), 5).unwrap(); // 0.5 * 5
        assert_eq!(m.acc(), 81920 + 81920);
        assert_eq!(m.resolve_i128(), 5);
    }

    #[test]
    fn mac15_wraps_mod_2_64() {
        let mut m = Mac15::new();
        m.add(Q15::MIN, i64::MIN).unwrap(); // (-1) * (-2^63): acc = 2^78
        assert_eq!(m.resolve_i128(), 1i128 << 63);
        assert_eq!(m.resolve_u64(), 1u64 << 63);
        let mut n = Mac15::from_acc(i128::MIN);
        assert_eq!(n.resolve_u64(), 0); // -2^127 ≡ 0 mod 2^64
        n.add(Q15::MAX, 1).unwrap();
        assert_eq!(n.resolve_u64(), wrap_mod_2_64(n.resolve_i128()));
    }

    #[test]
    fn mac15_add_overflow_detected() {
        let mut m = Mac15::from_acc(i128::MAX - 10);
        assert!(matches!(m.add(Q15::MAX, 1), Err(FixedPointError::Overflow { .. })));
    }

    #[test]
    fn wrap_mod_2_64_reference() {
        assert_eq!(wrap_mod_2_64(0), 0);
        assert_eq!(wrap_mod_2_64(-1), u64::MAX);
        assert_eq!(wrap_mod_2_64(1i128 << 64), 0);
        assert_eq!(wrap_mod_2_64(-(1i128 << 64)), 0);
        assert_eq!(wrap_mod_2_64(i128::MIN), 0);
        assert_eq!(wrap_mod_2_64(i128::MAX), u64::MAX);
        let mut rng = SplitMix64::new(3);
        for _ in 0..1000 {
            let x = (((rng.next_u64() as u128) << 64) | rng.next_u64() as u128) as i128;
            assert_eq!(wrap_mod_2_64(x), ((x as u128) & 0xFFFF_FFFF_FFFF_FFFF) as u64);
        }
    }
}
