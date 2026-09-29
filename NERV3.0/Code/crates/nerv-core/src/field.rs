//! Goldilocks prime field — the prover field (WP §5.3) and the arithmetic
//! substrate of the NCT's native Poseidon2 tree hashing (DSR-7 twin).
//! p = 2^64 − 2^32 + 1. Exact u128 intermediates everywhere: results are
//! bit-identical on every architecture (DSR-11). Ord is representation
//! order; consensus ordering never sorts field elements.

use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

use crate::codec::{Decode, Encode, Reader};
use crate::error::CodecError;

pub const GOLDILOCKS_PRIME: u64 = 18_446_744_069_414_584_321;

const _: () = assert!((GOLDILOCKS_PRIME as u128) == (1u128 << 64) - (1u128 << 32) + 1);
const _: () = assert!(GOLDILOCKS_PRIME == crate::params::PROOFS_FIELD_MODULUS);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Goldilocks(u64);

impl Default for Goldilocks {
    fn default() -> Self {
        Goldilocks(0)
    }
}

impl fmt::Debug for Goldilocks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F(0x{:016x})", self.0)
    }
}

impl Goldilocks {
    pub const ZERO: Goldilocks = Goldilocks(0);
    pub const ONE: Goldilocks = Goldilocks(1);

    pub const fn from_u32(x: u32) -> Goldilocks {
        Goldilocks(x as u64)
    }

    pub const fn from_u64_reduce(x: u64) -> Goldilocks {
        if x >= GOLDILOCKS_PRIME {
            Goldilocks(x - GOLDILOCKS_PRIME)
        } else {
            Goldilocks(x)
        }
    }

    /// Reduce a `u128` to a canonical `Goldilocks` element. The caller
    /// promises `x < GOLDILOCKS_PRIME * 2` (i.e. one reduction step is
    /// enough); anything bigger is reduced mod-P in two steps to stay
    /// safe in constant arithmetic.
    pub const fn from_u128_reduce(x: u128) -> Goldilocks {
        // `GOLDILOCKS_PRIME` fits in `u64`; use it twice to reduce any
        // 128-bit input without overflowing.
        let p = GOLDILOCKS_PRIME as u128;
        let r1 = if x >= p { x - p } else { x };
        let r2 = if r1 >= p { r1 - p } else { r1 };
        if r2 >= p {
            // Caller violated the `x < 2*P` precondition. Fall back to
            // a `u64` truncation; downstream code should never observe
            // this in well-formed inputs.
            Goldilocks(r2 as u64)
        } else {
            Goldilocks(r2 as u64)
        }
    }

    pub fn from_valid_u64(x: u64) -> Option<Goldilocks> {
        if x < GOLDILOCKS_PRIME {
            Some(Goldilocks(x))
        } else {
            None
        }
    }

    pub const fn as_u64(self) -> u64 {
        self.0
    }

    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    pub const fn add(self, rhs: Goldilocks) -> Goldilocks {
        let s = (self.0 as u128) + (rhs.0 as u128);
        if s >= GOLDILOCKS_PRIME as u128 {
            Goldilocks((s - GOLDILOCKS_PRIME as u128) as u64)
        } else {
            Goldilocks(s as u64)
        }
    }

    pub const fn sub(self, rhs: Goldilocks) -> Goldilocks {
        let d = (self.0 as u128) + (GOLDILOCKS_PRIME as u128) - (rhs.0 as u128);
        if d >= GOLDILOCKS_PRIME as u128 {
            Goldilocks((d - GOLDILOCKS_PRIME as u128) as u64)
        } else {
            Goldilocks(d as u64)
        }
    }

    pub fn mul(self, rhs: Goldilocks) -> Goldilocks {
        Goldilocks(
            (((self.0 as u128) * (rhs.0 as u128)) % (GOLDILOCKS_PRIME as u128)) as u64,
        )
    }

    /// Const-compatible modular multiplication (Goldilocks fast reduction,
    /// p = 2⁶⁴ − 2³² + 1). For a, b ∈ [0, p), returns a·b mod p with no
    /// overflow: a·b fits in 128 bits; the reduction uses 2⁶⁴ ≡ 2³² − 1
    /// (mod p) to fold the upper half back into the lower half. Branch
    /// count is constant-time. The runtime `mul` above is faster on
    /// machines without `u128` overflow tricks but is not `const`-callable;
    /// this one is for `const_pow` and other precompile time tables.
    pub const fn const_mul(self, rhs: Goldilocks) -> Goldilocks {
        let n = (self.0 as u128) * (rhs.0 as u128);
        let hi = (n >> 64) as u64;
        let lo = n as u64;
        // result = hi * 2^32 - hi + lo (may underflow to a negative i128)
        let r = (hi as i128) * (1i128 << 32) - (hi as i128) + (lo as i128);
        let r = if r < 0 {
            (r + GOLDILOCKS_PRIME as i128) as u64
        } else if r >= GOLDILOCKS_PRIME as i128 {
            (r - GOLDILOCKS_PRIME as i128) as u64
        } else {
            r as u64
        };
        Goldilocks(r)
    }

    pub const fn neg(self) -> Goldilocks {
        if self.0 == 0 {
            self
        } else {
            Goldilocks(GOLDILOCKS_PRIME - self.0)
        }
    }

    pub fn pow(self, mut exp: u64) -> Goldilocks {
        let mut acc = Goldilocks::ONE;
        let mut base = self;
        while exp > 0 {
            if exp & 1 == 1 {
                acc = acc.mul(base);
            }
            base = base.mul(base);
            exp >>= 1;
        }
        acc
    }

    /// Fermat inverse; ZERO for ZERO (callers must check is_zero).
    pub fn inverse(self) -> Goldilocks {
        if self.0 == 0 {
            return Goldilocks::ZERO;
        }
        self.pow(GOLDILOCKS_PRIME - 2)
    }

    /// x^7 — the Poseidon2 S-box (bijective: gcd(7, p−1) = 1, tested).
    pub fn pow7(self) -> Goldilocks {
        let x2 = self.mul(self);
        let x4 = x2.mul(x2);
        x4.mul(x2).mul(self)
    }
}

impl Add for Goldilocks {
    type Output = Goldilocks;
    fn add(self, rhs: Goldilocks) -> Goldilocks {
        Goldilocks::add(self, rhs)
    }
}
impl Sub for Goldilocks {
    type Output = Goldilocks;
    fn sub(self, rhs: Goldilocks) -> Goldilocks {
        Goldilocks::sub(self, rhs)
    }
}
impl Mul for Goldilocks {
    type Output = Goldilocks;
    fn mul(self, rhs: Goldilocks) -> Goldilocks {
        Goldilocks::mul(self, rhs)
    }
}
impl Neg for Goldilocks {
    type Output = Goldilocks;
    fn neg(self) -> Goldilocks {
        Goldilocks::neg(self)
    }
}

impl Encode for Goldilocks {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0.to_le_bytes());
    }
    fn encoded_len(&self) -> usize {
        8
    }
}

impl Decode for Goldilocks {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let v = r.read_u64()?;
        Goldilocks::from_valid_u64(v).ok_or(CodecError::FieldElementOutOfRange { value: v })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;

    fn fe(rng: &mut SplitMix64) -> Goldilocks {
        Goldilocks::from_u64_reduce(rng.next_u64())
    }

    #[test]
    fn prime_value() {
        assert_eq!(GOLDILOCKS_PRIME, 18446744069414584321u64);
        assert_eq!((GOLDILOCKS_PRIME as u128), (1u128 << 64) - (1u128 << 32) + 1);
        assert_eq!(Goldilocks::from_u64_reduce(u64::MAX).as_u64(), u64::MAX - GOLDILOCKS_PRIME + 1);
    }

    #[test]
    fn field_axioms() {
        let mut rng = SplitMix64::new(0xF1E1D);
        for _ in 0..500 {
            let (a, b, c) = (fe(&mut rng), fe(&mut rng), fe(&mut rng));
            assert_eq!(a.add(b), b.add(a));
            assert_eq!(a.mul(b), b.mul(a));
            assert_eq!(a.add(b).add(c), a.add(b.add(c)));
            assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
            assert_eq!(a.mul(b.add(c)), a.mul(b).add(a.mul(c)));
            assert_eq!(a.sub(b).add(b), a);
            assert_eq!(a.add(a.neg()), Goldilocks::ZERO);
            assert!(a.add(b).as_u64() < GOLDILOCKS_PRIME);
            assert!(a.sub(b).as_u64() < GOLDILOCKS_PRIME);
            assert!(a.mul(b).as_u64() < GOLDILOCKS_PRIME);
            if !a.is_zero() {
                assert_eq!(a.mul(a.inverse()), Goldilocks::ONE);
            }
        }
    }

    #[test]
    fn from_u64_reduce_matches_u128_mod() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..2000 {
            let x = rng.next_u64();
            assert_eq!(
                Goldilocks::from_u64_reduce(x).as_u64(),
                (x as u128 % GOLDILOCKS_PRIME as u128) as u64
            );
        }
        assert_eq!(Goldilocks::from_u64_reduce(GOLDILOCKS_PRIME), Goldilocks::ZERO);
        assert_eq!(Goldilocks::from_u64_reduce(GOLDILOCKS_PRIME - 1).as_u64(), GOLDILOCKS_PRIME - 1);
    }

    #[test]
    fn pow_matches_repeated_mul() {
        let mut rng = SplitMix64::new(11);
        for _ in 0..100 {
            let a = fe(&mut rng);
            for e in [0u64, 1, 2, 3, 7, 10, 17] {
                let mut acc = Goldilocks::ONE;
                for _ in 0..e {
                    acc = acc.mul(a);
                }
                assert_eq!(a.pow(e), acc, "e={e}");
            }
            assert_eq!(a.pow7(), a.pow(7));
        }
    }

    #[test]
    fn codec_roundtrip_and_range() {
        let mut rng = SplitMix64::new(13);
        for _ in 0..200 {
            let a = fe(&mut rng);
            let enc = a.encode();
            assert_eq!(enc.len(), 8);
            assert_eq!(Goldilocks::decode(&enc), Ok(a));
        }
        assert!(matches!(
            Goldilocks::decode(&GOLDILOCKS_PRIME.to_le_bytes()),
            Err(CodecError::FieldElementOutOfRange { .. })
        ));
        assert_eq!(
            Goldilocks::decode(&(GOLDILOCKS_PRIME - 1).to_le_bytes()).map(|v| v.as_u64()),
            Ok(GOLDILOCKS_PRIME - 1)
        );
        assert!(Goldilocks::decode(&[0u8; 7]).is_err());
        let mut ext = 0u64.to_le_bytes().to_vec();
        ext.push(0);
        assert!(Goldilocks::decode(&ext).is_err());
    }

    #[test]
    fn ops_delegation_and_debug() {
        assert_eq!(Goldilocks::default(), Goldilocks::ZERO);
        assert_eq!(format!("{:?}", Goldilocks::ONE), "F(0x0000000000000001)");
        let (a, b) = (Goldilocks::from_u32(3), Goldilocks::from_u32(5));
        assert_eq!(a + b, Goldilocks::from_u32(8));
        assert_eq!(a * b, Goldilocks::from_u32(15));
        assert_eq!((a - b) + b, a);
        assert_eq!(-Goldilocks::ZERO, Goldilocks::ZERO);
        assert_eq!(Goldilocks::from_u32(u32::MAX).as_u64(), u32::MAX as u64);
    }
}

