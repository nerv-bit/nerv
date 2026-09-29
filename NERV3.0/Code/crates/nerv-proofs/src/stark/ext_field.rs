//! The challenge field EF = GF(p)[X]/(X² − W): p the Goldilocks prime, W = 7
//! (WP §5.3's Goldilocks-class prover field; erratum 92's D=2 reading — a
//! 128-bit challenge field, |EF| = p² ≈ 2^128, exactly the terms
//! `security::Profile::wallet` models at modulus_bits = 128).
//!
//! W = 7 is the multiplicative generator of GF(p) (nerv-core field tests),
//! hence a quadratic non-residue — pinned by Euler's criterion below — so
//! X² − 7 is irreducible and the quotient is a field. All arithmetic is
//! componentwise nerv-core Goldilocks: exact u128 intermediates, identical
//! on every architecture (DSR-2, DSR-11). An element (c0, c1) denotes
//! c0 + c1·X with X² = 7.

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::error::CodecError;
use nerv_core::field::Goldilocks;

/// The defining non-residue: X² = 7.
pub const W: Goldilocks = Goldilocks::from_u32(7);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct ExtF {
    c0: Goldilocks,
    c1: Goldilocks,
}

impl ExtF {
    pub const ZERO: ExtF = ExtF { c0: Goldilocks::ZERO, c1: Goldilocks::ZERO };
    pub const ONE: ExtF = ExtF { c0: Goldilocks::ONE, c1: Goldilocks::ZERO };

    pub const fn new(c0: Goldilocks, c1: Goldilocks) -> ExtF {
        ExtF { c0, c1 }
    }

    pub const fn components(self) -> (Goldilocks, Goldilocks) {
        (self.c0, self.c1)
    }

    pub const fn from_base(b: Goldilocks) -> ExtF {
        ExtF { c0: b, c1: Goldilocks::ZERO }
    }

    /// The base element iff the X-component is zero.
    pub const fn to_base(self) -> Option<Goldilocks> {
        if self.c1.is_zero() {
            Some(self.c0)
        } else {
            None
        }
    }

    pub const fn add(self, rhs: ExtF) -> ExtF {
        ExtF { c0: self.c0.add(rhs.c0), c1: self.c1.add(rhs.c1) }
    }

    pub const fn sub(self, rhs: ExtF) -> ExtF {
        ExtF { c0: self.c0.sub(rhs.c0), c1: self.c1.sub(rhs.c1) }
    }

    pub const fn neg(self) -> ExtF {
        ExtF { c0: self.c0.neg(), c1: self.c1.neg() }
    }

    /// Karatsuba (3 base multiplications); pinned against the naive
    /// schoolbook in the tests.
    pub fn mul(self, rhs: ExtF) -> ExtF {
        let (a, b) = (self.c0, self.c1);
        let (c, d) = (rhs.c0, rhs.c1);
        let ac = a * c;
        let bd = b * d;
        let cross = (a + b) * (c + d) - ac - bd; // ad + bc
        ExtF { c0: ac + W * bd, c1: cross }
    }

    /// self · g with a base scalar (the DEEP/FRI linear-combination builder).
    pub fn scale(self, g: Goldilocks) -> ExtF {
        ExtF { c0: self.c0 * g, c1: self.c1 * g }
    }

    pub fn pow(self, mut e: u64) -> ExtF {
        let mut acc = ExtF::ONE;
        let mut base = self;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(base);
            }
            base = base.mul(base);
            e >>= 1;
        }
        acc
    }

    /// (c0, −c1). Norm = z·z̄ = c0² − W·c1² ∈ GF(p).
    pub const fn conjugate(self) -> ExtF {
        ExtF { c0: self.c0, c1: self.c1.neg() }
    }

    pub fn norm(self) -> Goldilocks {
        self.c0 * self.c0 - W * (self.c1 * self.c1)
    }

    /// Conjugate over norm; ZERO for ZERO (caller contract, matching
    /// `Goldilocks::inverse`). Nonzero elements always invert: the norm
    /// vanishes only at ZERO, else 7 would be a residue.
    pub fn inverse(self) -> ExtF {
        let n = self.norm();
        if n.is_zero() {
            return ExtF::ZERO;
        }
        let inv = n.inverse();
        ExtF { c0: self.c0 * inv, c1: self.c1 * inv }
    }

    pub const fn is_zero(self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }
}

impl Add for ExtF {
    type Output = ExtF;
    fn add(self, rhs: ExtF) -> ExtF {
        ExtF::add(self, rhs)
    }
}
impl Sub for ExtF {
    type Output = ExtF;
    fn sub(self, rhs: ExtF) -> ExtF {
        ExtF::sub(self, rhs)
    }
}
impl Mul for ExtF {
    type Output = ExtF;
    fn mul(self, rhs: ExtF) -> ExtF {
        ExtF::mul(self, rhs)
    }
}
impl Neg for ExtF {
    type Output = ExtF;
    fn neg(self) -> ExtF {
        ExtF::neg(self)
    }
}
impl AddAssign for ExtF {
    fn add_assign(&mut self, rhs: ExtF) {
        *self = ExtF::add(*self, rhs);
    }
}
impl SubAssign for ExtF {
    fn sub_assign(&mut self, rhs: ExtF) {
        *self = ExtF::sub(*self, rhs);
    }
}
impl MulAssign for ExtF {
    fn mul_assign(&mut self, rhs: ExtF) {
        *self = ExtF::mul(*self, rhs);
    }
}

// Canonical wire format: c0 ‖ c1, each u64 LE, both range-checked.
impl Encode for ExtF {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.c0.encode_into(out);
        self.c1.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        16
    }
}

impl Decode for ExtF {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let v0 = r.read_u64()?;
        let v1 = r.read_u64()?;
        let c0 = Goldilocks::from_valid_u64(v0)
            .ok_or(CodecError::FieldElementOutOfRange { value: v0 })?;
        let c1 = Goldilocks::from_valid_u64(v1)
            .ok_or(CodecError::FieldElementOutOfRange { value: v1 })?;
        Ok(ExtF::new(c0, c1))
    }
}

/// Montgomery batch inversion: one field inversion, n−1 multiplications.
/// All inputs must be nonzero — a zero input yields garbage (debug-asserted).
pub fn batch_inverse(vals: &mut [ExtF]) {
    let n = vals.len();
    if n == 0 {
        return;
    }
    let mut prefix = Vec::with_capacity(n);
    let mut acc = ExtF::ONE;
    for v in vals.iter() {
        debug_assert!(!v.is_zero(), "batch_inverse: zero input");
        acc = acc.mul(*v);
        prefix.push(acc);
    }
    let mut inv = prefix[n - 1].inverse();
    for i in (1..n).rev() {
        let orig = vals[i];
        vals[i] = inv.mul(prefix[i - 1]);
        inv = inv.mul(orig);
    }
    vals[0] = inv;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;
    use nerv_core::field::GOLDILOCKS_PRIME;

    fn fe(rng: &mut SplitMix64) -> Goldilocks {
        Goldilocks::from_u64_reduce(rng.next_u64())
    }

    fn ef(rng: &mut SplitMix64) -> ExtF {
        ExtF::new(fe(rng), fe(rng))
    }

    /// Independent reference: the naive schoolbook product.
    fn naive_mul(z: ExtF, w: ExtF) -> ExtF {
        let (a, b) = z.components();
        let (c, d) = w.components();
        ExtF::new(a * c + W * (b * d), a * d + b * c)
    }

    #[test]
    fn w_is_a_quadratic_nonresidue() {
        // Euler's criterion: 7^((p−1)/2) ≡ −1 ⇒ X² − 7 irreducible ⇒ EF is a field.
        let half = (GOLDILOCKS_PRIME - 1) / 2;
        assert_eq!(W.pow(half), Goldilocks::from_u64_reduce(GOLDILOCKS_PRIME - 1));
    }

    #[test]
    fn constants_and_embedding() {
        assert!(ExtF::ZERO.is_zero());
        assert!(!ExtF::ONE.is_zero());
        assert_eq!(ExtF::ONE.to_base(), Some(Goldilocks::ONE));
        assert_eq!(ExtF::from_base(Goldilocks::ONE), ExtF::ONE);
        assert_eq!(ExtF::from_base(Goldilocks::ZERO), ExtF::ZERO);
        assert_eq!(ExtF::new(Goldilocks::ONE, Goldilocks::ONE).to_base(), None);
        assert_eq!(ExtF::new(Goldilocks::ONE, Goldilocks::ZERO).to_base(), Some(Goldilocks::ONE));
    }

    #[test]
    fn field_axioms() {
        let mut rng = SplitMix64::new(0xE57);
        for _ in 0..500 {
            let (a, b, c) = (ef(&mut rng), ef(&mut rng), ef(&mut rng));
            assert_eq!(a + b, b + a);
            assert_eq!(a * b, b * a);
            assert_eq!((a + b) + c, a + (b + c));
            assert_eq!((a * b) * c, a * (b * c));
            assert_eq!(a * (b + c), a * b + a * c);
            assert_eq!((a - b) + b, a);
            assert_eq!(a + (-a), ExtF::ZERO);
            assert_eq!(a * ExtF::ONE, a);
            assert_eq!(a * ExtF::ZERO, ExtF::ZERO);
            if !a.is_zero() {
                assert_eq!(a * a.inverse(), ExtF::ONE);
            }
        }
    }

    #[test]
    fn karatsuba_matches_naive_schoolbook() {
        let mut rng = SplitMix64::new(0xCA);
        for _ in 0..1000 {
            let z = ef(&mut rng);
            let w = ef(&mut rng);
            assert_eq!(z * w, naive_mul(z, w));
            assert_eq!(z * z, naive_mul(z, z));
        }
    }

    #[test]
    fn conjugate_norm_inverse_identities() {
        let mut rng = SplitMix64::new(0x1D);
        for _ in 0..300 {
            let z = ef(&mut rng);
            assert_eq!(z * z.conjugate(), ExtF::from_base(z.norm()));
            if !z.is_zero() {
                let inv = z.inverse();
                assert_eq!(z * inv, ExtF::ONE);
                assert_eq!(inv * inv.inverse(), ExtF::ONE);
                assert_eq!(z.inverse().conjugate(), z.conjugate().inverse());
            }
        }
        assert_eq!(ExtF::ZERO.norm(), Goldilocks::ZERO);
        assert_eq!(ExtF::ZERO.inverse(), ExtF::ZERO);
    }

    #[test]
    fn frobenius_is_conjugate() {
        // z^p = z̄ for a degree-2 extension whose non-residue generates the
        // base multiplicative group — pins the Galois structure end to end.
        let mut rng = SplitMix64::new(0xFB);
        for _ in 0..50 {
            let z = ef(&mut rng);
            assert_eq!(z.pow(GOLDILOCKS_PRIME), z.conjugate());
        }
        assert_eq!(ExtF::ZERO.pow(GOLDILOCKS_PRIME), ExtF::ZERO);
        assert_eq!(ExtF::ONE.pow(GOLDILOCKS_PRIME), ExtF::ONE);
    }

    #[test]
    fn pow_matches_repeated_mul() {
        let mut rng = SplitMix64::new(0x90);
        for _ in 0..50 {
            let z = ef(&mut rng);
            let mut acc = ExtF::ONE;
            for e in 0u64..9 {
                assert_eq!(z.pow(e), acc, "e={e}");
                acc = acc * z;
            }
        }
    }

    #[test]
    fn scale_is_multiplication_by_the_lifted_base() {
        let mut rng = SplitMix64::new(0x5C);
        for _ in 0..200 {
            let z = ef(&mut rng);
            let g = fe(&mut rng);
            assert_eq!(z.scale(g), z * ExtF::from_base(g));
        }
    }

    #[test]
    fn base_embedding_is_a_ring_homomorphism() {
        let mut rng = SplitMix64::new(0xB4);
        for _ in 0..300 {
            let (a, b) = (fe(&mut rng), fe(&mut rng));
            assert_eq!(ExtF::from_base(a) + ExtF::from_base(b), ExtF::from_base(a + b));
            assert_eq!(ExtF::from_base(a) * ExtF::from_base(b), ExtF::from_base(a * b));
            assert_eq!(ExtF::from_base(a).to_base(), Some(a));
            assert_eq!((-ExtF::from_base(a)).to_base(), Some(-a));
        }
    }

    #[test]
    fn batch_inverse_matches_per_element() {
        let mut rng = SplitMix64::new(0xB7);
        for n in 0usize..17 {
            let mut vals: Vec<ExtF> = (0..n).map(|_| ef(&mut rng)).collect();
            let expect: Vec<ExtF> = vals.iter().map(|z| z.inverse()).collect();
            batch_inverse(&mut vals);
            assert_eq!(vals, expect, "n={n}");
        }
        // repeated elements still invert per-element
        let z = ef(&mut SplitMix64::new(99));
        let mut vals = vec![z; 5];
        batch_inverse(&mut vals);
        assert!(vals.iter().all(|v| *v == z.inverse()));
    }

    #[test]
    fn extremes() {
        let pm1 = Goldilocks::from_u64_reduce(GOLDILOCKS_PRIME - 1);
        let z_max = ExtF::new(pm1, pm1);
        assert_eq!(z_max + ExtF::ONE, ExtF::ZERO);
        assert_eq!(ExtF::ONE - ExtF::ONE, ExtF::ZERO);
        assert_eq!(ExtF::ZERO - ExtF::ONE, ExtF::from_base(pm1));
        // −1 is a base element: multiplying by it negates.
        let mut rng = SplitMix64::new(0xE9);
        for _ in 0..50 {
            let z = ef(&mut rng);
            assert_eq!(z * ExtF::from_base(pm1), -z);
        }
        let mut acc = ExtF::ONE;
        acc += ExtF::ONE;
        acc -= ExtF::ONE;
        acc *= ExtF::ONE;
        assert_eq!(acc, ExtF::ONE);
    }

    #[test]
    fn codec_roundtrip_and_strictness() {
        let mut rng = SplitMix64::new(0xC0);
        for _ in 0..200 {
            let z = ef(&mut rng);
            let enc = z.encode();
            assert_eq!(enc.len(), 16);
            assert_eq!(ExtF::decode(&enc), Ok(z));
            // wire layout: c0 ‖ c1, each u64 LE
            let (c0, c1) = z.components();
            let mut want = Vec::new();
            want.extend_from_slice(&c0.as_u64().to_le_bytes());
            want.extend_from_slice(&c1.as_u64().to_le_bytes());
            assert_eq!(enc, want);
            for cut in 0..16 {
                assert!(ExtF::decode(&enc[..cut]).is_err(), "cut={cut}");
            }
            let mut ext = enc.clone();
            ext.push(0);
            assert!(ExtF::decode(&ext).is_err());
        }
        // range rejection: either component ≥ p is non-canonical
        let z = ExtF::new(Goldilocks::ONE, Goldilocks::ONE);
        let mut bad = z.encode();
        bad[..8].copy_from_slice(&GOLDILOCKS_PRIME.to_le_bytes());
        assert!(matches!(
            ExtF::decode(&bad),
            Err(CodecError::FieldElementOutOfRange { .. })
        ));
        let mut bad2 = z.encode();
        bad2[8..].copy_from_slice(&GOLDILOCKS_PRIME.to_le_bytes());
        assert!(matches!(
            ExtF::decode(&bad2),
            Err(CodecError::FieldElementOutOfRange { .. })
        ));
    }
}

