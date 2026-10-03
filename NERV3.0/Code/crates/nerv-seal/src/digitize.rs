//! δ ↔ 512 × 8-bit digits and public carry resolution (WP §6.3.1–§6.3.2;
//! erratum 40).
//!
//! Each of the 64 coordinates (u64 mod-2^64 group elements) decomposes
//! into 8 canonical base-2^8 digits — 8·8 = 64 bits exactly, so position 7
//! is a full digit (bits 56–63) and the guard plane of the 9-bit scheme
//! is deleted: carry-out at position 7 *is* the mod-2^64 wrap. All 512
//! slots carry digits. The 8-bit width is the DKG-closure choice (erratum
//! 40: committee-summed keys at B_min = 128 admit scale 2^15 and a 9.49σ
//! honest decode margin only at 8-bit digits).
//!
//! Slot layout (frozen): digit k of coordinate j sits at slot 64·k + j;
//! planes 0–3 (slots 0–255) fill ring 0; planes 4–7 (slots 256–511) fill
//! ring 1.
//!
//! `resolve` is total: any per-slot digit sums evaluate in u128 to
//! Σ_k D_k·2^{8k} mod 2^64 — the exact identity behind "digit addition is
//! carry-free per slot, then carries resolve publicly" (the top-position
//! wrap is the embedding accumulator's group law, WP §7.5). Out-of-
//! envelope sums are decode-time detections owned by decrypt.
//!
//! Interop: nerv-codec's `Delta` is `[u64; 64]` — `digitize(&delta.0)` /
//! `Delta(coords)`; nerv-seal does not depend on nerv-codec (policy);
//! the 64-dimension seam is pinned by the conformance crate.

use crate::error::SealError;
use crate::ring::{Poly, Vec2, N};

pub const DIGIT_BITS: u32 = 8;
pub const DIGITS_PER_COORD: usize = 8;
pub const COORDS: usize = 64;
pub const SLOTS: usize = 512;
pub const SLOTS_USED: usize = 512;
pub const DIGIT_MAX: u64 = 255;
/// Canonical digit-vector serialization: 1,024 bytes (u16-LE, slot order).
pub const WIRE_SIZE: usize = SLOTS_USED * 2;

const _: () = assert!(SLOTS_USED == COORDS * DIGITS_PER_COORD);
const _: () = assert!(SLOTS == 2 * N);
const _: () = assert!(DIGIT_MAX == (1u64 << DIGIT_BITS) - 1);
const _: () = assert!(DIGITS_PER_COORD * DIGIT_BITS as usize == 64);

/// Slot of coordinate j's digit k: `64·k + j`.
pub const fn slot(j: usize, k: usize) -> usize {
    COORDS * k + j
}

/// The plaintext polynomial pair m ∈ R_q² carrying a digitized delta —
/// canonical by construction (every coefficient ≤ 255): the only
/// constructors are `digitize`, `from_digits`, `from_pair` (validated),
/// and `zero`.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Plaintext(Vec2);

impl nerv_core::codec::Encode for Plaintext {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.0.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        self.0.encoded_len()
    }
}

impl nerv_core::codec::Decode for Plaintext {
    fn decode_from(r: &mut nerv_core::codec::Reader<'_>) -> Result<Self, nerv_core::error::CodecError> {
        Ok(Plaintext(Vec2::decode_from(r)?))
    }
}

impl Plaintext {
    pub fn zero() -> Plaintext {
        Plaintext(Vec2::new([Poly::zero(), Poly::zero()]))
    }

    pub fn pair(&self) -> &Vec2 {
        &self.0
    }

    fn slot_value(&self, s: usize) -> u64 {
        if s < N {
            self.0.poly(0).coefficient(s)
        } else {
            self.0.poly(1).coefficient(s - N)
        }
    }

    pub fn digit(&self, j: usize, k: usize) -> u64 {
        self.slot_value(slot(j, k))
    }

    pub fn slot_values(&self) -> [u64; SLOTS] {
        let mut out = [0u64; SLOTS];
        for (s, o) in out.iter_mut().enumerate() {
            *o = self.slot_value(s);
        }
        out
    }

    /// Digits in slot order (index s = 64·k + j).
    pub fn digits(&self) -> [u16; SLOTS_USED] {
        let mut out = [0u16; SLOTS_USED];
        for (s, o) in out.iter_mut().enumerate() {
            let v = self.slot_value(s);
            debug_assert!(v <= DIGIT_MAX);
            *o = v as u16;
        }
        out
    }

    /// The delta this plaintext encodes (resolve of its own digits).
    pub fn coords(&self) -> [u64; COORDS] {
        resolve(&self.slot_values())
    }

    pub fn from_pair(pair: &Vec2) -> Result<Self, SealError> {
        for r in 0..2 {
            for (c, &v) in pair.poly(r).coefficients().iter().enumerate() {
                if v > DIGIT_MAX {
                    return Err(SealError::DigitOutOfRange { slot: r * N + c, value: v });
                }
            }
        }
        Ok(Plaintext(Vec2::new([*pair.poly(0), *pair.poly(1)])))
    }

    pub fn from_digits(digits: &[u16; SLOTS_USED]) -> Result<Self, SealError> {
        for (s, &d) in digits.iter().enumerate() {
            if u64::from(d) > DIGIT_MAX {
                return Err(SealError::DigitOutOfRange { slot: s, value: u64::from(d) });
            }
        }
        let mut p0 = [0u64; N];
        let mut p1 = [0u64; N];
        for (s, &d) in digits.iter().enumerate() {
            let v = u64::from(d);
            if s < N {
                p0[s] = v;
            } else {
                p1[s - N] = v;
            }
        }
        Ok(Plaintext(Vec2::new([Poly::new(p0), Poly::new(p1)])))
    }

    pub fn to_bytes(&self) -> [u8; WIRE_SIZE] {
        let digits = self.digits();
        let mut out = [0u8; WIRE_SIZE];
        for (s, d) in digits.iter().enumerate() {
            out[2 * s..2 * s + 2].copy_from_slice(&d.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SealError> {
        if bytes.len() != WIRE_SIZE {
            return Err(SealError::BadLength { len: bytes.len(), expected: WIRE_SIZE });
        }
        let mut digits = [0u16; SLOTS_USED];
        for (s, d) in digits.iter_mut().enumerate() {
            let mut b = [0u8; 2];
            b.copy_from_slice(&bytes[2 * s..2 * s + 2]);
            *d = u16::from_le_bytes(b);
        }
        Self::from_digits(&digits)
    }
}

/// δ → the plaintext pair: digit k of coordinate j at slot 64·k + j.
pub fn digitize(coords: &[u64; COORDS]) -> Plaintext {
    let mut p0 = [0u64; N];
    let mut p1 = [0u64; N];
    for j in 0..COORDS {
        for k in 0..DIGITS_PER_COORD {
            let d = (coords[j] >> (DIGIT_BITS * k as u32)) & DIGIT_MAX;
            let s = COORDS * k + j;
            if s < N {
                p0[s] = d;
            } else {
                p1[s - N] = d;
            }
        }
    }
    Plaintext(Vec2::new([Poly::new(p0), Poly::new(p1)]))
}

/// Public carry resolution (WP §6.3.2): per-slot digit sums → coordinate
/// sums mod 2^64. Total — exact u128 evaluation of Σ_k D_k·2^{8k}, low 64
/// bits kept.
pub fn resolve(sums: &[u64; SLOTS]) -> [u64; COORDS] {
    let mut out = [0u64; COORDS];
    for (j, o) in out.iter_mut().enumerate() {
        let mut acc: u128 = 0;
        for k in 0..DIGITS_PER_COORD {
            acc += u128::from(sums[COORDS * k + j]) << (DIGIT_BITS * k as u32);
        }
        *o = acc as u64;
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    // Independent formulation: explicit carry propagation in base 256,
    // the top position wrapping mod 2^8 into bits 56–63.
    fn resolve_reference_carry(sums: &[u64; SLOTS]) -> [u64; COORDS] {
        let base = u128::from(DIGIT_MAX + 1);
        let mut out = [0u64; COORDS];
        for (j, o) in out.iter_mut().enumerate() {
            let mut carry: u128 = 0;
            let mut acc: u64 = 0;
            for k in 0..DIGITS_PER_COORD {
                let t = u128::from(sums[COORDS * k + j]) + carry;
                acc += ((t % base) as u64) << (DIGIT_BITS * k as u32);
                carry = t / base;
            }
            *o = acc;
        }
        out
    }

    fn rand_coords(s: &mut u64) -> [u64; COORDS] {
        let mut c = [0u64; COORDS];
        for v in c.iter_mut() {
            *v = splitmix64(s);
        }
        c
    }

    #[test]
    fn slot_layout_pins() {
        assert_eq!(slot(0, 0), 0);
        assert_eq!(slot(63, 0), 63);
        assert_eq!(slot(0, 1), 64);
        assert_eq!(slot(63, 3), 255);
        assert_eq!(slot(0, 4), 256);
        assert_eq!(slot(63, 6), 447);
        assert_eq!(slot(0, 7), 448);
        assert_eq!(slot(63, 7), 511);
        assert_eq!(WIRE_SIZE, 1024);
        assert_eq!(SLOTS_USED, SLOTS);
    }

    #[test]
    fn digitize_is_the_base_256_decomposition() {
        let mut coords = [0u64; COORDS];
        coords[0] = u64::MAX;
        coords[1] = 255;
        coords[2] = 256;
        coords[63] = 0x0123_4567_89AB_CDEF;
        let pt = digitize(&coords);
        for j in 0..COORDS {
            for k in 0..DIGITS_PER_COORD {
                assert_eq!(pt.digit(j, k), (coords[j] >> (8 * k)) & 255);
            }
        }
        assert!(pt.slot_values().iter().all(|&v| v <= DIGIT_MAX));
        assert_eq!(pt.digit(0, 7), 255);
        assert_eq!(pt.digit(1, 0), 255);
        assert_eq!(pt.digit(1, 1), 0);
        assert_eq!(pt.digit(2, 0), 0);
        assert_eq!(pt.digit(2, 1), 1);
    }

    #[test]
    fn digitize_resolve_roundtrip_random_and_edges() {
        let mut s = 0xD1CE_0000u64;
        for _ in 0..200 {
            let coords = rand_coords(&mut s);
            let pt = digitize(&coords);
            assert_eq!(resolve(&pt.slot_values()), coords);
            assert_eq!(pt.coords(), coords);
            assert_eq!(Plaintext::from_pair(pt.pair()).unwrap(), pt);
        }
        assert_eq!(resolve(&digitize(&[0; COORDS]).slot_values()), [0; COORDS]);
        let ones = [u64::MAX; COORDS];
        assert_eq!(resolve(&digitize(&ones).slot_values()), ones);
    }

    #[test]
    fn zero_plaintext() {
        let z = Plaintext::zero();
        assert_eq!(z.to_bytes(), [0u8; WIRE_SIZE]);
        assert_eq!(z.coords(), [0u64; COORDS]);
        assert_eq!(z, Plaintext::default());
    }

    fn pair_with(slot: usize, value: u64) -> Vec2 {
        let mut p0 = [0u64; N];
        let mut p1 = [0u64; N];
        if slot < N {
            p0[slot] = value;
        } else {
            p1[slot - N] = value;
        }
        Vec2::new([Poly::new(p0), Poly::new(p1)])
    }

    #[test]
    fn from_pair_rejects_noncanonical_and_accepts_boundaries() {
        assert!(matches!(
            Plaintext::from_pair(&pair_with(0, 256)),
            Err(SealError::DigitOutOfRange { slot: 0, value: 256 })
        ));
        assert!(matches!(
            Plaintext::from_pair(&pair_with(511, 300)),
            Err(SealError::DigitOutOfRange { slot: 511, .. })
        ));
        assert!(Plaintext::from_pair(&pair_with(0, 255)).is_ok());
        assert!(Plaintext::from_pair(&pair_with(447, 255)).is_ok());
        assert!(Plaintext::from_pair(&pair_with(448, 255)).is_ok());
        assert!(Plaintext::from_pair(&pair_with(511, 255)).is_ok());
    }

    #[test]
    fn wire_and_digits_roundtrip_with_validation() {
        let mut s = 0xD1CE_0001u64;
        let coords = rand_coords(&mut s);
        let pt = digitize(&coords);
        let bytes = pt.to_bytes();
        assert_eq!(Plaintext::from_bytes(&bytes).unwrap(), pt);
        assert!(matches!(
            Plaintext::from_bytes(&bytes[..WIRE_SIZE - 1]),
            Err(SealError::BadLength { expected: 1024, .. })
        ));
        let digits = pt.digits();
        assert_eq!(Plaintext::from_digits(&digits).unwrap(), pt);
        let mut bad = digits;
        bad[0] = 256;
        assert!(matches!(
            Plaintext::from_digits(&bad),
            Err(SealError::DigitOutOfRange { slot: 0, .. })
        ));
    }

    #[test]
    fn aggregate_digit_sums_resolve_to_wrapping_sums() {
        // 128 legs: per-slot sums ≤ 128·255 = 32,640 — inside the decode
        // headroom 49,152 (noise.rs's budget, erratum 40).
        let mut s = 0xD1CE_0002u64;
        let mut sums = [0u64; SLOTS];
        let mut naive = [0u64; COORDS];
        for _ in 0..128 {
            let coords = rand_coords(&mut s);
            for (e, d) in sums.iter_mut().zip(digitize(&coords).slot_values().iter()) {
                *e += d;
            }
            for (n, &c) in naive.iter_mut().zip(coords.iter()) {
                *n = n.wrapping_add(c);
            }
        }
        assert_eq!(resolve(&sums), naive);
        for &d in sums.iter() {
            assert!(d <= 128 * 255);
        }
    }

    proptest! {
        #[test]
        fn prop_resolve_matches_carry_reference(
            realistic in prop::collection::vec(0u64..200_000, SLOTS..=SLOTS),
            adversarial in prop::collection::vec(any::<u64>(), SLOTS..=SLOTS),
        ) {
            let mut a = [0u64; SLOTS];
            let mut b = [0u64; SLOTS];
            a.copy_from_slice(&realistic);
            b.copy_from_slice(&adversarial);
            prop_assert_eq!(resolve(&a), resolve_reference_carry(&a));
            prop_assert_eq!(resolve(&b), resolve_reference_carry(&b));
        }

        #[test]
        fn prop_digitize_resolve_roundtrip(seed in any::<u64>()) {
            let mut s = seed ^ 0xD1CE;
            let coords = rand_coords(&mut s);
            let pt = digitize(&coords);
            prop_assert_eq!(resolve(&pt.slot_values()), coords);
            prop_assert_eq!(pt.coords(), coords);
        }
    }
}

