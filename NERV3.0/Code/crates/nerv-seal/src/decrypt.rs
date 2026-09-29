//! Decryption and the reveal surface (WP §6.3.1–§6.3.3; errata 34–35, 40).
//!
//! m̂ = v_B − s·u_B, decoded digit-by-digit; public carry resolution
//! (digitize::resolve) reconstructs coordinate sums mod 2^64 exactly
//! (§6.3.2). Frozen decode rule: for the centered representative
//! cl ∈ [−(Q−1)/2, (Q−1)/2], d = (cl + 2^14) div 2^15 — round-half-up of
//! cl/scale. The same su serves both components (erratum 20).
//!
//! The su seam: `decode_chunk` consumes the committee's combined partial
//! value — s·u_B plus smudging bounded per coefficient by
//! `noise::SMUDGING_BUDGET`, which the VPD proofs enforce. `decrypt_native`
//! is the single-key DSR-7 twin.
//!
//! The invalid-reveal surface (§6.3.4): the compile-time budget keeps even
//! worst-case adversarial noise inside the centered range, so decode is
//! unambiguous; the canonical envelope then classifies every reveal —
//! every slot ≤ legs·255 (erratum 40: no guard plane; legs ∈ [1,
//! CHUNK_MAX]). Out of envelope ⇒ RevealError: deterministic, public,
//! first offending slot (slot-ascending scan, frozen). Honest
//! false-rejection ≈ 10⁻²¹ per slot. Custody never consults a reveal.

use crate::digitize::{resolve, COORDS, DIGIT_MAX, SLOTS};
use crate::error::RevealError;
use crate::noise::{HALF_SCALE, SCALE};
use crate::ring::{Poly, Vec2, Vec8, N};

/// Maximum legs per chunk reveal (params' chunk_max; genesis 128). Larger
/// batches split into CHUNK_MAX-sized chunks and sum publicly (§6.3.2);
/// sub-minimum batches are committee-padded to exactly CHUNK_MAX legs
/// (§6.3.5). The privacy floor B_min is a ceremony policy, not a decode
/// rule — decode is a pure function of (legs, su, v) (erratum 35).
pub const CHUNK_MAX: u64 = nerv_core::params::SEAL_CHUNK_MAX as u64;

/// A verified chunk reveal: per-slot digit sums inside the canonical
/// envelope. Constructed only by `decode_chunk` / `decrypt_native`, and by
/// `from_bytes` (re-validated) — the type's invariant.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ChunkReveal {
    legs: u64,
    sums: [u64; SLOTS],
}

impl ChunkReveal {
    /// legs (u32 LE) ‖ 512 digit sums (u32 LE) — 2,052 bytes (erratum 36).
    pub const WIRE_SIZE: usize = 4 + SLOTS * 4;

    pub fn legs(&self) -> u64 {
        self.legs
    }

    pub fn sums(&self) -> &[u64; SLOTS] {
        &self.sums
    }

    /// The chunk's coordinate aggregate: public carry resolution.
    pub fn coords(&self) -> [u64; COORDS] {
        resolve(&self.sums)
    }

    pub fn to_bytes(&self) -> [u8; ChunkReveal::WIRE_SIZE] {
        let mut out = [0u8; ChunkReveal::WIRE_SIZE];
        out[..4].copy_from_slice(&(self.legs as u32).to_le_bytes());
        for (s, &d) in self.sums.iter().enumerate() {
            out[4 + 4 * s..8 + 4 * s].copy_from_slice(&(d as u32).to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<ChunkReveal, RevealError> {
        if bytes.len() != ChunkReveal::WIRE_SIZE {
            return Err(RevealError::BadLength { len: bytes.len(), expected: ChunkReveal::WIRE_SIZE });
        }
        let mut lb = [0u8; 4];
        lb.copy_from_slice(&bytes[..4]);
        let legs = u64::from(u32::from_le_bytes(lb));
        check_legs(legs)?;
        let mut sums = [0u64; SLOTS];
        for (s, d) in sums.iter_mut().enumerate() {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[4 + 4 * s..8 + 4 * s]);
            *d = u64::from(u32::from_le_bytes(b));
            check_slot(legs, s, *d)?;
        }
        Ok(ChunkReveal { legs, sums })
    }
}

fn check_legs(legs: u64) -> Result<(), RevealError> {
    if legs == 0 {
        Err(RevealError::EmptyChunk)
    } else if legs > CHUNK_MAX {
        Err(RevealError::ChunkTooLarge { legs, max: CHUNK_MAX })
    } else {
        Ok(())
    }
}

fn check_slot(legs: u64, slot: usize, du: u64) -> Result<(), RevealError> {
    let envelope = legs * DIGIT_MAX;
    if du > envelope {
        return Err(RevealError::DigitSumAboveEnvelope { slot, value: du, envelope });
    }
    Ok(())
}

fn decode_hat(
    legs: u64,
    hat: &Poly,
    base_slot: usize,
    sums: &mut [u64; SLOTS],
) -> Result<(), RevealError> {
    let cl = hat.centerlift();
    for (c, &v) in cl.iter().enumerate() {
        let slot = base_slot + c;
        let d = (v + HALF_SCALE).div_euclid(SCALE as i64);
        if d < 0 {
            return Err(RevealError::NegativeDigit { slot, value: d });
        }
        check_slot(legs, slot, d as u64)?;
        sums[slot] = d as u64;
    }
    Ok(())
}

/// Decodes an aggregate ciphertext under the committee's combined value
/// su ≈ s·u_B (VPD-verified; |su − s·u_B|∞ ≤ noise::SMUDGING_BUDGET).
/// m̂ = v − su componentwise; the scan is slot-ascending and reports the
/// first offending slot.
pub fn decode_chunk(legs: u64, su: &Poly, v_b: &Vec2) -> Result<ChunkReveal, RevealError> {
    check_legs(legs)?;
    let mut sums = [0u64; SLOTS];
    let hat0 = v_b.poly(0).sub(su);
    decode_hat(legs, &hat0, 0, &mut sums)?;
    let hat1 = v_b.poly(1).sub(su);
    decode_hat(legs, &hat1, N, &mut sums)?;
    Ok(ChunkReveal { legs, sums })
}

/// Single-key native decryption — the DSR-7 twin of the committee path.
pub fn decrypt_native(
    legs: u64,
    s: &Vec8,
    u_b: &Vec8,
    v_b: &Vec2,
) -> Result<ChunkReveal, RevealError> {
    decode_chunk(legs, &s.dot(u_b), v_b)
}

/// Δ_B = Σ_chunks resolve(chunk) mod 2^64 — the public sum of revealed
/// chunks (§6.3.2). Never sum digit sums across chunks: cross-chunk sums
/// overflow the decode headroom (erratum 38).
pub fn assemble_block_aggregate(reveals: &[ChunkReveal]) -> [u64; COORDS] {
    let mut out = [0u64; COORDS];
    for r in reveals {
        for (o, &c) in out.iter_mut().zip(r.coords().iter()) {
            *o = o.wrapping_add(c);
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::digitize::{digitize, Plaintext};
    use crate::encrypt::{derive_reference_keypair, Ciphertext};
    use crate::ring::Q;
    use crate::sampling::NoiseSeed;
    use proptest::prelude::*;

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn noise_seed(state: &mut u64) -> NoiseSeed {
        let mut b = [0u8; 32];
        for ch in b.chunks_mut(8) {
            ch.copy_from_slice(&splitmix64(state).to_le_bytes());
        }
        NoiseSeed::from_bytes(b)
    }

    fn coords(state: &mut u64) -> [u64; COORDS] {
        let mut c = [0u64; COORDS];
        for v in c.iter_mut() {
            *v = splitmix64(state);
        }
        c
    }

    fn raw_coeff(x: i64) -> u64 {
        let m = x.unsigned_abs() as u64;
        if x >= 0 { m } else { Q - m }
    }

    fn one_slot_hat(slot: usize, hat: i64) -> Vec2 {
        let mut p0 = [0u64; N];
        let mut p1 = [0u64; N];
        let c = raw_coeff(hat);
        if slot < N {
            p0[slot] = c;
        } else {
            p1[slot - N] = c;
        }
        Vec2::new([Poly::new(p0), Poly::new(p1)])
    }

    fn synthetic_v(sums: &[u64; SLOTS]) -> Vec2 {
        let mut p0 = [0u64; N];
        let mut p1 = [0u64; N];
        for (s, &d) in sums.iter().enumerate() {
            let v = d * SCALE % Q;
            if s < N {
                p0[s] = v;
            } else {
                p1[s - N] = v;
            }
        }
        Vec2::new([Poly::new(p0), Poly::new(p1)])
    }

    #[test]
    fn decode_rule_is_round_half_up_over_scale() {
        let pin = |slot, hat, want| {
            match decode_chunk(128, &Poly::zero(), &one_slot_hat(slot, hat)) {
                Ok(r) => assert_eq!(r.sums()[slot], want, "hat {hat}"),
                Err(e) => panic!("hat {hat}: {e:?}"),
            }
        };
        pin(0, 0, 0);
        pin(0, SCALE as i64, 1);
        pin(0, 255 * SCALE as i64, 255);
        pin(448, 255 * SCALE as i64, 255);
        pin(511, SCALE as i64, 1);
        pin(0, HALF_SCALE, 1);
        pin(0, HALF_SCALE - 1, 0);
        pin(0, SCALE as i64 - 1, 1);
        pin(0, -HALF_SCALE, 0);
        match decode_chunk(128, &Poly::zero(), &one_slot_hat(0, -HALF_SCALE - 1)) {
            Err(RevealError::NegativeDigit { slot: 0, value: -1 }) => {}
            other => panic!("{other:?}"),
        }
        pin(300, 7 * SCALE as i64, 7);
    }

    #[test]
    fn envelope_accepts_exact_boundaries() {
        let legs = 128u64;
        let mut sums = [0u64; SLOTS];
        for d in sums.iter_mut() {
            *d = legs * DIGIT_MAX;
        }
        let reveal = decode_chunk(legs, &Poly::zero(), &synthetic_v(&sums)).unwrap();
        assert_eq!(*reveal.sums(), sums);
        assert_eq!(reveal.legs(), legs);
    }

    #[test]
    fn envelope_rejects_each_violation() {
        let legs = 1u64;
        let v = one_slot_hat(0, 256 * SCALE as i64);
        match decode_chunk(legs, &Poly::zero(), &v) {
            Err(RevealError::DigitSumAboveEnvelope { slot: 0, value: 256, envelope: 255 }) => {}
            other => panic!("{other:?}"),
        }
        let v = one_slot_hat(511, 300 * SCALE as i64);
        match decode_chunk(legs, &Poly::zero(), &v) {
            Err(RevealError::DigitSumAboveEnvelope { slot: 511, value: 300, envelope: 255 }) => {}
            other => panic!("{other:?}"),
        }
        let v = one_slot_hat(0, -SCALE as i64);
        match decode_chunk(legs, &Poly::zero(), &v) {
            Err(RevealError::NegativeDigit { slot: 0, value: -1 }) => {}
            other => panic!("{other:?}"),
        }
        match decode_chunk(0, &Poly::zero(), &Vec2::zero()) {
            Err(RevealError::EmptyChunk) => {}
            other => panic!("{other:?}"),
        }
        match decode_chunk(CHUNK_MAX + 1, &Poly::zero(), &Vec2::zero()) {
            Err(RevealError::ChunkTooLarge { legs, max }) if legs == CHUNK_MAX + 1 && max == CHUNK_MAX => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn native_and_explicit_su_paths_agree() {
        let (pk, s) = derive_reference_keypair(&[0x71; 32]).unwrap();
        let a_ntt = pk.expand_a().unwrap().ntt();
        let t_ntt = pk.t().ntt();
        let mut st = 0xD2C0_0001u64;
        let c = coords(&mut st);
        let m = digitize(&c);
        let ct = Ciphertext::encrypt_cached(&a_ntt, &t_ntt, &noise_seed(&mut st), &m).unwrap();
        let via_native = decrypt_native(1, &s, ct.u(), ct.v()).unwrap();
        let via_su = decode_chunk(1, &s.dot(ct.u()), ct.v()).unwrap();
        assert_eq!(via_native, via_su);
        assert_eq!(*via_native.sums(), m.slot_values());
        assert_eq!(via_native.coords(), c);
    }

    #[test]
    fn chunk_reveal_wire_roundtrip_and_validation() {
        let legs = 128u64;
        let mut sums = [0u64; SLOTS];
        for d in sums.iter_mut() {
            *d = 255;
        }
        let reveal = decode_chunk(legs, &Poly::zero(), &synthetic_v(&sums)).unwrap();
        let bytes = reveal.to_bytes();
        assert_eq!(ChunkReveal::WIRE_SIZE, 2052);
        assert_eq!(ChunkReveal::from_bytes(&bytes).unwrap(), reveal);

        assert!(matches!(
            ChunkReveal::from_bytes(&bytes[..2051]),
            Err(RevealError::BadLength { expected: 2052, .. })
        ));
        let mut bad = bytes;
        bad[..4].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(ChunkReveal::from_bytes(&bad), Err(RevealError::EmptyChunk)));
        bad[..4].copy_from_slice(&129u32.to_le_bytes());
        assert!(matches!(ChunkReveal::from_bytes(&bad), Err(RevealError::ChunkTooLarge { .. })));
        let mut bad = bytes;
        bad[4..8].copy_from_slice(&40_000u32.to_le_bytes());
        assert!(matches!(
            ChunkReveal::from_bytes(&bad),
            Err(RevealError::DigitSumAboveEnvelope { slot: 0, value: 40_000, envelope: 32_640 })
        ));
    }

    #[test]
    fn assemble_sums_coords_mod_2_64() {
        let mk = |seed: u64| -> ChunkReveal {
            let mut st = seed;
            let mut sums = [0u64; SLOTS];
            for d in sums.iter_mut() {
                *d = splitmix64(&mut st) % 1000;
            }
            decode_chunk(128, &Poly::zero(), &synthetic_v(&sums)).unwrap()
        };
        let a = mk(1);
        let b = mk(2);
        let mut want = [0u64; COORDS];
        for (w, (&x, &y)) in want.iter_mut().zip(a.coords().iter().zip(b.coords().iter())) {
            *w = x.wrapping_add(y);
        }
        assert_eq!(assemble_block_aggregate(&[a.clone(), b.clone()]), want);
        assert_eq!(assemble_block_aggregate(&[]), [0u64; COORDS]);
        assert_eq!(assemble_block_aggregate(&[a.clone()]), a.coords());
    }

    proptest! {
        #[test]
        fn prop_synthetic_in_envelope_decodes_exactly(
            pos in prop::collection::vec(0u64..=2_000, SLOTS..=SLOTS),
        ) {
            let mut sums = [0u64; SLOTS];
            sums.copy_from_slice(&pos);
            // legs = 8: envelope 2,040.
            for &d in sums.iter() {
                prop_assume!(d <= 8 * DIGIT_MAX);
            }
            let v = synthetic_v(&sums);
            let reveal = decode_chunk(8, &Poly::zero(), &v).unwrap();
            prop_assert_eq!(*reveal.sums(), sums);
            prop_assert_eq!(reveal.legs(), 8);
            prop_assert_eq!(reveal.coords(), resolve(&sums));
        }

        #[test]
        fn prop_honest_small_chunks_roundtrip(legs in 1u64..=8, seed in any::<u64>()) {
            let (pk, s) = derive_reference_keypair(&[0x5A; 32]).unwrap();
            let a_ntt = pk.expand_a().unwrap().ntt();
            let t_ntt = pk.t().ntt();
            let mut st = seed ^ 0xD2C0D2;
            let mut agg = Ciphertext::zero();
            let mut expected = [0u64; SLOTS];
            let mut naive = [0u64; COORDS];
            for _ in 0..legs {
                let c = coords(&mut st);
                let m = digitize(&c);
                let ct = Ciphertext::encrypt_cached(&a_ntt, &t_ntt, &noise_seed(&mut st), &m).unwrap();
                agg = agg.add(&ct);
                for (e, d) in expected.iter_mut().zip(m.slot_values()) {
                    *e += d;
                }
                for (nv, &x) in naive.iter_mut().zip(c.iter()) {
                    *nv = nv.wrapping_add(x);
                }
            }
            let reveal = decrypt_native(legs, &s, agg.u(), agg.v()).unwrap();
            prop_assert_eq!(*reveal.sums(), expected);
            prop_assert_eq!(reveal.coords(), naive);
        }
    }

    #[test]
    fn zero_plaintext_decodes_to_a_zero_reveal() {
        let (pk, s) = derive_reference_keypair(&[0x66; 32]).unwrap();
        let a_ntt = pk.expand_a().unwrap().ntt();
        let t_ntt = pk.t().ntt();
        let zero = Plaintext::zero();
        let mut st = 0xD2C0_0002u64;
        let mut agg = Ciphertext::zero();
        for _ in 0..128 {
            let ct = Ciphertext::encrypt_cached(&a_ntt, &t_ntt, &noise_seed(&mut st), &zero).unwrap();
            agg = agg.add(&ct);
        }
        let reveal = decrypt_native(128, &s, agg.u(), agg.v()).unwrap();
        assert_eq!(*reveal.sums(), [0u64; SLOTS]);
        assert_eq!(reveal.coords(), [0u64; COORDS]);
    }
}

