//! The engine's single challenge-reduction layer (WP §5.7): every value
//! the prover and verifier derive from Fiat–Shamir — α, ζ, the DEEP
//! combination challenge, the FRI fold challenges, query indices, the
//! grinding witness — passes through here, so both sides derive identical
//! values by identical code (DSR-11). The `TAG_*` constants are the
//! engine's entire FS purpose vocabulary: call sites never mint tags.
//!
//! Reductions (frozen):
//! * `sample_ext` — ONE `FsTranscript` roll; both EF components
//!   rejection-sampled from the single stream (fs.rs's `challenge_f_n`
//!   law, total by its 64-attempt exact-reduction fallback): uniform on
//!   [0, p)².
//! * `sample_index` — one roll, the low `bits` of the squeezed u64:
//!   exactly uniform on [0, 2^bits).
//! * `absorb_instance` — the statement's shape description (widths,
//!   counts, heights, FRI configuration) as ONE framed part of LE u64
//!   words, absorbed before any commitment: two statements of different
//!   shape cannot share a transcript seed.
//! * grinding — absorb an 8-byte LE nonce; the first squeeze under the
//!   purpose must carry `bits` leading zero bits; the ground transcript
//!   continues from the post-squeeze state. `bits == 0` performs no
//!   absorb and no roll.

use crate::air::fs::FsTranscript;
use crate::stark::ext_field::ExtF;
use nerv_core::hash::Hash256;

/// The constraint-folding challenge (drawn after trace/preprocessed
/// commitments and public values, before the quotient).
pub const TAG_ALPHA: &[u8] = b"nerv.stark.alpha.v1";
/// The out-of-domain point (drawn after the quotient commitment and any
/// grinding).
pub const TAG_ZETA: &[u8] = b"nerv.stark.zeta.v1";
/// The DEEP combination challenge (drawn after the opened values).
pub const TAG_DEEP: &[u8] = b"nerv.stark.deep.v1";
/// Per-fold-round challenge; one roll per round, in round order.
pub const TAG_FRI_FOLD: &[u8] = b"nerv.stark.fri.fold.v1";
/// Query positions; one roll per query.
pub const TAG_FRI_QUERY: &[u8] = b"nerv.stark.fri.query.v1";
/// The final polynomial's coefficients (absorbed before queries).
pub const TAG_FRI_FINAL: &[u8] = b"nerv.stark.fri.final.v1";
/// The grinding check squeeze.
pub const TAG_POW: &[u8] = b"nerv.stark.pow.v1";

/// One uniform challenge in EF.
pub fn sample_ext(t: &mut FsTranscript, purpose: &[u8]) -> ExtF {
    let [c0, c1] = t.challenge_f_n::<2>(purpose);
    ExtF::new(c0, c1)
}

/// One uniform index in [0, 2^bits) — a FRI query position.
pub fn sample_index(t: &mut FsTranscript, purpose: &[u8], bits: usize) -> u64 {
    assert!(bits <= 64, "index width is at most 64 bits");
    let v = t.squeeze_u64(purpose);
    if bits == 64 { v } else { v & ((1u64 << bits) - 1) }
}

/// Absorb the statement's shape description as one framed LE-u64 part.
pub fn absorb_instance(t: &mut FsTranscript, fields: &[u64]) {
    let mut buf = Vec::with_capacity(8 * fields.len());
    for f in fields {
        buf.extend_from_slice(&f.to_le_bytes());
    }
    t.absorb_bytes(&buf);
}

fn leading_zero_bits(h: &Hash256) -> u32 {
    let mut n = 0u32;
    for &b in h.as_bytes() {
        if b == 0 {
            n += 8;
        } else {
            return n + b.leading_zeros();
        }
    }
    n
}

/// Prover-side grinding: find the least nonce whose check squeeze carries
/// `bits` leading zero bits, and leave `t` in the ground state. Expected
/// 2^bits transcript clones.
pub fn grind(t: &mut FsTranscript, purpose: &[u8], bits: usize) -> u64 {
    assert!(bits <= 32, "grinding beyond 32 bits is a misconfiguration");
    if bits == 0 {
        return 0;
    }
    let mut nonce = 0u64;
    loop {
        let mut cand = t.clone();
        cand.absorb_bytes(&nonce.to_le_bytes());
        if leading_zero_bits(&cand.squeeze_hash(purpose)) >= bits as u32 {
            *t = cand;
            return nonce;
        }
        nonce += 1;
    }
}

/// Verifier-side grinding check. On `false` the transcript has been left
/// rolled — the caller aborts verification.
pub fn check_grind(t: &mut FsTranscript, purpose: &[u8], bits: usize, nonce: u64) -> bool {
    assert!(bits <= 32, "grinding beyond 32 bits is a misconfiguration");
    if bits == 0 {
        return true;
    }
    t.absorb_bytes(&nonce.to_le_bytes());
    leading_zero_bits(&t.squeeze_hash(purpose)) >= bits as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use nerv_core::field::GOLDILOCKS_PRIME;

    fn seeded(seed: u64) -> FsTranscript {
        let mut t = FsTranscript::new();
        t.absorb_bytes(&seed.to_le_bytes());
        t
    }

    #[test]
    fn purpose_tags_are_distinct() {
        let tags = [
            TAG_ALPHA, TAG_ZETA, TAG_DEEP, TAG_FRI_FOLD, TAG_FRI_QUERY, TAG_FRI_FINAL, TAG_POW,
        ];
        for i in 0..tags.len() {
            for j in i + 1..tags.len() {
                assert_ne!(tags[i], tags[j]);
            }
        }
    }

    #[test]
    fn challenges_are_deterministic_and_purpose_bound() {
        let mut a = seeded(1);
        let mut b = seeded(1);
        for _ in 0..8 {
            assert_eq!(sample_ext(&mut a, TAG_ALPHA), sample_ext(&mut b, TAG_ALPHA));
        }
        assert_eq!(a.ops(), b.ops());

        let mut c = seeded(1);
        assert_ne!(sample_ext(&mut c, TAG_ALPHA), sample_ext(&mut c, TAG_ZETA));

        let mut d = seeded(2);
        let x1 = sample_ext(&mut d, TAG_FRI_FOLD);
        let x2 = sample_ext(&mut d, TAG_FRI_FOLD);
        assert_ne!(x1, x2, "repeated purpose rolls fresh state");

        let mut e = seeded(2);
        let mut f = e.clone();
        assert_eq!(sample_ext(&mut e, TAG_FRI_FOLD), x1);
        assert_eq!(sample_ext(&mut f, TAG_FRI_FOLD), x1);
    }

    #[test]
    fn ext_challenges_are_bounded_and_bit_balanced() {
        let mut t = seeded(0xB1);
        let mut seen0 = [false; 64];
        let mut seen1 = [false; 64];
        let mut distinct_pair = false;
        for _ in 0..512 {
            let (c0, c1) = sample_ext(&mut t, TAG_ALPHA).components();
            assert!(c0.as_u64() < GOLDILOCKS_PRIME);
            assert!(c1.as_u64() < GOLDILOCKS_PRIME);
            for (v, b) in [c0.as_u64(), c1.as_u64()].iter().flat_map(|x| (0..64).map(move |b| (*x, b))) {
                if (v >> b) & 1 == 1 { seen1[b] = true; } else { seen0[b] = true; }
            }
            if c0 != c1 {
                distinct_pair = true;
            }
        }
        for b in 0..64 {
            assert!(seen0[b] && seen1[b], "bit {b} never took both values");
        }
        assert!(distinct_pair);
    }

    #[test]
    fn ops_accounting() {
        let mut t = FsTranscript::new();
        assert_eq!(t.ops(), 0);
        absorb_instance(&mut t, &[7, 8]);
        assert_eq!(t.ops(), 1);
        let _ = sample_ext(&mut t, TAG_ALPHA);
        assert_eq!(t.ops(), 2);
        let _ = sample_index(&mut t, TAG_FRI_QUERY, 10);
        assert_eq!(t.ops(), 3);

        assert_eq!(grind(&mut t, TAG_POW, 0), 0);
        assert_eq!(t.ops(), 3);
        assert!(check_grind(&mut t, TAG_POW, 0, 0));
        assert_eq!(t.ops(), 3);

        let mut pre = t.clone();
        let n = grind(&mut t, TAG_POW, 8);
        assert_eq!(t.ops(), 5);
        assert!(check_grind(&mut pre, TAG_POW, 8, n));
        assert_eq!(pre.ops(), 5);
    }

    #[test]
    fn sample_index_bounds_and_edges() {
        let mut t = seeded(0x1D);
        for bits in 1usize..=24 {
            for _ in 0..64 {
                let i = sample_index(&mut t, TAG_FRI_QUERY, bits);
                assert!(i < (1u64 << bits), "bits={bits} i={i}");
            }
        }
        assert_eq!(sample_index(&mut t, TAG_FRI_QUERY, 0), 0);

        let mut a = seeded(0x2E);
        let mut b = a.clone();
        let full = sample_index(&mut a, TAG_FRI_QUERY, 64);
        assert_eq!(full, b.squeeze_u64(TAG_FRI_QUERY));
        let mut c = seeded(0x2E);
        assert_eq!(sample_index(&mut c, TAG_FRI_QUERY, 64), full);
    }

    #[test]
    #[should_panic(expected = "at most 64")]
    fn sample_index_rejects_oversized_bits() {
        let mut t = FsTranscript::new();
        let _ = sample_index(&mut t, TAG_FRI_QUERY, 65);
    }

    #[test]
    fn absorb_instance_framing() {
        // One part [a, b] is not two parts [a], [b].
        let mut one = seeded(3);
        absorb_instance(&mut one, &[10, 20]);
        let mut two = seeded(3);
        absorb_instance(&mut two, &[10]);
        absorb_instance(&mut two, &[20]);
        assert_ne!(sample_ext(&mut one, TAG_ZETA), sample_ext(&mut two, TAG_ZETA));

        // absorb_instance(&[a]) is exactly absorb_bytes(a.to_le_bytes()).
        let mut x = seeded(4);
        absorb_instance(&mut x, &[0x1122334455667788]);
        let mut y = seeded(4);
        y.absorb_bytes(&0x1122334455667788u64.to_le_bytes());
        assert_eq!(sample_ext(&mut x, TAG_ZETA), sample_ext(&mut y, TAG_ZETA));

        // The empty instance is still one distinct op.
        let mut e = seeded(5);
        absorb_instance(&mut e, &[]);
        let mut n = seeded(5);
        assert_ne!(sample_ext(&mut e, TAG_ZETA), sample_ext(&mut n, TAG_ZETA));
        assert_eq!(e.ops(), 1);
    }

    #[test]
    fn grind_is_replayable_and_synchronized() {
        let mut p = seeded(0x99);
        let mut v = seeded(0x99);
        let n = grind(&mut p, TAG_POW, 8);
        assert!(check_grind(&mut v, TAG_POW, 8, n));
        assert_eq!(sample_ext(&mut p, TAG_ZETA), sample_ext(&mut v, TAG_ZETA));
        assert_eq!(p.ops(), v.ops());

        let mut z = seeded(0x99);
        assert_eq!(grind(&mut z, TAG_POW, 0), 0);
        assert_eq!(z.ops(), 0);
        assert!(check_grind(&mut z, TAG_POW, 0, 0));
        assert_eq!(z.ops(), 0);
    }

    #[test]
    fn wrong_nonce_fails() {
        let bits = 16;
        let mut p = seeded(0x77);
        let n = grind(&mut p, TAG_POW, bits);
        let mut all_passed = true;
        for k in 1..=8u64 {
            let mut v = seeded(0x77);
            if !check_grind(&mut v, TAG_POW, bits, n + k) {
                all_passed = false;
                break;
            }
        }
        assert!(!all_passed, "eight forged nonces cannot all pass {bits} bits");
    }
}

