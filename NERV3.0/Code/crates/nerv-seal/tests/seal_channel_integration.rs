//! Cross-module integration closing the seal channel (chunk 8): the full
//! §6.3 ceremony path — reference keypair → per-leg encryption →
//! aggregation → decode → carry resolution — plus the invalid-reveal
//! surface on the combination seam, committee padding, smudged
//! combinations, and multi-chunk block assembly. Replaces chunk 7's
//! inline decode oracle (subsumption map in the chunk-8 close-out).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use nerv_seal::decrypt::{
    assemble_block_aggregate, decode_chunk, decrypt_native, ChunkReveal, CHUNK_MAX,
};
use nerv_seal::digitize::{digitize, Plaintext, COORDS};
use nerv_seal::encrypt::{derive_reference_keypair, Ciphertext};
use nerv_seal::noise::{SCALE, SMUDGING_BUDGET};
use nerv_seal::ring::{Mat2x8Ntt, Mat8x8Ntt, Poly, Vec8, N, Q};
use nerv_seal::sampling::NoiseSeed;
use nerv_seal::RevealError;

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

fn full_constant(c: u64) -> Poly {
    let mut a = [0u64; N];
    a.fill(c);
    Poly::new(a)
}

struct Ref {
    a_ntt: Mat8x8Ntt,
    t_ntt: Mat2x8Ntt,
    s: Vec8,
}

fn reference(tag: u8) -> Ref {
    let (pk, s) = derive_reference_keypair(&[tag; 32]).unwrap();
    Ref { a_ntt: pk.expand_a().unwrap().ntt(), t_ntt: pk.t().ntt(), s }
}

fn leg(r: &Ref, ns: &NoiseSeed, coords: &[u64; COORDS]) -> (Ciphertext, Plaintext) {
    let m = digitize(coords);
    let ct = Ciphertext::encrypt_cached(&r.a_ntt, &r.t_ntt, ns, &m).unwrap();
    (ct, m)
}

fn build_chunk(r: &Ref, seed: u64) -> (Ciphertext, [u64; SLOTS], [u64; COORDS]) {
    let mut st = seed;
    let mut agg = Ciphertext::zero();
    let mut expected = [0u64; SLOTS];
    let mut naive = [0u64; COORDS];
    for _ in 0..CHUNK_MAX {
        let c = coords(&mut st);
        let (ct, m) = leg(r, &noise_seed(&mut st), &c);
        agg = agg.add(&ct);
        for (e, d) in expected.iter_mut().zip(m.slot_values()) {
            *e += d;
        }
        for (nv, &x) in naive.iter_mut().zip(c.iter()) {
            *nv = nv.wrapping_add(x);
        }
    }
    (agg, expected, naive)
}

#[test]
fn genesis_chunk_ceremony_is_exact_and_deterministic() {
    assert_eq!(CHUNK_MAX, 128);
    let r = reference(0x42);
    let (agg, expected, naive) = build_chunk(&r, 0x5EED_C001);
    let (agg2, expected2, naive2) = build_chunk(&r, 0x5EED_C001);
    assert_eq!(agg.to_bytes(), agg2.to_bytes());
    assert_eq!(expected, expected2);
    assert_eq!(naive, naive2);

    let reveal = decrypt_native(CHUNK_MAX, &r.s, agg.u(), agg.v()).unwrap();
    assert_eq!(reveal.legs(), CHUNK_MAX);
    assert_eq!(*reveal.sums(), expected);
    assert_eq!(reveal.coords(), naive);

 
    }

    // The public record round-trips and re-validates.
    let bytes = reveal.to_bytes();
    assert_eq!(bytes.len(), 2052);
    assert_eq!(ChunkReveal::from_bytes(&bytes).unwrap(), reveal);
}

#[test]
fn block_aggregate_is_the_public_sum_of_chunk_reveals() {
    let r = reference(0x2B);
    let mut reveals = Vec::new();
    let mut naive = [0u64; COORDS];
    for k in 0..2u64 {
        let (agg, _, chunk_naive) = build_chunk(&r, 0x5EED_C100 + k);
        reveals.push(decrypt_native(CHUNK_MAX, &r.s, agg.u(), agg.v()).unwrap());
        for (nv, &x) in naive.iter_mut().zip(chunk_naive.iter()) {
            *nv = nv.wrapping_add(x);
        }
    }
    assert_eq!(assemble_block_aggregate(&reveals), naive);
    assert_ne!(assemble_block_aggregate(&reveals), reveals[0].coords());
    assert_eq!(assemble_block_aggregate(&[]), [0u64; COORDS]);
}

#[test]
fn corrupted_combination_is_detected_not_silent() {
    // An honest 128-leg aggregate, then three classes of su corruption on
    // the combination seam — the surface chunk 9's VPD proofs police;
    // decode's envelope is the public backstop (§6.3.4, erratum 39).
    let r = reference(0x5C);
    let (agg, _, _) = build_chunk(&r, 0x5EED_C002);
    let su = r.s.dot(agg.u());

    // +70,000-digit uniform shift: main-plane ceiling at slot 0.
    let evil = su.sub(&full_constant(70_000 * SCALE));
    match decode_chunk(CHUNK_MAX, &evil, agg.v()) {
        Err(RevealError::DigitSumAboveEnvelope { slot: 0, .. }) => {}
        other => panic!("A: {other:?}"),
    }
    // +100-digit shift: positions pass (≈ 32,740 ≤ 65,408); the guard
    // plane (≈ 64 + 100 > 128) trips.
    let evil = su.sub(&full_constant(100 * SCALE));
    match decode_chunk(CHUNK_MAX, &evil, agg.v()) {
        Err(RevealError::DigitSumAboveEnvelope { slot:0, .. }) if slot >= 448 => {}
        other => panic!("B: {other:?}"),
    }
    // −70,000-digit shift: negative decode at slot 0.
    let evil = su.add(&full_constant(70_000 * SCALE));
    match decode_chunk(CHUNK_MAX, &evil, agg.v()) {
        Err(RevealError::NegativeDigit { slot: 0, .. }) => {}
        other => panic!("C: {other:?}"),
    }
}

#[test]
fn committee_padding_completes_subminimum_batches() {
    // 100 real legs padded to 128 with well-formed zero-encryptions
    // (§6.3.5): the reveal is exact for the real activity.
    let r = reference(0x3A);
    let mut st = 0x5EED_C200u64;
    let real = 100u64;
    let mut agg = Ciphertext::zero();
    let mut expected = [0u64; SLOTS];
    let mut naive = [0u64; COORDS];
    for _ in 0..real {
        let c = coords(&mut st);
        let (ct, m) = leg(&r, &noise_seed(&mut st), &c);
        agg = agg.add(&ct);
        for (e, d) in expected.iter_mut().zip(m.slot_values()) {
            *e += d;
        }
        for (nv, &x) in naive.iter_mut().zip(c.iter()) {
            *nv = nv.wrapping_add(x);
        }
    }
    let zero = Plaintext::zero();
    for _ in real..CHUNK_MAX {
        let ct = Ciphertext::encrypt_cached(&r.a_ntt, &r.t_ntt, &noise_seed(&mut st), &zero).unwrap();
        agg = agg.add(&ct);
    }
    let reveal = decrypt_native(CHUNK_MAX, &r.s, agg.u(), agg.v()).unwrap();
    assert_eq!(*reveal.sums(), expected);
    assert_eq!(reveal.coords(), naive);
}

#[test]
fn smudged_combination_within_budget_decodes_exactly() {
    let r = reference(0x1D);
    let (agg, expected, _) = build_chunk(&r, 0x5EED_C300);
    let su = r.s.dot(agg.u());
    // ε within the budget: |ε|_∞ = 512 ≤ SMUDGING_BUDGET.
    assert!(512 <= SMUDGING_BUDGET);
    let mut eps = [0u64; N];
    for (i, c) in eps.iter_mut().enumerate() {
        *c = if i % 2 == 0 { 512 } else { Q - 512 };
    }
    let smudged = su.add(&Poly::new(eps));
    let clean = decode_chunk(CHUNK_MAX, &su, agg.v()).unwrap();
    let via_smudged = decode_chunk(CHUNK_MAX, &smudged, agg.v()).unwrap();
    assert_eq!(clean, via_smudged);
    assert_eq!(*via_smudged.sums(), expected);
}

#[test]
fn single_leg_decodes_exactly() {
    let r = reference(0x0F);
    let mut st = 0x5EED_C400u64;
    let c = coords(&mut st);
    let (ct, m) = leg(&r, &noise_seed(&mut st), &c);
    let reveal = decrypt_native(1, &r.s, ct.u(), ct.v()).unwrap();
    assert_eq!(reveal.legs(), 1);
    assert_eq!(*reveal.sums(), m.slot_values());
    assert_eq!(reveal.coords(), c);
}
