//! Native-checkable forms of proof statements 9–10 (WP §5.1) — the DSR-7
//! twins the seal chip (nerv-proofs, chunks 10–11) must mirror bit-for-bit
//! before anything folds. Nothing here is a proof; these are the frozen
//! relations and their native verifiers.
//!
//! Statement 9 — digit encoding: the plaintext m's 512 coefficients are
//! exactly the canonical 8-bit digits of the proven delta δ (erratum 40
//! layout: digit k of coordinate j at slot 64k+j). AIR-facing form: per
//! coordinate j, eight 8-bit range checks and the reconstruction identity
//! Σ_k d_k·2^{8k} ≡ δ_j (mod 2^64) — the position-7 carry is the wrap.
//!
//! Statement 10 — well-formedness: ct = (u, v) satisfies
//!   u = A·r + e₁   (8 rows),   v = T·r + e₂ + scale·m   (2 rows)
//! with A = XOF(ASeed), T the epoch public key, and in-circuit bounds
//! |r|∞, |e₁|∞, |e₂|∞ ≤ noise::CIRCUIT_NOISE_BOUND (±3) and m canonical
//! (≤ 255). The witness block order is frozen: r ‖ e₁ ‖ e₂ ‖ m (20 blocks).
//!
//! `epoch_key_identifier` realizes §5.1's public input "the seal epoch
//! key identifier": H(ASeed ‖ T) under `nerv.seal.stmt`.


use nerv_core::constants::SEAL_STMT;
use nerv_core::hash::Hash256;
use crate::digitize::{digitize, Plaintext, COORDS, DIGIT_BITS, DIGIT_MAX, DIGITS_PER_COORD};
use crate::error::CircuitStmtError;
use crate::noise::{CIRCUIT_NOISE_BOUND, SCALE};
use crate::ring::{Mat2x8, Mat2x8Ntt, Mat8x8Ntt, Poly, Vec2, Vec8};
use crate::sampling::{expand_matrix, ASeed};


/// Witness block count: r ‖ e₁ ‖ e₂ ‖ m.
pub const STATEMENT_10_BLOCKS: usize = 8 + 8 + 2 + 2;
/// Relation row count: u (8) ‖ v (2).
pub const STATEMENT_10_ROWS: usize = 10;
/// Short-block bound for r, e₁, e₂ (statement 10; WP §6.3.7's 3σ).
pub const SHORT_BOUND: u64 = CIRCUIT_NOISE_BOUND;


fn inf_norm(p: &Poly) -> u64 {
    p.centerlift().iter().map(|v| v.unsigned_abs()).max().unwrap_or(0)
}


// ---------------------------------------------------------------------------
// Statement 9
// ---------------------------------------------------------------------------


/// Native check: m encodes exactly the canonical digits of `delta`.
pub fn digits_encode_delta(delta: &[u64; COORDS], m: &Plaintext) -> bool {
    *m.pair() == *digitize(delta).pair()
}


/// Per-coordinate granularity (the AIR's unit): coordinate j's eight
/// digits reconstruct δ_j mod 2^64.
pub fn coordinate_encoding_holds(delta: &[u64; COORDS], m: &Plaintext, j: usize) -> bool {
    let mut acc: u128 = 0;
    for k in 0..DIGITS_PER_COORD {
        acc += u128::from(m.digit(j, k)) << (DIGIT_BITS * k as u32);
    }
    acc as u64 == delta[j]
}


/// The expected-digit vector the chip constrains m against (slot order).
pub fn expected_digits(delta: &[u64; COORDS]) -> [u16; 512] {
    digitize(delta).digits()
}


// ---------------------------------------------------------------------------
// Statement 10
// ---------------------------------------------------------------------------


/// Evaluates the statement-10 relations: (u, v) = (A·r + e₁, T·r + e₂ +
/// scale·m). The chip's algebra, natively.
pub fn apply_statement_10(
    a_seed: &ASeed,
    t: &Mat2x8,
    r: &Vec8,
    e1: &Vec8,
    e2: &Vec2,
    m: &Plaintext,
) -> Result<(Vec8, Vec2), CircuitStmtError> {
    let a_ntt: Mat8x8Ntt = expand_matrix(a_seed)?.ntt();
    let t_ntt: Mat2x8Ntt = t.ntt();
    let r_ntt = r.ntt();
    let u = a_ntt.mul_vec(&r_ntt).intt().add(e1);
    let scaled = Vec2::new([
        m.pair().poly(0).mul_scalar(SCALE),
        m.pair().poly(1).mul_scalar(SCALE),
    ]);
    let v = t_ntt.mul_vec(&r_ntt).intt().add(e2).add(&scaled);
    Ok((u, v))
}


/// The full native verifier for statement 10: bounds, then all ten
/// relations. Rows 0–7 are u, 8–9 are v.
pub fn check_statement_10(
    a_seed: &ASeed,
    t: &Mat2x8,
    u: &Vec8,
    v: &Vec2,
    m: &Plaintext,
    r: &Vec8,
    e1: &Vec8,
    e2: &Vec2,
) -> Result<(), CircuitStmtError> {
    for (b, p) in r.polys().iter().enumerate() {
        let n = inf_norm(p);
        if n > SHORT_BOUND {
            return Err(CircuitStmtError::NoiseBound { block: b, value: n, bound: SHORT_BOUND });
        }
    }
    for (b, p) in e1.polys().iter().enumerate() {
        let n = inf_norm(p);
        if n > SHORT_BOUND {
            return Err(CircuitStmtError::NoiseBound { block: 8 + b, value: n, bound: SHORT_BOUND });
        }
    }
    for (b, p) in e2.polys().iter().enumerate() {
        let n = inf_norm(p);
        if n > SHORT_BOUND {
            return Err(CircuitStmtError::NoiseBound { block: 16 + b, value: n, bound: SHORT_BOUND });
        }
    }
    for (s, &d) in m.slot_values().iter().enumerate() {
        if d > DIGIT_MAX {
            return Err(CircuitStmtError::DigitBound { slot: s, value: d });
        }
    }
    let (u2, v2) = apply_statement_10(a_seed, t, r, e1, e2, m)?;
    for i in 0..8 {
        if u.poly(i) != u2.poly(i) {
            return Err(CircuitStmtError::RelationFailed { row: i });
        }
    }
    for c in 0..2 {
        if v.poly(c) != v2.poly(c) {
            return Err(CircuitStmtError::RelationFailed { row: 8 + c });
        }
    }
    Ok(())
}


/// §5.1's seal epoch key identifier: H("nerv.seal.stmt" ‖ ASeed ‖ T).
pub fn epoch_key_identifier(a_seed: &ASeed, t: &Mat2x8) -> [u8; 32] {
    let mut msg = Vec::with_capacity(32 + 2 * Vec8::WIRE_SIZE);
    msg.extend_from_slice(a_seed.as_bytes());
    for row in t.rows().iter() {
        msg.extend_from_slice(&Vec8::new(*row).to_bytes());
    }
    *Hash256::concat(&SEAL_STMT, &msg).as_bytes()
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::encrypt::{derive_reference_keypair, Ciphertext};
    use crate::ring::Q;
    use crate::sampling::{derive_short_triplet, NoiseSeed};
    use proptest::prelude::*;


    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }


    fn coords(state: &mut u64) -> [u64; COORDS] {
        let mut c = [0u64; COORDS];
        for v in c.iter_mut() {
            *v = splitmix64(state);
        }
        c
    }


    fn noise_seed(state: &mut u64) -> NoiseSeed {
        let mut b = [0u8; 32];
        for ch in b.chunks_mut(8) {
            ch.copy_from_slice(&splitmix64(state).to_le_bytes());
        }
        NoiseSeed::from_bytes(b)
    }


    #[test]
    fn statement_9_holds_and_detects_tampering() {
        let mut st = 0x57A7_0000u64;
        let delta = coords(&mut st);
        let m = digitize(&delta);
        assert!(digits_encode_delta(&delta, &m));
        for j in 0..COORDS {
            assert!(coordinate_encoding_holds(&delta, &m, j));
        }
        assert_eq!(expected_digits(&delta), m.digits());


        // A tampered digit breaks its coordinate only.
        let mut polys = *m.pair().polys();
        let mut c = *polys[0].coefficients();
        c[0] = (c[0] + 1) % Q;
        polys[0] = Poly::new(c);
        let bad = Plaintext::from_pair(&Vec2::new(polys)).unwrap();
        assert!(!digits_encode_delta(&delta, &bad));
        assert!(!coordinate_encoding_holds(&delta, &bad, 0));
        assert!(coordinate_encoding_holds(&delta, &bad, 1));


        assert!(digits_encode_delta(&[0u64; COORDS], &Plaintext::zero()));
        let ones = [u64::MAX; COORDS];
        assert!(digits_encode_delta(&ones, &digitize(&ones)));
    }


    #[test]
    fn statement_10_holds_for_honest_encryption() {
        let (pk, _) = derive_reference_keypair(&[0x3B; 32]).unwrap();
        let mut st = 0x57A7_0001u64;
        let delta = coords(&mut st);
        let m = digitize(&delta);
        let ns = noise_seed(&mut st);
        let ct = Ciphertext::encrypt(&pk, &ns, &m).unwrap();
        let (r, e1, e2) = derive_short_triplet(&ns).unwrap();
        check_statement_10(
            pk.a_seed(), pk.t(), ct.u(), ct.v(), &m, &r, &e1, &e2,
        )
        .unwrap();
        assert!(digits_encode_delta(&delta, &m));
    }


    #[test]
    fn statement_10_detects_every_tamper_class() {
        let (pk, _) = derive_reference_keypair(&[0x3C; 32]).unwrap();
        let mut st = 0x57A7_0002u64;
        let delta = coords(&mut st);
        let m = digitize(&delta);
        let ns = noise_seed(&mut st);
        let ct = Ciphertext::encrypt(&pk, &ns, &m).unwrap();
        let (r, e1, e2) = derive_short_triplet(&ns).unwrap();
        let seed = pk.a_seed();
        let t = pk.t();
        let check = |u, v, m2, r2, e12, e22| {
            check_statement_10(seed, t, u, v, m2, r2, e12, e22)
        };


        // Tampered u / v.
        let mut polys = *ct.u().polys();
        let mut c = *polys[3].coefficients();
        c[7] = (c[7] + 1) % Q;
        polys[3] = Poly::new(c);
        let bad_u = Vec8::new(polys);
        assert!(matches!(
            check(&bad_u, ct.v(), &m, &r, &e1, &e2),
            Err(CircuitStmtError::RelationFailed { row: 3 })
        ));
        let mut polys = *ct.v().polys();
        let mut c = *polys[1].coefficients();
        c[0] = (c[0] + 1) % Q;
        polys[1] = Poly::new(c);
        let bad_v = Vec2::new(polys);
        assert!(matches!(
            check(ct.u(), &bad_v, &m, &r, &e1, &e2),
            Err(CircuitStmtError::RelationFailed { row: 9 })
        ));


        // Tampered witness components.
        let mut polys = *r.polys();
        let cl = polys[0].centerlift();
        let mut cl2 = cl;
        cl2[1] += 1;
        polys[0] = Poly::from_centered(&cl2);
        assert!(matches!(
            check(ct.u(), ct.v(), &m, &Vec8::new(polys), &e1, &e2),
            Err(CircuitStmtError::RelationFailed { .. })
        ));
        let mut polys = *e1.polys();
        let mut cl2 = polys[5].centerlift();
        cl2[9] += 1;
        polys[5] = Poly::from_centered(&cl2);
        assert!(matches!(
            check(ct.u(), ct.v(), &m, &r, &Vec8::new(polys), &e2),
            Err(CircuitStmtError::RelationFailed { .. })
        ));
        let mut polys = *e2.polys();
        let mut cl2 = polys[0].centerlift();
        cl2[2] -= 1;
        polys[0] = Poly::from_centered(&cl2);
        assert!(matches!(
            check(ct.u(), ct.v(), &m, &r, &e1, &Vec2::new(polys)),
            Err(CircuitStmtError::RelationFailed { row: 8 })
        ));


        // A different plaintext under the same ciphertext.
        let delta2 = coords(&mut st);
        let m2 = digitize(&delta2);
        assert!(matches!(
            check(ct.u(), ct.v(), &m2, &r, &e1, &e2),
            Err(CircuitStmtError::RelationFailed { .. })
        ));


        // Bound violation: r with a coefficient of 4 (> ±3).
        let mut polys = *r.polys();
        let mut vals = polys[0].centerlift();
        vals[0] = 4;
        polys[0] = Poly::from_centered(&vals);
        assert!(matches!(
            check(ct.u(), ct.v(), &m, &Vec8::new(polys), &e1, &e2),
            Err(CircuitStmtError::NoiseBound { block: 0, value: 4, bound: 3 })
        ));
        let mut polys = *e2.polys();
        let mut vals = polys[1].centerlift();
        vals[4] = -4;
        polys[1] = Poly::from_centered(&vals);
        assert!(matches!(
            check(ct.u(), ct.v(), &m, &r, &e1, &Vec2::new(polys)),
            Err(CircuitStmtError::NoiseBound { block: 17, value: 4, bound: 3 })
        ));
    }


    #[test]
    fn apply_statement_10_matches_encrypt() {
        let (pk, _) = derive_reference_keypair(&[0x3D; 32]).unwrap();
        let mut st = 0x57A7_0003u64;
        let m = digitize(&coords(&mut st));
        let ns = noise_seed(&mut st);
        let ct = Ciphertext::encrypt(&pk, &ns, &m).unwrap();
        let (r, e1, e2) = derive_short_triplet(&ns).unwrap();
        let (u, v) = apply_statement_10(pk.a_seed(), pk.t(), &r, &e1, &e2, &m).unwrap();
        assert_eq!(u.to_bytes(), ct.u().to_bytes());
        assert_eq!(v.to_bytes(), ct.v().to_bytes());
    }


    #[test]
    fn epoch_key_identifier_is_pinned_and_sensitive() {
        let (pk, _) = derive_reference_keypair(&[0x3E; 32]).unwrap();
        let id1 = epoch_key_identifier(pk.a_seed(), pk.t());
        assert_eq!(id1, epoch_key_identifier(pk.a_seed(), pk.t()));
        // Differential pin: H("nerv.seal.stmt" ‖ ASeed ‖ T wire).
        let mut msg = Vec::new();
        msg.extend_from_slice(SEAL_STMT.as_bytes());
        msg.extend_from_slice(pk.a_seed().as_bytes());
        for row in pk.t().rows().iter() {
            msg.extend_from_slice(&Vec8::new(*row).to_bytes());
        }
   
        let (pk2, _) = derive_reference_keypair(&[0x3F; 32]).unwrap();
        assert_ne!(id1, epoch_key_identifier(pk2.a_seed(), pk2.t()));
            // Same seed, tampered T: different identifier.
        let mut rows = [*pk.t().row(0), *pk.t().row(1)];
        let mut c = *rows[0][0].coefficients();
        c[0] = (c[0] + 1) % Q;
        rows[0][0] = Poly::new(c);
        assert_ne!(id1, epoch_key_identifier(pk.a_seed(), &Mat2x8::new(rows)));

    }




    proptest! {
        #[test]
        fn prop_encrypt_satisfies_both_statements(seed in any::<u64>()) {
            let (pk, _) = derive_reference_keypair(&[0x51; 32]).unwrap();
            let mut st = seed ^ 0x57A7;
            let delta = coords(&mut st);
            let m = digitize(&delta);
            let ns = noise_seed(&mut st);
            let ct = Ciphertext::encrypt(&pk, &ns, &m).unwrap();
            let (r, e1, e2) = derive_short_triplet(&ns).unwrap();
            prop_assert!(check_statement_10(pk.a_seed(), pk.t(), ct.u(), ct.v(), &m, &r, &e1, &e2).is_ok());
            prop_assert!(digits_encode_delta(&delta, &m));
        }
    }
}
