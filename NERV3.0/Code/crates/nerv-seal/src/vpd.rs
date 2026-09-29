//! Verifiable partial decryption (WP §6.3.3; errata 47–49).
//!
//! Per chunk, each member j of the participating t-subset S publishes
//!   p_j = λ_j·(s_j·u_B) + ε_j ∈ R
//! with λ_j the subset's Lagrange coefficient (folded into the partial,
//! WP-literal: combination is a plain sum) and ε_j the member's smudging,
//! |ε_j|∞ ≤ SMUDGE_BOUND = 128 — a witness bound, so t·128 = 896 ≤
//! noise::SMUDGING_BUDGET holds structurally for any t verified partials;
//! ε is never published, so combination needs no budget check (erratum 48).
//!
//! The proof (`dkg::sigma`, domain `nerv.seal.vpd`) attests, over the
//! short witness (s_j ‖ ρ̄_j ‖ ε_j) — 17 blocks, bounds (70 ‖ 10 ‖ 128):
//!   * rows 0–7: the public share commitment W_j = A_L·s_j + A_R·ρ̄_j
//!     (the DKG transcript's value — the partial's share IS a short
//!     opening of it), and
//!   * row 8: p_j = λ_j·(s_j·u_B) + ε_j, coefficients λ_j·u_B[k] public.
//! The FS context binds the full statement determiner: ASeed ‖ member ‖
//! n ‖ t ‖ subset(ascending) ‖ u_B — a partial is valid for exactly one
//! (key, subset, batch ciphertext) triple.
//!
//! Leakage posture (erratum 47, stated here because this file is where
//! the surface lives): the published partials and combined values are
//! LWE samples on the shares and the epoch key, with noise exactly the
//! smudging — dominated by (aggregate) or comparable to (per-partial)
//! the public-key MLWE instance, with sample abundance the unpriced
//! axis. Rotation + PSS bound it; the surface is a named M1 estimator
//! line-item; §9.4's kill-switch is the designed fallback.


use nerv_core::constants::SEAL_VPD;
use nerv_core::hash::Xof;


use crate::dkg::sigma;
use crate::dkg::{
    commit, lagrange_coefficients, CommitMatrix, Proof, ShareSecret, Statement, THRESHOLD,
};
use crate::error::{SealError, VpdError};
use crate::noise::{KEY_BOUND, SMUDGING_BUDGET};
use crate::ring::{Poly, Vec8, N};


pub use crate::dkg::SHARE_BOUND;
pub use crate::sampling::ASeed;


/// Per-member smudging bound: |ε_j|∞ ≤ 128. ρ̄_j is also an n-fold
/// ternary sum, hence noise::KEY_BOUND. The budget pin is compile-time.
pub const SMUDGE_BOUND: u64 = 128;


const _: () = assert!(THRESHOLD as u64 * SMUDGE_BOUND <= SMUDGING_BUDGET as u64);


// ---------------------------------------------------------------------------
// The partial and its wire form
// ---------------------------------------------------------------------------


#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PartialDecryption {
    pub member: u8,
    pub partial: Poly,
    pub proof: Proof,
}


impl PartialDecryption {
    pub fn to_bytes(&self) -> Vec<u8> {
        let pb = self.proof.to_bytes();
        let mut out = Vec::with_capacity(5 + 1024 + pb.len());
        out.push(self.member);
        out.extend_from_slice(&self.partial.to_bytes());
        out.extend_from_slice(&(pb.len() as u32).to_le_bytes());
        out.extend_from_slice(&pb);
        out
    }


    pub fn from_bytes(bytes: &[u8]) -> Result<PartialDecryption, VpdError> {
        const BODY: usize = 1 + 1024 + 4;
        if bytes.len() < BODY {
            return Err(VpdError::BadLength { len: bytes.len(), expected: BODY });
        }
        let member = bytes[0];
        let mut pb = [0u8; 1024];
        pb.copy_from_slice(&bytes[1..1025]);
        let partial = Poly::from_bytes(&pb)?;
        let mut lb = [0u8; 4];
        lb.copy_from_slice(&bytes[1025..1029]);
        let plen = u32::from_le_bytes(lb) as usize;
        if bytes.len() != BODY - 4 + plen {
            return Err(VpdError::BadLength { len: bytes.len(), expected: BODY - 4 + plen });
        }
        let proof = Proof::from_bytes(&bytes[1029..])?;
        Ok(PartialDecryption { member, partial, proof })
    }
}


// ---------------------------------------------------------------------------
// Smudging
// ---------------------------------------------------------------------------


fn eps_stream(member: u8, proof_seed: &[u8; 32], u_b: &Vec8, subset: &[u8]) -> Xof {
    let ub = u_b.to_bytes();
    let parts: [&[u8]; 4] = [&[member], proof_seed, &ub, subset];
    Xof::framed(&SEAL_VPD, &parts)
}


/// Uniform ε ∈ [−128, 128]^256 — exact sampling: u16 draws, reject only
/// 0xFFFF (257·255 = 65,535 = 2^16 − 1, so accepted draws are uniform on
/// 257 classes). Cap 64 per coefficient keeps the sampler total.
fn sample_eps(xof: &mut Xof) -> Result<Poly, SealError> {
    let mut vals = [0i64; N];
    for v in vals.iter_mut() {
        let mut settled = false;
        for _ in 0..64 {
            let x = u16::from_le_bytes(xof.read_array::<2>());
            if x != 0xFFFF {
                *v = i64::from(x % 257) - 128;
                settled = true;
                break;
            }
        }
        if !settled {
            return Err(SealError::SamplingExhausted { attempts: 64 });
        }
    }
    Ok(Poly::from_centered(&vals))
}


// ---------------------------------------------------------------------------
// The statement
// ---------------------------------------------------------------------------


fn build_statement(
    ac: &CommitMatrix,
    a_seed: &ASeed,
    member: u8,
    subset: &[u8],
    n: usize,
    t: usize,
    u_b: &Vec8,
    lambda_j: &Poly,
    w_j: &Vec8,
    p_j: &Poly,
) -> Statement {
    let mut ctx = Vec::with_capacity(32 + 3 + subset.len() + Vec8::WIRE_SIZE);
    ctx.extend_from_slice(a_seed.as_bytes());
    ctx.push(member);
    ctx.push(n as u8);
    ctx.push(t as u8);
    ctx.extend_from_slice(subset);
    ctx.extend_from_slice(&u_b.to_bytes());
    let mut bounds = vec![SHARE_BOUND; 8];
    bounds.extend_from_slice(&[KEY_BOUND; 8]);
    bounds.push(SMUDGE_BOUND);
    let mut stmt = Statement::new(bounds, ctx);
    for r in 0..8 {
        let mut entries = Vec::with_capacity(16);
        for k in 0..8 {
            entries.push((k, ac.left.row(r)[k]));
        }
        for m in 0..8 {
            entries.push((8 + m, ac.right.row(r)[m]));
        }
        stmt.push_equation(*w_j.poly(r), entries);
    }
    let mut entries = Vec::with_capacity(9);
    for k in 0..8 {
        entries.push((k, lambda_j.mul(u_b.poly(k))));
    }
    entries.push((16, Poly::one()));
    stmt.push_equation(*p_j, entries);
    stmt
}


// ---------------------------------------------------------------------------
// Prove / verify / combine
// ---------------------------------------------------------------------------


fn prove_with_eps(
    share: &ShareSecret,
    ac: &CommitMatrix,
    a_seed: &ASeed,
    u_b: &Vec8,
    subset: &[u8],
    n: usize,
    t: usize,
    eps: Poly,
    proof_seed: &[u8; 32],
) -> Result<PartialDecryption, VpdError> {
    let member = share.member;
    let idx = subset
        .iter()
        .position(|&m| m == member)
        .ok_or(VpdError::MemberNotInSubset { member })?;
    let lambdas = lagrange_coefficients(subset, n, t)?;
    let lambda_j = &lambdas[idx];
    let w_j = commit(ac, &share.share, &share.rho);
    let partial = share.share.dot(u_b).mul(lambda_j).add(&eps);
    let stmt = build_statement(ac, a_seed, member, subset, n, t, u_b, lambda_j, &w_j, &partial);
    let mut witness = Vec::with_capacity(17);
    witness.extend_from_slice(share.share.polys());
    witness.extend_from_slice(share.rho.polys());
    witness.push(eps);
    let proof = sigma::prove(&SEAL_VPD, &stmt, &witness, proof_seed)?;
    Ok(PartialDecryption { member, partial, proof })
}


/// Produces member j's partial for the chunk aggregate (u_B, ·) under the
/// subset S. Deterministic in (share, a_seed, u_b, subset, proof_seed);
/// production passes a fresh proof_seed per partial.
pub fn prove_partial(
    share: &ShareSecret,
    ac: &CommitMatrix,
    a_seed: &ASeed,
    u_b: &Vec8,
    subset: &[u8],
    n: usize,
    t: usize,
    proof_seed: &[u8; 32],
) -> Result<PartialDecryption, VpdError> {
    let eps = sample_eps(&mut eps_stream(share.member, proof_seed, u_b, subset))?;
    prove_with_eps(share, ac, a_seed, u_b, subset, n, t, eps, proof_seed)
}


/// Verifies a partial against the DKG transcript's share commitment W_j.
/// The partial is valid for exactly the (a_seed, subset, u_b) triple it
/// was proven under.
pub fn verify_partial(
    p: &PartialDecryption,
    ac: &CommitMatrix,
    a_seed: &ASeed,
    u_b: &Vec8,
    subset: &[u8],
    n: usize,
    t: usize,
    w_j: &Vec8,
) -> Result<(), VpdError> {
    let idx = subset
        .iter()
        .position(|&m| m == p.member)
        .ok_or(VpdError::MemberNotInSubset { member: p.member })?;
    let lambdas = lagrange_coefficients(subset, n, t)?;
    let stmt = build_statement(
        ac, a_seed, p.member, subset, n, t, u_b, &lambdas[idx], w_j, &p.partial,
    );
    Ok(sigma::verify(&SEAL_VPD, &stmt, &p.proof)?)
}


/// Verifies every partial (in subset order, against its W_j) and returns
/// Σ p_j — the committee's combined value for `decode_chunk`. The
/// smudging budget holds structurally: t proven bounds of 128 give
/// |Σp_j − s·u_B|∞ ≤ t·128 ≤ noise::SMUDGING_BUDGET.
pub fn combine_partials(
    partials: &[PartialDecryption],
    ac: &CommitMatrix,
    a_seed: &ASeed,
    u_b: &Vec8,
    subset: &[u8],
    w_js: &[Vec8],
    n: usize,
    t: usize,
) -> Result<Poly, VpdError> {
    if partials.len() != t || subset.len() != t || w_js.len() != t {
        return Err(VpdError::BadSubset { len: partials.len(), expected: t });
    }
    let mut su = Poly::zero();
    for (i, p) in partials.iter().enumerate() {
        if p.member != subset[i] {
            return Err(VpdError::MemberMismatch {
                index: i,
                partial: p.member,
                subset: subset[i],
            });
        }
        verify_partial(p, ac, a_seed, u_b, subset, n, t, &w_js[i])?;
        su = su.add(&p.partial);
    }
    Ok(su)
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::dkg::COMMITTEE_SIZE;
    use crate::error::SigmaError;
    use crate::ring::Q;


    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }


    fn synth_vec(bound: i64, st: &mut u64) -> Vec8 {
        let mut polys = [Poly::zero(); 8];
        for p in polys.iter_mut() {
            let mut vals = [0i64; N];
            for v in vals.iter_mut() {
                *v = (splitmix64(st) % (2 * bound as u64 + 1)) as i64 - bound;
            }
            *p = Poly::from_centered(&vals);
        }
        Vec8::new(polys)
    }


    fn synth_secret(member: u8, st: &mut u64) -> ShareSecret {
        ShareSecret {
            member,
            share: synth_vec(SHARE_BOUND as i64, st),
            rho: synth_vec(KEY_BOUND as i64, st),
        }
    }


    struct Fix {
        a_seed: ASeed,
        ac: CommitMatrix,
        u_b: Vec8,
        subset: Vec<u8>,
    }


    fn fixture(tag: u8) -> Fix {
        let a_seed = ASeed::from_bytes([tag; 32]);
        let ac = CommitMatrix::expand(&a_seed).unwrap();
        let mut st = 0xE651_0000u64 | u64::from(tag);
        let mut polys = [Poly::zero(); 8];
        for p in polys.iter_mut() {
            let mut a = [0u64; N];
            for c in a.iter_mut() {
                *c = splitmix64(&mut st) % Q;
            }
            *p = Poly::new(a);
        }
        Fix { a_seed, ac, u_b: Vec8::new(polys), subset: (1..=7).collect() }
    }


    #[test]
    fn proves_verifies_and_is_deterministic() {
        let f = fixture(0x21);
        let mut st = 0xE651_0021u64;
        let secret = synth_secret(3, &mut st);
        let w_j = commit(&f.ac, &secret.share, &secret.rho);
        let p1 = prove_partial(&secret, &f.ac, &f.a_seed, &f.u_b, &f.subset, COMMITTEE_SIZE, THRESHOLD, &[0x5E; 32]).unwrap();
        let p2 = prove_partial(&secret, &f.ac, &f.a_seed, &f.u_b, &f.subset, COMMITTEE_SIZE, THRESHOLD, &[0x5E; 32]).unwrap();
        assert_eq!(p1.to_bytes(), p2.to_bytes());
        verify_partial(&p1, &f.ac, &f.a_seed, &f.u_b, &f.subset, COMMITTEE_SIZE, THRESHOLD, &w_j).unwrap();
        // A different share commitment (wrong transcript) rejects.
        let other = commit(&f.ac, &synth_secret(4, &mut st).share, &synth_secret(5, &mut st).rho);
        assert!(matches!(
            verify_partial(&p1, &f.ac, &f.a_seed, &f.u_b, &f.subset, COMMITTEE_SIZE, THRESHOLD, &other),
            Err(VpdError::Sigma(SigmaError::EquationFailed { .. }))
        ));
        // Shape pins (erratum 48): 9 rows, 17 blocks, 27,661 B wire.
        assert_eq!(p1.proof.h.len(), 9);
        assert_eq!(p1.proof.z.len(), 17);
        assert_eq!(p1.to_bytes().len(), 1 + 1024 + 4 + 26_632);
    }


    #[test]
    fn eps_stream_is_domain_and_order_pinned() {
        let f = fixture(0x22);
        let seed = [0x7C; 32];
        let member = 3u8;
        let ub = f.u_b.to_bytes();
        let parts: [&[u8]; 4] = [&[member], &seed, &ub, &f.subset];
        let mut manual = Xof::framed(&SEAL_VPD, &parts);
        let mut got = eps_stream(member, &seed, &f.u_b, &f.subset);
        let mut a = [0u8; 64];
        let mut b = [0u8; 64];
        got.fill(&mut a);
        manual.fill(&mut b);
        assert_eq!(a, b);
    }


    #[test]
    fn member_not_in_subset_rejected() {
        let f = fixture(0x23);
        let mut st = 0xE651_0023u64;
        let secret = synth_secret(8, &mut st); // 8 ∉ [1..7]
        assert!(matches!(
            prove_partial(&secret, &f.ac, &f.a_seed, &f.u_b, &f.subset, COMMITTEE_SIZE, THRESHOLD, &[1; 32]),
            Err(VpdError::MemberNotInSubset { member: 8 })
        ));
        let inside = synth_secret(1, &mut st);
        let p = prove_partial(&inside, &f.ac, &f.a_seed, &f.u_b, &f.subset, COMMITTEE_SIZE, THRESHOLD, &[2; 32]).unwrap();
        let mut outsider = p.clone();
        outsider.member = 8;
        assert!(matches!(
            verify_partial(&outsider, &f.ac, &f.a_seed, &f.u_b, &f.subset, COMMITTEE_SIZE, THRESHOLD, &commit(&f.ac, &inside.share, &inside.rho)),
            Err(VpdError::MemberNotInSubset { member: 8 })
        ));
    }


    #[test]
    fn subset_binding_is_enforced_by_the_challenge() {
        let f = fixture(0x24);
        let mut st = 0xE651_0024u64;
        let secret = synth_secret(1, &mut st);
        let w_j = commit(&f.ac, &secret.share, &secret.rho);
        let p = prove_partial(&secret, &f.ac, &f.a_seed, &f.u_b, &f.subset, COMMITTEE_SIZE, THRESHOLD, &[3; 32]).unwrap();
        // Same member, different subset: λ_1 and the context change.
        let s2: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 8];
        assert!(matches!(
            verify_partial(&p, &f.ac, &f.a_seed, &f.u_b, &s2, COMMITTEE_SIZE, THRESHOLD, &w_j),
            Err(VpdError::Sigma(SigmaError::EquationFailed { .. }))
        ));
        // Same subset, different batch ciphertext.
        let mut polys = *f.u_b.polys();
        let mut coeffs = *polys[0].coefficients();
        coeffs[0] = (coeffs[0] + 1) % Q;
        polys[0] = Poly::new(coeffs);
        let u2 = Vec8::new(polys);
        assert!(matches!(
            verify_partial(&p, &f.ac, &f.a_seed, &u2, &f.subset, COMMITTEE_SIZE, THRESHOLD, &w_j),
            Err(VpdError::Sigma(SigmaError::EquationFailed { .. }))
        ));
    }


    #[test]
    fn tampered_partial_and_proof_fail() {
        let f = fixture(0x25);
        let mut st = 0xE651_0025u64;
        let secret = synth_secret(2, &mut st);
        let w_j = commit(&f.ac, &secret.share, &secret.rho);
        let p = prove_partial(&secret, &f.ac, &f.a_seed, &f.u_b, &f.subset, COMMITTEE_SIZE, THRESHOLD, &[4; 32]).unwrap();


        let mut bad = p.clone();
        let mut cl = bad.partial.centerlift();
        cl[9] += 1;
        bad.partial = Poly::from_centered(&cl);
        assert!(matches!(
            verify_partial(&bad, &f.ac, &f.a_seed, &f.u_b, &f.subset, COMMITTEE_SIZE, THRESHOLD, &w_j),
            Err(VpdError::Sigma(SigmaError::EquationFailed { .. }))
        ));


        let mut bad2 = p.clone();
        let mut zl = bad2.proof.z[0].centerlift();
        zl[3] += 1;
        bad2.proof.z[0] = Poly::from_centered(&zl);
        assert!(verify_partial(&bad2, &f.ac, &f.a_seed, &f.u_b, &f.subset, COMMITTEE_SIZE, THRESHOLD, &w_j).is_err());
    }


    #[test]
    fn cross_share_forgery_fails() {
        // A statement for member 1's published W, proven with member 2's
        // short witness: no single witness satisfies both row families.
        let f = fixture(0x26);
        let mut st = 0xE651_0026u64;
        let secret_a = synth_secret(1, &mut st);
        let secret_b = synth_secret(2, &mut st);
        let w_a = commit(&f.ac, &secret_a.share, &secret_a.rho);
        let lambdas = lagrange_coefficients(&f.subset, COMMITTEE_SIZE, THRESHOLD).unwrap();
        let partial = Poly::zero();
        let stmt = build_statement(
            &f.ac, &f.a_seed, 1, &f.subset, COMMITTEE_SIZE, THRESHOLD, &f.u_b, &lambdas[0], &w_a, &partial,
        );
        let mut witness = Vec::new();
        witness.extend_from_slice(secret_b.share.polys());
        witness.extend_from_slice(secret_b.rho.polys());
        witness.push(Poly::zero());
        let proof = sigma::prove(&SEAL_VPD, &stmt, &witness, &[0x99; 32]).unwrap();
        let p = PartialDecryption { member: 1, partial, proof };
        assert!(matches!(
            verify_partial(&p, &f.ac, &f.a_seed, &f.u_b, &f.subset, COMMITTEE_SIZE, THRESHOLD, &w_a),
            Err(VpdError::Sigma(SigmaError::EquationFailed { .. }))
        ));
    }


    #[test]
    fn witness_bounds_are_enforced() {
        let f = fixture(0x27);
        let mut st = 0xE651_0027u64;
        let secret = synth_secret(1, &mut st);
        let mut vals = [0i64; N];
        vals[0] = 129;
        let eps = Poly::from_centered(&vals);
        assert!(matches!(
            prove_with_eps(&secret, &f.ac, &f.a_seed, &f.u_b, &f.subset, COMMITTEE_SIZE, THRESHOLD, eps, &[1; 32]),
            Err(VpdError::Sigma(SigmaError::WitnessViolatesBound { block: 16 }))
        ));
        let mut bad_share = secret.clone();
        let mut cl = bad_share.share.poly(0).centerlift();
        cl[0] = 71;
        let mut polys = *bad_share.share.polys();
        polys[0] = Poly::from_centered(&cl);
        bad_share.share = Vec8::new(polys);
        assert!(matches!(
            prove_with_eps(&bad_share, &f.ac, &f.a_seed, &f.u_b, &f.subset, COMMITTEE_SIZE, THRESHOLD, Poly::zero(), &[1; 32]),
            Err(VpdError::Sigma(SigmaError::WitnessViolatesBound { block: 0 }))
        ));
        // Boundary ε (all ±128) proves and verifies.
        let mut vals = [0i64; N];
        for (i, v) in vals.iter_mut().enumerate() {
            *v = if i % 2 == 0 { 128 } else { -128 };
        }
        let eps = Poly::from_centered(&vals);
        let w_j = commit(&f.ac, &secret.share, &secret.rho);
        let p = prove_with_eps(&secret, &f.ac, &f.a_seed, &f.u_b, &f.subset, COMMITTEE_SIZE, THRESHOLD, eps, &[2; 32]).unwrap();
        verify_partial(&p, &f.ac, &f.a_seed, &f.u_b, &f.subset, COMMITTEE_SIZE, THRESHOLD, &w_j).unwrap();
    }


    #[test]
    fn wire_roundtrip_and_validation() {
        let f = fixture(0x28);
        let mut st = 0xE651_0028u64;
        let secret = synth_secret(5, &mut st);
        let p = prove_partial(&secret, &f.ac, &f.a_seed, &f.u_b, &f.subset, COMMITTEE_SIZE, THRESHOLD, &[6; 32]).unwrap();
        let bytes = p.to_bytes();
        assert_eq!(PartialDecryption::from_bytes(&bytes).unwrap(), p);
        assert!(matches!(
            PartialDecryption::from_bytes(&bytes[..bytes.len() - 1]),
            Err(VpdError::BadLength { .. })
        ));
        let mut bad = bytes.clone();
        let plen = u32::from_le_bytes([bad[1025], bad[1026], bad[1027], bad[1028]]) as usize;
        bad[1028] ^= 0xFF; // corrupt the declared proof length
        let plen2 = u32::from_le_bytes([bad[1025], bad[1026], bad[1027], bad[1028]]) as usize;
        if plen2 != plen {
            assert!(PartialDecryption::from_bytes(&bad).is_err());
        }
    }


    #[test]
    fn combine_sums_verified_partials_and_rejects_bad_sets() {
        let f = fixture(0x29);
        let mut st = 0xE651_0029u64;
        let mut partials = Vec::with_capacity(THRESHOLD);
        let mut w_js = Vec::with_capacity(THRESHOLD);
        for k in 0..THRESHOLD {
            let member = f.subset[k];
            let secret = synth_secret(member, &mut st);
            w_js.push(commit(&f.ac, &secret.share, &secret.rho));
            partials.push(
                prove_partial(&secret, &f.ac, &f.a_seed, &f.u_b, &f.subset, COMMITTEE_SIZE, THRESHOLD, &[0x40 + member; 32]).unwrap(),
            );
        }
        let su = combine_partials(&partials, &f.ac, &f.a_seed, &f.u_b, &f.subset, &w_js, COMMITTEE_SIZE, THRESHOLD).unwrap();
        let mut want = Poly::zero();
        for p in partials.iter() {
            want = want.add(&p.partial);
        }
        assert_eq!(su, want);


        // Tampered member partial: caught at combination.
        let mut bad = partials.clone();
        let mut cl = bad[3].partial.centerlift();
        cl[100] += 1;
        bad[3].partial = Poly::from_centered(&cl);
        assert!(combine_partials(&bad, &f.ac, &f.a_seed, &f.u_b, &f.subset, &w_js, COMMITTEE_SIZE, THRESHOLD).is_err());


        // Out-of-order members and short sets.
        let mut swapped = partials.clone();
        swapped.swap(0, 1);
        assert!(matches!(
            combine_partials(&swapped, &f.ac, &f.a_seed, &f.u_b, &f.subset, &w_js, COMMITTEE_SIZE, THRESHOLD),
            Err(VpdError::MemberMismatch { .. })
        ));
        let short = &partials[..THRESHOLD - 1];
        assert!(matches!(
            combine_partials(short, &f.ac, &f.a_seed, &f.u_b, &f.subset, &w_js, COMMITTEE_SIZE, THRESHOLD),
            Err(VpdError::BadSubset { expected: 7, .. })
        ));
    }
}
