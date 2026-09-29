//! The Lyubashevsky Fiat–Shamir-with-aborts engine (WP §6.3.3, §6.3.5) —
//! the shared primitive of both novel seal surfaces (DSR-6: patterned on
//! ML-DSA internals, in-house).
//!
//! Statement: knowledge of a short witness x ∈ R_q^k (per-block ∞-bounds
//! β_b, public) satisfying a sparse block-linear system
//!     M·x = y    (r equations; M's entries public arbitrary ring elements)
//! plus per-block norm bounds. Protocol (per attempt):
//!   1. w_b ← uniform per coefficient in [−γ, γ] (γ = 2^22; XOF-derived,
//!      modulo bias ≤ 2^−46 — stated, cryptographically void).
//!   2. h := M·w.
//!   3. c := sparse challenge (weight 32, ±1) = H(domain ‖ ctx ‖ y ‖ h).
//!   4. z := w + c·x; accept iff |z_b|∞ ≤ γ − weight·β_b for every block.
//! Accepted z is uniform on the public inner box, independent of x (the
//! standard box-rejection lemma, per coordinate); the transcript is (h, z)
//! with c recomputed by the verifier. Aborting repeats with fresh w;
//! the attempt cap keeps the prover total. Expected attempts ≈ 1.5 (DKG
//! statements, all-ternary witnesses) and ≈ 6 (VPD share blocks, part 2).
//!
//! Security (M1 dual-audit gate owns the full proofs): soundness by
//! forking — two accepting transcripts under one h give a short nonzero
//! MSIS solution for the stacked matrix (or the witness itself); zero
//! knowledge via rejection sampling + ROM programming of the challenge;
//! the FS binding requires the caller's `context` to uniquely determine
//! the matrix rows (documented contract; dkg and vpd build contexts from
//! the epoch seed and statement digest material).
//!
//! Transcript wire: (r u32, k u32) ‖ h blocks ‖ z blocks, each block
//! u32-LE — proof size = 8 + (r + k)·1024 bytes.

use nerv_core::constants::Domain;
use nerv_core::hash::Xof;
use crate::error::SigmaError;
use crate::ring::{Poly, PolyNtt, N};

pub const CHALLENGE_WEIGHT: usize = nerv_core::params::SEAL_SIGMA_CHALLENGE_WEIGHT as usize;
pub const UNIFORM_GAMMA: u64 = 1u64 << nerv_core::params::SEAL_SIGMA_UNIFORM_LOG2;
pub const ATTEMPT_CAP: u32 = nerv_core::params::SEAL_SIGMA_ATTEMPT_CAP as u32;

const _: () = assert!(CHALLENGE_WEIGHT >= 16 && CHALLENGE_WEIGHT <= 64);
const _: () = assert!(UNIFORM_GAMMA < crate::ring::Q / 2);
const _: () = assert!(ATTEMPT_CAP >= 64);

/// The per-block accept bound: γ − weight·β.
pub const fn accept_bound(beta: u64) -> u64 {
    UNIFORM_GAMMA - CHALLENGE_WEIGHT as u64 * beta
}

/// One equation: sparse entries (witness block, matrix coefficient), the
/// coefficients stored NTT-transformed.
#[derive(Clone, Debug)]
pub struct Row(pub Vec<(usize, PolyNtt)>);

/// A proof statement. Building order (equations) is caller-defined and
/// must be deterministic — the DKG pushes C-rows, F-rows, PK-rows in the
/// frozen order; the VPD (part 2) pushes commitment rows then the partial
/// equation.
#[derive(Clone, Debug)]
pub struct Statement {
    pub rows: Vec<Row>,
    pub targets: Vec<Poly>,
    pub bounds: Vec<u64>,
    pub context: Vec<u8>,
}

impl Statement {
    pub fn new(bounds: Vec<u64>, context: Vec<u8>) -> Statement {
        Statement { rows: Vec::new(), targets: Vec::new(), bounds, context }
    }

    pub fn blocks(&self) -> usize {
        self.bounds.len()
    }

    pub fn push_equation(&mut self, target: Poly, entries: Vec<(usize, Poly)>) {
        self.rows.push(Row(entries.into_iter().map(|(b, p)| (b, p.ntt())).collect()));
        self.targets.push(target);
    }

    /// M·x in the coefficient domain.
    fn apply(&self, x_ntt: &[PolyNtt]) -> Vec<Poly> {
        let mut out = Vec::with_capacity(self.rows.len());
        for row in self.rows.iter() {
            let mut acc = PolyNtt::zero();
            for &(b, ref entry) in row.0.iter() {
                acc = acc.add(&entry.mul_pointwise(&x_ntt[b]));
            }
            out.push(acc.intt());
        }
        out
    }

    fn validate(&self) -> Result<(), SigmaError> {
        if self.rows.len() != self.targets.len() {
            return Err(SigmaError::StructuralMismatch {
                h_len: self.rows.len(),
                z_len: self.targets.len(),
                rows: self.rows.len(),
                blocks: self.bounds.len(),
            });
        }
        for (b, &beta) in self.bounds.iter().enumerate() {
            if beta >= UNIFORM_GAMMA / CHALLENGE_WEIGHT as u64 {
                return Err(SigmaError::InfeasibleBound { block: b, bound: beta });
            }
        }
        for (r, row) in self.rows.iter().enumerate() {
            if row.0.is_empty() {
                return Err(SigmaError::EmptyRow { row: r });
            }
            let mut seen = std::collections::BTreeSet::new();
            for &(b, _) in row.0.iter() {
                if b >= self.bounds.len() {
                    return Err(SigmaError::EntryOutOfRange {
                        row: r,
                        block: b,
                        blocks: self.bounds.len(),
                    });
                }
                if !seen.insert(b) {
                    return Err(SigmaError::DuplicateEntry { row: r, block: b });
                }
            }
        }
        Ok(())
    }
}

/// A non-interactive proof: commitment h ∈ R^r and response z ∈ R^k; the
/// challenge is recomputed from the transcript.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Proof {
    pub h: Vec<Poly>,
    pub z: Vec<Poly>,
}

impl Proof {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + (self.h.len() + self.z.len()) * 1024);
        out.extend_from_slice(&(self.h.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.z.len() as u32).to_le_bytes());
        for p in self.h.iter().chain(self.z.iter()) {
            out.extend_from_slice(&p.to_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Proof, SigmaError> {
        if bytes.len() < 8 {
            return Err(SigmaError::BadLength { len: bytes.len(), expected: 8 });
        }
        let mut b4 = [0u8; 4];
        b4.copy_from_slice(&bytes[0..4]);
        let r = u32::from_le_bytes(b4) as usize;
        b4.copy_from_slice(&bytes[4..8]);
        let k = u32::from_le_bytes(b4) as usize;
        let expected = 8 + (r + k) * 1024;
        if bytes.len() != expected {
            return Err(SigmaError::BadLength { len: bytes.len(), expected });
        }
        let mut h = Vec::with_capacity(r);
        let mut z = Vec::with_capacity(k);
        // h blocks first, then z — indices computed explicitly.
        for i in 0..r {
            let at = 8 + i * 1024;
            let mut pb = [0u8; 1024];
            pb.copy_from_slice(&bytes[at..at + 1024]);
            h.push(Poly::from_bytes(&pb)?);
        }
        for i in 0..k {
            let at = 8 + (r + i) * 1024;
            let mut pb = [0u8; 1024];
            pb.copy_from_slice(&bytes[at..at + 1024]);
            z.push(Poly::from_bytes(&pb)?);
        }
        Ok(Proof { h, z })
    }
}

fn inf_norm(p: &Poly) -> u64 {
    p.centerlift().iter().map(|v| v.unsigned_abs()).max().unwrap_or(0)
}

fn sample_uniform_box(xof: &mut Xof) -> Poly {
    let modulus = 2 * UNIFORM_GAMMA + 1;
    let mut vals = [0i64; N];
    for v in vals.iter_mut() {
        let d = xof.next_u64();
        *v = (d % modulus) as i64 - UNIFORM_GAMMA as i64;
    }
    Poly::from_centered(&vals)
}

/// Sparse challenge: weight-32 ±1 coefficients at distinct positions from
/// the domain-keyed XOF over (context ‖ targets ‖ h). Total: if 4096
/// draws do not fill the weight (probability < 2^−2000), remaining
/// positions fill ascending — deterministic, documented.
fn challenge(domain: &Domain, context: &[u8], targets: &[Poly], h: &[Poly]) -> Poly {
    let mut msg = Vec::with_capacity(4 + context.len() + (targets.len() + h.len()) * 1024);
    msg.extend_from_slice(&(context.len() as u32).to_le_bytes());
    msg.extend_from_slice(context);
    for p in targets.iter().chain(h.iter()) {
        msg.extend_from_slice(&p.to_bytes());
    }
    let mut xof = Xof::new(domain, &msg);
    let mut signs = [0i64; N];
    let mut picked = 0usize;
    let mut draws = 0u32;
    while picked < CHALLENGE_WEIGHT && draws < 4096 {
        let d = xof.next_u64();
        draws += 1;
        let pos = (d % N as u64) as usize;
        if signs[pos] == 0 {
            signs[pos] = if (d >> 63) & 1 == 0 { 1 } else { -1 };
            picked += 1;
        }
    }
    if picked < CHALLENGE_WEIGHT {
        for pos in 0..signs.len() {
            if picked >= CHALLENGE_WEIGHT {
                break;
            }
            if signs[pos] == 0 {
                signs[pos] = 1;
                picked += 1;
            }
        }
    }
    Poly::from_centered(&signs)
}

/// Proves knowledge of a short `witness` for `stmt` under `domain`. The
/// prover's randomness is XOF-derived from `seed` — deterministic for
/// conformance vectors; fresh seeds per proof in production.
pub fn prove(
    domain: &Domain,
    stmt: &Statement,
    witness: &[Poly],
    seed: &[u8; 32],
) -> Result<Proof, SigmaError> {
    stmt.validate()?;
    if witness.len() != stmt.bounds.len() {
        return Err(SigmaError::StructuralMismatch {
            h_len: 0,
            z_len: witness.len(),
            rows: stmt.rows.len(),
            blocks: stmt.bounds.len(),
        });
    }
    for (b, x) in witness.iter().enumerate() {
        if inf_norm(x) > stmt.bounds[b] {
            return Err(SigmaError::WitnessViolatesBound { block: b });
        }
    }
    let mut xof = Xof::new(domain, seed);
    for _ in 0..ATTEMPT_CAP {
        let w: Vec<Poly> = (0..stmt.blocks()).map(|_| sample_uniform_box(&mut xof)).collect();
        let w_ntt: Vec<PolyNtt> = w.iter().map(|p| p.ntt()).collect();
        let h = stmt.apply(&w_ntt);
        let c = challenge(domain, &stmt.context, &stmt.targets, &h);
        let mut z = Vec::with_capacity(stmt.blocks());
        let mut ok = true;
        for (b, x_b) in witness.iter().enumerate() {
            let z_b = w[b].add(&c.mul(x_b));
            if inf_norm(&z_b) > accept_bound(stmt.bounds[b]) {
                ok = false;
                break;
            }
            z.push(z_b);
        }
        if ok {
            return Ok(Proof { h, z });
        }
    }
    Err(SigmaError::GenExhausted { attempts: ATTEMPT_CAP as u64 })
}

/// Verifies a proof: recomputes the challenge, checks M·z = h + c·y and
/// every response block's norm against its public accept bound.
pub fn verify(domain: &Domain, stmt: &Statement, proof: &Proof) -> Result<(), SigmaError> {
    stmt.validate()?;
    if proof.h.len() != stmt.rows.len() || proof.z.len() != stmt.bounds.len() {
        return Err(SigmaError::StructuralMismatch {
            h_len: proof.h.len(),
            z_len: proof.z.len(),
            rows: stmt.rows.len(),
            blocks: stmt.bounds.len(),
        });
    }
    let c = challenge(domain, &stmt.context, &stmt.targets, &proof.h);
    let z_ntt: Vec<PolyNtt> = proof.z.iter().map(|p| p.ntt()).collect();
    let lhs = stmt.apply(&z_ntt);
    for (r, lhs_r) in lhs.iter().enumerate() {
        let rhs = proof.h[r].add(&c.mul(&stmt.targets[r]));
        if *lhs_r != rhs {
            return Err(SigmaError::EquationFailed { row: r });
        }
    }
    for (b, z_b) in proof.z.iter().enumerate() {
        let bound = accept_bound(stmt.bounds[b]);
        let norm = inf_norm(z_b);
        if norm > bound {
            return Err(SigmaError::NormExceeded { block: b, value: norm, bound });
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use nerv_core::constants::{SEAL_DKG, SEAL_VPD};
    use proptest::prelude::*;

    fn const_poly(v: u64) -> Poly {
        let mut a = [0u64; 256];
        a[0] = v;
        Poly::new(a)
    }

    fn ternary_vec(len: usize, xof: &mut Xof) -> Vec<Poly> {
        (0..len).map(|_| {
            let mut vals = [0i64; 256];
            for v in vals.iter_mut() {
                *v = (xof.next_u64() % 3) as i64 - 1;
            }
            Poly::from_centered(&vals)
        })
        .collect()
    }

    /// 3 blocks (ternary), 2 equations: x0 + x1 = y0; c1·x1 + x2 = y1.
    fn toy_statement(witness: &[Poly]) -> (Statement, [u8; 32]) {
        let mut stmt = Statement::new(vec![1; witness.len()], b"sigma-toy".to_vec());
        let y0 = witness[0].add(&witness[1]);
        stmt.push_equation(y0, vec![(0, Poly::one()), (1, Poly::one())]);
        let c1 = const_poly(1_234_567);
        let mut y1 = witness[1].mul(&c1);
        y1 = y1.add(&witness[2]);
        stmt.push_equation(y1, vec![(1, c1), (2, Poly::one())]);
        (stmt, [0x5A; 32])
    }

    fn toy_witness() -> Vec<Poly> {
        let mut xof = Xof::new(&SEAL_DKG, b"toy witness");
        ternary_vec(3, &mut xof)
    }

    #[test]
    fn completeness_determinism_and_wire() {
        let witness = toy_witness();
        let (stmt, seed) = toy_statement(&witness);
        let p1 = prove(&SEAL_DKG, &stmt, &witness, &seed).unwrap();
        let p2 = prove(&SEAL_DKG, &stmt, &witness, &seed).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(p1.to_bytes(), p2.to_bytes());
        verify(&SEAL_DKG, &stmt, &p1).unwrap();
        let parsed = Proof::from_bytes(&p1.to_bytes()).unwrap();
        assert_eq!(parsed, p1);
        assert_eq!(p1.to_bytes().len(), 8 + (2 + 3) * 1024);
        assert!(Proof::from_bytes(&p1.to_bytes()[..10]).is_err());
    }

    #[test]
    fn tampering_fails() {
        let witness = toy_witness();
        let (stmt, seed) = toy_statement(&witness);
        let proof = prove(&SEAL_DKG, &stmt, &witness, &seed).unwrap();
        verify(&SEAL_DKG, &stmt, &proof).unwrap();

        // Tampered response.
        let mut bad = proof.clone();
        let mut cl = bad.z[0].centerlift();
        cl[0] += 1;
        bad.z[0] = Poly::from_centered(&cl);
        assert!(matches!(
            verify(&SEAL_DKG, &stmt, &bad),
            Err(SigmaError::EquationFailed { row: 0 }) | Err(SigmaError::NormExceeded { .. })
        ));

        // Tampered commitment: the challenge changes → some equation fails.
        let mut bad = proof.clone();
        let mut cl = bad.h[1].centerlift();
        cl[7] += 1;
        bad.h[1] = Poly::from_centered(&cl);
        assert!(verify(&SEAL_DKG, &stmt, &bad).is_err());

        // Wrong witness: a proof for a different witness fails against the
        // original statement's targets.
        let other = {
            let mut xof = Xof::new(&SEAL_DKG, b"other witness");
            ternary_vec(3, &mut xof)
        };
        let (stmt2, _) = toy_statement(&other);
        let p2 = prove(&SEAL_DKG, &stmt2, &other, &[0x11; 32]).unwrap();
        verify(&SEAL_DKG, &stmt2, &p2).unwrap();
        assert!(matches!(
            verify(&SEAL_DKG, &stmt, &p2),
            Err(SigmaError::EquationFailed { .. })
        ));

        // Structural mismatch.
        let mut short = proof.clone();
        short.z.pop();
        assert!(matches!(
            verify(&SEAL_DKG, &stmt, &short),
            Err(SigmaError::StructuralMismatch { .. })
        ));
    }

    #[test]
    fn domain_separation_is_binding() {
        let witness = toy_witness();
        let (stmt, seed) = toy_statement(&witness);
        let proof = prove(&SEAL_DKG, &stmt, &witness, &seed).unwrap();
        verify(&SEAL_DKG, &stmt, &proof).unwrap();
        // The same statement under a different FS domain: the challenge
        // differs, so the DKG proof does not verify as a VPD proof.
        assert!(matches!(
            verify(&SEAL_VPD, &stmt, &proof),
            Err(SigmaError::EquationFailed { .. })
        ));
    }

    #[test]
    fn witness_bound_is_enforced_on_the_prover() {
        let mut witness = toy_witness();
        let (stmt, seed) = toy_statement(&witness);
        // Overstating a bound makes the statement infeasible.
        let mut loose = Statement::new(vec![UNIFORM_GAMMA; 3], b"loose".to_vec());
        assert!(matches!(
            prove(&SEAL_DKG, &loose, &witness, &seed),
            Err(SigmaError::InfeasibleBound { .. })
        ));
        // A witness violating its bound is rejected.
        let mut cl = witness[0].centerlift();
        cl[0] = 2;
        witness[0] = Poly::from_centered(&cl);
        assert!(matches!(
            prove(&SEAL_DKG, &stmt, &witness, &seed),
            Err(SigmaError::WitnessViolatesBound { block: 0 })
        ));
    }

    #[test]
    fn challenge_is_sparse_weighted_and_deterministic() {
        let targets = vec![const_poly(5), const_poly(7)];
        let h = vec![const_poly(9), const_poly(11)];
        let c1 = challenge(&SEAL_DKG, b"ctx", &targets, &h);
        let c2 = challenge(&SEAL_DKG, b"ctx", &targets, &h);
        assert_eq!(c1, c2);
        assert_ne!(c1, challenge(&SEAL_DKG, b"ctx2", &targets, &h));
        let nz: Vec<i64> = c1.centerlift().iter().copied().filter(|&v| v != 0).collect();
        assert_eq!(nz.len(), CHALLENGE_WEIGHT);
        assert!(nz.iter().all(|&v| v.abs() == 1));
        let c3 = challenge(&SEAL_VPD, b"ctx", &targets, &h);
        assert_ne!(c1, c3);
    }

    proptest! {
        #[test]
        fn prop_roundtrip(len in 2usize..=4, seed in any::<u64>()) {
            let mut b = [0u8; 32];
            b[..8].copy_from_slice(&seed.to_le_bytes());
            let mut xof = Xof::new(&SEAL_DKG, &b);
            let witness = ternary_vec(len, &mut xof);
            let mut stmt = Statement::new(vec![1; len], b"prop".to_vec());
            // One dense equation: Σ c_b·x_b = y.
            let coeffs: Vec<Poly> = (0..len)
                .map(|_| {
                    let mut a = [0u64; 256];
                    for c in a.iter_mut() { *c = xof.next_u64() % crate::ring::Q; }
                    Poly::new(a)
                })
                .collect();
            let mut y = Poly::zero();
            for (w, x) in witness.iter().zip(coeffs.iter()) {
                y = y.add(&w.mul(x));
            }
            stmt.push_equation(y, coeffs.iter().enumerate().map(|(b, c)| (b, *c)).collect());
            let proof = prove(&SEAL_DKG, &stmt, &witness, &b).unwrap();
            prop_assert!(verify(&SEAL_DKG, &stmt, &proof).is_ok());
            prop_assert_eq!(Proof::from_bytes(&proof.to_bytes()).unwrap(), proof);
        }
    }
}
