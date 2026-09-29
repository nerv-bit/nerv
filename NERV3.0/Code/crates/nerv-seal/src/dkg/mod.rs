//! The dealerless DKG over R_q (WP §6.3.5; Appendix D.1; errata 42–46).
//!
//! Per shard, per epoch, the n-member committee (t = 7 of n = 10) jointly
//! establishes the threshold key:
//!
//! * Public matrix material is XOF-derived from the epoch's committed
//!   32-byte `ASeed`: the encryption matrix A (existing) and the Ajtai
//!   commitment pair `A_c = (A_L, A_R)` ∈ (R^{8×8})² under `nerv.seal.dkg`
//!   — commit(v, ρ) = A_L·v + A_R·ρ (MSIS binding, MLWE hiding;
//!   estimator-gated at M1).
//!
//! * Each member i holds a Shamir sharing polynomial
//!   f_i(X) = a_{i,0} + a_{i,1}X + … + a_{i,t−1}X^{t−1} over R^8 with
//!   ternary coefficients, and publishes: coefficient commitments
//!   C_{i,ℓ} = commit(a_{i,ℓ}, ρ_{i,ℓ}); fragment commitments
//!   F_{i,j} = commit(f_i(γ_j), ρ′_{i,j}) for every member j (γ_j = x^j —
//!   monomial evaluation points keep shares short, erratum 42); and the
//!   public-key contribution PK_i = a_{i,0}·A + E_{0,i} — the Ajtai-form
//!   commitment whose sum is literally the aggregate public key
//!   T = Σ_i PK_i = s·A + E₀ (WP: "the sum of share commitments").
//!
//! * One FS-with-aborts proof per member (`dkg::sigma`, domain
//!   `nerv.seal.dkg`) binds ALL published targets to a single short
//!   witness (a's, ρ's, E's) — the consistency proofs of §6.3.5. A member
//!   whose PK used a different secret than its fragments cannot prove.
//!
//! * Fragment values with their commitment randomness travel privately to
//!   each recipient (ML-KEM delivery is epoch.rs, part 3); the recipient
//!   verifies every opening against the public F_{i,j} (complaint =
//!   public fraud path) and assembles its share s_j = Σ_i f_i(γ_j) with
//!   the opening of the transcript-derived share commitment
//!   W_j = Σ_i F_{i,j} = commit(s_j, ρ̄_j) — the randomness sum ρ̄_j is
//!   known exactly to member j (erratum 44).
//!
//! * Reconstruction: Lagrange coefficients λ_j ∈ R_q over any t-subset;
//!   Σ_{j∈S} λ_j·s_j = s exactly. T − s·A is the summed committee noise
//!   (bounded by n — the budget's key contract, erratum 41).

pub mod sigma;

use nerv_core::constants::{SEAL_DKG, SEAL_NOISE};
use nerv_core::hash::{Hash256, Xof};

use crate::encrypt::PublicKey;
use crate::error::{DkgError, SealError, SigmaError};
use crate::ring::{Mat2x8, Mat8x8, Poly, Vec8};
use crate::sampling::{expand_matrix, sample_ternary_vec8, ASeed};

pub use sigma::{Proof, Statement};

/// Committee size (params).
pub const COMMITTEE_SIZE: usize = nerv_core::params::SEAL_COMMITTEE_N as usize;
/// Threshold (params).
pub const THRESHOLD: usize = nerv_core::params::SEAL_THRESHOLD_T as usize;
/// Honest share ∞-bound: n·t ternary contributions per coefficient.
pub const SHARE_BOUND: u64 = (COMMITTEE_SIZE * THRESHOLD) as u64;

const _: () = assert!(COMMITTEE_SIZE <= 255 && THRESHOLD >= 1 && THRESHOLD <= COMMITTEE_SIZE);

// ---------------------------------------------------------------------------
// The Ajtai commitment
// ---------------------------------------------------------------------------

/// The commitment matrix pair A_c = (A_L, A_R), XOF-expanded from the
/// epoch seed under `nerv.seal.dkg`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CommitMatrix {
    pub left: Mat8x8,
    pub right: Mat8x8,
}

impl CommitMatrix {
    pub fn expand(a_seed: &ASeed) -> Result<CommitMatrix, SealError> {
        let mut xof = Xof::new(&SEAL_DKG, a_seed.as_bytes());
        let expand_mat = |xof: &mut Xof| -> Result<Mat8x8, SealError> {
            let mut rows = [[Poly::zero(); 8]; 8];
            for row in rows.iter_mut() {
                for p in row.iter_mut() {
                    let mut a = [0u64; 256];
                    for c in a.iter_mut() {
                        let mut v = None;
                        for _ in 0..1024 {
                            let raw = u32::from_le_bytes(xof.read_array::<4>());
                            if u64::from(raw) < crate::ring::Q {
                                v = Some(u64::from(raw));
                                break;
                            }
                        }
                        *c = v.ok_or(SealError::SamplingExhausted { attempts: 1024 })?;
                    }
                    *p = Poly::new(a);
                }
            }
            Ok(Mat8x8::new(rows))
        };
        Ok(CommitMatrix { left: expand_mat(&mut xof)?, right: expand_mat(&mut xof)? })
    }
}

/// commit(v, ρ) = A_L·v + A_R·ρ ∈ R^8.
pub fn commit(ac: &CommitMatrix, value: &Vec8, rand: &Vec8) -> Vec8 {
    ac.left.mul_vec(value).add(&ac.right.mul_vec(rand))
}

// ---------------------------------------------------------------------------
// Member secrets and public transcripts
// ---------------------------------------------------------------------------

/// Member i's private DKG material (ternary throughout).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MemberSecret {
    pub member: u8,
    /// Shamir coefficients a_{i,ℓ}, ℓ = 0..t−1 (a[0] is the secret share
    /// contribution).
    pub a: Vec<Vec8>,
    /// Commitment randomness for the C_{i,ℓ}.
    pub rho_c: Vec<Vec8>,
    /// Fragment-commitment randomness, indexed by target member j−1.
    pub rho_f: Vec<Vec8>,
    /// The two E-rows of PK_i.
    pub e_pk: [Vec8; 2],
}

impl MemberSecret {
    /// Deterministic in (member, seed); frozen sampling order
    /// a ‖ rho_c ‖ rho_f ‖ e_pk under `nerv.seal.noise`.
    pub fn generate(member: u8, seed: &[u8; 32], n: usize, t: usize) -> Result<Self, DkgError> {
        validate_committee(n, t)?;
        if member == 0 || member as usize > n {
            return Err(DkgError::BadMember { member, n });
        }
        let mut xof = Xof::new(&SEAL_NOISE, seed);
        let sample = |xof: &mut Xof, count: usize| -> Vec<Vec8> {
            (0..count).map(|_| sample_ternary_vec8(xof)).collect()
        };
        Ok(MemberSecret {
            member,
            a: sample(&mut xof, t),
            rho_c: sample(&mut xof, t),
            rho_f: sample(&mut xof, n),
            e_pk: [sample_ternary_vec8(&mut xof), sample_ternary_vec8(&mut xof)],
        })
    }

    /// f_i(γ_j) = Σ_ℓ x^{jℓ}·a[ℓ] (rotations — coefficients stay ≤ t).
    fn eval_at(&self, j: usize) -> Vec8 {
        let mut out = Vec8::zero();
        for (l, coef) in self.a.iter().enumerate() {
            let mut rotated = [Poly::zero(); 8];
            for (c, p) in coef.polys().iter().enumerate() {
                rotated[c] = p.mul_monomial(j * l);
            }
            out = out.add(&Vec8::new(rotated));
        }
        out
    }

    /// The private payload for member j: (fragment, commitment randomness).
    /// Delivered under ML-KEM by epoch.rs; the recipient verifies it
    /// against the public F_{i,j} (`verify_fragment_opening`).
    pub fn fragment_for(&self, target: u8) -> (Vec8, Vec8) {
        (self.eval_at(target as usize), self.rho_f[target as usize - 1].clone())
    }

    /// The witness blocks in the statement's frozen order:
    /// a ‖ rho_c ‖ rho_f ‖ e_pk (128 + 8n blocks, all bound 1).
    pub fn witness_blocks(&self) -> Vec<Poly> {
        let mut out = Vec::with_capacity(128 + 8 * self.rho_f.len());
        for v in self.a.iter().chain(self.rho_c.iter()).chain(self.rho_f.iter()) {
            out.extend_from_slice(v.polys());
        }
        for row in self.e_pk.iter() {
            out.extend_from_slice(row.polys());
        }
        out
    }
}

/// Member i's public DKG transcript.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MemberPublic {
    pub member: u8,
    /// C_{i,ℓ}, ℓ = 0..t−1.
    pub commitments: Vec<Vec8>,
    /// F_{i,j}, j = 1..n (index j−1; includes the self-fragment F_{i,i}).
    pub fragments: Vec<Vec8>,
    /// PK_i's two rows: a_{i,0}·A + E.
    pub pk_rows: [Vec8; 2],
    pub proof: Proof,
}

impl MemberPublic {
    /// Builds the public transcript and proves it. Deterministic in
    /// (secret, proof_seed).
    pub fn build(
        secret: &MemberSecret,
        ac: &CommitMatrix,
        a_mat: &Mat8x8,
        a_seed: &ASeed,
        proof_seed: &[u8; 32],
    ) -> Result<MemberPublic, DkgError> {
        let n = secret.rho_f.len();
        let t = secret.a.len();
        let commitments: Vec<Vec8> = (0..t)
            .map(|l| commit(ac, &secret.a[l], &secret.rho_c[l]))
            .collect();
        let fragments: Vec<Vec8> = (1..=n)
            .map(|j| commit(ac, &secret.eval_at(j), &secret.rho_f[j - 1]))
            .collect();
        let base = a_mat.mul_vec_transpose(&secret.a[0]);
        let pk_rows =
            [base.add(&secret.e_pk[0]), base.add(&secret.e_pk[1])];
        let partial = MemberPublic {
            member: secret.member,
            commitments,
            fragments,
            pk_rows,
            proof: Proof { h: Vec::new(), z: Vec::new() },
        };
        let stmt = member_statement(&partial, ac, a_mat, a_seed)?;
        let witness = secret.witness_blocks();
        let proof = sigma::prove(&SEAL_DKG, &stmt, &witness, proof_seed)?;
        Ok(MemberPublic { proof, ..partial })
    }

    /// Full public verification: structural checks + the FS proof against
    /// the statement rebuilt from public data.
    pub fn verify(&self, a_seed: &ASeed, n: usize, t: usize) -> Result<(), DkgError> {
        validate_committee(n, t)?;
        if self.member == 0 || self.member as usize > n {
            return Err(DkgError::BadMember { member: self.member, n });
        }
        if self.commitments.len() != t || self.fragments.len() != n {
            return Err(DkgError::BadShareSet {
                len: self.commitments.len(),
                expected: t,
            });
        }
        let ac = CommitMatrix::expand(a_seed)?;
        let a_mat = expand_matrix(a_seed)?;
        let stmt = member_statement(self, &ac, &a_mat, a_seed)?;
        Ok(sigma::verify(&SEAL_DKG, &stmt, &self.proof)?)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(self.member);
        out.push(self.commitments.len() as u8);
        out.push(self.fragments.len() as u8);
        for v in self.commitments.iter().chain(self.fragments.iter()) {
            out.extend_from_slice(&v.to_bytes());
        }
        for row in self.pk_rows.iter() {
            out.extend_from_slice(&row.to_bytes());
        }
        let pb = self.proof.to_bytes();
        out.extend_from_slice(&(pb.len() as u32).to_le_bytes());
        out.extend_from_slice(&pb);
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<MemberPublic, DkgError> {
        if bytes.len() < 3 {
            return Err(SealError::BadLength { len: bytes.len(), expected: 3 }.into());
        }
        let member = bytes[0];
        let t = bytes[1] as usize;
        let n = bytes[2] as usize;
        let body = 3 + (t + n + 2) * 8192;
        if bytes.len() < body + 4 {
            return Err(SealError::BadLength { len: bytes.len(), expected: body + 4 }.into());
        }
        let take_vec = |at: usize| -> Result<Vec8, SealError> {
            let mut b = [0u8; 8192];
            b.copy_from_slice(&bytes[at..at + 8192]);
            Vec8::from_bytes(&b)
        };
        let mut at = 3;
        let mut commitments = Vec::with_capacity(t);
        for _ in 0..t {
            commitments.push(take_vec(at)?);
            at += 8192;
        }
        let mut fragments = Vec::with_capacity(n);
        for _ in 0..n {
            fragments.push(take_vec(at)?);
            at += 8192;
        }
        let mut pk_rows = [Vec8::zero(), Vec8::zero()];
        for row in pk_rows.iter_mut() {
            *row = take_vec(at)?;
            at += 8192;
        }
        let mut lb = [0u8; 4];
        lb.copy_from_slice(&bytes[at..at + 4]);
        at += 4;
        let plen = u32::from_le_bytes(lb) as usize;
        if bytes.len() != at + plen {
            return Err(SealError::BadLength { len: bytes.len(), expected: at + plen }.into());
        }
        let proof = Proof::from_bytes(&bytes[at..])?;
        Ok(MemberPublic { member, commitments, fragments, pk_rows, proof })
    }
}

/// The statement for one member's transcript: the C, F, and PK equations
/// over the shared witness (a ‖ rho_c ‖ rho_f ‖ e_pk). Context binds the
/// matrix material: ASeed ‖ member ‖ n ‖ t.
pub fn member_statement(
    mp: &MemberPublic,
    ac: &CommitMatrix,
    a_mat: &Mat8x8,
    a_seed: &ASeed,
) -> Result<Statement, SigmaError> {
    let n = mp.fragments.len();
    let t = mp.commitments.len();
    let blocks = 128 + 8 * n;
    let mut ctx = Vec::with_capacity(35);
    ctx.extend_from_slice(a_seed.as_bytes());
    ctx.push(mp.member);
    ctx.push(n as u8);
    ctx.push(t as u8);
    let mut stmt = Statement::new(vec![1; blocks], ctx);

    // C_{i,ℓ} = A_L·a_ℓ + A_R·ρ_ℓ.
    for l in 0..t {
        for r in 0..8 {
            let mut entries = Vec::with_capacity(16);
            for j in 0..8 {
                entries.push((l * 8 + j, ac.left.row(r)[j]));
            }
            for m in 0..8 {
                entries.push((56 + l * 8 + m, ac.right.row(r)[m]));
            }
            stmt.push_equation(*mp.commitments[l].poly(r), entries);
        }
    }
    // F_{i,j} = A_L·f(γ_j) + A_R·ρ′_j, with f(γ_j) = Σ_ℓ γ_j^ℓ·a_ℓ.
    for j in 1..=n {
        for r in 0..8 {
            let mut entries = Vec::with_capacity(64);
            for l in 0..t {
                for jp in 0..8 {
                    entries.push((l * 8 + jp, ac.left.row(r)[jp].mul_monomial(j * l)));
                }
            }
            for m in 0..8 {
                entries.push((112 + (j - 1) * 8 + m, ac.right.row(r)[m]));
            }
            stmt.push_equation(*mp.fragments[j - 1].poly(r), entries);
        }
    }
    // PK rows: Σ_j a_{0,j}·A[j][c] + e^{(i2)}_c.
    for i2 in 0..2 {
        for c in 0..8 {
            let mut entries = Vec::with_capacity(9);
            for j in 0..8 {
                entries.push((j, a_mat.row(j)[c]));
            }
            entries.push((112 + 8 * n + i2 * 8 + c, Poly::one()));
            stmt.push_equation(*mp.pk_rows[i2].poly(c), entries);
        }
    }
    Ok(stmt)
}

// ---------------------------------------------------------------------------
// Shares, reconstruction, public-key assembly
// ---------------------------------------------------------------------------

/// Verifies a received fragment opening against the published F.
pub fn verify_fragment_opening(
    ac: &CommitMatrix,
    published: &Vec8,
    fragment: &Vec8,
    rho: &Vec8,
) -> bool {
    commit(ac, fragment, rho) == *published
}

/// A member's assembled share secret: s_j and the opening randomness of
/// W_j = Σ_i F_{i,j}.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ShareSecret {
    pub member: u8,
    pub share: Vec8,
    pub rho: Vec8,
}

impl ShareSecret {
    /// Assembles member j's share from its own contribution and the
    /// received openings: sums fragments and randomnesses.
    pub fn assemble(
        member: u8,
        own: &(Vec8, Vec8),
        received: &[(u8, Vec8, Vec8)],
        n: usize,
        t: usize,
    ) -> Result<ShareSecret, DkgError> {
        validate_committee(n, t)?;
        if member == 0 || member as usize > n {
            return Err(DkgError::BadMember { member, n });
        }
        if received.len() != n - 1 {
            return Err(DkgError::BadShareSet { len: received.len() + 1, expected: n });
        }
        let mut share = own.0.clone();
        let mut rho = own.1.clone();
           for (from, frag, r) in received.iter() {
            let from = *from;
            if from == 0 || from as usize > n || from == member {
                return Err(DkgError::BadMember { member: from, n });
            }
            share = share.add(frag);
            rho = rho.add(r);
        }

        Ok(ShareSecret { member, share, rho })
    }
}

/// The transcript-derived share commitment W_j = Σ_i F_{i,j} (index j−1).
pub fn share_commitment(members: &[MemberPublic], j: u8, n: usize) -> Result<Vec8, DkgError> {
    if members.len() != n {
        return Err(DkgError::BadShareSet { len: members.len(), expected: n });
    }
    if j == 0 || j as usize > n {
        return Err(DkgError::BadMember { member: j, n });
    }
    let mut w = Vec8::zero();
    for m in members.iter() {
        if m.member == 0 || m.member as usize > n {
            return Err(DkgError::BadMember { member: m.member, n });
        }
        w = w.add(&m.fragments[j as usize - 1]);
    }
    Ok(w)
}

/// Checks a member's share secret against the public W_j.
pub fn check_share_commitment(ac: &CommitMatrix, w_j: &Vec8, share: &ShareSecret) -> bool {
    commit(ac, &share.share, &share.rho) == *w_j
}

/// Lagrange coefficients λ_j over a t-subset S of monomial points γ_j.
/// Self-checks the interpolation identities Σ_j λ_j·γ_j^ℓ = δ_{ℓ0}.
pub fn lagrange_coefficients(set: &[u8], n: usize, t: usize) -> Result<Vec<Poly>, DkgError> {
    validate_committee(n, t)?;
    if set.len() != t {
        return Err(DkgError::BadShareSet { len: set.len(), expected: t });
    }
    for (i, &m) in set.iter().enumerate() {
        if m == 0 || m as usize > n {
            return Err(DkgError::BadMember { member: m, n });
        }
        if set[..i].contains(&m) {
            return Err(DkgError::DuplicateMember { member: m });
        }
    }
    let mut out = Vec::with_capacity(t);
    for &j in set.iter() {
        let mut num = Poly::one();
        let mut den = Poly::one();
        for &k in set.iter() {
            if k == j {
                continue;
            }
            num = num.mul(&Poly::monomial(k as usize).neg());
            den = den.mul(&Poly::monomial(j as usize).sub(&Poly::monomial(k as usize)));
        }
        let inv = den.try_invert().ok_or(DkgError::NotInvertible { member: j })?;
        out.push(num.mul(&inv));
    }
    // Interpolation self-check (defense in depth).
    for l in 0..t {
        let mut acc = Poly::zero();
        for (idx, &j) in set.iter().enumerate() {
            acc = acc.add(&out[idx].mul(&Poly::monomial(j as usize * l)));
        }
        let want = if l == 0 { Poly::one() } else { Poly::zero() };
        if acc != want {
            return Err(DkgError::InterpolationCheckFailed);
        }
    }
    Ok(out)
}

/// Reconstructs the joint secret from exactly t shares:
/// s = Σ_{j∈S} λ_j·s_j.
pub fn reconstruct(
    set: &[u8],
    shares: &[(u8, Vec8)],
    n: usize,
    t: usize,
) -> Result<Vec8, DkgError> {
    let lambdas = lagrange_coefficients(set, n, t)?;
    if shares.len() != t {
        return Err(DkgError::BadShareSet { len: shares.len(), expected: t });
    }
    let mut s = Vec8::zero();
    for (i, &(m, ref share)) in shares.iter().enumerate() {
        if m != set[i] {
            return Err(DkgError::BadMember { member: m, n });
        }
        let lam = &lambdas[i];
        let mut scaled = [Poly::zero(); 8];
        for (c, p) in share.polys().iter().enumerate() {
            scaled[c] = lam.mul(p);
        }
        s = s.add(&Vec8::new(scaled));
    }
    Ok(s)
}

/// Assembles the epoch public key from all members' verified transcripts:
/// T = Σ_i PK_i.
pub fn assemble_public_key(members: &[MemberPublic], a_seed: &ASeed, n: usize) -> Result<PublicKey, DkgError> {
    if members.len() != n {
        return Err(DkgError::BadShareSet { len: members.len(), expected: n });
    }
    let mut row0 = Vec8::zero();
    let mut row1 = Vec8::zero();
    for m in members.iter() {
        row0 = row0.add(&m.pk_rows[0]);
        row1 = row1.add(&m.pk_rows[1]);
    }
    Ok(PublicKey::new(*a_seed, Mat2x8::new([*row0.polys(), *row1.polys()])))
}

/// The DKG transcript digest (beacon-committed under the shard's
/// params_root, WP §6.3.5).
pub fn transcript_digest(members: &[MemberPublic]) -> [u8; 32] {
    let mut msg = Vec::new();
    msg.extend_from_slice(&(members.len() as u32).to_le_bytes());
    for m in members.iter() {
        msg.extend_from_slice(&m.to_bytes());
    }
    *Hash256::concat(&SEAL_DKG, &msg).as_bytes()
}

fn validate_committee(n: usize, t: usize) -> Result<(), DkgError> {
    if n == 0 || n > 255 {
        return Err(DkgError::BadCommittee { n });
    }
    if t == 0 || t > n {
        return Err(DkgError::BadThreshold { t, n });
    }
    Ok(())
}
    
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::decrypt::decrypt_native;
    use crate::digitize::{digitize, COORDS};
    use crate::encrypt::Ciphertext;
    use crate::noise::{KEY_BOUND, SCALE};
    use crate::sampling::NoiseSeed;

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn seed_of(tag: u8, k: u64) -> [u8; 32] {
        let mut b = [0u8; 32];
        b[0] = tag;
        b[24..32].copy_from_slice(&k.to_le_bytes());
        b
    }

    struct Committee {
        a_seed: ASeed,
        ac: CommitMatrix,
        a_mat: Mat8x8,
        secrets: Vec<MemberSecret>,
        public: Vec<MemberPublic>,
    }

    fn build_committee(tag: u8) -> Committee {
        let a_seed = ASeed::from_bytes(seed_of(tag, 0));
        let ac = CommitMatrix::expand(&a_seed).unwrap();
        let a_mat = expand_matrix(&a_seed).unwrap();
        let mut secrets = Vec::new();
        let mut public = Vec::new();
        for i in 1..=COMMITTEE_SIZE as u8 {
            let secret = MemberSecret::generate(i, &seed_of(tag, u64::from(i)), COMMITTEE_SIZE, THRESHOLD).unwrap();
            let mp = MemberPublic::build(&secret, &ac, &a_mat, &a_seed, &seed_of(tag, 0x1000 + u64::from(i))).unwrap();
            secrets.push(secret);
            public.push(mp);
        }
        Committee { a_seed, ac, a_mat, secrets, public }
    }

    #[test]
    fn all_member_proofs_verify() {
        let c = build_committee(0x42);
        for mp in c.public.iter() {
            mp.verify(&c.a_seed, COMMITTEE_SIZE, THRESHOLD).unwrap();
        }
        // Determinism: rebuilding member 3 reproduces the transcript.
        let s3 = MemberSecret::generate(3, &seed_of(0x42, 3), COMMITTEE_SIZE, THRESHOLD).unwrap();
        let mp3 = MemberPublic::build(&s3, &c.ac, &c.a_mat, &c.a_seed, &seed_of(0x42, 0x1003)).unwrap();
        assert_eq!(mp3.to_bytes(), c.public[2].to_bytes());
    }

    #[test]
    fn tampered_targets_reject() {
        let c = build_committee(0x43);
        let mut bad = c.public[0].clone();
        // Flip one coefficient of a fragment commitment.
        let mut cl = bad.fragments[1].poly(0).centerlift();
        cl[0] += 1;
        let f = *bad.fragments[1].poly(0);
        let mut coeffs = *f.coefficients();
        coeffs[0] = (coeffs[0] + 1) % crate::ring::Q;
        let mut polys = *bad.fragments[1].polys();
        polys[0] = Poly::new(coeffs);
        bad.fragments[1] = Vec8::new(polys);
        assert!(bad.verify(&c.a_seed, COMMITTEE_SIZE, THRESHOLD).is_err());

        // PK built from a different secret than the fragments: recompute
        // member 1's PK rows from member 2's secret contribution.
        let mut bad2 = c.public[0].clone();
        let base = c.a_mat.mul_vec_transpose(&c.secrets[1].a[0]);
        bad2.pk_rows[0] = base.add(&c.secrets[0].e_pk[0]);
        bad2.pk_rows[1] = base.add(&c.secrets[0].e_pk[1]);
        // Re-prove with the honest witness: no single witness satisfies
        // both the F rows and these PK rows.
        let stmt = {
            let ac = &c.ac;
            let a_seed = &c.a_seed;
            let a_mat = &c.a_mat;
            use crate::dkg::member_statement;
            member_statement(&bad2, ac, a_mat, a_seed).unwrap()
        };
        let witness = c.secrets[0].witness_blocks();
        let proof = crate::dkg::sigma::prove(&SEAL_DKG, &stmt, &witness, &[9; 32]).unwrap();
        assert!(crate::dkg::sigma::verify(&SEAL_DKG, &stmt, &proof).is_err());
    }

    #[test]
    fn fragments_openings_and_share_commitments_close() {
        let c = build_committee(0x44);
        // Every (i → j) private opening matches its public F.
        for (i, si) in c.secrets.iter().enumerate() {
            for j in 1..=COMMITTEE_SIZE as u8 {
                let (frag, rho) = si.fragment_for(j);
                assert!(
                    verify_fragment_opening(&c.ac, &c.public[i].fragments[j as usize - 1], &frag, &rho),
                    "fragment {i}→{j}"
                );
            }
        }
        // Every member's assembled share opens W_j.
        for j in 1..=COMMITTEE_SIZE as u8 {
            let ji = j as usize - 1;
            let own = c.secrets[ji].fragment_for(j);
            let received: Vec<(u8, Vec8, Vec8)> = c.secrets
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != ji)
                .map(|(i, s)| {
                    let (f, r) = s.fragment_for(j);
                    (s.member, f, r)
                })
                .collect();
            let share = ShareSecret::assemble(j, &own, &received, COMMITTEE_SIZE, THRESHOLD).unwrap();
            let w_j = share_commitment(&c.public, j, COMMITTEE_SIZE).unwrap();
            assert!(check_share_commitment(&c.ac, &w_j, &share), "member {j}");
            // Share bound (erratum 42): |s_j|∞ ≤ n·t = 70.
            for p in share.share.polys() {
                for v in p.centerlift() {
                    assert!(v.unsigned_abs() <= SHARE_BOUND as u64);
                }
            }
        }
        // A wrong fragment fails the opening check (complaint evidence).
        let (mut frag, rho) = c.secrets[0].fragment_for(2);
        let mut cl = frag.poly(0).centerlift();
        cl[1] += 1;
        let mut coeffs = *frag.poly(0).coefficients();
        coeffs[1] = (coeffs[1] + 1) % crate::ring::Q;
        let mut polys = *frag.polys();
        polys[0] = Poly::new(coeffs);
        frag = Vec8::new(polys);
        assert!(!verify_fragment_opening(&c.ac, &c.public[0].fragments[1], &frag, &rho));
    }

    #[test]
    fn reconstruction_from_any_t_subset() {
        let c = build_committee(0x45);
        // Assemble all shares.
        let mut shares: Vec<(u8, Vec8)> = Vec::new();
        for j in 1..=COMMITTEE_SIZE as u8 {
            let ji = j as usize - 1;
            let own = c.secrets[ji].fragment_for(j);
            let received: Vec<(u8, Vec8, Vec8)> = c.secrets
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != ji)
                .map(|(i, s)| {
                    let (f, r) = s.fragment_for(j);
                    (s.member, f, r)
                })
                .collect();
            let share = ShareSecret::assemble(j, &own, &received, COMMITTEE_SIZE, THRESHOLD).unwrap();
            shares.push((j, share.share));
        }
        let base = reconstruct(&[1, 2, 3, 4, 5, 6, 7], &shares[..7].to_vec(), COMMITTEE_SIZE, THRESHOLD).unwrap();
        // Every t-subset (120 of them) reconstructs the same secret.
        let mut subsets: Vec<Vec<u8>> = Vec::new();
        let mut idx: Vec<usize> = (0..COMMITTEE_SIZE).collect();
        combine(&mut idx, THRESHOLD, &mut Vec::new(), &mut subsets);
        assert_eq!(subsets.len(), 120);
        for s in subsets.iter() {
            let set: Vec<u8> = s.iter().map(|&i| shares[i].0).collect();
            let subset_shares: Vec<(u8, Vec8)> = s.iter().map(|&i| shares[i].clone()).collect();
            let r = reconstruct(&set, &subset_shares, COMMITTEE_SIZE, THRESHOLD).unwrap();
            assert_eq!(r, base, "subset {set:?}");
        }
        // t−1 shares reconstruct something different (and are not the key).
        let six = reconstruct(&[1, 2, 3, 4, 5, 6], &shares[..6].to_vec(), COMMITTEE_SIZE, 6).unwrap();
        assert_ne!(six, base);
    }

    fn combine(pool: &[usize], k: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for (i, &x) in pool.iter().enumerate() {
            if cur.last().map_or(true, |&l| l < x) {
                cur.push(x);
                combine(&pool[i + 1..], k, cur, out);
                cur.pop();
            }
        }
    }

    #[test]
    fn assembled_pk_matches_reconstruction() {
        let c = build_committee(0x46);
        let shares: Vec<(u8, Vec8)> = (1..=COMMITTEE_SIZE as u8)
            .map(|j| {
                let ji = j as usize - 1;
                let own = c.secrets[ji].fragment_for(j);
                let received: Vec<(u8, Vec8, Vec8)> = c.secrets
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != ji)
                    .map(|(i, s)| {
                        let (f, r) = s.fragment_for(j);
                        (s.member, f, r)
                    })
                    .collect();
                let share = ShareSecret::assemble(j, &own, &received, COMMITTEE_SIZE, THRESHOLD).unwrap();
                (j, share.share)
            })
            .collect();
        let s = reconstruct(&[1, 2, 3, 4, 5, 6, 7], &shares[..7].to_vec(), COMMITTEE_SIZE, THRESHOLD).unwrap();
        let pk = assemble_public_key(&c.public, &c.a_seed, COMMITTEE_SIZE).unwrap();
        // T − s·A is the summed committee noise: |·|∞ ≤ n = KEY_BOUND.
        let base = c.a_mat.mul_vec_transpose(&s);
        for i2 in 0..2 {
            let e = Vec8::new(*pk.t().row(i2)).sub(&base);
            for p in e.polys() {
                for v in p.centerlift() {
                    assert!(v.unsigned_abs() <= KEY_BOUND, "T noise {v}");
                }
            }
        }
        // The transcript digest is deterministic and member-order-checked.
        let d1 = transcript_digest(&c.public);
        let mut swapped = c.public.clone();
        swapped.swap(0, 1);
        assert_ne!(d1, transcript_digest(&swapped));
    }

    #[test]
    fn end_to_end_dkg_key_encrypts_and_decrypts() {
        let c = build_committee(0x47);
        let pk = assemble_public_key(&c.public, &c.a_seed, COMMITTEE_SIZE).unwrap();
        let a_ntt = pk.expand_a().unwrap().ntt();
        let t_ntt = pk.t().ntt();
        let shares: Vec<(u8, Vec8)> = (1..=COMMITTEE_SIZE as u8)
            .map(|j| {
                let ji = j as usize - 1;
                let own = c.secrets[ji].fragment_for(j);
                let received: Vec<(u8, Vec8, Vec8)> = c.secrets
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != ji)
                    .map(|(i, s)| {
                        let (f, r) = s.fragment_for(j);
                        (s.member, f, r)
                    })
                    .collect();
                let share = ShareSecret::assemble(j, &own, &received, COMMITTEE_SIZE, THRESHOLD).unwrap();
                (j, share.share)
            })
            .collect();
        let s = reconstruct(&[1, 2, 3, 4, 5, 6, 7], &shares[..7].to_vec(), COMMITTEE_SIZE, THRESHOLD).unwrap();

        let mut st = 0xE2E0_0001u64;
        let mut agg = Ciphertext::zero();
        let mut expected = [0u64; 512];
        let mut naive = [0u64; COORDS];
        for _ in 0..128 {
            let mut coords = [0u64; COORDS];
            for v in coords.iter_mut() {
                *v = splitmix64(&mut st);
            }
            let m = digitize(&coords);
            let mut nb = [0u8; 32];
            nb[..8].copy_from_slice(&splitmix64(&mut st).to_le_bytes());
            let ns = NoiseSeed::from_bytes(nb);
            let ct = Ciphertext::encrypt_cached(&a_ntt, &t_ntt, &ns, &m).unwrap();
            agg = agg.add(&ct);
            for (e, d) in expected.iter_mut().zip(m.slot_values()) {
                *e += d;
            }
            for (nv, &x) in naive.iter_mut().zip(coords.iter()) {
                *nv = nv.wrapping_add(x);
            }
        }
        let reveal = decrypt_native(128, &s, agg.u(), agg.v()).unwrap();
        assert_eq!(*reveal.sums(), expected);
        assert_eq!(reveal.coords(), naive);
    }

    #[test]
    fn wire_roundtrip_of_member_public() {
        let c = build_committee(0x48);
        let bytes = c.public[3].to_bytes();
        let parsed = MemberPublic::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.to_bytes(), bytes);
        assert!(parsed.verify(&c.a_seed, COMMITTEE_SIZE, THRESHOLD).unwrap() == ());
        assert!(MemberPublic::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn committee_validation_errors() {
        assert!(matches!(
            MemberSecret::generate(0, &[0; 32], COMMITTEE_SIZE, THRESHOLD),
            Err(DkgError::BadMember { .. })
        ));
        assert!(matches!(
            MemberSecret::generate(1, &[0; 32], 0, THRESHOLD),
            Err(DkgError::BadCommittee { n: 0 })
        ));
        assert!(matches!(
            MemberSecret::generate(1, &[0; 32], 4, 5),
            Err(DkgError::BadThreshold { t: 5, n: 4 })
        ));
        assert!(matches!(
            lagrange_coefficients(&[1, 2, 1], COMMITTEE_SIZE, 3),
            Err(DkgError::DuplicateMember { member: 1 })
        ));
        assert!(matches!(
            lagrange_coefficients(&[1, 2, 3], COMMITTEE_SIZE, 2),
            Err(DkgError::BadShareSet { .. })
        ));
    }

    #[test]
    fn statement_shape_pins() {
        let c = build_committee(0x49);
        let stmt = member_statement(&c.public[0], &c.ac, &c.a_mat, &c.a_seed).unwrap();
        assert_eq!(stmt.blocks(), 128 + 8 * COMMITTEE_SIZE);
        assert_eq!(stmt.rows.len(), 56 + 8 * COMMITTEE_SIZE + 16);
        // Proof size (erratum 45): 8 + (r + k)·1024.
        let mp = &c.public[0];
        assert_eq!(
            mp.proof.to_bytes().len(),
            8 + (stmt.rows.len() + stmt.blocks()) * 1024
        );
    }
}


