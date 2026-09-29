
//! The ballot (WP §12.8; erratum 178): the shielded note-holder ballot
//! (voting nullifier + Ajtai weight commitment + sigma proof) and the
//! transparent validator vote.


use nerv_core::constants::{BALLOT_MATRIX, BALLOT_NULL, BALLOT_PROOF};
use nerv_core::hash::Hash256;
use nerv_core::types::Epoch;
use nerv_crypto::mldsa::{Signature, SigningKey, VerifyingKey};
use nerv_seal::dkg::sigma::{self, Proof, Statement};
use nerv_seal::dkg::CommitMatrix;
use nerv_seal::ring::{Poly, Vec8, N};
use nerv_seal::sampling::ASeed;


/// The per-limb bit width (erratum 178): 15 bits × 4 limbs = 60 bits.
pub const LIMB_BITS: u32 = 15;
pub const LIMB_BOUND: u64 = 1 << 15;
pub const LIMBS: usize = 4;
/// The blinding coefficient bound.
pub const BLINDING_BOUND: u64 = 1 << 10;
/// The maximum encodable weight: 2^60 − 1 nano-NERV.
pub const MAX_WEIGHT: u64 = (1u64 << (LIMB_BITS * LIMBS as u32)) - 1;


const _: () = assert!(LIMB_BOUND < 1 << 17, "the limb bound must be < 2^17 (sigma feasibility)");
const _: () = assert!(BLINDING_BOUND < 1 << 17);
const _: () = assert!(LIMBS <= 8);


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Choice {
    Yes,
    No,
    Abstain,
}


impl Choice {
    pub fn as_u8(self) -> u8 {
        match self {
            Choice::Yes => 0,
            Choice::No => 1,
            Choice::Abstain => 2,
        }
    }


    pub fn from_u8(b: u8) -> Option<Choice> {
        match b {
            0 => Some(Choice::Yes),
            1 => Some(Choice::No),
            2 => Some(Choice::Abstain),
            _ => None,
        }
    }
}


/// The one-time voting nullifier: H("nerv.ballot.null" ‖ nk ‖ referendum).
pub fn derive_voting_nullifier(nk: &[u8; 32], referendum_id: &Hash256) -> Hash256 {
    let mut msg = Vec::with_capacity(64);
    msg.extend_from_slice(nk);
    msg.extend_from_slice(referendum_id.as_bytes());
    Hash256::concat(&BALLOT_NULL, &msg)
}


/// The per-referendum Ajtai commitment matrix (erratum 178):
/// seeded from H("nerv.ballot.matrix" ‖ referendum_id).
pub fn ballot_matrix(referendum_id: &Hash256) -> Result<CommitMatrix, nerv_seal::SealError> {
    let seed = Hash256::concat(&BALLOT_MATRIX, referendum_id.as_bytes());
    CommitMatrix::expand(&ASeed::from_bytes(*seed.as_bytes()))
}


fn weight_limbs(weight: u64) -> [u64; LIMBS] {
    let mask = LIMB_BOUND - 1;
    [
        weight & mask,
        (weight >> LIMB_BITS) & mask,
        (weight >> (2 * LIMB_BITS)) & mask,
        (weight >> (3 * LIMB_BITS)) & mask,
    ]
}


fn limbs_to_weight(limbs: &[u64; LIMBS]) -> u64 {
    limbs[0]
        | (limbs[1] << LIMB_BITS)
        | (limbs[2] << (2 * LIMB_BITS))
        | (limbs[3] << (3 * LIMB_BITS))
}


fn limb_poly(v: u64) -> Poly {
    let mut coeffs = [0u64; N];
    coeffs[0] = v;
    Poly::new(coeffs)
}


fn zero_poly() -> Poly {
    Poly::zero()
}


/// The witness value vector: [limb_0, …, limb_3, 0, 0, 0, 0].
fn weight_value_vec(weight: u64) -> Vec8 {
    let limbs = weight_limbs(weight);
    let polys = [
        limb_poly(limbs[0]),
        limb_poly(limbs[1]),
        limb_poly(limbs[2]),
        limb_poly(limbs[3]),
        zero_poly(),
        zero_poly(),
        zero_poly(),
        zero_poly(),
    ];
    Vec8::new(polys)
}


/// The Ajtai weight commitment: commit(value, blinding).
pub fn commit_weight(
    weight: u64,
    blinding: &Vec8,
    matrix: &CommitMatrix,
) -> Vec8 {
    nerv_seal::dkg::commit(matrix, &weight_value_vec(weight), blinding)
}


/// Build the sigma statement for the weight proof (erratum 178).
fn weight_statement(
    commitment: &Vec8,
    matrix: &CommitMatrix,
    referendum_id: &Hash256,
) -> Statement {
    // 16 witness blocks: 8 value + 8 blinding.
    let mut bounds = vec![LIMB_BOUND; LIMBS];
    bounds.extend(vec![1u64; 8 - LIMBS]); // the zero positions
    bounds.extend(vec![BLINDING_BOUND; 8]);


    let mut ctx = Vec::with_capacity(32);
    ctx.extend_from_slice(referendum_id.as_bytes());


    let mut stmt = Statement::new(bounds, ctx);
    // 8 equations: one per output polynomial.
    for i in 0..8 {
        let mut entries = Vec::with_capacity(16);
        for j in 0..8 {
            entries.push((j, matrix.left.row(i)[j].clone()));
        }
        for j in 0..8 {
            entries.push((8 + j, matrix.right.row(i)[j].clone()));
        }
        stmt.push_equation(commitment.poly(i).clone(), entries);
    }
    stmt
}


/// The full witness: value ‖ blinding (16 polynomials).
fn weight_witness(weight: u64, blinding: &Vec8) -> Vec<Poly> {
    let value = weight_value_vec(weight);
    let mut w: Vec<Poly> = value.polys().to_vec();
    w.extend(blinding.polys().iter());
    w
}


/// Prove the weight commitment (erratum 178): knowledge of the opening
/// with the limb and blinding bounds.
pub fn prove_weight(
    weight: u64,
    blinding: &Vec8,
    commitment: &Vec8,
    matrix: &CommitMatrix,
    referendum_id: &Hash256,
    proof_seed: &[u8; 32],
) -> Result<Proof, nerv_seal::SigmaError> {
    let stmt = weight_statement(commitment, matrix, referendum_id);
    let witness = weight_witness(weight, blinding);
    sigma::prove(&BALLOT_PROOF, &stmt, &witness, proof_seed)
}


/// Verify the weight proof against the commitment.
pub fn verify_weight(
    commitment: &Vec8,
    matrix: &CommitMatrix,
    referendum_id: &Hash256,
    proof: &Proof,
) -> bool {
    let stmt = weight_statement(commitment, matrix, referendum_id);
    sigma::verify(&BALLOT_PROOF, &stmt, proof).is_ok()
}


/// The tally-phase weight opening: the limbs and blinding the voter
/// reveals for the aggregate computation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightOpening {
    pub limbs: [u64; LIMBS],
    pub blinding: Vec8,
}


impl WeightOpening {
    pub fn new(weight: u64, blinding: Vec8) -> WeightOpening {
        WeightOpening { limbs: weight_limbs(weight), blinding }
    }


    pub fn weight(&self) -> u64 {
        limbs_to_weight(&self.limbs)
    }


    /// Verify the opening against the commitment.
    pub fn verify(&self, commitment: &Vec8, matrix: &CommitMatrix) -> bool {
        let value = Vec8::new([
            limb_poly(self.limbs[0]),
            limb_poly(self.limbs[1]),
            limb_poly(self.limbs[2]),
            limb_poly(self.limbs[3]),
            zero_poly(),
            zero_poly(),
            zero_poly(),
            zero_poly(),
        ]);
        nerv_seal::dkg::commit(matrix, &value, &self.blinding) == *commitment
    }
}


/// Convenience: open a commitment given the weight and blinding directly.
pub fn open_weight(
    weight: u64,
    blinding: &Vec8,
    commitment: &Vec8,
    matrix: &CommitMatrix,
) -> bool {
    WeightOpening::new(weight, blinding.clone()).verify(commitment, matrix)
}


/// The shielded ballot for the note-holder chamber (erratum 178).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShieldedBallot {
    pub referendum_id: Hash256,
    pub choice: Choice,
    pub nullifier: Hash256,
    pub weight_commitment: Vec8,
    pub proof: Proof,
}


impl ShieldedBallot {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nk: &[u8; 32],
        referendum_id: Hash256,
        choice: Choice,
        weight: u64,
        blinding: Vec8,
        matrix: &CommitMatrix,
        proof_seed: &[u8; 32],
    ) -> Result<ShieldedBallot, crate::error::BallotError> {
        if weight > MAX_WEIGHT {
            return Err(crate::error::BallotError::WeightTooLarge {
                weight,
                max: MAX_WEIGHT,
            });
        }
        let nullifier = derive_voting_nullifier(nk, &referendum_id);
        let commitment = commit_weight(weight, &blinding, matrix);
        let proof = prove_weight(
            weight, &blinding, &commitment, matrix, &referendum_id, proof_seed,
        )?;
        Ok(ShieldedBallot {
            referendum_id,
            choice,
            nullifier,
            weight_commitment: commitment,
            proof,
        })
    }


    pub fn verify(&self, matrix: &CommitMatrix) -> bool {
        verify_weight(&self.weight_commitment, matrix, &self.referendum_id, &self.proof)
    }
}


/// The transparent validator vote.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatorVote {
    pub referendum_id: Hash256,
    pub vk: VerifyingKey,
    pub choice: Choice,
    pub signature: Signature,
}


impl ValidatorVote {
    fn message(referendum_id: &Hash256, choice: Choice) -> Vec<u8> {
        let mut m = Vec::with_capacity(33);
        m.extend_from_slice(referendum_id.as_bytes());
        m.push(choice.as_u8());
        m
    }


    pub fn new(
        sk: &SigningKey,
        referendum_id: Hash256,
        choice: Choice,
    ) -> Result<ValidatorVote, nerv_crypto::CryptoError> {
        let signature = sk.sign(&Self::message(&referendum_id, choice))?;
        Ok(ValidatorVote {
            referendum_id,
            vk: *sk.verifying_key(),
            choice,
            signature,
        })
    }


    pub fn verify(&self) -> bool {
        self.vk.verify(&Self::message(&self.referendum_id, self.choice), &self.signature)
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use nerv_seal::sampling::sample_short_vec8;
    use nerv_core::hash::Xof;


    fn referendum(seed: u64) -> Hash256 {
        Hash256::from_bytes([seed as u8; 32])
    }


    fn nk(seed: u64) -> [u8; 32] {
        let mut b = [0u8; 32];
        b[..8].copy_from_slice(&seed.to_le_bytes());
        b
    }


    fn signing_key(seed: u64) -> SigningKey {
        SigningKey::from_seed(&nk(seed)).unwrap()
    }


    fn blinding(seed: u64) -> Vec8 {
        let mut xof = Xof::new(&BALLOT_PROOF, &seed.to_le_bytes());
        let mut polys = [Poly::zero(); 8];
        for p in &mut polys {
            let mut vals = [0i64; N];
            for v in vals.iter_mut() {
                let x = xof.next_u64() % (2 * BLINDING_BOUND + 1);
                *v = x as i64 - BLINDING_BOUND as i64;
            }
            *p = Poly::from_centered(&vals);
        }
        Vec8::new(polys)
    }


    #[test]
    fn pins() {
        assert_eq!(LIMB_BITS, 15);
        assert_eq!(LIMBS, 4);
        assert_eq!(LIMB_BOUND, 32_768);
        assert_eq!(BLINDING_BOUND, 1_024);
        assert_eq!(MAX_WEIGHT, (1u64 << 60) - 1);
    }


    #[test]
    fn nullifier_is_domain_separated_and_pinned() {
        let r = referendum(7);
        let k = nk(42);
        let nf = derive_voting_nullifier(&k, &r);
        assert_eq!(nf, derive_voting_nullifier(&k, &r));
        assert_ne!(nf, derive_voting_nullifier(&nk(43), &r));
        assert_ne!(nf, derive_voting_nullifier(&k, &referendum(8)));
        let mut msg = Vec::new();
        msg.extend_from_slice(BALLOT_NULL.as_bytes());
        msg.extend_from_slice(&k);
        msg.extend_from_slice(r.as_bytes());
        assert_eq!(nf.as_bytes(), blake3::hash(&msg).as_bytes());
    }


    #[test]
    fn weight_limbs_roundtrip() {
        for w in [0u64, 1, 100, 32_767, 32_768, 32_769, 1_000_000, MAX_WEIGHT] {
            let limbs = weight_limbs(w);
            assert_eq!(limbs_to_weight(&limbs), w);
            assert!(limbs.iter().all(|&l| l < LIMB_BOUND));
        }
    }


    #[test]
    fn ballot_matrix_is_deterministic_and_referendum_seeded() {
        let r = referendum(9);
        let m1 = ballot_matrix(&r).unwrap();
        let m2 = ballot_matrix(&r).unwrap();
        assert_eq!(m1.left.rows()[0][0], m2.left.rows()[0][0]);
        let other = ballot_matrix(&referendum(10)).unwrap();
        assert_ne!(m1.left.rows()[0][0], other.left.rows()[0][0]);
    }


    #[test]
    fn ballot_lifecycle() {
        let r = referendum(11);
        let matrix = ballot_matrix(&r).unwrap();
        let voter_nk = nk(1);
        let blind = blinding(2);
        let proof_seed = [3u8; 32];


        let b = ShieldedBallot::new(
            &voter_nk, r, Choice::Yes, 50_000_000_000, blind.clone(),
            &matrix, &proof_seed,
        )
        .unwrap();
        assert_eq!(b.nullifier, derive_voting_nullifier(&voter_nk, &r));
        assert_eq!(b.choice, Choice::Yes);
        assert!(b.verify(&matrix));


        // Determinism.
        let b2 = ShieldedBallot::new(
            &voter_nk, r, Choice::Yes, 50_000_000_000, blind.clone(),
            &matrix, &proof_seed,
        )
        .unwrap();
        assert_eq!(b, b2);


        // A different voter or choice gives a different ballot.
        let other = ShieldedBallot::new(
            &nk(99), r, Choice::No, 50_000_000_000, blind.clone(),
            &matrix, &proof_seed,
        )
        .unwrap();
        assert_ne!(b.nullifier, other.nullifier);
        assert_ne!(b.choice, other.choice);


        // Weight overflow.
        assert!(matches!(
            ShieldedBallot::new(&voter_nk, r, Choice::Yes, MAX_WEIGHT + 1, blind, &matrix, &proof_seed),
            Err(crate::error::BallotError::WeightTooLarge { .. })
        ));
    }


    #[test]
    fn weight_opening_verifies() {
        let r = referendum(12);
        let matrix = ballot_matrix(&r).unwrap();
        let blind = blinding(5);
        let weight = 123_456_789_012;
        let commitment = commit_weight(weight, &blind, &matrix);
        let opening = WeightOpening::new(weight, blind.clone());
        assert!(opening.verify(&commitment, &matrix));
        assert_eq!(opening.weight(), weight);
        assert!(open_weight(weight, &blind, &commitment, &matrix));


        // Wrong weight.
        assert!(!open_weight(weight + 1, &blind, &commitment, &matrix));
        // Wrong blinding.
        let wrong_blind = blinding(99);
        assert!(!open_weight(weight, &wrong_blind, &commitment, &matrix));
        // Wrong commitment.
        let other_commit = commit_weight(weight + 1, &blind, &matrix);
        assert!(!opening.verify(&other_commit, &matrix));
    }


    #[test]
    fn tampered_proof_rejected() {
        let r = referendum(13);
        let matrix = ballot_matrix(&r).unwrap();
        let blind = blinding(6);
        let b = ShieldedBallot::new(
            &nk(7), r, Choice::Abstain, 1_000_000, blind, &matrix, &[7u8; 32],
        )
        .unwrap();
        assert!(b.verify(&matrix));


          // Tamper the commitment h: changes the challenge → the equations fail.
        let mut bad = b.clone();

        if !bad.proof.h.is_empty() {
            let cl = bad.proof.h[0].centerlift();
            let mut cl2 = cl;
            cl2[0] += 1;
            bad.proof.h[0] = Poly::from_centered(&cl2);
            assert!(!bad.verify(&matrix), "tampered commitment");
        }
    }


    #[test]
    fn validator_vote() {
        let sk = signing_key(20);
        let r = referendum(14);
        let v = ValidatorVote::new(&sk, r, Choice::Yes).unwrap();
        assert!(v.verify());
        assert_eq!(v.choice, Choice::Yes);


        let mut bad = v.clone();
        bad.choice = Choice::No;
        assert!(!bad.verify(), "the signature binds the choice");


        let other = ValidatorVote::new(&signing_key(21), r, Choice::No).unwrap();
        assert!(other.verify());
        assert_ne!(v.signature, other.signature);
    }


    #[test]
    fn multiple_weights_proof_and_open() {
        let r = referendum(15);
        let matrix = ballot_matrix(&r).unwrap();
        for (i, w) in [0u64, 1, 100, 32_767, 32_768, 1_000_000_000, MAX_WEIGHT]
            .iter()
            .enumerate()
        {
            let blind = blinding(100 + i as u64);
            let b = ShieldedBallot::new(
                &nk(200 + i as u64), r, Choice::Yes, *w, blind.clone(),
                &matrix, &[i as u8; 32],
            )
            .unwrap();
            assert!(b.verify(&matrix), "weight {w}");
            let opening = WeightOpening::new(*w, blind);
            assert!(opening.verify(&b.weight_commitment, &matrix), "weight {w}");
        }
    }
}
