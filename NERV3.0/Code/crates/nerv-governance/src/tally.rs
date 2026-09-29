//! The aggregate-only tally (WP §12.8; erratum 180): two-phase
//! (vote → reveal), per-choice sums, and the result type that never
//! carries individual weights.


use std::collections::BTreeMap;


use nerv_core::hash::Hash256;
use nerv_seal::dkg::CommitMatrix;


use crate::ballot::{Choice, ShieldedBallot, ValidatorVote, WeightOpening};
use crate::chambers::{ChamberTally, BootstrapPhase};
use crate::error::TallyError;


/// The participation floor: a referendum is decided only if the total
/// cast weight ≥ this permille of the chamber (genesis-config).
pub const PARTICIPATION_FLOOR_PERMILLE: u64 = 10; // 1%


/// The aggregate-only result (erratum 180): never carries individual
/// weights.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TallyResult {
    pub referendum_id_hash: Option<Hash256>,
    pub note_holder: ChamberTally,
    pub validator: ChamberTally,
    pub bootstrap: BootstrapPhase,
}


impl TallyResult {
    pub fn total_cast(&self) -> u128 {
        self.note_holder.total_cast() + self.validator.total_cast()
    }


    /// Both-chamber simple majority (§C.2).
    pub fn majority_yes(&self) -> bool {
        match self.bootstrap {
            BootstrapPhase::Bootstrap => self.validator.majority_yes(),
            BootstrapPhase::Both => {
                self.note_holder.majority_yes() && self.validator.majority_yes()
            }
        }
    }


    /// Both-chamber supermajority (§C.2: constitutional tier).
    pub fn supermajority_yes(&self) -> bool {
        match self.bootstrap {
            BootstrapPhase::Bootstrap => self.validator.supermajority_yes(),
            BootstrapPhase::Both => {
                self.note_holder.supermajority_yes()
                    && self.validator.supermajority_yes()
            }
        }
    }
}


/// Tally the note-holder chamber: voting phase (ballots verified, no
/// weights opened), then reveal phase (openings verified, sums computed).
pub fn tally_note_holder(
    ballots: &[ShieldedBallot],
    openings: &[WeightOpening],
    matrix: &CommitMatrix,
) -> Result<ChamberTally, TallyError> {
    if ballots.len() != openings.len() {
        return Err(TallyError::BelowFloor { cast: 0, required: 1 }); // length mismatch
    }
    // Phase 1: verify ballots (proofs + nullifier freshness).
    let mut seen = std::collections::BTreeSet::new();
    for (i, b) in ballots.iter().enumerate() {
        if !b.verify(matrix) {
            return Err(TallyError::Ballot {
                index: i,
                source: crate::error::BallotError::ProofFailed,
            });
        }
        if !seen.insert(*b.nullifier.as_bytes()) {
            return Err(TallyError::Ballot {
                index: i,
                source: crate::error::BallotError::DoubleVote { nf: b.nullifier },
            });
        }
    }
    // Phase 2: verify openings and sum.
    let mut tally = ChamberTally::default();
    for (i, (b, o)) in ballots.iter().zip(openings).enumerate() {
        if !o.verify(&b.weight_commitment, matrix) {
            return Err(TallyError::Opening {
                index: i,
                source: crate::error::BallotError::OpeningFailed,
            });
        }
        tally.add(b.choice, o.weight());
    }
    Ok(tally)
}


/// Tally the validator chamber: transparent stake-weighted votes.
pub fn tally_validator(
    votes: &[ValidatorVote],
    stake_of: &dyn Fn(&nerv_crypto::mldsa::VerifyingKey) -> u64,
) -> Result<ChamberTally, TallyError> {
    let mut seen = std::collections::BTreeSet::new();
    let mut tally = ChamberTally::default();
    for (i, v) in votes.iter().enumerate() {
        if !v.verify() {
            return Err(TallyError::Vote {
                index: i,
                source: crate::error::ChamberError::BadSignature,
            });
        }
        if !seen.insert(*v.vk.as_bytes()) {
            return Err(TallyError::Vote {
                index: i,
                source: crate::error::ChamberError::DoubleVote { vk: v.vk },
            });
        }
        let stake = stake_of(&v.vk);
        if stake == 0 {
            return Err(TallyError::Vote {
                index: i,
                source: crate::error::ChamberError::NoStake { vk: v.vk },
            });
        }
        tally.add(v.choice, stake);
    }
    Ok(tally)
}


/// The combined result (erratum 180).
pub fn combined_result(
    referendum_id: &Hash256,
    ballots: &[ShieldedBallot],
    openings: &[WeightOpening],
    matrix: &CommitMatrix,
    votes: &[ValidatorVote],
    stake_of: &dyn Fn(&nerv_crypto::mldsa::VerifyingKey) -> u64,
) -> Result<TallyResult, TallyError> {
    let note_holder = tally_note_holder(ballots, openings, matrix)?;
    let validator = tally_validator(votes, stake_of)?;
    let bootstrap = BootstrapPhase::from_participation(note_holder.total_cast());
    Ok(TallyResult {
        referendum_id_hash: Some(*referendum_id),
        note_holder,
        validator,
        bootstrap,
    })
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::ballot::{ballot_matrix, ShieldedBallot, ValidatorVote, WeightOpening};
    use crate::error::BallotError;
    use nerv_core::hash::Xof;
    use nerv_crypto::mldsa::SigningKey;
    use nerv_seal::ring::{Poly, Vec8, N};


    fn referendum(seed: u64) -> Hash256 {
        Hash256::from_bytes([seed as u8; 32])
    }


    fn nk(seed: u64) -> [u8; 32] {
        let mut b = [0u8; 32];
        b[..8].copy_from_slice(&seed.to_le_bytes());
        b
    }


    fn blinding(seed: u64) -> Vec8 {
        let mut xof = Xof::new(
            &nerv_core::constants::BALLOT_PROOF,
            &seed.to_le_bytes(),
        );
        let mut polys = [Poly::zero(); 8];
        for p in &mut polys {
            let mut vals = [0i64; N];
            for v in vals.iter_mut() {
                let x = xof.next_u64() % (2 * crate::ballot::BLINDING_BOUND + 1);
                *v = x as i64 - crate::ballot::BLINDING_BOUND as i64;
            }
            *p = Poly::from_centered(&vals);
        }
        Vec8::new(polys)
    }


    fn make_ballot(
        voter: u64, weight: u64, choice: Choice, r: &Hash256,
        matrix: &CommitMatrix,
    ) -> (ShieldedBallot, WeightOpening) {
        let b = ShieldedBallot::new(
            &nk(voter), *r, choice, weight, blinding(voter), matrix, &[voter as u8; 32],
        )
        .unwrap();
        let o = WeightOpening::new(weight, blinding(voter));
        (b, o)
    }


    #[test]
    fn note_holder_tally_aggregates() {
        let r = referendum(20);
        let m = ballot_matrix(&r).unwrap();
        let mut ballots = Vec::new();
        let mut openings = Vec::new();
        // 3 yes-voters with weights 100, 200, 300; 2 no-voters with 50, 150.
        for (v, w, c) in [
            (1u64, 100u64, Choice::Yes),
            (2, 200, Choice::Yes),
            (3, 300, Choice::Yes),
            (4, 50, Choice::No),
            (5, 150, Choice::No),
        ] {
            let (b, o) = make_ballot(v, w, c, &r, &m);
            ballots.push(b);
            openings.push(o);
        }
        let t = tally_note_holder(&ballots, &openings, &m).unwrap();
        assert_eq!(t.yes_nano, 600);
        assert_eq!(t.no_nano, 200);
        assert_eq!(t.abstain_nano, 0);
        assert_eq!(t.ballots, 5);
        assert_eq!(t.total_cast(), 800);
        assert!(t.majority_yes());
        assert!(t.supermajority_yes(), "600/800 = 75% ≥ 2/3");
    }


    #[test]
    fn note_holder_double_vote_rejected() {
        let r = referendum(21);
        let m = ballot_matrix(&r).unwrap();
        let (b1, o1) = make_ballot(1, 100, Choice::Yes, &r, &m);
        let (b2, o2) = make_ballot(1, 100, Choice::No, &r, &m); // same voter
        let err = tally_note_holder(&[b1, b2], &[o1, o2], &m).unwrap_err();
        assert!(matches!(
            err,
            TallyError::Ballot { source: BallotError::DoubleVote { .. }, .. }
        ));
    }


    #[test]
    fn note_holder_bad_proof_rejected() {
        let r = referendum(22);
        let m = ballot_matrix(&r).unwrap();
        let (b, o) = make_ballot(1, 100, Choice::Yes, &r, &m);
        let mut bad = b.clone();
        if !bad.proof.h.is_empty() {
            let cl = bad.proof.h[0].centerlift();
            let mut cl2 = cl;
            cl2[0] += 1;
            bad.proof.h[0] = Poly::from_centered(&cl2);
        }
        let err = tally_note_holder(&[bad], &[o], &m).unwrap_err();
        assert!(matches!(
            err,
            TallyError::Ballot { source: BallotError::ProofFailed, .. }
        ));
    }


    #[test]
    fn note_holder_bad_opening_rejected() {
        let r = referendum(23);
        let m = ballot_matrix(&r).unwrap();
        let (b, _) = make_ballot(1, 100, Choice::Yes, &r, &m);
        let wrong = WeightOpening::new(200, blinding(1)); // wrong weight
        let err = tally_note_holder(&[b], &[wrong], &m).unwrap_err();
        assert!(matches!(
            err,
            TallyError::Opening { source: BallotError::OpeningFailed, .. }
        ));
    }


    #[test]
    fn validator_tally() {
        let r = referendum(24);
        let sk1 = SigningKey::from_seed(&nk(1)).unwrap();
        let sk2 = SigningKey::from_seed(&nk(2)).unwrap();
        let v1 = ValidatorVote::new(&sk1, r, Choice::Yes).unwrap();
        let v2 = ValidatorVote::new(&sk2, r, Choice::No).unwrap();


        let stakes = BTreeMap::<[u8; 1952], u64>::new();
        let _ = stakes;
        let stake_fn = |vk: &nerv_crypto::mldsa::VerifyingKey| -> u64 {
            if *vk == *sk1.verifying_key() { 1_000 }
            else if *vk == *sk2.verifying_key() { 500 }
            else { 0 }
        };
        let t = tally_validator(&[v1, v2], &stake_fn).unwrap();
        assert_eq!(t.yes_nano, 1_000);
        assert_eq!(t.no_nano, 500);
        assert!(t.majority_yes());


        // Double vote.
        let v1b = ValidatorVote::new(&sk1, r, Choice::No).unwrap();
        let err = tally_validator(&[v1.clone(), v1b], &stake_fn).unwrap_err();
        assert!(matches!(
            err,
            TallyError::Vote { source: crate::error::ChamberError::DoubleVote { .. }, .. }
        ));


        // No stake.
        let sk3 = SigningKey::from_seed(&nk(3)).unwrap();
        let v3 = ValidatorVote::new(&sk3, r, Choice::Yes).unwrap();
        let err = tally_validator(&[v3], &stake_fn).unwrap_err();
        assert!(matches!(
            err,
            TallyError::Vote { source: crate::error::ChamberError::NoStake { .. }, .. }
        ));
    }


    #[test]
    fn combined_and_bootstrap() {
        let r = referendum(25);
        let m = ballot_matrix(&r).unwrap();


        // Bootstrap: no note-holder participation.
        let (vb, _) = make_ballot(1, 0, Choice::Yes, &r, &m);
        let vo = WeightOpening::new(0, blinding(1));
        let sk = SigningKey::from_seed(&nk(10)).unwrap();
        let vv = ValidatorVote::new(&sk, r, Choice::Yes).unwrap();
        let stake_fn = |vk: &nerv_crypto::mldsa::VerifyingKey| -> u64 {
            if *vk == *sk.verifying_key() { 1_000 } else { 0 }
        };
        let result = combined_result(&r, &[vb], &[vo], &m, &[vv], &stake_fn).unwrap();
        assert_eq!(result.bootstrap, BootstrapPhase::Bootstrap);
        assert!(result.majority_yes(), "the validator chamber alone decides in bootstrap");


        // Both chambers: note-holder participation active.
        let (nb, no) = make_ballot(1, 500, Choice::Yes, &r, &m);
        let result = combined_result(&r, &[nb], &[no], &m, &[vv], &stake_fn).unwrap();
        assert_eq!(result.bootstrap, BootstrapPhase::Both);
        assert!(result.majority_yes());
        assert_eq!(result.note_holder.yes_nano, 500);
        assert_eq!(result.validator.yes_nano, 1_000);
        assert_eq!(result.total_cast(), 1_500);
    }
}
