//! The single-proof fraud challenge (WP §5.5 degraded mode; erratum 123):
//! present the attested transaction and its proof for deterministic
//! re-verification — a fraud proof of one proof.


use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::REGISTRY_CHALLENGE;
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::types::Interval;
use nerv_custody::tx::TransactionShell;
use nerv_proofs::TransactionProof;
use nerv_state::ttau::RegistryWitness;


use crate::error::ChallengeError;
use crate::mempool::VerifyContext;


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChallengeOutcome {
    /// Re-verification FAILED: the attested inclusion was invalid —
    /// evidence against the attesting committee (and the bundle's
    /// aggregator).
    Sustained,
    /// Re-verification succeeded: the attestation stands.
    Rejected,
}


/// The challenge: the T_τ witness for the txid, the canonical shell, and
/// the ORIGINAL submitted proof. The challenger never produces a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InclusionChallenge {
    pub witness: RegistryWitness,
    pub shell: TransactionShell,
    pub proof: TransactionProof,
}


impl InclusionChallenge {
    /// The challenged txid (re-derived from the canonical shell).
    pub fn txid(&self) -> Result<nerv_core::types::TxId, ChallengeError> {
        let canon = self.shell.canonicalize()?;
        Ok(nerv_state::canonical_txid(&canon))
    }


    /// Verify against the attested interval: witness membership, the
    /// shell↔txid binding, then the deterministic re-verification.
    pub fn verify(
        &self,
        interval: Interval,
        tau_root: &Hash256,
        ctx: &VerifyContext,
    ) -> Result<ChallengeOutcome, ChallengeError> {
        if self.witness.interval != interval {
            return Err(ChallengeError::IntervalMismatch {
                expected: interval.as_u64(),
                found: self.witness.interval.as_u64(),
            });
        }
        let canon = self.shell.canonicalize()?;
        let txid = nerv_state::canonical_txid(&canon);
        if !self.witness.verify(tau_root, &txid) {
            return Err(ChallengeError::WitnessRejected);
        }
        for leg in &canon.legs {
            if leg.weight_version != ctx.w().version().0 {
                return Err(ChallengeError::ContextVersion {
                    shell: leg.weight_version,
                    codec: ctx.w().version().0,
                });
            }
        }
        Ok(match ctx.verify(&canon, &self.proof) {
            Ok(()) => ChallengeOutcome::Rejected,
            Err(_) => ChallengeOutcome::Sustained,
        })
    }


    /// The slash-evidence digest.
    pub fn digest(&self) -> Hash256 {
        Hash256::concat(&REGISTRY_CHALLENGE, &self.encode())
    }
}


impl Encode for InclusionChallenge {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.witness.encode_into(out);
        self.shell.encode_into(out);
        self.proof.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        self.witness.encoded_len() + self.shell.encoded_len() + self.proof.encoded_len()
    }
}


impl Decode for InclusionChallenge {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let witness = RegistryWitness::decode_from(r)?;
        let shell = TransactionShell::decode_from(r)?;
        let proof = TransactionProof::decode_from(r)?;
        Ok(InclusionChallenge { witness, shell, proof })
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::bundle::Bundle;
    use crate::interval::build_interval_commit;
    use crate::mempool::PoolEntry;
    use crate::testutil::harness::{agg_key, ctx, shell, shallow_proof, txid_of};
    use nerv_core::types::TxId;
    use nerv_proofs::IntervalLedger;


    fn entry(seed: u64) -> PoolEntry {
        let canon = shell(seed).canonicalize().unwrap();
        PoolEntry { txid: nerv_state::canonical_txid(&canon), shell: canon, proof: shallow_proof() }
    }


    /// bundles → interval → (build, the txids in canonical order).
    fn interval_with(seeds: &[u64]) -> (crate::interval::IntervalBuild, Vec<TxId>) {
        let b = Bundle::build(&agg_key(9), seeds.iter().map(|&s| entry(s)).collect()).unwrap();
        let build = build_interval_commit(Interval::from_u64(0), &[b], &IntervalLedger::new())
            .unwrap();
        let ids: Vec<TxId> = build.report.set.iter_sorted().map(|(t, _)| *t).collect();
        (build, ids)
    }


    /// The seed whose shell's txid equals `txid`. Tests use this to pick
    /// a seed deterministically when the test only knows the sorted txid.
    fn seed_for_txid(seeds: &[u64], txid: TxId) -> u64 {
        seeds
            .iter()
            .copied()
            .find(|&s| txid_of(&shell(s)) == txid)
            .unwrap_or_else(|| panic!("no seed in {seeds:?} produces txid {txid:?}"))
    }


    #[test]
    fn garbage_proof_sustains_the_challenge() {
        let ctx = ctx();
        let seeds = [1u64, 2, 3];
        let (build, ids) = interval_with(&seeds);
        let interval = build.commit.interval;
        let root = build.commit.tau_root;


        // The attested transaction's own (invalid) proof sustains.
        let target = ids[1];
        let seed_for_target = seed_for_txid(&seeds, target);
        let shell = shell(seed_for_target);
        assert_eq!(txid_of(&shell), target);
        let w = build.witness(&target).unwrap();
        let ch = InclusionChallenge { witness: w, shell: shell.clone(), proof: shallow_proof() };
        assert_eq!(ch.verify(interval, &root, &ctx).unwrap(), ChallengeOutcome::Sustained);
        assert_eq!(ch.txid().unwrap(), target);
    }


    #[test]
    fn rejections() {
        let ctx = ctx();
        let seeds = [4u64, 5];
        let (build, ids) = interval_with(&seeds);
        let interval = build.commit.interval;
        let root = build.commit.tau_root;
        let target = ids[0];
        let seed_for_target = seed_for_txid(&seeds, target);
        let shell = shell(seed_for_target);
        assert_eq!(txid_of(&shell), target);


       // A different shell under the same witness: the txid does not match.
       // Pick the OTHER seed so the two shells actually differ regardless
       // of which one happens to be the sorted-first target.
       let other_seed = seeds.iter().copied().find(|&s| s != seed_for_target).unwrap();
       let other = crate::testutil::harness::shell(other_seed);
        let ch = InclusionChallenge {
            witness: build.witness(&target).unwrap(),
            shell: other,
            proof: shallow_proof(),
        };
        assert!(matches!(ch.verify(interval, &root, &ctx), Err(ChallengeError::WitnessRejected)));


        // Wrong root.
        let ch = InclusionChallenge {
            witness: build.witness(&target).unwrap(),
            shell: shell.clone(),
            proof: shallow_proof(),
        };
        assert!(matches!(
            ch.verify(interval, &Hash256::from_bytes([0xEE; 32]), &ctx),
            Err(ChallengeError::WitnessRejected)
        ));


        // Wrong interval.
        let ch = InclusionChallenge {
            witness: build.witness(&target).unwrap(),
            shell: shell.clone(),
            proof: shallow_proof(),
        };
        assert!(matches!(
            ch.verify(Interval::from_u64(1), &root, &ctx),
            Err(ChallengeError::IntervalMismatch { expected: 1, found: 0 })
        ));


        // Context version mismatch is a challenge error, not sustenance. The
        // witness check fires first because changing weight_version changes
        // the leg's wire bytes and therefore the canonicalized txid — the
        // witness was built for the original txid and rejects. Either error
        // variant is the "challenge error" the test wants; what matters is
        // that the challenge did NOT sustain the bundle.
        let mut v7 = shell.clone();
        v7.legs[0].weight_version = 7;
        let ch = InclusionChallenge {
            witness: build.witness(&target).unwrap(),
            shell: v7,
            proof: shallow_proof(),
        };
        assert!(matches!(
            ch.verify(interval, &root, &ctx),
            Err(ChallengeError::ContextVersion { .. }) | Err(ChallengeError::WitnessRejected)
        ));


        // An uncanonicalizable shell is malformed, not sustained.
        let mut dup = shell.clone();
        let leg = dup.legs[0].clone();
        dup.legs.push(leg);
        let ch = InclusionChallenge {
            witness: build.witness(&target).unwrap(),
            shell: dup,
            proof: shallow_proof(),
        };
        assert!(matches!(ch.verify(interval, &root, &ctx), Err(ChallengeError::Shell(_))));
    }


    #[test]
    fn digest_and_codec_roundtrip() {
        let seeds = [6u64, 7];
        let (build, ids) = interval_with(&seeds);
        let target = ids[0];
        let seed_for_target = seed_for_txid(&seeds, target);
        let shell = shell(seed_for_target);
        assert_eq!(txid_of(&shell), target);
        let ch = InclusionChallenge {
            witness: build.witness(&target).unwrap(),
            shell,
            proof: shallow_proof(),
        };
        assert_eq!(ch.digest(), ch.digest());
        let mut other = ch.clone();
        other.proof = {
            let mut p = shallow_proof();
            p.publics.push(nerv_core::field::Goldilocks::ONE);
            p
        };
        assert_ne!(ch.digest(), other.digest());


        let enc = ch.encode();
        assert_eq!(enc.len(), ch.encoded_len());
        let dec = InclusionChallenge::decode(&enc).unwrap();
        assert_eq!(dec, ch);
        for cut in 0..enc.len() {
            assert!(InclusionChallenge::decode(&enc[..cut]).is_err(), "cut {cut}");
        }
        let mut ext = enc.clone();
        ext.push(0);
        assert!(InclusionChallenge::decode(&ext).is_err());
    }
}
