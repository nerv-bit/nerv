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


    /// bundles → interval → (build, the txids included).
    fn interval_with(seeds: &[u64]) -> (crate::interval::IntervalBuild, Vec<TxId>) {
        let b = Bundle::build(&agg_key(9), seeds.iter().map(|&s| entry(s)).collect()).unwrap();
        let build = build_interval_commit(Interval::from_u64(0), &[b], &IntervalLedger::new())
            .unwrap();
        let mut ids: Vec<TxId> = seeds.iter().map(|&s| txid_of(&shell(s))).collect();
        ids.sort();
        (build, ids)
    }


    #[test]
    fn garbage_proof_sustains_the_challenge() {
        let ctx = ctx();
        let (build, ids) = interval_with(&[1, 2, 3]);
        let interval = build.commit.interval;
        let root = build.commit.tau_root;


        // The attested transaction's own (invalid) proof sustains.
        let target = ids[1];
        let orig = build
            .report
            .set
            .iter_sorted()
            .find(|(t, _)| **t == target)
            .map(|(t, _)| *t)
            .unwrap();
        let shell = shell(2);
        assert_eq!(txid_of(&shell), orig);
        let w = build.witness(&orig).unwrap();
        let ch = InclusionChallenge { witness: w, shell: shell.clone(), proof: shallow_proof() };
        assert_eq!(ch.verify(interval, &root, &ctx).unwrap(), ChallengeOutcome::Sustained);
        assert_eq!(ch.txid().unwrap(), orig);
    }


    #[test]
    fn rejections() {
        let ctx = ctx();
        let (build, ids) = interval_with(&[4, 5]);
        let interval = build.commit.interval;
        let root = build.commit.tau_root;
        let target = ids[0];
        let shell = shell(4);
        assert_eq!(txid_of(&shell), target);


       // A different shell under the same witness: the txid does not match.
       let other = shell(5);
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


        // Context version mismatch is a challenge error, not sustenance.
        let mut v7 = shell.clone();
        v7.legs[0].weight_version = 7;
        let ch = InclusionChallenge {
            witness: build.witness(&target).unwrap(),
            shell: v7,
            proof: shallow_proof(),
        };
        assert!(matches!(
            ch.verify(interval, &root, &ctx),
            Err(ChallengeError::ContextVersion { shell: 7, codec: 1 })
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
        let (build, ids) = interval_with(&[6, 7]);
        let shell = shell(6);
        assert_eq!(txid_of(&shell), ids[0]);
        let ch = InclusionChallenge {
            witness: build.witness(&ids[0]).unwrap(),
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
