//! The committee-attestation fallback (WP §5.5; erratum 122): when the
//! prover market misses the folding deadline, the registry committee
//! attests the interval commit, and the interval finalizes only after an
//! unchallenged 30-second window.


use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::error::CodecError;
use nerv_core::types::Epoch;
use nerv_crypto::mldsa::{SigningKey, VerifyingKey};
use nerv_crypto::sigaggr::{validate_qc, vote_bytes, QuorumCertificate, VoteCollector};


use crate::challenge::ChallengeOutcome;
use crate::error::DegradedError;
use crate::interval::IntervalCommit;


pub const CHALLENGE_WINDOW_SECS: u64 = nerv_core::params::REGISTRY_CHALLENGE_WINDOW_SECS;


/// The committee's attestation: the interval commit and the 15-of-21 QC
/// over its digest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DegradedAttestation {
    pub commit: IntervalCommit,
    pub qc: QuorumCertificate,
}


impl DegradedAttestation {
    pub fn subject(&self) -> nerv_core::hash::Hash256 {
        self.commit.digest()
    }


    /// The committee's signing path: collect member signatures over the
    /// commit digest until quorum.
    pub fn attest(
        commit: IntervalCommit,
        epoch: Epoch,
        signers: &[(usize, &SigningKey)],
        committee: &[VerifyingKey],
        quorum: usize,
    ) -> Result<DegradedAttestation, DegradedError> {
        let subject = commit.digest();
        let mut vc = VoteCollector::new(epoch, subject);
        for &(i, sk) in signers {
            let sig = sk.sign(&vote_bytes(epoch, &subject))?;
            vc.add(i, sig, committee)?;
        }
        Ok(DegradedAttestation { commit, qc: vc.assemble(quorum)? })
    }


    /// Full validation against the committee roster: the QC's signatures,
    /// quorum, and subject binding to the commit digest.
    pub fn validate(
        &self,
        committee: &[VerifyingKey],
        quorum: usize,
    ) -> Result<(), DegradedError> {
        if self.qc.subject != self.subject() {
            return Err(DegradedError::SubjectMismatch);
        }
        validate_qc(&self.qc, committee, quorum)?;
        Ok(())
    }
}


impl Encode for DegradedAttestation {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.commit.encode_into(out);
        self.qc.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        self.commit.encoded_len() + self.qc.encoded_len()
    }
}


impl Decode for DegradedAttestation {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let commit = IntervalCommit::decode_from(r)?;
        let qc = QuorumCertificate::decode_from(r)?;
        Ok(DegradedAttestation { commit, qc })
    }
}


/// The 30-second challenge window (params; erratum 122).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChallengeWindow {
    pub opened_at: u64,
    pub closes_at: u64,
}


impl ChallengeWindow {
    pub fn open(at_secs: u64) -> ChallengeWindow {
        ChallengeWindow {
            opened_at: at_secs,
            closes_at: at_secs.saturating_add(CHALLENGE_WINDOW_SECS),
        }
    }


    pub fn closed(&self, now_secs: u64) -> bool {
        now_secs >= self.closes_at
    }


    pub fn remaining(&self, now_secs: u64) -> u64 {
        self.closes_at.saturating_sub(now_secs)
    }
}


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DegradedOutcome {
    /// The window is still open.
    Pending,
    /// The window closed unchallenged: finalize (beacon commits the
    /// interval; the ledger commits the set).
    Finalize,
    /// A sustained inclusion challenge voids the interval — it is never
    /// finalized (honest txids re-enter later intervals).
    Void,
}


/// The degraded-mode process state machine: attest → window → outcome.
#[derive(Clone, Debug)]
pub struct DegradedProcess {
    attestation: DegradedAttestation,
    window: ChallengeWindow,
    sustained: usize,
}


impl DegradedProcess {
    pub fn start(attestation: DegradedAttestation, at_secs: u64) -> DegradedProcess {
        DegradedProcess { attestation, window: ChallengeWindow::open(at_secs), sustained: 0 }
    }


    pub fn attestation(&self) -> &DegradedAttestation {
        &self.attestation
    }


    pub fn window(&self) -> &ChallengeWindow {
        &self.window
    }


    pub fn sustained_challenges(&self) -> usize {
        self.sustained
    }


    pub fn record(&mut self, outcome: ChallengeOutcome) {
        if matches!(outcome, ChallengeOutcome::Sustained) {
            self.sustained += 1;
        }
    }


    pub fn outcome(&self, now_secs: u64) -> DegradedOutcome {
        if !self.window.closed(now_secs) {
            DegradedOutcome::Pending
        } else if self.sustained > 0 {
            DegradedOutcome::Void
        } else {
            DegradedOutcome::Finalize
        }
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::challenge::ChallengeOutcome;
    use crate::interval::build_interval_commit;
    use crate::testutil::harness::{agg_key, committee, shell, shallow_proof};
    use nerv_core::types::Interval;
    use nerv_crypto::sigaggr::QcError;
    use nerv_proofs::IntervalLedger;


    fn commit() -> IntervalCommit {
        let entry = |seed: u64| {
            let canon = shell(seed).canonicalize().unwrap();
            crate::mempool::PoolEntry {
                txid: nerv_state::canonical_txid(&canon),
                shell: canon,
                proof: shallow_proof(),
            }
        };
        let b = crate::bundle::Bundle::build(&agg_key(1), vec![entry(1), entry(2)]).unwrap();
        build_interval_commit(Interval::from_u64(0), &[b], &IntervalLedger::new())
            .unwrap()
            .commit
    }


    #[test]
    fn attest_validate_with_a_real_committee() {
        let (sks, vks) = committee();
        let commit = commit();
        let epoch = Epoch::from_u64(3);
        let signers: Vec<(usize, &SigningKey)> =
            (0..15).map(|i| (i, &sks[i])).collect();
        let att = DegradedAttestation::attest(commit, epoch, &signers, &vks, 15).unwrap();
        assert_eq!(att.subject(), att.commit.digest());
        assert_eq!(att.qc.signer_count(), 15);
        att.validate(&vks, 15).unwrap();


        // Wrong roster: the signatures do not verify.
        let (_, other) = crate::testutil::harness::committee_offset(100);
        assert!(matches!(
            att.validate(&other, 15),
            Err(DegradedError::Qc(QcError::InvalidVote { .. }))
        ));


        // Sub-quorum validation fails.
        assert!(matches!(att.validate(&vks, 16), Err(DegradedError::Qc(QcError::QuorumNotMet { .. }))));


        // Tampered subject: the QC no longer binds the commit.
        let mut bad = att.clone();
        bad.qc.subject = nerv_core::hash::Hash256::from_bytes([0xEE; 32]);
        assert!(matches!(bad.validate(&vks, 15), Err(DegradedError::SubjectMismatch)));


        // Tampered commit: the subject check fires.
        let mut bad = att.clone();
        bad.commit.tau_root = nerv_core::hash::Hash256::from_bytes([0xEE; 32]);
        assert!(matches!(bad.validate(&vks, 15), Err(DegradedError::SubjectMismatch)));
    }


    #[test]
    fn window_and_outcomes() {
        assert_eq!(CHALLENGE_WINDOW_SECS, 30);
        let (sks, vks) = committee();
        let epoch = Epoch::from_u64(0);
        let signers: Vec<(usize, &SigningKey)> = (0..15).map(|i| (i, &sks[i])).collect();
        let att = DegradedAttestation::attest(commit(), epoch, &signers, &vks, 15).unwrap();


        let mut p = DegradedProcess::start(att, 1_000);
        assert_eq!(p.window().opened_at, 1_000);
        assert_eq!(p.window().closes_at, 1_030);
        assert!(!p.window().closed(1_029));
        assert_eq!(p.window().remaining(1_000), 30);
        assert_eq!(p.outcome(1_000), DegradedOutcome::Pending);
        assert_eq!(p.outcome(1_029), DegradedOutcome::Pending);
        assert_eq!(p.outcome(1_030), DegradedOutcome::Finalize);
        assert_eq!(p.outcome(2_000), DegradedOutcome::Finalize);


        p.record(ChallengeOutcome::Rejected);
        assert_eq!(p.outcome(1_030), DegradedOutcome::Finalize);
        p.record(ChallengeOutcome::Sustained);
        p.record(ChallengeOutcome::Rejected);
        assert_eq!(p.sustained_challenges(), 1);
        assert_eq!(p.outcome(1_029), DegradedOutcome::Pending);
        assert_eq!(p.outcome(1_030), DegradedOutcome::Void);
    }


    #[test]
    fn codec_roundtrip() {
        let (sks, vks) = committee();
        let signers: Vec<(usize, &SigningKey)> = (0..15).map(|i| (i, &sks[i])).collect();
        let att =
            DegradedAttestation::attest(commit(), Epoch::from_u64(7), &signers, &vks, 15).unwrap();
        let enc = att.encode();
        assert_eq!(enc.len(), att.encoded_len());
        let dec = DegradedAttestation::decode(&enc).unwrap();
        assert_eq!(dec, att);
        dec.validate(&vks, 15).unwrap();
        for cut in 0..enc.len() {
            assert!(DegradedAttestation::decode(&enc[..cut]).is_err(), "cut {cut}");
        }
        let mut ext = enc.clone();
        ext.push(0);
        assert!(DegradedAttestation::decode(&ext).is_err());
    }
}
