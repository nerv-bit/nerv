//! The referendum lifecycle (WP §12.8, §C.2; erratum 181): draft →
//! active → closed, the three tiers, and the two-epoch constitutional
//! confirmation mechanism.


use nerv_core::constants::REFERENDUM;
use nerv_core::hash::Hash256;
use nerv_core::types::Epoch;


use crate::tally::TallyResult;


/// What is being voted on (erratum 181).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReferendumSubject {
    /// A parameter change: (parameter id, proposed value).
    Parameter { parameter_id: u32, new_value: u64 },
    /// A codec W-epoch: the candidate W's commitment and the machine-check
    /// evidence (the certification's result, supplied at creation).
    WEpoch { w_commitment: Hash256, evidence: WEpochEvidence },
    /// A constitutional amendment: (amendment id, the text's hash).
    Constitutional { amendment_id: u32, text_hash: Hash256 },
}


impl ReferendumSubject {
    pub fn tier(&self) -> Tier {
        match self {
            ReferendumSubject::Parameter { .. } => Tier::Parameter,
            ReferendumSubject::WEpoch { .. } => Tier::WEpoch,
            ReferendumSubject::Constitutional { .. } => Tier::Constitutional,
        }
    }


    /// The subject's canonical encoding for the ID derivation.
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            ReferendumSubject::Parameter { parameter_id, new_value } => {
                out.push(0);
                out.extend_from_slice(&parameter_id.to_le_bytes());
                out.extend_from_slice(&new_value.to_le_bytes());
            }
            ReferendumSubject::WEpoch { w_commitment, evidence } => {
                out.push(1);
                out.extend_from_slice(w_commitment.as_bytes());
                out.extend_from_slice(&evidence.encode());
            }
            ReferendumSubject::Constitutional { amendment_id, text_hash } => {
                out.push(2);
                out.extend_from_slice(&amendment_id.to_le_bytes());
                out.extend_from_slice(text_hash.as_bytes());
            }
        }
        out
    }
}


/// The machine-check evidence for a W-epoch referendum (erratum 181):
/// the result of nerv-codec's certify function, supplied as data —
/// the governance layer checks completeness, never re-runs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WEpochEvidence {
    pub spark_certified: bool,
    pub norms_certified: bool,
    pub independence_certified: bool,
}


impl WEpochEvidence {
    pub fn complete(&self) -> bool {
        self.spark_certified && self.norms_certified && self.independence_certified
    }


    fn encode(&self) -> [u8; 3] {
        [
            u8::from(self.spark_certified),
            u8::from(self.norms_certified),
            u8::from(self.independence_certified),
        ]
    }
}


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier {
    Parameter,
    WEpoch,
    Constitutional,
}


impl Tier {
    pub fn name(self) -> &'static str {
        match self {
            Tier::Parameter => "parameter",
            Tier::WEpoch => "w-epoch",
            Tier::Constitutional => "constitutional",
        }
    }
}


/// The referendum's lifecycle state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LifecycleState {
    Draft,
    Active { epoch: Epoch },
    Closed { epoch: Epoch, passed: bool },
}


impl LifecycleState {
    pub fn name(&self) -> &'static str {
        match self {
            LifecycleState::Draft => "draft",
            LifecycleState::Active { .. } => "active",
            LifecycleState::Closed { .. } => "closed",
        }
    }
}


/// The referendum's ID: H("nerv.referendum" ‖ subject ‖ tier ‖ epoch ‖ nonce).
pub type ReferendumId = Hash256;


fn derive_id(subject: &ReferendumSubject, creation_epoch: Epoch, nonce: u64) -> ReferendumId {
    let mut msg = Vec::new();
    msg.extend_from_slice(&subject.encode());
    msg.push(match subject.tier() {
        Tier::Parameter => 0,
        Tier::WEpoch => 1,
        Tier::Constitutional => 2,
    });
    msg.extend_from_slice(&creation_epoch.as_u64().to_le_bytes());
    msg.extend_from_slice(&nonce.to_le_bytes());
    Hash256::concat(&REFERENDUM, &msg)
}


/// A referendum: the subject, the lifecycle, and the ID.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Referendum {
    pub id: ReferendumId,
    pub subject: ReferendumSubject,
    pub state: LifecycleState,
}


impl Referendum {
    /// Create a draft referendum. The W-epoch gate is checked at creation
    /// (erratum 181): incomplete evidence is a hard error.
    pub fn new(
        subject: ReferendumSubject,
        creation_epoch: Epoch,
        nonce: u64,
    ) -> Result<Referendum, crate::error::ReferendumError> {
        if let ReferendumSubject::WEpoch { evidence, .. } = &subject {
            if !evidence.complete() {
                return Err(crate::error::ReferendumError::WEpochGateIncomplete);
            }
        }
        Ok(Referendum {
            id: derive_id(&subject, creation_epoch, nonce),
            subject,
            state: LifecycleState::Draft,
        })
    }


    /// Activate at the given epoch.
    pub fn activate(&mut self, epoch: Epoch) -> Result<(), crate::error::ReferendumError> {
        match &self.state {
            LifecycleState::Draft => {
                self.state = LifecycleState::Active { epoch };
                Ok(())
            }
            other => Err(crate::error::ReferendumError::NotActive { state: other.name() }),
        }
    }


    /// Close with the tally result. Returns the passage decision.
    pub fn close(
        &mut self,
        epoch: Epoch,
        result: &TallyResult,
    ) -> Result<bool, crate::error::ReferendumError> {
        let passed = match self.subject.tier() {
            Tier::Parameter => result.majority_yes(),
            Tier::WEpoch => {
                // The W-epoch gate was checked at creation; the vote is
                // both-chamber majority.
                result.majority_yes()
            }
            Tier::Constitutional => result.supermajority_yes(),
        };
        match &self.state {
            LifecycleState::Active { .. } => {
                self.state = LifecycleState::Closed { epoch, passed };
                Ok(passed)
            }
            other => Err(crate::error::ReferendumError::NotActive { state: other.name() }),
        }
    }


    pub fn tier(&self) -> Tier {
        self.subject.tier()
    }
}


/// The two-epoch constitutional confirmation (erratum 181): the first
/// referendum passes at epoch E; the confirmation must pass at E+1 on
/// the SAME subject; adoption requires both.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstitutionalPair {
    pub subject: ReferendumSubject,
    pub first_epoch: Epoch,
    pub first_passed: bool,
    pub second_epoch: Epoch,
    pub second_passed: bool,
}


impl ConstitutionalPair {
    /// From a passed constitutional referendum: create the confirmation
    /// for the next epoch.
    pub fn from_first(first: &Referendum) -> Result<ConstitutionalPair, crate::error::ReferendumError> {
        let LifecycleState::Closed { epoch, passed: true } = &first.state else {
            return Err(crate::error::ReferendumError::NotActive {
                state: first.state.name(),
            });
        };
        if first.tier() != Tier::Constitutional {
            return Err(crate::error::ReferendumError::NotActive { state: "not constitutional" });
        }
        Ok(ConstitutionalPair {
            subject: first.subject.clone(),
            first_epoch: *epoch,
            first_passed: true,
            second_epoch: Epoch::from_u64(epoch.as_u64() + 1),
            second_passed: false,
        })
    }


    /// Record the confirmation's result: must be at second_epoch and on
    /// the same subject.
    pub fn confirm(
        &mut self,
        confirmation: &Referendum,
    ) -> Result<bool, crate::error::ReferendumError> {
        let LifecycleState::Closed { epoch, passed } = &confirmation.state else {
            return Err(crate::error::ReferendumError::NotActive {
                state: confirmation.state.name(),
            });
        };
        if *epoch != self.second_epoch {
            return Err(crate::error::ReferendumError::ConfirmationEpoch {
                expected: self.second_epoch.as_u64(),
                found: epoch.as_u64(),
            });
        }
        let expected_subject = self.subject.encode();
        if confirmation.subject.encode() != expected_subject {
            return Err(crate::error::ReferendumError::ConfirmationSubject {
                expected: Hash256::concat(&REFERENDUM, &expected_subject),
                found: Hash256::concat(&REFERENDUM, &confirmation.subject.encode()),
            });
        }
        self.second_passed = *passed;
        Ok(self.adopted())
    }


    /// The amendment is adopted iff both referenda passed.
    pub fn adopted(&self) -> bool {
        self.first_passed && self.second_passed
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::chambers::{BootstrapPhase, ChamberTally};
    use crate::tally::TallyResult;


    fn epoch(e: u64) -> Epoch {
        Epoch::from_u64(e)
    }


    fn param_subject() -> ReferendumSubject {
        ReferendumSubject::Parameter { parameter_id: 42, new_value: 1000 }
    }


    fn w_epoch_subject(complete: bool) -> ReferendumSubject {
        ReferendumSubject::WEpoch {
            w_commitment: Hash256::from_bytes([7u8; 32]),
            evidence: WEpochEvidence {
                spark_certified: complete,
                norms_certified: complete,
                independence_certified: complete,
            },
        }
    }


    fn const_subject() -> ReferendumSubject {
        ReferendumSubject::Constitutional {
            amendment_id: 1,
            text_hash: Hash256::from_bytes([9u8; 32]),
        }
    }


    fn tally(yes: u128, no: u128, validator_yes: u128, validator_no: u128) -> TallyResult {
        TallyResult {
            referendum_id_hash: None,
            note_holder: ChamberTally {
                yes_nano: yes, no_nano: no, abstain_nano: 0, ballots: 1,
            },
            validator: ChamberTally {
                yes_nano: validator_yes, no_nano: validator_no, abstain_nano: 0, ballots: 1,
            },
            bootstrap: BootstrapPhase::Both,
        }
    }


    #[test]
    fn parameter_lifecycle() {
        let mut r = Referendum::new(param_subject(), epoch(100), 0).unwrap();
        assert_eq!(r.tier(), Tier::Parameter);
        assert!(matches!(r.state, LifecycleState::Draft));


        r.activate(epoch(101)).unwrap();
        assert!(matches!(r.state, LifecycleState::Active { .. }));


        // Passage: simple majority in both chambers.
        let passed = r.close(epoch(102), &tally(600, 400, 100, 50)).unwrap();
        assert!(passed);
        assert!(matches!(r.state, LifecycleState::Closed { passed: true, .. }));


        // A failed referendum.
        let mut r2 = Referendum::new(param_subject(), epoch(100), 1).unwrap();
        r2.activate(epoch(101)).unwrap();
        let passed = r2.close(epoch(102), &tally(400, 600, 50, 100)).unwrap();
        assert!(!passed);


        // Transitions from wrong states are errors.
        let mut r3 = Referendum::new(param_subject(), epoch(100), 2).unwrap();
        assert!(r3.close(epoch(101), &tally(1, 0, 1, 0)).is_err());
        r3.activate(epoch(101)).unwrap();
        assert!(r3.activate(epoch(102)).is_err());
    }


    #[test]
    fn w_epoch_gate() {
        // Complete evidence: the referendum is creatable.
        let r = Referendum::new(w_epoch_subject(true), epoch(100), 0).unwrap();
        assert_eq!(r.tier(), Tier::WEpoch);


        // Incomplete evidence: rejected at creation.
        assert!(matches!(
            Referendum::new(w_epoch_subject(false), epoch(100), 0),
            Err(crate::error::ReferendumError::WEpochGateIncomplete)
        ));


        // Partially incomplete: also rejected.
        let partial = ReferendumSubject::WEpoch {
            w_commitment: Hash256::from_bytes([7u8; 32]),
            evidence: WEpochEvidence {
                spark_certified: true,
                norms_certified: true,
                independence_certified: false,
            },
        };
        assert!(Referendum::new(partial, epoch(100), 0).is_err());
    }


    #[test]
    fn constitutional_two_epoch_rule() {
        let mut first = Referendum::new(const_subject(), epoch(100), 0).unwrap();
        assert_eq!(first.tier(), Tier::Constitutional);
        first.activate(epoch(100)).unwrap();


        // First referendum: supermajority passes.
        // 700/(700+300) = 70% ≥ 2/3.
        let passed = first.close(epoch(100), &tally(700, 300, 70, 30)).unwrap();
        assert!(passed);


        // The pair: first passed, second pending at epoch+1.
        let mut pair = ConstitutionalPair::from_first(&first).unwrap();
        assert!(pair.first_passed);
        assert!(!pair.second_passed);
        assert!(!pair.adopted());
        assert_eq!(pair.second_epoch, epoch(101));


        // The confirmation at epoch 101: same subject, supermajority.
        let mut second = Referendum::new(const_subject(), epoch(101), 1).unwrap();
        second.activate(epoch(101)).unwrap();
        let passed = second.close(epoch(101), &tally(700, 300, 70, 30)).unwrap();
        assert!(passed);


        let adopted = pair.confirm(&second).unwrap();
        assert!(adopted);
        assert!(pair.adopted());


        // A failed confirmation: not adopted.
        let mut pair2 = ConstitutionalPair::from_first(&first).unwrap();
        let mut failed = Referendum::new(const_subject(), epoch(101), 2).unwrap();
        failed.activate(epoch(101)).unwrap();
        failed.close(epoch(101), &tally(300, 700, 30, 70)).unwrap();
        assert!(!pair2.confirm(&failed).unwrap());
    }


    #[test]
    fn constitutional_confirmation_guards() {
        let mut first = Referendum::new(const_subject(), epoch(100), 0).unwrap();
        first.activate(epoch(100)).unwrap();
        first.close(epoch(100), &tally(700, 300, 70, 30)).unwrap();
        let mut pair = ConstitutionalPair::from_first(&first).unwrap();


        // Wrong epoch.
        let mut wrong_epoch = Referendum::new(const_subject(), epoch(102), 0).unwrap();
        wrong_epoch.activate(epoch(102)).unwrap();
        wrong_epoch.close(epoch(102), &tally(700, 300, 70, 30)).unwrap();
        assert!(matches!(
            pair.confirm(&wrong_epoch),
            Err(crate::error::ReferendumError::ConfirmationEpoch { expected: 101, found: 102 })
        ));


        // Wrong subject.
        let mut wrong_subject = Referendum::new(param_subject(), epoch(101), 0).unwrap();
        wrong_subject.activate(epoch(101)).unwrap();
        wrong_subject.close(epoch(101), &tally(700, 300, 70, 30)).unwrap();
        assert!(matches!(
            pair.confirm(&wrong_subject),
            Err(crate::error::ReferendumError::ConfirmationSubject { .. })
        ));


        // Not closed.
        let mut not_closed = Referendum::new(const_subject(), epoch(101), 1).unwrap();
        not_closed.activate(epoch(101)).unwrap();
        assert!(pair.confirm(&not_closed).is_err());
    }


    #[test]
    fn non_constitutional_cannot_form_a_pair() {
        let mut r = Referendum::new(param_subject(), epoch(100), 0).unwrap();
        r.activate(epoch(100)).unwrap();
        r.close(epoch(100), &tally(600, 400, 60, 40)).unwrap();
        assert!(ConstitutionalPair::from_first(&r).is_err());
    }


    #[test]
    fn ids_are_deterministic_and_subject_sensitive() {
        let a = Referendum::new(param_subject(), epoch(100), 0).unwrap();
        let b = Referendum::new(param_subject(), epoch(100), 0).unwrap();
        assert_eq!(a.id, b.id);


        let c = Referendum::new(param_subject(), epoch(100), 1).unwrap();
        assert_ne!(a.id, c.id, "nonce-sensitive");


        let d = Referendum::new(
            ReferendumSubject::Parameter { parameter_id: 43, new_value: 1000 },
            epoch(100), 0,
        ).unwrap();
        assert_ne!(a.id, d.id, "subject-sensitive");


        let e = Referendum::new(param_subject(), epoch(101), 0).unwrap();
        assert_ne!(a.id, e.id, "epoch-sensitive");


        let w = Referendum::new(w_epoch_subject(true), epoch(100), 0).unwrap();
        assert_ne!(a.id, w.id, "tier-sensitive (via the subject encoding)");
    }


    #[test]
    fn supermajority_boundary() {
        let mut r = Referendum::new(const_subject(), epoch(100), 0).unwrap();
        r.activate(epoch(100)).unwrap();


        // Exactly 2/3: adopted (≥, not >).
        let passed = r.close(epoch(100), &tally(667, 333, 67, 33)).unwrap();
        assert!(passed, "667/(667+333) = 66.7% ≥ 2/3");


        // Just below 2/3: not adopted.
        let mut r2 = Referendum::new(const_subject(), epoch(100), 1).unwrap();
        r2.activate(epoch(100)).unwrap();
        let passed = r2.close(epoch(100), &tally(666, 334, 67, 33)).unwrap();
        assert!(!passed, "666/(666+334) = 66.6% < 2/3");
    }
}
