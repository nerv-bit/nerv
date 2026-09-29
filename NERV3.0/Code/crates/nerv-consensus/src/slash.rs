//! Slash-evidence types (WP §2.5, §11.4; erratum 129): the four provable
//! offense classes with self-contained verification. Escrow accounting is
//! nerv-economy's (chunk 17).

use nerv_core::codec::Encode;
use nerv_core::constants::SLASH;
use nerv_core::hash::Hash256;
use nerv_core::types::{Address, Interval};
use nerv_crypto::mldsa::VerifyingKey;
use nerv_registry::bundle::{verify_bundle, Bundle, BundleError};
use nerv_registry::challenge::{ChallengeOutcome, InclusionChallenge};
use nerv_registry::mempool::VerifyContext;
use nerv_state::fraud::{FraudError, FraudProof, ReexecutionFraud};
use nerv_state::{BeaconView, ChainSource, ShardState};

use crate::qc::{detect_double_sign, DoubleSignError, HeaderQc};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlashClass {
    InvalidBlock,
    InvalidInclusion,
    InvalidBundle,
    DoubleSign,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SlashEvidence {
    /// A settlement-invalid block — the minimal, beacon-verifiable class.
    InvalidBlock { proof: FraudProof },
    /// A settlement-invalid block — the reexecution class, verified
    /// against the predecessor state the verifier supplies.
    InvalidBlockReexec { fraud: ReexecutionFraud },
    /// A sustained inclusion challenge against the attested interval.
    InvalidInclusion { tau_root: Hash256, challenge: InclusionChallenge },
    /// An aggregator's bundle with a transaction that fails verification.
    InvalidBundle { bundle: Bundle, failing_index: usize },
    /// Two valid QCs over different headers at one (shard, height).
    DoubleSign { a: HeaderQc, b: HeaderQc },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Offender {
    /// The producer's header-committed payout address (§4.6).
    Payout(Address),
    Validator(VerifyingKey),
    Validators(Vec<VerifyingKey>),
}

/// Everything evidence verification consumes (erratum 129): the beacon
/// view, the transaction-verification context, and the optional
/// reexecution predecessor and DoubleSign committee.
pub struct SlashContext<'a> {
    pub view: &'a dyn BeaconView,
    pub verify: &'a VerifyContext,
    pub predecessor: Option<(&'a ShardState, &'a dyn ChainSource)>,
    pub committee: Option<(&'a [VerifyingKey], usize)>,
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SlashError {
    #[error(transparent)]
    Fraud(#[from] FraudError),
    #[error(transparent)]
    Challenge(#[from] nerv_registry::ChallengeError),
    #[error(transparent)]
    DoubleSign(#[from] DoubleSignError),
    #[error("the challenged inclusion was valid — not sustained")]
    NotSustained,
    #[error("the bundle's failing transaction does not match the claimed index: {0}")]
   BundleMismatch(BundleError),
    #[error("the bundle is valid — no offense")]
    BundleNotInvalid,
    #[error("the block is valid — no offense")]
    BlockNotInvalid,
    #[error("predecessor state unavailable for reexecution evidence")]
    PredecessorUnavailable,
    #[error("committee roster unavailable for double-sign evidence")]
    CommitteeUnavailable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedSlash {
    pub class: SlashClass,
    pub offenders: Vec<Offender>,
    pub digest: Hash256,
}

impl SlashEvidence {
    pub fn class(&self) -> SlashClass {
        match self {
            SlashEvidence::InvalidBlock { .. } | SlashEvidence::InvalidBlockReexec { .. } => {
                SlashClass::InvalidBlock
            }
            SlashEvidence::InvalidInclusion { .. } => SlashClass::InvalidInclusion,
            SlashEvidence::InvalidBundle { .. } => SlashClass::InvalidBundle,
            SlashEvidence::DoubleSign { .. } => SlashClass::DoubleSign,
        }
    }

    /// H("nerv.slash" ‖ class tag ‖ underlying digests) — the escrow
    /// ledger's dedup binding (erratum 129).
    pub fn digest(&self) -> Hash256 {
        let mut buf = Vec::new();
        match self {
            SlashEvidence::InvalidBlock { proof } => {
                buf.push(0);
                buf.extend_from_slice(proof.digest().as_bytes());
            }
            SlashEvidence::InvalidBlockReexec { fraud } => {
                buf.push(1);
                buf.extend_from_slice(fraud.digest().as_bytes());
            }
            SlashEvidence::InvalidInclusion { tau_root, challenge } => {
                buf.push(2);
                buf.extend_from_slice(tau_root.as_bytes());
                buf.extend_from_slice(challenge.digest().as_bytes());
            }
            SlashEvidence::InvalidBundle { bundle, failing_index } => {
                buf.push(3);
                bundle.summary().encode_into(&mut buf);
                buf.extend_from_slice(&failing_index.to_le_bytes());
            }
            SlashEvidence::DoubleSign { a, b } => {
                buf.push(4);
                buf.extend_from_slice(a.subject().as_bytes());
                buf.extend_from_slice(b.subject().as_bytes());
                buf.extend_from_slice(&a.header.height.as_u64().to_le_bytes());
                a.shard.encode_into(&mut buf);
                buf.extend_from_slice(&a.qc.epoch.as_u64().to_le_bytes());
            }
        }
        Hash256::concat(&SLASH, &buf)
    }

    pub fn verify(&self, ctx: &SlashContext) -> Result<VerifiedSlash, SlashError> {
        match self {
            SlashEvidence::InvalidBlock { proof } => {
                proof.verify(ctx.view)?;
                Ok(VerifiedSlash {
                    class: SlashClass::InvalidBlock,
                    offenders: vec![Offender::Payout(proof.block.header.producer_payout.clone())],
                    digest: self.digest(),
                })
            }
            SlashEvidence::InvalidBlockReexec { fraud } => {
                let Some((pred, chain)) = ctx.predecessor else {
                    return Err(SlashError::PredecessorUnavailable);
                };
                match fraud.verify(pred, ctx.view, chain) {
                    Ok(_) => Ok(VerifiedSlash {
                        class: SlashClass::InvalidBlock,
                        offenders: vec![Offender::Payout(
                            fraud.block.header.producer_payout.clone(),
                        )],
                        digest: self.digest(),
                    }),
                    Err(e) => Err(e.into()),
                }
            }
            SlashEvidence::InvalidInclusion { tau_root, challenge } => {
                let interval: Interval = challenge.witness.interval;
                match challenge.verify(interval, tau_root, ctx.verify) {
                    Ok(ChallengeOutcome::Sustained) => Ok(VerifiedSlash {
                        class: SlashClass::InvalidInclusion,
                        // The attesting committee's identity resolves from
                        // the economy's own records (erratum 129).
                        offenders: Vec::new(),
                        digest: self.digest(),
                    }),
                    Ok(ChallengeOutcome::Rejected) => Err(SlashError::NotSustained),
                    Err(e) => Err(e.into()),
                }
            }
            SlashEvidence::InvalidBundle { bundle, failing_index } => {
                match verify_bundle(bundle, ctx.verify) {
                    Err(e) => match e {
                        BundleError::Verification { index, .. } if index == *failing_index => {
                            Ok(VerifiedSlash {
                                class: SlashClass::InvalidBundle,
                                offenders: vec![Offender::Validator(bundle.aggregator)],
                                digest: self.digest(),
                            })
                        }
                        other => Err(SlashError::BundleMismatch(other)),
                    },
                    Ok(()) => Err(SlashError::BundleNotInvalid),
                }
            }
            SlashEvidence::DoubleSign { a, b } => {
                let Some((committee, quorum)) = ctx.committee else {
                    return Err(SlashError::CommitteeUnavailable);
                };
                let indices = detect_double_sign(a, b, committee, *quorum)?;
                Ok(VerifiedSlash {
                    class: SlashClass::DoubleSign,
                    offenders: indices
                        .into_iter()
                        .map(|i| Offender::Validator(committee[i]))
                        .collect(),
                    digest: self.digest(),
                })
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::harness::{
        garbage_proof, header, inclusion_fixture, keys, qc_for, reexec_fixture, verify_ctx,
        fee_fraud_fixture, TestView, NoChain,
    };
    use nerv_core::types::{Epoch, ShardSet};

    #[test]
    fn invalid_block_minimal() {
        let (proof, view) = fee_fraud_fixture(0xD1);
        let evidence = SlashEvidence::InvalidBlock { proof };
        assert_eq!(evidence.class(), SlashClass::InvalidBlock);
        let ctx = SlashContext {
            view: &view,
            verify: &verify_ctx(),
            predecessor: None,
            committee: None,
        };
        let verified = evidence.verify(&ctx).unwrap();
        assert_eq!(verified.class, SlashClass::InvalidBlock);
        assert_eq!(verified.offenders.len(), 1);
        assert_eq!(verified.digest, evidence.digest());

        // A view that rejects the facts rejects the evidence.
        let empty_view = TestView::default();
        let ctx = SlashContext {
            view: &empty_view,
            verify: &verify_ctx(),
            predecessor: None,
            committee: None,
        };
        assert!(matches!(
            evidence.verify(&ctx),
            Err(SlashError::Fraud(FraudError::FactsRejected))
        ));
    }

    #[test]
    fn invalid_block_reexecution() {
        let (fraud, pred, view) = reexec_fixture(0xD2);
        let evidence = SlashEvidence::InvalidBlockReexec { fraud };
        let chain = NoChain;
        let ctx = SlashContext {
            view: &view,
            verify: &verify_ctx(),
            predecessor: Some((&pred, &chain)),
            committee: None,
        };
        let verified = evidence.verify(&ctx).unwrap();
        assert_eq!(verified.class, SlashClass::InvalidBlock);
        assert_eq!(verified.offenders.len(), 1);

        // Without a predecessor: unavailable.
        let ctx = SlashContext { view: &view, verify: &verify_ctx(), predecessor: None, committee: None };
        assert!(matches!(
            evidence.verify(&ctx),
            Err(SlashError::PredecessorUnavailable)
        ));

        // Distinct digests per class tag.
        let (other, _, _) = reexec_fixture(0xD3);
        assert_ne!(evidence.digest(), SlashEvidence::InvalidBlockReexec { fraud: other }.digest());
    }

    #[test]
    fn invalid_inclusion() {
        let (challenge, tau_root, ctx_registry) = inclusion_fixture(0xD4);
        let evidence =
            SlashEvidence::InvalidInclusion { tau_root, challenge: challenge.clone() };
        assert_eq!(evidence.class(), SlashClass::InvalidInclusion);
        let view = TestView::default();
        let ctx = SlashContext {
            view: &view,
            verify: &ctx_registry,
            predecessor: None,
            committee: None,
        };
        let verified = evidence.verify(&ctx).unwrap();
        assert_eq!(verified.class, SlashClass::InvalidInclusion);
        assert!(verified.offenders.is_empty(), "committee resolution is the economy's");

        // A different attested root breaks the witness.
        let evidence = SlashEvidence::InvalidInclusion {
            tau_root: Hash256::from_bytes([0xEE; 32]),
            challenge,
        };
        assert!(matches!(
            evidence.verify(&ctx),
            Err(SlashError::Challenge(nerv_registry::ChallengeError::WitnessRejected))
        ));
    }

    #[test]
    fn invalid_bundle() {
        let (bundle, ctx_registry) = crate::testutil::harness::bundle_fixture(0xD5);
        let evidence = SlashEvidence::InvalidBundle { bundle: bundle.clone(), failing_index: 0 };
        assert_eq!(evidence.class(), SlashClass::InvalidBundle);
        let view = TestView::default();
        let ctx = SlashContext { view: &view, verify: &ctx_registry, predecessor: None, committee: None };
        let verified = evidence.verify(&ctx).unwrap();
        assert_eq!(verified.offenders, vec![Offender::Validator(bundle.aggregator)]);

        // A wrong claimed index does not verify.
        let evidence = SlashEvidence::InvalidBundle { bundle, failing_index: 1 };
        assert!(matches!(
            evidence.verify(&ctx),
            Err(SlashError::BundleMismatch(_))
        ));
    }

    #[test]
    fn double_sign_offenders() {
        let (keys, roster) = keys(21);
        let shard = ShardSet::genesis().ids()[7];
        let ha = header(shard, 5, 0xD6);
        let hb = header(shard, 5, 0xD7);
        let a = HeaderQc {
            shard,
            header: ha,
            qc: qc_for(Hash256::from_bytes([0u8; 32]), Epoch::from_u64(2), 0..15, &keys, 15),
        };
        let a = HeaderQc {
            qc: qc_for(a.header.header_hash(), Epoch::from_u64(2), 0..15, &keys, 15),
            ..a
        };
        let b = HeaderQc {
            shard,
            header: hb,
            qc: qc_for(Hash256::from_bytes([0u8; 32]), Epoch::from_u64(2), 5..20, &keys, 15),
        };
        let b = HeaderQc {
            qc: qc_for(b.header.header_hash(), Epoch::from_u64(2), 5..20, &keys, 15),
            ..b
        };
        let evidence = SlashEvidence::DoubleSign { a, b };
        let view = TestView::default();
        let ctx = SlashContext {
            view: &view,
            verify: &verify_ctx(),
            predecessor: None,
            committee: Some((&roster, 15)),
        };
        let verified = evidence.verify(&ctx).unwrap();
        assert_eq!(verified.class, SlashClass::DoubleSign);
        match &verified.offenders[..] {
            [Offender::Validators(vks)] => {
                assert_eq!(vks.len(), 10);
                for (i, vk) in vks.iter().enumerate() {
                    assert_eq!(*vk, roster[5 + i]);
                }
            }
            other => panic!("unexpected offenders: {other:?}"),
        }

        // Without the committee: unavailable.
        let ctx = SlashContext { view: &view, verify: &verify_ctx(), predecessor: None, committee: None };
        assert!(matches!(
            evidence.verify(&ctx),
            Err(SlashError::CommitteeUnavailable)
        ));
    }

    #[test]
    fn digests_bind_the_classes() {
        let (proof, view) = fee_fraud_fixture(0xD8);
        let (challenge, tau_root, _) = inclusion_fixture(0xD9);
        let a = SlashEvidence::InvalidBlock { proof };
        let b = SlashEvidence::InvalidInclusion { tau_root, challenge };
        assert_ne!(a.digest(), b.digest());
        assert_eq!(a.digest(), a.digest());
        let _ = view;
        let _ = garbage_proof();
    }
}
