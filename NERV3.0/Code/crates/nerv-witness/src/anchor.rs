//! The light-client anchor (WP §11.2, §11.4; erratum 176): the
//! epoch-attestation chain plus the current epoch's interval
//! attestations. The verifier's trust root and the source of the 𝔾 root
//! against which cold witnesses verify.


use std::collections::BTreeMap;


use nerv_core::codec::Encode;
use nerv_core::hash::Hash256;
use nerv_core::types::{Epoch, Interval};
use nerv_consensus::attestation::{
    verify_chain, AttestationError, EpochAttestation, IntervalAttestation,
};
use nerv_crypto::mldsa::VerifyingKey;
use nerv_crypto::sigaggr::validate_qc;


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AnchorError {
    #[error("attestation: {0}")]
    Attestation(#[from] AttestationError),
    #[error("qc: {0}")]
    Qc(#[from] nerv_crypto::QcError),
    #[error("interval {found} does not follow {expected}")]
    IntervalGap { expected: u64, found: u64 },
    #[error("epoch {found} does not follow {expected}")]
    EpochGap { expected: u64, found: u64 },
    #[error("epoch attestation {epoch} not found in the chain")]
    EpochMissing { epoch: u64 },
    #[error("no attestations in the anchor")]
    Empty,
}


/// The committee roster lookup: epoch → (signer set, quorum).
pub type CommitteeSource<'a> =
    dyn Fn(Epoch) -> Option<(Vec<VerifyingKey>, usize)> + 'a;


/// The light-client anchor (erratum 176).
#[derive(Clone, Debug, Default)]
pub struct LightAnchor {
    epoch_chain: Vec<EpochAttestation>,
    interval_chain: Vec<IntervalAttestation>,
}


impl LightAnchor {
    pub fn new() -> LightAnchor {
        LightAnchor::default()
    }


    /// From a trusted epoch attestation (a checkpoint; §11.4's "any
    /// trusted anchor" — the chain suffix verifies from it).
    pub fn from_checkpoint(att: EpochAttestation) -> LightAnchor {
        LightAnchor { epoch_chain: vec![att], interval_chain: Vec::new() }
    }


    pub fn epoch_chain(&self) -> &[EpochAttestation] {
        &self.epoch_chain
    }


    pub fn interval_chain(&self) -> &[IntervalAttestation] {
        &self.interval_chain
    }


    pub fn last_epoch(&self) -> Option<Epoch> {
        self.epoch_chain.last().map(|a| a.epoch)
    }


    pub fn last_interval(&self) -> Option<Interval> {
        self.interval_chain.last().map(|a| a.interval)
    }


    /// The latest 𝔾 root (from the latest interval attestation; if the
    /// epoch just rolled, from the epoch attestation's digest tree —
    /// the caller walks forward from there).
    pub fn g_root(&self) -> Option<Hash256> {
        self.interval_chain.last().map(|a| a.g_root)
    }


    /// Extend with an interval attestation: verify the chain link and
    /// the QC against the committee.
    pub fn extend_interval(
        &mut self,
        att: IntervalAttestation,
        committee: &CommitteeSource<'_>,
    ) -> Result<(), AnchorError> {
        let expected = match self.interval_chain.last() {
            Some(prev) => Interval::from_u64(prev.interval.as_u64() + 1),
            None => {
                // The first interval of the current epoch.
                let epoch = self
                    .epoch_chain
                    .last()
                    .ok_or(AnchorError::Empty)?
                    .epoch;
                epoch.first_interval().ok_or(AnchorError::Empty)?
            }
        };
        if att.interval != expected {
            return Err(AnchorError::IntervalGap {
                expected: expected.as_u64(),
                found: att.interval.as_u64(),
            });
        }
        if let Some(prev) = self.interval_chain.last() {
            if att.prev != prev.digest() {
                return Err(AttestationError::ChainBroken {
                    interval: att.interval.as_u64(),
                }
                .into());
            }
        }
        let (signers, quorum) = committee(att.interval.epoch())
            .ok_or(AttestationError::SignersUnavailable {
                interval: att.interval.as_u64(),
            })?;
        att.validate(&signers, quorum)?;
        self.interval_chain.push(att);
        Ok(())
    }


    /// Extend with an epoch attestation: verify the chain link and the QC.
    pub fn extend_epoch(
        &mut self,
        att: EpochAttestation,
        committee: &CommitteeSource<'_>,
    ) -> Result<(), AnchorError> {
        let expected = match self.epoch_chain.last() {
            Some(prev) => Epoch::from_u64(prev.epoch.as_u64() + 1),
            None => att.epoch, // genesis or a checkpoint start
        };
        if att.epoch != expected {
            return Err(AnchorError::EpochGap {
                expected: expected.as_u64(),
                found: att.epoch.as_u64(),
            });
        }
        if let Some(prev) = self.epoch_chain.last() {
            if att.prev != prev.digest() {
                return Err(AttestationError::ChainBroken {
                    interval: att.epoch.as_u64(),
                }
                .into());
            }
        }
        let (signers, quorum) = committee(att.epoch)
            .ok_or(AttestationError::SignersUnavailable {
                interval: att.epoch.as_u64(),
            })?;
        att.validate(&signers, quorum)?;
        self.epoch_chain.push(att);
        self.interval_chain.clear();
        Ok(())
    }


    /// Verify the full anchor: every epoch attestation's QC and chain,
    /// every interval attestation's QC and chain (the B6 cost).
    pub fn verify(&self, committee: &CommitteeSource<'_>) -> Result<(), AnchorError> {
        for w in self.epoch_chain.windows(2) {
            if w[1].prev != w[0].digest() {
                return Err(AttestationError::ChainBroken {
                    interval: w[1].epoch.as_u64(),
                }
                .into());
            }
        }
        for att in &self.epoch_chain {
            let (signers, quorum) = committee(att.epoch)
                .ok_or(AttestationError::SignersUnavailable {
                    interval: att.epoch.as_u64(),
                })?;
            att.validate(&signers, quorum)?;
        }
        for w in self.interval_chain.windows(2) {
            if w[1].prev != w[0].digest() {
                return Err(AttestationError::ChainBroken {
                    interval: w[1].interval.as_u64(),
                }
                .into());
            }
        }
        for att in &self.interval_chain {
            let (signers, quorum) = committee(att.interval.epoch())
                .ok_or(AttestationError::SignersUnavailable {
                    interval: att.interval.as_u64(),
                })?;
            att.validate(&signers, quorum)?;
        }
        Ok(())
    }


    /// Walk back to genesis: the full epoch-attestation chain (§11.2's
    /// ~18 MB/yr; one-time, self-contained). Returns the digests.
    pub fn walk_to_genesis(&self) -> Vec<Hash256> {
        self.epoch_chain.iter().map(|a| a.digest()).collect()
    }


    /// The anchor's serialized size (B6's < 100 KB budget).
    pub fn serialized_len(&self) -> usize {
        self.epoch_chain
            .iter()
            .map(|a| a.encoded_len())
            .sum::<usize>()
            + self.interval_chain
                .iter()
                .map(|a| a.encoded_len())
                .sum::<usize>()
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use nerv_consensus::attestation::interval_digests_root;
    use nerv_core::types::INTERVALS_PER_EPOCH;
    use nerv_crypto::mldsa::SigningKey;
    use nerv_crypto::sigaggr::{vote_bytes, VoteCollector};


    fn keys(seed: u64, n: u64) -> (Vec<SigningKey>, Vec<VerifyingKey>) {
        let sks: Vec<SigningKey> = (0..n)
            .map(|i| {
                let mut b = [0u8; 32];
                b[..8].copy_from_slice(&(seed + i).to_le_bytes());
                SigningKey::from_seed(&b).unwrap()
            })
            .collect();
        let vks: Vec<VerifyingKey> = sks.iter().map(|k| *k.verifying_key()).collect();
        (sks, vks)
    }


    fn committee_fn<'a>(
        vks: &'a Vec<VerifyingKey>,
        quorum: usize,
    ) -> impl Fn(Epoch) -> Option<(Vec<VerifyingKey>, usize)> + 'a {
        move |_| Some((vks.clone(), quorum))
    }


    fn interval_att(
        interval: u64,
        prev: Hash256,
        sks: &[SigningKey],
        vks: &[VerifyingKey],
        quorum: usize,
    ) -> IntervalAttestation {
        let signers: Vec<(usize, &SigningKey)> =
            (0..quorum).map(|i| (i, &sks[i])).collect();
        IntervalAttestation::build(
            Interval::from_u64(interval),
            Hash256::from_bytes([interval as u8; 32]),
            Hash256::from_bytes([(interval + 1) as u8; 32]),
            Hash256::from_bytes([(interval + 2) as u8; 32]),
            prev,
            &signers,
            vks,
            quorum,
        )
        .unwrap()
    }


    fn epoch_att(
        epoch: u64,
        digests: &[Hash256],
        prev: Hash256,
        sks: &[SigningKey],
        vks: &[VerifyingKey],
        quorum: usize,
    ) -> EpochAttestation {
        let signers: Vec<(usize, &SigningKey)> =
            (0..quorum).map(|i| (i, &sks[i])).collect();
        EpochAttestation::build(
            Epoch::from_u64(epoch),
            interval_digests_root(digests),
            prev,
            &signers,
            vks,
            quorum,
        )
        .unwrap()
    }


    #[test]
    fn anchor_lifecycle() {
        let (sks, vks) = keys(0xANCH0R, 21);
        let quorum = 15;
        let cf = committee_fn(&vks, quorum);
        let zero = Hash256::from_bytes([0u8; 32]);


        // Epoch 0: intervals 0..86399.
        let mut anchor = LightAnchor::new();
        let e0 = epoch_att(0, &[zero], zero, &sks, &vks, quorum);
        anchor.extend_epoch(e0.clone(), &cf).unwrap();
        assert_eq!(anchor.last_epoch(), Some(Epoch::from_u64(0)));


        let i0 = interval_att(0, zero, &sks, &vks, quorum);
        anchor.extend_interval(i0.clone(), &cf).unwrap();
        let i1 = interval_att(1, i0.digest(), &sks, &vks, quorum);
        anchor.extend_interval(i1.clone(), &cf).unwrap();
        assert_eq!(anchor.last_interval(), Some(Interval::from_u64(1)));
        assert_eq!(anchor.g_root(), Some(i1.g_root));


        // A gap is rejected.
        let i3 = interval_att(3, i1.digest(), &sks, &vks, quorum);
        assert!(matches!(
            anchor.extend_interval(i3, &cf),
            Err(AnchorError::IntervalGap { expected: 2, found: 3 })
        ));


        // A broken link is rejected.
        let i2_bad = interval_att(2, Hash256::from_bytes([9u8; 32]), &sks, &vks, quorum);
        assert!(anchor.extend_interval(i2_bad, &cf).is_err());


        // The correct i2.
        let i2 = interval_att(2, i1.digest(), &sks, &vks, quorum);
        anchor.extend_interval(i2.clone(), &cf).unwrap();


        // Epoch 1: roll the interval chain into the next epoch attestation.
        let digests = vec![i0.digest(), i1.digest(), i2.digest()];
        let e1 = epoch_att(1, &digests, e0.digest(), &sks, &vks, quorum);
        anchor.extend_epoch(e1, &cf).unwrap();
        assert_eq!(anchor.last_epoch(), Some(Epoch::from_u64(1)));
        assert!(anchor.interval_chain().is_empty(), "rolled");
        assert_eq!(anchor.g_root(), None, "no intervals yet in the new epoch");


        // The next epoch's intervals.
        let i86400 = interval_att(86_400, zero, &sks, &vks, quorum);
        anchor.extend_interval(i86400.clone(), &cf).unwrap();
        assert_eq!(anchor.g_root(), Some(i86400.g_root));


        // Full verification passes.
        anchor.verify(&cf).unwrap();
        assert_eq!(anchor.walk_to_genesis().len(), 2);
        assert!(anchor.serialized_len() > 0);
        assert!(anchor.serialized_len() < 100_000, "B6: < 100 KB, got {}", anchor.serialized_len());
    }


    #[test]
    fn from_checkpoint() {
        let (sks, vks) = keys(0xCHKP0NT, 21);
        let quorum = 15;
        let cf = committee_fn(&vks, quorum);
        let zero = Hash256::from_bytes([0u8; 32]);
        let e5 = epoch_att(5, &[zero], zero, &sks, &vks, quorum);
        let anchor = LightAnchor::from_checkpoint(e5.clone());
        assert_eq!(anchor.last_epoch(), Some(Epoch::from_u64(5)));
        anchor.verify(&cf).unwrap();
        assert_eq!(anchor.walk_to_genesis(), vec![e5.digest()]);
    }


    #[test]
    fn epoch_chain_and_error_paths() {
        let (sks, vks) = keys(0xEPOCH, 21);
        let quorum = 15;
        let cf = committee_fn(&vks, quorum);
        let zero = Hash256::from_bytes([0u8; 32]);
        let mut anchor = LightAnchor::new();


        // Empty anchor: extending an interval fails.
        assert!(matches!(
            anchor.extend_interval(
                interval_att(0, zero, &sks, &vks, quorum),
                &cf
            ),
            Err(AnchorError::Empty)
        ));


        // Epoch 0, then epoch 2 (skipping 1): gap.
        let e0 = epoch_att(0, &[zero], zero, &sks, &vks, quorum);
        anchor.extend_epoch(e0, &cf).unwrap();
        let e2 = epoch_att(2, &[zero], zero, &sks, &vks, quorum);
        assert!(matches!(
            anchor.extend_epoch(e2, &cf),
            Err(AnchorError::EpochGap { expected: 1, found: 2 })
        ));


        // A broken epoch chain link.
        let e1_bad = epoch_att(1, &[zero], Hash256::from_bytes([9u8; 32]), &sks, &vks, quorum);
        assert!(anchor.extend_epoch(e1_bad, &cf).is_err());


        // Wrong committee.
        let (_, other_vks) = keys(0xBAD, 21);
        let wrong_cf = committee_fn(&other_vks, quorum);
        let e1 = epoch_att(1, &[zero], zero, &sks, &vks, quorum);
        assert!(anchor.extend_epoch(e1, &wrong_cf).is_err());


        // Missing committee.
        let none_cf = |_: Epoch| -> Option<(Vec<VerifyingKey>, usize)> { None };
        assert!(anchor.extend_epoch(
            epoch_att(1, &[zero], zero, &sks, &vks, quorum),
            &none_cf
        ).is_err());
    }
}
