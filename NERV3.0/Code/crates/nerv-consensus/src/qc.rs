//! Quorum certificates in committee context (WP §4.6; erratum 130):
//! subject binding to the shard header, and same-slot equivocation
//! detection — the DoubleSign slash surface.

use nerv_core::hash::Hash256;
use nerv_core::types::ShardId;
use nerv_state::header::ShardHeader;
use nerv_crypto::mldsa::VerifyingKey;
use nerv_crypto::sigaggr::{
    validate_qc, QuorumCertificate, MAX_BITMAP_COMMITTEE, QcError,
};

/// The committee's signing subject (erratum 131): the header hash with
/// the qc_hash field zeroed — the unsigned header. The header's qc_hash
/// then commits to the resulting QC, binding (header, QC) without
/// circularity. The FULL header hash (with qc_hash) is the identity for
/// 𝔾 leaves, prev-chaining, and finality.
pub fn body_hash(header: &ShardHeader) -> Hash256 {
   let mut h = header.clone();
   h.qc_hash = Hash256::from_bytes([0u8; 32]);
   h.header_hash()
}


/// A QC bound to what it signed: `shard`'s header at the header's height.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeaderQc {
   pub shard: ShardId,
   pub header: ShardHeader,
   pub qc: QuorumCertificate,
}


impl HeaderQc {
   /// The QC's subject: the body hash (erratum 131).
   pub fn subject(&self) -> Hash256 {
       body_hash(&self.header)
   }


   /// The full header hash: the 𝔾-leaf and finality identity.
   pub fn header_hash(&self) -> Hash256 {
       self.header.header_hash()
   }


   /// Committee-context validation: the subject is the header's body
   /// hash and the QC carries a committee quorum.
   pub fn validate(&self, committee: &[VerifyingKey], quorum: usize) -> Result<(), QcError> {
       if self.header.height.as_u64() == 0 {
           return Err(QcError::MalformedCertificate("headers start at height 1"));
       }
       if self.qc.subject != self.subject() {
           return Err(QcError::MalformedCertificate("qc subject is not the header body hash"));
       }
       validate_qc(&self.qc, committee, quorum)?;
       Ok(())
   }
}


/// Member indices present in both signer bitmaps.
pub fn signer_intersection(a: &QuorumCertificate, b: &QuorumCertificate) -> Vec<usize> {
    (0..MAX_BITMAP_COMMITTEE)
        .filter(|&i| a.contains_signer(i) && b.contains_signer(i))
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DoubleSignError {
    #[error(transparent)]
    Qc(#[from] QcError),
    #[error("the two QCs are for different slots (shard or height)")]
    DifferentSlots,
    #[error("the two QCs are for the same header — no equivocation")]
    SameHeader,
    #[error("the two QCs are from different epochs — different committees")]
    DifferentEpochs,
    #[error("the signer sets do not overlap — no attributable offender")]
    NoOverlappingSigner,
}

/// Same-slot equivocation: two valid QCs, one committee, one
/// (shard, height), two different headers. Returns the offending member
/// indices (positions in `committee`).
pub fn detect_double_sign(
    a: &HeaderQc,
    b: &HeaderQc,
    committee: &[VerifyingKey],
    quorum: usize,
) -> Result<Vec<usize>, DoubleSignError> {
    if a.shard != b.shard || a.header.height != b.header.height {
        return Err(DoubleSignError::DifferentSlots);
    }
    if a.subject() == b.subject() {
        return Err(DoubleSignError::SameHeader);
    }
    if a.qc.epoch != b.qc.epoch {
        return Err(DoubleSignError::DifferentEpochs);
    }
    a.validate(committee, quorum)?;
    b.validate(committee, quorum)?;
    let intersection = signer_intersection(&a.qc, &b.qc);
    if intersection.is_empty() {
        return Err(DoubleSignError::NoOverlappingSigner);
    }
    Ok(intersection)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::harness::{header, keys, qc_for};
    use nerv_core::types::{Epoch, ShardSet};

    #[test]
    fn validate_binds_subject_and_committee() {
        let (keys, roster) = keys(21);
        let shard = ShardSet::genesis().ids()[7];
         let hdr = header(shard, 1, 0x51);
       let qc = qc_for(body_hash(&hdr), Epoch::from_u64(2), 0..15, &keys, 15);
        let hq = HeaderQc { shard, header: hdr.clone(), qc }.clone();
        drop(hdr);
        hq.validate(&roster, 15).unwrap();
        assert_eq!(hq.subject(), hq.header.header_hash());

        let mut bad = hq.clone();
        bad.qc.subject = bad.qc.subject(); // no-op for the borrow checker
        let forged = h(&bad.header);
        bad.qc.subject = forged;
        assert!(matches!(
            bad.validate(&roster, 15),
            Err(QcError::MalformedCertificate("qc subject is not the header hash"))
        ));

        let (_, other) = keys(21 + 7);
        assert!(matches!(
            hq.validate(&other, 15),
            Err(QcError::InvalidVote { .. })
        ));
        assert!(matches!(
            hq.validate(&roster, 16),
            Err(QcError::QuorumNotMet { .. })
        ));

        let hdr0 = header(shard, 0, 0x52);
       let qc0 = qc_for(body_hash(&hdr0), Epoch::from_u64(2), 0..15, &keys, 15);
        let hq0 = HeaderQc { shard, header: hdr0, qc: qc0 };
        assert!(matches!(
            hq0.validate(&roster, 15),
            Err(QcError::MalformedCertificate("headers start at height 1"))
        ));
    }

    #[test]
    fn detects_offenders() {
        let (keys, roster) = keys(21);
        let shard = ShardSet::genesis().ids()[7];
        let ha = header(shard, 5, 0x53);
        let hb = header(shard, 5, 0x54);
        assert_ne!(ha.header_hash(), hb.header_hash());
        let a = HeaderQc {
           shard,
           header: ha,
           qc: qc_for(body_hash(&ha), Epoch::from_u64(2), 0..15, &keys, 15),
       };
       let b = HeaderQc {
           shard,
           header: hb,
           qc: qc_for(body_hash(&hb), Epoch::from_u64(2), 5..20, &keys, 15),
       };
       let offenders = detect_double_sign(&a, &b, &roster, 15).unwrap();

        assert_eq!(offenders, (5..15).collect::<Vec<_>>());

        // Different height / different header / different epochs.
        let hc = header(shard, 6, 0x55);
        let c = HeaderQc {
           shard,
           header: hc,
           qc: qc_for(body_hash(&hc), Epoch::from_u64(2), 0..15, &keys, 15),
       };
       assert!(matches!(
           detect_double_sign(&a, &c, &roster, 15),
           Err(DoubleSignError::DifferentSlots)
       ));
       let a2 = HeaderQc {
           shard,
           header: a.header.clone(),
           qc: qc_for(body_hash(&a.header), Epoch::from_u64(3), 0..15, &keys, 15),
       };

        assert!(matches!(
            detect_double_sign(&a, &c, &roster, 15),
            Err(DoubleSignError::DifferentSlots)
        ));
        let a2 = HeaderQc {
            shard,
            header: a.header.clone(),
            qc: qc_for(a.header.header_hash(), Epoch::from_u64(3), 0..15, &keys, 15),
        };
        assert!(matches!(
            detect_double_sign(&a, &a2, &roster, 15),
            Err(DoubleSignError::DifferentEpochs)
        ));
        assert!(matches!(
            detect_double_sign(&a, &a, &roster, 15),
            Err(DoubleSignError::SameHeader)
        ));
    }

    #[test]
    fn disjoint_signers_are_not_attributable() {
        let (keys, roster) = keys(30);
        let shard = ShardSet::genesis().ids()[7];
        let ha = header(shard, 2, 0x56);
        let hb = header(shard, 2, 0x57);
        let a = HeaderQc {
           shard,
           header: ha,
           qc: qc_for(body_hash(&ha), Epoch::from_u64(1), 0..15, &keys, 15),
       };
       let b = HeaderQc {
           shard,
           header: hb,
           qc: qc_for(body_hash(&hb), Epoch::from_u64(1), 15..30, &keys, 15),
       };

        assert!(matches!(
            detect_double_sign(&a, &b, &roster, 15),
            Err(DoubleSignError::NoOverlappingSigner)
        ));
    }

    fn h(hq: &HeaderQc) -> Hash256 {
        // A digest distinct from the header hash, for subject forgery.
        let mut b = *hq.subject().as_bytes();
        b[0] ^= 1;
        Hash256::from_bytes(b)
    }
}
