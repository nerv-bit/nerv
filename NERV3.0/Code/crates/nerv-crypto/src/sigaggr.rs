//! Quorum-certificate assembly, validation, and hash compression (WP §4.6).
//! A QC is a set of ML-DSA-65 signatures over the vote message
//! "nerv.qc.vote" ‖ epoch(LE) ‖ subject from ≥ quorum members of an epoch's
//! committee; headers commit only QC_hash = BLAKE3("nerv.qc" ‖ canonical QC)
//! and the full certificate travels in the block body. Member indices refer
//! to the committee roster in the caller's canonical order (sortition rank
//! order — see sortition::select_committee).

use std::collections::BTreeMap;
use std::fmt;
use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::{QUORUM_CERT, QC_VOTE};
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::types::Epoch;

use crate::error::QcError;
use crate::mldsa::{Signature, VerifyingKey, SIG_LEN};

pub const MAX_BITMAP_COMMITTEE: usize = 32;

/// The bytes a committee member signs: domain ‖ epoch(LE) ‖ subject — all
/// fixed-width.
pub fn vote_bytes(epoch: Epoch, subject: &Hash256) -> Vec<u8> {
    let mut out = Vec::with_capacity(QC_VOTE.as_bytes().len() + 8 + 32);
    out.extend_from_slice(QC_VOTE.as_bytes());
    out.extend_from_slice(&epoch.as_u64().to_le_bytes());
    out.extend_from_slice(subject.as_bytes());
    out
}

#[derive(Clone, PartialEq, Eq)]
pub struct QuorumCertificate {
    pub epoch: Epoch,
    pub subject: Hash256,
    /// Bitmap over committee member indices (bit i = member i).
    pub signers: u32,
    /// Signatures in ascending signer-index order.
    pub signatures: Vec<Signature>,
}

impl QuorumCertificate {
    pub fn signer_count(&self) -> usize {
        self.signers.count_ones() as usize
    }

    pub fn signer_indices(&self) -> Vec<usize> {
        (0..MAX_BITMAP_COMMITTEE)
            .filter(|i| self.signers & (1 << i) != 0)
            .collect()
    }

    pub fn contains_signer(&self, member: usize) -> bool {
        member < MAX_BITMAP_COMMITTEE && self.signers & (1 << member) != 0
    }

    /// WP §4.6 hash compression: the value the header commits.
    pub fn qc_hash(&self) -> Hash256 {
        Hash256::concat(&QUORUM_CERT, &self.encode())
    }
}

impl fmt::Debug for QuorumCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuorumCertificate")
            .field("epoch", &self.epoch)
            .field("subject", &self.subject)
            .field("signers", &self.signer_count())
            .finish()
    }
}

/// Collects and immediately verifies votes until quorum, rejecting
/// duplicates and invalid signatures at entry.
pub struct VoteCollector {
    epoch: Epoch,
    subject: Hash256,
    votes: BTreeMap<usize, Signature>,
}

impl VoteCollector {
    pub fn new(epoch: Epoch, subject: Hash256) -> VoteCollector {
        VoteCollector { epoch, subject, votes: BTreeMap::new() }
    }

    pub fn epoch(&self) -> Epoch {
        self.epoch
    }

    pub fn subject(&self) -> Hash256 {
        self.subject
    }

    pub fn count(&self) -> usize {
        self.votes.len()
    }

    pub fn has_quorum(&self, quorum: usize) -> bool {
        self.votes.len() >= quorum
    }

    /// Add a vote; returns `false` (idempotently) if the member already
    /// voted. The signature is verified against `committee[member]` over the
    /// canonical vote message before insertion.
    pub fn add(
        &mut self,
        member: usize,
        signature: Signature,
        committee: &[VerifyingKey],
    ) -> Result<bool, QcError> {
        if committee.len() > MAX_BITMAP_COMMITTEE {
            return Err(QcError::CommitteeTooLarge { size: committee.len(), max: MAX_BITMAP_COMMITTEE });
        }
        if member >= committee.len() {
            return Err(QcError::MemberIndexOutOfRange { member, committee: committee.len() });
        }
        if self.votes.contains_key(&member) {
            return Ok(false);
        }
        if !committee[member].verify(&vote_bytes(self.epoch, &self.subject), &signature) {
            return Err(QcError::InvalidVote { member });
        }
        self.votes.insert(member, signature);
        Ok(true)
    }

    pub fn assemble(&self, quorum: usize) -> Result<QuorumCertificate, QcError> {
        if self.votes.len() < quorum {
            return Err(QcError::QuorumNotMet { have: self.votes.len(), need: quorum });
        }
        let mut signers = 0u32;
        for &i in self.votes.keys() {
            signers |= 1 << i;
        }
        Ok(QuorumCertificate {
            epoch: self.epoch,
            subject: self.subject,
            signers,
            signatures: self.votes.values().cloned().collect(),
        })
    }
}

/// Full validation against the committee roster for `qc.epoch`: bitmap/count
/// consistency, in-range signers, quorum, and every signature. Returns the
/// QC hash (the header value) on success.
pub fn validate_qc(
    qc: &QuorumCertificate,
    committee: &[VerifyingKey],
    quorum: usize,
) -> Result<Hash256, QcError> {
    if committee.len() > MAX_BITMAP_COMMITTEE {
        return Err(QcError::CommitteeTooLarge { size: committee.len(), max: MAX_BITMAP_COMMITTEE });
    }
    let indices = qc.signer_indices();
    if indices.len() != qc.signatures.len() {
        return Err(QcError::MalformedCertificate("signer bitmap does not match signature count"));
    }
    if indices.len() < quorum {
        return Err(QcError::QuorumNotMet { have: indices.len(), need: quorum });
    }
    let msg = vote_bytes(qc.epoch, &qc.subject);
    for (member, sig) in indices.iter().zip(&qc.signatures) {
        if *member >= committee.len() {
            return Err(QcError::MalformedCertificate("signer index outside the committee"));
        }
        if !committee[*member].verify(&msg, sig) {
            return Err(QcError::InvalidVote { member: *member });
        }
    }
    Ok(qc.qc_hash())
}

impl Encode for QuorumCertificate {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.epoch.encode_into(out);
        self.subject.encode_into(out);
        out.extend_from_slice(&self.signers.to_le_bytes());
        out.extend_from_slice(&(self.signatures.len() as u32).to_le_bytes());
        for s in &self.signatures {
            s.encode_into(out);
        }
    }
    fn encoded_len(&self) -> usize {
        8 + 32 + 4 + 4 + self.signatures.len() * SIG_LEN
    }
}

impl Decode for QuorumCertificate {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let epoch = Epoch::decode_from(r)?;
        let subject = Hash256::decode_from(r)?;
        let signers = r.read_u32()?;
        let n = r.read_seq_len()?;
        if n != signers.count_ones() as usize {
            return Err(CodecError::InvariantViolated("signer bitmap / signature count mismatch"));
        }
        r.enter()?;
        let mut signatures = Vec::with_capacity(n);
        for _ in 0..n {
            signatures.push(Signature::decode_from(r)?);
        }
        r.leave();
        Ok(QuorumCertificate { epoch, subject, signers, signatures })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mldsa::SigningKey;
    use crate::testutil::DetRng;

    fn key(seed: u64) -> SigningKey {
        let mut rng = DetRng::new(seed);
        SigningKey::from_seed(&rng.bytes32()).unwrap()
    }

    fn committee21() -> Vec<SigningKey> {
        (100..121).map(key).collect()
    }

    fn subject() -> Hash256 {
        Hash256::from_bytes(DetRng::new(55).bytes32())
    }

    #[test]
    fn full_flow_assemble_validate_hash() {
        let members = committee21();
        let roster: Vec<VerifyingKey> = members.iter().map(|k| *k.verifying_key()).collect();
        let epoch = Epoch::from_u64(9);
        let subj = subject();
        let mut collector = VoteCollector::new(epoch, subj);
        for (i, m) in members.iter().enumerate().take(15) {
            let sig = m.sign(&vote_bytes(epoch, &subj)).unwrap();
            assert!(collector.add(i, sig, &roster).unwrap());
        }
        assert_eq!(collector.count(), 15);
        assert!(collector.has_quorum(15));
        assert!(!collector.has_quorum(16));
        let qc = collector.assemble(15).unwrap();
        assert_eq!(qc.signer_count(), 15);
        assert_eq!(qc.signer_indices(), (0..15).collect::<Vec<_>>());
        assert!(qc.contains_signer(7));
        assert!(!qc.contains_signer(19));
        let h = validate_qc(&qc, &roster, 15).unwrap();
        assert_eq!(h, qc.qc_hash());
        // 15-of-21 wire size: 8+32+4+4+15*3309 = 49,703 B (WP §11.2's ~50 KB)
        assert_eq!(qc.encoded_len(), 8 + 32 + 4 + 4 + 15 * SIG_LEN);
        // super-quorum validation with the same certificate
        assert!(validate_qc(&qc, &roster, 16).is_err());
        assert!(validate_qc(&qc, &roster, 15).is_ok());
    }

    #[test]
    fn quorum_not_met() {
        let members = committee21();
        let roster: Vec<VerifyingKey> = members.iter().map(|k| *k.verifying_key()).collect();
        let epoch = Epoch::from_u64(1);
        let subj = subject();
        let mut collector = VoteCollector::new(epoch, subj);
        for (i, m) in members.iter().enumerate().take(14) {
            let sig = m.sign(&vote_bytes(epoch, &subj)).unwrap();
            collector.add(i, sig, &roster).unwrap();
        }
        assert!(matches!(
            collector.assemble(15),
            Err(QcError::QuorumNotMet { have: 14, need: 15 })
        ));
    }

    #[test]
    fn duplicate_votes_are_idempotent() {
        let members = committee21();
        let roster: Vec<VerifyingKey> = members.iter().map(|k| *k.verifying_key()).collect();
        let epoch = Epoch::from_u64(2);
        let subj = subject();
        let mut collector = VoteCollector::new(epoch, subj);
        let sig = members[3].sign(&vote_bytes(epoch, &subj)).unwrap();
        assert!(collector.add(3, sig.clone(), &roster).unwrap());
        assert!(!collector.add(3, sig, &roster).unwrap());
        assert_eq!(collector.count(), 1);
    }

    #[test]
    fn invalid_votes_rejected() {
        let members = committee21();
        let roster: Vec<VerifyingKey> = members.iter().map(|k| *k.verifying_key()).collect();
        let epoch = Epoch::from_u64(3);
        let subj = subject();
        let mut collector = VoteCollector::new(epoch, subj);
        // signed for a different subject
        let wrong_subject = Hash256::from_bytes(DetRng::new(56).bytes32());
        let sig = members[0].sign(&vote_bytes(epoch, &wrong_subject)).unwrap();
        assert!(matches!(
            collector.add(0, sig, &roster),
            Err(QcError::InvalidVote { member: 0 })
        ));
        // signed for a different epoch
        let sig = members[0].sign(&vote_bytes(Epoch::from_u64(4), &subj)).unwrap();
        assert!(matches!(
            collector.add(0, sig, &roster),
            Err(QcError::InvalidVote { member: 0 })
        ));
        // tampered signature
        let mut sig = members[0].sign(&vote_bytes(epoch, &subj)).unwrap();
        sig.0[500] ^= 1;
        assert!(matches!(
            collector.add(0, sig, &roster),
            Err(QcError::InvalidVote { member: 0 })
        ));
        assert_eq!(collector.count(), 0);
        // out-of-range member / oversized committee
        let sig = members[0].sign(&vote_bytes(epoch, &subj)).unwrap();
        assert!(matches!(
            collector.add(21, sig, &roster),
            Err(QcError::MemberIndexOutOfRange { member: 21, committee: 21 })
        ));
        let big: Vec<VerifyingKey> = (0..33).map(|i| *key(200 + i).verifying_key()).collect();
        let sig = key(200).sign(&vote_bytes(epoch, &subj)).unwrap();
        assert!(matches!(
            collector.add(0, sig, &big),
            Err(QcError::CommitteeTooLarge { size: 33, .. })
        ));
    }

    #[test]
    fn validate_rejects_tampered_certificate() {
        let members = committee21();
        let roster: Vec<VerifyingKey> = members.iter().map(|k| *k.verifying_key()).collect();
        let epoch = Epoch::from_u64(5);
        let subj = subject();
        let mut collector = VoteCollector::new(epoch, subj);
        for (i, m) in members.iter().enumerate().take(15) {
            let sig = m.sign(&vote_bytes(epoch, &subj)).unwrap();
            collector.add(i, sig, &roster).unwrap();
        }
        let qc = collector.assemble(15).unwrap();
        let mut tampered = qc.clone();
        tampered.signatures[7].0[1000] ^= 1;
        assert!(matches!(
            validate_qc(&tampered, &roster, 15),
            Err(QcError::InvalidVote { member: 7 })
        ));
        // subject substitution: signatures no longer match
        let mut other = qc.clone();
        other.subject = Hash256::from_bytes(DetRng::new(57).bytes32());
        assert!(validate_qc(&other, &roster, 15).is_err());
        // wrong roster: member keys do not verify
        let wrong: Vec<VerifyingKey> = (300..321).map(|i| *key(i).verifying_key()).collect();
        assert!(validate_qc(&qc, &wrong, 15).is_err());
    }

    #[test]
    fn codec_roundtrip_and_bitmap_consistency() {
        let members = committee21();
        let roster: Vec<VerifyingKey> = members.iter().map(|k| *k.verifying_key()).collect();
        let epoch = Epoch::from_u64(6);
        let subj = subject();
        let mut collector = VoteCollector::new(epoch, subj);
        for (i, m) in members.iter().enumerate().take(15) {
            let sig = m.sign(&vote_bytes(epoch, &subj)).unwrap();
            collector.add(i, sig, &roster).unwrap();
        }
        let qc = collector.assemble(15).unwrap();
        let enc = qc.encode();
        assert_eq!(enc.len(), qc.encoded_len());
        let decoded = QuorumCertificate::decode(&enc).unwrap();
        assert_eq!(decoded, qc);
        assert_eq!(validate_qc(&decoded, &roster, 15).unwrap(), qc.qc_hash());
        // bitmap/count mismatch on the wire
        let mut bad = enc.clone();
        let count_off = 8 + 32 + 4;
        bad[count_off] = bad[count_off].wrapping_add(1);
        assert!(QuorumCertificate::decode(&bad).is_err());
        // truncation and trailing bytes
        assert!(QuorumCertificate::decode(&enc[..enc.len() - 1]).is_err());
        let mut ext = enc.clone();
        ext.push(0);
        assert!(QuorumCertificate::decode(&ext).is_err());
    }

    #[test]
    fn qc_hash_is_binding_and_stable() {
        let members = committee21();
        let roster: Vec<VerifyingKey> = members.iter().map(|k| *k.verifying_key()).collect();
        let epoch = Epoch::from_u64(7);
        let subj = subject();
        let mut collector = VoteCollector::new(epoch, subj);
        for (i, m) in members.iter().enumerate().take(15) {
            let sig = m.sign(&vote_bytes(epoch, &subj)).unwrap();
            collector.add(i, sig, &roster).unwrap();
        }
        let qc = collector.assemble(15).unwrap();
        let h1 = qc.qc_hash();
        assert_eq!(h1, qc.qc_hash());
        let mut other = qc.clone();
        other.epoch = Epoch::from_u64(8);
        assert_ne!(h1, other.qc_hash());
        let mut fewer = qc.clone();
        fewer.signatures.pop();
        fewer.signers &= !(1 << 14);
        assert_ne!(h1, fewer.qc_hash());
        // vote bytes differ across epochs and subjects (domain separation)
        assert_ne!(vote_bytes(Epoch::from_u64(1), &subj), vote_bytes(Epoch::from_u64(2), &subj));
    }
}
