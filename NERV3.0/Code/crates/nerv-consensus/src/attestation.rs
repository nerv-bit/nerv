//! A_τ interval attestations and the epoch-attestation chain (WP §11.2;
//! erratum 126): the 𝔾 tree over per-shard header tips, the signed
//! interval body, the hash chain, and epoch compression.

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::{ATT, BEACON, Domain};
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::types::{Epoch, Interval};
use nerv_crypto::mldsa::{SigningKey, VerifyingKey};
use nerv_crypto::sigaggr::{
    validate_qc, vote_bytes, QuorumCertificate, VoteCollector, QcError,
};
use nerv_crypto::CryptoError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AttestationError {
    #[error(transparent)]
    Qc(#[from] QcError),
    #[error(transparent)]
    Crypto(#[from] CryptoError),
    #[error("qc subject does not match the attestation digest")]
    SubjectMismatch,
    #[error("qc epoch {qc} does not match the attested epoch {attested}")]
    EpochMismatch { qc: u64, attested: u64 },
    #[error("attestation chain gap: expected interval {expected}, found {found}")]
    ChainGap { expected: u64, found: u64 },
    #[error("attestation at interval {interval} does not chain to its predecessor")]
    ChainBroken { interval: u64 },
    #[error("signer set unavailable for interval {interval}")]
    SignersUnavailable { interval: u64 },
}

// ---------------------------------------------------------------------------
// The padded Merkle helper (erratum 126)
// ---------------------------------------------------------------------------

pub fn empty_leaf(domain: &Domain) -> Hash256 {
    Hash256::concat(domain, &[0u8; 32])
}

fn node(domain: &Domain, l: &Hash256, r: &Hash256) -> Hash256 {
    let mut msg = [0u8; 64];
    msg[..32].copy_from_slice(l.as_bytes());
    msg[32..].copy_from_slice(r.as_bytes());
    Hash256::concat(domain, &msg)
}

fn padded_len(n: usize) -> usize {
    n.max(1).next_power_of_two()
}

/// The complete binary tree over the pow2-padded leaves (empty → the
/// domain's empty leaf; a single leaf → itself).
pub fn merkle_root(domain: &Domain, leaves: &[Hash256]) -> Hash256 {
    if leaves.is_empty() {
        return empty_leaf(domain);
    }
    if leaves.len() == 1 {
        return leaves[0];
    }
    let mut level: Vec<Hash256> = leaves.to_vec();
    level.resize(padded_len(leaves.len()), empty_leaf(domain));
    while level.len() > 1 {
        level = level.chunks(2).map(|p| node(domain, &p[0], &p[1])).collect();
    }
    level[0]
}

pub fn merkle_witness(domain: &Domain, leaves: &[Hash256], index: usize) -> Option<Vec<Hash256>> {
    if index >= leaves.len() {
        return None;
    }
    if leaves.len() == 1 {
        return Some(Vec::new());
    }
    let mut level: Vec<Hash256> = leaves.to_vec();
    level.resize(padded_len(leaves.len()), empty_leaf(domain));
    let mut siblings = Vec::with_capacity(level.len().trailing_zeros() as usize);
    let mut i = index;
    while level.len() > 1 {
        siblings.push(level[i ^ 1]);
        level = level.chunks(2).map(|p| node(domain, &p[0], &p[1])).collect();
        i /= 2;
    }
    Some(siblings)
}

/// Adversarial input is rejected, never panics: the sibling count must
/// equal the depth implied by `leaf_count`.
pub fn verify_merkle_witness(
    domain: &Domain,
    root: &Hash256,
    leaf_count: usize,
    index: usize,
    leaf: &Hash256,
    siblings: &[Hash256],
) -> bool {
    if leaf_count == 0 || index >= leaf_count {
        return false;
    }
    if leaf_count == 1 {
        return siblings.is_empty() && leaf == root;
    }
    let depth = padded_len(leaf_count).trailing_zeros() as usize;
    if siblings.len() != depth {
        return false;
    }
    let mut cur = *leaf;
    let mut i = index;
    for s in siblings {
        cur = if i & 1 == 0 { node(domain, &cur, s) } else { node(domain, s, &cur) };
        i >>= 1;
    }
    cur == *root
}

// ---------------------------------------------------------------------------
// 𝔾 — the shard-header tree (§4.7, §11.2)
// ---------------------------------------------------------------------------

/// The 𝔾 tree: one leaf per active shard in canonical ShardSet order —
/// the shard's latest header hash in the interval, the domain's empty
/// leaf where a shard produced none (erratum 126).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GTree {
    tips: Vec<Hash256>,
}

impl GTree {
    pub fn new(shard_count: usize) -> GTree {
        GTree { tips: vec![empty_leaf(&BEACON); shard_count] }
    }

    pub fn shard_count(&self) -> usize {
        self.tips.len()
    }

    pub fn set_tip(&mut self, index: usize, tip: Hash256) {
        self.tips[index] = tip;
    }

    pub fn tip(&self, index: usize) -> Hash256 {
        self.tips[index]
    }

    pub fn root(&self) -> Hash256 {
        merkle_root(&BEACON, &self.tips)
    }

    pub fn witness(&self, index: usize) -> Option<Vec<Hash256>> {
        merkle_witness(&BEACON, &self.tips, index)
    }
}

pub fn verify_g_witness(
    root: &Hash256,
    shard_count: usize,
    index: usize,
    tip: &Hash256,
    siblings: &[Hash256],
) -> bool {
    verify_merkle_witness(&BEACON, root, shard_count, index, tip, siblings)
}

// ---------------------------------------------------------------------------
// A_τ — the interval attestation
// ---------------------------------------------------------------------------

/// The interval attestation (§11.2): the QC'd tuple (interval, 𝔾_τ, T_τ
/// root, DA root, previous attestation hash), signed by the interval's
/// 21-member subset of the beacon committee (E-001).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntervalAttestation {
    pub interval: Interval,
    pub g_root: Hash256,
    pub tau_root: Hash256,
    pub da_root: Hash256,
    /// The previous interval attestation's digest; zero for the first.
    pub prev: Hash256,
    pub qc: QuorumCertificate,
}

impl IntervalAttestation {
    fn body_into(interval: Interval, g: &Hash256, tau: &Hash256, da: &Hash256, prev: &Hash256, out: &mut Vec<u8>) {
        out.push(0u8);
        out.extend_from_slice(&interval.as_u64().to_le_bytes());
        out.extend_from_slice(g.as_bytes());
        out.extend_from_slice(tau.as_bytes());
        out.extend_from_slice(da.as_bytes());
        out.extend_from_slice(prev.as_bytes());
    }

    /// The signed content and the chain link (erratum 126).
    pub fn digest(&self) -> Hash256 {
        let mut buf = Vec::with_capacity(137);
        Self::body_into(self.interval, &self.g_root, &self.tau_root, &self.da_root, &self.prev, &mut buf);
        Hash256::concat(&ATT, &buf)
    }

    pub fn build(
        interval: Interval,
        g_root: Hash256,
        tau_root: Hash256,
        da_root: Hash256,
        prev: Hash256,
        signers: &[(usize, &SigningKey)],
        signer_set: &[VerifyingKey],
        quorum: usize,
    ) -> Result<IntervalAttestation, AttestationError> {
        let mut body = Vec::with_capacity(137);
        Self::body_into(interval, &g_root, &tau_root, &da_root, &prev, &mut body);
        let subject = Hash256::concat(&ATT, &body);
        let epoch = interval.epoch();
        let mut vc = VoteCollector::new(epoch, subject);
        for &(i, sk) in signers {
            let sig = sk.sign(&vote_bytes(epoch, &subject))?;
            vc.add(i, sig, signer_set)?;
        }
        Ok(IntervalAttestation {
            interval,
            g_root,
            tau_root,
            da_root,
            prev,
            qc: vc.assemble(quorum)?,
        })
    }

    pub fn validate(
        &self,
        signer_set: &[VerifyingKey],
        quorum: usize,
    ) -> Result<(), AttestationError> {
        if self.qc.subject != self.digest() {
            return Err(AttestationError::SubjectMismatch);
        }
        if self.qc.epoch != self.interval.epoch() {
            return Err(AttestationError::EpochMismatch {
                qc: self.qc.epoch.as_u64(),
                attested: self.interval.epoch().as_u64(),
            });
        }
        validate_qc(&self.qc, signer_set, quorum)?;
        Ok(())
    }
}

/// Walk an attestation suffix from a trusted anchor: contiguity, chain
/// links, and every QC against the interval's signer set (which the
/// caller derives from its committee state — epoch-boundary bookkeeping
/// is the beacon's, part 3).
pub fn verify_chain(
    anchor: Interval,
    anchor_digest: &Hash256,
    attestations: &[IntervalAttestation],
    signers_of: &dyn Fn(Interval) -> Option<Vec<VerifyingKey>>,
    quorum: usize,
) -> Result<(), AttestationError> {
    let mut prev_interval = anchor;
    let mut prev_digest = *anchor_digest;
    for att in attestations {
        if att.interval.as_u64() != prev_interval.as_u64() + 1 {
            return Err(AttestationError::ChainGap {
                expected: prev_interval.as_u64() + 1,
                found: att.interval.as_u64(),
            });
        }
        if att.prev != prev_digest {
            return Err(AttestationError::ChainBroken { interval: att.interval.as_u64() });
        }
        let Some(signers) = signers_of(att.interval) else {
            return Err(AttestationError::SignersUnavailable { interval: att.interval.as_u64() });
        };
        att.validate(&signers, quorum)?;
        prev_interval = att.interval;
        prev_digest = att.digest();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The epoch attestation
// ---------------------------------------------------------------------------

/// The epoch's interval-digest tree root (interval order).
pub fn interval_digests_root(digests: &[Hash256]) -> Hash256 {
    merkle_root(&ATT, digests)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpochAttestation {
    pub epoch: Epoch,
    pub interval_digests_root: Hash256,
    /// The previous epoch attestation's digest; zero for epoch 0.
    pub prev: Hash256,
    pub qc: QuorumCertificate,
}

impl EpochAttestation {
    pub fn digest(&self) -> Hash256 {
        let mut buf = Vec::with_capacity(73);
        buf.push(1u8);
        buf.extend_from_slice(&self.epoch.as_u64().to_le_bytes());
        buf.extend_from_slice(self.interval_digests_root.as_bytes());
        buf.extend_from_slice(self.prev.as_bytes());
        Hash256::concat(&ATT, &buf)
    }

    pub fn build(
        epoch: Epoch,
        interval_digests_root: Hash256,
        prev: Hash256,
        signers: &[(usize, &SigningKey)],
        signer_set: &[VerifyingKey],
        quorum: usize,
    ) -> Result<EpochAttestation, AttestationError> {
        let subject = {
            let probe = EpochAttestation {
                epoch,
                interval_digests_root,
                prev,
                qc: QuorumCertificate {
                    epoch,
                    subject: Hash256::from_bytes([0u8; 32]),
                    signers: 0,
                    signatures: Vec::new(),
                },
            };
            probe.digest()
        };
        let mut vc = VoteCollector::new(epoch, subject);
        for &(i, sk) in signers {
            let sig = sk.sign(&vote_bytes(epoch, &subject))?;
            vc.add(i, sig, signer_set)?;
        }
        Ok(EpochAttestation {
            epoch,
            interval_digests_root,
            prev,
            qc: vc.assemble(quorum)?,
        })
    }

    pub fn validate(&self, signer_set: &[VerifyingKey], quorum: usize) -> Result<(), AttestationError> {
        if self.qc.subject != self.digest() {
            return Err(AttestationError::SubjectMismatch);
        }
        if self.qc.epoch != self.epoch {
            return Err(AttestationError::EpochMismatch {
                qc: self.qc.epoch.as_u64(),
                attested: self.epoch.as_u64(),
            });
        }
        validate_qc(&self.qc, signer_set, quorum)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Codecs
// ---------------------------------------------------------------------------

impl Encode for IntervalAttestation {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.interval.as_u64().to_le_bytes());
        out.extend_from_slice(self.g_root.as_bytes());
        out.extend_from_slice(self.tau_root.as_bytes());
        out.extend_from_slice(self.da_root.as_bytes());
        out.extend_from_slice(self.prev.as_bytes());
        self.qc.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        8 + 4 * 32 + self.qc.encoded_len()
    }
}

impl Decode for IntervalAttestation {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let interval = Interval::from_u64(r.read_u64()?);
        let g_root = Hash256::decode_from(r)?;
        let tau_root = Hash256::decode_from(r)?;
        let da_root = Hash256::decode_from(r)?;
        let prev = Hash256::decode_from(r)?;
        let qc = QuorumCertificate::decode_from(r)?;
        Ok(IntervalAttestation { interval, g_root, tau_root, da_root, prev, qc })
    }
}

impl Encode for EpochAttestation {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.epoch.as_u64().to_le_bytes());
        out.extend_from_slice(self.interval_digests_root.as_bytes());
        out.extend_from_slice(self.prev.as_bytes());
        self.qc.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        8 + 2 * 32 + self.qc.encoded_len()
    }
}

impl Decode for EpochAttestation {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let epoch = Epoch::from_u64(r.read_u64()?);
        let interval_digests_root = Hash256::decode_from(r)?;
        let prev = Hash256::decode_from(r)?;
        let qc = QuorumCertificate::decode_from(r)?;
        Ok(EpochAttestation { epoch, interval_digests_root, prev, qc })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::committee::{attestation_signers, beacon_committee};
    use crate::testutil::harness::{keys, RandomnessMap};
    use crate::testutil::SplitMix64;
    use nerv_core::constants::BEACON;
    use proptest::prelude::*;

    fn digests(seed: u64, n: usize) -> Vec<Hash256> {
        let mut rng = SplitMix64::new(seed);
        (0..n).map(|_| Hash256::from_bytes(rng.bytes32())).collect()
    }

    #[test]
    fn merkle_construction_and_padding() {
        let d = &ATT;
        let e = empty_leaf(d);
        assert_eq!(merkle_root(d, &[]), e);
        let one = digests(1, 1);
        assert_eq!(merkle_root(d, &one), one[0]);
        let two = digests(2, 2);
        let mut pre = Vec::new();
        pre.extend_from_slice(d.as_bytes());
        pre.extend_from_slice(two[0].as_bytes());
        pre.extend_from_slice(two[1].as_bytes());
        assert_eq!(merkle_root(d, &two).as_bytes(), blake3::hash(&pre).as_bytes());
        // 3 leaves == the same tree as 4 with an empty pad.
        let three = digests(3, 3);
        assert_eq!(merkle_root(d, &three), merkle_root(d, &[three[0], three[1], three[2], e]));
        // Order matters.
        assert_ne!(merkle_root(d, &three), merkle_root(d, &[three[1], three[0], three[2]]));
        // The empty leaf is the literal formula.
        let mut ep = Vec::new();
        ep.extend_from_slice(d.as_bytes());
        ep.extend_from_slice(&[0u8; 32]);
        assert_eq!(e.as_bytes(), blake3::hash(&ep).as_bytes());
    }

    #[test]
    fn merkle_witnesses_verify() {
        for n in [2usize, 3, 5, 9, 64, 100] {
            let leaves = digests(0xA0 + n as u64, n);
            let root = merkle_root(&ATT, &leaves);
            for i in 0..n {
                let w = merkle_witness(&ATT, &leaves, i).unwrap();
                assert_eq!(w.len(), padded_len(n).trailing_zeros() as usize);
                assert!(verify_merkle_witness(&ATT, &root, n, i, &leaves[i], &w), "n={n} i={i}");
                let mut bad = w.clone();
                if !bad.is_empty() {
                    let mut b = *bad[0].as_bytes();
                    b[0] ^= 1;
                    bad[0] = Hash256::from_bytes(b);
                    assert!(!verify_merkle_witness(&ATT, &root, n, i, &leaves[i], &bad));
                }
                assert!(!verify_merkle_witness(&ATT, &root, n, (i + 1) % n, &leaves[i], &w));
                assert!(!verify_merkle_witness(
                    &ATT,
                    &Hash256::from_bytes([9u8; 32]),
                    n,
                    i,
                    &leaves[i],
                    &w
                ));
                let mut long = w.clone();
                long.push(empty_leaf(&ATT));
                assert!(!verify_merkle_witness(&ATT, &root, n, i, &leaves[i], &long));
            }
            assert!(merkle_witness(&ATT, &leaves, n).is_none());
            assert!(!verify_merkle_witness(&ATT, &root, n, n, &leaves[0], &[]));
        }
        let one = digests(7, 1);
        assert_eq!(merkle_witness(&ATT, &one, 0).unwrap(), Vec::<Hash256>::new());
        assert!(verify_merkle_witness(&ATT, &one[0], 1, 0, &one[0], &[]));
    }

    #[test]
    fn g_tree_positional_semantics() {
        let mut g = GTree::new(64);
        assert_eq!(g.shard_count(), 64);
        let empty_root = g.root();
        assert_eq!(g.tip(7), empty_leaf(&BEACON));
        let t7 = Hash256::from_bytes([7u8; 32]);
        let t40 = Hash256::from_bytes([40u8; 32]);
        g.set_tip(7, t7);
        g.set_tip(40, t40);
        assert_ne!(g.root(), empty_root);
        let root = g.root();
        for i in [0usize, 7, 40, 63] {
            let w = g.witness(i).unwrap();
            let leaf = if i == 7 { t7 } else if i == 40 { t40 } else { empty_leaf(&BEACON) };
            assert!(verify_g_witness(&root, 64, i, &leaf, &w), "i={i}");
        }
        // The absent-shard sentinel verifies as itself.
        assert!(verify_g_witness(&root, 64, 9, &empty_leaf(&BEACON), &g.witness(9).unwrap()));
        // A wrong shard count rejects.
        assert!(!verify_g_witness(&root, 65, 7, &t7, &g.witness(7).unwrap()));
        assert!(!verify_g_witness(&empty_root, 64, 7, &t7, &g.witness(7).unwrap()));
    }

 use std::sync::OnceLock;

    fn signer_set() -> &'static (Vec<nerv_crypto::mldsa::SigningKey>, Vec<VerifyingKey>) {
        static SET: OnceLock<(Vec<nerv_crypto::mldsa::SigningKey>, Vec<VerifyingKey>)> =
            OnceLock::new();
        SET.get_or_init(|| {
            let (keys, vks) = keys(21);
            (keys, vks)
        })
    }

    fn attestation_fixture(seed: u64, interval: u64, prev: Hash256) -> IntervalAttestation {
        let mut rng = SplitMix64::new(seed);
        let (keys, vks) = signer_set();
        let signers: Vec<(usize, &nerv_crypto::mldsa::SigningKey)> =
            (0..15).map(|i| (i, &keys[i])).collect();
        IntervalAttestation::build(
            Interval::from_u64(interval),
            Hash256::from_bytes(rng.bytes32()),
            Hash256::from_bytes(rng.bytes32()),
            Hash256::from_bytes(rng.bytes32()),
            prev,
            &signers,
            vks,
            15,
        )
        .unwrap()
    }

    #[test]
    fn interval_attestation_build_validate_digest() {
        let zero = Hash256::from_bytes([0u8; 32]);
        let a = attestation_fixture(0xA1, 86_400, zero);
        a.validate(&signer_set().1, 15).unwrap();
        assert_eq!(a.digest(), a.digest());

        // The literal digest formula.
        let mut pre = Vec::new();
        pre.extend_from_slice(ATT.as_bytes());
        let mut body = Vec::new();
        IntervalAttestation::body_into(
            a.interval, &a.g_root, &a.tau_root, &a.da_root, &a.prev, &mut body,
        );
        pre.extend_from_slice(&body);
        assert_eq!(a.digest().as_bytes(), blake3::hash(&pre).as_bytes());

        // Field sensitivity.
        let mut b = a.clone();
        b.g_root = Hash256::from_bytes([1u8; 32]);
        assert_ne!(a.digest(), b.digest());
        let mut b = a.clone();
        b.tau_root = Hash256::from_bytes([2u8; 32]);
        assert_ne!(a.digest(), b.digest());
        let mut b = a.clone();
        b.da_root = Hash256::from_bytes([3u8; 32]);
        assert_ne!(a.digest(), b.digest());
        let mut b = a.clone();
        b.prev = Hash256::from_bytes([4u8; 32]);
        assert_ne!(a.digest(), b.digest());
        let mut b = a.clone();
        b.interval = Interval::from_u64(86_401);
        assert_ne!(a.digest(), b.digest());

        // Wrong roster / quorum / subject / epoch.
        let (_, other) = keys(21 + 9);
        assert!(matches!(a.validate(&other, 15), Err(AttestationError::Qc(_))));
        assert!(matches!(a.validate(&signer_set().1, 16), Err(AttestationError::Qc(_))));
        let mut bad = a.clone();
        bad.qc.subject = Hash256::from_bytes([5u8; 32]);
        assert!(matches!(bad.validate(&signer_set().1, 15), Err(AttestationError::SubjectMismatch)));
        let mut bad = a.clone();
        bad.qc.epoch = Epoch::from_u64(a.qc.epoch.as_u64() + 1);
        assert!(matches!(
            bad.validate(&signer_set().1, 15),
            Err(AttestationError::EpochMismatch { .. })
        ));
    }

    #[test]
    fn chain_verification() {
        let zero = Hash256::from_bytes([0u8; 32]);
        let a = attestation_fixture(0xA2, 86_400, zero);
        let b = attestation_fixture(0xA3, 86_401, a.digest());
        let c = attestation_fixture(0xA4, 86_402, b.digest());
        let set = &signer_set().1;
        verify_chain(Interval::from_u64(86_399), &zero, &[a.clone(), b.clone(), c.clone()], &|_| Some(set.clone()), 15)
            .unwrap();

        // Broken link.
        assert!(matches!(
            verify_chain(Interval::from_u64(86_399), &zero, &[a.clone(), c.clone()], &|_| Some(set.clone()), 15),
            Err(AttestationError::ChainBroken { interval: 86_402 })
        ));
        // Gap.
        assert!(matches!(
            verify_chain(Interval::from_u64(86_399), &zero, &[a.clone(), c.clone(), b.clone()], &|_| Some(set.clone()), 15),
            Err(AttestationError::ChainGap { expected: 86_401, found: 86_402 })
        ));
        // Signers unavailable.
        assert!(matches!(
            verify_chain(Interval::from_u64(86_399), &zero, &[a.clone()], &|_| None, 15),
            Err(AttestationError::SignersUnavailable { interval: 86_400 })
        ));
        // A wrong anchor digest breaks the first link.
        assert!(matches!(
            verify_chain(Interval::from_u64(86_399), &Hash256::from_bytes([9u8; 32]), &[a.clone()], &|_| Some(set.clone()), 15),
            Err(AttestationError::ChainBroken { .. })
        ));
        // Empty suffix: trivially Ok.
        verify_chain(Interval::from_u64(86_399), &zero, &[], &|_| None, 15).unwrap();
    }

    #[test]
    fn epoch_attestation_and_compression() {
        let zero = Hash256::from_bytes([0u8; 32]);
        let a = attestation_fixture(0xA5, 86_400, zero);
        let b = attestation_fixture(0xA6, 86_401, a.digest());
        let digests = vec![a.digest(), b.digest()];
        let root = interval_digests_root(&digests);
        assert_eq!(root, merkle_root(&ATT, &digests));
        assert_ne!(root, interval_digests_root(&digestests_reversed(&digests)));

        let (keys, vks) = signer_set();
        let signers: Vec<(usize, &nerv_crypto::mldsa::SigningKey)> =
            (0..15).map(|i| (i, &keys[i])).collect();
        let e0 = EpochAttestation::build(Epoch::from_u64(0), root, zero, &signers, vks, 15).unwrap();
        e0.validate(vks, 15).unwrap();
        assert_eq!(e0.digest(), e0.digest());
        assert_ne!(e0.digest(), a.digest(), "the type tags separate the namespaces");
        let e1 = EpochAttestation::build(Epoch::from_u64(1), root, e0.digest(), &signers, vks, 15).unwrap();
        e1.validate(vks, 15).unwrap();
        assert_ne!(e0.digest(), e1.digest());

        let mut bad = e0.clone();
        bad.qc.subject = Hash256::from_bytes([7u8; 32]);
        assert!(matches!(bad.validate(vks, 15), Err(AttestationError::SubjectMismatch)));
        let mut bad = e0.clone();
        bad.epoch = Epoch::from_u64(1);
        assert!(matches!(bad.validate(vks, 15), Err(AttestationError::EpochMismatch { .. })));
        let (_, other) = keys(21 + 9);
        assert!(matches!(e0.validate(&other, 15), Err(AttestationError::Qc(_))));
        assert_eq!(interval_digests_root(&[]), empty_leaf(&ATT));
    }

    fn digestests_reversed(d: &[Hash256]) -> Vec<Hash256> {
        let mut v = d.to_vec();
        v.reverse();
        v
    }

    #[test]
    fn codecs_roundtrip() {
        let zero = Hash256::from_bytes([0u8; 32]);
        let a = attestation_fixture(0xA7, 86_400, zero);
        let enc = a.encode();
        assert_eq!(enc.len(), a.encoded_len());
        let dec = IntervalAttestation::decode(&enc).unwrap();
        assert_eq!(dec, a);
        assert_eq!(dec.digest(), a.digest());
        for cut in 0..enc.len() {
            assert!(IntervalAttestation::decode(&enc[..cut]).is_err(), "cut {cut}");
        }
        let mut ext = enc.clone();
        ext.push(0);
        assert!(IntervalAttestation::decode(&ext).is_err());

        let (keys, vks) = signer_set();
        let signers: Vec<(usize, &nerv_crypto::mldsa::SigningKey)> =
            (0..15).map(|i| (i, &keys[i])).collect();
        let e = EpochAttestation::build(
            Epoch::from_u64(2),
            interval_digests_root(&[a.digest()]),
            zero,
            &signers,
            vks,
            15,
        )
        .unwrap();
        let enc = e.encode();
        assert_eq!(enc.len(), e.encoded_len());
        assert_eq!(EpochAttestation::decode(&enc).unwrap(), e);
        assert!(EpochAttestation::decode(&enc[..enc.len() - 1]).is_err());
    }

    #[test]
    fn signed_by_the_real_interval_subset() {
        let (keys, roster) = keys(40);
        let mut rng = SplitMix64::new(0xA8);
        let rmap = RandomnessMap::with(
            (0..5).map(|e| (e, Hash256::from_bytes(rng.bytes32()))).collect(),
        );
        let epoch = Epoch::from_u64(3);
       let f = |e: Epoch| rmap.get(e);
       let beacon = beacon_committee(epoch, &roster, &f).unwrap();
        let interval = Interval::from_u64(3 * 86_400 + 42);
        let subset = attestation_signers(&beacon, &rmap.map[&3], interval);
        assert_eq!(subset.len(), 21);
        // Map subset pubkeys to their signing keys.
        let by_pk: std::collections::BTreeMap<[u8; nerv_crypto::mldsa::PK_LEN], usize> =
            roster.iter().enumerate().map(|(i, vk)| (*vk.as_bytes(), i)).collect();
        let subset_keys: Vec<nerv_crypto::mldsa::SigningKey> = subset
            .iter()
            .map(|vk| keys[by_pk[vk.as_bytes()]].clone())
            .collect();
        let signers: Vec<(usize, &nerv_crypto::mldsa::SigningKey)> =
            (0..15).map(|i| (i, &subset_keys[i])).collect();
        let att = IntervalAttestation::build(
            interval,
            GTree::new(64).root(),
            Hash256::from_bytes(rng.bytes32()),
            Hash256::from_bytes(rng.bytes32()),
            Hash256::from_bytes([0u8; 32]),
            &signers,
            &subset,
            15,
        )
        .unwrap();
        att.validate(&subset, 15).unwrap();
        // Not valid against the full beacon roster with a different index
        // mapping: the subset IS the committee for this attestation.
        let mut wrong: Vec<VerifyingKey> = subset.clone();
        wrong.reverse();
        assert!(matches!(att.validate(&wrong, 15), Err(AttestationError::Qc(_))));
    }

    proptest! {
        #[test]
        fn prop_merkle_witness(n in 2usize..33, seed in any::<u64>()) {
            let mut rng = SplitMix64::new(seed);
            let leaves: Vec<Hash256> =
                (0..n).map(|_| Hash256::from_bytes(rng.bytes32())).collect();
            let root = merkle_root(&ATT, &leaves);
            for i in 0..n {
                let w = merkle_witness(&ATT, &leaves, i).unwrap();
                prop_assert!(verify_merkle_witness(&ATT, &root, n, i, &leaves[i], &w));
            }
        }
    }
}

