//! The aggregator's wire submission (WP §5.5 tier 1; erratum 120): the
//! transaction list, the txid-set commitment, and the staked ML-DSA
//! attestation — the object slashed on failed verification.


use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::BUNDLE_ATTEST;
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::types::TxId;
use nerv_crypto::mldsa::{Signature, SigningKey, VerifyingKey};
use nerv_state::ttau::TauTree;


use crate::error::{BundleError, VerificationError};
use crate::mempool::PoolEntry;


pub const BUNDLE_MIN: usize = nerv_core::params::REGISTRY_BUNDLE_MIN as usize;
pub const BUNDLE_MAX: usize = nerv_core::params::REGISTRY_BUNDLE_MAX as usize;


/// The signed message: domain ‖ txid_root ‖ count (u32 LE).
pub fn bundle_attest_message(root: &Hash256, count: u32) -> Vec<u8> {
    let mut m = Vec::with_capacity(BUNDLE_ATTEST.as_bytes().len() + 36);
    m.extend_from_slice(BUNDLE_ATTEST.as_bytes());
    m.extend_from_slice(root.as_bytes());
    m.extend_from_slice(&count.to_le_bytes());
    m
}


#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BundleAttestation {
    pub txid_root: Hash256,
    pub count: u32,
    pub signature: Signature,
}


impl BundleAttestation {
    pub fn build(sk: &SigningKey, txid_root: Hash256, count: u32) -> Result<Self, BundleError> {
        Ok(BundleAttestation {
            txid_root,
            count,
            signature: sk.sign(&bundle_attest_message(&txid_root, count))?,
        })
    }


    pub fn verify(&self, vk: &VerifyingKey) -> bool {
        vk.verify(&bundle_attest_message(&self.txid_root, self.count), &self.signature)
    }
}


impl Encode for BundleAttestation {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self.txid_root.as_bytes());
        out.extend_from_slice(&self.count.to_le_bytes());
        self.signature.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        36 + self.signature.encoded_len()
    }
}


impl Decode for BundleAttestation {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let txid_root = Hash256::decode_from(r)?;
        let count = r.read_u32()?;
        let signature = Signature::decode_from(r)?;
        Ok(BundleAttestation { txid_root, count, signature })
    }
}


/// The arrival-order summary the interval commit carries — the slashing
/// evidence binding for a failed bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BundleSummary {
    pub aggregator: VerifyingKey,
    pub root: Hash256,
    pub count: u32,
    pub signature: Signature,
}


impl Encode for BundleSummary {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.aggregator.encode_into(out);
        out.extend_from_slice(self.root.as_bytes());
        out.extend_from_slice(&self.count.to_le_bytes());
        self.signature.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        self.aggregator.encoded_len() + 36 + self.signature.encoded_len()
    }
}


impl Decode for BundleSummary {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let aggregator = VerifyingKey::decode_from(r)?;
        let root = Hash256::decode_from(r)?;
        let count = r.read_u32()?;
        let signature = Signature::decode_from(r)?;
        Ok(BundleSummary { aggregator, root, count, signature })
    }
}


/// One bundle: the transactions (submission order) and the aggregator's
/// attestation over their txid-set root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bundle {
    pub aggregator: VerifyingKey,
    pub transactions: Vec<PoolEntry>,
    pub attestation: BundleAttestation,
}


impl Bundle {
    /// The aggregator's build path: structural checks, the set root, the
    /// staked signature.
    pub fn build(sk: &SigningKey, transactions: Vec<PoolEntry>) -> Result<Bundle, BundleError> {
        if transactions.is_empty() {
            return Err(BundleError::Empty);
        }
        if transactions.len() > BUNDLE_MAX {
            return Err(BundleError::TooLarge { count: transactions.len(), max: BUNDLE_MAX });
        }
        let txids = derive_txids(&transactions)?;
        let root = Self::set_root(&txids)?;
        let count = transactions.len() as u32;
        Ok(Bundle {
            aggregator: *sk.verifying_key(),
            attestation: BundleAttestation::build(sk, root, count)?,
            transactions,
        })
    }


    fn set_root(txids: &[TxId]) -> Result<Hash256, BundleError> {
   let mut sorted = txids.to_vec();
   sorted.sort();
   sorted.dedup();
   Ok(TauTree::from_sorted(&sorted)?.root())
}


    /// The submission-order txid list (fold::dedup's `BundleTxids` input).
    pub fn txids(&self) -> Vec<TxId> {
        self.transactions.iter().map(|e| e.txid).collect()
    }


    pub fn txid_root(&self) -> Hash256 {
        self.attestation.txid_root
    }


    pub fn summary(&self) -> BundleSummary {
        BundleSummary {
            aggregator: self.aggregator,
            root: self.attestation.txid_root,
            count: self.attestation.count,
            signature: self.attestation.signature.clone(),
        }
    }


    /// Structural validation with an explicit maximum (the registry's
    /// intake policy; the default is BUNDLE_MAX — erratum 120).
    pub fn validate_structure_with_max(&self, max: usize) -> Result<(), BundleError> {
        if self.transactions.is_empty() {
            return Err(BundleError::Empty);
        }
        if self.transactions.len() > max {
            return Err(BundleError::TooLarge { count: self.transactions.len(), max });
        }
        let txids = derive_txids(&self.transactions)?;
        let root = Self::set_root(&txids)?;
        if root != self.attestation.txid_root {
            return Err(BundleError::RootMismatch { expected: root, found: self.attestation.txid_root });
        }
        if self.attestation.count as usize != self.transactions.len() {
            return Err(BundleError::CountMismatch {
                count: self.attestation.count as usize,
                transactions: self.transactions.len(),
            });
        }
        if !self.attestation.verify(&self.aggregator) {
            return Err(BundleError::BadSignature);
        }
        Ok(())
    }


    pub fn validate_structure(&self) -> Result<(), BundleError> {
        self.validate_structure_with_max(BUNDLE_MAX)
    }
}


fn derive_txids(transactions: &[PoolEntry]) -> Result<Vec<TxId>, BundleError> {
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::with_capacity(transactions.len());
    for (i, e) in transactions.iter().enumerate() {
        let canon = e.shell.canonicalize()?;
        let txid = nerv_state::canonical_txid(&canon);
        if txid != e.txid {
            return Err(BundleError::TxidMismatch { index: i, txid: e.txid });
        }
        if !seen.insert(txid) {
            return Err(BundleError::DuplicateTxid { txid });
        }
        out.push(txid);
    }
    Ok(out)
}


/// The registry committee's bundle gate (WP §5.5 tier 2; erratum 120):
/// structure, then per-transaction proof verification. The first failing
/// transaction identifies the slash evidence.
pub fn verify_bundle(
    bundle: &Bundle,
    ctx: &crate::mempool::VerifyContext,
) -> Result<(), BundleError> {
    bundle.validate_structure()?;
    for (i, e) in bundle.transactions.iter().enumerate() {
        ctx.verify(&e.shell, &e.proof)
            .map_err(|source| BundleError::Verification { index: i, source })?;
    }
    Ok(())
}


impl Encode for Bundle {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.aggregator.encode_into(out);
        out.extend_from_slice(&(self.transactions.len() as u32).to_le_bytes());
        for t in &self.transactions {
            t.encode_into(out);
        }
        self.attestation.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        self.aggregator.encoded_len()
            + 4
            + self.transactions.iter().map(|t| t.encoded_len()).sum::<usize>()
            + self.attestation.encoded_len()
    }
}


impl Decode for Bundle {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let aggregator = VerifyingKey::decode_from(r)?;
        let n = r.read_seq_len()?;
        if n > BUNDLE_MAX {
            return Err(CodecError::SeqTooLarge { count: n, max: BUNDLE_MAX });
        }
        let mut transactions = Vec::with_capacity(n);
        for _ in 0..n {
            transactions.push(PoolEntry::decode_from(r)?);
        }
        let attestation = BundleAttestation::decode_from(r)?;
        Ok(Bundle { aggregator, transactions, attestation })
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::mempool::Mempool;
    use crate::testutil::harness::{agg_key, ctx, shell, shallow_proof, txid_of};
    use crate::testutil::SplitMix64;
    use nerv_core::constants::BUNDLE_ATTEST;


    fn entries(seeds: &[u64]) -> Vec<PoolEntry> {
        seeds
            .iter()
            .map(|&s| {
                let canon = shell(s).canonicalize().unwrap();
                PoolEntry { txid: nerv_state::canonical_txid(&canon), shell: canon, proof: shallow_proof() }
            })
            .collect()
    }


    #[test]
    fn build_validate_root_and_signature() {
        let sk = agg_key(1);
        let es = entries(&[1, 2, 3]);
        let root_source = {
            let mut t = es.iter().map(|e| e.txid).collect::<Vec<_>>();
            t.sort();
            TauTree::from_sorted(&t).unwrap().root()
        };
        let b = Bundle::build(&sk, es).unwrap();
        assert_eq!(b.txid_root(), root_source);
        assert_eq!(b.attestation.count, 3);
        assert_eq!(b.txids().len(), 3);
        b.validate_structure().unwrap();


        // The signed message is the literal formula.
        let mut m = Vec::new();
        m.extend_from_slice(BUNDLE_ATTEST.as_bytes());
        m.extend_from_slice(root_source.as_bytes());
        m.extend_from_slice(&3u32.to_le_bytes());
        assert!(sk.verifying_key().verify(&m, &b.attestation.signature));
        assert_eq!(b.attestation.verify(&sk.verifying_key()), true);


        // The summary matches the attestation.
        let s = b.summary();
        assert_eq!(s.root, root_source);
        assert_eq!(s.count, 3);
        assert_eq!(s.aggregator, *sk.verifying_key());
    }


    #[test]
    fn empty_and_cap_rejected() {
        let sk = agg_key(2);
        assert!(matches!(Bundle::build(&sk, vec![]), Err(BundleError::Empty)));
        let es = entries(&[10, 11, 12, 13, 14]);
        let b = Bundle::build(&sk, es).unwrap();
        assert!(matches!(
            b.validate_structure_with_max(4),
            Err(BundleError::TooLarge { count: 5, max: 4 })
        ));
        b.validate_structure_with_max(5).unwrap();
        assert!(BUNDLE_MIN >= 1 && BUNDLE_MAX >= BUNDLE_MIN);
    }


    #[test]
    fn duplicate_txid_rejected() {
        let sk = agg_key(3);
        let mut es = entries(&[20, 21]);
        es.push(es[0].clone());
        assert!(matches!(
            Bundle::build(&sk, es),
            Err(BundleError::DuplicateTxid { .. })
        ));
    }


    #[test]
    fn tampered_fields_rejected() {
        let sk = agg_key(4);
        let es = entries(&[30, 31]);
        let b = Bundle::build(&sk, es).unwrap();


        let mut bad = b.clone();
        bad.attestation.txid_root = Hash256::from_bytes([0xEE; 32]);
        assert!(matches!(bad.validate_structure(), Err(BundleError::RootMismatch { .. })));


        let mut bad = b.clone();
        bad.attestation.count = 9;
        assert!(matches!(bad.validate_structure(), Err(BundleError::CountMismatch { .. })));


        // Swap one transaction for another shell: txid mismatch.
        let mut bad = b.clone();
        let other = shell(99).canonicalize().unwrap();
        bad.transactions[0].shell = other;
        assert!(matches!(
            bad.validate_structure(),
            Err(BundleError::TxidMismatch { index: 0, .. })
        ));


        // A different aggregator's key: signature fails.
        let mut bad = b.clone();
        bad.aggregator = *agg_key(5).verifying_key();
        assert!(matches!(bad.validate_structure(), Err(BundleError::BadSignature)));


        // Tampered signature bytes.
        let mut bad = b.clone();
        let mut sig = bad.attestation.signature.as_bytes().clone();
        sig[100] ^= 1;
        bad.attestation.signature = Signature::from_bytes(
            sig.try_into().unwrap(),
        );
        assert!(matches!(bad.validate_structure(), Err(BundleError::BadSignature)));


        b.validate_structure().unwrap();
    }


    #[test]
    fn verify_bundle_runs_the_real_gate() {
        let ctx = ctx();
        let sk = agg_key(6);
        let b = Bundle::build(&sk, entries(&[40, 41])).unwrap();
        assert!(matches!(
            verify_bundle(&b, &ctx),
            Err(BundleError::Verification { index: 0, source: VerificationError::ProofRejected })
        ));
        // The failure leaves the structure valid (slash evidence is the
        // bundle + the failed index).
        b.validate_structure().unwrap();
    }


    #[test]
    fn codec_roundtrip_and_strictness() {
        let sk = agg_key(7);
        let b = Bundle::build(&sk, entries(&[50, 51, 52])).unwrap();
        let enc = b.encode();
        assert_eq!(enc.len(), b.encoded_len());
        let dec = Bundle::decode(&enc).unwrap();
        assert_eq!(dec, b);
        dec.validate_structure().unwrap();
        for cut in 0..enc.len() {
            assert!(Bundle::decode(&enc[..cut]).is_err(), "cut {cut}");
        }
        let mut ext = enc.clone();
        ext.push(0);
        assert!(Bundle::decode(&ext).is_err());
    }


    #[test]
    fn mempool_to_bundle_end_to_end() {
        let mut pool = Mempool::new(64);
        for s in 60..68u64 {
            pool.insert(shell(s), shallow_proof()).unwrap();
        }
        let sel = pool.select_bundle(5);
        assert_eq!(sel.len(), 5);
        let b = Bundle::build(&agg_key(8), sel).unwrap();
        assert_eq!(b.transactions.len(), 5);
        b.validate_structure().unwrap();
        // The bundle's txid set is a prefix of the pool's canonical order.
        let pool_ids: Vec<TxId> = pool.txids().collect();
        assert_eq!(b.txids(), pool_ids[..5].to_vec());
        let _ = SplitMix64::new(1);
        let _ = txid_of(&shell(60));
    }
}
