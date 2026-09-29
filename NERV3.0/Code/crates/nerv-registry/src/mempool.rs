//! The aggregator's validated-tx pool (WP §5.5 tier 1; erratum 119):
//! structural admission, txid dedup, canonical-order selection. Proof
//! verification is the registry's bundle gate — it re-establishes every
//! contained proof independently, so the pool's discretion is not
//! load-bearing; `admit` runs the full gate as the production path.


use std::collections::BTreeMap;


use nerv_codec::codec_w::CodecW;
use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::types::TxId;
use nerv_custody::tx::TransactionShell;
use nerv_proofs::{
    bind_transaction, canonical_nullifiers, shell_digest, verify_transaction, FsTranscript,
    FriShape, TransactionProof, TxPublicInputs,
};
use nerv_seal::circuit_stmt::epoch_key_identifier;
use nerv_seal::encrypt::PublicKey;


use crate::error::{MempoolError, VerificationError};


/// The verification context: everything a transaction proof is checked
/// against — the FRI shape, the epoch's frozen codec W, and the epoch's
/// seal public key.
#[derive(Clone)]
pub struct VerifyContext {
    fri: FriShape,
    w: CodecW,
    epoch_pk: PublicKey,
    epoch_key_id: Hash256,
}


impl VerifyContext {
    pub fn new(fri: FriShape, w: CodecW, epoch_pk: PublicKey) -> VerifyContext {
        let epoch_key_id =
            Hash256::from_bytes(epoch_key_identifier(epoch_pk.a_seed(), epoch_pk.t()));
        VerifyContext { fri, w, epoch_pk, epoch_key_id }
    }


    pub fn fri(&self) -> &FriShape {
        &self.fri
    }


    pub fn w(&self) -> &CodecW {
        &self.w
    }


    pub fn epoch_pk(&self) -> &PublicKey {
        &self.epoch_pk
    }


    pub fn epoch_key_id(&self) -> &Hash256 {
        &self.epoch_key_id
    }


    /// THE gate (erratum 119): the statement-11 binding, then the
    /// transaction STARK's verification against the anchored public
    /// inputs.
    pub fn verify(
        &self,
        shell: &TransactionShell,
        proof: &TransactionProof,
    ) -> Result<(), VerificationError> {
        let canon = shell.canonicalize()?;
        for leg in &canon.legs {
            if leg.weight_version != self.w.version().0 {
                return Err(VerificationError::WeightVersion {
                    shell: leg.weight_version,
                    codec: self.w.version().0,
                });
            }
        }
        let txid = nerv_state::canonical_txid(&canon);
        let digest = shell_digest(&canon)?;
        let nullifiers = canonical_nullifiers(&canon)?;
        let public = TxPublicInputs::derive(&canon, self.epoch_key_id)?;
        let (mut t, _, _) = bind_transaction(&nullifiers, &txid, &digest, &public);
        match verify_transaction(&self.fri, &canon, &self.w, &self.epoch_pk, proof, &mut t)? {
            true => Ok(()),
            false => Err(VerificationError::ProofRejected),
        }
    }
}


/// One pooled transaction: the canonical shell, its txid, and the proof
/// the aggregator verified at intake.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolEntry {
    pub shell: TransactionShell,
    pub txid: TxId,
    pub proof: TransactionProof,
}


impl Encode for PoolEntry {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.shell.encode_into(out);
        self.proof.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        self.shell.encoded_len() + self.proof.encoded_len()
    }
}


impl Decode for PoolEntry {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let shell = TransactionShell::decode_from(r)?;
        let proof = TransactionProof::decode_from(r)?;
        let canon = shell
            .canonicalize()
            .map_err(|_| CodecError::InvariantViolated("pooled shell failed canonicalization"))?;
        Ok(PoolEntry { txid: nerv_state::canonical_txid(&canon), shell: canon, proof })
    }
}


/// The aggregator's default pool capacity: four maximum bundles in flight.
pub const DEFAULT_CAPACITY: usize = 4 * crate::BUNDLE_MAX;


/// The validated-tx pool: txid dedup, capacity admission, canonical-order
/// selection and post-finalization pruning.
#[derive(Clone, Debug)]
pub struct Mempool {
    entries: BTreeMap<TxId, PoolEntry>,
    capacity: usize,
}


impl Default for Mempool {
    fn default() -> Self {
        Mempool::new(DEFAULT_CAPACITY)
    }
}


impl Mempool {
    pub fn new(capacity: usize) -> Mempool {
        Mempool { entries: BTreeMap::new(), capacity }
    }


    pub fn capacity(&self) -> usize {
        self.capacity
    }


    pub fn len(&self) -> usize {
        self.entries.len()
    }


    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }


    pub fn contains(&self, txid: &TxId) -> bool {
        self.entries.contains_key(txid)
    }


    pub fn get(&self, txid: &TxId) -> Option<&PoolEntry> {
        self.entries.get(txid)
    }


    /// Canonical byte order — the T_τ leaf order.
    pub fn txids(&self) -> impl Iterator<Item = TxId> + '_ {
        self.entries.keys().copied()
    }


    pub fn entries(&self) -> impl Iterator<Item = &PoolEntry> {
        self.entries.values()
    }


    /// Structural admission (erratum 119): canonicalize, dedup, capacity.
    /// The caller has verified the proof; the registry's bundle gate
    /// re-verifies independently.
    pub fn insert(
        &mut self,
        shell: TransactionShell,
        proof: TransactionProof,
    ) -> Result<bool, MempoolError> {
        let canon = shell.canonicalize().map_err(VerificationError::from)?;
        let txid = nerv_state::canonical_txid(&canon);
        if self.entries.contains_key(&txid) {
            return Ok(false);
        }
        if self.entries.len() >= self.capacity {
            return Err(MempoolError::Full { count: self.entries.len(), capacity: self.capacity });
        }
        self.entries.insert(txid, PoolEntry { shell: canon, txid, proof });
        Ok(true)
    }


    /// The production gate: full proof verification, then pooling.
    /// Idempotent on duplicates (the duplicate check precedes the gate).
    pub fn admit(
        &mut self,
        ctx: &VerifyContext,
        shell: TransactionShell,
        proof: TransactionProof,
    ) -> Result<bool, MempoolError> {
        let canon = shell.canonicalize().map_err(VerificationError::from)?;
        let txid = nerv_state::canonical_txid(&canon);
        if self.entries.contains_key(&txid) {
            return Ok(false);
        }
        ctx.verify(&canon, &proof)?;
        self.insert(shell, proof)
    }


    /// Canonical-order bundle candidates (within-bundle order is
    /// irrelevant to dedup — fold::dedup).
    pub fn select_bundle(&self, max: usize) -> Vec<PoolEntry> {
        self.entries.values().take(max).cloned().collect()
    }


    pub fn remove(&mut self, txid: &TxId) -> Option<PoolEntry> {
        self.entries.remove(txid)
    }


    /// Post-finalization pruning: drop every txid settled in a finalized
    /// interval set. Returns the number removed.
    pub fn prune_finalized(&mut self, set: &nerv_proofs::IntervalSet) -> usize {
        let before = self.entries.len();
        self.entries.retain(|txid, _| !set.contains(txid));
        before - self.entries.len()
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::harness::{ctx, deep_garbage, shell, shell2, shallow_proof, txid_of};
    use crate::testutil::SplitMix64;
    use proptest::prelude::*;


    #[test]
    fn insert_canonicalizes_and_dedups() {
        let mut pool = Mempool::new(100);
        let s = shell(1);
        assert!(pool.insert(s.clone(), shallow_proof()).unwrap());
        assert_eq!(pool.len(), 1);
        assert!(!pool.insert(s.clone(), shallow_proof()).unwrap(), "same shell → same txid");
        assert_eq!(pool.len(), 1);


        // Reordered legs canonicalize identically → the same txid.
        let two = shell2(2);
        let mut reversed = two.clone();
        reversed.legs.reverse();
        assert_eq!(txid_of(&two), txid_of(&reversed));
        assert!(pool.insert(two, shallow_proof()).unwrap());
        assert!(!pool.insert(reversed, shallow_proof()).unwrap());
        assert_eq!(pool.len(), 2);


        // Uncanonicalizable shells (duplicate shard legs) are rejected.
        let mut dup = shell(3);
        let clone_leg = dup.legs[0].clone();
        dup.legs.push(clone_leg);
        assert!(pool.insert(dup, shallow_proof()).is_err());
        assert_eq!(pool.len(), 2);
    }


    #[test]
    fn capacity_full_rejected() {
        let mut pool = Mempool::new(2);
        assert!(pool.insert(shell(4), shallow_proof()).unwrap());
        assert!(pool.insert(shell(5), shallow_proof()).unwrap());
        assert!(matches!(
            pool.insert(shell(6), shallow_proof()),
            Err(MempoolError::Full { count: 2, capacity: 2 })
        ));
        // A duplicate is a no-op even when full.
        assert!(!pool.insert(shell(4), shallow_proof()).unwrap());
        assert_eq!(pool.len(), 2);
    }


    #[test]
    fn selection_order_removal_and_pruning() {
        let mut pool = Mempool::new(100);
        let mut txids = Vec::new();
        for seed in 0..6u64 {
            let s = shell(10 + seed);
            txids.push(txid_of(&s));
            pool.insert(s, shallow_proof()).unwrap();
        }
        let mut sorted = txids.clone();
        sorted.sort();
        assert_eq!(pool.txids().collect::<Vec<_>>(), sorted);


        let sel = pool.select_bundle(3);
        assert_eq!(sel.len(), 3);
        assert_eq!(sel.iter().map(|e| e.txid).collect::<Vec<_>>(), sorted[..3].to_vec());
        assert_eq!(pool.select_bundle(10).len(), 6);
        assert!(pool.select_bundle(0).is_empty());


        pool.remove(&sorted[1]).unwrap();
        assert!(!pool.contains(&sorted[1]));
        assert_eq!(pool.len(), 5);


        // Pruning by a finalized interval set.
        let mut set = nerv_proofs::IntervalSet {
            interval: nerv_core::types::Interval::from_u64(9),
            txids: Default::default(),
        };
        set.txids.insert(sorted[0], 0);
        set.txids.insert(sorted[4], 1);
        assert_eq!(pool.prune_finalized(&set), 2);
        assert!(!pool.contains(&sorted[0]) && !pool.contains(&sorted[4]));
        assert_eq!(pool.len(), 3);
        assert_eq!(pool.prune_finalized(&set), 0);
    }


    #[test]
    fn admit_rejects_shallow_garbage() {
        let ctx = ctx();
        let mut pool = Mempool::new(100);
        let s = shell(20);
        assert!(matches!(
            pool.admit(&ctx, s.clone(), shallow_proof()),
            Err(MempoolError::Verification(VerificationError::ProofRejected))
        ));
        assert!(pool.is_empty());
        assert!(!pool.contains(&txid_of(&s)));
    }


    #[test]
    fn admit_rejects_weight_version_mismatch() {
        let ctx = ctx();
        let mut s = shell(21);
        s.legs[0].weight_version = 7;
        assert!(matches!(
            ctx.verify(&s, &shallow_proof()),
            Err(VerificationError::WeightVersion { shell: 7, codec: 1 })
        ));
        let mut pool = Mempool::new(10);
        assert!(matches!(
            pool.admit(&ctx, s, shallow_proof()),
            Err(MempoolError::Verification(VerificationError::WeightVersion { .. }))
        ));
        assert!(pool.is_empty());
    }


    /// The full-pipeline rejection: publics crafted so the shell ct-binding
    /// PASSES, the statement shape matches — verification runs gen_tx_prep
    /// and the engine, and rejects the proof at the STARK layer.
    #[test]
    fn admit_rejects_shape_consistent_garbage() {
        let ctx = ctx();
        let s = shell(22);
        let canon = s.canonicalize().unwrap();
        let proof = deep_garbage(&canon);
        assert!(matches!(
            ctx.verify(&canon, &proof),
            Err(VerificationError::ProofRejected)
        ));
        let mut pool = Mempool::new(10);
        assert!(matches!(
            pool.admit(&ctx, s, proof),
            Err(MempoolError::Verification(VerificationError::ProofRejected))
        ));
        assert!(pool.is_empty());
    }


    #[test]
    fn pool_entry_and_proof_codec_roundtrip() {
        let s = shell(23).canonicalize().unwrap();
        let entry = PoolEntry { txid: txid_of(&s), shell: s, proof: shallow_proof() };
        let enc = entry.encode();
        assert_eq!(enc.len(), entry.encoded_len());
        let dec = PoolEntry::decode(&enc).unwrap();
        assert_eq!(dec, entry);
        for cut in 0..enc.len() {
            assert!(PoolEntry::decode(&enc[..cut]).is_err(), "cut {cut}");
        }
        let mut ext = enc.clone();
        ext.push(0);
        assert!(PoolEntry::decode(&ext).is_err());
    }


    #[test]
    fn proof_codec_validates_shape() {
        let p = shallow_proof();
        let enc = p.encode();
        assert_eq!(TransactionProof::decode(&enc).unwrap(), p);
        let mut bad = enc.clone();
        bad[0..4].copy_from_slice(&0u32.to_le_bytes());
        assert!(TransactionProof::decode(&bad).is_err());
        let mut bad = enc.clone();
        bad[0..4].copy_from_slice(&33u32.to_le_bytes());
        assert!(TransactionProof::decode(&bad).is_err());
        let mut bad = enc.clone();
        bad[4..8].copy_from_slice(&0u32.to_le_bytes());
        assert!(TransactionProof::decode(&bad).is_err());
        assert!(TransactionProof::decode(&enc[..enc.len() - 1]).is_err());
    }


    proptest! {
        #[test]
        fn prop_insertion_order_invariant(seeds in prop::collection::vec(any::<u64>(), 0..40)) {
            let mut a = Mempool::new(1024);
            let mut b = Mempool::new(1024);
            for &s in &seeds {
                let _ = a.insert(shell(s), shallow_proof());
            }
            for &s in seeds.iter().rev() {
                let _ = b.insert(shell(s), shallow_proof());
            }
            let ia: Vec<TxId> = a.txids().collect();
            let ib: Vec<TxId> = b.txids().collect();
            prop_assert_eq!(ia, ib);
            prop_assert!(ia.windows(2).all(|w| w[0] < w[1]));
            let sa = a.select_bundle(1000);
            let sb = b.select_bundle(1000);
            prop_assert_eq!(sa.iter().map(|e| e.txid).collect::<Vec<_>>(),
                            sb.iter().map(|e| e.txid).collect::<Vec<_>>());
        }
    }


    #[test]
    fn harness_shells_are_distinct_and_two_legged() {
        let mut rng = SplitMix64::new(99);
        let _ = &mut rng;
        assert_ne!(txid_of(&shell(30)), txid_of(&shell(31)));
        assert_ne!(txid_of(&shell(30)), txid_of(&shell2(30)));
        let two = shell2(31);
        assert_eq!(two.legs.len(), 2);
        assert_ne!(two.legs[0].shard, two.legs[1].shard);
    }
}
