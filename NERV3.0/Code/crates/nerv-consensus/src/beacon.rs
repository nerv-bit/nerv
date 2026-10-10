//! The beacon state machine (WP §4.7, §11.2; erratum 133): per-shard
//! chain guards, the 𝔾 tree, interval attestations, epoch boundaries,
//! and randomness derivation.


use std::collections::BTreeMap;


use nerv_core::types::{Epoch, Interval, ShardId, ShardSet};
use nerv_crypto::mldsa::{SigningKey, VerifyingKey};
use nerv_proofs::{IntervalLedger, IntervalSet};


use crate::attestation::{
    interval_digests_root, EpochAttestation, GTree, IntervalAttestation,
};
use crate::committee::{attestation_signers, ATTESTATION_QUORUM};
use crate::finality::{
    FinalityViolation, IntervalFinality, ObserveOutcome, ShardFinality,
};


/// The beacon's per-shard genesis: C₀ for each shard.
pub type GenesisMap = BTreeMap<ShardId, nerv_core::hash::Hash256>;


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum BeaconError {
    #[error(transparent)]
    Interval(#[from] crate::finality::FinalityError),
    #[error(transparent)]
    Attestation(#[from] crate::attestation::AttestationError),
    #[error(transparent)]
    Registry(#[from] nerv_registry::IntervalError),
    #[error("shard {shard} is not active")]
    InactiveShard { shard: ShardId },
    #[error("interval {interval} is not the next after {last}")]
    NotNextInterval { interval: u64, last: u64 },
    #[error("interval {interval} is not in epoch {epoch}")]
    IntervalNotInEpoch { interval: u64, epoch: u64 },
    #[error("no interval attestations in epoch {epoch}")]
    EmptyEpoch { epoch: u64 },
    #[error(transparent)]
    Dedup(#[from] nerv_proofs::DedupError),
}


/// The beacon's state: per-shard guards, the interval watermark, the
/// registry ledger, and the attestation chain.
#[derive(Clone, Debug)]
pub struct BeaconState {
    active: ShardSet,
    guards: BTreeMap<ShardId, ShardFinality>,
    interval_finality: IntervalFinality,
    ledger: IntervalLedger,
    current_epoch: Epoch,
    epoch_interval_digests: Vec<nerv_core::hash::Hash256>,
    epoch_attestations: BTreeMap<Epoch, EpochAttestation>,
    interval_attestations: Vec<IntervalAttestation>,
}


impl BeaconState {
    pub fn genesis(active: ShardSet, genesis_cs: &GenesisMap) -> BeaconState {
        let guards = active
            .ids()
            .iter()
            .map(|id| {
                let c0 = genesis_cs.get(id).copied().unwrap_or_default();
                (*id, ShardFinality::new(*id, c0))
            })
            .collect();
        BeaconState {
            active,
            guards,
            interval_finality: IntervalFinality::new(),
            ledger: IntervalLedger::new(),
            current_epoch: Epoch::from_u64(0),
            epoch_interval_digests: Vec::new(),
            epoch_attestations: BTreeMap::new(),
            interval_attestations: Vec::new(),
        }
    }


    pub fn active(&self) -> &ShardSet {
        &self.active
    }


    pub fn current_epoch(&self) -> Epoch {
        self.current_epoch
    }


    pub fn last_finalized_interval(&self) -> Option<Interval> {
        self.interval_finality.last()
    }


    pub fn interval_attestations(&self) -> &[IntervalAttestation] {
        &self.interval_attestations
    }


    pub fn epoch_attestations(&self) -> &BTreeMap<Epoch, EpochAttestation> {
        &self.epoch_attestations
    }


    pub fn ledger(&self) -> &IntervalLedger {
        &self.ledger
    }


    pub fn guard(&self, shard: &ShardId) -> Option<&ShardFinality> {
        self.guards.get(shard)
    }


    /// The shard's tip (height, header hash) or None.
    pub fn shard_tip(&self, shard: &ShardId) -> Option<(u64, nerv_core::hash::Hash256)> {
        self.guards.get(shard).and_then(|g| g.tip())
    }


    /// Observe a QC-validated header (the caller's pipeline validates
    /// the QC before feeding the guard). Per erratum 103, the guard's
    /// parent linkage is the predecessor's C_t (`prev`) and the new
    /// header's own committed C_t is `link`. Both are required.
    pub fn observe_header(
        &mut self,
        shard: ShardId,
        height: u64,
        hash: nerv_core::hash::Hash256,
        prev: nerv_core::hash::Hash256,
        link: nerv_core::hash::Hash256,
    ) -> Result<ObserveOutcome, BeaconError> {
        let guard = self
            .guards
            .get_mut(&shard)
            .ok_or(BeaconError::InactiveShard { shard })?;
        Ok(guard.observe(height, hash, prev, link)?)
    }


    /// Finalize a shard through `height` (from committee finality).
    pub fn finalize_shard(
        &mut self,
        shard: ShardId,
        height: u64,
    ) -> Result<Vec<FinalityViolation>, BeaconError> {
        let guard = self
            .guards
            .get_mut(&shard)
            .ok_or(BeaconError::InactiveShard { shard })?;
        Ok(guard.finalize(height)?)
    }


    /// The 𝔾 tree from the guards' tips (one leaf per active shard,
    /// canonical order).
    pub fn g_tree(&self) -> GTree {
        let mut g = GTree::new(self.active.ids().len());
        for (i, id) in self.active.ids().iter().enumerate() {
            if let Some((_, tip)) = self.guards.get(id).and_then(|guard| guard.tip()) {
                g.set_tip(i, tip);
            }
        }
        g
    }


    /// The interval's signer subset: the ranked 21 of the beacon
    /// committee for the epoch (E-001). The caller supplies the beacon
    /// committee keys.
    pub fn interval_signers(
        &self,
        beacon_committee: &[VerifyingKey],
        epoch_randomness: &nerv_core::hash::Hash256,
        interval: Interval,
    ) -> Vec<VerifyingKey> {
        attestation_signers(beacon_committee, epoch_randomness, interval)
    }


    /// Close one interval: build the 𝔾 tree, produce the attestation,
    /// finalize the interval, commit the registry set. Returns the
    /// attestation.
    pub fn close_interval(
        &mut self,
        interval: Interval,
        tau_root: nerv_core::hash::Hash256,
        da_root: nerv_core::hash::Hash256,
        signers: &[(usize, &SigningKey)],
        signer_set: &[VerifyingKey],
    ) -> Result<IntervalAttestation, BeaconError> {
        // The interval must be in the current epoch (erratum 100).
        // Epoch-range check fires first so a far-future or far-past
        // interval returns a more informative error than the
        // contiguity check.
        if interval.epoch() != self.current_epoch {
            return Err(BeaconError::IntervalNotInEpoch {
                interval: interval.as_u64(),
                epoch: self.current_epoch.as_u64(),
            });
        }


        // The interval must be the next after the last finalized *within an
// epoch* (erratum 100). `interval_finality` does not advance across the
// epoch boundary — that's `close_epoch`'s job — so a new interval in
// a later epoch is allowed even though its raw index isn't `last + 1`.
        if let Some(last) = self.interval_finality.last() {
            if interval.epoch() == last.epoch() && interval.as_u64() != last.as_u64() + 1 {
                return Err(BeaconError::NotNextInterval {
                    interval: interval.as_u64(),
                    last: last.as_u64(),
                });
            }
        }


        let g = self.g_tree();
        let prev = self
            .interval_attestations
            .last()
            .map(|a| a.digest())
            .unwrap_or_else(|| nerv_core::hash::Hash256::from_bytes([0u8; 32]));


        let attestation = IntervalAttestation::build(
            interval,
            g.root(),
            tau_root,
            da_root,
            prev,
            signers,
            signer_set,
            ATTESTATION_QUORUM,
        )?;


        // Commit the (empty) registry set for this interval — the ledger
        // chain advances even with no bundles (erratum 121).
        let set = IntervalSet {
            interval,
            txids: Default::default(),
        };
        self.ledger.commit(&set)?;


        self.interval_finality.finalize(interval)?;
        self.epoch_interval_digests.push(attestation.digest());
        self.interval_attestations.push(attestation.clone());
        Ok(attestation)
    }


    /// Close the epoch: build the epoch attestation, derive the next
    /// epoch's randomness, reset the interval list. Returns the epoch
    /// attestation and the next epoch's randomness.
    pub fn close_epoch(
        &mut self,
        signers: &[(usize, &SigningKey)],
        signer_set: &[VerifyingKey],
    ) -> Result<(EpochAttestation, nerv_core::hash::Hash256), BeaconError> {
        if self.epoch_interval_digests.is_empty() {
            return Err(BeaconError::EmptyEpoch { epoch: self.current_epoch.as_u64() });
        }
        let root = interval_digests_root(&self.epoch_interval_digests);
        let prev = self
            .epoch_attestations
            .values()
            .last()
            .map(|e| e.digest())
            .unwrap_or_else(|| nerv_core::hash::Hash256::from_bytes([0u8; 32]));
        let epoch = self.current_epoch;
        let attestation = EpochAttestation::build(
            epoch,
            root,
            prev,
            signers,
            signer_set,
            ATTESTATION_QUORUM,
        )?;
        self.epoch_attestations.insert(epoch, attestation.clone());


        // The epoch attestation is the boundary's finality checkpoint:
        // advance the interval watermark so the next epoch's first
        // interval is acceptable (erratum 100).
        self.interval_finality.close_epoch(epoch);


        let next_randomness =
            crate::committee::epoch_randomness(&attestation.digest(), Epoch::from_u64(epoch.as_u64() + 1));
        self.current_epoch = Epoch::from_u64(epoch.as_u64() + 1);
        self.epoch_interval_digests.clear();
        Ok((attestation, next_randomness))
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::harness::keys;
    use crate::testutil::SplitMix64;
    use nerv_core::hash::Hash256;
    use nerv_core::types::ShardSet;


    fn h(seed: u64) -> Hash256 {
        Hash256::from_bytes(SplitMix64::new(seed).bytes32())
    }


    fn genesis_map(set: &ShardSet) -> GenesisMap {
        set.ids().iter().map(|id| (*id, h(0xC0))).collect()
    }


    fn signer_set() -> (Vec<SigningKey>, Vec<VerifyingKey>) {
        keys(21)
    }


    fn signers(keys: &[SigningKey]) -> Vec<(usize, &SigningKey)> {
        (0..crate::committee::ATTESTATION_QUORUM).map(|i| (i, &keys[i])).collect()
    }


    #[test]
    fn beacon_genesis_state() {
        let set = ShardSet::genesis();
        let b = BeaconState::genesis(set.clone(), &genesis_map(&set));
        assert_eq!(b.active().len(), 64);
        assert_eq!(b.current_epoch(), Epoch::from_u64(0));
        assert_eq!(b.last_finalized_interval(), None);
        assert!(b.interval_attestations().is_empty());
        assert!(b.epoch_attestations().is_empty());
        // Fresh genesis seeds every guard with the C₀ at height 0; the tip IS
// that block until the first observation extends the canon above it.
        assert_eq!(b.shard_tip(&set.ids()[7]), Some((0, h(0xC0))));
        // The 𝔾 tree's leaves are the C₀ hash for every shard — not
        // the empty-subtree sentinel, because each guard has the genesis
        // block at height 0 in its canon chain. Build the matching GTree
        // by stamping the same hash into every leaf.
        let mut expected_g = GTree::new(64);
        for i in 0..64 {
            expected_g.set_tip(i, h(0xC0));
        }
        let g = b.g_tree();
        assert_eq!(g.root(), expected_g.root());
    }


    #[test]
    fn observe_and_g_tree() {
        let set = ShardSet::genesis();
        let mut b = BeaconState::genesis(set.clone(), &genesis_map(&set));
        let s7 = set.ids()[7];
        let s40 = set.ids()[40];


        // Erratum 103: (height, hash, prev, link) — `prev` = parent's
        // C_t = genesis C_t at height 1; `link` = the header's own
        // committed C_t (a placeholder here).
        b.observe_header(s7, 1, h(1), h(0xC0), h(0xA1)).unwrap();
        b.observe_header(s7, 2, h(2), h(0xA1), h(0xB1)).unwrap();
        b.observe_header(s40, 1, h(3), h(0xC0), h(0xA3)).unwrap();


        assert_eq!(b.shard_tip(&s7), Some((2, h(2))));
        assert_eq!(b.shard_tip(&s40), Some((1, h(3))));


        let g = b.g_tree();
        let root = g.root();
        // The tips verify as 𝔾 leaves.
        assert!(crate::attestation::verify_g_witness(&root, 64, 7, &h(2), &g.witness(7).unwrap()));
        assert!(crate::attestation::verify_g_witness(&root, 64, 40, &h(3), &g.witness(40).unwrap()));
        // Shards with no observation still carry their genesis tip:
        // each guard starts with the C₀ block at height 0 in its canon
        // chain, so the 𝔾 leaf is the C₀ hash — not the empty-subtree
        // sentinel, which is reserved for shards that produced nothing
        // (erratum 126; see beacon_genesis_state).
        assert!(crate::attestation::verify_g_witness(&root, 64, 9, &h(0xC0), &g.witness(9).unwrap()));


        // An inactive shard is rejected.
        let inactive = ShardId::new(7, 100).unwrap();
        assert!(matches!(
            b.observe_header(inactive, 1, h(9), h(0xC0), h(0xA9)),
            Err(BeaconError::InactiveShard { .. })
        ));
    }


    #[test]
    fn interval_close_chain_and_epoch_boundary() {
        let set = ShardSet::genesis();
        let mut b = BeaconState::genesis(set.clone(), &genesis_map(&set));
        let (sks, vks) = signer_set();
        let signers = signers(&sks);


        // Interval 0 (epoch 0).
        let a0 = b.close_interval(Interval::from_u64(0), h(10), h(11), &signers, &vks).unwrap();
        assert_eq!(b.last_finalized_interval(), Some(Interval::from_u64(0)));
        assert_eq!(a0.interval, Interval::from_u64(0));
        assert_eq!(a0.prev, Hash256::from_bytes([0u8; 32]));
        a0.validate(&vks, ATTESTATION_QUORUM).unwrap();


        // Interval 1 chains to interval 0.
        let a1 = b.close_interval(Interval::from_u64(1), h(12), h(13), &signers, &vks).unwrap();
        assert_eq!(a1.prev, a0.digest());
        a1.validate(&vks, ATTESTATION_QUORUM).unwrap();


        // Non-contiguous intervals are rejected.
        assert!(matches!(
            b.close_interval(Interval::from_u64(3), h(14), h(15), &signers, &vks),
            Err(BeaconError::NotNextInterval { interval: 3, last: 1 })
        ));


        // An interval outside the current epoch is rejected.
        let far = Interval::from_u64(86_401);
        assert!(matches!(
            b.close_interval(far, h(16), h(17), &signers, &vks),
            Err(BeaconError::IntervalNotInEpoch { .. })
        ));


        // Close epoch 0.
        let (e0, r1) = b.close_epoch(&signers, &vks).unwrap();
        assert_eq!(e0.epoch, Epoch::from_u64(0));
        e0.validate(&vks, ATTESTATION_QUORUM).unwrap();
        assert_eq!(
            e0.interval_digests_root,
            interval_digests_root(&[a0.digest(), a1.digest()])
        );
        assert_eq!(b.current_epoch(), Epoch::from_u64(1));
        assert_eq!(
            r1,
            crate::committee::epoch_randomness(&e0.digest(), Epoch::from_u64(1))
        );


        // The next epoch's intervals.
        let next = Interval::from_u64(86_400);
        let a2 = b.close_interval(next, h(18), h(19), &signers, &vks).unwrap();
        assert_eq!(a2.interval.epoch(), Epoch::from_u64(1));
        a2.validate(&vks, ATTESTATION_QUORUM).unwrap();


        // Closing an empty epoch is rejected.
        let (e1, r2) = b.close_epoch(&signers, &vks).unwrap();
        assert_eq!(e1.epoch, Epoch::from_u64(1));
        assert_eq!(b.current_epoch(), Epoch::from_u64(2));
        assert_ne!(r1, r2);
        assert!(matches!(
            b.close_epoch(&signers, &vks),
            Err(BeaconError::EmptyEpoch { .. })
        ));
    }


    #[test]
    fn shard_finalization_flows_through() {
        let set = ShardSet::genesis();
        let mut b = BeaconState::genesis(set.clone(), &genesis_map(&set));
        let s7 = set.ids()[7];
        // Extend to height 2 with canon hash h(1) at height 1 and h(2)
        // at height 2 (each with a distinct placeholder link).
        b.observe_header(s7, 1, h(1), h(0xC0), h(0xA1)).unwrap();
        b.observe_header(s7, 2, h(2), h(0xA1), h(0xB1)).unwrap();
        // A competing fork at height 1: a strictly smaller hash
        // reorganizes the canon; the displaced canon becomes an alt.
        let mut smaller = h(1);
        let mut raw = *smaller.as_bytes();
        raw[31] = raw[31].wrapping_sub(1);
        smaller = Hash256::from_bytes(raw);
        assert!(smaller < h(1));
        let outcome = b.observe_header(s7, 1, smaller, h(0xC0), h(0xA2)).unwrap();
        assert_eq!(outcome, ObserveOutcome::Reorganized);
        assert_eq!(b.shard_tip(&s7), Some((1, smaller)));

        // A larger hash at height 2 (extending the NEW canon at h=1
        // via prev = the new link 0xA2) — not smaller than the existing
        // canon at h=2, so it's a non-reorganizing fork recorded as an
        // alt.
        let larger_h2 = {
            let mut raw = *h(2).as_bytes();
            raw[30] = raw[30].wrapping_add(1);
            Hash256::from_bytes(raw)
        };
        let out = b.observe_header(s7, 2, larger_h2, h(0xA2), h(0xB2)).unwrap();
        assert_eq!(out, ObserveOutcome::ForkRecorded);

        let violations = b.finalize_shard(s7, 2).unwrap();
        // Two violations: the displaced canon at h=1 (reorg), and the
        // larger h=2 fork.
        assert_eq!(violations.len(), 2, "got {:?}", violations);
        let h1_violation = violations.iter().find(|v| v.height == 1).expect("h=1 violation");
        // D1: finalized_hash at h=1 is the *new* canon (smaller), not the
        // parent at h=0 (genesis hash).
        assert_eq!(h1_violation.finalized_hash, smaller);
        assert_ne!(h1_violation.finalized_hash, h(0xC0));
        assert_eq!(h1_violation.conflicting_hash, h(1));
        // D6: the alt carries the displaced canon's parent C_t (= genesis)
        // and its own link (= the displaced h(1)'s link 0xA1).
        assert_eq!(h1_violation.conflicting_prev, h(0xC0));
        assert_eq!(h1_violation.conflicting_link, h(0xA1));
        let h2_violation = violations.iter().find(|v| v.height == 2).expect("h=2 violation");
        assert_eq!(h2_violation.finalized_hash, h(2));
        assert_eq!(h2_violation.conflicting_hash, larger_h2);
        assert_eq!(h2_violation.conflicting_prev, h(0xA2));
        assert_eq!(h2_violation.conflicting_link, h(0xB2));
    }
}
