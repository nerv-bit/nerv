//! Canonical txid-first-wins deduplication (WP §5.5, tier 2): "a txid
//! settles into the first interval in which a valid bundle contained
//! it." Within an interval, the lowest arrival index claims the txid;
//! across intervals, the settled-ledger drops re-submissions. Pure and
//! deterministic: the output is a function of (ledger state, bundle
//! index assignment) — never of slice order. Policy-free: bundle
//! size bounds and admission rules are the registry's (chunk 14).


use std::collections::{BTreeMap, BTreeSet};
use nerv_core::types::{Interval, TxId};
/// One bundle's committed txid list, in submission order. `index` is the
/// bundle's canonical arrival position within the interval — first-wins
/// is defined over it, so the index assignment, not slice order, is the
/// determinism carrier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BundleTxids {
    pub index: u32,
    pub txids: Vec<TxId>,
}


/// The deduplicated txid set of one interval, in canonical byte order
/// (the T_τ tree's leaf order, chunk 14), with per-txid provenance —
/// the contributing bundle's arrival index (slashing/attestation
/// evidence: which bundle vouched for each inclusion).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntervalSet {
    pub interval: Interval,
    pub txids: BTreeMap<TxId, u32>,
}


impl IntervalSet {
    pub fn len(&self) -> usize {
        self.txids.len()
    }


    pub fn is_empty(&self) -> bool {
        self.txids.is_empty()
    }


    pub fn contains(&self, txid: &TxId) -> bool {
        self.txids.contains_key(txid)
    }


    pub fn provenance(&self, txid: &TxId) -> Option<u32> {
        self.txids.get(txid).copied()
    }


    /// Canonical (byte-ascending) iteration — the T_τ leaf order.
    pub fn iter_sorted(&self) -> impl Iterator<Item = (&TxId, &u32)> {
        self.txids.iter()
    }


    /// Kept-txid counts per contributing bundle index.
    pub fn bundle_contributions(&self) -> BTreeMap<u32, usize> {
        let mut out = BTreeMap::new();
        for (_, &b) in &self.txids {
            *out.entry(b).or_insert(0) += 1;
        }
        out
    }
}


/// One interval's dedup outcome: the set plus the cross-interval
/// duplicates (txids already settled in a prior interval — dropped,
/// with their settling interval recorded).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DedupReport {
    pub set: IntervalSet,
    pub cross_interval: Vec<(TxId, Interval)>,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DedupError {
    #[error("bundle {bundle} lists txid {txid} twice — malformed (its txid-set commitment is over a set)")]
    IntraBundleDuplicate { bundle: u32, txid: TxId },
    #[error("two bundles share arrival index {index}")]
    DuplicateBundleIndex { index: u32 },
    #[error("dedup for interval {requested} is not after the ledger's {ledger} — forward-only processing")]
    IntervalOrder { ledger: Interval, requested: Interval },
    #[error("commit gap: expected interval {expected}, got {got}")]
    Gap { expected: Interval, got: Interval },
}


/// The settled-txid ledger: txid → its settling interval, plus the
/// committed-interval chain head. The cross-interval first-wins oracle.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IntervalLedger {
    current: Option<Interval>,
    settled: BTreeMap<TxId, Interval>,
}


impl IntervalLedger {
    pub fn new() -> IntervalLedger {
        IntervalLedger::default()
    }


    pub fn current_interval(&self) -> Option<Interval> {
        self.current
    }


    pub fn len(&self) -> usize {
        self.settled.len()
    }


    pub fn is_empty(&self) -> bool {
        self.settled.is_empty()
    }


    pub fn settled_in(&self, txid: &TxId) -> Option<Interval> {
        self.settled.get(txid).copied()
    }


    /// Dedups `bundles` for `interval`. Within the interval, the lowest
    /// bundle index containing a txid claims it; txids settled in prior
    /// intervals are dropped (recorded in the report); a txid listed
    /// twice within one bundle is a malformed bundle (error — the
    /// registry rejects the bundle, dedup never silently repairs it).
    /// `bundles` must already be validation-passed (WP: the registry
    /// includes any *valid* bundle; validity is the caller's gate).
    pub fn dedup(
        &self,
        interval: Interval,
        bundles: &[BundleTxids],
    ) -> Result<DedupReport, DedupError> {
        if let Some(c) = self.current {
            if interval <= c {
                return Err(DedupError::IntervalOrder { ledger: c, requested: interval });
            }
        }
        let mut sorted: Vec<&BundleTxids> = bundles.iter().collect();
        sorted.sort_unstable_by_key(|b| b.index);
        let mut seen_index = BTreeSet::new();
        let mut txids = BTreeMap::new();
        let mut cross = Vec::new();
        for bundle in sorted {
            if !seen_index.insert(bundle.index) {
                return Err(DedupError::DuplicateBundleIndex { index: bundle.index });
            }
            let mut seen_txid = BTreeSet::new();
            for txid in &bundle.txids {
                if !seen_txid.insert(*txid) {
                    return Err(DedupError::IntraBundleDuplicate {
                        bundle: bundle.index,
                        txid: *txid,
                    });
                }
                if let Some(prior) = self.settled_in(txid) {
                    cross.push((*txid, prior));
                } else {
                    txids.entry(*txid).or_insert(bundle.index);
                }
            }
        }
        Ok(DedupReport { set: IntervalSet { interval, txids }, cross_interval: cross })
    }


    /// Commits an interval's set into the ledger. The +1 chain is the
    /// integrity check: the registry commits one set per beacon interval
    /// (empty when no valid bundles arrived) — the chain must not skip.
    pub fn commit(&mut self, set: &IntervalSet) -> Result<(), DedupError> {
        if let Some(c) = self.current {
            let expected = Interval::from_u64(c.as_u64() + 1);
            if set.interval != expected {
                return Err(DedupError::Gap { expected, got: set.interval });
            }
        }
        for txid in set.txids.keys() {
            self.settled.insert(*txid, set.interval);
        }
        self.current = Some(set.interval);
        Ok(())
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;
    use nerv_core::hash::Hash256;


    fn txid(rng: &mut SplitMix64) -> TxId {
        TxId::from_hash(Hash256::from_bytes(rng.bytes32()))
    }


    fn txids(seed: u64, n: usize) -> Vec<TxId> {
        let mut rng = SplitMix64::new(seed);
        (0..n).map(|_| txid(&mut rng)).collect()
    }


    fn det_txid(i: u8) -> TxId {
        TxId::from_hash(Hash256::from_bytes([i; 32]))
    }


    #[test]
    fn first_wins_within_interval() {
        let ledger = IntervalLedger::new();
        let a = det_txid(1);
        let b = det_txid(2);
        let c = det_txid(3);
        let report = ledger
            .dedup(
                Interval::from_u64(5),
                &[
                    BundleTxids { index: 0, txids: vec![a, b] },
                    BundleTxids { index: 1, txids: vec![b, c] },
                ],
            )
            .unwrap();
        assert_eq!(report.set.len(), 3);
        assert_eq!(report.set.provenance(&a), Some(0));
        assert_eq!(report.set.provenance(&b), Some(0), "lowest index claims");
        assert_eq!(report.set.provenance(&c), Some(1));
        assert!(report.cross_interval.is_empty());
        assert_eq!(report.set.bundle_contributions(), BTreeMap::from([(0, 2), (1, 1)]));
    }


    #[test]
    fn slice_order_does_not_change_the_outcome() {
        let a = det_txid(1);
        let b = det_txid(2);
        let mut ledger = IntervalLedger::new();
        let r1 = ledger
            .dedup(
                Interval::from_u64(0),
                &[BundleTxids { index: 0, txids: vec![a, b] }],
            )
            .unwrap();
        let r2 = ledger
            .dedup(
                Interval::from_u64(0),
                &[BundleTxids { index: 0, txids: vec![b, a] }],
            )
            .unwrap();
        assert_eq!(r1, r2, "list order within a bundle is irrelevant");


        // Arrival index, not slice position, carries first-wins.
        let r3 = ledger
            .dedup(
                Interval::from_u64(1),
                &[
                    BundleTxids { index: 1, txids: vec![a] },
                    BundleTxids { index: 0, txids: vec![a] },
                ],
            )
            .unwrap();
        assert!(r3.set.is_empty(), "a settled in interval 0 — cross-interval drop");
        assert_eq!(r3.cross_interval, vec![(a, Interval::from_u64(0)), (a, Interval::from_u64(0))]);
        let mut ledger2 = IntervalLedger::new();
        let r4 = ledger2
            .dedup(
                Interval::from_u64(1),
                &[
                    BundleTxids { index: 7, txids: vec![b] },
                    BundleTxids { index: 2, txids: vec![b] },
                ],
            )
            .unwrap();
        assert_eq!(r4.set.provenance(&b), Some(2), "lowest index wins regardless of slice order");
    }


    #[test]
    fn cross_interval_first_wins_and_ledger_growth() {
        let mut ledger = IntervalLedger::new();
        let t = txids(1, 6);
        let r0 = ledger
            .dedup(Interval::from_u64(0), &[BundleTxids { index: 0, txids: t[..3].to_vec() }])
            .unwrap();
        ledger.commit(&r0.set).unwrap();
        assert_eq!(ledger.current_interval(), Some(Interval::from_u64(0)));
        for x in &t[..3] {
            assert_eq!(ledger.settled_in(x), Some(Interval::from_u64(0)));
        }
        assert_eq!(ledger.settled_in(&t[3]), None);


        // Interval 1: re-submits t[1] (settled), plus t[3], t[4].
        let r1 = ledger
            .dedup(
                Interval::from_u64(1),
                &[
                    BundleTxids { index: 0, txids: vec![t[1]] },
                    BundleTxids { index: 1, txids: vec![t[3], t[4], t[1]] },
                ],
            )
            .unwrap();
        assert_eq!(r1.set.len(), 2);
        assert_eq!(r1.set.provenance(&t[3]), Some(1));
        assert_eq!(r1.cross_interval.len(), 2);
        ledger.commit(&r1.set).unwrap();
        assert_eq!(ledger.len(), 5);


        // Interval 2: t[0] re-submitted once more — still settles at 0.
        let r2 = ledger
            .dedup(Interval::from_u64(2), &[BundleTxids { index: 0, txids: vec![t[0]] }])
            .unwrap();
        assert!(r2.set.is_empty());
        assert_eq!(r2.cross_interval, vec![(t[0], Interval::from_u64(0))]);
        ledger.commit(&r2.set).unwrap();
        assert_eq!(ledger.settled_in(&t[4]), Some(Interval::from_u64(1)));
    }


    #[test]
    fn canonical_byte_order_output() {
        let ledger = IntervalLedger::new();
        let t = txids(2, 64);
        let mut shuffled = t.clone();
        let mut rng = SplitMix64::new(99);
        for i in (1..shuffled.len()).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            shuffled.swap(i, j);
        }
        let report = ledger
            .dedup(
                Interval::from_u64(0),
                &[
                    BundleTxids { index: 0, txids: shuffled[..32].to_vec() },
                    BundleTxids { index: 1, txids: shuffled[32..].to_vec() },
                ],
            )
            .unwrap();
        let keys: Vec<&TxId> = report.set.iter_sorted().map(|(k, _)| k).collect();
        assert_eq!(keys.len(), 64);
        for w in keys.windows(2) {
            assert!(w[0] < w[1], "iter_sorted must be byte-ascending");
        }
        let mut sorted_t = t.clone();
        sorted_t.sort();
        assert_eq!(keys, sorted_t.iter().collect::<Vec<_>>());
    }


    #[test]
    fn malformed_bundles_rejected() {
        let ledger = IntervalLedger::new();
        let a = det_txid(1);
        let dup = BundleTxids { index: 3, txids: vec![a, det_txid(2), a] };
        assert!(matches!(
            ledger.dedup(Interval::from_u64(0), &[dup]),
            Err(DedupError::IntraBundleDuplicate { bundle: 3, .. })
        ));
        let r = ledger
            .dedup(
                Interval::from_u64(0),
                &[
                    BundleTxids { index: 5, txids: vec![a] },
                    BundleTxids { index: 5, txids: vec![det_txid(2)] },
                ],
            )
            .unwrap_err();
        assert!(matches!(r, DedupError::DuplicateBundleIndex { index: 5 }));
    }


    #[test]
    fn forward_only_processing_and_commit_chain() {
        let mut ledger = IntervalLedger::new();
        let a = det_txid(1);
        let r0 = ledger.dedup(Interval::from_u64(3), &[BundleTxids { index: 0, txids: vec![a] }]).unwrap();
        ledger.commit(&r0.set).unwrap();
        assert!(matches!(
            ledger.dedup(Interval::from_u64(3), &[BundleTxids { index: 0, txids: vec![a] }]),
            Err(DedupError::IntervalOrder { .. })
        ));
        assert!(matches!(
            ledger.dedup(Interval::from_u64(2), &[BundleTxids { index: 0, txids: vec![a] }]),
            Err(DedupError::IntervalOrder { .. })
        ));
        assert!(ledger.dedup(Interval::from_u64(4), &[]).is_ok());


        // Commit chain: skips are gaps; empty sets keep the chain.
        let r4 = ledger.dedup(Interval::from_u64(4), &[]).unwrap();
        ledger.commit(&r4.set).unwrap();
        let r6 = ledger.dedup(Interval::from_u64(6), &[]).unwrap();
        assert!(matches!(
            ledger.commit(&r6.set),
            Err(DedupError::Gap { expected: Interval, .. }) if expected == Interval::from_u64(5)
        ));
        let r5 = ledger.dedup(Interval::from_u64(5), &[]).unwrap();
        ledger.commit(&r5.set).unwrap();
        assert_eq!(ledger.current_interval(), Some(Interval::from_u64(5)));
        assert_eq!(ledger.len(), 1);
    }


    #[test]
    fn empty_inputs_are_policy_free() {
        let ledger = IntervalLedger::new();
        let r = ledger.dedup(Interval::from_u64(0), &[]).unwrap();
        assert!(r.set.is_empty() && r.cross_interval.is_empty());
        let r = ledger
            .dedup(Interval::from_u64(0), &[BundleTxids { index: 0, txids: vec![] }])
            .unwrap();
        assert!(r.set.is_empty());
    }


    #[test]
    fn many_bundles_provenance_bruteforce() {
        // 200 txids; txid i appears in bundles {i % 8, 7}. Expected claim:
        // min(i % 8, 7) = i % 8.
        let ledger = IntervalLedger::new();
        let t: Vec<TxId> = (0..200u8).map(det_txid).collect();
        let bundles: Vec<BundleTxids> = (0..8u32)
            .map(|j| BundleTxids {
                index: j,
                txids: t.iter()
                    .enumerate()
                    .filter(|&(i, _)| (i as u32 % 8) == j || j == 7)
                    .map(|(_, x)| *x)
                    .collect(),
            })
            .collect();
        let report = ledger.dedup(Interval::from_u64(0), &bundles).unwrap();
        assert_eq!(report.set.len(), 200);
        for (i, x) in t.iter().enumerate() {
            assert_eq!(report.set.provenance(x), Some((i as u32 % 8)), "txid {i}");
        }
        // Determinism across runs.
        let report2 = ledger.dedup(Interval::from_u64(0), &bundles).unwrap();
        assert_eq!(report, report2);
        // Contributions: bundle j < 7 keeps 25 each (i%8==j), bundle 7
        // keeps the 25 with i%8==7.
        let contrib = report.set.bundle_contributions();
        for j in 0..8u32 {
            assert_eq!(contrib.get(&j), Some(&25), "bundle {j}");
        }
    }
}
