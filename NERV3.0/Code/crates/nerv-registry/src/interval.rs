//! T_τ construction (WP §4.3 rule 1, §5.5 tier 2; erratum 121):
//! arrival-indexed bundles → first-wins dedup → the interval set's tree.


use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::REGISTRY_COMMIT;
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::types::{Interval, TxId};
use nerv_state::ttau::{empty_root, RegistryWitness, TauTree};


use crate::bundle::{Bundle, BundleSummary};
use crate::error::IntervalError;
use nerv_proofs::{BundleTxids, DedupReport, IntervalLedger};


/// The finalized interval record: the T_τ root and the bundle set that
/// produced it (arrival order) — the slashing evidence binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntervalCommit {
    pub interval: Interval,
    pub tau_root: Hash256,
    pub bundles: Vec<BundleSummary>,
}


impl IntervalCommit {
    /// The degraded-mode QC's subject (erratum 121).
    pub fn digest(&self) -> Hash256 {
        Hash256::concat(&REGISTRY_COMMIT, &self.encode())
    }
}


impl Encode for IntervalCommit {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.interval.as_u64().to_le_bytes());
        out.extend_from_slice(self.tau_root.as_bytes());
        out.extend_from_slice(&(self.bundles.len() as u32).to_le_bytes());
        for b in &self.bundles {
            b.encode_into(out);
        }
    }
    fn encoded_len(&self) -> usize {
        8 + 32 + 4 + self.bundles.iter().map(|b| b.encoded_len()).sum::<usize>()
    }
}


impl Decode for IntervalCommit {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let interval = Interval::from_u64(r.read_u64()?);
        let tau_root = Hash256::decode_from(r)?;
        let n = r.read_seq_len()?;
        if n > 65_536 {
            return Err(CodecError::SeqTooLarge { count: n, max: 65_536 });
        }
        let mut bundles = Vec::with_capacity(n);
        for _ in 0..n {
            bundles.push(BundleSummary::decode_from(r)?);
        }
        Ok(IntervalCommit { interval, tau_root, bundles })
    }
}


/// One built interval: the commit, the dedup report (cross-interval drops
/// and per-txid bundle provenance), and the T_τ tree (witness generation
/// for shard executors, rule 1).
#[derive(Clone, Debug)]
pub struct IntervalBuild {
    pub commit: IntervalCommit,
    pub report: DedupReport,
    pub tau: TauTree,
}


impl IntervalBuild {
    /// The rule-1 membership witness for a settled txid.
    pub fn witness(&self, txid: &TxId) -> Result<RegistryWitness, IntervalError> {
        let index = self
            .tau
            .position(txid)
            .ok_or(IntervalError::TxidAbsent { txid: *txid })?;
        Ok(RegistryWitness::new(self.commit.interval, self.tau.witness(index)?))
    }


    pub fn contains(&self, txid: &TxId) -> bool {
        self.report.set.contains(txid)
    }
}

/// [`build_interval_commit`] with an exclusion set: each bundle's txid
/// list is pre-filtered against `exclude` (txids pending in
/// not-yet-finalized intervals — erratum 133) before dedup. Excluded
/// txids are in flight elsewhere and require no report entry.
pub fn build_interval_commit_excluding(
   interval: Interval,
   bundles: &[Bundle],
   ledger: &IntervalLedger,
   exclude: &std::collections::BTreeSet<TxId>,
) -> Result<IntervalBuild, IntervalError> {
   let mut summaries = Vec::with_capacity(bundles.len());
   let mut txid_lists = Vec::with_capacity(bundles.len());
   for (i, b) in bundles.iter().enumerate() {
       b.validate_structure()?;
       summaries.push(b.summary());
       let txids: Vec<TxId> =
           b.txids().into_iter().filter(|t| !exclude.contains(t)).collect();
       txid_lists.push(BundleTxids { index: i as u32, txids });
   }
   let report = ledger.dedup(interval, &txid_lists)?;
   let sorted: Vec<TxId> = report.set.iter_sorted().map(|(t, _)| *t).collect();
   let tau = TauTree::from_sorted(&sorted)?;
   Ok(IntervalBuild {
       commit: IntervalCommit { interval, tau_root: tau.root(), bundles: summaries },
       report,
       tau,
   })
}



/// Build one interval's commit from arrival-ordered bundles. Structural
/// validation per bundle; the committee's proof verification is the
/// explicit preceding step (`verify_bundle`); the challenge window is the
/// public backstop (errata 120–122).
pub fn build_interval_commit(
    interval: Interval,
    bundles: &[Bundle],
    ledger: &IntervalLedger,
) -> Result<IntervalBuild, IntervalError> {
    let mut summaries = Vec::with_capacity(bundles.len());
    let mut txid_lists = Vec::with_capacity(bundles.len());
    for (i, b) in bundles.iter().enumerate() {
        b.validate_structure()?;
        summaries.push(b.summary());
        txid_lists.push(BundleTxids { index: i as u32, txids: b.txids() });
    }
    let report = ledger.dedup(interval, &txid_lists)?;
    let sorted: Vec<TxId> = report.set.iter_sorted().map(|(t, _)| *t).collect();
    let tau = TauTree::from_sorted(&sorted)?;
    Ok(IntervalBuild {
        commit: IntervalCommit { interval, tau_root: tau.root(), bundles: summaries },
        report,
        tau,
    })
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::harness::{agg_key, shell, shallow_proof, txid_of};
    use nerv_proofs::IntervalSet;


    fn entry(seed: u64) -> crate::mempool::PoolEntry {
        let canon = shell(seed).canonicalize().unwrap();
        crate::mempool::PoolEntry {
            txid: nerv_state::canonical_txid(&canon),
            shell: canon,
            proof: shallow_proof(),
        }
    }


    fn bundle(tag: u8, seeds: &[u64]) -> Bundle {
        Bundle::build(&agg_key(tag), seeds.iter().map(|&s| entry(s)).collect()).unwrap()
    }


    #[test]
    fn build_dedups_first_wins_and_builds_tau() {
        let ledger = IntervalLedger::new();
        let interval = Interval::from_u64(86_400);
        // Bundle 0: {1, 2, 3}; bundle 1: {3, 4, 5} — txid 3 contested, index 0 wins.
        let b0 = bundle(1, &[1, 2, 3]);
        let b1 = bundle(2, &[3, 4, 5]);
        let build = build_interval_commit(interval, &[b0.clone(), b1], &ledger).unwrap();


        let mut expected = vec![
            txid_of(&shell(1)),
            txid_of(&shell(2)),
            txid_of(&shell(3)),
            txid_of(&shell(4)),
            txid_of(&shell(5)),
        ];
        expected.sort();
        assert_eq!(build.report.set.len(), 5);
        assert_eq!(build.report.cross_interval.len(), 0);
        assert_eq!(build.commit.tau_root, TauTree::from_sorted(&expected).unwrap().root());
        assert_eq!(build.commit.bundles.len(), 2);
        assert_eq!(build.commit.bundles[0].root, b0.txid_root());
        // First-wins provenance.
        assert_eq!(build.report.set.provenance(&txid_of(&shell(3))), Some(0));
        assert_eq!(build.report.set.provenance(&txid_of(&shell(4))), Some(1));


        // Every included txid has a verifying rule-1 witness.
        for t in &expected {
            let w = build.witness(t).unwrap();
            assert!(w.verify(&build.commit.tau_root, t));
            assert_eq!(w.interval, interval);
        }
        assert!(matches!(
            build.witness(&txid_of(&shell(99))),
            Err(IntervalError::TxidAbsent { .. })
        ));
    }


    #[test]
    fn cross_interval_drops_after_commit() {
        let mut ledger = IntervalLedger::new();
        let i0 = Interval::from_u64(0);
        let build0 = build_interval_commit(i0, &[bundle(1, &[10, 11])], &ledger).unwrap();
        ledger.commit(&build0.report.set).unwrap();
        assert_eq!(ledger.current_interval(), Some(i0));


        // A later interval resubmitting txid 10 drops it.
        let i1 = Interval::from_u64(1);
        let build1 = build_interval_commit(i1, &[bundle(2, &[10, 12])], &ledger).unwrap();
        assert_eq!(build1.report.set.len(), 1);
        assert!(!build1.contains(&txid_of(&shell(10))));
        assert!(build1.contains(&txid_of(&shell(12))));
        assert_eq!(build1.report.cross_interval, vec![(txid_of(&shell(10)), i0)]);


        // Forward-only: replaying interval 1 is rejected by the ledger.
        assert!(build_interval_commit(i1, &[], &ledger).is_err());
    }


    #[test]
    fn empty_interval_commits_the_empty_root() {
        let ledger = IntervalLedger::new();
        let build = build_interval_commit(Interval::from_u64(0), &[], &ledger).unwrap();
        assert!(build.report.set.is_empty());
        assert_eq!(build.commit.tau_root, empty_root());
        assert!(build.commit.bundles.is_empty());
        assert_eq!(build.report.set.len(), 0);
    }


    #[test]
    fn invalid_bundle_rejected() {
        let ledger = IntervalLedger::new();
        let mut bad = bundle(3, &[20, 21]);
        bad.attestation.txid_root = Hash256::from_bytes([0xEE; 32]);
        assert!(matches!(
            build_interval_commit(Interval::from_u64(0), &[bad], &ledger),
            Err(IntervalError::Bundle(BundleError::RootMismatch { .. }))
        ));
    }


    #[test]
    fn digest_determinism_and_codec_roundtrip() {
        let ledger = IntervalLedger::new();
        let build = build_interval_commit(
            Interval::from_u64(5),
            &[bundle(4, &[30, 31]), bundle(5, &[32])],
            &ledger,
        )
        .unwrap();
        let c = &build.commit;
        assert_eq!(c.digest(), c.digest());
        let mut other = c.clone();
        other.tau_root = Hash256::from_bytes([1; 32]);
        assert_ne!(c.digest(), other.digest());
        other = c.clone();
        other.bundles.pop();
        assert_ne!(c.digest(), other.digest());
        other = c.clone();
        other.interval = Interval::from_u64(6);
        assert_ne!(c.digest(), other.digest());


        let enc = c.encode();
        assert_eq!(enc.len(), c.encoded_len());
        assert_eq!(IntervalCommit::decode(&enc).unwrap(), *c);
        assert_eq!(IntervalCommit::decode(&enc).unwrap().digest(), c.digest());
        for cut in 0..enc.len() {
            assert!(IntervalCommit::decode(&enc[..cut]).is_err(), "cut {cut}");
        }
        let mut ext = enc.clone();
        ext.push(0);
        assert!(IntervalCommit::decode(&ext).is_err());


        // The digest is the literal formula.
        let mut pre = Vec::new();
        pre.extend_from_slice(REGISTRY_COMMIT.as_bytes());
        pre.extend_from_slice(&c.encode());
        assert_eq!(
            c.digest().as_bytes(),
            blake3::hash(&pre).as_bytes()
        );
        let _ = IntervalSet::default();
    }


    use crate::error::BundleError;

     #[test]
   fn excluding_filters_in_flight_txids() {
       let ledger = IntervalLedger::new();
       let interval = Interval::from_u64(0);
       let b0 = bundle(1, &[40, 41]);
       let mut exclude = std::collections::BTreeSet::new();
       exclude.insert(txid_of(&shell(40)));
       let build = build_interval_commit_excluding(interval, &[b0], &ledger, &exclude).unwrap();
       assert_eq!(build.report.set.len(), 1);
       assert!(build.contains(&txid_of(&shell(41))));
       assert!(!build.contains(&txid_of(&shell(40))));
       assert_eq!(
           build.commit.tau_root,
           TauTree::from_sorted(&[txid_of(&shell(41))]).unwrap().root()
       );
       // No exclusions: identical to the plain build.
       let plain = build_interval_commit(interval, &[bundle(1, &[40, 41])], &ledger).unwrap();
       let none = build_interval_commit_excluding(interval, &[bundle(1, &[40, 41])], &ledger, &Default::default()).unwrap();
       assert_eq!(plain.commit.tau_root, none.commit.tau_root);
   }

}
