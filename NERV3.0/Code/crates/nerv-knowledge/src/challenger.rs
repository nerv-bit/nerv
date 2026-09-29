//! The challenger market (WP §10.4; erratum 172): an open market of
//! registered parameterizations competing out-of-sample against the
//! incumbent on sealed targets. Rewards (the useful-work pool) flow only
//! from the verified SkillRecord; the payout wiring is the node's.

use std::collections::BTreeMap;

use nerv_codec::codec_w::Delta;
use nerv_core::constants::DERIVED_PRED;
use nerv_core::hash::Hash256;

use crate::block_loop::BlockEvent;
use crate::forecaster::{predict, DIMS, Observation, Weights, Window};
use crate::huber;

pub const WINDOW_BLOCKS: u64 = nerv_core::params::OVERLAY_CHALLENGER_WINDOW_BLOCKS;
pub const MIN_COVERAGE_PERMILLE: u64 =
    nerv_core::params::OVERLAY_CHALLENGER_MIN_COVERAGE_PERMILLE;
/// The promotion margin: the challenger's window total must beat the
/// incumbent's by this permille, strictly (genesis-config; governance-set).
pub const MARGIN_PERMILLE: u64 = 50;

const _: () = assert!(WINDOW_BLOCKS == 2016);
const _: () = assert!(MIN_COVERAGE_PERMILLE == 900);
const _: () = assert!(MARGIN_PERMILLE < 1000);

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ChallengerError {
    #[error("challenger {id:?} is already registered")]
    AlreadyRegistered { id: [u8; 32] },
    #[error("challenger {id:?} is not registered")]
    NotRegistered { id: [u8; 32] },
    #[error("commit for height {height} arrived after its reveal")]
    CommitAfterReveal { height: u64 },
    #[error("commit for height {height} precedes the next height {expected}")]
    PrematureCommit { height: u64, expected: u64 },
    #[error("event height {found} does not follow {expected}")]
    NonContiguous { expected: u64, found: u64 },
}

/// The verified out-of-sample skill record — the useful-work pool's
/// payout input (WP §10.4, P7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkillRecord {
    pub challenger: [u8; 32],
    pub window_start: u64,
    pub window_end: u64,
    pub covered_blocks: u64,
    pub challenger_loss_q32: u128,
    pub incumbent_loss_q32: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EligibilityFailure {
    InsufficientCoverage { covered: u64, required: u64 },
    MissingIncumbentScore { block: u64 },
    MarginNotMet { challenger: u128, incumbent: u128 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateOutcome {
    /// The challenger earned promotion: install `weights` as the
    /// incumbent parameterization at the next epoch boundary (§10.4; the
    /// incumbent reverts to reference at every W-epoch anyway — §7.6).
    Promote { record: SkillRecord, weights: Weights },
    NotEligible(EligibilityFailure),
}

/// One registered challenger: its frozen parameterization, its shadow
/// copy of the public window, and its sealed-exam record.
#[derive(Clone, Debug)]
pub struct Challenger {
    id: [u8; 32],
    weights: Weights,
    window: Window,
    next_height: u64,
    commits: BTreeMap<u64, Hash256>,
    scores: BTreeMap<u64, u128>,
}

impl Challenger {
    pub fn register(id: [u8; 32], weights: Weights) -> Challenger {
        Challenger {
            id,
            weights,
            window: Window::new(),
            next_height: 1,
            commits: BTreeMap::new(),
            scores: BTreeMap::new(),
        }
    }

    pub fn id(&self) -> &[u8; 32] {
        &self.id
    }

    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn next_height(&self) -> u64 {
        self.next_height
    }

    pub fn commits(&self) -> &BTreeMap<u64, Hash256> {
        &self.commits
    }

    pub fn scores(&self) -> &BTreeMap<u64, u128> {
        &self.scores
    }

    /// The sealed commit for the upcoming block (§10.4): the deterministic
    /// prediction from the pre-reveal window, hashed under the incumbent's
    /// commitment domain. Idempotent; the timing discipline is structural
    /// (erratum 172).
    pub fn commit(&mut self, height: u64) -> Result<Hash256, ChallengerError> {
        if height < self.next_height {
            return Err(ChallengerError::CommitAfterReveal { height });
        }
        if height > self.next_height {
            return Err(ChallengerError::PrematureCommit { height, expected: self.next_height });
        }
        let pred = predict(&self.weights, &self.window);
        let h = Hash256::concat(&DERIVED_PRED, &pred.canonical_bytes());
        self.commits.insert(height, h);
        Ok(h)
    }

    /// One public event. On a reveal with a sealed, matching commit: the
    /// block is scored (the challenger's own registered scales). A
    /// mismatched commit is dropped — no score, no coverage credit.
    pub fn process(&mut self, event: BlockEvent) -> Result<Option<u128>, ChallengerError> {
        let height = match &event {
            BlockEvent::Reveal { height, .. } | BlockEvent::Miss { height, .. } => *height,
        };
        if height != self.next_height {
            return Err(ChallengerError::NonContiguous {
                expected: self.next_height,
                found: height,
            });
        }
        self.next_height += 1;
        match event {
            BlockEvent::Miss { .. } => Ok(None),
            BlockEvent::Reveal { delta, fee_sum, bucket, .. } => {
                let mut scored = None;
                if let Some(&committed) = self.commits.get(&height) {
                    let pred = predict(&self.weights, &self.window);
                    let h = Hash256::concat(&DERIVED_PRED, &pred.canonical_bytes());
                    if h == committed {
                        let r = huber::residual(&delta, &pred);
                        let mut scales = [0u64; DIMS];
                        for (j, s) in scales.iter_mut().enumerate() {
                            *s = self.weights.scale(j);
                        }
                        let loss = huber::total_loss(&r, &scales);
                        self.scores.insert(height, loss);
                        scored = Some(loss);
                    }
                }
                self.window.push(Observation { delta, fee_sum, bucket });
                Ok(scored)
            }
        }
    }

    /// The 2,016-block promotion gate (erratum 172): coverage, then the
    /// margin — both totals summed over the challenger's covered blocks.
    pub fn evaluate_gate(
        &self,
        start_height: u64,
        incumbent_losses: &BTreeMap<u64, u128>,
    ) -> GateOutcome {
        let end = start_height.saturating_add(WINDOW_BLOCKS - 1);
        let covered: Vec<(u64, u128)> =
            self.scores.range(start_height..=end).map(|(&h, &l)| (h, l)).collect();
        let required = (WINDOW_BLOCKS * MIN_COVERAGE_PERMILLE + 999) / 1000;
        if (covered.len() as u64) < required {
            return GateOutcome::NotEligible(EligibilityFailure::InsufficientCoverage {
                covered: covered.len() as u64,
                required,
            });
        }
        let mut challenger_total = 0u128;
        let mut incumbent_total = 0u128;
        for &(h, l) in &covered {
            challenger_total += l;
            match incumbent_losses.get(&h) {
                Some(&il) => incumbent_total += il,
                None => {
                    return GateOutcome::NotEligible(
                        EligibilityFailure::MissingIncumbentScore { block: h },
                    )
                }
            }
        }
        if challenger_total * 1000 < incumbent_total * (1000 - MARGIN_PERMILLE as u128) {
            GateOutcome::Promote {
                record: SkillRecord {
                    challenger: self.id,
                    window_start: start_height,
                    window_end: end,
                    covered_blocks: covered.len() as u64,
                    challenger_loss_q32: challenger_total,
                    incumbent_loss_q32: incumbent_total,
                },
                weights: self.weights.clone(),
            }
        } else {
            GateOutcome::NotEligible(EligibilityFailure::MarginNotMet {
                challenger: challenger_total,
                incumbent: incumbent_total,
            })
        }
    }
}

/// The open market: many challengers over the same public stream.
#[derive(Default)]
pub struct ChallengerMarket {
    challengers: BTreeMap<[u8; 32], Challenger>,
}

impl ChallengerMarket {
    pub fn new() -> ChallengerMarket {
        ChallengerMarket::default()
    }

    pub fn register(&mut self, id: [u8; 32], weights: Weights) -> Result<(), ChallengerError> {
        if self.challengers.contains_key(&id) {
            return Err(ChallengerError::AlreadyRegistered { id });
        }
        self.challengers.insert(id, Challenger::register(id, weights));
        Ok(())
    }

    pub fn commit(&mut self, id: &[u8; 32], height: u64) -> Result<Hash256, ChallengerError> {
        self.challengers
            .get_mut(id)
            .ok_or(ChallengerError::NotRegistered { id: *id })?
            .commit(height)
    }

    /// Every challenger processes the public event; (id, per-block score).
    pub fn process(
        &mut self,
        event: BlockEvent,
    ) -> Result<Vec<([u8; 32], Option<u128>)>, ChallengerError> {
        let mut out = Vec::with_capacity(self.challengers.len());
        for (&id, c) in self.challengers.iter_mut() {
            out.push((id, c.process(event.clone())?));
        }
        Ok(out)
    }

    pub fn challenger(&self, id: &[u8; 32]) -> Option<&Challenger> {
        self.challengers.get(id)
    }

    pub fn evaluate_gate(
        &self,
        id: &[u8; 32],
        start_height: u64,
        incumbent_losses: &BTreeMap<u64, u128>,
    ) -> Result<GateOutcome, ChallengerError> {
        let c = self
            .challengers
            .get(id)
            .ok_or(ChallengerError::NotRegistered { id: *id })?;
        Ok(c.evaluate_gate(start_height, incumbent_losses))
    }

    pub fn ids(&self) -> impl Iterator<Item = [u8; 32]> + '_ {
        self.challengers.keys().copied()
    }

    pub fn len(&self) -> usize {
        self.challengers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.challengers.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::block_loop::KnowledgeState;

    fn reveal(height: u64, v: u64) -> BlockEvent {
        BlockEvent::Reveal { height, delta: Delta([v; DIMS]), fee_sum: 1_000, bucket: 3 }
    }

    fn incumbent(loss: u128) -> BTreeMap<u64, u128> {
        (1..=WINDOW_BLOCKS).map(|h| (h, loss)).collect()
    }

    fn seeded(scores: &[(u64, u128)]) -> Challenger {
        let mut c = Challenger::register([8u8; 32], Weights::reference());
        for &(h, l) in scores {
            c.scores.insert(h, l);
        }
        c
    }

    #[test]
    fn layout_pins() {
        assert_eq!(WINDOW_BLOCKS, 2016);
        assert_eq!(MIN_COVERAGE_PERMILLE, 900);
        assert_eq!(MARGIN_PERMILLE, 50);
        // ⌈2016·900/1000⌉ = 1815.
        assert_eq!((WINDOW_BLOCKS * MIN_COVERAGE_PERMILLE + 999) / 1000, 1815);
    }

    #[test]
    fn commit_reveal_discipline() {
        let mut c = Challenger::register([1u8; 32], Weights::reference());
        assert_eq!(c.next_height(), 1);
        assert!(matches!(
            c.commit(2),
            Err(ChallengerError::PrematureCommit { height: 2, expected: 1 })
        ));
        let h = c.commit(1).unwrap();
        assert_eq!(h, c.commit(1).unwrap(), "idempotent");
        let pred = predict(c.weights(), c.window());
        assert_eq!(h, Hash256::concat(&DERIVED_PRED, &pred.canonical_bytes()));
        assert!(c.process(reveal(1, 500)).unwrap().is_some());
        assert_eq!(c.scores().len(), 1);
        assert!(matches!(
            c.commit(1),
            Err(ChallengerError::CommitAfterReveal { height: 1 })
        ));
        assert!(matches!(
            c.process(reveal(3, 1)),
            Err(ChallengerError::NonContiguous { expected: 2, found: 3 })
        ));
    }

    #[test]
    fn uncommitted_and_mismatched_blocks_score_nothing() {
        let mut c = Challenger::register([2u8; 32], Weights::reference());
        assert!(c.process(reveal(1, 100)).unwrap().is_none(), "no commit");
        assert!(c.scores().is_empty());
        let _ = c.commit(2);
        *c.commits.get_mut(&2).unwrap() = Hash256::from_bytes([0xEE; 32]);
        assert!(c.process(reveal(2, 100)).unwrap().is_none(), "mismatched commit");
        assert!(c.scores().is_empty());
        c.commit(3).unwrap();
        assert!(c.process(reveal(3, 100)).unwrap().is_some());
        assert_eq!(c.scores().len(), 1);
    }

    #[test]
    fn misses_carry_without_scoring() {
        let mut c = Challenger::register([3u8; 32], Weights::reference());
        c.commit(1).unwrap();
        c.process(reveal(1, 100)).unwrap();
        c.commit(2).unwrap();
        assert!(c.process(BlockEvent::Miss { height: 2, legs: 128 }).unwrap().is_none());
        assert!(c.process(reveal(3, 100)).unwrap().is_none(), "no commit for 3");
        assert_eq!(c.next_height(), 4);
        assert_eq!(c.scores().len(), 1, "the stale commit for the miss is inert");
    }

    #[test]
    fn scores_match_an_independent_computation() {
        let mut c = Challenger::register([4u8; 32], Weights::reference());
        for h in 1..=6u64 {
            c.commit(h).unwrap();
            c.process(reveal(h, 10_000 + h)).unwrap();
        }
        let mut w = Window::new();
        for h in 1..=6u64 {
            let pred = predict(c.weights(), &w);
            let r = huber::residual(&Delta([10_000 + h; DIMS]), &pred);
            let mut scales = [0u64; DIMS];
            for (j, s) in scales.iter_mut().enumerate() {
                *s = c.weights().scale(j);
            }
            assert_eq!(c.scores()[&h], huber::total_loss(&r, &scales), "block {h}");
            w.push(Observation { delta: Delta([10_000 + h; DIMS]), fee_sum: 1_000, bucket: 3 });
        }
    }

    #[test]
    fn determinism() {
        let mut a = Challenger::register([5u8; 32], Weights::reference());
        let mut b = Challenger::register([6u8; 32], Weights::reference());
        for h in 1..=5u64 {
            a.commit(h).unwrap();
            b.commit(h).unwrap();
            let e = reveal(h, 777 * h);
            a.process(e.clone()).unwrap();
            b.process(e).unwrap();
        }
        assert_eq!(a.scores(), b.scores());
        assert_eq!(a.commits(), b.commits());
    }

    #[test]
    fn gate_coverage_threshold() {
        let mk = |n: u64| seeded(&(1..=n).map(|h| (h, 100)).collect::<Vec<_>>());
        assert!(matches!(
            mk(1814).evaluate_gate(1, &incumbent(1_000)),
            GateOutcome::NotEligible(EligibilityFailure::InsufficientCoverage {
                covered: 1814,
                required: 1815,
            })
        ));
        match mk(1815).evaluate_gate(1, &incumbent(1_000)) {
            GateOutcome::Promote { record, .. } => {
                assert_eq!(record.covered_blocks, 1815);
                assert_eq!(record.challenger_loss_q32, 1815 * 100);
                assert_eq!(record.incumbent_loss_q32, 1815 * 1_000);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn gate_margin_arithmetic() {
        // Equal losses: the margin fails.
        let c = seeded(&(1..=WINDOW_BLOCKS).map(|h| (h, 500)).collect::<Vec<_>>());
        assert!(matches!(
            c.evaluate_gate(1, &incumbent(500)),
            GateOutcome::NotEligible(EligibilityFailure::MarginNotMet { .. })
        ));
        // Exactly at the 5% margin: not strictly less → fails.
        let at = 950u128;
        let c = seeded(&(1..=WINDOW_BLOCKS).map(|h| (h, at)).collect::<Vec<_>>());
        assert!(matches!(
            c.evaluate_gate(1, &incumbent(1_000)),
            GateOutcome::NotEligible(EligibilityFailure::MarginNotMet { .. })
        ));
        // One unit per block better: promotes.
        let c = seeded(&(1..=WINDOW_BLOCKS).map(|h| (h, at - 1)).collect::<Vec<_>>());
        match c.evaluate_gate(1, &incumbent(1_000)) {
            GateOutcome::Promote { record, weights } => {
                assert_eq!(record.covered_blocks, WINDOW_BLOCKS);
                assert_eq!(record.challenger_loss_q32, 2016 * 949);
                assert_eq!(record.incumbent_loss_q32, 2016 * 1_000);
                assert_eq!(record.window_start, 1);
                assert_eq!(record.window_end, WINDOW_BLOCKS);
                assert_eq!(record.challenger, [8u8; 32]);
                assert_eq!(weights, *c.weights());
            }
            other => panic!("{other:?}"),
        }
        // A perfect incumbent can never be beaten by a margin.
        let c = seeded(&(1..=WINDOW_BLOCKS).map(|h| (h, 0)).collect::<Vec<_>>());
        assert!(matches!(
            c.evaluate_gate(1, &incumbent(0)),
            GateOutcome::NotEligible(_)
        ));
    }

    #[test]
    fn gate_window_scope_and_missing_incumbent() {
        let mut scores: Vec<(u64, u128)> = (1..=1815).map(|h| (h, 100)).collect();
        scores.push((WINDOW_BLOCKS + 100, 0));
        match seeded(&scores).evaluate_gate(1, &incumbent(1_000)) {
            GateOutcome::Promote { record, .. } => {
                assert_eq!(record.covered_blocks, 1815, "the outside score is ignored");
                assert_eq!(record.challenger_loss_q32, 1815 * 100);
            }
            other => panic!("{other:?}"),
        }
        let c = seeded(&(1..=WINDOW_BLOCKS).map(|h| (h, 100)).collect::<Vec<_>>());
        let mut inc = incumbent(1_000);
        inc.remove(&1_000);
        assert!(matches!(
            c.evaluate_gate(1, &inc),
            GateOutcome::NotEligible(EligibilityFailure::MissingIncumbentScore { block: 1_000 })
        ));
        let empty = Challenger::register([9u8; 32], Weights::reference());
        assert!(matches!(
            empty.evaluate_gate(50, &incumbent(1)),
            GateOutcome::NotEligible(EligibilityFailure::InsufficientCoverage {
                covered: 0,
                required: 1815,
            })
        ));
    }

    #[test]
    fn the_market_dispatches() {
        let mut m = ChallengerMarket::new();
        assert!(m.is_empty());
        m.register([1u8; 32], Weights::reference()).unwrap();
        assert!(matches!(
            m.register([1u8; 32], Weights::reference()),
            Err(ChallengerError::AlreadyRegistered { .. })
        ));
        m.register([2u8; 32], Weights::zero()).unwrap();
        for h in 1..=3u64 {
            m.commit([1u8; 32], h).unwrap();
            let results = m.process(reveal(h, 500)).unwrap();
            assert_eq!(results.len(), 2);
            for (id, s) in &results {
                if *id == [1u8; 32] {
                    assert!(s.is_some());
                } else {
                    assert!(s.is_none(), "challenger 2 never committed");
                }
            }
        }
        assert!(matches!(
            m.commit([9u8; 32], 4),
            Err(ChallengerError::NotRegistered { .. })
        ));
        assert!(matches!(
            m.evaluate_gate([1u8; 32], 1, &incumbent(u128::MAX)).unwrap(),
            GateOutcome::NotEligible(EligibilityFailure::InsufficientCoverage { covered: 3, .. })
        ));
        assert!(m.challenger(&[2u8; 32]).is_some());
        assert_eq!(m.ids().count(), 2);
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn real_pipeline_short() {
        let mut state = KnowledgeState::genesis();
        let mut c = Challenger::register([0xA1; 32], Weights::reference());
        let mut incumbent_losses = BTreeMap::new();
        for h in 1..=6u64 {
            let e = reveal(h, 3_000 + h * 17);
            c.commit(h).unwrap();
            let rec = state.process(e.clone()).unwrap();
            c.process(e).unwrap();
            incumbent_losses.insert(h, rec.loss_q32.unwrap());
        }
        // Block 1: identical parameterization, identical window, identical
        // scales (both at the reference default) → identical losses.
        assert_eq!(c.scores()[&1], incumbent_losses[&1]);
        // Later blocks: the incumbent's scales evolve, the challenger's are
        // frozen — both Huber totals over the same residuals, recorded.
        for h in 2..=6 {
            assert!(c.scores().contains_key(&h));
            assert!(incumbent_losses.contains_key(&h));
        }
    }

    #[test]
    #[ignore = "the full-window end-to-end: minutes in debug; `cargo test -p nerv-knowledge --release -- --ignored`"]
    fn the_full_pipeline_promotes_the_perfect_challenger() {
        const C: u64 = 40_000;
        let mut w = Weights::zero();
        for j in 0..DIMS {
            w.set_bias(j, C as i64);
        }
        let mut state = KnowledgeState::genesis();
        let mut c = Challenger::register([0xBB; 32], w);
        let mut incumbent_losses = BTreeMap::new();
        for h in 1..=WINDOW_BLOCKS {
            let e =
                BlockEvent::Reveal { height: h, delta: Delta([C; DIMS]), fee_sum: 0, bucket: 0 };
            c.commit(h).unwrap();
            let rec = state.process(e.clone()).unwrap();
            c.process(e).unwrap();
            incumbent_losses.insert(h, rec.loss_q32.unwrap());
        }
        match c.evaluate_gate(1, &incumbent_losses) {
            GateOutcome::Promote { record, weights } => {
                assert_eq!(record.covered_blocks, WINDOW_BLOCKS);
                assert_eq!(record.challenger_loss_q32, 0, "the perfect challenger scores zero");
                assert!(record.incumbent_loss_q32 > 0);
                assert_eq!(weights.bias(0), C as i64);
            }
            other => panic!("expected promotion: {other:?}"),
        }
    }
}
