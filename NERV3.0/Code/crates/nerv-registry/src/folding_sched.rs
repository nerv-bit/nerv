
//! G_τ production deadlines and grace handling (WP §5.5; erratum 122).
//! Intake closes at the interval boundary; the folded proof is awaited
//! until boundary + grace; past that the committee degrades.


use nerv_core::params::{REGISTRY_FOLDING_GRACE_SECS, TIMING_BEACON_INTERVAL_SECS};
use nerv_core::types::Interval;


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldingDecision {
    /// Intake open: collect bundles.
    Collect,
    /// A folded G_τ is in hand: verify it and finalize (the steady path).
    Steady,
    /// Intake closed, inside the grace window, no proof yet.
    AwaitProof,
    /// Grace expired without delivery: the committee attests (degraded).
    Degraded,
}


/// One interval's folding timeline. All times are wall-clock seconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldingWindow {
    pub interval: Interval,
    pub intake_closes_secs: u64,
    pub grace_ends_secs: u64,
}


impl FoldingWindow {
    pub fn for_interval(interval: Interval) -> FoldingWindow {
        let end = interval.start_secs().saturating_add(TIMING_BEACON_INTERVAL_SECS);
        FoldingWindow {
            interval,
            intake_closes_secs: end,
            grace_ends_secs: end.saturating_add(REGISTRY_FOLDING_GRACE_SECS),
        }
    }


    pub fn decision(&self, now_secs: u64, proof_delivered: bool) -> FoldingDecision {
        if now_secs < self.intake_closes_secs {
            FoldingDecision::Collect
        } else if proof_delivered {
            FoldingDecision::Steady
        } else if now_secs < self.grace_ends_secs {
            FoldingDecision::AwaitProof
        } else {
            FoldingDecision::Degraded
        }
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;


    #[test]
    fn window_arithmetic() {
        assert_eq!(TIMING_BEACON_INTERVAL_SECS, 1);
        assert_eq!(REGISTRY_FOLDING_GRACE_SECS, 5);
        let w = FoldingWindow::for_interval(Interval::from_u64(86_400));
        assert_eq!(w.intake_closes_secs, 86_401);
        assert_eq!(w.grace_ends_secs, 86_406);
        let w0 = FoldingWindow::for_interval(Interval::from_u64(0));
        assert_eq!(w0.intake_closes_secs, 1);
        assert_eq!(w0.grace_ends_secs, 6);
    }


    #[test]
    fn decision_table() {
        let w = FoldingWindow::for_interval(Interval::from_u64(7));
        assert_eq!(w.decision(7, false), FoldingDecision::Collect);
        assert_eq!(w.decision(7, true), FoldingDecision::Collect);
        assert_eq!(w.decision(8, false), FoldingDecision::AwaitProof);
        assert_eq!(w.decision(8, true), FoldingDecision::Steady);
        assert_eq!(w.decision(12, false), FoldingDecision::AwaitProof);
        assert_eq!(w.decision(12, true), FoldingDecision::Steady);
        assert_eq!(w.decision(13, false), FoldingDecision::Degraded);
        assert_eq!(w.decision(13, true), FoldingDecision::Steady);
        assert_eq!(w.decision(u64::MAX, false), FoldingDecision::Degraded);
        assert_eq!(w.decision(u64::MAX, true), FoldingDecision::Steady);
    }
}
