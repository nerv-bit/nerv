//! Split/merge hysteresis on finalized load metrics (WP §8.5; erratum
//! 128): deterministic thresholds, the notice period, and the topology
//! engine that applies actions through ShardSet — never re-homing a note.

use std::collections::BTreeMap;

use nerv_core::params::{
    PROTOCOL_SHARD_COUNT_MAX, SHARDING_ADOPTION_NOTICE_DAYS,
    SHARDING_EMERGENCY_OVERLOAD_MULTIPLE, SHARDING_ENGINEERED_CEILING_LEGS_PER_SEC,
    SHARDING_MERGE_THRESHOLD_LEGS_PER_SEC, SHARDING_MERGE_WINDOW_DAYS,
    SHARDING_SPLIT_THRESHOLD_LEGS_PER_SEC, SHARDING_SPLIT_WINDOW_DAYS, TIMING_EPOCH_SECS,
};
use nerv_core::types::{Epoch, ShardId, ShardSet};

pub const SECS_PER_DAY: u64 = 86_400;
/// Days map 1:1 onto 24-hour epochs (params pin; erratum 128).
const _: () = assert!(TIMING_EPOCH_SECS == SECS_PER_DAY);

/// The sunset audit's residual threshold (1% of S_lo; genesis-config).
pub const SUNSET_NEGLIGIBLE_LEGS_PER_SEC: u64 =
    SHARDING_MERGE_THRESHOLD_LEGS_PER_SEC / 100;

/// One shard's finalized settled-leg counts, by epoch.
pub type ShardLoad = BTreeMap<Epoch, u64>;
pub type ShardLoads = BTreeMap<ShardId, ShardLoad>;

fn epochs_of(days: u64) -> u64 {
    days * (TIMING_EPOCH_SECS / SECS_PER_DAY)
}

/// The total over the trailing `window_days` epochs ending at `at`
/// (exclusive); requires every epoch present.
pub fn trailing_total(load: &ShardLoad, at: Epoch, window_days: u64) -> Option<u64> {
    let w = epochs_of(window_days);
    let mut total = 0u64;
    for k in 0..w {
        let e = Epoch::from_u64(at.as_u64().checked_sub(k + 1)?);
        total = total.checked_add(*load.get(&e)?)?;
    }
    Some(total)
}

pub fn trailing_legs_per_sec(load: &ShardLoad, at: Epoch, window_days: u64) -> Option<u64> {
    let total = trailing_total(load, at, window_days)?;
    Some(total / (epochs_of(window_days) * SECS_PER_DAY))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitDecision {
    None,
    /// The trailing average strictly exceeds S_hi.
    Propose,
    /// The trailing average strictly exceeds 2× the engineered ceiling —
    /// detection only; adoption is governance's (§C.2).
    Emergency,
}

pub fn evaluate_split(load: &ShardLoad, at: Epoch) -> SplitDecision {
    let Some(avg) = trailing_legs_per_sec(load, at, SHARDING_SPLIT_WINDOW_DAYS) else {
        return SplitDecision::None;
    };
    if avg > SHARDING_EMERGENCY_OVERLOAD_MULTIPLE * SHARDING_ENGINEERED_CEILING_LEGS_PER_SEC {
        SplitDecision::Emergency
    } else if avg > SHARDING_SPLIT_THRESHOLD_LEGS_PER_SEC {
        SplitDecision::Propose
    } else {
        SplitDecision::None
    }
}

/// The merge streak: every trailing `window` epoch's per-second rate is
/// strictly below S_lo (erratum 128).
pub fn merge_qualifies(load: &ShardLoad, at: Epoch) -> bool {
    let w = epochs_of(SHARDING_MERGE_WINDOW_DAYS);
    for k in 0..w {
        let Some(e) = at.as_u64().checked_sub(k + 1) else { return false };
        let Some(&legs) = load.get(&Epoch::from_u64(e)) else { return false };
        if legs / SECS_PER_DAY >= SHARDING_MERGE_THRESHOLD_LEGS_PER_SEC {
            return false;
        }
    }
    true
}

pub fn evaluate_merge(a: &ShardLoad, b: &ShardLoad, at: Epoch) -> bool {
    merge_qualifies(a, at) && merge_qualifies(b, at)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TopologyAction {
    /// Replace `target` with its two children (§8.2).
    Split { target: ShardId },
    /// Replace the sibling pair with their parent.
    Merge { a: ShardId, b: ShardId },
}

/// The topology state machine: the active set plus pending actions
/// keyed by activation epoch (erratum 128).
#[derive(Clone, Debug)]
pub struct TopologyEngine {
    active: ShardSet,
    pending: BTreeMap<Epoch, Vec<TopologyAction>>,
}

impl TopologyEngine {
    pub fn new(active: ShardSet) -> TopologyEngine {
        TopologyEngine { active, pending: BTreeMap::new() }
    }

    pub fn active(&self) -> &ShardSet {
        &self.active
    }

    pub fn pending(&self) -> Vec<(Epoch, TopologyAction)> {
        self.pending
            .iter()
            .flat_map(|(e, v)| v.iter().map(move |a| (*e, *a)))
            .collect()
    }

    fn has_pending_for(&self, shard: &ShardId) -> bool {
        self.pending.values().any(|v| {
            v.iter().any(|act| match act {
                TopologyAction::Split { target } => target == shard,
                TopologyAction::Merge { a, b } => a == shard || b == shard,
            })
        })
    }

    /// Evaluate the finalized loads at the boundary into `at`; enqueue
    /// routine proposals with the notice. Returns the proposals.
    pub fn evaluate(&mut self, at: Epoch, loads: &ShardLoads) -> Vec<TopologyAction> {
        let activation = Epoch::from_u64(at.as_u64() + epochs_of(SHARDING_ADOPTION_NOTICE_DAYS));
        let mut proposed = Vec::new();

        let mut split_budget = PROTOCOL_SHARD_COUNT_MAX.saturating_sub(self.active.len() as u64);
        for id in self.active.ids() {
            if split_budget == 0 {
                break;
            }
            let Some(load) = loads.get(id) else { continue };
            if matches!(evaluate_split(load, at), SplitDecision::Propose)
                && !self.has_pending_for(id)
            {
                proposed.push(TopologyAction::Split { target: *id });
                split_budget -= 1;
            }
        }

        let ids: Vec<ShardId> = self.active.ids().to_vec();
        for (i, &a) in ids.iter().enumerate() {
            for &b in &ids[i + 1..] {
                if a.sibling() != Some(b) {
                    continue;
                }
                let qualifies = match (loads.get(&a), loads.get(&b)) {
                    (Some(x), Some(y)) => evaluate_merge(x, y, at),
                    _ => false,
                };
                if qualifies && !self.has_pending_for(&a) && !self.has_pending_for(&b) {
                    proposed.push(TopologyAction::Merge { a, b });
                }
            }
        }

        if !proposed.is_empty() {
            self.pending.entry(activation).or_default().extend(proposed.iter().copied());
        }
        proposed
    }

    /// Emergency detection only (adoption is governance's; erratum 128).
    pub fn emergency_overloads(&self, at: Epoch, loads: &ShardLoads) -> Vec<ShardId> {
        self.active
            .ids()
            .iter()
            .filter(|id| {
                loads.get(id).is_some_and(|l| {
                    matches!(evaluate_split(l, at), SplitDecision::Emergency)
                })
            })
            .copied()
            .collect()
    }

    /// Apply every action due at or before `at`, in epoch order and
    /// sorted within an epoch; no-longer-applicable actions are skipped.
    pub fn advance(&mut self, at: Epoch) -> Vec<TopologyAction> {
        let mut applied = Vec::new();
        let due: Vec<Epoch> = self.pending.range(..=at).map(|(e, _)| *e).collect();
        for e in due {
            let mut actions = self.pending.remove(&e).unwrap_or_default();
            actions.sort();
            for act in actions {
                let applied_ok = match act {
                    TopologyAction::Split { target } => self
                        .active
                        .split(&target)
                        .map(|next| {
                            self.active = next;
                        })
                        .is_ok(),
                    TopologyAction::Merge { a, b } => {
                        self.active.contains(&a)
                            && self.active.contains(&b)
                            && a.sibling() == Some(b)
                            && self.active.merge(&a).map(|next| self.active = next).is_ok()
                    }
                };
                if applied_ok {
                    applied.push(act);
                }
            }
        }
        applied
    }
}

/// The sunset audit (§8.5): the trailing-7-epoch residual.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SunsetReport {
    pub avg_legs_per_sec: Option<u64>,
    pub negligible: bool,
}

pub fn sunset_report(load: &ShardLoad, at: Epoch) -> SunsetReport {
    let avg = trailing_legs_per_sec(load, at, SHARDING_SPLIT_WINDOW_DAYS);
    SunsetReport {
        avg_legs_per_sec: avg,
        negligible: avg.is_some_and(|v| v <= SUNSET_NEGLIGIBLE_LEGS_PER_SEC),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use nerv_core::types::ShardId;

    const S_HI: u64 = SHARDING_SPLIT_THRESHOLD_LEGS_PER_SEC;
    const CEILING: u64 = SHARDING_ENGINEERED_CEILING_LEGS_PER_SEC;
    const S_LO: u64 = SHARDING_MERGE_THRESHOLD_LEGS_PER_SEC;

    fn load_from(at: u64, window: u64, legs_per_epoch: u64) -> ShardLoad {
        (0..window)
            .map(|k| (Epoch::from_u64(at - 1 - k), legs_per_epoch))
            .collect()
    }

    fn set() -> ShardSet {
        ShardSet::genesis()
    }

    #[test]
    fn parameter_pins() {
        assert_eq!(S_HI, 4_000);
        assert_eq!(S_LO, 400);
        assert_eq!(CEILING, 10_000);
        assert_eq!(SHARDING_SPLIT_WINDOW_DAYS, 7);
        assert_eq!(SHARDING_MERGE_WINDOW_DAYS, 30);
        assert_eq!(SHARDING_ADOPTION_NOTICE_DAYS, 7);
        assert_eq!(SHARDING_EMERGENCY_OVERLOAD_MULTIPLE, 2);
        assert_eq!(SUNSET_NEGLIGIBLE_LEGS_PER_SEC, 4);
    }

    #[test]
    fn trailing_average_math() {
        // 345.6M legs/epoch over 7 epochs → 4000/s exactly.
        let l = load_from(10, 7, 345_600_000);
        assert_eq!(trailing_legs_per_sec(&l, Epoch::from_u64(10), 7), Some(4_000));
        assert_eq!(trailing_total(&l, Epoch::from_u64(10), 7), Some(7 * 345_600_000));
        // Not exceeding S_hi: no proposal at exactly the threshold.
        assert_eq!(evaluate_split(&l, Epoch::from_u64(10)), SplitDecision::None);
        // 346M/epoch → 4004/s → Propose.
        let l = load_from(10, 7, 346_000_000);
        assert_eq!(evaluate_split(&l, Epoch::from_u64(10)), SplitDecision::Propose);
        // 1.73B/epoch → 20,023/s → Emergency (> 20,000).
        let l = load_from(10, 7, 1_733_000_000);
        assert_eq!(evaluate_split(&l, Epoch::from_u64(10)), SplitDecision::Emergency);
        // Exactly 2× ceiling (1.728B/epoch) is NOT emergency ("beyond").
        let l = load_from(10, 7, 1_728_000_000);
        assert_eq!(evaluate_split(&l, Epoch::from_u64(10)), SplitDecision::Propose);
        // Missing epochs → None.
        let mut short = load_from(10, 7, 400_000_000);
        short.remove(&Epoch::from_u64(6));
        assert_eq!(trailing_legs_per_sec(&short, Epoch::from_u64(10), 7), None);
        assert_eq!(evaluate_split(&short, Epoch::from_u64(10)), SplitDecision::None);
        // Window boundary: at=10 uses 3..=9.
        let l = load_from(10, 7, 346_000_000);
        assert_eq!(l.contains_key(&Epoch::from_u64(3)), true);
        assert_eq!(l.contains_key(&Epoch::from_u64(9)), true);
        assert_eq!(l.contains_key(&Epoch::from_u64(10)), false);
        // Not enough history at all.
        assert_eq!(trailing_legs_per_sec(&ShardLoad::new(), Epoch::from_u64(3), 7), None);
    }

    #[test]
    fn merge_streak_rule() {
        // 30M legs/epoch → 347/s < 400 for all 30 epochs.
        let a = load_from(40, 30, 30_000_000);
        assert!(merge_qualifies(&a, Epoch::from_u64(40)));
        // One busy epoch breaks the streak.
        let mut broken = a.clone();
        broken.insert(Epoch::from_u64(39), 40_000_000);
        assert!(!merge_qualifies(&broken, Epoch::from_u64(40)));
        // Missing data disqualifies.
        let mut missing = a.clone();
        missing.remove(&Epoch::from_u64(20));
        assert!(!merge_qualifies(&missing, Epoch::from_u64(40)));
        // Both siblings must qualify.
        let b = load_from(40, 30, 30_000_000);
        assert!(evaluate_merge(&a, &b, Epoch::from_u64(40)));
        let mut hot_b = b.clone();
        hot_b.insert(Epoch::from_u64(15), 50_000_000);
        assert!(!evaluate_merge(&a, &hot_b, Epoch::from_u64(40)));
    }

    #[test]
    fn split_proposal_notice_and_activation() {
        let hot = set().ids()[7];
        let mut engine = TopologyEngine::new(set());
        let mut loads = ShardLoads::new();
        loads.insert(hot, load_from(10, 7, 400_000_000));
        let proposed = engine.evaluate(Epoch::from_u64(10), &loads);
        assert_eq!(proposed, vec![TopologyAction::Split { target: hot }]);
        assert_eq!(
            engine.pending(),
            vec![(Epoch::from_u64(17), TopologyAction::Split { target: hot })]
        );
        // Re-evaluation does not duplicate.
        assert!(engine.evaluate(Epoch::from_u64(11), &loads).is_empty());
        assert_eq!(engine.pending().len(), 1);
        // Before the notice: nothing.
        assert!(engine.advance(Epoch::from_u64(16)).is_empty());
        assert_eq!(engine.active().len(), 64);
        // At activation: the split applies.
        let applied = engine.advance(Epoch::from_u64(17));
        assert_eq!(applied, vec![TopologyAction::Split { target: hot }]);
        assert_eq!(engine.active().len(), 65);
        let (c0, c1) = (hot.child(false).unwrap(), hot.child(true).unwrap());
        assert!(engine.active().contains(&c0) && engine.active().contains(&c1));
        assert!(!engine.active().contains(&hot));
        assert!(engine.pending().is_empty());
        // Notes are never re-homed: the split target's old id is gone but
        // the set remains a valid prefix partition.
        assert_eq!(engine.active().ids().len(), 65);
    }

    #[test]
    fn merge_proposal_and_activation() {
        let g = set();
        let (a, b) = (g.ids()[6], g.ids()[7]);
        assert_eq!(a.sibling(), Some(b));
        let mut engine = TopologyEngine::new(g.clone());
        let mut loads = ShardLoads::new();
        loads.insert(a, load_from(40, 30, 30_000_000));
        loads.insert(b, load_from(40, 30, 30_000_000));
        let proposed = engine.evaluate(Epoch::from_u64(40), &loads);
        assert_eq!(proposed, vec![TopologyAction::Merge { a, b }]);
        assert!(engine.advance(Epoch::from_u64(46)).is_empty());
        let applied = engine.advance(Epoch::from_u64(47));
        assert_eq!(applied, vec![TopologyAction::Merge { a, b }]);
        assert_eq!(engine.active().len(), 63);
        let parent = a.parent().unwrap();
        assert!(engine.active().contains(&parent));
        assert!(!engine.active().contains(&a) && !engine.active().contains(&b));
    }

    #[test]
    fn cap_suppresses_splits() {
        let full: Vec<ShardId> = (0..1024u16)
            .map(|v| ShardId::new(10, v).unwrap())
            .collect();
        let engine = TopologyEngine::new(ShardSet::from_vec(full).unwrap());
        let hot = engine.active().ids()[7];
        let mut loads = ShardLoads::new();
        loads.insert(hot, load_from(10, 7, 400_000_000));
        assert!(engine.evaluate(Epoch::from_u64(10), &loads).is_empty());
        assert!(engine.pending().is_empty());
    }

    #[test]
    fn emergency_detection_and_sunset() {
        let hot = set().ids()[7];
        let mut engine = TopologyEngine::new(set());
        let mut loads = ShardLoads::new();
        loads.insert(hot, load_from(10, 7, 1_733_000_000));
        assert_eq!(engine.emergency_overloads(Epoch::from_u64(10), &loads), vec![hot]);
        // Emergency does not auto-propose.
        assert!(engine.evaluate(Epoch::from_u64(10), &loads).is_empty());

        let quiet = load_from(10, 7, 100_000); // ~1.157/s
        assert_eq!(sunset_report(&quiet, Epoch::from_u64(10)).avg_legs_per_sec, Some(1));
        assert!(sunset_report(&quiet, Epoch::from_u64(10)).negligible);
        let busy = load_from(10, 7, 400_000_000);
        assert!(!sunset_report(&busy, Epoch::from_u64(10)).negligible);
        assert!(!sunset_report(&ShardLoad::new(), Epoch::from_u64(3)).negligible);
    }

    #[test]
    fn stale_actions_are_skipped() {
        let g = set();
        let target = g.ids()[7];
        let mut engine = TopologyEngine::new(g.clone());
        let mut loads = ShardLoads::new();
        loads.insert(target, load_from(10, 7, 400_000_000));
        engine.evaluate(Epoch::from_u64(10), &loads);
        // Split early through a different path (governance-style direct
        // application is not this engine's, but advance is idempotent —
        // simulate by pre-splitting the set the engine holds).
        let split = engine.active.split(&target).unwrap();
        let mut engine2 = TopologyEngine { active: split, pending: engine.pending.clone() };
        let applied = engine2.advance(Epoch::from_u64(17));
        assert!(applied.is_empty(), "the target is no longer active");
        assert_eq!(engine2.active().len(), 65);
    }
}
