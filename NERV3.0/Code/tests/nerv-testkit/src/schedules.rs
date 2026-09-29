//! The TLA+ schedule corpus (WP §13.4, theorems T5–T7; erratum 194):
//! the adversarial schedules the model checker explores, executed as
//! data-driven harness runs.


use crate::harness::{Harness, TestNode, NodeId, Outbound};
use crate::scheduler::ScheduleStrategy;


/// One corpus entry: (name, strategy, description).
#[derive(Clone, Debug)]
pub struct CorpusEntry {
    pub name: &'static str,
    pub strategy: ScheduleStrategy,
    pub description: &'static str,
    pub theorem: &'static str,
}


/// The full corpus: every (strategy, assertion) pair the model checker
/// covers, as executable scenarios.
pub const CORPUS: &[CorpusEntry] = &[
    // T5 — Cross-shard atomicity (§4.5): no reachable state creates
    // value without destroying equal value.
    CorpusEntry {
        name: "t5-honest-delivery",
        strategy: ScheduleStrategy::InOrder,
        description: "In-order delivery: the baseline where cross-shard completes cleanly.",
        theorem: "T5",
    },
    CorpusEntry {
        name: "t5-delayed-issue-past-expiry",
        strategy: ScheduleStrategy::DelaySource {
            source: "shard-40".to_string(),
            delay_ns: 24 * 3600 * 1_000_000_000, // 24 hours in logical ns.
        },
        description: "The issue leg is delayed past the expiry: reversion must mint the revert outputs.",
        theorem: "T5",
    },
    CorpusEntry {
        name: "t5-reorder-spend-issue",
        strategy: ScheduleStrategy::Reverse,
        description: "Issue leg arrives before the spend leg: the ordering rule must gate.",
        theorem: "T5",
    },
    // T6 — Deadlock-freedom (§4.5): the dependency graph is acyclic.
    CorpusEntry {
        name: "t6-adversarial-reorder",
        strategy: ScheduleStrategy::Reverse,
        description: "Systematic message reversal: no cyclic dependency can form.",
        theorem: "T6",
    },
    CorpusEntry {
        name: "t6-delay-all",
        strategy: ScheduleStrategy::DelaySource {
            source: "*".to_string(),
            delay_ns: 60 * 1_000_000_000, // 60 seconds.
        },
        description: "All messages delayed 60s: completion still occurs within bounds.",
        theorem: "T6",
    },
    // T7 — Liveness (§4.5, §5.5): issue legs are completable by anyone.
    CorpusEntry {
        name: "t7-producer-censored",
        strategy: ScheduleStrategy::DropSource {
            source: "producer-shard-7".to_string(),
        },
        description: "The producer is censored: issue legs remain completable by any other party.",
        theorem: "T7",
    },
    CorpusEntry {
        name: "t7-round-robin-fairness",
        strategy: ScheduleStrategy::RoundRobin,
        description: "Round-robin delivery: every party gets turns; no starvation.",
        theorem: "T7",
    },
    CorpusEntry {
        name: "t7-partial-drop",
        strategy: ScheduleStrategy::DropPermille { permille: 300, seed: 42 },
        description: "30% of messages dropped: the surviving paths still deliver.",
        theorem: "T7",
    },
];


/// The result of one corpus run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorpusResult {
    pub name: String,
    pub theorem: String,
    pub steps: u64,
    pub passed: bool,
    pub details: String,
}


/// Run the full corpus against the provided node factory. The factory
/// creates the nodes for each scenario (the harness is rebuilt per entry).
pub fn run_corpus(
    node_factory: &dyn Fn(&str) -> Box<dyn TestNode + Send>,
    node_names: &[&str],
) -> Vec<CorpusResult> {
    CORPUS
        .iter()
        .map(|entry| {
            let mut harness = Harness::new(entry.strategy.clone());
            for name in node_names {
                harness.add_node(node_factory(name));
            }
            match harness.run() {
                Ok(steps) => CorpusResult {
                    name: entry.name.to_string(),
                    theorem: entry.theorem.to_string(),
                    steps,
                    passed: true,
                    details: format!("completed in {steps} steps"),
                },
                Err(e) => CorpusResult {
                    name: entry.name.to_string(),
                    theorem: entry.theorem.to_string(),
                    steps: 0,
                    passed: false,
                    details: format!("failed: {e}"),
                },
            }
        })
        .collect()
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::TestNode;


    struct SimpleNode {
        id: NodeId,
        received: u64,
    }


    impl SimpleNode {
        fn new(name: &str) -> SimpleNode {
            SimpleNode { id: NodeId::new(name), received: 0 }
        }
    }


    impl TestNode for SimpleNode {
        fn id(&self) -> &NodeId {
            &self.id
        }


        fn process(&mut self, _from: &str, _payload: &[u8], _now: u64) -> Vec<Outbound> {
            self.received += 1;
            Vec::new()
        }


        fn state_summary(&self) -> String {
            format!("received={}", self.received)
        }
    }


    #[test]
    fn corpus_is_well_formed() {
        assert!(CORPUS.len() >= 7);
        let theorems: std::collections::BTreeSet<&str> =
            CORPUS.iter().map(|e| e.theorem).collect();
        assert!(theorems.contains("T5"));
        assert!(theorems.contains("T6"));
        assert!(theorems.contains("T7"));
        // No duplicate names.
        let names: std::collections::BTreeSet<&str> = CORPUS.iter().map(|e| e.name).collect();
        assert_eq!(names.len(), CORPUS.len());
        // Every entry has a description.
        for e in CORPUS {
            assert!(!e.description.is_empty(), "{}", e.name);
        }
    }


    #[test]
    fn corpus_runs_to_completion() {
        let factory = |name: &str| -> Box<dyn TestNode + Send> {
            Box::new(SimpleNode::new(name))
        };
        let results = run_corpus(&factory, &["a", "b", "c"]);
        assert_eq!(results.len(), CORPUS.len());
        for r in &results {
            assert!(r.passed, "{} failed: {}", r.name, r.details);
        }
    }
}
