//! The three-tier proof fabric's folding layer (WP §5.5; design doc
//! fold/). Tier 1 (bundle) and tier 2 (global) AIRs land per erratum 85;
//! dedup is the canonical-set machinery both tiers and the registry
//! consume.


pub mod dedup;
pub use dedup::{BundleTxids, DedupError, DedupReport, IntervalLedger, IntervalSet};
