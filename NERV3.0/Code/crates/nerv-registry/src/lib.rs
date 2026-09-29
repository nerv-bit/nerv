//! # nerv-registry — the global validity registry (WP §5.5)
//!
//! * `mempool`       — the aggregator's validated-tx pool: admission,
//!                     dedup, canonical-order selection; the gate.
//! * `bundle`        — the aggregator's wire submission: the txid-set
//!                     commitment, the staked attestation, verification.
//! * `interval`      — T_τ construction: arrival-indexed bundles →
//!                     first-wins dedup → the interval set's tree.
//! * `folding_sched` — G_τ production deadlines and grace handling.
//! * `degraded`      — the committee-attestation fallback and the
//!                     30 s challenge window.
//! * `challenge`     — the single-proof fraud challenge (verify-one).
//!
//! The only proof check in this crate is nerv-proofs'
//! `verify_transaction` under the statement-11 binding. The T_τ tree is
//! nerv-state's (the executor's rule-1 verifier): one implementation for
//! builder and verifier. No floating point (P5).


#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic, clippy::float_cmp)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]


pub mod bundle;
pub mod challenge;
pub mod degraded;
pub mod error;
pub mod folding_sched;
pub mod interval;
pub mod mempool;


#[cfg(test)]
mod testutil;


pub use bundle::{verify_bundle, Bundle, BundleAttestation, BundleSummary, BUNDLE_MAX, BUNDLE_MIN};
pub use challenge::{ChallengeOutcome, InclusionChallenge};
pub use degraded::{DegradedAttestation, DegradedOutcome, DegradedProcess, ChallengeWindow};
pub use error::{BundleError, ChallengeError, DegradedError, IntervalError, MempoolError, VerificationError};
pub use folding_sched::{FoldingDecision, FoldingWindow};
pub use interval::{build_interval_commit, build_interval_commit_excluding, IntervalBuild, IntervalCommit};
pub use mempool::{Mempool, PoolEntry, VerifyContext};


const _: () = assert!(nerv_core::params::REGISTRY_BUNDLE_MAX >= nerv_core::params::REGISTRY_BUNDLE_MIN);
const _: () = assert!(nerv_core::params::REGISTRY_BUNDLE_MIN >= 1);
const _: () = assert!(nerv_core::params::REGISTRY_CHALLENGE_WINDOW_SECS > 0);
const _: () = assert!(nerv_core::params::REGISTRY_FOLDING_GRACE_SECS > 0);
// `&str == &str` is not const-stable; compare via byte-string match.
const _: () = match nerv_core::params::REGISTRY_DEGRADED_MODE.as_bytes() {
    b"committee-attestation" => (),
    _ => panic!("REGISTRY_DEGRADED_MODE must be \"committee-attestation\""),
};
