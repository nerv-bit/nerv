//! # nerv-consensus — the beacon, committees, and finality (WP §§4.6–4.7, 8.3, 11.2)
//!
//! * `committee`   — epoch sortition: role-separated randomness, the
//!                   staggered beacon cohorts, attestation signer subsets.
//! * `qc`          — quorum certificates in committee context; same-slot
//!                   equivocation detection (the DoubleSign surface).
//! * `attestation` — A_τ interval attestations, the 𝔾 tree, the
//!                   epoch-attestation hash chain.
//! * `finality`    — the interval watermark and the per-shard finalized
//!                   line; no reorg past a finalized interval.
//! * `topology`    — split/merge hysteresis on finalized load metrics.
//! * `slash`       — the four provable offense classes.
//!
//! (Part 3 adds `beacon` and `shard_chain`.) No floating point (P5).

#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic, clippy::float_cmp)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod attestation;
pub mod beacon;
pub mod committee;
pub mod finality;
pub mod qc;
pub mod shard_chain;
pub mod slash;
pub mod topology;


#[cfg(test)]
mod testutil;

pub use attestation::{
    verify_chain, verify_g_witness, EpochAttestation, GTree, IntervalAttestation,
};
pub use committee::{
    attestation_signers, beacon_cohort, beacon_committee, derive_randomness, epoch_randomness,
    genesis_randomness, select, Committees, ATTESTATION_QUORUM, ATTESTATION_SIGNERS,
    BEACON_COHORTS, BEACON_COMMITTEE_SIZE, DECRYPTION_COMMITTEE_SIZE, REGISTRY_COMMITTEE_SIZE,
    SHARD_COMMITTEE_SIZE, STAGGER_START,
};
pub use finality::{
    FinalityViolation, IntervalFinality, ObserveOutcome, ShardFinality,
};
pub use qc::{body_hash, detect_double_sign, signer_intersection, DoubleSignError, HeaderQc};
pub use slash::{Offender, SlashClass, SlashContext, SlashError, SlashEvidence, VerifiedSlash};
pub use beacon::{BeaconError, BeaconState, GenesisMap};
pub use shard_chain::{
   assemble_with_qc, propose_block, validate_and_sign,
   PipelineError, Proposal, SHARD_QUORUM,
};
pub use topology::{
   evaluate_merge, evaluate_split, merge_qualifies, sunset_report, SplitDecision, SunsetReport,
   TopologyAction, TopologyEngine, ShardLoad, ShardLoads, SUNSET_NEGLIGIBLE_LEGS_PER_SEC,
};

