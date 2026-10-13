//! # nerv-conformance — the conformance registry
//!
//! Spec authority (validate + hash specs/params.toml), the frozen vector
//! registry (freeze/verify with tamper, spec-drift, schema-drift, and
//! generator-drift detection), and reference implementations that are pure
//! functions of the frozen parameters (the genesis emission schedule).
//! Generated here: container_encoding, emission_schedule. The six whitepaper
//! families are schema-pinned as deferred and fill with their owning chunks,
//! freezing at M1 (WP §14.1).
//!
//! Cross-crate integration tests for the chunks 13–19 surface
//! (erratum 199). These are the tests the CI conformance job runs
//! on every PR.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod encoding;
pub mod error;
pub mod primality;
pub mod registry;
pub mod schedule;
pub mod schema;
pub mod spec;
pub mod util;
pub mod vectors;

pub const SPEC_FILE: &str = "specs/params.toml";
pub const VECTORS_DIR: &str = "specs/vectors";

/// Cross-crate conformance tests for Phase C of `specs/road_to_testnet.md`.
///
/// Each submodule is the wired-up regression coverage for one of the six
/// Phase C sub-phases:
///   * `tx_lifecycle`        — C1a (deterministic-replay prove+apply) and C1b
///                              (full mixnet path, scaffolded with `#[ignore]`
///                              for C3).
///   * `cross_shard_escrow`  — C2 (conditional-output → claim end-to-end).
///   * `mixnet_path`         — C3 (5-relay aggregator decode).
///   * `da_sampling`         — C4 (DA round-trip + sample).
///   * `qc_collection`       — C5 (15-of-21 assembly + equivocation rejection).
///   * `epoch_rotation`      — C6 (committee rotation across one epoch).
///
/// Plus the pre-existing cross-cutting conformance tests for spec/registry/
/// encoding coverage.
///
/// The harness is intentionally explicit and deterministic — every
/// cryptographic object is a real `nerv-crypto` / `nerv-custody` value,
/// never a hand-computed stub. The shared helpers
/// (`testkit::TestWorld`, `testkit::seal`, `testkit::register_tau`, …)
/// come from `nerv_state::testkit`, enabled via the `testkit` feature on
/// the `nerv-state` dev-dependency in `Cargo.toml`.
#[cfg(test)]
mod tests;
