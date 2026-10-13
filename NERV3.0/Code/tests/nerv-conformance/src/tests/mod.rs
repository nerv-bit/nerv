//! Cross-crate conformance tests for NERV 3.0 Phase C.
//!
//! Every submodule is a unit test of the `nerv-conformance` crate, but
//! compiled in `cfg(test)` mode so it has full access to the public API of
//! every peer crate. The setup intentionally uses the public surfaces
//! (`nerv_state::testkit`, `nerv_net::testutil`, `nerv_consensus::testutil`)
//! rather than reaching into crate internals — every test must remain
//! valid even if a peer crate reorganises its file layout.
//!
//! The pre-existing spec / registry / encoding conformance coverage
//! (`zero_drift`, `spec_validation`, `registry_cycle`, `encoding_schema`)
//! lives in `tests/nerv-conformance/tests/` as Cargo integration tests
//! — they were originally authored in that style and were never
//! registered as submodules of this lib. Session 10 moves them to the
//! conventional location so they actually run.
//!
//! All conformance tests share three discipline rules (see
//! `specs/road_to_testnet.md §10` and the Session 10 hard rules):
//!
//!   1. Production-grade only. No `unimplemented!()`, no hand-computed
//!      hashes, no compile-only stubs. If a code path under test would
//!      silently degrade, the test fails by design.
//!   2. Every fix lands with its regression test. The tests below are
//!      the regression surface; any change that breaks one is a
//!      correctness regression that the Session-9 / Session-10 review
//!      discipline flags.
//!   3. Cite `file:line` in the final report. Module doc-comments below
//!      document the primary citation for each test.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::items_after_test_module)]

// Phase C integration coverage (road_to_testnet.md §12.1–§12.5).
mod tx_lifecycle;
mod cross_shard_escrow;
mod mixnet_path;
mod da_sampling;
mod qc_collection;
mod epoch_rotation;
