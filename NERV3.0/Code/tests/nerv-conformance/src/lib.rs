//! # nerv-conformance — the conformance registry
//!
//! Spec authority (validate + hash specs/params.toml), the frozen vector
//! registry (freeze/verify with tamper, spec-drift, schema-drift, and
//! generator-drift detection), and reference implementations that are pure
//! functions of the frozen parameters (the genesis emission schedule).
//! Generated here: container_encoding, emission_schedule. The six whitepaper
//! families are schema-pinned as deferred and fill with their owning chunks,
//! freezing at M1 (WP §14.1).

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

//! Cross-crate integration tests for the chunks 13–19 surface
//! (erratum 199). These are the tests the CI conformance job runs
//! on every PR.


pub mod integration;


#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn conformance_crate_loads() {
        // The crate itself is the integration surface; the tests are
        // in the integration module.
        assert!(true);
    }
}
