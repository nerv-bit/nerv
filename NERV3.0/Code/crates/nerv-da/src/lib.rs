//! # nerv-da — data availability (WP §8.7)
//!
//! * `erasure`      — GF(2⁸) systematic Cauchy Reed–Solomon, any-k-of-n.
//! * `blobs`        — the 2k×2k extended square, commitments, blob sets.
//! * `sampling`     — deterministic position streams, cell authentication,
//!                    iterated reconstruction.
//! * `availability` — the B7 bound math and the bad-encoding fraud.
//!
//! Bytes in, bytes out: the DA layer never interprets block contents.
//! Depends only on nerv-core (erratum 139). No floating point (P5).


#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic, clippy::float_cmp)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]


pub mod availability;
pub mod blobs;
pub mod erasure;
pub mod error;
pub mod sampling;


#[cfg(test)]
mod testutil;


pub use availability::{default_samples, samples_needed, BadEncodingEvidence, B7_DETECTION_PERMILLE, B7_WITHHOLD_PERMILLE};
pub use blobs::{
    cell_leaf, tree_path, tree_root, verify_tree_path, BlobSet, SetCommitment, Square,
    CHUNK_LEN, MAX_BLOB_DATA, MAX_K,
};
pub use error::{AvailabilityError, DaError, ErasureError};
pub use sampling::{reconstruct_verified, CellAuth, SamplePos, SampleResult, SampleRound};


/// Compile-time invariants for the DA parameters. `==` on `&str` isn't
/// const-stable, so we compare byte slices (which are) inside a `const`
/// block; the `const _:` makes the assertions fire at type-check time.
const _: () = {
    let scheme = nerv_core::params::DA_ERASURE_SCHEME;
    let expected_scheme: &[u8] = b"2d-reed-solomon";
    assert!(
        scheme.as_bytes().len() == expected_scheme.len()
            && {
                let mut ok = true;
                let mut i = 0;
                while i < expected_scheme.len() {
                    if scheme.as_bytes()[i] != expected_scheme[i] {
                        ok = false;
                    }
                    i += 1;
                }
                ok
            },
        "DA erasure scheme must be 2d-reed-solomon",
    );
    assert!(
        nerv_core::params::DA_B7_WITHHOLD_PERMILLE == 200,
        "DA B7 withhold permille must be 200",
    );
    assert!(
        nerv_core::params::DA_B7_DETECTION_PERMILLE == 999,
        "DA B7 detection permille must be 999",
    );
    assert!(
        nerv_core::params::DA_B7_WINDOW_SECS == 30,
        "DA B7 window seconds must be 30",
    );
};


