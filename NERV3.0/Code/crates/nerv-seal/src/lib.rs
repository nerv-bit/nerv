//! # nerv-seal — the sealed-delta channel (WP §6.3)
//!
//! NERV-Seal is the standard module-LWE construction parameterized for
//! additive aggregation over a 64-dimensional plaintext: digitized deltas
//! ride as polynomial coefficients under a per-shard-epoch threshold key;
//! block producers add ciphertexts componentwise; committees open one
//! aggregate per chunk with verifiable partial decryptions.
//!
//! * `ring`     — R_q arithmetic, the negacyclic NTT, ring inversion.
//! * `sampling` — binomial/ternary short vectors; uniform XOF expansion.
//! * `digitize` — δ ↔ 512 × 8-bit digits; public carry resolution.
//! * `noise`    — the budget: scale, variance model, provable bounds,
//!   committee smudging allocation, compile-time closure.
//! * `encrypt`  — ciphertexts, the epoch public key, aggregate addition.
//! * `decrypt`  — round-decode, chunk reveals, invalid-reveal detection.
//! * `dkg`      — dealerless DKG over R_q (WP §6.3.5), with the shared
//!   Lyubashevsky FS-with-aborts engine (`dkg::sigma`).
//! * `vpd`      — verifiable partial decryption (WP §6.3.3).
//!
//! Parameter posture (P9, errata 30–31, 40–41, 47–49). No floating point
//! anywhere (P5, DSR-11).


#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic, clippy::float_cmp)]


pub mod circuit_stmt;
pub mod dkg;
pub mod decrypt;
pub mod digitize;
pub mod encrypt;
pub mod error;
pub mod noise;
pub mod ring;
pub mod sampling;
pub mod vpd;


pub use error::{CircuitStmtError, DkgError, RevealError, SealError, SigmaError, VpdError};


// Parameter pins — the codegen seam (names per specs/params.toml; erratum 57).
const _: () = assert!(nerv_core::params::SEAL_RING_DEGREE == 256);
const _: () = assert!(nerv_core::params::SEAL_Q == 4_293_918_721);
const _: () = assert!(nerv_core::params::SEAL_Q_BITS == 32);
const _: () = assert!(nerv_core::params::SEAL_MODULE_RANK == 8);
const _: () = assert!(nerv_core::params::SEAL_PLAINTEXT_RINGS == 2);
const _: () = assert!(nerv_core::params::SEAL_DIGIT_BITS == 8);
const _: () = assert!(nerv_core::params::SEAL_DIGITS_PER_COORDINATE == 8);
const _: () = assert!(nerv_core::params::SEAL_DIGIT_SLOTS == 512);
const _: () = assert!(nerv_core::params::SEAL_DIGITS_USED == 512);
const _: () = assert!(nerv_core::params::SEAL_NOISE_ETA == 2);
const _: () = assert!(nerv_core::params::SEAL_NOISE_IN_CIRCUIT_BOUND == 3);
const _: () = assert!(nerv_core::params::SEAL_DKG_NOISE_ETA == 1);
const _: () = assert!(nerv_core::params::SEAL_CHUNK_MIN == nerv_core::params::SEAL_CHUNK_MAX);
const _: () = assert!(nerv_core::params::SEAL_COMMITTEE_N == 10);
const _: () = assert!(nerv_core::params::SEAL_THRESHOLD_T == 7);
const _: () = assert!(nerv_core::params::SEAL_SIGMA_CHALLENGE_WEIGHT == 32);
const _: () = assert!(nerv_core::params::SEAL_SIGMA_UNIFORM_LOG2 == 22);
const _: () = assert!(nerv_core::params::SEAL_SIGMA_ATTEMPT_CAP == 1024);

