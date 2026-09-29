//! # nerv-crypto — the only door to raw cryptographic primitives (DSR-3)
//!
//! ML-DSA-65 (FIPS 204), ML-KEM-768 (FIPS 203), ChaCha20-Poly1305, BLAKE3-KDF
//! / HKDF-SHA512 / BIP-39 seed derivation, quorum-certificate assembly and
//! hash compression (WP §4.6), beacon-seeded hash sortition (DSR-5). All
//! external primitive calls live in the private `provider` module; the PQ
//! audit is that file plus the dependency list.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod aead;
pub mod error;
pub mod kdf;
pub mod mldsa;
pub mod mlkem;
pub mod sigaggr;
pub mod sortition;

mod provider;

#[cfg(test)]
mod testutil;

pub use error::{CryptoError, QcError};
