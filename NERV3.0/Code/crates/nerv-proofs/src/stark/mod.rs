//! The native nerv-stark engine (WP §5.3, §5.7; Option B, register 57):
//! transparent STARK proving over Goldilocks and its degree-2 extension,
//! BLAKE3 Fiat–Shamir (`air::fs::FsTranscript`), BLAKE3-Merkle FRI.

pub mod collector;
pub mod compose;
pub mod domain;
pub mod ext_field;
pub mod fft;
pub mod fri;
pub mod merkle;
pub mod proof;
pub mod prover;
pub mod transcript_common;
pub mod verifier;
