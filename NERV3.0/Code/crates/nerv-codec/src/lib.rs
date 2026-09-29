//! # nerv-codec — the identity layer (WP §7.2, §7.6)
//!
//! * `features`   — per-leg feature-vector construction and admissibility.
//! * `codec_w`    — the frozen linear codec `W`, deltas, versioning.
//! * `weight_gen` — beacon-XOF verifiable expansion and the per-epoch
//!                  machine certification (norms, rank checks, certificate).
//!
//! Canonical, frozen, governance-versioned; depends only on nerv-core.
//! Nothing here learns, adapts, or consults any model, and no floating-point
//! operation exists anywhere in this crate (P5, DSR-11).

#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic, clippy::float_cmp)]

pub mod codec_w;
pub mod features;
pub mod weight_gen;

