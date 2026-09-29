//! # nerv-witness — inclusion witnesses and light anchors (WP §11)
//!
//! * `witness` — the ~600 B client-held inclusion witness: the standard
//!   form (a shard-tracking verifier) and the cold form (a verifier that
//!   tracks nothing). Portfolio composition.
//! * `anchor`   — the light-client anchor: the epoch-attestation chain
//!   and the current epoch's intervals. The 𝔾 root; extension; the
//!   walk back to genesis.
//! * `regen`    — archival regeneration: any finalized witness from
//!   the DA-published block, forever (§11.6).
//!
//! The witness proves two things: the header is finalized (the 𝔾 path,
//! a cryptographic fact from the anchor) and the leg is in the block
//! (the leg-tree path, against the block's root). The root's binding to
//! the header requires the DA data — the full-node path (§11.4).


#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic, clippy::float_cmp)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]


pub mod anchor;
pub mod error;
pub mod regen;
pub mod witness;


pub use anchor::{AnchorError, LightAnchor};
pub use error::WitnessError;
pub use regen::regenerate;
pub use witness::{
    verify, verify_portfolio, GPath, InclusionWitness, LegWitness, PortfolioWitness,
};
