//! # nerv-core — the shared vocabulary
//!
//! * `constants`     — all domain-separation strings in one auditable file.
//! * `hash`          — [WRAP blake3]: domain-separated Hash256 / XOF / ToField.
//! * `codec`         — canonical length-prefixed-LE wire format, total order.
//! * `types`         — ShardId prefix trie + ShardSet topology (WP §8.2),
//!                     Height / Interval / Epoch, TxId, LegIndex, FeeSats,
//!                     LegKey canonical (txid, leg_index) order (WP §4.3).
//! * `fixed_point`   — Q15 (WP §7.2 W format), round-half-even, 128-bit
//!                     intermediates, mod-2^64 delta coordinates.
//! * `params`        — GENERATED from specs/params.toml (build.rs).
//! * `error`         — core error taxonomy.
//!
//! Runtime dependency floor: blake3 + thiserror (DSR-2). No floats (P5).

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
#![deny(clippy::float_arithmetic, clippy::float_cmp)]

pub mod codec;
pub mod constants;
pub mod error;
pub mod field;
pub mod fixed_point;
pub mod hash;
pub mod params {
    include!(concat!(env!("OUT_DIR"), "/params.rs"));
}
pub mod types;

#[cfg(test)]
mod params_tests;
#[cfg(test)]
mod testutil;

pub use codec::{canonical_cmp, Decode, Encode, Encoded, Reader};
pub use constants::Domain;
pub use error::{CodecError, Error, FixedPointError, TypeError};
pub use field::{Goldilocks, GOLDILOCKS_PRIME};
pub use fixed_point::{round_half_even, round_half_even_pow2, wrap_mod_2_64, Mac15, Q15};
pub use hash::{hash_to_goldilocks, Hash256, Xof};
pub use types::{
    kappa, Epoch, FeeSats, Height, Interval, LegIndex, LegKey, ShardId, ShardSet, TxId,
    INTERVALS_PER_EPOCH, MAX_SHARD_BITS,
};

