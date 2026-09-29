//! # nerv-state — deterministic execution and the composite state commitment (WP §4)
//!
//! * `ttau`       — the T_τ inclusion tree: rule 1's membership object.
//! * `anchor`     — the 64-header NCT-root freshness ring (rule 2).
//! * `commitment` — C_t (§4.2) and C₀ (§12.1 as amended by E-008).
//! * `header`     — the shard header (§4.3): field set, `nerv.hdr`
//!                  hashing, canonical codec.
//! * (part 2) `block`, `executor` — the block body, rules 1–7, BeaconView.
//! * (part 3) `fraud`, `store`    — public-hash fraud evidence, the store seam.
//!
//! DSR-4: D_t travels through this crate as 32 opaque bytes; the previous
//! block's revealed Δ_B as 512 opaque bytes. No floating point (P5).

#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic, clippy::float_cmp)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod anchor;
pub mod block;
pub mod commitment;
pub mod error;
pub mod executor;
pub mod fraud;
pub mod header;
pub mod store;
pub mod ttau;


#[cfg(test)]
mod testutil;

pub use anchor::AnchorRing;
pub use commitment::{genesis_commitment, state_commitment};
pub use block::{
   canonical_txid, ct_sum, verify_leg_witness, BlockLegTree, ClaimRecord, LegEvidence,
   LegTreeWitness, ResolvedLeg, ReversionRecord, SettledLeg, ShardBlock, TransitEvidence,
   LEG_TREE_CAPACITY, LEG_TREE_DEPTH, MAX_LEGS,
};
pub use error::{BlockError, ExecutorError, FraudError, RebuildError, StateError, StoreError};
pub use executor::{
   apply_block, propose, Applied, BeaconView, BlockBody, ChainSource, ComputedHeader,
   HeaderInputs, ShardState,
};

pub use fraud::{BeaconFacts, FraudCondition, FraudProof, ReexecutionFraud};
pub use store::{
   load_shard, rebuild_state, ArchiveChain, MemStore, NodeStore, ShardArchive, StoreArchive,
   SCHEMA_VERSION,
};

pub use header::{RegistryRef, ShardHeader, REVEAL_BYTES};
pub use ttau::{
    empty_root, verify_tau_witness, RegistryWitness, TauTree, TauWitness, LEAF_CAPACITY,
    MAX_DEPTH as TAU_MAX_DEPTH,
};

const _: () = assert!(nerv_core::params::CUSTODY_ANCHOR_WINDOW_HEADERS == 64);
