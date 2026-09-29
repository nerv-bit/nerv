//! # nerv-wallet — client-side everything (WP App A)
//!
//! * `keys`      — the diversified address set; shard coverage.
//! * `scan`       — home-shard note streams; trial-decryption.
//! * `construct`  — transaction assembly: note selection, outputs,
//!   legs, deltas, sealing, the shell.
//! * `prove`      — the prover driver: witness generation and STARK
//!   proving (§5.4's background job).
//! * (part 3) `send`, `complete`, `claim`, `rehome`, `witness_store`.

//! # nerv-wallet — client-side everything (WP App A)

#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic, clippy::float_cmp)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod claim;
pub mod complete;
pub mod construct;
pub mod keys;
pub mod prove;
pub mod rehome;
pub mod scan;
pub mod send;
pub mod witness_store;

#[cfg(test)]
mod testutil;

pub use claim::{ClaimCredentials, ClaimRequest};
pub use complete::{check_finalized, build_witness, CompletionPackage};
pub use construct::{
    construct_payment, ConstructedTx, ConstructError, PaymentSpec, WalletEntropy,
};
pub use keys::{AddressSet, WalletAddress, DEFAULT_SHARD_COVERAGE, MAX_GENERATION};
pub use prove::{prove, ProvedTx, ProveError};
pub use rehome::{rehome_notes, RehomeResult};
pub use scan::{NotePosition, ScannedNote, WalletNoteSet};
pub use send::{
    bucket_fee, run_send_pipeline, send_transaction, PipelineError, PipelineStage, SendConfig,
    SendError, SendPipeline, SendReport,
};
pub use witness_store::WitnessStore;
