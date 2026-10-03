//! # nerv-proofs — the proof fabric (WP §5)
//!
//! * `air::fs` — the protocol-level Fiat–Shamir layer: statement-11's
//!   binding challenge, the §5.1 public inputs, and the deterministic
//!   BLAKE3 transcript every proof composes through.
//! * `air::custody_air` + `air::chips` — statements 1–5 plus the D.3
//!   revert and burn wiring; the seal chip (statement 10's in-circuit
//!   NTT) and the encoder chip (statement 7) live in `chips`.
//! * `security` — D.2's FRI-security gate over the native conjectured
//!   model (Option B).
//! * `stark` (chunk 12.5) — the native prover/verifier engine: extension
//!   field, two-adic domains, FFT, BLAKE3 Merkle FRI, composition, and
//!   the prove/verify drivers over `prove`-shaped entrypoints.
//! * `air::delta_air` (statements 6–9δ) and `air::tx_air` (the composed
//!   whole-transaction AIR, statements 1–11 chained) are delivered; 
//!
//! DSR-7: every in-circuit relation lands with, and is differentially
//! tested against, its native twin (nerv-custody's Poseidon2/NCT for the
//! Merkle chip; nerv-seal's `circuit_stmt` for the seal chip;
//! nerv-codec's encoder for the delta module). No floating point (P5).

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic, clippy::float_cmp)]

pub mod air;
pub mod error;
pub mod witness_gen;
pub mod fold;
pub mod security;
pub mod stark;

#[cfg(test)]
mod testutil;

pub use air::{
    bind, bind_transaction, canonical_nullifiers, shell_digest, Air, AirBuilder, AirExpr,
    ConstraintFailure, FsTranscript, NativeEval, TxPublicInputs, MerkleChip, gen_merkle_trace,
    CustodyAir, CustodyTrace, CustodyWitness, InputWitness, OutputWitness, RevertWitness,
    BurnWitness, bind_witness_to_shell, gen_custody_trace, gen_cons_prep, gen_cons_trace,
    ConservationChip, ConstraintSet, SymBuilder, SymExpr, MeasureBuilder, measure,
    Blake3Compression, compress_native, hash_native, SealChip, gen_seal_trace, SealLegInput,
    SealTrace,
};
// Re-export the custody_air module path so callers can write
// `nerv_proofs::custody_air::*` (used by `nerv-wallet`'s construction
// path; previously only the deep types were re-exported, not the module
// path).
pub use air::custody_air;

pub use air::delta_air::{gen_delta_trace, DeltaAir, DeltaTrace, LegShape};


pub use air::tx_air::{
    build_tx_air, check_ct_binding, gen_tx_prep, gen_tx_trace, prove_transaction, serialize_uv,
    verify_transaction, TransactionProof, TxAir, TxError, TxTrace,
};


pub use fold::{BundleTxids, DedupError, DedupReport, IntervalLedger, IntervalSet};


pub use error::{FsError, WitnessError, WitnessGenError};

pub use security::{
    AirShape, FriShape, GateReport, Profile, conjectured, gate, search_min_fri, validate,
    D2_ALGEBRAIC_FLOOR_BITS, PQ_COLLISION_BITS,
};

pub use witness_gen::{DeltaLegWitness, DeltaWitness, SealLegWitness, TransactionWitness};

pub use stark::compose::{ComposedProof, ComposeError, Plan};

pub use stark::prover::{ProveError, Prover, Proved};

pub use stark::verifier::{Verifier, VerifyError};


