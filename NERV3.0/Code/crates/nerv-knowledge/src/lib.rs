//! # nerv-knowledge — the advisory overlay (WP §§7.5, 10)
//!
//! * `embedding`  — e_t: the per-shard derived accumulator, the miss
//!                  log, and the D_t composition.
//! * `forecaster` — the linear AR(1024): state, deterministic
//!                  inference, the reference initialization.
//! * `adam`       — the deterministic integer optimizer.
//! * `huber`      — the componentwise scoring and the scale EMA.
//! * `block_loop` — the §10.3 per-block loop: commit → reveal → score
//!                  → update → record.
//! * `replay`     — the D_t re-derivation and the advisory fault.
//! * `challenger` — the §10.4 market: registration, sealed commits,
//!                  the 2,016-block promotion gate.
//! * `anomaly`    — the §10.5 tail-residual advisory flags.
//!
//! THE FIREWALL (DSR-1): this crate depends only on nerv-core and
//! nerv-codec, and nothing depends on it. `tests/firewall.rs` is
//! Axiom 3's harness — the delete test that makes the separation a
//! build rule, not a review guideline. Deleting this crate leaves the
//! authority stack compiling and validating chains.

#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic, clippy::float_cmp)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod adam;
pub mod anomaly;
pub mod block_loop;
pub mod challenger;
pub mod embedding;
pub mod forecaster;
pub mod huber;
pub mod replay;

#[cfg(test)]
mod testutil;

pub use adam::{
    isqrt, step, AdamError, ALPHA_B, ALPHA_W, GRAD_CLIP_RAW, GRAD_CLIP_TRUE, K1, K2,
    RATIO_CLIP_RAW,
};
pub use anomaly::{AnomalyFlags, AnomalyTracker, CASCADE_DIMS, HISTORY, TAIL_MULT};
pub use block_loop::{BlockEvent, BlockRecord, Commitment, KnowledgeState, LoopError};
pub use challenger::{
    EligibilityFailure, Challenger, ChallengerError, ChallengerMarket, GateOutcome, SkillRecord,
    MARGIN_PERMILLE, MIN_COVERAGE_PERMILLE, WINDOW_BLOCKS,
};
pub use embedding::{derived_state_root, EmbedError, Embedding, MissedReveal};
pub use forecaster::{
    predict, Forecaster, ForecasterError, Moments, Observation, Weights, Window, AR_ORDER, DIMS,
    LEARNABLE, MOMENTS_CANONICAL_LEN, PER_CHANNEL, SCALE_DEFAULT, WEIGHTS_CANONICAL_LEN,
};
pub use huber::{
    blame, loss_per_dim, residual, scale_ema, total_loss, S_MAX, S_MIN, SCALE_EMA_SHIFT,
};
pub use replay::{replay, verify_d_t_chain, ReplayError, ReplayFault};

const _: () =
    assert!(nerv_core::params::OVERLAY_FORECASTER_AR_WINDOW as usize == forecaster::AR_ORDER);
const _: () = assert!(nerv_core::params::OVERLAY_CHALLENGER_WINDOW_BLOCKS == challenger::WINDOW_BLOCKS);
