//! AI-Native Consensus (Lightweight).
//!
//! Validators use the NWO Perceptron to predict the next embedding hash
//! via a dot product (microseconds). If ≥67% of active stake agrees,
//! instant probabilistic finality in <600ms. Disputes resolved via
//! Monte-Carlo simulation.
//!
//! Submodules:
//! - `predictor` — NWO Perceptron for sub-μs hash prediction
//! - `voting` — Weighted quorum (stake × reputation)
//! - `dispute` — Monte-Carlo resolution (kept for safety, rarely triggered)
//!
//!
//!
//! NERV v2.0 achieves sub-second finality via an **optimistic neural
//! prediction layer**: validators use the NWO Perceptron to predict
//! the next embedding hash after a batch. Agreement on the prediction
//! grants instant probabilistic finality; rare disagreements trigger
//! a Monte-Carlo dispute resolved in secure memory.
//!
//! # Fast Path (99.99% of blocks)
//!
//! ```text
//! 1. After homomorphic update, each validator computes:
//!    h_pred = BLAKE3(e_t + W·Δ_B + b)     ← dot product, <1μs
//!
//! 2. Validator broadcasts NeuralVote:
//!    { h_pred, partial BLS sig, reputation, weight_commitment }
//!
//! 3. If ≥67% of (stake × reputation) agrees on h_pred
//!    within 800ms → INSTANT probabilistic finality (~600ms median)
//!
//! 4. Full BLS threshold signature aggregated and committed
//! ```
//!
//! # Slow Path (<0.01% of blocks)
//!
//! ```text
//! 1. Any validator posts bonded challenge (0.5–1% stake)
//! 2. 32 VRF-chosen validators run 10,000 parallel Monte-Carlo sims
//! 3. Each sim: sample random batch subset → re-apply deltas → vote
//! 4. Resolution <650ms; winning root finalised
//! 5. Losing side slashed 0.5–5%; challenger bond returned or forfeited
//! ```

pub mod predictor;
pub mod voting;
pub mod dispute;

// Re-export primary types
pub use predictor::{EmbeddingPredictor, Prediction, NeuralVote};
pub use voting::{
    VoteMessage, VoteAggregator, QuorumResult, QuorumStatus,
    BLS_THRESHOLD_SIG_SIZE,
};
pub use dispute::{
    DisputeChallenge, DisputeState, DisputeResolution, MonteCarloResult,
    DisputeManager,
};

