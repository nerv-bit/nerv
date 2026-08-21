//! NWO Perceptron for Sub-Microsecond Embedding Hash Prediction.
//!
//! The prediction operation is the performance-critical path of
//! NERV's consensus. Because the NWO Perceptron is a single linear
//! layer (`f(x) = W·x + b`), predicting the next embedding root
//! is just a dot product and a hash:
//!
//! ```text
//! h_pred = BLAKE3(e_t + W·Δ_B + b)
//! ```
//!
//! This completes in <1μs on modern hardware, enabling sub-second
//! finality when ≥67% of validators agree.

use crate::{
    EMBEDDING_DIM, NervError, NervResult,
    EmbeddingRoot, BlockHeight, StakeAmount, ReputationScore, ValidatorId,
    TARGET_PREDICTION_US, TARGET_FINALITY_MS,
};
use crate::embedding::fixed_point::{FixedPoint64, EmbeddingVector};
use crate::embedding::perceptron::NwoPerceptron;
use crate::embedding::homomorphism::EmbeddingDelta;
use crate::utils::blake3_hash;
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ─── Prediction ──────────────────────────────────────────────────────────

/// A predicted next embedding root with metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prediction {
    /// The predicted embedding root hash.
    pub predicted_root: EmbeddingRoot,

    /// The block height this prediction is for.
    pub target_height: BlockHeight,

    /// The shard this prediction applies to.
    pub shard_id: u64,

    /// BLAKE3 hash of the weight matrix used for this prediction.
    pub weight_commitment: [u8; 32],

    /// Time taken to compute this prediction (microseconds).
    pub computation_time_us: u64,
}

impl Prediction {
    /// Create a new prediction.
    pub fn new(
        predicted_root: EmbeddingRoot,
        target_height: BlockHeight,
        shard_id: u64,
        weight_commitment: [u8; 32],
        computation_time_us: u64,
    ) -> Self {
        Self {
            predicted_root,
            target_height,
            shard_id,
            weight_commitment,
            computation_time_us,
        }
    }
}

// ─── Neural Vote ─────────────────────────────────────────────────────────

/// A validator's vote on the next embedding root.
///
/// This is the message broadcast during the fast path of consensus.
/// Votes are gossiped aggressively (only ~160 bytes per message).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeuralVote {
    /// The validator's identifier.
    pub validator_id: ValidatorId,

    /// The predicted embedding root hash.
    pub predicted_root: EmbeddingRoot,

    /// The block height being voted on.
    pub target_height: BlockHeight,

    /// The shard being voted on.
    pub shard_id: u64,

    /// The validator's stake (for weight computation).
    pub stake: StakeAmount,

    /// The validator's reputation score.
    pub reputation: ReputationScore,

    /// Partial BLS12-381 signature share (48 bytes compressed).
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub bls_sig_share: Vec<u8>,

    /// BLAKE3 hash of the weight matrix (for compatibility check).
    pub weight_commitment: [u8; 32],

    /// Monotonic timestamp (for freshness / replay protection).
    pub timestamp_ms: u64,
}

impl NeuralVote {
    /// Create a new neural vote.
    pub fn new(
        validator_id: ValidatorId,
        prediction: &Prediction,
        stake: StakeAmount,
        reputation: ReputationScore,
        bls_sig_share: Vec<u8>,
    ) -> Self {
        Self {
            validator_id,
            predicted_root: prediction.predicted_root,
            target_height: prediction.target_height,
            shard_id: prediction.shard_id,
            stake,
            reputation,
            bls_sig_share,
            weight_commitment: prediction.weight_commitment,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Compute the voting weight: stake × reputation.
    pub fn voting_weight(&self) -> crate::VotingWeight {
        crate::compute_voting_weight(self.stake, self.reputation)
    }

    /// Get the total message size in bytes (for gossip optimization).
    pub fn message_size(&self) -> usize {
        // Approximate: 32 (id) + 32 (root) + 8 (height) + 8 (shard)
        // + 8 (stake) + 4 (reputation) + 48 (sig) + 32 (commitment) + 8 (timestamp)
        180
    }

    /// Check if this vote is compatible with another (same root & height).
    pub fn is_compatible_with(&self, other: &NeuralVote) -> bool {
        self.predicted_root == other.predicted_root
            && self.target_height == other.target_height
            && self.shard_id == other.shard_id
            && self.weight_commitment == other.weight_commitment
    }
}

// ─── Embedding Predictor ─────────────────────────────────────────────────

/// The AI-native embedding root predictor.
///
/// Wraps the NWO Perceptron and adds consensus-specific logic:
/// timing, weight commitment tracking, and vote construction.
#[derive(Debug, Clone)]
pub struct EmbeddingPredictor {
    /// The NWO Perceptron used for prediction.
    pub perceptron: NwoPerceptron,

    /// BLAKE3 hash of the current weight matrix.
    pub weight_commitment: [u8; 32],

    /// Statistics.
    pub stats: PredictorStats,
}

/// Prediction statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictorStats {
    /// Total predictions made.
    pub predictions: u64,

    /// Total prediction time (microseconds).
    pub total_time_us: u64,

    /// Number of predictions under the target latency.
    pub under_target: u64,

    /// Maximum prediction time (microseconds).
    pub max_time_us: u64,
}

impl EmbeddingPredictor {
    /// Create a new predictor from an existing Perceptron.
    pub fn new(perceptron: NwoPerceptron) -> Self {
        // Compute weight commitment
        let weight_commitment = blake3_hash(&perceptron.weights.to_bytes());
        Self {
            perceptron,
            weight_commitment,
            stats: PredictorStats::default(),
        }
    }

    /// Predict the next embedding root after applying a batch delta.
    ///
    /// This is the critical path:
    /// ```text
    /// h_pred = BLAKE3(e_t + δ_B)
    /// ```
    ///
    /// Where `δ_B = W · Δ_S + b` is computed via the Perceptron.
    /// Total time: <1μs for dot product + hash.
    pub fn predict(
        &mut self,
        current_embedding: &EmbeddingVector,
        batch_delta: &EmbeddingDelta,
        target_height: BlockHeight,
        shard_id: u64,
    ) -> Prediction {
        let start = Instant::now();

        // Compute the predicted next root
        let predicted_root = self.perceptron.predict_next_root(
            current_embedding,
            batch_delta,
        );

        let computation_time_us = start.elapsed().as_micros() as u64;

        // Update statistics
        self.stats.predictions += 1;
        self.stats.total_time_us += computation_time_us;
        if computation_time_us <= TARGET_PREDICTION_US as u64 {
            self.stats.under_target += 1;
        }
        if computation_time_us > self.stats.max_time_us {
            self.stats.max_time_us = computation_time_us;
        }

        Prediction::new(
            predicted_root,
            target_height,
            shard_id,
            self.weight_commitment,
            computation_time_us,
        )
    }

    /// Predict using a batch of transaction features (full forward pass).
    pub fn predict_from_features(
        &mut self,
        current_embedding: &EmbeddingVector,
        batch_features: &[crate::embedding::TransactionFeatures],
        target_height: BlockHeight,
        shard_id: u64,
    ) -> NervResult<Prediction> {
        let start = Instant::now();

        // Compute the batch delta via the Perceptron
        let batch_delta = self.perceptron.compute_batch_delta(batch_features)?;
        let delta = EmbeddingDelta::from_vector(batch_delta);

        // Apply and hash
        let predicted_root = self.perceptron.predict_next_root(
            current_embedding,
            &delta,
        };

        let computation_time_us = start.elapsed().as_micros() as u64;

        self.stats.predictions += 1;
        self.stats.total_time_us += computation_time_us;
        if computation_time_us <= TARGET_PREDICTION_US as u64 {
            self.stats.under_target += 1;
        }
        if computation_time_us > self.stats.max_time_us {
            self.stats.max_time_us = computation_time_us;
        }

        Ok(Prediction::new(
            predicted_root,
            target_height,
            shard_id,
            self.weight_commitment,
            computation_time_us,
        ))
   *}

    /// Create a neural vote from a prediction.
    pub fn create_vote(
        &self,
        validator_id: ValidatorId,
        prediction: &Prediction,
        stake: StakeAmount,
        reputation: ReputationScore,
        bls_sig_share: Vec<u8>,
    ) -> NeuralVote {
        NeuralVote::new(validator_id, prediction, stake, reputation, bls_sig_share)
    }

    /// Get the average prediction time in microseconds.
    pub fn avg_prediction_time_us(&self) -> f64 {
        if self.stats.predictions == 0 {
            0.0
        } else {
            self.stats.total_time_us as f64 / self.stats.predictions as f64
        }
    }

    /// Get the percentage of predictions under target latency.
    pub fn under_target_pct(&self) -> f64 {
        if self.stats.predictions == 0 {
            100.0
        } else {
            (self.stats.under_target as f64 / self.stats.predictions as f64) * 100.0
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::{NwoWeights, DEFAULT_INPUT_DIM};

    fn make_predictor() -> EmbeddingPredictor {
        let weights = NwoWeights::new_constant(DEFAULT_INPUT_DIM, 0.01);
        let perceptron = NwoPerceptron::from_weights(weights);
        EmbeddingPredictor::new(perceptron)
    }

    #[test]
    fn test_prediction_creation() {
        let root = EmbeddingRoot::from_bytes([1u8; 32]);
        let pred = Prediction::new(root, BlockHeight::from(100), 0, [0u8; 32], 50);
        assert_eq!(pred.target_height, BlockHeight::from(100));
        assert_eq!(pred.computation_time_us, 50);
    }

    #[test]
    fn test_predictor_basic() {
        let mut predictor = make_predictor();
        let embedding = EmbeddingVector::ZERO;
        let delta = EmbeddingDelta::splat(FixedPoint64::from_int(1));

        let pred = predictor.predict(&embedding, &delta, BlockHeight::from(1), 0);
        assert_eq!(pred.target_height, BlockHeight::from(1));
        assert!(pred.computation_time_us < 10_000); // Should be fast
        assert_eq!(predictor.stats.predictions, 1);
    }

    #[test]
    fn test_predictor_deterministic() {
        let mut predictor = make_predictor();
        let embedding = EmbeddingVector::splat(FixedPoint64::from_int(10));
        let delta = EmbeddingDelta::splat(FixedPoint64::from_int(5));

        let pred1 = predictor.predict(&embedding, &delta, BlockHeight::from(1), 0);
        let pred2 = predictor.predict(&embedding, &delta, BlockHeight::from(1), 0);

        // Same inputs → same prediction
        assert_eq!(pred1.predicted_root, pred2.predicted_root);
    }

    #[test]
    fn test_neural_vote_creation() {
        let mut predictor = make_predictor();
        let embedding = EmbeddingVector::ZERO;
        let delta = EmbeddingDelta::ZERO;
        let pred = predictor.predict(&embedding, &delta, BlockHeight::from(1), 0);

        let vote = NeuralVote::new(
            ValidatorId::from_bytes([1u8; 32]),
            &pred,
            StakeAmount(1000),
            ReputationScore::PERFECT,
            vec![0u8; 48],
        );

        assert_eq!(vote.predicted_root, pred.predicted_root);
        assert_eq!(vote.target_height, BlockHeight::from(1));
        assert_eq!(vote.message_size(), 180);
    }

    #[test]
    fn test_voting_weight() {
        let mut predictor = make_predictor();
        let pred = predictor.predict(
            &EmbeddingVector::ZERO,
            &EmbeddingDelta::ZERO,
            BlockHeight::from(1), 0,
        );

        let vote = NeuralVote::new(
            ValidatorId::from_bytes([1u8; 32]),
            &pred,
            StakeAmount(1000 * crate::ONE_NERV),
            ReputationScore::from_f64(0.8),
            vec![0u8; 48],
        );

        let weight = vote.voting_weight();
        // weight = stake * reputation = 1000 * 10^9 * 0.8 = 800 * 10^9
        assert!(weight.0 > 0);
    }

    #[test]
    fn test_vote_compatibility() {
        let mut predictor = make_predictor();
        let pred = predictor.predict(
            &EmbeddingVector::ZERO,
            &EmbeddingDelta::ZERO,
            BlockHeight::from(1), 0,
        );

        let vote1 = NeuralVote::new(
            ValidatorId::from_bytes([1u8; 32]),
            &pred,
            StakeAmount(1000),
            ReputationScore::PERFECT,
            vec![0u8; 48],
        );

        let vote2 = NeuralVote::new(
            ValidatorId::from_bytes([2u8; 32]),
            &pred,
            StakeAmount(2000),
            ReputationScore::INITIAL,
            vec![0u8; 48],
        );

        // Same prediction → compatible
        assert!(vote1.is_compatible_with(&vote2));
    }

    #[test]
    fn test_predictor_stats() {
        let mut predictor = make_predictor();
        let embedding = EmbeddingVector::ZERO;
        let delta = EmbeddingDelta::ZERO;

        for _ in 0..100 {
            predictor.predict(&embedding, &delta, BlockHeight::from(1), 0);
        }

        assert_eq!(predictor.stats.predictions, 100);
        assert!(predictor.avg_prediction_time_us() >= 0.0);
        assert!(predictor.under_target_pct() > 0.0);
    }
}
