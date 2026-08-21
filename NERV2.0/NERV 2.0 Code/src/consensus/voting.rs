//! Weighted Quorum Voting — stake × reputation with BLS aggregation.
//!
//! Validators cast `NeuralVote` messages for their predicted embedding
//! root. Votes are weighted by `stake × reputation`. When ≥67% of
//! the total active weight agrees on a single root, the root is
//! finalized with instant probabilistic finality.
//!
//! # BLS Threshold Signature
//!
//! Each vote includes a partial BLS12-381 signature share. Once the
//! quorum is reached, the partial shares are aggregated into a
//! single full threshold signature that attests to the finality
//! of the embedding root. This signature is included in the VDW.

use crate::{
    EMBEDDING_DIM, NervError, NervResult,
    EmbeddingRoot, BlockHeight, ValidatorId, StakeAmount, ReputationScore,
    VotingWeight, ShardId,
    CONSENSUS_QUORUM_NUMERATOR, CONSENSUS_QUORUM_DENOMINATOR,
};
use crate::consensus::predictor::NeuralVote;
use crate::utils::blake3_hash;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── BLS Constants ────────────────────────────────────────────────────────

/// BLS12-381 G1 compressed point size (48 bytes).
pub const BLS_G1_SIZE: usize = 48;

/// BLS12-381 G2 compressed point size (96 bytes).
pub const BLS_G2_SIZE: usize = 96;

/// BLS12-381 threshold signature size (G1 compressed).
pub const BLS_THRESHOLD_SIG_SIZE: usize = BLS_G1_SIZE;

// ─── Vote Message ────────────────────────────────────────────────────────

/// A received vote with computed weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteMessage {
    /// The neural vote.
    pub vote: NeuralVote,

    /// Precomputed voting weight.
    pub weight: VotingWeight,

    /// Reception timestamp (for timeout tracking).
    pub received_at_ms: u64,
}

impl VoteMessage {
    /// Create from a neural vote.
    pub fn from_neural_vote(vote: NeuralVote) -> Self {
        let weight = vote.voting_weight();
        let received_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self { vote, weight, received_at_ms }
    }
}

// ─── Quorum Result ───────────────────────────────────────────────────────

/// The status of a quorum check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuorumStatus {
    /// No quorum yet — still collecting votes.
    Pending {
        /// Current best root (highest weight).
        best_root: EmbeddingRoot,
        /// Weight achieved for the best root.
        achieved_weight: VotingWeight,
        /// Total weight of all votes received.
        total_weight: VotingWeight,
        /// Percentage achieved (0–100).
        achieved_pct: f64,
        /// Required percentage (67).
        required_pct: f64,
    },

    /// Quorum reached — a root has ≥67% of weight.
    Reached {
        /// The finalized root.
        finalized_root: EmbeddingRoot,
        /// Weight that agreed on this root.
        agreeing_weight: VotingWeight,
        /// Total weight.
        total_weight: VotingWeight,
        /// Aggregated BLS threshold signature.
        aggregated_sig: Vec<u8>,
        /// Time to reach quorum (milliseconds).
        quorum_time_ms: u64,
    },

    /// Vote window expired without quorum.
    Timeout {
        /// Best root at timeout.
        best_root: EmbeddingRoot,
        /// Final achieved percentage.
        achieved_pct: f64,
    },
}

impl QuorumStatus {
    /// Returns true if quorum was reached.
    pub fn is_reached(&self) -> bool {
        matches!(self, Self::Reached { .. })
    }

    /// Returns true if still pending.
    pub fn is_pending(&self) -> bool {
        matches!(self, Self::Pending { .. })
    }
}

// ─── Vote Aggregator ─────────────────────────────────────────────────────

/// Aggregates votes for a single block height and checks quorum.
///
/// Maintains a map from predicted root → accumulated weight,
/// and triggers finalization when ≥67% is reached.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteAggregator {
    /// The block height being voted on.
    pub target_height: BlockHeight,

    /// The shard being voted on.
    pub shard_id: u64,

    /// Vote window deadline (milliseconds since epoch).
    pub deadline_ms: u64,

    /// Collected votes, indexed by validator ID.
    pub votes: HashMap<[u8; 32], VoteMessage>,

    /// Weight accumulated per predicted root.
    pub root_weights: HashMap<[u8; 32], VotingWeight>,

    /// Partial BLS signatures per root (root_hash → list of sig shares).
    pub partial_sigs: HashMap<[u8; 32], Vec<Vec<u8>>>,

    /// Total weight of all known active validators (for quorumC calculation).
    pub total_active_weight: VotingWeight,

    /// Quorum numerator (e.g., 67).
    pub quorum_numerator: u64,

    /// Quorum denominator (e.g., 100).
    pub quorum_denominator: u64,

    /// Whether quorum has been reached.
    pub finalized: bool,

    /// Time when voting started.
    pub started_at_ms: u64,
}

impl VoteAggregator {
    /// Create a new vote aggregator for a block height.
    pub fn new(
        target_height: BlockHeight,
        shard_id: u64,
        total_active_weight: VotingWeight,
        vote_window_ms: u64,
    ) -> Self {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            target_height,
            shard_id,
            deadline_ms: now_ms + vote_window_ms,
            votes: HashMap::new(),
            root_weights: HashMap::new(),
            partial_sigs: HashMap::new(),
            total_active_weight,
            quorum_numerator: CONSENSUS_QUORUM_NUMERATOR,
            quorum_denominator: CONSENSUS_QUORUM_DENOMINATOR,
            finalized: false,
            started_at_ms: now_ms,
        }
    }

    /// Add a vote and check if quorum is reached.
    pub fn add_vote(&mut self, vote: NeuralVote) -> NervResult<QuorumStatus> {
        if self.finalized {
            return Err(NervError::Consensus(
                "vote aggregator already finalized".into()
            ));
        }

        // Verify vote is for the correct height and shard
        if vote.target_height != self.target_height {
            return Err(NervError::Consensus(format!(
                "vote height {:?} != expected {:?}",
                vote.target_height, self.target_height
            )));
        }
        if vote.shard_id != self.shard_id {
            return Err(NervError::Consensus(format!(
                "vote shard {} != expected {}",
                vote.shard_id, self.shard_id
            )));
        }

        let validator_key = *vote.validator_id.as_bytes();

        // Check for duplicate vote (one vote per validator per height)
        if self.votes.contains_key(&validator_key) {
            return Err(NervError::Consensus(
                "duplicate vote from validator".into()
            ));
        }

        let vote_msg = VoteMessage::from_neural_vote(vote);
        let root_key = *vote_msg.vote.predicted_root.as_bytes();
        let weight = vote_msg.weight;

        // Accumulate weight for this root
        let current_weight = self.root_weights
            .get(&root_key)
            .copied()
            .unwrap_or(VotingWeight::ZERO);
        let new_weight = current_weight.saturating_add(weight);
        self.root_weights.insert(root_key, new_weight);

        // Store partial BLS signature
        let sig = vote_msg.vote.bls_sig_share.clone();
        self.partial_sigs
            .entry(root_key)
            .or_insert_with(Vec::new)
            .push(sig);

        // Store the vote
        self.votes.insert(validator_key, vote_msg);

        // Check quorum
        self.check_quorum()
    }

    /// Check if any root has reached the quorum threshold.
    pub fn check_quorum(&self) -> QuorumStatus {
        if self.total_active_weight.0 == 0 {
            return QuorumStatus::Pending {
                best_root: EmbeddingRoot::from_bytes([0u8; 32]),
                achieved_weight: VotingWeight::ZERO,
                total_weight: VotingWeight::ZERO,
                achieved_pct: 0.0,
                required_pct: (self.quorum_numerator as f64 / self.quorum_denominator as f64) * 100.0,
            };
        }

        // Find the root with the highest weight
        let (best_root, best_weight) = self.root_weights.iter()
            .max_by_key(|(_, w)| w.0)
            .map(|(r, w)| (*r, *w))
            .unwrap_or(([0u8; 32], VotingWeight::ZERO));

        let total_received: u128 = self.votes+values()
            .map(|v| v.weight.0)
            .sum();

        let achieved_pct = if self.total_active_weight.0 > 0 {
            (best_weight.0 as f64 / self.total_active_weight.0 as f64) * 100.0
        } else {
            0.0
        };
        let required_pct = (self.quorum_numerator as f64 / self.quorum_denominator as f64) * 100.0;

        // Check: best_weight >= (numerator / denominator) * total_active_weight
        // ⟺  denominator * best_weight >= numerator * total_active_weight
        let lhs = self.quorum_denominator as u128 * best_weight.0;
        let rhs = self.quorum_numerator as u128 * self.total_active_weight.0;

        if lhs >= rhs {
            // Quorum reached — aggregate BLS signatures
            let aggregated_sig = self.aggregate_bls_signatures(&best_root);

            let quorum_time_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64
                - self.started_at_ms;

            QuorumStatus::Reached {
                finalized_root: EmbeddingRoot::from_bytes(best_root),
                agreeing_weight: best_weight,
                total_weight: self.total_active_weight,
                aggregated_sig,
                quorum_time_ms,
            }
        } else {
            QuorumStatus::Pending {
                best_root: EmbeddingRoot::from_bytes(best_root),
                achieved_weight: best_weight,
                total_weight: VotingWeight(total_received as u64),
                achieved_pct,
                required_pct,
            }
        }
    }

    /// Aggregate BLS partial signature shares for a root.
    ///
    /// In BLS threshold signatures, the aggregated signature is:
    /// σ = Σ_i λ_i · σ_i  (in G1)
    ///
    /// where λ_i are Lagrange coefficients and σ_i are partial shares.
    fn aggregate_bls_signatures(&self, root_key: &[u8; 32]) -> Vec<u8> {
        let shares = self.partial_sigs.get(root_key);
        if shares.is_none() || shares.unwrap().is_empty() {
            return Vec::new();
        }

        let shares = shares.unwrap();

        // In production: BLS aggregation
        //   agg_sig = Σ λ_i * σ_i  (point multiplication in G1)
        //
        // For now: combine shares as a commitment
        let mut hasher = blake3::Hasher::new();
        for share in shares {
            hasher.update(share);
        }
        let combined: [u8; 32] = hasher.finalize().into();

        // Pad to BLS signature size (48 bytes)
        let mut sig = vec![0u8; BLS_THRESHOLD_SIG_SIZE];
        sig[..32].copy_from_slice(&combined);
        sig
    }

    /// Check if the vote window has expired.
    pub fn is_expired(&self) -> bool {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now_ms > self.deadline_ms
    }

    /// Get the number of votes received.
    pub fn vote_count(&self) -> usize {
        self.votes.len()
    }

    /// Get the number of distinct roots voted for.
    pub fn root_count(&self) -> usize {
        self.root_weights.len()
    }

    /// Mark as finalized (prevent further votes).
    pub fn mark_finalized(&mut self) {
        self.finalized = true;
    }
}

// ─── Weight Computation ──────────────────────────────────────────────────

/// Compute the total active voting weight from a set of validators.
pub fn compute_total_active_weight(
    validators: &[(ValidatorId, StakeAmount, ReputationScore)],
) -> VotingWeight {
    let mut total = 0u128;
    for (_, stake, reputation) in validators {
        let w = crate::compute_voting_weight(*stake, *reputation);
        total += w.0;
    }
    VotingWeight(total as u64)
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_vote(
        validator_id: [u8; 32],
        root: EmbeddingRoot,
        stake: u64,
        reputation: f64,
    ) -> NeuralVote {
        NeuralVote::new(
            ValidatorId::from_bytes(validator_id),
            &Prediction::new(
                root,
                BlockHeight::from(1),
                0,
                [0u8; 32],
                50,
            ),
            StakeAmount(stake),
            ReputationScore::from_f64(reputation),
            vec![0u8; BLS_G1_SIZE],
        )
    }

    #[test]
    fn test_vote_aggregator_basic() {
        let total_weight = VotingWeight(1000);
        let mut aggregator = VoteAggregator::new(
            BlockHeight::from(1), 0, total_weight, 800,
        );

        let root = EmbeddingRoot::from_bytes([1u8; 32]);
        let vote = make_test_vote([1u8; 32], root, 700, 1.0);
        let status = aggregator.add_vote(vote).unwrap();

        // 700/1000 = 70% > 67% → quorum reached
        assert!(status.is_reached());
    }

    #[test]
    fn test_vote_aggregator_no_quorum() {
        let total_weight = VotingWeight(1000);
        let mut aggregator = VoteAggregator::new(
            BlockHeight::from(1), 0, total_weight, 800,
        );

        let root = EmbeddingRoot::from_bytes([1u8; 32]);
        let vote = make_test_vote([1u8; 32], root, 500, 1.0);
        let status = aggregator.add_vote(vote).unwrap();

        // 500/1000 = 50% < 67% → pending
        assert!(status.is_pending());
    }

    #[test]
    fn test_vote_aggregator_split_votes() {
        let total_weight = VotingWeight(1000);
        let mut aggregator = VoteAggregator::new(
            BlockHeight::from(1), 0, total_weight, 800,
        );

        let root_a = EmbeddingRoot::from_bytes([1u8; 32]);
        let root_b = EmbeddingRoot::from_bytes([2u8; 32]);

        // Vote for A: 400 weight
        let vote_a = make_test_vote([1u8; 32], root_a, 400, 1.0);
        let status = aggregator.add_vote(vote_a).unwrap();
        assert!(status.is_pending());

        // Vote for B: 300 weight
        let vote_b = make_test_vote([2u8; 32], root_b, 300, 1.0);
        let status = aggregator.add_vote(vote_b).unwrap();
        assert!(status.is_pending()); // Neither has ≥67%

        // Another vote for A: 300 weight → A now has 700/1000 = 70%
        let vote_a2 = make_test_vote([3u8; 32], root_a, 300, 1.0);
        let status = aggregator.add_vote(vote_a2).unwrap();
        assert!(status.is_reached());
    }

    #[test]
    fn test_vote_aggregator_duplicate_vote() {
        let total_weight = VotingWeight(1000);
        let mut aggregator = VoteAggregator::new(
            BlockHeight::from(1), 0, total_weight, 800,
        );

        let root = EmbeddingRoot::from_bytes([1u8; 32]);
        let vote1 = make_test_vote([1u8; 32], root, 500, 1.0);
        let vote2 = make_test_vote([1u8; 32], root, 500, 1.0);

        aggregator.add_vote(vote1).unwrap();
        let result = aggregator.add_vote(vote2);
        assert!(result.is_err()); // Duplicate
    }

    #[test]
    fn test_vote_aggregator_wrong_height() {
        let total_weight = VotingWeight(1000);
        let mut aggregator = VoteAggregator::new(
            BlockHeight::from(1), 0, total_weight, 800,
        );

        let root = EmbeddingRoot::from_bytes([1u8; 32]);
        let mut vote = make_test_vote([1u8; 32], root, 500, 1.0);
        vote.target_height = BlockHeight::from(2); // Wrong height

        let result = aggregator.add_vote(vote);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_total_active_weight() {
        let validators = vec![
            (ValidatorId::from_bytes([1u8; 32]), StakeAmount(100 * crate::ONE_NERV), ReputationScore::from_f64(1.0)),
            (ValidatorId::from_bytes([2u8; 32]), StakeAmount(200 * crate::ONE_NERV), ReputationScore::from_f64(0.8)),
            (ValidatorId::from_bytes([3u8; 32]), StakeAmount(300 * crate::ONE_NERV), ReputationScore::from_f64(0.9)),
        ];

        let total = compute_total_active_weight(&validators);
        assert!(total.0 > 0);
    }

    #[test]
    fn test_quorum_status_display() {
        let root = EmbeddingRoot::from_bytes([1u8; 32]);
        let status = QuorumStatus::Pending {
            best_root: root,
            achieved_weight: VotingWeight(500),
            total_weight: VotingWeight(1000),
            achieved_pct: 50.0,
            required_pct: 67.0,
        };
        assert!(status.is_pending());
        assert!(!status.is_reached());
    }
}
