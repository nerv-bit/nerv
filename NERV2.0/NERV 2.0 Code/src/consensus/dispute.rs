//! Monte-Carlo Dispute Resolution.
//!
//! When a validator disagrees with the majority prediction (which should
//! be extremely rare with the exact NWO Perceptron), they can post a
//! bonded challenge. 32 VRF-chosen validators then run 10,000 parallel
//! Monte-Carlo simulations to determine the correct embedding root.
//!
//! # Resolution Process
//!
//! ```text
//! 1. Challenger posts DisputeChallenge with 0.5–1% bond
//! 2. 32 validators selected via VRF from active set
//! 3. Each validator runs 10,000 Monte-Carlo simulations:
//!    a. Sample random subset of the batch (30–70%)
//!    b. Apply deltas sequentially to current embedding
//!    c. Compute resulting root hash
//!    d. Vote: majority root from simulations
//! 4. Aggregate simulation votes
//! 5. Winning root is finalised cryptographically
//! 6. Losing side slashed 0.5–5%
//! 7. Challenger bond returned if correct, forfeited if wrong
//! ```
//!
//! # Timing
//!
//! - Total resolution: <650ms
//! - Per-simulation: <1μs (just delta application + hash)
//! - Parallel across 32 validators: 10,000/32 ≈ 313 sims each

use crate::{
    EMBEDDING_DIM, NervError, NervResult,
    EmbeddingRoot, BlockHeight, ValidatorId, StakeAmount,
    DISPUTE_SIMULATIONS, DISPUTE_VALIDATOR_COUNT,
    CHALLENGER_BOND_MIN_BPS, CHALLENGER_BOND_MAX_BPS,
    SLASH_MIN_BPS, SLASH_MAX_BPS,
};
use crate::embedding::fixed_point::{FixedPoint64, EmbeddingVector};
use crate::embedding::homomorphism::EmbeddingDelta;
use crate::utils::{blake3_hash, random_bytes, random_32bytes};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Dispute Challenge ────────────────────────────────────────────────────

/// A dispute challenge from a validator who disagrees with the majority.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisputeChallenge {
    /// The challenging validator.
    pub challenger_id: ValidatorId,

    /// The block height being disputed.
    pub disputed_height: BlockHeight,

    /// The shard being disputed.
    pub shard_id: u64,

    /// The majority's predicted root (being challenged).
    pub majority_root: EmbeddingRoot,

    /// The challenger's predicted root.
    pub challenger_root: EmbeddingRoot,

    /// The challenger's bonded stake (0.5–1% of total stake).
    pub bond: StakeAmount,

    /// Dilithium-3 signature over the challenge.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub signature: Vec<u8>,

    /// Timestamp of the challenge.
    pub timestamp_ms: u64,
}

impl DisputeChallenge {
    /// Create a new dispute challenge.
    pub fn new(
        challenger_id: ValidatorId,
        disputed_height: BlockHeight,
        shard_id: u64,
        majority_root: EmbeddingRoot,
        challenger_root: EmbeddingRoot,
        total_stake: StakeAmount,
    ) -> Self {
        // Bond is 0.75% of total stake (midpoint of 0.5–1%)
        let bond = StakeAmount(
            ((total_stake.0 as f64 * 0.0075) as u64).max(1)
        );

        Self {
            challenger_id,
            disputed_height,
            shard_id,
            majority_root,
            challenger_root,
            bond,
            signature: Vec::new(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Validate the challenge parameters.
    pub fn validate(&self, total_stake: StakeAmount) -> NervResult<()> {
        if self.majority_root == self.challenger_root {
            return Err(NervError::Consensus(
                "majority and challenger roots are identical".into()
            ));
        }

        // Verify bond is within range
        let min_bond = (total_stake.0 as f64 * CHALLENGER_BOND_MIN_BPS as f64 / 10000.0) as u64;
        let max_bond = (total_stake.0 as f64 * CHALLENGER_BOND_MAX_BPS as f64 / 10000.0) as u64;

        if self.bond.0 < min_bond {
            return Err(NervError::Consensus(
                format!("bond {} below minimum {}", self.bond.0, min_bond)
            ));
        }
        if self.bond.0 > max_bond {
            return Err(NervError::Consensus(
                format!("bond {} above maximum {}", self.bond.0, max_bond)
            ));
        }

        Ok(())
    }
}

// ─── Monte-Carlo Simulation ──────────────────────────────────────────────

/// Result of a single Monte-Carlo simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloResult {
    /// The resulting embedding root from this simulation.
    pub resulting_root: EmbeddingRoot,

    /// The fraction of the batch used in this simulation.
    pub batch_fraction: f64,

    /// The random seed used (for reproducibility/audit).
    pub seed: [u8; 32],
}

impl MonteCarloResult {
    /// Run a single Monte-Carlo simulation.
    ///
    /// 1. Sample a random subset of the batch (30–70%)
    /// 2. Apply deltas sequentially to the current embedding
    /// 3. Compute the resulting root hash
    pub fn simulate(
        current_embedding: &EmbeddingVector,
        batch_deltas: &[EmbeddingDelta],
        seed: [u8; 32],
    ) -> Self {
        // Determine batch fraction (30–70%)
        let seed_val = u32::from_le_bytes(seed[..4].try_into().unwrap_or([0u8; 4]));
        let fraction = 0.3 + (seed_val as f64 / u32::MAX as f64) * 0.4;

        // Determine subset size
        let subset_size = ((batch_deltas.len() as f64 * fraction).round() as usize)
            .max(1)
            .min(batch_deltas.len());

        // Select random subset indices using the seed
        let mut indices: Vec<usize> = (0..batch_deltas.len()).collect();
        // Simple Fisher-Yates shuffle with seed-derived randomness
        for i in (1..indices.len()).rev() {
            let offset = (i + 1) * 4;
            let j = if offset + 4 <= seed.len() {
                u32::from_le_bytes(seed[offset..offset + 4].try_into().unwrap_or([0u8; 4])) as usize % (i + 1)
            } else {
                i
            };
            indices.swap(i, j);
        }

        // Apply deltas from the selected subset
        let mut embedding = current_embedding.clone();
        for &idx in &indices[..subset_size] {
            if idx < batch_deltas.len() {
                embedding = crate::embedding::homomorphism::apply_delta(
                    &embedding,
                    &batch_deltas[idx],
                );
            }
        }

        let resulting_root = embedding.hash();

        Self {
            resulting_root,
            batch_fraction: fraction,
            seed,
        }
    }
}

/// Run multiple Monte-Carlo simulations in parallel.
pub fn run_monte_carlo_simulations(
    current_embedding: &EmbeddingVector,
    batch_deltas: &[EmbeddingDelta],
    num_simulations: usize,
) -> Vec<MonteCarloResult> {
    (0..num_simulations)
        .map(|_| {
            let seed = random_32bytes();
            MonteCarloResult::simulate(current_embedding, batch_deltas, seed)
        })
        .collect()
}

/// Tally Monte-Carlo simulation results to determine the majority root.
pub fn tally_simulations(results: &[MonteCarloResult]) -> (EmbeddingRoot, usize, usize) {
    let mut tally: HashMap<[u8; 32], usize> = HashMap::new();
    for result in results {
        let key = *result.resulting_root.as_bytes();
        *tally.entry(key).or_insert(0) += 1;
    }

    let (winning_root, winning_count) = tally.iter()
        .max_by_key(|(_, count)| *count)
        .map(|(r, c)| (*r, *c))
        .unwrap_or(([0u8; 32], 0));

    (EmbeddingRoot::from_bytes(winning_root), winning_count, results.len())
}

// ─── Dispute State ───────────────────────────────────────────────────────

/// The state of a dispute resolution process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisputeState {
    /// Challenge submitted, awaiting validators.
    Challenged,
    /// Simulations in progress.
    Simulating,
    /// Resolution complete.
    Resolved,
    /// Dispute timed out.
    TimedOut,
}

// ─── Dispute Resolution ──────────────────────────────────────────────────

/// The result of a dispute resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisputeResolution {
    /// The winning embedding root.
    pub winning_root: EmbeddingRoot,

    /// Number of simulations that agreed with the winner.
    pub winning_sim_count: usize,

    /// Total simulations run.
    pub total_sim_count: usize,

    /// Whether the challenger was correct.
    pub challenger_correct: bool,

    /// Amount slashed from the losing side.
    pub slash_amount: StakeAmount,

    /// Whether the challenger's bond is returned.
    pub bond_returned: bool,

    /// Time taken for resolution (microseconds).
    pub resolution_time_us: u64,
}

impl DisputeResolution {
    /// Resolve a dispute using Monte-Carlo simulations.
    pub fn resolve(
        challenge: &DisputeChallenge,
        current_embedding: &EmbeddingVector,
        batch_deltas: &[EmbeddingDelta],
        num_simulations: usize,
        total_stake: StakeAmount,
    ) -> Self {
        let start = std::time::Instant::now();

        // Run Monte-Carlo simulations
        let results = run_monte_carlo_simulations(
            current_embedding,
            batch_deltas,
            num_simulations,
        );

        // Tally results
        let (winning_root, winning_count, total_count) = tally_simulations(&results);

        // Determine if challenger was correct
        let challenger_correct = winning_root == challenge.challenger_root;

        // Compute slashing
        let slash_pct = if challenger_correct {
            // Majority was wrong → slash majority side
            // Rate: 2–5% (higher if they were confidently wrong)
            let majority_agreement_pct = winning_count as f64 / total_count as f64;
            if majority_agreement_pct > 0.8 {
                0.05 // Strong dispute evidence → 5% slash
            } else {
                0.02 // Moderate evidence → 2% slash
            }
        } else {
            // Challenger was wrong → slash challenger
            // Rate: 0.5–1%
            0.01
        };

        let slash_amount = StakeAmount(
            ((total_stake.0 as f64) * slash_pct) as u64
        );

        // Bond is returned if challenger was correct
        let bond_returned = challenger_correct;

        let resolution_time_us = start.elapsed().as_micros() as u64;

        Self {
            winning_root,
            winning_sim_count: winning_count,
            total_sim_count: total_count,
            challenger_correct,
            slash_amount,
            bond_returned,
            resolution_time_us,
        }
    }
}

// ─── Dispute Manager ─────────────────────────────────────────────────────

/// Manages dispute challenges and resolutions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisputeManager {
    /// Active disputes (height → challenge).
    pub active_disputes: HashMap<u64, DisputeChallenge>,

    /// Dispute states (height → state).
    pub dispute_states: HashMap<u64, DisputeState>,

    /// Completed resolutions (height → resolution).
    pub resolutions: HashMap<u64, DisputeResolution>,

    /// Maximum concurrent disputes.
    pub max_concurrent: usize,

    /// Number of simulations per dispute.
    pub num_simulations: usize,

    /// Number of validators per dispute.
    pub num_validators: usize,
}

impl DisputeManager {
    /// Create a new dispute manager.
    pub fn new() -> Self {
        Self {
            active_disputes: HashMap::new(),
            dispute_states: HashMap::new(),
            resolutions: HashMap::new(),
            max_concurrent: 10,
            num_simulations: DISPUTE_SIMULATIONS,
            num_validators: DISPUTE_VALIDATOR_COUNT,
        }
    }

    /// Submit a dispute challenge.
    pub fn submit_challenge(&mut self, challenge: DisputeChallenge) -> NervResult<()> {
        let height = challenge.disputed_height.into();
        if self.active_disputes.contains_key(&height) {
            return Err(NervError::Consensus(
                "dispute already active for this height".into()
            ));
        }
        if self.active_disputes.len() >= self.max_concurrent {
            return Err(NervError::Consensus(
                format!("max concurrent disputes ({}) reached", self.max_concurrent)
            ));
        }
        if self.resolutions.contains_key(&height) {
            return Err(NervError::Consensus(
                "dispute already resolved for this height".into()
            ));
        }

        self.active_disputes.insert(height, challenge);
        self.dispute_states.insert(height, DisputeState::Challenged);
         Ok(())
    }

    /// Check if a dispute is active for a height.
    pub fn is_disputed(&self, height: BlockHeight) -> bool {
        self.active_disputes.contains_key(&height.into())
    }

    /// Get the dispute state for a height.
    pub fn get_state(&self, height: BlockHeight) -> Option<DisputeState> {
        self.dispute_states.get(&height.into()).copied()
    }

    /// Record a completed resolution.
    pub fn record_resolution(&mut self, height: BlockHeight, resolution: DisputeResolution) {
        let h: u64 = height.into();
        self.resolutions.insert(h, resolution);
        self.dispute_states.insert(h, DisputeState::Resolved);
        self.active_disputes.remove(&h);
    }

    /// Get the number of active disputes.
    pub fn active_count(&self) -> usize {
        self.active_disputes.len()
   }
}

impl std::default::Default for DisputeManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_embedding() -> EmbeddingVector {
        EmbeddingVector::splat(FixedPoint64::from_int(100))
    }

    fn make_test_deltas() -> Vec<EmbeddingDelta> {
        (0..10).map(|i| {
            EmbeddingDelta::splat(FixedPoint64::from_int(i as i64))
        }).collect()
    }

    #[test]
    fn test_dispute_challenge_creation() {
        let challenge = DisputeChallenge::new(
            ValidatorId::from_bytes([1u8; 32]),
            BlockHeight::from(100),
            0,
            EmbeddingRoot::from_bytes([1u8; 32]),
            EmbeddingRoot::from_bytes([2u8; 32]),
            StakeAmount(1000 * crate::ONE_NERV),
        );

        assert_eq!(challenge.disputed_height, BlockHeight::from(100));
        assert!(challenge.bond.0 > 0);
    }

    #[test]
    fn test_dispute_challenge_validation() {
        let challenge = DisputeChallenge::new(
            ValidatorId::from_bytes([1u8; 32]),
            BlockHeight::from(100),
            0,
            EmbeddingRoot::from_bytes([1u8; 32]),
            EmbeddingRoot::from_bytes([2u8; 32]),
            StakeAmount(1000 * crate::ONE_NERV),
        );

        let result = challenge.validate(StakeAmount(1000 * crate::ONE_NERV));
        assert!(result.is_ok());
    }

    #[test]
    fn test_dispute_challenge_same_roots() {
        let challenge = DisputeChallenge {
            challenger_id: ValidatorId::from_bytes([1u8; 32]),
            disputed_height: BlockHeight::from(100),
            shard_id: 0,
            majority_root: EmbeddingRoot::from_bytes([1u8; 32]),
            challenger_root: EmbeddingRoot::from_bytes([1u8; 32]), // Same!
            bond: StakeAmount(10),
            signature: Vec::new(),
            timestamp_ms: 0,
        };

        let result = challenge.validate(StakeAmount(1000 * crate::ONE_NERV));
        assert!(result.is_err());
    }

    #[test]
    fn test_monte_carlo_simulation() {
        let embedding = make_test_embedding();
        let deltas = make_test_deltas();
        let seed = [42u8; 32];

        let result = MonteCarloResult::simulate(&embedding, &deltas, seed);
        assert!(!result.resulting_root.is_genesis());
        assert!(result.batch_fraction >= 0.3);
        assert!(result.batch_fraction <= 0.7);
    }

    #[test]
    fn test_monte_carlo_deterministic() {
        let embedding = make_test_embedding();
        let deltas = make_test_deltas();
        let seed = [42u8; 32];

        let result1 = MonteCarloResult::simulate(&embedding, &deltas, seed);
        let result2 = MonteCarloResult::simulate(&embedding, &deltas, seed);
        assert_eq!(result1.resulting_root, result2.resulting_root);
    }

    #[test]
    fn test_monte_carlo_different_seeds() {
        let embedding = make_test_embedding();
        let deltas = make_test_deltas();

        let result1 = MonteCarloResult::simulate(&embedding, &deltas, [1u8; 32]);
        let result2 = MonteCarloResult::simulate(&embedding, &deltas, [2u8; 32]);
        // Different seeds may produce different results (not guaranteed, but likely)
        // We just verify both%both run without panicking
    }

    #[test]
    fn test_tally_simulations() {
        let root_a = EmbeddingRoot::from_bytes([1u8; 32]);
        let root_b = EmbeddingRoot::from_bytes([2u8; 32]);

        let results = vec![
            MonteCarloResult { resulting_root: root_a, batch_fraction: 0.5, seed: [0u8; 32] },
            MonteCarloResult { resulting_root: root_a, batch_fraction: 0.5, seed: [0u8; 32] },
            MonteCarloResult { resulting_root: root_a, batch_fraction: 0.5, seed: [0u8; 32] },
            MonteCarloResult { resulting_root: root_b, batch_fraction: 0.5, seed: [0u8; 32] },
        ];

        let (winner, count, total) = tally_simulations(&results);
        assert_eq!(winner, root_a);
        assert_eq!(count, 3);
        assert_eq!(total, 4);
    }

    #[test]
    fn test_dispute_resolution() {
        let embedding = make_test_embedding();
        let deltas = make_test_deltas();

        let challenge = DisputeChallenge::new(
            ValidatorId::from_bytes([1u8; 32]),
            BlockHeight::from(100),
            0,
            EmbeddingRoot::from_bytes([1u8; 32]),
            EmbeddingRoot::from_bytes([2u8; 32]),
            StakeAmount(1000 * crate::ONE_NERV),
        );

        let resolution = DisputeResolution::resolve(
            &challenge,
            &embedding,
            &deltas,
            100, // Small number for test speed
            StakeAmount(1000 * crate::ONE_NERV),
        );

        assert_eq!(resolution.total_sim_count, 100);
        assert_eq!(resolution.resolution_time_us < 10_000_000); // <10s
    }

    #[test]
    fn test_dispute_manager() {
        let mut manager = DisputeManager::new();
        assert_eq!(manager.active_count(), 0);

        let challenge = DisputeChallenge::new(
            ValidatorId::from_bytes([1u8; 32]),
            BlockHeight::from(100),
            0,
            EmbeddingRoot::from_bytes([1u8; 32]),
            EmbeddingRoot::from_bytes([2u8; 32]),
            StakeAmount(1000 * crate::ONE_NERV),
        );

        manager.submit_challenge(challenge).unwrap();
        assert_eq!(manager.active_count(), 1);
        assert!(manager.is_disputed(BlockHeight::from(100)));
        assert_eq!(manager.get_state(BlockHeight::from(100)), Some(DisputeState::Challenged));
    }

    #[test]
    fn test_dispute_manager_duplicate() {
        let mut manager = DisputeManager::new();

        let challenge1 = DisputeChallenge::new(
            ValidatorId::from_bytes([1u8; 32]),
            BlockHeight::from(100), 0,
            EmbeddingRoot::from_bytes([1u8; 32]),
            EmbeddingRoot::from_bytes([2u8; 32]),
            StakeAmount(1000 * crate::ONE_NERV),
        );
        let challenge2 = DisputeChallenge::new(
            ValidatorId::from_bytes([2u8; 32]),
            BlockHeight::from(100), 0,
            EmbeddingRoot::from_bytes([1u8; 32]),
            EmbeddingRoot::from_bytes([3u8; 32]),
            StakeAmount(1000 * crate::ONE_NERV),
        );

        manager.submit_challenge(challenge1).unwrap();
        let result = manager.submit_challenge(challenge2);
        assert!(result.is_err()); // Duplicate height
    }
}
