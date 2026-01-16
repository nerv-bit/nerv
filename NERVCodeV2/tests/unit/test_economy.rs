// tests/unit/test_economy.rs
// ============================================================================
// ECONOMY MODULE UNIT TESTS
// ============================================================================

use nerv_bit::economy::{
    EconomyManager, EconomyConfig, EconomyMetrics, NodeInfo,
    shapley::{ShapleyComputer, ShapleyConfig, GradientContribution},
    fl_aggregation::{FLAggregator, AggregationConfig, GradientUpdate, QualityAssessor},
    rewards::{RewardDistributor, RewardConfig, IndividualReward, VestingSchedule},
    encoder_updater::EncoderUpdater,
};
use nerv_bit::embedding::encoder::{NeuralEncoder, EncoderConfig};
use nerv_bit::crypto::CryptoProvider;
use nerv_bit::consensus::ConsensusEngine; // Mock or minimal trait if needed
use chrono::{Utc, Duration};
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::test]
async fn test_economy_manager_creation_and_basic_ops() {
    let config = EconomyConfig::default();
    let manager = EconomyManager::new(config).await.unwrap();

    // Epoch info
    let epoch_info = manager.get_epoch_info().await;
    assert_eq!(epoch_info.current_epoch, 0);
    assert_eq!(epoch_info.total_gradients_processed, 0);

    // Metrics
    let metrics = manager.get_metrics().await;
    assert_eq!(metrics.total_gradients_submitted, 0);

    // Node registry (add a test node)
    manager.register_node("test_node".to_string()).await.unwrap();
    let nodes = manager.get_registered_nodes().await;
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].node_id, "test_node");
    assert!((nodes[0].reputation_score - 0.5).abs() < 1e-9);
}

#[test]
fn test_economy_config_defaults() {
    let config = EconomyConfig::default();

    assert_eq!(config.base_emission_per_epoch, 1_000_000);
    assert_eq!(config.dp_clip_norm, 1.0);
    assert!((config.reputation_decay - 0.95).abs() < 1e-9);
    assert!(config.rewards_enabled);
}

#[tokio::test]
async fn test_shapley_computer_basic_computation() {
    let config = ShapleyConfig {
        monte_carlo_samples: 200, // Small but sufficient for test
        privacy_budget: 0.0,      // Disable DP for deterministic test
        max_coalition_size: 10,
        enable_parallel: false,
        tee_backed: false,
        cache_ttl_seconds: 3600,
        min_confidence_threshold: 0.1,
    };

    let computer = ShapleyComputer::new(config);

    // Minimal contributions (2 nodes)
    let mut contributions = std::collections::HashMap::new();
    contributions.insert(
        "node1".to_string(),
        GradientContribution::new("node1".to_string(), vec![1.0, 0.0], 0.9, 2.0, vec![]),
    );
    contributions.insert(
        "node2".to_string(),
        GradientContribution::new("node2".to_string(), vec![0.0, 1.0], 0.8, 1.5, vec![]),
    );

    let result = computer.compute_shapley(&contributions).await.unwrap();

    assert_eq!(result.len(), 2);
    let total: f64 = result.iter().map(|v| v.value).sum();
    // Efficiency axiom: sum should be ~1.0 (normalized value function)
    assert!((total - 1.0).abs() < 0.2); // Tolerance due to Monte Carlo

    // Higher quality node should get higher value
    let node1_val = result.iter().find(|v| v.node_id == "node1").unwrap().value;
    let node2_val = result.iter().find(|v| v.node_id == "node2").unwrap().value;
    assert!(node1_val > node2_val);
}

#[tokio::test]
async fn test_fl_aggregator_creation_and_gradient_submission() {
    let crypto = Arc::new(CryptoProvider::new().unwrap());
    let config = AggregationConfig::default();
    let aggregator = FLAggregator::new(config, crypto);

    // Submit a simple gradient
    let gradient = GradientUpdate::new(vec![0.1, -0.2, 0.3], vec![]);
    aggregator.submit_gradient("test_node".to_string(), gradient).await.unwrap();

    let submissions = aggregator.submissions.read().await;
    assert_eq!(submissions.len(), 1);
    assert!(submissions.contains_key("test_node"));
}

#[tokio::test]
async fn test_quality_assessor() {
    let config = AggregationConfig::default();
    let assessor = QualityAssessor::new(config);

    let good_gradient = GradientUpdate::new(vec![0.5; 10], vec![]); // Reasonable norm
    let quality = assessor.assess(&good_gradient).await.unwrap();
    assert!(quality > 0.5);

    let bad_gradient = GradientUpdate::new(vec![1000.0], vec![]); // Huge norm → low quality
    let quality_bad = assessor.assess(&bad_gradient).await.unwrap();
    assert!(quality_bad < 0.5);
}

#[tokio::test]
async fn test_reward_distributor_basic_flow() {
    let config = RewardConfig::default();
    let distributor = RewardDistributor::new(config);

    // Simulate Shapley values
    let mut shapley_map = std::collections::HashMap::new();
    shapley_map.insert("node1".to_string(), 0.6);
    shapley_map.insert("node2".to_string(), 0.4);

    // Distribute rewards (total pool 10000)
    let rewards = distributor
        .calculate_rewards(shapley_map, 10_000.0)
        .await
        .unwrap();

    assert_eq!(rewards.len(), 2);
    let node1_reward = rewards.iter().find(|r| r.node_id == "node1").unwrap();
    assert!(node1_reward.final_amount > 5000.0 && node1_reward.final_amount < 7000.0);

    // Apply vesting
    distributor.apply_vesting("node1".to_string(), node1_reward.final_amount as u128);
    let schedules = distributor.vesting_schedules.read().await;
    assert!(schedules.contains_key("node1"));
}

#[test]
fn test_vesting_schedule_logic() {
    let start = Utc::now();
    let schedule = VestingSchedule {
        node_id: "test".to_string(),
        total_amount: 1_000_000,
        start_date: start,
        cliff_months: 3,
        vesting_months: 12,
        claimed_amount: 0,
        created_at: start,
    };

    // Before cliff
    let before_cliff = start + Duration::days(60);
    assert_eq!(schedule.vested_amount(before_cliff), 0);

    // After cliff, partial
    let partial = start + Duration::days(180); // ~6 months total
    let vested = schedule.vested_amount(partial);
    assert!(vested > 0 && vested < 1_000_000);

    // Full vesting
    let full = start + Duration::days(500);
    assert_eq!(schedule.vested_amount(full), 1_000_000);
}

#[tokio::test]
async fn test_encoder_updater_lifecycle() {
    // Minimal mocks
    let encoder = Arc::new(RwLock::new(NeuralEncoder::new(EncoderConfig::default()).unwrap()));
    let aggregator = Arc::new(FLAggregator::new(AggregationConfig::default(), Arc::new(CryptoProvider::new().unwrap())));
    let consensus = Arc::new(MockConsensusEngine::new()); // Define a simple mock below

    let updater = EncoderUpdater::new(encoder, aggregator, consensus, 10);

    // Run a short cycle (won't actually update without gradients, but tests spawn)
    tokio::spawn(async move {
        updater.run(0).await;
    });

    // Sleep briefly to allow loop iteration
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    // No panic → success
}

// Simple mock for ConsensusEngine to allow compilation
struct MockConsensusEngine;
impl MockConsensusEngine {
    fn new() -> Self { Self }
}
#[async_trait::async_trait]
impl ConsensusEngine for MockConsensusEngine {
    async fn current_height(&self) -> u64 { 100 }
    async fn propose_parameter_change(&self, _: ParameterProposal) -> Result<()> { Ok(()) }
    // Minimal impl - add more if needed
}
use async_trait::async_trait;
use crate::consensus::ParameterProposal; // Adjust import as needed