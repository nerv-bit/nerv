Rust// tests/unit/test_sharding.rs
// ============================================================================
// SHARDING MODULE UNIT TESTS
// ============================================================================

use nerv_bit::sharding::{
    ShardingManager, ShardingConfig, ShardState, ShardOperation,
    lstm_predictor::{LstmLoadPredictor, LoadPrediction, ShardLoadMetrics, PredictionConfig},
    bisection::{EmbeddingBisection, ShardBoundary, EmbeddingPartition},
    erasure::{ReedSolomonEncoder, ReedSolomonDecoder, ErasureCodingConfig, EncoderStats},
};
use nerv_bit::embedding::{EmbeddingVec, EmbeddingHash};
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn test_sharding_manager_creation_and_basic_ops() {
    let config = ShardingConfig::default();
    let manager = ShardingManager::new(config).await.unwrap();

    // Initial state
    assert_eq!(manager.get_active_shards().await.len(), 1); // Genesis shard
    assert_eq!(manager.current_shard_count().await, 1);

    // Add mock load metrics
    let metrics = ShardLoadMetrics {
        shard_id: 0,
        tx_count: 1000,
        embedding_count: 500,
        storage_bytes: 1_000_000,
        cpu_load: 0.8,
        timestamp: std::time::SystemTime::now(),
    };
    manager.record_load_metrics(metrics).await.unwrap();

    let stats = manager.get_statistics().await;
    assert_eq!(stats.total_transactions, 1000);
    assert!(stats.avg_load_per_shard > 0.0);
}

#[test]
fn test_sharding_config_defaults() {
    let config = ShardingConfig::default();

    assert_eq!(config.max_shards, 1024);
    assert_eq!(config.min_shards, 1);
    assert_eq!(config.target_load_per_shard, 10_000);
    assert!(config.split_threshold > 1.0);
    assert!(config.merge_threshold < 1.0);
    assert!(config.erasure_coding.enabled);
    assert_eq!(config.erasure_coding.data_shards, 5);
    assert_eq!(config.erasure_coding.parity_shards, 2);
}

#[tokio::test]
async fn test_shard_split_proposal() {
    let mut config = ShardingConfig::default();
    config.target_load_per_shard = 100; // Low threshold for test
    config.split_threshold = 1.5;

    let manager = ShardingManager::new(config).await.unwrap();

    // Overload genesis shard
    for _ in 0..200 {
        let metrics = ShardLoadMetrics {
            shard_id: 0,
            tx_count: 200,
            embedding_count: 100,
            storage_bytes: 500_000,
            cpu_load: 0.9,
            timestamp: std::time::SystemTime::now(),
        };
        manager.record_load_metrics(metrics).await.unwrap();
    }

    // Trigger prediction and check for split proposal
    manager.run_prediction_cycle().await.unwrap();

    let operations = manager.get_pending_operations().await;
    assert!(!operations.is_empty());
    assert!(operations.iter().any(|op| matches!(op, ShardOperation::Split { .. })));
}

#[tokio::test]
async fn test_shard_merge_proposal() {
    let mut config = ShardingConfig::default();
    config.target_load_per_shard = 1000;
    config.merge_threshold = 0.3;

    let manager = ShardingManager::new(config).await.unwrap();

    // Manually create underloaded shards (simulate post-split)
    manager.force_split(0).await.unwrap(); // Creates shard 1

    // Record very low load
    for shard_id in 0..2 {
        let metrics = ShardLoadMetrics {
            shard_id,
            tx_count: 10,
            embedding_count: 5,
            storage_bytes: 10_000,
            cpu_load: 0.1,
            timestamp: std::time::SystemTime::now(),
        };
        manager.record_load_metrics(metrics).await.unwrap();
    }

    manager.run_prediction_cycle().await.unwrap();

    let operations = manager.get_pending_operations().await;
    assert!(operations.iter().any(|op| matches!(op, ShardOperation::Merge { .. })));
}

#[test]
fn test_embedding_bisection_simple_partition() {
    let mut embeddings = vec![];
    let mut rng = thread_rng();

    // Cluster 1: around [0.0; 512]
    for _ in 0..50 {
        let mut vec = [0.0f64; 512];
        for v in vec.iter_mut() {
            *v = rng.gen_range(-0.1..0.1);
        }
        embeddings.push(EmbeddingVec::new(vec));
    }

    // Cluster 2: around [1.0; 512]
    for _ in 0..50 {
        let mut vec = [1.0f64; 512];
        for v in vec.iter_mut() {
            *v += rng.gen_range(-0.1..0.1);
        }
        embeddings.push(EmbeddingVec::new(vec));
    }

    let bisection = EmbeddingBisection::new();
    let partitions = bisection.partition_embeddings(&embeddings, 2).unwrap();

    assert_eq!(partitions.len(), 2);
    // Each partition should have roughly 50 embeddings
    assert!(partitions[0].embeddings.len() >= 40 && partitions[0].embeddings.len() <= 60);
    assert!(partitions[1].embeddings.len() >= 40 && partitions[1].embeddings.len() <= 60);

    // Centroids should be far apart
    let dist = partitions[0].boundary.centroid.distance(&partitions[1].boundary.centroid);
    assert!(dist > 0.5);
}

#[test]
fn test_reed_solomon_encoding_decoding() {
    let config = ErasureCodingConfig::default();
    let encoder = ReedSolomonEncoder::new(config.clone());

    let data = vec![vec![42u8; 1024]; 5]; // 5 data shards

    let shards = encoder.encode(&data).unwrap();
    assert_eq!(shards.len(), 7); // 5 data + 2 parity

    let stats = encoder.stats();
    assert_eq!(stats.successful_encodes, 1);
    assert!(stats.avg_encode_time_ms > 0.0);

    // Test reconstruction with 2 missing shards
    let mut incomplete = shards.clone();
    incomplete[0] = vec![]; // Missing data shard
    incomplete[5] = vec![]; // Missing parity shard

    let decoder = ReedSolomonDecoder::new(config);
    let reconstructed = decoder.reconstruct(&incomplete).unwrap();

    assert_eq!(reconstructed.len(), 5);
    for i in 0..5 {
        assert_eq!(reconstructed[i], data[i]);
    }
}

#[test]
fn test_reed_solomon_cache() {
    let config = ErasureCodingConfig::default();
    let mut encoder = ReedSolomonEncoder::new(config);

    encoder.cache.max_size = 2;

    let data1 = vec![vec![1u8; 512]; 5];
    let hash1 = blake3::hash(&bincode::serialize(&data1).unwrap()).into();

    let shards1 = encoder.encode(&data1).unwrap();
    // Cache hit on second encode
    let shards1_cached = encoder.encode(&data1).unwrap();
    assert_eq!(shards1, shards1_cached);

    let data2 = vec![vec![2u8; 512]; 5];
    encoder.encode(&data2).unwrap();

    // Cache should evict oldest (data1) when full
    let data3 = vec![vec![3u8; 512]; 5];
    encoder.encode(&data3).unwrap();

    // data1 should be evicted
    assert!(encoder.cache.get(&hash1).is_none());
}

#[tokio::test]
async fn test_lstm_predictor_mock_fallback() {
    // Use invalid model path to trigger fallback
    let mut config = PredictionConfig::default();
    config.model_path = "nonexistent.onnx".to_string();

    let predictor = LstmLoadPredictor::new(config).await.unwrap();
    assert!(predictor.fallback_mode);

    let history = vec![1000.0f32; 60]; // 1 hour of data

    let prediction = predictor.predict_load(0, &history).unwrap();
    assert!(prediction.predicted_load > 0.0);
    assert!(prediction.confidence > 0.5); // Fallback heuristic

    // Record feedback
    predictor.record_feedback(0, prediction.predicted_load, 1100.0).await;
    let accuracy = predictor.get_accuracy_metrics();
    assert!(accuracy.accuracy_10_percent > 0.0);
}

#[tokio::test]
async fn test_sharding_state_persistence() {
    let config = ShardingConfig::default();
    let manager = ShardingManager::new(config).await.unwrap();

    // Modify state
    manager.force_split(0).await.unwrap();

    // Save
    manager.save_state().await.unwrap();

    // Create new manager and load
    let manager2 = ShardingManager::new(config).await.unwrap();
    manager2.load_state().await.unwrap();

    assert_eq!(manager2.current_shard_count().await, 2);
}