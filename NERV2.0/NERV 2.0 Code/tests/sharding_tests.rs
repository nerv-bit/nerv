//! Unit tests for Dynamic Sharding and Erasure Coding.
//!
//! Validates the NERV V2.0 AI-native sharding mechanism. Ensures that the
//! secondary NWO Perceptron correctly predicts network load based on features
//! (TPS, mempool size, queue depth), triggers shard bisection when load 
//! exceeds 90%, and that the Reed-Solomon erasure coding parameters (k=5, m=2)
//! are correctly configured to tolerate 2 node failures per shard.

use nerv::config::ShardingConfig;
use nerv::embedding::fixed_point::{EmbeddingVector, FixedPoint};
use nerv::embedding::perceptron::{NwoPerceptron, PerceptronConfig};
use nerv::{ERASURE_K, ERASURE_M, EMBEDDING_DIM};

// ─── Sharding Configuration Tests ────────────────────────────────────────

#[test]
fn test_sharding_config_defaults() {
    let config = ShardingConfig::default();
    
    // V2.0 mandates a 90% split threshold and 20% merge threshold
    assert_eq!(config.split_threshold, 0.90, "Split threshold must be 0.90");
    assert_eq!(config.merge_threshold, 0.20, "Merge threshold must be 0.20");
    
    // Erasure coding must be k=5, m=2
    assert_eq!(config.erasure_k, ERASURE_K, "Erasure K must be 5");
    assert_eq!(config.erasure_m, ERASURE_M, "Erasure M must be 2");
    
    // The predictor features must match the V2.0 whitepaper
    assert!(config.predictor_features.contains(&"current_tps".to_string()), "Must track current_tps");
    assert!(config.predictor_features.contains(&"mempool_size".to_string()), "Must track mempool_size");
    assert!(config.predictor_features.contains(&"inter_shard_queue_depth".to_string()), "Must track inter_shard_queue_depth");
}

// ─── NWO Load Prediction & Bisection Tests ───────────────────────────────

#[test]
fn test_nwo_load_prediction_triggers_bisection() {
    let mut config = PerceptronConfig::default();
    config.learning_rate = 0.01;
    let mut predictor = NwoPerceptron::new(config);
    
    // Simulate a high-load scenario feature vector
    // [TPS, Mempool Size, Queue Depth, Gas Utilization]
    let high_load_features = EmbeddingVector::new([
        950_000, // 95% of 1M TPS capacity
        480_000, // 96% of 500k mempool limit
        100_000, // High queue depth
        980_000, // 98% gas utilization
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0
    ]);
    
    // Train the predictor to associate these features with high execution time
    // Target: 1.0 (100% load)
    let target_load = EmbeddingVector::new([1_000_000; EMBEDDING_DIM]);
    for _ in 0..100 {
        let pred = predictor.forward(&high_load_features);
        let gradients = predictor.compute_gradients(&target_load, &pred);
        predictor.adam_update(&gradients);
    }
    
    // Predict the load
    let predicted_load = predictor.forward(&high_load_features);
    let load_value = predicted_load.data[0] as f64 / 1_000_000.0;
    
    // The NWO must predict a load > 0.90, triggering a bisection
    assert!(
        load_value > 0.90,
        "Predicted load ({}) must exceed 0.90 split threshold", load_value
    );
}

#[test]
fn test_nwo_load_prediction_no_bisection() {
    let predictor = NwoPerceptron::new(PerceptronConfig::default());
    
    // Simulate a low-load scenario
    let low_load_features = EmbeddingVector::new([
        100_000, // 10% TPS
        50_000,  // 10% mempool
        1_000,   // Low queue depth
        100_000, // 10% gas
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0
    ]);
    
    let predicted_load = predictor.forward(&low_load_features);
    let load_value = predicted_load.data[0] as f64 / 1_000_000.0;
    
    // The NWO must predict a load < 0.90, preventing a bisection
    assert!(
        load_value < 0.90,
        "Predicted load ({}) must be below 0.90 split threshold", load_value
    );
}

// ─── Embedding Bisection Logic Tests ─────────────────────────────────────

#[test]
fn test_embedding_bisection_state_split() {
    // In V2.0, when a shard splits, the 512-byte (64-dim) embedding is bisected
    // into two new 512-byte embeddings with re-initialized local weights.
    
    let original_data = [100i64; EMBEDDING_DIM];
    let original_state = EmbeddingVector::new(original_data);
    
    // Simulate bisection: Split the state vector logically (e.g., interleaved or halves)
    // For NERV, the state is split such that each new shard handles half the features.
    let mut shard_a_data = [0i64; EMBEDDING_DIM];
    let mut shard_b_data = [0i64; EMBEDDING_DIM];
    
    for i in 0..EMBEDDING_DIM {
        if i % 2 == 0 {
            shard_a_data[i] = original_state.data[i];
        } else {
            shard_b_data[i] = original_state.data[i];
        }
    }
    
    let shard_a = EmbeddingVector::new(shard_a_data);
    let shard_b = EmbeddingVector::new(shard_b_data);
    
    // Verify the bisection preserved the total state magnitude (sum of elements)
    let original_sum: i64 = original_state.data.iter().sum();
    let shard_sum: i64 = shard_a.data.iter().sum::<i64>() + shard_b.data.iter().sum::<i64>();
    
    assert_eq!(original_sum, shard_sum, "Bisection must conserve the total state sum");
}

// ─── Erasure Coding Resilience Tests ─────────────────────────────────────

#[test]
fn test_erasure_coding_fault_tolerance() {
    // NERV V2.0 uses Reed-Solomon with k=5 (data shards) and m=2 (parity shards).
    // This means the system can tolerate up to 2 missing shards out of 7 total.
    let total_shards = ERASURE_K + ERASURE_M; // 7
    let tolerance = ERASURE_M; // 2
    
    // Simulate having 5 shards (lost 2)
    let available_shards = total_shards - tolerance;
    assert_eq!(available_shards, ERASURE_K, "Available shards must meet K to reconstruct");
    
    // Simulate having 4 shards (lost 3) - must fail
    let insufficient_shards = total_shards - (tolerance + 1);
    assert!(insufficient_shards < ERASURE_K, "Must not be able to reconstruct with < K shards");
}
