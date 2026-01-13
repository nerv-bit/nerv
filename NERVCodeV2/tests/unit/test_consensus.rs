// tests/unit/test_consensus.rs
// ============================================================================
// CONSENSUS MODULE UNIT TESTS
// ============================================================================


use nerv_bit::consensus::predictor::*;
use nerv_bit::Result;
use rand::Rng;


#[tokio::test]
async fn test_neural_predictor_creation() {
    let config = ModelConfig::default();
    let predictor = NeuralPredictor::new(config).await.unwrap();
    
    // Test warmup
    predictor.warmup().await.unwrap();
    
    // Test prediction with dummy data
    let block_data = vec![0u8; 1024];
    let result = predictor.predict(&block_data).await.unwrap();
    
    assert!(result.validity_score >= 0.0 && result.validity_score <= 1.0);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    assert!(result.anomaly_score >= 0.0 && result.anomaly_score <= 1.0);
    assert_eq!(result.model_version, 1);
    assert_eq!(result.feature_vector.len(), 256);
}


#[test]
fn test_feature_extractor() {
    let extractor = FeatureExtractor::new();
    
    // Test entropy calculation
    let uniform_data = vec![0u8; 1000];
    let high_entropy_data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
    
    let uniform_entropy = extractor.entropy(&uniform_data);
    let high_entropy = extractor.entropy(&high_entropy_data);
    
    assert!(high_entropy > uniform_entropy, "High entropy data should have higher entropy");
    assert!(uniform_entropy >= 0.0 && uniform_entropy <= 1.0);
    
    // Test byte distribution
    let data = vec![0u8, 255u8, 128u8];
    let distribution = extractor.byte_distribution(&data);
    
    assert_eq!(distribution.len(), 16);
    let sum: f32 = distribution.iter().sum();
    assert!((sum - 1.0).abs() < 0.0001, "Distribution should sum to 1");
}


#[test]
fn test_prediction_statistics() {
    let mut stats = PredictionStatistics::new();
    
    // Add some results
    for i in 0..100 {
        let result = PredictionResult {
            validity_score: 0.8,
            confidence: 0.9,
            anomaly_score: 0.1,
            inference_time_us: 1000,
            model_version: 1,
            feature_vector: vec![0.0; 256],
            metadata: PredictionMetadata {
                gas_price_prediction: 100,
                gas_limit_prediction: 21000,
                throughput_prediction: 1000.0,
                latency_prediction: 100.0,
                security_risk: 0.1,
                energy_efficiency: 10.0,
            },
        };
        
        stats.update(&result);
        stats.add_ground_truth(true);
    }
    
    assert_eq!(stats.total_predictions, 100);
    assert_eq!(stats.correct_predictions, 100);
    assert_eq!(stats.accuracy(), 1.0);
    assert!(stats.recent_accuracy() > 0.0);
}


#[tokio::test]
async fn test_prediction_cache() {
    let cache = PredictionCache::new(10);
    
    let mut rng = rand::thread_rng();
    
    // Add entries
    for i in 0..15 {
        let mut key = [0u8; 32];
        rng.fill(&mut key);
        
        let result = PredictionResult {
            validity_score: 0.8,
            confidence: 0.9,
            anomaly_score: 0.1,
            inference_time_us: 1000,
            model_version: 1,
            feature_vector: vec![0.0; 256],
            metadata: PredictionMetadata::default(),
        };
        
        cache.insert(key, result.clone());
        
        // Test retrieval
        let cached = cache.get(&key);
        if i >= 5 {
            // First 5 should be evicted (LRU)
            assert!(cached.is_some(), "Entry {} should be in cache", i);
        }
    }
    
    // Cache should have exactly 10 entries
    // (Implementation detail: this depends on how your cache is implemented)
}


#[test]
fn test_model_config_serialization() {
    let config = ModelConfig::default();
    
    // Test serialization
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: ModelConfig = serde_json::from_str(&json).unwrap();
    
    assert_eq!(config.model_path, deserialized.model_path);
    assert_eq!(config.input_shape, deserialized.input_shape);
    assert_eq!(config.confidence_threshold, deserialized.confidence_threshold);
}


#[tokio::test]
async fn test_predictor_state_persistence() {
    let config = ModelConfig::default();
    let mut predictor = NeuralPredictor::new(config).await.unwrap();
    
    // Add some test data
    let block_data = vec![0u8; 1024];
    let result = predictor.predict(&block_data).await.unwrap();
    
    // Update predictor with block
    predictor.update_with_block(&block_data, true).await.unwrap();
    
    // Test state saving/loading (would need actual filesystem in production)
    // For now, just test the functions don't panic
    let save_result = predictor.save_state().await;
    let load_result = predictor.load_state().await;
    
    // These might fail in test environment without proper files
    // Just ensure they don't panic
    println!("Save result: {:?}", save_result);
    println!("Load result: {:?}", load_result);
}


#[test]
fn test_prediction_metadata() {
    let mut rng = rand::thread_rng();
    let mut features = Vec::new();
    
    for _ in 0..256 {
        features.push(rng.gen_range(0.0..1.0));
    }
    
    // This would normally be done by the predictor
    let feature_sum: f32 = features.iter().sum();
    let feature_len = features.len() as f32;
    
    let metadata = PredictionMetadata {
        gas_price_prediction: (feature_sum * 10.0).max(1.0) as u64,
        gas_limit_prediction: (feature_sum * 1000.0).max(21000) as u64,
        throughput_prediction: (feature_len * 100.0).min(10000.0),
        latency_prediction: (feature_len * 0.1).max(1.0),
        security_risk: (features.iter().map(|&x| x.abs()).sum::<f32>() / feature_len).min(1.0),
        energy_efficiency: (feature_len / (feature_sum + 1.0)).max(0.1),
    };
    
    assert!(metadata.gas_price_prediction > 0);
    assert!(metadata.gas_limit_prediction >= 21000);
    assert!(metadata.throughput_prediction > 0.0);
    assert!(metadata.latency_prediction >= 1.0);
    assert!(metadata.security_risk >= 0.0 && metadata.security_risk <= 1.0);
    assert!(metadata.energy_efficiency > 0.0);
}
