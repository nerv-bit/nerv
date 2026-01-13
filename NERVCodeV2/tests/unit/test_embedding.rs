// tests/unit/test_embedding.rs
// ============================================================================
// EMBEDDING MODULE UNIT TESTS
// ============================================================================


use crate::unit::TestResult;
use nerv_bit::embedding::encoder::*;
use nerv_bit::embedding::homomorphism::*;
use nerv_bit::Result;
use rand::Rng;


#[test]
fn test_neural_encoder_creation() {
    let config = EncoderConfig::default();
    let encoder = NeuralEncoder::new(config).unwrap();
    
    assert_eq!(encoder.config.embedding_dim, 512);
    assert_eq!(encoder.layers.len(), 24);
    assert_eq!(encoder.epoch, 0);
    
    let param_count = encoder.parameter_count();
    assert!(param_count > 0);
    println!("Encoder parameters: {}", param_count);
    
    let size_mb = encoder.size_mb();
    assert!(size_mb > 0.0);
    assert!(size_mb < 100.0, "Encoder too large: {}MB", size_mb);
}


#[test]
fn test_encoder_quantization() {
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let quantized = quantize_tensor(&data, 8);
    
    assert_eq!(quantized.bits, 8);
    assert_eq!(quantized.data.len(), 5);
    
    // Dequantize and check
    let dequantized = quantized.dequantize();
    for i in 0..5 {
        assert!((data[i] - dequantized[i]).abs() < 0.1);
    }
}


#[test]
fn test_encoder_serialization() {
    let temp_dir = std::env::temp_dir();
    let test_path = temp_dir.join("test_encoder_serialization.json");
    
    let encoder = create_default_encoder().unwrap();
    
    // Save encoder
    encoder.save(&test_path).unwrap();
    assert!(test_path.exists());
    
    // Load encoder
    let loaded = NeuralEncoder::load(&test_path).unwrap();
    
    // Compare properties
    assert_eq!(encoder.config.embedding_dim, loaded.config.embedding_dim);
    assert_eq!(encoder.layers.len(), loaded.layers.len());
    assert_eq!(encoder.epoch, loaded.epoch);
    
    // Clean up
    std::fs::remove_file(&test_path).ok();
}


#[tokio::test]
async fn test_homomorphism_transfer() {
    let mut rng = rand::thread_rng();
    
    // Create account manager
    let manager = AccountEmbeddingManager::new();
    
    // Create test accounts
    let account1 = [1u8; 32];
    let account2 = [2u8; 32];
    
    // Create transfer transaction
    let transaction = TransferTransaction {
        sender: account1,
        receiver: account2,
        amount: FixedPoint32_16::from_float(100.0),
        nonce: 1,
        timestamp: 1234567890,
        balance_proof: None,
    };
    
    // This should fail since we don't have embeddings
    let result = manager.compute_transfer_delta(transaction).await;
    assert!(result.is_err());
}


#[test]
fn test_transfer_delta_computation() {
    let mut rng = rand::thread_rng();
    
    // Create test embeddings
    let mut sender_embedding = Vec::new();
    let mut receiver_embedding = Vec::new();
    let mut bias_terms = Vec::new();
    
    for i in 0..512 {
        sender_embedding.push(FixedPoint32_16::from_float(rng.gen_range(-1.0..1.0)));
        receiver_embedding.push(FixedPoint32_16::from_float(rng.gen_range(-1.0..1.0)));
        bias_terms.push(FixedPoint32_16::from_float(rng.gen_range(-0.1..0.1)));
    }
    
    // Create transaction
    let transaction = TransferTransaction {
        sender: [1u8; 32],
        receiver: [2u8; 32],
        amount: FixedPoint32_16::from_float(100.0),
        nonce: 1,
        timestamp: 1234567890,
        balance_proof: None,
    };
    
    // Create delta
    let delta = TransferDelta::new(
        transaction,
        &sender_embedding,
        &receiver_embedding,
        &bias_terms,
    ).unwrap();
    
    assert_eq!(delta.delta.len(), 512);
    
    // Verify delta computation
    let verified = delta.verify(
        &sender_embedding,
        &receiver_embedding,
        &bias_terms,
    ).unwrap();
    
    assert!(verified);
}


#[test]
fn test_homomorphism_verifier() {
    let mut verifier = HomomorphismVerifier::new();
    
    // Create test embeddings
    let mut old_embedding = Vec::new();
    let mut new_embedding = Vec::new();
    let mut delta_values = Vec::new();
    
    for i in 0..512 {
        let old_val = FixedPoint32_16::from_float(i as f64);
        let delta_val = FixedPoint32_16::from_float(1.0);
        let new_val = FixedPoint32_16::from_float(i as f64 + 1.0);
        
        old_embedding.push(old_val);
        new_embedding.push(new_val);
        delta_values.push(delta_val);
    }
    
    // Create delta
    let transaction = TransferTransaction {
        sender: [1u8; 32],
        receiver: [2u8; 32],
        amount: FixedPoint32_16::from_float(1.0),
        nonce: 1,
        timestamp: 1234567890,
        balance_proof: None,
    };
    
    let delta = TransferDelta {
        delta: delta_values,
        transaction,
        error_bound: FixedPoint32_16::from_float(1e-9),
        signature: None,
    };
    
    // Verify homomorphism (should pass)
    let result = verifier.verify_single_transfer(
        &old_embedding,
        &delta,
        &new_embedding,
    ).unwrap();
    
    assert!(result);
    assert_eq!(verifier.stats().successful_verifications, 1);
}


#[test]
fn test_embedding_utils() {
    use nerv_bit::embedding::homomorphism::utils;
    
    // Create test embedding
    let mut embedding = Vec::new();
    for i in 0..512 {
        embedding.push(FixedPoint32_16::from_float(i as f64));
    }
    
    // Test serialization
    let bytes = utils::embedding_to_bytes(&embedding);
    let reconstructed = utils::bytes_to_embedding(&bytes).unwrap();
    
    assert_eq!(embedding.len(), reconstructed.len());
    
    // Test hashing
    let hash1 = utils::hash_embedding(&embedding);
    assert_eq!(hash1.len(), 32);
    
    // Test normalization
    let normalized = utils::normalize_embedding(&embedding);
    let mut norm_sq = 0.0;
    for &fp in &normalized {
        let val = fp.to_float();
        norm_sq += val * val;
    }
    
    let norm = norm_sq.sqrt();
    assert!((norm - 1.0).abs() < 1e-6, "Normalized embedding should have unit norm");
}


#[tokio::test]
async fn test_encoder_trainer() {
    use nerv_bit::embedding::encoder::create_default_encoder;
    
    let encoder = create_default_encoder().unwrap();
    let encoder_arc = std::sync::Arc::new(encoder);
    
    let mut trainer = EncoderTrainer::new(encoder_arc);
    
    // Run training epoch (should handle empty dataset)
    let loss = trainer.train_epoch().await.unwrap();
    assert_eq!(loss, 0.0);
    
    // Test with random examples
    let test_results = trainer.test_homomorphism(10).await.unwrap();
    assert_eq!(test_results.total_tests, 10);
    println!("Test pass rate: {:.1}%", test_results.pass_rate() * 100.0);
}


#[test]
fn test_batch_delta_aggregation() {
    let mut rng = rand::thread_rng();
    
    // Create multiple deltas
    let mut deltas = Vec::new();
    
    for i in 0..10 {
        // Create test embeddings
        let mut sender_embedding = Vec::new();
        let mut receiver_embedding = Vec::new();
        let bias_terms = vec![FixedPoint32_16::from_float(0.0); 512];
        
        for j in 0..512 {
            sender_embedding.push(FixedPoint32_16::from_float(
                rng.gen_range(-1.0..1.0)
            ));
            receiver_embedding.push(FixedPoint32_16::from_float(
                rng.gen_range(-1.0..1.0)
            ));
        }
        
        let transaction = TransferTransaction {
            sender: [i as u8; 32],
            receiver: [(i + 1) as u8; 32],
            amount: FixedPoint32_16::from_float(rng.gen_range(1.0..100.0)),
            nonce: i as u64,
            timestamp: 1234567890 + i as u64,
            balance_proof: None,
        };
        
        let delta = TransferDelta::new(
            transaction,
            &sender_embedding,
            &receiver_embedding,
            &bias_terms,
        ).unwrap();
        
        deltas.push(delta);
    }
    
    // Create batch delta
    let batch = BatchDelta::new(deltas).unwrap();
    
    assert_eq!(batch.transactions.len(), 10);
    assert_eq!(batch.aggregated_delta.len(), 512);
    
    // Test compression ratio
    let compression = batch.compression_ratio(100_000);
    assert!(compression > 100.0, "Compression ratio should be >100x, got {}", compression);
}
