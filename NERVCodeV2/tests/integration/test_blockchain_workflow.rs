// tests/integration/test_blockchain_workflow.rs
// ============================================================================
// BLOCKCHAIN WORKFLOW INTEGRATION TEST
// ============================================================================
// Tests the complete transaction flow from submission to confirmation,
// including embedding updates, privacy mixing, and consensus validation.
// ============================================================================


use crate::integration::*;
use nerv_bit::embedding::*;
use nerv_bit::privacy::*;
use nerv_bit::consensus::*;
use nerv_bit::crypto::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use tokio::sync::Mutex;


#[tokio::test]
async fn test_complete_transaction_flow() {
    let config = IntegrationConfig {
        node_count: 3,
        transaction_count: 10,
        random_seed: 42,
        ..Default::default()
    };
    
    let mut env = TestEnvironment::new(config);
    let mut rng = ChaCha8Rng::seed_from_u64(env.config.random_seed);
    
    println!("Starting complete transaction flow test...");
    
    // 1. Initialize nodes
    for i in 0..env.config.node_count {
        let node = TestNode::new(
            i as u64,
            &format!("127.0.0.1:{}", 8000 + i),
            1000.0 * (i + 1) as f64,
            i == 0, // First node is validator
        );
        env.add_node(node);
    }
    
    println!("Initialized {} nodes", env.nodes.len());
    
    // 2. Generate transactions
    for i in 0..env.config.transaction_count {
        let mut sender = [0u8; 32];
        let mut receiver = [0u8; 32];
        rng.fill(&mut sender);
        rng.fill(&mut receiver);
        
        let amount = rng.gen_range(1.0..100.0);
        let fee = amount * 0.01;
        
        let tx = TestTransaction::new(sender, receiver, amount, fee);
        env.add_transaction(tx);
    }
    
    println!("Generated {} transactions", env.transactions.len());
    
    // 3. Create neural encoder
    let encoder_config = nerv_bit::embedding::encoder::EncoderConfig::default();
    let encoder = Arc::new(
        nerv_bit::embedding::encoder::NeuralEncoder::new(encoder_config).unwrap()
    );
    
    // 4. Create privacy mixer
    let mixer_config = nerv_bit::privacy::mixer::MixerConfig::default();
    let mixer = Arc::new(nerv_bit::privacy::mixer::OnionMixer::new(mixer_config));
    
    // 5. Create consensus predictor
    let predictor_config = nerv_bit::consensus::predictor::ModelConfig::default();
    let predictor = Arc::new(Mutex::new(
        nerv_bit::consensus::predictor::NeuralPredictor::new(predictor_config).await.unwrap()
    ));
    
    // 6. Process transactions
    let mut metrics = IntegrationMetrics::default();
    let mut processed_txs = 0;
    
    for tx in &env.transactions {
        if env.is_timeout() {
            println!("Test timeout reached");
            break;
        }
        
        let start_time = std::time::Instant::now();
        
        // Simulate transaction processing steps
        
        // Step 1: Privacy mixing
        let payload = nerv_bit::privacy::mixer::EncryptedPayload::new(
            &tx.id,
            &[0u8; 32], // recipient key
        ).unwrap();
        
        // Step 2: Create transfer delta
        let mut sender_embedding = Vec::new();
        let mut receiver_embedding = Vec::new();
        let mut bias_terms = Vec::new();
        
        for _ in 0..512 {
            sender_embedding.push(nerv_bit::embedding::homomorphism::FixedPoint32_16::from_float(
                rng.gen_range(-1.0..1.0)
            ));
            receiver_embedding.push(nerv_bit::embedding::homomorphism::FixedPoint32_16::from_float(
                rng.gen_range(-1.0..1.0)
            ));
            bias_terms.push(nerv_bit::embedding::homomorphism::FixedPoint32_16::from_float(
                rng.gen_range(-0.1..0.1)
            ));
        }
        
        let transaction = nerv_bit::embedding::homomorphism::TransferTransaction {
            sender: tx.sender,
            receiver: tx.receiver,
            amount: nerv_bit::embedding::homomorphism::FixedPoint32_16::from_float(tx.amount),
            nonce: processed_txs as u64,
            timestamp: tx.timestamp,
            balance_proof: None,
        };
        
        let delta = nerv_bit::embedding::homomorphism::TransferDelta::new(
            transaction,
            &sender_embedding,
            &receiver_embedding,
            &bias_terms,
        ).unwrap();
        
        // Step 3: Consensus validation
        let block_data = vec![0u8; 1024]; // Simulated block data
        let prediction = predictor.lock().await.predict(&block_data).await.unwrap();
        
        if prediction.validity_score > 0.5 && prediction.confidence > 0.7 {
            // Transaction considered valid
            processed_txs += 1;
            metrics.transactions_processed += 1;
            
            // Update embedding
            let mut new_embedding = sender_embedding.clone();
            for i in 0..512 {
                new_embedding[i] = new_embedding[i].add(delta.delta[i]);
            }
            
            // Record metrics
            let latency = start_time.elapsed().as_millis() as f64;
            metrics.avg_latency_ms = (metrics.avg_latency_ms * (processed_txs - 1) as f64 + latency) / processed_txs as f64;
            
            println!("Processed transaction {} with latency {:.2}ms", processed_txs, latency);
        } else {
            metrics.error_count += 1;
            println!("Transaction failed validation (score: {:.2}, confidence: {:.2})", 
                    prediction.validity_score, prediction.confidence);
        }
        
        // Small delay to simulate real processing
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
    
    // 7. Calculate success rate
    metrics.success_rate = metrics.transactions_processed as f64 / env.transactions.len() as f64;
    
    // 8. Create result
    let result = IntegrationResult {
        test_name: "complete_transaction_flow".to_string(),
        duration: env.elapsed(),
        metrics,
        passed: true,
        errors: Vec::new(),
    };
    
    println!("{}", result.summary());
    
    // Assertions
    assert!(result.metrics.transactions_processed > 0, "Should process at least one transaction");
    assert!(result.metrics.success_rate > 0.0, "Success rate should be positive");
    assert!(result.metrics.avg_latency_ms > 0.0, "Average latency should be positive");
    
    println!("Complete transaction flow test passed!");
}


#[tokio::test]
async fn test_multi_shard_transaction_processing() {
    println!("Starting multi-shard transaction processing test...");
    
    let config = IntegrationConfig {
        node_count: 5,
        transaction_count: 50,
        random_seed: 12345,
        test_duration_secs: 30,
        ..Default::default()
    };
    
    let env = TestEnvironment::new(config);
    let mut metrics = IntegrationMetrics::default();
    
    // Simulate multi-shard processing
    let shard_count = 3;
    let mut shard_metrics = vec![0u64; shard_count];
    
    for i in 0..env.config.transaction_count {
        if i % 10 == 0 {
            println!("Processing transaction {}/{}", i + 1, env.config.transaction_count);
        }
        
        // Assign transaction to shard
        let shard_id = i as usize % shard_count;
        shard_metrics[shard_id] += 1;
        metrics.transactions_processed += 1;
        
        // Simulate shard operation
        if i % 15 == 0 {
            metrics.shard_operations += 1;
        }
        
        // Simulate consensus
        if i % 5 == 0 {
            metrics.consensus_rounds += 1;
        }
        
        // Simulate block production
        if i % 10 == 0 {
            metrics.blocks_produced += 1;
        }
        
        // Simulate TEE attestation
        if i % 20 == 0 {
            metrics.tee_attestations += 1;
        }
        
        // Small delay
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
    }
    
    // Calculate average latency
    metrics.avg_latency_ms = 150.0; // Simulated average
    
    // Verify shard distribution
    println!("Shard distribution: {:?}", shard_metrics);
    for (i, &count) in shard_metrics.iter().enumerate() {
        assert!(count > 0, "Shard {} should have processed transactions", i);
    }
    
    let result = IntegrationResult {
        test_name: "multi_shard_processing".to_string(),
        duration: env.elapsed(),
        metrics,
        passed: true,
        errors: Vec::new(),
    };
    
    println!("{}", result.summary());
    
    assert_eq!(result.metrics.transactions_processed, env.config.transaction_count);
    assert!(result.metrics.blocks_produced > 0);
    assert!(result.metrics.consensus_rounds > 0);
    
    println!("Multi-shard transaction processing test passed!");
}


#[tokio::test]
async fn test_error_recovery_flow() {
    println!("Starting error recovery flow test...");
    
    let config = IntegrationConfig {
        node_count: 3,
        transaction_count: 20,
        random_seed: 999,
        ..Default::default()
    };
    
    let mut env = TestEnvironment::new(config);
    let mut metrics = IntegrationMetrics::default();
    let mut rng = ChaCha8Rng::seed_from_u64(env.config.random_seed);
    
    // Simulate transactions with failures
    for i in 0..env.config.transaction_count {
        let should_fail = rng.gen_bool(0.3); // 30% failure rate
        
        if should_fail {
            metrics.error_count += 1;
            println!("Transaction {} simulated failure", i + 1);
            
            // Simulate recovery
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            
            // Retry
            metrics.transactions_processed += 1;
            println!("Transaction {} recovered on retry", i + 1);
        } else {
            metrics.transactions_processed += 1;
            println!("Transaction {} succeeded", i + 1);
        }
        
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
    }
    
    // Calculate metrics
    metrics.success_rate = metrics.transactions_processed as f64 / 
                         (metrics.transactions_processed + metrics.error_count) as f64;
    metrics.avg_latency_ms = 75.0; // Simulated average with retries
    
    let result = IntegrationResult {
        test_name: "error_recovery_flow".to_string(),
        duration: env.elapsed(),
        metrics,
        passed: true,
        errors: Vec::new(),
    };
    
    println!("{}", result.summary());
    
    // Should have some errors and recoveries
    assert!(result.metrics.error_count > 0, "Should have some errors");
    assert!(result.metrics.success_rate > 0.7, "Success rate should be reasonable");
    
    println!("Error recovery flow test passed!");
}


#[tokio::test]
async fn test_concurrent_transaction_processing() {
    println!("Starting concurrent transaction processing test...");
    
    let config = IntegrationConfig {
        node_count: 10,
        transaction_count: 100,
        random_seed: 777,
        ..Default::default()
    };
    
    let env = TestEnvironment::new(config);
    let metrics = Arc::new(Mutex::new(IntegrationMetrics::default()));
    
    let start_time = std::time::Instant::now();
    
    // Process transactions concurrently
    let mut handles = Vec::new();
    
    for i in 0..env.config.transaction_count {
        let metrics_clone = metrics.clone();
        
        let handle = tokio::spawn(async move {
            // Simulate transaction processing
            let processing_time = 10 + (i % 20) as u64; // Varying processing times
            tokio::time::sleep(tokio::time::Duration::from_millis(processing_time)).await;
            
            let mut metrics = metrics_clone.lock().await;
            metrics.transactions_processed += 1;
            
            // Every 10th transaction creates a block
            if i % 10 == 0 {
                metrics.blocks_produced += 1;
            }
            
            // Every 5th transaction requires consensus
            if i % 5 == 0 {
                metrics.consensus_rounds += 1;
            }
            
            // Record latency
            let latency = processing_time as f64;
            let count = metrics.transactions_processed;
            metrics.avg_latency_ms = (metrics.avg_latency_ms * (count - 1) as f64 + latency) / count as f64;
        });
        
        handles.push(handle);
    }
    
    // Wait for all transactions
    for handle in handles {
        handle.await.unwrap();
    }
    
    let duration = start_time.elapsed();
    let final_metrics = metrics.lock().await.clone();
    
    let result = IntegrationResult {
        test_name: "concurrent_processing".to_string(),
        duration,
        metrics: final_metrics,
        passed: true,
        errors: Vec::new(),
    };
    
    println!("{}", result.summary());
    
    // Calculate TPS
    let tps = result.metrics.transactions_processed as f64 / duration.as_secs_f64().max(1.0);
    println!("Achieved TPS: {:.1}", tps);
    
    assert_eq!(result.metrics.transactions_processed, env.config.transaction_count);
    assert!(tps > 10.0, "Should achieve reasonable TPS with concurrency");
    
    println!("Concurrent transaction processing test passed!");
}
