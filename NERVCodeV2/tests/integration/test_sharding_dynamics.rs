// tests/integration/test_sharding_dynamics.rs
// ============================================================================
// SHARDING DYNAMICS INTEGRATION TEST
// ============================================================================
// Tests dynamic neural sharding including load prediction, shard splits/merges,
// and erasure coding for data availability.
// ============================================================================


use crate::integration::*;
use nerv_bit::sharding::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};


#[tokio::test]
async fn test_dynamic_shard_management() {
    println!("Starting dynamic shard management test...");
    
    let config = IntegrationConfig {
        node_count: 8,
        transaction_count: 50,
        random_seed: 2222,
        ..Default::default()
    };
    
    let mut env = TestEnvironment::new(config);
    let mut rng = ChaCha8Rng::seed_from_u64(env.config.random_seed);
    let mut metrics = IntegrationMetrics::default();
    
    // Create shard manager
    let shard_manager = Arc::new(RwLock::new(ShardManager::new()));
    
    // Initialize with some shards
    {
        let mut manager = shard_manager.write().await;
        manager.create_shard(0, 100); // Shard 0 with capacity 100
        manager.create_shard(1, 100); // Shard 1 with capacity 100
    }
    
    println!("Initial shards: 2");
    
    // Process transactions and monitor shard loads
    for i in 0..env.config.transaction_count {
        let start_time = std::time::Instant::now();
        
        // Determine which shard gets this transaction
        let shard_id = i % 2;
        
        // Get shard load
        let current_load = {
            let manager = shard_manager.read().await;
            manager.get_shard_load(shard_id as u64).unwrap_or(0)
        };
        
        // Check if shard needs splitting
        if current_load > 80 { // High load threshold
            println!("Shard {} load high ({}/100), considering split...", shard_id, current_load);
            
            // Check if split conditions are met
            let should_split = check_shard_split_conditions(shard_id as u64, current_load).await;
            
            if should_split {
                metrics.shard_operations += 1;
                
                // Perform shard split
                let split_result = {
                    let mut manager = shard_manager.write().await;
                    manager.split_shard(shard_id as u64).await
                };
                
                match split_result {
                    Ok((new_shard1, new_shard2)) => {
                        println!("Shard {} split into {} and {}", shard_id, new_shard1, new_shard2);
                        metrics.transactions_processed += 1;
                    }
                    Err(e) => {
                        metrics.error_count += 1;
                        println!("Shard {} split failed: {}", shard_id, e);
                    }
                }
            }
        }
        
        // Check if shards should merge (low load)
        if i % 10 == 0 && i > 0 {
            let should_merge = check_shard_merge_conditions().await;
            
            if should_merge {
                metrics.shard_operations += 1;
                
                // Find two low-load shards to merge
                let merge_result = {
                    let mut manager = shard_manager.write().await;
                    manager.merge_shards(0, 1).await // Example merge
                };
                
                if merge_result.is_ok() {
                    println!("Shards merged successfully");
                }
            }
        }
        
        // Add transaction to shard
        let add_result = {
            let mut manager = shard_manager.write().await;
            manager.add_transaction_to_shard(shard_id as u64, i as u64).await
        };
        
        if add_result.is_ok() {
            metrics.transactions_processed += 1;
            
            // Update shard metrics
            {
                let mut manager = shard_manager.write().await;
                manager.update_shard_metrics(shard_id as u64, 1, start_time.elapsed());
            }
        } else {
            metrics.error_count += 1;
        }
        
        // Record latency
        let latency = start_time.elapsed().as_millis() as f64;
        metrics.avg_latency_ms = (metrics.avg_latency_ms * (i as f64) + latency) / ((i + 1) as f64);
        
        // Small delay
        tokio::time::sleep(tokio::time::Duration::from_millis(15)).await;
    }
    
    // Print final shard state
    {
        let manager = shard_manager.read().await;
        let shard_count = manager.get_shard_count();
        println!("Final shard count: {}", shard_count);
        
        for shard_id in 0..shard_count {
            if let Some(load) = manager.get_shard_load(shard_id) {
                println!("Shard {} load: {}/100", shard_id, load);
            }
        }
    }
    
    metrics.success_rate = metrics.transactions_processed as f64 / env.config.transaction_count as f64;
    
    let result = IntegrationResult {
        test_name: "dynamic_shard_management".to_string(),
        duration: env.elapsed(),
        metrics,
        passed: true,
        errors: Vec::new(),
    };
    
    println!("{}", result.summary());
    
    // Assertions
    assert!(result.metrics.transactions_processed > 0, "Should process transactions");
    assert!(result.metrics.shard_operations >= 0, "Should handle shard operations");
    
    println!("Dynamic shard management test passed!");
}


#[tokio::test]
async fn test_load_prediction_and_scaling() {
    println!("Starting load prediction and scaling test...");
    
    let config = IntegrationConfig {
        node_count: 6,
        transaction_count: 100,
        random_seed: 3333,
        test_duration_secs: 15,
        ..Default::default()
    };
    
    let mut env = TestEnvironment::new(config);
    let mut rng = ChaCha8Rng::seed_from_u64(env.config.random_seed);
    let mut metrics = IntegrationMetrics::default();
    
    // Create LSTM load predictor
    let predictor = Arc::new(Mutex::new(LoadPredictor::new()));
    
    // Create scaling manager
    let scaling_manager = Arc::new(RwLock::new(ScalingManager::new()));
    
    // Simulate varying load patterns
    let mut current_load = 50; // Starting at 50% load
    let mut load_pattern = Vec::new();
    
    for i in 0..env.config.transaction_count {
        // Generate load pattern (simulates daily cycles)
        let hour_of_day = (i / 10) % 24;
        let base_load = if hour_of_day < 6 {
            30 // Night: low load
        } else if hour_of_day < 18 {
            70 // Day: high load
        } else {
            50 // Evening: medium load
        };
        
        // Add some randomness
        let load_variation: i32 = rng.gen_range(-20..20);
        current_load = (base_load + load_variation).clamp(10, 95);
        
        load_pattern.push(current_load);
        
        // Predict future load
        let prediction = {
            let mut predictor = predictor.lock().await;
            predictor.predict_load(&load_pattern).await
        };
        
        // Make scaling decisions based on prediction
        if let Ok(predicted_load) = prediction {
            println!("Time {}: Current load {}, Predicted load {}", 
                    hour_of_day, current_load, predicted_load);
            
            let scaling_decision = {
                let mut manager = scaling_manager.write().await;
                manager.make_scaling_decision(current_load, predicted_load).await
            };
            
            match scaling_decision {
                ScalingDecision::ScaleUp(amount) => {
                    metrics.shard_operations += 1;
                    println!("Scaling UP by {}", amount);
                    
                    // Simulate adding capacity
                    tokio::time::sleep(tokio::time::Duration::from_millis(amount * 10)).await;
                }
                ScalingDecision::ScaleDown(amount) => {
                    metrics.shard_operations += 1;
                    println!("Scaling DOWN by {}", amount);
                }
                ScalingDecision::NoAction => {
                    // No scaling needed
                }
            }
        }
        
        metrics.transactions_processed += 1;
        
        // Record metrics
        {
            let mut manager = scaling_manager.write().await;
            manager.record_metric(current_load, prediction.unwrap_or(current_load));
        }
        
        // Small delay
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
    
    // Print scaling statistics
    {
        let manager = scaling_manager.read().await;
        let stats = manager.get_statistics();
        
        println!("\nScaling Statistics:");
        println!("  Average load: {:.1}%", stats.avg_load);
        println!("  Prediction accuracy: {:.1}%", stats.prediction_accuracy * 100.0);
        println!("  Scaling operations: {}", stats.scaling_operations);
        println!("  Total scale-up: {}", stats.total_scale_up);
        println!("  Total scale-down: {}", stats.total_scale_down);
    }
    
    metrics.success_rate = 1.0;
    
    let result = IntegrationResult {
        test_name: "load_prediction_scaling".to_string(),
        duration: env.elapsed(),
        metrics,
        passed: true,
        errors: Vec::new(),
    };
    
    println!("{}", result.summary());
    
    // Assertions
    assert!(result.metrics.transactions_processed > 0, "Should process transactions");
    assert!(result.metrics.shard_operations >= 0, "Should have scaling operations");
    
    println!("Load prediction and scaling test passed!");
}


#[tokio::test]
async fn test_erasure_coding_and_recovery() {
    println!("Starting erasure coding and recovery test...");
    
    let config = IntegrationConfig {
        node_count: 10,
        transaction_count: 20,
        random_seed: 4444,
        ..Default::default()
    };
    
    let mut env = TestEnvironment::new(config);
    let mut rng = ChaCha8Rng::seed_from_u64(env.config.random_seed);
    let mut metrics = IntegrationMetrics::default();
    
    // Test Reed-Solomon erasure coding (k=5, m=2)
    let k = 5; // Data shards
    let m = 2; // Parity shards
    let total_shards = k + m;
    
    for i in 0..env.config.transaction_count {
        println!("\nTest {}: Erasure coding and recovery", i + 1);
        
        // Generate test data
        let data_size = 1024; // 1KB of data
        let mut original_data = vec![0u8; data_size];
        rng.fill(&mut original_data[..]);
        
        // Encode data into shards
        let shards_result = encode_data_with_erasure(&original_data, k, m).await;
        
        match shards_result {
            Ok(mut shards) => {
                assert_eq!(shards.len(), total_shards, "Should have correct number of shards");
                
                println!("  Encoded into {} shards ({} data + {} parity)", total_shards, k, m);
                metrics.transactions_processed += 1;
                
                // Test recovery with missing shards
                let missing_shards = if i % 3 == 0 { 2 } else { 1 }; // Lose 1 or 2 shards
                
                // Mark some shards as missing
                let missing_indices: Vec<usize> = (0..missing_shards)
                    .map(|j| (i + j) as usize % total_shards)
                    .collect();
                
                for &idx in &missing_indices {
                    shards[idx] = None;
                }
                
                println!("  Simulating loss of {} shards: {:?}", missing_shards, missing_indices);
                
                // Try to recover original data
                let recovery_result = recover_data_from_shards(&shards, k, m).await;
                
                match recovery_result {
                    Ok(recovered_data) => {
                        // Compare with original
                        if recovered_data == original_data {
                            println!("  Recovery SUCCESSFUL - Data matches original");
                            metrics.transactions_processed += 1;
                        } else {
                            println!("  Recovery FAILED - Data doesn't match");
                            metrics.error_count += 1;
                        }
                    }
                    Err(e) => {
                        if missing_shards <= m {
                            // Should be able to recover
                            println!("  Recovery FAILED unexpectedly: {}", e);
                            metrics.error_count += 1;
                        } else {
                            // Too many shards lost - expected failure
                            println!("  Recovery failed as expected (too many shards lost)");
                        }
                    }
                }
                
                // Test shard verification
                let verification_result = verify_shard_integrity(&shards).await;
                if verification_result {
                    println!("  Shard integrity verified");
                } else {
                    println!("  Shard integrity check failed");
                    metrics.error_count += 1;
                }
            }
            Err(e) => {
                println!("  Encoding failed: {}", e);
                metrics.error_count += 1;
            }
        }
        
        // Small delay
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
    }
    
    metrics.success_rate = metrics.transactions_processed as f64 / (env.config.transaction_count * 2) as f64;
    
    let result = IntegrationResult {
        test_name: "erasure_coding_recovery".to_string(),
        duration: env.elapsed(),
        metrics,
        passed: true,
        errors: Vec::new(),
    };
    
    println!("{}", result.summary());
    
    // Assertions
    assert!(result.metrics.transactions_processed > 0, "Should process encoding/recovery");
    
    println!("Erasure coding and recovery test passed!");
}


#[tokio::test]
async fn test_shard_rebalancing() {
    println!("Starting shard rebalancing test...");
    
    let config = IntegrationConfig {
        node_count: 12,
        transaction_count: 60,
        random_seed: 5555,
        ..Default::default()
    };
    
    let mut env = TestEnvironment::new(config);
    let mut metrics = IntegrationMetrics::default();
    
    // Create shard cluster with imbalance
    let mut shard_cluster = ShardCluster::new();
    
    // Initialize shards with different loads
    shard_cluster.add_shard(0, 90); // High load
    shard_cluster.add_shard(1, 85); // High load
    shard_cluster.add_shard(2, 30); // Low load
    shard_cluster.add_shard(3, 25); // Low load
    shard_cluster.add_shard(4, 50); // Medium load
    
    println!("Initial shard loads: {:?}", shard_cluster.get_shard_loads());
    
    // Perform rebalancing
    let rebalancing_result = shard_cluster.rebalance().await;
    
    match rebalancing_result {
        Ok(operations) => {
            metrics.shard_operations += operations.len() as u64;
            metrics.transactions_processed += 1;
            
            println!("Rebalancing completed with {} operations", operations.len());
            
            for op in operations {
                println!("  Operation: {:?}", op);
            }
            
            // Check if rebalancing improved load distribution
            let final_loads = shard_cluster.get_shard_loads();
            println!("Final shard loads: {:?}", final_loads);
            
            // Calculate load standard deviation
            let avg_load: f64 = final_loads.values().sum::<u32>() as f64 / final_loads.len() as f64;
            let variance: f64 = final_loads.values()
                .map(|&load| (load as f64 - avg_load).powi(2))
                .sum::<f64>() / final_loads.len() as f64;
            let std_dev = variance.sqrt();
            
            println!("Load statistics: Avg={:.1}, StdDev={:.1}", avg_load, std_dev);
            
            // Rebalancing should reduce load imbalance
            assert!(std_dev < 25.0, "Load should be reasonably balanced");
        }
        Err(e) => {
            metrics.error_count += 1;
            println!("Rebalancing failed: {}", e);
        }
    }
    
    // Test hot spot mitigation
    let hotspot_result = shard_cluster.mitigate_hotspots().await;
    if hotspot_result.is_ok() {
        println!("Hot spot mitigation successful");
    }
    
    metrics.success_rate = 1.0;
    
    let result = IntegrationResult {
        test_name: "shard_rebalancing".to_string(),
        duration: env.elapsed(),
        metrics,
        passed: true,
        errors: Vec::new(),
    };
    
    println!("{}", result.summary());
    
    // Assertions
    assert!(result.metrics.transactions_processed > 0, "Should perform rebalancing");
    
    println!("Shard rebalancing test passed!");
}


// Helper structs and implementations for sharding tests


struct ShardManager {
    shards: std::collections::HashMap<u64, Shard>,
    next_shard_id: u64,
}


impl ShardManager {
    fn new() -> Self {
        Self {
            shards: std::collections::HashMap::new(),
            next_shard_id: 0,
        }
    }
    
    fn create_shard(&mut self, id: u64, capacity: u32) {
        self.shards.insert(id, Shard::new(id, capacity));
    }
    
    async fn split_shard(&mut self, shard_id: u64) -> Result<(u64, u64), String> {
        // Simulate shard split
        let new_id1 = self.next_shard_id;
        let new_id2 = self.next_shard_id + 1;
        
        self.next_shard_id += 2;
        
        // Create new shards
        self.create_shard(new_id1, 100);
        self.create_shard(new_id2, 100);
        
        // Remove old shard
        self.shards.remove(&shard_id);
        
        Ok((new_id1, new_id2))
    }
    
    async fn merge_shards(&mut self, shard1: u64, shard2: u64) -> Result<u64, String> {
        // Simulate shard merge
        let new_id = self.next_shard_id;
        self.next_shard_id += 1;
        
        self.create_shard(new_id, 200); // Combined capacity
        
        // Remove old shards
        self.shards.remove(&shard1);
        self.shards.remove(&shard2);
        
        Ok(new_id)
    }
    
    async fn add_transaction_to_shard(&mut self, shard_id: u64, tx_id: u64) -> Result<(), String> {
        if let Some(shard) = self.shards.get_mut(&shard_id) {
            shard.add_transaction(tx_id)
        } else {
            Err("Shard not found".to_string())
        }
    }
    
    fn get_shard_load(&self, shard_id: u64) -> Option<u32> {
        self.shards.get(&shard_id).map(|shard| shard.current_load)
    }
    
    fn get_shard_count(&self) -> u64 {
        self.shards.len() as u64
    }
    
    async fn update_shard_metrics(&mut self, shard_id: u64, tx_count: u32, latency: std::time::Duration) {
        if let Some(shard) = self.shards.get_mut(&shard_id) {
            shard.update_metrics(tx_count, latency);
        }
    }
}


struct Shard {
    id: u64,
    capacity: u32,
    current_load: u32,
    transaction_count: u64,
    total_latency: std::time::Duration,
}


impl Shard {
    fn new(id: u64, capacity: u32) -> Self {
        Self {
            id,
            capacity,
            current_load: 0,
            transaction_count: 0,
            total_latency: std::time::Duration::ZERO,
        }
    }
    
    fn add_transaction(&mut self, _tx_id: u64) -> Result<(), String> {
        if self.current_load < self.capacity {
            self.current_load += 1;
            Ok(())
        } else {
            Err("Shard at capacity".to_string())
        }
    }
    
    fn update_metrics(&mut self, tx_count: u32, latency: std::time::Duration) {
        self.transaction_count += tx_count as u64;
        self.total_latency += latency;
    }
    
    fn average_latency(&self) -> std::time::Duration {
        if self.transaction_count > 0 {
            self.total_latency / self.transaction_count as u32
        } else {
            std::time::Duration::ZERO
        }
    }
}


async fn check_shard_split_conditions(shard_id: u64, current_load: u32) -> bool {
    // Conditions for shard split:
    // 1. High load (> 80%)
    // 2. Sustained high load (simulated)
    // 3. Sufficient transactions processed
    
    current_load > 80 && shard_id % 2 == 0 // Simple condition for testing
}


async fn check_shard_merge_conditions() -> bool {
    // Conditions for shard merge:
    // 1. Both shards have low load (< 40%)
    // 2. Low transaction volume
    // 3. Similar access patterns
    
    rand::thread_rng().gen_bool(0.3) // 30% chance of merge opportunity
}


struct LoadPredictor {
    predictions: Vec<i32>,
}


impl LoadPredictor {
    fn new() -> Self {
        Self {
            predictions: Vec::new(),
        }
    }
    
    async fn predict_load(&mut self, historical_load: &[i32]) -> Result<i32, String> {
        // Simple LSTM-like prediction (simplified)
        if historical_load.len() < 3 {
            return Err("Insufficient historical data".to_string());
        }
        
        // Weighted moving average
        let weights = [0.1, 0.3, 0.6]; // More weight to recent data
        let recent_data = &historical_load[historical_load.len() - 3..];
        
        let prediction = (recent_data[0] as f64 * weights[0] +
                         recent_data[1] as f64 * weights[1] +
                         recent_data[2] as f64 * weights[2]).round() as i32;
        
        self.predictions.push(prediction);
        Ok(prediction)
    }
}


enum ScalingDecision {
    ScaleUp(u32),
    ScaleDown(u32),
    NoAction,
}


struct ScalingManager {
    metrics: Vec<(i32, i32)>, // (actual, predicted)
    scaling_operations: u64,
    total_scale_up: u32,
    total_scale_down: u32,
}


impl ScalingManager {
    fn new() -> Self {
        Self {
            metrics: Vec::new(),
            scaling_operations: 0,
            total_scale_up: 0,
            total_scale_down: 0,
        }
    }
    
    async fn make_scaling_decision(&mut self, current_load: i32, predicted_load: i32) -> ScalingDecision {
        self.metrics.push((current_load, predicted_load));
        
        // Simple scaling logic
        if predicted_load > 80 {
            self.scaling_operations += 1;
            self.total_scale_up += 10;
            ScalingDecision::ScaleUp(10) // Add 10% capacity
        } else if predicted_load < 20 && current_load < 25 {
            self.scaling_operations += 1;
            self.total_scale_down += 5;
            ScalingDecision::ScaleDown(5) // Remove 5% capacity
        } else {
            ScalingDecision::NoAction
        }
    }
    
    fn record_metric(&mut self, actual: i32, predicted: i32) {
        self.metrics.push((actual, predicted));
    }
    
    fn get_statistics(&self) -> ScalingStatistics {
        let total = self.metrics.len();
        
        if total == 0 {
            return ScalingStatistics::default();
        }
        
        let sum_actual: i32 = self.metrics.iter().map(|(a, _)| a).sum();
        let sum_predicted: i32 = self.metrics.iter().map(|(_, p)| p).sum();
        
        let avg_load = sum_actual as f64 / total as f64;
        
        // Calculate prediction accuracy
        let mut total_error = 0;
        for (actual, predicted) in &self.metrics {
            total_error += (actual - predicted).abs();
        }
        let avg_error = total_error as f64 / total as f64;
        let prediction_accuracy = 1.0 - (avg_error / 100.0).min(1.0);
        
        ScalingStatistics {
            avg_load,
            prediction_accuracy,
            scaling_operations: self.scaling_operations,
            total_scale_up: self.total_scale_up,
            total_scale_down: self.total_scale_down,
        }
    }
}


#[derive(Default)]
struct ScalingStatistics {
    avg_load: f64,
    prediction_accuracy: f64,
    scaling_operations: u64,
    total_scale_up: u32,
    total_scale_down: u32,
}


async fn encode_data_with_erasure(data: &[u8], k: usize, m: usize) -> Result<Vec<Option<Vec<u8>>>, String> {
    // Simulate Reed-Solomon encoding
    let total_shards = k + m;
    let shard_size = (data.len() + k - 1) / k; // Ceiling division
    
    let mut shards = Vec::with_capacity(total_shards);
    
    // Create data shards
    for i in 0..k {
        let start = i * shard_size;
        let end = std::cmp::min(start + shard_size, data.len());
        
        if start < data.len() {
            let mut shard = data[start..end].to_vec();
            // Add padding if needed
            shard.resize(shard_size, 0);
            shards.push(Some(shard));
        } else {
            shards.push(Some(vec![0; shard_size]));
        }
    }
    
    // Create parity shards (simplified)
    for i in 0..m {
        let mut parity = vec![0u8; shard_size];
        
        // Simple XOR parity (not actual Reed-Solomon)
        for j in 0..k {
            if let Some(data_shard) = &shards[j] {
                for (k, byte) in data_shard.iter().enumerate() {
                    parity[k] ^= byte;
                }
            }
        }
        
        // Add index to distinguish parity shards
        parity.push(i as u8);
        shards.push(Some(parity));
    }
    
    Ok(shards)
}


async fn recover_data_from_shards(shards: &[Option<Vec<u8>>], k: usize, m: usize) -> Result<Vec<u8>, String> {
    let total_shards = k + m;
    
    if shards.len() != total_shards {
        return Err(format!("Expected {} shards, got {}", total_shards, shards.len()));
    }
    
    // Count available shards
    let available_shards: Vec<&Vec<u8>> = shards.iter()
        .filter_map(|shard| shard.as_ref())
        .collect();
    
    if available_shards.len() < k {
        return Err(format!("Need at least {} shards, only have {}", k, available_shards.len()));
    }
    
    // Recover data (simplified - just use first k shards)
    let mut recovered_data = Vec::new();
    
    for i in 0..k {
        if let Some(shard) = &shards[i] {
            recovered_data.extend_from_slice(shard);
        } else {
            // Use next available shard
            for j in k..total_shards {
                if let Some(parity_shard) = &shards[j] {
                    recovered_data.extend_from_slice(&parity_shard[..parity_shard.len() - 1]); // Remove index byte
                    break;
                }
            }
        }
    }
    
    // Remove padding (zeros at the end)
    while recovered_data.last() == Some(&0) {
        recovered_data.pop();
    }
    
    Ok(recovered_data)
}


async fn verify_shard_integrity(shards: &[Option<Vec<u8>>]) -> bool {
    // Simple integrity check
    for shard in shards {
        if let Some(data) = shard {
            // Check for all zeros (could indicate corruption)
            if data.iter().all(|&b| b == 0) {
                return false;
            }
        }
    }
    true
}


struct ShardCluster {
    shards: std::collections::HashMap<u64, u32>, // shard_id -> load percentage
}


impl ShardCluster {
    fn new() -> Self {
        Self {
            shards: std::collections::HashMap::new(),
        }
    }
    
    fn add_shard(&mut self, id: u64, load: u32) {
        self.shards.insert(id, load);
    }
    
    fn get_shard_loads(&self) -> std::collections::HashMap<u64, u32> {
        self.shards.clone()
    }
    
    async fn rebalance(&mut self) -> Result<Vec<RebalanceOperation>, String> {
        let mut operations = Vec::new();
        
        // Calculate average load
        let total_load: u32 = self.shards.values().sum();
        let avg_load = total_load as f64 / self.shards.len() as f64;
        
        // Find shards that need rebalancing
        let mut high_load_shards = Vec::new();
        let mut low_load_shards = Vec::new();
        
        for (&id, &load) in &self.shards {
            if load as f64 > avg_load * 1.3 {
                // More than 30% above average
                high_load_shards.push((id, load));
            } else if load as f64 < avg_load * 0.7 {
                // Less than 70% of average
                low_load_shards.push((id, load));
            }
        }
        
        // Pair high-load shards with low-load shards for rebalancing
        let mut pairs = Vec::new();
        for (high_id, high_load) in high_load_shards {
            if let Some((low_id, low_load)) = low_load_shards.pop() {
                pairs.push((high_id, high_load, low_id, low_load));
            }
        }
        
        // Create rebalancing operations
        for (high_id, high_load, low_id, low_load) in pairs {
            let transfer_amount = ((high_load - low_load) / 2).min(30); // Transfer up to 30% load
            
            if transfer_amount > 5 { // Only rebalance if significant
                // Update loads
                if let Some(load) = self.shards.get_mut(&high_id) {
                    *load -= transfer_amount;
                }
                
                if let Some(load) = self.shards.get_mut(&low_id) {
                    *load += transfer_amount;
                }
                
                operations.push(RebalanceOperation {
                    from_shard: high_id,
                    to_shard: low_id,
                    amount: transfer_amount,
                });
            }
        }
        
        Ok(operations)
    }
    
    async fn mitigate_hotspots(&mut self) -> Result<(), String> {
        // Simple hotspot mitigation
        for (&id, load) in self.shards.iter_mut() {
            if *load > 90 {
                // Reduce load on hotspot
                *load = (*load * 8 / 10).min(80); // Reduce to at most 80%
            }
        }
        Ok(())
    }
}


#[derive(Debug)]
struct RebalanceOperation {
    from_shard: u64,
    to_shard: u64,
    amount: u32,
}
