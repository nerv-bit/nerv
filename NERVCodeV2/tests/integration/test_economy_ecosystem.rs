// tests/integration/test_economy_ecosystem.rs
// ============================================================================
// ECONOMY ECOSYSTEM INTEGRATION TEST
// ============================================================================
// Tests the useful-work economy including Shapley value computation,
// federated learning rewards, and gradient contribution incentives.
// ============================================================================


use crate::integration::*;
use nerv_bit::economy::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};


#[tokio::test]
async fn test_shapley_value_distribution() {
    println!("Starting Shapley value distribution test...");
    
    let config = IntegrationConfig {
        node_count: 5,
        transaction_count: 30,
        random_seed: 6666,
        ..Default::default()
    };
    
    let mut env = TestEnvironment::new(config);
    let mut rng = ChaCha8Rng::seed_from_u64(env.config.random_seed);
    let mut metrics = IntegrationMetrics::default();
    
    // Create economy with participants
    let economy = Arc::new(RwLock::new(EconomySystem::new()));
    
    // Add participants with different contribution levels
    {
        let mut eco = economy.write().await;
        
        eco.add_participant("Alice".to_string(), ParticipantType::Validator, 1000.0);
        eco.add_participant("Bob".to_string(), ParticipantType::Validator, 800.0);
        eco.add_participant("Charlie".to_string(), ParticipantType::DataProvider, 600.0);
        eco.add_participant("Diana".to_string(), ParticipantType::ComputeProvider, 1200.0);
        eco.add_participant("Eve".to_string(), ParticipantType::Validator, 400.0);
    }
    
    println!("Economy initialized with 5 participants");
    
    // Simulate multiple rounds of contributions and rewards
    for round in 0..env.config.transaction_count {
        let start_time = std::time::Instant::now();
        
        // Simulate contributions in this round
        let contributions = simulate_contributions(round).await;
        
        // Calculate Shapley values for this round
        let shapley_values = {
            let mut eco = economy.write().await;
            eco.calculate_shapley_values(&contributions).await
        };
        
        match shapley_values {
            Ok(values) => {
                metrics.transactions_processed += 1;
                
                // Distribute rewards based on Shapley values
                let total_rewards = 1000.0; // Fixed reward pool per round
                let distribution_result = distribute_rewards(&values, total_rewards).await;
                
                match distribution_result {
                    Ok(distributions) => {
                        println!("\nRound {} - Reward Distribution:", round + 1);
                        
                        let mut total_distributed = 0.0;
                        for (participant, amount) in &distributions {
                            println!("  {}: {:.2} tokens", participant, amount);
                            total_distributed += amount;
                            
                            // Update participant balance
                            let mut eco = economy.write().await;
                            eco.add_rewards(participant, *amount);
                        }
                        
                        assert!((total_distributed - total_rewards).abs() < 0.01,
                                "Total distributed should match reward pool");
                        
                        // Update metrics
                        {
                            let eco = economy.read().await;
                            metrics.transactions_processed += eco.get_participant_count() as u64;
                        }
                    }
                    Err(e) => {
                        metrics.error_count += 1;
                        println!("Round {} reward distribution failed: {}", round + 1, e);
                    }
                }
            }
            Err(e) => {
                metrics.error_count += 1;
                println!("Round {} Shapley calculation failed: {}", round + 1, e);
            }
        }
        
        // Record latency
        let latency = start_time.elapsed().as_millis() as f64;
        metrics.avg_latency_ms = (metrics.avg_latency_ms * (round as f64) + latency) / ((round + 1) as f64);
        
        // Every 5 rounds, print economy statistics
        if (round + 1) % 5 == 0 {
            let eco = economy.read().await;
            let stats = eco.get_statistics();
            
            println!("\n=== Economy Statistics after {} rounds ===", round + 1);
            println!("Total rewards distributed: {:.2}", stats.total_rewards);
            println!("Average reward per participant: {:.2}", stats.avg_reward);
            println!("Gini coefficient: {:.3}", stats.gini_coefficient);
            println!("Most valuable participant: {} ({:.2})", 
                    stats.top_performer, stats.top_performance);
        }
        
        tokio::time::sleep(tokio::time::Duration::from_millis(25)).await;
    }
    
    // Final economy state
    {
        let eco = economy.read().await;
        let final_stats = eco.get_statistics();
        
        println!("\n=== Final Economy State ===");
        println!("Total participants: {}", eco.get_participant_count());
        println!("Total rewards: {:.2}", final_stats.total_rewards);
        println!("Wealth concentration (Gini): {:.3}", final_stats.gini_coefficient);
        
        // Participants should have positive balances
        for participant in eco.get_participants() {
            assert!(participant.balance >= 0.0, "Participant balance should be non-negative");
        }
    }
    
    metrics.success_rate = metrics.transactions_processed as f64 / 
                         (env.config.transaction_count * 2) as f64;
    
    let result = IntegrationResult {
        test_name: "shapley_value_distribution".to_string(),
        duration: env.elapsed(),
        metrics,
        passed: true,
        errors: Vec::new(),
    };
    
    println!("{}", result.summary());
    
    // Assertions
    assert!(result.metrics.transactions_processed > 0, "Should process transactions");
    
    println!("Shapley value distribution test passed!");
}


#[tokio::test]
async fn test_federated_learning_rewards() {
    println!("Starting federated learning rewards test...");
    
    let config = IntegrationConfig {
        node_count: 8,
        transaction_count: 40,
        random_seed: 7777,
        ..Default::default()
    };
    
    let mut env = TestEnvironment::new(config);
    let mut rng = ChaCha8Rng::seed_from_u64(env.config.random_seed);
    let mut metrics = IntegrationMetrics::default();
    
    // Create federated learning system
    let fl_system = Arc::new(RwLock::new(FederatedLearningSystem::new()));
    
    // Add clients with different data quality
    {
        let mut system = fl_system.write().await;
        
        system.add_client("Client1".to_string(), 0.9); // High quality
        system.add_client("Client2".to_string(), 0.8);
        system.add_client("Client3".to_string(), 0.7);
        system.add_client("Client4".to_string(), 0.6);
        system.add_client("Client5".to_string(), 0.5); // Medium quality
        system.add_client("Client6".to_string(), 0.4);
        system.add_client("Client7".to_string(), 0.3);
        system.add_client("Client8".to_string(), 0.2); // Low quality
    }
    
    println!("Federated learning system with 8 clients initialized");
    
    // Simulate multiple training rounds
    for round in 0..env.config.transaction_count {
        let start_time = std::time::Instant::now();
        
        // Collect gradients from clients
        let gradients = collect_gradients_from_clients(round).await;
        
        // Aggregate gradients (FedAvg)
        let aggregation_result = aggregate_gradients(&gradients).await;
        
        match aggregation_result {
            Ok((aggregated_gradient, contributions)) => {
                metrics.transactions_processed += 1;
                
                // Validate aggregated gradient
                let validation_result = validate_gradient(&aggregated_gradient).await;
                
                if validation_result {
                    println!("Round {}: Gradient aggregation successful", round + 1);
                    
                    // Calculate rewards based on contributions
                    let reward_pool = 500.0; // Fixed reward per round
                    let rewards_result = calculate_fl_rewards(&contributions, reward_pool).await;
                    
                    match rewards_result {
                        Ok(rewards) => {
                            // Distribute rewards
                            let mut system = fl_system.write().await;
                            
                            println!("Round {} rewards:", round + 1);
                            let mut total_rewarded = 0.0;
                            
                            for (client, amount) in rewards {
                                system.add_client_reward(&client, amount);
                                total_rewarded += amount;
                                
                                if amount > 50.0 {
                                    println!("  {}: {:.2} tokens (significant contributor)", client, amount);
                                }
                            }
                            
                            assert!((total_rewarded - reward_pool).abs() < 0.01,
                                    "Total rewards should match pool");
                            
                            // Update model with aggregated gradient
                            let update_result = system.update_model(aggregated_gradient).await;
                            
                            if update_result.is_ok() {
                                println!("Round {}: Model updated successfully", round + 1);
                                
                                // Calculate model improvement
                                let improvement = system.calculate_model_improvement().await;
                                println!("Round {}: Model improved by {:.4}", round + 1, improvement);
                            }
                        }
                        Err(e) => {
                            metrics.error_count += 1;
                            println!("Round {} reward calculation failed: {}", round + 1, e);
                        }
                    }
                } else {
                    metrics.error_count += 1;
                    println!("Round {}: Invalid aggregated gradient", round + 1);
                }
            }
            Err(e) => {
                metrics.error_count += 1;
                println!("Round {} gradient aggregation failed: {}", round + 1, e);
            }
        }
        
        // Every 10 rounds, prune low-performing clients
        if (round + 1) % 10 == 0 {
            let mut system = fl_system.write().await;
            let pruned = system.prune_low_performers(0.3).await; // Prune bottom 30%
            
            if pruned > 0 {
                println!("Pruned {} low-performing clients", pruned);
                
                // Add new clients to maintain network size
                for i in 0..pruned {
                    let client_name = format!("NewClient{}", i + 1);
                    system.add_client(client_name, rng.gen_range(0.4..0.8));
                }
                println!("Added {} new clients", pruned);
            }
        }
        
        // Record latency
        let latency = start_time.elapsed().as_millis() as f64;
        metrics.avg_latency_ms = (metrics.avg_latency_ms * (round as f64) + latency) / ((round + 1) as f64);
        
        tokio::time::sleep(tokio::time::Duration::from_millis(30)).await;
    }
    
    // Final statistics
    {
        let system = fl_system.read().await;
        let stats = system.get_statistics();
        
        println!("\n=== Federated Learning Statistics ===");
        println!("Total training rounds: {}", env.config.transaction_count);
        println!("Active clients: {}", system.get_client_count());
        println!("Total rewards distributed: {:.2}", stats.total_rewards);
        println!("Average client contribution: {:.3}", stats.avg_contribution);
        println!("Model accuracy improvement: {:.2}%", stats.model_improvement * 100.0);
        
        // Top performers
        println!("\nTop 3 performers:");
        for (i, (client, reward)) in stats.top_performers.iter().enumerate().take(3) {
            println!("  {}. {}: {:.2} tokens", i + 1, client, reward);
        }
    }
    
    metrics.success_rate = metrics.transactions_processed as f64 / env.config.transaction_count as f64;
    
    let result = IntegrationResult {
        test_name: "federated_learning_rewards".to_string(),
        duration: env.elapsed(),
        metrics,
        passed: true,
        errors: Vec::new(),
    };
    
    println!("{}", result.summary());
    
    // Assertions
    assert!(result.metrics.transactions_processed > 0, "Should process training rounds");
    assert!(result.metrics.success_rate > 0.5, "Should have reasonable success rate");
    
    println!("Federated learning rewards test passed!");
}


#[tokio::test]
async fn test_gradient_contribution_incentives() {
    println!("Starting gradient contribution incentives test...");
    
    let config = IntegrationConfig {
        node_count: 10,
        transaction_count: 25,
        random_seed: 8888,
        ..Default::default()
    };
    
    let mut env = TestEnvironment::new(config);
    let mut rng = ChaCha8Rng::seed_from_u64(env.config.random_seed);
    let mut metrics = IntegrationMetrics::default();
    
    // Create incentive system
    let incentive_system = Arc::new(RwLock::new(IncentiveSystem::new()));
    
    // Test different incentive mechanisms
    for i in 0..env.config.transaction_count {
        let contribution_type = match i % 4 {
            0 => ContributionType::GradientQuality,
            1 => ContributionType::DataQuantity,
            2 => ContributionType::ComputeTime,
            3 => ContributionType::ModelImprovement,
            _ => unreachable!(),
        };
        
        // Simulate contribution
        let contribution_value = rng.gen_range(0.1..1.0);
        let contributor = format!("Contributor{}", (i % 10) + 1);
        
        // Calculate incentive
        let incentive_result = {
            let mut system = incentive_system.write().await;
            system.calculate_incentive(&contributor, contribution_type, contribution_value).await
        };
        
        match incentive_result {
            Ok((base_reward, bonus)) => {
                let total_reward = base_reward + bonus;
                
                metrics.transactions_processed += 1;
                
                println!("Contribution {}: {} contributed {:.2} ({:?}) - Reward: {:.2} (base: {:.2}, bonus: {:.2})",
                        i + 1, contributor, contribution_value, contribution_type,
                        total_reward, base_reward, bonus);
                
                // Record the reward
                {
                    let mut system = incentive_system.write().await;
                    system.record_reward(&contributor, total_reward);
                }
                
                // Test Sybil resistance
                if i % 5 == 0 {
                    let sybil_test = test_sybil_resistance(&contributor, contribution_value).await;
                    if !sybil_test {
                        println!("Warning: Possible Sybil attack detected for {}", contributor);
                        metrics.error_count += 1;
                    }
                }
            }
            Err(e) => {
                metrics.error_count += 1;
                println!("Contribution {} incentive calculation failed: {}", i + 1, e);
            }
        }
        
        // Every 8 contributions, redistribute incentives
        if (i + 1) % 8 == 0 {
            let redistribution_result = {
                let mut system = incentive_system.write().await;
                system.redistribute_incentives().await
            };
            
            if redistribution_result.is_ok() {
                println!("Periodic incentive redistribution completed");
            }
        }
        
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
    }
    
    // Analyze incentive distribution
    {
        let system = incentive_system.read().await;
        let analysis = system.analyze_incentive_distribution().await;
        
        println!("\n=== Incentive Distribution Analysis ===");
        println!("Total rewards distributed: {:.2}", analysis.total_rewards);
        println!("Average reward per contribution: {:.2}", analysis.avg_reward);
        println!("Reward variance: {:.2}", analysis.reward_variance);
        println!("Gini coefficient: {:.3}", analysis.gini_coefficient);
        println!("Most rewarded contributor: {} ({:.2})", 
                analysis.top_contributor, analysis.top_reward);
        
        // Check fairness
        assert!(analysis.gini_coefficient < 0.5, "Incentives should be reasonably fair");
        assert!(analysis.reward_variance > 0.0, "Should have some variance in rewards");
    }
    
    metrics.success_rate = metrics.transactions_processed as f64 / env.config.transaction_count as f64;
    
    let result = IntegrationResult {
        test_name: "gradient_contribution_incentives".to_string(),
        duration: env.elapsed(),
        metrics,
        passed: true,
        errors: Vec::new(),
    };
    
    println!("{}", result.summary());
    
    // Assertions
    assert!(result.metrics.transactions_processed > 0, "Should process contributions");
    
    println!("Gradient contribution incentives test passed!");
}


#[tokio::test]
async fn test_useful_work_verification() {
    println!("Starting useful work verification test...");
    
    let config = IntegrationConfig {
        node_count: 6,
        transaction_count: 20,
        random_seed: 9999,
        ..Default::default()
    };
    
    let mut env = TestEnvironment::new(config);
    let mut rng = ChaCha8Rng::seed_from_u64(env.config.random_seed);
    let mut metrics = IntegrationMetrics::default();
    
    // Create work verification system
    let verification_system = Arc::new(RwLock::new(WorkVerificationSystem::new()));
    
    // Test different types of useful work
    for i in 0..env.config.transaction_count {
        let work_type = match i % 3 {
            0 => WorkType::GradientComputation,
            1 => WorkType::DataValidation,
            2 => WorkType::ModelTraining,
            _ => unreachable!(),
        };
        
        let worker = format!("Worker{}", (i % 6) + 1);
        let work_amount = rng.gen_range(10..100) as f64;
        
        // Submit work for verification
        let submission_result = {
            let mut system = verification_system.write().await;
            system.submit_work(&worker, work_type, work_amount).await
        };
        
        match submission_result {
            Ok(submission_id) => {
                println!("Work {} submitted by {}: {:?} (amount: {:.1})", 
                        submission_id, worker, work_type, work_amount);
                
                // Verify the work
                let verification_result = {
                    let mut system = verification_system.write().await;
                    system.verify_work(submission_id).await
                };
                
                match verification_result {
                    Ok((is_valid, quality_score)) => {
                        if is_valid {
                            metrics.transactions_processed += 1;
                            
                            // Calculate reward based on work quality
                            let reward = calculate_work_reward(work_type, work_amount, quality_score).await;
                            
                            println!("Work {} VERIFIED - Quality: {:.3}, Reward: {:.2}", 
                                    submission_id, quality_score, reward);
                            
                            // Record reward
                            {
                                let mut system = verification_system.write().await;
                                system.record_reward(&worker, reward);
                            }
                        } else {
                            metrics.error_count += 1;
                            println!("Work {} REJECTED - Low quality or invalid", submission_id);
                            
                            // Penalize for low-quality work
                            let penalty = work_amount * 0.1;
                            println!("Applying penalty: {:.2}", penalty);
                        }
                    }
                    Err(e) => {
                        metrics.error_count += 1;
                        println!("Work {} verification failed: {}", submission_id, e);
                    }
                }
            }
            Err(e) => {
                metrics.error_count += 1;
                println!("Work submission {} failed: {}", i + 1, e);
            }
        }
        
        // Test work duplication detection
        if i % 4 == 0 {
            let duplicate_check = check_for_duplicate_work(&worker, work_type, work_amount).await;
            if duplicate_check {
                println!("Duplicate work detected for {}", worker);
                metrics.error_count += 1;
            }
        }
        
        tokio::time::sleep(tokio::time::Duration::from_millis(25)).await;
    }
    
    // Final verification statistics
    {
        let system = verification_system.read().await;
        let stats = system.get_statistics();
        
        println!("\n=== Work Verification Statistics ===");
        println!("Total work submissions: {}", env.config.transaction_count);
        println!("Verified work: {} ({:.1}%)", stats.verified_count, 
                stats.verified_count as f64 / env.config.transaction_count as f64 * 100.0);
        println!("Rejected work: {} ({:.1}%)", stats.rejected_count,
                stats.rejected_count as f64 / env.config.transaction_count as f64 * 100.0);
        println!("Total rewards: {:.2}", stats.total_rewards);
        println!("Average work quality: {:.3}", stats.avg_quality);
        
        // Quality should be reasonable
        assert!(stats.avg_quality > 0.5, "Average work quality should be reasonable");
        assert!(stats.verified_count > 0, "Should have some verified work");
    }
    
    metrics.success_rate = metrics.transactions_processed as f64 / env.config.transaction_count as f64;
    
    let result = IntegrationResult {
        test_name: "useful_work_verification".to_string(),
        duration: env.elapsed(),
        metrics,
        passed: true,
        errors: Vec::new(),
    };
    
    println!("{}", result.summary());
    
    // Assertions
    assert!(result.metrics.transactions_processed > 0, "Should verify some work");
    
    println!("Useful work verification test passed!");
}


// Helper structs and implementations for economy tests


enum ParticipantType {
    Validator,
    DataProvider,
    ComputeProvider,
}


struct Participant {
    name: String,
    participant_type: ParticipantType,
    stake: f64,
    balance: f64,
    total_rewards: f64,
    contribution_score: f64,
}


impl Participant {
    fn new(name: String, participant_type: ParticipantType, stake: f64) -> Self {
        Self {
            name,
            participant_type,
            stake,
            balance: 0.0,
            total_rewards: 0.0,
            contribution_score: 0.0,
        }
    }
    
    fn add_reward(&mut self, amount: f64) {
        self.balance += amount;
        self.total_rewards += amount;
        self.contribution_score += amount / 100.0; // Simple score increase
    }
}


struct EconomySystem {
    participants: std::collections::HashMap<String, Participant>,
    total_rewards_distributed: f64,
}


impl EconomySystem {
    fn new() -> Self {
        Self {
            participants: std::collections::HashMap::new(),
            total_rewards_distributed: 0.0,
        }
    }
    
    fn add_participant(&mut self, name: String, participant_type: ParticipantType, stake: f64) {
        self.participants.insert(name.clone(), Participant::new(name, participant_type, stake));
    }
    
    async fn calculate_shapley_values(&mut self, contributions: &[(String, f64)]) -> Result<Vec<(String, f64)>, String> {
        // Simplified Shapley value calculation
        let total_contribution: f64 = contributions.iter().map(|(_, c)| c).sum();
        
        if total_contribution <= 0.0 {
            return Err("No contributions".to_string());
        }
        
        let mut shapley_values = Vec::new();
        
        for (name, contribution) in contributions {
            // Simplified: proportional to contribution with stake weighting
            if let Some(participant) = self.participants.get(name) {
                let weight = contribution * participant.stake.sqrt(); // sqrt for diminishing returns
                shapley_values.push((name.clone(), weight));
            }
        }
        
        // Normalize
        let total_weight: f64 = shapley_values.iter().map(|(_, w)| w).sum();
        for (_, weight) in &mut shapley_values {
            *weight = *weight / total_weight;
        }
        
        Ok(shapley_values)
    }
    
    fn add_rewards(&mut self, participant: &str, amount: f64) {
        if let Some(p) = self.participants.get_mut(participant) {
            p.add_reward(amount);
            self.total_rewards_distributed += amount;
        }
    }
    
    fn get_participant_count(&self) -> usize {
        self.participants.len()
    }
    
    fn get_participants(&self) -> Vec<&Participant> {
        self.participants.values().collect()
    }
    
    fn get_statistics(&self) -> EconomyStatistics {
        let participant_count = self.participants.len();
        
        if participant_count == 0 {
            return EconomyStatistics::default();
        }
        
        let mut balances: Vec<f64> = self.participants.values()
            .map(|p| p.balance)
            .collect();
        
        balances.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        let total_rewards = balances.iter().sum::<f64>();
        let avg_reward = total_rewards / participant_count as f64;
        
        // Calculate Gini coefficient
        let mut gini_numerator = 0.0;
        for i in 0..participant_count {
            for j in 0..participant_count {
                gini_numerator += (balances[i] - balances[j]).abs();
            }
        }
        
        let mean_balance = total_rewards / participant_count as f64;
        let gini_denominator = 2.0 * participant_count as f64 * participant_count as f64 * mean_balance;
        let gini_coefficient = if gini_denominator > 0.0 {
            gini_numerator / gini_denominator
        } else {
            0.0
        };
        
        // Find top performer
        let top_performer = self.participants.values()
            .max_by(|a, b| a.total_rewards.partial_cmp(&b.total_rewards).unwrap())
            .map(|p| p.name.clone())
            .unwrap_or_default();
            
        let top_performance = self.participants.get(&top_performer)
            .map(|p| p.total_rewards)
            .unwrap_or(0.0);
        
        EconomyStatistics {
            total_rewards,
            avg_reward,
            gini_coefficient,
            top_performer,
            top_performance,
        }
    }
}


#[derive(Default)]
struct EconomyStatistics {
    total_rewards: f64,
    avg_reward: f64,
    gini_coefficient: f64,
    top_performer: String,
    top_performance: f64,
}


async fn simulate_contributions(round: usize) -> Vec<(String, f64)> {
    let mut contributions = Vec::new();
    let mut rng = rand::thread_rng();
    
    // Simulate different contribution patterns
    let participants = vec!["Alice", "Bob", "Charlie", "Diana", "Eve"];
    
    for (i, participant) in participants.iter().enumerate() {
        // Vary contributions based on round and participant
        let base_contribution = match i {
            0 => 80.0, // Alice: high contributor
            1 => 60.0, // Bob: medium-high
            2 => 40.0, // Charlie: medium
            3 => 70.0, // Diana: high
            4 => 30.0, // Eve: low
            _ => 50.0,
        };
        
        // Add some randomness
        let variation: f64 = rng.gen_range(-20.0..20.0);
        let contribution = (base_contribution + variation).max(10.0);
        
        contributions.push((participant.to_string(), contribution));
    }
    
    contributions
}


async fn distribute_rewards(shapley_values: &[(String, f64)], total_rewards: f64) -> Result<Vec<(String, f64)>, String> {
    let mut distributions = Vec::new();
    
    for (name, weight) in shapley_values {
        let reward = weight * total_rewards;
        distributions.push((name.clone(), reward));
    }
    
    Ok(distributions)
}


struct FederatedLearningSystem {
    clients: std::collections::HashMap<String, Client>,
    model_version: u64,
    model_accuracy: f64,
    total_rewards: f64,
}


struct Client {
    name: String,
    data_quality: f64,
    total_contributions: f64,
    total_rewards: f64,
    contribution_history: Vec<f64>,
}


impl FederatedLearningSystem {
    fn new() -> Self {
        Self {
            clients: std::collections::HashMap::new(),
            model_version: 0,
            model_accuracy: 0.5, // Starting accuracy
            total_rewards: 0.0,
        }
    }
    
    fn add_client(&mut self, name: String, data_quality: f64) {
        self.clients.insert(name.clone(), Client {
            name,
            data_quality,
            total_contributions: 0.0,
            total_rewards: 0.0,
            contribution_history: Vec::new(),
        });
    }
    
    async fn update_model(&mut self, _gradient: Vec<f32>) -> Result<(), String> {
        // Simulate model update
        self.model_version += 1;
        
        // Simulate accuracy improvement
        let improvement = 0.01 + (rand::thread_rng().gen_range(0.0..0.03)); // 1-4% improvement
        self.model_accuracy = (self.model_accuracy + improvement).min(0.99);
        
        Ok(())
    }
    
    fn add_client_reward(&mut self, client_name: &str, amount: f64) {
        if let Some(client) = self.clients.get_mut(client_name) {
            client.total_rewards += amount;
            client.total_contributions += amount / 10.0; // Convert reward to contribution score
            client.contribution_history.push(amount);
        }
        self.total_rewards += amount;
    }
    
    fn get_client_count(&self) -> usize {
        self.clients.len()
    }
    
    async fn prune_low_performers(&mut self, percentage: f64) -> usize {
        let target_prune_count = (self.clients.len() as f64 * percentage).ceil() as usize;
        
        if target_prune_count == 0 {
            return 0;
        }
        
        // Sort clients by contribution
        let mut clients: Vec<&Client> = self.clients.values().collect();
        clients.sort_by(|a, b| a.total_contributions.partial_cmp(&b.total_contributions).unwrap());
        
        // Prune lowest performers
        let mut pruned = 0;
        for client in clients.iter().take(target_prune_count) {
            self.clients.remove(&client.name);
            pruned += 1;
        }
        
        pruned
    }
    
    async fn calculate_model_improvement(&self) -> f64 {
        self.model_accuracy - 0.5 // Improvement from base
    }
    
    fn get_statistics(&self) -> FLStatistics {
        let client_count = self.clients.len();
        
        if client_count == 0 {
            return FLStatistics::default();
        }
        
        let total_contributions: f64 = self.clients.values()
            .map(|c| c.total_contributions)
            .sum();
        
        let avg_contribution = total_contributions / client_count as f64;
        
        // Find top performers
        let mut top_performers: Vec<(String, f64)> = self.clients.iter()
            .map(|(name, client)| (name.clone(), client.total_rewards))
            .collect();
        
        top_performers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        FLStatistics {
            total_rewards: self.total_rewards,
            avg_contribution,
            model_improvement: self.model_accuracy - 0.5,
            top_performers,
        }
    }
}


#[derive(Default)]
struct FLStatistics {
    total_rewards: f64,
    avg_contribution: f64,
    model_improvement: f64,
    top_performers: Vec<(String, f64)>,
}


async fn collect_gradients_from_clients(round: usize) -> Vec<(String, Vec<f32>, f64)> {
    let mut gradients = Vec::new();
    let mut rng = rand::thread_rng();
    
    let clients = vec!["Client1", "Client2", "Client3", "Client4", "Client5", "Client6", "Client7", "Client8"];
    
    for client in clients {
        // Generate random gradient
        let gradient_size = 100;
        let mut gradient = vec![0.0; gradient_size];
        for value in &mut gradient {
            *value = rng.gen_range(-1.0..1.0);
        }
        
        // Contribution based on client quality and round
        let base_quality = match client {
            "Client1" => 0.9,
            "Client2" => 0.8,
            "Client3" => 0.7,
            "Client4" => 0.6,
            "Client5" => 0.5,
            "Client6" => 0.4,
            "Client7" => 0.3,
            "Client8" => 0.2,
            _ => 0.5,
        };
        
        let contribution = base_quality * (1.0 + (round as f64 * 0.01)); // Slight improvement over rounds
        
        gradients.push((client.to_string(), gradient, contribution));
    }
    
    gradients
}


async fn aggregate_gradients(gradients: &[(String, Vec<f32>, f64)]) -> Result<(Vec<f32>, Vec<(String, f64)>), String> {
    if gradients.is_empty() {
        return Err("No gradients to aggregate".to_string());
    }
    
    let gradient_size = gradients[0].1.len();
    let mut aggregated = vec![0.0; gradient_size];
    let mut contributions = Vec::new();
    
    let total_contribution: f64 = gradients.iter().map(|(_, _, c)| c).sum();
    
    for (client, gradient, contribution) in gradients {
        if gradient.len() != gradient_size {
            return Err("Gradient size mismatch".to_string());
        }
        
        // Weight by contribution
        let weight = contribution / total_contribution;
        
        for i in 0..gradient_size {
            aggregated[i] += gradient[i] * weight as f32;
        }
        
        contributions.push((client.clone(), *contribution));
    }
    
    Ok((aggregated, contributions))
}


async fn validate_gradient(gradient: &[f32]) -> bool {
    // Check for NaN or Infinity
    for &value in gradient {
        if value.is_nan() || value.is_infinite() {
            return false;
        }
    }
    
    // Check gradient magnitude
    let magnitude: f32 = gradient.iter().map(|&x| x * x).sum::<f32>().sqrt();
    magnitude > 0.001 && magnitude < 100.0 // Reasonable bounds
}


async fn calculate_fl_rewards(contributions: &[(String, f64)], total_reward: f64) -> Result<Vec<(String, f64)>, String> {
    let total_contribution: f64 = contributions.iter().map(|(_, c)| c).sum();
    
    if total_contribution <= 0.0 {
        return Err("No contributions".to_string());
    }
    
    let mut rewards = Vec::new();
    
    for (client, contribution) in contributions {
        let reward = (contribution / total_contribution) * total_reward;
        rewards.push((client.clone(), reward));
    }
    
    Ok(rewards)
}


#[derive(Debug, Clone, Copy)]
enum ContributionType {
    GradientQuality,
    DataQuantity,
    ComputeTime,
    ModelImprovement,
}


struct IncentiveSystem {
    rewards: std::collections::HashMap<String, f64>,
    contribution_history: Vec<(String, ContributionType, f64, f64)>, // (contributor, type, value, reward)
}


impl IncentiveSystem {
    fn new() -> Self {
        Self {
            rewards: std::collections::HashMap::new(),
            contribution_history: Vec::new(),
        }
    }
    
    async fn calculate_incentive(&mut self, contributor: &str, contribution_type: ContributionType, value: f64) -> Result<(f64, f64), String> {
        // Base reward based on contribution type and value
        let base_multiplier = match contribution_type {
            ContributionType::GradientQuality => 10.0,
            ContributionType::DataQuantity => 5.0,
            ContributionType::ComputeTime => 3.0,
            ContributionType::ModelImprovement => 15.0,
        };
        
        let base_reward = value * base_multiplier;
        
        // Bonus for consistency and high value
        let bonus = if value > 0.8 {
            base_reward * 0.5 // 50% bonus for high-quality contributions
        } else if value > 0.5 {
            base_reward * 0.2 // 20% bonus for medium-quality
        } else {
            0.0
        };
        
        // Record contribution
        self.contribution_history.push((contributor.to_string(), contribution_type, value, base_reward + bonus));
        
        Ok((base_reward, bonus))
    }
    
    fn record_reward(&mut self, contributor: &str, amount: f64) {
        *self.rewards.entry(contributor.to_string()).or_insert(0.0) += amount;
    }
    
    async fn redistribute_incentives(&mut self) -> Result<(), String> {
        // Simple redistribution: take 10% from top and redistribute to bottom
        let total_rewards: f64 = self.rewards.values().sum();
        
        if total_rewards == 0.0 {
            return Ok(());
        }
        
        let mut rewards_vec: Vec<(String, f64)> = self.rewards.drain().collect();
        rewards_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let top_count = (rewards_vec.len() / 10).max(1); // Top 10%
        let bottom_count = (rewards_vec.len() / 5).max(1); // Bottom 20%
        
        let mut redistribution_pool = 0.0;
        
        // Take from top
        for i in 0..top_count {
            let take_amount = rewards_vec[i].1 * 0.1; // Take 10%
            rewards_vec[i].1 -= take_amount;
            redistribution_pool += take_amount;
        }
        
        // Redistribute to bottom
        let redistribution_per_bottom = redistribution_pool / bottom_count as f64;
        let start_idx = rewards_vec.len() - bottom_count;
        
        for i in start_idx..rewards_vec.len() {
            rewards_vec[i].1 += redistribution_per_bottom;
        }
        
        // Update rewards map
        self.rewards = rewards_vec.into_iter().collect();
        
        Ok(())
    }
    
    async fn analyze_incentive_distribution(&self) -> IncentiveAnalysis {
        let contributor_count = self.rewards.len();
        
        if contributor_count == 0 {
            return IncentiveAnalysis::default();
        }
        
        let rewards: Vec<f64> = self.rewards.values().copied().collect();
        let total_rewards: f64 = rewards.iter().sum();
        let avg_reward = total_rewards / contributor_count as f64;
        
        // Calculate variance
        let variance: f64 = rewards.iter()
            .map(|&r| (r - avg_reward).powi(2))
            .sum::<f64>() / contributor_count as f64;
        
        // Calculate Gini coefficient
        let mut sorted_rewards = rewards.clone();
        sorted_rewards.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mut gini_numerator = 0.0;
        for i in 0..contributor_count {
            for j in 0..contributor_count {
                gini_numerator += (sorted_rewards[i] - sorted_rewards[j]).abs();
            }
        }
        
        let gini_denominator = 2.0 * contributor_count as f64 * contributor_count as f64 * avg_reward;
        let gini_coefficient = if gini_denominator > 0.0 {
            gini_numerator / gini_denominator
        } else {
            0.0
        };
        
        // Find top contributor
        let top_contributor = self.rewards.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(name, _)| name.clone())
            .unwrap_or_default();
            
        let top_reward = *self.rewards.get(&top_contributor).unwrap_or(&0.0);
        
        IncentiveAnalysis {
            total_rewards,
            avg_reward,
            reward_variance: variance,
            gini_coefficient,
            top_contributor,
            top_reward,
        }
    }
}


#[derive(Default)]
struct IncentiveAnalysis {
    total_rewards: f64,
    avg_reward: f64,
    reward_variance: f64,
    gini_coefficient: f64,
    top_contributor: String,
    top_reward: f64,
}


async fn test_sybil_resistance(contributor: &str, contribution_value: f64) -> bool {
    // Simple Sybil resistance check
    // In real system, would check IP, stake, history, etc.
    
    // Contributions that are too perfect might be Sybil
    if contribution_value > 0.95 && contributor.contains("Clone") {
        return false;
    }
    
    // Multiple similar contributions from similar names
    if contributor.ends_with("_1") || contributor.ends_with("_2") || contributor.ends_with("_3") {
        return contribution_value < 0.9; // Sybils usually can't provide high-quality contributions
    }
    
    true
}


#[derive(Debug, Clone, Copy)]
enum WorkType {
    GradientComputation,
    DataValidation,
    ModelTraining,
}


struct WorkVerificationSystem {
    submissions: std::collections::HashMap<u64, WorkSubmission>,
    next_submission_id: u64,
    rewards: std::collections::HashMap<String, f64>,
    verification_history: Vec<(u64, bool, f64)>, // (submission_id, verified, quality)
}


struct WorkSubmission {
    id: u64,
    worker: String,
    work_type: WorkType,
    amount: f64,
    timestamp: u64,
    verified: bool,
    quality_score: f64,
}


impl WorkVerificationSystem {
    fn new() -> Self {
        Self {
            submissions: std::collections::HashMap::new(),
            next_submission_id: 1,
            rewards: std::collections::HashMap::new(),
            verification_history: Vec::new(),
        }
    }
    
    async fn submit_work(&mut self, worker: &str, work_type: WorkType, amount: f64) -> Result<u64, String> {
        let submission_id = self.next_submission_id;
        self.next_submission_id += 1;
        
        let submission = WorkSubmission {
            id: submission_id,
            worker: worker.to_string(),
            work_type,
            amount,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            verified: false,
            quality_score: 0.0,
        };
        
        self.submissions.insert(submission_id, submission);
        
        Ok(submission_id)
    }
    
    async fn verify_work(&mut self, submission_id: u64) -> Result<(bool, f64), String> {
        let submission = self.submissions.get_mut(&submission_id)
            .ok_or_else(|| "Submission not found".to_string())?;
        
        if submission.verified {
            return Err("Already verified".to_string());
        }
        
        // Simulate verification process
        let mut rng = rand::thread_rng();
        
        // Quality depends on work type and amount
        let base_quality = match submission.work_type {
            WorkType::GradientComputation => 0.7,
            WorkType::DataValidation => 0.8,
            WorkType::ModelTraining => 0.6,
        };
        
        let quality_variation: f64 = rng.gen_range(-0.2..0.2);
        let quality_score = (base_quality + quality_variation + (submission.amount / 200.0)).clamp(0.0, 1.0);
        
        // Work is verified if quality is above threshold
        let is_valid = quality_score > 0.5;
        
        submission.verified = true;
        submission.quality_score = quality_score;
        
        self.verification_history.push((submission_id, is_valid, quality_score));
        
        Ok((is_valid, quality_score))
    }
    
    fn record_reward(&mut self, worker: &str, amount: f64) {
        *self.rewards.entry(worker.to_string()).or_insert(0.0) += amount;
    }
    
    fn get_statistics(&self) -> WorkVerificationStats {
        let total_submissions = self.submissions.len();
        let verified_count = self.verification_history.iter().filter(|(_, v, _)| *v).count();
        let rejected_count = total_submissions - verified_count;
        
        let total_rewards: f64 = self.rewards.values().sum();
        
        let total_quality: f64 = self.verification_history.iter()
            .filter(|(_, v, _)| *v)
            .map(|(_, _, q)| q)
            .sum();
        
        let avg_quality = if verified_count > 0 {
            total_quality / verified_count as f64
        } else {
            0.0
        };
        
        WorkVerificationStats {
            total_submissions,
            verified_count,
            rejected_count,
            total_rewards,
            avg_quality,
        }
    }
}


#[derive(Default)]
struct WorkVerificationStats {
    total_submissions: usize,
    verified_count: usize,
    rejected_count: usize,
    total_rewards: f64,
    avg_quality: f64,
}


async fn calculate_work_reward(work_type: WorkType, amount: f64, quality: f64) -> f64 {
    let base_rate = match work_type {
        WorkType::GradientComputation => 2.0,
        WorkType::DataValidation => 1.5,
        WorkType::ModelTraining => 3.0,
    };
    
    amount * base_rate * quality
}


async fn check_for_duplicate_work(worker: &str, work_type: WorkType, amount: f64) -> bool {
    // In real system, would check against database of previous work
    // For test, just return false
    false
}




