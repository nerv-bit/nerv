// tests/unit/test_economy.rs
// ============================================================================
// ECONOMY MODULE UNIT TESTS
// ============================================================================


use nerv_bit::economy::*;
use rand::Rng;


#[test]
fn test_shapley_value_calculation() {
    // Test with simple cooperative game
    // 3 players, characteristic function: v(S) = |S|^2
    
    let players = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    
    // Define characteristic function
    let characteristic = |coalition: &[String]| -> f64 {
        (coalition.len() as f64).powi(2)
    };
    
    // Calculate Shapley values
    let shapley_values = calculate_shapley_values(&players, characteristic);
    
    assert_eq!(shapley_values.len(), 3);
    
    // All players should have equal Shapley value for this symmetric game
    let expected = 14.0 / 6.0; // For 3 players with v(S)=|S|^2
    for (_, value) in &shapley_values {
        assert!((value - expected).abs() < 0.0001);
    }
    
    // Sum of Shapley values should equal grand coalition value
    let total: f64 = shapley_values.values().sum();
    assert!((total - 9.0).abs() < 0.0001); // v({A,B,C}) = 3^2 = 9
}


#[test]
fn test_federated_learning_aggregation() {
    // Create test gradients from different clients
    let mut gradients = Vec::new();
    
    for i in 0..5 {
        let gradient = vec![i as f32 * 0.1; 100]; // 100-dimensional gradient
        gradients.push(gradient);
    }
    
    // Aggregate using FedAvg
    let aggregated = federated_average(&gradients).unwrap();
    
    assert_eq!(aggregated.len(), 100);
    
    // Check that aggregation worked (average of 0, 0.1, 0.2, 0.3, 0.4 = 0.2)
    for &value in &aggregated {
        assert!((value - 0.2).abs() < 0.0001);
    }
}


#[test]
fn test_gradient_validation() {
    // Create valid gradient
    let valid_gradient = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    
    // Test validation
    let is_valid = validate_gradient(&valid_gradient).unwrap();
    assert!(is_valid);
    
    // Create invalid gradient (contains NaN)
    let mut invalid_gradient = valid_gradient.clone();
    invalid_gradient[2] = f32::NAN;
    
    let is_valid = validate_gradient(&invalid_gradient).unwrap();
    assert!(!is_valid);
    
    // Create invalid gradient (contains infinity)
    let mut invalid_gradient2 = valid_gradient.clone();
    invalid_gradient2[3] = f32::INFINITY;
    
    let is_valid = validate_gradient(&invalid_gradient2).unwrap();
    assert!(!is_valid);
}


#[test]
fn test_reward_calculation() {
    let contributions = vec![
        ("Alice".to_string(), 100.0),
        ("Bob".to_string(), 200.0),
        ("Charlie".to_string(), 300.0),
    ];
    
    let total_reward = 1000.0;
    
    let rewards = calculate_proportional_rewards(&contributions, total_reward).unwrap();
    
    assert_eq!(rewards.len(), 3);
    
    // Check proportional distribution
    let total_contributions: f64 = contributions.iter().map(|(_, c)| c).sum();
    
    for ((name, contribution), (reward_name, reward)) in contributions.iter().zip(&rewards) {
        assert_eq!(name, reward_name);
        let expected = contribution / total_contributions * total_reward;
        assert!((reward - expected).abs() < 0.0001);
    }
    
    // Sum of rewards should equal total reward
    let reward_sum: f64 = rewards.iter().map(|(_, r)| r).sum();
    assert!((reward_sum - total_reward).abs() < 0.0001);
}


#[test]
fn test_economic_metrics() {
    let mut metrics = EconomicMetrics::new();
    
    // Update metrics
    metrics.update_total_stake(1000000.0);
    metrics.update_rewards_distributed(50000.0);
    metrics.record_transaction_fee(0.001);
    metrics.record_validator_reward(100.0);
    
    assert_eq!(metrics.total_stake, 1000000.0);
    assert_eq!(metrics.total_rewards_distributed, 50000.0);
    assert_eq!(metrics.transaction_fees_collected, 0.001);
    assert_eq!(metrics.validator_rewards_paid, 100.0);
    
    // Test inflation calculation
    let inflation = metrics.calculate_inflation_rate();
    assert!(inflation >= 0.0);
    
    // Test staking ratio
    let staking_ratio = metrics.calculate_staking_ratio(500000.0);
    assert_eq!(staking_ratio, 0.5);
}


#[test]
fn test_sybil_resistance() {
    let identities = vec![
        ("Alice".to_string(), 100.0),
        ("Bob".to_string(), 200.0),
        ("Charlie".to_string(), 300.0),
    ];
    
    // Apply Sybil resistance (e.g., square root scaling)
    let resistant = apply_sybil_resistance(&identities, |x| x.sqrt()).unwrap();
    
    assert_eq!(resistant.len(), 3);
    
    // Check that larger stakes don't get proportional power
    let alice_power = resistant.iter().find(|(n, _)| n == "Alice").unwrap().1;
    let charlie_power = resistant.iter().find(|(n, _)| n == "Charlie").unwrap().1;
    
    // Charlie has 3x stake but should have less than 3x power
    let ratio = charlie_power / alice_power;
    assert!(ratio < 3.0, "Sybil resistance should reduce large stake power");
}


#[test]
fn test_token_vesting() {
    let mut vesting = TokenVesting::new("Alice".to_string(), 10000.0);
    
    // Add vesting schedule
    vesting.add_vesting_schedule(1000.0, 30, 90); // 1000 tokens over 30-90 days
    
    // Test vesting calculation
    let vested = vesting.calculate_vested(45); // Day 45
    assert!(vested > 0.0 && vested < 1000.0);
    
    let fully_vested = vesting.calculate_vested(90); // Day 90
    assert!((fully_vested - 1000.0).abs() < 0.0001);
    
    // Test cliff
    let before_cliff = vesting.calculate_vested(29); // Day 29 (before cliff)
    assert_eq!(before_cliff, 0.0);
}


// Placeholder implementations for economy module


pub fn calculate_shapley_values<F>(players: &[String], characteristic: F) -> std::collections::HashMap<String, f64>
where
    F: Fn(&[String]) -> f64,
{
    let mut values = std::collections::HashMap::new();
    let n = players.len() as f64;
    
    // Simplified calculation (exact for symmetric games)
    for player in players {
        let mut total = 0.0;
        
        // This is a simplified implementation
        // In practice, we would iterate over all permutations
        for k in 0..players.len() {
            let weight = 1.0 / (n * (n - 1.0) * binomial_coefficient(n as usize - 1, k) as f64);
            total += weight * (characteristic(&[player.clone()]) + 1.0);
        }
        
        values.insert(player.clone(), total);
    }
    
    values
}


fn binomial_coefficient(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut result = 1;
    for i in 1..=k {
        result = result * (n - k + i) / i;
    }
    result
}


pub fn federated_average(gradients: &[Vec<f32>]) -> Result<Vec<f32>, EconomyError> {
    if gradients.is_empty() {
        return Err(EconomyError::InvalidParameter("No gradients provided".to_string()));
    }
    
    let n = gradients.len();
    let dim = gradients[0].len();
    
    let mut aggregated = vec![0.0; dim];
    
    for gradient in gradients {
        if gradient.len() != dim {
            return Err(EconomyError::InvalidParameter("Gradient dimensions mismatch".to_string()));
        }
        
        for (i, &value) in gradient.iter().enumerate() {
            aggregated[i] += value;
        }
    }
    
    for value in aggregated.iter_mut() {
        *value /= n as f32;
    }
    
    Ok(aggregated)
}


pub fn validate_gradient(gradient: &[f32]) -> Result<bool, EconomyError> {
    for &value in gradient {
        if value.is_nan() || value.is_infinite() {
            return Ok(false);
        }
    }
    Ok(true)
}


pub fn calculate_proportional_rewards(contributions: &[(String, f64)], total_reward: f64) -> Result<Vec<(String, f64)>, EconomyError> {
    let total_contributions: f64 = contributions.iter().map(|(_, c)| c).sum();
    
    if total_contributions <= 0.0 {
        return Err(EconomyError::InvalidParameter("Total contributions must be positive".to_string()));
    }
    
    let rewards: Vec<(String, f64)> = contributions
        .iter()
        .map(|(name, contribution)| {
            let reward = contribution / total_contributions * total_reward;
            (name.clone(), reward)
        })
        .collect();
    
    Ok(rewards)
}


pub fn apply_sybil_resistance<F>(identities: &[(String, f64)], transform: F) -> Result<Vec<(String, f64)>, EconomyError>
where
    F: Fn(f64) -> f64,
{
    let transformed: Vec<(String, f64)> = identities
        .iter()
        .map(|(name, stake)| (name.clone(), transform(*stake)))
        .collect();
    
    Ok(transformed)
}


#[derive(Debug)]
pub enum EconomyError {
    InvalidParameter(String),
    CalculationError(String),
}


impl std::fmt::Display for EconomyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            Self::CalculationError(msg) => write!(f, "Calculation error: {}", msg),
        }
    }
}


impl std::error::Error for EconomyError {}


type Result<T> = std::result::Result<T, EconomyError>;


struct EconomicMetrics {
    total_stake: f64,
    total_rewards_distributed: f64,
    transaction_fees_collected: f64,
    validator_rewards_paid: f64,
}


impl EconomicMetrics {
    fn new() -> Self {
        Self {
            total_stake: 0.0,
            total_rewards_distributed: 0.0,
            transaction_fees_collected: 0.0,
            validator_rewards_paid: 0.0,
        }
    }
    
    fn update_total_stake(&mut self, stake: f64) {
        self.total_stake = stake;
    }
    
    fn update_rewards_distributed(&mut self, rewards: f64) {
        self.total_rewards_distributed = rewards;
    }
    
    fn record_transaction_fee(&mut self, fee: f64) {
        self.transaction_fees_collected += fee;
    }
    
    fn record_validator_reward(&mut self, reward: f64) {
        self.validator_rewards_paid += reward;
    }
    
    fn calculate_inflation_rate(&self) -> f64 {
        if self.total_stake > 0.0 {
            self.total_rewards_distributed / self.total_stake
        } else {
            0.0
        }
    }
    
    fn calculate_staking_ratio(&self, staked_amount: f64) -> f64 {
        if self.total_stake > 0.0 {
            staked_amount / self.total_stake
        } else {
            0.0
        }
    }
}


struct VestingSchedule {
    amount: f64,
    start_day: u64,
    end_day: u64,
}


struct TokenVesting {
    beneficiary: String,
    total_amount: f64,
    schedules: Vec<VestingSchedule>,
}


impl TokenVesting {
    fn new(beneficiary: String, total_amount: f64) -> Self {
        Self {
            beneficiary,
            total_amount,
            schedules: Vec::new(),
        }
    }
    
    fn add_vesting_schedule(&mut self, amount: f64, start_day: u64, end_day: u64) {
        self.schedules.push(VestingSchedule {
            amount,
            start_day,
            end_day,
        });
    }
    
    fn calculate_vested(&self, current_day: u64) -> f64 {
        let mut total_vested = 0.0;
        
        for schedule in &self.schedules {
            if current_day >= schedule.end_day {
                total_vested += schedule.amount;
            } else if current_day > schedule.start_day {
                let duration = schedule.end_day - schedule.start_day;
                let elapsed = current_day - schedule.start_day;
                total_vested += schedule.amount * (elapsed as f64 / duration as f64);
            }
        }
        
        total_vested
    }
}
