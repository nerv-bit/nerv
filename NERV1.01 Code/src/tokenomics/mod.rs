// src/tokenomics/mod.rs
// ============================================================================
// TOKENOMICS AND ECONOMIC MODEL
// ============================================================================
// This module implements NERV's fair launch tokenomics as specified in the
// whitepaper Section 8:
// 1. Genesis allocation (no pre-mine, fair distribution)
// 2. Emission schedule (10-year emission with 0.5% perpetual tail)
// 3. Useful-work rewards and Shapley-value incentives
// 4. Vesting schedules for early contributors
// ============================================================================

use crate::{Result, NervError};
use crate::params::{
    TOTAL_SUPPLY, EMISSION_YEAR_1_2_PERCENT, EMISSION_YEAR_3_5_PERCENT,
    EMISSION_YEAR_6_10_PERCENT, TAIL_EMISSION_PERCENT,
    GENESIS_USEFUL_WORK_PERCENT, GENESIS_CODE_CONTRIB_PERCENT,
    GENESIS_AUDIT_BOUNTY_PERCENT, GENESIS_RESEARCH_PERCENT,
    GENESIS_EARLY_DONOR_PERCENT, GENESIS_TREASURY_PERCENT,
    GENESIS_VISIONARY_PERCENT,
    REWARD_GRADIENT_PERCENT, REWARD_VALIDATION_PERCENT, REWARD_PUBLIC_GOODS_PERCENT,
};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// Re-export public types
pub use emission::EmissionSchedule;
pub use allocation::{GenesisAllocation, AllocationRecord, VestingSchedule};
pub use rewards::{RewardCalculator, GradientReward, ValidationReward};

// Module declarations
pub mod emission;
pub mod allocation;
pub mod rewards;
pub mod vesting;

// ============================================================================
// CORE TOKENOMICS TYPES
// ============================================================================

/// NERV token (smallest unit: 1 wei = 10^-18 NERV)
pub type TokenAmount = u128;

/// Convert NERV to wei (10^18 wei per NERV)
pub const TOKEN_DECIMALS: u32 = 18;
pub const WEI_PER_NERV: TokenAmount = 1_000_000_000_000_000_000;

/// Address type for token holders
pub type Address = [u8; 20];

/// Tokenomics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenomicsConfig {
    /// Total supply (10 billion NERV)
    pub total_supply: TokenAmount,
    
    /// Emission schedule configuration
    pub emission_schedule: EmissionSchedule,
    
    /// Genesis allocation configuration
    pub genesis_allocation: GenesisAllocation,
    
    /// Reward distribution configuration
    pub reward_distribution: RewardDistribution,
    
    /// Vesting configuration
    pub vesting_config: VestingConfig,
    
    /// Transaction fee parameters
    pub fee_params: FeeParams,
    
    /// Inflation parameters
    pub inflation_params: InflationParams,
}

impl Default for TokenomicsConfig {
    fn default() -> Self {
        Self {
            total_supply: TOTAL_SUPPLY as TokenAmount * WEI_PER_NERV,
            emission_schedule: EmissionSchedule::default(),
            genesis_allocation: GenesisAllocation::default(),
            reward_distribution: RewardDistribution::default(),
            vesting_config: VestingConfig::default(),
            fee_params: FeeParams::default(),
            inflation_params: InflationParams::default(),
        }
    }
}

/// Reward distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardDistribution {
    /// Gradient contribution percentage (60%)
    pub gradient_percent: f64,
    
    /// Validation percentage (30%)
    pub validation_percent: f64,
    
    /// Public goods percentage (10%)
    pub public_goods_percent: f64,
}

impl Default for RewardDistribution {
    fn default() -> Self {
        Self {
            gradient_percent: REWARD_GRADIENT_PERCENT,
            validation_percent: REWARD_VALIDATION_PERCENT,
            public_goods_percent: REWARD_PUBLIC_GOODS_PERCENT,
        }
    }
}

/// Vesting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VestingConfig {
    /// Visionary vesting period (2 years)
    pub visionary_vesting_years: u64,
    
    /// Early contributor vesting cliff (6 months)
    pub early_contributor_cliff_months: u64,
    
    /// Treasury unlock schedule (quarterly)
    pub treasury_unlock_quarters: u64,
    
    /// Allow accelerated vesting for community vote
    pub allow_accelerated_vesting: bool,
}

impl Default for VestingConfig {
    fn default() -> Self {
        Self {
            visionary_vesting_years: 2,
            early_contributor_cliff_months: 6,
            treasury_unlock_quarters: 16, // 4 years
            allow_accelerated_vesting: true,
        }
    }
}

/// Transaction fee parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeParams {
    /// Base fee per transaction (wei)
    pub base_fee: TokenAmount,
    
    /// Fee per byte (wei)
    pub fee_per_byte: TokenAmount,
    
    /// Priority fee multiplier
    pub priority_fee_multiplier: f64,
    
    /// Burn percentage of fees (0-100%)
    pub burn_percentage: u8,
    
    /// Fee distribution to validators
    pub validator_fee_share: u8,
}

impl Default for FeeParams {
    fn default() -> Self {
        Self {
            base_fee: 100, // 100 wei
            fee_per_byte: 1, // 1 wei per byte
            priority_fee_multiplier: 1.5,
            burn_percentage: 20, // 20% burned
            validator_fee_share: 80, // 80% to validators
        }
    }
}

/// Inflation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InflationParams {
    /// Initial inflation rate (38% first year)
    pub initial_inflation_rate: f64,
    
    /// Inflation decay rate per year
    pub inflation_decay_rate: f64,
    
    /// Target tail inflation (0.5%)
    pub tail_inflation_rate: f64,
    
    /// Years to reach tail inflation
    pub years_to_tail: u64,
    
    /// Maximum supply (hard cap)
    pub max_supply: TokenAmount,
}

impl Default for InflationParams {
    fn default() -> Self {
        Self {
            initial_inflation_rate: 0.38,
            inflation_decay_rate: 0.3, // 30% decay per year
            tail_inflation_rate: TAIL_EMISSION_PERCENT,
            years_to_tail: 10,
            max_supply: TOTAL_SUPPLY as TokenAmount * WEI_PER_NERV * 2, // 2x buffer
        }
    }
}

// ============================================================================
// TOKENOMICS ENGINE - Main interface
// ============================================================================

/// Main tokenomics engine for managing emissions, rewards, and allocations
pub struct TokenomicsEngine {
    /// Configuration
    config: TokenomicsConfig,
    
    /// Current block height
    current_height: u64,
    
    /// Genesis timestamp (seconds since Unix epoch)
    genesis_timestamp: u64,
    
    /// Total emitted tokens
    total_emitted: TokenAmount,
    
    /// Total burned tokens
    total_burned: TokenAmount,
    
    /// Vesting schedules
    vesting_schedules: HashMap<Address, Vec<VestingSchedule>>,
    
    /// Reward pools
    reward_pools: RewardPools,
    
    /// Fee collector
    fee_collector: FeeCollector,
}

impl TokenomicsEngine {
    /// Create a new tokenomics engine
    pub fn new(config: TokenomicsConfig, genesis_timestamp: u64) -> Self {
        Self {
            config,
            current_height: 0,
            genesis_timestamp,
            total_emitted: 0,
            total_burned: 0,
            vesting_schedules: HashMap::new(),
            reward_pools: RewardPools::default(),
            fee_collector: FeeCollector::default(),
        }
    }
    
    /// Get current block reward (emission for this block)
    pub fn current_block_reward(&self) -> TokenAmount {
        let year = self.current_year();
        let emission = self.config.emission_schedule.emission_for_year(year);
        
        // Convert annual emission to per-block emission
        let blocks_per_year = self.blocks_per_year();
        emission / blocks_per_year
    }
    
    /// Distribute block rewards
    pub fn distribute_rewards(
        &mut self,
        gradient_contributors: &[GradientReward],
        validators: &[ValidationReward],
        public_goods_proposals: &[PublicGoodsProposal],
    ) -> Result<DistributionResult> {
        let block_reward = self.current_block_reward();
        
        // Calculate reward pools
        let gradient_pool = (block_reward as f64 * self.config.reward_distribution.gradient_percent) as TokenAmount;
        let validation_pool = (block_reward as f64 * self.config.reward_distribution.validation_percent) as TokenAmount;
        let public_goods_pool = (block_reward as f64 * self.config.reward_distribution.public_goods_percent) as TokenAmount;
        
        // Distribute gradient rewards (Shapley-value weighted)
        let gradient_distribution = self.distribute_gradient_rewards(gradient_contributors, gradient_pool)?;
        
        // Distribute validation rewards (stake × reputation weighted)
        let validation_distribution = self.distribute_validation_rewards(validators, validation_pool)?;
        
        // Select and fund public goods proposals
        let public_goods_distribution = self.select_public_goods(public_goods_proposals, public_goods_pool)?;
        
        // Update total emitted
        self.total_emitted += block_reward;
        
        // Check inflation bounds
        self.validate_inflation()?;
        
        Ok(DistributionResult {
            block_reward,
            gradient_distribution,
            validation_distribution,
            public_goods_distribution,
            total_emitted: self.total_emitted,
        })
    }
    
    /// Calculate transaction fee
    pub fn calculate_fee(&self, tx_size_bytes: usize, priority: bool) -> TokenAmount {
        let mut fee = self.config.fee_params.base_fee
            + (tx_size_bytes as TokenAmount * self.config.fee_params.fee_per_byte);
        
        if priority {
            fee = (fee as f64 * self.config.fee_params.priority_fee_multiplier) as TokenAmount;
        }
        
        fee
    }
    
    /// Process transaction fee (burn and distribute)
    pub fn process_fee(&mut self, fee: TokenAmount) -> FeeProcessingResult {
        let burn_amount = (fee as f64 * (self.config.fee_params.burn_percentage as f64 / 100.0)) as TokenAmount;
        let validator_amount = (fee as f64 * (self.config.fee_params.validator_fee_share as f64 / 100.0)) as TokenAmount;
        
        // Burn tokens
        self.total_burned += burn_amount;
        
        // Add to validator fee pool
        self.fee_collector.add_to_validator_pool(validator_amount);
        
        FeeProcessingResult {
            total_fee: fee,
            burned: burn_amount,
            to_validators: validator_amount,
            total_burned: self.total_burned,
        }
    }
    
    /// Check vesting status for an address
    pub fn vested_amount(&self, address: &Address, current_time: u64) -> TokenAmount {
        let mut total_vested = 0;
        
        if let Some(schedules) = self.vesting_schedules.get(address) {
            for schedule in schedules {
                total_vested += schedule.vested_amount(current_time);
            }
        }
        
        total_vested
    }
    
    /// Get circulating supply (total emitted - total burned - locked in vesting)
    pub fn circulating_supply(&self, current_time: u64) -> TokenAmount {
        // Calculate total locked in vesting
        let total_locked: TokenAmount = self.vesting_schedules.values()
            .flat_map(|schedules| schedules.iter())
            .map(|schedule| schedule.total_amount - schedule.vested_amount(current_time))
            .sum();
        
        self.total_emitted - self.total_burned - total_locked
    }
    
    /// Get current inflation rate
    pub fn current_inflation_rate(&self) -> f64 {
        let circulating = self.circulating_supply(self.current_timestamp());
        if circulating == 0 {
            return 0.0;
        }
        
        let annual_emission = self.config.emission_schedule.emission_for_year(self.current_year());
        let blocks_per_year = self.blocks_per_year();
        let daily_emission = annual_emission / 365;
        
        (daily_emission as f64 * 365.0) / circulating as f64
    }
    
    /// Update block height (called when new block is produced)
    pub fn update_block_height(&mut self, height: u64) {
        self.current_height = height;
    }
    
    // Helper methods
    fn current_year(&self) -> u64 {
        let elapsed_seconds = self.current_timestamp() - self.genesis_timestamp;
        elapsed_seconds / (365 * 24 * 60 * 60) + 1 // Year 1, 2, 3...
    }
    
    fn blocks_per_year(&self) -> TokenAmount {
        // Assuming 0.6 second block time
        (365 * 24 * 60 * 60) / 600 * 100 // Convert to TokenAmount
    }
    
    fn current_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
    
    fn distribute_gradient_rewards(
        &self,
        contributors: &[GradientReward],
        total_pool: TokenAmount,
    ) -> Result<Vec<(Address, TokenAmount)>> {
        if contributors.is_empty() {
            return Ok(Vec::new());
        }
        
        // Calculate total Shapley value
        let total_shapley: f64 = contributors.iter().map(|c| c.shapley_value).sum();
        
        if total_shapley == 0.0 {
            return Err(NervError::Config("Total Shapley value is zero".to_string()));
        }
        
        // Distribute proportionally to Shapley value
        let mut distribution = Vec::new();
        for contributor in contributors {
            let share = (contributor.shapley_value / total_shapley) * total_pool as f64;
            distribution.push((contributor.address, share as TokenAmount));
        }
        
        Ok(distribution)
    }
    
    fn distribute_validation_rewards(
        &self,
        validators: &[ValidationReward],
        total_pool: TokenAmount,
    ) -> Result<Vec<(Address, TokenAmount)>> {
        if validators.is_empty() {
            return Ok(Vec::new());
        }
        
        // Calculate total weight (stake × reputation)
        let total_weight: f64 = validators.iter()
            .map(|v| v.stake as f64 * v.reputation_score)
            .sum();
        
        if total_weight == 0.0 {
            return Err(NervError::Config("Total validator weight is zero".to_string()));
        }
        
        // Distribute proportionally to weight
        let mut distribution = Vec::new();
        for validator in validators {
            let weight = validator.stake as f64 * validator.reputation_score;
            let share = (weight / total_weight) * total_pool as f64;
            distribution.push((validator.address, share as TokenAmount));
        }
        
        Ok(distribution)
    }
    
    fn select_public_goods(
        &self,
        proposals: &[PublicGoodsProposal],
        total_pool: TokenAmount,
    ) -> Result<Vec<(Address, TokenAmount, String)>> {
        // Simple selection: fund top proposals until pool is exhausted
        let mut selected = Vec::new();
        let mut remaining_pool = total_pool;
        
        // Sort proposals by score (community vote score)
        let mut sorted_proposals = proposals.to_vec();
        sorted_proposals.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        for proposal in sorted_proposals {
            if proposal.requested_amount <= remaining_pool {
                selected.push((proposal.recipient, proposal.requested_amount, proposal.description.clone()));
                remaining_pool -= proposal.requested_amount;
            } else if remaining_pool > 0 {
                // Partial funding
                selected.push((proposal.recipient, remaining_pool, proposal.description.clone()));
                remaining_pool = 0;
                break;
            }
        }
        
        Ok(selected)
    }
    
    fn validate_inflation(&self) -> Result<()> {
        let inflation_rate = self.current_inflation_rate();
        let max_inflation = self.config.inflation_params.initial_inflation_rate;
        
        if inflation_rate > max_inflation * 1.1 { // 10% tolerance
            return Err(NervError::Config(format!(
                "Inflation rate {} exceeds maximum {}",
                inflation_rate, max_inflation
            )));
        }
        
        let total_supply = self.total_emitted;
        let max_supply = self.config.inflation_params.max_supply;
        
        if total_supply > max_supply {
            return Err(NervError::Config(format!(
                "Total supply {} exceeds maximum {}",
                total_supply, max_supply
            )));
        }
        
        Ok(())
    }
}

// ============================================================================
// SUPPORTING TYPES
// ============================================================================

/// Gradient reward for useful-work contributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientReward {
    /// Contributor address
    pub address: Address,
    
    /// Shapley value contribution
    pub shapley_value: f64,
    
    /// Gradient quality score (0-1)
    pub quality_score: f64,
    
    /// Differential privacy epsilon used
    pub dp_epsilon: f64,
    
    /// Timestamp of contribution
    pub timestamp: u64,
}

/// Validation reward for honest consensus participation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReward {
    /// Validator address
    pub address: Address,
    
    /// Staked amount
    pub stake: TokenAmount,
    
    /// Reputation score (0-1)
    pub reputation_score: f64,
    
    /// Uptime percentage (0-1)
    pub uptime: f64,
    
    /// Number of votes participated in
    pub votes_count: u64,
    
    /// Slashing history (negative for penalties)
    pub slashing_adjustment: i64,
}

/// Public goods funding proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicGoodsProposal {
    /// Proposal ID
    pub proposal_id: u64,
    
    /// Recipient address
    pub recipient: Address,
    
    /// Amount requested
    pub requested_amount: TokenAmount,
    
    /// Description of the public good
    pub description: String,
    
    /// Community vote score (0-1)
    pub score: f64,
    
    /// Category (research, development, education, etc.)
    pub category: ProposalCategory,
    
    /// Duration (months)
    pub duration_months: u64,
}

/// Proposal category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProposalCategory {
    Research,
    Development,
    Education,
    Community,
    Security,
    Infrastructure,
    Other,
}

/// Reward distribution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionResult {
    /// Total block reward
    pub block_reward: TokenAmount,
    
    /// Gradient rewards distribution
    pub gradient_distribution: Vec<(Address, TokenAmount)>,
    
    /// Validation rewards distribution
    pub validation_distribution: Vec<(Address, TokenAmount)>,
    
    /// Public goods funding distribution
    pub public_goods_distribution: Vec<(Address, TokenAmount, String)>,
    
    /// Total emitted tokens so far
    pub total_emitted: TokenAmount,
}

/// Fee processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeProcessingResult {
    /// Total fee collected
    pub total_fee: TokenAmount,
    
    /// Amount burned
    pub burned: TokenAmount,
    
    /// Amount to validators
    pub to_validators: TokenAmount,
    
    /// Total burned tokens so far
    pub total_burned: TokenAmount,
}

/// Reward pools
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RewardPools {
    gradient_pool: TokenAmount,
    validation_pool: TokenAmount,
    public_goods_pool: TokenAmount,
}

/// Fee collector
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct FeeCollector {
    validator_fee_pool: TokenAmount,
    treasury_fee_pool: TokenAmount,
    burned_fees: TokenAmount,
}

impl FeeCollector {
    fn add_to_validator_pool(&mut self, amount: TokenAmount) {
        self.validator_fee_pool += amount;
    }
    
    fn add_to_treasury_pool(&mut self, amount: TokenAmount) {
        self.treasury_fee_pool += amount;
    }
    
    fn add_burned_fees(&mut self, amount: TokenAmount) {
        self.burned_fees += amount;
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Calculate annual percentage rate (APR) for staking
pub fn calculate_staking_apr(
    total_staked: TokenAmount,
    annual_validation_rewards: TokenAmount,
    fee_revenue: TokenAmount,
) -> f64 {
    if total_staked == 0 {
        return 0.0;
    }
    
    let total_rewards = annual_validation_rewards + fee_revenue;
    (total_rewards as f64 / total_staked as f64) * 100.0
}

/// Calculate Shapley value for gradient contributions
pub fn calculate_shapley_value(
    gradients: &[Vec<f32>], // List of gradient vectors from contributors
    baseline_loss: f64,     // Loss without any contributions
    final_loss: f64,        // Loss with all contributions
) -> Vec<f64> {
    // Simplified Shapley calculation (Monte Carlo approximation)
    // In production, this would use secure multi-party computation in TEEs
    
    let n = gradients.len();
    if n == 0 {
        return Vec::new();
    }
    
    let mut shapley_values = vec![0.0; n];
    
    // Monte Carlo approximation (1000 samples)
    let samples = 1000.min(1 << n); // Limit to 2^n for small n
    
    for _ in 0..samples {
        // Random permutation of contributors
        let mut permutation: Vec<usize> = (0..n).collect();
        // Fisher-Yates shuffle
        for i in 0..n {
            let j = rand::random::<usize>() % (n - i) + i;
            permutation.swap(i, j);
        }
        
        // Calculate marginal contributions
        let mut current_loss = baseline_loss;
        for &contributor_idx in &permutation {
            // Simulate adding this contributor's gradient
            // In production, this would actually recompute the loss
            let marginal_improvement = (current_loss - final_loss) / n as f64;
            shapley_values[contributor_idx] += marginal_improvement;
            current_loss -= marginal_improvement;
        }
    }
    
    // Average over samples
    for value in &mut shapley_values {
        *value /= samples as f64;
    }
    
    shapley_values
}

/// Convert NERV to human-readable string
pub fn format_nerv_amount(amount: TokenAmount) -> String {
    let nerv = amount as f64 / WEI_PER_NERV as f64;
    format!("{:.8} NERV", nerv)
}

/// Convert human-readable NERV string to wei
pub fn parse_nerv_amount(s: &str) -> Result<TokenAmount> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() != 2 || parts[1].to_uppercase() != "NERV" {
        return Err(NervError::Config("Invalid format, expected 'X.XX NERV'".to_string()));
    }
    
    let nerv: f64 = parts[0].parse()
        .map_err(|e| NervError::Config(format!("Invalid number: {}", e)))?;
    
    Ok((nerv * WEI_PER_NERV as f64) as TokenAmount)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_token_conversions() {
        assert_eq!(WEI_PER_NERV, 1_000_000_000_000_000_000);
        
        let amount = 1.5 * WEI_PER_NERV as f64;
        let formatted = format_nerv_amount(amount as TokenAmount);
        assert_eq!(formatted, "1.50000000 NERV");
        
        let parsed = parse_nerv_amount("1.50000000 NERV").unwrap();
        assert_eq!(parsed, amount as TokenAmount);
    }
    
    #[test]
    fn test_fee_calculation() {
        let config = TokenomicsConfig::default();
        let engine = TokenomicsEngine::new(config, 0);
        
        let fee = engine.calculate_fee(256, false); // 256 byte transaction
        assert!(fee > 0);
        
        let priority_fee = engine.calculate_fee(256, true);
        assert!(priority_fee >= fee); // Priority fee should be at least base fee
    }
    
    #[test]
    fn test_shapley_calculation() {
        // Mock gradients (3 contributors)
        let gradients = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];
        
        let baseline_loss = 10.0;
        let final_loss = 5.0;
        
        let shapley_values = calculate_shapley_value(&gradients, baseline_loss, final_loss);
        
        assert_eq!(shapley_values.len(), 3);
        
        // Values should sum to total improvement
        let total_improvement = baseline_loss - final_loss;
        let sum: f64 = shapley_values.iter().sum();
        assert!((sum - total_improvement).abs() < 0.1);
        
        // All values should be positive
        for &value in &shapley_values {
            assert!(value >= 0.0);
        }
    }
    
    #[test]
    fn test_reward_distribution() {
        let config = TokenomicsConfig::default();
        let mut engine = TokenomicsEngine::new(config, 0);
        
        // Mock gradient contributors
        let gradient_contributors = vec![
            GradientReward {
                address: [1u8; 20],
                shapley_value: 0.6,
                quality_score: 0.9,
                dp_epsilon: 3.0,
                timestamp: 0,
            },
            GradientReward {
                address: [2u8; 20],
                shapley_value: 0.4,
                quality_score: 0.8,
                dp_epsilon: 3.0,
                timestamp: 0,
            },
        ];
        
        // Mock validators
        let validators = vec![
            ValidationReward {
                address: [3u8; 20],
                stake: 1000 * WEI_PER_NERV,
                reputation_score: 0.95,
                uptime: 0.99,
                votes_count: 1000,
                slashing_adjustment: 0,
            },
            ValidationReward {
                address: [4u8; 20],
                stake: 2000 * WEI_PER_NERV,
                reputation_score: 0.90,
                uptime: 0.98,
                votes_count: 2000,
                slashing_adjustment: -100, // Was slashed
            },
        ];
        
        // Mock public goods proposals
        let proposals = vec![
            PublicGoodsProposal {
                proposal_id: 1,
                recipient: [5u8; 20],
                requested_amount: 5000 * WEI_PER_NERV,
                description: "Research paper".to_string(),
                score: 0.8,
                category: ProposalCategory::Research,
                duration_months: 12,
            },
        ];
        
        // Distribute rewards
        let result = engine.distribute_rewards(
            &gradient_contributors,
            &validators,
            &proposals,
        ).unwrap();
        
        assert!(result.block_reward > 0);
        assert_eq!(result.gradient_distribution.len(), 2);
        assert_eq!(result.validation_distribution.len(), 2);
        
        // Check that gradient distribution is proportional to Shapley values
        let total_gradient: TokenAmount = result.gradient_distribution.iter()
            .map(|(_, amount)| amount)
            .sum();
        
        assert!(total_gradient > 0);
        
        // Check that validation distribution is proportional to stake × reputation
        let total_validation: TokenAmount = result.validation_distribution.iter()
            .map(|(_, amount)| amount)
            .sum();
        
        assert!(total_validation > 0);
    }
    
    #[test]
    fn test_fee_processing() {
        let config = TokenomicsConfig::default();
        let mut engine = TokenomicsEngine::new(config, 0);
        
        let fee = 1000 * WEI_PER_NERV;
        let result = engine.process_fee(fee);
        
        assert_eq!(result.total_fee, fee);
        
        // Check burn amount (20% of 1000 = 200)
        let expected_burn = (fee as f64 * 0.2) as TokenAmount;
        assert_eq!(result.burned, expected_burn);
        
        // Check validator amount (80% of 1000 = 800)
        let expected_validator = (fee as f64 * 0.8) as TokenAmount;
        assert_eq!(result.to_validators, expected_validator);
        
        assert_eq!(result.total_burned, expected_burn);
    }
    
    #[test]
    fn test_inflation_calculation() {
        let config = TokenomicsConfig::default();
        let engine = TokenomicsEngine::new(config, 0);
        
        let inflation_rate = engine.current_inflation_rate();
        
        // Initial inflation should be high (38%)
        assert!(inflation_rate > 0.0);
        assert!(inflation_rate <= 0.38);
    }
}
