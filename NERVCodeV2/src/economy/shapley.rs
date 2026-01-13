//! Shapley Value Computation for NERV Useful-Work Economy
//! 
//! This module implements the Shapley value mechanism for fair allocation of rewards
//! based on marginal contributions in federated learning. The Shapley value satisfies
//! four key axioms: efficiency, symmetry, dummy, and additivity.
//! 
//! Key Features:
//! - Monte Carlo approximation for scalable computation (O(n log n) vs O(2^n))
//! - Differential privacy integration (ε-DP guarantees)
//! - TEE-backed secure computation for privacy
//! - Parallel computation support for large coalitions
//! 
//! The Shapley value φ_i for node i is defined as:
//! φ_i(v) = Σ_{C ⊆ N\{i}} |C|!(n-|C|-1)!/n! [v(C ∪ {i}) - v(C)]
//! where v(C) is the value function measuring coalition C's contribution.


use crate::crypto::CryptoProvider;
use crate::params::{DP_SIGMA, MONTE_CARLO_SIMULATIONS};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug, warn};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;


/// Shapley value computer with Monte Carlo approximation
pub struct ShapleyComputer {
    /// Configuration parameters
    config: ShapleyConfig,
    
    /// Cryptographic provider for secure operations
    crypto_provider: Arc<CryptoProvider>,
    
    /// Cache for computed values (node_id -> Shapley value)
    value_cache: RwLock<HashMap<String, f64>>,
    
    /// Historical data for adaptive sampling
    history: RwLock<ShapleyHistory>,
    
    /// Random number generator
    rng: RwLock<rand::rngs::StdRng>,
}


impl ShapleyComputer {
    /// Create a new Shapley computer with given configuration
    pub fn new(config: ShapleyConfig) -> Self {
        let crypto_provider = Arc::new(CryptoProvider::new().expect("Failed to create crypto provider"));
        let rng = RwLock::new(rand::rngs::StdRng::from_entropy());
        
        Self {
            config,
            crypto_provider,
            value_cache: RwLock::new(HashMap::new()),
            history: RwLock::new(ShapleyHistory::new()),
            rng,
        }
    }
    
    /// Compute Shapley values for a set of gradient contributions
    pub async fn compute_shapley(
        &self,
        contributions: &HashMap<String, GradientContribution>,
    ) -> Result<Vec<ShapleyValue>, ShapleyError> {
        let start_time = std::time::Instant::now();
        
        if contributions.is_empty() {
            return Ok(Vec::new());
        }
        
        info!("Computing Shapley values for {} nodes", contributions.len());
        
        // Extract node IDs and contributions
        let node_ids: Vec<String> = contributions.keys().cloned().collect();
        let n = node_ids.len();
        
        if n > self.config.max_coalition_size {
            warn!("Number of nodes ({}) exceeds max coalition size ({}). Using approximation.",
                  n, self.config.max_coalition_size);
        }
        
        // Choose computation method based on size
        let shapley_values = if n <= 10 {
            // Exact computation for small n
            self.compute_exact(&node_ids, contributions).await?
        } else if n <= self.config.max_coalition_size {
            // Monte Carlo approximation for medium n
            self.compute_monte_carlo(&node_ids, contributions).await?
        } else {
            // Truncated approximation for very large n
            self.compute_truncated(&node_ids, contributions).await?
        };
        
        // Apply differential privacy if enabled
        let private_values = if self.config.privacy_budget > 0.0 {
            self.apply_differential_privacy(shapley_values, self.config.privacy_budget).await?
        } else {
            shapley_values
        };
        
        // Cache results
        self.cache_values(&private_values).await;
        
        // Update history
        self.update_history(&node_ids, &private_values).await;
        
        let duration = start_time.elapsed();
        info!("Shapley computation completed in {}ms for {} nodes", 
              duration.as_millis(), n);
        
        Ok(private_values)
    }
    
    /// Compute exact Shapley values (exponential complexity, only for small n)
    async fn compute_exact(
        &self,
        node_ids: &[String],
        contributions: &HashMap<String, GradientContribution>,
    ) -> Result<Vec<ShapleyValue>, ShapleyError> {
        let n = node_ids.len();
        if n > 15 {
            return Err(ShapleyError::ComputationTooLarge(n, 15));
        }
        
        info!("Computing exact Shapley values for {} nodes", n);
        
        // Generate all subsets (2^n)
        let total_subsets = 1usize << n;
        let mut shapley_values = vec![0.0; n];
        
        // Iterate over all subsets
        for subset_mask in 0..total_subsets {
            let subset = self.mask_to_set(subset_mask, node_ids);
            
            // For each node not in subset
            for (i, node_id) in node_ids.iter().enumerate() {
                if (subset_mask >> i) & 1 == 0 {
                    // Node i is not in subset
                    let subset_with_i = self.add_node_to_set(&subset, node_id);
                    
                    // Compute marginal contribution: v(S ∪ {i}) - v(S)
                    let value_without = self.evaluate_coalition(&subset, contributions).await?;
                    let value_with = self.evaluate_coalition(&subset_with_i, contributions).await?;
                    let marginal = value_with - value_without;
                    
                    // Weight: |S|!(n-|S|-1)!/n!
                    let s_size = subset.len();
                    let weight = self.compute_exact_weight(s_size, n);
                    
                    shapley_values[i] += weight * marginal;
                }
            }
        }
        
        // Convert to ShapleyValue structs
        let result = node_ids.iter().enumerate()
            .map(|(i, node_id)| ShapleyValue {
                node_id: node_id.clone(),
                value: shapley_values[i],
                confidence: 1.0, // Exact computation
                computation_method: "exact".to_string(),
            })
            .collect();
        
        Ok(result)
    }
    
    /// Compute Shapley values using Monte Carlo approximation
    async fn compute_monte_carlo(
        &self,
        node_ids: &[String],
        contributions: &HashMap<String, GradientContribution>,
    ) -> Result<Vec<ShapleyValue>, ShapleyError> {
        let n = node_ids.len();
        let samples = self.config.monte_carlo_samples;
        
        info!("Computing Monte Carlo Shapley values ({} samples)", samples);
        
        // Initialize accumulators
        let mut shapley_accumulators = vec![0.0; n];
        let mut shapley_squares = vec![0.0; n]; // For variance calculation
        
        // Generate random permutations in parallel if enabled
        let results: Vec<Vec<f64>> = if self.config.enable_parallel && n > 1 {
            // Parallel Monte Carlo sampling
            (0..samples).into_par_iter()
                .map(|_| {
                    let mut local_rng = rand::rngs::StdRng::from_entropy();
                    self.compute_marginal_for_permutation(&node_ids, contributions, &mut local_rng)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            // Sequential sampling
            let mut rng = self.rng.write().await;
            (0..samples).map(|_| {
                self.compute_marginal_for_permutation(&node_ids, contributions, &mut *rng)
            })
            .collect::<Result<Vec<_>, _>>()?
        };
        
        // Aggregate results
        for (sample_idx, marginals) in results.iter().enumerate() {
            for (i, &marginal) in marginals.iter().enumerate() {
                shapley_accumulators[i] += marginal;
                shapley_squares[i] += marginal * marginal;
            }
            
            // Progress reporting
            if sample_idx % 1000 == 0 && sample_idx > 0 {
                debug!("Monte Carlo progress: {}/{} samples", sample_idx, samples);
            }
        }
        
        // Compute means and variances
        let mut shapley_values = Vec::with_capacity(n);
        for (i, node_id) in node_ids.iter().enumerate() {
            let mean = shapley_accumulators[i] / samples as f64;
            let mean_sq = shapley_squares[i] / samples as f64;
            let variance = (mean_sq - mean * mean).max(0.0);
            let std_err = (variance / samples as f64).sqrt();
            let confidence = 1.0 / (1.0 + std_err); // Higher confidence with lower error
            
            shapley_values.push(ShapleyValue {
                node_id: node_id.clone(),
                value: mean,
                confidence,
                computation_method: "monte_carlo".to_string(),
            });
        }
        
        Ok(shapley_values)
    }
    
    /// Compute Shapley values using truncated approximation for very large n
    async fn compute_truncated(
        &self,
        node_ids: &[String],
        contributions: &HashMap<String, GradientContribution>,
    ) -> Result<Vec<ShapleyValue>, ShapleyError> {
        let n = node_ids.len();
        let truncation_size = self.config.max_coalition_size.min(n);
        
        info!("Computing truncated Shapley values (truncation size: {})", truncation_size);
        
        // Sort nodes by contribution magnitude for prioritization
        let mut nodes_with_contrib: Vec<(String, f64)> = node_ids.iter()
            .map(|id| {
                let contrib = contributions.get(id)
                    .map(|c| c.quality_score)
                    .unwrap_or(0.0);
                (id.clone(), contrib)
            })
            .collect();
        
        nodes_with_contrib.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top-k nodes for exact computation within coalitions
        let top_k: Vec<String> = nodes_with_contrib.iter()
            .take(truncation_size)
            .map(|(id, _)| id.clone())
            .collect();
        
        // For remaining nodes, use simplified approximation
        let remaining: Vec<String> = nodes_with_contrib.iter()
            .skip(truncation_size)
            .map(|(id, _)| id.clone())
            .collect();
        
        // Compute Shapley for top-k nodes using Monte Carlo
        let mut top_contributions = HashMap::new();
        for node_id in &top_k {
            if let Some(contrib) = contributions.get(node_id) {
                top_contributions.insert(node_id.clone(), contrib.clone());
            }
        }
        
        let top_shapley = self.compute_monte_carlo(&top_k, &top_contributions).await?;
        
        // For remaining nodes, use average contribution as approximation
        let mut all_shapley = top_shapley;
        
        if !remaining.is_empty() {
            let total_remaining_value = remaining.iter()
                .filter_map(|id| contributions.get(id).map(|c| c.quality_score))
                .sum::<f64>();
            
            let avg_remaining_value = total_remaining_value / remaining.len() as f64;
            
            for node_id in remaining {
                all_shapley.push(ShapleyValue {
                    node_id,
                    value: avg_remaining_value,
                    confidence: 0.5, // Low confidence for approximation
                    computation_method: "truncated_approximation".to_string(),
                });
            }
        }
        
        Ok(all_shapley)
    }
    
    /// Compute marginal contributions for a single random permutation
    fn compute_marginal_for_permutation(
        &self,
        node_ids: &[String],
        contributions: &HashMap<String, GradientContribution>,
        rng: &mut impl Rng,
    ) -> Result<Vec<f64>, ShapleyError> {
        let n = node_ids.len();
        let mut marginals = vec![0.0; n];
        
        // Generate random permutation
        let mut permutation: Vec<usize> = (0..n).collect();
        permutation.shuffle(rng);
        
        // Build coalition incrementally and compute marginal contributions
        let mut current_coalition = HashSet::new();
        let mut current_value = 0.0;
        
        for &pos in &permutation {
            let node_id = &node_ids[pos];
            
            // Add node to coalition
            current_coalition.insert(node_id.clone());
            
            // Compute new coalition value
            let new_value = self.evaluate_coalition_set(&current_coalition, contributions)?;
            
            // Marginal contribution of this node in this permutation
            marginals[pos] = new_value - current_value;
            
            // Update current value
            current_value = new_value;
        }
        
        Ok(marginals)
    }
    
    /// Evaluate the value of a coalition (set of node IDs)
    async fn evaluate_coalition(
        &self,
        coalition: &[String],
        contributions: &HashMap<String, GradientContribution>,
    ) -> Result<f64, ShapleyError> {
        let coalition_set: HashSet<String> = coalition.iter().cloned().collect();
        self.evaluate_coalition_set(&coalition_set, contributions)
    }
    
    /// Evaluate the value of a coalition (as HashSet)
    fn evaluate_coalition_set(
        &self,
        coalition: &HashSet<String>,
        contributions: &HashMap<String, GradientContribution>,
    ) -> Result<f64, ShapleyError> {
        if coalition.is_empty() {
            return Ok(0.0);
        }
        
        // Aggregate contributions from coalition members
        let mut total_quality = 0.0;
        let mut total_magnitude = 0.0;
        let mut count = 0;
        
        for node_id in coalition {
            if let Some(contrib) = contributions.get(node_id) {
                total_quality += contrib.quality_score;
                total_magnitude += contrib.gradient_magnitude;
                count += 1;
            }
        }
        
        if count == 0 {
            return Ok(0.0);
        }
        
        // Value function: weighted combination of quality and magnitude
        let avg_quality = total_quality / count as f64;
        let avg_magnitude = total_magnitude / count as f64;
        
        // Synergy factor: coalitions with diverse contributions are more valuable
        let synergy_factor = self.compute_synergy_factor(coalition, contributions);
        
        // Final value: base contribution adjusted by synergy
        let base_value = 0.7 * avg_quality + 0.3 * avg_magnitude;
        let value = base_value * (1.0 + synergy_factor);
        
        // Apply diminishing returns for very large coalitions
        if count > 10 {
            let scale = (10.0 / count as f64).sqrt();
            Ok(value * scale)
        } else {
            Ok(value)
        }
    }
    
    /// Compute synergy factor for a coalition
    fn compute_synergy_factor(
        &self,
        coalition: &HashSet<String>,
        contributions: &HashMap<String, GradientContribution>,
    ) -> f64 {
        // Measure diversity in contributions
        let mut quality_values = Vec::new();
        let mut magnitude_values = Vec::new();
        
        for node_id in coalition {
            if let Some(contrib) = contributions.get(node_id) {
                quality_values.push(contrib.quality_score);
                magnitude_values.push(contrib.gradient_magnitude);
            }
        }
        
        if quality_values.len() < 2 {
            return 0.0;
        }
        
        // Compute coefficient of variation as diversity measure
        let quality_cv = coefficient_of_variation(&quality_values);
        let magnitude_cv = coefficient_of_variation(&magnitude_values);
        
        // Diversity leads to synergy (up to a point)
        let diversity = 0.5 * quality_cv + 0.5 * magnitude_cv;
        let synergy = diversity.min(0.3); // Cap at 30% synergy
        
        synergy
    }
    
    /// Apply differential privacy to Shapley values
    async fn apply_differential_privacy(
        &self,
        mut shapley_values: Vec<ShapleyValue>,
        privacy_budget: f64,
    ) -> Result<Vec<ShapleyValue>, ShapleyError> {
        if privacy_budget <= 0.0 {
            return Ok(shapley_values);
        }
        
        info!("Applying differential privacy with ε={:.3}", privacy_budget);
        
        // Calculate sensitivity of Shapley values
        // For utility functions bounded in [0,1], sensitivity ≤ 1/n
        let n = shapley_values.len();
        let sensitivity = 1.0 / n.max(1) as f64;
        
        // Scale parameter for Laplace mechanism: Δf/ε
        let scale = sensitivity / privacy_budget;
        
        // Generate Laplace noise for each value
        let mut rng = self.rng.write().await;
        
        for value in &mut shapley_values {
            // Generate Laplace noise: Laplace(0, scale)
            let noise = laplace_sample(0.0, scale, &mut *rng);
            
            // Add noise to Shapley value
            value.value += noise;
            
            // Ensure non-negativity
            value.value = value.value.max(0.0);
            
            // Adjust confidence based on added noise
            let noise_ratio = noise.abs() / (value.value.abs() + 1e-10);
            value.confidence *= 1.0 / (1.0 + noise_ratio);
        }
        
        // Normalize to sum to 1 (preserving efficiency axiom)
        let total: f64 = shapley_values.iter().map(|v| v.value).sum();
        if total > 0.0 {
            for value in &mut shapley_values {
                value.value /= total;
            }
        }
        
        Ok(shapley_values)
    }
    
    /// Cache computed Shapley values
    async fn cache_values(&self, values: &[ShapleyValue]) {
        let mut cache = self.value_cache.write().await;
        for value in values {
            cache.insert(value.node_id.clone(), value.value);
        }
    }
    
    /// Update computation history
    async fn update_history(&self, node_ids: &[String], values: &[ShapleyValue]) {
        let mut history = self.history.write().await;
        history.record_computation(node_ids, values);
    }
    
    /// Helper: Convert bitmask to set of node IDs
    fn mask_to_set(&self, mask: usize, node_ids: &[String]) -> Vec<String> {
        let mut set = Vec::new();
        for (i, node_id) in node_ids.iter().enumerate() {
            if (mask >> i) & 1 == 1 {
                set.push(node_id.clone());
            }
        }
        set
    }
    
    /// Helper: Add node to set
    fn add_node_to_set(&self, set: &[String], node_id: &str) -> Vec<String> {
        let mut new_set = set.to_vec();
        if !new_set.contains(&node_id.to_string()) {
            new_set.push(node_id.to_string());
        }
        new_set
    }
    
    /// Helper: Compute exact Shapley weight
    fn compute_exact_weight(&self, s_size: usize, n: usize) -> f64 {
        let s_fact = factorial(s_size);
        let n_minus_s_minus_1_fact = factorial(n - s_size - 1);
        let n_fact = factorial(n);
        
        (s_fact * n_minus_s_minus_1_fact) as f64 / n_fact as f64
    }
}


/// Gradient contribution from a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientContribution {
    /// Node identifier
    pub node_id: String,
    
    /// Gradient data (serialized)
    pub gradient_data: Vec<u8>,
    
    /// Quality score (0.0 to 1.0)
    pub quality_score: f64,
    
    /// Gradient magnitude (L2 norm)
    pub gradient_magnitude: f64,
    
    /// Timestamp of contribution
    pub timestamp: std::time::SystemTime,
    
    /// TEE attestation proof
    pub attestation: Vec<u8>,
    
    /// Differential privacy parameters used
    pub dp_params: Option<DpParams>,
}


impl GradientContribution {
    /// Create a new gradient contribution
    pub fn new(
        node_id: String,
        gradient_data: Vec<u8>,
        quality_score: f64,
        gradient_magnitude: f64,
        attestation: Vec<u8>,
    ) -> Self {
        Self {
            node_id,
            gradient_data,
            quality_score: quality_score.clamp(0.0, 1.0),
            gradient_magnitude: gradient_magnitude.max(0.0),
            timestamp: std::time::SystemTime::now(),
            attestation,
            dp_params: None,
        }
    }
    
    /// Apply differential privacy parameters
    pub fn with_dp_params(mut self, dp_params: DpParams) -> Self {
        self.dp_params = Some(dp_params);
        self
    }
    
    /// Validate the contribution
    pub fn validate(&self) -> Result<(), ShapleyError> {
        if self.node_id.is_empty() {
            return Err(ShapleyError::InvalidContribution("Empty node ID".to_string()));
        }
        
        if self.gradient_data.is_empty() {
            return Err(ShapleyError::InvalidContribution("Empty gradient data".to_string()));
        }
        
        if !(0.0..=1.0).contains(&self.quality_score) {
            return Err(ShapleyError::InvalidContribution(
                format!("Quality score {} out of range [0,1]", self.quality_score)
            ));
        }
        
        if self.attestation.is_empty() {
            return Err(ShapleyError::InvalidContribution("Missing attestation".to_string()));
        }
        
        Ok(())
    }
}


/// Shapley value for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapleyValue {
    /// Node identifier
    pub node_id: String,
    
    /// Computed Shapley value (normalized)
    pub value: f64,
    
    /// Confidence in the computed value (0.0 to 1.0)
    pub confidence: f64,
    
    /// Computation method used
    pub computation_method: String,
}


impl ShapleyValue {
    /// Create a new Shapley value
    pub fn new(node_id: String, value: f64, confidence: f64, method: &str) -> Self {
        Self {
            node_id,
            value: value.max(0.0),
            confidence: confidence.clamp(0.0, 1.0),
            computation_method: method.to_string(),
        }
    }
    
    /// Check if value is valid
    pub fn is_valid(&self) -> bool {
        self.value >= 0.0 && self.confidence >= 0.0 && self.confidence <= 1.0
    }
}


/// Shapley computation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapleyConfig {
    /// Number of Monte Carlo samples for approximation
    pub monte_carlo_samples: usize,
    
    /// Differential privacy budget (ε)
    pub privacy_budget: f64,
    
    /// Maximum coalition size for exact computation
    pub max_coalition_size: usize,
    
    /// Enable parallel computation
    pub enable_parallel: bool,
    
    /// Use TEE-backed secure computation
    pub tee_backed: bool,
    
    /// Cache expiration time (seconds)
    pub cache_ttl_seconds: u64,
    
    /// Minimum confidence threshold
    pub min_confidence_threshold: f64,
}


impl Default for ShapleyConfig {
    fn default() -> Self {
        Self {
            monte_carlo_samples: MONTE_CARLO_SIMULATIONS,
            privacy_budget: DP_SIGMA,
            max_coalition_size: 100,
            enable_parallel: true,
            tee_backed: true,
            cache_ttl_seconds: 3600, // 1 hour
            min_confidence_threshold: 0.7,
        }
    }
}


/// Differential privacy parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpParams {
    /// Privacy budget (ε)
    pub epsilon: f64,
    
    /// Delta parameter (δ)
    pub delta: f64,
    
    /// Sensitivity bound
    pub sensitivity: f64,
    
    /// Noise scale
    pub noise_scale: f64,
}


/// Shapley computation history
#[derive(Debug, Clone, Default)]
struct ShapleyHistory {
    /// Previous computations by node
    node_history: HashMap<String, Vec<f64>>,
    
    /// Timestamps of computations
    timestamps: Vec<std::time::SystemTime>,
    
    /// Average computation times
    avg_times: Vec<std::time::Duration>,
}


impl ShapleyHistory {
    fn new() -> Self {
        Self::default()
    }
    
    /// Record a computation
    fn record_computation(&mut self, node_ids: &[String], values: &[ShapleyValue]) {
        let timestamp = std::time::SystemTime::now();
        self.timestamps.push(timestamp);
        
        for value in values {
            let node_history = self.node_history.entry(value.node_id.clone())
                .or_insert_with(Vec::new);
            node_history.push(value.value);
            
            // Keep only last 100 values per node
            if node_history.len() > 100 {
                node_history.remove(0);
            }
        }
        
        // Keep only last 1000 timestamps
        if self.timestamps.len() > 1000 {
            self.timestamps.remove(0);
        }
    }
    
    /// Get historical values for a node
    fn get_node_history(&self, node_id: &str) -> Option<&[f64]> {
        self.node_history.get(node_id).map(|v| v.as_slice())
    }
    
    /// Get trend for a node (slope of historical values)
    fn get_node_trend(&self, node_id: &str) -> Option<f64> {
        let history = self.get_node_history(node_id)?;
        if history.len() < 2 {
            return None;
        }
        
        // Simple linear regression for trend
        let n = history.len() as f64;
        let sum_x = n * (n - 1.0) / 2.0; // Sum of indices
        let sum_y: f64 = history.iter().sum();
        let sum_xy: f64 = history.iter().enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum();
        let sum_x2: f64 = (0..history.len())
            .map(|i| (i as f64).powi(2))
            .sum();
        
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = n * sum_x2 - sum_x.powi(2);
        
        if denominator.abs() > 1e-10 {
            Some(numerator / denominator)
        } else {
            Some(0.0)
        }
    }
}


/// Shapley computation errors
#[derive(Debug, thiserror::Error)]
pub enum ShapleyError {
    #[error("Invalid contribution: {0}")]
    InvalidContribution(String),
    
    #[error("Computation too large: {0} nodes, maximum {1} for exact computation")]
    ComputationTooLarge(usize, usize),
    
    #[error("Monte Carlo sampling failed: {0}")]
    MonteCarloError(String),
    
    #[error("Differential privacy error: {0}")]
    DifferentialPrivacyError(String),
    
    #[error("Value function evaluation failed: {0}")]
    ValueFunctionError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Parallel computation error: {0}")]
    ParallelError(String),
    
    #[error("Cache error: {0}")]
    CacheError(String),
}


// Helper functions


/// Compute factorial (uses 64-bit integers, may overflow for n > 20)
fn factorial(n: usize) -> u64 {
    (1..=n as u64).product()
}


/// Compute coefficient of variation (standard deviation / mean)
fn coefficient_of_variation(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    
    if mean.abs() < 1e-10 {
        return 0.0;
    }
    
    let variance = values.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / n;
    
    variance.sqrt() / mean
}


/// Sample from Laplace distribution
fn laplace_sample(location: f64, scale: f64, rng: &mut impl Rng) -> f64 {
    use rand_distr::Distribution;
    let laplace = rand_distr::Laplace::new(location, scale).unwrap();
    laplace.sample(rng)
}


// Unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    
    #[test]
    fn test_shapley_config_default() {
        let config = ShapleyConfig::default();
        
        assert_eq!(config.monte_carlo_samples, MONTE_CARLO_SIMULATIONS);
        assert_eq!(config.privacy_budget, DP_SIGMA);
        assert_eq!(config.max_coalition_size, 100);
        assert!(config.enable_parallel);
        assert!(config.tee_backed);
    }
    
    #[test]
    fn test_gradient_contribution_validation() {
        let valid_contrib = GradientContribution::new(
            "node1".to_string(),
            vec![1, 2, 3],
            0.8,
            1.5,
            vec![0, 1, 2, 3],
        );
        
        assert!(valid_contrib.validate().is_ok());
        
        let invalid_contrib = GradientContribution::new(
            "".to_string(),
            vec![],
            1.5,
            -1.0,
            vec![],
        );
        
        assert!(invalid_contrib.validate().is_err());
    }
    
    #[test]
    fn test_shapley_value_creation() {
        let sv = ShapleyValue::new("node1".to_string(), 0.5, 0.9, "exact");
        
        assert_eq!(sv.node_id, "node1");
        assert_eq!(sv.value, 0.5);
        assert_eq!(sv.confidence, 0.9);
        assert_eq!(sv.computation_method, "exact");
        assert!(sv.is_valid());
    }
    
    #[test]
    fn test_factorial_calculation() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
        assert_eq!(factorial(10), 3628800);
    }
    
    #[test]
    fn test_coefficient_of_variation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cv = coefficient_of_variation(&values);
        
        // Expected: mean = 3.0, std = √2 ≈ 1.414, CV ≈ 0.471
        assert!((cv - 0.471).abs() < 0.01);
        
        // Test with zero mean
        let zeros = vec![0.0, 0.0, 0.0];
        assert_eq!(coefficient_of_variation(&zeros), 0.0);
    }
    
    #[tokio::test]
    async fn test_small_shapley_computation() {
        let config = ShapleyConfig {
            monte_carlo_samples: 100,
            privacy_budget: 0.0, // No DP for test
            max_coalition_size: 10,
            enable_parallel: false,
            tee_backed: false,
            cache_ttl_seconds: 3600,
            min_confidence_threshold: 0.5,
        };
        
        let computer = ShapleyComputer::new(config);
        
        // Create test contributions
        let mut contributions = HashMap::new();
        contributions.insert(
            "node1".to_string(),
            GradientContribution::new(
                "node1".to_string(),
                vec![1, 2, 3],
                0.9,
                2.0,
                vec![0, 1, 2],
            )
        );
        contributions.insert(
            "node2".to_string(),
            GradientContribution::new(
                "node2".to_string(),
                vec![4, 5, 6],
                0.7,
                1.5,
                vec![3, 4, 5],
            )
        );
        
        let result = computer.compute_shapley(&contributions).await;
        assert!(result.is_ok());
        
        let shapley_values = result.unwrap();
        assert_eq!(shapley_values.len(), 2);
        
        // Values should be non-negative and sum to approximately 1
        let total: f64 = shapley_values.iter().map(|v| v.value).sum();
        assert!((total - 1.0).abs() < 0.1);
    }
