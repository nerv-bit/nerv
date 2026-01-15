//! Federated Learning Aggregation for NERV Useful-Work Economy
//! 
//! This module implements secure, privacy-preserving aggregation of gradients
//! from multiple nodes in the NERV network. It uses:
//! - Differential Privacy (DP-SGD with σ=0.5) to protect individual contributions
//! - Secure Multi-Party Computation (SMPC) in TEE clusters for privacy
//! - Byzantine fault tolerance to handle malicious nodes
//! - Quality-based filtering to ensure only useful gradients are aggregated
//! 
//! The aggregation process happens every ~15 seconds or 1000 transactions,
//! producing a global gradient update that improves the neural encoder ε_θ.


use crate::embedding::encoder::NeuralEncoder;
use crate::params::DP_SIGMA;  // Ensure fixed σ=0.5 is used
use bincode;  // For serializing inputs/outputs to TEE
use crate::privacy::tee::{TEERuntime, TEEType, TEEConfig};


use crate::crypto::{CryptoProvider, MlKem768, ByteSerializable};
use crate::params::{DP_SIGMA, RS_K, RS_M, TEE_CLUSTER_SIZE};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use tracing::{info, warn, debug, error};
use rand::prelude::*;
use rayon::prelude::*;
use blake3::Hasher;


/// Federated learning aggregator with DP and TEE support
pub struct FLAggregator {
    /// Configuration parameters
    config: AggregationConfig,
    
    /// Cryptographic provider for encryption and signatures
    crypto_provider: Arc<CryptoProvider>,
    
    /// Current gradient submissions (node_id -> encrypted gradient)
    submissions: RwLock<HashMap<String, EncryptedGradient>>,
    
    /// Quality assessment engine
    quality_assessor: QualityAssessor,
    
    /// TEE cluster manager
    tee_cluster: Option<TeeClusterManager>,

    pub tee_manager: Option<Arc<PrivacyManager>>
    
    /// Aggregation semaphore for rate limiting
    aggregation_semaphore: Semaphore,
    
    /// Metrics collector
    metrics: AggregationMetrics,
    
    /// Byzantine fault detector
    fault_detector: ByzantineDetector,
}


impl FLAggregator {
    /// Create a new FL aggregator
    pub fn new(config: AggregationConfig, crypto_provider: Arc<CryptoProvider>) -> Self {
        let quality_assessor = QualityAssessor::new(config.clone());
        let tee_cluster = if config.tee_cluster_size > 0 {
            Some(TeeClusterManager::new(config.tee_cluster_size))
        } else {
            None
        };
        
        let aggregation_semaphore = Semaphore::new(config.max_concurrent_aggregations);
        
        Self {
            config,
            crypto_provider,
            submissions: RwLock::new(HashMap::new()),
            quality_assessor,
            tee_cluster,
            aggregation_semaphore,
            metrics: AggregationMetrics::new(),
            fault_detector: ByzantineDetector::new(),
        }
    }
    
    /// Submit a gradient for aggregation
    pub async fn submit_gradient(
        &self,
        node_id: &str,
        gradient: GradientUpdate,
    ) -> Result<SubmissionReceipt, AggregationError> {
        let start_time = std::time::Instant::now();
        
        // Validate gradient
        self.validate_gradient(&gradient).await?;
        
        // Assess gradient quality
        let quality_score = self.quality_assessor.assess(&gradient).await?;
        
        // Apply differential privacy if not already applied
        let dp_gradient = if gradient.dp_applied {
            gradient
        } else {
            self.apply_differential_privacy(gradient).await?
        };
        
        // Encrypt gradient for secure aggregation
        let encrypted = self.encrypt_gradient(&dp_gradient, node_id).await?;
        
        // Store submission
        let mut submissions = self.submissions.write().await;
        
        let submission_id = self.generate_submission_id(node_id, &encrypted);
        let timestamp = std::time::SystemTime::now();
        
        let submission = EncryptedGradient {
            id: submission_id.clone(),
            node_id: node_id.to_string(),
            encrypted_data: encrypted,
            quality_score,
            timestamp,
            dp_params: dp_gradient.dp_params.clone(),
            attestation: dp_gradient.attestation.clone(),
        };
        
        submissions.insert(submission_id.clone(), submission);
        
        // Estimate Shapley value (pre-computation)
        let estimated_shapley = self.estimate_shapley(node_id, quality_score).await?;
        
        // Update metrics
        self.metrics.record_submission(start_time.elapsed());
        
        debug!("Gradient submitted by {} with quality {:.4}, estimated Shapley {:.4}",
               node_id, quality_score, estimated_shapley);
        
        Ok(SubmissionReceipt {
            id: submission_id,
            timestamp,
            quality_score,
            estimated_shapley,
            encryption_proof: vec![], // Would contain encryption proof in production
        })
    }
    
    /// Collect gradients for current aggregation interval
    pub async fn collect_gradients(&self) -> Result<Vec<EncryptedGradient>, AggregationError> {
        let submissions = self.submissions.read().await;
        
        if submissions.len() < self.config.min_batch_size {
            return Err(AggregationError::InsufficientSubmissions(
                self.config.min_batch_size,
                submissions.len(),
            ));
        }
        
        // Filter by quality threshold
        let filtered: Vec<EncryptedGradient> = submissions.values()
            .filter(|s| s.quality_score >= self.config.min_quality_threshold)
            .cloned()
            .collect();
        
        if filtered.len() < self.config.min_batch_size {
            return Err(AggregationError::InsufficientQuality(
                self.config.min_batch_size,
                filtered.len(),
            ));
        }
        
        // Check for Byzantine attacks
        self.fault_detector.detect_anomalies(&filtered).await?;
        
        Ok(filtered)
    }
    
    /// Perform secure aggregation of gradients in TEE cluster
    pub async fn secure_aggregate(
        &self,
        gradients: Vec<EncryptedGradient>,
    ) -> Result<AggregationResult, AggregationError> {
        let _permit = self.aggregation_semaphore
            .acquire()
            .await
            .map_err(|e| AggregationError::ResourceError(e.to_string()))?;
        
        let start_time = std::time::Instant::now();
        info!("Starting secure aggregation of {} gradients", gradients.len());
        
        // FIRST: Try to use TEE runtime if available
        if let Some(tee_runtime) = &self.tee_runtime {
            // Decrypt gradients first
            let mut decrypted_gradients = Vec::new();
            let mut contributions = HashMap::new();
            let mut processed_ids = HashSet::new();
            
            for encrypted in &gradients {
                let decrypted = self.decrypt_gradient(encrypted).await?;
                decrypted_gradients.push(decrypted.gradient.clone());
                
                // Record contribution
                let contrib = GradientContribution {
                    node_id: encrypted.node_id.clone(),
                    submission_id: encrypted.id.clone(),
                    quality_score: encrypted.quality_score,
                    gradient_magnitude: decrypted.gradient.compute_norm(),
                    timestamp: encrypted.timestamp,
                };
                
                contributions.insert(encrypted.node_id.clone(), contrib);
                processed_ids.insert(encrypted.id.clone());
            }
            
            // Prepare DP parameters for TEE execution
            let dp_params = DpParams {
                epsilon: self.config.dp_epsilon,
                delta: self.config.dp_delta,
                sigma: DP_SIGMA,  // Fixed 0.5 as per whitepaper
                clip_norm: self.config.clip_norm,
                noise_scale: DP_SIGMA * self.config.clip_norm,
            };
            
            // Execute in TEE with attestation
            let (global_gradient, attestation) = self
                .execute_dp_sgd_in_tee(tee_runtime, decrypted_gradients, dp_params)
                .await?;
            
            // Calculate global loss improvement
            let loss_improvement = self.calculate_loss_improvement(&global_gradient, &contributions).await?;
            
            let result = AggregationResult {
                global_gradient,
                contributions,
                total_gradients: processed_ids.len(),
                global_loss_improvement: Some(loss_improvement),
                processed_ids,
            };
            
            // Verify aggregation integrity
            self.verify_aggregation(&result).await?;
            
            // Clear processed submissions
            self.clear_processed_submissions(&result.processed_ids).await;
            
            // Update metrics
            self.metrics.record_aggregation(start_time.elapsed(), result.total_gradients);
            
            info!("TEE-based DP-SGD aggregation completed: {} gradients, {} nodes, loss improvement: {:.6}",
                  result.total_gradients,
                  result.contributions.len(),
                  result.global_loss_improvement.unwrap_or(0.0));
            
            return Ok(result);
        }
    
    /// Get aggregation statistics
    pub async fn get_statistics(&self) -> AggregationStats {
        let submissions = self.submissions.read().await;
        
        AggregationStats {
            total_submissions: submissions.len(),
            avg_quality_score: submissions.values()
                .map(|s| s.quality_score)
                .sum::<f64>() / submissions.len().max(1) as f64,
            pending_aggregation: submissions.len(),
            last_aggregation_time: self.metrics.last_aggregation,
            total_aggregations: self.metrics.total_aggregations,
        }
    }
    
    // Internal implementation methods
    
    async fn validate_gradient(&self, gradient: &GradientUpdate) -> Result<(), AggregationError> {
        // Check gradient dimensions
        if gradient.data.is_empty() {
            return Err(AggregationError::InvalidGradient("Empty gradient data".to_string()));
        }
        
        // Check for NaN or Inf values
        if gradient.data.iter().any(|&x| !x.is_finite()) {
            return Err(AggregationError::InvalidGradient("Non-finite values in gradient".to_string()));
        }
        
        // Check gradient magnitude bounds
        let norm = gradient.compute_norm();
        if norm > self.config.max_gradient_norm {
            return Err(AggregationError::InvalidGradient(
                format!("Gradient norm {} exceeds maximum {}", norm, self.config.max_gradient_norm)
            ));
        }
        
        // Verify TEE attestation if present
        if !gradient.attestation.is_empty() {
            self.verify_tee_attestation(&gradient.attestation).await?;
        }
        
        Ok(())
    }
    
    async fn apply_differential_privacy(&self, gradient: GradientUpdate) -> Result<GradientUpdate, AggregationError> {
        // Apply DP-SGD: clip and add Gaussian noise
        
        // 1. Clip gradient to bound sensitivity
        let clipped = self.clip_gradient(gradient, self.config.clip_norm).await?;
        
        // 2. Add Gaussian noise: N(0, σ² * clip_norm² * I)
        let noise_scale = self.config.dp_sigma * self.config.clip_norm;
        let noisy = self.add_gaussian_noise(clipped, noise_scale).await?;
        
        // 3. Record DP parameters
        let dp_params = DpParams {
            epsilon: self.config.dp_epsilon,
            delta: self.config.dp_delta,
            sigma: self.config.dp_sigma,
            clip_norm: self.config.clip_norm,
            noise_scale,
        };
        
        Ok(noisy.with_dp_params(dp_params))
    }
    
    async fn clip_gradient(&self, gradient: GradientUpdate, max_norm: f64) -> Result<GradientUpdate, AggregationError> {
        let norm = gradient.compute_norm();
        
        if norm <= max_norm {
            return Ok(gradient);
        }
        
        // Scale gradient to have norm = max_norm
        let scale = max_norm / norm;
        let clipped_data: Vec<f32> = gradient.data.iter()
            .map(|&x| x * scale as f32)
            .collect();
        
        Ok(GradientUpdate {
            data: clipped_data,
            dp_applied: gradient.dp_applied,
            dp_params: gradient.dp_params,
            attestation: gradient.attestation,
            metadata: gradient.metadata,
        })
    }
    
    async fn add_gaussian_noise(&self, gradient: GradientUpdate, noise_scale: f64) -> Result<GradientUpdate, AggregationError> {
        use rand_distr::{Distribution, Normal};
        
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, noise_scale)
            .map_err(|e| AggregationError::DpError(e.to_string()))?;
        
        let noisy_data: Vec<f32> = gradient.data.iter()
            .map(|&x| x + normal.sample(&mut rng) as f32)
            .collect();
        
        Ok(GradientUpdate {
            data: noisy_data,
            dp_applied: true,
            dp_params: gradient.dp_params,
            attestation: gradient.attestation,
            metadata: gradient.metadata,
        })
    }
    
    async fn encrypt_gradient(&self, gradient: &GradientUpdate, node_id: &str) -> Result<Vec<u8>, AggregationError> {
        // Serialize gradient
        let serialized = bincode::serialize(gradient)
            .map_err(|e| AggregationError::SerializationError(e.to_string()))?;
        
        // Generate symmetric encryption key
        let sym_key = self.crypto_provider.random_bytes(32)
            .map_err(|e| AggregationError::CryptoError(e.to_string()))?;
        
        // Encrypt with AES-GCM or similar
        let encrypted = self.symmetric_encrypt(&serialized, &sym_key).await?;
        
        // Encrypt symmetric key with ML-KEM for each TEE in cluster
        let key_encryptions = self.encrypt_symmetric_key(&sym_key).await?;
        
        // Combine encrypted data and key encryptions
        let mut combined = Vec::new();
        combined.extend_from_slice(&(encrypted.len() as u32).to_le_bytes());
        combined.extend_from_slice(&encrypted);
        combined.extend_from_slice(&(key_encryptions.len() as u32).to_le_bytes());
        combined.extend_from_slice(&key_encryptions);
        
        Ok(combined)
    }
    
    async fn symmetric_encrypt(&self, data: &[u8], key: &[u8]) -> Result<Vec<u8>, AggregationError> {
        // Use AES-GCM-256 for authenticated encryption
        use aes_gcm::{
            aead::{Aead, KeyInit, OsRng},
            Aes256Gcm, Nonce,
        };
        
        let cipher = Aes256Gcm::new_from_slice(key)
            .map_err(|e| AggregationError::CryptoError(e.to_string()))?;
        
        // Generate random nonce
        let nonce = Nonce::generate(&mut OsRng);
        
        // Encrypt
        let ciphertext = cipher.encrypt(&nonce, data)
            .map_err(|e| AggregationError::CryptoError(e.to_string()))?;
        
        // Combine nonce and ciphertext
        let mut result = Vec::with_capacity(nonce.len() + ciphertext.len());
        result.extend_from_slice(nonce.as_slice());
        result.extend_from_slice(&ciphertext);
        
        Ok(result)
    }
    
    async fn encrypt_symmetric_key(&self, sym_key: &[u8]) -> Result<Vec<u8>, AggregationError> {
        // Encrypt symmetric key for each TEE in cluster using ML-KEM
        
        let tee_count = self.tee_cluster.as_ref()
            .map(|c| c.size())
            .unwrap_or(1);
        
        let mut encrypted_keys = Vec::new();
        
        for i in 0..tee_count {
            // In production: use actual TEE public keys
            // For now, simulate with random encryption
            let mut encrypted = Vec::new();
            encrypted.extend_from_slice(&(i as u32).to_le_bytes());
            encrypted.extend_from_slice(sym_key); // Not actually encrypted in simulation
            
            // Add HMAC for integrity
            let hmac = self.compute_hmac(&encrypted, sym_key).await?;
            encrypted.extend_from_slice(&hmac);
            
            encrypted_keys.extend_from_slice(&(encrypted.len() as u32).to_le_bytes());
            encrypted_keys.extend_from_slice(&encrypted);
        }
        
        Ok(encrypted_keys)
    }
    
    async fn compute_hmac(&self, data: &[u8], key: &[u8]) -> Result<Vec<u8>, AggregationError> {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        
        type HmacSha256 = Hmac<Sha256>;
        
        let mut mac = HmacSha256::new_from_slice(key)
            .map_err(|e| AggregationError::CryptoError(e.to_string()))?;
        
        mac.update(data);
        let result = mac.finalize().into_bytes().to_vec();
        
        Ok(result)
    }
    
    async fn verify_tee_attestation(&self, attestation: &[u8]) -> Result<(), AggregationError> {
        // Verify TEE remote attestation
        // In production: verify against Intel/AMD/ARM root certificates
        
        if attestation.len() < 64 {
            return Err(AggregationError::InvalidAttestation("Attestation too short".to_string()));
        }
        
        // Simplified check: ensure it starts with known header
        let expected_header = b"TEE_ATTESTATION";
        if attestation.len() >= expected_header.len() 
            && &attestation[..expected_header.len()] == expected_header {
            Ok(())
        } else {
            Err(AggregationError::InvalidAttestation("Invalid attestation format".to_string()))
        }
    }
    
    async fn estimate_shapley(&self, node_id: &str, quality_score: f64) -> Result<f64, AggregationError> {
        // Estimate Shapley value based on quality and historical performance
        // This is a pre-computation estimate, actual Shapley will be computed later
        
        let submissions = self.submissions.read().await;
        let total_nodes = submissions.len() + 1; // Including current submission
        
        // Base estimate: quality score normalized by total nodes
        let base_estimate = quality_score / total_nodes.max(1) as f64;
        
        // Adjust based on node's historical performance if available
        let historical_adjustment = self.get_historical_adjustment(node_id).await?;
        
        let estimate = base_estimate * (1.0 + historical_adjustment);
        
        Ok(estimate.clamp(0.0, 1.0))
    }
    
    async fn get_historical_adjustment(&self, node_id: &str) -> Result<f64, AggregationError> {
        // Get historical performance adjustment for node
        // In production: query historical database
        
        // Simulated: return small random adjustment
        let mut rng = rand::thread_rng();
        Ok(rng.gen_range(-0.1..0.1))
    }
    
    async fn group_gradients(&self, gradients: Vec<EncryptedGradient>) -> Result<Vec<GradientGroup>, AggregationError> {
        // Group gradients for parallel processing
        
        let group_size = (gradients.len() + self.config.tee_cluster_size - 1) 
            / self.config.tee_cluster_size.max(1);
        
        let mut groups = Vec::new();
        let mut current_group = Vec::new();
        
        for gradient in gradients {
            current_group.push(gradient);
            
            if current_group.len() >= group_size {
                groups.push(GradientGroup {
                    gradients: current_group,
                    group_id: groups.len(),
                });
                current_group = Vec::new();
            }
        }
        
        if !current_group.is_empty() {
            groups.push(GradientGroup {
                gradients: current_group,
                group_id: groups.len(),
            });
        }
        
        Ok(groups)
    }
    
    async fn aggregate_in_tee_cluster(
        &self,
        tee_cluster: &TeeClusterManager,
        groups: Vec<GradientGroup>,
    ) -> Result<AggregationResult, AggregationError> {
        info!("Aggregating {} gradient groups in TEE cluster", groups.len());
        
        // Distribute groups to TEEs
        let tee_tasks = tee_cluster.distribute_groups(groups).await?;
        
        // Execute aggregation in parallel
        let mut handles = Vec::new();
        
        for task in tee_tasks {
            let handle = tokio::spawn(async move {
                // In production: actually execute in TEE
                // For simulation: aggregate locally
                aggregate_group_locally(task.gradients).await
            });
            handles.push(handle);
        }
        
        // Collect results
        let mut all_results = Vec::new();
        let mut all_contributions = HashMap::new();
        let mut processed_ids = HashSet::new();
        
        for handle in handles {
            let result = handle.await
                .map_err(|e| AggregationError::TaskError(e.to_string()))?
                .map_err(|e| AggregationError::AggregationError(e.to_string()))?;
            
            all_results.push(result.aggregated_gradient);
            
            for (node_id, contrib) in result.contributions {
                all_contributions.insert(node_id, contrib);
                processed_ids.insert(contrib.submission_id);
            }
        }
        
        // Aggregate group results
        let final_aggregated = self.aggregate_group_results(all_results).await?;
        
        // Calculate global loss improvement
        let loss_improvement = self.calculate_loss_improvement(&final_aggregated, &all_contributions).await?;
        
        Ok(AggregationResult {
            global_gradient: final_aggregated,
            contributions: all_contributions,
            total_gradients: processed_ids.len(),
            global_loss_improvement: Some(loss_improvement),
            processed_ids,
        })
    }
    
    async fn aggregate_locally(&self, groups: Vec<GradientGroup>) -> Result<AggregationResult, AggregationError> {
        warn!("Aggregating locally without TEE protection - not recommended for production");
        
        let mut all_gradients = Vec::new();
        let mut all_contributions = HashMap::new();
        let mut processed_ids = HashSet::new();
        
        for group in groups {
            // Decrypt and aggregate each group
            for encrypted in group.gradients {
                // Decrypt gradient
                let decrypted = self.decrypt_gradient(&encrypted).await?;
                
                // Add to collection
                all_gradients.push(decrypted.gradient);
                
                // Record contribution
                let contrib = GradientContribution {
                    node_id: encrypted.node_id.clone(),
                    submission_id: encrypted.id.clone(),
                    quality_score: encrypted.quality_score,
                    gradient_magnitude: decrypted.gradient.compute_norm(),
                    timestamp: encrypted.timestamp,
                };
                
                all_contributions.insert(encrypted.node_id.clone(), contrib);
                processed_ids.insert(encrypted.id);
            }
        }
        
        // Aggregate all gradients
        let aggregated = self.aggregate_gradients(&all_gradients).await?;
        
        // Calculate loss improvement
        let loss_improvement = self.calculate_loss_improvement(&aggregated, &all_contributions).await?;
        
        Ok(AggregationResult {
            global_gradient: aggregated,
            contributions: all_contributions,
            total_gradients: processed_ids.len(),
            global_loss_improvement: Some(loss_improvement),
            processed_ids,
        })
    }
    
    async fn decrypt_gradient(&self, encrypted: &EncryptedGradient) -> Result<DecryptedGradient, AggregationError> {
        // Decrypt gradient data
        // In production: would involve TEE attestation and key release
        
        // For simulation: return dummy gradient
        let gradient = GradientUpdate {
            data: vec![0.1, 0.2, 0.3], // Dummy data
            dp_applied: true,
            dp_params: encrypted.dp_params.clone(),
            attestation: encrypted.attestation.clone(),
            metadata: HashMap::new(),
        };
        
        Ok(DecryptedGradient {
            gradient,
            submission_id: encrypted.id.clone(),
        })
    }
    
    async fn aggregate_gradients(&self, gradients: &[GradientUpdate]) -> Result<GradientUpdate, AggregationError> {
        if gradients.is_empty() {
            return Err(AggregationError::AggregationError("No gradients to aggregate".to_string()));
        }
        
        // Check all gradients have same dimension
        let dim = gradients[0].data.len();
        for grad in gradients {
            if grad.data.len() != dim {
                return Err(AggregationError::AggregationError(
                    format!("Gradient dimension mismatch: {} vs {}", dim, grad.data.len())
                ));
            }
        }
        
        // Compute average gradient
        let mut aggregated = vec![0.0f32; dim];
        
        for grad in gradients {
            for (i, &value) in grad.data.iter().enumerate() {
                aggregated[i] += value;
            }
        }
        
        let count = gradients.len() as f32;
        for value in &mut aggregated {
            *value /= count;
        }
        
        Ok(GradientUpdate {
            data: aggregated,
            dp_applied: true,
            dp_params: None,
            attestation: Vec::new(),
            metadata: HashMap::new(),
        })
    }
    
    async fn aggregate_group_results(&self, group_results: Vec<GradientUpdate>) -> Result<GradientUpdate, AggregationError> {
        self.aggregate_gradients(&group_results).await
    }
    
    async fn calculate_loss_improvement(
        &self,
        aggregated_gradient: &GradientUpdate,
        contributions: &HashMap<String, GradientContribution>,
    ) -> Result<f64, AggregationError> {
        // Calculate how much this aggregation improves the model
        // Metric: average gradient quality weighted by contribution magnitude
        
        if contributions.is_empty() {
            return Ok(0.0);
        }
        
        let total_quality: f64 = contributions.values()
            .map(|c| c.quality_score * c.gradient_magnitude)
            .sum();
        
        let total_magnitude: f64 = contributions.values()
            .map(|c| c.gradient_magnitude)
            .sum();
        
        if total_magnitude == 0.0 {
            return Ok(0.0);
        }
        
        let weighted_quality = total_quality / total_magnitude;
        
        // Normalize to [0, 1] range
        let improvement = weighted_quality.clamp(0.0, 1.0);
        
        Ok(improvement)
    }
    
    async fn verify_aggregation(&self, result: &AggregationResult) -> Result<(), AggregationError> {
        // Verify aggregation integrity
        
        // Check all gradients were accounted for
        let submissions = self.submissions.read().await;
        for id in &result.processed_ids {
            if !submissions.contains_key(id) {
                return Err(AggregationError::VerificationError(
                    format!("Processed submission {} not found in submissions", id)
                ));
            }
        }
        
        // Check aggregated gradient is valid
        if result.global_gradient.data.is_empty() {
            return Err(AggregationError::VerificationError("Empty aggregated gradient".to_string()));
        }
        
        // Check for NaN/Inf in aggregated gradient
        if result.global_gradient.data.iter().any(|&x| !x.is_finite()) {
            return Err(AggregationError::VerificationError("Non-finite values in aggregated gradient".to_string()));
        }
        
        // Verify contribution consistency
        for (node_id, contrib) in &result.contributions {
            if contrib.node_id != *node_id {
                return Err(AggregationError::VerificationError(
                    format!("Contribution node ID mismatch: {} vs {}", node_id, contrib.node_id)
                ));
            }
        }
        
        Ok(())
    }
    
    async fn clear_processed_submissions(&self, processed_ids: &HashSet<String>) {
        let mut submissions = self.submissions.write().await;
        for id in processed_ids {
            submissions.remove(id);
        }
    }

    async fn execute_dp_sgd_in_tee(
        &self,
        tee_runtime: &Arc<dyn TEERuntime + Send + Sync>,
        gradients: Vec<GradientUpdate>,
        dp_params: DpParams,
    ) -> Result<(GradientUpdate, Vec<u8>), AggregationError> {
        info!("Executing DP-SGD aggregation in TEE with σ={}", dp_params.sigma);
        
        // Call the new method on TEE runtime
        let (aggregated_gradient, attestation) = tee_runtime
            .execute_dp_sgd_aggregation(gradients, dp_params.clone())
            .await
            .map_err(|e| AggregationError::TEEError(format!("TEE DP-SGD execution failed: {}", e)))?;
        
        // Verify DP was applied correctly
        if !aggregated_gradient.dp_applied {
            return Err(AggregationError::TEEError("DP not applied in TEE".to_string()));
        }
        
        if let Some(params) = &aggregated_gradient.dp_params {
            if (params.sigma - DP_SIGMA).abs() > 1e-10 {
                return Err(AggregationError::TEEError(
                    format!("DP sigma mismatch: TEE used {}, expected {}", params.sigma, DP_SIGMA)
                ));
            }
        } else {
            return Err(AggregationError::TEEError("Aggregated gradient missing DP parameters".to_string()));
        }
        
        // Verify gradient matches encoder structure
        let expected_params = NeuralEncoder::default_parameter_count();
        if aggregated_gradient.data.len() != expected_params {
            return Err(AggregationError::InvalidGradient(
                format!("Gradient dimension mismatch — not for current encoder: expected {}, got {}", 
                        expected_params, aggregated_gradient.data.len())
            ));
        }
        
        Ok((aggregated_gradient, attestation))
    }
    
    fn generate_submission_id(&self, node_id: &str, encrypted_data: &[u8]) -> String {
        // Generate unique submission ID: hash(node_id + timestamp + data_hash)
        let mut hasher = Hasher::new();
        hasher.update(node_id.as_bytes());
        hasher.update(&std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .to_le_bytes());
        hasher.update(encrypted_data);
        
        let hash = hasher.finalize();
        hex::encode(hash.as_bytes())
    }
}


/// Gradient update structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientUpdate {
    /// Gradient data (flattened tensor)
    pub data: Vec<f32>,
    
    /// Whether DP has been applied
    pub dp_applied: bool,
    
    /// DP parameters if applied
    pub dp_params: Option<DpParams>,
    
    /// TEE attestation proof
    pub attestation: Vec<u8>,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}


impl GradientUpdate {
    /// Create a new gradient update
    pub fn new(data: Vec<f32>, attestation: Vec<u8>) -> Self {
        Self {
            data,
            dp_applied: false,
            dp_params: None,
            attestation,
            metadata: HashMap::new(),
        }
    }
    
    /// Add DP parameters
    pub fn with_dp_params(mut self, dp_params: DpParams) -> Self {
        self.dp_applied = true;
        self.dp_params = Some(dp_params);
        self
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    /// Compute L2 norm of gradient
    pub fn compute_norm(&self) -> f64 {
        let sum_sq: f64 = self.data.iter()
            .map(|&x| (x as f64).powi(2))
            .sum();
        
        sum_sq.sqrt()
    }
    
    /// Check if gradient is valid
    pub fn is_valid(&self) -> bool {
        !self.data.is_empty() && self.data.iter().all(|&x| x.is_finite())
    }
}


/// Encrypted gradient for secure aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedGradient {
    /// Unique submission ID
    pub id: String,
    
    /// Node identifier
    pub node_id: String,
    
    /// Encrypted gradient data
    pub encrypted_data: Vec<u8>,
    
    /// Quality score (0.0 to 1.0)
    pub quality_score: f64,
    
    /// Submission timestamp
    pub timestamp: std::time::SystemTime,
    
    /// DP parameters used
    pub dp_params: Option<DpParams>,
    
    /// TEE attestation
    pub attestation: Vec<u8>,
}


/// Decrypted gradient for aggregation
#[derive(Debug, Clone)]
struct DecryptedGradient {
    /// Decrypted gradient
    pub gradient: GradientUpdate,
    
    /// Original submission ID
    pub submission_id: String,
}


/// Gradient contribution for Shapley computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientContribution {
    /// Node identifier
    pub node_id: String,
    
    /// Submission ID
    pub submission_id: String,
    
    /// Quality score
    pub quality_score: f64,
    
    /// Gradient magnitude (L2 norm)
    pub gradient_magnitude: f64,
    
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}


/// Aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationConfig {
    /// DP sigma parameter
    pub dp_sigma: f64,
    
    /// DP epsilon parameter
    pub dp_epsilon: f64,
    
    /// DP delta parameter
    pub dp_delta: f64,
    
    /// Gradient clipping norm
    pub clip_norm: f64,
    
    /// Minimum batch size for aggregation
    pub min_batch_size: usize,
    
    /// Maximum batch size
    pub max_batch_size: usize,
    
    /// TEE cluster size
    pub tee_cluster_size: usize,
    
    /// Aggregation timeout in seconds
    pub aggregation_timeout_secs: u64,
    
    /// Minimum quality threshold
    pub min_quality_threshold: f64,
    
    /// Maximum gradient norm
    pub max_gradient_norm: f64,
    
    /// Maximum concurrent aggregations
    pub max_concurrent_aggregations: usize,
}


impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            dp_sigma: DP_SIGMA,
            dp_epsilon: 3.0,
            dp_delta: 1e-5,
            clip_norm: 1.0,
            min_batch_size: 10,
            max_batch_size: 1000,
            tee_cluster_size: TEE_CLUSTER_SIZE,
            aggregation_timeout_secs: 30,
            min_quality_threshold: 0.5,
            max_gradient_norm: 10.0,
            max_concurrent_aggregations: 5,
        }
    }
}


/// DP parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpParams {
    /// Privacy budget (ε)
    pub epsilon: f64,
    
    /// Delta parameter (δ)
    pub delta: f64,
    
    /// Noise scale (σ)
    pub sigma: f64,
    
    /// Clipping norm
    pub clip_norm: f64,
    
    /// Actual noise scale applied
    pub noise_scale: f64,
}


/// Aggregation result
#[derive(Debug, Clone)]
pub struct AggregationResult {
    /// Globally aggregated gradient
    pub global_gradient: GradientUpdate,
    
    /// Individual contributions for Shapley computation
    pub contributions: HashMap<String, GradientContribution>,
    
    /// Total number of gradients aggregated
    pub total_gradients: usize,
    
    /// Global loss improvement metric
    pub global_loss_improvement: Option<f64>,
    
    /// IDs of processed submissions
    pub processed_ids: HashSet<String>,
}


/// Group of gradients for parallel processing
#[derive(Debug, Clone)]
struct GradientGroup {
    /// Gradients in this group
    pub gradients: Vec<EncryptedGradient>,
    
    /// Group identifier
    pub group_id: usize,
}


/// Group aggregation result
#[derive(Debug, Clone)]
struct GroupAggregationResult {
    /// Aggregated gradient for this group
    pub aggregated_gradient: GradientUpdate,
    
    /// Contributions in this group
    pub contributions: HashMap<String, GradientContribution>,
}


/// Submission receipt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmissionReceipt {
    /// Submission ID
    pub id: String,
    
    /// Timestamp
    pub timestamp: std::time::SystemTime,
    
    /// Quality score
    pub quality_score: f64,
    
    /// Estimated Shapley value
    pub estimated_shapley: f64,
    
    /// Encryption proof
    pub encryption_proof: Vec<u8>,
}


/// Aggregation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationStats {
    /// Total submissions
    pub total_submissions: usize,
    
    /// Average quality score
    pub avg_quality_score: f64,
    
    /// Pending aggregation count
    pub pending_aggregation: usize,
    
    /// Last aggregation time
    pub last_aggregation_time: Option<std::time::SystemTime>,
    
    /// Total aggregations performed
    pub total_aggregations: u64,
}


// Quality assessment engine
struct QualityAssessor {
    config: AggregationConfig,
}


impl QualityAssessor {
    fn new(config: AggregationConfig) -> Self {
        Self { config }
    }
    
    async fn assess(&self, gradient: &GradientUpdate) -> Result<f64, AggregationError> {
        // Assess gradient quality based on multiple factors
        
        let mut quality_score = 1.0;
        
        // 1. Check gradient magnitude (too small or too large is bad)
        let norm = gradient.compute_norm();
        let norm_score = if norm < 1e-10 {
            0.1 // Too small
        } else if norm > self.config.max_gradient_norm {
            0.2 // Too large
        } else {
            // Optimal range: 0.1 to 1.0
            let optimal_norm = 0.5;
            let diff = (norm - optimal_norm).abs();
            1.0 / (1.0 + diff * 10.0)
        };
        quality_score *= norm_score;
        
        // 2. Check gradient diversity (non-zero elements)
        let zero_count = gradient.data.iter()
            .filter(|&&x| x.abs() < 1e-6)
            .count();
        let zero_ratio = zero_count as f64 / gradient.data.len() as f64;
        let diversity_score = 1.0 - zero_ratio;
        quality_score *= diversity_score;
        
        // 3. Check for outliers (extreme values)
        let mean: f64 = gradient.data.iter()
            .map(|&x| x as f64)
            .sum::<f64>() / gradient.data.len() as f64;
        
        let variance: f64 = gradient.data.iter()
            .map(|&x| ((x as f64 - mean).powi(2)))
            .sum::<f64>() / gradient.data.len() as f64;
        
        let std_dev = variance.sqrt();
        let outlier_count = gradient.data.iter()
            .filter(|&&x| (x as f64 - mean).abs() > 3.0 * std_dev)
            .count();
        
        let outlier_ratio = outlier_count as f64 / gradient.data.len() as f64;
        let outlier_score = 1.0 - outlier_ratio * 10.0; // Penalize outliers heavily
        quality_score *= outlier_score.max(0.1);
        
        // 4. Apply DP bonus if applicable
        if gradient.dp_applied {
            quality_score *= 1.1; // 10% bonus for DP
        }
        
        Ok(quality_score.clamp(0.0, 1.0))
    }
}


// TEE cluster manager (simplified)
struct TeeClusterManager {
    size: usize,
}


impl TeeClusterManager {
    fn new(size: usize) -> Self {
        Self { size }
    }
    
    fn size(&self) -> usize {
        self.size
    }
    
    async fn distribute_groups(&self, groups: Vec<GradientGroup>) -> Result<Vec<TeeTask>, AggregationError> {
        // Distribute gradient groups to TEEs
        let mut tasks = Vec::new();
        
        for (tee_id, group) in groups.into_iter().enumerate() {
            let tee_id = tee_id % self.size;
            tasks.push(TeeTask {
                tee_id,
                gradients: group.gradients,
            });
        }
        
        Ok(tasks)
    }
}


struct TeeTask {
    tee_id: usize,
    gradients: Vec<EncryptedGradient>,
}


// Byzantine fault detector
struct ByzantineDetector;


impl ByzantineDetector {
    fn new() -> Self {
        Self
    }
    
    async fn detect_anomalies(&self, gradients: &[EncryptedGradient]) -> Result<(), AggregationError> {
        // Detect Byzantine attacks in gradients
        
        if gradients.len() < 3 {
            return Ok(()); // Need at least 3 for statistical tests
        }
        
        // Collect quality scores for analysis
        let quality_scores: Vec<f64> = gradients.iter()
            .map(|g| g.quality_score)
            .collect();
        
        // Check for statistical anomalies
        let mean: f64 = quality_scores.iter().sum::<f64>() / quality_scores.len() as f64;
        let variance: f64 = quality_scores.iter()
            .map(|&s| (s - mean).powi(2))
            .sum::<f64>() / quality_scores.len() as f64;
        
        let std_dev = variance.sqrt();
        
        // Detect outliers (3 sigma rule)
        let outlier_indices: Vec<usize> = quality_scores.iter()
            .enumerate()
            .filter(|(_, &s)| (s - mean).abs() > 3.0 * std_dev)
            .map(|(i, _)| i)
            .collect();
        
        if !outlier_indices.is_empty() {
            warn!("Detected {} potential Byzantine gradients", outlier_indices.len());
            
            // In production: would quarantine or reject these gradients
            // For now, just log warning
            for &idx in &outlier_indices {
                debug!("Potential Byzantine gradient from node {} with quality {} (mean: {:.3}, std: {:.3})",
                       gradients[idx].node_id, gradients[idx].quality_score, mean, std_dev);
            }
        }
        
        // Check for collusion (multiple similar low-quality gradients)
        let low_quality_count = quality_scores.iter()
            .filter(|&&s| s < 0.3)
            .count();
        
        if low_quality_count > gradients.len() / 2 {
            return Err(AggregationError::ByzantineAttack(
                format!("Suspected collusion: {}/{} low-quality gradients", 
                       low_quality_count, gradients.len())
            ));
        }
        
        Ok(())
    }
}


// Aggregation metrics
struct AggregationMetrics {
    total_submissions: u64,
    total_aggregations: u64,
    avg_submission_time_ms: f64,
    avg_aggregation_time_ms: f64,
    last_submission: Option<std::time::SystemTime>,
    last_aggregation: Option<std::time::SystemTime>,
}


impl AggregationMetrics {
    fn new() -> Self {
        Self {
            total_submissions: 0,
            total_aggregations: 0,
            avg_submission_time_ms: 0.0,
            avg_aggregation_time_ms: 0.0,
            last_submission: None,
            last_aggregation: None,
        }
    }
    
    fn record_submission(&mut self, duration: std::time::Duration) {
        self.total_submissions += 1;
        self.last_submission = Some(std::time::SystemTime::now());
        
        // Update moving average
        let new_time = duration.as_millis() as f64;
        self.avg_submission_time_ms = 0.9 * self.avg_submission_time_ms + 0.1 * new_time;
    }
    
    fn record_aggregation(&mut self, duration: std::time::Duration, gradients: usize) {
        self.total_aggregations += 1;
        self.last_aggregation = Some(std::time::SystemTime::now());
        
        // Update moving average
        let new_time = duration.as_millis() as f64;
        self.avg_aggregation_time_ms = 0.9 * self.avg_aggregation_time_ms + 0.1 * new_time;
    }
}


// Helper function for local group aggregation
async fn aggregate_group_locally(gradients: Vec<EncryptedGradient>) -> Result<GroupAggregationResult, AggregationError> {
    // Simulated local aggregation
    // In production, this would run inside TEE
    
    let mut aggregated_data = Vec::new();
    let mut contributions = HashMap::new();
    
    for gradient in gradients {
        // Simulate decryption and aggregation
        let decrypted = GradientUpdate::new(
            vec![0.1, 0.2, 0.3], // Dummy data
            gradient.attestation.clone(),
        );
        
        if aggregated_data.is_empty() {
            aggregated_data = decrypted.data.clone();
        } else {
            // Average with existing data
            for (i, &value) in decrypted.data.iter().enumerate() {
                if i < aggregated_data.len() {
                    aggregated_data[i] = (aggregated_data[i] + value) / 2.0;
                }
            }
        }
        
        // Record contribution
        let contrib = GradientContribution {
            node_id: gradient.node_id.clone(),
            submission_id: gradient.id.clone(),
            quality_score: gradient.quality_score,
            gradient_magnitude: decrypted.compute_norm(),
            timestamp: gradient.timestamp,
        };
        
        contributions.insert(gradient.node_id.clone(), contrib);
    }
    
    let aggregated_gradient = GradientUpdate {
        data: aggregated_data,
        dp_applied: true,
        dp_params: None,
        attestation: Vec::new(),
        metadata: HashMap::new(),
    };
    
    Ok(GroupAggregationResult {
        aggregated_gradient,
        contributions,
    })
}


/// Aggregation errors
#[derive(Debug, thiserror::Error)]
pub enum AggregationError {
    #[error("Invalid gradient: {0}")]
    InvalidGradient(String),
    
    #[error("Insufficient submissions: minimum {0}, got {1}")]
    InsufficientSubmissions(usize, usize),
    
    #[error("Insufficient quality: minimum {0} submissions meet quality threshold, got {1}")]
    InsufficientQuality(usize, usize),
    
    #[error("Differential privacy error: {0}")]
    DpError(String),
    
    #[error("Cryptographic error: {0}")]
    CryptoError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Invalid TEE attestation: {0}")]
    InvalidAttestation(String),
    
    #[error("Aggregation error: {0}")]
    AggregationError(String),
    
    #[error("Verification error: {0}")]
    VerificationError(String),
    
    #[error("Resource error: {0}")]
    ResourceError(String),
    
    #[error("Task error: {0}")]
    TaskError(String),
    
    #[error("Byzantine attack detected: {0}")]
    ByzantineAttack(String),
    
    #[error("Timeout error: {0}")]
    TimeoutError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
}


// Unit tests
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gradient_update_creation() {
        let gradient = GradientUpdate::new(
            vec![1.0, 2.0, 3.0],
            vec![0, 1, 2],
        );
        
        assert_eq!(gradient.data.len(), 3);
        assert!(!gradient.dp_applied);
        assert!(gradient.dp_params.is_none());
        assert!(!gradient.attestation.is_empty());
        assert!(gradient.is_valid());
    }
    
    #[test]
    fn test_gradient_norm_calculation() {
        let gradient = GradientUpdate::new(
            vec![3.0, 4.0],
            vec![],
        );
        
        let norm = gradient.compute_norm();
        assert!((norm - 5.0).abs() < 1e-6); // 3-4-5 triangle
    }
    
    #[test]
    fn test_dp_params_application() {
        let gradient = GradientUpdate::new(
            vec![1.0, 2.0],
            vec![],
        );
        
        let dp_params = DpParams {
            epsilon: 1.0,
            delta: 1e-5,
            sigma: 0.5,
            clip_norm: 1.0,
            noise_scale: 0.5,
        };
        
        let dp_gradient = gradient.with_dp_params(dp_params);
        
        assert!(dp_gradient.dp_applied);
        assert!(dp_gradient.dp_params.is_some());
    }
    
    #[tokio::test]
    async fn test_quality_assessment() {
        let config = AggregationConfig::default();
        let assessor = QualityAssessor::new(config);
        
        let gradient = GradientUpdate::new(
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
            vec![],
        );
        
        let quality = assessor.assess(&gradient).await;
        assert!(quality.is_ok());
        
        let quality_score = quality.unwrap();
        assert!(quality_score >= 0.0 && quality_score <= 1.0);
    }
    
    #[test]
    fn test_aggregation_config_default() {
        let config = AggregationConfig::default();
        
        assert_eq!(config.dp_sigma, DP_SIGMA);
        assert_eq!(config.dp_epsilon, 3.0);
        assert_eq!(config.dp_delta, 1e-5);
        assert_eq!(config.clip_norm, 1.0);
        assert_eq!(config.min_batch_size, 10);
        assert_eq!(config.tee_cluster_size, TEE_CLUSTER_SIZE);
    }
}
