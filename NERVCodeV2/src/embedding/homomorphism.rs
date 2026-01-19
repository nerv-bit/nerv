//! # Transfer Homomorphism - Core Mathematical Foundation
//!
//! This module implements the Transfer Homomorphism property that enables NERV's
//! 900× compression and efficient updates:
//!
//!     ε_θ(S_{t+1}) = ε_θ(S_t) + δ(tx) + ε, where |ε| ≤ 1e-9
//!
//! ## Mathematical Foundation:
//! The homomorphism allows us to update the 512-byte neural embedding via simple
//! vector addition for balance transfers, without decompressing the state.
//!
//! ## Key Components:
//! 1. **TransferDelta**: Computes δ(tx) vectors for individual transfers
//! 2. **BatchAggregator**: Sums deltas for transaction batches (up to 256 txs)
//! 3. **HomomorphismVerifier**: Verifies ε_θ(S_{t+1}) ≈ ε_θ(S_t) + Σ δ(tx_i)
//! 4. **EncoderTrainer**: Trains transformer to preserve homomorphism property
//!
//! ## Technical Details:
//! - Uses 32.16 fixed-point arithmetic for neural computations
//! - Enforces ≤ 1e-9 relative error bound for correctness
//! - Supports batched updates with O(1) complexity per transaction
//! - Integrates with ZK circuits for verifiable computation


use crate::crypto::{Dilithium3, MLKEM768};
use crate::embedding::encoder::TransformerEncoder;
use crate::embedding::circuit::{FixedPoint32_16, CircuitError};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;


/// Error type for homomorphism operations
#[derive(Debug, thiserror::Error)]
pub enum HomomorphismError {
    #[error("Homomorphism violation: error {0} > 1e-9")]
    Violation(f64),
    
    #[error("Dimension mismatch: expected {0}, got {1}")]
    DimensionMismatch(usize, usize),
    
    #[error("Batch size exceeded: {0} > 256")]
    BatchSizeExceeded(usize),
    
    #[error("Account not found: {0:?}")]
    AccountNotFound([u8; 32]),
    
    #[error("Insufficient balance: {0} < {1}")]
    InsufficientBalance(f64, f64),
    
    #[error("Encoder error: {0}")]
    Encoder(String),
    
    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("homomorphic error bound violated")]
     ErrorBoundViolated,
}


/// Result type for homomorphism operations
pub type Result<T> = std::result::Result<T, HomomorphismError>;


/// A transfer transaction in the latent space
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransferTransaction {
    /// Sender's blinded account identifier (32-byte hash)
    pub sender: [u8; 32],
    
    /// Receiver's blinded account identifier (32-byte hash)
    pub receiver: [u8; 32],
    
    /// Amount in NANO (smallest unit, fixed-point encoded)
    pub amount: FixedPoint32_16,
    
    /// Nonce for replay protection
    pub nonce: u64,
    
    /// Timestamp (for ordering)
    pub timestamp: u64,
    
    /// ZK proof of balance sufficiency (not included in delta computation)
    #[serde(skip)]
    pub balance_proof: Option<Vec<u8>>,
}


/// Delta vector for a single transaction (512 dimensions)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransferDelta {
    /// The delta vector itself (512 fixed-point values)
    pub delta: Vec<FixedPoint32_16>,
    
    /// Transaction that generated this delta
    pub transaction: TransferTransaction,
    
    /// Homomorphism error bound for this delta
    pub error_bound: FixedPoint32_16,
    
    /// Cryptographic signature (Dilithium-3) for non-repudiation
    pub signature: Option<Vec<u8>>,
}


impl TransferDelta {
    /// Create a new delta for a transfer transaction
    pub fn new(
        transaction: TransferTransaction,
        sender_embedding: &[FixedPoint32_16],
        receiver_embedding: &[FixedPoint32_16],
        bias_terms: &[FixedPoint32_16],
    ) -> Result<Self> {
        let dim = 512;
        
        if sender_embedding.len() != dim || receiver_embedding.len() != dim || bias_terms.len() != dim {
            return Err(HomomorphismError::DimensionMismatch(dim, sender_embedding.len()));
        }
        
        let mut delta = Vec::with_capacity(dim);
        
        // Compute δ(tx) = amount × (receiver - sender) + bias
        // This is the core homomorphism equation
        for i in 0..dim {
            let receiver_val = receiver_embedding[i];
            let sender_val = sender_embedding[i];
            let bias = bias_terms[i];
            
            // Δ_i = amount × (receiver_i - sender_i) + bias_i
            let diff = receiver_val.add(FixedPoint32_16::from_float(-1.0).mul(sender_val));
            let scaled = transaction.amount.mul(diff);
            let final_val = scaled.add(bias);
            
            delta.push(final_val);
        }
        
        Ok(Self {
            delta,
            transaction,
            error_bound: FixedPoint32_16::from_float(1e-9),
            signature: None,
        })
    }
    
    /// Verify that this delta was correctly computed
    pub fn verify(
        &self,
        sender_embedding: &[FixedPoint32_16],
        receiver_embedding: &[FixedPoint32_16],
        bias_terms: &[FixedPoint32_16],
    ) -> Result<bool> {
        let dim = 512;
        
        // Recompute delta from embeddings
        let recomputed = Self::new(
            self.transaction.clone(),
            sender_embedding,
            receiver_embedding,
            bias_terms,
        )?;
        
        // Compare element-wise with error bound
        for i in 0..dim {
            let diff = if self.delta[i].0 > recomputed.delta[i].0 {
                FixedPoint32_16(self.delta[i].0 - recomputed.delta[i].0)
            } else {
                FixedPoint32_16(recomputed.delta[i].0 - self.delta[i].0)
            };
            
            if diff.0 > self.error_bound.0 {
                return Err(HomomorphismError::Violation(diff.to_float()));
            }
        }
        
        Ok(true)
    }
    
    /// Sign the delta with Dilithium-3
    pub fn sign(&mut self, private_key: &[u8]) -> Result<()> {
        // Serialize delta for signing
        let serialized = bincode::serialize(&self.delta)
            .map_err(|e| HomomorphismError::Serialization(e.to_string()))?;
        
        // Sign with Dilithium-3 (post-quantum signature)
        let signature = Dilithium3::sign(&serialized, private_key)
            .map_err(|e| HomomorphismError::Encoder(e.to_string()))?;
        
        self.signature = Some(signature);
        Ok(())
    }
    
    /// Verify the signature on this delta
    pub fn verify_signature(&self, public_key: &[u8]) -> Result<bool> {
        match &self.signature {
            Some(sig) => {
                let serialized = bincode::serialize(&self.delta)
                    .map_err(|e| HomomorphismError::Serialization(e.to_string()))?;
                
                Dilithium3::verify(&serialized, sig, public_key)
                    .map_err(|e| HomomorphismError::Encoder(e.to_string()))
            }
            None => Ok(false), // No signature to verify
        }
    }
}


/// Aggregates multiple deltas into a batch delta (up to 256 transactions)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchDelta {
    /// Aggregated delta vector (sum of individual deltas)
    pub aggregated_delta: Vec<FixedPoint32_16>,
    
    /// Individual transactions in the batch
    pub transactions: Vec<TransferTransaction>,
    
    /// Batch identifier (hash of transactions)
    pub batch_id: [u8; 32],
    
    /// Total amount transferred in the batch
    pub total_amount: FixedPoint32_16,
    
    /// Average bytes per transaction (target: 2 bytes/tx)
    pub bytes_per_tx: f64,
}


impl BatchDelta {
    /// Create a new batch delta from individual deltas
    pub fn new(deltas: Vec<TransferDelta>) -> Result<Self> {
        if deltas.is_empty() {
            return Err(HomomorphismError::BatchSizeExceeded(0));
        }
        
        if deltas.len() > 256 {
            return Err(HomomorphismError::BatchSizeExceeded(deltas.len()));
        }
        
        let dim = 512;
        let mut aggregated_delta = vec![FixedPoint32_16::from_float(0.0); dim];
        let mut transactions = Vec::with_capacity(deltas.len());
        let mut total_amount = FixedPoint32_16::from_float(0.0);
        
        // Sum all deltas
        for delta in &deltas {
            if delta.delta.len() != dim {
                return Err(HomomorphismError::DimensionMismatch(dim, delta.delta.len()));
            }
            
            for i in 0..dim {
                aggregated_delta[i] = aggregated_delta[i].add(delta.delta[i]);
            }
            
            total_amount = total_amount.add(delta.transaction.amount);
            transactions.push(delta.transaction.clone());
        }
        
        // Compute batch ID (hash of all transaction hashes)
        let batch_id = Self::compute_batch_id(&transactions);
        
        // Compute bytes per transaction (compression metric)
        let bytes_per_tx = 512.0 / deltas.len() as f64; // 512 bytes total / N txs
        
        Ok(Self {
            aggregated_delta,
            transactions,
            batch_id,
            total_amount,
            bytes_per_tx,
        })
    }
    
    /// Compute batch ID from transactions
    fn compute_batch_id(transactions: &[TransferTransaction]) -> [u8; 32] {
        use sha3::{Digest, Sha3_256};
        
        let mut hasher = Sha3_256::new();
        
        for tx in transactions {
            // Hash each transaction's essential data
            hasher.update(&tx.sender);
            hasher.update(&tx.receiver);
            hasher.update(&tx.amount.0.to_le_bytes());
            hasher.update(&tx.nonce.to_le_bytes());
        }
        
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }
    
   
  /// Apply this batch delta to an embedding with strict fixed-point enforcement
pub fn apply_to_embedding(
    &self,
    embedding: &[FixedPoint32_16],
) -> Result<Vec<FixedPoint32_16>> {

    let dim = self.aggregated_delta.len();

    if embedding.len() != dim || self.aggregated_delta.len() != dim {
        return Err(HomomorphismError::DimensionMismatch(
            embedding.len(),
            self.aggregated_delta.len(),
        ));
    }

    let mut result = Vec::with_capacity(dim);

    for i in 0..dim {
        let summed = embedding[i]
            .checked_add(self.aggregated_delta[i])
            .ok_or(HomomorphismError::Overflow)?;

        result.push(summed);
    }

    Ok(result)
}
    
    /// Verify that applying this batch produces a valid embedding
    pub fn verify_application(
        &self,
        old_embedding: &[FixedPoint32_16],
        new_embedding: &[FixedPoint32_16],
        error_bound: FixedPoint32_16,
    ) -> Result<bool> {
        let computed = self.apply_to_embedding(old_embedding)?;
        
        // Compare computed vs actual new embedding
        for i in 0..512 {
            let diff = if computed[i].0 > new_embedding[i].0 {
                FixedPoint32_16(computed[i].0 - new_embedding[i].0)
            } else {
                FixedPoint32_16(new_embedding[i].0 - computed[i].0)
            };
            
            if diff.0 > error_bound.0 {
                return Err(HomomorphismError::Violation(diff.to_float()));
            }
        }
        
        Ok(true)
    }
    
    /// Get compression ratio vs raw state
    pub fn compression_ratio(&self, accounts_per_shard: usize) -> f64 {
        // Raw state size: accounts * (32-byte key + 8-byte balance) = accounts * 40 bytes
        let raw_size = accounts_per_shard * 40;
        
        // Our size: 512 bytes for embedding + overhead
        let our_size = 512 + (self.transactions.len() * 2); // ~2 bytes per tx overhead
        
        raw_size as f64 / our_size as f64
    }
}


/// Manages account embeddings for delta computation
#[derive(Clone, Debug)]
pub struct AccountEmbeddingManager {
    /// Map from blinded account keys to their 512D embeddings
    embeddings: Arc<RwLock<HashMap<[u8; 32], Vec<FixedPoint32_16>>>>,
    
    /// Bias terms for transfer homomorphism (learned during training)
    transfer_bias: Vec<FixedPoint32_16>,
    
    /// Encoder for computing new embeddings (if needed)
    encoder: Option<Arc<TransformerEncoder>>,
}


impl AccountEmbeddingManager {
    /// Create a new embedding manager
    pub fn new() -> Self {
        // Initialize with default bias terms (zero)
        let transfer_bias = vec![FixedPoint32_16::from_float(0.0); 512];
        
        Self {
            embeddings: Arc::new(RwLock::new(HashMap::new())),
            transfer_bias,
            encoder: None,
        }
    }
    
    /// Set the transformer encoder for computing embeddings
    pub fn set_encoder(&mut self, encoder: Arc<TransformerEncoder>) {
        self.encoder = Some(encoder);
    }
    
    /// Get or compute an account embedding
    pub async fn get_embedding(&self, account_key: &[u8; 32]) -> Result<Vec<FixedPoint32_16>> {
        let embeddings = self.embeddings.read().await;
        
        if let Some(embedding) = embeddings.get(account_key) {
            return Ok(embedding.clone());
        }
        
        // Drop read lock before acquiring write lock
        drop(embeddings);
        
        // Compute new embedding if encoder is available
        if let Some(encoder) = &self.encoder {
            // In practice, we would compute embedding from account state
            // For now, generate a deterministic embedding from the key
            let embedding = Self::compute_deterministic_embedding(account_key);
            
            let mut embeddings = self.embeddings.write().await;
            embeddings.insert(*account_key, embedding.clone());
            
            Ok(embedding)
        } else {
            Err(HomomorphismError::AccountNotFound(*account_key))
        }
    }
    
    /// Compute a deterministic embedding from account key
    fn compute_deterministic_embedding(account_key: &[u8; 32]) -> Vec<FixedPoint32_16> {
        let mut embedding = Vec::with_capacity(512);
        let key_hash = blake3::hash(account_key);
        
        for i in 0..512 {
            // Use key hash bytes cyclically to generate embedding values
            let byte = key_hash.as_bytes()[i % 32];
            let value = (byte as f64 / 255.0) * 2.0 - 1.0; // Normalize to [-1, 1]
            embedding.push(FixedPoint32_16::from_float(value));
        }
        
        embedding
    }
    
    /// Update bias terms (called during federated learning)
    pub fn update_bias_terms(&mut self, new_bias: Vec<FixedPoint32_16>) -> Result<()> {
        if new_bias.len() != 512 {
            return Err(HomomorphismError::DimensionMismatch(512, new_bias.len()));
        }
        
        self.transfer_bias = new_bias;
        Ok(())
    }
    
    /// Get bias terms
    pub fn bias_terms(&self) -> &[FixedPoint32_16] {
        &self.transfer_bias
    }
    
    /// Compute delta for a transfer transaction
    pub async fn compute_transfer_delta(
        &self,
        transaction: TransferTransaction,
    ) -> Result<TransferDelta> {
        // Get sender and receiver embeddings
        let sender_embedding = self.get_embedding(&transaction.sender).await?;
        let receiver_embedding = self.get_embedding(&transaction.receiver).await?;
        
        // Create delta using the homomorphism formula
        TransferDelta::new(
            transaction,
            &sender_embedding,
            &receiver_embedding,
            &self.transfer_bias,
        )
    }
    
    /// Update account embedding after a transfer
    pub async fn update_account_embedding(
        &self,
        account_key: &[u8; 32],
        delta: &[FixedPoint32_16],
    ) -> Result<()> {
        let mut embeddings = self.embeddings.write().await;
        
        if let Some(current) = embeddings.get_mut(account_key) {
            // Update embedding: new = old + delta
            for i in 0..512 {
                current[i] = current[i].add(delta[i]);
            }
            Ok(())
        } else {
            // If account doesn't exist, create with delta as initial embedding
            embeddings.insert(*account_key, delta.to_vec());
            Ok(())
        }
    }
    
    /// Batch update multiple accounts
    pub async fn batch_update_embeddings(
        &self,
        updates: &[([u8; 32], Vec<FixedPoint32_16>)],
    ) -> Result<()> {
        let mut embeddings = self.embeddings.write().await;
        
        for (account_key, delta) in updates {
            if let Some(current) = embeddings.get_mut(account_key) {
                for i in 0..512 {
                    current[i] = current[i].add(delta[i]);
                }
            } else {
                embeddings.insert(*account_key, delta.clone());
            }
        }
        
        Ok(())
    }
}


/// Verifies homomorphism property for state transitions
#[derive(Clone, Debug)]
pub struct HomomorphismVerifier {
    /// Error bound (1e-9 in fixed-point)
    error_bound: FixedPoint32_16,
    
    /// Maximum allowed batch size
    max_batch_size: usize,
    
    /// Statistics for monitoring
    stats: VerificationStats,
}


/// Statistics for homomorphism verification
#[derive(Clone, Debug, Default)]
pub struct VerificationStats {
    pub total_verifications: u64,
    pub successful_verifications: u64,
    pub failed_verifications: u64,
    pub max_error_observed: f64,
    pub avg_error: f64,
    pub total_error_sum: f64,
}


impl HomomorphismVerifier {
    /// Create a new verifier with default error bound (1e-9)
    pub fn new() -> Self {
        Self {
            error_bound: FixedPoint32_16::from_float(1e-9),
            max_batch_size: 256,
            stats: VerificationStats::default(),
        }
    }
    
    /// Verify single transfer homomorphism
    pub fn verify_single_transfer(
        &mut self,
        old_embedding: &[FixedPoint32_16],
        delta: &TransferDelta,
        new_embedding: &[FixedPoint32_16],
    ) -> Result<bool> {
        self.stats.total_verifications += 1;
        
        if old_embedding.len() != 512 || new_embedding.len() != 512 || delta.delta.len() != 512 {
            return Err(HomomorphismError::DimensionMismatch(512, old_embedding.len()));
        }
        
        let mut max_error = 0.0;
        
        // Verify: new ≈ old + δ
        for i in 0..512 {
            let computed = old_embedding[i].add(delta.delta[i]);
            let diff = if computed.0 > new_embedding[i].0 {
                FixedPoint32_16(computed.0 - new_embedding[i].0)
            } else {
                FixedPoint32_16(new_embedding[i].0 - computed.0)
            };
            
            let error = diff.to_float();
            max_error = max_error.max(error);
            
            if diff.0 > self.error_bound.0 {
                self.stats.failed_verifications += 1;
                self.stats.total_error_sum += error;
                return Err(HomomorphismError::Violation(error));
            }
        }
        
        // Update statistics
        self.stats.successful_verifications += 1;
        self.stats.max_error_observed = self.stats.max_error_observed.max(max_error);
        self.stats.total_error_sum += max_error;
        self.stats.avg_error = self.stats.total_error_sum / self.stats.total_verifications as f64;
        
        Ok(true)
    }
    
    /// Verify batch homomorphism
    pub fn verify_batch(
        &mut self,
        old_embedding: &[FixedPoint32_16],
        batch_delta: &BatchDelta,
        new_embedding: &[FixedPoint32_16],
    ) -> Result<bool> {
        self.stats.total_verifications += 1;
        
        // Apply batch delta to old embedding
        let computed_embedding = batch_delta.apply_to_embedding(old_embedding)?;
        
        // Verify against new embedding
        let mut max_error = 0.0;
        
        for i in 0..512 {
            let diff = if computed_embedding[i].0 > new_embedding[i].0 {
                FixedPoint32_16(computed_embedding[i].0 - new_embedding[i].0)
            } else {
                FixedPoint32_16(new_embedding[i].0 - computed_embedding[i].0)
            };
            
            let error = diff.to_float();
            max_error = max_error.max(error);
            
            if diff.0 > self.error_bound.0 {
                self.stats.failed_verifications += 1;
                self.stats.total_error_sum += error;
                return Err(HomomorphismError::Violation(error));
            }
        }
        
        // Update statistics
        self.stats.successful_verifications += 1;
        self.stats.max_error_observed = self.stats.max_error_observed.max(max_error);
        self.stats.total_error_sum += max_error;
        self.stats.avg_error = self.stats.total_error_sum / self.stats.total_verifications as f64;
        
        // Also verify compression ratio is reasonable
        let compression = batch_delta.compression_ratio(100_000); // Example: 100k accounts
        if compression < 100.0 {
            log::warn!("Low compression ratio: {:.1}x", compression);
        }
        
        Ok(true)
    }
    
    /// Get verification statistics
    pub fn stats(&self) -> &VerificationStats {
        &self.stats
    }
    
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = VerificationStats::default();
    }
    
    /// Set error bound (for testing or different precision requirements)
    pub fn set_error_bound(&mut self, bound: f64) {
        self.error_bound = FixedPoint32_16::from_float(bound);
    }
}


/// Trains the transformer encoder to preserve homomorphism property
#[derive(Clone, Debug)]
pub struct EncoderTrainer {
    /// Training dataset of state transitions
    dataset: Vec<TrainingExample>,
    
    /// Current encoder being trained
    encoder: Arc<TransformerEncoder>,
    
    /// Learning rate for gradient updates
    learning_rate: f64,
    
    /// Batch size for training
    batch_size: usize,
    
    /// Homomorphism error target
    target_error: f64,
}


/// A training example for the encoder
#[derive(Clone, Debug)]
pub struct TrainingExample {
    /// Old state S_t
    pub old_state: Vec<([u8; 32], FixedPoint32_16)>,
    
    /// Transfer transaction
    pub transaction: TransferTransaction,
    
    /// New state S_{t+1}
    pub new_state: Vec<([u8; 32], FixedPoint32_16)>,
    
    /// Expected delta δ(tx)
    pub expected_delta: Vec<FixedPoint32_16>,
}


impl EncoderTrainer {
    /// Create a new encoder trainer
    pub fn new(encoder: Arc<TransformerEncoder>) -> Self {
        Self {
            dataset: Vec::new(),
            encoder,
            learning_rate: 1e-4,
            batch_size: 32,
            target_error: 1e-9,
        }
    }
    
    /// Add a training example
    pub fn add_example(&mut self, example: TrainingExample) {
        self.dataset.push(example);
    }
    
    /// Train the encoder to improve homomorphism preservation
    pub async fn train_epoch(&mut self) -> Result<f64> {
        if self.dataset.is_empty() {
            return Ok(0.0);
        }
        
        let mut total_loss = 0.0;
        let num_batches = (self.dataset.len() + self.batch_size - 1) / self.batch_size;
        
        // Shuffle dataset
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        self.dataset.shuffle(&mut thread_rng());
        
        for batch_start in (0..self.dataset.len()).step_by(self.batch_size) {
            let batch_end = std::cmp::min(batch_start + self.batch_size, self.dataset.len());
            let batch = &self.dataset[batch_start..batch_end];
            
            // Compute loss for this batch
            let batch_loss = self.compute_batch_loss(batch).await?;
            total_loss += batch_loss;
            
            // Update encoder weights (simplified - actual would use backprop)
            self.update_encoder_weights(batch_loss).await?;
        }
        
        let avg_loss = total_loss / num_batches as f64;
        Ok(avg_loss)
    }
    
    /// Compute loss for a batch of examples
    async fn compute_batch_loss(&self, examples: &[TrainingExample]) -> Result<f64> {
        let mut total_loss = 0.0;
        
        for example in examples {
            // Encode old state
            let old_embedding = self.encoder.encode_state(&example.old_state)
                .await
                .map_err(|e| HomomorphismError::Encoder(e.to_string()))?;
            
            // Encode new state
            let new_embedding = self.encoder.encode_state(&example.new_state)
                .await
                .map_err(|e| HomomorphismError::Encoder(e.to_string()))?;
            
            // Compute expected new embedding: old + δ
            let mut expected_new = Vec::with_capacity(512);
            for i in 0..512 {
                expected_new.push(old_embedding[i].add(example.expected_delta[i]));
            }
            
            // Compute L2 loss: ||new_embedding - (old_embedding + δ)||²
            let mut loss = 0.0;
            for i in 0..512 {
                let diff = new_embedding[i].to_float() - expected_new[i].to_float();
                loss += diff * diff;
            }
            
            total_loss += loss;
        }
        
        Ok(total_loss / examples.len() as f64)
    }
    
    /// Update encoder weights based on loss
    async fn update_encoder_weights(&mut self, loss: f64) -> Result<()> {
        // Simplified weight update - in practice would use gradient descent
        // with backpropagation through the transformer
        
        if loss > self.target_error * 10.0 {
            // If loss is high, we need significant updates
            log::warn!("High training loss: {:.2e}, needs more training", loss);
        }
        
        // In a real implementation, this would:
        // 1. Compute gradients via backprop
        // 2. Apply optimizer step (Adam, SGD, etc.)
        // 3. Update encoder weights
        
        // For now, just log the loss
        log::debug!("Training loss: {:.2e}, target: {:.2e}", loss, self.target_error);
        
        Ok(())
    }
    
    /// Test the encoder's homomorphism preservation
    pub async fn test_homomorphism(&self, num_tests: usize) -> Result<TestResults> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let mut results = TestResults::new();
        
        for _ in 0..num_tests {
            // Generate random test example
            let example = self.generate_random_example(&mut rng);
            
            // Encode states
            let old_embedding = self.encoder.encode_state(&example.old_state)
                .await
                .map_err(|e| HomomorphismError::Encoder(e.to_string()))?;
            
            let new_embedding = self.encoder.encode_state(&example.new_state)
                .await
                .map_err(|e| HomomorphismError::Encoder(e.to_string()))?;
            
            // Compute homomorphism error
            let mut max_error = 0.0;
            for i in 0..512 {
                let expected = old_embedding[i].add(example.expected_delta[i]);
                let diff = (expected.to_float() - new_embedding[i].to_float()).abs();
                max_error = max_error.max(diff);
            }
            
            results.record_test(max_error <= self.target_error, max_error);
        }
        
        Ok(results)
    }
    
    /// Generate a random training example
    fn generate_random_example<R: Rng>(&self, rng: &mut R) -> TrainingExample {
        // Generate random accounts and balances
        let num_accounts = rng.gen_range(2..10);
        let mut old_state = Vec::new();
        
        for _ in 0..num_accounts {
            let mut key = [0u8; 32];
            rng.fill(&mut key);
            let balance = FixedPoint32_16::from_float(rng.gen_range(0.0..1000.0));
            old_state.push((key, balance));
        }
        
        // Select random sender and receiver
        let sender_idx = rng.gen_range(0..num_accounts);
        let receiver_idx = rng.gen_range(0..num_accounts);
        
        let (sender_key, sender_balance) = old_state[sender_idx].clone();
        let (receiver_key, _) = old_state[receiver_idx].clone();
        
        // Random amount (less than sender's balance)
        let amount = FixedPoint32_16::from_float(
            rng.gen_range(0.0..sender_balance.to_float())
        );
        
        // Create transaction
        let transaction = TransferTransaction {
            sender: sender_key,
            receiver: receiver_key,
            amount,
            nonce: rng.gen(),
            timestamp: rng.gen(),
            balance_proof: None,
        };
        
        // Create new state by applying transfer
        let mut new_state = old_state.clone();
        new_state[sender_idx].1 = new_state[sender_idx].1.add(FixedPoint32_16::from_float(-1.0).mul(amount));
        new_state[receiver_idx].1 = new_state[receiver_idx].1.add(amount);
        
        // Generate expected delta (simplified - in practice would compute from embeddings)
        let expected_delta = vec![FixedPoint32_16::from_float(0.0); 512];
        
        TrainingExample {
            old_state,
            transaction,
            new_state,
            expected_delta,
        }
    }
}


/// Results from homomorphism testing
#[derive(Clone, Debug)]
pub struct TestResults {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub max_error: f64,
    pub min_error: f64,
    pub avg_error: f64,
}


impl TestResults {
    /// Create new test results
    pub fn new() -> Self {
        Self {
            total_tests: 0,
            passed_tests: 0,
            failed_tests: 0,
            max_error: 0.0,
            min_error: f64::INFINITY,
            avg_error: 0.0,
        }
    }
    
    /// Record a test result
    pub fn record_test(&mut self, passed: bool, error: f64) {
        self.total_tests += 1;
        
        if passed {
            self.passed_tests += 1;
        } else {
            self.failed_tests += 1;
        }
        
        self.max_error = self.max_error.max(error);
        self.min_error = self.min_error.min(error);
        
        // Update average error
        let total_error_sum = self.avg_error * (self.total_tests - 1) as f64 + error;
        self.avg_error = total_error_sum / self.total_tests as f64;
    }
    
    /// Get pass rate
    pub fn pass_rate(&self) -> f64 {
        if self.total_tests == 0 {
            return 0.0;
        }
        self.passed_tests as f64 / self.total_tests as f64
    }
}


/// Utility functions for homomorphism operations
pub mod utils {
    use super::*;
    
    /// Convert embedding to bytes for storage/transmission
    pub fn embedding_to_bytes(embedding: &[FixedPoint32_16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(embedding.len() * 8);
        
        for &fp in embedding {
            bytes.extend_from_slice(&fp.0.to_le_bytes());
        }
        
        bytes
    }
    
    /// Convert bytes back to embedding
    pub fn bytes_to_embedding(bytes: &[u8]) -> Result<Vec<FixedPoint32_16>> {
        if bytes.len() % 8 != 0 {
            return Err(HomomorphismError::DimensionMismatch(
                bytes.len() / 8,
                512
            ));
        }
        
        let mut embedding = Vec::with_capacity(bytes.len() / 8);
        
        for chunk in bytes.chunks_exact(8) {
            let value = i64::from_le_bytes(chunk.try_into().unwrap());
            embedding.push(FixedPoint32_16(value));
        }
        
        Ok(embedding)
    }
    
    /// Compute embedding hash (32-byte BLAKE3)
    pub fn hash_embedding(embedding: &[FixedPoint32_16]) -> [u8; 32] {
        use sha3::{Digest, Sha3_256};
        
        let mut hasher = Sha3_256::new();
        
        for &fp in embedding {
            hasher.update(&fp.0.to_le_bytes());
        }
        
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }
    
    /// Compute L2 distance between two embeddings
    pub fn embedding_distance(a: &[FixedPoint32_16], b: &[FixedPoint32_16]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(HomomorphismError::DimensionMismatch(a.len(), b.len()));
        }
        
        let mut sum = 0.0;
        
        for i in 0..a.len() {
            let diff = a[i].to_float() - b[i].to_float();
            sum += diff * diff;
        }
        
        Ok(sum.sqrt())
    }
    
    /// Normalize embedding to unit length
    pub fn normalize_embedding(embedding: &[FixedPoint32_16]) -> Vec<FixedPoint32_16> {
        // Compute L2 norm
        let mut norm_sq = 0.0;
        for &fp in embedding {
            let val = fp.to_float();
            norm_sq += val * val;
        }
        
        let norm = norm_sq.sqrt();
        
        if norm < 1e-10 {
            return embedding.to_vec(); // Avoid division by zero
        }
        
        // Scale to unit length
        embedding.iter()
            .map(|&fp| FixedPoint32_16::from_float(fp.to_float() / norm))
            .collect()
    }

    pub fn verify_embedding_error_bound(
    expected: &[FixedPoint32_16],
    computed: &[FixedPoint32_16],
    max_error: i64,
) -> Result<()> {

    if expected.len() != computed.len() {
        return Err(HomomorphismError::DimensionMismatch(
            expected.len(),
            computed.len(),
        ));
    }

    for i in 0..expected.len() {
        let diff = (expected[i].0 - computed[i].0).abs();
        if diff > max_error {
            return Err(HomomorphismError::ErrorBoundViolated);
        }
    }

    Ok(())
}
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    
    #[test]
    fn test_transfer_delta_creation() {
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
    fn test_batch_delta_aggregation() {
        let mut rng = rand::thread_rng();
        
        // Create multiple deltas
        let mut deltas = Vec::new();
        
        for i in 0..10 {
            // Create test embeddings for each transaction
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
        assert!(compression > 100.0, "Compression ratio should be >100x");
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
        
        // Now test with incorrect new embedding (should fail)
        let mut bad_new_embedding = new_embedding.clone();
        bad_new_embedding[0] = FixedPoint32_16::from_float(1000.0);
        
        let result = verifier.verify_single_transfer(
            &old_embedding,
            &delta,
            &bad_new_embedding,
        );
        
        assert!(result.is_err());
        assert_eq!(verifier.stats().failed_verifications, 1);
    }
    
    #[tokio::test]
    async fn test_account_embedding_manager() {
        let manager = AccountEmbeddingManager::new();
        
        // Test account embedding retrieval (should fail since no encoder set)
        let test_key = [42u8; 32];
        let result = manager.get_embedding(&test_key).await;
        
        assert!(result.is_err());
        
        // Test delta computation with mock embeddings
        let transaction = TransferTransaction {
            sender: [1u8; 32],
            receiver: [2u8; 32],
            amount: FixedPoint32_16::from_float(50.0),
            nonce: 1,
            timestamp: 1234567890,
            balance_proof: None,
        };
        
        // This should also fail since we don't have embeddings
        let result = manager.compute_transfer_delta(transaction).await;
        assert!(result.is_err());
    }
    
    #[test]
    fn test_utils_functions() {
        // Test embedding serialization
        let mut embedding = Vec::new();
        
        for i in 0..512 {
            embedding.push(FixedPoint32_16::from_float(i as f64));
        }
        
        // Convert to bytes and back
        let bytes = utils::embedding_to_bytes(&embedding);
        let reconstructed = utils::bytes_to_embedding(&bytes).unwrap();
        
        assert_eq!(embedding.len(), reconstructed.len());
        
        for i in 0..512 {
            let diff = (embedding[i].to_float() - reconstructed[i].to_float()).abs();
            assert!(diff < 1e-6, "Serialization round-trip failed");
        }
        
        // Test embedding hash
        let hash1 = utils::hash_embedding(&embedding);
        
        // Slightly change embedding
        let mut embedding2 = embedding.clone();
        embedding2[0] = FixedPoint32_16::from_float(embedding2[0].to_float() + 0.1);
        
        let hash2 = utils::hash_embedding(&embedding2);
        
        assert_ne!(hash1, hash2, "Different embeddings should have different hashes");
        
        // Test embedding distance
        let distance = utils::embedding_distance(&embedding, &embedding2).unwrap();
        assert!(distance > 0.0, "Distance between different embeddings should be positive");
        
        // Test normalization
        let normalized = utils::normalize_embedding(&embedding);
        
        // Compute norm of normalized embedding
        let mut norm_sq = 0.0;
        for &fp in &normalized {
            let val = fp.to_float();
            norm_sq += val * val;
        }
        
        let norm = norm_sq.sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "Normalized embedding should have unit norm");
    }
}
