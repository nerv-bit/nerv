// src/embedding/mod.rs
// ============================================================================
// NEURAL STATE EMBEDDINGS MODULE
// ============================================================================
// This module implements the core innovation of NERV: replacing Merkle trees
// with 512-byte neural state embeddings produced by a transformer encoder.
// Key components:
// 1. NeuralEncoder: 24-layer transformer for state compression
// 2. TransferHomomorphism: Linear updates for balance transfers
// 3. DeltaComputation: Pre-computed deltas for transaction batches
// 4. LatentLedgerCircuit: Halo2 ZK circuit for verifiable encoding
// ============================================================================

use crate::{Embedding, Hash, DeltaVector, Result, NervError};
use crate::params::{
    EMBEDDING_SIZE, BATCH_SIZE, HOMO_ERROR_BOUND, TRANSFORMER_LAYERS,
    TRANSFORMER_HEADS, TRANSFORMER_EMBEDDING_DIM, TRANSFORMER_FF_DIM,
    FIXED_POINT_SCALE,
};
use crate::utils::serialization::EmbeddingVec;

// Re-export public types and functions
pub use encoder::NeuralEncoder;
pub use homomorphism::{TransferHomomorphism, DeltaComputation};
pub use circuit::{LatentLedgerCircuit, EmbeddingProof};

// Module declarations
pub mod encoder;
pub mod homomorphism;
pub mod circuit;
pub mod training;

// ============================================================================
// CORE TYPES AND TRAITS
// ============================================================================

/// State representation at a specific height
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    /// Block height
    pub height: u64,
    
    /// Shard identifier
    pub shard_id: u64,
    
    /// Neural state embedding (512 bytes)
    pub embedding: EmbeddingVec,
    
    /// Hash of the embedding (BLAKE3)
    pub embedding_hash: Hash,
    
    /// Metadata for verification
    pub metadata: StateMetadata,
}

/// Metadata for state verification
#[derive(Debug, Clone)]
pub struct StateMetadata {
    /// Epoch number (encoder version)
    pub epoch: u64,
    
    /// Timestamp of last update
    pub timestamp: u64,
    
    /// Number of accounts in this state
    pub account_count: u64,
    
    /// Total balance in this shard
    pub total_balance: u128,
    
    /// Proof of correctness (Halo2 proof)
    pub proof: Option<Vec<u8>>,
}

/// Batch of transactions for processing
#[derive(Debug, Clone)]
pub struct TransactionBatch {
    /// Batch identifier
    pub batch_id: u64,
    
    /// List of transaction deltas
    pub deltas: Vec<DeltaVector>,
    
    /// Aggregated delta (sum of all deltas)
    pub aggregated_delta: DeltaVector,
    
    /// Source shard identifier
    pub shard_id: u64,
    
    /// Batch processing timestamp
    pub timestamp: u64,
}

impl TransactionBatch {
    /// Create a new transaction batch
    pub fn new(batch_id: u64, shard_id: u64) -> Self {
        Self {
            batch_id,
            deltas: Vec::with_capacity(BATCH_SIZE),
            aggregated_delta: [0.0; EMBEDDING_SIZE],
            shard_id,
            timestamp: crate::utils::current_timestamp_ms(),
        }
    }
    
    /// Add a transaction delta to the batch
    pub fn add_delta(&mut self, delta: DeltaVector) -> Result<()> {
        if self.deltas.len() >= BATCH_SIZE {
            return Err(NervError::State(
                format!("Batch size exceeded: {}", BATCH_SIZE)
            ));
        }
        
        // Update aggregated delta
        for i in 0..EMBEDDING_SIZE {
            self.aggregated_delta[i] += delta[i];
        }
        
        self.deltas.push(delta);
        Ok(())
    }
    
    /// Get the number of transactions in the batch
    pub fn size(&self) -> usize {
        self.deltas.len()
    }
    
    /// Check if the batch is full
    pub fn is_full(&self) -> bool {
        self.deltas.len() >= BATCH_SIZE
    }
    
    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.deltas.is_empty()
    }
    
    /// Clear the batch (for reuse)
    pub fn clear(&mut self) {
        self.deltas.clear();
        self.aggregated_delta = [0.0; EMBEDDING_SIZE];
    }
}

/// State update result
#[derive(Debug, Clone)]
pub struct StateUpdate {
    /// Old state snapshot
    pub old_state: StateSnapshot,
    
    /// New state snapshot
    pub new_state: StateSnapshot,
    
    /// Transaction batch that caused the update
    pub batch: TransactionBatch,
    
    /// Homomorphism error (should be ≤ HOMO_ERROR_BOUND)
    pub homomorphism_error: f64,
    
    /// Proof of correct update (Halo2 proof)
    pub update_proof: Vec<u8>,
}

impl StateUpdate {
    /// Validate that the homomorphism error is within bounds
    pub fn validate_error_bound(&self) -> Result<()> {
        if self.homomorphism_error > HOMO_ERROR_BOUND * self.batch.size() as f64 {
            return Err(NervError::State(format!(
                "Homomorphism error {} exceeds bound {}",
                self.homomorphism_error,
                HOMO_ERROR_BOUND * self.batch.size() as f64
            )));
        }
        Ok(())
    }
}

// ============================================================================
// EMBEDDING MANAGER - Main interface for embedding operations
// ============================================================================

/// Manages neural state embeddings for a shard
pub struct EmbeddingManager {
    /// Current state snapshot
    current_state: StateSnapshot,
    
    /// Neural encoder for this epoch
    encoder: NeuralEncoder,
    
    /// Transfer homomorphism calculator
    homomorphism: TransferHomomorphism,
    
    /// Delta computation engine
    delta_computer: DeltaComputation,
    
    /// Pending transaction batches
    pending_batches: Vec<TransactionBatch>,
    
    /// Epoch number
    epoch: u64,
    
    /// Shard identifier
    shard_id: u64,
}

impl EmbeddingManager {
    /// Create a new embedding manager for a shard
    pub fn new(
        shard_id: u64,
        initial_embedding: EmbeddingVec,
        encoder: NeuralEncoder,
        homomorphism: TransferHomomorphism,
        delta_computer: DeltaComputation,
    ) -> Result<Self> {
        let embedding_hash = crate::utils::blake3_hash(&Self::embedding_to_bytes(&initial_embedding));
        
        let current_state = StateSnapshot {
            height: 0,
            shard_id,
            embedding: initial_embedding,
            embedding_hash,
            metadata: StateMetadata {
                epoch: 0,
                timestamp: crate::utils::current_timestamp_ms(),
                account_count: 0,
                total_balance: 0,
                proof: None,
            },
        };
        
        Ok(Self {
            current_state,
            encoder,
            homomorphism,
            delta_computer,
            pending_batches: Vec::new(),
            epoch: 0,
            shard_id,
        })
    }
    
    /// Get the current state
    pub fn current_state(&self) -> &StateSnapshot {
        &self.current_state
    }
    
    /// Create a new transaction batch
    pub fn create_batch(&mut self) -> u64 {
        let batch_id = crate::utils::current_timestamp_ms(); // Use timestamp as batch ID
        let batch = TransactionBatch::new(batch_id, self.shard_id);
        self.pending_batches.push(batch);
        batch_id
    }
    
    /// Add a transaction to a batch
    pub fn add_transaction(
        &mut self,
        batch_id: u64,
        sender_key: &[u8],
        receiver_key: &[u8],
        amount: u64,
    ) -> Result<()> {
        let batch = self.pending_batches
            .iter_mut()
            .find(|b| b.batch_id == batch_id)
            .ok_or_else(|| NervError::State("Batch not found".to_string()))?;
        
        // Compute delta for this transaction
        let delta = self.delta_computer.compute_delta(
            sender_key,
            receiver_key,
            amount,
        )?;
        
        batch.add_delta(delta)?;
        Ok(())
    }
    
    /// Apply a batch to the current state
    pub fn apply_batch(&mut self, batch_id: u64) -> Result<StateUpdate> {
        // Find and remove the batch
        let batch_index = self.pending_batches
            .iter()
            .position(|b| b.batch_id == batch_id)
            .ok_or_else(|| NervError::State("Batch not found".to_string()))?;
        
        let batch = self.pending_batches.remove(batch_index);
        
        if batch.is_empty() {
            return Err(NervError::State("Cannot apply empty batch".to_string()));
        }
        
        // Create old state snapshot
        let old_state = self.current_state.clone();
        
        // Apply homomorphic update
        let new_embedding = self.homomorphism.apply_update(
            &old_state.embedding,
            &batch.aggregated_delta,
        )?;
        
        // Compute homomorphism error
        let error = self.homomorphism.compute_error(
            &old_state.embedding,
            &new_embedding,
            &batch.aggregated_delta,
        )?;
        
        // Update current state
        let new_height = old_state.height + 1;
        let new_embedding_hash = crate::utils::blake3_hash(&Self::embedding_to_bytes(&new_embedding));
        
        let new_state = StateSnapshot {
            height: new_height,
            shard_id: self.shard_id,
            embedding: new_embedding,
            embedding_hash: new_embedding_hash,
            metadata: StateMetadata {
                epoch: self.epoch,
                timestamp: crate::utils::current_timestamp_ms(),
                account_count: old_state.metadata.account_count, // TODO: Update based on transactions
                total_balance: old_state.metadata.total_balance, // TODO: Update based on transactions
                proof: None, // Will be filled by ZK circuit
            },
        };
        
        // Generate proof (in production, this would be done in TEE)
        let update_proof = self.generate_update_proof(&old_state, &new_state, &batch)?;
        
        // Update current state
        self.current_state = new_state.clone();
        
        let update = StateUpdate {
            old_state,
            new_state,
            batch,
            homomorphism_error: error,
            update_proof,
        };
        
        // Validate error bound
        update.validate_error_bound()?;
        
        Ok(update)
    }
    
    /// Generate a ZK proof for the state update
    fn generate_update_proof(
        &self,
        old_state: &StateSnapshot,
        new_state: &StateSnapshot,
        batch: &TransactionBatch,
    ) -> Result<Vec<u8>> {
        // In production, this would generate a Halo2 proof
        // For now, return a placeholder
        Ok(vec![])
    }
    
    /// Convert embedding to bytes for hashing
    fn embedding_to_bytes(embedding: &EmbeddingVec) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(EMBEDDING_SIZE * 4);
        for &value in &embedding.0 {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
    
    /// Verify a state update using the proof
    pub fn verify_update(&self, update: &StateUpdate) -> Result<bool> {
        // Verify homomorphism error bound
        update.validate_error_bound()?;
        
        // Verify that new embedding = old embedding + aggregated delta
        let expected_embedding = self.homomorphism.apply_update(
            &update.old_state.embedding,
            &update.batch.aggregated_delta,
        )?;
        
        if !update.new_state.embedding.approx_eq(&expected_embedding, HOMO_ERROR_BOUND as f32) {
            return Ok(false);
        }
        
        // Verify embedding hash
        let computed_hash = crate::utils::blake3_hash(
            &Self::embedding_to_bytes(&update.new_state.embedding)
        );
        
        if computed_hash != update.new_state.embedding_hash {
            return Ok(false);
        }
        
        // TODO: Verify ZK proof when implemented
        
        Ok(true)
    }
    
    /// Get the epoch number
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    
    /// Update to a new encoder (epoch transition)
    pub fn update_encoder(&mut self, new_encoder: NeuralEncoder) -> Result<()> {
        // Verify that the new encoder preserves homomorphism
        let test_delta = [1.0; EMBEDDING_SIZE];
        let test_embedding = EmbeddingVec::zeros();
        
        let new_embedding = self.homomorphism.apply_update_with_encoder(
            &test_embedding,
            &test_delta,
            &new_encoder,
        )?;
        
        let error = self.homomorphism.compute_error_with_encoder(
            &test_embedding,
            &new_embedding,
            &test_delta,
            &new_encoder,
        )?;
        
        if error > HOMO_ERROR_BOUND {
            return Err(NervError::State(format!(
                "New encoder error {} exceeds bound {}",
                error, HOMO_ERROR_BOUND
            )));
        }
        
        // Update encoder and increment epoch
        self.encoder = new_encoder;
        self.epoch += 1;
        
        Ok(())
    }
}

// ============================================================================
// FIXED-POINT ARITHMETIC FOR EMBEDDINGS
// ============================================================================

/// Convert floating-point embedding to fixed-point representation
pub fn embedding_to_fixed(embedding: &EmbeddingVec) -> Vec<i32> {
    let mut fixed = Vec::with_capacity(EMBEDDING_SIZE);
    for &value in &embedding.0 {
        let fixed_value = (value * FIXED_POINT_SCALE) as i32;
        fixed.push(fixed_value);
    }
    fixed
}

/// Convert fixed-point embedding back to floating-point
pub fn fixed_to_embedding(fixed: &[i32]) -> EmbeddingVec {
    let mut values = [0.0; EMBEDDING_SIZE];
    for i in 0..EMBEDDING_SIZE.min(fixed.len()) {
        values[i] = fixed[i] as f64 / FIXED_POINT_SCALE;
    }
    EmbeddingVec::new(values)
}

/// Add two fixed-point embeddings (for homomorphic updates)
pub fn add_fixed_embeddings(a: &[i32], b: &[i32]) -> Vec<i32> {
    let mut result = Vec::with_capacity(EMBEDDING_SIZE);
    for i in 0..EMBEDDING_SIZE {
        result.push(a[i].saturating_add(b[i]));
    }
    result
}

/// Check if two fixed-point embeddings are equal within epsilon
pub fn fixed_embeddings_approx_eq(a: &[i32], b: &[i32], epsilon: i32) -> bool {
    for i in 0..EMBEDDING_SIZE {
        if (a[i] - b[i]).abs() > epsilon {
            return false;
        }
    }
    true
}

// ============================================================================
// EMBEDDING UTILITIES
// ============================================================================

/// Compute the L2 norm (Euclidean distance) between two embeddings
pub fn embedding_distance(a: &EmbeddingVec, b: &EmbeddingVec) -> f64 {
    let mut sum = 0.0;
    for i in 0..EMBEDDING_SIZE {
        let diff = a.0[i] - b.0[i];
        sum += diff * diff;
    }
    sum.sqrt()
}

/// Compute the cosine similarity between two embeddings
pub fn embedding_similarity(a: &EmbeddingVec, b: &EmbeddingVec) -> f64 {
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    
    for i in 0..EMBEDDING_SIZE {
        dot_product += a.0[i] * b.0[i];
        norm_a += a.0[i] * a.0[i];
        norm_b += b.0[i] * b.0[i];
    }
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}

/// Normalize an embedding to unit length
pub fn normalize_embedding(embedding: &mut EmbeddingVec) {
    let norm = {
        let mut sum = 0.0;
        for &value in &embedding.0 {
            sum += value * value;
        }
        sum.sqrt()
    };
    
    if norm > 0.0 {
        for value in &mut embedding.0 {
            *value /= norm;
        }
    }
}

/// Interpolate between two embeddings (for smooth state transitions)
pub fn interpolate_embeddings(
    a: &EmbeddingVec,
    b: &EmbeddingVec,
    t: f32,
) -> EmbeddingVec {
    let mut result = [0.0; EMBEDDING_SIZE];
    for i in 0..EMBEDDING_SIZE {
        result[i] = a.0[i] as f32 * (1.0 - t) + b.0[i] as f32 * t;
    }
    EmbeddingVec::new(result.map(|x| x as f64))
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_transaction_batch() {
        let mut batch = TransactionBatch::new(1, 1);
        assert!(batch.is_empty());
        assert!(!batch.is_full());
        
        // Add a delta
        let delta = [1.0; EMBEDDING_SIZE];
        batch.add_delta(delta).unwrap();
        assert_eq!(batch.size(), 1);
        
        // Check aggregated delta
        for &value in &batch.aggregated_delta {
            assert!((value - 1.0).abs() < 0.0001);
        }
        
        // Clear batch
        batch.clear();
        assert!(batch.is_empty());
        for &value in &batch.aggregated_delta {
            assert!((value - 0.0).abs() < 0.0001);
        }
    }
    
    #[test]
    fn test_fixed_point_conversion() {
        let mut values = [0.0; EMBEDDING_SIZE];
        values[0] = 1.5;
        values[1] = -2.3;
        values[2] = 0.0;
        
        let embedding = EmbeddingVec::new(values);
        let fixed = embedding_to_fixed(&embedding);
        let restored = fixed_to_embedding(&fixed);
        
        // Check that conversion is approximately reversible
        assert!(embedding.approx_eq(&restored, 1e-4 as f32));
    }
    
    #[test]
    fn test_fixed_point_addition() {
        let a: Vec<i32> = (0..EMBEDDING_SIZE).map(|i| i as i32).collect();
        let b: Vec<i32> = (0..EMBEDDING_SIZE).map(|i| (i * 2) as i32).collect();
        
        let sum = add_fixed_embeddings(&a, &b);
        
        for i in 0..EMBEDDING_SIZE {
            assert_eq!(sum[i], a[i] + b[i]);
        }
    }
    
    #[test]
    fn test_embedding_operations() {
        let a = EmbeddingVec::zeros();
        let mut b_values = [0.0; EMBEDDING_SIZE];
        for i in 0..EMBEDDING_SIZE {
            b_values[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let b = EmbeddingVec::new(b_values);
        
        // Test addition
        let sum = a.add(&b);
        assert!(sum.approx_eq(&b, 0.0001));
        
        // Test subtraction
        let diff = sum.sub(&b);
        assert!(diff.approx_eq(&a, 0.0001));
        
        // Test distance
        let distance = embedding_distance(&a, &b);
        assert!(distance > 0.0);
        
        // Test similarity (orthogonal vectors should have similarity ~0)
        let similarity = embedding_similarity(&a, &b);
        assert!(similarity.abs() < 0.0001);
    }
    
    #[test]
    fn test_normalize_embedding() {
        let mut values = [0.0; EMBEDDING_SIZE];
        values[0] = 3.0;
        values[1] = 4.0;
        
        let mut embedding = EmbeddingVec::new(values);
        normalize_embedding(&mut embedding);
        
        // Check that norm is approximately 1
        let norm = {
            let mut sum = 0.0;
            for &value in &embedding.0 {
                sum += value * value;
            }
            sum.sqrt()
        };
        
        assert!((norm - 1.0).abs() < 0.0001);
    }
    
    #[test]
    fn test_interpolate_embeddings() {
        let a = EmbeddingVec::zeros();
        let b_values = [1.0; EMBEDDING_SIZE];
        let b = EmbeddingVec::new(b_values);
        
        let interpolated = interpolate_embeddings(&a, &b, 0.5);
        
        for i in 0..EMBEDDING_SIZE {
            assert!((interpolated.0[i] - 0.5).abs() < 0.0001);
        }
    }
    
    #[test]
    fn test_state_update_validation() {
        let update = StateUpdate {
            old_state: StateSnapshot {
                height: 0,
                shard_id: 1,
                embedding: EmbeddingVec::zeros(),
                embedding_hash: [0u8; 32],
                metadata: StateMetadata {
                    epoch: 0,
                    timestamp: 0,
                    account_count: 0,
                    total_balance: 0,
                    proof: None,
                },
            },
            new_state: StateSnapshot {
                height: 1,
                shard_id: 1,
                embedding: EmbeddingVec::zeros(),
                embedding_hash: [0u8; 32],
                metadata: StateMetadata {
                    epoch: 0,
                    timestamp: 0,
                    account_count: 0,
                    total_balance: 0,
                    proof: None,
                },
            },
            batch: TransactionBatch::new(1, 1),
            homomorphism_error: HOMO_ERROR_BOUND * 0.5, // Within bounds
            update_proof: vec![],
        };
        
        // Should not error
        assert!(update.validate_error_bound().is_ok());
        
        // Test with error exceeding bounds
        let mut bad_update = update.clone();
        bad_update.homomorphism_error = HOMO_ERROR_BOUND * 3.0;
        assert!(bad_update.validate_error_bound().is_err());
    }
}
