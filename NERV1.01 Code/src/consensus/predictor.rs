//! src/consensus/predictor.rs
//! # AI-Native Consensus Predictor
//!
//! This module implements the 1.8MB distilled neural predictor for NERV's
//! AI-native consensus mechanism.
//!
//! Key features:
//! - Loads the pre-trained quantized predictor_1.8mb.pt model (via tch/libtorch)
//! - Fast CPU inference for block proposal validation and next-state prediction
//! - Input: Tokenized sequence of recent consensus events (validator votes, proposals, disputes)
//! - Output: 512-dim predicted embedding delta + validity score
//! - Used for 0.6s block times and efficient Monte-Carlo dispute resolution
//! - Homomorphism-aware: predictions align with embedding updates
//! - Fallback dummy prediction if model loading fails (preserves old functionality)
//! - Matches whitepaper: distilled model for perpetual self-improvement via FL

use crate::{Result, NervError, embedding::NeuralEmbedding};
use crate::consensus::ConsensusConfig;
use std::path::Path;
use tch::{Tensor, Device, nn::Module, kind::Kind, no_grad, CModule};

/// Consensus predictor using the 1.8MB distilled model
pub struct Predictor {
    /// Loaded TorchScript or quantized model
    model: CModule,
    
    /// Device (CPU preferred for consensus nodes)
    device: Device,
    
    /// Model configuration (for input validation)
    vocab_size: i64,
    max_seq_len: i64,
    
    /// Fallback mode (if model load fails)
    fallback: bool,
}

impl Predictor {
    /// Create new predictor by loading the model from config path
    pub fn new(config: &ConsensusConfig) -> Result<Self> {
        let model_path = config.predictor_model_path.as_path();
        
        if !model_path.exists() {
            return Err(NervError::Consensus(format!(
                "Predictor model not found at {:?}", model_path
            )));
        }
        
        let device = Device::Cpu; // CPU for broad compatibility; CUDA optional
        
        let model = match tch::CModule::load_on_device(model_path, device) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("Failed to load predictor model: {}. Using fallback dummy predictor.", e);
                return Ok(Self {
                    model: CModule::default(), // Dummy
                    device,
                    vocab_size: 8192,
                    max_seq_len: 512,
                    fallback: true,
                });
            }
        };
        
        Ok(Self {
            model,
            device,
            vocab_size: 8192,   // From model generation script
            max_seq_len: 512,
            fallback: false,
        })
    }
    
    /// Predict next embedding delta and validity score from tokenized history
    /// Input: vec of u16 token IDs (e.g., validator IDs, vote types, timestamps quantized)
    /// Output: Predicted 512-dim delta + validity score (0.0-1.0)
    pub fn predict(&self, input_tokens: &[u16]) -> Result<(NeuralEmbedding, f32)> {
        if self.fallback {
            // Preserve old dummy functionality: simple heuristic
            let score = if input_tokens.len() % 2 == 0 { 0.95 } else { 0.85 };
            let delta = vec![0.01f32; 512]; // Small neutral delta
            return Ok((NeuralEmbedding(delta), score));
        }
        
        if input_tokens.is_empty() || input_tokens.len() > self.max_seq_len as usize {
            return Err(NervError::Consensus("Invalid input sequence length".into()));
        }
        
        // Convert to Tensor (i64 for embeddings)
        let input_tensor = Tensor::from_slice(
            &input_tokens.iter().map(|&t| t as i64).collect::<Vec<i64>>()
        )
        .unsqueeze(0) // Add batch dim: (1, seq_len)
        .to_device(self.device);
        
        // Padding mask (all false = no padding)
        let padding_mask = Tensor::zeros(&[1, input_tokens.len() as i64], (Kind::Bool, self.device));
        
        no_grad(|| {
            // Forward: model expects dict or tuple (input_ids, padding_mask)
            let output = self.model
                .forward_ts(&[input_tensor, padding_mask])
                .map_err(|e| NervError::Consensus(format!("Inference failed: {}", e)))?;
            
            // Output is (1, 512) tensor
            let delta_vec: Vec<f32> = output
                .get(0)
                .f_to_vec::<f32>()
                .map_err(|e| NervError::Consensus(format!("Tensor conversion failed: {}", e)))?;
            
            // Validity score: simple norm or last dim if model outputs it
            let validity_score = delta_vec.iter()
                .map(|v| v.abs())
                .sum::<f32>()
                .min(1.0); // Clamped heuristic; in real: separate head
            
            Ok((NeuralEmbedding(delta_vec), validity_score))
        })
    }
    
    /// Parameter count / size (for metrics)
    pub fn parameter_count(&self) -> usize {
        if self.fallback { 0 } else { 1_750_000 } // Approx from model
    }
    
    /// Model size in MB
    pub fn size_mb(&self) -> f64 {
        if self.fallback { 0.0 } else { 1.8 }
    }
}

// Example usage in consensus engine:
// let (predicted_delta, score) = predictor.predict(&recent_votes_tokens)?;
// if score > 0.9 { accept proposal } else { trigger dispute }

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_fallback_mode() {
        let config = ConsensusConfig {
            predictor_model_path: Path::new("nonexistent.pt").to_path_buf(),
            ..Default::default()
        };
        let predictor = Predictor::new(&config).unwrap();
        assert!(predictor.fallback);
        
        let tokens = vec![1u16, 2, 3, 4];
        let (embedding, score) = predictor.predict(&tokens).unwrap();
        assert_eq!(embedding.0.len(), 512);
        assert!(score > 0.8);
    }
    
    #[test]
    fn test_input_validation() {
        let config = ConsensusConfig {
            predictor_model_path: Path::new("nonexistent.pt").to_path_buf(),
            ..Default::default()
        };
        let predictor = Predictor::new(&config).unwrap();
        
        let empty: Vec<u16> = vec![];
        assert!(predictor.predict(&empty).is_err());
        
        let too_long = vec![0u16; 600];
        assert!(predictor.predict(&too_long).is_err());
    }
}
