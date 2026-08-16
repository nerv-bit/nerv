//! Reed-Solomon erasure coding for shard data availability
//! 
//! This module implements Reed-Solomon (k=5, m=2) erasure coding:
//! - 5 data fragments + 2 parity fragments = 7 total fragments
//! - Can tolerate loss of any 2 fragments
//! - 40% storage overhead for 100% data availability
//! - Optimized for blockchain state storage


use crate::Result;
use serde::{Deserialize, Serialize};
use reed_solomon_erasure::galois_8::ReedSolomon;
use std::sync::Arc;
use tokio::sync::RwLock;


/// Erasure coding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErasureCodingConfig {
    /// Number of data shards (k)
    pub data_shards: usize,
    
    /// Number of parity shards (m)
    pub parity_shards: usize,
    
    /// Total shards (k + m)
    pub total_shards: usize,
    
    /// Shard size in bytes
    pub shard_size: usize,
    
    /// Maximum shard size (for padding)
    pub max_shard_size: usize,
    
    /// Recovery threshold (minimum shards needed)
    pub recovery_threshold: usize,
    
    /// Enable/disable erasure coding
    pub enabled: bool,
    
    /// Storage backend type
    pub storage_backend: StorageBackend,
    
    /// Verification mode
    pub verification_mode: VerificationMode,
}


impl Default for ErasureCodingConfig {
    fn default() -> Self {
        Self {
            data_shards: 5,    // k
            parity_shards: 2,  // m
            total_shards: 7,   // k + m
            shard_size: 65536, // 64KB per shard
            max_shard_size: 131072, // 128KB max
            recovery_threshold: 5, // Need at least k shards
            enabled: true,
            storage_backend: StorageBackend::Distributed,
            verification_mode: VerificationMode::OnDemand,
        }
    }
}


/// Storage backend for erasure coded data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageBackend {
    /// Store on IPFS/Arweave
    Distributed,
    
    /// Store on validator nodes
    Validator,
    
    /// Store on dedicated storage nodes
    StorageNodes,
    
    /// Hybrid approach
    Hybrid,
}


/// Verification mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationMode {
    /// Verify on demand (when reading)
    OnDemand,
    
    /// Verify periodically
    Periodic,
    
    /// Verify always (on write and read)
    Always,
    
    /// No verification (trust-based)
    None,
}


/// Encoded shard data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedShard {
    /// Shard index (0-based)
    pub index: usize,
    
    /// Shard data
    pub data: Vec<u8>,
    
    /// Checksum (SHA256)
    pub checksum: [u8; 32],
    
    /// Storage location
    pub storage_location: StorageLocation,
    
    /// Timestamp
    pub timestamp: u64,
    
    /// Version
    pub version: u32,
    
    /// Proof of storage (for verification)
    pub proof_of_storage: Option<Vec<u8>>,
}


/// Storage location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageLocation {
    /// Node ID storing this shard
    pub node_id: [u8; 20],
    
    /// IP address (if known)
    pub ip_address: Option<String>,
    
    /// Storage path/URI
    pub path: String,
    
    /// Availability score (0-1)
    pub availability_score: f64,
    
    /// Last ping timestamp
    pub last_ping: u64,
}


/// Reed-Solomon encoder
#[derive(Debug, Clone)]
pub struct ReedSolomonEncoder {
    /// Reed-Solomon encoder instance
    encoder: Arc<ReedSolomon>,
    
    /// Configuration
    config: ErasureCodingConfig,
    
    /// Statistics
    stats: Arc<RwLock<EncoderStats>>,
    
    /// Cache for recent encodings
    encoding_cache: Arc<RwLock<EncodingCache>>,
}


impl ReedSolomonEncoder {
    /// Create a new Reed-Solomon encoder
    pub fn new(config: ErasureCodingConfig) -> Self {
        let encoder = ReedSolomon::new(config.data_shards, config.parity_shards)
            .expect("Failed to create Reed-Solomon encoder");
        
        Self {
            encoder: Arc::new(encoder),
            config,
            stats: Arc::new(RwLock::new(EncoderStats::default())),
            encoding_cache: Arc::new(RwLock::new(EncodingCache::new(1000))),
        }
    }
    
    /// Encode data into shards
    pub fn encode(&self, data: &[u8], config: &ErasureCodingConfig) -> Result<Vec<Vec<u8>>> {
        let start_time = std::time::Instant::now();
        
        // Check cache first
        let cache_key = blake3::hash(data);
        if let Some(cached) = self.encoding_cache.blocking_read().get(&cache_key) {
            return Ok(cached.clone());
        }
        
        // Pad data to multiple of shard size if needed
        let padded_data = self.pad_data(data, config)?;
        
        // Split data into shards
        let shard_size = (padded_data.len() + config.data_shards - 1) / config.data_shards;
        let mut shards = self.split_into_shards(&padded_data, shard_size, config.data_shards)?;
        
        // Extend with empty parity shards
        shards.resize(config.total_shards, vec![0u8; shard_size]);
        
        // Encode parity
        self.encoder.encode(&mut shards)
            .map_err(|e| ShardingError::ErasureCodingError(format!("Encoding failed: {}", e)))?;
        
        // Update statistics
        let mut stats = self.stats.blocking_write();
        stats.update_encode(data.len(), shard_size, start_time.elapsed());
        
        // Cache result
        self.encoding_cache.blocking_write().insert(cache_key, shards.clone());
        
        Ok(shards)
    }
    
    /// Reconstruct missing shards
    pub fn reconstruct(&self, shards: &[Option<Vec<u8>>], config: &ErasureCodingConfig) -> Result<Vec<Vec<u8>>> {
        let start_time = std::time::Instant::now();
        
        // Check if we have enough shards for reconstruction
        let present_shards: usize = shards.iter()
            .filter(|shard| shard.is_some())
            .count();
        
        if present_shards < config.recovery_threshold {
            return Err(ShardingError::ErasureCodingError(
                format!("Insufficient shards for reconstruction: {}/{}", 
                       present_shards, config.recovery_threshold)
            ).into());
        }
        
        // Convert to mutable shards for reconstruction
        let mut shards_vec: Vec<Option<Vec<u8>>> = shards.to_vec();
        
        // Determine shard size from first present shard
        let shard_size = shards_vec.iter()
            .find_map(|shard| shard.as_ref().map(|s| s.len()))
            .unwrap_or(config.shard_size);
        
        // Ensure all shards have correct size
        for shard in &mut shards_vec {
            if let Some(data) = shard {
                if data.len() < shard_size {
                    data.resize(shard_size, 0);
                }
            } else {
                *shard = Some(vec![0u8; shard_size]);
            }
        }
        
        // Reconstruct missing shards
        self.encoder.reconstruct(&mut shards_vec)
            .map_err(|e| ShardingError::ErasureCodingError(format!("Reconstruction failed: {}", e)))?;
        
        // Convert back to Vec<Vec<u8>>
        let reconstructed: Vec<Vec<u8>> = shards_vec.into_iter()
            .map(|shard| shard.unwrap_or_else(|| vec![0u8; shard_size]))
            .collect();
        
        // Update statistics
        let mut stats = self.stats.blocking_write();
        stats.update_reconstruct(present_shards, config.total_shards, start_time.elapsed());
        
        Ok(reconstructed)
    }
    
    /// Verify shard integrity
    pub fn verify_shards(&self, shards: &[Vec<u8>], config: &ErasureCodingConfig) -> Result<bool> {
        if shards.len() != config.total_shards {
            return Err(ShardingError::ErasureCodingError(
                format!("Invalid number of shards: {}/{}", shards.len(), config.total_shards)
            ).into());
        }
        
        // Check shard sizes
        let shard_size = shards[0].len();
        for (i, shard) in shards.iter().enumerate() {
            if shard.len() != shard_size {
                return Err(ShardingError::ErasureCodingError(
                    format!("Shard {} has incorrect size: {}/{}", i, shard.len(), shard_size)
                ).into());
            }
        }
        
        // Verify Reed-Solomon parity
        let mut shards_copy = shards.to_vec();
        let result = self.encoder.verify(&shards_copy);
        
        // Update statistics
        let mut stats = self.stats.blocking_write();
        stats.update_verify(result);
        
        Ok(result)
    }
    
    /// Get encoder statistics
    pub fn get_stats(&self) -> EncoderStats {
        self.stats.blocking_read().clone()
    }
    
    // Private helper methods
    
    fn pad_data(&self, data: &[u8], config: &ErasureCodingConfig) -> Result<Vec<u8>> {
        let total_size = config.data_shards * config.shard_size;
        
        if data.len() > total_size {
            return Err(ShardingError::ErasureCodingError(
                format!("Data too large: {} > {}", data.len(), total_size)
            ).into());
        }
        
        let mut padded = data.to_vec();
        
        // Add padding if needed
        if padded.len() < total_size {
            let padding_size = total_size - padded.len();
            padded.extend(vec![0u8; padding_size]);
        }
        
        Ok(padded)
    }
    
    fn split_into_shards(&self, data: &[u8], shard_size: usize, num_shards: usize) -> Result<Vec<Vec<u8>>> {
        let mut shards = Vec::with_capacity(num_shards);
        
        for i in 0..num_shards {
            let start = i * shard_size;
            let end = (i + 1) * shard_size;
            
            if start < data.len() {
                let end = end.min(data.len());
                shards.push(data[start..end].to_vec());
            } else {
                shards.push(vec![0u8; shard_size]);
            }
        }
        
        Ok(shards)
    }
}


/// Reed-Solomon decoder
#[derive(Debug, Clone)]
pub struct ReedSolomonDecoder {
    /// Reed-Solomon decoder instance
    decoder: Arc<ReedSolomon>,
    
    /// Configuration
    config: ErasureCodingConfig,
    
    /// Statistics
    stats: Arc<RwLock<DecoderStats>>,
}


impl ReedSolomonDecoder {
    /// Create a new Reed-Solomon decoder
    pub fn new(config: ErasureCodingConfig) -> Self {
        let decoder = ReedSolomon::new(config.data_shards, config.parity_shards)
            .expect("Failed to create Reed-Solomon decoder");
        
        Self {
            decoder: Arc::new(decoder),
            config,
            stats: Arc::new(RwLock::new(DecoderStats::default())),
        }
    }
    
    /// Decode data from shards
    pub fn decode(&self, shards: &[Vec<u8>], config: &ErasureCodingConfig) -> Result<Vec<u8>> {
        let start_time = std::time::Instant::now();
        
        // Verify we have enough shards
        if shards.len() < config.recovery_threshold {
            return Err(ShardingError::ErasureCodingError(
                format!("Insufficient shards for decoding: {}/{}", 
                       shards.len(), config.recovery_threshold)
            ).into());
        }
        
        // Convert to option shards (some may be missing)
        let mut option_shards: Vec<Option<Vec<u8>>> = Vec::with_capacity(config.total_shards);
        
        for i in 0..config.total_shards {
            if i < shards.len() && !shards[i].is_empty() {
                option_shards.push(Some(shards[i].clone()));
            } else {
                option_shards.push(None);
            }
        }
        
        // Reconstruct if necessary
        let present_count = option_shards.iter().filter(|s| s.is_some()).count();
        
        let reconstructed_shards = if present_count < config.total_shards {
            // Need reconstruction
            self.decoder.reconstruct(&mut option_shards)
                .map_err(|e| ShardingError::ErasureCodingError(format!("Reconstruction failed: {}", e)))?;
            
            option_shards.into_iter()
                .map(|s| s.unwrap_or_else(|| vec![0u8; config.shard_size]))
                .collect()
        } else {
            // All shards present
            option_shards.into_iter()
                .map(|s| s.unwrap())
                .collect()
        };
        
        // Extract data shards
        let data_shards = &reconstructed_shards[..config.data_shards];
        
        // Combine data shards
        let combined = self.combine_shards(data_shards)?;
        
        // Remove padding
        let decoded = self.remove_padding(&combined)?;
        
        // Update statistics
        let mut stats = self.stats.blocking_write();
        stats.update_decode(present_count, config.total_shards, start_time.elapsed());
        
        Ok(decoded)
    }
    
    /// Check if data can be recovered from available shards
    pub fn can_recover(&self, available_shards: usize) -> bool {
        available_shards >= self.config.recovery_threshold
    }
    
    /// Get optimal shard distribution for given reliability target
    pub fn get_optimal_distribution(&self, reliability_target: f64) -> (usize, usize) {
        // Calculate required parity shards for given reliability
        // This is a simplified calculation
        let base_parity = 2; // Minimum parity
        let additional_parity = ((1.0 - reliability_target) * 10.0).ceil() as usize;
        
        let data_shards = 5; // Fixed k=5
        let parity_shards = base_parity + additional_parity;
        
        (data_shards, parity_shards)
    }
    
    /// Get decoder statistics
    pub fn get_stats(&self) -> DecoderStats {
        self.stats.blocking_read().clone()
    }
    
    // Private helper methods
    
    fn combine_shards(&self, shards: &[Vec<u8>]) -> Result<Vec<u8>> {
        if shards.is_empty() {
            return Ok(Vec::new());
        }
        
        let shard_size = shards[0].len();
        let total_size = shards.len() * shard_size;
        
        let mut combined = Vec::with_capacity(total_size);
        
        for shard in shards {
            combined.extend_from_slice(shard);
        }
        
        Ok(combined)
    }
    
    fn remove_padding(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Find first zero byte (assuming null-terminated or similar)
        // In production: use actual padding scheme
        let mut end = data.len();
        
        for (i, &byte) in data.iter().enumerate().rev() {
            if byte != 0 {
                end = i + 1;
                break;
            }
        }
        
        Ok(data[..end].to_vec())
    }
}


/// Encoder statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderStats {
    /// Total encodings performed
    pub total_encodings: u64,
    
    /// Total data encoded (bytes)
    pub total_data_encoded: u64,
    
    /// Total parity generated (bytes)
    pub total_parity_generated: u64,
    
    /// Average encoding time (ms)
    pub avg_encoding_time_ms: f64,
    
    /// Encoding success rate
    pub encoding_success_rate: f64,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
    
    /// Storage overhead (parity/data ratio)
    pub storage_overhead: f64,
    
    /// Last encoding timestamp
    pub last_encoding_time: u64,
}


impl Default for EncoderStats {
    fn default() -> Self {
        Self {
            total_encodings: 0,
            total_data_encoded: 0,
            total_parity_generated: 0,
            avg_encoding_time_ms: 0.0,
            encoding_success_rate: 1.0,
            cache_hit_rate: 0.0,
            storage_overhead: 0.4, // 40% overhead for k=5, m=2
            last_encoding_time: 0,
        }
    }
}


impl EncoderStats {
    fn update_encode(&mut self, data_size: usize, shard_size: usize, duration: std::time::Duration) {
        self.total_encodings += 1;
        self.total_data_encoded += data_size as u64;
        self.total_parity_generated += (shard_size * 2) as u64; // m=2 parity shards
        self.avg_encoding_time_ms = (self.avg_encoding_time_ms * (self.total_encodings - 1) as f64 +
                                     duration.as_millis() as f64) / self.total_encodings as f64;
        self.last_encoding_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }
    
    fn update_reconstruct(&mut self, present_shards: usize, total_shards: usize, duration: std::time::Duration) {
        // Update reconstruction statistics
        // (This would be extended in production)
    }
    
    fn update_verify(&mut self, success: bool) {
        // Update verification statistics
        let alpha = 0.01;
        self.encoding_success_rate = alpha * (success as u8 as f64) + 
                                    (1.0 - alpha) * self.encoding_success_rate;
    }
}


/// Decoder statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderStats {
    /// Total decodings performed
    pub total_decodings: u64,
    
    /// Total reconstructions performed
    pub total_reconstructions: u64,
    
    /// Successful decodings
    pub successful_decodings: u64,
    
    /// Failed decodings
    pub failed_decodings: u64,
    
    /// Average decoding time (ms)
    pub avg_decoding_time_ms: f64,
    
    /// Average reconstruction time (ms)
    pub avg_reconstruction_time_ms: f64,
    
    /// Data recovery rate
    pub data_recovery_rate: f64,
    
    /// Last decoding timestamp
    pub last_decoding_time: u64,
}


impl Default for DecoderStats {
    fn default() -> Self {
        Self {
            total_decodings: 0,
            total_reconstructions: 0,
            successful_decodings: 0,
            failed_decodings: 0,
            avg_decoding_time_ms: 0.0,
            avg_reconstruction_time_ms: 0.0,
            data_recovery_rate: 1.0,
            last_decoding_time: 0,
        }
    }
}


impl DecoderStats {
    fn update_decode(&mut self, present_shards: usize, total_shards: usize, duration: std::time::Duration) {
        self.total_decodings += 1;
        self.successful_decodings += 1;
        
        if present_shards < total_shards {
            self.total_reconstructions += 1;
            self.avg_reconstruction_time_ms = (self.avg_reconstruction_time_ms * (self.total_reconstructions - 1) as f64 +
                                             duration.as_millis() as f64) / self.total_reconstructions as f64;
        } else {
            self.avg_decoding_time_ms = (self.avg_decoding_time_ms * (self.total_decodings - self.total_reconstructions - 1) as f64 +
                                        duration.as_millis() as f64) / (self.total_decodings - self.total_reconstructions) as f64;
        }
        
        self.last_decoding_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }
}


/// Encoding cache
#[derive(Debug, Clone)]
struct EncodingCache {
    max_size: usize,
    entries: Vec<CacheEntry>,
    index: std::collections::HashMap<[u8; 32], usize>,
}


impl EncodingCache {
    fn new(max_size: usize) -> Self {
        Self {
            max_size,
            entries: Vec::with_capacity(max_size),
            index: std::collections::HashMap::new(),
        }
    }
    
    fn get(&self, key: &[u8; 32]) -> Option<Vec<Vec<u8>>> {
        self.index.get(key)
            .and_then(|&idx| self.entries.get(idx))
            .map(|entry| entry.shards.clone())
    }
    
    fn insert(&mut self, key: [u8; 32], shards: Vec<Vec<u8>>) {
        // Remove oldest if cache is full
        if self.entries.len() >= self.max_size {
            if let Some(oldest) = self.entries.first() {
                self.index.remove(&oldest.key);
            }
            self.entries.remove(0);
        }
        
        // Add new entry
        let idx = self.entries.len();
        self.entries.push(CacheEntry { key, shards });
        self.index.insert(key, idx);
    }
}


/// Cache entry
#[derive(Debug, Clone)]
struct CacheEntry {
    key: [u8; 32],
    shards: Vec<Vec<u8>>,
}
