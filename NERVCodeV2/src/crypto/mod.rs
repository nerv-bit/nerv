//! Post-Quantum Cryptography Module for NERV Blockchain
//! 
//! This module implements the NIST-standardized post-quantum cryptographic
//! algorithms selected for NERV's security against quantum attacks:
//! 
//! 1. CRYSTALS-Dilithium-3: Primary digital signatures (ML-DSA-87)
//! 2. ML-KEM-768: Key encapsulation mechanism (Kyber-768)
//! 3. SPHINCS+-SHA256-192s: Backup stateless hash-based signatures
//! 
//! All implementations are constant-time, side-channel resistant, and 
//! follow NIST FIPS 203/204/205 standards.

mod dilithium;
mod ml_kem;
mod sphincs;

pub use dilithium::{Dilithium3, DilithiumKeypair, DilithiumSignature, DilithiumError};
pub use ml_kem::{MlKem768, MlKemKeypair, MlKemCiphertext, MlKemError};
pub use sphincs::{SphincsPlus, SphincsKeypair, SphincsSignature, SphincsError};

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Main cryptographic provider for NERV
pub struct CryptoProvider {
    /// Dilithium-3 signature scheme (primary)
    pub dilithium: Dilithium3,
    
    /// ML-KEM-768 key encapsulation (for encryption)
    pub ml_kem: MlKem768,
    
    /// SPHINCS+ backup signature scheme
    pub sphincs: SphincsPlus,
    
    /// Configuration
    pub config: CryptoConfig,
    
    /// Random number generator state
    rng_state: RngState,
}

impl CryptoProvider {
    /// Create a new cryptographic provider with default configuration
    pub fn new() -> Result<Self, CryptoError> {
        Self::with_config(CryptoConfig::default())
    }
    
    /// Create a new cryptographic provider with custom configuration
    pub fn with_config(config: CryptoConfig) -> Result<Self, CryptoError> {
        // Initialize RNG
        let rng_state = RngState::new()?;
        
        // Initialize algorithms
        let dilithium = Dilithium3::new(config.dilithium_config.clone())?;
        let ml_kem = MlKem768::new(config.ml_kem_config.clone())?;
        let sphincs = SphincsPlus::new(config.sphincs_config.clone())?;
        
        Ok(Self {
            dilithium,
            ml_kem,
            sphincs,
            config,
            rng_state,
        })
    }
    
    /// Generate a new keypair for signatures (uses Dilithium-3 by default)
    pub fn generate_signature_keypair(&mut self) -> Result<DilithiumKeypair, CryptoError> {
        self.dilithium.generate_keypair(&mut self.rng_state)
    }
    
    /// Generate a new keypair for encryption (uses ML-KEM-768)
    pub fn generate_encryption_keypair(&mut self) -> Result<MlKemKeypair, CryptoError> {
        self.ml_kem.generate_keypair(&mut self.rng_state)
    }
    
    /// Generate a backup signature keypair (uses SPHINCS+)
    pub fn generate_backup_keypair(&mut self) -> Result<SphincsKeypair, CryptoError> {
        self.sphincs.generate_keypair(&mut self.rng_state)
    }
    
    /// Sign a message with Dilithium-3
    pub fn sign(&self, message: &[u8], secret_key: &[u8]) -> Result<DilithiumSignature, CryptoError> {
        self.dilithium.sign(message, secret_key, &mut self.rng_state)
    }
    
    /// Verify a Dilithium-3 signature
    pub fn verify(&self, message: &[u8], signature: &DilithiumSignature, public_key: &[u8]) -> Result<bool, CryptoError> {
        self.dilithium.verify(message, signature, public_key)
    }
    
    /// Encapsulate a symmetric key using ML-KEM
    pub fn encapsulate(&self, public_key: &[u8]) -> Result<(MlKemCiphertext, Vec<u8>), CryptoError> {
        self.ml_kem.encapsulate(public_key, &mut self.rng_state)
    }
    
    /// Decapsulate a symmetric key using ML-KEM
    pub fn decapsulate(&self, ciphertext: &MlKemCiphertext, secret_key: &[u8]) -> Result<Vec<u8>, CryptoError> {
        self.ml_kem.decapsulate(ciphertext, secret_key)
    }
    
    /// Sign a message with SPHINCS+ (backup)
    pub fn sign_backup(&self, message: &[u8], secret_key: &[u8]) -> Result<SphincsSignature, CryptoError> {
        self.sphincs.sign(message, secret_key, &mut self.rng_state)
    }
    
    /// Verify a SPHINCS+ signature
    pub fn verify_backup(&self, message: &[u8], signature: &SphincsSignature, public_key: &[u8]) -> Result<bool, CryptoError> {
        self.sphincs.verify(message, signature, public_key)
    }
    
    /// Generate a keypair for hybrid signatures (Dilithium + SPHINCS)
    pub fn generate_hybrid_keypair(&mut self) -> Result<HybridKeypair, CryptoError> {
        let dilithium_kp = self.generate_signature_keypair()?;
        let sphincs_kp = self.generate_backup_keypair()?;
        
        Ok(HybridKeypair {
            dilithium: dilithium_kp,
            sphincs: sphincs_kp,
        })
    }
    
    /// Create a hybrid signature (both Dilithium and SPHINCS)
    pub fn sign_hybrid(&mut self, message: &[u8], hybrid_kp: &HybridKeypair) -> Result<HybridSignature, CryptoError> {
        let dilithium_sig = self.dilithium.sign(message, &hybrid_kp.dilithium.secret_key, &mut self.rng_state)?;
        let sphincs_sig = self.sphincs.sign(message, &hybrid_kp.sphincs.secret_key, &mut self.rng_state)?;
        
        Ok(HybridSignature {
            dilithium: dilithium_sig,
            sphincs: sphincs_sig,
        })
    }
    
    /// Verify a hybrid signature
    pub fn verify_hybrid(&self, message: &[u8], signature: &HybridSignature, hybrid_pk: &HybridPublicKey) -> Result<bool, CryptoError> {
        let dilithium_valid = self.dilithium.verify(message, &signature.dilithium, &hybrid_pk.dilithium)?;
        let sphincs_valid = self.sphincs.verify(message, &signature.sphincs, &hybrid_pk.sphincs)?;
        
        Ok(dilithium_valid && sphincs_valid)
    }
    
    /// Generate secure random bytes
    pub fn random_bytes(&mut self, len: usize) -> Result<Vec<u8>, CryptoError> {
        self.rng_state.generate_bytes(len)
    }
    
    /// Get cryptographic configuration
    pub fn config(&self) -> &CryptoConfig {
        &self.config
    }
    
    /// Update configuration (reinitializes algorithms if needed)
    pub fn update_config(&mut self, config: CryptoConfig) -> Result<(), CryptoError> {
        // Reinitialize if algorithm parameters changed
        if config.dilithium_config != self.config.dilithium_config {
            self.dilithium = Dilithium3::new(config.dilithium_config.clone())?;
        }
        
        if config.ml_kem_config != self.config.ml_kem_config {
            self.ml_kem = MlKem768::new(config.ml_kem_config.clone())?;
        }
        
        if config.sphincs_config != self.config.sphincs_config {
            self.sphincs = SphincsPlus::new(config.sphincs_config.clone())?;
        }
        
        self.config = config;
        Ok(())
    }
}

/// Cryptographic configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CryptoConfig {
    /// Dilithium-3 configuration
    pub dilithium_config: DilithiumConfig,
    
    /// ML-KEM-768 configuration
    pub ml_kem_config: MlKemConfig,
    
    /// SPHINCS+ configuration
    pub sphincs_config: SphincsConfig,
    
    /// Enable/disable hybrid signatures
    pub enable_hybrid_signatures: bool,
    
    /// Strict validation mode (reject non-compliant inputs)
    pub strict_validation: bool,
    
    /// Enable constant-time verification
    pub constant_time_verification: bool,
    
    /// Maximum signature size allowed
    pub max_signature_size: usize,
    
    /// Maximum public key size allowed
    pub max_public_key_size: usize,
    
    /// Security level (1=128-bit, 2=192-bit, 3=256-bit)
    pub security_level: u8,
}

impl Default for CryptoConfig {
    fn default() -> Self {
        Self {
            dilithium_config: DilithiumConfig::default(),
            ml_kem_config: MlKemConfig::default(),
            sphincs_config: SphincsConfig::default(),
            enable_hybrid_signatures: true,
            strict_validation: true,
            constant_time_verification: true,
            max_signature_size: 5000, // Bytes
            max_public_key_size: 2000, // Bytes
            security_level: 3, // 256-bit security
        }
    }
}

/// Dilithium configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DilithiumConfig {
    /// Security parameter (2, 3, or 5)
    pub security_parameter: u8,
    
    /// Enable/disable deterministic signing
    pub deterministic_signing: bool,
    
    /// Enable/disable compression
    pub enable_compression: bool,
    
    /// Use robust mode (more secure but larger signatures)
    pub robust_mode: bool,
    
    /// Custom parameters (if any)
    pub custom_params: Option<DilithiumParams>,
}

impl Default for DilithiumConfig {
    fn default() -> Self {
        Self {
            security_parameter: 3, // Dilithium-3
            deterministic_signing: true,
            enable_compression: true,
            robust_mode: false,
            custom_params: None,
        }
    }
}

/// ML-KEM configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MlKemConfig {
    /// Security parameter (512, 768, or 1024)
    pub security_parameter: u16,
    
    /// Enable/disable CPA-secure mode
    pub cpa_secure: bool,
    
    /// Use deterministic key generation
    pub deterministic_keygen: bool,
    
    /// Custom parameters (if any)
    pub custom_params: Option<MlKemParams>,
}

impl Default for MlKemConfig {
    fn default() -> Self {
        Self {
            security_parameter: 768, // ML-KEM-768
            cpa_secure: true,
            deterministic_keygen: false,
            custom_params: None,
        }
    }
}

/// SPHINCS+ configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SphincsConfig {
    /// Security parameter (128, 192, or 256)
    pub security_parameter: u16,
    
    /// Hash function (SHAKE256 or SHA256)
    pub hash_function: SphincsHashFunction,
    
    /// Tree height
    pub tree_height: u8,
    
    /// Winternitz parameter
    pub wots_w: u16,
    
    /// Enable/disable fast signing mode
    pub fast_signing: bool,
    
    /// Custom parameters (if any)
    pub custom_params: Option<SphincsParams>,
}

impl Default for SphincsConfig {
    fn default() -> Self {
        Self {
            security_parameter: 192, // SPHINCS+-SHA256-192s
            hash_function: SphincsHashFunction::Sha256,
            tree_height: 16,
            wots_w: 16,
            fast_signing: true,
            custom_params: None,
        }
    }
}

/// SPHINCS+ hash function choice
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SphincsHashFunction {
    Sha256,
    Shake256,
}

/// Hybrid keypair (Dilithium + SPHINCS)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridKeypair {
    pub dilithium: DilithiumKeypair,
    pub sphincs: SphincsKeypair,
}

/// Hybrid public key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridPublicKey {
    pub dilithium: Vec<u8>,
    pub sphincs: Vec<u8>,
}

/// Hybrid signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSignature {
    pub dilithium: DilithiumSignature,
    pub sphincs: SphincsSignature,
}

/// Random number generator state
struct RngState {
    /// System RNG (for seeding)
    system_rng: Option<Box<dyn RngCore + Send + Sync>>,
    
    /// Deterministic RNG (for testing/reproducibility)
    deterministic_rng: Option<ChaCha20Rng>,
    
    /// Use deterministic mode
    deterministic_mode: bool,
    
    /// Entropy accumulator
    entropy_pool: EntropyPool,
}

impl RngState {
    fn new() -> Result<Self, CryptoError> {
        let mut entropy_pool = EntropyPool::new();
        
        // Collect initial entropy
        entropy_pool.mix_system_entropy()?;
        entropy_pool.mix_timing_entropy();
        
        // Initialize system RNG
        let system_rng = Some(Box::new(rand::rngs::OsRng) as Box<dyn RngCore + Send + Sync>);
        
        Ok(Self {
            system_rng,
            deterministic_rng: None,
            deterministic_mode: false,
            entropy_pool,
        })
    }
    
    fn generate_bytes(&mut self, len: usize) -> Result<Vec<u8>, CryptoError> {
        if self.deterministic_mode {
            // Use deterministic RNG
            let rng = self.deterministic_rng.as_mut()
                .ok_or_else(|| CryptoError::RngError("Deterministic RNG not initialized".into()))?;
            
            let mut bytes = vec![0u8; len];
            rng.fill_bytes(&mut bytes);
            Ok(bytes)
        } else {
            // Use system RNG
            let rng = self.system_rng.as_mut()
                .ok_or_else(|| CryptoError::RngError("System RNG not available".into()))?;
            
            let mut bytes = vec![0u8; len];
            rng.fill_bytes(&mut bytes);
            
            // Mix generated bytes into entropy pool
            self.entropy_pool.mix_bytes(&bytes);
            
            Ok(bytes)
        }
    }
    
    fn set_deterministic_mode(&mut self, seed: &[u8]) {
        self.deterministic_mode = true;
        let seed_array = Self::bytes_to_seed(seed);
        self.deterministic_rng = Some(ChaCha20Rng::from_seed(seed_array));
    }
    
    fn set_system_mode(&mut self) {
        self.deterministic_mode = false;
        self.deterministic_rng = None;
    }
    
    fn bytes_to_seed(seed: &[u8]) -> [u8; 32] {
        let mut seed_array = [0u8; 32];
        let len = seed.len().min(32);
        seed_array[..len].copy_from_slice(&seed[..len]);
        
        // If seed is shorter than 32 bytes, pad with zeros
        // In production: use key derivation function
        seed_array
    }
}

/// Entropy pool for cryptographic randomness
struct EntropyPool {
    pool: [u8; 64],
    pool_index: usize,
    entropy_count: u64,
}

impl EntropyPool {
    fn new() -> Self {
        Self {
            pool: [0u8; 64],
            pool_index: 0,
            entropy_count: 0,
        }
    }
    
    fn mix_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.pool[self.pool_index] ^= byte;
            self.pool_index = (self.pool_index + 1) % 64;
            self.entropy_count = self.entropy_count.wrapping_add(byte as u64);
        }
    }
    
    fn mix_system_entropy(&mut self) -> Result<(), CryptoError> {
        let mut system_bytes = vec![0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut system_bytes)
            .map_err(|e| CryptoError::RngError(format!("Failed to get system entropy: {}", e)))?;
        
        self.mix_bytes(&system_bytes);
        Ok(())
    }
    
    fn mix_timing_entropy(&mut self) {
        // Use timing as additional entropy source
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        
        let micros = now.as_micros().to_le_bytes();
        self.mix_bytes(&micros);
    }
    
    fn extract(&mut self, len: usize) -> Vec<u8> {
        let mut output = Vec::with_capacity(len);
        
        for i in 0..len {
            output.push(self.pool[(self.pool_index + i) % 64]);
        }
        
        // Stir the pool
        self.stir();
        
        output
    }
    
    fn stir(&mut self) {
        // Simple mixing function
        for i in 0..64 {
            self.pool[i] = self.pool[i].wrapping_add(self.pool[(i + 1) % 64]);
            self.pool[i] = self.pool[i].rotate_left(3);
        }
    }
}

/// Cryptographic errors
#[derive(Debug, Error)]
pub enum CryptoError {
    #[error("Dilithium error: {0}")]
    DilithiumError(#[from] DilithiumError),
    
    #[error("ML-KEM error: {0}")]
    MlKemError(#[from] MlKemError),
    
    #[error("SPHINCS+ error: {0}")]
    SphincsError(#[from] SphincsError),
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Invalid key size: expected {0}, got {1}")]
    InvalidKeySize(usize, usize),
    
    #[error("Invalid signature size: expected {0}, got {1}")]
    InvalidSignatureSize(usize, usize),
    
    #[error("Verification failed")]
    VerificationFailed,
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
    
    #[error("Random number generation error: {0}")]
    RngError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Unsupported algorithm: {0}")]
    UnsupportedAlgorithm(String),
}

/// Custom parameter types (simplified for this example)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DilithiumParams {
    // Implementation-specific parameters
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MlKemParams {
    // Implementation-specific parameters
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SphincsParams {
    // Implementation-specific parameters
}

// Re-export rand traits for internal use
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

// Trait for objects that can be serialized to/from bytes
pub trait ByteSerializable: Sized {
    fn to_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError>;
}

// Trait for objects that can be hex-encoded
pub trait HexEncodable {
    fn to_hex(&self) -> String;
    fn from_hex(hex: &str) -> Result<Self, CryptoError> where Self: Sized;
}

/// Utility functions for cryptographic operations
pub mod utils {
    use super::*;
    use blake3::Hasher as Blake3;
    use sha2::{Sha256, Sha512, Digest};
    use sha3::{Sha3_256, Sha3_512};
    
    /// Hash data using SHA3-256 (NERV standard)
    pub fn sha3_256(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(data);
        hasher.finalize().into()
    }
    
    /// Hash data using SHA3-512
    pub fn sha3_512(data: &[u8]) -> [u8; 64] {
        let mut hasher = Sha3_512::new();
        hasher.update(data);
        hasher.finalize().into()
    }
    
    /// Hash data using BLAKE3 (for performance)
    pub fn blake3_hash(data: &[u8]) -> [u8; 32] {
        let mut hasher = Blake3::new();
        hasher.update(data);
        *hasher.finalize().as_bytes()
    }
    
    /// Hash data using SHA256 (for compatibility)
    pub fn sha256(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().into()
    }
    
    /// Hash data using SHA512
    pub fn sha512(data: &[u8]) -> [u8; 64] {
        let mut hasher = Sha512::new();
        hasher.update(data);
        hasher.finalize().into()
    }
    
    /// Constant-time comparison of byte arrays
    pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        
        let mut result = 0u8;
        for (x, y) in a.iter().zip(b.iter()) {
            result |= x ^ y;
        }
        
        result == 0
    }
    
    /// Generate a key derivation using HKDF
    pub fn hkdf(ikm: &[u8], salt: &[u8], info: &[u8], output_len: usize) -> Result<Vec<u8>, CryptoError> {
        use hkdf::Hkdf;
        use sha2::Sha256;
        
        let hk = Hkdf::<Sha256>::new(Some(salt), ikm);
        let mut okm = vec![0u8; output_len];
        
        hk.expand(info, &mut okm)
            .map_err(|e| CryptoError::RngError(format!("HKDF expansion failed: {}", e)))?;
        
        Ok(okm)
    }
    
    /// XOR two byte arrays (constant-time)
    pub fn xor_bytes(a: &[u8], b: &[u8]) -> Result<Vec<u8>, CryptoError> {
        if a.len() != b.len() {
            return Err(CryptoError::InvalidParameter(
                format!("Arrays must have same length: {} != {}", a.len(), b.len())
            ));
        }
        
        let mut result = Vec::with_capacity(a.len());
        for (x, y) in a.iter().zip(b.iter()) {
            result.push(x ^ y);
        }
        
        Ok(result)
    }
    
    /// Encode bytes to hex string
    pub fn bytes_to_hex(bytes: &[u8]) -> String {
        hex::encode(bytes)
    }
    
    /// Decode hex string to bytes
    pub fn hex_to_bytes(hex: &str) -> Result<Vec<u8>, CryptoError> {
        hex::decode(hex).map_err(|e| CryptoError::DeserializationError(e.to_string()))
    }
    
    /// Base64 encode bytes
    pub fn base64_encode(bytes: &[u8]) -> String {
        base64::encode(bytes)
    }
    
    /// Base64 decode string
    pub fn base64_decode(encoded: &str) -> Result<Vec<u8>, CryptoError> {
        base64::decode(encoded).map_err(|e| CryptoError::DeserializationError(e.to_string()))
    }
    
    /// Pad data to block size
    pub fn pad_to_block(data: &[u8], block_size: usize) -> Vec<u8> {
        let mut padded = data.to_vec();
        let padding_len = block_size - (data.len() % block_size);
        
        if padding_len != block_size {
            padded.extend(vec![0u8; padding_len]);
        }
        
        padded
    }
    
    /// Unpad data from block size
    pub fn unpad_from_block(data: &[u8], block_size: usize) -> Result<Vec<u8>, CryptoError> {
        if data.len() % block_size != 0 {
            return Err(CryptoError::InvalidParameter(
                format!("Data length {} not multiple of block size {}", data.len(), block_size)
            ));
        }
        
        // Find first zero byte from the end
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
