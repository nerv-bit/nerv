//! src/lib.rs
//! # NERV - Neural State Embedding Blockchain
//!
//! This crate implements the NERV blockchain: a private, post-quantum,
//! infinitely scalable blockchain via neural state embeddings and useful-work.
//!
//! ## Core Architecture Components:
//! - **Neural State Embeddings**: 512-byte transformer-encoded ledger states
//! - **Homomorphic Updates**: Linear delta operations on embeddings
//! - **TEE-Bound Privacy**: 5-hop onion routing, cover traffic/jitter, hardware enclaves
//! - **AI-Native Consensus**: Neural prediction voting with Monte-Carlo disputes
//! - **Dynamic Neural Sharding**: Embedding bisection + LSTM load prediction
//! - **Post-Quantum Cryptography**: Dilithium/ML-KEM/SPHINCS+ from genesis
//! - **Useful-Work Economy**: Shapley-value federated learning rewards
//! - **Tokenomics**: Fair launch, emission schedule, vesting, rewards
//! - **Infinite Recursion**: Nova-based IVC for unbounded history proofs
//!
//! ## Key Innovations (from whitepaper):
//! 1. **Transfer Homomorphism**: ε_θ(S_{t+1}) = ε_θ(S_t) + δ(tx) with ≤1e-9 error
//! 2. **LatentLedger Circuit**: 7.9M-constraint Halo2 circuit + Nova folding
//! 3. **5-Hop TEE Mixer**: k-anonymity >1,000,000 with verifiable delay witnesses
//! 4. **Useful-Work Economy**: Shapley-value incentives for gradient contributions
// External crates (re-exports for convenience)
pub use anyhow::Result as AnyResult;
pub use thiserror::Error;
// Module declarations (complete set from all provided documents)
pub mod embedding;
pub mod privacy;
pub mod consensus;
pub mod sharding;
pub mod crypto;
pub mod economy;
pub mod network;
pub mod utils;
pub mod tokenomics;
// Re-export commonly used types for easier crate access
pub use embedding::{
    Encoder, NeuralState, Delta, TransferDelta, TransferTransaction,
    homomorphism::{TransferDelta as HomomorphismDelta, HomomorphismError},
    circuit::{DeltaCircuit, LatentLedgerCircuit, RecursiveProver, FixedPoint32_16, NervCircuitParams},
};
pub use privacy::{
    Mixer, MixConfig, VDWGenerator, VDWVerifier, TEERuntime, TEEAttestation,
    tee::{SealedState, SealingError},
    PrivacyManager,
};
pub use consensus::{Validator, Predictor, ConsensusEngine, DisputeManager, DisputeError};
pub use sharding::{ShardingManager, LstmLoadPredictor, EmbeddingBisection, ReedSolomonEncoder};
pub use crypto::{CryptoProvider, Dilithium3, MlKem768, SphincsPlus};
pub use economy::{EconomyManager, ShapleyComputer, FLAggregator, RewardDistributor};
pub use network::{NetworkManager, MempoolManager, GossipManager, DhtManager};
pub use tokenomics::{TokenomicsEngine, EmissionSchedule, VestingManager, RewardCalculator};
pub use utils::metrics::{MetricsCollector, Timer};
// Core error type for the entire library (unified across all modules)
#[derive(Debug, Error)]
pub enum NervError {
    #[error("Embedding error: {0}")]
    Embedding(String),
   #[error("Homomorphism error: {0}")]
    Homomorphism(String),
   #[error("Circuit error: {0}")]
    Circuit(String),
   #[error("Privacy/TEE error: {0}")]
    Privacy(String),
   #[error("VDW error: {0}")]
    VDW(String),
   #[error("TEE attestation failed: {0}")]
    TEEAttestation(String),
   #[error("Consensus error: {0}")]
    Consensus(String),
   #[error("Sharding error: {0}")]
    Sharding(String),
   #[error("Cryptographic error: {0}")]
    Crypto(String),
   #[error("Economy error: {0}")]
    Economy(String),
   #[error("Network error: {0}")]
    Network(String),
   #[error("Tokenomics error: {0}")]
    Tokenomics(String),
   #[error("Metrics error: {0}")]
    Metrics(String),
   #[error("Serialization error: {0}")]
    Serialization(String),
   #[error("IO error: {0}")]
    Io(String),
   #[error("Other error: {0}")]
    Other(String),
}
/// Result type alias for NERV operations
pub type Result = std::result::Result;
/// Global configuration for the NERV node (complete, covering all subsystems)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Config {
    /// Network ID (mainnet, testnet, devnet)
    pub network_id: String,
   /// Node role (full, light, relay, archive)
    pub node_role: NodeRole,
   /// Path to TEE attestation certificates (required for full/relay nodes)
    pub tee_cert_path: Option,
   /// Consensus parameters
    pub consensus: ConsensusConfig,
   /// Sharding parameters
    pub sharding: ShardingConfig,
   /// Privacy parameters
    pub privacy: PrivacyConfig,
   /// Economy parameters
    pub economy: EconomyConfig,
   /// Tokenomics parameters
    pub tokenomics: TokenomicsConfig,
   /// Network parameters
    pub network: network::NetworkConfig,
   /// Storage/data directory
    pub storage_path: std::path::PathBuf,
   /// Logging and metrics configuration
    pub logging_level: tracing::Level,
   /// Metrics exporter enabled
    pub enable_metrics: bool,
}
/// Node role enumeration
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum NodeRole {
    Full,
    Light,
    Relay,
    Archive,
}
/// Consensus-specific configuration
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConsensusConfig {
    pub min_stake: u128,
    pub reputation_decay: f64,
    pub predictor_model_path: std::path::PathBuf, // 1.8MB distilled model
    pub dispute_monte_carlo_samples: usize,
    pub dispute_tee_count: usize,
    pub target_block_time_ms: u64,
    pub finality_threshold: u64,
}
/// Sharding-specific configuration
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ShardingConfig {
    pub initial_shards: usize,
    pub lstm_model_path: std::path::PathBuf, // 1.1MB load predictor
    pub split_threshold: f64,
    pub merge_threshold_tps: u32,
    pub merge_time_window_secs: u64,
    pub erasure_k: usize,
    pub erasure_m: usize,
}
/// Privacy-specific configuration
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PrivacyConfig {
    pub mixer_hops: usize,
    pub target_anonymity_set: usize,
    pub cover_traffic_ratio: f64,
    pub jitter_mu_ms: u32,
    pub jitter_sigma_ms: u32,
    pub enable_vdw: bool,
    pub vdw_storage_backend: VDWStorageBackend, // Arweave/IPFS
}
/// VDW storage backend
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum VDWStorageBackend {
    Arweave,
    IPFS,
    Local,
}
/// Economy-specific configuration
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EconomyConfig {
    pub gradient_interval_sec: u64,
    pub dp_sigma: f64,
    pub shapley_samples: usize,
}
/// Main NERV node structure (integrates all subsystems)
pub struct NervNode {
    config: Config,
    state: NodeState,
    crypto_provider: crypto::CryptoProvider,
    network_manager: Arc,
    consensus_engine: Arc,
    sharding_manager: Arc,
    privacy_manager: Arc,
    economy_manager: Arc,
    tokenomics_engine: Arc,
    metrics_collector: Arc,
}
#[derive(Clone, Debug)]
pub struct NodeState {
    pub assigned_shards: Vec,
    pub validator_status: Option,
    pub reputation: f64,
    pub stake: u128,
    pub embedding_roots: std::collections::HashMap,
    pub node_id: [u8; 20],
}
impl NervNode {
    /// Create a new NERV node with the given configuration
    pub async fn new(config: Config) -> Result {
        // Crypto first (dependency for most components)
        let crypto_provider = crypto::CryptoProvider::new()
            .map_err(|e| NervError::Crypto(e.to_string()))?;
       // Metrics collector
        let metrics_collector = Arc::new(utils::metrics::MetricsCollector::new(config.clone().into())?);
       // Network layer
        let network_manager = Arc::new(network::NetworkManager::new(config.network.clone(), crypto_provider.clone()).await?);
       // Privacy manager (TEE + mixer + VDW)
        let privacy_manager = Arc::new(privacy::PrivacyManager::new(config.privacy.clone(), crypto_provider.clone()).await?);
       // Consensus engine
        let consensus_engine = Arc::new(consensus::ConsensusEngine::new(config.consensus.clone(), crypto_provider.clone()).await?);
       // Sharding manager
        let sharding_manager = Arc::new(sharding::ShardingManager::new(config.sharding.clone()).await?);
       // Economy manager
        let economy_manager = Arc::new(economy::EconomyManager::new(config.economy.clone()).await?);
       // Tokenomics engine
        let tokenomics_engine = Arc::new(tokenomics::TokenomicsEngine::new(config.tokenomics.clone())?);
       let state = NodeState {
            assigned_shards: Vec::new(),
            validator_status: None,
            reputation: 1.0,
            stake: 0,
            embedding_roots: std::collections::HashMap::new(),
            node_id: [0u8; 20], // Set properly in production via key management
        };
       Ok(Self {
            config,
            state,
            crypto_provider,
            network_manager,
            consensus_engine,
            sharding_manager,
            privacy_manager,
            economy_manager,
            tokenomics_engine,
            metrics_collector,
        })
    }
   /// Start the node and all subsystems
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting NERV node ({}) on {}", self.config.node_role, self.config.network_id);
       self.network_manager.join_network().await?;
        self.privacy_manager.start_mixer().await?;
        self.sharding_manager.discover_and_join_shards().await?;
        self.consensus_engine.start_validation().await?;
        self.economy_manager.start_epoch_monitoring().await?;
       tracing::info!("NERV node fully operational");
        Ok(())
    }
   /// Graceful shutdown
    pub async fn shutdown(&mut self) -> Result<()> {
        tracing::info!("Shutting down NERV node...");
        self.network_manager.leave_network().await?;
        self.privacy_manager.stop_mixer().await?;
        Ok(())
    }
}



// ============================================================================
// NERV BLOCKCHAIN CORE LIBRARY
// ============================================================================
// This file re-exports all public APIs and defines the top-level module structure.
// All modules are feature-gated to allow minimal compilation for light clients.
// ============================================================================

#![cfg_attr(feature = "tee-sgx", feature(sgx_platform))]
#![deny(
    missing_docs,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications,
    warnings
)]
#![allow(clippy::too_many_arguments)] // Cryptographic functions need many params
#![allow(clippy::type_complexity)] // ZK circuits are complex by nature

// ============================================================================
// RE-EXPORTED TYPES - Core types used throughout the codebase
// ============================================================================

/// 512-byte neural state embedding (compressed representation of ledger state)
pub type Embedding = [f32; 512];

/// 32-byte hash (BLAKE3 output) used for embedding roots and transaction IDs
pub type Hash = [u8; 32];

/// Shard identifier (16 bytes)
pub type ShardId = [u8; 16];

/// Transaction delta vector (512 dimensions)
pub type DeltaVector = [f32; 512];

/// Fixed-point 32.16 representation for embedding arithmetic
pub type FixedPoint = i32; // 16 bits integer, 16 bits fractional

// ============================================================================
// GLOBAL ERROR TYPE - Unified error handling across all modules
// ============================================================================

/// Unified error type for the NERV blockchain
#[derive(thiserror::Error, Debug)]
pub enum NervError {
    /// ZK proof generation/verification failed
    #[error("ZK proof error: {0}")]
    ZkProof(String),
    
    /// Neural network inference failed
    #[error("Neural network error: {0}")]
    NeuralNetwork(String),
    
    /// Cryptographic operation failed
    #[error("Cryptographic error: {0}")]
    Crypto(String),
    
    /// TEE attestation or sealing failed
    #[error("TEE error: {0}")]
    Tee(String),
    
    /// Network communication error
    #[error("Network error: {0}")]
    Network(String),
    
    /// Invalid state or consensus violation
    #[error("State error: {0}")]
    State(String),
    
    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
}

/// Result type alias for NERV operations
pub type Result<T> = std::result::Result<T, NervError>;

// ============================================================================
// MODULE DECLARATIONS - Feature-gated module exports
// ============================================================================

// Core modules always included
pub mod params;
pub mod utils;

// Embedding module (required for all node types)
#[cfg(feature = "embedding")]
pub mod embedding;

// Privacy module (TEE and VDW) - requires TEE features
#[cfg(any(feature = "tee-sgx", feature = "tee-sev", feature = "tee-arm"))]
pub mod privacy;

// Consensus module (required for validator nodes)
#[cfg(feature = "consensus")]
pub mod consensus;

// Sharding module (required for full nodes)
#[cfg(feature = "sharding")]
pub mod sharding;

// Cryptography module (always included for light clients)
pub mod crypto;

// Economy module (useful-work rewards)
#[cfg(feature = "economy")]
pub mod economy;

// Network module (P2P networking)
#[cfg(feature = "network")]
pub mod network;

// ============================================================================
// RE-EXPORTS - Public API surface
// ============================================================================

// Core types
pub use params::{EMBEDDING_SIZE, BATCH_SIZE, HOMO_ERROR_BOUND, EPOCH_DAYS};
pub use utils::serialization::{ProtobufEncoder, ProtobufDecoder};

// Embedding APIs
#[cfg(feature = "embedding")]
pub use embedding::{NeuralEncoder, TransferHomomorphism, DeltaComputation};

// Privacy APIs
#[cfg(any(feature = "tee-sgx", feature = "tee-sev", feature = "tee-arm"))]
pub use privacy::{TeeMixer, VerifiableDelayWitness, VdwVerifier};

// Consensus APIs
#[cfg(feature = "consensus")]
pub use consensus::{NeuralPredictor, ConsensusEngine, WeightedQuorum};

// Sharding APIs
#[cfg(feature = "sharding")]
pub use sharding::{ShardManager, EmbeddingBisection, LoadPredictor};

// Crypto APIs
pub use crypto::{DilithiumSigner, MlKemEncryptor, SphincsBackup};

// Economy APIs
#[cfg(feature = "economy")]
pub use economy::{ShapleyComputer, GradientRewards, FederatedLearning};

// Network APIs
#[cfg(feature = "network")]
pub use network::{P2PNode, DhtRegistry, GossipProtocol};

// ============================================================================
// INITIALIZATION FUNCTION - Global blockchain initialization
// ============================================================================

/// Initialize the NERV blockchain with given configuration
/// 
/// # Arguments
/// * `config` - Blockchain configuration
/// 
/// # Returns
/// * `Result<()>` - Success or error
/// 
/// # Example
/// ```no_run
/// use nerv::initialize;
/// use nerv::params::Config;
/// 
/// let config = Config::default();
/// initialize(config).unwrap();
/// ```
pub fn initialize(config: params::Config) -> Result<()> {
    // Initialize logging
    utils::logging::init_logging(&config.log_level)?;
    
    // Log initialization
    tracing::info!(
        "Initializing NERV blockchain v{}",
        env!("CARGO_PKG_VERSION")
    );
    tracing::info!("Embedding size: {} bytes", EMBEDDING_SIZE);
    tracing::info!("Batch size: {} transactions", BATCH_SIZE);
    tracing::info!("Homomorphism error bound: {}", HOMO_ERROR_BOUND);
    
    // Initialize cryptography
    crypto::initialize_crypto_rng()?;
    
    // Initialize TEE if enabled
    #[cfg(any(feature = "tee-sgx", feature = "tee-sev", feature = "tee-arm"))]
    privacy::tee::initialize_tee(&config.tee_config)?;
    
    // Initialize neural models
    #[cfg(feature = "embedding")]
    embedding::initialize_models(&config.model_path)?;
    
    Ok(())
}

// ============================================================================
// VERSION INFORMATION
// ============================================================================

/// Get the current version of the NERV blockchain
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Get the target fair launch date
pub fn launch_date() -> &'static str {
    "June 2028"
}

/// Get the git commit hash at compile time
pub fn git_commit() -> &'static str {
    env!("GIT_HASH")
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }
    
    #[test]
    fn test_launch_date() {
        assert_eq!(launch_date(), "June 2028");
    }
}
