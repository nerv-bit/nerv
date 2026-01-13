// src/params.rs
// ============================================================================
// NERV GLOBAL PARAMETERS AND CONFIGURATION
// ============================================================================
// This file defines all constants, configuration structures, and global settings
// for the NERV blockchain as specified in the whitepaper v1.01.
// ============================================================================

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ============================================================================
// FIXED CONSTANTS - Never change after genesis
// ============================================================================

/// Size of neural state embeddings in bytes (512 dimensions × 4 bytes/f32)
pub const EMBEDDING_SIZE: usize = 512;

/// Maximum transactions per batch (optimal for homomorphic updates)
pub const BATCH_SIZE: usize = 256;

/// Homomorphism error bound (10^-9) as specified in Theorem 1
pub const HOMO_ERROR_BOUND: f64 = 1e-9;

/// Epoch length in days for encoder updates (30 days)
pub const EPOCH_DAYS: u64 = 30;

/// Number of hops in the TEE mixer (5-hop anonymity)
pub const MIXER_HOPS: usize = 5;

/// Number of confirmations for deep reorg safety (10 blocks)
pub const DEEP_REORG_CONFIRMATIONS: u64 = 10;

/// Shard split threshold (92% overload probability)
pub const SHARD_SPLIT_THRESHOLD: f64 = 0.92;

/// Shard merge threshold (10 TPS for 10 minutes)
pub const SHARD_MERGE_THRESHOLD_TPS: f64 = 10.0;
pub const SHARD_MERGE_DURATION_MINUTES: u64 = 10;

/// Useful-work gradient contribution interval (15 seconds or 1000 txs)
pub const GRADIENT_INTERVAL_SECONDS: u64 = 15;
pub const GRADIENT_INTERVAL_TXS: u64 = 1000;

/// VDW size limits (average 1.4KB, max 1.8KB)
pub const VDW_AVERAGE_SIZE: usize = 1400; // 1.4KB
pub const VDW_MAX_SIZE: usize = 1800;     // 1.8KB

/// Light client sync size (<100KB forever)
pub const LIGHT_CLIENT_MAX_SIZE: usize = 100 * 1024; // 100KB

/// Reed-Solomon erasure coding parameters (k=5, m=2 for 40% fault tolerance)
pub const RS_K: usize = 5;
pub const RS_M: usize = 2;
pub const RS_TOTAL: usize = RS_K + RS_M; // 7 total replicas

/// Monte-Carlo dispute resolution parameters
pub const MONTE_CARLO_SIMULATIONS: usize = 10_000;
pub const MONTE_CARLO_TEES: usize = 32;

/// Consensus thresholds (67% for finality)
pub const CONSENSUS_THRESHOLD: f64 = 0.67;
pub const CHALLENGE_BOND_PERCENT: f64 = 0.01; // 1% bond for challenges

// ============================================================================
// POST-QUANTUM CRYPTOGRAPHY PARAMETERS
// ============================================================================

/// Dilithium-3 parameters (NIST Level 3)
pub const DILITHIUM_PUBLIC_KEY_SIZE: usize = 1809;  // ~1.8KB
pub const DILITHIUM_SECRET_KEY_SIZE: usize = 4000;  // ~4KB
pub const DILITHIUM_SIGNATURE_SIZE: usize = 3293;   // ~3.3KB
pub const DILITHIUM_VERIFY_TIME_US: u64 = 58;       // 58 microseconds

/// ML-KEM-768 parameters (formerly Kyber-768)
pub const ML_KEM_PUBLIC_KEY_SIZE: usize = 1184;     // ~1.2KB
pub const ML_KEM_SECRET_KEY_SIZE: usize = 2400;     // ~2.4KB
pub const ML_KEM_CIPHERTEXT_SIZE: usize = 1088;     // ~1.1KB
pub const ML_KEM_ENCAPSULATION_TIME_US: u64 = 42;   // 42 microseconds

/// SPHINCS+-SHA256-192s-robust parameters
pub const SPHINCS_PUBLIC_KEY_SIZE: usize = 48;      // 48 bytes
pub const SPHINCS_SECRET_KEY_SIZE: usize = 96;      // 96 bytes
pub const SPHINCS_SIGNATURE_SIZE: usize = 41000;    // ~41KB
pub const SPHINCS_SIGN_TIME_US: u64 = 1_000_000;    // 1 second (slow)

// ============================================================================
// NEURAL NETWORK PARAMETERS
// ============================================================================

/// Transformer encoder architecture (24 layers)
pub const TRANSFORMER_LAYERS: usize = 24;
pub const TRANSFORMER_HEADS: usize = 8;
pub const TRANSFORMER_EMBEDDING_DIM: usize = 512;
pub const TRANSFORMER_FF_DIM: usize = 2048;

/// Distilled predictor model size (1.8MB)
pub const PREDICTOR_MODEL_SIZE: usize = 1_800_000;  // 1.8MB

/// LSTM load predictor size (1.1MB)
pub const LSTM_PREDICTOR_SIZE: usize = 1_100_000;   // 1.1MB

/// Fixed-point arithmetic parameters (32.16 format)
pub const FIXED_POINT_INTEGER_BITS: usize = 16;
pub const FIXED_POINT_FRACTIONAL_BITS: usize = 16;
pub const FIXED_POINT_TOTAL_BITS: usize = 32;
pub const FIXED_POINT_SCALE: f64 = 65536.0; // 2^16

/// Differential privacy parameters (DP-SGD with σ=0.5)
pub const DP_SIGMA: f64 = 0.5;
pub const DP_CLIP_NORM: f64 = 1.0;
pub const DP_EPSILON_TARGET: f64 = 3.0;  // ε=3.0 privacy budget

// ============================================================================
// NETWORK PARAMETERS
// ============================================================================

/// Default P2P port
pub const DEFAULT_P2P_PORT: u16 = 4242;

/// Default RPC port
pub const DEFAULT_RPC_PORT: u16 = 8545;

/// Default metrics port
pub const DEFAULT_METRICS_PORT: u16 = 9090;

/// Gossip parameters
pub const GOSSIP_FANOUT: usize = 6;
pub const GOSSIP_TTL: u32 = 7;
pub const GOSSIP_INTERVAL_MS: u64 = 100; // 100ms

/// DHT parameters
pub const DHT_BUCKET_SIZE: usize = 20;
pub const DHT_REPLICATION_FACTOR: usize = 5;

/// Mixer cover traffic ratio (1-10x real traffic)
pub const COVER_TRAFFIC_RATIO_MIN: f64 = 1.0;
pub const COVER_TRAFFIC_RATIO_MAX: f64 = 10.0;

/// Mixer timing jitter parameters (μ=100ms, σ=200ms)
pub const MIXER_BASE_DELAY_MS: u64 = 100;
pub const MIXER_JITTER_MS: u64 = 200;

// ============================================================================
// ECONOMIC PARAMETERS (From whitepaper Section 8)
// ============================================================================

/// Total supply (10 billion NERV)
pub const TOTAL_SUPPLY: u64 = 10_000_000_000;

/// Emission schedule percentages
pub const EMISSION_YEAR_1_2_PERCENT: f64 = 0.38;  // 38%
pub const EMISSION_YEAR_3_5_PERCENT: f64 = 0.34;  // 34%
pub const EMISSION_YEAR_6_10_PERCENT: f64 = 0.28; // 28%
pub const TAIL_EMISSION_PERCENT: f64 = 0.005;     // 0.5% per year forever

/// Reward distribution (from useful-work economy)
pub const REWARD_GRADIENT_PERCENT: f64 = 0.60;    // 60% to gradient contributors
pub const REWARD_VALIDATION_PERCENT: f64 = 0.30;  // 30% to honest validators
pub const REWARD_PUBLIC_GOODS_PERCENT: f64 = 0.10; // 10% to retroactive grants

/// Genesis allocation (from whitepaper Section 8.2)
pub const GENESIS_USEFUL_WORK_PERCENT: f64 = 0.48;    // 48%
pub const GENESIS_CODE_CONTRIB_PERCENT: f64 = 0.25;   // 25%
pub const GENESIS_AUDIT_BOUNTY_PERCENT: f64 = 0.10;   // 10%
pub const GENESIS_RESEARCH_PERCENT: f64 = 0.04;       // 4%
pub const GENESIS_EARLY_DONOR_PERCENT: f64 = 0.05;    // 5%
pub const GENESIS_TREASURY_PERCENT: f64 = 0.03;       // 3%
pub const GENESIS_VISIONARY_PERCENT: f64 = 0.05;      // 5% (vesting 2 years)

// ============================================================================
// CONFIGURATION STRUCTURES - Runtime configuration
// ============================================================================

/// Main blockchain configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Node type (full, light, archive, relay)
    pub node_type: NodeType,
    
    /// TEE configuration
    pub tee_config: TeeConfig,
    
    /// Network configuration
    pub network_config: NetworkConfig,
    
    /// Storage configuration
    pub storage_config: StorageConfig,
    
    /// Consensus configuration
    pub consensus_config: ConsensusConfig,
    
    /// Sharding configuration
    pub sharding_config: ShardingConfig,
    
    /// Economic configuration
    pub economic_config: EconomicConfig,
    
    /// Logging level
    pub log_level: LogLevel,
    
    /// Path to model files
    pub model_path: PathBuf,
    
    /// Genesis block hash
    pub genesis_hash: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            node_type: NodeType::Full,
            tee_config: TeeConfig::default(),
            network_config: NetworkConfig::default(),
            storage_config: StorageConfig::default(),
            consensus_config: ConsensusConfig::default(),
            sharding_config: ShardingConfig::default(),
            economic_config: EconomicConfig::default(),
            log_level: LogLevel::Info,
            model_path: PathBuf::from("./models"),
            genesis_hash: "0x0000000000000000000000000000000000000000000000000000000000000000".to_string(),
        }
    }
}

/// Node type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    Full,      // Full validator node with TEE
    Light,     // Light client (headers only)
    Archive,   // Archive node (full history)
    Relay,     // Mixer relay node
}

/// TEE configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeeConfig {
    /// TEE type (SGX, SEV-SNP, ARM CCA)
    pub tee_type: TeeType,
    
    /// Enclave binary path
    pub enclave_path: PathBuf,
    
    /// Attestation service URL
    pub attestation_url: String,
    
    /// Enclave memory size (MB)
    pub enclave_memory_mb: usize,
}

impl Default for TeeConfig {
    fn default() -> Self {
        Self {
            tee_type: TeeType::Sgx,
            enclave_path: PathBuf::from("./enclave/target/release/enclave.signed.so"),
            attestation_url: "https://api.trustedservices.intel.com/sgx/dev".to_string(),
            enclave_memory_mb: 256,
        }
    }
}

/// TEE type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TeeType {
    Sgx,      // Intel SGX
    SevSnp,   // AMD SEV-SNP
    ArmCca,   // ARM CCA/TrustZone
    Apple,    // Apple Secure Enclave
    Nvidia,   // NVIDIA confidential computing
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// P2P listen address
    pub p2p_listen_addr: String,
    
    /// RPC listen address
    pub rpc_listen_addr: String,
    
    /// Metrics listen address
    pub metrics_listen_addr: String,
    
    /// Bootstrap nodes
    pub bootstrap_nodes: Vec<String>,
    
    /// Maximum peers
    pub max_peers: usize,
    
    /// Minimum peers
    pub min_peers: usize,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            p2p_listen_addr: format!("/ip4/0.0.0.0/tcp/{}", DEFAULT_P2P_PORT),
            rpc_listen_addr: format!("0.0.0.0:{}", DEFAULT_RPC_PORT),
            metrics_listen_addr: format!("0.0.0.0:{}", DEFAULT_METRICS_PORT),
            bootstrap_nodes: vec![],
            max_peers: 50,
            min_peers: 5,
        }
    }
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Database path
    pub db_path: PathBuf,
    
    /// Arweave configuration
    pub arweave_config: Option<ArweaveConfig>,
    
    /// IPFS configuration
    pub ipfs_config: Option<IpfsConfig>,
    
    /// Cache size (MB)
    pub cache_size_mb: usize,
    
    /// Prune old data (days)
    pub prune_after_days: u64,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("./data"),
            arweave_config: None,
            ipfs_config: None,
            cache_size_mb: 1024, // 1GB
            prune_after_days: 365 * 5, // 5 years
        }
    }
}

/// Arweave storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArweaveConfig {
    /// Arweave wallet file
    pub wallet_path: PathBuf,
    
    /// Arweave gateway URL
    pub gateway_url: String,
    
    /// Pinning cost (AR per MB)
    pub cost_per_mb: f64,
}

/// IPFS storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpfsConfig {
    /// IPFS API endpoint
    pub api_endpoint: String,
    
    /// Pinata API key (optional)
    pub pinata_api_key: Option<String>,
    
    /// Web3.storage token (optional)
    pub web3_storage_token: Option<String>,
}

/// Consensus configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Minimum stake required to validate
    pub min_stake: u64,
    
    /// Slashing percentage for misbehavior
    pub slashing_percent: f64,
    
    /// Unbonding period (blocks)
    pub unbonding_period: u64,
    
    /// Block time target (seconds)
    pub block_time_target: u64,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            min_stake: 1000, // 1000 NERV tokens
            slashing_percent: 0.05, // 5% slashing
            unbonding_period: 100_800, // ~7 days at 0.6s blocks
            block_time_target: 600, // 0.6 seconds
        }
    }
}

/// Sharding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingConfig {
    /// Initial shard count
    pub initial_shard_count: usize,
    
    /// Maximum shard size (accounts)
    pub max_shard_size: u64,
    
    /// Minimum shard size (accounts)
    pub min_shard_size: u64,
    
    /// Cross-shard delay (ms)
    pub cross_shard_delay_ms: u64,
}

impl Default for ShardingConfig {
    fn default() -> Self {
        Self {
            initial_shard_count: 16,
            max_shard_size: 100_000_000, // 100M accounts
            min_shard_size: 1_000_000,   // 1M accounts
            cross_shard_delay_ms: 180,   // 180ms median
        }
    }
}

/// Economic configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomicConfig {
    /// Inflation rate (annual)
    pub inflation_rate: f64,
    
    /// Transaction fee parameters
    pub fee_params: FeeParams,
    
    /// Reward distribution
    pub reward_distribution: RewardDistribution,
}

impl Default for EconomicConfig {
    fn default() -> Self {
        Self {
            inflation_rate: 0.38, // 38% first year
            fee_params: FeeParams::default(),
            reward_distribution: RewardDistribution::default(),
        }
    }
}

/// Transaction fee parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeParams {
    /// Base fee per transaction
    pub base_fee: u64,
    
    /// Fee per byte
    pub fee_per_byte: u64,
    
    /// Priority fee multiplier
    pub priority_fee_multiplier: f64,
}

impl Default for FeeParams {
    fn default() -> Self {
        Self {
            base_fee: 100, // 100 wei
            fee_per_byte: 1, // 1 wei per byte
            priority_fee_multiplier: 1.5,
        }
    }
}

/// Reward distribution parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardDistribution {
    /// Gradient contribution percentage
    pub gradient_percent: f64,
    
    /// Validation percentage
    pub validation_percent: f64,
    
    /// Public goods percentage
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

/// Logging level enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Calculate epoch number from block height
pub fn epoch_from_height(height: u64) -> u64 {
    const BLOCKS_PER_DAY: u64 = 24 * 60 * 60 / 600; // 0.6s blocks
    const BLOCKS_PER_EPOCH: u64 = EPOCH_DAYS * BLOCKS_PER_DAY;
    
    height / BLOCKS_PER_EPOCH
}

/// Calculate embedding update error bound for batch size
pub fn batch_error_bound(batch_size: usize) -> f64 {
    batch_size as f64 * HOMO_ERROR_BOUND
}

/// Calculate VDW generation time (150ms per batch)
pub fn vdw_generation_time_ms(batch_size: usize) -> u64 {
    // Base time plus linear scaling
    150 + (batch_size as u64 * 2)
}

/// Calculate mixer anonymity set size
pub fn mixer_anonymity_set(active_relays: usize, adversary_control: f64) -> f64 {
    let k = MIXER_HOPS as f64;
    let n = active_relays as f64;
    let f = adversary_control;
    
    // Formula from ProVerif analysis in Appendix D
    n.powf(k) * (1.0 - f).powf(k)
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_constants() {
        assert_eq!(EMBEDDING_SIZE, 512);
        assert_eq!(BATCH_SIZE, 256);
        assert_eq!(HOMO_ERROR_BOUND, 1e-9);
        assert_eq!(MIXER_HOPS, 5);
        assert_eq!(RS_TOTAL, 7); // k=5, m=2
    }
    
    #[test]
    fn test_epoch_calculation() {
        let epoch = epoch_from_height(0);
        assert_eq!(epoch, 0);
        
        // Should be deterministic
        let epoch2 = epoch_from_height(1_000_000);
        assert!(epoch2 > 0);
    }
    
    #[test]
    fn test_batch_error_bound() {
        let error = batch_error_bound(256);
        assert!((error - 2.56e-7).abs() < 1e-10);
    }
    
    #[test]
    fn test_mixer_anonymity() {
        let anonymity = mixer_anonymity_set(100, 0.33);
        assert!(anonymity > 100_000.0); // >100k as claimed in whitepaper
    }
    
    #[test]
    fn test_config_defaults() {
        let config = Config::default();
        assert!(matches!(config.node_type, NodeType::Full));
        assert_eq!(config.log_level, LogLevel::Info);
    }
}
