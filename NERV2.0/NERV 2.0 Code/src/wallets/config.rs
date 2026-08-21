//! Wallet Configuration System.
//!
//! Defines the configuration parameters for the NERV light wallet, including
//! data directories, trusted RPC endpoints, and synchronization settings.
//! The wallet relies on HTTP/3 for privacy-preserving range requests with
//! random padding to prevent timing correlation attacks.

use crate::NervResult;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level wallet configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WalletConfig {
    /// Human-readable wallet name (e.g., "Main Wallet", "Validator Funds").
    pub wallet_name: String,

    /// Data directory for wallet state (keys, cached notes, VDWs).
    pub data_dir: PathBuf,

    /// Network configuration for the wallet.
    pub network: WalletNetworkConfig,

    /// Synchronization configuration.
    pub sync: SyncConfig,

    /// Privacy configuration (Sphinx routing, cover traffic).
    pub privacy: WalletPrivacyConfig,
}

impl Default for WalletConfig {
    fn default() -> Self {
        Self {
            wallet_name: "NERV Main Wallet".into(),
            data_dir: PathBuf::from("./nerv-wallet"),
            network: WalletNetworkConfig::default(),
            sync: SyncConfig::default(),
            privacy: WalletPrivacyConfig::default(),
        }
    }
}

impl WalletConfig {
    /// Load configuration from a TOML file.
    pub fn load_from_file(path: &std::path::Path) -> NervResult<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::NervError::Config(format!("failed to read wallet config: {}", e)))?;
        let config: Self = toml::from_str(&content)
            .map_err(|e| crate::NervError::Config(format!("failed to parse wallet config TOML: {}", e)))?;
        Ok(config)
    }

    /// Ensure the wallet data directory exists.
    pub fn ensure_data_dir(&self) -> NervResult<()> {
        if !self.data_dir.exists() {
            std::fs::create_dir_all(&self.data_dir).map_err(|e| {
                crate::NervError::Config(format!("failed to create wallet data_dir: {}", e))
            })?;
        }
        Ok(())
    }
}

/// Network configuration for the wallet's light-client RPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WalletNetworkConfig {
    /// The trusted RPC endpoint for fetching embedding roots and VDWs.
    /// In production, this should be a local node or a trusted gateway.
    pub rpc_endpoint: String,

    /// Enable HTTP/3 (QUIC) for RPC requests.
    pub enable_http3: bool,

    /// Request timeout in seconds.
    pub request_timeout_secs: u64,
}

impl Default for WalletNetworkConfig {
    fn default() -> Self {
        Self {
            rpc_endpoint: "https://rpc.nerv.network".into(),
            enable_http3: true,
            request_timeout_secs: 15,
        }
    }
}

/// Light-client synchronization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SyncConfig {
    /// The maximum number of blocks to fetch per sync request.
    pub batch_size: u64,

    /// Interval in seconds for polling new embedding roots.
    pub poll_interval_secs: u64,

    /// Enable dummy cover traffic requests during sync to prevent timing analysis.
    /// (Crucial for NERV V2.0 pure-crypto privacy model).
    pub enable_cover_traffic: bool,

    /// The amount of random padding (in bytes) added to sync requests.
    pub request_padding_bytes: usize,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            poll_interval_secs: 10,
            enable_cover_traffic: true,
            request_padding_bytes: 1024, // 1KB padding
        }
    }
}

/// Wallet-level privacy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WalletPrivacyConfig {
    /// Number of Sphinx onion-routing hops for outgoing transactions.
    /// NERV V2.0 mandates exactly 5 hops using ML-KEM-768.
    pub sphinx_hops: usize,

    /// Minimum relay stake required for the wallet to consider a relay trustworthy.
    /// (Fetched from the network and cached locally).
    pub min_relay_stake_nano: u64,

    /// Whether to attach dummy cover transactions during periods of inactivity.
    pub generate_cover_txs: bool,
}

impl Default for WalletPrivacyConfig {
    fn default() -> Self {
        Self {
            sphinx_hops: crate::SPHINX_HOPS, // 5
            min_relay_stake_nano: 100 * crate::ONE_NERV,
            generate_cover_txs: true,
        }
    }
}
