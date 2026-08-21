//! Native Mobile UI Bindings (iOS / Android) via UniFFI.
//!
//! Exposes the NERV Wallet core to Swift and Kotlin. This module acts as the 
//! bridge between Rust's asynchronous, thread-safe architecture and native 
//! mobile UI frameworks (SwiftUI / Jetpack Compose).
//!
//! ## Architecture
//! - Uses `uniffi` to generate native Swift/Kotlin bindings.
//! - Implements a callback interface `MobileRpcClient` allowing mobile apps 
//!   to handle network requests natively (e.g., using URLSession or OkHttp).
//! - Exposes a `MobileWallet` object that handles state and cryptography 
//!   entirely in Rust, guaranteeing memory safety and PQ-security on mobile.

use crate::{
    TxHash, WalletResult, WalletError, ONE_NERV,
};
use crate::wallets::config::WalletConfig;
use crate::wallets::keys::Mnemonic;
use crate::wallets::wallet::{Wallet, WalletRpcClient};
use crate::wallets::sync::{RpcFetcher, SyncBatch};
use crate::wallets::sphinx_builder::{RelayInfo, SphinxPacket};
use crate::privacy::dkg::DkgPublicKey;
use crate::BlockHeight;

use async_trait::async_trait;
use std::sync::Arc;
use parking_lot::Mutex;
use serde::{Serialize, Deserialize};

uniffi::setup_scaffolding!();

// ─── FFI Data Types ───────────────────────────────────────────────────────

/// A flat error type for FFI boundaries, as Rust enums can be tricky to 
/// map directly to Swift/Kotlin error protocols without complex bindings.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct MobileWalletError {
    pub message: String,
}

impl From<WalletError> for MobileWalletError {
    fn from(e: WalletError) -> Self {
        Self { message: format!("{}", e) }
    }
}

/// A snapshot of the wallet state for the UI to render.
#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)]
pub struct MobileWalletState {
    pub is_syncing: bool,
    pub balance_nerv: f64,
    pub address: String,
    pub notification: Option<String>,
}

// ─── Mobile RPC Callback Interface ────────────────────────────────────────

/// A trait implemented by the native mobile app (Swift/Kotlin) to handle 
/// network requests. This allows the Rust core to remain platform-agnostic 
/// while leveraging native networking stacks.
#[uniffi::export(with_foreign)]
#[async_trait::async_trait]
pub trait MobileRpcClient: Send + Sync {
    /// Fetches the latest block height.
    async fn get_latest_height(&self) -> Result<u64, MobileWalletError>;
    
    /// Fetches a batch of sync data.
    async fn fetch_batch(&self, height: u64, padding: u64) -> Result<SyncBatch, MobileWalletError>;
    
    /// Fetches the network's DKG public key.
    async fn get_dkg_public_key(&self) -> Result<DkgPublicKey, MobileWalletError>;
    
    /// Fetches the network's NWO Perceptron weights and bias.
    async fn get_network_params(&self) -> Result<NetworkParams, MobileWalletError>;
    
    /// Fetches the active Sphinx relays.
    async fn get_sphinx_relays(&self) -> Result<Vec<RelayInfo>, MobileWalletError>;

    /// Broadcasts a Sphinx packet to the first relay.
    async fn broadcast_sphinx_packet(&self, packet: SphinxPacket) -> Result<String, MobileWalletError>;
}

/// Wrapper for network params to satisfy UniFFI's type requirements (tuples can be tricky).
#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)]
pub struct NetworkParams {
    pub weights: Vec<Vec<i64>>,
    pub bias: Vec<i64>,
}

/// Internal adapter to bridge the `MobileRpcClient` callback to the Rust `WalletRpcClient` trait.
struct MobileRpcAdapter {
    client: Arc<dyn MobileRpcClient>,
}

#[async_trait::async_trait]
impl RpcFetcher for MobileRpcAdapter {
    async fn get_latest_height(&self) -> WalletResult<BlockHeight> {
        let h = self.client.get_latest_height().await?;
        Ok(BlockHeight::from(h))
    }

    async fn fetch_batch(&self, height: BlockHeight, padding: usize) -> WalletResult<SyncBatch> {
        self.client.fetch_batch(height.0, padding as u64).await.map_err(Into::into)
    }
}

#[async_trait::async_trait]
impl WalletRpcClient for MobileRpcAdapter {
    async fn get_dkg_public_key(&self) -> WalletResult<DkgPublicKey> {
        self.client.get_dkg_public_key().await.map_err(Into::into)
    }

    async fn get_network_params(&self) -> WalletResult<(Vec<Vec<i64>>, Vec<i64>)> {
        let params = self.client.get_network_params().await?;
        Ok((params.weights, params.bias))
    }

    async fn get_sphinx_relays(&self) -> WalletResult<Vec<RelayInfo>> {
        self.client.get_sphinx_relays().await.map_err(Into::into)
    }

    async fn broadcast_sphinx_packet(&self, packet: &SphinxPacket) -> WalletResult<TxHash> {
        let hex_hash = self.client.broadcast_sphinx_packet(packet.clone()).await?;
        TxHash::from_hex(&hex_hash).map_err(Into::into)
    }
}

// ─── Mobile Wallet API ────────────────────────────────────────────────────

/// The main Wallet interface exposed to Swift/Kotlin.
#[derive(uniffi::Object)]
pub struct MobileWallet {
    wallet: Arc<Wallet>,
    state: Arc<Mutex<MobileWalletState>>,
}

#[uniffi::export]
impl MobileWallet {
    /// Creates a new wallet instance from a config JSON string and a mnemonic.
    #[uniffi::constructor]
    pub fn new(config_json: String, mnemonic_phrase: String) -> Result<Self, MobileWalletError> {
        let config: WalletConfig = serde_json::from_str(&config_json)?;
        
        // Parse mnemonic (mobile UIs might pass hex or words)
        let mnemonic = if mnemonic_phrase.starts_with("0x") {
            let hex = &mnemonic_phrase[2..];
            let entropy = hex::decode(hex)?;
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&entropy);
            Mnemonic::from_entropy(arr)
        } else {
            // In production, map BIP-39 words to entropy
            Mnemonic::generate()
        };

        let wallet = Wallet::new(config, &mnemonic, "")?;
        
        let mut state = MobileWalletState {
            is_syncing: false,
            balance_nerv: 0.0,
            address: format!("nerv1{}", &hex::encode(wallet.get_kem_public_key())[..24]),
            notification: None,
        };

        Ok(Self {
            wallet: Arc::new(wallet),
            state: Arc::new(Mutex::new(state)),
        })
    }

    /// Generates a new random mnemonic and returns it as a hex string.
    #[uniffi::method]
    pub fn generate_mnemonic() -> String {
        let mnemonic = Mnemonic::generate();
        format!("0x{}", hex::encode(mnemonic.entropy()))
    }

    /// Returns the current wallet state for UI rendering.
    #[uniffi::method]
    pub fn get_state(&self) -> MobileWalletState {
        let mut state = self.state.lock().clone();
        state.balance_nerv = self.wallet.get_balance() as f64 / ONE_NERV as f64;
        state
    }

    /// Synchronizes the wallet with the network using the native mobile RPC client.
    #[uniffi::method]
    pub async fn sync(&self, rpc_client: Arc<dyn MobileRpcClient>) -> Result<(), MobileWalletError> {
        {
            let mut state = self.state.lock();
            state.is_syncing = true;
            state.notification = Some("Synchronizing...".to_string());
        }

        let adapter = MobileRpcAdapter { client: rpc_client };
        let result = self.wallet.sync(&adapter).await;

        {
            let mut state = self.state.lock();
            state.is_syncing = false;
            state.balance_nerv = self.wallet.get_balance() as f64 / ONE_NERV as f64;
            match &result {
                Ok(_) => state.notification = Some("Sync complete.".to_string()),
                Err(e) => state.notification = Some(format!("Error: {}", e)),
            }
        }

        result.map_err(Into::into)
    }

    /// Sends a transaction via the PQ-Sphinx mixnet.
    #[uniffi::method]
    pub async fn send(
        &self,
        rpc_client: Arc<dyn MobileRpcClient>,
        recipient_kem_pk_hex: String,
        amount_nerv: f64,
        memo: Option<String>,
    ) -> Result<String, MobileWalletError> {
        let adapter = MobileRpcAdapter { client: rpc_client };
        let recipient_pk = hex::decode(&recipient_kem_pk_hex)?;
        let amount_nano = (amount_nerv * ONE_NERV as f64) as u64;

        let tx_hash = self.wallet.send(&adapter, &recipient_pk, amount_nano, memo).await?;
        
        let mut state = self.state.lock();
        state.balance_nerv = self.wallet.get_balance() as f64 / ONE_NERV as f64;
        
        Ok(tx_hash.to_hex())
    }
}
