//! Main Wallet Orchestrator.
//!
//! Ties together all wallet subsystems: key management, note tracking,
//! VDW caching, light-client synchronization, and transaction construction.
//! The UI layer interacts exclusively with this `Wallet` struct.

use crate::{
    TxHash, WalletResult, WalletError,
    EMBEDDING_DIM,
};
use crate::wallets::config::WalletConfig;
use crate::wallets::keys::{Mnemonic, WalletKeys};
use crate::wallets::memo::{Memo, EncryptedMemo, DisclosureProof};
use crate::wallets::notes::NoteTracker;
use crate::wallets::sync::{SyncEngine, SyncState, RpcFetcher};
use crate::wallets::transaction::{TransactionBuilder, ThresholdEncryptedTx};
use crate::wallets::vdw_cache::VdwCache;
use crate::wallets::sphinx_builder::{PqSphinxBuilder, RelayInfo, SphinxPacket};
use crate::privacy::dkg::DkgPublicKey;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use tracing::{info, warn};

/// Combined RPC trait required by the Wallet orchestrator.
#[async_trait]
pub trait WalletRpcClient: RpcFetcher {
    /// Fetches the network's current DKG public key.
    async fn get_dkg_public_key(&self) -> WalletResult<DkgPublicKey>;
    
    /// Fetches the network's current NWO Perceptron weights and bias.
    async fn get_network_params(&self) -> WalletResult<(Vec<Vec<i64>>, Vec<i64>)>;
    
    /// Fetches the active Sphinx relays for packet routing.
    async fn get_sphinx_relays(&self) -> WalletResult<Vec<RelayInfo>>;

    /// Broadcasts a constructed Sphinx packet to the first relay.
    async fn broadcast_sphinx_packet(&self, packet: &SphinxPacket) -> WalletResult<TxHash>;
}

/// The main wallet interface.
pub struct Wallet {
    config: WalletConfig,
    keys: WalletKeys,
    notes: Arc<Mutex<NoteTracker>>,
    vdw_cache: Arc<VdwCache>,
    sync_engine: Arc<SyncEngine>,
    // Transaction builder is wrapped in a Mutex because it depends on 
    // network parameters (weights/bias) that can update during sync.
    tx_builder: Arc<Mutex<Option<TransactionBuilder>>>,
}

impl Wallet {
    /// Initializes a new wallet instance from a mnemonic seed.
    /// 
    /// This creates the data directory, initializes the VDW cache,
    /// and prepares the sync engine.
    pub fn new(config: WalletConfig, mnemonic: &Mnemonic, passphrase: &str) -> WalletResult<Self> {
        config.ensure_data_dir()?;
        
        let seed = mnemonic.to_seed(passphrase);
        let keys = WalletKeys::new_from_seed(&seed)?;
        let notes = Arc::new(Mutex::new(NoteTracker::new()));
        
        let vdw_path = config.data_dir.join("vdw_cache");
        let vdw_cache = Arc::new(VdwCache::open(&vdw_path)?);
        
        let sync_engine = Arc::new(SyncEngine::new(config.clone()));
        
        Ok(Self {
            config,
            keys,
            notes,
            vdw_cache,
            sync_engine,
            tx_builder: Arc::new(Mutex::new(None)),
        })
    }

    /// Returns a reference to the wallet's public KEM key.
    /// Used for receiving funds.
    pub fn get_kem_public_key(&self) -> &[u8] {
        &self.keys.kem_keypair.public_key
    }

    /// Returns the current available balance in nano-NERV.
    pub fn get_balance(&self) -> u64 {
        self.notes.lock().get_balance()
    }

    /// Returns the current available balance as a formatted string.
    pub fn get_formatted_balance(&self) -> String {
        self.notes.lock().get_formatted_balance()
    }

    /// Runs a single synchronization step.
    /// 
    /// Fetches the latest blocks, trial-decrypts notes, and updates the
    /// local balance. Also updates the network parameters (weights/DKG PK).
    pub async fn sync<R: WalletRpcClient>(&self, rpc: &R) -> WalletResult<()> {
        // 1. Update network parameters if necessary
        if self.tx_builder.lock().is_none() {
            let dkg_pk = rpc.get_dkg_public_key().await?;
            let (weights, bias) = rpc.get_network_params().await?;
            
            let builder = TransactionBuilder::new(
                dkg_pk,
                Arc::new(weights),
                Arc::new(bias),
            );
            *self.tx_builder.lock() = Some(builder);
            info!("Network parameters loaded and transaction builder initialized.");
        }

        // 2. Run the sync engine
        self.sync_engine.sync(rpc, &self.keys, &self.notes, &self.vdw_cache).await?;
        Ok(())
    }

    /// Constructs, encrypts, and routes a private transaction.
    /// 
    /// # Arguments
    /// * `rpc` - The RPC client to fetch relays and broadcast.
    /// * `recipient_kem_pk` - The recipient's ML-KEM-768 public key.
    /// * `amount` - The amount to send in nano-NERV.
    /// * `memo` - Optional plaintext memo to encrypt and attach.
    pub async fn send<R: WalletRpcClient>(
        &self,
        rpc: &R,
        recipient_kem_pk: &[u8],
        amount: u64,
        memo: Option<String>,
    ) -> WalletResult<TxHash> {
        // 1. Build the threshold-encrypted transaction
        let encrypted_tx = {
            let builder_lock = self.tx_builder.lock();
            let builder = builder_lock.as_ref()
                .ok_or_else(|| WalletError::Sync("Transaction builder not initialized. Run sync first.".into()))?;
            
            builder.build(&self.keys, &self.notes, recipient_kem_pk, amount)?
        };

        // 2. Handle Memo (if provided)
        // In a complete implementation, the memo ciphertext would be appended to the 
        // transaction payload before DKG encryption. Here we just generate it.
        if let Some(memo_text) = memo {
            let _memo = Memo::new(memo_text)?;
            // Encrypt memo using the shared secret derived from the recipient's KEM PK
            // (Simplified: actual implementation attaches to PrivateTransaction)
            warn!("Memo attachment is structurally supported but not fully wired to DKG payload in this iteration.");
        }

        // 3. Fetch Sphinx relays and construct the onion packet
        let relays = rpc.get_sphinx_relays().await?;
        if relays.len() != crate::SPHINX_HOPS {
            return Err(WalletError::Sphinx(format!(
                "RPC returned {} relays, expected {}",
                relays.len(),
                crate::SPHINX_HOPS
            )));
        }

        let sphinx_builder = PqSphinxBuilder::new(relays)?;
        let payload_bytes = bincode::serialize(&encrypted_tx)?;
        let sphinx_packet = sphinx_builder.build(&payload_bytes)?;

        // 4. Broadcast the Sphinx packet
        let tx_hash = rpc.broadcast_sphinx_packet(&sphinx_packet).await?;
        
        info!(tx_hash = %tx_hash, "Transaction successfully broadcast to mixnet");
        Ok(tx_hash)
    }

    /// Retrieves a cached VDW for a specific transaction.
    pub fn get_vdw(&self, tx_hash: &TxHash) -> WalletResult<Option<crate::privacy::vdw::Vdw>> {
        self.vdw_cache.get_vdw(tx_hash)
    }

    /// Exports a memo disclosure proof for a third party.
    /// 
    /// Note: In a complete implementation, the wallet would need to retrieve
    /// the specific shared secret used for that transaction. This requires
    /// storing ephemeral KEM secrets locally until the transaction is confirmed.
    pub fn export_memo_disclosure(
        &self,
        _tx_hash: &TxHash,
        _shared_secret: &[u8],
        _encrypted_memo: &EncryptedMemo,
    ) -> WalletResult<DisclosureProof> {
        Err(WalletError::Ui("Memo disclosure export not fully implemented in this iteration.".into()))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wallets::config::{WalletConfig, WalletNetworkConfig, SyncConfig, WalletPrivacyConfig};
    use crate::wallets::sync::SyncBatch;
    use crate::BlockHeight;
    use std::sync::Arc;

    struct MockWalletRpc {
        latest_height: BlockHeight,
    }

    #[async_trait]
    impl RpcFetcher for MockWalletRpc {
        async fn get_latest_height(&self) -> WalletResult<BlockHeight> {
            Ok(self.latest_height)
        }

        async fn fetch_batch(&self, height: BlockHeight, _padding: usize) -> WalletResult<SyncBatch> {
            Ok(SyncBatch {
                height,
                previous_root: crate::EmbeddingRoot::GENESIS,
                new_root: crate::EmbeddingRoot::from_bytes([height.0 as u8; 32]),
                encrypted_notes: vec![],
                vdws: vec![],
            })
        }
    }

    #[async_trait]
    impl WalletRpcClient for MockWalletRpc {
        async fn get_dkg_public_key(&self) -> WalletResult<DkgPublicKey> {
            // Return a dummy DKG PK for testing
            Ok(DkgPublicKey {
                point: vec![0u8; 48],
                hash: [0u8; 32],
                session_id: [0u8; 32],
                threshold: 3,
                num_participants: 5,
            })
        }

        async fn get_network_params(&self) -> WalletResult<(Vec<Vec<i64>>, Vec<i64>)> {
            Ok((vec![vec![1i64; 1]; EMBEDDING_DIM], vec![0i64; EMBEDDING_DIM]))
        }

        async fn get_sphinx_relays(&self) -> WalletResult<Vec<RelayInfo>> {
            // Generate 5 dummy relays
            Ok((0..crate::SPHINX_HOPS).map(|_| RelayInfo {
                kem_pk: vec![0u8; 1184], // ML-KEM-768 PK size
                next_addr: "/ip4/127.0.0.1/tcp/4000".into(),
            }).collect())
        }

        async fn broadcast_sphinx_packet(&self, _packet: &SphinxPacket) -> WalletResult<TxHash> {
            Ok(TxHash::from_bytes([1u8; 32]))
        }
    }

    #[tokio::test]
    async fn test_wallet_initialization_and_sync() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let config = WalletConfig {
            wallet_name: "Test Wallet".into(),
            data_dir: tmp_dir.path().to_path_buf(),
            network: WalletNetworkConfig::default(),
            sync: SyncConfig::default(),
            privacy: WalletPrivacyConfig::default(),
        };

        let mnemonic = Mnemonic::generate();
        let wallet = Wallet::new(config, &mnemonic, "").unwrap();

        let rpc = MockWalletRpc {
            latest_height: BlockHeight::from(1),
        };

        // Run sync - should initialize tx_builder
        let result = wallet.sync(&rpc).await;
        assert!(result.is_ok(), "Sync failed: {:?}", result.err());
        
        // Verify tx_builder was initialized
        assert!(wallet.tx_builder.lock().is_some());
    }
}

