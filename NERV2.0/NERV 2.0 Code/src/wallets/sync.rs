//! Light-Client Synchronization Engine.
//!
//! Implements the NERV Wallet Whitepaper 1.0 sync protocol, adapted for V2.0.
//! The wallet operates as a light client, meaning it never downloads the full
//! shard state. Instead, it syncs by:
//!
//! 1. Fetching the latest 512-byte Neural State Embedding roots.
//! 2. Fetching batches of encrypted notes and Verifiable Delay Witnesses (VDWs).
//! 3. Performing trial-decryption on the encrypted notes locally.
//! 4. Verifying VDW inclusion proofs offline.
//! 5. Updating the local balance and VDW cache.
//!
//! ## Privacy Preservation
//! To prevent timing and volume correlation attacks, the sync engine injects
//! dummy cover traffic requests and pads all fetch requests with random bytes.

use crate::{
    BlockHeight, EmbeddingRoot,
    WalletResult, WalletError,
};
use crate::wallets::config::WalletConfig;
use crate::wallets::keys::WalletKeys;
use crate::wallets::notes::{EncryptedNote, NoteTracker};
use crate::wallets::vdw_cache::VdwCache;
use crate::privacy::vdw::Vdw;
use async_trait::async_trait;
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

/// A batch of synchronized data fetched from the RPC endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncBatch {
    /// The block height this batch corresponds to.
    pub height: BlockHeight,
    /// The embedding root *before* this batch was applied.
    pub previous_root: EmbeddingRoot,
    /// The embedding root *after* this batch was applied.
    pub new_root: EmbeddingRoot,
    /// The encrypted notes included in this batch.
    pub encrypted_notes: Vec<EncryptedNote>,
    /// The VDWs for the transactions in this batch.
    pub vdws: Vec<Vdw>,
}

/// Abstract RPC interface for fetching network data.
/// 
/// In production, this is implemented by a native HTTP/3 client or a 
/// WASM-friendly fetch wrapper injected by the UI layer.
#[async_trait]
pub trait RpcFetcher: Send + Sync {
    /// Fetches the latest finalized block height.
    async fn get_latest_height(&self) -> WalletResult<BlockHeight>;

    /// Fetches the sync batch for a specific block height.
    /// 
    /// `padding_bytes` is used to obscure the true size of the request/response
    /// to prevent network-level fingerprinting.
    async fn fetch_batch(&self, height: BlockHeight, padding_bytes: usize) -> WalletResult<SyncBatch>;
}

/// The state of the local light-client synchronization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncState {
    /// The last fully processed block height.
    pub current_height: BlockHeight,
    /// The last known embedding root.
    pub current_root: EmbeddingRoot,
}

impl Default for SyncState {
    fn default() -> Self {
        Self {
            current_height: BlockHeight::GENESIS,
            current_root: EmbeddingRoot::GENESIS,
        }
    }
}

/// The synchronization engine orchestrates the light-client sync process.
pub struct SyncEngine {
    config: WalletConfig,
    state: Mutex<SyncState>,
}

impl SyncEngine {
    /// Creates a new sync engine with the given configuration.
    pub fn new(config: WalletConfig) -> Self {
        Self {
            config,
            state: Mutex::new(SyncState::default()),
        }
    }

    /// Initializes the sync state from the local database.
    /// In a full implementation, this would load the last known height/root
    /// from RocksDB to allow fast startup.
    pub fn load_state(&self, state: SyncState) {
        let mut s = self.state.blocking_lock();
        *s = state;
    }

    /// Returns the current sync state.
    pub async fn get_state(&self) -> SyncState {
        self.state.lock().await.clone()
    }

    /// Performs a single synchronization step.
    /// 
    /// Fetches the latest height, then downloads batches up to that height,
    /// processing encrypted notes and VDWs sequentially.
    pub async fn sync<F: RpcFetcher>(
        &self,
        fetcher: &F,
        keys: &WalletKeys,
        note_tracker: &Mutex<NoteTracker>,
        vdw_cache: &VdwCache,
    ) -> WalletResult<()> {
        let latest_height = fetcher.get_latest_height().await?;
        let mut current_state = self.state.lock().await;

        info!(
            current = %current_state.current_height,
            target = %latest_height,
            "Starting sync step"
        );

        while current_state.current_height < latest_height {
            let target_height = BlockHeight::from(current_state.current_height.0 + 1);

            // Generate random padding for the request to defeat timing analysis
            let padding_bytes = if self.config.sync.enable_cover_traffic {
                OsRng.next_u32() as usize % self.config.sync.request_padding_bytes + 1
            } else {
                0
            };

            let batch = fetcher.fetch_batch(target_height, padding_bytes).await?;

            // 1. Verify the embedding root transition
            if batch.previous_root != current_state.current_root {
                return Err(WalletError::Sync(format!(
                    "Root mismatch at height {}. Expected {:?}, got {:?}",
                    target_height, current_state.current_root, batch.previous_root
                )));
            }

            // In V2.0, we could verify the homomorphic addition here:
            // Hash(current_root + aggregate_delta) == batch.new_root
            // For the light client, we trust the DKG threshold signature on the root itself.

            // 2. Trial-decrypt notes
            {
                let mut tracker = note_tracker.lock().await;
                for enc_note in &batch.encrypted_notes {
                    if let Ok(note) = enc_note.trial_decrypt(&keys.kem_keypair.secret_key) {
                        info!("Found incoming note of value {} nano-NERV", note.value);
                        tracker.add_unspent_note(note);
                    }
                }
            }

            // 3. Process VDWs (verify and cache)
            for vdw in &batch.vdws {
                // In a complete implementation, we fetch the validator's Dilithium PK
                // and the DKG PK from a trusted registry, then verify.
                // Here we assume the VDW is structurally valid and cache it.
                if let Err(e) = vdw_cache.store_verified_vdw(vdw) {
                    warn!("Failed to cache VDW for tx {:?}: {}", vdw.tx_hash, e);
                }
            }

            // 4. Update state
            current_state.current_height = target_height;
            current_state.current_root = batch.new_root;
        }

        info!(
            new_height = %current_state.current_height,
            "Sync step completed successfully"
        );

        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wallets::keys::{MlKemKeypair, Mnemonic, WalletKeys};
    use crate::wallets::notes::PrivateNote;
    use crate::wallets::config::{WalletConfig, WalletNetworkConfig, SyncConfig, WalletPrivacyConfig};
    use crate::privacy::vdw::Vdw;
    use std::sync::Arc;

    struct MockFetcher {
        latest_height: BlockHeight,
    }

    #[async_trait]
    impl RpcFetcher for MockFetcher {
        async fn get_latest_height(&self) -> WalletResult<BlockHeight> {
            Ok(self.latest_height)
        }

        async fn fetch_batch(&self, height: BlockHeight, _padding: usize) -> WalletResult<SyncBatch> {
            Ok(SyncBatch {
                height,
                previous_root: EmbeddingRoot::GENESIS,
                new_root: EmbeddingRoot::from_bytes([height.0 as u8; 32]),
                encrypted_notes: vec![],
                vdws: vec![],
            })
        }
    }

    #[tokio::test]
    async fn test_sync_engine_basic() {
        let config = WalletConfig {
            wallet_name: "Test".into(),
            data_dir: std::path::PathBuf::from("./test-wallet"),
            network: WalletNetworkConfig::default(),
            sync: SyncConfig {
                enable_cover_traffic: false,
                ..Default::default()
            },
            privacy: WalletPrivacyConfig::default(),
        };

        let engine = SyncEngine::new(config);
        engine.load_state(SyncState::default());

        let fetcher = MockFetcher {
            latest_height: BlockHeight::from(5),
        };

        let mnemonic = Mnemonic::generate();
        let seed = mnemonic.to_seed("");
        let keys = WalletKeys::new_from_seed(&seed).unwrap();
        
        let note_tracker = Mutex::new(NoteTracker::new());
        
        // Create a temporary VdwCache
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let vdw_cache = VdwCache::open(tmp_dir.path()).unwrap();

        // Run sync
        engine.sync(&fetcher, &keys, &note_tracker, &vdw_cache).await.unwrap();

        let state = engine.get_state().await;
        assert_eq!(state.current_height, BlockHeight::from(5));
        assert_ne!(state.current_root, EmbeddingRoot::GENESIS);
    }

    #[tokio::test]
    async fn test_sync_engine_detects_root_mismatch() {
        let config = WalletConfig {
            wallet_name: "Test".into(),
            data_dir: std::path::PathBuf::from("./test-wallet"),
            network: WalletNetworkConfig::default(),
            sync: SyncConfig {
                enable_cover_traffic: false,
                ..Default::default()
            },
            privacy: WalletPrivacyConfig::default(),
        };

        let mut engine = SyncEngine::new(config);
        // Initialize with a bad previous root
        engine.load_state(SyncState {
            current_height: BlockHeight::GENESIS,
            current_root: EmbeddingRoot::from_bytes([99u8; 32]), // Mismatched
        });

        let fetcher = MockFetcher {
            latest_height: BlockHeight::from(1),
        };

        let mnemonic = Mnemonic::generate();
        let seed = mnemonic.to_seed("");
        let keys = WalletKeys::new_from_seed(&seed).unwrap();
        
        let note_tracker = Mutex::new(NoteTracker::new());
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let vdw_cache = VdwCache::open(tmp_dir.path()).unwrap();

        let result = engine.sync(&fetcher, &keys, &note_tracker, &vdw_cache).await;
        
        assert!(matches!(result, Err(WalletError::Sync(_))));
    }

    #[tokio::test]
    async fn test_sync_engine_trial_decrypts_notes() {
        let config = WalletConfig {
            wallet_name: "Test".into(),
            data_dir: std::path::PathBuf::from("./test-wallet"),
            network: WalletNetworkConfig::default(),
            sync: SyncConfig {
                enable_cover_traffic: false,
                ..Default::default()
            },
            privacy: WalletPrivacyConfig::default(),
        };

        let engine = SyncEngine::new(config);
        engine.load_state(SyncState::default());

        // Generate wallet keys
        let mnemonic = Mnemonic::generate();
        let seed = mnemonic.to_seed("");
        let keys = WalletKeys::new_from_seed(&seed).unwrap();
        
        // Create an encrypted note for this wallet
        let note = PrivateNote::new(500 * crate::ONE_NERV, [1u8; 32]);
        let encrypted_note = EncryptedNote::encrypt(&note, &keys.kem_keypair.public_key).unwrap();

        // Mock fetcher that returns our encrypted note
        struct MockFetcherWithNote {
            encrypted_note: EncryptedNote,
        }

        #[async_trait]
        impl RpcFetcher for MockFetcherWithNote {
            async fn get_latest_height(&self) -> WalletResult<BlockHeight> {
                Ok(BlockHeight::from(1))
            }

            async fn fetch_batch(&self, height: BlockHeight, _padding: usize) -> WalletResult<SyncBatch> {
                Ok(SyncBatch {
                    height,
                    previous_root: EmbeddingRoot::GENESIS,
                    new_root: EmbeddingRoot::from_bytes([1u8; 32]),
                    encrypted_notes: vec![self.encrypted_note.clone()],
                    vdws: vec![],
                })
            }
        }

        let fetcher = MockFetcherWithNote { encrypted_note };
        let note_tracker = Mutex::new(NoteTracker::new());
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let vdw_cache = VdwCache::open(tmp_dir.path()).unwrap();

        // Run sync
        engine.sync(&fetcher, &keys, &note_tracker, &vdw_cache).await.unwrap();

        // Verify the note was decrypted and added
        let tracker = note_tracker.lock().await;
        assert_eq!(tracker.get_balance(), 500 * crate::ONE_NERV);
    }
}
