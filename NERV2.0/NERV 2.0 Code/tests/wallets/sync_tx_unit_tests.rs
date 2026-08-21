//! Unit tests for Sync Engine and Transaction Builder.
//!
//! Validates the NERV V2.0 light-client synchronization state machine and
//! the transaction construction pipeline. Ensures that root transitions
//! are verified strictly, notes are decrypted correctly during sync, and
//! transactions accurately calculate fees, select notes, compute deltas,
//! and encrypt the payload for the DKG mempool.

use nerv::wallets::config::{WalletConfig, WalletNetworkConfig, SyncConfig, WalletPrivacyConfig};
use nerv::wallets::keys::{Mnemonic, MlKemKeypair, WalletKeys};
use nerv::wallets::notes::{EncryptedNote, NoteTracker, PrivateNote};
use nerv::wallets::sync::{RpcFetcher, SyncBatch, SyncEngine, SyncState};
use nerv::wallets::transaction::TransactionBuilder;
use nerv::wallets::vdw_cache::VdwCache;
use nerv::wallets::error::WalletError;
use nerv::privacy::dkg::{DkgPublicKey, DkgScalar, FeldmanCommitment};
use nerv::{BlockHeight, EmbeddingRoot, ONE_NERV, EMBEDDING_DIM};
use bls12_381::Scalar as BlsScalar;
use parking_lot::Mutex;
use std::sync::Arc;
use async_trait::async_trait;

// ─── Mock RPC Fetcher for Sync Tests ─────────────────────────────────────

struct MockFetcher {
    latest_height: BlockHeight,
    batch_to_return: SyncBatch,
}

#[async_trait]
impl RpcFetcher for MockFetcher {
    async fn get_latest_height(&self) -> Result<BlockHeight, WalletError> {
        Ok(self.latest_height)
    }

    async fn fetch_batch(&self, height: BlockHeight, _padding: usize) -> Result<SyncBatch, WalletError> {
        // Return a cloned batch for the requested height
        let mut batch = self.batch_to_return.clone();
        batch.height = height;
        Ok(batch)
    }
}

// ─── Sync Engine Tests ───────────────────────────────────────────────────

fn get_test_config() -> WalletConfig {
    WalletConfig {
        wallet_name: "Test Wallet".into(),
        data_dir: std::path::PathBuf::from("./test-wallet-sync"),
        network: WalletNetworkConfig::default(),
        sync: SyncConfig {
            enable_cover_traffic: false,
            ..Default::default()
        },
        privacy: WalletPrivacyConfig::default(),
    }
}

#[tokio::test]
async fn test_sync_engine_advances_state() {
    let config = get_test_config();
    let engine = SyncEngine::new(config);
    engine.load_state(SyncState::default());

    let fetcher = MockFetcher {
        latest_height: BlockHeight::from(3),
        batch_to_return: SyncBatch {
            height: BlockHeight::from(1),
            previous_root: EmbeddingRoot::GENESIS,
            new_root: EmbeddingRoot::from_bytes([1u8; 32]),
            encrypted_notes: vec![],
            vdws: vec![],
        },
    };

    let mnemonic = Mnemonic::generate();
    let keys = WalletKeys::new_from_seed(&mnemonic.to_seed("")).unwrap();
    let note_tracker = Mutex::new(NoteTracker::new());
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let vdw_cache = VdwCache::open(tmp_dir.path()).unwrap();

    engine.sync(&fetcher, &keys, &note_tracker, &vdw_cache).await.unwrap();

    let state = engine.get_state().await;
    assert_eq!(state.current_height, BlockHeight::from(3), "Engine must advance to latest height");
    assert_eq!(state.current_root, EmbeddingRoot::from_bytes([3u8; 32]), "Root must update to the latest");
}

#[tokio::test]
async fn test_sync_engine_detects_root_mismatch() {
    let config = get_test_config();
    let engine = SyncEngine::new(config);
    
    // Initialize with a bad previous root
    engine.load_state(SyncState {
        current_height: BlockHeight::GENESIS,
        current_root: EmbeddingRoot::from_bytes([99u8; 32]), // Mismatched
    });

    let fetcher = MockFetcher {
        latest_height: BlockHeight::from(1),
        batch_to_return: SyncBatch {
            height: BlockHeight::from(1),
            previous_root: EmbeddingRoot::GENESIS,
            new_root: EmbeddingRoot::from_bytes([1u8; 32]),
            encrypted_notes: vec![],
            vdws: vec![],
        },
    };

    let mnemonic = Mnemonic::generate();
    let keys = WalletKeys::new_from_seed(&mnemonic.to_seed("")).unwrap();
    let note_tracker = Mutex::new(NoteTracker::new());
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let vdw_cache = VdwCache::open(tmp_dir.path()).unwrap();

    let result = engine.sync(&fetcher, &keys, &note_tracker, &vdw_cache).await;
    
    assert!(matches!(result, Err(WalletError::Sync(_))), "Sync must fail on root mismatch");
}

#[tokio::test]
async fn test_sync_engine_trial_decrypts_notes() {
    let config = get_test_config();
    let engine = SyncEngine::new(config);
    engine.load_state(SyncState::default());

    let mnemonic = Mnemonic::generate();
    let keys = WalletKeys::new_from_seed(&mnemonic.to_seed("")).unwrap();
    
    // Create an encrypted note specifically for this wallet
    let note = PrivateNote::new(500 * ONE_NERV, [1u8; 32]);
    let encrypted_note = EncryptedNote::encrypt(&note, &keys.kem_keypair.public_key).unwrap();

    let fetcher = MockFetcher {
        latest_height: BlockHeight::from(1),
        batch_to_return: SyncBatch {
            height: BlockHeight::from(1),
            previous_root: EmbeddingRoot::GENESIS,
            new_root: EmbeddingRoot::from_bytes([1u8; 32]),
            encrypted_notes: vec![encrypted_note],
            vdws: vec![],
        },
    };

    let note_tracker = Mutex::new(NoteTracker::new());
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let vdw_cache = VdwCache::open(tmp_dir.path()).unwrap();

    engine.sync(&fetcher, &keys, &note_tracker, &vdw_cache).await.unwrap();

    let tracker = note_tracker.lock();
    assert_eq!(tracker.get_balance(), 500 * ONE_NERV, "Trial-decrypted note must add to balance");
}

// ─── Transaction Builder Tests ───────────────────────────────────────────

fn get_test_dkg_pk() -> DkgPublicKey {
    let s = DkgScalar::from_bls_scalar(&BlsScalar::from(42));
    let commitment = FeldmanCommitment::commit(&s);
    DkgPublicKey {
        point: commitment.point,
        hash: nerv::utils::blake3_hash(&commitment.point),
        session_id: [0u8; 32],
        threshold: 3,
        num_participants: 5,
    }
}

#[test]
fn test_transaction_fee_calculation() {
    // Base fee is 0.001 NERV (1,000,000 nano), plus 0.0001 NERV (100,000 nano) per note
    assert_eq!(TransactionBuilder::calculate_fee(1, 2), 1_300_000); // 1 in, 2 out
    assert_eq!(TransactionBuilder::calculate_fee(3, 2), 1_700_000); // 3 in, 2 out
    assert_eq!(TransactionBuilder::calculate_fee(0, 1), 1_100_000); // 0 in, 1 out
}

#[tokio::test]
async fn test_transaction_builder_success() {
    let dkg_pk = get_test_dkg_pk();
    let weights = Arc::new(vec![vec![1i64; 1]; EMBEDDING_DIM]);
    let bias = Arc::new(vec![0i64; EMBEDDING_DIM]);
    
    let builder = TransactionBuilder::new(dkg_pk, weights, bias);
    
    let mnemonic = Mnemonic::generate();
    let keys = WalletKeys::new_from_seed(&mnemonic.to_seed("")).unwrap();
    
    let tracker = Mutex::new(NoteTracker::new());
    {
        let mut t = tracker.lock();
        t.add_unspent_note(PrivateNote::new(1000 * ONE_NERV, [1u8; 32]));
    }
    
    let recipient_kp = MlKemKeypair::generate().unwrap();
    
    let result = builder.build(&keys, &tracker, &recipient_kp.public_key, 500 * ONE_NERV);
    assert!(result.is_ok(), "Transaction building failed: {:?}", result.err());
    
    let encrypted_tx = result.unwrap();
    assert!(!encrypted_tx.ciphertext.c3.is_empty(), "Threshold ciphertext must exist");
    
    // Verify local state mutation (balance reduced by amount + fee)
    let expected_balance = 1000 * ONE_NERV - 500 * ONE_NERV - 1_300_000; // 499.9987 NERV
    assert_eq!(tracker.lock().get_balance(), expected_balance, "Local balance must reflect spent notes and change");
}

#[tokio::test]
async fn test_transaction_builder_insufficient_funds() {
    let dkg_pk = get_test_dkg_pk();
    let weights = Arc::new(vec![vec![1i64; 1]; EMBEDDING_DIM]);
    let bias = Arc::new(vec![0i64; EMBEDDING_DIM]);
    
    let builder = TransactionBuilder::new(dkg_pk, weights, bias);
    
    let mnemonic = Mnemonic::generate();
    let keys = WalletKeys::new_from_seed(&mnemonic.to_seed("")).unwrap();
    
    let tracker = Mutex::new(NoteTracker::new());
    {
        let mut t = tracker.lock();
        t.add_unspent_note(PrivateNote::new(100 * ONE_NERV, [1u8; 32]));
    }
    
    let recipient_kp = MlKemKeypair::generate().unwrap();
    
    let result = builder.build(&keys, &tracker, &recipient_kp.public_key, 200 * ONE_NERV);
    
    assert!(matches!(result, Err(WalletError::InsufficientFunds { .. })), "Must fail on insufficient funds");
}

#[tokio::test]
async fn test_transaction_builder_zero_amount_fails() {
    let dkg_pk = get_test_dkg_pk();
    let weights = Arc::new(vec![vec![1i64; 1]; EMBEDDING_DIM]);
    let bias = Arc::new(vec![0i64; EMBEDDING_DIM]);
    
    let builder = TransactionBuilder::new(dkg_pk, weights, bias);
    
    let mnemonic = Mnemonic::generate();
    let keys = WalletKeys::new_from_seed(&mnemonic.to_seed("")).unwrap();
    
    let tracker = Mutex::new(NoteTracker::new());
    let recipient_kp = MlKemKeypair::generate().unwrap();
    
    let result = builder.build(&keys, &tracker, &recipient_kp.public_key, 0);
    
    assert!(matches!(result, Err(WalletError::Transaction(_)))), "Must fail on zero amount");
}
