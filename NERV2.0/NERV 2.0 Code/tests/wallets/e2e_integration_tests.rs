//! End-to-End Integration Tests for NERV V2.0 Wallet.
//!
//! Validates the complete lifecycle of a private transaction across two
//! distinct wallet instances (Sender and Receiver) without a live network.
//! 
//! Flow:
//! 1. Initialize Sender and Receiver wallets.
//! 2. Fund the Sender with mock unspent notes.
//! 3. Sender constructs a transaction to the Receiver.
//! 4. The transaction is threshold-encrypted (simulating mixnet entry).
//! 5. The test simulates the DKG threshold decryption ceremony to extract
//!    the recipient's `EncryptedNote` from the transaction payload.
//! 6. A mock `SyncBatch` containing the note is fed to the Receiver's sync engine.
//! 7. Receiver trial-decrypts the note and updates their balance.
//! 8. Sender's change note is similarly synced back to the Sender.

use nerv::wallets::config::{WalletConfig, WalletNetworkConfig, SyncConfig, WalletPrivacyConfig};
use nerv::wallets::keys::{Mnemonic, MlKemKeypair, WalletKeys};
use nerv::wallets::notes::{EncryptedNote, NoteTracker, PrivateNote};
use nerv::wallets::sync::{RpcFetcher, SyncBatch, SyncEngine, SyncState};
use nerv::wallets::transaction::{TransactionBuilder, PrivateTransaction};
use nerv::wallets::vdw_cache::VdwCache;
use nerv::privacy::dkg::{DkgPublicKey, DkgScalar, FeldmanCommitment, ThresholdCiphertext};
use nerv::{BlockHeight, EmbeddingRoot, ONE_NERV, EMBEDDING_DIM, NervResult};
use bls12_381::{Scalar as BlsScalar, G1Affine, G1Projective};
use group::Group;
use parking_lot::Mutex;
use std::sync::Arc;
use async_trait::async_trait;
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use chacha20poly1305::aead::{Aead, KeyInit};

// ─── Test Setup Helpers ──────────────────────────────────────────────────

fn get_test_config(dir: &str) -> WalletConfig {
    WalletConfig {
        wallet_name: dir.into(),
        data_dir: std::path::PathBuf::from(dir),
        network: WalletNetworkConfig::default(),
        sync: SyncConfig { enable_cover_traffic: false, ..Default::default() },
        privacy: WalletPrivacyConfig::default(),
    }
}

fn get_test_dkg_pk_and_secret() -> (DkgPublicKey, DkgScalar) {
    let secret = DkgScalar::from_bls_scalar(&BlsScalar::from(42));
    let commitment = FeldmanCommitment::commit(&secret);
    let pk = DkgPublicKey {
        point: commitment.point,
        hash: nerv::utils::blake3_hash(&commitment.point),
        session_id: [0u8; 32],
        threshold: 3,
        num_participants: 5,
    };
    (pk, secret)
}

/// Helper to simulate the network's DKG threshold decryption ceremony.
/// In production, this happens securely across multiple validators. 
/// Here, we use the collective secret to decrypt the payload for the sync batch.
fn simulate_dkg_decryption(ct: &ThresholdCiphertext, secret: &DkgScalar) -> Vec<u8> {
    let c1 = ct.c1_point();
    let shared_point = G1Projective::from(c1) * secret.to_bls_scalar();
    let shared_bytes = G1Affine::from(shared_point).to_compressed();
    let key = nerv::utils::blake3_hash(shared_bytes.as_ref());
    
    let cipher = ChaCha20Poly1305::new(Key::from_slice(&key));
    let nonce = Nonce::from_slice(&[0u8; 12]); // As used in ThresholdCiphertext::encrypt
    
    cipher.decrypt(nonce, ct.c3.as_ref()).expect("Mock decryption failed")
}

// ─── Mock RPC Fetcher ────────────────────────────────────────────────────

struct MockE2EFetcher {
    latest_height: BlockHeight,
    batch_to_return: SyncBatch,
}

#[async_trait]
impl RpcFetcher for MockE2EFetcher {
    async fn get_latest_height(&self) -> Result<BlockHeight, nerv::WalletError> {
        Ok(self.latest_height)
    }

    async fn fetch_batch(&self, height: BlockHeight, _padding: usize) -> Result<SyncBatch, nerv::WalletError> {
        let mut batch = self.batch_to_return.clone();
        batch.height = height;
        Ok(batch)
    }
}

// ─── E2E Integration Test ────────────────────────────────────────────────

#[tokio::test]
async fn test_e2e_transaction_lifecycle() {
    // 1. Initialize Network Parameters
    let (dkg_pk, dkg_secret) = get_test_dkg_pk_and_secret();
    let weights = Arc::new(vec![vec![1i64; 1]; EMBEDDING_DIM]);
    let bias = Arc::new(vec![0i64; EMBEDDING_DIM]);

    // 2. Initialize Wallets
    let sender_mnemonic = Mnemonic::generate();
    let sender_keys = WalletKeys::new_from_seed(&sender_mnemonic.to_seed("")).unwrap();
    let sender_tracker = Arc::new(Mutex::new(NoteTracker::new()));
    
    let receiver_mnemonic = Mnemonic::generate();
    let receiver_keys = WalletKeys::new_from_seed(&receiver_mnemonic.to_seed("")).unwrap();
    let receiver_tracker = Arc::new(Mutex::new(NoteTracker::new()));

    // 3. Fund the Sender (e.g., 1000 NERV)
    {
        let mut t = sender_tracker.lock();
        t.add_unspent_note(PrivateNote::new(1000 * ONE_NERV, [1u8; 32]));
    }
    assert_eq!(sender_tracker.lock().get_balance(), 1000 * ONE_NERV);

    // 4. Sender constructs a transaction to Receiver for 250 NERV
    let builder = TransactionBuilder::new(dkg_pk.clone(), weights.clone(), bias.clone());
    let amount_to_send = 250 * ONE_NERV;
    
    let encrypted_tx_result = builder.build(
        &sender_keys,
        &sender_tracker,
        &receiver_keys.kem_keypair.public_key,
        amount_to_send,
    );
    assert!(encrypted_tx_result.is_ok(), "TX build failed: {:?}", encrypted_tx_result.err());
    
    let encrypted_tx = encrypted_tx_result.unwrap();

    // 5. Simulate network decryption to extract the notes for the sync batch
    let payload_bytes = simulate_dkg_decryption(&encrypted_tx.ciphertext, &dkg_secret);
    let private_tx: PrivateTransaction = bincode::deserialize(&payload_bytes).unwrap();

    // The transaction should have 2 outputs: Receiver's note and Sender's change note
    assert_eq!(private_tx.outputs.len(), 2, "TX should have 2 outputs");

    // Identify which note belongs to whom via trial-decryption
    let mut receiver_enc_notes = Vec::new();
    let mut sender_change_notes = Vec::new();

    for output in &private_tx.outputs {
        // Try receiver
        if output.encrypted_note.trial_decrypt(&receiver_keys.kem_keypair.secret_key).is_ok() {
            receiver_enc_notes.push(output.encrypted_note.clone());
        }
        // Try sender (change)
        else if output.encrypted_note.trial_decrypt(&sender_keys.kem_keypair.secret_key).is_ok() {
            sender_change_notes.push(output.encrypted_note.clone());
        }
    }

    assert_eq!(receiver_enc_notes.len(), 1, "Receiver should have 1 note");
    assert_eq!(sender_change_notes.len(), 1, "Sender should have 1 change note");

    // 6. Receiver syncs and decrypts their note
    {
        let config = get_test_config("./test-e2e-receiver");
        let engine = SyncEngine::new(config);
        engine.load_state(SyncState::default());
        
        let fetcher = MockE2EFetcher {
            latest_height: BlockHeight::from(1),
            batch_to_return: SyncBatch {
                height: BlockHeight::from(1),
                previous_root: EmbeddingRoot::GENESIS,
                new_root: EmbeddingRoot::from_bytes([1u8; 32]),
                encrypted_notes: receiver_enc_notes,
                vdws: vec![],
            },
        };

        let tmp_dir = tempfile::TempDir::new().unwrap();
        let vdw_cache = VdwCache::open(tmp_dir.path()).unwrap();

        engine.sync(&fetcher, &receiver_keys, &receiver_tracker, &vdw_cache).await.unwrap();
    }

    // 7. Sender syncs their change note
    {
        let config = get_test_config("./test-e2e-sender");
        let engine = SyncEngine::new(config);
        engine.load_state(SyncState::default());
        
        let fetcher = MockE2EFetcher {
            latest_height: BlockHeight::from(1),
            batch_to_return: SyncBatch {
                height: BlockHeight::from(1),
                previous_root: EmbeddingRoot::GENESIS,
                new_root: EmbeddingRoot::from_bytes([1u8; 32]),
                encrypted_notes: sender_change_notes,
                vdws: vec![],
            },
        };

        let tmp_dir = tempfile::TempDir::new().unwrap();
        let vdw_cache = VdwCache::open(tmp_dir.path()).unwrap();

        engine.sync(&fetcher, &sender_keys, &sender_tracker, &vdw_cache).await.unwrap();
    }

    // 8. Final Assertions
    let receiver_balance = receiver_tracker.lock().get_balance();
    assert_eq!(receiver_balance, 250 * ONE_NERV, "Receiver balance must be exactly 250 NERV");

    // Sender balance: 1000 - 250 (sent) - 0.0013 (fee) = 749.9987 NERV
    let expected_sender_balance = 1000 * ONE_NERV - 250 * ONE_NERV - 1_300_000;
    let sender_balance = sender_tracker.lock().get_balance();
    assert_eq!(sender_balance, expected_sender_balance, "Sender balance must reflect change and fee");
}

