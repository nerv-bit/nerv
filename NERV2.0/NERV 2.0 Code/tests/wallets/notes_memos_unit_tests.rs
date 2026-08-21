//! Unit tests for Private Notes, Trial-Decryption, and Memos.
//!
//! Validates the NERV V2.0 shielded transaction outputs. Ensures that notes
//! are correctly committed, securely encrypted for specific recipients, and
//! that local balance reconstruction via trial-decryption behaves accurately.
//! Also validates the selective disclosure proofs for encrypted memos.

use nerv::wallets::keys::MlKemKeypair;
use nerv::wallets::memo::{Memo, DisclosureProof};
use nerv::wallets::notes::{PrivateNote, EncryptedNote, NoteTracker};
use nerv::ONE_NERV;
use nerv::WalletError;

// ─── Private Note Tests ──────────────────────────────────────────────────

#[test]
fn test_private_note_creation_and_commitment() {
    let receiver = [1u8; 32];
    let note1 = PrivateNote::new(100 * ONE_NERV, receiver);
    let note2 = PrivateNote::new(100 * ONE_NERV, receiver);
    
    // Blinding factors and nullifier seeds must be cryptographically random
    assert_ne!(note1.blinding_factor, note2.blinding_factor, "Blinding factors must differ");
    assert_ne!(note1.nullifier_seed, note2.nullifier_seed, "Nullifier seeds must differ");
    
    // Commitments must be unique even for identical values and receivers
    assert_ne!(note1.compute_commitment(), note2.compute_commitment(), "Commitments must be unique");
}

#[test]
fn test_note_nullifier_computation() {
    let receiver = [2u8; 32];
    let note = PrivateNote::new(50 * ONE_NERV, receiver);
    
    let det_key1 = [10u8; 32];
    let det_key2 = [20u8; 32];
    
    let nullifier1 = note.compute_nullifier(&det_key1);
    let nullifier2 = note.compute_nullifier(&det_key2);
    
    // Same note with different detection keys must yield different nullifiers
    assert_ne!(nullifier1, nullifier2, "Nullifiers must differ across detection keys");
    
    // Must be deterministic for the same key
    let nullifier1_again = note.compute_nullifier(&det_key1);
    assert_eq!(nullifier1, nullifier1_again, "Nullifier must be deterministic");
}

// ─── Encrypted Note & Trial Decryption Tests ─────────────────────────────

#[test]
fn test_note_encryption_and_trial_decryption_success() {
    let keypair = MlKemKeypair::generate().unwrap();
    let receiver = [3u8; 32];
    let note = PrivateNote::new(500 * ONE_NERV, receiver);
    
    // Encrypt the note for the recipient's KEM public key
    let encrypted_note = EncryptedNote::encrypt(&note, &keypair.public_key).unwrap();
    
    // Recipient trial-decrypts using their KEM secret key
    let decrypted_note = encrypted_note.trial_decrypt(&keypair.secret_key).unwrap();
    
    // The decrypted note must perfectly match the original
    assert_eq!(note, decrypted_note, "Trial decryption must yield the exact original note");
}

#[test]
fn test_note_trial_decryption_failure_wrong_key() {
    let keypair1 = MlKemKeypair::generate().unwrap();
    let keypair2 = MlKemKeypair::generate().unwrap();
    let receiver = [4u8; 32];
    let note = PrivateNote::new(500 * ONE_NERV, receiver);
    
    // Encrypt for keypair1
    let encrypted_note = EncryptedNote::encrypt(&note, &keypair1.public_key).unwrap();
    
    // Attempt to decrypt with keypair2's secret key
    let result = encrypted_note.trial_decrypt(&keypair2.secret_key);
    
    // Must fail securely without panicking
    assert!(result.is_err(), "Trial decryption must fail for wrong secret key");
}

// ─── Note Tracker & Balance Reconstruction Tests ─────────────────────────

#[test]
fn test_note_tracker_balance_and_spending() {
    let mut tracker = NoteTracker::new();
    let det_key = [5u8; 32];
    
    let note1 = PrivateNote::new(100 * ONE_NERV, [1u8; 32]);
    let note2 = PrivateNote::new(250 * ONE_NERV, [2u8; 32]);
    
    // Add unspent notes
    tracker.add_unspent_note(note1.clone());
    tracker.add_unspent_note(note2.clone());
    
    // Verify total balance
    assert_eq!(tracker.get_balance(), 350 * ONE_NERV, "Balance must sum unspent notes");
    
    // Spend note1
    let commitment1 = note1.compute_commitment();
    tracker.spend_note(&commitment1, &det_key).unwrap();
    
    // Verify balance updated
    assert_eq!(tracker.get_balance(), 250 * ONE_NERV, "Balance must reflect spent note");
    
    // Verify nullifier tracking
    let nullifier1 = note1.compute_nullifier(&det_key);
    assert!(tracker.is_nullifier_spent(&nullifier1), "Nullifier must be marked as spent");
    
    // Verify spending a non-existent note fails
    let bad_commitment = [99u8; 32];
    assert!(tracker.spend_note(&bad_commitment, &det_key).is_err(), "Spending unknown note must fail");
}

#[test]
fn test_note_tracker_select_notes_sufficient() {
    let mut tracker = NoteTracker::new();
    tracker.add_unspent_note(PrivateNote::new(100 * ONE_NERV, [1u8; 32]));
    tracker.add_unspent_note(PrivateNote::new(50 * ONE_NERV, [2u8; 32]));
    tracker.add_unspent_note(PrivateNote::new(300 * ONE_NERV, [3u8; 32]));
    
    // Select notes to cover 120 NERV
    let selected = tracker.select_notes(120 * ONE_NERV).unwrap();
    
    // Largest-first selection should pick the 300 NERV note
    assert_eq!(selected.len(), 1, "Should select exactly 1 note (largest-first)");
    assert_eq!(selected[0].value, 300 * ONE_NERV, "Should select the highest value note");
    
    // Test selecting exact amount requiring multiple notes
    let selected_multi = tracker.select_notes(150 * ONE_NERV).unwrap();
    assert_eq!(selected_multi.len(), 2, "Should select 2 notes to cover 150");
    assert_eq!(selected_multi.iter().map(|n| n.value).sum::<u64>(), 350 * ONE_NERV, "Total selected must cover target");
}

#[test]
fn test_note_tracker_select_notes_insufficient() {
    let mut tracker = NoteTracker::new();
    tracker.add_unspent_note(PrivateNote::new(100 * ONE_NERV, [1u8; 32]));
    
    // Attempt to select more than available
    let result = tracker.select_notes(200 * ONE_NERV);
    
    assert!(matches!(result, Err(WalletError::InsufficientFunds { .. })), "Must return InsufficientFunds error");
}

// ─── Memo & Selective Disclosure Tests ───────────────────────────────────

#[test]
fn test_memo_encryption_and_decryption() {
    let shared_secret = [1u8; 32];
    let memo = Memo::new("Invoice #123".into()).unwrap();
    
    let encrypted = memo.encrypt(&shared_secret).unwrap();
    let decrypted = encrypted.decrypt(&shared_secret).unwrap();
    
    assert_eq!(memo.text, decrypted.text, "Decrypted memo must match original");
}

#[test]
fn test_memo_decryption_wrong_key() {
    let shared_secret = [1u8; 32];
    let wrong_secret = [2u8; 32];
    let memo = Memo::new("Secret data".into()).unwrap();
    
    let encrypted = memo.encrypt(&shared_secret).unwrap();
    let result = encrypted.decrypt(&wrong_secret);
    
    assert!(result.is_err(), "Decryption with wrong key must fail");
}

#[test]
fn test_memo_size_limit() {
    // 257 characters (exceeds 256 byte limit)
    let long_text = "A".repeat(257);
    let result = Memo::new(long_text);
    
    assert!(result.is_err(), "Memo exceeding size limit must be rejected");
}

#[test]
fn test_selective_disclosure_proof() {
    let shared_secret = [3u8; 32];
    let memo = Memo::new("Payment for consulting".into()).unwrap();
    let encrypted = memo.encrypt(&shared_secret).unwrap();
    
    // Create the proof to share with a third party
    let proof = DisclosureProof::create(&encrypted, &shared_secret);
    
    // Third party verifies and decrypts using only the proof
    let decrypted = proof.verify_and_decrypt().unwrap();
    assert_eq!(decrypted.text, "Payment for consulting", "Third party must decrypt memo from proof");
    
    // Verify base64 export/import functionality
    let exported = proof.to_base64().unwrap();
    let imported = DisclosureProof::from_base64(&exported).unwrap();
    let decrypted_imported = imported.verify_and_decrypt().unwrap();
    assert_eq!(decrypted_imported.text, "Payment for consulting", "Imported proof must decrypt correctly");
}
