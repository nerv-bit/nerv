// tests/wallet_balance_tests.rs
//! Unit tests for Epic 2: Private Note Detection and Balance Computation
//!
//! Tests cover:
//! - Note decryption with correct/incorrect keys
//! - Balance accumulation
//! - Duplicate/spent note handling
//! - Multiple address scanning


#[cfg(test)]
mod tests {
    use super::super::{balance::BalanceTracker, keys::AccountKeys, types::{Output, Note}};
    use nerv::crypto::ml_kem::MlKem768;
    use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
    use hkdf::Hkdf;
    use sha2::Sha256;
    use rand::thread_rng;


    fn create_test_output(enc_pk: &[u8], amount: u128, memo: &str) -> Output {
        let (ct, ss) = MlKem768::encapsulate(enc_pk).unwrap();


        let mut sym_key = [0u8; 32];
        Hkdf::<Sha256>::new(None, &ss).expand(b"nerv-note-encryption-v1", &mut sym_key).unwrap();
        let cipher = Aes256Gcm::new(&sym_key.into());
        let nonce = thread_rng().gen::<[u8; 12]>();


        let payload = bincode::serialize(&super::super::tx::NotePayload { amount, memo: memo.to_string() }).unwrap();
        let encrypted_payload = cipher.encrypt(Nonce::from_slice(&nonce), payload.as_ref()).unwrap();


        Output {
            ct,
            encrypted_payload,
            nonce,
            height: 100,
        }
    }


    #[test]
    fn test_note_detection_and_balance() {
        let wallet = super::super::keys::HdWallet::generate("").unwrap();
        let account = wallet.derive_account(0, 3).unwrap();
        let keys: Vec<&AccountKeys> = account.keys.iter().collect();


        let mut tracker = BalanceTracker::new();


        // Create outputs for first two addresses
        let out1 = create_test_output(&keys[0].enc_pk, 5000, "test1");
        let out2 = create_test_output(&keys[1].enc_pk, 3000, "test2");
        let wrong_out = create_test_output(&[0u8; MlKem768::PUBLIC_KEY_SIZE], 1000, "wrong"); // Wrong pk


        let outputs = vec![out1, out2, wrong_out];
        let new_notes = tracker.scan_outputs(&outputs, &keys);


        assert_eq!(new_notes.len(), 2);
        assert_eq!(tracker.get_balance(), 8000);
        assert_eq!(tracker.get_history().len(), 2);
    }


    #[test]
    fn test_spent_note_handling() {
        let wallet = super::super::keys::HdWallet::generate("").unwrap();
        let account = wallet.derive_account(0, 1).unwrap();
        let keys = vec![&account.keys[0]];


        let mut tracker = BalanceTracker::new();
        let out = create_test_output(&keys[0].enc_pk, 10000, "initial");


        tracker.scan_outputs(&[out], &keys);
        assert_eq!(tracker.get_balance(), 10000);


        let nullifier = blake3::hash(&bincode::serialize(&out).unwrap()).into();
        tracker.spend_notes(&[nullifier]);


        assert_eq!(tracker.get_balance(), 0);
        assert!(tracker.notes.is_empty());
    }
}
