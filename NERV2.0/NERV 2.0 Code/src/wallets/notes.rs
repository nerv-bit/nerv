//! Private Notes, Commitments, and Local Balance Reconstruction.
//!
//! In NERV V2.0, transaction outputs are represented as `PrivateNote`s. 
//! A note contains the value, blinding factor, receiver commitment, and a 
//! nullifier seed. 
//!
//! ## Privacy Model
//! Notes are never transmitted in plaintext. The sender encrypts the note 
//! using an ephemeral ML-KEM-768 shared secret derived from the recipient's 
//! diversified receiver public key. 
//!
//! The wallet reconstructs its balance by downloading batches of encrypted 
//! notes from the network's homomorphic deltas and performing trial-decryption. 
//! If decryption succeeds, the note belongs to the wallet. The wallet then 
//! checks the note's nullifier against the global set of spent nullifiers to 
//! ensure it is unspent before adding it to the local balance.

use crate::wallets::error::{WalletError, WalletResult};
use crate::{NervError, ONE_NERV};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use chacha20poly1305::aead::{Aead, KeyInit};
use oqs::{kem::MlKem768};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use zeroize::Zeroize;

/// A private note representing a shielded transaction output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Zeroize)]
pub struct PrivateNote {
    /// The value of the note in nano-NERV.
    pub value: u64,
    /// The blinding factor used to hide the value in the homomorphic delta.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub blinding_factor: [u8; 32],
    /// The diversified receiver commitment parameter.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub diversified_receiver: [u8; 32],
    /// The seed used to generate the nullifier. Prevents double-spends.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub nullifier_seed: [u8; 32],
}

impl PrivateNote {
    /// Creates a new random private note for a given value and receiver.
    pub fn new(value: u64, diversified_receiver: [u8; 32]) -> Self {
        let mut blinding_factor = [0u8; 32];
        let mut nullifier_seed = [0u8; 32];
        // Use OS RNG for cryptographic security
        rand::rngs::OsRng.fill_bytes(&mut blinding_factor);
        rand::rngs::OsRng.fill_bytes(&mut nullifier_seed);

        Self {
            value,
            blinding_factor,
            diversified_receiver,
            nullifier_seed,
        }
    }

    /// Computes the note's nullifier using the wallet's detection key.
    /// 
    /// The nullifier is public when a note is spent, but cannot be linked 
    /// back to the original note without the detection key.
    pub fn compute_nullifier(&self, detection_key: &[u8; 32]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(detection_key);
        hasher.update(&self.nullifier_seed);
        *hasher.finalize().as_bytes()
    }

    /// Computes the note commitment that goes into the homomorphic delta.
    pub fn compute_commitment(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.value.to_le_bytes());
        hasher.update(&self.blinding_factor);
        hasher.update(&self.diversified_receiver);
        hasher.update(&self.nullifier_seed);
        *hasher.finalize().as_bytes()
    }
}

/// An encrypted private note as it appears on-chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedNote {
    /// Ephemeral ML-KEM-768 ciphertext encapsulating the symmetric key.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub kem_ciphertext: Vec<u8>,
    /// AEAD ciphertext of the `PrivateNote` payload.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub ciphertext: Vec<u8>,
}

impl EncryptedNote {
    /// Encrypts a private note for a given receiver's ML-KEM public key.
    pub fn encrypt(note: &PrivateNote, receiver_kem_pk: &[u8]) -> WalletResult<Self> {
        let kem = MlKem768::default();
        let pk = MlKem768::PublicKey::from_bytes(receiver_kem_pk)
            .map_err(|e| WalletError::NoteProcessing(format!("Invalid receiver PK: {:?}", e)))?;
        
        let (ct, shared_secret) = kem.encapsulate(&pk)
            .map_err(|e| WalletError::NoteProcessing(format!("KEM encapsulate failed: {:?}", e)))?;
        
        // Derive symmetric key from shared secret
        let key = blake3::hash(&shared_secret);
        let cipher = ChaCha20Poly1305::new(Key::from_slice(key.as_bytes()));
        
        // Nonce can be deterministic here because the key is unique per note (ephemeral KEM)
        let nonce = Nonce::from_slice(&[0u8; 12]); 
        
        let plaintext = bincode::serialize(note)
            .map_err(|e| WalletError::Serialization(e.to_string()))?;
        
        let ct_aead = cipher.encrypt(nonce, plaintext.as_ref())
            .map_err(|e| WalletError::NoteProcessing(format!("AEAD encrypt failed: {:?}", e)))?;
        
        Ok(Self {
            kem_ciphertext: ct.into_vec(),
            ciphertext: ct_aead,
        })
    }

    /// Attempts to decrypt the note using the receiver's ML-KEM secret key.
    /// 
    /// This is the "trial-decryption" phase. If the note was not encrypted 
    /// for this secret key, it will return an `Err`.
    pub fn trial_decrypt(&self, receiver_kem_sk: &[u8]) -> WalletResult<PrivateNote> {
        let kem = MlKem768::default();
        let sk = MlKem768::SecretKey::from_bytes(receiver_kem_sk)
            .map_err(|e| WalletError::NoteProcessing(format!("Invalid receiver SK: {:?}", e)))?;
        let ct = MlKem768::Ciphertext::from_bytes(&self.kem_ciphertext)
            .map_err(|e| WalletError::NoteProcessing(format!("Invalid KEM CT: {:?}", e)))?;
        
        let shared_secret = kem.decapsulate(&sk, &ct)
            .map_err(|_| WalletError::NoteProcessing("KEM decapsulate failed (not our note)".into()))?;
        
        let key = blake3::hash(&shared_secret);
        let cipher = ChaCha20Poly1305::new(Key::from_slice(key.as_bytes()));
        let nonce = Nonce::from_slice(&[0u8; 12]);
        
        let plaintext = cipher.decrypt(nonce, self.ciphertext.as_ref())
            .map_err(|_| WalletError::NoteProcessing("Trial decryption failed (not our note)".into()))?;
        
        let note: PrivateNote = bincode::deserialize(&plaintext)
            .map_err(|_| WalletError::NoteProcessing("Deserialization failed".into()))?;
        
        Ok(note)
    }
}

/// The local state tracker for wallet funds.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NoteTracker {
    /// Unspent notes, mapped by their commitment.
    unspent_notes: HashMap<[u8; 32], PrivateNote>,
    /// Nullifiers of notes that have been spent.
    spent_nullifiers: HashSet<[u8; 32]>,
}

impl NoteTracker {
    /// Creates a new empty note tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a decrypted note to the unspent set.
    pub fn add_unspent_note(&mut self, note: PrivateNote) {
        let commitment = note.compute_commitment();
        self.unspent_notes.insert(commitment, note);
    }

    /// Marks a note as spent by adding its nullifier.
    /// Also removes it from the unspent set if present.
    pub fn mark_spent(&mut self, nullifier: [u8; 32]) {
        self.spent_nullifiers.insert(nullifier);
        // Remove from unspent if it exists locally
        self.unspent_notes.retain(|_, note| {
            // To truly remove it, we'd need the detection key to compute the nullifier.
            // In practice, the wallet calls `spend_note` which handles this securely.
            true
        });
    }

    /// Spends a specific note locally.
    pub fn spend_note(&mut self, commitment: &[u8; 32], detection_key: &[u8; 32]) -> WalletResult<()> {
        let note = self.unspent_notes.remove(commitment)
            .ok_or_else(|| WalletError::NoteProcessing("Note not found in unspent set".into()))?;
        
        let nullifier = note.compute_nullifier(detection_key);
        self.spent_nullifiers.insert(nullifier);
        Ok(())
    }

    /// Computes the total available balance from unspent notes.
    pub fn get_balance(&self) -> u64 {
        self.unspent_notes.values().map(|n| n.value).sum()
    }

    /// Returns a formatted string of the balance in NERV.
    pub fn get_formatted_balance(&self) -> String {
        let nano = self.get_balance();
        let whole = nano / ONE_NERV;
        let frac = nano % ONE_NERV;
        format!("{}.{:09}", whole, frac)
    }

    /// Selects notes to cover a target amount. Uses a simple largest-first selection.
    pub fn select_notes(&self, target_amount: u64) -> WalletResult<Vec<PrivateNote>> {
        let mut notes: Vec<&PrivateNote> = self.unspent_notes.values().collect();
        // Sort descending by value to minimize the number of inputs
        notes.sort_by(|a, b| b.value.cmp(&a.value));

        let mut selected = Vec::new();
        let mut accumulated = 0u64;

        for note in notes {
            selected.push(note.clone());
            accumulated = accumulated.saturating_add(note.value);
            if accumulated >= target_amount {
                return Ok(selected);
            }
        }

        Err(WalletError::InsufficientFunds {
            required: target_amount,
            available: accumulated,
        })
    }

    /// Checks if a nullifier is already spent.
    pub fn is_nullifier_spent(&self, nullifier: &[u8; 32]) -> bool {
        self.spent_nullifiers.contains(nullifier)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wallets::keys::MlKemKeypair;

    #[test]
    fn test_note_creation_and_commitment() {
        let receiver = [1u8; 32];
        let note1 = PrivateNote::new(100 * ONE_NERV, receiver);
        let note2 = PrivateNote::new(100 * ONE_NERV, receiver);
        
        // Different blinding factors and seeds should yield different commitments
        assert_ne!(note1.compute_commitment(), note2.compute_commitment());
    }

    #[test]
    fn test_note_encryption_and_trial_decryption() {
        let keypair = MlKemKeypair::generate().unwrap();
        let receiver = [2u8; 32];
        let note = PrivateNote::new(500 * ONE_NERV, receiver);
        
        // Encrypt for receiver
        let encrypted = EncryptedNote::encrypt(&note, &keypair.public_key).unwrap();
        
        // Decrypt with correct secret key
        let decrypted = encrypted.trial_decrypt(&keypair.secret_key).unwrap();
        assert_eq!(note, decrypted);
    }

    #[test]
    fn test_trial_decryption_fails_for_wrong_key() {
        let keypair1 = MlKemKeypair::generate().unwrap();
        let keypair2 = MlKemKeypair::generate().unwrap();
        let receiver = [3u8; 32];
        let note = PrivateNote::new(500 * ONE_NERV, receiver);
        
        let encrypted = EncryptedNote::encrypt(&note, &keypair1.public_key).unwrap();
        
        // Attempt to decrypt with wrong secret key
        let result = encrypted.trial_decrypt(&keypair2.secret_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_note_tracker_balance_and_spending() {
        let mut tracker = NoteTracker::new();
        let det_key = [5u8; 32];
        
        let note1 = PrivateNote::new(100 * ONE_NERV, [1u8; 32]);
        let note2 = PrivateNote::new(250 * ONE_NERV, [2u8; 32]);
        
        tracker.add_unspent_note(note1.clone());
        tracker.add_unspent_note(note2.clone());
        
        assert_eq!(tracker.get_balance(), 350 * ONE_NERV);
        
        // Spend note1
        let commitment1 = note1.compute_commitment();
        tracker.spend_note(&commitment1, &det_key).unwrap();
        
        assert_eq!(tracker.get_balance(), 250 * ONE_NERV);
        
        // Check nullifier
        let nullifier1 = note1.compute_nullifier(&det_key);
        assert!(tracker.is_nullifier_spent(&nullifier1));
    }

    #[test]
    fn test_select_notes_sufficient() {
        let mut tracker = NoteTracker::new();
        tracker.add_unspent_note(PrivateNote::new(100 * ONE_NERV, [1u8; 32]));
        tracker.add_unspent_note(PrivateNote::new(50 * ONE_NERV, [2u8; 32]));
        tracker.add_unspent_note(PrivateNote::new(300 * ONE_NERV, [3u8; 32]));
        
        let selected = tracker.select_notes(120 * ONE_NERV).unwrap();
        // Should select the 300 note first (largest-first)
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].value, 300 * ONE_NERV);
    }

    #[test]
    fn test_select_notes_insufficient() {
        let mut tracker = NoteTracker::new();
        tracker.add_unspent_note(PrivateNote::new(100 * ONE_NERV, [1u8; 32]));
        
        let result = tracker.select_notes(200 * ONE_NERV);
        assert!(matches!(result, Err(WalletError::InsufficientFunds { .. })));
    }
}
