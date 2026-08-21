//! Encrypted Memos and Selective Disclosure.
//!
//! Implements the NERV Wallet Whitepaper 1.0 memo and selective disclosure
//! features, adapted for V2.0's pure-cryptographic privacy model.
//!
//! ## Functionality
//! - **Encrypted Memos**: Senders can attach free text (e.g., "Invoice #123")
//!   to a private transaction. The memo is encrypted using ChaCha20-Poly1305
//!   with a key derived from the note's ephemeral shared secret.
//! - **Selective Disclosure**: A user can export a `DisclosureProof` containing
//!   the encrypted memo and the specific derived memo key. This allows a third
//!   party to decrypt and read the memo *without* gaining access to the funds
//!   or the full transaction history.

use crate::wallets::error::{WalletError, WalletResult};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use chacha20poly1305::aead::{Aead, KeyInit};
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

/// The maximum size of a memo payload (in bytes).
pub const MAX_MEMO_SIZE: usize = 256;

/// A plaintext memo attached to a transaction.
#[derive(Debug, Clone, Serialize, Deserialize, Zeroize)]
pub struct Memo {
    /// The free-text content of the memo.
    pub text: String,
}

impl Memo {
    /// Creates a new memo from a string.
    pub fn new(text: String) -> WalletResult<Self> {
        if text.len() > MAX_MEMO_SIZE {
            return Err(WalletError::Transaction(format!(
                "Memo exceeds maximum size of {} bytes",
                MAX_MEMO_SIZE
            )));
        }
        Ok(Self { text })
    }

    /// Encrypts the memo using a key derived from the transaction's shared secret.
    /// 
    /// We use a distinct derivation path ("nerv:memo") so that disclosing the
    /// memo key does not compromise the note's value encryption.
    pub fn encrypt(&self, shared_secret: &[u8]) -> WalletResult<EncryptedMemo> {
        let memo_key = blake3::derive_key("nerv:memo:v1", shared_secret);
        let key = Key::from_slice(&memo_key);
        
        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let cipher = ChaCha20Poly1305::new(key);
        let plaintext = self.text.as_bytes();
        
        let ciphertext = cipher.encrypt(nonce, plaintext)
            .map_err(|e| WalletError::Transaction(format!("Memo encryption failed: {:?}", e)))?;

        Ok(EncryptedMemo {
            ciphertext,
            nonce: nonce_bytes.to_vec(),
        })
    }
}

/// An encrypted memo as it appears alongside the `EncryptedNote`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedMemo {
    /// The AEAD ciphertext of the memo text.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub ciphertext: Vec<u8>,
    /// The 12-byte nonce used for encryption.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub nonce: Vec<u8>,
}

impl EncryptedMemo {
    /// Decrypts the memo using the shared secret.
    pub fn decrypt(&self, shared_secret: &[u8]) -> WalletResult<Memo> {
        let memo_key = blake3::derive_key("nerv:memo:v1", shared_secret);
        let key = Key::from_slice(&memo_key);
        
        let nonce = Nonce::from_slice(&self.nonce);
        let cipher = ChaCha20Poly1305::new(key);
        
        let plaintext = cipher.decrypt(nonce, self.ciphertext.as_ref())
            .map_err(|_| WalletError::NoteProcessing("Memo decryption failed".into()))?;
        
        let text = String::from_utf8(plaintext)
            .map_err(|_| WalletError::NoteProcessing("Invalid UTF-8 in memo".into()))?;
        
        Ok(Memo { text })
    }
}

/// A proof that can be shared with a third party to selectively disclose a memo.
/// 
/// This contains the encrypted memo and the derived memo key. It does NOT contain
/// the full shared secret, meaning the third party cannot decrypt the transaction
/// value or spend the funds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisclosureProof {
    /// The encrypted memo payload.
    pub encrypted_memo: EncryptedMemo,
    /// The 32-byte derived memo key.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub memo_key: [u8; 32],
}

impl DisclosureProof {
    /// Creates a disclosure proof from an encrypted memo and its original shared secret.
    pub fn create(encrypted_memo: &EncryptedMemo, shared_secret: &[u8]) -> Self {
        let memo_key = blake3::derive_key("nerv:memo:v1", shared_secret);
        Self {
            encrypted_memo: encrypted_memo.clone(),
            memo_key,
        }
    }

    /// Verifies and decrypts the memo using the disclosed key.
    pub fn verify_and_decrypt(&self) -> WalletResult<Memo> {
        let key = Key::from_slice(&self.memo_key);
        let nonce = Nonce::from_slice(&self.encrypted_memo.nonce);
        let cipher = ChaCha20Poly1305::new(key);
        
        let plaintext = cipher.decrypt(nonce, self.encrypted_memo.ciphertext.as_ref())
            .map_err(|_| WalletError::VdwVerification("Disclosure proof verification failed".into()))?;
        
        let text = String::from_utf8(plaintext)
            .map_err(|_| WalletError::NoteProcessing("Invalid UTF-8 in memo".into()))?;
        
        Ok(Memo { text })
    }

    /// Exports the proof to a portable base64 string.
    pub fn to_base64(&self) -> WalletResult<String> {
        bincode::serialize(self)
            .map(|bytes| base64::engine::general_purpose::STANDARD.encode(&bytes))
            .map_err(|e| WalletError::Serialization(e.to_string()))
    }

    /// Imports a proof from a base64 string.
    pub fn from_base64(s: &str) -> WalletResult<Self> {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(s)
            .map_err(|e| WalletError::Serialization(format!("base64 decode failed: {}", e)))?;
        
        bincode::deserialize(&bytes)
            .map_err(|e| WalletError::Serialization(e.to_string()))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memo_encryption_decryption() {
        let shared_secret = [1u8; 32];
        let memo = Memo::new("Invoice #123".into()).unwrap();
        
        let encrypted = memo.encrypt(&shared_secret).unwrap();
        let decrypted = encrypted.decrypt(&shared_secret).unwrap();
        
        assert_eq!(memo.text, decrypted.text);
    }

    #[test]
    fn test_memo_decryption_wrong_key() {
        let shared_secret = [1u8; 32];
        let wrong_secret = [2u8; 32];
        let memo = Memo::new("Secret data".into()).unwrap();
        
        let encrypted = memo.encrypt(&shared_secret).unwrap();
        let result = encrypted.decrypt(&wrong_secret);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_memo_size_limit() {
        let long_text = "A".repeat(MAX_MEMO_SIZE + 1);
        let result = Memo::new(long_text);
        assert!(result.is_err());
    }

    #[test]
    fn test_selective_disclosure() {
        let shared_secret = [3u8; 32];
        let memo = Memo::new("Payment for consulting".into()).unwrap();
        let encrypted = memo.encrypt(&shared_secret).unwrap();
        
        // Create the proof
        let proof = DisclosureProof::create(&encrypted, &shared_secret);
        
        // Third party verifies and decrypts
        let decrypted = proof.verify_and_decrypt().unwrap();
        assert_eq!(decrypted.text, "Payment for consulting");
    }

    #[test]
    fn test_disclosure_proof_base64() {
        let shared_secret = [4u8; 32];
        let memo = Memo::new("Test proof".into()).unwrap();
        let encrypted = memo.encrypt(&shared_secret).unwrap();
        
        let proof = DisclosureProof::create(&encrypted, &shared_secret);
        let exported = proof.to_base64().unwrap();
        
        let imported = DisclosureProof::from_base64(&exported).unwrap();
        let decrypted = imported.verify_and_decrypt().unwrap();
        
        assert_eq!(decrypted.text, "Test proof");
    }
}
