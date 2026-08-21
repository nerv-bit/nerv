//! Post-Quantum Hierarchical Deterministic (HD) Key Management.
//!
//! Implements the NERV wallet key hierarchy using a BIP-39 compatible mnemonic
//! seed (PQ-hardened) to derive spending and detection keys. Because NERV V2.0
//! uses Post-Quantum cryptography (Dilithium-3 and ML-KEM-768), standard ECC
//! derivation (BIP-32) is replaced with a BLAKE3/HMAC-SHA512 based derivation
//! tree. 
//!
//! ## Derivation Path
//! `m / purpose' / coin_type' / account' / change / index`
//! - `purpose` = 32' (Shielded NERV)
//! - `coin_type` = 133' (NERV)
//!
//! ## Key Types
//! - **Spending Key**: Dilithium-3 keypair for transaction non-repudiation.
//! - **KEM Key**: ML-KEM-768 keypair for note/memo encryption and decryption.
//! - **Detection Key**: Derived symmetric key for trial-decryption of blinded commitments.

use crate::{
    DILITHIUM3_PK_BYTES, DILITHIUM3_SK_BYTES, 
    ML_KEM768_PK_BYTES, ML_KEM768_SK_BYTES,
    NervError, NervResult,
};
use hmac::{Hmac, Mac};
use oqs::{kem::MlKem768, sig::Dilithium3};
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use sha2::Sha512;
use zeroize::Zeroize;

type HmacSha512 = Hmac<Sha512>;

/// NERV BIP-39 purpose code for shielded derivation.
const PURPOSE_SHIELDED: u32 = 32;
/// NERV SLIP-44 coin type.
const COIN_TYPE_NERV: u32 = 133;

// ─── Mnemonic ─────────────────────────────────────────────────────────────

/// A PQ-hardened mnemonic seed (256-bit entropy).
/// 
/// In production, this 32-byte entropy is mapped to a 24-word BIP-39 mnemonic
/// phrase for user backup. The wallet UI layer handles the word mapping.
#[derive(Clone, Zeroize)]
pub struct Mnemonic {
    entropy: [u8; 32],
}

impl Mnemonic {
    /// Generates a new random mnemonic using the OS CSPRNG.
    pub fn generate() -> Self {
        let mut entropy = [0u8; 32];
        OsRng.fill_bytes(&mut entropy);
        Self { entropy }
    }

    /// Creates a mnemonic from raw 32-byte entropy.
    pub fn from_entropy(entropy: [u8; 32]) -> Self {
        Self { entropy }
    }

    /// Derives the master seed using PBKDF2-HMAC-SHA512 (BIP-39 standard).
    /// The passphrase is optional and can be used for additional security.
    pub fn to_seed(&self, passphrase: &str) -> [u8; 64] {
        // In a full BIP-39 implementation, the mnemonic words are the password.
        // Here, we use the hex of the entropy directly to keep the crypto core pure.
        let mnemonic_hex = hex::encode(self.entropy);
        let salt = format!("mnemonic{}", passphrase);
        
        let mut seed = [0u8; 64];
        pbkdf2::pbkdf2::<HmacSha512>(mnemonic_hex.as_bytes(), salt.as_bytes(), 2048, &mut seed);
        seed
    }

    /// Returns the raw entropy.
    pub fn entropy(&self) -> &[u8; 32] {
        &self.entropy
    }
}

impl Drop for Mnemonic {
    fn drop(&mut self) {
        self.entropy.zeroize();
    }
}

// ─── Derivation Path ──────────────────────────────────────────────────────

/// Represents an HD derivation path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivationPath {
    pub purpose: u32,
    pub coin_type: u32,
    pub account: u32,
    pub change: u32,
    pub index: u32,
}

impl Default for DerivationPath {
    fn default() -> Self {
        Self {
            purpose: PURPOSE_SHIELDED,
            coin_type: COIN_TYPE_NERV,
            account: 0,
            change: 0,
            index: 0,
        }
    }
}

impl DerivationPath {
    /// Serializes the path to bytes for HMAC derivation.
    /// Hardened indices (bit 31 set) are handled by standard serialization.
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(20);
        bytes.extend_from_slice(&self.purpose.to_be_bytes());
        bytes.extend_from_slice(&self.coin_type.to_be_bytes());
        bytes.extend_from_slice(&self.account.to_be_bytes());
        bytes.extend_from_slice(&self.change.to_be_bytes());
        bytes.extend_from_slice(&self.index.to_be_bytes());
        bytes
    }
}

// ─── Dilithium-3 Keypair Wrapper ──────────────────────────────────────────

/// A Dilithium-3 keypair for spending (signing transactions).
#[derive(Clone, Serialize, Deserialize)]
pub struct DilithiumKeypair {
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub public_key: Vec<u8>,
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub secret_key: Vec<u8>,
}

impl DilithiumKeypair {
    /// Generates a new Dilithium-3 keypair using the OS CSPRNG.
    /// Note: `oqs` handles key generation securely, but this is not deterministic.
    /// For HD derivation, the wallet generates the keypair once and encrypts it 
    /// with the derived secret.
    pub fn generate() -> NervResult<Self> {
        let sig = Dilithium3::default();
        let (pk, sk) = sig.keypair()
            .map_err(|e| NervError::Crypto(format!("Dilithium keygen failed: {:?}", e)))?;
        
        Ok(Self {
            public_key: pk.into_vec(),
            secret_key: sk.into_vec(),
        })
    }

    /// Signs a message with the secret key.
    pub fn sign(&self, msg: &[u8]) -> NervResult<Vec<u8>> {
        let sig = Dilithium3::default();
        let pk = Dilithium3::PublicKey::from_bytes(&self.public_key)
            .map_err(|e| NervError::Crypto(format!("Invalid Dilithium PK: {:?}", e)))?;
        let sk = Dilithium3::SecretKey::from_bytes(&self.secret_key)
            .map_err(|e| NervError::Crypto(format!("Invalid Dilithium SK: {:?}", e)))?;
        
        let signature = sig.sign(msg, &sk)
            .map_err(|e| NervError::Crypto(format!("Dilithium signing failed: {:?}", e)))?;
        Ok(signature.into_vec())
    }

    /// Verifies a signature.
    pub fn verify(&self, msg: &[u8], signature: &[u8]) -> bool {
        let sig = Dilithium3::default();
        let pk = match Dilithium3::PublicKey::from_bytes(&self.public_key) {
            Ok(pk) => pk,
            Err(_) => return false,
        };
        let s = match Dilithium3::Signature::from_bytes(signature) {
            Ok(s) => s,
            Err(_) => return false,
        };
        sig.verify(msg, &s, &pk).is_ok()
    }
}

impl Drop for DilithiumKeypair {
    fn drop(&mut self) {
        self.secret_key.zeroize();
    }
}

// ─── ML-KEM-768 Keypair Wrapper ───────────────────────────────────────────

/// An ML-KEM-768 keypair for note/memo encryption.
#[derive(Clone, Serialize, Deserialize)]
pub struct MlKemKeypair {
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub public_key: Vec<u8>,
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub secret_key: Vec<u8>,
}

impl MlKemKeypair {
    /// Generates a new ML-KEM-768 keypair.
    pub fn generate() -> NervResult<Self> {
        let kem = MlKem768::default();
        let (pk, sk) = kem.keypair()
            .map_err(|e| NervError::Crypto(format!("ML-KEM keygen failed: {:?}", e)))?;
        
        Ok(Self {
            public_key: pk.into_vec(),
            secret_key: sk.into_vec(),
        })
    }

    /// Encapsulates a shared secret using the public key.
    pub fn encapsulate(&self) -> NervResult<(Vec<u8>, [u8; 32])> {
        let kem = MlKem768::default();
        let pk = MlKem768::PublicKey::from_bytes(&self.public_key)
            .map_err(|e| NervError::Crypto(format!("Invalid ML-KEM PK: {:?}", e)))?;
        
        let (ct, ss) = kem.encapsulate(&pk)
            .map_err(|e| NervError::Crypto(format!("ML-KEM encapsulate failed: {:?}", e)))?;
        
        Ok((ct.into_vec(), ss))
    }

    /// Decapsulates a shared secret using the secret key.
    pub fn decapsulate(&self, ciphertext: &[u8]) -> NervResult<[u8; 32]> {
        let kem = MlKem768::default();
        let sk = MlKem768::SecretKey::from_bytes(&self.secret_key)
            .map_err(|e| NervError::Crypto(format!("Invalid ML-KEM SK: {:?}", e)))?;
        let ct = MlKem768::Ciphertext::from_bytes(ciphertext)
            .map_err(|e| NervError::Crypto(format!("Invalid ML-KEM CT: {:?}", e)))?;
        
        let ss = kem.decapsulate(&sk, &ct)
            .map_err(|e| NervError::Crypto(format!("ML-KEM decapsulate failed: {:?}", e)))?;
        
        let mut shared_secret = [0u8; 32];
        shared_secret.copy_from_slice(&ss);
        Ok(shared_secret)
    }
}

impl Drop for MlKemKeypair {
    fn drop(&mut self) {
        self.secret_key.zeroize();
    }
}

// ─── Wallet Keys ──────────────────────────────────────────────────────────

/// The complete set of keys managed by a NERV wallet.
#[derive(Clone, Serialize, Deserialize)]
pub struct WalletKeys {
    /// The master seed used for deterministic derivation of detection keys.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub master_seed: Vec<u8>,
    /// The Dilithium-3 keypair for spending.
    pub spending_keypair: DilithiumKeypair,
    /// The ML-KEM-768 keypair for receiving encrypted notes.
    pub kem_keypair: MlKemKeypair,
}

impl WalletKeys {
    /// Creates a new `WalletKeys` instance from a mnemonic seed.
    /// Generates fresh PQ keypairs and initializes the master seed.
    pub fn new_from_seed(seed: &[u8; 64]) -> NervResult<Self> {
        Ok(Self {
            master_seed: seed.to_vec(),
            spending_keypair: DilithiumKeypair::generate()?,
            kem_keypair: MlKemKeypair::generate()?,
        })
    }

    /// Derives a symmetric detection key for a specific path.
    /// 
    /// This key is used to trial-decrypt private note commitments in the 
    /// homomorphic delta batches without revealing the spending key.
    pub fn derive_detection_key(&self, path: &DerivationPath) -> [u8; 32] {
        let mut mac = HmacSha512::new_from_slice(&self.master_seed)
            .expect("HMAC accepts any key size");
        mac.update(&path.to_bytes());
        
        let result = mac.finalize().into_bytes();
        let mut key = [0u8; 32];
        key.copy_from_slice(&result[..32]);
        key
    }

    /// Derives a diversified receiver commitment parameter.
    /// 
    /// Used to generate unique blinded commitment params for each address index,
    /// preventing public key reuse while only requiring a single PQ keypair.
    pub fn derive_diversified_receiver(&self, index: u32) -> [u8; 32] {
        let path = DerivationPath {
            index,
            ..Default::default()
        };
        self.derive_detection_key(&path)
    }
}

impl Drop for WalletKeys {
    fn drop(&mut self) {
        self.master_seed.zeroize();
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnemonic_generation_and_seed() {
        let mnemonic = Mnemonic::generate();
        let seed1 = mnemonic.to_seed("");
        let seed2 = mnemonic.to_seed("");
        assert_eq!(seed1, seed2, "Seeds should be deterministic");

        let seed3 = mnemonic.to_seed("extra_passphrase");
        assert_ne!(seed1, seed3, "Different passphrases should yield different seeds");
    }

    #[test]
    fn test_dilithium_signing() {
        let keypair = DilithiumKeypair::generate().unwrap();
        let msg = b"test transaction payload";
        
        let signature = keypair.sign(msg).unwrap();
        assert!(keypair.verify(msg, &signature), "Signature must verify");
        
        let bad_msg = b"tampered message";
        assert!(!keypair.verify(bad_msg, &signature), "Tampered message must fail");
    }

    #[test]
    fn test_ml_kem_encapsulation() {
        let keypair = MlKemKeypair::generate().unwrap();
        
        let (ciphertext, shared_secret_sender) = keypair.encapsulate().unwrap();
        let shared_secret_receiver = keypair.decapsulate(&ciphertext).unwrap();
        
        assert_eq!(
            shared_secret_sender, shared_secret_receiver,
            "Encapsulated shared secrets must match"
        );
    }

    #[test]
    fn test_wallet_keys_derivation() {
        let mnemonic = Mnemonic::generate();
        let seed = mnemonic.to_seed("");
        let wallet_keys = WalletKeys::new_from_seed(&seed).unwrap();

        let path1 = DerivationPath::default();
        let path2 = DerivationPath { index: 1, ..Default::default() };

        let det_key1 = wallet_keys.derive_detection_key(&path1);
        let det_key2 = wallet_keys.derive_detection_key(&path2);
        let det_key1_again = wallet_keys.derive_detection_key(&path1);

        assert_ne!(det_key1, det_key2, "Different paths must yield different keys");
        assert_eq!(det_key1, det_key1_again, "Same path must yield same key");
    }
}
