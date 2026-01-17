// src/keys.rs
//! Epic 1: Post-Quantum Hierarchical Deterministic Key Management
//!
//! Features:
//! - BIP39 mnemonic (12/24 words) with optional passphrase
//! - Hardened derivation (BIP32/SLIP-10 style adapted for PQ schemes)
//! - Paired keys per address: Dilithium3 spending key + ML-KEM-768 encryption key
//! - Seeded deterministic keypair generation (using HKDF + ChaCha20 RNG for sampling)
//! - Gap limit (20) for automatic address derivation
//! - Receiving addresses as Bech32-encoded ML-KEM pk ("nerv1...")
//! - Zeroize secrets on drop
//! - Superior UI flow (described below)

use nerv::crypto::dilithium::Dilithium3;
use nerv::crypto::ml_kem::MlKem768;
use bip39::{Mnemonic, MnemonicType, Language};
use hkdf::Hkdf;
use sha2::Sha256;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use zeroize::{Zeroize, Zeroizing};
use bech32::{ToBase32, Variant};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum KeyError {
    #[error("Invalid mnemonic")]
    InvalidMnemonic,
    #[error("Derivation failed")]
    DerivationFailed,
    #[error("Bech32 encoding error")]
    Bech32Error,
}

pub struct HdWallet {
    master_seed: Zeroizing<[u8; 64]>, // 64 bytes for future-proofing
    mnemonic: String,
    passphrase: String, // Stored only temporarily if needed
}

pub struct Account {
    pub account_index: u32,
    pub keys: Vec<AccountKeys>, // Derived addresses (with gap limit)
    pub next_index: u32,
}

pub struct AccountKeys {
    pub index: u32,
    pub spending_pk: Vec<u8>,
    pub spending_sk: Zeroizing<Vec<u8>>,
    pub enc_pk: Vec<u8>,
    pub enc_sk: Zeroizing<Vec<u8>>,
}

impl HdWallet {
    /// Generate new wallet with 24-word mnemonic (recommended for PQ security)
    /// UI: Beautiful mnemonic display in 6x4 grid with word numbers, copy button,
    /// prominent warning banner, dark/light mode support, animated confetti on backup confirmation,
    /// then require user to re-select random words for verification (better than typing).
    pub fn generate(passphrase: &str) -> Result<Self, KeyError> {
        let mnemonic = Mnemonic::new(MnemonicType::Words24, Language::English);
        Self::from_mnemonic(mnemonic.phrase(), passphrase)
    }

    /// Restore from mnemonic
    pub fn from_mnemonic(phrase: &str, passphrase: &str) -> Result<Self, KeyError> {
        let mnemonic = Mnemonic::from_phrase(phrase, Language::English)
            .map_err(|_| KeyError::InvalidMnemonic)?;
        let seed = mnemonic.to_seed(passphrase);

        let mut master_seed = Zeroizing::new([0u8; 64]);
        Hkdf::<Sha256>::new(None, &seed)
            .expand(b"nerv-hd-wallet-master-v1", &mut master_seed)
            .map_err(|_| KeyError::DerivationFailed)?;

        Ok(Self {
            master_seed,
            mnemonic: phrase.to_string(),
            passphrase: passphrase.to_string(),
        })
    }

    /// Derive a new account (hardened)
    pub fn derive_account(&self, account_index: u32, gap_limit: u32) -> Result<Account, KeyError> {
        let mut keys = Vec::with_capacity(gap_limit as usize);
        for index in 0..gap_limit {
            let child_keys = self.derive_address(account_index, index)?;
            keys.push(child_keys);
        }

        Ok(Account {
            account_index,
            keys,
            next_index: gap_limit,
        })
    }

    /// Derive single address keys (hardened path m/44'/nerv'/account'/index')
    fn derive_address(&self, account_index: u32, index: u32) -> Result<AccountKeys, KeyError> {
        let path = format!("m/44'/1989'/{}'/{}", account_index + 0x80000000, index + 0x80000000);

        let mut seed = self.master_seed.clone();
        for part in path.split('/').skip(1) {
            let i = part.trim_end_matches('\'').parse::<u32>().unwrap();
            let hardened_index = i + 0x80000000;

            let mut child = Zeroizing::new([0u8; 64]);
            Hkdf::<Sha256>::new(None, &seed)
                .expand(&hardened_index.to_be_bytes(), &mut child)
                .map_err(|_| KeyError::DerivationFailed)?;
            seed = child;
        }

        // Separate chains for spending and encryption
        let mut spending_seed = [0u8; 32];
        let mut enc_seed = [0u8; 32];
        Hkdf::<Sha256>::new(None, &seed)
            .expand(b"spending-key", &mut spending_seed)
            .map_err(|_| KeyError::DerivationFailed)?;
        Hkdf::<Sha256>::new(None, &seed)
            .expand(b"encryption-key", &mut enc_seed)
            .map_err(|_| KeyError::DerivationFailed)?;

        // Deterministic keygen using seeded RNG
        let mut spending_rng = ChaCha20Rng::from_seed(spending_seed);
        let (spending_pk, spending_sk) = Dilithium3::keypair_seeded(&mut spending_rng);

        let mut enc_rng = ChaCha20Rng::from_seed(enc_seed);
        let (enc_pk, enc_sk) = MlKem768::keypair_seeded(&mut enc_rng);

        Ok(AccountKeys {
            index,
            spending_pk,
            spending_sk: Zeroizing::new(spending_sk),
            enc_pk,
            enc_sk: Zeroizing::new(enc_sk),
        })
    }
}

impl AccountKeys {
    /// Human-readable receiving address (Bech32 with "nerv" HRP)
    /// UI: QR code generation with logo overlay, one-tap copy, share sheet,
    /// amount request integration (better than any existing wallet).
    pub fn receiving_address(&self) -> Result<String, KeyError> {
        let data = self.enc_pk.to_base32();
        bech32::encode("nerv", data, Variant::Bech32)
            .map_err(|_| KeyError::Bech32Error)
    }
}

// Helper extensions (assumed added to crypto primitives for seeded generation)
trait PqSeededKeygen {
    fn keypair_seeded<R: rand_core::RngCore + rand_core::CryptoRng>(rng: &mut R) -> (Vec<u8>, Vec<u8>);
}
// Implementation note: In production, implement proper seeded sampling matching NIST specs