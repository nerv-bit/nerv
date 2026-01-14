//! ML-KEM-768 (Kyber-768) using audited pqcrypto crate
//!
//! This provides NIST-standardized post-quantum key encapsulation (FIPS 203).
//! Security: Level 3 (~128-bit quantum security)
//! Ciphertext size: 1088 bytes
//! Public key size: 1184 bytes
//! Secret key size: 2400 bytes
//! Shared secret: 32 bytes
//!
//! We use kyber768 which is equivalent to ML-KEM-768.

use pqcrypto_kyber::kyber768::{
    keypair, encapsulate, decapsulate,
    public_key_bytes, secret_key_bytes, ciphertext_bytes, shared_secret_bytes
};
use pqcrypto_traits::kem::{PublicKey, SecretKey, Ciphertext, SharedSecret};
use rand::{CryptoRng, RngCore};
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct MlKem768;

#[derive(Debug, Error)]
pub enum MlKemError {
    #[error("Key generation failed")]
    KeyGenFailed,
    #[error("Encapsulation failed")]
    EncapsulationFailed,
    #[error("Decapsulation failed")]
    DecapsulationFailed,
    #[error("Invalid key length")]
    InvalidKey,
    #[error("Invalid ciphertext length")]
    InvalidCiphertext,
}

impl MlKem768 {
    /// Generate a new keypair
    pub fn keypair<R: CryptoRng + RngCore>(rng: &mut R) -> Result<(Vec<u8>, Vec<u8>), MlKemError> {
        let (pk, sk) = keypair(rng);
        Ok((pk.as_bytes().to_vec(), sk.as_bytes().to_vec()))
    }

    /// Encapsulate to public key → (ciphertext, shared_secret)
    pub fn encapsulate(pk: &[u8]) -> Result<(Vec<u8>, Vec<u8>), MlKemError> {
        let public_key = pqcrypto_kyber::kyber768::PublicKey::from_bytes(pk)
            .map_err(|_| MlKemError::InvalidKey)?;
        let (ct, ss) = encapsulate(&public_key);
        Ok((ct.as_bytes().to_vec(), ss.as_bytes().to_vec()))
    }

    /// Decapsulate ciphertext with secret key → shared_secret
    pub fn decapsulate(sk: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, MlKemError> {
        let secret_key = pqcrypto_kyber::kyber768::SecretKey::from_bytes(sk)
            .map_err(|_| MlKemError::InvalidKey)?;
        let ct = pqcrypto_kyber::kyber768::Ciphertext::from_bytes(ciphertext)
            .map_err(|_| MlKemError::InvalidCiphertext)?;
        let ss = decapsulate(&ct, &secret_key)
            .map_err(|_| MlKemError::DecapsulationFailed)?;
        Ok(ss.as_bytes().to_vec())
    }

    /// Constant sizes (matches FIPS 203 ML-KEM-768)
    pub const PUBLIC_KEY_SIZE: usize = public_key_bytes();
    pub const SECRET_KEY_SIZE: usize = secret_key_bytes();
    pub const CIPHERTEXT_SIZE: usize = ciphertext_bytes();
    pub const SHARED_SECRET_SIZE: usize = shared_secret_bytes();
}
