//! CRYSTALS-Dilithium-3 (ML-DSA) using audited pqcrypto crate
//! 
//! This provides NIST-standardized post-quantum digital signatures.
//! Security: ~128-bit quantum security (Level 3)
//! Signature size: ~2420-3293 bytes (depending on variant)
//! We use dilithium3 (recommended balanced parameter set)

use pqcrypto_dilithium::dilithium3::{
    detached_sign, verify_detached_signature, 
    keypair, public_key_bytes, secret_key_bytes, signature_bytes
};
use pqcrypto_traits::sign::{PublicKey, SecretKey, DetachedSignature};
use rand::{CryptoRng, RngCore};
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct Dilithium3;

#[derive(Debug, Error)]
pub enum DilithiumError {
    #[error("Signature verification failed")]
    VerificationFailed,
    #[error("Invalid key length")]
    InvalidKey,
    #[error("Signing failed")]
    SigningFailed,
}

impl Dilithium3 {
    /// Generate a new keypair
    pub fn keypair<R: CryptoRng + RngCore>(rng: &mut R) -> (Vec<u8>, Vec<u8>) { // (pk, sk)
        let (pk, sk) = keypair(rng);
        (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
    }

    /// Sign a message (deterministic mode available via pqcrypto)
    pub fn sign(sk: &[u8], message: &[u8]) -> Result<Vec<u8>, DilithiumError> {
        let secret_key = pqcrypto_dilithium::dilithium3::SecretKey::from_bytes(sk)
            .map_err(|_| DilithiumError::InvalidKey)?;
        let signature = detached_sign(message, &secret_key);
        Ok(signature.as_bytes().to_vec())
    }

    /// Verify a signature
    pub fn verify(pk: &[u8], message: &[u8], signature: &[u8]) -> Result<bool, DilithiumError> {
        let public_key = pqcrypto_dilithium::dilithium3::PublicKey::from_bytes(pk)
            .map_err(|_| DilithiumError::InvalidKey)?;
        let sig = pqcrypto_dilithium::dilithium3::DetachedSignature::from_bytes(signature)
            .map_err(|_| DilithiumError::InvalidKey)?;
        
        Ok(verify_detached_signature(&sig, message, &public_key).is_ok())
    }

    /// Constant sizes
    pub const PUBLIC_KEY_SIZE: usize = public_key_bytes();
    pub const SECRET_KEY_SIZE: usize = secret_key_bytes();
    pub const SIGNATURE_SIZE: usize = signature_bytes();
}
