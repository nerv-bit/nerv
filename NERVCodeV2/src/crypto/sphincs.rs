//! SPHINCS+-SHA256-192s-simple using audited pqcrypto crate
//!
//! This provides NIST-standardized post-quantum stateless hash-based signatures (FIPS 205).
//! Used as backup/fallback signature scheme in NERV (primary is Dilithium-3).
//! Parameter set: SHA256-192s-simple
//!   - Security: Level 3 (~192-bit classical, ~96-bit quantum)
//!   - Signature size: ~16 KB (17088 bytes)
//!   - Public key size: 48 bytes
//!   - Secret key size: 128 bytes
//!   - Fully deterministic (simple variant) – ideal for blockchain reproducibility
//!   - Stateless – no signing nonce management needed

use pqcrypto_sphincsplus::sphincsplussha256192ssimple::{
    keypair, sign, verify,
    public_key_bytes, secret_key_bytes, signature_bytes
};
use pqcrypto_traits::sign::{PublicKey, SecretKey, SignedMessage};
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct SphincsPlus;

#[derive(Debug, Error)]
pub enum SphincsError {
    #[error("Signature verification failed")]
    VerificationFailed,
    #[error("Invalid public key length")]
    InvalidPublicKey,
    #[error("Invalid secret key length")]
    InvalidSecretKey,
    #[error("Invalid signature length")]
    InvalidSignature,
}

impl SphincsPlus {
    /// Generate a new keypair (deterministic – no RNG needed for simple variant)
    pub fn keypair() -> (Vec<u8>, Vec<u8>) { // (pk, sk)
        let (pk, sk) = keypair();
        (pk.as_bytes().to_vec(), sk.as_bytes().to_vec())
    }

    /// Sign a message (fully deterministic for simple variant)
    pub fn sign(sk: &[u8], message: &[u8]) -> Result<Vec<u8>, SphincsError> {
        let secret_key = pqcrypto_sphincsplus::sphincsplussha256192ssimple::SecretKey::from_bytes(sk)
            .map_err(|_| SphincsError::InvalidSecretKey)?;
        let signed = sign(message, &secret_key);
        Ok(signed.as_bytes().to_vec())
    }

    /// Verify a signature (returns true if valid)
    pub fn verify(pk: &[u8], message: &[u8], signature: &[u8]) -> Result<bool, SphincsError> {
        let public_key = pqcrypto_sphincsplus::sphincsplussha256192ssimple::PublicKey::from_bytes(pk)
            .map_err(|_| SphincsError::InvalidPublicKey)?;
        
        // SignedMessage = signature || message
        let mut signed_message = Vec::with_capacity(signature.len() + message.len());
        signed_message.extend_from_slice(signature);
        signed_message.extend_from_slice(message);
        
        let signed = pqcrypto_sphincsplus::sphincsplussha256192ssimple::SignedMessage::from_bytes(&signed_message)
            .map_err(|_| SphincsError::InvalidSignature)?;
        
        Ok(verify(&signed, &public_key).is_ok())
    }

    /// Constant sizes (exact for SHA256-192s-simple)
    pub const PUBLIC_KEY_SIZE: usize = public_key_bytes();   // 48
    pub const SECRET_KEY_SIZE: usize = secret_key_bytes();   // 128
    pub const SIGNATURE_SIZE: usize = signature_bytes();     // 17088
}
