//! SPHINCS+-SHA256-192f-simple — Stateless Hash-Based Signature Scheme.
//!
//! SPHINCS+ is used exclusively for **cold/genesis keys** — keys that
//! are used very rarely (e.g., genesis block signing, emergency
//! governance actions) but must remain secure forever without state.
//!
//! | Parameter | Size |
//! |-----------|------|
//! | Public key | 48 bytes |
//! | Secret key | 96 bytes |
//! | Signature | ~16 224 bytes |
//!
//! # Why SPHINCS+ for Cold Keys?
//!
//! - **Stateless**: No need to track which signatures have been used
//! - **Conservative**: Pure hash-based security (no lattice assumptions)
//! - **Small keys**: 48-byte public keys fit easily on paper/metal backups
//! - **Post-quantum**: Secure against Shor's algorithm
//! - **Trade-off**: Large signatures (~16 KB), but acceptable for rare use
//!
//! # Warning
//!
//! SPHINCS+ signatures are ~5× larger than Dilithium-3. Do NOT use
//! for high-frequency operations (validator attestations, transactions).
//! Use Dilithium-3 for those.

use crate::{
    SPHINCS_PK_BYTES, SPHINCS_SIG_BYTES,
    NervError, NervResult,
};
use crate::crypto::KeyId;
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;
use zeroize::Zeroize;

// ─── SPHINCS+ Public Key ─────────────────────────────────────────────────

/// A SPHINCS+-SHA256-192f-simple public key (48 bytes).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SphincsPublicKey(
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub Vec<u8>,
);

impl SphincsPublicKey {
    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> NervResult<Self> {
        if bytes.len() != SPHINCS_PK_BYTES {
            return Err(NervError::Crypto(format!(
                "SPHINCS+ public key must be {} bytes, got {}",
                SPHINCS_PK_BYTES, bytes.len()
            )));
        }
        Ok(Self(bytes.to_vec()))
    }

    /// Create from a Vec.
    pub fn from_vec(vec: Vec<u8>) -> NervResult<Self> {
        if vec.len() != SPHINCS_PK_BYTES {
            return Err(NervError::Crypto(format!(
                "SPHINCS+ public key must be {} bytes, got {}",
                SPHINCS_PK_BYTES, vec.len()
            )));
        }
        Ok(Self(vec))
    }

    /// Return the raw bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Compute the KeyId.
    pub fn key_id(&self) -> KeyId {
        KeyId::from_pk_bytes(&self.0)
    }

    /// Convert to hexadecimal.
    pub fn to_hex(&self) -> String {
        hex::encode(&self.0)
    }

    /// Size in bytes.
    #[inline]
    pub const fn size(&self) -> usize {
        SPHINCS_PK_BYTES
    }
}

impl AsRef<[u8]> for SphincsPublicKey {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

// ─── SPHINCS+ Secret Key ─────────────────────────────────────────────────

/// A SPHINCS+-SHA256-192f-simple secret key.
///
/// **Security**: Zeroized on drop. This key is typically stored
/// offline (hardware wallet, paper backup, HSM).
#[derive(Debug, Clone, Zeroize, Serialize, Deserialize)]
#[zeroize(drop)]
pub struct SphincsSecretKey(
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub Vec<u8>,
);

/// SPHINCS+ secret key size (96 bytes for 192f-simple).
const SPHINCS_SK_SIZE: usize = 96;

impl SphincsSecretKey {
    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> NervResult<Self> {
        if bytes.len() != SPHINCS_SK_SIZE {
            return Err(NervError::Crypto(format!(
                "SPHINCS+ secret key must be {} bytes, got {}",
                SPHINCS_SK_SIZE, bytes.len()
            )));
        }
        Ok(Self(bytes.to_vec()))
    }

    /// Create from a Vec.
    pub fn from_vec(vec: Vec<u8>) -> NervResult<Self> {
        if vec.len() != SPHINCS_SK_SIZE {
            return Err(NervError::Crypto(format!(
                "SPHINCS+ secret key must be {} bytes, got {}",
                SPHINCS_SK_SIZE, vec.len()
            )));
        }
        Ok(Self(vec))
    }

    /// Return the raw bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Size in bytes.
    #[inline]
    pub const fn size(&self) -> usize {
        SPHINCS_SK_SIZE
    }
}

// ─── SPHINCS+ Signature ──────────────────────────────────────────────────

/// A SPHINCS+-SHA256-192f-simple signature.
///
/// **Note**: The actual OQS signature size may differ slightly from
/// `SPHINCS_SIG_BYTES` depending on the exact parameter set. This
/// type accepts signatures up to the maximum expected size.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SphincsSignature(
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub Vec<u8>,
);

impl SphincsSignature {
    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> NervResult<Self> {
        if bytes.len() > SPHINCS_SIG_BYTES * 2 {
            return Err(NervError::Crypto(format!(
                "SPHINCS+ signature too large: {} bytes (max {})",
                bytes.len(), SPHINCS_SIG_BYTES * 2
            )));
        }
        if bytes.is_empty() {
            return Err(NervError::Crypto("SPHINCS+ signature is empty".into()));
        }
        Ok(Self(bytes.to_vec()))
    }

    /// Create from a Vec.
    pub fn from_vec(vec: Vec<u8>) -> NervResult<Self> {
        Self::from_bytes(&vec)
    }

    /// Return the raw bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Convert to hexadecimal.
    pub fn to_hex(&self) -> String {
        hex::encode(&self.0)
    }

    /// Size in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.0.len()
    }
}

impl AsRef<[u8]> for SphincsSignature {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

// ─── SPHINCS+ Keypair ────────────────────────────────────────────────────

/// A SPHINCS+-SHA256-192f-simple keypair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphincsKeypair {
    /// The public key.
    pub public_key: SphincsPublicKey,
    /// The secret key (zeroized on drop).
    pub secret_key: SphincsSecretKey,
}

impl SphincsKeypair {
    /// Generate a new random SPHINCS+ keypair.
    pub fn generate() -> NervResult<Self> {
        let keypair = oqs::sig::SphincsSha256192fSimple::keypair()
            .map_err(|e| NervError::Crypto(format!("SPHINCS+ keygen failed: {e}")))?;

        let pk_bytes = keypair.pk().as_ref().to_vec();
        let sk_bytes = keypair.sk().as_ref().to_vec();

        Ok(Self {
            public_key: SphincsPublicKey::from_vec(pk_bytes)?,
            secret_key: SphincsSecretKey::from_vec(sk_bytes)?,
        })
    }

    /// Construct from existing keys.
    pub fn from_keys(
        public_key: SphincsPublicKey,
        secret_key: SphincsSecretKey,
    ) -> Self {
        Self { public_key, secret_key }
    }

    /// Get the key ID.
    pub fn key_id(&self) -> KeyId {
        self.public_key.key_id()
    }
}

// ─── Signing & Verification ──────────────────────────────────────────────

/// Sign a message with a SPHINCS+ secret key.
///
/// **Warning**: SPHINCS+ signatures are ~16 KB. Use only for
/// cold/genesis keys. For all other signing, use Dilithium-3.
pub fn sphincs_sign(
    message: &[u8],
    secret_key: &SphincsSecretKey,
) -> NervResult<SphincsSignature> {
    let sk = oqs::sig::SphincsSha256192fSimple::SecretKey::from_bytes(secret_key.as_bytes())
        .map_err(|e| NervError::Crypto(format!("invalid SPHINCS+ secret key: {e}")))?;

    let sig = oqs::sig::SphincsSha256192fSimple::sign(message, &sk)
        .map_err(|e| NervError::Crypto(format!("SPHINCS+ sign failed: {e}")))?;

    SphincsSignature::from_vec(sig.as_ref().to_vec())
}

/// Verify a SPHINCS+ signature.
pub fn sphincs_verify(
    message: &[u8],
    signature: &SphincsSignature,
    public_key: &SphincsPublicKey,
) -> bool {
    let pk = match oqs::sig::SphincsSha256192fSimple::PublicKey::from_bytes(public_key.as_bytes()) {
        Ok(pk) => pk,
        Err(_) => return false,
    };

    let sig = match oqs::sig::SphincsSha256192fSimple::Signature::from_bytes(signature.as_bytes()) {
        Ok(sig) => sig,
        Err(_) => return false,
    };

    oqs::sig::SphincsSha256192fSimple::verify(message, &sig, &pk)
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphincs_keygen() {
        let keypair = SphincsKeypair::generate().unwrap();
        assert_eq!(keypair.public_key.as_bytes().len(), SPHINCS_PK_BYTES);
        assert_eq!(keypair.secret_key.as_bytes().len(), SPHINCS_SK_SIZE);
    }

    #[test]
    fn test_sphincs_sign_verify() {
        let keypair = SphincsKeypair::generate().unwrap();
        let message = b"NERV genesis block signature";
        let signature = sphincs_sign(message, &keypair.secret_key).unwrap();

        assert!(sphincs_verify(message, &signature, &keypair.public_key));
    }

    #[test]
    fn test_sphincs_sign_verify_wrong_message() {
        let keypair = SphincsKeypair::generate().unwrap();
        let message = b"original";
        let signature = sphincs_sign(message, &keypair.secret_key).unwrap();

        assert!(!sphincs_verify(b"wrong", &signature, &keypair.public_key));
    }

    #[test]
    fn test_sphincs_sign_verify_wrong_key() {
        let keypair1 = SphincsKeypair::generate().unwrap();
        let keypair2 = SphincsKeypair::generate().unwrap();
        let message = b"test";
        let signature = sphincs_sign(message, &keypair1.secret_key).unwrap();

        assert!(!sphincs_verify(message, &signature, &keypair2.public_key));
    }

    #[test]
    fn test_sphincs_signature_size() {
        let keypair = SphincsKeypair::generate().unwrap();
        let signature = sphincs_sign(b"test", &keypair.secret_key).unwrap();
        // SPHINCS+ signatures should be large (thousands of bytes)
        assert!(signature.size() > 1000);
    }

    #[test]
    fn test_sphincs_public_key_small() {
        let keypair = SphincsKeypair::generate().unwrap();
        // SPHINCS+ public keys are very small (48 bytes)
        assert_eq!(keypair.public_key.size(), 48);
    }

    #[test]
    fn test_sphincs_key_id() {
        let keypair = SphincsKeypair::generate().unwrap();
        let key_id = keypair.key_id();
        assert!(!key_id.is_null());
    }

    #[test]
    fn test_sphincs_public_key_from_bytes() {
        let keypair = SphincsKeypair::generate().unwrap();
        let pk_bytes = keypair.public_key.as_bytes();
        let restored = SphincsPublicKey::from_bytes(pk_bytes).unwrap();
        assert_eq!(restored.as_bytes(), pk_bytes);
    }

    #[test]
    fn test_sphincs_public_key_wrong_size() {
        let bad_bytes = vec![0u8; 50];
        assert!(SphincsPublicKey::from_bytes(&bad_bytes).is_err());
    }
}
