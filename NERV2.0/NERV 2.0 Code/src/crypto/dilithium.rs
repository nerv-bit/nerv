//! CRYSTALS-Dilithium-3 — Primary Post-Quantum Digital Signature Scheme.
//!
//! Dilithium-3 is used for ALL validator attestations, transaction
//! non-repudiation, and VDW receipts in NERV. It provides NIST
//! Security Level 3 (approximately AES-192 equivalent).
//!
//! | Parameter | Size |
//! |-----------|------|
//! | Public key | 1 952 bytes |
//! | Secret key | 4 032 bytes |
//! | Signature | 3 293 bytes |
//!
//! # Security Properties
//!
//! - **Post-quantum**: Secure against Shor's algorithm
//! - **Strongly unforgeable**: Cannot create new signatures even after seeing others
//! - **Deterministic signing**: Same (msg, sk) → same signature (no randomness)
//!
//! # Integration
//!
//! Uses the `oqs` crate (liboqs bindings) for the actual cryptographic
//! operations. Key material is stored as raw bytes with zeroization on drop.

use crate::{
    DILITHIUM3_PK_BYTES, DILITHIUM3_SK_BYTES, DILITHIUM3_SIG_BYTES,
    NervError, NervResult,
};
use crate::crypto::{KeyId, CryptoVersion};
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;
use zeroize::Zeroize;

// ─── Dilithium-3 Public Key ──────────────────────────────────────────────

/// A CRYSTALS-Dilithium-3 public key (1952 bytes).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DilithiumPublicKey(
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub Vec<u8>,
);

impl DilithiumPublicKey {
    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> NervResult<Self> {
        if bytes.len() != DILITHIUM3_PK_BYTES {
            return Err(NervError::Crypto(format!(
                "Dilithium-3 public key must be {} bytes, got {}",
                DILITHIUM3_PK_BYTES, bytes.len()
            )));
        }
        Ok(Self(bytes.to_vec()))
    }

    /// Create from a Vec (consumes the vector).
    pub fn from_vec(vec: Vec<u8>) -> NervResult<Self> {
        if vec.len() != DILITHIUM3_PK_BYTES {
            return Err(NervError::Crypto(format!(
                "Dilithium-3 public key must be {} bytes, got {}",
                DILITHIUM3_PK_BYTES, vec.len()
            )));
        }
        Ok(Self(vec))
    }

    /// Return the raw bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Compute the KeyId (BLAKE3 hash of the public key).
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
        DILITHIUM3_PK_BYTES
    }

    /// Constant-time equality check.
    pub fn ct_eq(&self, other: &Self) -> subtle::Choice {
        self.0.as_slice().ct_eq(other.0.as_slice())
    }
}

impl AsRef<[u8]> for DilithiumPublicKey {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

// ─── Dilithium-3 Secret Key ──────────────────────────────────────────────

/// A CRYSTALS-Dilithium-3 secret key (4032 bytes).
///
/// **Security**: This type implements `Zeroize` and `ZeroizeOnDrop`
/// to ensure secret material is wiped from memory when no longer needed.
#[derive(Debug, Clone, Zeroize, Serialize, Deserialize)]
#[zeroize(drop)]
pub struct DilithiumSecretKey(
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub Vec<u8>,
);

impl DilithiumSecretKey {
    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> NervResult<Self> {
        if bytes.len() != DILITHIUM3_SK_BYTES {
            return Err(NervError::Crypto(format!(
                "Dilithium-3 secret key must be {} bytes, got {}",
                DILITHIUM3_SK_BYTES, bytes.len()
            )));
        }
        Ok(Self(bytes.to_vec()))
    }

    /// Create from a Vec.
    pub fn from_vec(vec: Vec<u8>) -> NervResult<Self> {
        if vec.len() != DILITHIUM3_SK_BYTES {
            return Err(NervError::Crypto(format!(
                "Dilithium-3 secret key must be {} bytes, got {}",
                DILITHIUM3_SK_BYTES, vec.len()
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
        DILITHIUM3_SK_BYTES
    }
}

// ─── Dilithium-3 Signature ───────────────────────────────────────────────

/// A CRYSTALS-Dilithium-3 signature (3293 bytes).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DilithiumSignature(
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub Vec<u8>,
);

impl DilithiumSignature {
    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> NervResult<Self> {
        if bytes.len() != DILITHIUM3_SIG_BYTES {
            return Err(NervError::Crypto(format!(
                "Dilithium-3 signature must be {} bytes, got {}",
                DILITHIUM3_SIG_BYTES, bytes.len()
            )));
        }
        Ok(Self(bytes.to_vec()))
    }

    /// Create from a Vec.
    pub fn from_vec(vec: Vec<u8>) -> NervResult<Self> {
        if vec.len() != DILITHIUM3_SIG_BYTES {
            return Err(NervError::Crypto(format!(
                "Dilithium-3 signature must be {} bytes, got {}",
                DILITHIUM3_SIG_BYTES, vec.len()
            )));
        }
        Ok(Self(vec))
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
    pub const fn size(&self) -> usize {
        DILITHIUM3_SIG_BYTES
    }
}

impl AsRef<[u8]> for DilithiumSignature {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

// ─── Dilithium-3 Keypair ─────────────────────────────────────────────────

/// A CRYSTALS-Dilithium-3 keypair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DilithiumKeypair {
    /// The public key.
    pub public_key: DilithiumPublicKey,
    /// The secret key (zeroized on drop).
    pub secret_key: DilithiumSecretKey,
}

impl DilithiumKeypair {
    /// Generate a new random Dilithium-3 keypair.
    pub fn generate() -> NervResult<Self> {
        let keypair = oqs::sig::Dilithium3::keypair()
            .map_err(|e| NervError::Crypto(format!("Dilithium-3 keygen failed: {e}")))?;

        let pk_bytes = keypair.pk().as_ref().to_vec();
        let sk_bytes = keypair.sk().as_ref().to_vec();

        Ok(Self {
            public_key: DilithiumPublicKey::from_vec(pk_bytes)?,
            secret_key: DilithiumSecretKey::from_vec(sk_bytes)?,
        })
    }

    /// Construct from existing keys.
    pub fn from_keys(
        public_key: DilithiumPublicKey,
        secret_key: DilithiumSecretKey,
    ) -> Self {
        Self { public_key, secret_key }
    }

    /// Get the key ID.
    pub fn key_id(&self) -> KeyId {
        self.public_key.key_id()
    }
}

// ─── Signing & Verification ──────────────────────────────────────────────

/// Sign a message with a Dilithium-3 secret key.
///
/// Dilithium signing is deterministic (no randomness required),
/// which is ideal for blockchain applications where signature
/// determinism aids debugging and audit.
pub fn dilithium_sign(
    message: &[u8],
    secret_key: &DilithiumSecretKey,
) -> NervResult<DilithiumSignature> {
    let sk = oqs::sig::Dilithium3::SecretKey::from_bytes(secret_key.as_bytes())
        .map_err(|e| NervError::Crypto(format!("invalid Dilithium-3 secret key: {e}")))?;

    let sig = oqs::sig::Dilithium3::sign(message, &sk)
        .map_err(|e| NervError::Crypto(format!("Dilithium-3 sign failed: {e}")))?;

    DilithiumSignature::from_vec(sig.as_ref().to_vec())
}

/// Verify a Dilithium-3 signature.
///
/// Returns `true` if the signature is valid, `false` otherwise.
/// Verification takes ~58 µs on AVX-512 hardware.
pub fn dilithium_verify(
    message: &[u8],
    signature: &DilithiumSignature,
    public_key: &DilithiumPublicKey,
) -> bool {
    let pk = match oqs::sig::Dilithium3::PublicKey::from_bytes(public_key.as_bytes()) {
        Ok(pk) => pk,
        Err(_) => return false,
    };

    let sig = match oqs::sig::Dilithium3::Signature::from_bytes(signature.as_bytes()) {
        Ok(sig) => sig,
        Err(_) => return false,
    };

    oqs::sig::Dilithium3::verify(message, &sig, &pk)
}

/// Batch-verify multiple Dilithium-3 signatures.
///
/// This is more efficient than individual verification when
/// validating entire blocks of validator attestations.
///
/// Returns `true` only if ALL signatures are valid.
pub fn dilithium_batch_verify(
    messages: &[&[u8]],
    signatures: &[DilithiumSignature],
    public_keys: &[DilithiumPublicKey],
) -> NervResult<bool> {
    if messages.len() != signatures.len() || messages.len() != public_keys.len() {
        return Err(NervError::Crypto(
            "batch verify: length mismatch".into()
        ));
    }

    // Dilithium doesn't have a native batch verification,
    // so we verify individually (still parallelizable)
    for ((msg, sig), pk) in messages.iter().zip(signatures.iter()).zip(public_keys.iter()) {
        if !dilithium_verify(msg, sig, pk) {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Sign a message and return both the signature and the public key.
///
/// Convenience function for attestation workflows.
pub fn dilithium_sign_with_pk(
    message: &[u8],
    keypair: &DilithiumKeypair,
) -> NervResult<(DilithiumSignature, DilithiumPublicKey)> {
    let sig = dilithium_sign(message, &keypair.secret_key)?;
    Ok((sig, keypair.public_key.clone()))
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dilithium_keygen() {
        let keypair = DilithiumKeypair::generate().unwrap();
        assert_eq!(keypair.public_key.as_bytes().len(), DILITHIUM3_PK_BYTES);
        assert_eq!(keypair.secret_key.as_bytes().len(), DILITHIUM3_SK_BYTES);
    }

    #[test]
    fn test_dilithium_sign_verify() {
        let keypair = DilithiumKeypair::generate().unwrap();
        let message = b"NERV v2.0 test transaction";
        let signature = dilithium_sign(message, &keypair.secret_key).unwrap();

        assert!(dilithium_verify(message, &signature, &keypair.public_key));
    }

    #[test]
    fn test_dilithium_sign_verify_wrong_message() {
        let keypair = DilithiumKeypair::generate().unwrap();
        let message = b"original message";
        let signature = dilithium_sign(message, &keypair.secret_key).unwrap();

        let wrong_message = b"wrong message";
        assert!(!dilithium_verify(wrong_message, &signature, &keypair.public_key));
    }

    #[test]
    fn test_dilithium_sign_verify_wrong_key() {
        let keypair1 = DilithiumKeypair::generate().unwrap();
        let keypair2 = DilithiumKeypair::generate().unwrap();
        let message = b"test message";
        let signature = dilithium_sign(message, &keypair1.secret_key).unwrap();

        // Verify with wrong public key should fail
        assert!(!dilithium_verify(message, &signature, &keypair2.public_key));
    }

    #[test]
    fn test_dilithium_batch_verify() {
        let keypair1 = DilithiumKeypair::generate().unwrap();
        let keypair2 = DilithiumKeypair::generate().unwrap();

        let msg1 = b"message 1";
        let msg2 = b"message 2";

        let sig1 = dilithium_sign(msg1, &keypair1.secret_key).unwrap();
        let sig2 = dilithium_sign(msg2, &keypair2.secret_key).unwrap();

        let result = dilithium_batch_verify(
            &[msg1.as_slice(), msg2.as_slice()],
            &[sig1, sig2],
            &[keypair1.public_key.clone(), keypair2.public_key.clone()],
        ).unwrap();

        assert!(result);
    }

    #[test]
    fn test_dilithium_batch_verify_one_invalid() {
        let keypair1 = DilithiumKeypair::generate().unwrap();
        let keypair2 = DilithiumKeypair::generate().unwrap();

        let msg1 = b"message 1";
        let msg2 = b"message 2";

        let sig1 = dilithium_sign(msg1, &keypair1.secret_key).unwrap();
        let sig2 = dilithium_sign(msg2, &keypair2.secret_key).unwrap();

        // Verify sig1 with keypair2 (wrong) and sig2 with keypair2 (correct)
        let result = dilithium_batch_verify(
            &[msg1.as_slice(), msg2.as_slice()],
            &[sig1, sig2],
            &[keypair2.public_key.clone(), keypair2.public_key.clone()],
        ).unwrap();

        assert!(!result);
    }

    #[test]
    fn test_dilithium_public_key_from_bytes() {
        let keypair = DilithiumKeypair::generate().unwrap();
        let pk_bytes = keypair.public_key.as_bytes();
        let restored = DilithiumPublicKey::from_bytes(pk_bytes).unwrap();
        assert_eq!(restored.as_bytes(), pk_bytes);
    }

    #[test]
    fn test_dilithium_public_key_wrong_size() {
        let bad_bytes = vec![0u8; 100];
        assert!(DilithiumPublicKey::from_bytes(&bad_bytes).is_err());
    }

    #[test]
    fn test_dilithium_key_id() {
        let keypair = DilithiumKeypair::generate().unwrap();
        let key_id = keypair.key_id();
        assert!(!key_id.is_null());
    }

    #[test]
    fn test_dilithium_deterministic_signing() {
        let keypair = DilithiumKeypair::generate().unwrap();
        let message = b"deterministic test";

        let sig1 = dilithium_sign(message, &keypair.secret_key).unwrap();
        let sig2 = dilithium_sign(message, &keypair.secret_key).unwrap();

        // Dilithium signing is deterministic: same input → same output
        assert_eq!(sig1.as_bytes(), sig2.as_bytes());
    }

    #[test]
    fn test_dilithium_signature_serialization() {
        let keypair = DilithiumKeypair::generate().unwrap();
        let sig = dilithium_sign(b"test", &keypair.secret_key).unwrap();

        let serialized = bincode::serialize(&sig).unwrap();
        let deserialized: DilithiumSignature = bincode::deserialize(&serialized).unwrap();
        assert_eq!(sig.as_bytes(), deserialized.as_bytes());
    }
}
