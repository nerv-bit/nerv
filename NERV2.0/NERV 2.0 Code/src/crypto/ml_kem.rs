//! ML-KEM-768 — Post-Quantum Key Encapsulation Mechanism.
//!
//! ML-KEM-768 (formerly Kyber-768) is used for ALL encryption in NERV:
//! - 5-hop Sphinx onion routing (each hop uses ML-KEM-768 encapsulation)
//! - DKG share distribution (encrypted shares between validators)
//! - Client-to-validator encrypted transaction submission
//! - Threshold mempool encryption (combined with DKG)
//!
//! | Parameter | Size |
//! |-----------|------|
//! | Public key | 1 184 bytes |
//! | Secret key | 2 400 bytes |
//! | Ciphertext | 1 088 bytes |
//! | Shared secret | 32 bytes |
//!
//! # IND-CCA2 Security
//!
//! ML-KEM-768 provides IND-CCA2 security (chosen-ciphertext attack
//! resistance) in the quantum random oracle model. This is the
//! strongest security notion for a KEM.

use crate::{
    ML_KEM768_PK_BYTES, ML_KEM768_SK_BYTES, ML_KEM768_CT_BYTES, ML_KEM768_SS_BYTES,
    NervError, NervResult,
};
use crate::crypto::KeyId;
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;
use zeroize::Zeroize;

// ─── ML-KEM-768 Public Key ───────────────────────────────────────────────

/// An ML-KEM-768 public key (1184 bytes).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlKemPublicKey(
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub Vec<u8>,
);

impl MlKemPublicKey {
    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> NervResult<Self> {
        if bytes.len() != ML_KEM768_PK_BYTES {
            return Err(NervError::Crypto(format!(
                "ML-KEM-768 public key must be {} bytes, got {}",
                ML_KEM768_PK_BYTES, bytes.len()
            )));
        }
        Ok(Self(bytes.to_vec()))
    }

    /// Create from a Vec.
    pub fn from_vec(vec: Vec<u8>) -> NervResult<Self> {
        if vec.len() != ML_KEM768_PK_BYTES {
            return Err(NervError::Crypto(format!(
                "ML-KEM-768 public key must be {} bytes, got {}",
                ML_KEM768_PK_BYTES, vec.len()
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
        ML_KEM768_PK_BYTES
    }

    /// Constant-time equality check.
    pub fn ct_eq(&self, other: &Self) -> subtle::Choice {
        self.0.as_slice().ct_eq(other.0.as_slice())
    }
}

impl AsRef<[u8]> for MlKemPublicKey {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

// ─── ML-KEM-768 Secret Key ───────────────────────────────────────────────

/// An ML-KEM-768 secret key (2400 bytes).
///
/// **Security**: Zeroized on drop.
#[derive(Debug, Clone, Zeroize, Serialize, Deserialize)]
#[zeroize(drop)]
pub struct MlKemSecretKey(
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub Vec<u8>,
);

impl MlKemSecretKey {
    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> NervResult<Self> {
        if bytes.len() != ML_KEM768_SK_BYTES {
            return Err(NervError::Crypto(format!(
                "ML-KEM-768 secret key must be {} bytes, got {}",
                ML_KEM768_SK_BYTES, bytes.len()
            )));
        }
        Ok(Self(bytes.to_vec()))
    }

    /// Create from a Vec.
    pub fn from_vec(vec: Vec<u8>) -> NervResult<Self> {
        if vec.len() != ML_KEM768_SK_BYTES {
            return Err(NervError::Crypto(format!(
                "ML-KEM-768 secret key must be {} bytes, got {}",
                ML_KEM768_SK_BYTES, vec.len()
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
        ML_KEM768_SK_BYTES
    }
}

// ─── ML-KEM-768 Ciphertext ───────────────────────────────────────────────

/// An ML-KEM-768 ciphertext (1088 bytes).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlKemCiphertext(
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub Vec<u8>,
);

impl MlKemCiphertext {
    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> NervResult<Self> {
        if bytes.len() != ML_KEM768_CT_BYTES {
            return Err(NervError::Crypto(format!(
                "ML-KEM-768 ciphertext must be {} bytes, got {}",
                ML_KEM768_CT_BYTES, bytes.len()
            )));
        }
        Ok(Self(bytes.to_vec()))
    }

    /// Create from a Vec.
    pub fn from_vec(vec: Vec<u8>) -> NervResult<Self> {
        if vec.len() != ML_KEM768_CT_BYTES {
            return Err(NervError::Crypto(format!(
                "ML-KEM-768 ciphertext must be {} bytes, got {}",
                ML_KEM768_CT_BYTES, vec.len()
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
        ML_KEM768_CT_BYTES
    }

    /// BLAKE3 hash for integrity checking.
    pub fn hash(&self) -> [u8; 32] {
        blake3::hash(&self.0).into()
    }
}

impl AsRef<[u8]> for MlKemCiphertext {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

// ─── ML-KEM-768 Shared Secret ────────────────────────────────────────────

/// An ML-KEM-768 shared secret (32 bytes).
///
/// **Security**: Zeroized on drop. This is the most sensitive
/// cryptographic material — it protects all encrypted communications.
#[derive(Debug, Clone, Zeroize, PartialEq, Eq, Serialize, Deserialize)]
#[zeroize(drop)]
pub struct MlKemSharedSecret(
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub Vec<u8>,
);

impl MlKemSharedSecret {
    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> NervResult<Self> {
        if bytes.len() != ML_KEM768_SS_BYTES {
            return Err(NervError::Crypto(format!(
                "ML-KEM-768 shared secret must be {} bytes, got {}",
                ML_KEM768_SS_BYTES, bytes.len()
            )));
        }
        Ok(Self(bytes.to_vec()))
    }

    /// Create from a Vec.
    pub fn from_vec(vec: Vec<u8>) -> NervResult<Self> {
        if vec.len() != ML_KEM768_SS_BYTES {
            return Err(NervError::Crypto(format!(
                "ML-KEM-768 shared secret must be {} bytes, got {}",
                ML_KEM768_SS_BYTES, vec.len()
            )));
        }
        Ok(Self(vec))
    }

    /// Return the raw bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Return as a fixed-size array.
    pub fn as_array(&self) -> &[u8; 32] {
        self.0.as_slice().try_into().unwrap_or(&[0u8; 32])
    }

    /// Size in bytes.
    #[inline]
    pub const fn size(&self) -> usize {
        ML_KEM768_SS_BYTES
    }

    /// Constant-time equality.
    pub fn ct_eq(&self, other: &Self) -> subtle::Choice {
        self.0.as_slice().ct_eq(other.0.as_slice())
    }
}

// ─── ML-KEM-768 Keypair ──────────────────────────────────────────────────

/// An ML-KEM-768 keypair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlKemKeypair {
    /// The public key.
    pub public_key: MlKemPublicKey,
    /// The secret key (zeroized on drop).
    pub secret_key: MlKemSecretKey,
}

impl MlKemKeypair {
    /// Generate a new random ML-KEM-768 keypair.
    pub fn generate() -> NervResult<Self> {
        let keypair = oqs::kem::MlKem768::keypair()
            .map_err(|e| NervError::Crypto(format!("ML-KEM-768 keygen failed: {e}")))?;

        let pk_bytes = keypair.pk().as_ref().to_vec();
        let sk_bytes = keypair.sk().as_ref().to_vec();

        Ok(Self {
            public_key: MlKemPublicKey::from_vec(pk_bytes)?,
            secret_key: MlKemSecretKey::from_vec(sk_bytes)?,
        })
    }

    /// Construct from existing keys.
    pub fn from_keys(
        public_key: MlKemPublicKey,
        secret_key: MlKemSecretKey,
    ) -> Self {
        Self { public_key, secret_key }
    }

    /// Get the key ID.
    pub fn key_id(&self) -> KeyId {
        self.public_key.key_id()
    }
}

// ─── Encapsulation & Decapsulation ───────────────────────────────────────

/// Encapsulate: generate a shared secret and ciphertext for a public key.
///
/// This is the KEM equivalent of "encrypting a key". The resulting
/// ciphertext can only be decapsulated by the holder of the
/// corresponding secret key.
///
/// Returns `(ciphertext, shared_secret)`.
pub fn ml_kem_encapsulate(
    public_key: &MlKemPublicKey,
) -> NervResult<(MlKemCiphertext, MlKemSharedSecret)> {
    let pk = oqs::kem::MlKem768::PublicKey::from_bytes(public_key.as_bytes())
        .map_err(|e| NervError::Crypto(format!("invalid ML-KEM-768 public key: {e}")))?;

    let (ct, ss) = oqs::kem::MlKem768::encapsulate(&pk)
        .map_err(|e| NervError::Crypto(format!("ML-KEM-768 encapsulate failed: {e}")))?;

    let ciphertext = MlKemCiphertext::from_vec(ct.as_ref().to_vec())?;
    let shared_secret = MlKemSharedSecret::from_vec(ss.as_ref().to_vec())?;

    Ok((ciphertext, shared_secret))
}

/// Decapsulate: recover the shared secret from a ciphertext.
///
/// Only the holder of the secret key can successfully decapsulate.
/// Returns the 32-byte shared secret.
pub fn ml_kem_decapsulate(
    ciphertext: &MlKemCiphertext,
    secret_key: &MlKemSecretKey,
) -> NervResult<MlKemSharedSecret> {
    let ct = oqs::kem::MlKem768::Ciphertext::from_bytes(ciphertext.as_bytes())
        .map_err(|e| NervError::Crypto(format!("invalid ML-KEM-768 ciphertext: {e}")))?;

    let sk = oqs::kem::MlKem768::SecretKey::from_bytes(secret_key.as_bytes())
        .map_err(|e| NervError::Crypto(format!("invalid ML-KEM-768 secret key: {e}")))?;

    let ss = oqs::kem::MlKem768::decapsulate(&ct, &sk)
        .map_err(|e| NervError::Crypto(format!("ML-KEM-768 decapsulate failed: {e}")))?;

    MlKemSharedSecret::from_vec(ss.as_ref().to_vec())
}

/// Full KEM round-trip: generate keypair, encapsulate, decapsulate.
///
/// Returns `(keypair, ciphertext, shared_secret)`.
/// Useful for Sphinx hop key derivation.
pub fn ml_kem_round_trip() -> NervResult<(
    MlKemKeypair,
    MlKemCiphertext,
    MlKemSharedSecret,
)> {
    let keypair = MlKemKeypair::generate()?;
    let (ciphertext, ss_enc) = ml_kem_encapsulate(&keypair.public_key)?;
    let ss_dec = ml_kem_decapsulate(&ciphertext, &keypair.secret_key)?;

    // Verify the shared secrets match (constant-time)
    if ss_enc.as_bytes().ct_eq(ss_dec.as_bytes()).into() {
        Ok((keypair, ciphertext, ss_enc))
    } else {
        Err(NervError::Crypto(
            "ML-KEM-768 round-trip: shared secret mismatch".into()
        ))
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_kem_keygen() {
        let keypair = MlKemKeypair::generate().unwrap();
        assert_eq!(keypair.public_key.as_bytes().len(), ML_KEM768_PK_BYTES);
        assert_eq!(keypair.secret_key.as_bytes().len(), ML_KEM768_SK_BYTES);
    }

    #[test]
    fn test_ml_kem_encapsulate_decapsulate() {
        let keypair = MlKemKeypair::generate().unwrap();
        let (ciphertext, ss_enc) = ml_kem_encapsulate(&keypair.public_key).unwrap();
        let ss_dec = ml_kem_decapsulate(&ciphertext, &keypair.secret_key).unwrap();

        // Shared secrets must match
        assert_eq!(ss_enc.as_bytes(), ss_dec.as_bytes());
    }

    #[test]
    fn test_ml_kem_ciphertext_size() {
        let keypair = MlKemKeypair::generate().unwrap();
        let (ciphertext, _) = ml_kem_encapsulate(&keypair.public_key).unwrap();
        assert_eq!(ciphertext.as_bytes().len(), ML_KEM768_CT_BYTES);
    }

    #[test]
    fn test_ml_kem_shared_secret_size() {
        let keypair = MlKemKeypair::generate().unwrap();
        let (_, ss) = ml_kem_encapsulate(&keypair.public_key).unwrap();
        assert_eq!(ss.as_bytes().len(), ML_KEM768_SS_BYTES);
    }

    #[test]
    fn test_ml_kem_wrong_key_decapsulate() {
        let keypair1 = MlKemKeypair::generate().unwrap();
        let keypair2 = MlKemKeypair::generate().unwrap();

        let (ciphertext, ss_enc) = ml_kem_encapsulate(&keypair1.public_key).unwrap();

        // Decapsulate with wrong secret key
        let ss_dec = ml_kem_decapsulate(&ciphertext, &keypair2.secret_key).unwrap();

        // Shared secrets must NOT match (IND-CCA2 security)
        assert_ne!(ss_enc.as_bytes(), ss_dec.as_bytes());
    }

    #[test]
    fn test_ml_kem_round_trip() {
        let result = ml_kem_round_trip();
        assert!(result.is_ok());
    }

    #[test]
    fn test_ml_kem_key_id() {
        let keypair = MlKemKeypair::generate().unwrap();
        let key_id = keypair.key_id();
        assert!(!key_id.is_null());
    }

    #[test]
    fn test_ml_kem_public_key_from_bytes() {
        let keypair = MlKemKeypair::generate().unwrap();
        let pk_bytes = keypair.public_key.as_bytes();
        let restored = MlKemPublicKey::from_bytes(pk_bytes).unwrap();
        assert_eq!(restored.as_bytes(), pk_bytes);
    }

    #[test]
    fn test_ml_kem_public_key_wrong_size() {
        let bad_bytes = vec![0u8; 100];
        assert!(MlKemPublicKey::from_bytes(&bad_bytes).is_err());
    }

    #[test]
    fn test_ml_kem_ciphertext_hash() {
        let keypair = MlKemKeypair::generate().unwrap();
        let (ct, _) = ml_kem_encapsulate(&keypair.public_key).unwrap();
        let h1 = ct.hash();
        let h2 = ct.hash();
        assert_eq!(h1, h2);
    }

   C    #[test]
    fn test_ml_kem_different_encapsulations() {
        let keypair = MlKemKeypair::generate().unwrap();
        let (ct1, ss1) = ml_kem_encapsulate(&keypair.public_key).unwrap();
        let (ct2, ss2) = ml_kem_encapsulate(&keypair.public_key).unwrap();

        // Different encapsulations should produce different ciphertexts
        // and different shared secrets (with overwhelming probability)
        assert_ne!(ct1.as_bytes(), ct2.as_bytes());
        assert_ne!(ss1.as_bytes(), ss2.as_bytes());
    }
}

