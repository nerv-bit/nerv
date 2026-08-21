//! Post-Quantum Cryptography Suite.
//!
//! All cryptographic primitives are NIST-standardized lattice-based
//! algorithms from genesis. No ECDSA, no RSA, no curves vulnerable
//! to Shor's algorithm.
//!
//! Submodules:
//! - `dilithium` — CRYSTALS-Dilithium-3 (signatures)
//! - `ml_kem` — ML-KEM-768 (key encapsulation for Sphinx & DKG)
//! - `sphincs` — SPHINCS+-SHA256-192s (stateless fallback signatures)
//! - `hash` — BLAKE3 for embedding roots and general hashing
//!
//! Post-Quantum Cryptography Suite — Zero legacy elliptic curves from genesis.
//!
//! NERV v2.0 uses exclusively NIST-standardized lattice-based cryptography
//! on every critical path. No ECDSA, EdDSA, or secp256k1 anywhere.
//!
//! | Function | Primitive | NIST Level | Size | Speed (AVX-512) |
//! |----------|-----------|------------|------|------------------|
//! | Signatures | CRYSTALS-Dilithium-3 | Level 3 | pk 1952 B, sig 3293 B | ~58 µs verify |
//! | Encryption | ML-KEM-768 | Level 3 | ct 1088 B | ~42 µs decaps |
//! | Cold keys | SPHINCS+-192f-simple | Level 1 | pk 48 B, sig ~16 KB | Slow (hash-only) |
//! | Hashing | BLAKE3 + SHA3-256 | — | 32 B | ~1 ns/byte |
//!
//! # Cryptographic Agility
//!
//! A `CryptoVersion` enum + 180-day governance vote allows future
//! migration (e.g., Dilithium-5, Falcon-1024) without breaking
//! historic verification. Old signatures remain verifiable forever.

pub mod dilithium;
pub mod ml_kem;
pub mod sphincs;
pub mod hash;

// ─── Re-exports ─────────────────────────────────────────────────────────

pub use dilithium::{
    DilithiumPublicKey, DilithiumSecretKey, DilithiumSignature, DilithiumKeypair,
};
pub use ml_kem::{
    MlKemPublicKey, MlKemSecretKey, MlKemCiphertext, MlKemSharedSecret, MlKemKeypair,
};
pub use sphincs::{
    SphincsPublicKey, SphincsSecretKey, SphincsSignature, SphincsKeypair,
};
pub use hash::*;

use crate::{
    DILITHIUM3_PK_BYTES, DILITHIUM3_SK_BYTES, DILITHIUM3_SIG_BYTES,
    ML_KEM768_PK_BYTES, ML_KEM768_SK_BYTES, ML_KEM768_CT_BYTES, ML_KEM768_SS_BYTES,
    SPHINCS_PK_BYTES, SPHINCS_SIG_BYTES,
    NervError, NervResult,
};
use serde::{Deserialize, Serialize};

// ─── Crypto Version (Agility) ────────────────────────────────────────────

/// Cryptographic algorithm version — enables in-place migration.
///
/// Version 0 is the genesis configuration (Dilithium-3 + ML-KEM-768).
/// Future versions are activated via 180-day on-chain governance vote.
/// Old versions remain verifiable forever (forward-compatibility).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CryptoVersion {
    /// Genesis: Dilithium-3 + ML-KEM-768 + SPHINCS+-192f-simple + BLAKE3/SHA3
    V0Genesis,

    /// Placeholder for Dilithium-5 upgrade (NIST Level 5).
    V1Dilithium5,

    /// Placeholder for Falcon-1024 upgrade (compact sigs, Level 1).
    V2Falcon1024,

    /// Placeholder for ML-KEM-1024 upgrade (NIST Level 5 KEM).
    V3MlKem1024,

    /// Placeholder for a future yet-unknown algorithm.
    V4Reserved,
}

impl CryptoVersion {
    /// Current active version.
    pub const CURRENT: Self = Self::V0Genesis;

    /// Governance voting period in blocks (~180 days at 400ms blocks).
    /// 180 * 365.25/2 * 24 * 3600 / 0.4 ≈ 7_095_600 blocks for 6 months
    /// Full 180-day vote: ~14_191_200 blocks
    pub const GOVERNANCE_VOTE_BLOCKS: u64 = 14_191_200;

    /// Required quorum for crypto migration (80% of total stake).
    pub const MIGRATION_QUORUM_BPS: u64 = 8000;

    /// Get the signature algorithm name for this version.
    pub fn sig_algorithm(&self) -> &'static str {
        match self {
            Self::V0Genesis => "Dilithium-3",
            Self::V1Dilithium5 => "Dilithium-5",
            Self::V2Falcon1024 => "Falcon-1024",
            Self::V3MlKem1024 => "Dilithium-3",
            Self::V4Reserved => "Unknown",
        }
    }

    /// Get the KEM algorithm name for this version.
    pub fn kem_algorithm(&self) -> &'static str {
        match self {
            Self::V0Genesis => "ML-KEM-768",
            Self::V1Dilithium5 => "ML-KEM-768",
            Self::V2Falcon1024 => "ML-KEM-768",
            Self::V3MlKem1024 => "ML-KEM-1024",
            Self::V4Reserved => "Unknown",
        }
    }

    /// Get the public key size for the signature algorithm.
    pub fn sig_pk_bytes(&self) -> usize {
        match self {
            Self::V0Genesis => DILITHIUM3_PK_BYTES,
            Self::V1Dilithium5 => 2592,   // Dilithium-5 pk size
            Self::V2Falcon1024 => 1793,    // Falcon-1024 pk size
            Self::V3MlKem1024 => DILITHIUM3_PK_BYTES,
            Self::V4Reserved => 0,
        }
    }

    /// Get the signature size for the signature algorithm.
    pub fn sig_bytes(&self) -> usize {
        match self {
            Self::V0Genesis => DILITHIUM3_SIG_BYTES,
            Self::V1Dilithium5 => 4595,   // Dilithium-5 sig size
            Self::V2Falcon1024 => 1462,   // Falcon-1024 sig size
            Self::V3MlKem1024 => DILITHIUM3_SIG_BYTES,
            Self::V4Reserved => 0,
        }
    }

    /// Get the KEM public key size.
    pub fn kem_pk_bytes(&self) -> usize {
        match self {
            Self::V3MlKem1024 => 1568,    // ML-KEM-1024 pk
            _ => ML_KEM768_PK_BYTES,
        }
    }

    /// Get the KEM ciphertext size.
    pub fn kem_ct_bytes(&self) -> usize {
        match self {
            Self::V3MlKem1024 => 1568,    // ML-KEM-1024 ct
            _ => ML_KEM768_CT_BYTES,
        }
    }

    /// Get the SPHINCS+ variant name for this version.
    pub fn sphincs_variant(&self) -> &'static str {
        match self {
            Self::V0Genesis => "SPHINCS+-SHA256-192f-simple",
            _ => "SPHINCS+-SHA256-192f-simple",
        }
    }
}

impl Default for CryptoVersion {
    fn default() -> Self {
        Self::CURRENT
    }
}

impl std::fmt::Display for CryptoVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::V0Genesis => write!(f, "v0-genesis"),
            Self::V1Dilithium5 => write!(f, "v1-dilithium5"),
            Self::V2Falcon1024 => write!(f, "v2-falcon1024"),
            Self::V3MlKem1024 => write!(f, "v3-mlkem1024"),
            Self::V4Reserved => write!(f, "v4-reserved"),
        }
    }
}

// ─── Key ID ──────────────────────────────────────────────────────────────

/// A unique identifier for a public key, computed as BLAKE3(pk_bytes).
///
/// Key IDs are used throughout NERV as compact, collision-resistant
/// references to public keys (32 bytes vs. 1952 bytes for Dilithium-3).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord,
    Serialize, Deserialize,
)]
pub struct KeyId(
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub [u8; 32],
);

impl KeyId {
    /// Compute a KeyId from raw public key bytes.
    pub fn from_pk_bytes(pk_bytes: &[u8]) -> Self {
        let hash = blake3::hash(pk_bytes);
        Self(hash.into())
    }

    /// The null/zero key ID.
    pub const NULL: Self = Self([0u8; 32]);

    /// Return the raw 32-byte array.
    #[inline]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Convert to hexadecimal.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Check if this is the null key ID.
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0 == [0u8; 32]
    }
}

impl std::fmt::Display for KeyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.to_hex()[..16]) // Short form: first 16 hex chars
    }
}

impl AsRef<[u8]> for KeyId {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

// ─── Crypto Suite ────────────────────────────────────────────────────────

/// The active cryptographic suite configuration.
///
/// This is determined by the current `CryptoVersion` and provides
/// size parameters for all algorithms. It is loaded at node startup
/// and updated only via governance vote.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoSuite {
    /// Current algorithm version.
    pub version: CryptoVersion,

    /// Whether this node has voted for a pending migration.
    pub migration_vote: Option<CryptoVersion>,

    /// Blocks remaining in the current migration vote (if active).
    pub migration_blocks_remaining: u64,
}

impl CryptoSuite {
    /// Create the genesis crypto suite.
    pub fn genesis() -> Self {
        Self {
            version: CryptoVersion::V0Genesis,
            migration_vote: None,
            migration_blocks_remaining: 0,
        }
    }

    /// Get the current version.
    pub fn version(&self) -> CryptoVersion {
        self.version
    }

    /// Cast a vote for a crypto migration.
    pub fn vote_for_migration(&mut self, target: CryptoVersion) -> NervResult<()> {
        if target == self.version {
            return Err(NervError::Crypto("cannot vote for current version".into()));
        }
        self.migration_vote = Some(target);
        Ok(())
    }

    /// Check if a migration has completed (quorum reached + voting period elapsed).
    pub fn check_migration_complete(
        &mut self,
        vote_weight_bps: u64,
        blocks_elapsed: u64,
    ) -> bool {
        if let Some(target) = self.migration_vote {
            if vote_weight_bps >= CryptoVersion::MIGRATION_QUORUM_BPS
                && blocks_elapsed >= CryptoVersion::GOVERNANCE_VOTE_BLOCKS
            {
                self.version = target;
                self.migration_vote = None;
                self.migration_blocks_remaining = 0;
                return true;
            }
        }
        false
    }
}

impl Default for CryptoSuite {
    fn default() -> Self {
        Self::genesis()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crypto_version_current() {
        assert_eq!(CryptoVersion::CURRENT, CryptoVersion::V0Genesis);
    }

    #[test]
    fn test_crypto_version_algorithms() {
        let v0 = CryptoVersion::V0Genesis;
        assert_eq!(v0.sig_algorithm(), "Dilithium-3");
        assert_eq!(v0.kem_algorithm(), "ML-KEM-768");
        assert_eq!(v0.sphincs_variant(), "SPHINCS+-SHA256-192f-simple");
    }

    #[test]
    fn test_crypto_version_sizes() {
        let v0 = CryptoVersion::V0Genesis;
        assert_eq!(v0.sig_pk_bytes(), 1952);
        assert_eq!(v0.sig_bytes(), 3293);
        assert_eq!(v0.kem_pk_bytes(), 1184);
        assert_eq!(v0.kem_ct_bytes(), 1088);
    }

    #[test]
    fn test_crypto_version_display() {
        assert_eq!(CryptoVersion::V0Genesis.to_string(), "v0-genesis");
        assert_eq!(CryptoVersion::V1Dilithium5.to_string(), "v1-dilithium5");
    }

    #[test]
    fn test_key_id_from_pk() {
        let pk = vec![1u8; 1952];
        let id1 = KeyId::from_pk_bytes(&pk);
        let id2 = KeyId::from_pk_bytes(&pk);
        assert_eq!(id1, id2);
        assert!(!id1.is_null());
    }

    #[test]
    fn test_key_id_different_pks() {
        let pk1 = vec![1u8; 1952];
        let pk2 = vec![2u8; 1952];
        let id1 = KeyId::from_pk_bytes(&pk1);
        let id2 = KeyId::from_pk_bytes(&pk2);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_crypto_suite_genesis() {
        let suite = CryptoSuite::genesis();
        assert_eq!(suite.version(), CryptoVersion::V0Genesis);
    }

    #[test]
    fn test_crypto_suite_migration_vote() {
        let mut suite = CryptoSuite::genesis();
        assert!(suite.vote_for_migration(CryptoVersion::V1Dilithium5).is_ok());
        assert!(suite.vote_for_migration(CryptoVersion::V0Genesis).is_err()); // Same as current
    }

    #[test]
    fn test_key_id_null() {
        assert!(KeyId::NULL.is_null());
    }

    #[test]
    fn test_key_id_display() {
        let pk = vec![42u8; 1952];
        let id = KeyId::from_pk_bytes(&pk);
        let display = id.to_string();
        assert_eq!(display.len(), 16); // Short form
    }
}

