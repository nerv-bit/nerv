//! Distributed Key Generation (DKG) for Threshold Mempool Encryption.
//!
//! The DKG protocol allows N validators to jointly generate a public
//! encryption key and corresponding secret key shares, such that:
//!
//! - **No single validator** knows the full secret key
//! - **Any t out of N** validators can collaboratively decrypt
//! - **Fewer than t** validators cannot learn anything about the key
//!
//! # Protocol (Feldman VSS-based DKG)
//!
//! ```text
//! Phase 1 — Share Distribution:
//!   Each validator i:
//!     1. Generates random polynomial f_i(x) of degree t-1
//!     2. Computes shares s_ij = f_i(j) for all j
//!     3. Broadcasts Feldman commitments C_k = g^(a_k) for each coefficient
//!     4. Sends encrypted share s_ij to validator j (via ML-KEM-768)
//!
//! Phase 2 — Share Verification & Aggregation:
//!   Each validator j:
//!     1. Receives shares s_ij from all i
//!     2. Verifies each share against Feldman commitments
//!     3. Computes final share: s_j = Σ_i s_ij
//!     4. Broadcasts confirmation
//!
//! Phase 3 — Public Key Computation:
//!   All validators compute: PK = Σ_i C_i[0] = Σ_i g^(f_i(0))
//!   This is the collective public key — no single party knows the
//!   corresponding secret key s = Σ_i f_i(0).
//! ```
//!
//! # Security Properties
//!
//! - **Threshold security**: Any t shares can decrypt; t-1 cannot
//! - **No trusted dealer**: The secret is never known to any single party
//! - **Verifiable shares**: Feldman commitments allow public verification
//! - **Post-quantum transport**: Share delivery uses ML-KEM-768 encryption

use crate::{
    EMBEDDING_DIM, ML_KEM768_PK_BYTES, ML_KEM768_CT_BYTES, ML_KEM768_SS_BYTES,
    NervError, NervResult, ValidatorId,
};
use crate::utils::{blake3_hash, random_bytes};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── DKG Constants ───────────────────────────────────────────────────────

/// Minimum number of DKG participants.
pub const DKG_MIN_PARTICIPANTS: usize = 3;

/// Default threshold (2/3 of participants for BFT safety).
pub const DKG_DEFAULT_THRESHOLD_FRACTION: f64 = 2.0 / 3.0;

/// Maximum number of DKG participants.
pub const DKG_MAX_PARTICIPANTS: usize = 256;

/// Polynomial coefficient representation size (32 bytes for BLS12-381 scalar).
pub const SCALAR_BYTES: usize = 32;

/// G1 point compressed size (48 bytes for BLS12-381).
pub const G1_COMPRESSED_BYTES: usize = 48;

// ─── Scalar Field Arithmetic Helpers ──────────────────────────────────────

/// A wrapper around a 256-bit scalar value for the DKG polynomial.
///
/// In production, this maps to `bls12_381::Scalar`. For portability
/// and to avoid direct dependency on the curve library in all contexts,
/// we store the scalar as raw bytes and convert as needed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DkgScalar(
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub Vec<u8>,
);

impl DkgScalar {
    /// The zero scalar.
    pub const ZERO: Self = Self(vec![0u8; SCALAR_BYTES]);

    /// The unit scalar (1).
    pub const ONE: Self = Self({
        let mut v = vec![0u8; SCALAR_BYTES];
        v[0] = 1;
        v
    });

    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> NervResult<Self> {
        if bytes.len() != SCALAR_BYTES {
            return Err(NervError::Privacy(format!(
                "scalar must be {} bytes, got {}", SCALAR_BYTES, bytes.len()
            )));
        }
        Ok(Self(bytes.to_vec()))
    }

    /// Generate a random scalar.
    pub fn random() -> Self {
        // In production: use bls12_381::Scalar::random()
        Self(random_bytes(SCALAR_BYTES))
    }

    /// Convert to BLS12-381 Scalar.
    pub fn to_bls_scalar(&self) -> bls12_381::Scalar {
        let mut repr = bls12_381::ScalarRepr::default();
        let bytes: [u8; 32] = self.0.clone().try_into().unwrap_or([0u8; 32]);
        repr.as_mut().copy_from_slice(&bytes);
        bls12_381::Scalar::from_repr(repr).unwrap_or(bls12_381::Scalar::ZERO)
    }

    /// Create from a BLS12-381 Scalar.
    pub fn from_bls_scalar(s: &bls12_381::Scalar) -> Self {
        let repr = s.to_repr();
        Self(repr.as_ref().to_vec())
    }

    /// Add two scalars (mod p).
    pub fn add(&self, other: &Self) -> Self {
        let a = self.to_bls_scalar();
        let b = other.to_bls_scalar();
        Self::from_bls_scalar(&(a + b))
    }

    /// Multiply two scalars (mod p).
    pub fn mul(&self, other: &Self) -> Self {
        let a = self.to_bls_scalar();
        let b = other.to_bls_scalar();
        Self::from_bls_scalar(&(a * b))
    }

    /// Negate a scalar.
    pub fn neg(&self) -> Self {
        let a = self.to_bls_scalar();
        Self::from_bls_scalar(&(-a))
    }

    /// Compute modular inverse.
    pub fn invert(&self) -> Option<Self> {
        let a = self.to_bls_scalar();
        a.invert().map(|inv| Self::from_bls_scalar(&inv)).into()
    }

    /// Check if zero.
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }
}

// ─── Polynomial ──────────────────────────────────────────────────────────

/// A polynomial over the BLS12-381 scalar field.
///
/// Represented as coefficients `[a_0, a_1, ..., a_{t-1}]` where:
/// ```text
/// f(x) = a_0 + a_1·x + a_2·x² + ... + a_{t-1}·x^{t-1}
/// ```
///
/// The constant term `a_0 = f(0)` is the secret.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Polynomial {
    /// Coefficients in ascending order of degree.
    pub coefficients: Vec<DkgScalar>,
}

impl Polynomial {
    /// Create a random polynomial of degree `degree` with the given constant term.
    pub fn random(degree: usize, constant_term: DkgScalar) -> Self {
        let mut coefficients = Vec::with_capacity(degree + 1);
        coefficients.push(constant_term);
        for _ in 0..degree {
            coefficients.push(DkgScalar::random());
        }
        Self { coefficients }
    }

    /// Create a random polynomial with a random constant term.
    pub fn random_full(degree: usize) -> Self {
        let constant_term = DkgScalar::random();
        Self::random(degree, constant_term)
    }

    /// Evaluate the polynomial at point x.
    pub fn evaluate(&self, x: &DkgScalar) -> DkgScalar {
        // Horner's method: f(x) = a_0 + x*(a_1 + x*(a_2 + ...))
        let mut result = DkgScalar::ZERO;
        for coeff in self.coefficients.iter().rev() {
            result = result.mul(x).add(coeff);
        }
        result
    }

    /// Get the degree of the polynomial.
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() {
            0
        } else {
            self.coefficients.len() - 1
        }
    }

    /// Get the constant term (the secret).
    pub fn constant_term(&self) -> &DkgScalar {
        self.coefficients.first().unwrap_or(&DkgScalar::ZERO)
    }
}

// ─── Feldman Commitment ──────────────────────────────────────────────────

/// A Feldman VSS commitment: C_k = g^(a_k) in G1.
///
/// These commitments allow public verification of shares
/// without revealing the coefficients.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeldmanCommitment {
    /// The G1 point commitment (compressed, 48 bytes).
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub point: Vec<u8>,
}

impl FeldmanCommitment {
    /// Compute a commitment for a coefficient: C = g^a in G1.
    pub fn commit(coefficient: &DkgScalar) -> Self {
        let scalar = coefficient.to_bls_scalar();
        let g = bls12_381::G1Affine::generator();
        let point = bls12_381::G1Projective::from(g) * scalar;
        let compressed = bls12_381::G1Affine::from(point).to_compressed();
        Self {
            point: compressed.as_ref().to_vec(),
        }
    }

    /// Verify a share against Feldman commitments.
    ///
    /// Checks that: g^(share_i) == Π_k (C_k)^(i^k)
    pub fn verify_share(
        commitments: &[FeldmanCommitment],
        participant_index: u32,
        share: &DkgScalar,
    ) -> bool {
        if commitments.is_empty() {
            return false;
        }

        // Left side: g^share
        let g = bls12_381::G1Affine::generator();
        let lhs = bls12_381::G1Projective::from(g) * share.to_bls_scalar();

        // Right side: Π_k C_k^(i^k)
        let index_scalar = bls12_381::Scalar::from(participant_index as u64);
        let mut rhs = bls12_381::G1Projective::identity();
        let mut power = bls12_381::Scalar::ONE;

        for commitment in commitments {
            let c_point = bls12_381::G1Affine::from_compressed(
                &commitment.point.clone().try_into().unwrap_or([0u8; 48])
            );
            if let Some(p) = c_point.into_option() {
                rhs += bls12_381::G1Projective::from(p) * power;
            }
            power *= index_scalar;
        }

        bls12_381::G1Affine::from(lhs) == bls12_381::G1Affine::from(rhs)
    }
}

/// A set of Feldman commitments for one participant's polynomial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentSet {
    /// The participant who generated these commitments.
    pub participant_id: ValidatorId,

    /// The commitments C_0, C_1, ..., C_{t-1}.
    pub commitments: Vec<FeldmanCommitment>,
}

// ─── DKG Deal ────────────────────────────────────────────────────────────

/// A share sent from one participant to another during the DKG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DkgDeal {
    /// The sender's participant ID.
    pub from: ValidatorId,

    /// The recipient's participant ID.
    pub to: ValidatorId,

    /// The encrypted share (ML-KEM-768 ciphertext + ChaCha20-Poly1305).
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub encrypted_share: Vec<u8>,

    /// The sender's Feldman commitments (for verification).
    pub commitments: CommitmentSet,

    /// Session ID for this DKG round.
    pub session_id: [u8; 32],
}

// ─── DKG Participant ────────────────────────────────────────────────────

/// State of a participant in the DKG protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DkgParticipant {
    /// Unique identifier.
    pub id: ValidatorId,

    /// Index in the participant set (1-based for polynomial evaluation).
    pub index: u32,

    /// ML-KEM-768 public key for receiving encrypted shares.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub kem_pk: Vec<u8>,

    /// Whether this participant has confirmed their shares.
    pub confirmed: bool,
}

// ─── DKG Session ────────────────────────────────────────────────────────

/// The phase of a DKG session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DkgPhase {
    /// Waiting for all participants to join.
    Initialization,
    /// Distributing shares and commitments.
    ShareDistribution,
    /// Verifying shares and aggregating.
    Verification,
    /// Computing the collective public key.
    KeyComputation,
    /// DKG complete — public key and shares are finalized.
    Completed,
}

impl std::fmt::Display for DkgPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Initialization => write!(f, "initialization"),
            Self::ShareDistribution => write!(f, "share_distribution"),
            Self::Verification => write!(f, "verification"),
            Self::KeyComputation => write!(f, "key_computation"),
            Self::Completed => write!(f, "completed"),
        }
    }
}

/// An active DKG session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DkgSession {
    /// Unique session ID.
    pub session_id: [u8; 32],

    /// The threshold t (minimum shares for decryption).
    pub threshold: usize,

    /// Total number of participants n.
    pub num_participants: usize,

    /// Current phase.
    pub phase: DkgPhase,

    /// Registered participants.
    pub participants: Vec<DkgParticipant>,

    /// Received deals (from → to → deal).
    pub deals: HashMap<[u8; 32], Vec<DkgDeal>>,

    /// Received commitment sets (from → commitments).
    pub commitment_sets: HashMap<[u8; 32], CommitmentSet>,

    /// Computed shares for this participant (from_i → share_ij).
    pub received_shares: HashMap<[u8; 32], DkgScalar>,

    /// This participant's aggregated secret share.
    pub secret_share: Option<DkgScalar>,

    /// The collective public key (computed in KeyComputation phase).
    pub collective_pk: Option<DkgPublicKey>,

    /// Timestamp of session creation.
    pub created_at: u64,
}

impl DkgSession {
    /// Create a new DKG session.
    pub fn new(threshold: usize, num_participants: usize) -> NervResult<Self> {
        if num_participants < DKG_MIN_PARTICIPANTS {
            return Err(NervError::Privacy(format!(
                "need at least {} participants, got {}",
                DKG_MIN_PARTICIPANTS, num_participants
            )));
        }
        if threshold > num_participants {
            return Err(NervError::Privacy(format!(
                "threshold {} exceeds participants {}", threshold, num_participants
            )));
        }
        if threshold < 2 {
            return Err(NervError::Privacy("threshold must be at least 2".into()));
        }

        Ok(Self {
            session_id: crate::utils::random_32bytes(),
            threshold,
            num_participants,
            phase: DkgPhase::Initialization,
            participants: Vec::new(),
            deals: HashMap::new(),
            commitment_sets: HashMap::new(),
            received_shares: HashMap::new(),
            secret_share: None,
            collective_pk: None,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }

    /// Create with default threshold (⌈2n/3⌉).
    pub fn new_default(num_participants: usize) -> NervResult<Self> {
        let threshold = ((num_participants as f64) * DKG_DEFAULT_THRESHOLD_FRACTION)
            .ceil() as usize;
        Self::new(threshold, num_participants)
    }

    /// Register a participant.
    pub fn add_participant(&mut self, participant: DkgParticipant) -> NervResult<()> {
        if self.phase != DkgPhase::Initialization {
            return Err(NervError::Privacy(
                "cannot add participants after initialization".into()
            ));
        }
        if self.participants.len() >= self.num_participants {
            return Err(NervError::Privacy("all participants already registered".into()));
        }
        self.participants.push(participant);
        Ok(())
    }

    /// Check if all participants have registered and transition to ShareDistribution.
    pub fn check_ready(&mut self) -> bool {
        if self.participants.len() == self.num_participants {
            self.phase = DkgPhase::ShareDistribution;
            true
        } else {
            false
        }
    }

    /// Get the session ID as hex.
    pub fn session_id_hex(&self) -> String {
        hex::encode(self.session_id)
    }

    /// Get the threshold.
    pub fn threshold(&self) -> usize {
        self.threshold
    }

    /// Check if the DKG is complete.
    pub fn is_complete(&self) -> bool {
        self.phase == DkgPhase::Completed
    }
}

// ─── DKG Public Key ──────────────────────────────────────────────────────

/// The collective DKG public key (in G1).
///
/// This is used to encrypt mempool transactions. No single
/// participant knows the corresponding secret key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DkgPublicKey {
    /// The public key point in G1 (compressed, 48 bytes).
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub point: Vec<u8>,

    /// BLAKE3 hash of the point for quick comparison.
    pub hash: [u8; 32],

    /// The DKG session that produced this key.
    pub session_id: [u8; 32],

    /// The threshold t.
    pub threshold: usize,

    /// The number of participants n.
    pub num_participants: usize,
}

impl DkgPublicKey {
    /// Compute the collective public key from Feldman commitments.
    ///
    /// PK = Σ_i C_i[0] = Σ_i g^(f_i(0))
    pub fn from_commitments(
        commitment_sets: &[CommitmentSet],
        session_id: [u8; 32],
        threshold: usize,
        num_participants: usize,
    ) -> Self {
        let mut pk_point = bls12_381::G1Projective::identity();

        for cs in commitment_sets {
            if let Some(c0) = cs.commitments.first() {
                if let Some(point) = bls12_381::G1Affine::from_compressed(
                    &c0.point.clone().try_into().unwrap_or([0u8; 48])
                ).into_option() {
                    pk_point += bls12_381::G1Projective::from(point);
                }
            }
        }

        let compressed = bls12_381::G1Affine::from(pk_point).to_compressed();
        let point_bytes = compressed.as_ref().to_vec();
        let hash = blake3_hash(&point_bytes);

        Self {
            point: point_bytes,
            hash,
            session_id,
            threshold,
            num_participants,
        }
    }

    /// Get the G1 affine point.
    pub fn to_g1_affine(&self) -> bls12_381::G1Affine {
        bls12_381::G1Affine::from_compressed(
            &self.point.clone().try_into().unwrap_or([0u8; 48])
        ).into_option().unwrap_or(bls12_381::G1Affine::identity())
    }

    /// Check if this is the identity point (invalid key).
    pub fn is_identity(&self) -> bool {
        self.to_g1_affine().is_identity().into()
    }
}

// ─── DKG Share ──────────────────────────────────────────────────────────

/// A participant's final aggregated secret share.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DkgSecretShare {
    /// The participant's ID.
    pub participant_id: ValidatorId,

    /// The participant's index (1-based).
    pub index: u32,

    /// The secret share value.
    pub share: DkgScalar,

    /// The session this share belongs to.
    pub session_id: [u8; 32],
}

impl DkgSecretShare {
    /// Create a new secret share.
    pub fn new(participant_id: ValidatorId, index: u32, share: DkgScalar, session_id: [u8; 32]) -> Self {
        Self { participant_id, index, share, session_id }
    }

    /// Get the G1 public share: g^share.
    pub fn public_share(&self) -> FeldmanCommitment {
        FeldmanCommitment::commit(&self.share)
    }
}

// ─── Lagrange Interpolation ──────────────────────────────────────────────

/// Compute a Lagrange coefficient for threshold reconstruction.
///
/// Given a set of participant indices S and a target index j,
/// compute: λ_j = Π_{k ∈ S, k ≠ j} k / (k - j)  (mod p)
///
/// This is used for both:
/// - Threshold decryption (combining partial decryptions)
/// - Secret reconstruction (recovering the secret from t shares)
pub fn lagrange_coefficient(
    participant_indices: &[u32],
    target_index: u32,
) -> DkgScalar {
    let j = bls12_381::Scalar::from(target_index as u64);
    let mut numerator = bls12_381::Scalar::ONE;
    let mut denominator = bls12_381::Scalar::ONE;

    for &k in participant_indices {
        if k == target_index {
            continue;
        }
        let k_scalar = bls12_381::Scalar::from(k as u64);
        numerator *= k_scalar;
        denominator *= (k_scalar - j);
    }

    match denominator.invert() {
        Some(inv) => DkgScalar::from_bls_scalar(&(numerator * inv)),
        None => DkgScalar::ZERO,
    }
}

/// Reconstruct a secret from t shares using Lagrange interpolation.
///
/// Given shares {(i_1, s_1), ..., (i_t, s_t)}, compute:
/// s = Σ_j λ_j · s_j  where λ_j are Lagrange coefficients
pub fn reconstruct_secret(shares: &[(u32, DkgScalar)]) -> DkgScalar {
    let indices: Vec<u32> = shares.iter().map(|(i, _)| *i).collect();
    let mut result = bls12_381::Scalar::ZERO;

    for (index, share) in shares {
        let lambda = lagrange_coefficient(&indices, *index);
        let lambda_scalar = lambda.to_bls_scalar();
        let share_scalar = share.to_bls_scalar();
        result += lambda_scalar * share_scalar;
    }

    DkgScalar::from_bls_scalar(&result)
}

// ─── Threshold Encryption ────────────────────────────────────────────────

/// A ciphertext encrypted under the DKG collective public key.
///
/// Uses a hybrid ElGamal-like scheme:
/// - C1 = r * G (ephemeral public key in G1)
/// - C2 = K (symmetric key encrypted under r * PK)
/// - C3 = AEAD_Encrypt(payload, K) (authenticated encryption of the data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdCiphertext {
    /// Ephemeral public key C1 = r * G (48 bytes compressed).
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub c1: Vec<u8>,

    /// Encrypted symmetric key C2.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub c2: Vec<u8>,

    /// AEAD-encrypted payload C3.
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub c3: Vec<u8>,

    /// The DKG session ID.
    pub session_id: [u8; 32],
}

impl ThresholdCiphertext {
    /// Encrypt a payload under the DKG collective public key.
    pub fn encrypt(
        payload: &[u8],
        dkg_pk: &DkgPublicKey,
        session_id: [u8; 32],
    ) -> NervResult<Self> {
        // 1. Generate random scalar r
        let r = DkgScalar::random();
        let r_scalar = r.to_bls_scalar();

        // 2. Compute C1 = r * G
        let g = bls12_381::G1Affine::generator();
        let c1_point = bls12_381::G1Projective::from(g) * r_scalar;
        let c1 = bls12_381::G1Affine::from(c1_point).to_compressed();

        // 3. Compute r * PK (shared secret point)
        let pk_point = dkg_pk.to_g1_affine();
        let shared_point = bls12_381::G1Projective::from(pk_point) * r_scalar;

        // 4. Derive symmetric key from shared secret
        let shared_bytes = bls12_381::G1Affine::from(shared_point).to_compressed();
        let key = blake3_hash(shared_bytes.as_ref());

        // 5. Encrypt the payload with ChaCha20-Poly1305
        let c3 = crate::privacy::sphinx::aead_encrypt_internal(
            payload, &key, 0,
        )?;

        // 6. C2 is empty (the key is derived from the ECDH-like exchange)
        Ok(Self {
            c1: c1.as_ref().to_vec(),
            c2: Vec::new(),
            c3,
            session_id,
        })
    }

    /// Get the C1 point as G1Affine.
    pub fn c1_point(&self) -> bls12_381::G1Affine {
        bls12_381::G1Affine::from_compressed(
            &self.c1.clone().try_into().unwrap_or([0u8; 48])
        ).into_option().unwrap_or(bls12_381::G1Affine::identity())
    }
}

// Internal AEAD helper (reuses the sphinx module's logic)
mod aead_internal {
    use crate::NervResult;
    use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
    use chacha20poly1305::aead::{Aead, KeyInit};

    pub fn encrypt(plaintext: &[u8], key: &[u8; 32], hop: u8) -> crate::NervResult<Vec<u8>> {
        let nonce_bytes = crate::utils::blake3_derive_key("nerv:threshold:nonce", &[key, &[hop]].concat());
        let nonce = Nonce::from_slice(&nonce_bytes[..12]);
        let cipher = ChaCha20Poly1305::new(Key::from_slice(key));
        let ct = cipher.encrypt(nonce, plaintext)
            .map_err(|e| crate::NervError::Crypto(format!("AEAD encrypt: {e}")))?;
        let mut result = nonce_bytes[..12].to_vec();
        result.extend_from_slice(&ct);
        Ok(result)
    }

    pub fn decrypt(ciphertext: &[u8], key: &[u8; 32], hop: u8) -> crate::NervResult<Vec<u8>> {
        if ciphertext.len() < 12 + 16 { return Err(crate::NervError::Crypto("too short".into())); }
        let nonce = Nonce::from_slice(&ciphertext[..12]);
        let cipher = ChaCha20Poly1305::new(Key::from_slice(key));
        let pt = cipher.decrypt(nonce, &ciphertext[12..])
            .map_err(|e| crate::NervError::Crypto(format!("AEAD decrypt: {e}")))?;
        Ok(pt)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dkg_scalar_add() {
        let a = DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(3));
        let b = DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(7));
        let sum = a.add(&b);
        let expected = bls12_381::Scalar::from(10);
        assert_eq!(sum.to_bls_scalar(), expected);
    }

    #[test]
    fn test_dkg_scalar_mul() {
        let a = DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(3));
        let b = DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(7));
        let product = a.mul(&b);
        let expected = bls12_381::Scalar::from(21);
        assert_eq!(product.to_bls_scalar(), expected);
    }

    #[test]
    fn test_polynomial_evaluate() {
        // f(x) = 5 + 3x → f(2) = 5 + 6 = 11
        let poly = Polynomial {
            coefficients: vec![
                DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(5)),
                DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(3)),
            ],
        };
        let x = DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(2));
        let result = poly.evaluate(&x);
        assert_eq!(result.to_bls_scalar(), bls12_381::Scalar::from(11));
    }

    #[test]
    fn test_polynomial_evaluate_quadratic() {
        // f(x) = 1 + 2x + 3x² → f(2) = 1 + 4 + 12 = 17
        let poly = Polynomial {
            coefficients: vec![
                DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(1)),
                DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(2)),
                DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(3)),
            ],
        };
        let x = DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(2));
        let result = poly.evaluate(&x);
        assert_eq!(result.to_bls_scalar(), bls12_381::Scalar::from(17));
    }

    #[test]
    fn test_feldman_commitment() {
        let scalar = DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(42));
        let commitment = FeldmanCommitment::commit(&scalar);
        assert_eq!(commitment.point.len(), G1_COMPRESSED_BYTES);
    }

    #[test]
    fn test_lagrange_coefficient() {
        // For indices [1, 2, 3] and target 1:
        // λ_1 = (2 * 3) / ((2-1) * (3-1)) = 6 / 2 = 3
        let indices = vec![1u32, 2, 3];
        let lambda1 = lagrange_coefficient(&indices, 1);
        assert_eq!(lambda1.to_bls_scalar(), bls12_381::Scalar::from(3));
    }

    #[test]
    fn test_secret_reconstruction() {
        // Create a degree-2 polynomial: f(x) = 7 + 3x + 2x²
        // Shares: f(1)=12, f(2)=21, f(3)=34
        let poly = Polynomial {
            coefficients: vec![
                DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(7)),
                DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(3)),
                DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(2)),
            ],
        };

        let shares: Vec<(u32, DkgScalar)> = vec![
            (1, poly.evaluate(&DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(1)))),
            (2, poly.evaluate(&DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(2)))),
            (3, poly.evaluate(&DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(3)))),
        ];

        // Reconstruct the secret (f(0) = 7)
        let secret = reconstruct_secret(&shares);
        assert_eq!(secret.to_bls_scalar(), bls12_381::Scalar::from(7));
    }

    #[test]
    fn test_secret_reconstruction_any_subset() {
        // Any t=3 out of n=5 shares should reconstruct the same secret
        let poly = Polynomial {
            coefficients: vec![
                DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(42)),
                DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(5)),
                DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(3)),
            ],
        };

        let all_shares: Vec<(u32, DkgScalar)> = (1..=5).map(|i| {
            (i, poly.evaluate(&DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(i as u64)))))
        ).collect();

        // Reconstruct with different subsets of 3 shares
        let secret1 = reconstruct_secret(&all_shares[0..3]);
        let secret2 = reconstruct_secret(&all_shares[2..5]);
        let secret3 = reconstruct_secret(&vec![all_shares[0].clone(), all_shares[3].clone(), all_shares[4].clone()]);

        assert_eq!(secret1, secret2);
        assert_eq!(secret2, secret3);
        assert_eq!(secret1.to_bls_scalar(), bls12_381::Scalar::from(42));
    }

    #[test]
    fn test_dkg_session_creation() {
        let session = DkgSession::new(5, 7).unwrap();
        assert_eq!(session.threshold, 5);
        assert_eq!(session.num_participants, 7);
        assert_eq!(session.phase, DkgPhase::Initialization);
    }

    #[test]
    fn test_dkg_session_default_threshold() {
        let session = DkgSession::new_default(21).unwrap();
        // ⌈21 * 2/3⌉ = 14
        assert_eq!(session.threshold, 14);
    }

    #[test]
    fn test_dkg_session_invalid_params() {
        // Threshold > participants
        assert!(DkgSession::new(10, 5).is_err());
        // Too few participants
        assert!(DkgSession::new(2, 2).is_err());
    }

    #[test]
    fn test_dkg_public_key_from_commitments() {
        // Create commitments for two participants
        let s1 = DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(10));
        let s2 = DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(20));
        let c1 = FeldmanCommitment::commit(&s1);
        let c2 = FeldmanCommitment::commit(&s2);

        let cs1 = CommitmentSet {
            participant_id: ValidatorId::from_bytes([1u8; 32]),
            commitments: vec![c1],
        };
        let cs2 = CommitmentSet {
            participant_id: ValidatorId::from_bytes([2u8; 32]),
            commitments: vec![c2],
        };

        let pk = DkgPublicKey::from_commitments(
            &[cs1, cs2], [0u8; 32], 2, 2,
        );

        // PK = g^10 + g^20 = g^30
        let expected = FeldmanCommitment::commit(
            &DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(30))
        );
        assert_eq!(pk.point, expected.point);
    }

    #[test]
    fn test_threshold_ciphertext_encrypt() {
        let s = DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(42));
        let commitment = FeldmanCommitment::commit(&s);
        let dkg_pk = DkgPublicKey {
            point: commitment.point,
            hash: blake3_hash(&commitment.point),
            session_id: [0u8; 32],
            threshold: 3,
            num_participants: 5,
        };

        let payload = b"private transaction data";
        let ct = ThresholdCiphertext::encrypt(payload, &dkg_pk, [0u8; 32]);
        assert!(ct.is_ok());

        let ct = ct.unwrap();
        assert_eq!(ct.c1.len(), G1_COMPRESSED_BYTES);
        assert!(!ct.c3.is_empty());
    }
}
