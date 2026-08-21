//! Threshold Decryption Ceremony — Mempool privacy before execution.
//!
//! The threshold decryption ceremony is the critical privacy gate:
//! transactions sit in the mempool fully encrypted under the DKG
//! public key, and **no single node can read them**. Only when a
//! block producer assembles a batch and ≥t validators participate
//! can the transactions be decrypted — and only within the
//! validator's secure memory space.
//!
//! # Ceremony Flow
//!
//! ```text
//! 1. Block producer selects batch of encrypted txs from mempool
//! 2. Each validator i computes partial decryption: P_i = C1^(share_i)
//! 3. Validators broadcast their partial decryptions
//! 4. Once ≥t partial decryptions collected, combine via Lagrange:
//!    shared_secret = Π P_j^(λ_j)
//! 5. Derive symmetric key K from shared_secret
//! 6. Decrypt each transaction with K
//! 7. Execute the batch
//! 8. Zeroize all plaintext from memory immediately after execution
//! ```
//!
//! # Security Guarantees
//!
//! - **Pre-execution privacy**: No node reads any transaction before the ceremony
//! - **Threshold security**: t-1 partial decryptions reveal nothing
//! - **Post-execution privacy**: Plaintext zeroized after execution
//! - **No TEE dependency**: Security is purely mathematical

use crate::{EMBEDDING_DIM, NervError, NervResult, ValidatorId};
use crate::privacy::dkg::{
    DkgScalar, DkgSecretShare, DkgPublicKey, ThresholdCiphertext,
    FeldmanCommitment, lagrange_coefficient,
};
use crate::utils::{blake3_hash, secure_zero};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Partial Decryption ──────────────────────────────────────────────────

/// A validator's partial decryption share for a single ciphertext.
///
/// P_i = C1^(share_i) in G1, where C1 is the ephemeral public key
/// from the ciphertext and share_i is the validator's DKG share.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartialDecryption {
    /// The validator who computed this partial decryption.
    pub validator_id: ValidatorId,

    /// The validator's index in the DKG (1-based).
    pub validator_index: u32,

    /// The partial decryption point in G1 (compressed, 48 bytes).
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub partial_point: Vec<u8>,

    /// The ciphertext hash this partial decryption corresponds to.
    pub ciphertext_hash: [u8; 32],

    /// Dilithium-3 signature over the partial decryption (attestation).
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub signature: Vec<u8>,
}

impl PartialDecryption {
    /// Compute a partial decryption from a ciphertext and secret share.
    pub fn compute(
        ciphertext: &ThresholdCiphertext,
        share: &DkgSecretShare,
        validator_id: ValidatorId,
    ) -> Self {
        // P_i = C1^(share_i)
        let c1 = ciphertext.c1_point();
        let share_scalar = share.share.to_bls_scalar();
        let partial = bls12_381::G1Projective::from(c1) * share_scalar;
        let compressed = bls12_381::G1Affine::from(partial).to_compressed();

        let ciphertext_hash = blake3_hash(&[
            &ciphertext.c1,
            &ciphertext.c2,
            &ciphertext.c3,
        ].concat());

        Self {
            validator_id,
            validator_index: share.index,
            partial_point: compressed.as_ref().to_vec(),
            ciphertext_hash,
            signature: Vec::new(), // Signed in production with Dilithium-3
        }
    }

    /// Get the G1 affine point.
    pub fn to_g1_affine(&self) -> bls12_381::G1Affine {
        bls12_381::G1Affine::from_compressed(
            &self.partial_point.clone().try_into().unwrap_or([0u8; 48])
        ).into_option().unwrap_or(bls12_381::G1Affine::identity())
    }
}

// ─── Decryption Ceremony ─────────────────────────────────────────────────

/// State of a threshold decryption ceremony.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecryptionCeremony {
    /// The ciphertext being decrypted.
    pub ciphertext: ThresholdCiphertext,

    /// Collected partial decryptions (validator_index → partial).
    pub partials: HashMap<u32, PartialDecryption>,

    /// Required threshold.
    pub threshold: usize,

    /// Ceremony state.
    pub state: CeremonyState,

    /// Timestamp of ceremony start.
    pub started_at: u64,
}

/// State of a decryption ceremony.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CeremonyState {
    /// Waiting for partial decryptions.
    Collecting,
    /// Enough partials collected; ready to combine.
    Ready,
    /// Decryption complete.
    Completed,
    /// Ceremony failed.
    Failed,
}

impl DecryptionCeremony {
    /// Create a new ceremony for a ciphertext.
    pub fn new(ciphertext: ThresholdCiphertext, threshold: usize) -> Self {
        Self {
            ciphertext,
            partials: HashMap::new(),
            threshold,
            state: CeremonyState::Collecting,
            started_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Add a partial decryption.
    pub fn add_partial(&mut self, partial: PartialDecryption) -> NervResult<()> {
        // Verify the partial corresponds to our ciphertext
        let ct_hash = blake3_hash(&[
            &self.ciphertext.c1,
            &self.ciphertext.c2,
            &self.ciphertext.c3,
        ].concat());

        if partial.ciphertext_hash != ct_hash {
            return Err(NervError::Privacy(
                "partial decryption corresponds to different ciphertext".into()
            ));
        }

        self.partials.insert(partial.validator_index, partial);

        // Check if we have enough partials
        if self.partials.len() >= self.threshold {
            self.state = CeremonyState::Ready;
        }

        Ok(())
    }

    /// Check if ready to combine.
    pub fn is_ready(&self) -> bool {
        self.state == CeremonyState::Ready || self.state == CeremonyState::Completed
    }

    /// Combine partial decryptions to recover the plaintext.
    ///
    /// Returns the decrypted payload bytes.
    ///
    /// **Security**: The caller is responsible for zeroizing the
    /// returned plaintext after use.
    pub fn combine(&mut self) -> NervResult<Vec<u8>> {
        if self.partials.len() < self.threshold {
            return Err(NervError::Privacy(format!(
                "not enough partials: {} < {}",
                self.partials.len(), self.threshold
            )));
        }

        // Select the first `threshold` partials
        let selected: Vec<&PartialDecryption> = self.partials.values()
            .take(self.threshold)
            .collect();

        let indices: Vec<u32> = selected.iter().map(|p| p.validator_index).collect();

        // Combine: shared_secret_point = Π P_j^(λ_j)
        let mut combined = bls12_381::G1Projective::identity();
        for partial in selected {
            let lambda = lagrange_coefficient(&indices, partial.validator_index);
            let lambda_scalar = lambda.to_bls_scalar();
            let partial_point = partial.to_g1_affine();
            combined += bls12_381::G1Projective::from(partial_point) * lambda_scalar;
        }

        // Derive the symmetric key from the combined point
        let combined_affine = bls12_381::G1Affine::from(combined);
        let combined_bytes = combined_affine.to_compressed();
        let key = blake3_hash(combined_bytes.as_ref());

        // Decrypt the payload
        let plaintext = aead_internal::decrypt(
            &self.ciphertext.c3, &key, 0,
        )?;

        self.state = CeremonyState::Completed;
        Ok(plaintext)
    }

    /// Get the number of collected partials.
    pub fn partial_count(&self) -> usize {
        self.partials.len()
    }
}

// ─── Batch Decryption ────────────────────────────────────────────────────

/// Result of decrypting a batch of encrypted transactions.
#[derive(Debug, Clone)]
pub struct BatchDecryptionResult {
    /// Successfully decrypted transactions.
    pub decrypted: Vec<Vec<u8>>,

    /// Transactions that failed to decrypt.
    pub failed: Vec<([u8; 32], NervError)>,

    /// Time taken for the ceremony (microseconds).
    pub ceremony_time_us: u64,

    /// Number of validators who participated.
    pub validator_count: usize,
}

/// Decrypt a batch of ciphertexts using threshold decryption.
///
/// This is the primary function called by block producers after
/// assembling a batch from the encrypted mempool.
///
/// # Security
///
/// After this function returns, the caller MUST zeroize the
/// decrypted transactions after execution. Use `secure_zero`
/// on each Vec<u8> after processing.
pub fn decrypt_batch(
    ciphertexts: &[ThresholdCiphertext],
    partials: &[PartialDecryption],
    threshold: usize,
) -> NervResult<BatchDecryptionResult> {
    let start = std::time::Instant::now();
    let mut decrypted = Vec::new();
    let mut failed = Vec::new();
    let mut validator_indices = std::collections::HashSet::new();

    for ct in ciphertexts {
        let ct_hash = blake3_hash(&[&ct.c1, &ct.c2, &ct.c3].concat());

        let mut ceremony = DecryptionCeremony::new(ct.clone(), threshold);

        // Add relevant partials
        for partial in partials {
            if partial.ciphertext_hash == ct_hash {
                if let Err(e) = ceremony.add_partial(partial.clone()) {
                    continue;
                }
                validator_indices.insert(partial.validator_index);
            }
        }

        if ceremony.is_ready() {
            match ceremony.combine() {
                Ok(plaintext) => decrypted.push(plaintext),
                Err(e) => failed.push((ct_hash, e)),
            }
        } else {
            failed.push((ct_hash, NervError::Privacy("insufficient partials".into())));
        }
    }

    Ok(BatchDecryptionResult {
        decrypted,
        failed,
        ceremony_time_us: start.elapsed().as_micros() as u64,
        validator_count: validator_indices.len(),
    })
}

/// Securely zeroize decrypted transaction data after execution.
///
/// **MUST** be called after transactions are executed and applied
/// to the embedding. Failure to call this leaks plaintext.
pub fn zeroize_decrypted_batch(batch: &mut BatchDecryptionResult) {
    for tx in &mut batch.decrypted {
        secure_zero(tx);
    }
    batch.decrypted.clear();
    batch.decrypted.shrink_to_fit();
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::dkg::{DkgSession, Polynomial};

    #[test]
    fn test_partial_decryption_computation() {
        // Create a simple DKG setup
        let secret = DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(42));
        let commitment = FeldmanCommitment::commit(&secret);
        let dkg_pk = DkgPublicKey {
            point: commitment.point,
            hash: blake3_hash(&commitment.point),
            session_id: [0u8; 32],
            threshold: 2,
            num_participants: 3,
        };

        let payload = b"test threshold encryption";
        let ct = ThresholdCiphertext::encrypt(payload, &dkg_pk, [0u8; 32]).unwrap();

        // Create a share
        let share = DkgSecretShare::new(
            ValidatorId::from_bytes([1u8; 32]),
            1,
            secret,
            [0u8; 32],
        );

        let partial = PartialDecryption::compute(
            &ct, &share, ValidatorId::from_bytes([1u8; 32]),
        );

        assert_eq!(partial.partial_point.len(), 48);
        assert_eq!(partial.validator_index, 1);
    }

    #[test]
    fn test_decryption_ceremony_state_machine() {
        let secret = DkgScalar::from_bls_scalar(&bls12_381::Scalar::from(42));
        let commitment = FeldmanCommitment::commit(&secret);
        let dkg_pk = DkgPublicKey {
            point: commitment.point,
            hash: blake3_hash(&commitment.point),
            session_id: [0u8; 32],
            threshold: 2,
            num_participants: 3,
        };

        let ct = ThresholdCiphertext::encrypt(b"test", &dkg_pk, [0u8; 32]).unwrap();
        let mut ceremony = DecryptionCeremony::new(ct, 2);

        assert_eq!(ceremony.state, CeremonyState::Collecting);
        assert!(!ceremony.is_ready());
    }

    #[test]
    fn test_zeroize_decrypted_batch() {
        let mut result = BatchDecryptionResult {
            decrypted: vec![vec![1u8, 2, 3, 4], vec![5, 6, 7, 8]],
            failed: vec![],
            ceremony_time_us: 100,
            validator_count: 3,
        };

        zeroize_decrypted_batch(&mut result);
        assert!(result.decrypted.is_empty());
    }

    #[test]
    fn test_lagrange_coefficients_sum_to_one() {
        // For any set of indices, the Lagrange coefficients should sum to 1
        let indices = vec![1u32, 2, 3];
        let mut sum = bls12_381::Scalar::ZERO;
        for &i in &indices {
            let lambda = lagrange_coefficient(&indices, i);
            sum += lambda.to_bls_scalar();
        }
        assert_eq!(sum, bls12_381::Scalar::ONE);
    }

    #[test]
    fn test_lagrange_coefficients_sum_to_one_larger() {
        let indices = vec![1u32, 3, 5, 7];
        let mut sum = bls12_381::Scalar::ZERO;
        for &i in &indices {
            let lambda = lagrange_coefficient(&indices, i);
            sum += lambda.to_bls_scalar();
        }
        assert_eq!(sum, bls12_381::Scalar::ONE);
    }
}
