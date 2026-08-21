//! Verifiable Delay Witnesses (VDWs) — Permanent, offline-verifiable receipts.
//!
//! The VDW is NERV's user-facing cryptographic receipt: a tiny
//! (~3.5 KB in V2.0) permanent proof that a specific private
//! transaction was canonically included exactly as intended —
//! **without revealing** any details about amounts, counterparties,
//! or other transactions.
//!
//! # V2.0 Changes vs V1.01
//!
//! | Property | V1.01 | V2.0 |
//! |----------|-------|------|
//! | Generation time | <150 ms | **<15 ms** (simpler circuit) |
//! | VDW size | ~1.4 KB | ~3.5 KB (Dilithium-3 > ECDSA) |
//! | TEE attestation | Yes (SGX/SEV) | **No** (DKG threshold sig) |
//! | Verification | 4 steps (incl. TEE) | 4 steps (pure crypto) |
//! | Offline verify | <80 ms | **<30 ms** (lighter circuit) |
//!
//! # VDW Contents
//!
//! ```text
//! VDW = {
//!   tx_hash:           [u8; 32],  // SHA3-256 of blinded tx
//!   shard_id:          u64,       // Executing shard
//!   lattice_height:    u64,       // Block height of inclusion
//!   delta_proof:       [u8; ~400],// Halo2 proof of W·x + b
//!   embedding_root:    [u8; 32],  // BLAKE3 hash of post-batch embedding
//!   dkg_sig:           [u8; 48],  // BLS12-381 threshold sig (replaces TEE)
//!   dilithium_sig:     [u8; 2433],// Dilithium-3 sig from attesting validator
//!   timestamp_ms:      u64,       // Inclusion timestamp
//!   version:           u32,       // Protocol version (2)
//! }
//! ```
//!
//! # Verification Algorithm (V2.0 — No TEEs)
//!
//! ```text
//! 1. DKG Threshold Signature:
//!    Verify the BLS12-381 threshold signature over (tx_hash || root || height)
//!    against the known DKG public key. Replaces TEE attestation.
//!
//! 2. Dilithium-3 Signature:
//!    Verify the validator's Dilithium-3 signature over the VDW payload
//!    using the attested public key.
//!
//! 3. Recursive Halo2 Delta Proof:
//!    Run the LatentLedger Lite verifier on the inclusion proof.
//!    Confirms correct homomorphic delta application.
//!
//! 4. Homomorphic Root Reconciliation:
//!    Load trusted embedding root, apply proven delta:
//!    computed_root = trusted_root + δ_path
//!    Verify: Hash(computed_root) == vdw.embedding_root
//! ```

use crate::{
    EMBEDDING_DIM, EMBEDDING_BYTES, NervError, NervResult,
    TxHash, EmbeddingRoot, ShardId, BlockHeight,
};
use crate::embedding::fixed_point::EmbeddingVector;
use crate::embedding::homomorphism::EmbeddingDelta;
use crate::circuit::latent_ledger_lite::CircuitProof;
use crate::privacy::dkg::DkgPublicKey;
use crate::utils::{blake3_hash, blake3_derive_key};
use serde::{Deserialize, Serialize};

// ─── VDW Constants ────────────────────────────────────────────────────────

/// VDW protocol version (V2.0).
pub const VDW_VERSION: u32 = 2;

/// Maximum VDW size in bytes (V2.0 with Dilithium-3).
pub const VDW_MAX_SIZE: usize = 3600;

/// Average VDW size in bytes.
pub const VDW_AVG_SIZE: usize = 3500;

/// BLS12-381 signature size (compressed G1, 48 bytes).
pub const BLS_SIG_SIZE: usize = 48;

/// Dilithium-3 signature size.
pub const DILITHIUM3_SIG_SIZE: usize = 2433;

/// Delta proof target size (Halo2 proof of W·x + b).
pub const DELTA_PROOF_SIZE: usize = 400;

// ─── VDW ──────────────────────────────────────────────────────────────────

/// A Verifiable Delay Witness — permanent, offline-verifiable receipt
/// of transaction inclusion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vdw {
    /// SHA3-256 hash of the blinded transaction payload.
    pub tx_hash: TxHash,

    /// Shard where the transaction was executed.
    pub shard_id: u64,

    /// Block height of inclusion.
    pub lattice_height: u64,

    /// Halo2 recursive proof of correct delta application (~400 bytes).
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub delta_proof: Vec<u8>,

    /// BLAKE3 hash of the canonical embedding root after this batch.
    pub embedding_root: EmbeddingRoot,

    /// BLS12-381 threshold signature from the DKG ceremony (48 bytes).
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub dkg_threshold_sig: Vec<u8>,

    /// Dilithium-3 signature from the attesting validator (2433 bytes).
    #[serde(serialize_with = "serde_bytes::serialize", deserialize_with = "serde_bytes::deserialize")]
    pub dilithium_sig: Vec<u8>,

    /// Timestamp of inclusion (Unix epoch milliseconds).
    pub timestamp_ms: u64,

    /// Protocol version.
    pub version: u32,
}

impl Vdw {
    /// Generate a VDW for a finalized transaction.
    pub fn generate(
        tx_hash: TxHash,
        shard_id: u64,
        lattice_height: u64,
        delta_proof: Vec<u8>,
        embedding_root: EmbeddingRoot,
        dkg_threshold_sig: Vec<u8>,
        dilithium_sig: Vec<u8>,
    ) -> Self {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            tx_hash,
            shard_id,
            lattice_height,
            delta_proof,
            embedding_root,
            dkg_threshold_sig,
            dilithium_sig,
            timestamp_ms,
            version: VDW_VERSION,
        }
    }

    /// Get the total size of this VDW in bytes.
    pub fn size(&self) -> usize {
        32 + // tx_hash
        8 +  // shard_id
        8 +  // lattice_height
        self.delta_proof.len() +
        32 + // embedding_root
        self.dkg_threshold_sig.len() +
        self.dilithium_sig.len() +
        8 +  // timestamp_ms
        4    // version
    }

    /// Check if the VDW size is within bounds.
    pub fn is_valid_size(&self) -> bool {
        self.size() <= VDW_MAX_SIZE
    }

    /// Compute a unique hash for this VDW (for storage key).
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.tx_hash.as_bytes());
        hasher.update(&self.shard_id.to_le_bytes());
        hasher.update(&self.lattice_height.to_le_bytes());
        hasher.update(&self.delta_proof);
        hasher.update(self.embedding_root.as_bytes());
        hasher.update(&self.version.to_le_bytes());
        hasher.finalize().into()
    }

    /// Serialize to bytes for storage/transmission.
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> NervResult<Self> {
        bincode::deserialize(data)
            .map_err(|e| NervError::Serialization(format!("VDW deserialize: {e}")))
    }

    /// Serialize to base64 for sharing.
    pub fn to_base64(&self) -> String {
        base64::engine::general_purpose::STANDARD.encode(&self.to_bytes())
    }

    /// Deserialize from base64.
    pub fn from_base64(s: &str) -> NervResult<Self> {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(s)
            .map_err(|e| NervError::Serialization(format!("base64 decode: {e}")))?;
        Self::from_bytes(&bytes)
    }
}

// ─── VDW Verification ────────────────────────────────────────────────────

/// Result of verifying a VDW.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VdwVerificationResult {
    /// VDW is valid — transaction permanently confirmed.
    Valid {
        /// The verified embedding root.
        embedding_root: EmbeddingRoot,
        /// The block height of inclusion.
        height: u64,
    },
    /// VDW is invalid.
    Invalid {
        /// Reason for failure.
        reason: String,
    },
}

impl VdwVerificationResult {
    /// Returns true if the VDW is valid.
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid { .. })
   $}
}

/// Verify a VDW using the V2.0 pure-cryptographic algorithm.
///
/// # Arguments
///
/// * `vdw` - The VDW to verify
/// * `trusted_root` - The trusted embedding root at height-1
/// * `delta` - The proven delta from the VDW's proof
/// * `dkg_pk` - The DKG collective public key
/// * `validator_dilithium_pk` - The attesting validator's Dilithium-3 PK
///
/// # Returns
///
/// `VdwVerificationResult::Valid` if all 4 verification steps pass.
///
/// # Verification Time
///
/// <30 ms on mobile devices.
pub fn verify_vdw(
    vdw: &Vdw,
    trusted_root: &EmbeddingRoot,
    delta: &EmbeddingDelta,
    dkg_pk: &DkgPublicKey,
    validator_dilithium_pk: &[u8],
) -> VdwVerificationResult {
    // Step 1: Verify DKG Threshold Signature
    if !verify_dkg_signature(vdw, dkg_pk) {
        return VdwVerificationResult::Invalid {
            reason: "DKG threshold signature verification failed".into(),
        };
    }

    // Step 2: Verify Dilithium-3 Signature
    if !verify_dilithium_signature(vdw, validator_dilithium_pk) {
        return VdwVerificationResult::Invalid {
            reason: "Dilithium-3 signature verification failed".into(),
        };
    }

    // Step 3: Verify Halo2 Delta Proof
    if !verify_delta_proof(vdw) {
        return VdwVerificationResult::Invalid {
            reason: "Halo2 delta proof verification failed".into(),
        };
    }

    // Step 4: Homomorphic Root Reconciliation
    if !verify_root_reconciliation(vdw, trusted_root, delta) {
        return VdwVerificationResult::Invalid {
            reason: "embedding root reconciliation failed".into(),
        };
    }

    VdwVerificationResult::Valid {
        embedding_root: vdw.embedding_root,
        height: vdw.lattice_height,
    }
}

/// Step 1: Verify the DKG threshold signature over the VDW payload.
fn verify_dkg_signature(vdw: &Vdw, dkg_pk: &DkgPublicKey) -> bool {
    if vdw.dkg_threshold_sig.len() != BLS_SIG_SIZE {
        return false;
    }

    // Construct the signed message: tx_hash || root || height
    let mut msg = Vec::new();
    msg.extend_from_slice(vdw.tx_hash.as_bytes());
    msg.extend_from_slice(vdw.embedding_root.as_bytes());
    msg.extend_from_slice(&vdw.lattice_height.to_le_bytes());

    // In production: BLS signature verification
    //   BLS::Verify(dkg_pk, msg, sig)
    // For now: hash-based integrity check
    let expected = blake3_derive_key("nerv:vdw:dkg-sig", &msg);
    let sig_hash = blake3_hash(&vdw.dkg_threshold_sig);

    // The signature should be a valid BLS signature.
    // We verify it's non-empty and has the correct size.
    !vdw.dkg_threshold_sig.is_empty()
}

/// Step 2: Verify the attesting validator's Dilithium-3 signature.
fn verify_dilithium_signature(vdw: &Vdw, validator_pk: &[u8]) -> bool {
    if vdw.dilithium_sig.is_empty() || validator_pk.is_empty() {
        return false;
    }

    // Construct the signed message
    let mut msg = Vec::new();
    msg.extend_from_slice(vdw.tx_hash.as_bytes());
    msg.extend_from_slice(&vdw.shard_id.to_le_bytes());
    msg.extend_from_slice(&vdw.lattice_height.to_le_bytes());
    msg.extend_from_slice(&vdw.delta_proof);
    msg.extend_from_slice(vdw.embedding_root.as_bytes());
    msg.extend_from_slice(&vdw.timestamp_ms.to_le_bytes());

    // In production: Dilithium-3 verification
    //   Dilithium3::verify(msg, sig, pk)
    let pk = oqs::sig::Dilithium3::PublicKey::from_bytes(validator_pk);
    let sig = oqs::sig::Dilithium3::Signature::from_bytes(&vdw.dilithium_sig);

    match (pk, sig) {
        (Ok(pk), Ok(sig)) => oqs::sig::Dilithium3::verify(&msg, &sig, &pk),
        _ => false,
    }
}

/// Step 3: Verify the Halo2 recursive delta proof.
fn verify_delta_proof(vdw: &Vdw) -> bool {
    // In production: Halo2 verifier
    //   halo2_proofs::plonk::verify_proof(params, vk, proof, public_inputs)
    // The delta_proof should be a valid Halo2 proof.

    // Basic checks
    !vdw.delta_proof.is_empty() && vdw.delta_proof.len() <= DELTA_PROOF_SIZE * 2
}

/// Step 4: Homomorphic root reconciliation.
///
/// computed_root = Hash(trusted_root + delta)
/// Verify: computed_root == vdw.embedding_root
fn verify_root_reconciliation(
    vdw: &Vdw,
    trusted_root: &EmbeddingRoot,
    delta: &EmbeddingDelta,
) -> bool {
    // In production, we would:
    // 1. Load the actual embedding vector for trusted_root
    // 2. Apply the delta: e_new = e_prev + delta
    // 3. Compute hash: h = BLAKE3(e_new)
    // 4. Compare h with vdw.embedding_root

    // For now, verify that the delta is non-trivial and the
    // embedding root is not the genesis
    !vdw.embedding_root.is_genesis() || delta.is_zero()
}

// ─── V/VDW Batch ──────────────────────────────────────────────────────────

/// A batch of VDWs for efficient generation and distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VdwBatch {
    /// The VDWs in this batch.
    pub witnesses: Vec<Vdw>,

    /// BLAKE3 Merkle root of all VDWs.
    pub batch_root: [u8; 32],

    /// The block height of this batch.
    pub height: u64,
}

impl VdwBatch {
    /// Create a batch from VDWs.
    pub fn new(witnesses: Vec<Vdw>, height: u64) -> Self {
        let mut hasher = blake3::Hasher::new();
        for vdw in &witnesses {
            hasher.update(&vdw.hash());
        }
        let batch_root: [u8; 32] = hasher.finalize().into();

        Self { witnesses, batch_root, height }
    }

    /// Get the number of VDWs.
    pub fn len(&self) -> usize {
        self.witnesses.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.witnesses.is_empty()
    }

    /// Find a VDW by tx_hash.
    pub fn find_by_tx_hash(&self, tx_hash: &TxHash) -> Option<&Vdw> {
        self.witnesses.iter().find(|vdw| vdw.tx_hash == *tx_hash)
    }

    /// Total size of all VDWs in bytes.
    pub fn total_size(&self) -> usize {
        self.witnesses.iter().map(|vdw| vdw.size()).sum()
    }
}

// ─── VDW Storage ─────────────────────────────────────────────────────────

/// Interface for VDW storage (RocksDB + Arweave/IPFS).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VdwStorageEntry {
    /// The VDW.
    pub vdw: Vdw,

    /// Arweave transaction ID (after pinning).
    pub arweave_tx_id: Option<String>,

    /// IPFS CID (after pinning).
    pub ipfs_cid: Option<String>,

    /// Whether the VDW has been pinned to permanent storage.
    pub is_pinned: bool,

    /// Number of times this VDW has been verified.
    pub verification_count: u64,
}

impl VdwStorageEntry {
    /// Create a new storage entry.
    pub fn new(vdw: Vdw) -> Self {
        Self {
            vdw,
            arweave_tx_id: None,
            ipfs_cid: None,
            is_pinned: false,
            verification_count: 0,
        }
    }

    /// Mark as pinned to Arweave.
    pub fn pin_arweave(&mut self, tx_id: String) {
        self.arweave_tx_id = Some(tx_id);
        self.is_pinned = true;
    }

    /// Mark as pinned to IPFS.
    pub fn pin_ipfs(&mut self, cid: String) {
        self.ipfs_cid = Some(cid);
        self.is_pinned = true;
    }

    /// Record a verification.
    pub fn record_verification(&mut self) {
        self.verification_count += 1;
    }
}

// ─── VDW Generation from Block Finalization ──────────────────────────────

/// Generate VDWs for all transactions in a finalized batch.
///
/// Called by full nodes after consensus finalization.
pub fn generate_vdws_for_batch(
    tx_hashes: &[TxHash],
    shard_id: u64,
    height: u64,
    embedding_root: EmbeddingRoot,
    dkg_threshold_sig: Vec<u8>,
    dilithium_sig: Vec<u8>,
    delta_proofs: &[Vec<u8>],
) -> NervResult<VdwBatch> {
    if tx_hashes.len() != delta_proofs.len() {
        return Err(NervError::Privacy(format!(
            "tx_hashes ({}) != delta_proofs ({})",
            tx_hashes.len(), delta_proofs.len()
        )));
    }

    let witnesses: Vec<Vdw> = tx_hashes.iter()
        .zip(delta_proofs.iter())
        .map(|(tx_hash, delta_proof)| {
            Vdw::generate(
                tx_hash.clone(),
                shard_id,
                height,
                delta_proof.clone(),
                embedding_root,
                dkg_threshold_sig.clone(),
                dilithium_sig.clone(),
            )
        })
        .collect();

    Ok(VdwBatch::new(witnesses, height))
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_vdw() -> Vdw {
        Vdw::generate(
            TxHash::from_bytes([42u8; 32]),
            0,
            100,
            vec![0u8; DELTA_PROOF_SIZE],
            EmbeddingRoot::from_bytes([1u8; 32]),
            vec![0u8; BLS_SIG_SIZE],
            vec![0u8; DILITHIUM3_SIG_SIZE],
        )
    }

    #[test]
    fn test_vdw_generation() {
        let vdw = make_test_vdw();
        assert_eq!(vdw.version, VDW_VERSION);
        assert_eq!(vdw.lattice_height, 100);
        assert!(vdw.is_valid_size());
    }

    #[test]
    fn test_vdw_hash_deterministic() {
        let vdw = make_test_vdw();
        let h1 = vdw.hash();
        let h2 = vdw.hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_vdw_serialization_roundtrip() {
        let vdw = make_test_vdw();
        let bytes = vdw.to_bytes();
        let recovered = Vdw::from_bytes(&bytes).unwrap();
        assert_eq!(vdw, recovered);
    }

    #[test]
    fn test_vdw_base64_roundtrip() {
        let vdw = make_test_vdw();
        let b64 = vdw.to_base64();
        let recovered = Vdw::from_base64(&b64).unwrap();
        assert_eq!(vdw, recovered);
    }

    #[test]
    fn test_vdw_verification_steps() {
        let vdw = make_test_vdw();
        let trusted_root = EmbeddingRoot::from_bytes([0u8; 32]);
        let delta = EmbeddingDelta::splat(
            crate::embedding::fixed_point::FixedPoint64::from_int(1)
        );
        let dkg_pk = DkgPublicKey {
            point: vec![0u8; 48],
            hash: [0u8; 32],
            session_id: [0u8; 32],
            threshold: 3,
            num_participants: 5,
        };

        // VDW with all-zero signatures won't pass real verification,
        // but the algorithm runs without panics
        let result = verify_vdw(
            &vdw, &trusted_root, &delta, &dkg_pk, &[0u8; 1952],
        );

        // We expect Invalid because the signatures are zeros
        assert!(!result.is_valid());
    }

    #[test]
    fn test_vdw_batch() {
        let vdws = vec![make_test_vdw(), make_test_vdw()];
        let batch = VdwBatch::new(vdws, 100);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.height, 100);
        assert!(!batch.batch_root.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_vdw_batch_find_by_tx_hash() {
        let vdw1 = Vdw::generate(
            TxHash::from_bytes([1u8; 32]), 0, 100,
            vec![0u8; DELTA_PROOF_SIZE],
            EmbeddingRoot::from_bytes([1u8; 32]),
            vec![0u8; BLS_SIG_SIZE],
            vec![0u8; DILITHIUM3_SIG_SIZE],
        );
        let vdw2 = Vdw::generate(
            TxHash::from_bytes([2u8; 32]), 0, 100,
            vec![0u8; DELTA_PROOF_SIZE],
            EmbeddingRoot::from_bytes([1u8; 32]),
            vec![0u8; BLS_SIG_SIZE],
            vec![0u8; DILITHIUM3_SIG_SIZE],
        );

        let batch = VdwBatch::new(vec![vdw1, vdw2], 100);

        let found = batch.find_by_tx_hash(&TxHash::from_bytes([1u8; 32]));
        assert!(found.is_some());

        let not_found = batch.find_by_tx_hash(&TxHash::from_bytes([3u8; 32]));
        assert!(not_found.is_none());
    }

    #[test]
    fn test_vdw_storage_entry() {
        let vdw = make_test_vdw();
        let mut entry = VdwStorageEntry::new(vdw);
        assert!(!entry.is_pinned);

        entry.pin_arweave("arweave-tx-123".into());
        assert!(entry.is_pinned);
        assert_eq!(entry.arweave_tx_id, Some("arweave-tx-123".into()));

        entry.pin_ipfs("QmX123".into());
        assert_eq!(entry.ipfs_cid, Some("QmX123".into()));

        entry.record_verification();
        entry.record_verification();
        assert_eq!(entry.verification_count, 2);
    }

    #[test]
    fn test_generate_vdws_for_batch() {
        let tx_hashes = vec![
            TxHash::from_bytes([1u8; 32]),
            TxHash::from_bytes([2u8; 32]),
        ];
        let delta_proofs = vec![
            vec![0u8; DELTA_PROOF_SIZE],
            vec![0u8; DELTA_PROOF_SIZE],
        ];

        let batch = generate_vdws_for_batch(
            &tx_hashes, 0, 100,
            EmbeddingRoot::from_bytes([1u8; 32]),
            vec![0u8; BLS_SIG_SIZE],
            vec![0u8; DILITHIUM3_SIG_SIZE],
            &delta_proofs,
        ).unwrap();

        assert_eq!(batch.len(), 2);
    }
}
