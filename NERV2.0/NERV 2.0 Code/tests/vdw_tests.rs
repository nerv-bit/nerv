//! Unit tests for Verifiable Delay Witnesses (VDWs).
//!
//! Validates the NERV V2.0 pure-cryptographic receipt system. Ensures that
//! VDWs are correctly generated, serialized, and that the offline verification
//! pipeline (DKG signature check -> Dilithium signature check -> Halo2 proof
//! check -> Homomorphic root reconciliation) accurately accepts valid witnesses
//! and securely rejects tampered ones.

use nerv::privacy::vdw::Vdw;
use nerv::privacy::dkg::DkgPublicKey;
use nerv::wallets::keys::DilithiumKeypair;
use nerv::wallets::vdw_cache::{verify_vdw_offline, VdwStatus};
use nerv::{TxHash, EmbeddingRoot, NervResult};
use bls12_381::{G1Affine, G2Affine, pairing, G2Projective};
use group::Group;

// ─── VDW Generation & Serialization Tests ────────────────────────────────

#[test]
fn test_vdw_generation_and_size() {
    let vdw = Vdw::generate(
        TxHash::from_bytes([1u8; 32]),
        0, // shard_id
        100, // lattice_height
        vec![0u8; 400], // delta_proof
        EmbeddingRoot::from_bytes([2u8; 32]), // embedding_root
        vec![0u8; 48], // dkg_threshold_sig
        vec![0u8; 3293], // dilithium_sig
    );

    assert_eq!(vdw.version, 2, "VDW must be version 2");
    assert_eq!(vdw.shard_id, 0);
    assert_eq!(vdw.lattice_height, 100);
    assert!(vdw.size() <= 3600, "VDW must not exceed 3600 bytes");
    assert!(vdw.is_valid_size(), "VDW must pass valid size check");
}

#[test]
fn test_vdw_serialization_and_base64() {
    let vdw = Vdw::generate(
        TxHash::from_bytes([3u8; 32]),
        1,
        200,
        vec![1u8; 400],
        EmbeddingRoot::from_bytes([4u8; 32]),
        vec![2u8; 48],
        vec![3u8; 3293],
    );

    // Serialize to bytes
    let bytes = vdw.to_bytes();
    assert!(!bytes.is_empty());

    // Deserialize from bytes
    let deserialized = Vdw::from_bytes(&bytes).unwrap();
    assert_eq!(vdw, deserialized, "Deserialized VDW must match original");

    // Base64 export/import
    let b64 = vdw.to_base64();
    let from_b64 = Vdw::from_base64(&b64).unwrap();
    assert_eq!(vdw, from_b64, "Base64 imported VDW must match original");
}

// ─── Offline Verification Pipeline Tests ─────────────────────────────────

/// Helper to create a structurally valid VDW that will pass all cryptographic checks.
/// We use BLS generators for the DKG signature to bypass complex DKG setup,
/// ensuring the pairing math evaluates to true.
fn create_valid_vdw_and_keys() -> (Vdw, DkgPublicKey, Vec<u8>, [u8; 32], [u8; 32]) {
    let validator_kp = DilithiumKeypair::generate().unwrap();
    
    // Use BLS generators to mock a valid DKG signature pair
    // e(sig, G2::gen) == e(pk, H(msg))  -->  e(G1::gen, G2::gen) == e(G1::gen, G2::gen)
    let g1_gen = G1Affine::generator();
    let dkg_pk_point = g1_gen.to_compressed().as_ref().to_vec();
    let dkg_sig = g1_gen.to_compressed().as_ref().to_vec();
    
    let dkg_pk = DkgPublicKey {
        point: dkg_pk_point,
        hash: nerv::utils::blake3_hash(&dkg_pk_point),
        session_id: [0u8; 32],
        threshold: 3,
        num_participants: 5,
    };

    let trusted_prev_root = EmbeddingRoot::from_bytes([10u8; 32]);
    let proven_delta = [20u8; 32];
    
    // Compute the correct new root: Hash(prev_root + delta)
    let mut concat = Vec::with_capacity(64);
    concat.extend_from_slice(trusted_prev_root.as_bytes());
    concat.extend_from_slice(&proven_delta);
    let new_root = nerv::utils::blake3_hash(&concat);

    let mut vdw = Vdw::generate(
        TxHash::from_bytes([1u8; 32]),
        0,
        100,
        vec![0u8; 400],
        EmbeddingRoot::from_bytes(new_root),
        dkg_sig,
        vec![], // Placeholder, will be signed below
    );

    // Sign the VDW payload with Dilithium-3
    // Message: tx_hash || shard_id || height || timestamp
    let mut msg = Vec::with_capacity(56);
    msg.extend_from_slice(vdw.tx_hash.as_bytes());
    msg.extend_from_slice(&vdw.shard_id.to_le_bytes());
    msg.extend_from_slice(&vdw.lattice_height.to_le_bytes());
    msg.extend_from_slice(&vdw.timestamp_ms.to_le_bytes());
    
    vdw.dilithium_sig = validator_kp.sign(&msg).unwrap();

    (vdw, dkg_pk, validator_kp.public_key, trusted_prev_root.0, proven_delta)
}

#[test]
fn test_vdw_offline_verification_valid() {
    let (vdw, dkg_pk, validator_pk, prev_root, delta) = create_valid_vdw_and_keys();
    
    let status = verify_vdw_offline(
        &vdw,
        &dkg_pk,
        &validator_pk,
        &EmbeddingRoot::from_bytes(prev_root),
        &delta,
    );
    
    assert!(matches!(status, VdwStatus::Valid), "Valid VDW must pass all verification steps. Got: {:?}", status);
}

#[test]
fn test_vdw_offline_verification_root_mismatch() {
    let (mut vdw, dkg_pk, validator_pk, prev_root, delta) = create_valid_vdw_and_keys();
    
    // Tamper with the embedding root
    vdw.embedding_root = EmbeddingRoot::from_bytes([99u8; 32]);
    
    // Need to re-sign because the root changed? No, the Dilithium sig doesn't cover the root in our mock,
    // but the homomorphic reconciliation will catch it.
    let status = verify_vdw_offline(
        &vdw,
        &dkg_pk,
        &validator_pk,
        &EmbeddingRoot::from_bytes(prev_root),
        &delta,
    );
    
    assert!(matches!(status, VdwStatus::Invalid(ref r) if r.contains("root reconciliation failed")), "Must fail on root mismatch. Got: {:?}", status);
}

#[test]
fn test_vdw_offline_verification_dilithium_tampered() {
    let (mut vdw, dkg_pk, validator_pk, prev_root, delta) = create_valid_vdw_and_keys();
    
    // Tamper with the Dilithium signature
    vdw.dilithium_sig[0] ^= 0xFF;
    
    let status = verify_vdw_offline(
        &vdw,
        &dkg_pk,
        &validator_pk,
        &EmbeddingRoot::from_bytes(prev_root),
        &delta,
    );
    
    assert!(matches!(status, VdwStatus::Invalid(ref r) if r.contains("Dilithium-3")), "Must fail on tampered Dilithium signature. Got: {:?}", status);
}

#[test]
fn test_vdw_offline_verification_dkg_tampered() {
    let (mut vdw, dkg_pk, validator_pk, prev_root, delta) = create_valid_vdw_and_keys();
    
    // Tamper with the DKG threshold signature (make it an invalid G1 point)
    vdw.dkg_threshold_sig[0] ^= 0xFF;
    
    let status = verify_vdw_offline(
        &vdw,
        &dkg_pk,
        &validator_pk,
        &EmbeddingRoot::from_bytes(prev_root),
        &delta,
    );
    
    assert!(matches!(status, VdwStatus::Invalid(ref r) if r.contains("DKG threshold signature")), "Must fail on tampered DKG signature. Got: {:?}", status);
}

#[test]
fn test_vdw_offline_verification_wrong_validator_pk() {
    let (vdw, dkg_pk, _validator_pk, prev_root, delta) = create_valid_vdw_and_keys();
    
    // Generate a different validator keypair
    let wrong_kp = DilithiumKeypair::generate().unwrap();
    
    let status = verify_vdw_offline(
        &vdw,
        &dkg_pk,
        &wrong_kp.public_key,
        &EmbeddingRoot::from_bytes(prev_root),
        &delta,
    );
    
    assert!(matches!(status, VdwStatus::Invalid(ref r) if r.contains("Dilithium-3")), "Must fail with wrong validator PK. Got: {:?}", status);
}
