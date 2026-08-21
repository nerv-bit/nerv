//! Unit tests for DKG and Threshold Decryption.
//!
//! Validates the NERV V2.0 pure-cryptographic privacy layer. Ensures that
//! the Feldman VSS polynomial math correctly computes shares, that Lagrange
//! interpolation accurately reconstructs secrets, and that the threshold
//! decryption ceremony successfully decrypts a payload only when >= t
//! validators participate.

use nerv::privacy::dkg::{
    DkgScalar, Polynomial, FeldmanCommitment, DkgPublicKey, 
    ThresholdCiphertext, lagrange_coefficient, reconstruct_secret,
};
use nerv::privacy::threshold_dec::{PartialDecryption, DecryptionCeremony};
use nerv::ValidatorId;
use bls12_381::Scalar as BlsScalar;
use crate::NervResult;

// ─── Polynomial & Scalar Math Tests ──────────────────────────────────────

#[test]
fn test_polynomial_evaluation() {
    // f(x) = 5 + 3x
    let poly = Polynomial {
        coefficients: vec![
            DkgScalar::from_bls_scalar(&BlsScalar::from(5)),
            DkgScalar::from_bls_scalar(&BlsScalar::from(3)),
        ],
    };
    
    // f(2) = 5 + 3(2) = 11
    let x = DkgScalar::from_bls_scalar(&BlsScalar::from(2));
    let result = poly.evaluate(&x);
    assert_eq!(result.to_bls_scalar(), BlsScalar::from(11));
}

#[test]
fn test_feldman_commitment_verify_share() {
    // f(x) = 10 + 4x
    let poly = Polynomial {
        coefficients: vec![
            DkgScalar::from_bls_scalar(&BlsScalar::from(10)),
            DkgScalar::from_bls_scalar(&BlsScalar::from(4)),
        ],
    };
    
    // Generate commitments for coefficients
    let commitments: Vec<FeldmanCommitment> = poly.coefficients.iter()
        .map(FeldmanCommitment::commit)
        .collect();
    
    // Verify share for participant 1: f(1) = 14
    let share_1 = poly.evaluate(&DkgScalar::from_bls_scalar(&BlsScalar::from(1)));
    assert!(
        FeldmanCommitment::verify_share(&commitments, 1, &share_1),
        "Valid share must pass Feldman verification"
    );
    
    // Verify share for participant 2: f(2) = 18
    let share_2 = poly.evaluate(&DkgScalar::from_bls_scalar(&BlsScalar::from(2)));
    assert!(
        FeldmanCommitment::verify_share(&commitments, 2, &share_2),
        "Valid share for participant 2 must pass verification"
    );
    
    // Tampered share must fail
    let bad_share = DkgScalar::from_bls_scalar(&BlsScalar::from(99));
    assert!(
        !FeldmanCommitment::verify_share(&commitments, 1, &bad_share),
        "Tampered share must fail verification"
    );
}

// ─── Lagrange Interpolation & Secret Reconstruction Tests ────────────────

#[test]
fn test_lagrange_coefficients_sum_to_one() {
    // For any set of indices, the Lagrange coefficients must sum to 1
    let indices = vec![1u32, 2, 3, 4];
    let mut sum = BlsScalar::ZERO;
    
    for &i in &indices {
        let lambda = lagrange_coefficient(&indices, i);
        sum += lambda.to_bls_scalar();
    }
    
    assert_eq!(sum, BlsScalar::ONE, "Lagrange coefficients must sum to 1");
}

#[test]
fn test_secret_reconstruction_any_subset() {
    // Create a degree-2 polynomial (requires t=3 shares to reconstruct)
    // f(x) = 42 + 5x + 3x^2
    let poly = Polynomial {
        coefficients: vec![
            DkgScalar::from_bls_scalar(&BlsScalar::from(42)),
            DkgScalar::from_bls_scalar(&BlsScalar::from(5)),
            DkgScalar::from_bls_scalar(&BlsScalar::from(3)),
        ],
    };
    
    // Generate 5 shares
    let all_shares: Vec<(u32, DkgScalar)> = (1..=5).map(|i| {
        let x = DkgScalar::from_bls_scalar(&BlsScalar::from(i as u64));
        (i as u32, poly.evaluate(&x))
    }).collect();
    
    // Reconstruct using different subsets of 3 shares
    let secret1 = reconstruct_secret(&all_shares[0..3]);
    let secret2 = reconstruct_secret(&all_shares[2..5]);
    let secret3 = reconstruct_secret(&vec![all_shares[0].clone(), all_shares[3].clone(), all_shares[4].clone()]);
    
    // All reconstructions must yield the original secret (f(0) = 42)
    assert_eq!(secret1.to_bls_scalar(), BlsScalar::from(42), "Subset 1 must reconstruct secret");
    assert_eq!(secret2.to_bls_scalar(), BlsScalar::from(42), "Subset 2 must reconstruct secret");
    assert_eq!(secret3.to_bls_scalar(), BlsScalar::from(42), "Subset 3 must reconstruct secret");
}

// ─── Threshold Decryption Ceremony Tests ─────────────────────────────────

#[test]
fn test_threshold_encryption_and_decryption_success() {
    // 1. Setup DKG with threshold t=3, n=5
    let secret = DkgScalar::from_bls_scalar(&BlsScalar::from(42));
    let commitment = FeldmanCommitment::commit(&secret);
    let dkg_pk = DkgPublicKey {
        point: commitment.point,
        hash: nerv::utils::blake3_hash(&commitment.point),
        session_id: [0u8; 32],
        threshold: 3,
        num_participants: 5,
    };
    
    // Generate polynomial shares for 3 participants
    let poly = Polynomial {
        coefficients: vec![
            secret,
            DkgScalar::from_bls_scalar(&BlsScalar::from(10)),
            DkgScalar::from_bls_scalar(&BlsScalar::from(20)),
        ],
    };
    
    let shares: Vec<(u32, DkgScalar)> = (1..=3).map(|i| {
        let x = DkgScalar::from_bls_scalar(&BlsScalar::from(i as u64));
        (i as u32, poly.evaluate(&x))
    }).collect();
    
    // 2. Encrypt a mock mempool payload
    let payload = b"private transaction data block 12345";
    let ciphertext = ThresholdCiphertext::encrypt(payload, &dkg_pk, [0u8; 32]).unwrap();
    
    // 3. Compute partial decryptions from 3 validators
    let mut ceremony = DecryptionCeremony::new(ciphertext.clone(), 3);
    
    for (index, share) in &shares {
        let val_id = ValidatorId::from_bytes([*index as u8; 32]);
        let dkg_share = nerv::privacy::dkg::DkgSecretShare::new(val_id.clone(), *index, share.clone(), [0u8; 32]);
        let partial = PartialDecryption::compute(&ciphertext, &dkg_share, val_id);
        ceremony.add_partial(partial).unwrap();
    }
    
    assert!(ceremony.is_ready(), "Ceremony must be ready with t partials");
    
    // 4. Combine partials to decrypt
    let decrypted_payload = ceremony.combine().unwrap();
    
    assert_eq!(
        decrypted_payload, payload.to_vec(),
        "Threshold decryption must yield the exact original payload"
    );
}

#[test]
fn test_threshold_decryption_fails_with_insufficient_partials() {
    let secret = DkgScalar::from_bls_scalar(&BlsScalar::from(42));
    let commitment = FeldmanCommitment::commit(&secret);
    let dkg_pk = DkgPublicKey {
        point: commitment.point,
        hash: nerv::utils::blake3_hash(&commitment.point),
        session_id: [0u8; 32],
        threshold: 3,
        num_participants: 5,
    };
    
    let payload = b"secret tx";
    let ciphertext = ThresholdCiphertext::encrypt(payload, &dkg_pk, [0u8; 32]).unwrap();
    
    // Attempt to decrypt with only 2 partials (t-1)
    let mut ceremony = DecryptionCeremony::new(ciphertext, 3);
    
    // Add only 2 partials
    let poly = Polynomial {
        coefficients: vec![
            secret,
            DkgScalar::from_bls_scalar(&BlsScalar::from(10)),
            DkgScalar::from_bls_scalar(&BlsScalar::from(20)),
        ],
    };
    
    for i in 1..=2 {
        let share = poly.evaluate(&DkgScalar::from_bls_scalar(&BlsScalar::from(i as u64)));
        let val_id = ValidatorId::from_bytes([i as u8; 32]);
        let dkg_share = nerv::privacy::dkg::DkgSecretShare::new(val_id.clone(), i as u32, share, [0u8; 32]);
        let partial = PartialDecryption::compute(&ceremony.ciphertext, &dkg_share, val_id);
        ceremony.add_partial(partial).unwrap();
    }
    
    assert!(!ceremony.is_ready(), "Ceremony must not be ready with < t partials");
    
    let result = ceremony.combine();
    assert!(result.is_err(), "Combination must fail with insufficient partials");
}

