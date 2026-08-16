Rust// tests/unit/test_crypto.rs
// ============================================================================
// CRYPTO MODULE UNIT TESTS
// ============================================================================

use nerv_bit::crypto::{
    dilithium::{Dilithium3, DilithiumError},
    ml_kem::{MlKem768, MlKemError},
    sphincs::{SphincsPlus, SphincsError},
    hash::{NervHasher, HashStrategy},
    CryptoProvider, CryptoError,
};
use pasta_curves::pallas::Scalar as Fp;
use rand::{thread_rng, RngCore};
use rand_core::OsRng;

#[test]
fn test_dilithium_keypair_sign_verify() {
    let mut rng = OsRng;

    let (pk, sk) = Dilithium3::keypair(&mut rng);

    assert_eq!(pk.len(), Dilithium3::PUBLIC_KEY_SIZE);
    assert_eq!(sk.len(), Dilithium3::SECRET_KEY_SIZE);

    let message = b"NERV Dilithium test message - the future is post-quantum";

    let signature = Dilithium3::sign(&sk, message).unwrap();
    assert!(signature.len() <= Dilithium3::SIGNATURE_SIZE);

    // Valid signature
    assert!(Dilithium3::verify(&pk, message, &signature).unwrap());

    // Tampered message
    let tampered = b"NERV Dilithium tampered message";
    assert!(!Dilithium3::verify(&pk, tampered, &signature).unwrap());

    // Invalid signature length (truncated)
    let truncated = &signature[..signature.len() / 2];
    assert!(Dilithium3::verify(&pk, message, truncated).is_err());
}

#[test]
fn test_ml_kem_keypair_encaps_decaps() {
    let mut rng = OsRng;

    let (pk, sk) = MlKem768::keypair(&mut rng).unwrap();

    assert_eq!(pk.len(), MlKem768::PUBLIC_KEY_SIZE);
    assert_eq!(sk.len(), MlKem768::SECRET_KEY_SIZE);

    let (ciphertext, shared_secret_sender) = MlKem768::encapsulate(&pk).unwrap();

    assert_eq!(ciphertext.len(), MlKem768::CIPHERTEXT_SIZE);
    assert_eq!(shared_secret_sender.len(), MlKem768::SHARED_SECRET_SIZE);

    let shared_secret_receiver = MlKem768::decapsulate(&sk, &ciphertext).unwrap();

    assert_eq!(shared_secret_sender, shared_secret_receiver);

    // Invalid ciphertext (truncated)
    let bad_ct = &ciphertext[..ciphertext.len() / 2];
    assert!(MlKem768::decapsulate(&sk, bad_ct).is_err());

    // Invalid pk length
    let bad_pk = &pk[..pk.len() / 2];
    assert!(MlKem768::encapsulate(bad_pk).is_err());
}

#[test]
fn test_sphincs_keypair_sign_verify() {
    // SPHINCS+-simple is deterministic - no RNG needed
    let (pk, sk) = SphincsPlus::keypair();

    assert_eq!(pk.len(), SphincsPlus::PUBLIC_KEY_SIZE);   // 48
    assert_eq!(sk.len(), SphincsPlus::SECRET_KEY_SIZE);   // 128

    let message = b"NERV SPHINCS+ backup signature test";

    let signature = SphincsPlus::sign(&sk, message).unwrap();
    assert_eq!(signature.len(), SphincsPlus::SIGNATURE_SIZE); // 17088

    // Valid
    assert!(SphincsPlus::verify(&pk, message, &signature).unwrap());

    // Tampered message
    let tampered = b"tampered";
    assert!(!SphincsPlus::verify(&pk, tampered, &signature).unwrap());

    // Invalid lengths
    let bad_pk = &pk[..pk.len() / 2];
    assert!(SphincsPlus::verify(bad_pk, message, &signature).is_err());

    let bad_sig = &signature[..signature.len() / 2];
    assert!(SphincsPlus::verify(&pk, message, bad_sig).is_err());
}

#[test]
fn test_hash_blake3_strategy() {
    let hasher = NervHasher::new(HashStrategy::Blake3);

    let data1 = b"identical input";
    let data2 = b"different input";

    let hash1_a = hasher.hash_bytes(data1);
    let hash1_b = hasher.hash_bytes(data1);
    let hash2 = hasher.hash_bytes(data2);

    assert_eq!(hash1_a, hash1_b);
    assert_ne!(hash1_a, hash2);
}

#[test]
fn test_hash_poseidon_embedding() {
    let hasher = NervHasher::new(HashStrategy::Poseidon);

    // Zero embedding
    let embedding_zero: Vec<Fp> = vec![Fp::zero(); 512];

    // Modified embedding
    let mut embedding_modified = embedding_zero.clone();
    embedding_modified[0] = Fp::one();
    embedding_modified[511] = Fp::from(42);

    let hash_zero_a = hasher.hash_embedding(&embedding_zero);
    let hash_zero_b = hasher.hash_embedding(&embedding_zero);
    let hash_mod = hasher.hash_embedding(&embedding_modified);

    assert_eq!(hash_zero_a, hash_zero_b);
    assert_ne!(hash_zero_a, hash_mod);

    // Wrong length
    let short = vec![Fp::zero(); 511];
    assert!(hasher.hash_embedding(&short).is_err());
}

#[test]
fn test_hash_poseidon_bytes() {
    let hasher = NervHasher::new(HashStrategy::Poseidon);

    let data1 = b"poseidon byte hash test";
    let data2 = b"different data";

    let hash1_a = hasher.hash_bytes(data1);
    let hash1_b = hasher.hash_bytes(data1);
    let hash2 = hasher.hash_bytes(data2);

    assert_eq!(hash1_a, hash1_b);
    assert_ne!(hash1_a, hash2);
}

#[test]
fn test_crypto_provider_initialization_and_keygen() {
    let mut provider = CryptoProvider::new().unwrap();

    // Signature keypair (Dilithium)
    let sig_kp = provider.generate_signature_keypair().unwrap();
    assert_eq!(sig_kp.pk.len(), Dilithium3::PUBLIC_KEY_SIZE);

    // We don't have a direct generate_encryption_keypair in the snippet,
    // but we can test the underlying ml_kem via direct call
    let mut rng = OsRng;
    let (enc_pk, _enc_sk) = MlKem768::keypair(&mut rng).unwrap();
    assert_eq!(enc_pk.len(), MlKem768::PUBLIC_KEY_SIZE);
}

#[test]
fn test_crypto_provider_utilities() {
    let provider = CryptoProvider::new().unwrap();

    // HKDF
    let ikm = vec![0u8; 32];
    let salt = vec![1u8; 16];
    let info = b"NERV HKDF test";
    let okm = provider.hkdf(&salt, &ikm, info, 64).unwrap();
    assert_eq!(okm.len(), 64);

    // XOR
    let a = vec![0xff, 0xaa, 0x55];
    let b = vec![0x0f, 0xaa, 0x00];
    let expected = vec![0xf0, 0x00, 0x55];
    assert_eq!(provider.xor_bytes(&a, &b).unwrap(), expected);

    // Length mismatch error
    let c = vec![0u8; 2];
    assert!(provider.xor_bytes(&a, &c).is_err());

    // Hex / Base64
    let bytes = vec![0x00, 0x01, 0xde, 0xad, 0xbe, 0xef];
    let hex_str = CryptoProvider::bytes_to_hex(&bytes);
    assert_eq!(hex_str, "0001deadbeef");
    assert_eq!(CryptoProvider::hex_to_bytes(&hex_str).unwrap(), bytes);

    let b64_str = CryptoProvider::base64_encode(&bytes);
    assert_eq!(CryptoProvider::base64_decode(&b64_str).unwrap(), bytes);

    // Invalid hex
    assert!(CryptoProvider::hex_to_bytes("invalid_hex").is_err());

    // Padding
    let data = vec![1, 2, 3, 4, 5];
    let block_size = 8;
    let padded = CryptoProvider::pad_to_block(&data, block_size);
    assert_eq!(padded.len(), 8);
    assert_eq!(&padded[5..], &[0u8; 3]);

    let unpadded = CryptoProvider::unpad_from_block(&padded, block_size).unwrap();
    assert_eq!(unpadded, data);

    // No padding needed (exact multiple)
    let exact = vec![0u8; 16];
    let padded_exact = CryptoProvider::pad_to_block(&exact, block_size);
    assert_eq!(padded_exact.len(), 24); // Adds full block of zeros if already multiple

    // Invalid unpad (not multiple)
    let bad = vec![0u8; 9];
    assert!(CryptoProvider::unpad_from_block(&bad, block_size).is_err());
}