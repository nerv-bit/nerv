// tests/unit/test_crypto.rs
// ============================================================================
// CRYPTO MODULE UNIT TESTS
// ============================================================================


use nerv_bit::crypto::*;
use nerv_bit::crypto::dilithium::*;
use nerv_bit::crypto::ml_kem::*;
use rand::Rng;


#[test]
fn test_dilithium_key_generation() {
    let config = DilithiumConfig::default();
    let dilithium = Dilithium3::new(config).unwrap();
    
    // Generate random bytes for RNG
    let mut rng_bytes = [0u8; 64];
    rand::thread_rng().fill(&mut rng_bytes);
    let mut rng_state = RngState::new(&rng_bytes).unwrap();
    
    // Generate keypair
    let keypair = dilithium.generate_keypair(&mut rng_state).unwrap();
    
    assert_eq!(keypair.public_key.rho.len(), 32);
    assert_eq!(keypair.public_key.t1.len() > 0, true);
    assert_eq!(keypair.public_key.hash.len(), 32);
    
    // Test serialization
    let pk_bytes = keypair.public_key.to_bytes();
    let sk_bytes = keypair.secret_key.to_bytes();
    
    assert!(pk_bytes.len() >= 1472, "Public key too small: {} bytes", pk_bytes.len());
    assert!(sk_bytes.len() >= 3504, "Secret key too small: {} bytes", sk_bytes.len());
    
    // Test deserialization
    let pk_restored = DilithiumPublicKey::from_bytes(&pk_bytes).unwrap();
    let sk_restored = DilithiumSecretKey::from_bytes(&sk_bytes).unwrap();
    
    assert_eq!(pk_restored.rho, keypair.public_key.rho);
    assert_eq!(sk_restored.rho, keypair.secret_key.rho);
}


#[test]
fn test_ml_kem_key_encapsulation() {
    let config = MlKemConfig::default();
    let ml_kem = MlKem768::new(config).unwrap();
    
    // Generate random bytes for RNG
    let mut rng_bytes = [0u8; 64];
    rand::thread_rng().fill(&mut rng_bytes);
    let mut rng_state = RngState::new(&rng_bytes).unwrap();
    
    // Generate keypair
    let keypair = ml_kem.generate_keypair(&mut rng_state).unwrap();
    
    // Test encapsulation
    let (ciphertext, shared_secret1) = ml_kem.encapsulate(&keypair.public_key.to_bytes(), &mut rng_state).unwrap();
    
    assert!(ciphertext.u.len() > 0);
    assert!(ciphertext.v.len() > 0);
    assert_eq!(shared_secret1.len(), 32);
    
    // Test decapsulation
    let shared_secret2 = ml_kem.decapsulate(&ciphertext, &keypair.secret_key.to_bytes()).unwrap();
    
    assert_eq!(shared_secret1, shared_secret2, "Shared secrets should match");
    
    // Test with wrong ciphertext (should fail)
    let mut wrong_ciphertext = ciphertext.clone();
    wrong_ciphertext.u[0] = wrong_ciphertext.u[0].wrapping_add(1);
    
    let result = ml_kem.decapsulate(&wrong_ciphertext, &keypair.secret_key.to_bytes());
    assert!(result.is_err(), "Should fail with wrong ciphertext");
}


#[test]
fn test_crypto_utils() {
    // Test SHA3-256
    let data = b"hello world";
    let hash = utils::sha3_256(data);
    assert_eq!(hash.len(), 32);
    
    // Test BLAKE3
    let blake3_hash = utils::blake3_hash(data);
    assert_eq!(blake3_hash.len(), 32);
    
    // Test constant-time comparison
    let a = [1u8, 2, 3, 4];
    let b = [1u8, 2, 3, 4];
    let c = [1u8, 2, 3, 5];
    
    assert!(utils::constant_time_eq(&a, &b));
    assert!(!utils::constant_time_eq(&a, &c));
}


#[test]
fn test_signature_verification() {
    let config = DilithiumConfig::default();
    let dilithium = Dilithium3::new(config).unwrap();
    
    // Generate keypair
    let mut rng_bytes = [0u8; 64];
    rand::thread_rng().fill(&mut rng_bytes);
    let mut rng_state = RngState::new(&rng_bytes).unwrap();
    
    let keypair = dilithium.generate_keypair(&mut rng_state).unwrap();
    
    // Create message
    let message = b"Test message for Dilithium signature";
    
    // Sign message
    let signature = dilithium.sign(message, &keypair.secret_key.to_bytes(), &mut rng_state).unwrap();
    
    // Verify signature
    let is_valid = dilithium.verify(message, &signature, &keypair.public_key.to_bytes()).unwrap();
    assert!(is_valid, "Signature should be valid");
    
    // Test with wrong message
    let wrong_message = b"Wrong message";
    let is_valid = dilithium.verify(wrong_message, &signature, &keypair.public_key.to_bytes()).unwrap();
    assert!(!is_valid, "Should fail with wrong message");
    
    // Test with tampered signature
    let mut wrong_signature = signature.clone();
    if let Some(first_byte) = wrong_signature.z.first_mut() {
        *first_byte = first_byte.wrapping_add(1);
    }
    
    let is_valid = dilithium.verify(message, &wrong_signature, &keypair.public_key.to_bytes()).unwrap();
    assert!(!is_valid, "Should fail with tampered signature");
}


#[test]
fn test_key_derivation() {
    // Test public key derivation from secret key
    let config = DilithiumConfig::default();
    let dilithium = Dilithium3::new(config).unwrap();
    
    let mut rng_bytes = [0u8; 64];
    rand::thread_rng().fill(&mut rng_bytes);
    let mut rng_state = RngState::new(&rng_bytes).unwrap();
    
    let keypair = dilithium.generate_keypair(&mut rng_state).unwrap();
    
    // Derive public key from secret key
    let derived_pk = dilithium.public_key_from_secret(&keypair.secret_key.to_bytes()).unwrap();
    
    assert_eq!(derived_pk, keypair.public_key.to_bytes());
}


#[test]
fn test_crypto_configurations() {
    // Test different security levels
    let config_low = DilithiumConfig {
        security_parameter: 2,
        ..Default::default()
    };
    
    let config_high = DilithiumConfig {
        security_parameter: 5,
        ..Default::default()
    };
    
    assert!(Dilithium3::new(config_low).is_err(), "Should reject wrong security parameter");
    assert!(Dilithium3::new(config_high).is_err(), "Should reject wrong security parameter");
    
    // Test ML-KEM configurations
    let config_kem = MlKemConfig {
        security_parameter: 768,
        cpa_secure: true,
        deterministic_keygen: false,
    };
    
    assert!(MlKem768::new(config_kem).is_ok());
    
    let wrong_config = MlKemConfig {
        security_parameter: 512,
        ..Default::default()
    };
    
    assert!(MlKem768::new(wrong_config).is_err(), "Should reject wrong security parameter");
}
