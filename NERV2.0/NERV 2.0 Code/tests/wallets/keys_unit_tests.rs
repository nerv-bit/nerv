//! Unit tests for Post-Quantum Key Management.
//!
//! Validates the NERV V2.0 HD wallet key generation, mnemonic seed derivation,
//! and the core Post-Quantum cryptographic primitives (Dilithium-3 and ML-KEM-768).

use nerv::wallets::keys::{Mnemonic, WalletKeys, DerivationPath, DilithiumKeypair, MlKemKeypair};

#[test]
fn test_mnemonic_generation_and_entropy() {
    // Generate a new mnemonic
    let mnemonic = Mnemonic::generate();
    
    // Entropy must be exactly 32 bytes (256-bit) for BIP-39 24-word phrases
    let entropy = mnemonic.entropy();
    assert_eq!(entropy.len(), 32, "Mnemonic entropy must be 32 bytes");
    
    // Two consecutive generations must not yield the same entropy
    let mnemonic2 = Mnemonic::generate();
    assert_ne!(
        mnemonic.entropy(),
        mnemonic2.entropy(),
        "Mnemonic entropy must be cryptographically random"
    );
}

#[test]
fn test_mnemonic_to_seed_determinism() {
    let mnemonic = Mnemonic::generate();
    
    // Deriving the seed without a passphrase must be deterministic
    let seed1 = mnemonic.to_seed("");
    let seed2 = mnemonic.to_seed("");
    assert_eq!(seed1, seed2, "Seeds without passphrase must match");
    
    // Deriving with a passphrase must yield a different seed
    let seed3 = mnemonic.to_seed("extra_passphrase");
    assert_ne!(seed1, seed3, "Seeds with different passphrases must differ");
    
    // Same passphrase must yield the same seed
    let seed4 = mnemonic.to_seed("extra_passphrase");
    assert_eq!(seed3, seed4, "Seeds with same passphrase must match");
}

#[test]
fn test_mnemonic_from_entropy() {
    let mut entropy = [0u8; 32];
    // Fill with dummy data for deterministic test
    for i in 0..32 {
        entropy[i] = i as u8;
    }
    
    let mnemonic = Mnemonic::from_entropy(entropy);
    assert_eq!(mnemonic.entropy(), &entropy, "Reconstructed mnemonic entropy must match");
    
    let seed = mnemonic.to_seed("");
    assert_eq!(seed.len(), 64, "Seed must be 64 bytes");
}

#[test]
fn test_dilithium_signing_and_verification() {
    // Generate keypair
    let keypair = DilithiumKeypair::generate().expect("Dilithium keygen failed");
    
    // Message to sign
    let msg = b"NERV V2.0: The Self-Evolving Blockchain";
    
    // Sign the message
    let signature = keypair.sign(msg).expect("Dilithium signing failed");
    
    // Verify the signature
    assert!(
        keypair.verify(msg, &signature),
        "Valid signature must verify successfully"
    );
    
    // Tampered message must fail verification
    let bad_msg = b"NERV V1.01: The Old Blockchain";
    assert!(
        !keypair.verify(bad_msg, &signature),
        "Tampered message must fail verification"
    );
    
    // Tampered signature must fail verification
    let mut bad_sig = signature.clone();
    bad_sig[0] ^= 0xFF;
    assert!(
        !keypair.verify(msg, &bad_sig),
        "Tampered signature must fail verification"
    );
}

#[test]
fn test_ml_kem_encapsulation_and_decapsulation() {
    // Generate keypair
    let keypair = MlKemKeypair::generate().expect("ML-KEM keygen failed");
    
    // Encapsulate a shared secret using the Public Key
    let (ciphertext, shared_secret_sender) = keypair.encapsulate().expect("ML-KEM encapsulate failed");
    
    // Decapsulate the shared secret using the Secret Key
    let shared_secret_receiver = keypair.decapsulate(&ciphertext).expect("ML-KEM decapsulate failed");
    
    // Both parties must have the exact same shared secret
    assert_eq!(
        shared_secret_sender, shared_secret_receiver,
        "Encapsulated and decapsulated shared secrets must match"
    );
}

#[test]
fn test_ml_kem_decapsulation_fails_with_wrong_ct() {
    let keypair = MlKemKeypair::generate().unwrap();
    let (mut ciphertext, _) = keypair.encapsulate().unwrap();
    
    // Tamper with the ciphertext
    ciphertext[0] ^= 0xFF;
    
    // Decapsulation must fail for a tampered ciphertext
    let result = keypair.decapsulate(&ciphertext);
    assert!(
        result.is_err(),
        "ML-KEM decapsulation must fail for tampered ciphertext"
    );
}

#[test]
fn test_wallet_keys_initialization() {
    let mnemonic = Mnemonic::generate();
    let seed = mnemonic.to_seed("");
    
    // Initialize wallet keys from the seed
    let wallet_keys = WalletKeys::new_from_seed(&seed).expect("WalletKeys init failed");
    
    // Master seed must match the input seed
    assert_eq!(wallet_keys.master_seed, seed.to_vec(), "Master seed must be stored correctly");
    
    // Spending and KEM keypairs must be populated
    assert!(!wallet_keys.spending_keypair.public_key.is_empty(), "Dilithium PK must exist");
    assert!(!wallet_keys.spending_keypair.secret_key.is_empty(), "Dilithium SK must exist");
    assert!(!wallet_keys.kem_keypair.public_key.is_empty(), "ML-KEM PK must exist");
    assert!(!wallet_keys.kem_keypair.secret_key.is_empty(), "ML-KEM SK must exist");
}

#[test]
fn test_detection_key_derivation_determinism() {
    let mnemonic = Mnemonic::generate();
    let seed = mnemonic.to_seed("");
    let wallet_keys = WalletKeys::new_from_seed(&seed).unwrap();
    
    let path = DerivationPath::default();
    
    // Derive the key twice
    let det_key1 = wallet_keys.derive_detection_key(&path);
    let det_key2 = wallet_keys.derive_detection_key(&path);
    
    // Must be 32 bytes
    assert_eq!(det_key1.len(), 32, "Detection key must be 32 bytes");
    
    // Must be deterministic
    assert_eq!(det_key1, det_key2, "Same derivation path must yield same detection key");
}

#[test]
fn test_detection_key_derivation_uniqueness() {
    let mnemonic = Mnemonic::generate();
    let seed = mnemonic.to_seed("");
    let wallet_keys = WalletKeys::new_from_seed(&seed).unwrap();
    
    let path1 = DerivationPath { index: 0, ..Default::default() };
    let path2 = DerivationPath { index: 1, ..Default::default() };
    let path3 = DerivationPath { account: 1, ..Default::default() };
    
    let det_key1 = wallet_keys.derive_detection_key(&path1);
    let det_key2 = wallet_keys.derive_detection_key(&path2);
    let det_key3 = wallet_keys.derive_detection_key(&path3);
    
    // Different paths must yield different keys
    assert_ne!(det_key1, det_key2, "Different indices must yield different keys");
    assert_ne!(det_key1, det_key3, "Different accounts must yield different keys");
    assert_ne!(det_key2, det_key3, "Keys must be distinct");
}

#[test]
fn test_diversified_receiver_derivation() {
    let mnemonic = Mnemonic::generate();
    let seed = mnemonic.to_seed("");
    let wallet_keys = WalletKeys::new_from_seed(&seed).unwrap();
    
    let receiver1 = wallet_keys.derive_diversified_receiver(0);
    let receiver2 = wallet_keys.derive_diversified_receiver(1);
    let receiver3 = wallet_keys.derive_diversified_receiver(0);
    
    assert_eq!(receiver1.len(), 32, "Diversified receiver must be 32 bytes");
    assert_ne!(receiver1, receiver2, "Different indices must yield different receivers");
    assert_eq!(receiver1, receiver3, "Same index must yield same receiver (deterministic)");
}
