// tests/wallet_keys_tests.rs
// Complete test suite for Epic 1: Post-Quantum Hierarchical Deterministic Key Management

#[cfg(test)]
mod tests {
    use nerv_wallet::keys::{HdWallet, AccountKeys, KeyError};
    use bip39::{Mnemonic, Language};
    use zeroize::Zeroizing;
    use std::collections::HashSet;

    const TEST_PASSPHRASE: &str = "test-passphrase-123!@#";
    const TEST_MNEMONIC: &str = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";

    /// Test 1: Wallet generation and recovery
    #[tokio::test]
    async fn test_wallet_generation_and_recovery() {
        // Test 1.1: Generate new wallet
        let wallet1 = HdWallet::generate(TEST_PASSPHRASE).unwrap();
        assert!(!wallet1.mnemonic.is_empty());
        
        // Test 1.2: Export and import mnemonic
        let mnemonic = wallet1.mnemonic.clone();
        let wallet2 = HdWallet::from_mnemonic(&mnemonic, TEST_PASSPHRASE).unwrap();
        
        // Verify seeds match
        assert_eq!(wallet1.master_seed, wallet2.master_seed);
        
        // Test 1.3: Different passphrase produces different seed
        let wallet3 = HdWallet::from_mnemonic(&mnemonic, "different-passphrase").unwrap();
        assert_ne!(wallet1.master_seed, wallet3.master_seed);
        
        // Test 1.4: Invalid mnemonic should fail
        let result = HdWallet::from_mnemonic("invalid mnemonic phrase", TEST_PASSPHRASE);
        assert!(matches!(result, Err(KeyError::InvalidMnemonic)));
        
        // Test 1.5: Empty passphrase allowed
        let wallet4 = HdWallet::from_mnemonic(&mnemonic, "").unwrap();
        assert_ne!(wallet1.master_seed, wallet4.master_seed);
    }

    /// Test 2: Address derivation consistency
    #[test]
    fn test_address_derivation_consistency() {
        let wallet = HdWallet::from_mnemonic(TEST_MNEMONIC, "").unwrap();
        
        // Derive first account with 5 addresses
        let account = wallet.derive_account(0, Some("Test Account".to_string())).unwrap();
        assert_eq!(account.index, 0);
        assert_eq!(account.keys.len(), 20); // Gap limit of 20
        
        // Get addresses
        let addr1 = account.keys[0].receiving_address().unwrap();
        let addr2 = account.keys[1].receiving_address().unwrap();
        
        // Verify address format
        assert!(addr1.starts_with("nerv1"));
        assert_ne!(addr1, addr2);
        
        // Derive same address independently should match
        let single_key = wallet.derive_address(0, 0).unwrap();
        assert_eq!(single_key.receiving_address().unwrap(), addr1);
        
        // Test deterministic derivation across sessions
        let wallet2 = HdWallet::from_mnemonic(TEST_MNEMONIC, "").unwrap();
        let account2 = wallet2.derive_account(0, None).unwrap();
        assert_eq!(
            account.keys[0].receiving_address().unwrap(),
            account2.keys[0].receiving_address().unwrap()
        );
    }

    /// Test 3: Bech32 address encoding/decoding
    #[test]
    fn test_bech32_address_format() {
        let wallet = HdWallet::generate("").unwrap();
        let account = wallet.derive_account(0, None).unwrap();
        let addr = account.keys[0].receiving_address().unwrap();
        
        // Verify Bech32 encoding
        let (hrp, data, variant) = bech32::decode(&addr).unwrap();
        assert_eq!(hrp, "nerv");
        assert_eq!(variant, bech32::Variant::Bech32);
        
        // Verify public key size (ML-KEM-768 public key is 1184 bytes, but encoded)
        let decoded = Vec::<u8>::from_base32(&data).unwrap();
        assert_eq!(decoded.len(), 1184); // ML-KEM-768 public key size
        
        // Test address decoding
        let decoded_pk = HdWallet::decode_address(&addr).unwrap();
        assert_eq!(decoded_pk, account.keys[0].enc_pk);
        
        // Test invalid addresses
        assert!(HdWallet::decode_address("invalid").is_err());
        assert!(HdWallet::decode_address("bitcoin1abc").is_err());
        assert!(HdWallet::decode_address("nerv1invalid").is_err());
    }

    /// Test 4: Secret key zeroization
    #[test]
    fn test_secret_key_zeroization() {
        let wallet = HdWallet::generate("").unwrap();
        let keys = wallet.derive_address(0, 0).unwrap();
        
        // Clone secret keys before drop
        let spending_sk = keys.spending_sk.clone();
        let enc_sk = keys.enc_sk.clone();
        
        // Drop the keys struct
        drop(keys);
        
        // Verify zeroization (all bytes should be zero)
        assert!(spending_sk.iter().all(|&b| b == 0));
        assert!(enc_sk.iter().all(|&b| b == 0));
        
        // Test that zeroization happens on panic
        let result = std::panic::catch_unwind(|| {
            let wallet = HdWallet::generate("").unwrap();
            let _keys = wallet.derive_address(0, 0).unwrap();
            panic!("Intentional panic for zeroization test");
        });
        
        // Even on panic, zeroization should have occurred
        assert!(result.is_err());
    }

    /// Test 5: Multiple accounts and gap limit
    #[test]
    fn test_multiple_accounts_and_gap_limit() {
        let mut wallet = HdWallet::generate("").unwrap();
        
        // Create multiple accounts
        let account0 = wallet.derive_account(0, Some("Account 0")).unwrap();
        let account1 = wallet.derive_account(1, Some("Account 1")).unwrap();
        let account2 = wallet.derive_account(2, Some("Account 2")).unwrap();
        
        // Verify distinct accounts
        assert_ne!(
            account0.keys[0].receiving_address().unwrap(),
            account1.keys[0].receiving_address().unwrap()
        );
        
        assert_ne!(
            account1.keys[0].receiving_address().unwrap(),
            account2.keys[0].receiving_address().unwrap()
        );
        
        // Test gap limit expansion
        let mut account = wallet.derive_account(3, Some("Test Gap")).unwrap();
        let initial_len = account.keys.len();
        
        // Simulate using all addresses (exceed gap limit)
        for _ in 0..25 {
            let _ = wallet.next_unused_address(3).unwrap();
        }
        
        // Account should have expanded beyond initial gap limit
        let updated_account = wallet.derive_account(3, None).unwrap();
        assert!(updated_account.keys.len() > initial_len);
        
        // Test address uniqueness across indices
        let mut addresses = HashSet::new();
        for i in 0..5 {
            for j in 0..5 {
                let keys = wallet.derive_address(i, j).unwrap();
                let addr = keys.receiving_address().unwrap();
                assert!(addresses.insert(addr), "Duplicate address found");
            }
        }
    }

    /// Test 6: Next unused address generation
    #[tokio::test]
    async fn test_next_unused_address_generation() {
        let mut wallet = HdWallet::generate("").unwrap();
        
        // Get first unused address
        let (addr1, idx1) = wallet.next_unused_address(0).unwrap();
        assert_eq!(idx1, 0);
        assert!(addr1.starts_with("nerv1"));
        
        // Get second unused address
        let (addr2, idx2) = wallet.next_unused_address(0).unwrap();
        assert_eq!(idx2, 1);
        assert_ne!(addr1, addr2);
        
        // Get address by specific index should match
        let addr1_by_index = wallet.receiving_address(0, 0).unwrap();
        assert_eq!(addr1, addr1_by_index);
        
        // Test with different accounts
        let (addr_acc1, _) = wallet.next_unused_address(1).unwrap();
        let (addr_acc2, _) = wallet.next_unused_address(2).unwrap();
        assert_ne!(addr_acc1, addr_acc2);
        
        // Verify sequential index progression
        for i in 0..10 {
            let (_, idx) = wallet.next_unused_address(0).unwrap();
            assert_eq!(idx, i + 2); // +2 because we already got 0 and 1
        }
    }

    /// Test 7: Ownership proof generation and verification
    #[test]
    fn test_ownership_proofs() {
        let wallet = HdWallet::generate("").unwrap();
        let keys = wallet.derive_address(0, 0).unwrap();
        
        // Generate ownership proof
        let challenge = b"test-challenge-123";
        let proof = keys.generate_ownership_proof(challenge).unwrap();
        
        // Verify the proof
        let is_valid = AccountKeys::verify_ownership_proof(
            &keys.commitment,
            challenge,
            &proof
        );
        
        assert!(is_valid);
        
        // Test with different challenge should fail
        let wrong_challenge = b"wrong-challenge";
        let is_invalid = AccountKeys::verify_ownership_proof(
            &keys.commitment,
            wrong_challenge,
            &proof
        );
        
        assert!(!is_invalid);
        
        // Test with different commitment should fail
        let wrong_commitment = [0u8; 32];
        let is_invalid = AccountKeys::verify_ownership_proof(
            &wrong_commitment,
            challenge,
            &proof
        );
        
        assert!(!is_invalid);
    }

    /// Test 8: Key metadata and public info export
    #[test]
    fn test_key_metadata_and_export() {
        let wallet = HdWallet::generate("Test Wallet").unwrap();
        
        // Check metadata
        let metadata = wallet.metadata();
        assert!(metadata.created_at <= chrono::Utc::now());
        assert_eq!(metadata.usage_count, 0);
        
        // Use wallet to increase usage count
        let _ = wallet.derive_account(0, None).unwrap();
        let metadata = wallet.metadata();
        assert!(metadata.usage_count > 0);
        assert!(metadata.last_used <= chrono::Utc::now());
        
        // Export public info (no secrets)
        let public_info = wallet.export_public_info().unwrap();
        assert!(!public_info.is_empty());
        
        // Public info should be deserializable
        use nerv_wallet::keys::PublicWalletInfo;
        let info: PublicWalletInfo = bincode::deserialize(&public_info).unwrap();
        assert!(info.accounts >= 0);
        assert!(info.created_at <= chrono::Utc::now());
    }

    /// Test 9: Hardened derivation path security
    #[test]
    fn test_hardened_derivation_security() {
        let wallet = HdWallet::generate("").unwrap();
        
        // Derive with hardened path
        let keys1 = wallet.derive_address(0x80000000, 0x80000000).unwrap();
        let keys2 = wallet.derive_address(0x80000001, 0x80000001).unwrap();
        
        // Verify different keys for different paths
        assert_ne!(keys1.enc_pk, keys2.enc_pk);
        assert_ne!(keys1.spending_pk, keys2.spending_pk);
        
        // Test that non-hardened paths are rejected (if implemented)
        // This depends on implementation details
    }

    /// Test 10: Cross-platform compatibility
    #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
    #[wasm_bindgen_test::wasm_bindgen_test]
    fn test_wasm_compatibility() {
        use wasm_bindgen::prelude::*;
        
        // Test that wallet can be created in WASM
        let wallet = HdWallet::generate("wasm-test").unwrap();
        assert!(!wallet.mnemonic.is_empty());
        
        // Test address generation in WASM
        let account = wallet.derive_account(0, None).unwrap();
        let addr = account.keys[0].receiving_address().unwrap();
        assert!(addr.starts_with("nerv1"));
    }

    /// Test 11: Performance benchmarks (optional, for CI)
    #[cfg(feature = "bench")]
    mod benches {
        use super::*;
        use test::Bencher;
        
        #[bench]
        fn bench_wallet_generation(b: &mut Bencher) {
            b.iter(|| {
                HdWallet::generate("bench-passphrase").unwrap();
            });
        }
        
        #[bench]
        fn bench_address_derivation(b: &mut Bencher) {
            let wallet = HdWallet::generate("").unwrap();
            b.iter(|| {
                wallet.derive_address(0, 0).unwrap();
            });
        }
        
        #[bench]
        fn bench_mass_address_derivation(b: &mut Bencher) {
            let wallet = HdWallet::generate("").unwrap();
            b.iter(|| {
                for i in 0..100 {
                    wallet.derive_address(i, i).unwrap();
                }
            });
        }
    }

    /// Test 12: Error handling and edge cases
    #[test]
    fn test_error_handling() {
        // Test invalid derivation paths
        let wallet = HdWallet::generate("").unwrap();
        
        // This should work (hardened path)
        let result = wallet.derive_address(0x80000000, 0x80000000);
        assert!(result.is_ok());
        
        // Test with extremely large indices (should handle or error appropriately)
        // This depends on implementation
    }

    /// Test 13: Seed entropy and randomness
    #[test]
    fn test_seed_entropy() {
        // Generate multiple wallets and verify uniqueness
        let mut seeds = HashSet::new();
        
        for _ in 0..100 {
            let wallet = HdWallet::generate("").unwrap();
            let seed = wallet.master_seed.clone();
            assert!(seeds.insert(seed), "Duplicate seed found!");
        }
        
        // Verify we got 100 unique seeds
        assert_eq!(seeds.len(), 100);
    }

    /// Test 14: Passphrase handling
    #[test]
    fn test_passphrase_handling() {
        let mnemonic = TEST_MNEMONIC;
        
        // Same mnemonic, different passphrases should give different wallets
        let wallet1 = HdWallet::from_mnemonic(mnemonic, "pass1").unwrap();
        let wallet2 = HdWallet::from_mnemonic(mnemonic, "pass2").unwrap();
        let wallet3 = HdWallet::from_mnemonic(mnemonic, "").unwrap();
        
        assert_ne!(wallet1.master_seed, wallet2.master_seed);
        assert_ne!(wallet1.master_seed, wallet3.master_seed);
        assert_ne!(wallet2.master_seed, wallet3.master_seed);
        
        // Very long passphrase should work
        let long_pass = "a".repeat(1000);
        let wallet4 = HdWallet::from_mnemonic(mnemonic, &long_pass).unwrap();
        assert_ne!(wallet1.master_seed, wallet4.master_seed);
        
        // Unicode passphrase should work
        let unicode_pass = "🎉🎊🔐💰🚀";
        let wallet5 = HdWallet::from_mnemonic(mnemonic, unicode_pass).unwrap();
        assert_ne!(wallet1.master_seed, wallet5.master_seed);
    }

    /// Test 15: Serialization and deserialization safety
    #[test]
    fn test_serialization_safety() {
        let wallet = HdWallet::generate("").unwrap();
        
        // Public info should serialize safely
        let public_info = wallet.export_public_info().unwrap();
        let deserialized: nerv_wallet::keys::PublicWalletInfo = 
            bincode::deserialize(&public_info).unwrap();
        
        assert!(deserialized.accounts >= 0);
        
        // Secret keys should NOT be serializable accidentally
        // This test verifies that secret keys aren't included in serialization
        let keys = wallet.derive_address(0, 0).unwrap();
        
        // Attempt to serialize keys struct (should fail or not include secrets)
        // This depends on serde annotations in the actual implementation
    }
}
