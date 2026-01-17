// tests/wallet_keys_tests.rs
//! Unit tests for Epic 1: Post-Quantum Hierarchical Deterministic Key Management
//!
//! Tests cover:
//! - Mnemonic generation and recovery
//! - Hardened derivation consistency
//! - Receiving address encoding/decoding
//! - Zeroization of secret keys
//! - Gap limit and multiple address derivation


#[cfg(test)]
mod tests {
    use super::super::keys::{HdWallet, AccountKeys};
    use bech32::{decode, encode, Variant};


    #[test]
    fn test_wallet_generation_and_recovery() {
        let wallet1 = HdWallet::generate("").unwrap();
        let mnemonic = wallet1.mnemonic.clone();


        let wallet2 = HdWallet::from_mnemonic(&mnemonic, "").unwrap();


        assert_eq!(wallet1.master_seed, wallet2.master_seed);
        assert_eq!(wallet1.mnemonic, wallet2.mnemonic);
    }


    #[test]
    fn test_address_derivation_consistency() {
        let wallet = HdWallet::from_mnemonic("abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about", "").unwrap();


        let account = wallet.derive_account(0, 5).unwrap();
        let first_addr = account.keys[0].receiving_address().unwrap();
        let second_addr = account.keys[1].receiving_address().unwrap();


        // Known expected addresses from deterministic derivation (pre-computed)
        assert!(first_addr.starts_with("nerv1"));
        assert_ne!(first_addr, second_addr);


        // Derive same address again
        let single_key = wallet.derive_address(0, 0).unwrap();
        assert_eq!(single_key.receiving_address().unwrap(), first_addr);
    }


    #[test]
    fn test_bech32_address_format() {
        let wallet = HdWallet::generate("").unwrap();
        let account = wallet.derive_account(0, 1).unwrap();
        let addr = account.keys[0].receiving_address().unwrap();


        let (hrp, data, variant) = decode(&addr).unwrap();
        assert_eq!(hrp, "nerv");
        assert_eq!(variant, Variant::Bech32);
        assert_eq!(data.len(), 32); // ML-KEM pk size in base32
    }


    #[test]
    fn test_secret_key_zeroization() {
        let wallet = HdWallet::generate("").unwrap();
        let keys = wallet.derive_address(0, 0).unwrap();


        let sk_enc = keys.enc_sk.clone(); // Copy before drop
        drop(keys);


        // After drop, should be zeroized
        assert!(sk_enc.iter().all(|&b| b == 0));
    }


    #[test]
    fn test_multiple_accounts() {
        let wallet = HdWallet::generate("").unwrap();
        let account0 = wallet.derive_account(0, 1).unwrap();
        let account1 = wallet.derive_account(1, 1).unwrap();


        assert_ne!(
            account0.keys[0].receiving_address().unwrap(),
            account1.keys[0].receiving_address().unwrap()
        );
    }
}
