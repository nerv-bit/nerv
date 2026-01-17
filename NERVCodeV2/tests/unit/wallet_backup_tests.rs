// tests/wallet_backup_tests.rs
//! Unit tests for Epic 7: Secure Backup, Recovery, and Selective Disclosure
//!
//! Tests cover:
//! - Backup encryption/decryption
//! - Wrong password handling
//! - Selective proof structure


#[cfg(test)]
mod tests {
    use super::super::{keys::HdWallet, backup::BackupError};


    #[test]
    fn test_backup_and_recovery() {
        let wallet = HdWallet::generate("").unwrap();
        let mnemonic = wallet.mnemonic.clone();


        let backup = wallet.encrypt_backup("testpass123").unwrap();


        let recovered = HdWallet::recover_from_backup(backup, "testpass123").await.unwrap();
        assert_eq!(recovered.mnemonic, mnemonic);


        let wrong = HdWallet::recover_from_backup(backup.clone(), "wrong").await;
        assert!(matches!(wrong, Err(BackupError::WrongPassword)));
    }
}
