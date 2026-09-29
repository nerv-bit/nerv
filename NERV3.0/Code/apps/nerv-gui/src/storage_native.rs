
//! Native (desktop) storage: OS keychain with encrypted-seed fallback
//! to a file (erratum 204).

use nerv_wallet_core::{EncryptedSeed, FileStorage, StorageError, WalletStorage};

/// OS keychain storage via the `keyring` crate (macOS Keychain,
/// Linux secret-service, Windows Credential Manager).
pub struct KeyringStorage {
    service: String,
    account: String,
    fallback: FileStorage,
}

impl KeyringStorage {
    pub fn new(data_dir: impl AsRef<std::path::Path>) -> KeyringStorage {
        KeyringStorage {
            service: "nerv.wallet".to_string(),
            account: "master_seed".to_string(),
            fallback: FileStorage::new(data_dir),
        }
    }
}

impl WalletStorage for KeyringStorage {
    fn store_seed(&mut self, seed: &[u8; 32], password: &str) -> Result<(), StorageError> {
        let env = EncryptedSeed::seal(seed, password);
        let entry = keyring::Entry::new(&self.service, &self.account)
            .map_err(|e| StorageError::Backend(format!("keyring: {e}")))?;
        entry
            .set_password(&env.to_base64())
            .map_err(|e| {
                // Keychain unavailable (headless, no D-Bus): fall back to file.
                let _ = self.fallback.store_seed(seed, password);
                StorageError::Backend(format!("keychain write failed ({e}); fell back to file"))
            })?;
        Ok(())
    }

    fn load_seed(&self, password: &str) -> Result<Option<[u8; 32]>, StorageError> {
        let entry = keyring::Entry::new(&self.service, &self.account)
            .map_err(|e| StorageError::Backend(format!("keyring: {e}")))?;
        match entry.get_password() {
            Ok(b64) => {
                let env = EncryptedSeed::from_base64(&b64)?;
                Ok(Some(env.open(password)?))
            }
            Err(keyring::Error::NoEntry) => {
                // Not in the keychain; try the file fallback.
                self.fallback.load_seed(password)
            }
            Err(e) => {
                // Keychain unavailable; try the file fallback.
                self.fallback.load_seed(password)
                    .map_err(|_| StorageError::Backend(format!("keychain: {e}")))
            }
        }
    }

    fn delete_seed(&mut self) -> Result<(), StorageError> {
        let entry = keyring::Entry::new(&self.service, &self.account)
            .map_err(|e| StorageError::Backend(format!("keyring: {e}")))?;
        let _ = entry.delete_credential();
        let _ = self.fallback.delete_seed();
        Ok(())
    }

    fn has_seed(&self) -> bool {
        if let Ok(entry) = keyring::Entry::new(&self.service, &self.account) {
            if entry.get_password().is_ok() {
                return true;
            }
        }
        self.fallback.has_seed()
    }

    fn store_meta(&mut self, key: &str, value: &str) -> Result<(), StorageError> {
        self.fallback.store_meta(key, value)
    }

    fn load_meta(&self, key: &str) -> Result<Option<String>, StorageError> {
        self.fallback.load_meta(key)
    }
}
