//! WASM (web) storage: encrypted localStorage (erratum 204).
//! The password encryption is the only protection in the browser.

use nerv_wallet_core::{EncryptedSeed, StorageError, WalletStorage};

pub struct WebStorage {
    prefix: String,
}

impl WebStorage {
    pub fn new() -> WebStorage {
        WebStorage { prefix: "nerv.wallet".to_string() }
    }

    fn ls_get(&self, key: &str) -> Option<String> {
        let full = format!("{}.{}", self.prefix, key);
        web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten())
            .and_then(|ls| ls.get_item(&full).ok().flatten())
    }

    fn ls_set(&self, key: &str, value: &str) -> Result<(), StorageError> {
        let full = format!("{}.{}", self.prefix, key);
        web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten())
            .ok_or_else(|| StorageError::Backend("localStorage unavailable".into()))?
            .set_item(&full, value)
            .map_err(|e| StorageError::Backend(format!("localStorage: {e:?}")))
    }

    fn ls_remove(&self, key: &str) {
        let full = format!("{}.{}", self.prefix, key);
        if let Some(ls) = web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten())
        {
            let _ = ls.remove_item(&full);
        }
    }
}

impl Default for WebStorage {
    fn default() -> Self {
        WebStorage::new()
    }
}

impl WalletStorage for WebStorage {
    fn store_seed(&mut self, seed: &[u8; 32], password: &str) -> Result<(), StorageError> {
        let env = EncryptedSeed::seal(seed, password);
        self.ls_set("seed", &env.to_base64())
    }

    fn load_seed(&self, password: &str) -> Result<Option<[u8; 32]>, StorageError> {
        match self.ls_get("seed") {
            Some(b64) => {
                let env = EncryptedSeed::from_base64(&b64)?;
                Ok(Some(env.open(password)?))
            }
            None => Ok(None),
        }
    }

    fn delete_seed(&mut self) -> Result<(), StorageError> {
        self.ls_remove("seed");
        Ok(())
    }

    fn has_seed(&self) -> bool {
        self.ls_get("seed").is_some()
    }

    fn store_meta(&mut self, key: &str, value: &str) -> Result<(), StorageError> {
        self.ls_set(&format!("meta.{key}"), value)
    }

    fn load_meta(&self, key: &str) -> Result<Option<String>, StorageError> {
        Ok(self.ls_get(&format!("meta.{key}")))
    }
}
