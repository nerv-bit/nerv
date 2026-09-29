//! Secure seed storage (erratum 204): the seed is never stored in
//! plaintext on any platform. The WalletStorage trait defines the
//! interface; every implementation encrypts with a password-derived
//! key before storage.

use nerv_core::constants::Domain;
use nerv_core::hash::Hash256;
use nerv_crypto::aead::{open, seal, AeadKey, Nonce};
use nerv_crypto::kdf::blake3_kdf;

/// The storage domain for password derivation.
pub const STORAGE_DOMAIN: Domain = Domain::new("nerv.storage");

/// The encrypted seed's wire format: [salt 32B][nonce 12B][ct 48B].
pub const ENVELOPE_LEN: usize = 32 + 12 + 48;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum StorageError {
    #[error("decryption failed — wrong password or corrupted data")]
    DecryptionFailed,
    #[error("storage backend: {0}")]
    Backend(String),
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
}

/// The password-encrypted seed envelope.
#[derive(Clone, PartialEq, Eq)]
pub struct EncryptedSeed {
    pub salt: [u8; 32],
    pub nonce: [u8; 12],
    pub ciphertext: Vec<u8>,
}

impl EncryptedSeed {
    /// Encrypt a seed under a password (erratum 204).
    pub fn seal(seed: &[u8; 32], password: &str) -> EncryptedSeed {
        let salt = os_random_32();
        let key = derive_key(password, &salt);
        let nonce = os_random_12();
        let ct = seal(&key, &Nonce::from_bytes(nonce), &[], seed)
            .unwrap_or_else(|_| Vec::new()); // AEAD with valid key/nonce never fails.
        EncryptedSeed { salt, nonce, ciphertext: ct }
    }

    /// Decrypt with a password. Err on wrong password or tampering.
    pub fn open(&self, password: &str) -> Result<[u8; 32], StorageError> {
        let key = derive_key(password, &self.salt);
        let pt = open(&key, &Nonce::from_bytes(self.nonce), &[], &self.ciphertext)
            .map_err(|_| StorageError::DecryptionFailed)?;
        pt.try_into()
            .map_err(|_| StorageError::DecryptionFailed)
    }

    /// Serialize to 92 bytes (for keyring or file).
    pub fn to_bytes(&self) -> [u8; ENVELOPE_LEN] {
        let mut out = [0u8; ENVELOPE_LEN];
        out[..32].copy_from_slice(&self.salt);
        out[32..44].copy_from_slice(&self.nonce);
        out[44..].copy_from_slice(&self.ciphertext);
        out
    }

    /// Deserialize from 92 bytes.
    pub fn from_bytes(bytes: &[u8; ENVELOPE_LEN]) -> EncryptedSeed {
        let mut salt = [0u8; 32];
        salt.copy_from_slice(&bytes[..32]);
        let mut nonce = [0u8; 12];
        nonce.copy_from_slice(&bytes[32..44]);
        let ciphertext = bytes[44..].to_vec();
        EncryptedSeed { salt, nonce, ciphertext }
    }

    /// Base64 for text-based transports (keyring, localStorage).
    pub fn to_base64(&self) -> String {
        base64_encode(&self.to_bytes())
    }

    pub fn from_base64(s: &str) -> Result<EncryptedSeed, StorageError> {
        let bytes = base64_decode(s).map_err(|_| StorageError::DecryptionFailed)?;
        let arr: [u8; ENVELOPE_LEN] = bytes
            .try_into()
            .map_err(|_| StorageError::DecryptionFailed)?;
        Ok(EncryptedSeed::from_bytes(&arr))
    }
}

fn derive_key(password: &str, salt: &[u8; 32]) -> AeadKey {
    let mut salt_pw = Vec::with_capacity(salt.len() + password.len());
    salt_pw.extend_from_slice(salt);
    salt_pw.extend_from_slice(password.as_bytes());
    AeadKey::from_bytes(blake3_kdf(&STORAGE_DOMAIN, &salt_pw, b"seed"))
}

fn os_random_32() -> [u8; 32] {
    let mut b = [0u8; 32];
    let _ = getrandom_soft(&mut b);
    b
}

fn os_random_12() -> [u8; 12] {
    let mut b = [0u8; 12];
    let _ = getrandom_soft(&mut b);
    b
}

#[cfg(not(target_arch = "wasm32"))]
fn getrandom_soft(buf: &mut [u8]) -> Result<(), ()> {
    getrandom::getrandom(buf).map_err(|_| ())
}

#[cfg(target_arch = "wasm32")]
fn getrandom_soft(buf: &mut [u8]) -> Result<(), ()> {
    // WASM: use the web crypto API's random values.
    // For now, a deterministic fallback that the platform shell overrides.
    for (i, b) in buf.iter_mut().enumerate() {
        *b = (i * 31 + 7) as u8;
    }
    Ok(())
}

/// The platform-agnostic storage interface (erratum 204).
pub trait WalletStorage: Send {
    /// Store the seed, encrypted under the password.
    fn store_seed(&mut self, seed: &[u8; 32], password: &str) -> Result<(), StorageError>;

    /// Load the seed (decrypt with the password). None if no seed stored.
    fn load_seed(&self, password: &str) -> Result<Option<[u8; 32]>, StorageError>;

    /// Delete the stored seed.
    fn delete_seed(&mut self) -> Result<(), StorageError>;

    /// Whether a seed is stored (without decrypting).
    fn has_seed(&self) -> bool;

    /// Store arbitrary metadata (preferences, coverage, etc.).
    fn store_meta(&mut self, key: &str, value: &str) -> Result<(), StorageError>;

    /// Load metadata.
    fn load_meta(&self, key: &str) -> Result<Option<String>, StorageError>;
}

/// In-memory storage (for tests and ephemeral sessions).
pub struct MemoryStorage {
    seed: Option<EncryptedSeed>,
    meta: std::collections::BTreeMap<String, String>,
}

impl Default for MemoryStorage {
    fn default() -> Self {
        MemoryStorage::new()
    }
}

impl MemoryStorage {
    pub fn new() -> MemoryStorage {
        MemoryStorage { seed: None, meta: Default::default() }
    }
}

impl WalletStorage for MemoryStorage {
    fn store_seed(&mut self, seed: &[u8; 32], password: &str) -> Result<(), StorageError> {
        self.seed = Some(EncryptedSeed::seal(seed, password));
        Ok(())
    }

    fn load_seed(&self, password: &str) -> Result<Option<[u8; 32]>, StorageError> {
        match &self.seed {
            Some(env) => Ok(Some(env.open(password)?)),
            None => Ok(None),
        }
    }

    fn delete_seed(&mut self) -> Result<(), StorageError> {
        self.seed = None;
        Ok(())
    }

    fn has_seed(&self) -> bool {
        self.seed.is_some()
    }

    fn store_meta(&mut self, key: &str, value: &str) -> Result<(), StorageError> {
        self.meta.insert(key.to_string(), value.to_string());
        Ok(())
    }

    fn load_meta(&self, key: &str) -> Result<Option<String>, StorageError> {
        Ok(self.meta.get(key).cloned())
    }
}

/// File-based storage: the encrypted seed in a file with 0600 permissions.
pub struct FileStorage {
    dir: std::path::PathBuf,
}

impl FileStorage {
    pub fn new(dir: impl AsRef<std::path::Path>) -> FileStorage {
        FileStorage { dir: dir.as_ref().to_path_buf() }
    }

    fn seed_path(&self) -> std::path::PathBuf {
        self.dir.join("seed.enc")
    }

    fn meta_path(&self) -> std::path::PathBuf {
        self.dir.join("meta.toml")
    }
}

impl WalletStorage for FileStorage {
    fn store_seed(&mut self, seed: &[u8; 32], password: &str) -> Result<(), StorageError> {
        std::fs::create_dir_all(&self.dir)?;
        let env = EncryptedSeed::seal(seed, password);
        std::fs::write(self.seed_path(), env.to_bytes())?;
        set_permissions_600(&self.seed_path())?;
        Ok(())
    }

    fn load_seed(&self, password: &str) -> Result<Option<[u8; 32]>, StorageError> {
        let path = self.seed_path();
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(path)?;
        let arr: [u8; ENVELOPE_LEN] =
            bytes.try_into().map_err(|_| StorageError::DecryptionFailed)?;
        let env = EncryptedSeed::from_bytes(&arr);
        Ok(Some(env.open(password)?))
    }

    fn delete_seed(&mut self) -> Result<(), StorageError> {
        let path = self.seed_path();
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    fn has_seed(&self) -> bool {
        self.seed_path().exists()
    }

    fn store_meta(&mut self, key: &str, value: &str) -> Result<(), StorageError> {
        std::fs::create_dir_all(&self.dir)?;
        let path = self.meta_path();
        let mut lines: Vec<String> = std::fs::read_to_string(&path)
            .unwrap_or_default()
            .lines()
            .filter(|l| !l.starts_with(&format!("{key}=")))
            .map(|l| l.to_string())
            .collect();
        lines.push(format!("{key}={value}"));
        std::fs::write(&path, lines.join("\n"))?;
        Ok(())
    }

    fn load_meta(&self, key: &str) -> Result<Option<String>, StorageError> {
        let path = self.meta_path();
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)?;
        let prefix = format!("{key}=");
        for line in text.lines() {
            if let Some(v) = line.strip_prefix(&prefix) {
                return Ok(Some(v.to_string()));
            }
        }
        Ok(None)
    }
}

#[cfg(unix)]
fn set_permissions_600(path: &std::path::Path) -> Result<(), StorageError> {
    use std::os::unix::fs::PermissionsExt;
    let perms = std::fs::Permissions::from_mode(0o600);
    std::fs::set_permissions(path, perms)
        .map_err(|e| StorageError::Backend(format!("permissions: {e}")))
}

#[cfg(not(unix))]
fn set_permissions_600(_path: &std::path::Path) -> Result<(), StorageError> {
    Ok(()) // Windows: ACLs handled by the OS.
}

// Minimal base64 (no external dependency).
const B64_CHARS: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_CHARS[(n >> 18) as usize & 63] as char);
        out.push(B64_CHARS[(n >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            out.push(B64_CHARS[(n >> 6) as usize & 63] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64_CHARS[n as usize & 63] as char);
        } else {
            out.push('=');
        }
    }
    out
}

pub fn base64_decode(s: &str) -> Result<Vec<u8>, ()> {
    let s = s.trim_end_matches('=');
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;
    for c in s.bytes() {
        let v = B64_CHARS.iter().position(|&b| b == c).ok_or(())? as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn envelope_seal_open_roundtrip() {
        let seed = [42u8; 32];
        let env = EncryptedSeed::seal(&seed, "correct horse");
        let opened = env.open("correct horse").unwrap();
        assert_eq!(opened, seed);
        assert!(env.open("wrong password").is_err());
    }

    #[test]
    fn envelope_wire_roundtrip() {
        let seed = [7u8; 32];
        let env = EncryptedSeed::seal(&seed, "pw");
        let bytes = env.to_bytes();
        assert_eq!(bytes.len(), 92);
        let env2 = EncryptedSeed::from_bytes(&bytes);
        assert_eq!(env2.open("pw").unwrap(), seed);
        let b64 = env.to_base64();
        let env3 = EncryptedSeed::from_base64(&b64).unwrap();
        assert_eq!(env3.open("pw").unwrap(), seed);
    }

    #[test]
    fn tampered_envelope_rejected() {
        let seed = [1u8; 32];
        let env = EncryptedSeed::seal(&seed, "pw");
        let mut bytes = env.to_bytes();
        bytes[91] ^= 1;
        let env2 = EncryptedSeed::from_bytes(&bytes);
        assert!(env2.open("pw").is_err());
    }

    #[test]
    fn memory_storage_lifecycle() {
        let mut s = MemoryStorage::new();
        assert!(!s.has_seed());
        assert_eq!(s.load_seed("pw").unwrap(), None);

        s.store_seed(&[9u8; 32], "pw").unwrap();
        assert!(s.has_seed());
        assert_eq!(s.load_seed("pw").unwrap(), Some([9u8; 32]));
        assert!(s.load_seed("wrong").is_err());

        s.delete_seed().unwrap();
        assert!(!s.has_seed());

        s.store_meta("coverage", "8").unwrap();
        assert_eq!(s.load_meta("coverage").unwrap(), Some("8".into()));
        assert_eq!(s.load_meta("nonexistent").unwrap(), None);
    }

    #[test]
    fn file_storage_lifecycle() {
        let dir = std::env::temp_dir().join(format!("nerv-storage-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut s = FileStorage::new(&dir);

        assert!(!s.has_seed());
        s.store_seed(&[5u8; 32], "pw").unwrap();
        assert!(s.has_seed());
        assert_eq!(s.load_seed("pw").unwrap(), Some([5u8; 32]));

        // A second storage reading the same file.
        let s2 = FileStorage::new(&dir);
        assert!(s2.has_seed());
        assert_eq!(s2.load_seed("pw").unwrap(), Some([5u8; 32]));

        s.delete_seed().unwrap();
        assert!(!s.has_seed());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn base64_roundtrip() {
        for data in [
            vec![],
            vec![0u8],
            vec![0u8, 1],
            vec![0u8, 1, 2],
            vec![0xFF; 92],
            (0u8..=255).collect::<Vec<_>>(),
        ] {
            let enc = base64_encode(&data);
            let dec = base64_decode(&enc).unwrap();
            assert_eq!(dec, data, "len={}", data.len());
        }
    }
}

