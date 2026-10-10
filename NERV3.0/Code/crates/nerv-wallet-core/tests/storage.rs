//! Storage-layer round-trip tests (erratum 206).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use nerv_wallet_core::storage::{
    EncryptedSeed, FileStorage, MemoryStorage, WalletStorage, ENVELOPE_LEN,
};

#[test]
fn seal_open_with_correct_password() {
    for i in 0..100 {
        let seed = [i as u8; 32];
        let password = format!("password-{i}");
        let env = EncryptedSeed::seal(&seed, &password);
        assert_eq!(env.open(&password).unwrap(), seed);
    }
}

#[test]
fn seal_open_with_wrong_password_fails() {
    let seed = [42u8; 32];
    let env = EncryptedSeed::seal(&seed, "correct");
    for wrong in ["", "a", "correct ", "correctt", "password", "12345678", "\u{4e16}"] {
        assert!(
            env.open(wrong).is_err(),
            "wrong password {wrong:?} must fail"
        );
    }
}

#[test]
fn different_passwords_different_ciphertexts() {
    let seed = [7u8; 32];
    let a = EncryptedSeed::seal(&seed, "one");
    let b = EncryptedSeed::seal(&seed, "two");
    assert_ne!(a.ciphertext, b.ciphertext);
    assert_ne!(a.salt, b.salt, "fresh salt per encryption");
    assert_ne!(a.nonce, b.nonce, "fresh nonce per encryption");
}

#[test]
fn wire_format_is_exactly_92_bytes() {
    let env = EncryptedSeed::seal(&[1u8; 32], "pw");
    assert_eq!(env.to_bytes().len(), ENVELOPE_LEN);
    assert_eq!(ENVELOPE_LEN, 92);
}

#[test]
fn memory_storage_full_lifecycle() {
    let mut s = MemoryStorage::new();
    assert!(!s.has_seed());

    s.store_seed(&[1u8; 32], "pw").unwrap();
    assert!(s.has_seed());
    assert_eq!(s.load_seed("pw").unwrap(), Some([1u8; 32]));
    assert!(s.load_seed("wrong").is_err());

    s.delete_seed().unwrap();
    assert!(!s.has_seed());
    assert_eq!(s.load_seed("pw").unwrap(), None);

    s.store_meta("key", "value").unwrap();
    assert_eq!(s.load_meta("key").unwrap(), Some("value".into()));
    assert_eq!(s.load_meta("missing").unwrap(), None);
}

#[test]
fn file_storage_survives_process_restart() {
    let dir = std::env::temp_dir().join(format!(
        "nerv-storage-rt-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    // Write.
    {
        let mut s = FileStorage::new(&dir);
        s.store_seed(&[9u8; 32], "pw").unwrap();
        s.store_meta("theme", "dark").unwrap();
    }

    // Read (as if from a new process).
    {
        let s = FileStorage::new(&dir);
        assert!(s.has_seed());
        assert_eq!(s.load_seed("pw").unwrap(), Some([9u8; 32]));
        assert_eq!(s.load_meta("theme").unwrap(), Some("dark".into()));
    }

    // Delete.
    {
        let mut s = FileStorage::new(&dir);
        s.delete_seed().unwrap();
        assert!(!s.has_seed());
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn encrypted_seed_base64_roundtrip() {
    let seed = [0xAB; 32];
    let env = EncryptedSeed::seal(&seed, "pw");
    let b64 = env.to_base64();
    let decoded = EncryptedSeed::from_base64(&b64).unwrap();
    assert_eq!(decoded.open("pw").unwrap(), seed);
    assert!(EncryptedSeed::from_base64("invalid!").is_err());
    assert!(EncryptedSeed::from_base64("").is_err());
}


