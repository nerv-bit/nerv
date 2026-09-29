//! Key-derivation functions: BLAKE3-KDF (derive_key mode, length-framed
//! ikm/info), HKDF-SHA512 (RFC 5869), and the BIP-39 seed derivation
//! (PBKDF2-HMAC-SHA512, 2048 rounds, salt "mnemonic" ‖ passphrase; the
//! phrase must be NFKD-normalized by the caller — ASCII English mnemonics
//! are normalization-invariant).

use blake3::Hasher;
use hkdf::Hkdf;
use nerv_core::constants::Domain;
use sha2::Sha512;

use crate::error::CryptoError;

const HKDF_SHA512_MAX: usize = 255 * 64;

pub fn blake3_kdf(domain: &Domain, ikm: &[u8], info: &[u8]) -> [u8; 32] {
    let mut h = Hasher::new_derive_key(domain.as_str());
    h.update(&(ikm.len() as u32).to_le_bytes());
    h.update(ikm);
    h.update(&(info.len() as u32).to_le_bytes());
    h.update(info);
    *h.finalize().as_bytes()
}

/// Arbitrary-length derived stream from the same KDF state.
pub fn blake3_kdf_stream(domain: &Domain, ikm: &[u8], info: &[u8], out: &mut [u8]) {
    let mut h = Hasher::new_derive_key(domain.as_str());
    h.update(&(ikm.len() as u32).to_le_bytes());
    h.update(ikm);
    h.update(&(info.len() as u32).to_le_bytes());
    h.update(info);
    h.finalize_xof().fill(out);
}

pub fn hkdf_sha512(salt: &[u8], ikm: &[u8], info: &[u8], okm: &mut [u8]) -> Result<(), CryptoError> {
    let hk = Hkdf::<Sha512>::new(Some(salt), ikm);
    hk.expand(info, okm)
        .map_err(|_| CryptoError::KdfOutputTooLong { len: okm.len(), max: HKDF_SHA512_MAX })
}

pub fn bip39_seed(phrase: &str, passphrase: &str) -> [u8; 64] {
    let mut salt = Vec::with_capacity(8 + passphrase.len());
    salt.extend_from_slice(b"mnemonic");
    salt.extend_from_slice(passphrase.as_bytes());
    let mut seed = [0u8; 64];
    pbkdf2::pbkdf2_hmac::<Sha512>(phrase.as_bytes(), &salt, 2048, &mut seed);
    seed
}

#[cfg(test)]
mod tests {
    use super::*;
    use nerv_core::constants::{DERIVED_STATE, NOTE_KDF, SHARD_HOMING};
    use crate::testutil::DetRng;

    #[test]
    fn blake3_kdf_deterministic_and_separated() {
        let a = blake3_kdf(&NOTE_KDF, b"shared secret", b"");
        assert_eq!(a, blake3_kdf(&NOTE_KDF, b"shared secret", b""));
        assert_ne!(a, blake3_kdf(&DERIVED_STATE, b"shared secret", b""));
        assert_ne!(a, blake3_kdf(&NOTE_KDF, b"other ikm", b""));
        assert_ne!(a, blake3_kdf(&NOTE_KDF, b"shared secret", b"subkey"));
        let zeros = [0u8; 32];
        assert_ne!(a, blake3_kdf(&NOTE_KDF, &zeros, b""));
    }

    #[test]
    fn blake3_kdf_framing_unambiguous() {
        assert_ne!(blake3_kdf(&NOTE_KDF, b"ab", b""), blake3_kdf(&NOTE_KDF, b"a", b"b"));
        assert_ne!(blake3_kdf(&NOTE_KDF, b"", b"ab"), blake3_kdf(&NOTE_KDF, b"ab", b""));
    }

    #[test]
    fn blake3_kdf_stream_deterministic() {
        let mut a = vec![0u8; 100];
        let mut b = vec![0u8; 100];
        blake3_kdf_stream(&SHARD_HOMING, b"ikm", b"info", &mut a);
        blake3_kdf_stream(&SHARD_HOMING, b"ikm", b"info", &mut b);
        assert_eq!(a, b);
        blake3_kdf_stream(&SHARD_HOMING, b"ikm", b"other", &mut b);
        assert_ne!(a, b);
        let mut small = vec![0u8; 5];
        blake3_kdf_stream(&SHARD_HOMING, b"ikm", b"info", &mut small);
        assert_eq!(small, a[..5]);
    }

    #[test]
    fn hkdf_deterministic_and_sensitive() {
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        hkdf_sha512(b"salt", b"ikm", b"info", &mut a).unwrap();
        hkdf_sha512(b"salt", b"ikm", b"info", &mut b).unwrap();
        assert_eq!(a, b);
        hkdf_sha512(b"salt", b"ikm", b"info2", &mut b).unwrap();
        assert_ne!(a, b);
        hkdf_sha512(b"salt2", b"ikm", b"info", &mut b).unwrap();
        assert_ne!(a, b);
        hkdf_sha512(b"salt", b"ikm2", b"info", &mut b).unwrap();
        assert_ne!(a, b);
        let mut long = [0u8; 200];
        assert!(hkdf_sha512(b"s", b"i", b"n", &mut long).is_ok());
    }

    #[test]
    fn hkdf_length_limit() {
        let mut too_long = vec![0u8; HKDF_SHA512_MAX + 1];
        assert!(matches!(
            hkdf_sha512(b"s", b"i", b"n", &mut too_long),
            Err(CryptoError::KdfOutputTooLong { .. })
        ));
    }

    #[test]
    fn bip39_seed_properties() {
        let phrase = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let a = bip39_seed(phrase, "TREZOR");
        assert_eq!(a, bip39_seed(phrase, "TREZOR"));
        assert_ne!(a, bip39_seed(phrase, ""));
        assert_ne!(a, bip39_seed("different phrase", "TREZOR"));
        let mut rng = DetRng::new(77);
        let random = rng.bytes32();
        let from_random = bip39_seed(core::str::from_utf8(&random[..]).unwrap_or("x"), "");
        assert_ne!(a, from_random);
    }
}

