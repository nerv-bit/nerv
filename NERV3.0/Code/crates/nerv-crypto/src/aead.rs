//! ChaCha20-Poly1305 one-shot AEAD (WP §3.2 note encryption, §6.2 onion
//! layers). Key law: the key must be single-use or the nonce unique per
//! (key, nonce) pair. Note keys are freshly derived per ML-KEM encapsulation,
//! so the all-zero nonce is the standard call for note encryption.

use std::fmt;
use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce as CpNonce};
use zeroize::Zeroize;

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::error::CodecError;

use crate::error::CryptoError;

pub const KEY_LEN: usize = 32;
pub const NONCE_LEN: usize = 12;
pub const TAG_LEN: usize = 16;

/// AEAD key; zeroized on drop.
#[derive(Clone, PartialEq, Eq)]
pub struct AeadKey([u8; KEY_LEN]);

impl Drop for AeadKey {
    fn drop(&mut self) {
        self.0.as_mut_slice().zeroize();
    }
}

impl fmt::Debug for AeadKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AeadKey(<redacted, {KEY_LEN}B>)")
    }
}

impl AeadKey {
    pub const fn from_bytes(bytes: [u8; KEY_LEN]) -> AeadKey {
        AeadKey(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; KEY_LEN] {
        &self.0
    }
}

impl Encode for AeadKey {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0);
    }
    fn encoded_len(&self) -> usize {
        KEY_LEN
    }
}

impl Decode for AeadKey {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(AeadKey(r.take_array::<KEY_LEN>()?))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Nonce([u8; NONCE_LEN]);

impl Nonce {
    pub const ZERO: Nonce = Nonce([0u8; NONCE_LEN]);

    pub const fn from_bytes(bytes: [u8; NONCE_LEN]) -> Nonce {
        Nonce(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; NONCE_LEN] {
        &self.0
    }
}

impl Encode for Nonce {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0);
    }
    fn encoded_len(&self) -> usize {
        NONCE_LEN
    }
}

impl Decode for Nonce {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(Nonce(r.take_array::<NONCE_LEN>()?))
    }
}

/// One-shot seal: output = ciphertext ‖ 16-byte Poly1305 tag.
pub fn seal(
    key: &AeadKey,
    nonce: &Nonce,
    aad: &[u8],
    plaintext: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    let cipher = ChaCha20Poly1305::new(Key::from_slice(key.as_bytes()));
    cipher
        .encrypt(
            CpNonce::from_slice(nonce.as_bytes()),
            Payload { msg: plaintext, aad },
        )
        .map_err(|_| CryptoError::AeadAuthentication)
}

/// One-shot open: input = ciphertext ‖ tag; AAD and any tampering must match
/// exactly or authentication fails.
pub fn open(
    key: &AeadKey,
    nonce: &Nonce,
    aad: &[u8],
    ciphertext: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    let cipher = ChaCha20Poly1305::new(Key::from_slice(key.as_bytes()));
    cipher
        .decrypt(
            CpNonce::from_slice(nonce.as_bytes()),
            Payload { msg: ciphertext, aad },
        )
        .map_err(|_| CryptoError::AeadAuthentication)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::DetRng;

    fn key(seed: u64) -> AeadKey {
        AeadKey::from_bytes(DetRng::new(seed).bytes32())
    }

    #[test]
    fn roundtrip_and_lengths() {
        let k = key(1);
        let aad = b"commitment bytes";
        let pt = b"note plaintext v rho d memo";
        let sealed = seal(&k, &Nonce::ZERO, aad, pt).unwrap();
        assert_eq!(sealed.len(), pt.len() + TAG_LEN);
        assert_eq!(open(&k, &Nonce::ZERO, aad, &sealed).unwrap(), pt);
    }

    #[test]
    fn empty_plaintext_and_large_aad() {
        let k = key(2);
        let aad = vec![5u8; 10_000];
        let sealed = seal(&k, &Nonce::ZERO, &aad, b"").unwrap();
        assert_eq!(sealed.len(), TAG_LEN);
        assert_eq!(open(&k, &Nonce::ZERO, &aad, &sealed).unwrap(), b"");
        assert!(open(&k, &Nonce::ZERO, &aad[..9999], &sealed).is_err());
    }

    #[test]
    fn authentication_failures() {
        let k = key(3);
        let aad = b"aad";
        let sealed = seal(&k, &Nonce::ZERO, aad, b"payload").unwrap();
        assert!(open(&key(4), &Nonce::ZERO, aad, &sealed).is_err());
        assert!(open(&k, &Nonce::from_bytes([1u8; NONCE_LEN]), aad, &sealed).is_err());
        assert!(open(&k, &Nonce::ZERO, b"other", &sealed).is_err());
        let mut tampered = sealed.clone();
        tampered[0] ^= 1;
        assert!(open(&k, &Nonce::ZERO, aad, &tampered).is_err());
        let mut tampered_tag = sealed.clone();
        let n = tampered_tag.len();
        tampered_tag[n - 1] ^= 1;
        assert!(open(&k, &Nonce::ZERO, aad, &tampered_tag).is_err());
        assert!(open(&k, &Nonce::ZERO, aad, &sealed[..sealed.len() - 1]).is_err());
    }

    #[test]
    fn codec_roundtrips() {
        let k = key(5);
        assert_eq!(AeadKey::decode(&k.encode()).unwrap(), k);
        assert_eq!(k.encoded_len(), KEY_LEN);
        assert!(AeadKey::decode(&[0u8; KEY_LEN - 1]).is_err());
        let nonce = Nonce::from_bytes([9u8; NONCE_LEN]);
        assert_eq!(Nonce::decode(&nonce.encode()).unwrap(), nonce);
        assert_eq!(format!("{k:?}"), "AeadKey(<redacted, 32B>)");
    }
}
