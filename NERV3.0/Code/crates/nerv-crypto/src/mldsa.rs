//! ML-DSA-65 (FIPS 204, NIST level 3): deterministic keygen from a 32-byte
//! seed, sign/verify over raw message bytes (callers domain-separate),
//! parallel batch verification. FIPS wire: pk 1952 B, sig 3309 B.

use std::fmt;
use rayon::prelude::*;
use zeroize::Zeroize;

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::error::CodecError;

use crate::error::CryptoError;
use crate::provider;

pub const PK_LEN: usize = provider::DSA_PK_LEN;
pub const SIG_LEN: usize = provider::DSA_SIG_LEN;
pub const SEED_LEN: usize = 32;

/// ML-DSA-65 signing key. Secret material (the serialized keypair) is
/// zeroized on drop.
#[derive(Clone)]
pub struct SigningKey {
    kp: Vec<u8>,
    vk: VerifyingKey,
}

impl Drop for SigningKey {
    fn drop(&mut self) {
        self.kp.zeroize();
    }
}

impl fmt::Debug for SigningKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SigningKey").field("vk", &self.vk).finish()
    }
}

impl SigningKey {
    pub fn from_seed(seed: &[u8; SEED_LEN]) -> Result<Self, CryptoError> {
        let kp = provider::dsa_keypair_from_seed(seed).map_err(CryptoError::Provider)?;
        Ok(SigningKey { vk: extract_vk(&kp)?, kp })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        provider::dsa_keypair_from_bytes(bytes).map_err(CryptoError::Provider)?;
        let kp = bytes.to_vec();
        Ok(SigningKey { vk: extract_vk(&kp)?, kp })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.kp.clone()
    }

    pub const fn verifying_key(&self) -> &VerifyingKey {
        &self.vk
    }

    pub fn sign(&self, message: &[u8]) -> Result<Signature, CryptoError> {
        let sig = provider::dsa_sign(&self.kp, message).map_err(CryptoError::Provider)?;
        Ok(Signature(provider::to_fixed::<SIG_LEN>(sig, "ML-DSA-65 signature")?))
    }
}

fn extract_vk(kp: &[u8]) -> Result<VerifyingKey, CryptoError> {
    let vk = provider::dsa_vk_of_keypair(kp).map_err(CryptoError::Provider)?;
    Ok(VerifyingKey(provider::to_fixed::<PK_LEN>(vk, "ML-DSA-65 public key")?))
}

/// ML-DSA-65 public key (FIPS 332-byte… 1952-byte wire form).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VerifyingKey([u8; PK_LEN]);

impl fmt::Debug for VerifyingKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlDsa65Vk({:02x}{:02x}…)", self.0[0], self.0[1])
    }
}

impl VerifyingKey {
    pub const fn from_bytes(bytes: [u8; PK_LEN]) -> VerifyingKey {
        VerifyingKey(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; PK_LEN] {
        &self.0
    }

    pub fn from_slice(bytes: &[u8]) -> Result<VerifyingKey, CryptoError> {
        Ok(VerifyingKey(provider::to_fixed::<PK_LEN>(
            bytes.to_vec(),
            "ML-DSA-65 public key",
        )?))
    }

    pub fn verify(&self, message: &[u8], signature: &Signature) -> bool {
        provider::dsa_verify(&self.0, message, &signature.0)
    }
}

impl Encode for VerifyingKey {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0);
    }
    fn encoded_len(&self) -> usize {
        PK_LEN
    }
}

impl Decode for VerifyingKey {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(VerifyingKey(r.take_array::<PK_LEN>()?))
    }
}

/// ML-DSA-65 signature (3309-byte FIPS wire form).
#[derive(Clone, PartialEq, Eq)]
pub struct Signature(pub(crate) [u8; SIG_LEN]);

impl fmt::Debug for Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlDsa65Sig({:02x}{:02x}…, {SIG_LEN}B)", self.0[0], self.0[1])
    }
}

impl Signature {
    pub const fn from_bytes(bytes: [u8; SIG_LEN]) -> Signature {
        Signature(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; SIG_LEN] {
        &self.0
    }
}

impl Encode for Signature {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0);
    }
    fn encoded_len(&self) -> usize {
        SIG_LEN
    }
}

impl Decode for Signature {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(Signature(r.take_array::<SIG_LEN>()?))
    }
}

/// Parallel batch verification: FIPS 204 defines no signature aggregation, so
/// batching is rayon-parallel independent verification (the production-honest
/// meaning for lattice signatures).
pub fn verify_batch<'a>(items: &[(&'a VerifyingKey, &'a [u8], &'a Signature)]) -> bool {
    items.par_iter().all(|(vk, msg, sig)| vk.verify(msg, sig))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::DetRng;

    fn key(seed: u64) -> SigningKey {
        let mut rng = DetRng::new(seed);
        SigningKey::from_seed(&rng.bytes32()).unwrap()
    }

    #[test]
    fn deterministic_keygen() {
        let mut rng = DetRng::new(1);
        let seed = rng.bytes32();
        let a = SigningKey::from_seed(&seed).unwrap();
        let b = SigningKey::from_seed(&seed).unwrap();
        assert_eq!(a.verifying_key(), b.verifying_key());
        assert_eq!(a.to_bytes(), b.to_bytes());
        assert_ne!(a.verifying_key(), key(2).verifying_key());
    }

    #[test]
    fn sign_verify_and_failures() {
        let sk = key(3);
        let msg: &[u8] = b"block header digest bytes";
        let sig = sk.sign(msg).unwrap();
        assert_eq!(sig.as_bytes().len(), SIG_LEN);
        assert_eq!(sk.verifying_key().as_bytes().len(), PK_LEN);
        assert!(sk.verifying_key().verify(msg, &sig));
        assert!(!sk.verifying_key().verify(b"other message", &sig));
        assert!(!key(4).verifying_key().verify(msg, &sig));
        let mut tampered = sig;
        tampered.0[0] ^= 1;
        assert!(!sk.verifying_key().verify(msg, &tampered));
        let mut tampered_end = sk.sign(msg).unwrap();
        let n = tampered_end.0.len();
        tampered_end.0[n - 1] ^= 0x80;
        assert!(!sk.verifying_key().verify(msg, &tampered_end));
    }

    #[test]
    fn signing_key_bytes_roundtrip() {
        let sk = key(5);
        let bytes = sk.to_bytes();
        let restored = SigningKey::from_bytes(&bytes).unwrap();
        assert_eq!(restored.verifying_key(), sk.verifying_key());
        let msg = b"roundtrip";
        assert!(restored.verifying_key().verify(msg, &restored.sign(msg).unwrap()));
        assert!(SigningKey::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        assert!(SigningKey::from_bytes(&[0u8; 8]).is_err());
    }

    #[test]
    fn codec_roundtrips() {
        let vk = *key(6).verifying_key();
        let enc = vk.encode();
        assert_eq!(enc.len(), PK_LEN);
        assert_eq!(VerifyingKey::decode(&enc).unwrap(), vk);
        assert!(VerifyingKey::decode(&enc[..100]).is_err());
        let mut ext = enc.clone();
        ext.push(0);
        assert!(VerifyingKey::decode(&ext).is_err());

        let sig = key(6).sign(b"x").unwrap();
        let enc = sig.encode();
        assert_eq!(enc.len(), SIG_LEN);
        assert_eq!(Signature::decode(&enc).unwrap(), sig);
        assert!(Signature::decode(&enc[..enc.len() - 1]).is_err());
    }

    #[test]
    fn from_slice_length_checked() {
        assert!(VerifyingKey::from_slice(&[0u8; PK_LEN]).is_ok());
        assert!(VerifyingKey::from_slice(&[0u8; PK_LEN - 1]).is_err());
    }

    #[test]
    fn batch_verification() {
        let keys: Vec<SigningKey> = (10..13).map(key).collect();
        let msgs: [&[u8]; 3] = [b"alpha", b"beta", b"gamma"];
        let sigs: Vec<Signature> = keys.iter().zip(msgs).map(|(k, m)| k.sign(m).unwrap()).collect();
        let items: Vec<(&VerifyingKey, &[u8], &Signature)> = keys
            .iter()
            .zip(msgs)
            .zip(&sigs)
            .map(|((k, m), s)| (k.verifying_key(), m, s))
            .collect();
        assert!(verify_batch(&items));
        let mut bad = sigs[1].clone();
        bad.0[100] ^= 1;
        let items_bad: Vec<(&VerifyingKey, &[u8], &Signature)> = vec![
            (keys[0].verifying_key(), msgs[0], &sigs[0]),
            (keys[1].verifying_key(), msgs[1], &bad),
            (keys[2].verifying_key(), msgs[2], &sigs[2]),
        ];
        assert!(!verify_batch(&items_bad));
        assert!(verify_batch(&[]));
    }
}
