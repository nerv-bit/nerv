//! ML-KEM-768 (FIPS 203, NIST level 3): deterministic keygen from a 64-byte
//! seed (d‖z) — the primitive diversified delivery keys are derived from —
//! explicit-randomness encapsulation, decapsulation with implicit rejection
//! (a wrong-key decapsulation returns a pseudorandom secret, never an error;
//! only malformed input bytes error). FIPS wire: ek 1184, dk 2400, ct 1088,
//! ss 32.

use std::fmt;
use zeroize::Zeroize;

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::error::CodecError;

use crate::error::CryptoError;
use crate::provider;

pub const EK_LEN: usize = provider::KEM_EK_LEN;
pub const DK_LEN: usize = provider::KEM_DK_LEN;
pub const CT_LEN: usize = provider::KEM_CT_LEN;
pub const SS_LEN: usize = provider::KEM_SS_LEN;
pub const SEED_LEN: usize = 64;
pub const ENCAPS_RANDOMNESS_LEN: usize = 32;

pub fn keypair_from_seed(seed: &[u8; SEED_LEN]) -> Result<(EncapsulationKey, DecapsulationKey), CryptoError> {
    let (ek, dk) = provider::kem_keypair_from_seed(seed).map_err(CryptoError::Provider)?;
    Ok((
        EncapsulationKey(provider::to_fixed::<EK_LEN>(ek, "ML-KEM-768 encapsulation key")?),
        DecapsulationKey(provider::to_fixed::<DK_LEN>(dk, "ML-KEM-768 decapsulation key")?),
    ))
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EncapsulationKey([u8; EK_LEN]);

impl fmt::Debug for EncapsulationKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlKem768Ek({:02x}{:02x}…)", self.0[0], self.0[1])
    }
}

impl EncapsulationKey {
    pub const fn from_bytes(bytes: [u8; EK_LEN]) -> EncapsulationKey {
        EncapsulationKey(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; EK_LEN] {
        &self.0
    }

    pub fn from_slice(bytes: &[u8]) -> Result<EncapsulationKey, CryptoError> {
        Ok(EncapsulationKey(provider::to_fixed::<EK_LEN>(
            bytes.to_vec(),
            "ML-KEM-768 encapsulation key",
        )?))
    }

    /// Encapsulate with explicit 32-byte randomness (deterministic and
    /// testable; wallets derive it from their own entropy).
    pub fn encapsulate(
        &self,
        randomness: &[u8; ENCAPS_RANDOMNESS_LEN],
    ) -> Result<(SharedSecret, CipherText), CryptoError> {
        let (ss, ct) = provider::kem_encapsulate(&self.0, randomness).map_err(CryptoError::Provider)?;
        Ok((
            SharedSecret(provider::to_fixed::<SS_LEN>(ss, "ML-KEM-768 shared secret")?),
            CipherText(provider::to_fixed::<CT_LEN>(ct, "ML-KEM-768 ciphertext")?),
        ))
    }
}

/// Secret decapsulation key; zeroized on drop.
#[derive(Clone, PartialEq, Eq)]
pub struct DecapsulationKey([u8; DK_LEN]);

impl Drop for DecapsulationKey {
    fn drop(&mut self) {
        self.0.as_mut_slice().zeroize();
    }
}

impl fmt::Debug for DecapsulationKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlKem768Dk(<redacted, {DK_LEN}B>)")
    }
}

impl DecapsulationKey {
    pub const fn from_bytes(bytes: [u8; DK_LEN]) -> DecapsulationKey {
        DecapsulationKey(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; DK_LEN] {
        &self.0
    }

    pub fn from_slice(bytes: &[u8]) -> Result<DecapsulationKey, CryptoError> {
        Ok(DecapsulationKey(provider::to_fixed::<DK_LEN>(
            bytes.to_vec(),
            "ML-KEM-768 decapsulation key",
        )?))
    }

    pub fn decapsulate(&self, ct: &CipherText) -> Result<SharedSecret, CryptoError> {
        let ss = provider::kem_decapsulate(&self.0, &ct.0).map_err(CryptoError::Provider)?;
        Ok(SharedSecret(provider::to_fixed::<SS_LEN>(ss, "ML-KEM-768 shared secret")?))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CipherText([u8; CT_LEN]);

impl fmt::Debug for CipherText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlKem768Ct({:02x}{:02x}…, {CT_LEN}B)", self.0[0], self.0[1])
    }
}

impl CipherText {
    pub const fn from_bytes(bytes: [u8; CT_LEN]) -> CipherText {
        CipherText(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; CT_LEN] {
        &self.0
    }
}

/// Shared secret; zeroized on drop, never appears in Debug output, has no
/// codec (never persisted).
#[derive(Clone, PartialEq, Eq)]
pub struct SharedSecret([u8; SS_LEN]);

impl Drop for SharedSecret {
    fn drop(&mut self) {
        self.0.as_mut_slice().zeroize();
    }
}

impl fmt::Debug for SharedSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SharedSecret(<redacted, {SS_LEN}B>)")
    }
}

impl SharedSecret {
    pub const fn as_bytes(&self) -> &[u8; SS_LEN] {
        &self.0
    }

    pub const fn from_bytes(bytes: [u8; SS_LEN]) -> SharedSecret {
        SharedSecret(bytes)
    }
}

impl Encode for EncapsulationKey {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0);
    }
    fn encoded_len(&self) -> usize {
        EK_LEN
    }
}
impl Decode for EncapsulationKey {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(EncapsulationKey(r.take_array::<EK_LEN>()?))
    }
}

impl Encode for DecapsulationKey {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0);
    }
    fn encoded_len(&self) -> usize {
        DK_LEN
    }
}
impl Decode for DecapsulationKey {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(DecapsulationKey(r.take_array::<DK_LEN>()?))
    }
}

impl Encode for CipherText {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0);
    }
    fn encoded_len(&self) -> usize {
        CT_LEN
    }
}
impl Decode for CipherText {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(CipherText(r.take_array::<CT_LEN>()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::DetRng;

    fn kp(seed: u64) -> (EncapsulationKey, DecapsulationKey) {
        keypair_from_seed(&DetRng::new(seed).bytes64()).unwrap()
    }

    #[test]
    fn keygen_deterministic() {
        let seed = DetRng::new(5).bytes64();
        let (ek1, dk1) = keypair_from_seed(&seed).unwrap();
        let (ek2, dk2) = keypair_from_seed(&seed).unwrap();
        assert_eq!(ek1, ek2);
        assert_eq!(dk1, dk2);
        let (ek3, _) = kp(6);
        assert_ne!(ek1, ek3);
        assert_eq!(ek1.as_bytes().len(), EK_LEN);
        assert_eq!(dk1.as_bytes().len(), DK_LEN);
    }

    #[test]
    fn encapsulate_deterministic_and_decapsulates() {
        let (ek, dk) = kp(7);
        let m = DetRng::new(8).bytes32();
        let (ss1, ct1) = ek.encapsulate(&m).unwrap();
        let (ss2, ct2) = ek.encapsulate(&m).unwrap();
        assert_eq!(ss1, ss2);
        assert_eq!(ct1, ct2);
        assert_eq!(ct1.as_bytes().len(), CT_LEN);
        let m2 = DetRng::new(9).bytes32();
        let (ss3, ct3) = ek.encapsulate(&m2).unwrap();
        assert_ne!(ct1, ct3);
        assert_ne!(ss1, ss3);
        assert_eq!(dk.decapsulate(&ct1).unwrap(), ss1);
        assert_eq!(dk.decapsulate(&ct3).unwrap(), ss3);
    }

    #[test]
    fn implicit_rejection_on_wrong_key() {
        let (ek, dk) = kp(10);
        let (_, dk_wrong) = kp(11);
        let (ss, ct) = ek.encapsulate(&DetRng::new(12).bytes32()).unwrap();
        let rejected = dk_wrong.decapsulate(&ct).unwrap();
        assert_ne!(rejected, ss);
        assert_eq!(dk.decapsulate(&ct).unwrap(), ss);
    }

    #[test]
    fn codec_roundtrips() {
        let (ek, dk) = kp(13);
        let (.., ct) = ek.encapsulate(&DetRng::new(14).bytes32()).unwrap();
        for len in [EK_LEN, DK_LEN, CT_LEN] {
            assert!(EncapsulationKey::decode(&vec![0u8; len - 1]).is_err());
        }
        assert_eq!(EncapsulationKey::decode(&ek.encode()).unwrap(), ek);
        assert_eq!(DecapsulationKey::decode(&dk.encode()).unwrap(), dk);
        assert_eq!(CipherText::decode(&ct.encode()).unwrap(), ct);
        let mut ext = ek.encode();
        ext.push(0);
        assert!(EncapsulationKey::decode(&ext).is_err());
        assert!(EncapsulationKey::from_slice(&[0u8; EK_LEN + 1]).is_err());
        assert!(DecapsulationKey::from_slice(&[0u8; 7]).is_err());
    }

    #[test]
    fn debug_redacts_secrets() {
        let (_, dk) = kp(15);
        let (ss, _) = kp(15).0.encapsulate(&DetRng::new(16).bytes32()).unwrap();
        let dk_dbg = format!("{dk:?}");
        let ss_dbg = format!("{ss:?}");
        assert!(dk_dbg.contains("redacted") && !dk_dbg.contains('x'));
        assert!(ss_dbg.contains("redacted"));
    }
}

