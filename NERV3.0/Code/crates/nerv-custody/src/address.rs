//! Key hierarchy, diversified delivery keys, address homing, and the
//! per-address nullifier keys (WP §3.2, §8.2; erratum 72). κ is computed
//! over the delivery key; the shard tag is the generation-time longest
//! active prefix of κ. Addresses carry pk_n = H("nerv.nf.pk" ‖ nk_j) —
//! the nullifier-key commitment the sender places in the note commitment.

use std::fmt;
use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::{KEY_DELIVERY, KEY_DIVERSIFY, KEY_NULLIFIER, KEY_SPEND};
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::types::{kappa, ShardId, ShardSet};
use nerv_crypto::kdf::{bip39_seed, blake3_kdf, blake3_kdf_stream};
use nerv_crypto::mlkem::{DecapsulationKey, EncapsulationKey, EK_LEN};
use zeroize::Zeroize;

use crate::commitment::nullifier_pk;
use crate::error::CustodyError;

#[derive(Clone)]
pub struct MasterSeed([u8; 32]);

impl fmt::Debug for MasterSeed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("MasterSeed(<redacted>)")
    }
}

impl Drop for MasterSeed {
    fn drop(&mut self) {
        self.0.as_mut_slice().zeroize();
    }
}

impl MasterSeed {
    pub const fn from_bytes(bytes: [u8; 32]) -> MasterSeed {
        MasterSeed(bytes)
    }

    pub fn from_bip39(phrase: &str, passphrase: &str) -> MasterSeed {
        let mut seed = bip39_seed(phrase, passphrase);
        let mut master = [0u8; 32];
        master.copy_from_slice(&seed[..32]);
        seed.as_mut_slice().zeroize();
        MasterSeed(master)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone)]
pub struct WalletKeys {
    spend: [u8; 32],
    nullifier: [u8; 32],
    detection: DetectionSeed,
}

impl fmt::Debug for WalletKeys {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("WalletKeys(<redacted: sk_spend, nk, detection>)")
    }
}

impl Drop for WalletKeys {
    fn drop(&mut self) {
        self.spend.as_mut_slice().zeroize();
        self.nullifier.as_mut_slice().zeroize();
        self.detection.0.as_mut_slice().zeroize();
    }
}

impl WalletKeys {
    pub fn from_master(seed: &MasterSeed) -> WalletKeys {
        WalletKeys {
            spend: blake3_kdf(&KEY_SPEND, seed.as_bytes(), &[]),
            nullifier: blake3_kdf(&KEY_NULLIFIER, seed.as_bytes(), &[]),
            detection: DetectionSeed(blake3_kdf(&KEY_DIVERSIFY, seed.as_bytes(), &[])),
        }
    }

    pub fn spend_seed(&self) -> &[u8; 32] {
        &self.spend
    }

    /// The nullifier-key tree seed (the master-level nk; erratum 72).
    pub fn nullifier_key(&self) -> &[u8; 32] {
        &self.nullifier
    }

    /// The per-address nullifier key nk_j = KDF(KEY_NULLIFIER, nk, j) —
    /// the key whose H(nk_j) is committed in notes received at address j.
    pub fn nullifier_key_at(&self, index: u64) -> [u8; 32] {
        let mut out = [0u8; 32];
        blake3_kdf_stream(&KEY_NULLIFIER, &self.nullifier, &index.to_le_bytes(), &mut out);
        out
    }

    pub const fn detection(&self) -> &DetectionSeed {
        &self.detection
    }

    pub fn viewing(&self) -> DetectionSeed {
        self.detection.clone()
    }
}

#[derive(Clone)]
pub struct DetectionSeed([u8; 32]);

impl fmt::Debug for DetectionSeed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DetectionSeed(<redacted>)")
    }
}

impl Drop for DetectionSeed {
    fn drop(&mut self) {
        self.0.as_mut_slice().zeroize();
    }
}

impl DetectionSeed {
    pub const fn from_bytes(bytes: [u8; 32]) -> DetectionSeed {
        DetectionSeed(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn delivery_keypair(&self, index: u64) -> Result<DeliveryKeyPair, nerv_crypto::CryptoError> {
        let mut kem_seed = [0u8; 64];
        blake3_kdf_stream(&KEY_DELIVERY, self.as_bytes(), &index.to_le_bytes(), &mut kem_seed);
        let (ek, dk) = nerv_crypto::mlkem::keypair_from_seed(&kem_seed)?;
        kem_seed.as_mut_slice().zeroize();
        Ok(DeliveryKeyPair { index, ek, dk })
    }
}

#[derive(Clone, Debug)]
pub struct DeliveryKeyPair {
    pub index: u64,
    pub ek: EncapsulationKey,
    pub dk: DecapsulationKey,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Address {
    delivery: EncapsulationKey,
    tag: ShardId,
    pk_n: [u8; 32],
}

impl fmt::Debug for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Address")
            .field("tag", &self.tag.to_string())
            .field("delivery", &self.delivery)
            .field("pk_n", &"..32B".to_string())
            .finish()
    }
}

impl Address {
    pub fn new(
        delivery: EncapsulationKey,
        tag: ShardId,
        pk_n: [u8; 32],
    ) -> Result<Address, CustodyError> {
        if !tag.is_kappa_home(&Self::kappa_of(&delivery)) {
            return Err(CustodyError::InvalidShardTag);
        }
        Ok(Address { delivery, tag, pk_n })
    }

    /// Generate address `index`: delivery key from the detection seed,
    /// nk_j from the nullifier tree seed, pk_n = H(nk_j), tag = home(κ).
    pub fn generate(
        detection: &DetectionSeed,
        nk_tree: &[u8; 32],
        index: u64,
        set: &ShardSet,
    ) -> Result<Address, CustodyError> {
        let kp = detection
            .delivery_keypair(index)
            .map_err(|_| CustodyError::Derivation("ml-kem delivery keygen failed"))?;
        let mut nk = [0u8; 32];
        blake3_kdf_stream(&KEY_NULLIFIER, nk_tree, &index.to_le_bytes(), &mut nk);
        let tag = set
            .home_kappa(&kappa(kp.ek.as_bytes()))
            .map_err(|_| CustodyError::Derivation("homing lookup failed"))?;
        Ok(Address { delivery: kp.ek, tag, pk_n: nullifier_pk(&nk) })
    }

    fn kappa_of(delivery: &EncapsulationKey) -> Hash256 {
        kappa(delivery.as_bytes())
    }

    pub fn kappa(&self) -> Hash256 {
        Self::kappa_of(&self.delivery)
    }

    pub const fn delivery(&self) -> &EncapsulationKey {
        &self.delivery
    }

    pub const fn tag(&self) -> ShardId {
        self.tag
    }

    pub const fn pk_n(&self) -> &[u8; 32] {
        &self.pk_n
    }

    pub fn home_under(&self, set: &ShardSet) -> Result<ShardId, CustodyError> {
        set.home_kappa(&self.kappa())
            .map_err(|_| CustodyError::Derivation("homing lookup failed"))
    }
}

impl Encode for Address {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.delivery.encode_into(out);
        self.tag.encode_into(out);
        out.extend_from_slice(&self.pk_n);
    }
    fn encoded_len(&self) -> usize {
        EK_LEN + 3 + 32
    }
}

impl Decode for Address {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let delivery = EncapsulationKey::decode_from(r)?;
        let tag = ShardId::decode_from(r)?;
        let pk_n = r.take_array::<32>()?;
        Address::new(delivery, tag, pk_n)
            .map_err(|_| CodecError::InvariantViolated("shard tag is not a prefix of the homing key"))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;
    use std::collections::HashSet;

    fn wallet(seed: u64) -> WalletKeys {
        WalletKeys::from_master(&MasterSeed::from_bytes(SplitMix64::new(seed).bytes32()))
    }

    const P1: &str = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";

    #[test]
    fn hierarchy_deterministic_and_separated() {
        let m = MasterSeed::from_bytes([7u8; 32]);
        let a = WalletKeys::from_master(&m);
        let b = WalletKeys::from_master(&m);
        assert_eq!(a.spend_seed(), b.spend_seed());
        assert_eq!(a.nullifier_key(), b.nullifier_key());
        assert_eq!(a.detection().as_bytes(), b.detection().as_bytes());
        let c = WalletKeys::from_master(&MasterSeed::from_bytes([8u8; 32]));
        assert_ne!(a.nullifier_key(), c.nullifier_key());
        assert_ne!(a.spend_seed(), a.nullifier_key());
        assert_ne!(a.spend_seed(), a.detection().as_bytes());
        assert_ne!(a.nullifier_key(), a.detection().as_bytes());
        assert_eq!(a.nullifier_key(), &blake3_kdf(&KEY_NULLIFIER, m.as_bytes(), &[]));
    }

    #[test]
    fn nullifier_keys_per_address() {
        let w = wallet(1);
        assert_eq!(w.nullifier_key_at(5), w.nullifier_key_at(5));
        let mut seen = HashSet::new();
        for i in 0..64u64 {
            assert!(seen.insert(w.nullifier_key_at(i)), "nk_j collision at {i}");
        }
        let other = wallet(2);
        assert_ne!(w.nullifier_key_at(5), other.nullifier_key_at(5));
        // pk_n is a one-way commitment of nk_j: distinct per index, and the
        // tree seed alone does not reveal it.
        assert_ne!(nullifier_pk(&w.nullifier_key_at(0)), nullifier_pk(&w.nullifier_key_at(1)));
        assert_ne!(nullifier_pk(&w.nullifier_key_at(0)), *w.nullifier_key());
    }

    #[test]
    fn bip39_path() {
        let a = MasterSeed::from_bip39(P1, "pass");
        let b = MasterSeed::from_bip39(P1, "pass");
        assert_eq!(a.as_bytes(), b.as_bytes());
        assert_ne!(a.as_bytes(), MasterSeed::from_bip39(P1, "other").as_bytes());
    }

    #[test]
    fn address_generation_homing_and_pk_n() {
        let w = wallet(4);
        let genesis = ShardSet::genesis();
        let mut tags = HashSet::new();
        for i in 0..12u64 {
            let addr = Address::generate(w.detection(), w.nullifier_key(), i, &genesis).unwrap();
            assert_eq!(addr.tag().bits(), 6);
            assert_eq!(addr.tag(), genesis.home_kappa(&addr.kappa()).unwrap());
            assert_eq!(addr.pk_n(), &nullifier_pk(&w.nullifier_key_at(i)));
            assert_eq!(
                Address::new(addr.delivery(), addr.tag(), addr.pk_n).unwrap().pk_n(),
                addr.pk_n()
            );
            tags.insert(addr.tag());
        }
        assert!(tags.len() > 1);
    }

    #[test]
    fn address_rejects_forged_tag() {
        let w = wallet(5);
        let genesis = ShardSet::genesis();
        let addr = Address::generate(w.detection(), w.nullifier_key(), 0, &genesis).unwrap();
        let wrong = ShardId::new(6, addr.tag().value() ^ 1).unwrap();
        assert!(matches!(
            Address::new(addr.delivery(), wrong, addr.pk_n()),
            Err(CustodyError::InvalidShardTag)
        ));
        let parent = addr.tag().parent().unwrap();
        assert!(Address::new(addr.delivery(), parent, addr.pk_n()).is_ok());
    }

    #[test]
    fn legacy_tag_survives_split() {
        let w = wallet(6);
        let genesis = ShardSet::genesis();
        let addr = Address::generate(w.detection(), w.nullifier_key(), 5, &genesis).unwrap();
        let children = genesis.split(&addr.tag()).unwrap();
        assert!(Address::new(addr.delivery(), addr.tag(), addr.pk_n()).is_ok());
        assert!(addr.home_under(&children).unwrap().is_prefix_of(&addr.tag()));
    }

    #[test]
    fn address_codec_roundtrip() {
        let w = wallet(7);
        let genesis = ShardSet::genesis();
        let addr = Address::generate(w.detection(), w.nullifier_key(), 2, &genesis).unwrap();
        let enc = addr.encode();
        assert_eq!(enc.len(), EK_LEN + 3 + 32);
        assert_eq!(Address::decode(&enc).unwrap(), addr);
        assert!(Address::decode(&enc[..enc.len() - 33]).is_err());
        let mut forged = enc.clone();
        forged[EK_LEN + 1] ^= 1;
        assert!(Address::decode(&forged).is_err());
    }

    #[test]
    fn debug_redacts() {
        let w = wallet(8);
        assert!(format!("{w:?}").contains("redacted"));
        assert!(format!("{:?}", MasterSeed::from_bytes([1u8; 32])).contains("redacted"));
    }
}
