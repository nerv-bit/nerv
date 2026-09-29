//! The wallet's diversified address set (WP §8.6, §3.2; erratum 184):
//! deterministic generation from the detection seed, natural homing
//! across shards, the coverage map, and the per-address key material.


use std::collections::BTreeMap;


use nerv_core::types::{kappa, ShardId, ShardSet};
use nerv_crypto::mlkem::{DecapsulationKey, EncapsulationKey};
use nerv_custody::{
    Address, DetectionSeed, MasterSeed, WalletKeys,
};


/// The default number of distinct shards the wallet covers (§8.6).
pub const DEFAULT_SHARD_COVERAGE: usize = 8;
/// The generation cap: the coupon-collector bound at 64 shards and 8
/// coverage needs ~22 attempts on average; 128 is a generous ceiling.
pub const MAX_GENERATION: u64 = 128;


/// One wallet address: the public address plus its private key material.
/// The nk never leaves the wallet; the dk never leaves the wallet.
#[derive(Clone, Debug)]
pub struct WalletAddress {
    pub index: u64,
    pub address: Address,
    pub nk: [u8; 32],
    pub dk: DecapsulationKey,
    pub ek: EncapsulationKey,
}


impl WalletAddress {
    pub fn shard(&self) -> ShardId {
        self.address.tag()
    }
}


impl PartialEq for WalletAddress {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.address == other.address
    }
}


impl Eq for WalletAddress {}


/// The diversified address set (erratum 184): deterministic from the
/// wallet's keys; the first address in each distinct shard until the
/// coverage target is met.
#[derive(Clone, Debug)]
pub struct AddressSet {
    addresses: Vec<WalletAddress>,
    by_shard: BTreeMap<ShardId, Vec<u64>>,
    coverage_target: usize,
}


impl AddressSet {
    /// Generate the address set: sequential indices from the detection
    /// seed, keeping the first per distinct shard until the coverage
    /// target is met or the generation cap is hit.
    pub fn generate(
        keys: &WalletKeys,
        active: &ShardSet,
    ) -> Result<AddressSet, nerv_custody::CustodyError> {
        Self::generate_with_coverage(keys, active, DEFAULT_SHARD_COVERAGE)
    }


    pub fn generate_with_coverage(
        keys: &WalletKeys,
        active: &ShardSet,
        coverage: usize,
    ) -> Result<AddressSet, nerv_custody::CustodyError> {
        let mut addresses = Vec::new();
        let mut by_shard: BTreeMap<ShardId, Vec<u64>> = BTreeMap::new();
        let detection = keys.viewing();


        for index in 0..MAX_GENERATION {
            if by_shard.len() >= coverage {
                break;
            }
            let nk = keys.nullifier_key_at(index);
            let address = Address::generate(detection, keys.nullifier_key(), index, active)?;
            let kp = detection.delivery_keypair(index)?;
            let shard = address.tag();
            let is_new_shard = !by_shard.contains_key(&shard);
            if is_new_shard || by_shard.is_empty() {
                by_shard.entry(shard).or_default().push(index);
            }
            addresses.push(WalletAddress {
                index,
                address,
                nk,
                dk: kp.dk,
                ek: kp.ek,
            });
            // Only extend the map for new shards.
            if !is_new_shard {
                // The address was generated for determinism but doesn't
                // extend coverage; the map only tracks the first per shard.
                by_shard.entry(shard).or_default();
            }
        }
        Ok(AddressSet { addresses, by_shard, coverage_target: coverage })
    }


    /// All addresses, in generation order.
    pub fn addresses(&self) -> &[WalletAddress] {
        &self.addresses
    }


    /// The address indices homed to a shard.
    pub fn indices_for_shard(&self, shard: &ShardId) -> &[u64] {
        self.by_shard.get(shard).map_or(&[], |v| v.as_slice())
    }


    /// The addresses homed to a shard.
    pub fn addresses_for_shard(&self, shard: &ShardId) -> Vec<&WalletAddress> {
        self.indices_for_shard(shard)
            .iter()
            .filter_map(|&idx| self.addresses.iter().find(|a| a.index == idx))
            .collect()
    }


    pub fn covered_shards(&self) -> Vec<ShardId> {
        self.by_shard.keys().copied().collect()
    }


    pub fn coverage(&self) -> usize {
        self.by_shard.len()
    }


    pub fn coverage_target(&self) -> usize {
        self.coverage_target
    }


    pub fn len(&self) -> usize {
        self.addresses.len()
    }


    pub fn is_empty(&self) -> bool {
        self.addresses.is_empty()
    }


    pub fn get(&self, index: u64) -> Option<&WalletAddress> {
        self.addresses.iter().find(|a| a.index == index)
    }


    /// The address set for a specific shard: generates additional
    /// addresses if the shard is not yet covered.
    pub fn ensure_shard(
        &mut self,
        keys: &WalletKeys,
        active: &ShardSet,
        shard: ShardId,
    ) -> Result<(), nerv_custody::CustodyError> {
        if self.by_shard.contains_key(&shard) {
            return Ok(());
        }
        let detection = keys.viewing();
        for index in self.addresses.len() as u64..MAX_GENERATION {
            let address = Address::generate(detection, keys.nullifier_key(), index, active)?;
            if address.tag() == shard {
                let nk = keys.nullifier_key_at(index);
                let kp = detection.delivery_keypair(index)?;
                self.by_shard.insert(shard, vec![index]);
                self.addresses.push(WalletAddress {
                    index, address, nk, dk: kp.dk, ek: kp.ek,
                });
                return Ok(());
            }
            // Not the target shard; only add if it's a new shard.
            let tag = address.tag();
            if !self.by_shard.contains_key(&tag) {
                let nk = keys.nullifier_key_at(index);
                let kp = detection.delivery_keypair(index)?;
                self.by_shard.insert(tag, vec![index]);
                self.addresses.push(WalletAddress {
                    index, address, nk, dk: kp.dk, ek: kp.ek,
                });
            }
        }
        Err(nerv_custody::CustodyError::Derivation("generation cap exceeded"))
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;


    fn wallet(seed: u64) -> WalletKeys {
        WalletKeys::from_master(&MasterSeed::from_bytes(SplitMix64::new(seed).bytes32()))
    }


    #[test]
    fn pins() {
        assert_eq!(DEFAULT_SHARD_COVERAGE, 8);
        assert!(MAX_GENERATION >= 64);
    }


    #[test]
    fn deterministic_generation() {
        let set = ShardSet::genesis();
        let w1 = wallet(0xA1);
        let a1 = AddressSet::generate(&w1, &set).unwrap();
        let a2 = AddressSet::generate(&w1, &set).unwrap();
        assert_eq!(a1.len(), a2.len());
        assert_eq!(a1.coverage(), a2.coverage());
        for (x, y) in a1.addresses().iter().zip(a2.addresses()) {
            assert_eq!(x.index, y.index);
            assert_eq!(x.address, y.address);
            assert_eq!(x.nk, y.nk);
        }
        // Different wallets get different sets.
        let w2 = wallet(0xA2);
        let b = AddressSet::generate(&w2, &set).unwrap();
        assert_ne!(a1.addresses()[0].address, b.addresses()[0].address);
    }


    #[test]
    fn coverage_and_shard_distribution() {
        let set = ShardSet::genesis();
        let w = wallet(0xA3);
        let a = AddressSet::generate(&w, &set).unwrap();
        assert!(a.coverage() >= DEFAULT_SHARD_COVERAGE || a.len() >= MAX_GENERATION as usize);
        assert!(a.coverage() <= 64, "at most 64 distinct shards at genesis");
        assert!(!a.is_empty());
        // Every covered shard has at least one address.
        for shard in a.covered_shards() {
            assert!(!a.indices_for_shard(&shard).is_empty());
            let addrs = a.addresses_for_shard(&shard);
            assert!(!addrs.is_empty());
            assert!(addrs.iter().all(|x| x.shard() == shard));
        }
        // The address indices are sequential from 0.
        for (i, a) in a.addresses().iter().enumerate() {
            assert_eq!(a.index, i as u64);
        }
    }


    #[test]
    fn custom_coverage() {
        let set = ShardSet::genesis();
        let w = wallet(0xA4);
        let a = AddressSet::generate_with_coverage(&w, &set, 1).unwrap();
        assert!(a.coverage() >= 1);
        assert!(a.len() < AddressSet::generate(&w, &set).unwrap().len());
        let a3 = AddressSet::generate_with_coverage(&w, &set, 3).unwrap();
        assert!(a3.coverage() >= 3);
    }


    #[test]
    fn ensure_shard() {
        let set = ShardSet::genesis();
        let w = wallet(0xA5);
        let mut a = AddressSet::generate_with_coverage(&w, &set, 1).unwrap();
        let covered = a.covered_shards()[0];
        assert!(a.ensure_shard(&w, &set, covered).is_ok(), "already covered");
        // Find an uncovered shard.
        let target = set.ids().iter().find(|s| !a.by_shard.contains_key(s)).copied().unwrap();
        a.ensure_shard(&w, &set, target).unwrap();
        assert!(a.indices_for_shard(&target).len() >= 1);
        assert!(a.coverage() > 1);
    }


    #[test]
    fn nullifier_keys_are_per_address() {
        let set = ShardSet::genesis();
        let w = wallet(0xA6);
        let a = AddressSet::generate(&w, &set).unwrap();
        let mut seen = std::collections::BTreeSet::new();
        for addr in a.addresses() {
            assert!(seen.insert(addr.nk), "nk collision at index {}", addr.index);
        }
        // The nk differs from the master nk.
        for addr in a.addresses() {
            assert_ne!(addr.nk, *w.nullifier_key());
        }
    }


    #[test]
    fn address_shard_matches_kappa() {
        let set = ShardSet::genesis();
        let w = wallet(0xA7);
        let a = AddressSet::generate(&w, &set).unwrap();
        for addr in a.addresses() {
            let k = kappa(addr.ek.as_bytes());
            assert_eq!(addr.shard(), set.home_kappa(&k).unwrap());
        }
    }
}
