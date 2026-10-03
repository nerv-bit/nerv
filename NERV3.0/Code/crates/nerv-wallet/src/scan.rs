//! The note scanner (WP §3.2, §6.1; erratum 185): trial-decryption of
//! home-shard note streams, the wallet's note set, nullifier management.


use std::collections::{BTreeMap, BTreeSet};


use nerv_core::codec::Decode;
use nerv_core::hash::Hash256;
use nerv_core::types::{Height, LegIndex, ShardId, TxId};
use nerv_custody::nct::NctDigest;
use nerv_custody::note::{SealedNote, trial_decrypt};
use nerv_custody::NoteOpening;
use nerv_custody::nullifier::derive_nullifier;


use crate::keys::WalletAddress;


/// A scanned note: everything the wallet learned from one successful
/// trial-decryption (erratum 185).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScannedNote {
    pub opening: NoteOpening,
    pub nullifier_key: [u8; 32],
    pub address_index: u64,
    /// The precomputed nullifier (nk ‖ ρ → H("nerv.nf" ‖ nk ‖ ρ)).
    pub nullifier: Hash256,
    pub memo: Option<Vec<u8>>,
}


/// The wallet's note set: received notes with spend tracking.
#[derive(Clone, Debug, Default)]
pub struct WalletNoteSet {
    /// Keyed by commitment.
    notes: BTreeMap<[u8; 32], ScannedNote>,
    /// The wallet's local spent-nullifier set.
    spent: BTreeSet<[u8; 32]>,
    /// The spending txid per nullifier.
    spent_by: BTreeMap<[u8; 32], TxId>,
    /// NCT positions: commitment → (leaf_index, anchor, witness).
    positions: BTreeMap<[u8; 32], NotePosition>,
}


/// The note's position in the NCT and its Merkle witness.
#[derive(Clone, Debug)]
pub struct NotePosition {
    pub leaf_index: u64,
    pub anchor: NctDigest,
    pub siblings: Vec<nerv_core::field::Goldilocks>,
    pub shard: ShardId,
    pub received_height: u64,
}


impl WalletNoteSet {
    pub fn new() -> WalletNoteSet {
        WalletNoteSet::default()
    }


    pub fn len(&self) -> usize {
        self.notes.len()
    }


    pub fn is_empty(&self) -> bool {
        self.notes.is_empty()
    }


    pub fn unspent_len(&self) -> usize {
        self.notes.len() - self.spent.len()
    }


    /// The total unspent value.
    pub fn unspent_value(&self) -> u64 {
        self.notes
            .values()
            .filter(|n| !self.spent.contains(n.nullifier.as_bytes()))
            .map(|n| n.opening.value)
            .sum()
    }



    /// Insert a scanned note (from trial-decryption).
    pub fn insert(&mut self, note: ScannedNote) -> bool {
        let key = match note.opening.commitment() {
            Ok(cm) => *cm.as_bytes(),
            Err(_) => return false,
        };
        if self.notes.contains_key(&key) {
            return false;
        }
        self.notes.insert(key, note);
        true
    }


    /// Mark a nullifier spent.
    pub fn mark_spent(&mut self, nullifier: &Hash256, txid: TxId) -> bool {
        let key = *nullifier.as_bytes();
        if self.spent.insert(key) {
            self.spent_by.insert(key, txid);
            true
        } else {
            false
        }
    }


    /// Is the nullifier spent (locally)?
    pub fn is_spent(&self, nullifier: &Hash256) -> bool {
        self.spent.contains(nullifier.as_bytes())
    }


    /// Get a note by its commitment.
    pub fn get(&self, commitment: &[u8; 32]) -> Option<&ScannedNote> {
        self.notes.get(commitment)
    }


    /// Get the note's NCT position.
    pub fn position(&self, commitment: &[u8; 32]) -> Option<&NotePosition> {
        self.positions.get(commitment)
    }


    /// Record the NCT position for a note.
    pub fn set_position(&mut self, commitment: [u8; 32], pos: NotePosition) {
        self.positions.insert(commitment, pos);
    }


    /// All unspent notes (in deterministic order).
    pub fn unspent(&self) -> Vec<(&[u8; 32], &ScannedNote)> {
        self.notes
            .iter()
            .filter(|(_, n)| !self.spent.contains(n.nullifier.as_bytes()))
            .map(|(cm, n)| (cm, n))
            .collect()
    }


    /// Unspent notes homed to a specific shard.
    pub fn unspent_for_shard(&self, shard: &ShardId) -> Vec<(&[u8; 32], &ScannedNote)> {
        self.notes
            .iter()
            .filter(|(cm, n)| {
                !self.spent.contains(n.nullifier.as_bytes())
                    && self.positions.get(*cm).is_some_and(|p| &p.shard == shard)
            })
            .map(|(cm, n)| (cm, n))
            .collect()
    }



    /// The wallet's nullifiers (all notes, spent and unspent).
    pub fn nullifiers(&self) -> Vec<Hash256> {
        self.notes.values().map(|n| n.nullifier).collect()
    }


    /// The wallet's spent nullifiers (for local double-spend prevention).
    pub fn spent_nullifiers(&self) -> Vec<Hash256> {
        self.spent
            .iter()
            .filter_map(|b| {
                let arr: [u8; 32] = b[..32].try_into().ok()?;
                Some(Hash256::from_bytes(arr))
            })
            .collect()
    }
}


/// Scan one sealed note: try every address homed to the note's shard.
/// Returns the scanned note if any address matches (erratum 185).
pub fn scan_sealed_note(
    addresses: &[WalletAddress],
    sealed_note_bytes: &[u8],
    published_cm: &Hash256,
    note_shard: ShardId,
) -> Option<ScannedNote> {
    let sealed = SealedNote::decode(sealed_note_bytes).ok()?;
    for addr in addresses {
        if addr.shard() != note_shard {
            continue;
        }
        let kp = nerv_custody::DeliveryKeyPair {
            index: addr.index,
            ek: addr.ek,
            dk: addr.dk.clone(),
        };

        let pk_n = addr.address.pk_n();
        if let Ok(note) = trial_decrypt(&sealed, &kp, published_cm, &pk_n) {
            let opening = NoteOpening {
                value: note.value,
                rho: note.rho,
                delivery: note.delivery,
                blinding: note.blinding,
                pk_n: note.pk_n,
            };
            let nullifier = derive_nullifier(&addr.nk, &note.rho);
            return Some(ScannedNote {
                opening,
                nullifier_key: addr.nk,
                address_index: addr.index,
                nullifier,
                memo: note.memo().map(|m| m.to_vec()),
            });
        }
    }
    None
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::keys::AddressSet;
    use crate::testutil::SplitMix64;
    use nerv_core::types::ShardSet;
    use nerv_custody::{
        MasterSeed, NotePlaintext, WalletKeys, seal_note,
    };


    fn wallet(seed: u64) -> WalletKeys {
        WalletKeys::from_master(&MasterSeed::from_bytes(SplitMix64::new(seed).bytes32()))
    }


    fn make_and_seal(
        keys: &WalletKeys,
        addr_set: &AddressSet,
        value: u64,
        seed: u64,
    ) -> (Hash256, Vec<u8>, ShardId) {
        let addr = &addr_set.addresses()[0];
        let mut rng = SplitMix64::new(seed);
        let pt = NotePlaintext::new(
            value,
            rng.bytes32(),
            rng.bytes32(),
            None,
        )
        .unwrap();
        let kem_rand = rng.bytes32();
        let (cm, sealed) = seal_note(&pt, &addr.address, &kem_rand).unwrap();
        (cm, sealed.encode(), addr.shard())
    }


    #[test]
    fn scan_finds_note_for_correct_address() {
        let set = ShardSet::genesis();
        let w = wallet(0x5CA1);
        let addrs = AddressSet::generate(&w, &set).unwrap();
        let (cm, sealed_bytes, shard) = make_and_seal(&w, &addrs, 1_000_000_000, 0x5EED);


        let scanned = scan_sealed_note(addrs.addresses(), &sealed_bytes, &cm, shard);
        assert!(scanned.is_some());
        let sn = scanned.unwrap();
        assert_eq!(sn.opening.value, 1_000_000_000);
        assert_eq!(sn.address_index, addrs.addresses()[0].index);
        assert_eq!(sn.nullifier_key, addrs.addresses()[0].nk);
        assert_eq!(
            sn.nullifier,
            derive_nullifier(&sn.nullifier_key, &sn.opening.rho)
        );
        // The opening's commitment matches.
        assert_eq!(sn.opening.commitment().unwrap(), cm);
    }


    #[test]
    fn scan_rejects_foreign_note() {
        let set = ShardSet::genesis();
        let w1 = wallet(0x5CA2);
        let w2 = wallet(0x5CA3);
        let a1 = AddressSet::generate(&w1, &set).unwrap();
        let a2 = AddressSet::generate(&w2, &set).unwrap();


        // A note sealed to wallet 2's first address.
        let (cm, sealed_bytes, shard) = make_and_seal(&w2, &a2, 500, 0x5EE2);


        // Wallet 1 tries to scan it: no match (assuming different first
        // addresses — overwhelmingly likely with different seeds).
        let a1_first_shard_addrs: Vec<&WalletAddress> =
            a1.addresses_for_shard(&shard).into_iter().chain(
                a1.addresses().iter().filter(|a| a.shard() == shard)
            ).collect();
        let all_a1: Vec<WalletAddress> = a1.addresses().to_vec();
        let result = scan_sealed_note(&all_a1, &sealed_bytes, &cm, shard);
        // With overwhelming probability, wallet 1 has no key for this note.
        if a1.addresses()[0].address != a2.addresses()[0].address {
            assert!(result.is_none(), "foreign note must not decrypt");
        }
    }


    #[test]
    fn scan_wrong_shard_returns_none() {
        let set = ShardSet::genesis();
        let w = wallet(0x5CA4);
        let addrs = AddressSet::generate(&w, &set).unwrap();
        let (cm, sealed_bytes, _shard) = make_and_seal(&w, &addrs, 100, 0x5EE3);


        // Try with a different shard: the addresses are filtered by shard,
        // so the scan returns None.
        let other_shard = set.ids()[40];
        if other_shard != addrs.addresses()[0].shard() {
            assert!(scan_sealed_note(addrs.addresses(), &sealed_bytes, &cm, other_shard).is_none());
        }
    }


    #[test]
    fn scan_malformed_bytes() {
        let set = ShardSet::genesis();
        let w = wallet(0x5CA5);
        let addrs = AddressSet::generate(&w, &set).unwrap();
        let cm = Hash256::from_bytes([0u8; 32]);
        let shard = addrs.addresses()[0].shard();
        assert!(scan_sealed_note(addrs.addresses(), &[], &cm, shard).is_none());
        assert!(scan_sealed_note(addrs.addresses(), &[0xFF; 10], &cm, shard).is_none());
    }


    #[test]
    fn note_set_lifecycle() {
        let mut ns = WalletNoteSet::new();
        assert!(ns.is_empty());
        assert_eq!(ns.unspent_len(), 0);


        let set = ShardSet::genesis();
        let w = wallet(0x5CA6);
        let addrs = AddressSet::generate(&w, &set).unwrap();
        let (cm1, sealed1, shard) = make_and_seal(&w, &addrs, 100, 0x5EE4);
        let (cm2, sealed2, _) = make_and_seal(&w, &addrs, 200, 0x5EE5);


        let sn1 = scan_sealed_note(addrs.addresses(), &sealed1, &cm1, shard).unwrap();
        let sn2 = scan_sealed_note(addrs.addresses(), &sealed2, &cm2, shard).unwrap();


        assert!(ns.insert(sn1));
        assert_eq!(ns.len(), 1);
        assert_eq!(ns.unspent_value(), 100);
        assert!(ns.insert(sn2));
        assert_eq!(ns.len(), 2);
        assert_eq!(ns.unspent_value(), 300);


        // Duplicate insert (same commitment) is a no-op.
        let sn1_again = scan_sealed_note(addrs.addresses(), &sealed1, &cm1, shard).unwrap();
        assert!(!ns.insert(sn1_again));
        assert_eq!(ns.len(), 2);


        // Spend one.
        let txid = TxId::from_hash(Hash256::from_bytes([9u8; 32]));
        let nf1 = ns.get(cm1.as_bytes()).unwrap().nullifier;
        assert!(ns.mark_spent(&nf1, txid));
        assert_eq!(ns.unspent_len(), 1);
        assert_eq!(ns.unspent_value(), 200);
        assert!(!ns.mark_spent(&nf1, txid), "double-spend rejected");


        // The unspent list excludes the spent note.
        let unspent = ns.unspent();
        assert_eq!(unspent.len(), 1);
        assert_eq!(unspent[0].1.opening.value, 200);


        // Nullifiers.
        assert_eq!(ns.nullifiers().len(), 2);
        assert_eq!(ns.spent_nullifiers().len(), 1);
    }


    #[test]
    fn note_set_with_positions() {
        let mut ns = WalletNoteSet::new();
        let set = ShardSet::genesis();
        let w = wallet(0x5CA7);
        let addrs = AddressSet::generate(&w, &set).unwrap();
        let (cm, sealed, shard) = make_and_seal(&w, &addrs, 500, 0x5EE6);
        let sn = scan_sealed_note(addrs.addresses(), &sealed, &cm, shard).unwrap();
        let key = *cm.as_bytes();
        ns.insert(sn);


        let pos = NotePosition {
            leaf_index: 42,
            anchor: nerv_custody::nct::NoteCommitmentTree::new().root(),
            siblings: vec![],
            shard,
            received_height: 100,
        };
        ns.set_position(key, pos);
        assert!(ns.position(&key).is_some());
        assert_eq!(ns.position(&key).unwrap().leaf_index, 42);
        assert!(ns.position(&key).unwrap().received_height == 100);
    }
}
