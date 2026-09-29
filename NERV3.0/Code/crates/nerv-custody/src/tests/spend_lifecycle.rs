#![allow(clippy::unwrap_used, clippy::expect_used)]

use nerv_core::hash::Hash256;
use nerv_core::types::{LegIndex, ShardSet, TxId};
use nerv_custody::address::{Address, MasterSeed, WalletKeys};
use nerv_custody::burn::BurnCommitment;
use nerv_custody::error::CustodyError;
use nerv_custody::nullifier::{derive_nullifier, NullifierSet};
use nerv_custody::note::{seal_note, trial_decrypt, NotePlaintext};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn bytes32(&mut self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for c in out.chunks_exact_mut(8) {
            c.copy_from_slice(&self.next().to_le_bytes());
        }
        out
    }
}

const ALICE: &str = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";

#[test]
fn notes_to_nullifiers_to_spent_set() {
    let alice = WalletKeys::from_master(&MasterSeed::from_bip39(ALICE, "alice"));
    let genesis = ShardSet::genesis();
    let addr = Address::generate(alice.detection(), 3, &genesis).unwrap();
    let kp = alice.detection().delivery_keypair(3).unwrap();
    let mut rng = Rng(0x5EED_11FE);

    // two notes sealed to Alice at the same address
    let n1 = NotePlaintext::new(700_000_000_000, rng.bytes32(), rng.bytes32(), None).unwrap();
    let n2 = NotePlaintext::new(300_000_000_000, rng.bytes32(), rng.bytes32(), None).unwrap();
    let (cm1, s1) = seal_note(&n1, addr.delivery(), &rng.bytes32()).unwrap();
    let (cm2, s2) = seal_note(&n2, addr.delivery(), &rng.bytes32()).unwrap();
    let note1 = trial_decrypt(&s1, &kp, &cm1).unwrap();
    let note2 = trial_decrypt(&s2, &kp, &cm2).unwrap();

    // nullifiers derive from Alice's nk and each note's unique ρ
    let nf1 = derive_nullifier(alice.nullifier_key(), &note1.rho);
    let nf2 = derive_nullifier(alice.nullifier_key(), &note2.rho);
    assert_ne!(nf1, nf2, "distinct ρ ⇒ distinct nullifiers (one-time tags)");
    // nf is unlinkable to cm: it is a fresh BLAKE3 tag over secret material
    assert_ne!(nf1.as_hash(), &cm1);
    assert_ne!(nf1.as_hash(), &cm2);

    // spend note 1 at height 512: the shard's spent set
    let mut spent = NullifierSet::new();
    assert!(!spent.contains(&nf1));
    let fresh = spent.non_membership_proof(&nf1);
    assert!(fresh.verify_non_membership(&spent.root(), &nf1));

    spent.insert(&nf1, 512).unwrap();
    assert!(spent.contains(&nf1));
    assert!(!spent.contains(&nf2));

    // note 2 remains spendable: non-membership against the updated root
    let root = spent.root();
    let fresh2 = spent.non_membership_proof(&nf2);
    assert!(fresh2.verify_non_membership(&root, &nf2));
    // note 1's membership is publicly provable (fraud/audit path)
    let spent_proof = spent.membership_proof(&nf1).unwrap();
    assert!(spent_proof.verify_membership(&root, &nf1, 512));

    // double-spend of note 1 is rejected — at the tree, and in-batch
    assert!(matches!(
        spent.insert(&nf1, 513),
        Err(CustodyError::NullifierAlreadySpent { .. })
    ));
    let nf3 = derive_nullifier(alice.nullifier_key(), &Hash256::from_bytes(rng.bytes32()).as_bytes());
    assert!(matches!(
        spent.insert_batch(&[nf2, nf3, nf2], 514),
        Err(CustodyError::DuplicateNullifier { .. })
    ));
    assert_eq!(spent.len(), 1, "failed batch changed nothing");
    spent.validate_consistency().unwrap();
}

#[test]
fn burn_exit_from_a_spent_note() {
    let mut rng = Rng(0xB04A);
    // a burn leg: spend with no output; the public commitment binds value
    let legs = b"canonical serialization of the burn transaction's legs";
    let txid = TxId::hash_canonical(legs);
    let leg = LegIndex::from_u8(0);
    let value = 250_000_000_000u64;
    let burn = BurnCommitment::new(&txid, leg, value).unwrap();
    assert!(burn.verify(&txid, leg, value));
    // anyone recomputes it from public leg data
    assert!(!burn.verify(&txid, leg, value + 1));
    // serialization survives the wire
    let enc = burn.encode();
    assert_eq!(BurnCommitment::decode(&enc).unwrap().verify(&txid, leg, value), true);
}
