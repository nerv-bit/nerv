#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashSet;

use nerv_core::codec::{Decode, Encode};
use nerv_core::hash::Hash256;
use nerv_core::types::ShardSet;
use nerv_custody::address::{Address, MasterSeed, WalletKeys};
use nerv_custody::error::NoteError;
use nerv_custody::nct::{verify_witness, NoteCommitmentTree};
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

const BOB_PHRASE: &str = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
const CAROL_PHRASE: &str = "legal winner thank year wave sausage worth useful legal winner thank yellow";

#[test]
fn end_to_end_payment_and_recovery() {
    let bob = WalletKeys::from_master(&MasterSeed::from_bip39(BOB_PHRASE, "bob"));
    let genesis = ShardSet::genesis();

    let bob_addrs: Vec<Address> =
        (0..6).map(|i| Address::generate(bob.detection(), i, &genesis).unwrap()).collect();
    let mut eks = HashSet::new();
    for a in &bob_addrs {
        assert_eq!(a.tag().bits(), 6);
        assert!(eks.insert(*a.delivery().as_bytes()));
        assert_eq!(a.tag(), genesis.home_kappa(&a.kappa()).unwrap());
    }
    assert_eq!(eks.len(), 6, "diversified delivery keys must be distinct");

    // the recipient publishes address #3; it travels over the wire intact
    let wire = bob_addrs[3].encode();
    let addr = Address::decode(&wire).unwrap();
    assert_eq!(addr, bob_addrs[3]);

    // Alice seals a note to it (value, rho, blinding, KEM randomness all
    // caller-supplied — the wallet's entropy)
    let mut rng = Rng(0xB0B5EED);
    let plaintext = NotePlaintext::new(
        2_500_000_000_000,
        rng.bytes32(),
        rng.bytes32(),
        Some(b"invoice 42".to_vec()),
    )
    .unwrap();
    let (cm, sealed) = seal_note(&plaintext, addr.delivery(), &rng.bytes32()).unwrap();
    assert!(sealed.wire_len() <= 1300);

    // the shard settles the output commitment among decoys
    let mut nct = NoteCommitmentTree::new();
    for _ in 0..4 {
        nct.append(&Hash256::from_bytes(rng.bytes32())).unwrap();
    }
    let idx = nct.append(&cm).unwrap();
    nct.append(&Hash256::from_bytes(rng.bytes32())).unwrap();
    let root = nct.root();
    let w = nct.witness(idx).unwrap();
    assert!(verify_witness(&root, idx, &cm, &w.siblings));

    // Bob scans the shard's note stream: exactly one delivery key opens it
    let mut recovered = 0;
    for i in 0..6u64 {
        let kp = bob.detection().delivery_keypair(i).unwrap();
        match trial_decrypt(&sealed, &kp, &cm) {
            Ok(note) => {
                recovered += 1;
                assert_eq!(i, 3);
                assert_eq!(note.commitment().unwrap(), cm);
                assert_eq!(note.value, 2_500_000_000_000);
                assert_eq!(note.rho, plaintext.rho);
                assert_eq!(note.blinding, plaintext.blinding);
                assert_eq!(note.memo(), Some(&b"invoice 42"[..]));
                // the recovered note is the tree's member — spendable
                assert!(verify_witness(&root, idx, &note.commitment().unwrap(), &w.siblings));
                // the note's plaintext re-seals to the same commitment
                let p2 = note.plaintext();
                let (cm2, _) = seal_note(&p2, addr.delivery(), &rng.bytes32()).unwrap();
                assert_eq!(cm2, cm);
            }
            Err(NoteError::DecryptionFailed) => {}
            Err(e) => panic!("unexpected error for key {i}: {e}"),
        }
    }
    assert_eq!(recovered, 1);
}

#[test]
fn wrong_wallet_finds_nothing() {
    let bob = WalletKeys::from_master(&MasterSeed::from_bip39(BOB_PHRASE, "bob"));
    let carol = WalletKeys::from_master(&MasterSeed::from_bip39(CAROL_PHRASE, "carol"));
    let genesis = ShardSet::genesis();

    let target = Address::generate(bob.detection(), 0, &genesis).unwrap();
    let mut rng = Rng(0xCA401);
    let plaintext = NotePlaintext::new(999, rng.bytes32(), rng.bytes32(), None).unwrap();
    let (cm, sealed) = seal_note(&plaintext, target.delivery(), &rng.bytes32()).unwrap();

    for i in 0..4u64 {
        let kp = carol.detection().delivery_keypair(i).unwrap();
        assert!(matches!(
            trial_decrypt(&sealed, &kp, &cm),
            Err(NoteError::DecryptionFailed)
        ));
    }
    // and Bob still opens it with his key for that index
    let kp = bob.detection().delivery_keypair(0).unwrap();
    assert!(trial_decrypt(&sealed, &kp, &cm).is_ok());
}

#[test]
fn tampered_blob_fails_for_the_rightful_owner() {
    let bob = WalletKeys::from_master(&MasterSeed::from_bip39(BOB_PHRASE, "bob"));
    let genesis = ShardSet::genesis();
    let addr = Address::generate(bob.detection(), 1, &genesis).unwrap();
    let mut rng = Rng(0x7A471);
    let plaintext = NotePlaintext::new(7, rng.bytes32(), rng.bytes32(), None).unwrap();
    let (cm, sealed) = seal_note(&plaintext, addr.delivery(), &rng.bytes32()).unwrap();

    let kp = bob.detection().delivery_keypair(1).unwrap();
    assert!(trial_decrypt(&sealed, &kp, &cm).is_ok());

    let mut tampered = sealed.clone();
    let n = tampered.sealed.len();
    tampered.sealed[n - 1] ^= 1; // tag byte
    assert!(matches!(
        trial_decrypt(&tampered, &kp, &cm),
        Err(NoteError::DecryptionFailed)
    ));
}

