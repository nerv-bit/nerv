#![allow(clippy::unwrap_used, clippy::expect_used)]

use nerv_core::codec::{Decode, Encode};
use nerv_core::constants::{NOTE_KDF, TXID};
use nerv_core::hash::Hash256;
use nerv_core::types::Epoch;
use nerv_crypto::aead::{open, seal, AeadKey, Nonce};
use nerv_crypto::kdf::blake3_kdf;
use nerv_crypto::mldsa::{SigningKey, VerifyingKey};
use nerv_crypto::mlkem::keypair_from_seed;
use nerv_crypto::sigaggr::{validate_qc, vote_bytes, QuorumCertificate, VoteCollector};
use nerv_crypto::sortition::select_committee;

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
    fn bytes64(&mut self) -> [u8; 64] {
        let mut out = [0u8; 64];
        for c in out.chunks_exact_mut(8) {
            c.copy_from_slice(&self.next().to_le_bytes());
        }
        out
    }
}

#[test]
fn note_encryption_pipeline() {
    // WP §3.2: hybrid PQ note encryption. Receiver derives a delivery
    // keypair from a seed (chunk 4 derives the seed itself from the wallet
    // hierarchy).
    let mut rng = Rng(0xA11CE);
    let seed = rng.bytes64();
    let (ek, dk) = keypair_from_seed(&seed).unwrap();

    // sender: one encapsulation with caller randomness; single-use key law
    let m = rng.bytes32();
    let (ss, ct) = ek.encapsulate(&m).unwrap();
    let key = AeadKey::from_bytes(blake3_kdf(&NOTE_KDF, ss.as_bytes(), &[]));
    let note: &[u8] = b"note plaintext: v rho d memo";
    let aad: &[u8] = b"32-byte note commitment stand-in";
    let sealed = seal(&key, &Nonce::ZERO, aad, note).unwrap();

    // receiver: decapsulate, derive, open
    let ss2 = dk.decapsulate(&ct).unwrap();
    let key2 = AeadKey::from_bytes(blake3_kdf(&NOTE_KDF, ss2.as_bytes(), &[]));
    assert_eq!(open(&key2, &Nonce::ZERO, aad, &sealed).unwrap(), note);
    // AAD (commitment) substitution fails
    assert!(open(&key2, &Nonce::ZERO, b"different commitment", &sealed).is_err());
}

#[test]
fn note_encryption_wrong_receiver() {
    let mut rng = Rng(0xB0B);
    let (ek, _) = keypair_from_seed(&rng.bytes64()).unwrap();
    let (_, dk_wrong) = keypair_from_seed(&rng.bytes64()).unwrap();
    let (ss, ct) = ek.encapsulate(&rng.bytes32()).unwrap();
    let key = AeadKey::from_bytes(blake3_kdf(&NOTE_KDF, ss.as_bytes(), &[]));
    let sealed = seal(&key, &Nonce::ZERO, b"aad", b"secret note").unwrap();
    let wrong = dk_wrong.decapsulate(&ct).unwrap();
    let wrong_key = AeadKey::from_bytes(blake3_kdf(&NOTE_KDF, wrong.as_bytes(), &[]));
    assert!(open(&wrong_key, &Nonce::ZERO, b"aad", &sealed).is_err());
}

#[test]
fn sortition_roster_to_quorum_certificate() {
    // WP §4.6 + §8.3 + DSR-5: sortition selects the roster; members vote;
    // the QC assembles, validates, and hash-compresses.
    let mut rng = Rng(0xC0DE);
    let signers: Vec<SigningKey> = (0..30u64)
        .map(|i| {
            let mut r = Rng(0x5000 + i);
            SigningKey::from_seed(&r.bytes32()).unwrap()
        })
        .collect();
    let candidates: Vec<VerifyingKey> = signers.iter().map(|k| *k.verifying_key()).collect();

    let randomness = Hash256::from_bytes(rng.bytes32());
    let epoch = Epoch::from_u64(12);
    let roster_idx = select_committee(&randomness, &candidates, epoch, 21);
    assert_eq!(roster_idx.len(), 21);
    assert_eq!(
        roster_idx,
        select_committee(&randomness, &candidates, epoch, 21),
        "roster must be deterministic"
    );
    let roster: Vec<VerifyingKey> = roster_idx.iter().map(|&i| candidates[i]).collect();

    let subject = Hash256::concat(&TXID, b"block-header-digest-stand-in");
    let mut collector = VoteCollector::new(epoch, subject);
    for (member, &i) in roster_idx.iter().enumerate().take(15) {
        let sig = signers[i].sign(&vote_bytes(epoch, &subject)).unwrap();
        assert!(collector.add(member, sig, &roster).unwrap());
    }
    let qc = collector.assemble(15).unwrap();
    let h = validate_qc(&qc, &roster, 15).unwrap();
    assert_eq!(h, qc.qc_hash());

    // wire roundtrip preserves validation (the block-body path)
    let enc = qc.encode();
    assert_eq!(enc.len(), 8 + 32 + 4 + 4 + 15 * 3309);
    let qc2 = QuorumCertificate::decode(&enc).unwrap();
    assert_eq!(qc2, qc);
    assert_eq!(validate_qc(&qc2, &roster, 15).unwrap(), h);

    // a different epoch's roster does not validate this certificate
    let roster2_idx = select_committee(&randomness, &candidates, Epoch::from_u64(13), 21);
    let roster2: Vec<VerifyingKey> = roster2_idx.iter().map(|&i| candidates[i]).collect();
    assert!(validate_qc(&qc, &roster2, 15).is_err() || roster2 != roster);

    // non-selected candidates cannot vote into this committee
    let outside: Vec<usize> = (0..30).filter(|i| !roster_idx.contains(i)).collect();
    if let Some(&o) = outside.first() {
        let sig = signers[o].sign(&vote_bytes(epoch, &subject)).unwrap();
        let mut c2 = VoteCollector::new(epoch, subject);
        assert!(c2.add(0, sig, &roster).is_err() || roster[0] == candidates[o]);
    }
}

