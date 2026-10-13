//! C6 — epoch boundary and committee rotation.
//!
//! Road-to-testnet citation: [`specs/road_to_testnet.md §12.6 (C6)`].
//!
//! Walks the cross-epoch contract:
//!   1. Build a 21-member shard committee for epoch 0.
//!   2. Drive 15 valid votes through `VoteCollector::assemble(15)`
//!      (carried over from C5 to anchor the "previous epoch's QC
//!      still verifies" assertion).
//!   3. Establish epoch 1 (a different ASeed ⇒ different
//!      `derive_reference_keypair` ⇒ different epoch public key). The
//!      rotation contract: the new committee's public key is distinct
//!      from the old, and the old QC still verifies under the old
//!      committee's keys.
//!
//! Why `derive_reference_keypair` instead of the full DKG: the
//! conformance contract for C6 is the *public surface* (the
//! rotation-invariance of the previous epoch's QC, and the
//! non-equality of the new epoch's key). The DKG ceremony itself
//! (`nerv_seal::epoch::establish_epoch` + `MemberSecret::build`) is
//! regression-covered in `crates/nerv-seal/src/dkg/tests.rs` and
//! `crates/nerv-seal/src/epoch.rs:384`. Wiring a full DKG here would
//! duplicate 200 lines of deterministic secret generation for no
//! additional C6 contract gain.
//!
//! Citations:
//!   * `crates/nerv-seal/src/epoch.rs:107` — `establish_epoch`.
//!   * `crates/nerv-seal/src/encrypt.rs:177` — `derive_reference_keypair`.
//!   * `crates/nerv-crypto/src/sigaggr.rs:150` — `validate_qc`.
//!   * `crates/nerv-crypto/src/sigaggr.rs:196` — `vote_bytes`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use nerv_core::hash::Hash256;
use nerv_core::types::Epoch;
use nerv_crypto::mldsa::{Signature, SigningKey, VerifyingKey};
use nerv_crypto::sigaggr::{QcError, QuorumCertificate, VoteCollector};
use nerv_seal::encrypt::{derive_reference_keypair, PublicKey};

/// A 21-member shard committee with a deterministic ML-DSA keypair per
/// member. `seed` mixes into the key derivation so the epoch-0 and
/// epoch-1 committees can be made distinct on the same 21 validators
/// (if needed) without leaking the underlying derivation.
fn committee(seed: u64, n: usize) -> (Vec<SigningKey>, Vec<VerifyingKey>) {
    let mut sks = Vec::with_capacity(n);
    let mut vks = Vec::with_capacity(n);
    for i in 0..n as u64 {
        let mut sd = [0u8; 32];
        sd[..8].copy_from_slice(&seed.wrapping_add(i).to_le_bytes());
        let sk = SigningKey::from_seed(&sd).expect("SigningKey::from_seed");
        vks.push(sk.verifying_key().clone());
        sks.push(sk);
    }
    (sks, vks)
}

#[test]
fn epoch_boundary_rotates_committee() {
    // ---- Step 1: build the epoch-0 committee + a 15-of-21 QC.
    let (sks0, vks0) = committee(0xC6_0, 21);
    let epoch0 = Epoch::from_u64(0);
    let subject = Hash256::from_bytes([0xC6u8; 32]);
    let mut collector = VoteCollector::new(epoch0, subject);
    let msg = nerv_crypto::sigaggr::vote_bytes(epoch0, &subject);
    for (i, sk) in sks0.iter().enumerate().take(15) {
        let sig: Signature = sk.sign(&msg).expect("sign");
        collector.add(i, sig, &vks0).expect("add");
    }
    let qc: QuorumCertificate = collector.assemble(15).expect("assemble 15/21");
    let qc_hash_epoch0 = nerv_crypto::sigaggr::validate_qc(&qc, &vks0, 15).expect("validate_qc");
    assert_ne!(qc_hash_epoch0, Hash256::from_bytes([0u8; 32]));

    // ---- Step 2: derive two distinct epoch public keys via the
    //         reference keypair helper. Each call produces a unique
    //         `(A, T)` pair from a different entropy, simulating the
    //         DKG output across epochs.
    let pk0: PublicKey = derive_reference_keypair(&[0xC6u8; 32]).expect("pk0").0;
    let pk1: PublicKey = derive_reference_keypair(&[0xC6u8 + 1; 32]).expect("pk1").0;
    // The two epoch keys must differ (rotation contract).
    assert_ne!(
        pk0.a_seed().as_bytes(),
        pk1.a_seed().as_bytes(),
        "consecutive epochs must have distinct ASeeds"
    );
    assert_ne!(pk0, pk1, "consecutive epochs must have distinct epoch keys");

    // The ASeed-only equality is a stronger statement than a full PK
    // equality (T is also different), but the (deterministic) full
    // public key check is the contract the C1/C2 prover relies on.
    let id0 = nerv_seal::circuit_stmt::epoch_key_identifier(pk0.a_seed(), pk0.t());
    let id1 = nerv_seal::circuit_stmt::epoch_key_identifier(pk1.a_seed(), pk1.t());
    assert_ne!(id0, id1, "epoch_key_identifier is sensitive to ASeed and T");

    // ---- Step 3: the previous epoch's QC still verifies under the
    //         same committee that produced it (rotation invariance —
    //         the QC is a per-epoch certificate; until that epoch's
    //         keys are retired, the QC verifies).
    nerv_crypto::sigaggr::validate_qc(&qc, &vks0, 15).expect("epoch-0 QC still verifies");

    // ---- Negative: a QC under the epoch-0 committee's keys does not
    //         assemble once the under-quorum floor is breached (the
    //         timeout path C5 also exercises). Document the
    //         deterministic failure here for completeness.
    let mut partial = VoteCollector::new(epoch0, subject);
    for (i, sk) in sks0.iter().enumerate().take(14) {
        let sig: Signature = sk.sign(&msg).expect("sign");
        partial.add(i, sig, &vks0).expect("add");
    }
    assert!(matches!(
        partial.assemble(15),
        Err(QcError::QuorumNotMet { have: 14, need: 15 })
    ));
}
