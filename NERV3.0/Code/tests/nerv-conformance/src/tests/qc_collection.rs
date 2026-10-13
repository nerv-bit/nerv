//! C5 — QC collection: quorum assembly and equivocation/duplicate rejection.
//!
//! Road-to-testnet citation: [`specs/road_to_testnet.md §12.5 (C5)`].
//!
//! Drives a 21-member shard committee (the spec's default shard size) and
//! asserts:
//!   * 15 distinct valid votes from members 0..14 produce a
//!     `QuorumCertificate` via `VoteCollector::assemble`.
//!   * 16th vote submitted under a DIFFERENT member's signature
//!     (equivocation attempt — signature doesn't match the
//!     submitting-member's key) is rejected with
//!     `QcError::InvalidVote`.
//!   * Duplicate vote from the same member 0 a second time is rejected
//!     idempotently (`Ok(false)`); the count stays at 15.
//!   * 14 votes is below the 15-of-21 quorum floor, so
//!     `assemble(15)` returns `QcError::QuorumNotMet { have: 14, need: 15 }`
//!     (this is the conformance analog of the spec's "14 votes +
//!     timeout" — `VoteCollector` does not have a dedicated `Timeout`
//!     variant; the under-quorum state is the surface it surfaces).
//!
//! Citations:
//!   * `crates/nerv-crypto/src/sigaggr.rs:78` — `VoteCollector`.
//!   * `crates/nerv-crypto/src/sigaggr.rs:108` — `VoteCollector::add`.
//!   * `crates/nerv-crypto/src/sigaggr.rs:130` — `VoteCollector::assemble`.
//!   * `crates/nerv-crypto/src/sigaggr.rs:150` — `validate_qc`.
//!   * `crates/nerv-crypto/src/sigaggr.rs:196` — `vote_bytes`.
//!   * `crates/nerv-crypto/src/sigaggr.rs:17`  — `QcError` variants.
//!   * `crates/nerv-crypto/src/mldsa.rs:40`   — `SigningKey::from_seed`.
//!   * `crates/nerv-consensus/src/attestation.rs:IntervalAttestation::build` —
//!     the production attestation builder; not directly exercised here
//!     because C5's contract is the *aggregation gate*, not the per-block
//!     attestation surface.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use nerv_core::hash::Hash256;
use nerv_core::types::Epoch;
use nerv_crypto::mldsa::{Signature, SigningKey, VerifyingKey};
use nerv_crypto::sigaggr::{QcError, QuorumCertificate, VoteCollector};

/// Build a 21-member committee (the shard size from params.toml) where
/// each member `i` has a deterministic ML-DSA keypair derived from
/// `seed.wrapping_add(i)`. Returns `(signers, verifiers)`.
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
fn quorum_assembly_and_equivocation_rejection() {
    let epoch = Epoch::from_u64(7);
    let subject = Hash256::from_bytes([0xC5u8; 32]);
    let (sks, vks) = committee(0xC5, 21);
    assert_eq!(vks.len(), 21);

    let mut collector = VoteCollector::new(epoch, subject);
    let msg = nerv_crypto::sigaggr::vote_bytes(epoch, &subject);

    // ---- (a) 15 valid votes from members 0..14 → assemble(15) succeeds.
    for (member, sk) in sks.iter().enumerate().take(15) {
        let sig: Signature = sk.sign(&msg).expect("sign");
        let added = collector.add(member, sig, &vks).expect("add");
        assert!(added, "first vote from member {member} must be added");
    }
    assert_eq!(collector.count(), 15);
    let qc: QuorumCertificate = collector.assemble(15).expect("assemble 15/21");
    assert_eq!(qc.epoch, epoch);
    assert_eq!(qc.subject, subject);
    assert_eq!(qc.signatures.len(), 15);
    // Every signature verifies under `validate_qc`.
    let qc_hash = nerv_crypto::sigaggr::validate_qc(&qc, &vks, 15).expect("validate_qc");
    assert_ne!(qc_hash, Hash256::from_bytes([0u8; 32]));

    // ---- (b) Equivocation: member 15 submits a signature that was
    //         actually produced by member 0. The signature does not
    //         verify under member 15's key → QcError::InvalidVote.
    let stolen_sig: Signature = sks[0].sign(&msg).expect("sign");
    let err = collector.add(15, stolen_sig, &vks);
    assert!(
        matches!(err, Err(QcError::InvalidVote { member: 15 })),
        "equivocation (stolen signature) must surface QcError::InvalidVote, got {err:?}"
    );
    assert_eq!(collector.count(), 15, "invalid vote must not be inserted");

    // ---- (c) Duplicate: member 0 votes a SECOND time. The collector
    //         rejects idempotently (Ok(false)); count stays at 15.
    let dup_sig: Signature = sks[0].sign(&msg).expect("sign");
    let added = collector.add(0, dup_sig, &vks).expect("duplicate add is not a QcError");
    assert!(!added, "duplicate from member 0 must be rejected (Ok(false))");
    assert_eq!(collector.count(), 15, "duplicate must not change count");

    // ---- (d) Under-quorum: 14 votes is below the 15 floor.
    let mut low_collector = VoteCollector::new(epoch, subject);
    for (member, sk) in sks.iter().enumerate().take(14) {
        let sig: Signature = sk.sign(&msg).expect("sign");
        low_collector.add(member, sig, &vks).expect("add");
    }
    assert_eq!(low_collector.count(), 14);
    let err = low_collector.assemble(15);
    assert!(
        matches!(err, Err(QcError::QuorumNotMet { have: 14, need: 15 })),
        "under-quorum assemble must surface QuorumNotMet, got {err:?}"
    );
    // Note: the spec's "14 votes + a timeout event" — VoteCollector has
    // no dedicated `Timeout` variant. The under-quorum state is what the
    // aggregator surfaces in that case; the higher-level timeout path
    // (the consensus state machine) decides whether to escalate to a
    // timeout certificate. C5's conformance contract pins the collector
    // surface only.
}
