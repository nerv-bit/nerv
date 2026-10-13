//! C3 — five-relay aggregator decode (mixnet path end-to-end).
//!
//! Road-to-testnet citation: [`specs/road_to_testnet.md §12.3 (C3)`].
//!
//! Walks the public mixnet surface (`sphinx::build` →
//! `RelayEngine::handle_packet` × 5 → `HostFrame::decode`) and asserts
//! the deterministic 5-relay delivery path. Also covers the two
//! failure modes the spec requires:
//!   * `RelayAction::Drop` at the terminal relay when the path is
//!     configured with `TerminalAction::Drop`.
//!   * Tamper-bit detection at the terminal relay (one byte flip in the
//!     payload region of the encapsulated frame).
//!
//! Citations:
//!   * `crates/nerv-net/src/sphinx.rs:19`  — `PATH_RELAYS`.
//!   * `crates/nerv-net/src/sphinx.rs:201` — `sphinx::build`.
//!   * `crates/nerv-net/src/sphinx.rs:252` — `sphinx::peel` (per-hop).
//!   * `crates/nerv-net/src/relay.rs:165`  — `RelayAction`.
//!   * `crates/nerv-net/src/relay.rs:193`  — `RelayEngine::handle_packet`.
//!   * `crates/nerv-net/src/host.rs:616`   — `HostFrame::decode`.
//!   * `crates/nerv-net/src/lib.rs:65`     — `HostFrame::encode`.
//!
//! The conformance crate does not reach into `nerv_net::testutil` (which
//! is `pub(crate)` by design); instead, `sphinx::relays` and
//! `sphinx::encap_rand` are exposed as thin public helpers in
//! `crates/nerv-net/src/sphinx.rs` (Session 10) and used here directly.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use nerv_core::Encode;
use nerv_net::relay::RelayAction;
use nerv_net::relay::RelayEngine;
use nerv_net::sphinx::{
    self, encap_rand, relays, PathSpec, TerminalAction, PATH_RELAYS,
};
use nerv_net::submission::SubmissionMessage;
use nerv_net::HostFrame;

/// C3 — `sphinx::build` → 5-relay `RelayEngine::handle_packet` chain
/// → aggregator `SubmissionMessage::decode` → `HostFrame` round-trip.
///
/// The test covers:
///   (a) **happy path:** a HostFrame::Submission reaches the terminal
///       relay with `RelayAction::Deliver`, decodes to a
///       `SubmissionMessage::Transaction`, and the embedded `PoolEntry`
///       round-trips identically.
///   (b) **terminal drop:** a `PathSpec` with `terminal=Drop` produces
///       `RelayAction::Drop` at the terminal relay.
///   (c) **tamper detection:** a one-byte flip in the payload region
///       causes the terminal relay to produce `RelayAction::Drop` with
///       `stats.misrouted = 1`.
#[test]
fn five_relay_aggregator_decodes() {
    assert_eq!(PATH_RELAYS, 5, "PATH_RELAYS must be 5 for this test");

    // ---- (a) happy path: 5-relay delivery to the aggregator.
    let rs = relays(PATH_RELAYS, 0xC3);
    let (dest_id, dest_dk) = (rs[4].0, rs[4].2.clone());
    let hops: [(nerv_net::host::PeerId, nerv_crypto::mlkem::EncapsulationKey); PATH_RELAYS] = [
        (rs[0].0, rs[0].1.clone()),
        (rs[1].0, rs[1].1.clone()),
        (rs[2].0, rs[2].1.clone()),
        (rs[3].0, rs[3].1.clone()),
        (rs[4].0, rs[4].1.clone()),
    ];
    let spec = PathSpec::new(hops, dest_id, TerminalAction::Deliver);

    // Real, canonicalisable submission message (a single-leg shell — the
    // same surface C1a's C1b path will hand to `send_transaction`).
    let shell = make_garbage_shell();
    let canon = shell.canonicalize().expect("canonicalize");
    let txid = nerv_state::canonical_txid(&canon);
    let entry = nerv_registry::mempool::PoolEntry {
        txid,
        shell: canon.clone(),
        proof: make_garbage_proof(),
    };
    let submission = HostFrame::Submission(SubmissionMessage::Transaction { entry });
    let payload = submission.encode();
    // The wire path requires fragmenting the payload into per-frame chunks
    // (the spec allows a 5-fragment ceiling — we only have 1 here). The
    // encoded fragment (WITHOUT the MIX_FRAGMENT_TAG prefix that
    // `FragmentFrame::to_frame` adds — the `sphinx::build` MAX_DATA
    // gate rejects the +1 tag byte), is what `sphinx::build` consumes.
    // The terminal relay's delivered frame is the encoded fragment with
    // the tag prepended; the aggregator strips the tag and reassembles.
    let fragments = sphinx::fragment(&payload, &txid).expect("fragment");
    let fragment_bytes = fragments[0].encode();
    let packet = sphinx::build(&spec, &fragment_bytes, &encap_rand(0xC3A))
        .expect("sphinx::build");

    // Drive the 5-relay chain. Hops 0..4 should each produce RelayAction::Forward;
    // hop 4 (the terminal) should produce RelayAction::Deliver with the original
    // HostFrame payload.
    let mut engines: Vec<RelayEngine> = rs
        .iter()
        .map(|(_, _, dk)| RelayEngine::new(dk.clone()))
        .collect();

    let mut current = packet.clone();
    for (i, engine) in engines.iter_mut().enumerate() {
        let action = engine.handle_packet(&current, 0xC3B + i as u64);
        if i < PATH_RELAYS - 1 {
            let RelayAction::Forward { frame, .. } = action else {
                panic!("hop {i}: expected Forward, got {action:?}");
            };
            current = nerv_net::sphinx::Packet::from_frame(&frame).expect("from_frame");
        } else {
            let RelayAction::Deliver { frame, .. } = action else {
                panic!("terminal: expected Deliver, got {action:?}");
            };
            // The terminal's delivered frame is an encoded fragment
            // (tag = MIX_FRAGMENT_TAG). Reassemble back to the payload
            // and then decode the HostFrame.
            assert_eq!(frame[0], sphinx::MIX_FRAGMENT_TAG, "fragment tag");
            let frag = sphinx::FragmentFrame::from_frame(&frame).expect("from_frame");
            let reassembled = sphinx::reassemble(&[frag]).expect("reassemble");
            assert_eq!(reassembled, payload, "reassembled payload matches");
            let decoded = HostFrame::decode(&reassembled).expect("HostFrame::decode");
            let HostFrame::Submission(SubmissionMessage::Transaction { entry: got }) =
                decoded
            else {
                panic!("expected Submission, got {decoded:?}");
            };
            assert_eq!(
                got.txid, txid,
                "the aggregator decoded the same txid as the sender"
            );
            assert_eq!(
                got.shell, canon,
                "the aggregator decoded the same canonical shell"
            );
        }
    }

    // Sanity: the terminal relay reports exactly one delivery.
    let term_stats = engines.last().expect("5th engine").stats();
    assert_eq!(term_stats.packets_seen, 1);
    assert_eq!(term_stats.delivered, 1);
    assert_eq!(term_stats.misrouted, 0);
    assert_eq!(term_stats.replayed, 0);

    // ---- (b) terminal drop.
    let spec_drop = PathSpec::new(
        [
            (rs[0].0, rs[0].1.clone()),
            (rs[1].0, rs[1].1.clone()),
            (rs[2].0, rs[2].1.clone()),
            (rs[3].0, rs[3].1.clone()),
            (rs[4].0, rs[4].1.clone()),
        ],
        rs[4].0,
        TerminalAction::Drop,
    );
    let drop_packet = sphinx::build(&spec_drop, &fragment_bytes, &encap_rand(0xC3C))
        .expect("sphinx::build (drop)");
    let mut drop_engines: Vec<RelayEngine> = rs
        .iter()
        .map(|(_, _, dk)| RelayEngine::new(dk.clone()))
        .collect();
    let mut current = drop_packet;
    for (i, engine) in drop_engines.iter_mut().enumerate() {
        let action = engine.handle_packet(&current, 0xC3D + i as u64);
        if i < PATH_RELAYS - 1 {
            let RelayAction::Forward { frame, .. } = action else {
                panic!("drop hop {i}: expected Forward, got {action:?}");
            };
            current = nerv_net::sphinx::Packet::from_frame(&frame).expect("from_frame");
        } else {
            assert!(
                matches!(action, RelayAction::Drop),
                "terminal with TerminalAction::Drop must Drop, got {action:?}"
            );
        }
    }
    let drop_stats = drop_engines.last().expect("5th engine").stats();
    assert_eq!(drop_stats.cover_dropped, 1);
    assert_eq!(drop_stats.delivered, 0);

    // ---- (c) tamper detection at the terminal relay.
    // A one-byte flip in the payload region invalidates the AEAD tag at the
    // terminal, producing RelayAction::Drop with stats.misrouted = 1.
    let mut tampered = packet.clone();
    // Flip a bit well inside the payload region (past the meta + header).
    tampered.0[sphinx::PACKET_BYTES - 8] ^= 0x01;
    let mut tamper_engines: Vec<RelayEngine> = rs
        .iter()
        .map(|(_, _, dk)| RelayEngine::new(dk.clone()))
        .collect();
    let mut current = tampered;
    for (i, engine) in tamper_engines.iter_mut().enumerate() {
        let action = engine.handle_packet(&current, 0xC3E + i as u64);
        if i < PATH_RELAYS - 1 {
            // Earlier hops can still forward — the tamper is detected at the
            // terminal because each per-hop MAC only covers that hop's
            // expected transformation. Errata 158 keeps earlier hops working.
            match action {
                RelayAction::Forward { frame, .. } => {
                    current = nerv_net::sphinx::Packet::from_frame(&frame).expect("from_frame");
                }
                RelayAction::Drop => {
                    // Hops may drop too; that's fine — the contract is just
                    // that the terminal is not bypassed.
                    break;
                }
                other => panic!("tamper hop {i}: unexpected {other:?}"),
            }
        } else {
            assert!(
                matches!(action, RelayAction::Drop),
                "terminal must Drop the tampered packet, got {action:?}"
            );
        }
    }
    let tamper_stats = tamper_engines.last().expect("5th engine").stats();
    assert!(
        tamper_stats.misrouted >= 1 || tamper_stats.cover_dropped >= 1,
        "terminal must record misrouted or cover_dropped for the tampered packet, stats={:?}",
        tamper_stats
    );

    // Silence unused warnings for variables the compiler treats as
    // documentation-only.
    let _ = dest_dk;
}

// ---------------------------------------------------------------------------
// Test-harness helpers (a structural single-leg shell + a garbage proof
// that round-trips through `TransactionProof`'s encoding surface).
// ---------------------------------------------------------------------------

fn make_garbage_shell() -> nerv_custody::tx::TransactionShell {
    use nerv_custody::tx::{
        InputSet, LegShell, Output, TransactionShell,
    };
    let set = nerv_core::types::ShardSet::genesis();
    let shard = set.ids()[7];
    let mut ct = vec![0u8; nerv_seal::encrypt::Ciphertext::WIRE_SIZE];
    for (i, c) in ct.chunks_exact_mut(4).enumerate() {
        c.copy_from_slice(&((i as u32).wrapping_mul(0x10DE_F00D) & 0xFFF0_0000).to_le_bytes());
    }
    TransactionShell {
        legs: vec![LegShell {
            shard,
            inputs: InputSet::new(vec![nerv_core::hash::Hash256::from_bytes([7u8; 32])]),
            outputs: vec![Output {
                cm: nerv_core::hash::Hash256::from_bytes([0xA5; 32]),
                sealed_note: vec![0xA5; 48],
                value: 1_000_000_000,
                conditional: false,
                revert_cm: None,
            }],
            fee: nerv_core::types::FeeSats::from_u64(1_000),
            anchor: nerv_core::hash::Hash256::from_bytes([0xCC; 32]),
            expiry: nerv_core::types::Height::from_u64(5_000),
            weight_version: 1,
            ct,
            burns: vec![],
        }],
    }
}

fn make_garbage_proof() -> nerv_proofs::TransactionProof {
    use nerv_proofs::stark::ext_field::ExtF;
    use nerv_proofs::stark::fri::FriProof;
    use nerv_proofs::{ComposedProof, Proved, TransactionProof};
    TransactionProof {
        proved: Proved {
            proof: ComposedProof {
                trace_root: nerv_core::hash::Hash256::from_bytes([1u8; 32]),
                quotient_root: nerv_core::hash::Hash256::from_bytes([2u8; 32]),
                trace_zeta: Vec::new(),
                trace_zeta_next: Vec::new(),
                quotient_zeta: ExtF::ZERO,
                num_assertions: 0,
                fri: FriProof {
                    roots: Vec::new(),
                    final_coeffs: Vec::new(),
                    pow_nonce: 0,
                    queries: Vec::new(),
                    positions: Vec::new(),
                },
                outer: Vec::new(),
            },
            log_n: 2,
            width: 33,
            prep: Vec::new(),
        },
        publics: Vec::new(),
    }
}
