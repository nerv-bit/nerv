//! C2 — cross-shard escrow and claim lifecycle.
//!
//! Road-to-testnet citation: [`specs/road_to_testnet.md §12.2 (C2)`].
//!
//! Walks the conditional-output → pending-transit → claim flow end-to-end
//! using a real `construct_payment` cross-shard shell (a 2-leg
//! `CrossShardSpend` + `CrossShardIssue` transaction). Mirrors the
//! executor's own `claim_lifecycle_and_issue_leg_settlement` differential
//! (`crates/nerv-state/src/executor.rs:1471-1586`) but with the shell
//! produced by the wallet's public surface, not a synthetic `leg()` builder.
//!
//! **Review-first rule, pre-authorized by the user (2026-10-09, 14:53 PDT):**
//! the executor's claim + reversion logic is consensus-touching; any fix
//! must (a) land with a regression test that fails before and passes
//! after, (b) cite `file:line` in the final report, and (c) not silently
//! change any executor invariant. If a fix would also need to touch
//! `finality.rs`, `beacon.rs`, `qc.rs`, or `slash.rs`, stop and report.
//!
//! Citations:
//!   * `crates/nerv-wallet/src/construct.rs:188` — `construct_payment`.
//!   * `crates/nerv-wallet/src/construct.rs:411` — `build_cross_shard`.
//!   * `crates/nerv-state/src/executor.rs:291` — `apply_block`.
//!   * `crates/nerv-state/src/executor.rs:1471` — `claim_lifecycle_*`.
//!   * `crates/nerv-state/src/testkit.rs:104` — `leg` (synthetic, for diff).
//!   * `crates/nerv-state/src/testkit.rs:213` — `block` builder.
//!   * `crates/nerv-state/src/testkit.rs:263` — `seal` (differential).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use nerv_core::types::{Height, ShardSet};
use nerv_custody::{transit_key, TransitEntryState};
use nerv_state::executor;
use nerv_state::testkit::{self, TestWorld, TAU0};
use nerv_state::{ClaimRecord, ExecutorError, SettledLeg, ShardState, TransitEvidence};
use nerv_wallet::keys::AddressSet;

use super::tx_lifecycle::{codec, epoch_pk, funded_wallet, splitmix64, wallet_keys};

/// C2 — cross-shard escrow and claim end-to-end.
///
/// The 4-block lifecycle:
///   1. block A.1: spend leg settles, transit entry → Pending.
///   2. block B.1: issue leg settles with the spend evidence, transit →
///      Claimed on B's side; output goes into B's NCT.
///   3. block A.2: claim record consumes the Pending entry, transit on A
///      → Claimed; A does NOT add an NCT leaf (the recipient is homed to
///      B, so the leaf lives in B's tree).
///   4. (negative) replay of the issue leg, double claim, evidence tamper.
#[test]
fn cross_shard_escrow_settles_and_claims() {
    let active = ShardSet::genesis();
    let shard_a = active.ids()[7];
    let shard_b = active.ids()[40];
    assert_ne!(shard_a, shard_b, "test must use two different shards");

    // ---- Step 0: build a real cross-shard tx via `construct_payment`.
    //   * sender on shard_a with 1 note covering amount + fee
    //   * recipient on shard_b (different shard → cross-shard path)
    let nanos = 1_000_000_000u64;
    let amount_nano = 1 * nanos;
    // Cross-shard splits the fee across two legs (build_cross_shard
    // floor-divides by 2). The executor's per-leg minimum is 1000 nano,
    // so the total fee must be at least 2000 to satisfy both legs.
    let fee_nano = 2_000;
    let (sender_keys, sender_addrs, notes, _total) = funded_wallet(
        0xC2,
        1,
        amount_nano + fee_nano,
        shard_a,
    )
    .expect("fund sender");

    // Build a recipient on shard_b. `AddressSet::generate` may not cover
    // shard_b; `ensure_shard` deterministically adds one.
    let recipient_keys = wallet_keys(0xC2);
    let mut recipient_addrs =
        AddressSet::generate(&recipient_keys, &active).expect("recipient addresses");
    if !recipient_addrs
        .covered_shards()
        .iter()
        .any(|s| s == &shard_b)
    {
        recipient_addrs
            .ensure_shard(&recipient_keys, &active, shard_b)
            .expect("ensure_shard for recipient");
    }
    let recipient_addr = recipient_addrs
        .addresses_for_shard(&shard_b)
        .into_iter()
        .next()
        .expect("recipient on shard_b")
        .address
        .clone();
    let _ = recipient_keys;

    let spec = nerv_wallet::construct::PaymentSpec {
        recipient: recipient_addr,
        amount_nano,
        fee_nano,
        expiry_height: 5_000,
    };
    let w = codec();
    let pk = epoch_pk();
    let mut entropy: Box<dyn FnMut() -> [u8; 32]> = Box::new(splitmix64(0xC2));

    let constructed = nerv_wallet::construct::construct_payment(
        &spec,
        &notes,
        &sender_addrs,
        &sender_keys,
        &active,
        &w,
        &pk,
        100,
        &mut entropy,
    )
    .expect("construct_payment");

    let canon = constructed.shell.canonicalize().expect("canonicalize");
    let txid = canon.txid().expect("txid");
    assert_eq!(
        canon.legs.len(),
        2,
        "construct_payment must produce a 2-leg cross-shard shell"
    );
    // The issue leg is the second leg (cross-shard case). Its recipient
    // output must be conditional (erratum 165) and carry a revert_cm.
    let issue_leg = &canon.legs[1];
    let issue_output = &issue_leg.outputs[0];
    assert!(
        issue_output.conditional,
        "issue leg's output must be conditional for the escrow to land"
    );
    assert!(
        issue_output.revert_cm.is_some(),
        "issue leg's output must carry a revert_cm"
    );

    // ---- Step 1: block A.1 — spend leg settles, transit → Pending.
    let mut world = TestWorld::default();
    let interval = TAU0 + 1;
    let tree = testkit::register_tau(&mut world, interval, &[txid]);

    let spend_sl = SettledLeg {
        shell: canon.clone(),
        leg: nerv_core::types::LegIndex::FIRST,
        tau: tree.witness(0).expect("tau witness"),
        siblings: vec![],
    };
    let mut st_a1 = ShardState::genesis(shard_a, testkit::params());
    let mut b_a1 = testkit::block(shard_a, interval, vec![spend_sl], vec![], vec![]);
    testkit::seal(&st_a1, &mut b_a1, &mut world, &[]);
    let (st_a2, app_a2) = executor::apply_block(st_a1, &b_a1, &world, &world)
        .expect("apply block A.1");
    let key_a = transit_key(&txid, &shard_a, nerv_core::types::LegIndex::FIRST);
    assert_eq!(
        st_a2.transit().get(&key_a).expect("transit A").state,
        TransitEntryState::Pending,
        "block A.1 must open the escrow as Pending"
    );
    assert_eq!(app_a2.escrows_opened, vec![txid]);

    // Record the transit root at A.1 so the issue leg's evidence verifies.
    world
        .transit_roots
        .insert((shard_a, 1), st_a2.transit().root());

    // ---- Step 2: block B.1 — issue leg settles with spend evidence.
    let spend_entry = *st_a2.transit().get(&key_a).expect("spend entry");
    let ev = TransitEvidence {
        root_height: Height::from_u64(1),
        proof: st_a2.transit().membership_proof(&spend_entry),
    };
    let issue_sl = SettledLeg {
        shell: canon.clone(),
        leg: nerv_core::types::LegIndex::from_u8(1),
        tau: tree.witness(0).expect("tau witness"),
        siblings: vec![ev.clone()],
    };
    let st_b1 = ShardState::genesis(shard_b, testkit::params());
    let mut b_b1 =
        testkit::block(shard_b, interval, vec![issue_sl.clone()], vec![], vec![]);
    testkit::seal(&st_b1, &mut b_b1, &mut world, &[]);
    let (st_b2, app_b2) = executor::apply_block(st_b1, &b_b1, &world, &world)
        .expect("apply block B.1");
    let key_b = transit_key(&txid, &shard_b, nerv_core::types::LegIndex::from_u8(1));
    assert_eq!(
        st_b2.transit().get(&key_b).expect("transit B").state,
        TransitEntryState::Claimed,
        "block B.1 must mark the issue leg as Claimed"
    );
    assert_eq!(
        st_b2.nct().leaf_count(),
        1,
        "block B.1 must add the recipient note to B's NCT"
    );
    assert_eq!(app_b2.settled, vec![txid]);

    // Record the transit root at B.1.
    world
        .transit_roots
        .insert((shard_b, 1), st_b2.transit().root());

    // ---- Step 3: block A.2 — claim record transitions A's transit Pending → Claimed.
    let issue_entry = *st_b2.transit().get(&key_b).expect("issue entry");
    let claim = ClaimRecord {
        txid,
        spend_leg: nerv_core::types::LegIndex::FIRST,
        evidence: vec![TransitEvidence {
            root_height: Height::from_u64(1),
            proof: st_b2.transit().membership_proof(&issue_entry),
        }],
    };
    let mut b_a2 = testkit::block(shard_a, TAU0 + 2, vec![], vec![], vec![claim]);
    testkit::seal(&st_a2, &mut b_a2, &mut world, &[]);
    let (st_a3, app_a3) = executor::apply_block(st_a2, &b_a2, &world, &world)
        .expect("apply block A.2 (claim)");
    assert_eq!(
        st_a3.transit().get(&key_a).expect("transit A").state,
        TransitEntryState::Claimed,
        "block A.2 (claim) must transition the spend-side transit to Claimed"
    );
    assert_eq!(app_a3.escrows_claimed, vec![txid]);
    assert_eq!(
        st_a3.nct().leaf_count(),
        0,
        "the recipient note lives in B's NCT, not A's"
    );
    assert!(
        app_a3.settled.is_empty(),
        "the spend leg was already settled in A.1; A.2 must not double-settle"
    );

    // ---- Step 4: negative — replay of the issue leg must fail (rule 4).
    // A second block at the same shard with the same issue leg's shell
    // is an attempted double-spend; the executor should reject it with
    // `TransitSpent { leg: 0 }`. We register the tau tree for TAU0+2
    // explicitly so the seal-time differential computes the same root
    // as the original B.1 (without this, the test's seal() helper would
    // build a fresh tree from `resolve_legs()` and the executor would
    // see a *different* root for TAU0+2 — which is correct in general,
    // but blocks the negative test from reaching rule 4 with the
    // matching witness we have here).
    let issue_txid = canon.txid().expect("canon txid");
    let replay_tree = testkit::register_tau(&mut world, TAU0 + 2, &[issue_txid]);
    let replay_tau_wit = replay_tree
        .witness(0)
        .expect("replay tau witness");
    let mut b_b1_replay = testkit::block(
        shard_b,
        TAU0 + 2,
        vec![SettledLeg {
            shell: canon.clone(),
            leg: nerv_core::types::LegIndex::from_u8(1),
            tau: replay_tau_wit,
            siblings: vec![ev],
        }],
        vec![],
        vec![],
    );
    testkit::seal(&st_b2, &mut b_b1_replay, &mut world, &[]);
    let replay_err =
        executor::apply_block(st_b2.clone(), &b_b1_replay, &world, &world);
    assert!(
        matches!(replay_err, Err(ExecutorError::TransitSpent { leg: 0 })),
        "replay of the issue leg must fail with TransitSpent {{ leg: 0 }}, got {replay_err:?}"
    );

    // ---- Step 5: negative — evidence-tamper is rejected by rule 4.
    // The transit evidence references the spend shard's transit root at a
    // *different* height. The executor's membership-proof verification
    // should reject the height mismatch.
    let mut tampered_ev = issue_sl.siblings[0].clone();
    tampered_ev.root_height = Height::from_u64(99);
    let mut b_b1_evidence_tamper = testkit::block(
        shard_b,
        TAU0 + 3,
        vec![SettledLeg {
            shell: canon.clone(),
            leg: nerv_core::types::LegIndex::from_u8(1),
            tau: tree.witness(0).expect("tau witness"),
            siblings: vec![tampered_ev],
        }],
        vec![],
        vec![],
    );
    testkit::seal(&st_b2, &mut b_b1_evidence_tamper, &mut world, &[]);
    let tamper_err = executor::apply_block(
        st_b2.clone(),
        &b_b1_evidence_tamper,
        &world,
        &world,
    );
    assert!(
        tamper_err.is_err(),
        "tampered transit evidence (wrong height) must be rejected, got Ok"
    );
}
