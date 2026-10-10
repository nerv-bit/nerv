//! Test scaffolding for the executor (`nerv-state::executor::apply_block`).
//!
//! This module exposes the deterministic test world used by the executor's
//! own unit tests so that integration suites, the cross-crate smoke test in
//! `nerv-wallet`, and any downstream harness can drive `apply_block` from
//! outside the crate without re-implementing the helpers.
//!
//! # Build gates
//!
//! The module is compiled when **either**:
//!
//! * `cfg(test)` is active — every crate-internal test gets the helpers, or
//! * the `testkit` Cargo feature is enabled — external callers (the
//!   `smoke_cross_crate` example, future integration tests in
//!   `tests/cross_crate.rs`) opt in.
//!
//! # What lives here
//!
//! | Item             | Purpose                                          |
//! |------------------|--------------------------------------------------|
//! | `params()`       | The single byte-string every test uses as `params_root`. |
//! | `TAU0`           | The genesis-interval offset for tau trees.       |
//! | `Rng`            | Public alias for `crate::testutil::SplitMix64`.  |
//! | `h`              | A random `Hash256` from an `Rng`.                |
//! | `dummy_ct`       | A random ciphertext of the canonical wire size.  |
//! | `leg`            | A `LegShell` with random commitments.            |
//! | `payout`         | A cached, kappa-valid producer-payout Address.   |
//! | `shared_qc`      | A cached, valid 15-of-21 ML-DSA QuorumCertificate.|
//! | `TestWorld`      | `BeaconView + ChainSource` over `BTreeMap`s.     |
//! | `block`          | A `ShardBlock` with the right header defaults.   |
//! | `register_tau`   | Register a txid-set's tau tree for an interval.  |
//! | `seal`           | Independently compute the post-effects header.   |
//! | `advance`        | Advance the chain by N empty blocks.             |
//! | `genesis`        | Build a `ShardState::genesis`.                   |
//! | `empty_anchor`   | The empty NoteCommitmentTree root.               |
//! | `one_leg_block`  | Convenience: one-leg block + its txid + first nf.|
//!
//! # Production guarantees
//!
//! This is **not** a mock. Every helper that emits a cryptographic object
//! (`payout`, `shared_qc`) uses the real `nerv-crypto` and `nerv-custody`
//! primitives; `Address::new` enforces the kappa-tag binding, and
//! `VoteCollector::assemble` enforces the 15-of-21 quorum threshold. The
//! "test" in "testkit" means "deterministic test scaffolding", not "fake".

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::OnceLock;

use nerv_core::field::Goldilocks;
use nerv_core::hash::Hash256;
use nerv_core::types::{
    Epoch, FeeSats, Height, Interval, LegIndex, ShardId, ShardSet, TxId, kappa,
};
use nerv_crypto::mldsa::{SigningKey, VerifyingKey};
use nerv_crypto::mlkem::EncapsulationKey;
use nerv_crypto::sigaggr::{QuorumCertificate, VoteCollector, vote_bytes};
use nerv_custody::tx::{InputSet, LegShell, Output, TransactionShell};
use nerv_custody::{transit_key, Address, NoteCommitmentTree, TransitEntry, TransitEntryState};
use nerv_seal::encrypt::Ciphertext;

use crate::block::{
    ct_sum, ClaimRecord, ResolvedLeg, ReversionRecord, SettledLeg, ShardBlock,
};
use crate::executor::{apply_block, BeaconView, ChainSource};
use crate::header::{RegistryRef, ShardHeader};
use crate::ttau::TauTree;
use crate::ShardState;

/// The single byte-string every test uses as `params_root` so block headers
/// are byte-stable across runs. Returned by the `params()` function (not a
/// `const`) because `Hash256::from_bytes` is not a `const fn`.
pub fn params() -> Hash256 {
    Hash256::from_bytes([0x9A; 32])
}

/// The genesis-interval offset for tau trees: every shard starts at
/// `TAU0 + height` for its registry interval, matching the convention
/// used throughout the executor's unit tests.
pub const TAU0: u64 = 86_400;

/// Public alias for the deterministic `SplitMix64` PRNG used by every
/// helper. Re-exported so external callers don't have to reach into the
/// crate-private `testutil` module.
pub type Rng = crate::testutil::SplitMix64;

/// A random 32-byte hash drawn from the PRNG.
pub fn h(rng: &mut Rng) -> Hash256 {
    Hash256::from_bytes(rng.bytes32())
}

/// A random ciphertext of the canonical wire size (`Ciphertext::WIRE_SIZE`).
pub fn dummy_ct(rng: &mut Rng) -> Vec<u8> {
    let mut out = vec![0u8; Ciphertext::WIRE_SIZE];
    for c in out.chunks_exact_mut(4) {
        c.copy_from_slice(&(rng.next_u32() & 0xFFF0_0000).to_le_bytes());
    }
    out
}

/// Build a `LegShell` with random commitments and a deterministic ct body.
#[allow(clippy::too_many_arguments)]
pub fn leg(
    rng: &mut Rng,
    shard: ShardId,
    inputs: Vec<Hash256>,
    outputs: Vec<(u64, bool)>,
    fee: u64,
    expiry: u64,
    anchor: Hash256,
) -> LegShell {
    let outs = outputs
        .into_iter()
        .map(|(v, cond)| Output {
            cm: h(rng),
            sealed_note: vec![0xA5; 48],
            value: v,
            conditional: cond,
            revert_cm: cond.then(|| h(rng)),
        })
        .collect();
    LegShell {
        shard,
        inputs: InputSet::new(inputs),
        outputs: outs,
        fee: FeeSats::from_u64(fee),
        anchor,
        expiry: Height::from_u64(expiry),
        weight_version: 1,
        ct: dummy_ct(rng),
        burns: vec![],
    }
}

/// A cached, kappa-valid producer-payout Address. Computed once with seed
/// `0xFA0` so successive calls share the same Address (which means tests
/// can assert on header bytes without re-deriving them).
pub fn payout() -> Address {
    static PAYOUT: OnceLock<Address> = OnceLock::new();
    PAYOUT
        .get_or_init(|| {
            let mut rng = Rng::new(0xFA0);
            let set = ShardSet::genesis();
            let mut d = [0u8; 1184];
            for c in d.chunks_exact_mut(8) {
                c.copy_from_slice(&rng.next_u64().to_le_bytes());
            }
            let ek = EncapsulationKey::from_bytes(d);
            let tag = set.home_kappa(&kappa(ek.as_bytes())).unwrap();
            Address::new(ek, tag, rng.bytes32()).unwrap()
        })
        .clone()
}

/// A cached, valid 15-of-21 ML-DSA QuorumCertificate. Generated once with
/// seed `0xB0C` so blocks are byte-stable.
pub fn shared_qc() -> QuorumCertificate {
    static QC: OnceLock<QuorumCertificate> = OnceLock::new();
    QC.get_or_init(|| {
        let mut rng = Rng::new(0xB0C);
        let keys: Vec<SigningKey> = (0..21u64)
            .map(|i| {
                let mut b = [0u8; 32];
                b[..8].copy_from_slice(&rng.next_u64().to_le_bytes());
                b[24..32].copy_from_slice(&i.to_le_bytes());
                SigningKey::from_seed(&b).unwrap()
            })
            .collect();
        let roster: Vec<VerifyingKey> = keys.iter().map(|k| *k.verifying_key()).collect();
        let epoch = Epoch::from_u64(3);
        let subject = h(&mut rng);
        let mut vc = VoteCollector::new(epoch, subject);
        for (i, k) in keys.iter().enumerate().take(15) {
            vc.add(i, k.sign(&vote_bytes(epoch, &subject)).unwrap(), &roster).unwrap();
        }
        vc.assemble(15).unwrap()
    })
    .clone()
}

/// Deterministic in-memory test world: maps tau roots, transit-log roots,
/// and resolved legs onto the three things `apply_block` needs to read from
/// `BeaconView` + `ChainSource`.
#[derive(Default)]
pub struct TestWorld {
    /// interval → tau root
    pub tau: BTreeMap<u64, Hash256>,
    /// (shard, height) → transit-log root
    pub transit_roots: BTreeMap<(ShardId, u64), Hash256>,
    /// (height, leg-key) → resolved leg
    pub chain: BTreeMap<(u64, nerv_core::types::LegKey), ResolvedLeg>,
}

impl BeaconView for TestWorld {
    fn tau_root(&self, interval: Interval) -> Option<Hash256> {
        self.tau.get(&interval.as_u64()).copied()
    }
    fn transit_root(&self, shard: ShardId, height: Height) -> Option<Hash256> {
        self.transit_roots.get(&(shard, height.as_u64())).copied()
    }
}

impl ChainSource for TestWorld {
    fn settled_leg(&self, height: Height, key: &nerv_core::types::LegKey) -> Option<ResolvedLeg> {
        self.chain.get(&(height.as_u64(), *key)).cloned()
    }
}

/// Build a `ShardBlock` with the test-world defaults filled in. `legs`,
/// `reversions`, and `claims` are caller-supplied; everything else
/// (header defaults, payout address, qc) is the cached testkit value.
pub fn block(
    shard: ShardId,
    interval: u64,
    legs: Vec<SettledLeg>,
    reversions: Vec<ReversionRecord>,
    claims: Vec<ClaimRecord>,
) -> ShardBlock {
    ShardBlock {
        shard,
        header: ShardHeader {
            prev: Hash256::from_bytes([0u8; 32]),
            height: Height::ZERO,
            nct_root: nerv_custody::NctDigest::from_elements(&[Goldilocks::ZERO; 4]),
            nullifier_root: Hash256::from_bytes([0u8; 32]),
            transit_root: Hash256::from_bytes([0u8; 32]),
            params_root: params(),
            derived: [0u8; 32],
            ct_batch_hash: Hash256::from_bytes([0u8; 32]),
            prev_reveal: None,
            registry: RegistryRef {
                interval: Interval::from_u64(interval),
                root: Hash256::from_bytes([0u8; 32]),
            },
            fee_total: FeeSats::ZERO,
            producer_payout: payout(),
            qc_hash: Hash256::from_bytes([0u8; 32]),
        },
        legs,
        reversions,
        claims,
        qc: shared_qc(),
    }
}

/// Register the tau tree for `interval` containing the sorted `txids` in
/// the test world's `tau` map. Returns the tree so callers can build
/// witnesses via `tree.witness(tree.position(&txid).unwrap())`.
pub fn register_tau(world: &mut TestWorld, interval: u64, txids: &[TxId]) -> TauTree {
    let mut v = txids.to_vec();
    v.sort();
    let tree = TauTree::from_sorted(&v).unwrap();
    world.tau.insert(interval, tree.root());
    tree
}

/// Independently complete the block's header from the pre-state and the
/// block's effects (the differential the executor must agree with on
/// valid blocks). Effect failures are ignored — invalid blocks never reach
/// the post-root checks, so their sealed roots are irrelevant.
#[allow(clippy::too_many_arguments)]
pub fn seal(st: &ShardState, b: &mut ShardBlock, world: &mut TestWorld, mints: &[Vec<Hash256>]) {
    let h = Height::from_u64(st.height().as_u64() + 1);
    let resolved = b.resolve_legs().ok();
    let interval = b.header.registry.interval.as_u64();
    world.tau.entry(interval).or_insert_with(|| {
        let txids: Vec<TxId> = match &resolved {
            Some(rs) => {
                let mut v: Vec<TxId> = rs.iter().map(|r| r.txid).collect();
                v.sort();
                v
            }
            None => vec![],
        };
        let tree = TauTree::from_sorted(&txids).unwrap();
        tree.root()
    });
    b.header.prev = st.state_commitment();
    b.header.height = h;
    b.header.registry.root = world.tau[&interval];
    b.header.qc_hash = b.qc.qc_hash();
    let Some(resolved) = resolved else { return };

    let mut nct = st.nct().clone();
    let mut nfs = st.nullifiers().clone();
    let mut log = st.transit().clone();
    let mut fee = 0u64;
    for (sl, r) in b.legs.iter().zip(resolved.iter()) {
        let l = r.leg_shell();
        for out in &l.outputs {
            let _ = nct.append(&out.cm);
        }
        for nf in &l.inputs.nullifiers {
            let _ = nfs.insert(nf, h.as_u64());
        }
        let has_cond = r.canon.legs.iter().any(|x| x.outputs.iter().any(|o| o.conditional));
        let estate = if !l.inputs.nullifiers.is_empty() && has_cond {
            TransitEntryState::Pending
        } else {
            TransitEntryState::Claimed
        };
        let _ = log.insert_pending(TransitEntry {
            txid: r.txid,
            shard: st.shard(),
            leg: sl.leg,
            state: estate,
            height: h,
            expiry: l.expiry,
        });
        fee += l.fee.as_u64();
    }
    for (rec, mint) in b.reversions.iter().zip(mints.iter()) {
        let _ = log.revert(&transit_key(&rec.txid, &st.shard(), rec.spend_leg), h);
        for cm in mint {
            let _ = nct.append(cm);
        }
    }
    for rec in &b.claims {
        let _ = log.claim(&transit_key(&rec.txid, &st.shard(), rec.spend_leg), h);
    }
    b.header.nct_root = nct.root();
    b.header.nullifier_root = nfs.root();
    b.header.transit_root = log.root();
    b.header.fee_total = FeeSats::from_u64(fee);
    b.header.ct_batch_hash = Hash256::concat(&nerv_core::constants::CT_BATCH, &ct_sum(&resolved).unwrap().to_bytes());
    for r in &resolved {
        world.chain.insert((h.as_u64(), r.key), r.clone());
    }
}

/// Advance the chain by `n` empty blocks. Every step seals an empty block
/// (no legs, no reversions, no claims) and calls `apply_block`.
pub fn advance(mut st: ShardState, world: &mut TestWorld, n: u64) -> ShardState {
    for _ in 0..n {
        let h = st.height().as_u64() + 1;
        let mut b = block(st.shard(), TAU0 + h, vec![], vec![], vec![]);
        seal(&st, &mut b, world, &[]);
        let (s, _) = apply_block(st, &b, world, world).unwrap();
        st = s;
    }
    st
}

/// Build a `ShardState::genesis` for the given shard with the testkit's
/// params_root.
pub fn genesis(shard: ShardId) -> ShardState {
    ShardState::genesis(shard, params())
}

/// The empty `NoteCommitmentTree` root — useful as a no-history anchor.
pub fn empty_anchor() -> Hash256 {
    NoteCommitmentTree::new().root().as_hash256()
}

/// Convenience builder: returns a fully-sealed one-leg block, its txid,
/// and the first nullifier it spent.
pub fn one_leg_block(
    rng: &mut Rng,
    shard: ShardId,
    st: &ShardState,
    world: &mut TestWorld,
    inputs: Vec<Hash256>,
    outputs: Vec<(u64, bool)>,
) -> (ShardBlock, TxId, Hash256) {
    let nf = inputs.first().copied().unwrap_or_else(|| h(rng));
    let l = leg(
        rng,
        shard,
        if inputs.is_empty() { vec![nf] } else { inputs.clone() },
        outputs.clone(),
        1000,
        5_000,
        empty_anchor(),
    );
    let shell = TransactionShell { legs: vec![l] };
    let txid = shell.txid().unwrap();
    let interval = TAU0 + st.height().as_u64() + 1;
    let tau = register_tau(world, interval, &[txid]);
    let sl = SettledLeg {
        shell,
        leg: LegIndex::FIRST,
        tau: tau.witness(tau.position(&txid).unwrap()).unwrap(),
        siblings: vec![],
    };
    let mut b = block(shard, interval, vec![sl], vec![], vec![]);
    seal(st, &mut b, world, &[]);
    (b, txid, nf)
}

/// Legacy helper: a one-spend-leg block with two non-conditional outputs
/// (1B and 2B), a 1000-nano fee, anchored to the state's NCT root. Returns
/// the sealed block and its txid. Used by `fraud.rs::tests` and
/// `store.rs::tests` for header / store round-trips where the body
/// contents aren't under test.
pub fn valid_block(st: &ShardState, world: &mut TestWorld, seed: u64) -> (ShardBlock, TxId) {
    let mut rng = Rng::new(seed ^ 0x5EED_0000);
    let shard = st.shard();
    let __nf_a = h(&mut rng);
    let shell = TransactionShell {
        legs: vec![leg(
            &mut rng,
            shard,
            vec![__nf_a],
            vec![(1_000_000_000, false), (2_000_000_000, false)],
            1000,
            5_000,
            st.nct().root().as_hash256(),
        )],
    };
    let txid = shell.txid().unwrap();
    let interval = TAU0 + st.height().as_u64() + 1;
    let tree = register_tau(world, interval, &[txid]);
    let sl = SettledLeg {
        shell,
        leg: LegIndex::FIRST,
        tau: tree.witness(tree.position(&txid).unwrap()).unwrap(),
        siblings: vec![],
    };
    let mut b = block(shard, interval, vec![sl], vec![], vec![]);
    seal(st, &mut b, world, &[]);
    (b, txid)
}

/// Legacy helper: build a chain of `n` valid blocks. `states[i]` is the
/// state after `i` blocks (so `states[0]` is genesis). Distinct seeds
/// give distinct chains.
pub fn build_chain(shard: ShardId, n: u64, seed: u64) -> (Vec<ShardBlock>, Vec<ShardState>, TestWorld) {
    let mut world = TestWorld::default();
    let mut st = ShardState::genesis(shard, params());
    let mut blocks = Vec::new();
    let mut states = vec![st.clone()];
    for _ in 0..n {
        let (b, _) = valid_block(&st, &mut world, seed ^ (states.len() as u64));
        let (s, _) = apply_block(st, &b, &world, &world).unwrap();
        st = s;
        blocks.push(b);
        states.push(st.clone());
    }
    (blocks, states, world)
}

// End of testkit module.
