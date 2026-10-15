//! Criterion bench for `nerv_proofs::witness_gen::TransactionWitness::generate`.
// Bench harnesses call deterministic operations with fixtures they fully
// control, so a panic on `unwrap`/`expect` is the appropriate failure mode
// for the bench itself — production code never sees these.
#![allow(clippy::unwrap_used, clippy::clone_on_copy, clippy::expect_used)]
//!
//! Per D4 §12.3: this is the per-transaction witness assembler that runs
//! before any STARK prove. It binds the custody witness to the shell,
//! builds each leg's `FeatureVector` (per-leg move), runs `W·ΔS`, and
//! produces the seal statement-10 triplets. The bench uses the same
//! 2-leg cross-shard fixture the conformance C1/C2 path uses, so a
//! regression here propagates to C1a timings.

use criterion::{criterion_group, criterion_main, Criterion};
use nerv_codec::codec_w::{CodecW, WeightVersion};
use nerv_codec::weight_gen::{certify, expand, BeaconRandomness, CertConfig};
use nerv_core::hash::Hash256;
use nerv_core::types::{FeeSats, Height, ShardSet};
use nerv_custody::commitment::NoteOpening;
use nerv_custody::nct::NoteCommitmentTree;
use nerv_custody::nullifier::derive_nullifier;
use nerv_custody::tx::{InputSet, LegShell, Output, TransactionShell};
use nerv_custody::{Address, MasterSeed, WalletKeys};
use nerv_proofs::air::custody_air::{
    CustodyWitness, InputWitness, OutputWitness, RevertWitness,
};
use nerv_proofs::witness_gen::TransactionWitness;
use nerv_seal::encrypt::{derive_reference_keypair, PublicKey};
use nerv_seal::sampling::NoiseSeed;

#[derive(Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn bytes32(&mut self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for c in out.chunks_exact_mut(8) {
            c.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        out
    }
}

struct Fixture {
    shell: TransactionShell,
    custody: CustodyWitness,
    seeds: Vec<NoiseSeed>,
    w: CodecW,
    pk: PublicKey,
}

fn make_fixture(seed: u64) -> Fixture {
    let mut rng = SplitMix64::new(seed);
    let wk = WalletKeys::from_master(&MasterSeed::from_bytes(rng.bytes32()));
    let det = wk.viewing();
    let g = ShardSet::genesis();
    let addr = |i: u64| Address::generate(&det, wk.nullifier_key(), i, &g).unwrap();
    let mut opening = |v: u64, i: u64| NoteOpening {
        value: v,
        rho: rng.bytes32(),
        delivery: *addr(i).delivery().as_bytes(),
        blinding: rng.bytes32(),
        pk_n: *addr(i).pk_n(),
    };
    let nanos = 1_000_000_000u64;
    let in1 = opening(50 * nanos, 0);
    let in2 = opening(35 * nanos, 1);
    let out_carol = opening(25 * nanos, 2);
    let out_change = opening(9_999 * nanos / 1000, 3);
    let out_bob = opening(50 * nanos, 4);
    let revert_bob = opening(50 * nanos, 4);

    let mut tree = NoteCommitmentTree::new();
    for _ in 0..6 {
        tree.append(&Hash256::from_bytes(rng.bytes32())).unwrap();
    }
    let idx1 = tree.append(&in1.commitment().unwrap()).unwrap();
    let idx2 = tree.append(&in2.commitment().unwrap()).unwrap();
    let root = tree.root();
    let sib = |idx: u64| -> Vec<[nerv_core::field::Goldilocks; 4]> {
        tree.witness(idx)
            .unwrap()
            .siblings
            .iter()
            .map(|d| d.to_elements().unwrap())
            .collect()
    };
    let inputs = vec![
        InputWitness {
            opening: in1,
            nullifier_key: wk.nullifier_key_at(0),
            leaf_index: idx1,
            siblings: sib(idx1),
            anchor: root,
        },
        InputWitness {
            opening: in2,
            nullifier_key: wk.nullifier_key_at(1),
            leaf_index: idx2,
            siblings: sib(idx2),
            anchor: root,
        },
    ];
    let nf1 = derive_nullifier(&inputs[0].nullifier_key, &inputs[0].opening.rho);
    let nf2 = derive_nullifier(&inputs[1].nullifier_key, &inputs[1].opening.rho);
    let mk_out = |o: &NoteOpening, cond: bool| Output {
        cm: o.commitment().unwrap(),
        sealed_note: vec![0xA5; 48],
        value: o.value,
        conditional: cond,
        revert_cm: if cond { Some(revert_bob.commitment().unwrap()) } else { None },
    };

    // Pinned to the first two genesis shards so the fixture is
    // shard-deterministic (matches the C1/C2 test path).
    let leg7 = LegShell {
        shard: g.ids()[0],
        inputs: InputSet::new(vec![nf1, nf2]),
        outputs: vec![mk_out(&out_carol, false), mk_out(&out_change, false)],
        fee: FeeSats::from_u64(600_000),
        anchor: root.as_hash256(),
        expiry: Height::from_u64(5_000),
        weight_version: 1,
        ct: vec![],
        burns: vec![],
    };
    let leg40 = LegShell {
        shard: g.ids()[1],
        inputs: InputSet::new(vec![]),
        outputs: vec![mk_out(&out_bob, true)],
        fee: FeeSats::from_u64(400_000),
        anchor: Hash256::from_bytes(rng.bytes32()),
        expiry: Height::from_u64(5_000),
        weight_version: 1,
        ct: vec![],
        burns: vec![],
    };

    let shell = TransactionShell { legs: vec![leg7, leg40] };
    let custody = CustodyWitness {
        inputs,
        outputs: vec![
            OutputWitness { opening: out_carol },
            OutputWitness { opening: out_change },
            OutputWitness { opening: out_bob },
        ],
        reverts: vec![RevertWitness { opening: revert_bob }],
        burns: vec![],
        fees: vec![600_000, 400_000],
    };
    let seeds = vec![NoiseSeed::from_bytes(rng.bytes32()), NoiseSeed::from_bytes(rng.bytes32())];

    let w = expand(&BeaconRandomness::from_bytes([0x57; 32]), WeightVersion(1));
    certify(&w, CertConfig { spark_samples_per_size: 2, column_rail_samples: 1 }).unwrap();
    let pk = derive_reference_keypair(&[0xE0; 32]).unwrap().0;

    Fixture { shell, custody, seeds, w, pk }
}

fn bench_witness_gen(c: &mut Criterion) {
    let f = make_fixture(0x57A1);
    c.bench_function("witness_gen::TransactionWitness::generate (2-leg)", |b| {
        b.iter(|| TransactionWitness::generate(&f.shell, f.custody.clone(), &f.w, &f.pk, &f.seeds).unwrap())
    });
}

criterion_group!(benches, bench_witness_gen);
criterion_main!(benches);
