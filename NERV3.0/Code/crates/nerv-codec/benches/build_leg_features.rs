//! Criterion bench for `nerv-codec::features::build_leg_features`.
//!
//! Per D4 §12.3: the leg-feature builder is on the per-block hot path (every
//! leg of every transaction re-encodes its own `FeatureVector`). The
//! benchmark measures both the minimal 1-in/1-out (single-shard payment)
//! and a fuller 2-in/3-out (cross-shard escrow) leg shape so that a
//! regression in the 12-rail loop is visible.

use criterion::{criterion_group, criterion_main, Criterion};
use nerv_codec::features::{build_leg_features, LegKind, LegMovement};

fn minimal_leg() -> LegMovement {
    // 1 input, 1 output, small value — the smallest admissible leg.
    LegMovement {
        inputs: vec![(vec![0xAA; 32], 1_000)],
        outputs: vec![(vec![0xBB; 32], 900)],
        fee_nano: 100,
        kind: LegKind::SingleShard,
        expiry_height: 42,
        epoch_length_blocks: 1000,
    }
}

fn fuller_leg() -> LegMovement {
    // 2 inputs, 3 outputs, mid-range values — the 2-in/3-out shape used by
    // cross-shard escrow and other multi-output legs.
    LegMovement {
        inputs: vec![(vec![0x11; 32], 50_000), (vec![0x22; 32], 35_000)],
        outputs: vec![
            (vec![0x33; 32], 25_000),
            (vec![0x44; 32], 50_000),
            (vec![0x55; 32], 9_999),
        ],
        fee_nano: 1,
        kind: LegKind::CrossShardSpend,
        expiry_height: 7,
        epoch_length_blocks: 1000,
    }
}

fn bench_minimal(c: &mut Criterion) {
    let m = minimal_leg();
    c.bench_function("codec::build_leg_features (1in/1out)", |b| {
        b.iter(|| build_leg_features(&m).unwrap())
    });
}

fn bench_fuller(c: &mut Criterion) {
    let m = fuller_leg();
    c.bench_function("codec::build_leg_features (2in/3out)", |b| {
        b.iter(|| build_leg_features(&m).unwrap())
    });
}

criterion_group!(benches, bench_minimal, bench_fuller);
criterion_main!(benches);
