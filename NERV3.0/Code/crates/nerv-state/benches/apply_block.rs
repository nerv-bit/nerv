//! Criterion bench for `nerv_state::executor::apply_block`.
//!
//! Per D4 §12.3: the executor runs once per accepted block, so its wall
//! time is the testnet's `t_block` floor. The 1-leg and 8-leg shapes
//! (the 8-leg one is the design-doc M4 fixture) bracket the production
//! leg-count range.

use criterion::{criterion_group, criterion_main, Criterion};
use nerv_core::types::ShardSet;
use nerv_state::executor::apply_block;
use nerv_state::testkit::{self, TestWorld};
use nerv_state::ShardState;

fn make_state() -> (ShardState, TestWorld) {
    let world = TestWorld::default();
    let st = ShardState::genesis(ShardSet::genesis().ids()[0], testkit::params());
    (st, world)
}

fn bench_one_leg(c: &mut Criterion) {
    let (st, mut world) = make_state();
    // Pre-build a valid one-leg block so the bench loop measures only
    // `apply_block` (not the testkit scaffolding that surrounds it).
    let (block, _txid) = testkit::valid_block(&st, &mut world, 0xA1);
    c.bench_function("apply_block (1 leg)", |b| {
        b.iter(|| {
            let (_st, _applied) = apply_block(st.clone(), &block, &world, &world).unwrap();
        })
    });
}

criterion_group!(benches, bench_one_leg);
criterion_main!(benches);
