//! Criterion bench for `nerv_net::sphinx::build` and `peel`.
// Bench harnesses call deterministic operations with fixtures they fully
// control, so a panic on `unwrap`/`expect` is the appropriate failure mode
// for the bench itself — production code never sees these.
#![allow(clippy::unwrap_used, clippy::clone_on_copy, clippy::expect_used)]
//!
//! Per D4 §12.3: the Sphinx packet build is on every submission fan-out's
//! hot path, and `peel` runs on every inbound packet. The 5-relay
//! production shape is the only one we ship (PATH_RELAYS is a const
//! frozen by the design doc), so we bench the production shape and a
//! half-payload reference for comparison.

use criterion::{criterion_group, criterion_main, Criterion};
use nerv_net::sphinx::{build, encap_rand, node_keys, peel, PathSpec, TerminalAction};
use nerv_net::host::PeerId;

fn make_spec(seed: u64) -> (PathSpec, PeerId, nerv_crypto::mlkem::DecapsulationKey) {
    // Production path: 5 relays (PATH_RELAYS). Use node_keys to mint each
    // relay's (peer id, ek, dk) from a deterministic seed; the wallet
    // (PeerId, ek) tuples are what `PathSpec::new` consumes, and we
    // remember relay 0's decapsulation key so we can `peel` on it.
    let relays: Vec<(PeerId, _, _)> = (1..=5u64)
        .map(|i| {
            let (sk, ek, dk) = node_keys(seed.wrapping_add(i));
            (PeerId::of(sk.verifying_key()), ek, dk)
        })
        .collect();
    let dest_sk = nerv_crypto::mldsa::SigningKey::from_seed(&[0x42u8; 32]).unwrap();
    let dest = PeerId::of(dest_sk.verifying_key());

    let relay_tuples: [(PeerId, _); 5] = [
        (relays[0].0.clone(), relays[0].1.clone()),
        (relays[1].0.clone(), relays[1].1.clone()),
        (relays[2].0.clone(), relays[2].1.clone()),
        (relays[3].0.clone(), relays[3].1.clone()),
        (relays[4].0.clone(), relays[4].1.clone()),
    ];
    let spec = PathSpec::new(relay_tuples, dest, TerminalAction::Deliver);
    (spec, dest, relays[0].2.clone())
}

fn bench_build_small(c: &mut Criterion) {
    let (spec, _dest, _dk) = make_spec(0xA17);
    let data = vec![0u8; 64];
    let r = encap_rand(0xB17);
    c.bench_function("sphinx::build (64B payload, 5 relays)", |b| {
        b.iter(|| build(&spec, &data, &r).unwrap())
    });
}

fn bench_build_full(c: &mut Criterion) {
    let (spec, _dest, _dk) = make_spec(0xA17);
    // Full payload — the production MAX_DATA limit.
    let data = vec![0u8; 14_153];
    let r = encap_rand(0xB17);
    c.bench_function("sphinx::build (14_153B payload, 5 relays)", |b| {
        b.iter(|| build(&spec, &data, &r).unwrap())
    });
}

fn bench_peel(c: &mut Criterion) {
    let (spec, _dest, dk0) = make_spec(0xA17);
    let data = vec![0u8; 14_153];
    let r = encap_rand(0xB17);
    let pkt = build(&spec, &data, &r).unwrap();
    c.bench_function("sphinx::peel (1st relay, 5-relay packet)", |b| {
        b.iter(|| peel(&pkt, &dk0).unwrap())
    });
}

criterion_group!(benches, bench_build_small, bench_build_full, bench_peel);
criterion_main!(benches);
