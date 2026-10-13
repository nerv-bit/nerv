//! Criterion smoke benchmark for `nerv-seal::ring::ntt` and `intt`.
//!
//! Sanity baseline only: a few iterations of the 256-coefficient NTT/INTT
//! paths. The real regression-threshold baseline is recorded on a quieter
//! box. D4 plan §12.3 specifies three sample sizes; this file is the
//! minimal first pass and is expanded as the M4 gate builds out.

use criterion::{criterion_group, criterion_main, Criterion};
use nerv_seal::ring::Poly;

const N: usize = 256;

/// Deterministic, non-trivial 256-coeff polynomial (small balanced
/// representatives) so the bench is reproducible and exercises the
/// full 8-stage Cooley-Tukey loop instead of benching a zero input.
fn make_poly() -> Poly {
    let mut coeffs = [0i64; N];
    for (i, c) in coeffs.iter_mut().enumerate() {
        // Wrap into roughly ±2^30 so we never mod-out via a quiet path.
        *c = ((i as i64).wrapping_mul(0x12345) - 100_000) & 0x3FFFFFFF;
    }
    Poly::from_centered(&coeffs)
}

fn bench_poly_ntt(c: &mut Criterion) {
    let p = make_poly();
    c.bench_function("ring::Poly::ntt (256 coeffs)", |b| b.iter(|| p.ntt()));
}

fn bench_poly_intt(c: &mut Criterion) {
    // Start from the NTT form so the inverse has a real signal to undo.
    let p_ntt = make_poly().ntt();
    c.bench_function("ring::Poly::intt (256 coeffs)", |b| b.iter(|| p_ntt.intt()));
}

criterion_group!(benches, bench_poly_ntt, bench_poly_intt);
criterion_main!(benches);
