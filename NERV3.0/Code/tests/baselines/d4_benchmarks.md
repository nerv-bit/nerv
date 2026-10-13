# D4 criterion baseline (Session 13, 2026-10-12)

Captured on a Windows-10 8-GB box, `C:\NERV` (junction) + `C:\NERV\target`,
Rust 1.85.0 gnu, `cargo bench` (release profile, default `opt-level=3`).

The bench source is at:

- `crates/nerv-seal/benches/ring_ntt.rs`
- `crates/nerv-codec/benches/build_leg_features.rs`
- `crates/nerv-net/benches/sphinx.rs`
- `crates/nerv-proofs/benches/witness_gen.rs`
- `crates/nerv-state/benches/apply_block.rs` (requires the `testkit` feature)

The CI workflow `.github/workflows/ci.yml` `bench:` job re-runs all five
nightly. 5% slowdown on any of the 5 production hot paths is the failure
gate; a >10% slowdown is a hard alert.

| Bench | Variant | Median (95% CI) | Sample size |
|-------|---------|-----------------|-------------|
| `ring::Poly::ntt`        | 256 coeffs               | 25.5 µs  (24.6 – 27.7)  | 10 |
| `ring::Poly::intt`       | 256 coeffs               | 29.8 µs  (29.5 – 30.1)  | 10 |
| `codec::build_leg_features` | 1-in / 1-out           | 1.07 µs  (0.95 – 1.18)  | 10 |
| `codec::build_leg_features` | 2-in / 3-out           | 1.86 µs  (1.84 – 1.87)  | 10 |
| `sphinx::build`          | 64 B payload, 5 relays   | 939 µs   (930 – 947)    | 10 |
| `sphinx::build`          | 14_153 B payload, 5 relays | 1.17 ms (1.02 – 1.38) | 10 |
| `sphinx::peel`           | 1st relay, 5-relay packet | 288 µs  (230 – 355)    | 10 |
| `witness_gen::TransactionWitness::generate` | 2-leg fixture | 27.6 ms (26.4 – 28.5) | 10 |
| `apply_block`            | 1-leg block              | 5.83 ms  (5.78 – 5.88)  | 10 |

Each bench ran with `--warm-up-time 1 --measurement-time 1 --sample-size 10`.
The witness_gen bench bumped itself to 1.5 s of measurement time (criterion's
"Warning: Unable to complete 10 samples in 1.0s") — that is expected for a
multi-millisecond function; 10 samples is enough to bracket the median.

**Unblocking the `dlltool` blocker.** Criterion's transitive dep `windows-sys`
runs MinGW `dlltool` on the workspace path, which used to split the space in
`C:\Users\prosp\Documents\MiniMax NERV`. The fix is a junction at
`C:\NERV -> C:\Users\prosp\Documents\MiniMax NERV`, created once with:

```powershell
cmd.exe /c 'mklink /J "C:\NERV" "C:\Users\prosp\Documents\MiniMax NERV"'
```

All cargo invocations for D4 must be run from `C:\NERV` (the junction)
or the dlltool step fails with the same path-split error. Future sessions
should `cd C:\NERV` before `cargo bench`.
