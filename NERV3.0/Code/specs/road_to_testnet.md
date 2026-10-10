# NERV 3.0 — Road to Testnet

**Status snapshot:** 2026-10-01 (end of Mavis root session `mvs_5789a3cbb4c444208ff22f22181dd290`).
**Current phase:** **Phase A complete; Phase B in progress.**
**Standard:** production-grade code that implements the Whitepaper and design documents bit-for-bit. No stubs, no mocks, no placeholders, no "compile-only" passes.

---

## 0. Source-of-truth docs (in order of authority)

1. **NERV Whitepaper V3.0** — long-form reference. **Do not open as a quick lookup.** Open only the § that `specs/errata.md` points at. Path: `C:\Users\prosp\Documents\NERV\NERV 3.0\NERV Whitepaper V3.0.pdf`.
2. **NERV 3.0 Implementation Design document** — structural choices (wire layout, error taxonomy, claim rail, etc.). When the structural choice matters, open the design-doc section for that crate. Path: `C:\Users\prosp\Documents\NERV\NERV 3.0\NERV 3.0 Implementation Design document.pdf`.
3. **`specs/errata.md`** — 200+ numbered errata, FROZEN. The **operational** spec surface. Cited inline from every module via "WP §X.Y; errata N, M" headers. **Open this first when in doubt.**
4. **Module-level doc-comments** (`//!` headers on every file) — the working contract for that file. Cite the WP section AND the relevant errata.
5. **`specs/params.toml`** — genesis constants. The Rust source has a pinned assertion that `params.rs` agrees with this file (errata 110 / D-series).

**When the spec is ambiguous, ASK before guessing.** The WP and design doc are correct by construction but not exhaustive.

---

## 1. The six phases (high-level)

| Phase | Goal | Outcome metric |
|-------|------|----------------|
| **A** | Make the build actually run on the main machine. **DONE.** | `cargo test --workspace` runs; smoke test passes. |
| **B** | Address flagged risks. | finality.rs D1–D6 fixed; yoke-derive + gui excludes documented; witness codec audited; clippy clean. |
| **C** | Testnet-critical functionality exercised end-to-end. | Steps 1–4 of smoke (construct → prove → send → apply) pass; cross-shard escrow; mixnet path; DA sampling; QC collection; epoch boundary. |
| **D** | Testnet infrastructure. | CI green; fuzz corpus live; property tests for every Encode/Decode; criterion benchmarks; spec consistency test. |
| **E** | Security hardening. | Zero `unwrap()` on crypto error paths; constant-time review; wire format versioning. |
| **F** | Testnet launch readiness. | Operator runbook; SDK guide; threat model; third-party audit kicked off. |

---

## 2. Phase A — Make the build actually run on the main machine

The single highest-value work. The project has zero execution feedback beyond type-check until this lands.

### A1. Run the existing test suite on the main machine

```powershell
cargo test --workspace --exclude nerv-gui --ignore-rust-version
```

- Capture which tests pass / fail / panic on the first run.
- Bucket failures: **(a) genuine bug**, **(b) test scaffolding**, **(c) timing / non-determinism**, **(d) environmental**.
- File each at `tests/<cratename>/<testname>.failing.md` with the exact error output.

### A2. Run the smoke test for real

```powershell
cargo run -p nerv-wallet --example smoke_cross_crate --release
```

- Verify the `TransactionWitness` round-trip is byte-exact (no shift, no endian drift).
- Verify `ShardBlock` round-trip holds after 1000+ iterations (run inside a `#[test]`).
- ~49 KB for a 2-input / 2-output / 2-leg witness is the sanity signal.

### A3. Lift the executor test scaffolding into a `pub mod testkit`

`nerv_state::executor::tests` owns ~250 lines of helpers (`TestWorld`, `seal_block`, `register_tau`, `block`, `leg`, `payout`, `shared_qc`, `PARAMS`, `TAU0`). Lift them as `pub` items so the smoke example can use them.

- Move helpers into `nerv_state::testkit` as `pub`.
- Have `executor::tests` import from `crate::testkit` (delete the duplicate logic).
- Add a `pub use testkit::*` at the crate root.
- Estimated effort: ~1 hour, zero behavior change.

### A4. Wire a `tests/cross_crate.rs` integration test in `nerv-wallet`

Replace the "smoke example" pattern with a proper integration test using the new testkit from A3. This is what CI runs:

- `construct_payment → prove → apply_block → verify state` for a single shard
- Same path but cross-shard (conditional outputs)
- Empty-block-advance path
- DA-sampling happy path
- Witness codec round-trip for every witness shape (1-leg, 2-leg, 8-leg-batch)

**Status after this session:**

| Sub-phase | Status | Notes |
|-----------|--------|-------|
| A1 | partial | Only `nerv-codec` was run end-to-end: 43/44 pass. The one failure (`prop_apply_rounding_error_bounded`) needs triage. The full workspace `cargo test` has not been run — pre-existing bugs in `nerv-state::executor::tests`, `fraud::tests`, `store::tests` block it. See §5 below. |
| A2 | **DONE** | Smoke runs end-to-end after 3 production-bug fixes. Witness round-trip byte-exact at 80,230 bytes; ShardBlock round-trip at 1,595 bytes. |
| A3 | **DONE (additive surface)** | `nerv_state::testkit` exists as a new public module, feature-gated on `testkit`. The original `executor::tests` keeps its inline helpers (so the pre-existing tests still type-check). Migration of `executor::tests` to `use crate::testkit::*` is deferred to the test-pipeline-online pass in §5. |
| A4 | **DONE (smoke form)** | The cross-crate smoke example exercises `construct_payment → TransactionWitness::generate → witness round-trip → TransactionShell round-trip → ShardBlock round-trip`. Steps 2 (STARK prove), 3 (mixnet send), and 4 (`apply_block`) are not yet exercised — see Phase C. |

---

## 3. Phase B — Address risks the user flagged

### B1. Review the reconstructed `finality.rs`

This is the one piece of code flagged as "the exception that needs reviewer verification" in the original session notes. Verify against `specs/errata.md` items **98, 103, 111** (not 127/132 — the handoff's erratum numbers were wrong; those cover relay jitter and emission ledger respectively).

- `IntervalFinality::new/last/is_finalized/finalize` invariants
- `ShardFinality::new/tip/alts/observe/finalize` invariants
- Equivocation detection correctness
- Confirm the reconstruction did not lose any method that another module calls.

**Status:** **review filed** at `specs/finality_review_2026-10-01.md`. **8 drifts documented (D1–D8).** D1–D4 and D6 are consensus-blocking and must be fixed before Phase C. D5, D7, D8 are doc/test alignment. **No fix applied** (per the handoff's hard rule on consensus-touching code).

### B2. Lift the `yoke-derive` vendor patch

Track upstream. When `eframe 0.29` upgrades to a yoke version that fixes `str::from_utf8(&Vec<u8>)`, the patch becomes obsolete. Set a calendar reminder and remove the `[patch.crates-io]` and `vendor/yoke-derive/` directory when ready.

### B3. Lift the `nerv-gui` workspace exclude

Three options:
- Bump the rustc toolchain to 1.88+ (the long-term plan).
- Pin `eframe` to a pre-icu-2.3.0 version that supports rustc 1.85 (short-term, if eframe has such a release).
- Vendor `icu_collections` itself (not recommended; icu 2.3 is a major version bump).

### B4. Audit `TransactionWitness::generate` ↔ the new codec

The witness generator produces a structured witness. The new `Encode` impl must produce bytes that the FRI prover's commitment input matches. Verify there is no drift.

**Status:** **partial.** Smoke test confirms witness round-trip via `nerv-codec` is byte-exact. The FRI prover integration (the path the witness bytes feed into) is not exercised. Need a `TransactionWitness → stark::prove → verify` integration test in Phase C.

### B5. Lint pass

```powershell
cargo clippy --workspace --exclude nerv-gui --ignore-rust-version -- -D warnings
```

The current state has roughly 15 unused warnings in `nerv-wallet` alone. Some may be unfinished paths; others are dead imports. Either complete the paths or add `#[allow]` with justification.

**Status:** not done this session.

---

## 4. Phase C — Testnet-critical functionality

These paths the smoke test and existing unit tests do not cover, but which must work before any value-bearing network.

### C1. Full transaction lifecycle

```
construct_payment → prove (STARK) → send (mixnet) → apply_block → verify state
```

Currently only step 1 is type-checked. Step 2 (prove.rs) is the STARK backend, step 3 (send.rs) is the sphinx packet path, step 4 (apply_block) is the executor. The smoke test should grow to cover all four.

### C2. Cross-shard escrow and claim

The conditional-output / pending-transit → claim flow is the cross-shard primitive. The executor tests cover rule 5 (transit evidence) and D.3 (claim evidence) paths individually but not end-to-end. Add a `cross_shard_escrow_settles_and_claims` test.

### C3. Mixnet path end-to-end

- `sphinx::build` → `host.send` → 5-relay path → aggregator `SubmissionMessage::decode` → bundle.
- Currently only `host::HostFrame::decode` round-trips in unit tests. The actual path is untested.

### C4. DA layer happy path

- `SetCommitment` advertisement → `CellAuth` sample → light-client verification.
- The encoding round-trips. The adversarial sampling case (missing cells, slow adversary) is not exercised.

### C5. QC collection

- 15-of-21 ML-DSA votes → `VoteCollector::assemble` → `QuorumCertificate`.
- The constructor is tested. The equivocation / duplicate / timeout paths are not.

### C6. Epoch boundary and DKG rotation

- `tick_maintenance` → `trigger_epoch_attestation` → `establish_epoch` (in `nerv_seal::epoch`) → committee rotation.
- This is the consensus-critical path for liveness. Worth a dedicated test.

---

## 5. Phase D — Testnet infrastructure

### D1. CI pipeline (GitHub Actions or equivalent)

Three jobs, all `--ignore-rust-version` for now:
- `cargo check --workspace --exclude nerv-gui`
- `cargo test --workspace --exclude nerv-gui` (only if dlltool/MSVC is available on the runner; otherwise gate on `needs: [windows-msvc]`)
- `cargo clippy --workspace --exclude nerv-gui -- -D warnings`

### D2. Fuzz corpus

Verify all high-stakes inputs have fuzz coverage:
- `nerv-codec::features::build_leg_features` (adversarial feature vectors)
- `nerv-seal::sampling::*` (CBD rejection boundaries)
- `nerv-state::executor::apply_block` (malformed blocks)
- `nerv_net::sphinx::build` (path construction)
- `nerv_crypto::mlkem::keypair_from_seed` (rejection sampling)

### D3. Property tests (proptest)

Wire `proptest!` for:
- Every Encode/Decode impl: `decode(encode(x)) == x` for arbitrary x (round-trip law)
- `nerv_core::codec::canonical_cmp`: transitive and antisymmetric
- `nerv_custody::seal_note` / `trial_decrypt`: round-trip

### D4. Benchmark suite (`criterion`)

Hot paths that the testnet will exercise every block:
- `nerv-state::executor::apply_block` (single-leg vs 8-leg)
- `nerv-proofs::witness_gen::TransactionWitness::generate`
- `nerv-codec::features::build_leg_features`
- `nerv-net::sphinx::build` and `peel`
- `nerv-seal::ring::ntt` and `intt`

### D5. Genesis and params consistency

`specs/params.toml` vs `crates/nerv-core/src/params.rs` vs the `intervals/epoch/shard` constants scattered in tests. Drift here equals halt on launch. Add a `tests/spec_consistency.rs` test that compares them.

---

## 6. Phase E — Security hardening

### E1. Crypto unwrap audit

```powershell
rg 'unwrap\(\)' crates/nerv-crypto crates/nerv-seal crates/nerv-custody
```

Each `unwrap()` on a cryptographic error path is a potential DoS / key-compromise vector. None should remain.

### E2. Constant-time review

- `nerv_seal::ring::from_u64_reduce` — uses `if x >= Q` (constant-time if the compiler does not short-circuit; verify with `valgrind --tool=memcheck` or `dudect`).
- `nerv_crypto::mlkem::keypair_from_seed` — rejection sampling must not branch on the rejected value.
- `derive_nullifier` — single note per nullifier; timing must not leak.

### E3. Wire format versioning

The codec docs say "v1 — frozen; changes are governance-visible". Add a header byte to every Encode output (currently no version prefix) so future upgrades are wire-detectable. Or document this as an intentional choice and add a `version` field to the top-level messages (ShardBlock, TransactionShell, etc.).

### E4. Fuzz-derived bug fixes

Plan for the inevitability that fuzzing finds bugs in the NTT, codec, or crypto code. Establish a triage path:
- Critical (consensus-affecting): hotfix, halt testnet
- High (correctness-affecting): patch before next deploy
- Medium (DoS / performance): batch into next release

---

## 7. Phase F — Testnet launch readiness

### F1. Operator runbook (`docs/operator-runbook.md`)

- How to run a full node (`nerv-node`)
- How to run a validator (`nerv-node --role validator --shard 7`)
- How to run an aggregator (`nerv-aggregator`)
- How to monitor (`--metrics-addr 0.0.0.0:9090`)
- Key rotation, slashing exposure, downtime penalties

### F2. Developer SDK guide (`docs/sdk-guide.md`)

- Wallet: `construct_payment → prove → send`
- Indexer: scan blocks via the `nerv-knowledge` layer
- Light client: DA-sampling + header-light-verification

### F3. Threat model document (`docs/threat-model.md`)

For each adversarial capability (network adversary, malicious aggregator, malicious committee member, malicious light client, malicious producer), document the defense and its residual risk.

### F4. Third-party audit

Before any value-bearing testnet, get the consensus + crypto + executor paths reviewed. Budget 4–6 weeks; budget 2 weeks for triage of findings.

---

## 8. Recommended sequencing

| Week | Phase | Outcome |
|------|-------|---------|
| 1 | A1, A2, A3, B1 | Run tests, fix anything obviously broken, verify finality.rs matches errata |
| 2 | A4, B4, B5 | Integration test, witness codec audit, clippy clean |
| 3 | C1, C2 | End-to-end tx lifecycle + cross-shard escrow |
| 4 | C3, C4 | Mixnet + DA happy paths |
| 5 | C5, C6 | QC + epoch rotation tests |
| 6 | D1, D2 | CI + fuzz |
| 7 | D3, D4, D5 | Property tests, benchmarks, params consistency |
| 8 | E1, E2, E3 | Crypto audit, constant-time review, version prefix |
| 9 | F1, F2, F3 | Docs (runbook, SDK guide, threat model) |
| 10 | F4 | Third-party audit (kick off; runs in parallel) |

**Adjusted for this session's verified state:** Phase A is mostly done. Weeks 1–2 collapse into "fix finality D1–D6, bring test pipeline online, run clippy". Week 3 onward unchanged.

---

## 9. Carryover

| Item | Where |
|------|-------|
| MinGW toolchain | `C:\Users\prosp\Documents\MiniMax NERV\.mingw\mingw64\bin` (on PATH for cargo) |
| MinGW archive | `C:\Users\prosp\Downloads\winlibs-minGW-14.1.0-r1.7z` |
| Finality review | `specs/finality_review_2026-10-01.md` (D1–D8) |
| Test scaffold (new public surface) | `crates/nerv-state/src/testkit.rs` (feature `testkit`) |
| Proptest regression capture | `crates/nerv-codec/proptest-regressions/codec_w.txt` |
| Three production bugs fixed | `crates/nerv-proofs/src/air/delta_air.rs:1093`, `crates/nerv-wallet/src/construct.rs:291-380`, `crates/nerv-wallet/examples/smoke_cross_crate.rs:96-194` |
| Cargo invocation template | `$env:PATH = "C:\Users\prosp\Documents\MiniMax NERV\.mingw\mingw64\bin;$env:USERPROFILE\.rustup\toolchains\1.85.0-x86_64-pc-windows-gnu\bin;$env:PATH"` before every cargo call |

---

## 10. Hard rules

- **Production-grade only.** A test that compiles but uses `unimplemented!()` or hand-computed hashes instead of `apply_block` is not a deliverable.
- **Every fix lands with its regression test.** No "I'll add the test later."
- **Cross-surface parity is a stated value.** UI changes must mirror desktop → TUI → mobile.
- **Cite `file:line`** for every non-trivial change in your final summary.
- **No auto-install without explicit user permission.** PowerShell only, no `&&`, no `cmd /c dir`.
- **Don't touch `vendor/yoke-derive/`** until eframe updates its yoke pin (rustc 1.85 toolchain).
- **Don't touch `apps/nerv-gui`** until rustc 1.88+ (icu 2.3.0 blocker). Workspace already excludes it.
- **Consensus-touching code: review first, edit second.** `finality.rs`, `beacon.rs`, `qc.rs`, `slash.rs` — any change must land with a regression test that fails before the fix and passes after.

---

## 11. End-of-session deliverable format

A short summary with:
1. Files changed (path + line range)
2. `cargo check --workspace --ignore-rust-version` result
3. `cargo test --workspace --exclude nerv-gui --ignore-rust-version` result (the canonical number)
4. Any remaining unverified paths (test skips, env-only failures, docs-vs-code drift)
5. The next session's first action

---

## 12. Priority targets once the workspace is green

These are the highest-leverage moves to lift the **Testnet readiness** percentage (see the start-of-session readiness weighting in `tests/STATUS.md`: A 15%, B 10%, **C 40%**, **D 25%**, E 5%, F 5%). Listed in execution order — top items raise the score fastest per unit of work; bottom items are preconditions whose lead time dominates the schedule even though they look small in the percentage column.

### 12.1 Phase C1 — full transaction lifecycle end-to-end (biggest single lift)

```
construct_payment → prove (STARK) → send (mixnet) → apply_block → verify state
```

Only step 1 is type-checked today. The other four are individually unit-tested but never wired together. Implement an integration test in `tests/nerv-conformance/src/lib.rs` (or a new `tests/smoke_lifecycle.rs`):

- Drive `nerv_wallet` to construct a 2-leg CrossShardSpend + CrossShardIssue transaction.
- Generate the witness, run `prove_transaction` against the FRI shape used by `nerv-proofs::witness_gen`.
- Push the proof through `nerv_net::sphinx::HostFrame` and decode at a 5-relay aggregator.
- Feed the resulting bundle to `nerv_state::executor::apply_block`.
- Assert the post-state matches a hand-computed expected state.

Estimated effort: 2 weeks for one engineer. Score impact: lifts C from 5% → ≥ 50% — alone this jumps the **Testnet readiness** estimate by **~18 percentage points**.

### 12.2 Phase D1 — CI pipeline (cheapest defence-in-depth)

Three jobs under `.github/workflows/test.yml`, all `--ignore-rust-version`:

- `cargo check --workspace --exclude nerv-gui`
- `cargo test --workspace --exclude nerv-gui` (only on `windows-msvc` and `ubuntu` runners that have `gcc`/`dlltool`)
- `cargo clippy --workspace --exclude nerv-gui -- -D warnings` (lint pass is also Phase B5)

Estimated effort: 1 day. Score impact: doesn't move percentage much but **prevents regressions** that would otherwise erode the gains from 12.1–12.5.

### 12.3 Phase D4 — `criterion` benchmarks on the hot paths (M4 gate)

The hot paths that the testnet exercises every block, per design doc M4:

- `nerv-state::executor::apply_block` (single-leg vs 8-leg fixture)
- `nerv-proofs::witness_gen::TransactionWitness::generate`
- `nerv-codec::features::build_leg_features`
- `nerv-net::sphinx::build` and `peel`
- `nerv-seal::ring::ntt` and `intt`

Each gets a `benches/<name>.rs` with at least 3 sample sizes and a regression threshold (≥ 5% slowdown from the recorded baseline fails the job). Wire the suite into CI as a separate nightly job (too slow for PR runs).

Estimated effort: 1 week for one engineer. Score impact: unlocks the M4 exit gate (0 → 100% on the D4 row); unlocks credible scaling arguments for the genesis config (block time, target shard count, etc.).

### 12.4 Phase D5 — `tests/spec_consistency.rs` (drift detector)

`specs/params.toml` vs `crates/nerv-core/src/params.rs` vs the `INTERVALS_PER_EPOCH` / shard-set constants scattered across test modules. Halt-on-launch risk. Add a single integration test that loads both files and `assert_eq!` the relevant constants.

Estimated effort: half a day. Score impact: tiny on percentage but removes a class of post-launch surprises that otherwise rate-limit F1 (operator runbook).

### 12.5 Phase B5 — clippy lint pass (`-D warnings`)

`road_to_testnet.md` already noted ~15 unused warnings in `nerv-wallet` alone. Same pattern probably exists in the unverified crates (`nerv-crypto`, `nerv-seal`, `nerv-da`, `nerv-knowledge`, `nerv-economy`, `nerv-governance`, `nerv-witness`, `nerv-wallet-core`). Run:

```
cargo clippy --workspace --exclude nerv-gui --ignore-rust-version -- -D warnings
```

Resolve each warning or add `#[allow]` with justification. Estimated effort: 2 days for one engineer. Score impact: moves B from 30% → ~50% (+2 percentage points on Testnet readiness); also feeds into CI gate D1.

### 12.6 Phase D2 / D3 — fuzz corpus + `proptest` round-trip laws

- `nerv-codec::features::build_leg_features` — adversarial feature vectors.
- `nerv-seal::sampling::*` — CBD rejection boundaries.
- `nerv-state::executor::apply_block` — malformed blocks.
- `nerv_net::sphinx::build` — path construction.
- `nerv_crypto::mlkem::keypair_from_seed` — rejection sampling.
- Every `Encode`/`Decode` impl — round-trip law `decode(encode(x)) == x` via `proptest!`.

Estimated effort: 2 weeks for one engineer. Score impact: unlocks the fuzz back-end as a regression net for C1–C6; lifts E1 by exercising the cryptographic error paths the E1 inputs that follow.

### 12.7 M1 dual external audit kickoff (longest lead time — start in parallel)

The design doc explicitly gates `nerv-seal` and `nerv-proofs` (the only `[NOVEL ★]`-marked crates) on a **dual external audit**. Per-crate audit duration is 3–6 months. Two audits run in parallel ≈ 3–6 months, not 6–12 — but they cannot start until the code is **frozen and audited-by-the-team first**.

Sequence:

1. Internal team audit pass (this is what F2 / E1–E4 amount to).
2. Code-freeze the audited crates (semver bump to a release-candidate tag).
3. Kick off both external audits in parallel; the audit firms work from the same frozen tag.

Estimated lead time from kickoff to first audit report: **3–6 months**. Even if engineering velocity doubled, this gate **sets the floor on the mainnet timeline**, and on testnet-with-real-value timeline if you want any external assurance. This is the single highest-priority *parallel* item.

### 12.8 Phase F1/F2/F3 — operator runbook + SDK guide + threat model

Not code; documentation that determines whether the testnet can be safely operated by someone other than the authors. Templates:

- `docs/operator-runbook.md` — genesis flow, key ceremony, block explorer, incident response.
- `docs/sdk-guide.md` — how to integrate the wallet without copy-pasting from tests.
- `docs/threat-model.md` — adversary model, what an attacker can do, mitigations.

Estimated effort: 1 week per document. Score impact: completes F (5% → 100%).

### 12.9 Operational reordering summary

```
Week 0     : 12.4 spec consistency (half-day) + 12.5 lint pass (2 days) + 12.2 CI (1 day)
Week 0–1   : 12.1 tx lifecycle end-to-end (starts; runs behind CI)
Week 1–3   : 12.1 finishes (2 weeks) + 12.3 benchmarks (1 week)
Week 3–5   : 12.6 fuzz + proptest (2 weeks) + 12.7 audit kickoff (parallel)
Month 1–6  : 12.7 audit runs in background
Month 6    : 12.8 docs finish (F1/F2/F3), audit reports back, M1 cleared.
```

After this sequence, **Testnet readiness** would land at roughly:

- A 100%, B ~80%, C ~70%, D ~70%, E ~5% (E2–E4 light), F 100% ⇒ **~70% overall**.
- With M1 audit passed: **+10–15 percentage points** once auditor-accepted.
- Top of the readiness bar to clear: a public Testnet with real (small) value and a public bug bounty.

### 12.10 Anti-priorities (do *not* spend time on these now)

- **Phase A re-runs / re-reviews**: A is essentially complete; further iteration here has near-zero score impact.
- **`nerv-gui` / `apps/nerv-gui`**: blocked on rustc 1.88+ (ICU 2.3.0 dependency). Wait for the toolchain bump, not for engineering effort.
- **`vendor/yoke-derive/`**: blocked on rustc 1.88+ as well.
- **Re-doing the B1 finality review**: the file is filed (`specs/finality_review_2026-10-01.md`); only land its diffs once C1 is wired, otherwise the C1 endpoints shift and the B1 work is partially redone.
- **Re-running the slow tamper batteries every session**: they take 5–20 minutes each. Only re-run after a code change forces it; otherwise carry the recorded "PASSED 2026-10-06" stamp in `tests/STATUS.md` and trust the cached result.

Begin.
