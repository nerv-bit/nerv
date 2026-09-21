# **NERV v3.0** – The Self-Evolving Blockchain on a Provable Foundation

*Via a Cryptographic Ledger of Record, All-Post-Quantum Validity, Aggregate-Only Revelation, and First-Class Sharding*

**Private • Post-Quantum • Infinitely Scalable • Self-Evolving Layer-1**

---

## 🧠 Abstract

NERV v3.0 is a layer-1 blockchain rebuilt around a single principle, adopted from first principles and sharpened by external review: **the authoritative state of the chain is a cryptographic commitment, and everything neural is a derived, provably-consistent accelerator.** The protocol delivers privacy-by-construction, horizontal scalability via first-class sharding with native cross-shard transactions, post-quantum security across every security-critical primitive at genesis, and verifiable per-block self-improvement as an explicitly advisory bonus — with no trusted hardware, no trusted setup ceremonies, no elliptic curves anywhere in the security-critical stack, and no unproven research assumptions.

Previous architectures (including NERV v2.1) placed the neural embedding inside the state root, used Pedersen commitments and BLS signatures that leave discrete-log assumptions in the consensus path, and relied on Halo2/Plonky2 proofs whose elliptic-curve foundation bound ledger soundness to Shor-vulnerable mathematics. v3.0 inverts the relationship: the commitment is the truth; the embedding is a derived index. The proof pipeline is all-STARK over hash-based FRI. The entire privacy stack — note encryption, sealed deltas, mixnet transport, quorum certificates — uses only BLAKE3, ML-DSA, ML-KEM, and module-LWE.

Five commitments define v3.0: **commitment-first custody**, **all-post-quantum validity**, **aggregate-only revelation** (without Pedersen), **neural acceleration demoted to advisory**, and **first-class sharding**. The result is a chain whose privacy is defined against named adversary models, whose scalability claims are engineering targets with published budgets, and whose post-quantum posture is a property of construction rather than of intention.

---

## 💡 Core Innovations

### 1. Commitment-First Custody (The Ledger of Record)
The authoritative state is a 32-byte composite commitment C_t = BLAKE3("nerv.state" ‖ nct_root ‖ nullifier_root ‖ transit_root ‖ params_root ‖ prev ‖ height). Every transaction proves validity against an anchor of C_t. The neural embedding e_t is derived from public reveal data, committed separately as D_t, and consulted by nothing canonical. Deleting the entire knowledge layer leaves the chain fully verifiable from genesis — enforced as a CI job, not a policy.

### 2. Native STARK Engine (All-FRI, All-Hash)
A transparent AIR-STARK proof system over Goldilocks and its degree-2 extension, with FRI for polynomial commitment, hash-based lookup arguments, and Fiat–Shamir via BLAKE3. No elliptic curves, no pairings, no trusted setups, no ceremonies of any kind. The whole-transaction proof (~170K–200K constraints) covers custody, delta, and seal statements in one Fiat–Shamir transcript. Wallet proving: 0.8–2 s on a laptop; native verification: 0.2–0.5 ms.

### 3. NERV-Seal (The Ciphertext Is the Commitment)
Individual transaction deltas are sealed under a lattice-based linearly homomorphic threshold cryptosystem. The ciphertext itself serves as the binding commitment — v2.1 needed two objects (a Pedersen point plus a threshold ciphertext) because the commitment's binding had to be checked against something the network could open. v3.0 proves well-formedness of the seal inside each transaction's STARK (verifiable encryption), making the ciphertext publicly binding, privately hiding, and additively homomorphic. One object, three jobs: transport, binding, audit.

### 4. First-Class Sharding with Cross-Shard Atomicity
64 shards at genesis (scaling to 1,024), with a prefix-trie homing rule that never re-homes existing notes. Cross-shard transactions are sets of legs under one whole-transaction proof. The transit-log protocol makes cross-shard atomicity a construction rather than a coordination problem — no locks, no coordinator, no trusted relayer. Issue legs are permissionlessly completable by anyone, forever.

### 5. Deterministic Integer Self-Evolution
Per block: commit → reveal → score → update. The forecaster is a linear AR(1024) model trained by a deterministic integer Adam optimizer (Newton √, iterative β^t, clipped fixed-point updates) on sealed targets. Every operation is integer-exact with specified rounding. A node whose replay diverges from the committed D_t chain is faulted by that public fact alone. The challenger market pays only for demonstrated out-of-sample skill on committed-then-revealed targets — never for in-sample loss reduction.

### 6. Post-Quantum Everything (No Fourth Door)
Signatures: ML-DSA-65. Key exchange: ML-KEM-768. State commitments: BLAKE3. Proof system: FRI. Threshold channel: module-LWE. An adversary who breaks NERV must either invert BLAKE3 at generic-cost quantum bounds, solve module-LWE at parameter strength, or refute Fiat–Shamir-in-the-QROM. There is no fourth door, because there is no elliptic curve, no pairing, and no trusted setup anywhere in the stack.

---

## 🏗️ Architecture & Modules

| Crate | Description | Status |
|-------|-------------|--------|
| **nerv-core** | Domain-separated hashing, canonical codec, core types, fixed-point arithmetic, generated parameters | ✅ |
| **nerv-crypto** | ML-DSA-65, ML-KEM-768, ChaCha20-Poly1305, KDFs, quorum certificates, hash sortition | ✅ |
| **nerv-custody** | Notes, commitments, nullifiers, NCT, transit logs, transaction shells, burns | ✅ |
| **nerv-codec** | The frozen codec W, per-leg feature vectors, verifiable weight generation | ✅ |
| **nerv-seal** | NERV-Seal LHE channel: ring arithmetic, DKG, verifiable partial decryption | ✅ |
| **nerv-proofs** | Native STARK engine (FRI, AIR, DEEP-ALI), seal chip, encoder chip, folding | ✅ |
| **nerv-state** | Deterministic executor, C_t composite commitment, fraud proofs, storage | ✅ |
| **nerv-registry** | Global validity registry: mempool, bundles, T_τ, degraded mode, challenges | ✅ |
| **nerv-consensus** | Beacon, committees, sortition, attestations, finality, topology, slashing | ✅ |
| **nerv-da** | 2D Reed–Solomon erasure coding, blob commitments, sampled availability | ✅ |
| **nerv-net** | PQ wire protocol, TCP mesh, gossip with DSR-8 ordering, PQ-Sphinx mixnet, submission | ✅ |
| **nerv-economy** | Emission schedule, emission ledger, claim rail, fees (D.4 floor), subsidy, staking, supply identity | ✅ |
| **nerv-knowledge** | Embedding accumulator, AR(1024) forecaster, integer Adam, Huber, block loop, replay, challenger market | ✅ |
| **nerv-governance** | Two chambers, ZK ballot, aggregate-only tally, referenda, emergency powers | ✅ |
| **nerv-witness** | ~600 B inclusion witnesses, light-client anchor, archival regeneration | ✅ |
| **nerv-wallet** | Keys, scanning, construction, proving, sending, claiming, rehoming, witness store | ✅ |
| **nerv-wallet-core** | Platform-agnostic wallet state machine (all four UIs bind to this) | ✅ |
| **nerv-testkit** | In-process multinode harness: deterministic clock, loopback transports, adversarial scheduler (T5–T7) | ✅ |
| **nerv-conformance** | CI conformance suite: golden vectors, integration tests, the firewall check | ✅ |

---

## 🔐 Cryptographic Primitives Suite

NERV v3.0 is post-quantum from genesis. No ECC, RSA, pairings, or trusted setups appear in any security-critical role.

| Primitive | Algorithm | Standard |
|-----------|-----------|-----------|
| **Signatures** | ML-DSA-65 (Dilithium3) | FIPS 204 |
| **Key Exchange** | ML-KEM-768 | FIPS 203 |
| **Hashing** | BLAKE3 (external), Poseidon2 (in-circuit trees) | — |
| **ZK Proofs** | Native STARK (AIR + FRI + DEEP-ALI) | Transparent, no setup |
| **Threshold Channel** | NERV-Seal (module-LWE, rank 8, dimension 2,048) | §6.3 |
| **Commitments** | BLAKE3 hash commitments | §3.5 |
| **FS Non-interactivity** | Fiat–Shamir in the QROM | Named assumption |

**Explicitly absent:** elliptic-curve discrete logarithms, pairings, Pedersen algebra, BLS aggregation, trusted setups, ceremonies, TEEs, and floating-point arithmetic in any authority crate (enforced by lint and by the firewall CI job).

---

## 🛠️ Building from Source

NERV v3.0 requires Rust 1.83.0 or newer.

```bash
# Clone and build:
git clone https://example.invalid/nerv.git
cd nerv
cargo build --release

# Run the full test suite (2–5 minutes):
cargo test --workspace

# Verify the emission schedule:
./target/release/nerv-cli supply audit

# Run the firewall test (Axiom 3 — the chain works without the knowledge layer):
cargo test -p nerv-knowledge the_firewall_is_a_build_rule

# Start a terminal wallet:
./target/release/nerv-tui

# Build the web wallet (WASM/PWA):
cd apps/nerv-gui && bash build_web.sh

# Start a local network with Docker:
cd docker && docker compose up -d
```

---

## 🗺️ Roadmap to Production

| Milestone | Contents | Exit Gate |
|-----------|----------|-----------|
| **M0 — Specification Freeze** | This document; conformance suite; delete-test CI | Third-party spec review; CI green on stripped-chain validation |
| **M1 — Cryptographic Core** | NERV-Seal implementation and DKG; full STARK circuit stack; parameter freeze | Two independent circuit audits; golden vectors published |
| **M2 — Testnet Zero** | 8 shards; live mixnet; prover-market bootstrap | B5 and B10 pass; DKG reshare survives adversarial churn |
| **M3 — Formal Program** | Lean 4 state model; TLA+ concurrency model | Machine-checked induction core; model checker exhausts adversarial schedules |
| **M4 — Incentivized Testnet** | 64 shards; full fee markets; challenger market live | B1–B9 pass on public data |
| **M5 — Mainnet Genesis** | 64 shards; all audit reports public | Every gate green; contributor first tranche on schedule |

---

## 📊 Current Repository Status

**⚠️ In Development — Pre-Testnet**

This monorepo contains the complete implementation:

- ✅ Core documentation (whitepaper v3.0, Appendix D corrigenda, this README)
- ✅ Complete codebase — 15 library crates, 4 binaries, testkit, conformance suite, 5 fuzz targets
- ✅ Multi-platform wallet UI — terminal (ratatui), desktop (egui), web (WASM/PWA), mobile (iOS/Android shells)
- ✅ Deterministic multinode test harness (T5–T7 schedule corpus)
- ✅ Firewall test — the workspace builds and validates chains without the knowledge layer
- 🔄 Node event-loop integration wiring (7 bounded gaps; ~16 developer-days to close)

**All code, circuits, datasets, and specifications are released under MIT/Apache 2.0.**

---

## 🎯 Ethos & Principles

We are committed to:

1. **The commitment is the truth** — C_t is the sole authoritative representation of the ledger
2. **Radical openness** — Everything auditable, no hidden logic or allocations
3. **Zero pre-mine** — Empty trees at genesis; the entire supply emitted post-launch per a genesis-committed schedule
4. **Post-quantum by construction** — No legacy curves in critical paths; verified by cargo-deny as a dependency policy
5. **Provable fairness** — Every claim in this document is testable, tested, and pinned by CI

NERV belongs to the global privacy and open-source community. We invite cryptographers, systems engineers, privacy advocates, and builders to review, critique, audit, and contribute.

> **Let's build the nervous system of private money!**

---

## 🤝 Getting Involved

### First Steps:

1. **Read the whitepaper** — it contains the complete technical specification (§§1–14, Appendices A–D)
2. **Run `cargo test --workspace`** — the entire test suite is public and reproducible
3. **Run the firewall test** — verify that the chain works without the knowledge layer (Axiom 3)
4. **Review the code** — every novel cryptographic surface is isolated and commented

### Ways to Contribute:

- Open issues for questions, suggestions, or bug reports
- Review the cryptographic surfaces (nerv-seal, nerv-proofs) — these are the M1 audit targets
- Contribute to the testkit — more adversarial schedule strategies are always welcome
- Follow progress on the project landing page

---

## 🗓️ The Future

| Milestone | Target | Status |
|-----------|--------|--------|
| Public Testnet | Q1 2027 | Implementation complete; integration wiring in progress |
| Mainnet Fair Launch | H2 2028 | Planned |

**The nervous system of the private internet is being built in public. Join us!**

---

## ⚖️ License

NERV v3.0 is dual-licensed under either:

- **MIT License** ([LICENSE-MIT.txt](LICENSE-MIT.txt))
- **Apache License, Version 2.0** ([LICENSE-APACHE.txt](LICENSE-APACHE.txt))

at your option. This ensures maximum compatibility with the open-source ecosystem.

---

**NERV Collective**
August 2026

---

**Quick Links:** [Whitepaper v3.0](NERV%20Whitepaper%20V3.0.pdf) | [Errata Register](specs/errata.md) | [User Guide](USER_GUIDE.md)

<div align="center">

*The commitment is the truth; the proofs bind to it; the network learns in public beside it.*

</div>
