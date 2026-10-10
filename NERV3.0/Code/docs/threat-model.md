# NERV 3.0 Threat Model

**Audience:** protocol reviewers, implementers, operators, and testnet participants  
**Basis:** NERV Whitepaper V3.0, Implementation Design, `specs/errata.md`, and the current repository. Reviewed 2026-10-07.

## 1. Scope and claim discipline

This document describes defenses against five requested capabilities: a network adversary, malicious aggregator, malicious committee member, malicious light client, and malicious block producer. It distinguishes (a) protocol/design guarantees, (b) mechanisms present in code, and (c) end-to-end wiring that is incomplete or unverified. Presence of a library type or passing unit tests is not equivalent to a deployed, integrated defense.

NERV does not claim unconditional anonymity, perfect implementation security, or immunity to threshold collusion. The Whitepaper's threat model is explicitly property- and adversary-specific (§§2.5–2.6). Security-critical primitives are hash- or lattice-based, with the STARK Fiat–Shamir quantum-random-oracle-model assumption called out. The advisory seal has a lower stated security tier than the note layer; anomaly/forecast output is operationally advisory and is not an automatic slash trigger.

### Security properties in scope

- **Safety/correctness:** invalid state transitions, proofs, or quorum certificates must not become authoritative merely because an adversary proposes them.
- **Confidentiality:** note values, ownership, and counterparties are protected by the shielded note system; sealed delta information is protected under the seal's threshold assumptions.
- **Availability/liveness:** peers may withhold, delay, censor, or partition data and messages. Cryptography alone cannot force a peer to respond.
- **Accountability:** attributable, verifiable offense evidence can be validated and mapped to a slash class; unauthenticated suspicion is not proof.

## 2. Capability analysis

### 2.1 Network adversary

**Capability.** Observe, delay, reorder, replay, drop, selectively route, or partition packets; control a subset of relays/peers; correlate timing, size, and IP metadata; eclipse a wallet/light client; flood public ingress; or deny data requests.

**Designed defenses.** Wallet submissions use a five-relay Sphinx path and fan out to multiple aggregators. Relay packet layers use post-quantum ML-KEM-derived protection and authenticated forwarding; packet sizes/classes, jitter, and cover traffic reduce simple traffic correlation. Domain-separated hashes bind packet/session objects. Light clients sample authenticated DA cells from a commitment and should derive sample positions from beacon randomness committed after the blob is fixed. The DA layer uses 2D Reed–Solomon expansion and hash trees so returned cells can be checked without trusting the peer. Committee QCs and state-transition verification protect safety even when the gossip network is hostile. See Whitepaper §§2.3, 2.5, 6.4, 8.7 and `specs/errata.md` relay/DA rules.

**Code evidence and maturity.** Sphinx build/fragment/send APIs are present in [`crates/nerv-net/src/sphinx.rs`](../crates/nerv-net/src/sphinx.rs) and [`crates/nerv-wallet/src/send.rs`](../crates/nerv-wallet/src/send.rs); relay registration/path selection and a relay runtime exist in `nerv-net`. DA sampling and authenticated-cell verification are in [`crates/nerv-da/src/sampling.rs`](../crates/nerv-da/src/sampling.rs). However, the roadmap still records the end-to-end 5-relay submission path and DA adversarial-sampling behavior as needing integration validation. The current aggregator binary does not relay its built bundle to the registry. No current operator metrics endpoint reports relay failures.

**Residual risk.** A sufficiently broad observer can correlate ingress/egress timing, especially if it controls enough relays or watches endpoints. Traffic analysis, global passive observation, compromised endpoints, malicious OS/runtime, Sybil peers, eclipse attacks, and network partitions remain possible. Sphinx cannot prevent a relay from dropping traffic; fanout helps only when routes are independent and at least one aggregator path is honest/reachable. DA sampling is probabilistic and depends on unbiased, unpredictable sample selection and sufficient independent responses; withholding may cause a detectable availability failure, but a light client still needs retries and a trusted header/finality anchor. The adversary can cause denial of service without forging a proof.

**Operational response.** Use independent relays and aggregator operators, diverse network paths, rate limits, anti-eclipse peer policy, authenticated monitoring, and enough DA peers. Do not interpret anonymity guarantees as protection against a compromised wallet device or global traffic observer.

### 2.2 Malicious aggregator

**Capability.** Censor or delay submissions, reorder/drop entries, provide malformed bundles, misstate transaction counts, submit invalid proofs, replay data, or collude with a producer to favor some transactions.

**Designed defenses.** An aggregator is not a validity authority. Registry verification checks the bundle's transaction/proof data against the public STARK statement; the aggregator's signature/attestation binds the submitted bundle but does not make an invalid proof valid. Independent aggregators and multi-aggregator wallet fanout reduce single-operator censorship. Canonical transaction ordering limits order manipulation and reduces profitable reordering. Open challenge/inclusion evidence is intended to make omission detectable and slashable. Whitepaper §§2.5, 5.5, 11 and `specs/errata.md` items 98, 129, 188, and 197 describe the relevant verification and accountability surfaces.

**Code evidence and maturity.** Bundle construction and registry-side `verify_bundle`/per-entry verification are implemented in `nerv-registry`; the wallet send code supports a configurable aggregator list/fanout. The current `nerv-aggregator` executable, however, only decodes submissions, deduplicates, builds a bundle, and logs “bundle ready”; it does not currently perform proof verification or send the bundle to the registry. The current wallet report is a transport-level send report, not an inclusion receipt. The full omission/challenge/slash flow is not established by that binary alone. See [`bin/nerv-aggregator/src/main.rs`](../bin/nerv-aggregator/src/main.rs), [`crates/nerv-registry/src/bundle.rs`](../crates/nerv-registry/src/bundle.rs), and [`crates/nerv-wallet/src/send.rs`](../crates/nerv-wallet/src/send.rs).

**Residual risk.** An aggregator can always censor its own ingress or refuse to bundle. If wallet fanout collapses to one operator, the operator can suppress a transaction. An invalid-proof bundle can consume compute and delay users even when rejected. Collusion among aggregators and producers can shape availability or inclusion timing. Cryptographic verification cannot prove that an unobserved transaction was omitted; a censorship remedy requires a committed submission record, independent witnesses/challenges, available evidence, and an operationally live registry. Those integrated conditions remain a testnet readiness item.

**Operational response.** Operate multiple independent aggregator endpoints, monitor submission acknowledgments separately from finality, retain signed request IDs and transaction IDs, and retry through independent routes only after checking whether the original submission is pending. Do not promise a fee-for-inclusion service until the registry path is operational.

### 2.3 Malicious committee member

Committee roles have distinct consequences and should not be conflated.

#### Consensus committee

**Capability.** Refuse votes, equivocate, sign conflicting headers, collude to censor, or attempt to certify an invalid transition.

**Designed defenses.** Quorum certificates bind a subject to an epoch/committee context and require a threshold of ML-DSA signatures. With fewer than one-third Byzantine committee members, the Whitepaper claims BFT safety; invalid blocks are intended to be natively checkable by validators from public state updates. Same-slot conflicting valid QCs provide double-sign evidence; an invalid block, invalid inclusion, or invalid bundle has separate verifiable evidence types. Committee selection and randomness are epoch-bound and deterministically derived. See Whitepaper §§2.5, 4.6–4.7, 8.3, 11.2; `specs/errata.md` items 98, 103, 111, and 129; implementations in [`qc.rs`](../crates/nerv-consensus/src/qc.rs), [`committee.rs`](../crates/nerv-consensus/src/committee.rs), and [`slash.rs`](../crates/nerv-consensus/src/slash.rs).

**Residual risk.** At or above the BFT fault threshold, safety/liveness guarantees no longer hold. Below the threshold, a coalition may still censor or halt progress. Key theft, software bugs shared across validators, bad committee rosters, network partitions, weak randomness, or failure to connect evidence verification to the live stake ledger can defeat practical assumptions. The roadmap's finality review documents consensus risks that require closure before launch; do not infer all paths are deployed because the QC/evidence primitives exist.

#### Seal/decryption committee

**Capability.** Withhold partial decryptions, submit invalid shares, attempt to learn plaintexts, or collude to reconstruct a committee secret.

**Designed defenses.** The epoch key is produced through distributed key generation with verifiable commitments/proofs; partial decryptions carry public verification data. Fewer than the threshold `t` members cannot reconstruct the key or decrypt an aggregate. Epoch rotation establishes fresh key material; bounded handoff, backup delivery, reveal-count caps, and superseded-share erasure limit stale-share exposure. The code documents `t=7` for the current `n=10` seal committee and the handoff flow in [`crates/nerv-seal/src/epoch.rs`](../crates/nerv-seal/src/epoch.rs) and `dkg` modules. See Whitepaper §§6.3.5–6.3.7, Appendix D.1/D.6 and relevant seal errata.

**Residual risk.** A collusion of at least `t` current members can decrypt individual sealed aggregates; the Whitepaper acknowledges detection is primarily ex post through governance forensics, stake, and rotation. Withholding sufficient shares can delay a reveal or trigger a missed-reveal path. DKG transcript, ML-KEM delivery, handoff timing, share erasure, and governance-cap integration must all operate correctly. A vulnerable endpoint can expose a member's share without breaking the threshold mathematics.

**Economic exposure.** The code's evidence-based slash table assigns 100% to double-sign and invalid-block offenses, 25% to invalid-inclusion, and 10% to invalid-bundle evidence (`nerv-economy/src/staking.rs`; erratum 137). The consensus evidence verifier and economy ledger are separate layers. There is no generic “malicious committee member” or downtime slash without a matching, verified evidence class and a wired settlement path.

### 2.4 Malicious light client

**Capability.** Return false balances/confirmations to its own user, omit transactions, lie about locally stored keys/state, skip verification, choose biased DA samples, or present an untrusted checkpoint as canonical. A malicious serving peer may collude with it or exploit its implementation bugs.

**Designed defenses.** A correctly implemented light client checks cell authentication against the committed DA root, verifies headers/QCs against the correct epoch committee, validates inclusion paths and finality, and uses a post-commit unpredictable sample seed. The peer supplies bytes, not authority. Hash trees, ML-DSA committee signatures, and public inclusion/state evidence let the client reject fabricated responses. Whitepaper §§2.5, 8.7 and 11 describe DA sampling and compact inclusion witnesses; `nerv-da` implements authenticated cell proofs and reconstruction.

**Code evidence and maturity.** `nerv-da::CellAuth::verify`, `SampleRound`, and `reconstruct_verified` are present. The node has a `light` role that omits the executor, but the repository does not yet expose a complete light-client fetch/verify service or an integrated header-light verifier. DA cell authentication alone does not validate a block QC, state transition, or wallet balance. See [`roles.rs`](../bin/nerv-node/src/roles.rs), [`crates/nerv-da/src/sampling.rs`](../crates/nerv-da/src/sampling.rs), and the Phase C DA/light-client work in `specs/road_to_testnet.md`.

**Residual risk.** A malicious client can lie to its own user; consensus cannot force a dishonest client to run verification. An honest but buggy/eclipsed client may accept a stale chain view or fail to notice withheld data. A light client that trusts the first peer, a hard-coded committee key, a manipulated sample seed, or an unverified checkpoint inherits that peer's authority. Sampling is probabilistic; it does not provide full-node re-execution or prove global availability from a finite sample.

**Operational response.** Use a maintained client build, authenticate genesis/checkpoint data out of band, connect to diverse peers, verify committee context and finality, retry missing samples, and distinguish “data unavailable” from “invalid proof.” For high-value actions, independently verify with an archival/full node until the integrated light verifier is reviewed.

### 2.5 Malicious block producer

**Capability.** Propose an invalid state transition, omit/censor transactions, reorder transactions, withhold block data, publish malformed DA commitments, attempt a double-sign, or direct rewards to a chosen payout address.

**Designed defenses.** Producers cannot make an invalid transition authoritative by signing it: validators re-execute/check public state updates and require a valid QC. The executor's `apply_block` is the state authority, and public fraud/re-execution evidence is used for invalid-block challenges. Canonical ordering and fee/transaction-ID sorting restrict arbitrary reorder gains. DA commitments plus authenticated cells expose unavailable/malformed data; invalid-encoding evidence is separately defined. Same-slot conflicting QCs can identify equivocation. Evidence classes map to slash amounts in the staking ledger. See Whitepaper §§2.5, 4, 8.7, 11, and 12; `specs/errata.md` items 129, 137, and DA/fraud rules; code in [`crates/nerv-state/src/executor.rs`](../crates/nerv-state/src/executor.rs), [`crates/nerv-consensus/src/slash.rs`](../crates/nerv-consensus/src/slash.rs), and [`bin/nerv-node/src/producer.rs`](../bin/nerv-node/src/producer.rs).

**Residual risk.** A producer can censor or withhold data, delaying users or reducing liveness even if validators reject invalid blocks. A correlated validator/producer coalition, consensus-threshold compromise, bug in state validation, unavailable fraud evidence, or unintegrated slash settlement raises the risk. The node's current producer seed/stake wiring is local and does not provide a production keystore or complete registration/payout path. The repository does not implement an automatic downtime-penalty class; missed participation may reduce expected rewards or liveness but must not be described as a code-enforced fine.

**Operational response.** Validators should independently check proposals and reject missing or invalid DA/state data. Producers should publish complete data and preserve proposal/signature logs. Operators must never reuse a signing key across conflicting slots; isolate signing keys and halt signing if the node's chain view diverges. Capture both signed headers/QCs and committee context for equivocation evidence.

## 3. Cross-cutting assumptions and failure modes

### Cryptographic and implementation assumptions

- BLAKE3 collision/second-preimage resistance and domain separation protect commitments and identifiers.
- ML-DSA signatures and ML-KEM confidentiality/authentication rely on their selected lattice parameters and correct implementations.
- STARK soundness relies on the specified FRI parameters and Fiat–Shamir in the quantum random-oracle model. The proof system does not remove implementation, transcript-binding, or parameter-selection risk.
- Seal confidentiality depends on threshold assumptions and fresh/erased DKG shares; its stated security tier is lower than the custody note layer.
- Constant-time behavior, secret erasure, entropy quality, compiler behavior, memory safety across dependencies, and host/OS integrity require independent review.
- Privacy claims are layered. Metadata, endpoint compromise, user behavior, side channels, and global traffic analysis are not solved by note encryption.

### Economics and accountability

Evidence-based slashing does not itself guarantee honest behavior: deterrence depends on stake value, evidence publication, correct offender attribution, and settlement execution. A malicious or unavailable operator may evade consequences if evidence cannot be collected. Conversely, false or ambiguous evidence must not trigger a slash. Anomaly flags and forecasts are not proof and must never be wired directly to penalties. Current source search found no automatic downtime penalty class.

### Current integration caveats

The `nerv-aggregator` executable does not yet verify and submit bundles end to end; the node's metrics option does not launch an exporter; node data storage is not persistent; the requested shard CLI selection is not implemented; and the light-client/header verification path is incomplete. Phase C integration tests, Phase D CI/fuzzing, Phase E audits, and Phase F readiness work in `specs/road_to_testnet.md` therefore remain material to these guarantees. Do not treat the Whitepaper design as proof that each production path is already running.

## 4. Sources

- NERV Whitepaper V3.0: §§2.5–2.6, 3, 4, 5.4–5.7, 6.3.5–6.4, 8.7, 10.3–10.5, 11, 12.3, 12.8.
- NERV 3.0 Implementation Design: crate/role boundaries and risk/status notes, especially pp. 3–9.
- [`specs/errata.md`](../specs/errata.md): consensus/fraud/slash items 98, 103, 111, 129; relay item 129; DA rules; staking/emission items 130–139; wallet/operator items 185–197.
- [`specs/road_to_testnet.md`](../specs/road_to_testnet.md): Phases C–F, including end-to-end send, DA sampling, CI, security hardening, and operator/threat documents.
