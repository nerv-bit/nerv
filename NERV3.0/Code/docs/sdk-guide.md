# NERV 3.0 Developer SDK Guide

**Audience:** wallet, indexer, and light-client developers  
**Status:** API guide for the current Rust workspace, reviewed 2026-10-07. The crate APIs are evolving; this repository does not yet provide a stable remote wallet/node RPC.

## 1. Architecture and trust boundaries

The Rust workspace separates transaction handling (`nerv-wallet`, `nerv-custody`, `nerv-proofs`, `nerv-net`), authoritative chain validation (`nerv-state`, `nerv-consensus`, `nerv-registry`), data availability (`nerv-da`), and the derived advisory knowledge model (`nerv-knowledge`). Keep those roles separate in client code:

- Use finalized headers, proofs, and the authoritative state commitment to decide whether a payment or block is valid.
- Treat `nerv-knowledge` forecasts, anomaly flags, and learned statistics as advisory. They are deliberately outside the authority stack and cannot authorize a payment or slash.
- Treat a local wallet's note/nullifier cache as local state. Reconcile it with authenticated chain data before declaring a transaction final.

Useful entry points are [`nerv-wallet`](../crates/nerv-wallet/src/lib.rs), [`nerv-da`](../crates/nerv-da/src/lib.rs), and [`nerv-knowledge`](../crates/nerv-knowledge/src/lib.rs). The Whitepaper's architecture and privacy scope are in §§2–3, 5, 8, and 10.

## 2. Wallet SDK: construct → prove → send

The native Rust wallet flow has three explicit operations:

1. **Construct.** `construct_payment` selects unspent notes, creates recipient/change notes, builds transaction legs and the public shell, and computes the sealed delta. The caller supplies the scanned wallet note set, diversified addresses and keys, active shard set, codec, current epoch public key, current height, and a cryptographically secure fresh-entropy callback.
2. **Prove.** `prove` canonicalizes the shell, derives public inputs/nullifiers, generates the witness, and runs the transaction STARK prover. Proof generation is compute-intensive; applications should run it off the UI thread and keep secrets/witnesses out of logs and telemetry.
3. **Send.** `send_transaction` fragments the proved transaction and submits through Sphinx paths to the configured aggregator fanout. `SendReport` describes relay/host send success; it is not a registry receipt, block inclusion proof, or finality certificate.

The convenience orchestration API is `run_send_pipeline` with a preassembled `SendPipeline`:

```rust
let mut entropy = nerv_wallet::construct::os_entropy;
let report = nerv_wallet::run_send_pipeline(
    &pipeline, // application supplies wallet, network, epoch, and proof context
    &mut entropy,
    |stage| tracing::info!(stage = stage.label(), "wallet send progress"),
)?;
tracing::info!(txid = ?report.txid, aggregators_reached = report.aggregators_reached,
               "submission sent; waiting for chain confirmation");
```

`pipeline` must contain a `PaymentSpec` (recipient, amount, fee, expiry), wallet notes and note positions, address set, `WalletKeys`, active shard topology, certified `CodecW`, epoch public key, frozen `FriShape`, current height, connected `Host`, relay registry, aggregator peers/fanout, and the wallet seed used for deterministic relay-path selection. Generate a fresh independent entropy draw for every construction/proof use; do not substitute fixed test seeds. The API definitions are in [`construct.rs`](../crates/nerv-wallet/src/construct.rs), [`prove.rs`](../crates/nerv-wallet/src/prove.rs), and [`send.rs`](../crates/nerv-wallet/src/send.rs).

The lower-level calls, in order, are:

```rust
let constructed = nerv_wallet::construct_payment(
    &spec, &notes, &addresses, &keys, &active, &codec, &epoch_pk,
    current_height, &mut entropy,
)?;
let proved = nerv_wallet::prove(&constructed, &codec, &epoch_pk, &fri)?;
let sent = nerv_wallet::send_transaction(
    &host, &proved, &send_config, &wallet_seed, &mut entropy,
)?;
```

These calls require caller-owned state and network setup; they are not a remote RPC. Errors should be surfaced by stage, and a sent transaction must still be tracked through verified inclusion and finality. Cross-shard payments may become two-leg transactions with conditional outputs and a later claim/reversion path; do not report them settled merely because submission returned successfully.

### CLI availability

`nerv-cli keys generate` can generate a wallet master seed and diversified addresses; protect its seed output. `nerv-cli supply audit`, `nerv-cli vectors emission-schedule`, and `nerv-cli witness regen` support local audits/tooling. The CLI's `tx build` and `tx prove` commands currently return an error stating that they require scanned-note/node-wallet-RPC state; they are not yet usable transaction commands. See [`bin/nerv-cli/src/main.rs`](../bin/nerv-cli/src/main.rs) and [`commands.rs`](../bin/nerv-cli/src/commands.rs).

## 3. Wallet scanning and indexing

### Scanning sealed outputs

For each output in authenticated block data, the wallet can trial-decrypt the sealed note against diversified addresses on the output's shard. `scan_sealed_note(addresses, sealed_note_bytes, published_commitment, shard)` returns a `ScannedNote` on successful decryption and commitment validation. Store the associated authenticated note-commitment-tree position, anchor, sibling path, shard, and received height using `WalletNoteSet::set_position`; update local spent tracking only from accepted transactions/nullifiers. APIs: [`scan.rs`](../crates/nerv-wallet/src/scan.rs), [`keys.rs`](../crates/nerv-wallet/src/keys.rs), and [`witness_store.rs`](../crates/nerv-wallet/src/witness_store.rs).

### What `nerv-knowledge` does (and does not do)

`nerv-knowledge` is **not** a canonical block indexer or database adapter. It does not scan a node's block store, discover wallet outputs, or verify consensus. It maintains the derived embedding/forecast state from ordered reveal/miss events. Its firewall is intentional: the crate depends on core/codec, and the authority stack does not depend on it.

An indexer should therefore implement this pipeline:

1. Obtain blocks/headers from a trusted local archival source or authenticated node interface. The current repository does not expose a general block-stream RPC.
2. Verify the beacon/shard QC, epoch/committee context, previous-state link, and applicable fraud/state-transition rules before treating a block as authoritative.
3. Index transaction shells, commitments, nullifiers, note positions, and inclusion witnesses from accepted blocks. Provide authenticated data to wallet scanners, and handle reorg/finality boundaries explicitly.
4. Separately derive `nerv-knowledge::BlockEvent::Reveal` or `Miss` only from verified reveal/interval inputs, then call `KnowledgeState::process`. Persist and replay this state as derived data. It must never override the canonical index or consensus verdict.

`KnowledgeState::commit` is calculated before a reveal; `reveal` rechecks that state commitment before applying the delta and updating its forecaster. `derived_root` is a derived commitment, not the authoritative state root. See [`block_loop.rs`](../crates/nerv-knowledge/src/block_loop.rs), [`embedding.rs`](../crates/nerv-knowledge/src/embedding.rs), and the crate-level firewall documentation.

## 4. Light client: DA sampling and header verification

The DA crate provides 2D Reed–Solomon blob encoding, Merkle-authenticated cells, deterministic sample-position derivation, and reconstruction from authenticated cells. A `CellAuth` verifies that a returned cell is at the requested blob/row/column and is included under the advertised commitment. `SampleRound::run` records verified and missing samples; `reconstruct_verified` reconstructs a blob from a sufficient authenticated set and checks its committed root.

Conceptual API flow:

```rust
let positions = nerv_da::sampling::sample_positions(&commitment, &beacon_seed, per_blob);
let round = nerv_da::sampling::SampleRound::run(
    &commitment,
    &beacon_seed,
    per_blob,
    |position| da_peer.fetch_authenticated_cell(position),
);
if !round.all_verified() {
    return Err("DA sampling incomplete or unauthenticated".into());
}
```

The sampling seed must come from the protocol's unpredictable, committed beacon context after the blob commitment is fixed; never let the serving peer choose it. A production client should retry diverse peers, track timeouts separately from invalid proofs, and avoid treating “all requested samples answered” as proof of finality.

DA cell verification is only one check. Before displaying a transaction as confirmed, a light client also needs to authenticate the header/QC against the correct epoch committee and trusted genesis/checkpoint, verify inclusion and state-transition evidence appropriate to the query, and wait for the protocol's finality rule. The current `nerv-da` crate is a data-availability library, not a complete light-client network service; no integrated header-light verifier or peer-fetching API is exposed here. See [`sampling.rs`](../crates/nerv-da/src/sampling.rs), [`blobs.rs`](../crates/nerv-da/src/blobs.rs), and Whitepaper §8.7.

## 5. Wallet user manual

This manual describes the desktop/web/mobile wallet surfaces present in the repository. Screen names and state transitions are shared across the GUI shells; exact account import, RPC connection, and production key-storage behavior depend on platform integration. The web wallet is a PWA shell around the shared GUI and uses a tap-to-unlock gate; do not equate that gate with hardware-backed authentication.

### Create and protect a wallet

1. Use the wallet's key-generation flow or the offline `nerv-cli keys generate` tool to create a wallet from OS-provided randomness.
2. Record the master seed using a private, offline backup method. Verify recovery on a separate test installation before placing assets at risk. Never send the seed to support staff or enter it on a website reached from a message.
3. Create diversified receiving addresses for the shards you use. An address is associated with a home shard; confirm the selected shard/network before sharing it.
4. Lock the wallet when leaving the device. The GUI state machine clears its in-memory seed and transient draft/history state on lock. Locking does not erase a seed backup or guarantee that the operating system has no copies in swap/crash dumps.

### Receive and sync

1. Open **Receive**, choose the network and address/shard, and copy/share the displayed address through a trusted channel.
2. Wait for the wallet to sync authenticated chain data and scan sealed outputs. A displayed balance depends on the wallet's local note and position state.
3. For large or cross-shard transfers, wait for the chain's required finality and any cross-shard claim/settlement step. A network “submitted” state is not settlement.

### Send

1. Open **Send**, choose/paste a recipient address, amount, and fee. Review the fee and expiry before proceeding.
2. Confirm that the spendable balance covers amount plus fee and that selected inputs can be spent together. The current constructor rejects inputs spanning multiple shards; it creates change on the input shard.
3. Review the final transaction summary. The wallet constructs and proves locally, then sends through the configured relay/aggregator path.
4. Keep the transaction ID. Track inclusion and finality independently; a successful relay send only means the wallet reached at least one aggregator path at the transport layer.
5. If a cross-shard output is conditional, follow the claim/reversion status until the receiving-shard condition is finalized. Do not retry blindly while the original transaction may still be accepted.

### Claim, history, and producer controls

- **Claim:** choose the available emission bucket/claim amount, review the claim window and fee, and submit only while the claim is valid. Claim credentials are secret. Verify the resulting chain inclusion; the CLI supply audit is not a claim transaction.
- **History:** use transaction IDs and authenticated block references to reconcile local status. Locally cached “spent” state is not, by itself, proof that the network finalized the spend.
- **Producer:** producer identity and staking controls shown by a UI are not proof of an on-chain registration. The current node's producer stake configuration is local wiring, and the registry submission/payout lifecycle is not a complete operator workflow.
- **Settings/help:** check the network and software version before signing. Never paste a recovery seed into diagnostics, screenshots, or a support ticket. Lock before handing the device to another person.

For shell details and known wallet limitations, see [`apps/nerv-gui/web/wallet/README.md`](../apps/nerv-gui/web/wallet/README.md) and [`nerv-wallet-core`](../crates/nerv-wallet-core/src/lib.rs). The GUI presents Dashboard, Send, Receive, Claim, Producer, History, Settings, and Help areas; a screen existing does not imply that every backend/network service is production-wired.

## 6. Further references

- Whitepaper: §§3.2–3.6, 5.4–5.5, 8.7, 10.3, 11, 12.3.
- `specs/errata.md`: items 185–188, 191, 197, and 205; consult errata for the frozen API/wire invariants.
- Implementation Design: crate boundaries and operator surfaces, pp. 3–7.
