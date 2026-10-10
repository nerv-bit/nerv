# NERV 3.0 Operator Runbook

**Audience:** node, validator, aggregator, and network operations teams  
**Status:** Testnet implementation guide, reviewed against the repository on 2026-10-07. This is not a declaration of production readiness.

## 1. Read this before deployment

The repository contains useful node, consensus, economy, and networking components, but several operator surfaces are incomplete. Treat the current binaries as development/testnet prototypes and do not place value or production keys at risk.

| Requested operation | Current repository behavior |
| --- | --- |
| Persistent full/archive node | `nerv-node` has an archival role, but `data_dir` is only parsed/logged; chain state uses `MemoryChain`. Durable block/state recovery is not wired. |
| Select validator shard | `--shard` is not a node CLI option. Role parsing defaults validators and producers to shard `(bits=6, value=0)`. A `shard` config entry is parsed but does not currently affect role selection. |
| Prometheus monitoring | `--metrics` is accepted but not consumed by the runtime. `--metrics-addr` is not a recognized flag, and no Prometheus endpoint is started. |
| Aggregator submission | The binary receives and deduplicates transaction frames, builds a bundle, and logs that bundle is ready. It does not currently verify each transaction at the aggregator or deliver the bundle to the registry. `--registry` is required but only included in the log. |
| Governance, staking, emission operations | These are Rust library/state-machine APIs, not complete operator services or a set of online administration commands. |

Relevant implementation: [`nerv-node` CLI](../bin/nerv-node/src/main.rs), [configuration](../bin/nerv-node/src/config.rs), [role dispatch](../bin/nerv-node/src/roles.rs), [node wiring](../bin/nerv-node/src/wiring.rs), [`nerv-aggregator`](../bin/nerv-aggregator/src/main.rs), and [`nerv-cli`](../bin/nerv-cli/src/main.rs).

## 2. Build and launch

Use binaries built from a reviewed, pinned revision. For a local development build, build the desired binary from the repository root with the project's approved Rust toolchain. The examples below use executable names after build/install; use `cargo run -p nerv-node -- ...` or `cargo run -p nerv-aggregator -- ...` for local development.

Before exposing a process to a network, confirm its actual flags with `nerv-node --help` or `nerv-aggregator --help`, run it in an isolated test environment, and restrict inbound access to the peer network. The current process does not provide the persistence, metrics, key-store, or fully wired registry behavior expected of a production service.

### 2.1 Full-history / archival role

The role intended to retain and serve full history is `archival`:

```sh
nerv-node --role archival --listen 0.0.0.0:7000 --data-dir /var/lib/nerv --log-level info
```

`--listen` is a socket address; `--peers` accepts comma-separated peer socket addresses. `--data-dir` currently only appears in configuration and startup logging. Do not assume it persists data or that restart resumes from the previous tip. The current implementation's chain source is in-memory.

For service supervision, run under a dedicated unprivileged account, use a read-only system image, grant write access only to the future data/log paths, restart on failure, and capture stdout/stderr with the host service manager. The checked-in [`deploy/nerv-node.service`](../deploy/nerv-node.service) is a starting point, not proof that archival persistence or shard selection is complete.

### 2.2 Validator role

The requested invocation `nerv-node --role validator --shard 7` is not supported by the current CLI and will be rejected because `--shard` is unknown. The current validator command is:

```sh
nerv-node --role validator --listen 0.0.0.0:7000 --peers 10.0.0.11:7000,10.0.0.12:7000 --log-level info
```

This runs the default validator role on shard `(6, 0)`; it does **not** run shard 7. The node currently generates fresh ML-DSA and ML-KEM keys from OS entropy at startup. There is no operator keystore or stable validator identity option in this CLI. Restarting therefore changes its identity. Do not count this as a production validator deployment.

The role dispatcher knows validator, producer, committee, archival, and light roles. A config file accepts selected settings such as `listen`, `data_dir`, `shard`, `producer_seed`, and `producer_stake_nano`, but shard config is not connected to role selection. See [`config.rs`](../bin/nerv-node/src/config.rs) and [`roles.rs`](../bin/nerv-node/src/roles.rs).

### 2.3 Aggregator role

The current binary requires a registry identifier and listens on port 7020 by default:

```sh
nerv-aggregator --listen 0.0.0.0:7020 --registry <registry-peer-id-hex> --bundle-size 512 --interval 1 --log-level info
```

Use this only for isolated development of ingress and bundle construction. The process creates fresh signing and KEM keys at every start. Its current event loop decodes transaction submissions, deduplicates by transaction ID, drains up to `--bundle-size` entries on each timer tick, builds a bundle, then logs its encoded size and configured registry ID. It does not currently verify the entries at ingress or send the bundle to the registry. A “bundle ready” log is not a successful registry submission or transaction inclusion receipt. The checked-in systemd template also omits the required `--registry` argument and will exit during CLI parsing until that is supplied. See [`bin/nerv-aggregator/src/main.rs`](../bin/nerv-aggregator/src/main.rs) and [`deploy/nerv-aggregator.service`](../deploy/nerv-aggregator.service).

## 3. Monitoring and incident response

The requested `--metrics-addr 0.0.0.0:9090` flag is not implemented. The node recognizes `--metrics`, but the parsed address is not used to bind an HTTP listener or publish Prometheus metrics. The aggregator has no metrics flag. Do not configure a Prometheus scrape target for either process until a real exporter and health contract have landed.

For current development runs, monitor the process externally:

```sh
systemctl status nerv-node
journalctl -u nerv-node -f
ss -lntp
```

Also monitor host CPU, memory, disk, file descriptors, network errors, process restarts, peer reachability, and test-chain height using an external supervisor or host monitoring agent. Log messages are diagnostic and are not a stable health API. If a node rejects a block or stops advancing, preserve logs, the exact binary/revision, genesis/config inputs, peer list, and the relevant block/header/evidence bytes; do not delete local state while investigating.

When a metrics endpoint is implemented, bind it to loopback or a dedicated monitoring interface by default. Exposing `0.0.0.0:9090` permits every reachable host to query it; enforce firewall rules, avoid secret/high-cardinality labels, and document whether it exposes peer addresses or transaction metadata.

## 4. Governance operations

`nerv-governance` is a library implementing referenda, ballots, chamber tallies, and emergency governance. There is no current `nerv-governance` daemon or operator command for proposing, voting, executing, or rolling back a live network change. Do not edit `specs/params.toml` on a running network and describe that as a governance action.

The specified governance model has two chambers: shielded note-holder participation and transparent validator-stake-weighted participation. During bootstrap, when the note-holder chamber is empty, the validator chamber carries parameter governance within constitutional constraints. Outside bootstrap, the prescribed chamber thresholds apply. Emergency powers are a separate restricted path. See Whitepaper §12.8 and `specs/errata.md` governance sections; implementation entry points are [`crates/nerv-governance`](../crates/nerv-governance/src/lib.rs), [`referenda.rs`](../crates/nerv-governance/src/referenda.rs), and [`emergency.rs`](../crates/nerv-governance/src/emergency.rs).

For any future live proposal, operators should publish the exact parameter diff, old/new spec digest, activation epoch, compatibility plan, test vectors, rollback conditions, and independent review before voting. Keep a signed record of the proposal and the finalized decision. The on-chain execution and parameter activation workflow must be verified before operators use this procedure on a real testnet.

## 5. Token generation, emission, claims, and supply checks

Distinguish three operations:

1. **Wallet-key generation** creates a secret used to control a wallet. It does not create NERV.
2. **Scheduled emission** is governed by the genesis-committed bucket schedule and ledger rules. It is not an operator “mint” command.
3. **Claiming** is a wallet action that proves control of the relevant claim key and an available emission commitment. The claim window, bucket, nullifier, and ledger bounds apply.

Useful offline tooling in the current CLI includes:

```sh
nerv-cli keys generate --coverage 8
nerv-cli supply audit
nerv-cli vectors emission-schedule --output ./emission-vectors.json
```

`keys generate` prints a 32-byte master seed to stdout (and can write it with `--output`); protect terminal history, logs, output files, backups, and crash dumps. Never use a seed pasted into a shell command on a shared host. `supply audit` checks the local schedule/supply audit path; it is not a transaction that issues funds. The vectors command emits conformance data, not spendable tokens.

The schedule is deterministic and validated from the network parameters. Emission credits are bounded per bucket and epoch; validator subsidies are split in canonical active-shard order, with the remainder assigned deterministically. Unclaimed-window burns and other burn categories are ledger events, not a discretionary operator balance adjustment. See Whitepaper §§12.2–12.6, `specs/errata.md` items 130–139, and [`crates/nerv-economy`](../crates/nerv-economy/src/lib.rs). The wallet crate includes claim construction, but the repository does not currently expose a complete wallet RPC or a live emission operator service.

## 6. Keys, rotation, staking, and penalties

### Key custody and rotation

The network design rotates seal-committee keys by epoch: incoming members establish fresh DKG material, pending reveals are handed off in a bounded window, and superseded share material is erased. This is distinct from rotating an operator's node identity key. See Whitepaper §§6.3.5–6.3.7, Appendix D.1/D.6, and [`crates/nerv-seal/src/epoch.rs`](../crates/nerv-seal/src/epoch.rs).

The current node CLI has no persistent validator keystore or command for planned identity-key rotation. Its node identity is regenerated on process startup; producer configuration can provide a seed, but this is not an encrypted key-management workflow. Until key-store support exists, use disposable development keys only. The production design should support offline recovery-tested backups, encrypted at-rest storage or an HSM, least-privilege process access, audited rotation, and revocation/committee re-registration. Never put seed material in service arguments, environment dumps, issue trackers, or logs.

### Staking and withdrawal

The economy library's `StakeLedger` is keyed by an ML-DSA verifying key. It supports deposits, delayed epoch-boundary withdrawals, and evidence-digest-deduplicated slashing. The current node/CLI does not provide an end-to-end on-chain staking, unstaking, or balance-query workflow. `producer_stake_nano` is local producer wiring and is not evidence that a live network has accepted or escrowed stake.

### Slashing exposure and downtime

The specified evidence-based slash table is:

| Provable offense | Stake fraction |
| --- | ---: |
| Double-sign | 100% |
| Invalid block | 100% |
| Invalid inclusion | 25% |
| Invalid bundle | 10% |

Slash consumes bonded stake first and pending withdrawals second; withdrawing does not evade an offense that predates the withdrawal boundary. The evidence verifier is in [`nerv-consensus/src/slash.rs`](../crates/nerv-consensus/src/slash.rs); the accounting table and ledger are in [`nerv-economy/src/staking.rs`](../crates/nerv-economy/src/staking.rs). Only verifiable evidence classes are slashed by this path.

No automatic downtime-penalty class or operator downtime enforcement path was found in the current code. Do not advertise a missed-block fine or assume that uptime earns guaranteed rewards. Downtime can still cost participation, liveness, availability, and expected rewards; monitor and restore service promptly. A protocol-defined penalty must be implemented and specified before operators rely on it.

## 7. Operational references

- Whitepaper: §§2.5–2.6, 6.3.5–6.4, 8.7, 12.2–12.8.
- Implementation Design: node/aggregator/DA/economy architecture, especially pp. 3–7.
- `specs/road_to_testnet.md`: Phase F1–F3 and current launch-readiness work.
- `specs/errata.md`: items 129–139, 163, 170, 185–197, and the governance sections.
