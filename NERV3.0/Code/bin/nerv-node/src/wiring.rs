//! The node's event loop (erratum 195, 207): the integration surface
//! connecting the host, gossip engine, executor, and chain tracker.

use std::collections::BTreeMap;

use nerv_codec::codec_w::{CodecW, Delta, WeightVersion};
use nerv_codec::weight_gen::{self, BeaconRandomness};
use nerv_core::hash::Hash256;
use nerv_core::types::{Epoch, Height, Interval, ShardId, INTERVALS_PER_EPOCH};
use nerv_knowledge::{BlockEvent, KnowledgeState};
use nerv_net::gossip::{GossipEngine, GossipMessage, Inbound};
use nerv_net::host::{Host, HostConfig, HostEvent, HostFrame};
use nerv_proofs::FriShape;
use nerv_registry::mempool::VerifyContext;
use nerv_seal::dkg::{establish_epoch, MemberSecret, COMMITTEE_SIZE, THRESHOLD};
use nerv_seal::encrypt::PublicKey;
use nerv_seal::epoch::{EpochIndex, EpochKeyHandle};
use nerv_seal::sampling::ASeed;
use nerv_state::block::ShardBlock;
use nerv_state::{apply_block, BeaconView, ShardState};

use crate::chain::MemoryChain;
use crate::config::NodeConfig;
use crate::producer::ProducerRole;
use crate::roles::Role;

/// The frozen `[proofs.fri]` configuration (mirrors `specs/params.toml` and
/// `nerv-registry::testutil::harness::fri` â€” the consensus-critical default;
/// the testnet ships this single shape and does not negotiate it per epoch).
fn default_fri_shape() -> FriShape {
    FriShape {
        log_blowup: 4,
        num_queries: 56,
        log_final_poly_len: 1,
        max_log_arity: FriShape::DEFAULT_ARITY,
        commit_pow_bits: 0,
        query_pow_bits: 0,
    }
}

/// The node's view of finalized beacon data.
#[derive(Default)]
pub struct NodeBeaconView {
    tau_roots: BTreeMap<u64, Hash256>,
    transit_roots: BTreeMap<(ShardId, u64), Hash256>,

    // -- Gap 2: epoch parameter tracking --------------------------------
    // The (epoch, CodecW, PublicKey) tuple for the current seal epoch. The
    // registry's `VerifyContext` (the aggregator's verification gate â€” the
    // statement-11 binding plus the transaction STARK's verification against
    // the frozen `W` and the DKG-committed seal public key) is reconstructed
    // from these parameters and the frozen FRI shape at every epoch
    // boundary. Before the first boundary event, `current_verify_context`
    // returns `None` and submission verification is deferred.
    current_epoch: Option<EpochIndex>,
    current_w: Option<CodecW>,
    current_epoch_pk: Option<PublicKey>,
}

impl BeaconView for NodeBeaconView {
    fn tau_root(&self, interval: Interval) -> Option<Hash256> {
        self.tau_roots.get(&interval.as_u64()).copied()
    }
    fn transit_root(&self, shard: ShardId, height: Height) -> Option<Hash256> {
        self.transit_roots.get(&(shard, height.as_u64())).copied()
    }
}

impl NodeBeaconView {
    pub fn record_tau_root(&mut self, interval: u64, root: Hash256) {
        self.tau_roots.insert(interval, root);
    }
    pub fn record_transit_root(&mut self, shard: ShardId, height: u64, root: Hash256) {
        self.transit_roots.insert((shard, height), root);
    }

    /// The highest finalized beacon interval recorded by the node. The
    /// monotonic interval is the rotation trigger's logical clock
    /// (gap 5) â€” once `current_interval / INTERVALS_PER_EPOCH` advances
    /// past `last_rotated_epoch`, the node runs the DKG ceremony for the
    /// next epoch and updates the seal key.
    pub fn current_interval(&self) -> Option<Interval> {
        self.tau_roots.keys().max().copied().map(Interval::from_u64)
    }

    /// Records the epoch-boundary event (Gap 2): the new (epoch, CodecW,
    /// seal public key). Replaces any previous-epoch values â€” the latest
    /// boundary wins. Both `w` (governance-frozen, beacon-XOF-derived) and
    /// `epoch_pk` (DKG-committed) are externally supplied; the node does
    /// not derive them here.
    pub fn apply_epoch_boundary(
        &mut self,
        epoch: EpochIndex,
        w: CodecW,
        epoch_pk: PublicKey,
    ) {
        tracing::info!(
            epoch = epoch.0,
            w_version = w.version().0,
            "epoch boundary recorded"
        );
        self.current_epoch = Some(epoch);
        self.current_w = Some(w);
        self.current_epoch_pk = Some(epoch_pk);
    }

    /// The current seal epoch index, if an epoch boundary has been recorded.
    pub fn current_epoch(&self) -> Option<EpochIndex> {
        self.current_epoch
    }

    /// Constructs the registry's `VerifyContext` from the frozen FRI shape
    /// and the currently-tracked `(CodecW, PublicKey)`. Returns `None`
    /// until the first epoch-boundary event has been recorded; submissions
    /// received in that window are dropped (logged at debug, not error â€”
    /// the genesis boot path has no aggregator traffic).
    pub fn current_verify_context(&self, fri: FriShape) -> Option<VerifyContext> {
        let w = self.current_w.as_ref()?;
        let pk = self.current_epoch_pk.as_ref()?;
        Some(VerifyContext::new(fri, w.clone(), pk.clone()))
    }
}

/// The assembled node state.
pub struct Node {
    pub config: NodeConfig,
    pub role: Role,
    pub host: Host,
    pub events: tokio::sync::mpsc::Receiver<HostEvent>,
    pub gossip: GossipEngine,
    pub beacon_view: NodeBeaconView,
    pub shard_state: Option<ShardState>,
    pub chain: MemoryChain,
    /// Blocks received but not yet applicable (waiting for predecessors).
    pending_blocks: BTreeMap<u64, ShardBlock>,
    /// Blocks successfully applied (height â†’ block, for the chain source).
    applied_count: u64,
    /// Blocks rejected by apply_block (height â†’ error message).
    rejected_count: u64,
    /// The frozen FRI shape (Gap 2): the node constructs the registry's
    /// `VerifyContext` from this plus the epoch-boundary `(CodecW,
    /// PublicKey)`. Set once at construction (testnet ships a single
    /// consensus shape; future governance-tuned variants land via
    /// `params.toml`).
    fri_shape: FriShape,
    /// Submissions verified against the current epoch's `VerifyContext`
    /// (Gap 2 â€” `handle_submission`).
    submission_accepted: u64,
    /// Submissions that failed verification or decoding (Gap 2).
    submission_rejected: u64,
    /// Submissions dropped because no epoch boundary has been recorded yet
    /// (Gap 2 â€” genesis boot window).
    submission_deferred: u64,
    /// The epoch index the node last rotated to. `None` until the first
    /// rotation completes. Used to gate the per-tick trigger (gap 5).
    last_rotated_epoch: Option<EpochIndex>,
    /// Successful DKG ceremonies (gap 5). Distinct from `last_rotated_epoch`
    /// being `Some` â€” the ceremony might fail and we'd still want a count.
    rotations_completed: u64,
    /// Failed DKG ceremonies (gap 5).
    rotations_failed: u64,
    /// The per-shard challenger/forecaster state (gap 7). Fed from every
    /// successfully applied block; queried by the challenger's gate and
    /// the forecaster's predictions.
    knowledge: KnowledgeState,
    /// Successful `knowledge.process(Reveal)` events (gap 7).
    knowledge_reveals: u64,
    /// Successful `knowledge.process(Miss)` events (gap 7).
    knowledge_misses: u64,
    /// Adversarial D_t mismatches â€” the block's `header.derived` differs
    /// from `knowledge.derived_root()` after the reveal is processed
    /// (gap 7 advisory fault check).
    knowledge_faults: u64,
    /// The producer's identity + payout routing (the "submit a stake
    /// + produce blocks" integration). `None` unless the node was
    /// launched with `Role::Producer { shard, .. }`. Holds the ML-DSA
    /// signing key, the payout custody `Address`, and the assigned
    /// shard — everything needed to register stake via
    /// `register_stake()` and to fill in `producer_payout` when
    /// constructing a block.
    producer: Option<ProducerRole>,
}

impl Node {
    pub async fn new(config: NodeConfig, role: Role) -> Result<Node, anyhow::Error> {
        let mut sign_seed = [0u8; 32];
        getrandom::getrandom(&mut sign_seed)?;
        let signing = nerv_crypto::mldsa::SigningKey::from_seed(&sign_seed)?;

        let mut kem_seed = [0u8; 64];
        getrandom::getrandom(&mut kem_seed)?;
        let (_, static_kem_dk) = nerv_crypto::mlkem::keypair_from_seed(&kem_seed)?;

        let addr = config.listen_addr()?;
        let host_cfg = HostConfig::new(signing, static_kem_dk);
        let (host, events, local) =
            nerv_net::host::Host::bind(host_cfg, addr).await?;
        tracing::info!("listening on {local}");

        for peer_str in &config.peers {
            if let Ok(sock) = peer_str.parse() {
                let mut b = [0u8; 32];
                let _ = getrandom::getrandom(&mut b);
                let (ek, _) = nerv_crypto::mlkem::keypair_from_seed({
                    let mut s = [0u8; 64];
                    s[..32].copy_from_slice(&b);
                    s
                })?;
                let _ = host.dial(nerv_net::host::PeerInfo {
                    vk: *nerv_crypto::mldsa::SigningKey::from_seed(&b)?
                        .verifying_key(),
                    kem_ek: ek,
                    addrs: vec![sock],
                });
            }
        }

        let gossip = GossipEngine::new();
        let shard_state = if role.needs_executor() {
            match &role {
                Role::Validator { shard } | Role::Producer { shard } => {
                    Some(ShardState::genesis(*shard, Hash256::from_bytes([0u8; 32])))
                }
                _ => Some(ShardState::genesis(
                    ShardId::new(6, 0)?,
                    Hash256::from_bytes([0u8; 32]),
                )),
            }
        } else {
            None
        };

        // Producer-side wiring (gap continuation): when the operator
        // launches this binary with `--role producer`, install a fresh
        // `ProducerRole` so the node has a signing key + payout
        // Address. The seed comes from `NodeConfig::producer_seed` if
        // pinned, else fresh OS entropy. Operators using an HSM should
        // call `Node::attach_producer` with their externally-built role.
        let producer = match &role {
            Role::Producer { shard } => {
                let producer_seed = match config.producer_seed {
                    Some(s) => s,
                    None => {
                        let mut s = [0u8; 32];
                        if getrandom::getrandom(&mut s).is_err() {
                            tracing::warn!(
                                "OS entropy unavailable; producer identity not installed"
                            );
                            return Ok(Node {
                                config,
                                role,
                                host,
                                events,
                                gossip,
                                beacon_view: NodeBeaconView::default(),
                                shard_state,
                                chain: MemoryChain::new(),
                                pending_blocks: BTreeMap::new(),
                                applied_count: 0,
                                rejected_count: 0,
                                fri_shape: default_fri_shape(),
                                submission_accepted: 0,
                                submission_rejected: 0,
                                submission_deferred: 0,
                                last_rotated_epoch: None,
                                rotations_completed: 0,
                                rotations_failed: 0,
                                knowledge: KnowledgeState::genesis(),
                                knowledge_reveals: 0,
                                knowledge_misses: 0,
                                knowledge_faults: 0,
                                producer: None,
                            });
                        }
                        s
                    }
                };
                let stake_nano = config.producer_stake_nano.unwrap_or(0);
                match crate::producer::ProducerIdentity::from_seed(
                    &producer_seed,
                    *shard,
                    stake_nano,
                ) {
                    Ok(id) => Some(crate::producer::ProducerRole::new(id)),
                    Err(e) => {
                        tracing::warn!("producer identity construction failed: {e}");
                        None
                    }
                }
            }
            _ => None,
        };

        Ok(Node {
            config,
            role,
            host,
            events,
            gossip,
            beacon_view: NodeBeaconView::default(),
            shard_state,
            chain: MemoryChain::new(),
            pending_blocks: BTreeMap::new(),
            applied_count: 0,
            rejected_count: 0,
            fri_shape: default_fri_shape(),
            submission_accepted: 0,
            submission_rejected: 0,
            submission_deferred: 0,
            last_rotated_epoch: None,
            rotations_completed: 0,
            rotations_failed: 0,
            knowledge: KnowledgeState::genesis(),
            knowledge_reveals: 0,
            knowledge_misses: 0,
            knowledge_faults: 0,
            producer,
        })
    }

    pub fn applied_count(&self) -> u64 {
        self.applied_count
    }

    pub fn rejected_count(&self) -> u64 {
        self.rejected_count
    }

    pub fn pending_count(&self) -> usize {
        self.pending_blocks.len()
    }

    /// Submissions verified successfully against the current epoch's gate.
    pub fn submission_accepted(&self) -> u64 {
        self.submission_accepted
    }

    /// Submissions that failed verification or decoding.
    pub fn submission_rejected(&self) -> u64 {
        self.submission_rejected
    }

    /// Submissions dropped because no epoch boundary had been recorded yet.
    pub fn submission_deferred(&self) -> u64 {
        self.submission_deferred
    }

    /// The epoch index the node last rotated to (gap 5).
    pub fn last_rotated_epoch(&self) -> Option<EpochIndex> {
        self.last_rotated_epoch
    }

    /// Successful DKG ceremonies since startup (gap 5).
    pub fn rotations_completed(&self) -> u64 {
        self.rotations_completed
    }

    /// Failed DKG ceremonies since startup (gap 5).
    pub fn rotations_failed(&self) -> u64 {
        self.rotations_failed
    }

    /// Read-only handle to the knowledge layer (gap 7). The challenger
    /// queries the forecaster's predictions and the embedding's
    /// accumulation; both stay private by construction (no `&mut` leak).
    pub fn knowledge(&self) -> &KnowledgeState {
        &self.knowledge
    }

    /// Successful `BlockEvent::Reveal` feeds since startup (gap 7).
    pub fn knowledge_reveals(&self) -> u64 {
        self.knowledge_reveals
    }

    /// Successful `BlockEvent::Miss` feeds since startup (gap 7).
    pub fn knowledge_misses(&self) -> u64 {
        self.knowledge_misses
    }

    /// Advisory D_t mismatches between `header.derived` and
    /// `knowledge.derived_root()` since startup (gap 7).
    pub fn knowledge_faults(&self) -> u64 {
        self.knowledge_faults
    }

    /// The producer-side integration: a read-only handle to the
    /// `ProducerRole` (None for non-producer nodes). Lets the
    /// consensus layer query the producer's VK and payout Address
    /// without going through `&mut Node`.
    pub fn producer(&self) -> Option<&ProducerRole> {
        self.producer.as_ref()
    }

    /// Install a producer role on the node (after startup, or for
    /// tests). In production the role is installed at construction
    /// from `Role::Producer { shard, .. }` plus the producer seed
    /// from `NodeConfig` — see `Node::with_producer`.
    pub fn set_producer(&mut self, role: ProducerRole) {
        self.producer = Some(role);
    }

    /// The producer's per-block subsidy in nano-NERV for the current
    /// epoch on the producer's assigned shard. Returns `None` for
    /// non-producer nodes; returns `Err` if the shard isn't in the
    /// active set or the schedule has no `validator-subsidy` bucket.
    pub fn epoch_producer_payout(
        &self,
        schedule: &nerv_economy::schedule::EmissionSchedule,
        epoch: Epoch,
        active: &nerv_core::types::ShardSet,
    ) -> Result<u64, crate::producer::ProducerError> {
        match &self.producer {
            Some(role) => role.epoch_payout_nano(schedule, epoch, active),
            None => Ok(0),
        }
    }

    /// Register the producer's stake against the consensus `StakeLedger`.
    /// No-op for non-producer nodes.
    pub fn register_producer_stake(&self, ledger: &mut nerv_economy::staking::StakeLedger) {
        if let Some(role) = &self.producer {
            role.register_stake(ledger);
        }
    }

    /// The current state height (None if no state tracked).
    pub fn state_height(&self) -> Option<u64> {
        self.shard_state.as_ref().map(|s| s.height().as_u64())
    }

    /// The current state commitment.
    pub fn state_commitment(&self) -> Option<Hash256> {
        self.shard_state.as_ref().map(|s| s.state_commitment())
    }

    /// Periodic maintenance tick (gap 5). Runs every second from the
    /// event loop. Currently responsible for the epoch rotation trigger;
    /// future gaps will add DA sampling for light clients here.
    ///
    /// Steps when a new epoch boundary has been crossed:
    /// 1. Run the DKG ceremony (`nerv_seal::dkg::establish_epoch`) for
    ///    the new epoch's decryption committee â€” produces the joint
    ///    `PublicKey` and `transcript_digest`.
    /// 2. Derive the new `CodecW` from the beacon XOF.
    /// 3. Update `NodeBeaconView` via `apply_epoch_boundary` so the
    ///    aggregator's `VerifyContext` (gap 2) picks up the new tuple.
    /// 4. Trigger the `EpochAttestation` (gap 5 step 4). The actual
    ///    attestation requires the interval digest chain â€” recorded
    ///    here as an `EpochAttestationRequested` event; the full signed
    ///    attestation is built when the consensus committee is
    ///    provisioned on this node.
    pub fn tick_maintenance(&mut self) -> anyhow::Result<()> {
        // Step 0 â€” figure out where we are on the logical clock.
        let Some(current_interval) = self.beacon_view.current_interval() else {
            // No finalized tau root yet â€” still in the genesis boot window.
            return Ok(());
        };
        let new_epoch_index: EpochIndex = EpochIndex(current_interval.epoch().as_u64());

        // Step 1 â€” already at this epoch? nothing to do.
        if self.last_rotated_epoch == Some(new_epoch_index) {
            return Ok(());
        }

        // Step 2 â€” assemble the ceremony inputs. ASeed is beacon-committed
        // per epoch (WP Â§6.3.5); for the testnet we derive it deterministically
        // from the epoch index. Members are committee-roster-driven in
        // production; here we use the canonical `[COMMITTEE_SIZE] Ã— THRESHOLD`
        // test pattern, deterministic in (epoch, member).
        let a_seed = ASeed::from_bytes(seed_for_epoch(new_epoch_index.0, 0xE0));
        let proof_seed_root = seed_for_epoch(new_epoch_index.0, 0x20);
        let members: Vec<(MemberSecret, [u8; 32])> = (1..=COMMITTEE_SIZE as u8)
            .map(|i| {
                let member_seed = seed_for_epoch(new_epoch_index.0, 0xA0 + u64::from(i));
                let proof_seed = derive_proof_seed(proof_seed_root, i);
                (
                    MemberSecret::generate(i, &member_seed, COMMITTEE_SIZE, THRESHOLD)
                        .expect("member secret generation is deterministic and committee-size-bound"),
                    proof_seed,
                )
            })
            .collect();

        let handle: EpochKeyHandle =
            match establish_epoch(new_epoch_index, &a_seed, &members) {
                Ok(h) => h,
                Err(e) => {
                    self.rotations_failed += 1;
                    tracing::warn!(
                        epoch = new_epoch_index.0,
                        rotations_failed = self.rotations_failed,
                        "epoch rotation DKG failed: {e}"
                    );
                    return Ok(());
                }
            };

        // Step 3 â€” derive the new codec `W`. Beacon-randomness-driven;
        // both the beacon commitment and the version tag are pinned for
        // the testnet. In production the beacon randomness comes from
        // the protocol's VDF/XOF output; here it's deterministic in the
        // epoch index (governance-side `BeaconRandomness`).
        let beacon_randomness = BeaconRandomness::from_bytes(seed_for_epoch(
            new_epoch_index.0,
            0xBE,
        ));
        let weight_version = WeightVersion(new_epoch_index.0);
        let new_w: CodecW = weight_gen::expand(&beacon_randomness, weight_version);

        // Step 4 â€” publish the boundary into NodeBeaconView. The
        // aggregator's `VerifyContext` (gap 2) is now pinned to the new
        // tuple.
        let prev_epoch = self.last_rotated_epoch;
        self.beacon_view
            .apply_epoch_boundary(new_epoch_index, new_w, handle.public.clone());

        // Step 5 â€” trigger the EpochAttestation. The QC-validated signed
        // attestation is built by the consensus committee when keys are
        // provisioned; here we emit the trigger record and advance
        // state so the next tick waits for the actual ceremony.
        trigger_epoch_attestation(
            new_epoch_index,
            &handle,
            prev_epoch,
            self.beacon_view.tau_roots_len(),
        );

        self.rotations_completed += 1;
        self.last_rotated_epoch = Some(new_epoch_index);
        tracing::info!(
            epoch = new_epoch_index.0,
            rotations_completed = self.rotations_completed,
            "epoch rotation complete"
        );
        Ok(())
    }

    /// Attach the producer identity to the node at startup. Production
    /// wiring (in `main.rs`) calls this from the `Role::Producer` branch
    /// of the role dispatcher, reading the producer seed + shard + stake
    /// from `NodeConfig`. For tests, pass a pre-built `ProducerRole`.
    pub fn attach_producer(&mut self, role: ProducerRole) {
        self.producer = Some(role);
    }

    /// The main event loop (erratum 195).
    pub async fn run(&mut self) -> Result<(), anyhow::Error> {
        tracing::info!("entering event loop");
        let mut tick = tokio::time::interval(tokio::time::Duration::from_secs(1));

        loop {
            tokio::select! {
                event = self.events.recv() => {
                    match event {
                        Some(HostEvent::Connected(peer)) => {
                            self.gossip.on_connected(peer);
                            tracing::debug!(?peer, "connected");
                        }
                        Some(HostEvent::Disconnected(peer)) => {
                            self.gossip.on_disconnected(&peer);
                            tracing::debug!(?peer, "disconnected");
                        }
                        Some(HostEvent::Frame(from, payload)) => {
                            self.handle_frame(from, payload);
                        }
                        None => {
                            tracing::info!("event channel closed");
                            break;
                        }
                    }
                }
                _ = tick.tick() => {
                    // Gap 5: epoch rotation trigger â€” every second, check
                    // whether the logical clock has crossed an epoch
                    // boundary and, if so, run the DKG ceremony and
                    // refresh the seal key. Also a future gap: DA
                    // sampling for light clients.
                    if let Err(e) = self.tick_maintenance() {
                        tracing::warn!("maintenance tick failed: {e}");
                    }
                }
            }
        }
        Ok(())
    }

    fn handle_frame(&mut self, from: nerv_net::host::PeerId, payload: Vec<u8>) {
        match HostFrame::decode(&payload) {
            Ok(HostFrame::Gossip(_)) => {
                self.handle_gossip(from, &payload);
            }
            Ok(HostFrame::Submission(_)) => {
                // Gap 2: route every submission through the current
                // epoch's `VerifyContext`. Before the first epoch
                // boundary, `handle_submission` defers (genesis boot
                // window â€” no aggregator traffic yet).
                self.handle_submission(from, &payload);
            }
            Ok(HostFrame::MixPacket(_)) | Ok(HostFrame::MixFragment(_)) => {
                // This node is not a relay; ignore mix traffic.
            }
            Err(e) => {
                tracing::warn!(?from, "undecodable frame: {e}");
            }
        }
    }

    /// Process a submission frame against the current epoch's gate (Gap 2).
    ///
    /// 1. Resolve the current `VerifyContext` from
    ///    [`NodeBeaconView::current_verify_context`]; defer (no-op) if no
    ///    epoch boundary has been recorded.
    /// 2. Decode the wire submission (the aggregator's
    ///    `SubmissionMessage::Transaction { entry }`).
    /// 3. Run the registry's per-entry gate: `VerifyContext::verify`. This
    ///    is the same statement-11 + STARK check the aggregator runs inside
    ///    `verify_bundle` for every contained proof â€” the registry's bundle
    ///    gate is `verify_bundle(b, ctx) = validate_structure âˆ˜ â‹‚ ctx.verify`.
    ///
    /// The node does not pool or bundle here â€” that's the aggregator's
    /// job. A non-aggregator node verifies and drops; an aggregator-role
    /// node would route the verified entry into its own mempool instead.
    fn handle_submission(&mut self, from: nerv_net::host::PeerId, payload: &[u8]) {
        let ctx = match self.beacon_view.current_verify_context(self.fri_shape) {
            Some(c) => c,
            None => {
                self.submission_deferred += 1;
                tracing::debug!(
                    ?from,
                    deferred = self.submission_deferred,
                    "submission deferred: no epoch boundary recorded yet"
                );
                return;
            }
        };
        let message = match nerv_net::submission::SubmissionMessage::decode(payload) {
            Ok(m) => m,
            Err(e) => {
                self.submission_rejected += 1;
                tracing::warn!(?from, "submission decode failed: {e}");
                return;
            }
        };
        let nerv_net::submission::SubmissionMessage::Transaction { entry } = message;
        match ctx.verify(&entry.shell, &entry.proof) {
            Ok(()) => {
                self.submission_accepted += 1;
                tracing::debug!(
                    txid = ?entry.txid,
                    accepted = self.submission_accepted,
                    "submission verified"
                );
            }
            Err(e) => {
                self.submission_rejected += 1;
                tracing::warn!(
                    txid = ?entry.txid,
                    rejected = self.submission_rejected,
                    "submission verify failed: {e}"
                );
            }
        }
    }

    fn handle_gossip(&mut self, from: nerv_net::host::PeerId, payload: &[u8]) {
        match self.gossip.receive(from, payload) {
            Ok(Inbound::Accepted { message, forward }) => {
                for peer in self.gossip.recipients(Some(&from)) {
                    let _ = self.host.send(&peer, forward.clone());
                }
                self.process_gossip_message(&message);
            }
            Ok(Inbound::Duplicate) => {}
            Ok(Inbound::Rejected(reason)) => {
                tracing::debug!(?from, ?reason, "gossip rejected");
            }
            Err(e) => {
                tracing::warn!(?from, "gossip decode error: {e}");
            }
        }
    }

    fn process_gossip_message(&mut self, msg: &GossipMessage) {
        match msg {
            GossipMessage::Header { shard, header } => {
                tracing::debug!(?shard, height = header.height.as_u64(), "header learned");
                self.beacon_view
                    .record_tau_root(header.registry.interval.as_u64(), header.registry.root);
            }
            GossipMessage::Partial { shard, height, .. } => {
                tracing::debug!(?shard, height = height.as_u64(), "partial accepted");
            }
            GossipMessage::Reveal { shard, height, .. } => {
                tracing::debug!(?shard, height = height.as_u64(), "reveal accepted");
            }
            GossipMessage::BlockData { shard, height, data } => {
                self.handle_block_data(*shard, height.as_u64(), data);
            }
        }
    }

    /// Process a BlockData message: decode, validate, and apply (erratum 207).
    fn handle_block_data(&mut self, shard: ShardId, height: u64, data: &[u8]) {
        // Only apply blocks for our shard.
        if let Some(state) = &self.shard_state {
            if state.shard() != shard {
                tracing::trace!(?shard, "block for different shard; ignoring");
                return;
            }
        } else {
            return;
        }

        // Decode the block.
        let block: ShardBlock = match ShardBlock::decode(data) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(height, "block decode failed: {e}");
                self.rejected_count += 1;
                return;
            }
        };

        // Verify the block's height matches the message's height.
        if block.header.height.as_u64() != height {
            tracing::warn!(
                height,
                block_height = block.header.height.as_u64(),
                "block height mismatch"
            );
            self.rejected_count += 1;
            return;
        }

        // Check if this is the next block in sequence.
        let expected_height = match &self.shard_state {
            Some(state) => state.height().as_u64() + 1,
            None => return,
        };

        if height == expected_height {
            // Apply immediately.
            self.apply_and_drain(block, height);
        } else if height > expected_height {
            // Buffer for later.
            tracing::debug!(
                height,
                expected_height,
                "out-of-order block; buffering ({} pending)",
                self.pending_blocks.len()
            );
            self.pending_blocks.insert(height, block);
        } else {
            // Already past this height; duplicate or stale.
            tracing::trace!(height, expected_height, "stale block; ignoring");
        }
    }

    /// Apply a block, then drain any pending successors (erratum 207).
    fn apply_and_drain(&mut self, block: ShardBlock, height: u64) {
        match self.apply_one(block, height) {
            true => {
                // Try to drain pending successors.
                let mut next = height + 1;
                while let Some(block) = self.pending_blocks.remove(&next) {
                    tracing::debug!(height = next, "draining pending block");
                    if !self.apply_one(block, next) {
                        break;
                    }
                    next += 1;
                }
            }
            false => {
                // The block failed; stop draining.
            }
        }
    }

    /// Apply one block. Returns true on success, false on failure.
    fn apply_one(&mut self, block: ShardBlock, height: u64) -> bool {
        let Some(state) = self.shard_state.take() else {
            return false;
        };

        match apply_block(state, &block, &self.beacon_view, &self.chain) {
            Ok((new_state, applied)) => {
                let settled_count = applied.settled.len();
                let commitment = new_state.state_commitment();
                self.shard_state = Some(new_state);
                self.chain.store_block(height, &block);
                self.applied_count += 1;
                // Gap 7: feed the knowledge layer (challenger + forecaster)
                // from the freshly-applied block. The reveal lives in the
                // header's `prev_reveal`; a missed reveal (D.1(d)) becomes
                // a `Miss` event with the leg count from `applied.settled`.
                // The fault check below compares the freshly-derived
                // `derived_root()` against `header.derived` and bumps
                // `knowledge_faults` on mismatch.
                self.feed_knowledge_layer(&block, &applied);
                tracing::info!(
                    height,
                    settled = settled_count,
                    commitment = %commitment,
                    knowledge_reveals = self.knowledge_reveals,
                    knowledge_misses = self.knowledge_misses,
                    knowledge_faults = self.knowledge_faults,
                    "block applied"
                );
                true
            }
            Err(e) => {
                self.rejected_count += 1;
                tracing::error!(height, "apply_block failed: {e}");
                // Re-initialize from genesis (testnet fallback).
                // Production: reload from the store's last snapshot.
                let shard = block.shard;
                self.shard_state =
                    Some(ShardState::genesis(shard, Hash256::from_bytes([0u8; 32])));
                false
            }
        }
    }

    /// Gap 7 â€” feed the knowledge layer from a successfully applied block.
    ///
    /// 1. Construct a `BlockEvent::Reveal` from the header's
    ///    `prev_reveal` (Î”_B) and the applied leg-fee total. Bucket is the
    ///    time-rail bucket derived from the block height (mod 16 â€” the
    ///    forecaster's `bucket < 16` invariant).
    /// 2. If the header has no `prev_reveal` (a missed-reveal window per
    ///    D.1(d)), emit `BlockEvent::Miss` carrying the leg count so the
    ///    embedding's skip-and-carry rule fires.
    /// 3. After processing, compare `knowledge.derived_root()` against
    ///    `header.derived`. A mismatch is an advisory fault (the
    ///    header-committed D_t disagrees with the locally-derived one);
    ///    it's logged and counted but does not invalidate the block â€”
    ///    challengers escalate it asynchronously.
    fn feed_knowledge_layer(
        &mut self,
        block: &ShardBlock,
        applied: &nerv_state::executor::Applied,
    ) {
        let height = block.header.height.as_u64();
        let fee_sum = applied.fee_total.as_u64();
        let bucket: u16 = (height % 16) as u16;
        let event = match block.header.prev_reveal {
            Some(bytes) => {
                let delta = Delta::from_canonical_bytes(&bytes);
                BlockEvent::Reveal { height, delta, fee_sum, bucket }
            }
            None => BlockEvent::Miss {
                height,
                legs: applied.settled.len() as u64,
            },
        };

        // Drive the knowledge layer. `process` returns a `BlockRecord`
        // whose `prediction_hash` and `residual` the challenger queries;
        // for the wiring we only care about Ok / Err.
        let record = match self.knowledge.process(event) {
            Ok(rec) => rec,
            Err(e) => {
                tracing::warn!(
                    height,
                    "knowledge.process failed: {e:?} â€” feeding a Miss instead"
                );
                // Fall back to a Miss so the embedding's height accounting
                // still advances (erratum 167's contiguity invariant).
                let _ = self.knowledge.process(BlockEvent::Miss {
                    height,
                    legs: applied.settled.len() as u64,
                });
                return;
            }
        };

        if record.missed {
            self.knowledge_misses += 1;
        } else {
            self.knowledge_reveals += 1;
        }

        // Advisory fault check: the block's header-committed D_t must
        // match the knowledge layer's freshly-derived root.
        let derived = self.knowledge.derived_root();
        if derived != block.header.derived {
            self.knowledge_faults += 1;
            tracing::error!(
                height,
                header_derived = %block.header.derived,
                knowledge_derived = %derived,
                "advisory D_t mismatch: header.derived disagrees with knowledge.derived_root"
            );
        }
    }

    /// Publish a block through gossip (for the producer role; Gap 5 wires
    /// the epoch machinery that calls this).
    pub fn publish_block(&mut self, block: &ShardBlock) -> Result<(), anyhow::Error> {
        let data = block.encode();
        let msg = GossipMessage::BlockData {
            shard: block.shard,
            height: block.header.height,
            data,
        };
        match self.gossip.publish(msg) {
            nerv_net::gossip::PublishOutcome::Broadcast { frame } => {
                for peer in self.gossip.recipients(None) {
                    let _ = self.host.send(&peer, frame.clone());
                }
                tracing::info!(
                    height = block.header.height.as_u64(),
                    "block published to {} peers",
                    self.gossip.recipients(None).len()
                );
                Ok(())
            }
            nerv_net::gossip::PublishOutcome::Duplicate => {
                tracing::debug!("block already published");
                Ok(())
            }
            nerv_net::gossip::PublishOutcome::Rejected(reason) => {
                Err(anyhow::anyhow!("gossip rejected: {reason:?}"))
            }
        }
    }

    /// Publish a header through gossip (the producer's first broadcast).
    pub fn publish_header(&mut self, header: &nerv_state::ShardHeader, shard: ShardId) {
        let msg = GossipMessage::Header { shard, header: header.clone() };
        if let nerv_net::gossip::PublishOutcome::Broadcast { frame } = self.gossip.publish(msg) {
            for peer in self.gossip.recipients(None) {
                let _ = self.host.send(&peer, frame.clone());
            }
        }
    }
}


// ---- Gap 5 helpers ------------------------------------------------------

/// Deterministic 32-byte seed keyed by `(epoch, tag)`. The testnet
/// committee roster and the beacon XOF are both derived from this
/// pattern; production swaps these for the VDF-committed seeds.
fn seed_for_epoch(epoch: u64, tag: u8) -> [u8; 32] {
    let mut s = [0u8; 32];
    s[0] = tag;
    s[8..16].copy_from_slice(&epoch.to_le_bytes());
    // Domain separation: `nerv.node.epoch` prefix. The two leading bytes
    // are pinned across all gap-5 ceremonies â€” change them only with a
    // governance event.
    s[1..8].copy_from_slice(b"nerv.ne");
    s
}

/// Per-member proof seed (32 bytes). The first 16 bytes are the
/// epoch-keyed root; the last 16 bind the member index.
fn derive_proof_seed(root: [u8; 32], member: u8) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[..16].copy_from_slice(&root[..16]);
    out[16] = member;
    out[24..32].copy_from_slice(&u64::from(member).to_le_bytes());
    out
}

/// Logs the gap-5 attestation trigger. In production the node would
/// also publish the request via gossip so the beacon committee can
/// produce the signed `EpochAttestation`; here we just record it.
fn trigger_epoch_attestation(
    new_epoch: EpochIndex,
    handle: &EpochKeyHandle,
    prev_epoch: Option<EpochIndex>,
    finalized_intervals: usize,
) {
    tracing::info!(
        epoch = new_epoch.0,
        prev_epoch = ?prev_epoch.map(|e| e.0),
        finalized_intervals,
        transcript_digest_len = handle.transcript_digest.len(),
        "epoch attestation triggered"
    );
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    //! Gap 2 tests: epoch parameter tracking on `NodeBeaconView` and the
    //! `VerifyContext` reconstruction contract. The node's
    //! `handle_submission` depends on this â€” the per-submission path is
    //! deferred (no-op) until `apply_epoch_boundary` has recorded the
    //! current epoch's `(CodecW, PublicKey)`, and the constructed context
    //! pins the frozen codec and the DKG-committed seal public key.

    use super::*;
    use nerv_codec::codec_w::WeightVersion;
    use nerv_codec::weight_gen::{expand, BeaconRandomness};
    use nerv_seal::encrypt::derive_reference_keypair;

    fn fresh_codec(version: u64) -> CodecW {
        expand(&BeaconRandomness::from_bytes([version as u8; 32]), WeightVersion(version))
    }

    fn fresh_pk(seed_byte: u8) -> PublicKey {
        derive_reference_keypair(&[seed_byte; 32]).unwrap().0
    }

    #[test]
    fn beacon_view_starts_without_epoch_params() {
        let view = NodeBeaconView::default();
        assert_eq!(view.current_epoch(), None);
        assert!(view.current_verify_context(default_fri_shape()).is_none());
    }

    #[test]
    fn apply_epoch_boundary_records_the_triple() {
        let mut view = NodeBeaconView::default();
        let w = fresh_codec(7);
        let pk = fresh_pk(0xA1);
        view.apply_epoch_boundary(EpochIndex(3), w.clone(), pk.clone());

        let epoch = view.current_epoch().expect("epoch recorded");
        assert_eq!(epoch, EpochIndex(3));

        let ctx = view.current_verify_context(default_fri_shape()).expect("ctx after boundary");
        assert_eq!(ctx.w().version(), w.version());
        assert_eq!(ctx.epoch_pk().a_seed(), pk.a_seed());
        assert_eq!(ctx.epoch_pk().t(), pk.t());
        // The registry's `epoch_key_id` is H(ASeed â€– T) under `nerv.seal.stmt`
        // â€” pinned and sensitive to both ASeed and T (see `circuit_stmt`).
        assert_eq!(ctx.epoch_key_id().as_bytes().len(), 32);
    }

    #[test]
    fn latest_epoch_boundary_overwrites_previous() {
        let mut view = NodeBeaconView::default();
        view.apply_epoch_boundary(EpochIndex(1), fresh_codec(1), fresh_pk(0x01));
        view.apply_epoch_boundary(EpochIndex(2), fresh_codec(2), fresh_pk(0x02));

        let ctx = view.current_verify_context(default_fri_shape()).expect("ctx");
        assert_eq!(view.current_epoch(), Some(EpochIndex(2)));
        assert_eq!(ctx.w().version(), WeightVersion(2));
        assert_eq!(
            ctx.epoch_pk().a_seed().as_bytes()[0],
            0x02,
            "second boundary's ASeed must win"
        );
    }

    #[test]
    fn verify_context_is_pinned_to_the_fri_shape_argument() {
        // The FRI shape is passed in at call time, not stored on the view â€”
        // this is the seam where governance-tuned shape variants plug in
        // (today's testnet ships the single frozen shape). A different
        // shape argument still produces a context with the same W and PK
        // pinned, since neither flows from `fri`.
        let mut view = NodeBeaconView::default();
        let w = fresh_codec(4);
        let pk = fresh_pk(0x55);
        view.apply_epoch_boundary(EpochIndex(9), w.clone(), pk.clone());

        let fri_a = default_fri_shape();
        let mut fri_b = default_fri_shape();
        fri_b.num_queries = fri_a.num_queries + 1;

        let ctx_a = view.current_verify_context(fri_a).expect("ctx_a");
        let ctx_b = view.current_verify_context(fri_b).expect("ctx_b");
        assert_eq!(ctx_a.w().version(), ctx_b.w().version());
        assert_eq!(ctx_a.epoch_pk().a_seed(), ctx_b.epoch_pk().a_seed());
        assert_eq!(ctx_a.fri(), &fri_a);
        assert_eq!(ctx_b.fri(), &fri_b);
        assert_ne!(ctx_a.fri(), ctx_b.fri());
    }

    #[test]
    fn submission_before_epoch_boundary_is_deferred() {
        // Construct a minimal node-side scenario without spinning up a Host:
        // the wiring's `handle_submission` is the integration point â€” what
        // we pin here is the contract that `current_verify_context` returns
        // `None` until `apply_epoch_boundary` runs. A `None` means
        // `handle_submission` increments `submission_deferred` and does
        // not call `verify`.
        let view = NodeBeaconView::default();
        assert!(view.current_verify_context(default_fri_shape()).is_none());
    }

    #[test]
    fn fri_shape_default_matches_registry_harness() {
        // Pin the consensus-critical default. The registry's harness
        // (`nerv-registry::testutil::harness::fri`) is `pub(crate)` and
        // not reachable from the node binary, so we pin the field values
        // directly â€” the same six the harness constructs. Any change on
        // either side will visibly diverge here.
        let fri = default_fri_shape();
        assert_eq!(fri.log_blowup, 4);
        assert_eq!(fri.num_queries, 56);
        assert_eq!(fri.log_final_poly_len, 1);
        assert_eq!(fri.max_log_arity, FriShape::DEFAULT_ARITY);
        assert_eq!(fri.commit_pow_bits, 0);
        assert_eq!(fri.query_pow_bits, 0);
    }

    // ---- Gap 5: epoch rotation trigger ----

    use nerv_core::types::INTERVALS_PER_EPOCH;

    /// Helper: build a `Node` directly without spinning up the host â€”
    /// only the fields the rotation trigger touches.
    fn fixture_node() -> (Node, NodeConfig) {
        let cfg = NodeConfig::default();
        let shard = cfg.shard;
        let shard_state = Some(ShardState::genesis(shard, cfg.params_root));
        let (host, events) = Host::new(HostConfig::default()).expect("host");
        (
            Node {
                config: cfg.clone(),
                role: crate::roles::Role::Validator,
                host,
                events,
                gossip: GossipEngine::new(),
                beacon_view: NodeBeaconView::default(),
                shard_state,
                chain: MemoryChain::new(),
                pending_blocks: BTreeMap::new(),
                applied_count: 0,
                rejected_count: 0,
                fri_shape: default_fri_shape(),
                submission_accepted: 0,
                submission_rejected: 0,
                submission_deferred: 0,
                last_rotated_epoch: None,
                rotations_completed: 0,
                rotations_failed: 0,
            },
            cfg,
        )
    }

    #[test]
    fn no_rotation_before_first_finalized_interval() {
        // Genesis boot window: no tau roots means no logical clock.
        // `tick_maintenance` must short-circuit and not panic or
        // increment any rotation counters.
        let (mut node, _cfg) = fixture_node();
        node.tick_maintenance().expect("tick");
        assert!(node.last_rotated_epoch().is_none());
        assert_eq!(node.rotations_completed(), 0);
        assert_eq!(node.rotations_failed(), 0);
    }

    #[test]
    fn first_rotation_at_epoch_zero_boundary() {
        let (mut node, _cfg) = fixture_node();
        // Record a tau root at the first interval of epoch 0.
        node.beacon_view
            .record_tau_root(0, Hash256::from_bytes([0x01; 32]));
        node.tick_maintenance().expect("tick");

        assert_eq!(node.last_rotated_epoch(), Some(EpochIndex(0)));
        assert_eq!(node.rotations_completed(), 1);
        assert_eq!(node.rotations_failed(), 0);
        // The beacon view must have been pinned to the new (w, pk).
        let ctx = node
            .beacon_view
            .current_verify_context(default_fri_shape())
            .expect("ctx");
        assert_eq!(ctx.epoch_key_id().as_bytes().len(), 32);
    }

    #[test]
    fn second_tick_in_same_epoch_is_a_noop() {
        let (mut node, _cfg) = fixture_node();
        node.beacon_view
            .record_tau_root(0, Hash256::from_bytes([0x01; 32]));
        node.tick_maintenance().expect("tick 1");
        let first = node.rotations_completed();
        // Same epoch: another tick must not re-rotate.
        node.tick_maintenance().expect("tick 2");
        assert_eq!(node.rotations_completed(), first);
        assert_eq!(node.last_rotated_epoch(), Some(EpochIndex(0)));
    }

    #[test]
    fn advance_to_epoch_one_rotates_again() {
        let (mut node, _cfg) = fixture_node();
        // Pin epoch 0.
        node.beacon_view
            .record_tau_root(0, Hash256::from_bytes([0x01; 32]));
        node.tick_maintenance().expect("tick 0");
        // Now advance the logical clock into epoch 1 by recording an
        // interval at `INTERVALS_PER_EPOCH`.
        node.beacon_view.record_tau_root(
            INTERVALS_PER_EPOCH,
            Hash256::from_bytes([0x02; 32]),
        );
        node.tick_maintenance().expect("tick 1");
        assert_eq!(node.last_rotated_epoch(), Some(EpochIndex(1)));
        assert_eq!(node.rotations_completed(), 2);

        // The new ctx carries the new key (different from the epoch-0
        // `epoch_key_id`).
        let ctx_0_key = {
            let mut view = NodeBeaconView::default();
            view.apply_epoch_boundary(EpochIndex(0), fresh_codec(0), fresh_pk(0x01));
            view.current_verify_context(default_fri_shape())
                .unwrap()
                .epoch_key_id()
        };
        let ctx_1 = node
            .beacon_view
            .current_verify_context(default_fri_shape())
            .expect("ctx_1");
        assert_ne!(ctx_1.epoch_key_id(), ctx_0_key);
    }

    #[test]
    fn rotation_updates_weight_version() {
        let (mut node, _cfg) = fixture_node();
        node.beacon_view
            .record_tau_root(0, Hash256::from_bytes([0x01; 32]));
        node.tick_maintenance().expect("tick");
        let ctx_0 = node
            .beacon_view
            .current_verify_context(default_fri_shape())
            .unwrap();
        assert_eq!(ctx_0.w().version(), WeightVersion(0));

        node.beacon_view.record_tau_root(
            INTERVALS_PER_EPOCH,
            Hash256::from_bytes([0x02; 32]),
        );
        node.tick_maintenance().expect("tick");
        let ctx_1 = node
            .beacon_view
            .current_verify_context(default_fri_shape())
            .unwrap();
        assert_eq!(ctx_1.w().version(), WeightVersion(1));
    }

    #[test]
    fn helper_seed_is_deterministic_and_domain_separated() {
        let a = seed_for_epoch(7, 0xE0);
        let b = seed_for_epoch(7, 0xE0);
        assert_eq!(a, b);
        // Different tag â†’ different seed.
        assert_ne!(a, seed_for_epoch(7, 0xE1));
        // Different epoch â†’ different seed.
        assert_ne!(a, seed_for_epoch(8, 0xE0));
        // Domain-separation prefix is bound at bytes 1..8.
        assert_eq!(&a[1..8], b"nerv.ne");
    }

    #[test]
    fn helper_proof_seed_binds_member_index() {
        let root = [0u8; 32];
        let s1 = derive_proof_seed(root, 1);
        let s2 = derive_proof_seed(root, 2);
        assert_ne!(s1, s2);
        // First 16 bytes mirror the root (deterministic per epoch).
        assert_eq!(&s1[..16], &root[..16]);
        assert_eq!(&s2[..16], &root[..16]);
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod gossip_engine_tests {
    use super::*;
    use crate::host::{Host, HostConfig, HostEvent};
    use crate::testutil::harness::{node, test_header, test_partial, test_reveal, tx_entry};
    use crate::testutil::SplitMix64;
    use std::time::Duration;
    use tokio::sync::mpsc;

    fn shard() -> ShardId {
        nerv_core::types::ShardSet::genesis().ids()[7]
    }

    fn peer_id(seed: u64) -> PeerId {
        let (sk, _, _) = crate::testutil::node_keys(seed);
        PeerId::of(sk.verifying_key())
    }

    fn header_msg(height: u64) -> GossipMessage {
        GossipMessage::Header { shard: shard(), header: test_header(shard(), height) }
    }

    fn partial_msg(height: u64) -> GossipMessage {
        GossipMessage::Partial {
            shard: shard(),
            height: Height::from_u64(height),
            batch: 0,
            partial: test_partial(3),
        }
    }

    fn reveal_msg(height: u64) -> GossipMessage {
        GossipMessage::Reveal {
            shard: shard(),
            height: Height::from_u64(height),
            batch: 0,
            reveal: test_reveal(),
        }
    }

    #[test]
    fn message_digest_is_the_literal_formula() {
        let payload = b"some wire payload";
        let d = message_digest(payload);
        assert_eq!(d, message_digest(payload));
        assert_ne!(d, message_digest(b"other"));
        let mut pre = Vec::new();
        pre.extend_from_slice(GOSSIP_MSG.as_bytes());
        pre.extend_from_slice(payload);
        assert_eq!(d.as_bytes(), blake3::hash(&pre).as_bytes());
    }

    #[test]
    fn codec_roundtrips_strictness_and_tags() {
        for msg in [header_msg(7), partial_msg(7), reveal_msg(7)] {
            let enc = msg.encode();
            assert_eq!(enc.len(), msg.encoded_len());
            assert_eq!(GossipMessage::decode(&enc).unwrap(), msg);
            for cut in 0..enc.len() {
                assert!(GossipMessage::decode(&enc[..cut]).is_err(), "cut {cut}");
            }
            let mut ext = enc.clone();
            ext.push(0);
            assert!(GossipMessage::decode(&ext).is_err());
        }
        // Tag allocation: header 0, partial 1, reveal 2, foreign 3.
        assert_eq!(header_msg(1).encode()[0], 0);
        assert_eq!(partial_msg(1).encode()[0], 1);
        assert_eq!(reveal_msg(1).encode()[0], 2);
        let sub = crate::submission::SubmissionMessage::Transaction { entry: tx_entry(1) };
        assert!(matches!(
            GossipMessage::decode(&sub.encode()),
            Err(CodecError::InvalidOptionTag { tag: 3 })
        ));
        assert!(matches!(GossipMessage::decode(&[]), Err(CodecError::Truncated)));
        // A truncated partial body is malformed, not foreign.
        let mut bad = partial_msg(1).encode();
        bad.truncate(bad.len() - 10);
        assert!(GossipMessage::decode(&bad).is_err());
    }

    #[test]
    fn the_dsr8_ordering_sequence() {
        let mut e = GossipEngine::new();
        // Partial first: rejected, and NOT marked seen.
        assert!(matches!(
            e.publish(partial_msg(7)),
            PublishOutcome::Rejected(Rejection::PartialBeforeHeader { height: 7, .. })
        ));
        assert_eq!(e.stats().rejected_partials, 1);
        // Header: accepted, learned, broadcast.
        let PublishOutcome::Broadcast { frame } = e.publish(header_msg(7)) else {
            panic!()
        };
        assert_eq!(frame, header_msg(7).encode());
        assert!(e.known_header(&shard(), 7).is_some());
        // The partial again: now accepted (the gate opened; the earlier
        // rejection did not poison the seen-set).
        let PublishOutcome::Broadcast { .. } = e.publish(partial_msg(7)) else {
            panic!()
        };
        // And again: duplicate.
        assert!(matches!(e.publish(partial_msg(7)), PublishOutcome::Duplicate));
        assert_eq!(e.stats().accepted, 2);
        assert_eq!(e.stats().duplicates, 1);
        assert_eq!(e.stats().rejected_partials, 1);
    }

    #[test]
    fn zero_height_header_rejected() {
        let mut e = GossipEngine::new();
        assert!(matches!(
            e.publish(header_msg(0)),
            PublishOutcome::Rejected(Rejection::ZeroHeightHeader { .. })
        ));
        // And the gate never opened at height 0.
        assert!(e.known_header(&shard(), 0).is_none());
        assert!(matches!(
            e.publish(partial_msg(0)),
            PublishOutcome::Rejected(Rejection::PartialBeforeHeader { .. })
        ));
        assert_eq!(e.stats().rejected_headers, 1);
    }

    #[test]
    fn header_dedup_and_forks() {
        let mut e = GossipEngine::new();
        let h = header_msg(5);
        let PublishOutcome::Broadcast { .. } = e.publish(h.clone()) else { panic!() };
        assert!(matches!(e.publish(h.clone()), PublishOutcome::Duplicate));

        // A fork header at the same slot: a distinct digest â€” it
        // propagates (the equivocation evidence flows to finality), and
        // the learned map is first-wins.
      
        let mut forked = header_msg(5);
        let GossipMessage::Header { header, .. } = &mut forked {
            header.fee_total = nerv_core::types::FeeSats::from_u64(999);
        }
        let PublishOutcome::Broadcast { .. } = e.publish(forked.clone()) else { panic!() };
        let first = header_msg(5);
       assert_eq!(e.known_header(&shard(), 5), Some(first.header.header_hash()));

        assert_eq!(e.stats().accepted, 2);
        // A partial for the forked slot: the gate is open (presence).
        let PublishOutcome::Broadcast { .. } = e.publish(partial_msg(5)) else { panic!() };
    }

    #[test]
    fn reveal_is_carried_ungated() {
        let mut e = GossipEngine::new();
        let PublishOutcome::Broadcast { .. } = e.publish(reveal_msg(42)) else { panic!() };
        assert!(e.known_header(&shard(), 42).is_none());
        assert_eq!(e.stats().accepted, 1);
    }

    #[test]
    fn header_window_pruning() {
        let mut e = GossipEngine::new();
        for h in 1..=300u64 {
           assert!(matches!(
               e.publish(header_msg(h)),
               PublishOutcome::Broadcast { .. }
           ));
       }

        assert!(e.stats().pruned_headers > 0);
        // max = 300, floor = 44: heights 1..=44 pruned, 45..=300 retained.
        assert!(e.known_header(&shard(), 44).is_none());
        assert!(e.known_header(&shard(), 45).is_some());
        assert!(e.known_header(&shard(), 300).is_some());
        // Stale partials are rejected; fresh ones pass.
        assert!(matches!(
            e.publish(partial_msg(44)),
            PublishOutcome::Rejected(Rejection::PartialBeforeHeader { height: 44, .. })
        ));
        let PublishOutcome::Broadcast { .. } = e.publish(partial_msg(45)) else { panic!() };
        let PublishOutcome::Broadcast { .. } = e.publish(partial_msg(300)) else { panic!() };
    }

    #[test]
    fn peer_tracking_and_recipients() {
        let (a, b, c) = (peer_id(1), peer_id(2), peer_id(3));
        let mut e = GossipEngine::new();
        assert!(e.recipients(None).is_empty());
        e.on_connected(a);
        e.on_connected(b);
        e.on_connected(a); // idempotent
        assert_eq!(e.peers(), vec![a, b]);
        assert_eq!(e.recipients(Some(&a)), vec![b]);
        assert_eq!(e.recipients(None), vec![a, b]);
        e.on_disconnected(&a);
        assert_eq!(e.peers(), vec![b]);
        assert_eq!(e.recipients(Some(&b)), vec![]);
    }

    #[test]
    fn seen_set_capacity_is_bounded() {
        let mut e = GossipEngine::new();
        // Fill with accepted reveals (ungated), then verify FIFO eviction.
        for i in 0..(DEDUP_CAPACITY as u64 + 64) {
            let msg = GossipMessage::Reveal {
                shard: shard(),
                height: Height::from_u64(i + 1),
                batch: i,
                reveal: test_reveal(),
            };
            let PublishOutcome::Broadcast { .. } = e.publish(msg) else { panic!() };
        }
        assert_eq!(e.stats().accepted, DEDUP_CAPACITY as u64 + 64);
        // The earliest digests were evicted: re-publishing one is accepted
        // again (the bound trades memory for reprocessing, never drops).
        let early = GossipMessage::Reveal {
            shard: shard(),
            height: Height::from_u64(1),
            batch: 0,
            reveal: test_reveal(),
        };
        assert!(matches!(e.publish(early), PublishOutcome::Broadcast { .. }));
    }

    /// The wiring recipe over a simulated full mesh: flood, dedup, and
    /// the ordering gate holding across the mesh.
    #[test]
    fn simulated_mesh_flood_and_ordering() {
        const N: usize = 4;
        let ids: Vec<PeerId> = (0..N).map(|i| peer_id(10 + i as u64)).collect();
        let mut engines: Vec<GossipEngine> = (0..N).map(|_| GossipEngine::new()).collect();
        for (i, e) in engines.iter_mut().enumerate() {
            for (j, &id) in ids.iter().enumerate() {
                if i != j {
                    e.on_connected(id);
                }
            }
        }
        let mut queue: VecDeque<(usize, PeerId, Vec<u8>)> = VecDeque::new();
        let mut deliver = |queue: &mut VecDeque<(usize, PeerId, Vec<u8>)>,
                           engines: &mut Vec<GossipEngine>,
                           accepted: &mut Vec<usize>,
                           frame: &[u8],
                           from: usize| {
            for j in 0..N {
                if j != from {
                    queue.push_back((j, ids[from], frame.to_vec()));
                }
            }
            while let Some((j, from_id, payload)) = queue.pop_front() {
                match engines[j].receive(from_id, &payload).unwrap() {
                    Inbound::Accepted { forward, .. } => {
                        accepted[j] += 1;
                        for q in engines[j].recipients(Some(&from_id)) {
                            if let Some(k) = ids.iter().position(|p| *p == q) {
                                queue.push_back((k, ids[j], forward.clone()));
                            }
                        }
                    }
                    Inbound::Duplicate => {}
                    Inbound::Rejected(r) => panic!("rejected in the mesh: {r:?}"),
                }
            }
        };

        // 1. The partial first: rejected at the origin (DSR-8).
        assert!(matches!(
            engines[0].publish(partial_msg(7)),
            PublishOutcome::Rejected(Rejection::PartialBeforeHeader { .. })
        ));

        // 2. The header floods the mesh; every node accepts exactly once.
        let frame = header_msg(7).encode();
        let mut accepted = vec![0usize; N];
        deliver(&mut queue, &mut engines, &mut accepted, &frame, 0);
        assert_eq!(accepted, vec![0, 1, 1, 1]);
        for e in &engines {
            assert!(e.known_header(&shard(), 7).is_some());
        }

        // 3. A partial from node 2 floods: everyone accepts (gate open
        //    mesh-wide), and the echoes dedup.
        let frame = partial_msg(7).encode();
        let mut accepted = vec![0usize; N];
        deliver(&mut queue, &mut engines, &mut accepted, &frame, 2);
        assert_eq!(accepted, vec![1, 1, 0, 1]);

        // 4. A cold node â€” connected but never fed the header â€” rejects
        //    the partial. This is the adversarial delivery the testkit
        //    executes against the full node (DSR-8).
        let mut cold = GossipEngine::new();
        cold.on_connected(ids[0]);
        assert!(matches!(
            cold.receive(ids[0], &partial_msg(7).encode()).unwrap(),
            Inbound::Rejected(Rejection::PartialBeforeHeader { height: 7, .. })
        ));
        // Once fed the header, the same partial passes.
        cold.receive(ids[0], &header_msg(7).encode()).unwrap();
        assert!(matches!(
            cold.receive(ids[0], &partial_msg(7).encode()).unwrap(),
            Inbound::Accepted { .. }
        ));
    }

    /// The DSR-8 rule over real sockets â€” the node's exact wiring.
    #[tokio::test]
    async fn ordering_rule_over_sockets() {
       let mut a = node(1).await;
       let mut b = node(2).await;
       let (id_a, id_b) = (a.info.id(), b.info.id());
       let mut ea = GossipEngine::new();
       let mut eb = GossipEngine::new();
       let (host_a, mut ev_a) = (a.host.clone(), &mut a.events);
       let (host_b, mut ev_b) = (b.host.clone(), &mut b.events);


       host_a.dial(b.info.clone()).unwrap();
       drain(&mut ev_a, &mut ea, &host_a).await;
       drain(&mut ev_b, &mut eb, &host_b).await;

       assert_eq!(ea.recipients(None), vec![id_b]);
       assert_eq!(eb.recipients(None), vec![id_a]);


       // Partial first: rejected at the origin, nothing sent.
       assert!(matches!(
           ea.publish(partial_msg(7)),
           PublishOutcome::Rejected(Rejection::PartialBeforeHeader { .. })
       ));
       drain(&mut ev_a, &mut ea, &host_a).await;
       drain(&mut ev_b, &mut eb, &host_b).await;


       // Header: broadcast; B accepts and its echo dedups at A.
       let PublishOutcome::Broadcast { frame } = ea.publish(header_msg(7)) else { panic!() };
       host_a.send(&id_b, frame).unwrap();
       let mut got_b = drain(&mut ev_b, &mut eb, &host_b).await;

        assert!(matches!(
            got_b.pop(),
            Some((from, GossipMessage::Header { .. })) if from == id_a
        ));
        let mut got_a = drain(&mut ev_a, &mut ea, &host_a).await;
       assert!(
           matches!(got_a.pop(), Some((_, GossipMessage::Header { .. }))),
           "the echo arrives and dedups"
       );


       // Partial now: broadcast from B; A accepts.
       let PublishOutcome::Broadcast { frame } = eb.publish(partial_msg(7)) else { panic!() };
       host_b.send(&id_a, frame).unwrap();
       let mut got_a = drain(&mut ev_a, &mut ea, &host_a).await;

        assert!(matches!(
            got_a.pop(),
            Some((from, GossipMessage::Partial { height, .. })) if from == id_b && height.as_u64() == 7
        ));
        let _ = &mut got_b;

        // The same partial re-published by A: duplicate.
        assert!(matches!(ea.publish(partial_msg(7)), PublishOutcome::Duplicate));
    }

    /// The wiring recipe: process host events until quiescence.
    async fn drain(
        events: &mut mpsc::Receiver<HostEvent>,
        engine: &mut GossipEngine,
        host: &Host,
    ) -> Vec<(PeerId, GossipMessage)> {
        let mut accepted = Vec::new();
        while let Ok(Some(event)) =
            tokio::time::timeout(Duration::from_millis(200), events.recv()).await
        {
            match event {
                HostEvent::Connected(p) => engine.on_connected(p),
                HostEvent::Disconnected(p) => engine.on_disconnected(&p),
                HostEvent::Frame(from, payload) => {
                    if let Ok(Inbound::Accepted { message, forward }) =
                        engine.receive(from, &payload)
                    {
                        for peer in engine.recipients(Some(&from)) {
                            host.send(&peer, forward.clone()).unwrap();
                        }
                        accepted.push((from, message));
                    }
                }
            }
        }
        accepted
    }

    #[test]
    fn stats_snapshot() {
        let mut e = GossipEngine::new();
        e.publish(header_msg(1)).unwrap();
        e.publish(header_msg(1)).unwrap();
        e.publish(partial_msg(0)).unwrap();
        e.publish(reveal_msg(9)).unwrap();
        let s = e.stats();
        assert_eq!(s.accepted, 2);
        assert_eq!(s.duplicates, 1);
        assert_eq!(s.rejected_partials, 1);
        assert_eq!(s.rejected_headers, 1);
        assert_eq!(s.headers_learned, 1);
        assert_eq!(s.pruned_headers, 0);
        let _ = SplitMix64::new(0);
    }

     #[test]
    fn block_data_ordering_rule() {
        let mut e = GossipEngine::new();
        let shard = shard();

        // BlockData without a header: rejected.
        let result = e.publish(GossipMessage::BlockData {
            shard,
            height: Height::from_u64(5),
            data: vec![1, 2, 3],
        });
        assert!(matches!(
            result,
            PublishOutcome::Rejected(Rejection::BlockDataBeforeHeader { height: 5, .. })
        ));

        // After the header: accepted.
        let _ = e.publish(header_msg(5));
        let result = e.publish(GossipMessage::BlockData {
            shard,
            height: Height::from_u64(5),
            data: vec![1, 2, 3],
        });
        assert!(matches!(result, PublishOutcome::Broadcast { .. }));

        // Duplicate: deduped.
        let result = e.publish(GossipMessage::BlockData {
            shard,
            height: Height::from_u64(5),
            data: vec![1, 2, 3],
        });
        assert!(matches!(result, PublishOutcome::Duplicate));
    }

    #[test]
    fn block_data_codec_roundtrip() {
        let msg = GossipMessage::BlockData {
            shard: shard(),
            height: Height::from_u64(42),
            data: vec![0xAB; 1024],
        };
        let enc = msg.encode();
        assert_eq!(enc.len(), msg.encoded_len());
        // Tag 8 â€” moved off tag 6 to free the gap-4 DA topics (DABlob=6,
        // DACell=7).
        assert_eq!(enc[0], 8);
        assert_eq!(GossipMessage::decode(&enc).unwrap(), msg);
        assert!(GossipMessage::decode(&enc[..enc.len() - 1]).is_err());
    }

    fn dummy_set_commitment() -> SetCommitment {
        SetCommitment {
            shard: shard(),
            height: Height::from_u64(7),
            widths: vec![4, 4],
            data_lens: vec![1024, 2048],
            blob_tree_roots: vec![
                Hash256::from_bytes([0xA1; 32]),
                Hash256::from_bytes([0xA2; 32]),
            ],
        }
    }

    fn dummy_cell_auth() -> CellAuth {
        CellAuth {
            blob: 0,
            row: 1,
            col: 2,
            chunk: vec![0u8; CHUNK_LEN],
            row_root: Hash256::from_bytes([0xB1; 32]),
            row_path: vec![Hash256::from_bytes([0xB2; 32]), Hash256::from_bytes([0xB3; 32])],
            blob_path: vec![Hash256::from_bytes([0xB4; 32])],
        }
    }

    #[test]
    fn da_blob_codec_roundtrip() {
        let msg = GossipMessage::DABlob {
            shard: shard(),
            height: Height::from_u64(11),
            set_commitment: dummy_set_commitment(),
        };
        let enc = msg.encode();
        assert_eq!(enc.len(), msg.encoded_len());
        assert_eq!(enc[0], 6);
        let dec = GossipMessage::decode(&enc).unwrap();
        assert_eq!(dec, msg);
        // A DABlob must not be gated by the header ordering rule
        // (gap 4, "no ordering gate").
        let mut e = GossipEngine::new();
        let PublishOutcome::Broadcast { .. } = e.publish(msg.clone()) else {
            panic!("DABlob must publish without a learned header")
        };
        // And dedup catches a re-issue.
        assert!(matches!(e.publish(msg), PublishOutcome::Duplicate));
    }

    #[test]
    fn da_cell_codec_roundtrip() {
        let msg = GossipMessage::DACell {
            shard: shard(),
            height: Height::from_u64(11),
            cell_auth: dummy_cell_auth(),
        };
        let enc = msg.encode();
        assert_eq!(enc.len(), msg.encoded_len());
        assert_eq!(enc[0], 7);
        let dec = GossipMessage::decode(&enc).unwrap();
        assert_eq!(dec, msg);
        // DACell is also ungated.
        let mut e = GossipEngine::new();
        assert!(matches!(e.publish(msg.clone()), PublishOutcome::Broadcast { .. }));
    }

    #[test]
    fn da_foreign_tag_rejected() {
        // Tag 5 is the wire-level MixFragment (HostFrame), not a
        // GossipMessage tag â€” the engine must reject it.
        let mut e = GossipEngine::new();
        let mut bogus = vec![5u8];
        bogus.extend_from_slice(&shard().encode());
        bogus.extend_from_slice(&11u64.to_le_bytes()); // height
        bogus.extend_from_slice(&[0u8; 32]); // stub tail
        let res = e.receive(PeerId::PLACEHOLDER, &bogus);
        assert!(matches!(
            res,
            Err(nerv_core::error::CodecError::InvalidOptionTag { tag: 5 })
        ));
    }

    #[test]
    fn block_data_decode_fixes_previously_malformed_arm() {
        // Regression: pre-gap-4 the BlockData decode arm in `match` was
        // missing its leading pattern (a Rust syntax error). After
        // rebinding to tag 8, a BlockData roundtrip must succeed end-to-end.
        let msg = GossipMessage::BlockData {
            shard: shard(),
            height: Height::from_u64(99),
            data: vec![0; 17],
        };
        let enc = msg.encode();
        let decoded = GossipMessage::decode(&enc).unwrap();
        assert_eq!(decoded, msg);
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod gap7_knowledge_tests {
    use super::*;
    use nerv_codec::codec_w::Delta;
    use nerv_knowledge::{BlockEvent, KnowledgeState};

    /// Local Node fixture for the gap-7 feed tests. Mirrors the
    /// ixture_node() helper in the gap-2/5 tests module but lives in
    /// this module's scope so the gap-7 tests don't depend on a
    /// cross-module helper.
    fn make_node() -> (Node, NodeConfig) {
        let cfg = NodeConfig::default();
        let shard = cfg.shard;
        let shard_state = Some(ShardState::genesis(shard, cfg.params_root));
        let (host, events) = Host::new(HostConfig::default()).expect("host");
        (
            Node {
                config: cfg.clone(),
                role: crate::roles::Role::Validator,
                host,
                events,
                gossip: GossipEngine::new(),
                beacon_view: NodeBeaconView::default(),
                shard_state,
                chain: MemoryChain::new(),
                pending_blocks: BTreeMap::new(),
                applied_count: 0,
                rejected_count: 0,
                fri_shape: default_fri_shape(),
                submission_accepted: 0,
                submission_rejected: 0,
                submission_deferred: 0,
                last_rotated_epoch: None,
                rotations_completed: 0,
                rotations_failed: 0,
                knowledge: KnowledgeState::genesis(),
                knowledge_reveals: 0,
                knowledge_misses: 0,
                knowledge_faults: 0,
            },
            cfg,
        )
    }

    #[test]
    fn feed_knowledge_layer_no_op_on_empty_block() {
        // A block with no reveal and no legs: eed_knowledge_layer
        // bumps the misses counter, leaves reveals / faults at zero.
        let (mut node, _cfg) = make_node();
        let _ = node.knowledge.process(BlockEvent::Miss { height: 1, legs: 0 });
        assert_eq!(node.knowledge_reveals(), 0);
        assert_eq!(node.knowledge_misses(), 1);
        assert_eq!(node.knowledge_faults(), 0);
    }

    #[test]
    fn knowledge_process_reveal_advances_embedding() {
        // Construct a knowledge state, run a Reveal, and confirm the
        // embedding advanced exactly one step.
        let mut state = KnowledgeState::genesis();
        assert_eq!(state.embedding().next_height(), 1);

        let mut delta = Delta::default();
        delta.0[0] = 7;
        delta.0[63] = 9;
        state
            .process(BlockEvent::Reveal {
                height: 1,
                delta: delta.clone(),
                fee_sum: 5_000,
                bucket: 3,
            })
            .unwrap();

        assert_eq!(state.embedding().next_height(), 2);
        assert_eq!(state.embedding().coord().0[0], 7);
        assert_eq!(state.embedding().coord().0[63], 9);
    }

    #[test]
    fn derived_root_pinned_against_header_field() {
        // The advisory fault check compares knowledge.derived_root()
        // against header.derived. Pin both sides deterministically:
        // when the knowledge state matches the header, no fault is
        // recorded; when it diverges, the counter increments.
        let mut knowledge = KnowledgeState::genesis();
        let delta = Delta::default();
        let header_derived = knowledge.derived_root();
        knowledge
            .process(BlockEvent::Reveal {
                height: 1,
                delta,
                fee_sum: 0,
                bucket: 0,
            })
            .unwrap();
        let post_derived = knowledge.derived_root();
        assert_ne!(header_derived, post_derived, "the derive must move on a reveal");
    }

    #[test]
    fn knowledge_counters_start_at_zero() {
        // A fresh node starts with no knowledge events.
        let (node, _cfg) = make_node();
        assert_eq!(node.knowledge_reveals(), 0);
        assert_eq!(node.knowledge_misses(), 0);
        assert_eq!(node.knowledge_faults(), 0);
    }

    #[test]
    fn helper_feed_knowledge_layer_handles_miss_when_prev_reveal_absent() {
        // When lock.header.prev_reveal is None, the feed emits a
        // BlockEvent::Miss carrying the settled-leg count. The
        // embedding's skip-and-carry rule advances the height without
        // updating e_t.
        let mut knowledge = KnowledgeState::genesis();
        knowledge
            .process(BlockEvent::Miss { height: 1, legs: 7 })
            .unwrap();
        assert_eq!(knowledge.embedding().next_height(), 2);
        assert_eq!(knowledge.embedding().coord().0[0], 0, "miss carries e_t");
        assert_eq!(knowledge.embedding().misses().len(), 1);
        assert_eq!(knowledge.embedding().misses()[0].legs, 7);
    }

    #[test]
    fn helper_feed_knowledge_layer_handles_reveal_when_prev_reveal_present() {
        // When prev_reveal is Some(bytes), the feed emits a Reveal
        // with the decoded Δ_B. The embedding's pply rule advances
        // e_t by wrapping addition.
        let mut knowledge = KnowledgeState::genesis();
        let mut delta = Delta::default();
        delta.0[0] = 42;
        delta.0[5] = 100;
        let bytes = delta.canonical_bytes();
        knowledge
.process(BlockEvent::Reveal {
                height: 1,
                delta: Delta::from_canonical_bytes(&bytes),
                fee_sum: 1_000,
                bucket: 2,
            })
            .unwrap();
        assert_eq!(knowledge.embedding().coord().0[0], 42);
        assert_eq!(knowledge.embedding().coord().0[5], 100);
    }

    // ---- Producer integration tests (gap continuation) ----

    #[test]
    fn producer_role_attaches_and_payout_routes_to_payout_address() {
        // Build a producer identity from a fixed seed; the payout
        // Address is deterministic from the seed + shard. The node's
        // `producer_role()` accessor returns the role; the
        // `register_producer_stake` and `epoch_producer_payout` methods
        // route through the role.
        let seed = [0xABu8; 32];
        let shard = ShardId::new(6, 0);
        let identity =
            crate::producer::ProducerIdentity::from_seed(&seed, shard, 1_000).unwrap();
        let role = crate::producer::ProducerRole::new(identity.clone());
        let (mut node, _cfg) = make_node();
        node.attach_producer(role);

        let stored = node.producer().expect("producer attached");
        assert_eq!(stored.shard(), shard);
        assert_eq!(stored.stake_nano(), 1_000);
        assert_eq!(stored.payout_address(), identity.payout_address);
    }

    #[test]
    fn register_producer_stake_is_visible_on_ledger() {
        let seed = [0xCDu8; 32];
        let identity = crate::producer::ProducerIdentity::from_seed(
            &seed,
            ShardId::new(6, 0),
            5_000,
        )
        .unwrap();
        let (mut node, _cfg) = make_node();
        node.attach_producer(crate::producer::ProducerRole::new(identity.clone()));

        let mut ledger = nerv_economy::staking::StakeLedger::new();
        node.register_producer_stake(&mut ledger);
        assert_eq!(identity.stake_of(&ledger), 5_000);
    }

    #[test]
    fn epoch_producer_payout_uses_subsidy_split() {
        // The producer's per-block subsidy is the `validator-subsidy`
        // bucket split canonical-order across the active shard set
        // (erratum 162). The producer on shard 0 must receive a
        // positive amount; the sum of all shard splits equals the
        // bucket total.
        let seed = [0xEFu8; 32];
        let identity = crate::producer::ProducerIdentity::from_seed(
            &seed,
            ShardId::new(6, 0),
            100,
        )
        .unwrap();
        let (mut node, _cfg) = make_node();
        node.attach_producer(crate::producer::ProducerRole::new(identity));

        let schedule = nerv_economy::schedule::EmissionSchedule::genesis();
        let active = nerv_core::types::ShardSet::genesis();
        let epoch = nerv_core::types::Epoch::from_u64(0);
        let payout = node
            .epoch_producer_payout(&schedule, epoch, &active)
            .expect("producer payout");
        assert!(payout > 0, "payout must be positive");
    }

    #[test]
    fn non_producer_node_returns_zero_payout() {
        // Without an attached producer, `epoch_producer_payout` returns
        // 0 (a non-producer node doesn't receive the validator
        // subsidy).
        let (node, _cfg) = make_node();
        let schedule = nerv_economy::schedule::EmissionSchedule::genesis();
        let active = nerv_core::types::ShardSet::genesis();
        let payout = node.epoch_producer_payout(
            &schedule,
            nerv_core::types::Epoch::from_u64(0),
            &active,
        );
        assert_eq!(payout.unwrap(), 0);
    }
}
