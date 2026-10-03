//! The relay runtime (WP §6.2; errata 153–155): peel → jitter → forward,
//! the replay cache, and the cover emitter. The engine is pure logic;
//! `run_relay` transports.

use std::collections::{BTreeSet, VecDeque};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use nerv_core::constants::SPHINX;
use nerv_core::hash::Xof;
use nerv_crypto::mlkem::DecapsulationKey;

use crate::host::{Host, HostEvent, PeerId};
use crate::relay_registry::{select_path, RelayRegistry};
use crate::sphinx::{
    build, forward, peel, replay_tag, Action, Packet, PathSpec, TerminalAction, CAPSULE_LEN, CT_LEN,
    ENCAPS_LEN, HEADER_LEN, MAX_DATA, MIX_FRAGMENT_TAG, MIX_PACKET_TAG, PATH_RELAYS,
};

pub const JITTER_MEAN_MS: u64 = nerv_core::params::MIXNET_JITTER_MEAN_MS;
pub const JITTER_CAP_MS: u64 = nerv_core::params::MIXNET_JITTER_CAP_MS;
pub const REPLAY_CAPACITY: usize = 65_536;

const RATIO: u64 = JITTER_CAP_MS / JITTER_MEAN_MS;
const _: () = assert!(JITTER_MEAN_MS > 0);
const _: () = assert!(JITTER_CAP_MS >= JITTER_MEAN_MS);
const _: () = assert!(JITTER_CAP_MS % JITTER_MEAN_MS == 0);
const _: () = assert!(RATIO >= 1 && RATIO <= 12);
const _: () = assert!(RATIO == 5, "genesis jitter shape: 500 ms cap at 100 ms mean (E-004)");

const Q: u128 = 1u128 << 64;

const fn qmul(a: i128, b: i128) -> i128 {
    (a * b) >> 64
}

const LN2_Q64: i128 = const_ln2();
const E_R_Q64: u128 = const_exp_neg(RATIO);

const fn const_ln2() -> i128 {
    let t = (Q as i128) / 3;
    let t2 = qmul(t, t);
    let mut term = t;
    let mut sum = t;
    let mut n = 1i128;
    while n < 45 {
        term = qmul(term, t2);
        if term == 0 {
            break;
        }
        sum += term / (2 * n + 1);
        n += 1;
    }
    2 * sum
}

const fn const_exp_neg(r: u64) -> u128 {
    let mut sum = Q as i128;
    let mut term = Q as i128;
    let mut n = 1i128;
    while n <= 90 {
        term = -(term * (r as i128)) / n;
        if term == 0 {
            break;
        }
        sum += term;
        n += 1;
    }
    sum as u128
}

/// ln(y) in Q64 for y_q = y·2^64 ∈ [1, 2^64] (erratum 153).
pub(crate) fn ln_q64(y_q: u128) -> i128 {
    debug_assert!(y_q >= 1);
    if y_q >= Q {
        return 0;
    }
    let k = (y_q.leading_zeros() - 64) as i128;
    let m_q = y_q << (k as u32);
    let num = ((m_q as i128) - (Q as i128)) << 64;
    let t = num / ((m_q + Q) as i128);
    let t2 = qmul(t, t);
    let mut term = t;
    let mut sum = t;
    for n in 1..45i128 {
        term = qmul(term, t2);
        if term == 0 {
            break;
        }
        sum += term / (2 * n + 1);
    }
    2 * sum - k * LN2_Q64
}

/// The truncated-exponential delay in milliseconds, deterministic in the
/// draw (erratum 153).
pub fn jitter_millis(draw: u64) -> u64 {
    let span = Q - E_R_Q64;
    let y_q = Q - ((draw as u128 * span) >> 64);
    let ln_y = ln_q64(y_q);
    let delay_q = ((-ln_y) as u128) * JITTER_MEAN_MS as u128;
    (((delay_q + Q / 2) >> 64) as u64).min(JITTER_CAP_MS)
}

// ---------------------------------------------------------------------------
// The replay cache (erratum 154)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ReplayCache {
    seen: BTreeSet<[u8; 16]>,
    order: VecDeque<[u8; 16]>,
    capacity: usize,
}

impl ReplayCache {
    pub fn new(capacity: usize) -> ReplayCache {
        ReplayCache { seen: BTreeSet::new(), order: VecDeque::new(), capacity }
    }

    /// False = replay.
    pub fn check_and_insert(&mut self, tag: [u8; 16]) -> bool {
        if !self.seen.insert(tag) {
            return false;
        }
        self.order.push_back(tag);
        while self.order.len() > self.capacity {
            if let Some(old) = self.order.pop_front() {
                self.seen.remove(&old);
            }
        }
        true
    }

    pub fn contains(&self, tag: &[u8; 16]) -> bool {
        self.seen.contains(tag)
    }

    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

// ---------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RelayStats {
    pub packets_seen: u64,
    pub forwarded: u64,
    pub delivered: u64,
    pub cover_dropped: u64,
    pub misrouted: u64,
    pub replayed: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RelayAction {
    Forward { to: PeerId, frame: Vec<u8>, delay_ms: u64 },
    Deliver { to: PeerId, frame: Vec<u8>, delay_ms: u64 },
    Drop,
}

pub struct RelayEngine {
    dk: DecapsulationKey,
    replay: ReplayCache,
    stats: RelayStats,
}

impl RelayEngine {
    pub fn new(dk: DecapsulationKey) -> RelayEngine {
        RelayEngine { dk, replay: ReplayCache::new(REPLAY_CAPACITY), stats: RelayStats::default() }
    }

    pub fn stats(&self) -> RelayStats {
        self.stats
    }

    pub fn replay_cache(&self) -> &ReplayCache {
        &self.replay
    }

    /// One packet: replay-check (pre-decapsulation), peel, jitter, and the
    /// next action. `entropy` drives the jitter draw and the forward pads.
    pub fn handle_packet(&mut self, packet: &Packet, entropy: u64) -> RelayAction {
        self.stats.packets_seen += 1;
        let ct0: &[u8; CT_LEN] = packet.0[..CT_LEN].try_into().expect("ct slice");
        let capsule0 = &packet.0[HEADER_LEN..HEADER_LEN + CAPSULE_LEN];
        let tag = replay_tag(ct0, capsule0);
        if !self.replay.check_and_insert(tag) {
            self.stats.replayed += 1;
            return RelayAction::Drop;
        }
        let mut xof = Xof::new(&SPHINX, &entropy.to_le_bytes());
        let jitter_draw = xof.next_u64();
        let ct_pad = xof.read_array::<CT_LEN>();
        let cap_pad = xof.read_array::<CAPSULE_LEN>();
        let delay_ms = jitter_millis(jitter_draw);
        match peel(packet, &self.dk) {
            Err(_) => {
                self.stats.misrouted += 1;
                RelayAction::Drop
            }
            Ok(p) => match p.action {
                Action::Relay => {
                    let Some(parts) = &p.forward else {
                        self.stats.misrouted += 1;
                        return RelayAction::Drop;
                    };
                    match forward(parts, &ct_pad, &cap_pad) {
                        Ok(next) => {
                            self.stats.forwarded += 1;
                            RelayAction::Forward { to: p.next, frame: next.to_frame(), delay_ms }
                        }
                        Err(_) => {
                            self.stats.misrouted += 1;
                            RelayAction::Drop
                        }
                    }
                }
                Action::Deliver { data } => {
                    let mut frame = Vec::with_capacity(1 + data.len());
                    frame.push(MIX_FRAGMENT_TAG);
                    frame.extend_from_slice(&data);
                    self.stats.delivered += 1;
                    RelayAction::Deliver { to: p.next, frame, delay_ms }
                }
                Action::Drop => {
                    self.stats.cover_dropped += 1;
                    RelayAction::Drop
                }
            },
        }
    }
}

// ---------------------------------------------------------------------------
// The runtime
// ---------------------------------------------------------------------------

fn os_draw() -> u64 {
    let mut b = [0u8; 8];
    if getrandom::getrandom(&mut b).is_ok() {
        u64::from_le_bytes(b)
    } else {
        u64::MAX // fail-safe: the maximum delay, never zero (erratum 153)
    }
}

/// The relay's transport loop: mix packets in, (jittered) actions out.
/// Non-mix frames are ignored — the relay binary is not a validator.
pub async fn run_relay(
    mut engine: RelayEngine,
    host: Host,
    mut events: mpsc::Receiver<HostEvent>,
    mut entropy: Box<dyn FnMut() -> u64 + Send>,
) {
    while let Some(event) = events.recv().await {
        if let HostEvent::Frame(_, payload) = event {
            if payload.first() != Some(&MIX_PACKET_TAG) {
                continue;
            }
            let Ok(packet) = Packet::from_frame(&payload) else {
                continue;
            };
            match engine.handle_packet(&packet, entropy()) {
                RelayAction::Forward { to, frame, delay_ms }
                | RelayAction::Deliver { to, frame, delay_ms } => {
                    if delay_ms > 0 {
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    }
                    let _ = host.send(&to, frame);
                }
                RelayAction::Drop => {}
            }
        }
    }
}

pub fn spawn_relay(
    engine: RelayEngine,
    host: Host,
    events: mpsc::Receiver<HostEvent>,
) -> JoinHandle<()> {
    tokio::spawn(run_relay(engine, host, events, Box::new(os_draw)))
}

// ---------------------------------------------------------------------------
// Cover traffic (§6.2, §10.5; erratum 155)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CoverError {
    #[error(transparent)]
    Registry(#[from] crate::relay_registry::RegistryError),
    #[error(transparent)]
    Sphinx(#[from] crate::sphinx::SphinxError),
}

/// A real 5-hop packet with a Drop terminal and a uniformly random
/// payload — indistinguishable on the wire (the payload region is
/// stream-XORed per hop either way).
pub fn cover_packet(path: &PathSpec, seed: u64) -> Result<Packet, CoverError> {
    let mut xof = Xof::new(&SPHINX, &seed.to_le_bytes());
    let mut enc = [[0u8; ENCAPS_LEN]; PATH_RELAYS];
    for e in &mut enc {
        *e = xof.read_array::<ENCAPS_LEN>();
    }
    let mut payload = vec![0u8; MAX_DATA];
    xof.fill(&mut payload);
    Ok(build(path, &payload, &enc)?)
}

/// Emit `count` cover packets over seeded registry paths, re-rolling
/// paths whose first hop is the sender itself. Returns the packets
/// actually sent.
pub fn emit_cover(
    host: &Host,
    registry: &RelayRegistry,
    self_id: Option<&PeerId>,
    seed: &[u8; 32],
    count: usize,
) -> Result<usize, CoverError> {
    let mut xof = Xof::new(&SPHINX, seed);
    let mut sent = 0;
    let mut attempts = 0;
    while sent < count && attempts < 32 {
        attempts += 1;
        let mut path_seed = [0u8; 32];
        xof.fill(&mut path_seed);
        let path = select_path(registry, &path_seed)?;
        if Some(&path[0].0) == self_id {
            continue;
        }
        let first = path[0].0;
        let spec = PathSpec::new(path, first, TerminalAction::Drop);
        let packet = cover_packet(&spec, xof.next_u64())?;
        if host.send(&first, packet.to_frame()).is_ok() {
            sent += 1;
        }
    }
    Ok(sent)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::host::HostEvent;
    use crate::relay_registry::{RelayRecord, SignedRelayRecord};
    use crate::submission::{ingest, IngestOutcome, SubmissionMessage};
    use crate::testutil::harness::{node_full, tx_entry, verify_ctx};
    use crate::testutil::{node_keys, SplitMix64};
    use nerv_core::hash::Hash256;
    use nerv_core::types::TxId;
    use nerv_registry::mempool::Mempool;
    use std::time::Duration;
    use tokio::sync::mpsc;

    fn relays(n: u64) -> Vec<(PeerId, nerv_crypto::mlkem::EncapsulationKey, DecapsulationKey)> {
        (1..=n)
            .map(|i| {
                let (sk, ek, dk) = node_keys(i);
                (PeerId::of(sk.verifying_key()), ek, dk)
            })
            .collect()
    }

    fn encap_rand(seed: u64) -> [[u8; ENCAPS_LEN]; PATH_RELAYS] {
        let mut rng = SplitMix64::new(seed);
        let mut out = [[0u8; ENCAPS_LEN]; PATH_RELAYS];
        for r in &mut out {
            for c in r.chunks_mut(8) {
                c.copy_from_slice(&rng.next_u64().to_le_bytes());
            }
        }
        out
    }

    fn txid(seed: u64) -> TxId {
        TxId::from_hash(Hash256::from_bytes(SplitMix64::new(seed).bytes32()))
    }

    fn spec_over(
        r: &[(PeerId, nerv_crypto::mlkem::EncapsulationKey, DecapsulationKey)],
        dest: PeerId,
        terminal: TerminalAction,
    ) -> PathSpec {
        let arr = core::array::from_fn(|i| (r[i].0, r[i].1.clone()));
        PathSpec::new(arr, dest, terminal)
    }

    // -- the jitter -----------------------------------------------------------

    #[test]
    fn jitter_pins() {
        assert_eq!(JITTER_MEAN_MS, 100);
        assert_eq!(JITTER_CAP_MS, 500);
        assert_eq!(jitter_millis(0), 0);
        assert!(jitter_millis(u64::MAX) <= 500);
        let mid = jitter_millis(1u64 << 63);
        assert!((66..=71).contains(&mid), "median ≈ 68.7, got {mid}");
        // Monotone in the draw.
        let mut prev = 0;
        for i in 0..2_000u64 {
            let d = jitter_millis(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            assert!(d >= prev, "monotonicity broke at {i}: {d} < {prev}");
            prev = d;
        }
        // Determinism.
        assert_eq!(jitter_millis(12345), jitter_millis(12345));
    }

    #[test]
    #[allow(clippy::float_arithmetic, clippy::float_cmp)]
    fn ln_matches_the_f64_oracle() {
        let mut rng = SplitMix64::new(0x319);
        for _ in 0..500 {
            let y_q = rng.next_u64() | 1;
            let got = ln_q64(y_q) as f64 / 18446744073709551616.0;
            let want = (y_q as f64 / 18446744073709551616.0).ln();
            assert!((got - want).abs() < 1e-12, "y_q={y_q}: {got} vs {want}");
        }
        assert_eq!(ln_q64(Q), 0);
        let l = ln_q64(1);
        assert!((l as f64 / 18446744073709551616.0 - (-44.36141955583650)).abs() < 1e-10);
        let ln2 = LN2_Q64 as f64 / 18446744073709551616.0;
        assert!((ln2 - 0.6931471805599453).abs() < 1e-15, "{ln2}");
        let e5 = E_R_Q64 as f64 / 18446744073709551616.0;
        assert!((e5 - 0.006737946999085467).abs() < 1e-17, "{e5}");
    }

    #[test]
    #[allow(clippy::float_arithmetic, clippy::float_cmp)]
    fn jitter_matches_the_f64_oracle() {
        let mut rng = SplitMix64::new(0x318);
        for _ in 0..500 {
            let u = rng.next_u64();
            let expected = -100.0
                * (1.0 - (u as f64 / 18446744073709551616.0) * (1.0 - (-5.0f64).exp())).ln();
            let got = jitter_millis(u) as f64;
            assert!((got - expected).abs() <= 1.0, "u={u}: {got} vs {expected}");
        }
    }

    #[test]
    fn jitter_distribution_battery() {
        let mut rng = SplitMix64::new(0x317);
        let n = 100_000u64;
        let mut sum: u128 = 0;
        let mut le100 = 0;
        let mut zeros = 0;
        let mut max = 0;
        let mut delays = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let d = jitter_millis(rng.next_u64());
            sum += d as u128;
            if d <= 100 {
                le100 += 1;
            }
            if d == 0 {
                zeros += 1;
            }
            max = max.max(d);
            delays.push(d);
        }
        let mean = (sum / n as u128) as u64;
        assert!((93..100).contains(&mean), "mean ≈ 96.6, got {mean}");
        delays.sort_unstable();
        let median = delays[(n / 2) as usize];
        assert!((65..=73).contains(&median), "median ≈ 68.7, got {median}");
        assert!((63_000..=64_300).contains(&le100), "P(≤100ms) ≈ 0.636, got {le100}");
        assert!((150..1_000).contains(&zeros), "P(0ms) ≈ 0.5%, got {zeros}");
        assert_eq!(max, 500, "the cap is reachable");
        assert!(delays[(n as usize * 999 / 1000)] >= 350, "the tail reaches deep");
    }

    // -- the replay cache ------------------------------------------------------

    #[test]
    fn replay_cache_semantics() {
        let mut c = ReplayCache::new(8);
        let t = |i: u8| [i; 16];
        assert!(c.check_and_insert(t(1)));
        assert!(!c.check_and_insert(t(1)));
        assert!(c.len() == 1 && c.contains(&t(1)));
        for i in 2..=10u8 {
            assert!(c.check_and_insert(t(i)));
        }
        assert_eq!(c.len(), 8, "FIFO eviction at capacity");
        assert!(!c.contains(&t(1)) && !c.contains(&t(2)), "the oldest evicted");
        assert!(c.check_and_insert(t(1)), "an evicted tag re-admits");
        assert!(ReplayCache::new(0).check_and_insert(t(9)), "zero capacity still admits");
        assert!(!ReplayCache::new(0).contains(&t(9)));
    }

    // -- the engine ------------------------------------------------------------

    #[test]
    fn engine_full_path_forward_and_deliver() {
        let r = relays(5);
        let (sk_d, _, _) = node_keys(99);
        let dest = PeerId::of(sk_d.verifying_key());
        let spec = spec_over(&r, dest, TerminalAction::Deliver);
        let payload = b"the wire payload".to_vec();
        let frames = crate::sphinx::fragment(&payload, &txid(1)).unwrap();
        let packet = build(&spec, &frames[0].encode(), &encap_rand(7)).unwrap();

        let mut engines: Vec<RelayEngine> = r.iter().map(|x| RelayEngine::new(x.2.clone())).collect();
        let mut current = packet;
        let mut delivered = None;
        for (i, e) in engines.iter_mut().enumerate() {
            let action = e.handle_packet(&current, 1_000 + i as u64);
            if i < 4 {
                let RelayAction::Forward { to, frame, delay_ms } = action else {
                    panic!("hop {i}: {action:?}")
                };
                assert_eq!(to, r[i + 1].0);
                assert!(delay_ms <= JITTER_CAP_MS);
                current = Packet::from_frame(&frame).unwrap();
            } else {
                let RelayAction::Deliver { to, frame, delay_ms } = action else {
                    panic!("terminal: {action:?}")
                };
                assert_eq!(to, dest);
                assert!(delay_ms <= JITTER_CAP_MS);
                delivered = Some(frame);
            }
        }
        let frame = delivered.unwrap();
        assert_eq!(frame[0], MIX_FRAGMENT_TAG);
        let frag = crate::sphinx::FragmentFrame::from_frame(&frame).unwrap();
        assert_eq!(crate::sphinx::reassemble(&[frag]).unwrap(), payload);
        for (i, e) in engines.iter().enumerate() {
            let s = e.stats();
            assert_eq!(s.packets_seen, 1);
            if i < 4 {
                assert_eq!(s.forwarded, 1);
                assert_eq!(s.delivered, 0);
            } else {
                assert_eq!(s.delivered, 1);
            }
            assert_eq!(s.replayed + s.misrouted + s.cover_dropped, 0);
        }
    }

    #[test]
    fn engine_replay_misroute_and_cover() {
        let r = relays(5);
        let spec = spec_over(&r, r[0].0, TerminalAction::Deliver);
        let packet = build(&spec, b"x", &encap_rand(8)).unwrap();

         // A foreign relay: BadCapsule → Drop.
       let (_, _, dk_f) = node_keys(50);
       let mut foreign = RelayEngine::new(dk_f);

        assert!(matches!(
            foreign.handle_packet(&packet, 1),
            RelayAction::Drop
        ));
        assert_eq!(foreign.stats().misrouted, 1);

        // Replay at the correct first relay.
        let mut e0 = RelayEngine::new(r[0].2.clone());
        assert!(matches!(e0.handle_packet(&packet, 2), RelayAction::Forward { .. }));
        assert!(matches!(e0.handle_packet(&packet, 3), RelayAction::Drop));
        assert_eq!(e0.stats().replayed, 1);
        assert_eq!(e0.stats().forwarded, 1);

        // A cover terminal: Drop with the cover stat.
        let cover_spec = spec_over(&r, r[0].0, TerminalAction::Drop);
        let cover = build(&cover_spec, &[0u8; 32], &encap_rand(9)).unwrap();
        let mut e4 = RelayEngine::new(r[4].2.clone());
        assert!(matches!(e4.handle_packet(&cover, 4), RelayAction::Drop));
        assert_eq!(e4.stats().cover_dropped, 1);
        assert_eq!(e4.stats().packets_seen, 1);

        // A tampered packet is a misroute, and its tag is consumed.
        let mut tampered = packet.clone();
        tampered.0[10] ^= 1;
        let mut e1 = RelayEngine::new(r[0].2.clone());
        assert!(matches!(e1.handle_packet(&tampered, 5), RelayAction::Drop));
        assert_eq!(e1.stats().misrouted, 1);
        // The pristine packet still flows (distinct tag).
        assert!(matches!(e1.handle_packet(&packet, 6), RelayAction::Forward { .. }));
        
    }

    // -- the socket-level integration -----------------------------------------

    struct Mesh {
        client: Host,
        agg: Host,
        agg_events: mpsc::Receiver<HostEvent>,
        relay_hosts: Vec<Host>,
        r1: PeerId,
        r5: PeerId,
        agg_id: PeerId,
        registry: RelayRegistry,
        path: [(PeerId, nerv_crypto::mlkem::EncapsulationKey); PATH_RELAYS],
    }

    async fn await_connected(events: &mut mpsc::Receiver<HostEvent>, n: usize) {
        let mut got = 0;
        while got < n {
            let ev = tokio::time::timeout(Duration::from_secs(5), events.recv())
                .await
                .expect("connect timeout")
                .expect("channel open");
            if matches!(ev, HostEvent::Connected(_)) {
                got += 1;
            }
        }
    }

    #[allow(clippy::type_complexity)]
    async fn mesh(zero_jitter: bool) -> Mesh {
        let mut nodes = Vec::new();
        let mut sks = Vec::new();
        let mut dks = Vec::new();
        for seed in 1..=7u64 {
            let (n, sk, dk) = node_full(seed).await;
            nodes.push(n);
            sks.push(sk);
            dks.push(dk);
        }
        // nodes[0]=client, [1..=5]=relays, [6]=aggregator.
        nodes[0].host.dial(nodes[1].info.clone()).unwrap();
        for i in 1..=5usize {
            for j in (i + 1)..=5usize {
                nodes[i].host.dial(nodes[j].info.clone()).unwrap();
            }
        }
        nodes[5].host.dial(nodes[6].info.clone()).unwrap();
        for (n, want) in nodes.iter_mut().zip([1usize, 5, 4, 4, 4, 5, 1]) {
            await_connected(&mut n.events, want).await;
        }

        let mut registry = RelayRegistry::new();
        for i in 1..=5usize {
            let mut operator = [0u8; crate::relay_registry::OPERATOR_LEN];
            operator[0] = i as u8;
            let rec = RelayRecord {
                vk: nodes[i].info.vk,
                kem_ek: nodes[i].info.kem_ek.clone(),
                addrs: nodes[i].info.addrs.clone(),
                operator,
                stake_nerv: 1_000,
            };
            registry.insert(SignedRelayRecord::register(&sks[i], rec).unwrap()).unwrap();
        }

        let mut relay_hosts = Vec::new();
        for i in 1..=5usize {
            let host = nodes[i].host.clone();
            let events = std::mem::replace(&mut nodes[i].events, mpsc::channel(1).1);
            let engine = RelayEngine::new(dks[i].clone());
            let entropy: Box<dyn FnMut() -> u64 + Send> = if zero_jitter {
                Box::new(|| 0)
            } else {
                let mut sm = SplitMix64::new(0x5E1A_0000 + i as u64);
                Box::new(move || sm.next_u64())
            };
            tokio::spawn(run_relay(engine, host.clone(), events, entropy));
            relay_hosts.push(host);
        }

        let ids: Vec<PeerId> = (1..=5usize).map(|i| nodes[i].info.id()).collect();
        let path = core::array::from_fn(|k| (ids[k], nodes[k + 1].info.kem_ek.clone()));
        let mut nodes = nodes;
        let client = nodes.remove(0);
        let agg = nodes.pop().unwrap();
        Mesh {
            client: client.host,
            agg: agg.host,
            agg_events: agg.events,
            relay_hosts,
            r1: ids[0],
            r5: ids[4],
            agg_id: agg.info.id(),
            registry,
            path,
        }
    }

    async fn next_agg_frame(events: &mut mpsc::Receiver<HostEvent>, timeout_ms: u64) -> (PeerId, Vec<u8>) {
        let ev = tokio::time::timeout(Duration::from_millis(timeout_ms), events.recv())
            .await
            .expect("delivery timeout")
            .expect("channel open");
        match ev {
            HostEvent::Frame(from, payload) => (from, payload),
            other => panic!("unexpected event: {other:?}"),
        }
    }

    async fn assert_agg_silent(events: &mut mpsc::Receiver<HostEvent>, ms: u64) {
        assert!(
            tokio::time::timeout(Duration::from_millis(ms), events.recv())
                .await
                .is_err(),
            "unexpected aggregator event"
        );
    }

    fn deliver_one(m: &Mesh, entry_seed: u64, encap_seed: u64) -> Vec<u8> {
        let entry = tx_entry(entry_seed);
        let payload = SubmissionMessage::Transaction { entry: entry.clone() }.encode();
        let frames = crate::sphinx::fragment(&payload, &entry.txid).unwrap();
        let spec = PathSpec::new(m.path, m.agg_id, TerminalAction::Deliver);
        let packet = build(&spec, &frames[0].encode(), &encap_rand(encap_seed)).unwrap();
        m.client.send(&m.r1, packet.to_frame()).unwrap();
        payload
    }

    #[tokio::test]
    async fn end_to_end_through_the_mix() {
        let mut m = mesh(false).await;

        let payload = deliver_one(&m, 42, 0xE2E);
        let (from, frame) = next_agg_frame(&mut m.agg_events, 10_000).await;
        assert_eq!(from, m.r5);
        let frag = crate::sphinx::FragmentFrame::from_frame(&frame).unwrap();
        assert_eq!(crate::sphinx::reassemble(&[frag]).unwrap(), payload);

        // The aggregator's gate runs on the reassembled submission.
        let ctx = verify_ctx();
        let mut pool = Mempool::new(64);
        match ingest(&ctx, &mut pool, &payload).unwrap() {
            IngestOutcome::Rejected(_) => {}
            other => panic!("garbage proof must reject: {other:?}"),
        }

        // A replayed packet is dropped at the first relay.
        let entry = tx_entry(42);
        let frames = crate::sphinx::fragment(
            &SubmissionMessage::Transaction { entry: entry.clone() }.encode(),
            &entry.txid,
        )
        .unwrap();
        let spec = PathSpec::new(m.path, m.agg_id, TerminalAction::Deliver);
        let packet = build(&spec, &frames[0].encode(), &encap_rand(0xE2E)).unwrap();
        m.client.send(&m.r1, packet.to_frame()).unwrap();
        assert_agg_silent(&mut m.agg_events, 500).await;

        // A distinct transaction still flows through the same path.
        let payload2 = deliver_one(&m, 43, 0xE2F);
        let (_, frame2) = next_agg_frame(&mut m.agg_events, 10_000).await;
        let frag2 = crate::sphinx::FragmentFrame::from_frame(&frame2).unwrap();
        assert_eq!(crate::sphinx::reassemble(&[frag2]).unwrap(), payload2);
    }

    #[tokio::test]
    async fn cover_traffic_is_silent_and_the_mesh_survives() {
        let mut m = mesh(true).await;

        let sent = emit_cover(&m.relay_hosts[0], &m.registry, Some(&m.r1), &[9u8; 32], 3).unwrap();
       assert_eq!(sent, 3, "re-rolling finds non-self first hops");

        assert_agg_silent(&mut m.agg_events, 300).await;

        // The mesh still delivers afterwards.
        let payload = deliver_one(&m, 44, 0xE30);
        let (_, frame) = next_agg_frame(&mut m.agg_events, 5_000).await;
        let frag = crate::sphinx::FragmentFrame::from_frame(&frame).unwrap();
        assert_eq!(crate::sphinx::reassemble(&[frag]).unwrap(), payload);

        // A cover path that starts elsewhere also stays silent.
        let sent2 = emit_cover(&m.relay_hosts[2], &m.registry, Some(&m.r5), &[8u8; 32], 2).unwrap();
        assert!(sent2 >= 1);
        assert_agg_silent(&mut m.agg_events, 300).await;
    }
}
