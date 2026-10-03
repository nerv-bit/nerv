//! The TCP mesh host (errata 143, 145): peer identity, the address book,
//! the connection registry, and the event surface.


use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};


use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::task::JoinSet;


use nerv_core::codec::Decode;
use nerv_core::constants::NET_PEER;
use nerv_core::CodecError;
use nerv_core::hash::Hash256;
use nerv_crypto::mlkem::{DecapsulationKey, EncapsulationKey};
use nerv_crypto::mldsa::{SigningKey, VerifyingKey};

use crate::gossip::GossipMessage;
use crate::sphinx::{FragmentFrame, Packet, MIX_FRAGMENT_TAG, MIX_PACKET_TAG};
use crate::submission::SubmissionMessage;
use crate::wire::{accept_handshake, dial_handshake, Session, WireError};


/// H("nerv.net.peer" ‖ vk) — the authoritative message-layer identity.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PeerId(Hash256);


impl PeerId {
   pub fn of(vk: &VerifyingKey) -> PeerId {
       PeerId(Hash256::concat(&NET_PEER, vk.as_bytes()))
   }


   pub fn from_hash(h: Hash256) -> PeerId {
       PeerId(h)
   }


   /// Borrow the underlying 32-byte hash.
   pub fn as_hash(&self) -> Hash256 {
       self.0
   }
}


impl std::fmt::Display for PeerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}


/// One dialable peer: its identity, static ML-KEM key, and addresses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeerInfo {
    pub vk: VerifyingKey,
    pub kem_ek: EncapsulationKey,
    pub addrs: Vec<SocketAddr>,
}


impl PeerInfo {
    pub fn id(&self) -> PeerId {
        PeerId::of(&self.vk)
    }
}


/// The static address book (erratum 145): merge semantics deduplicate
/// addresses; a later insert rotates the keys.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PeerBook {
    entries: BTreeMap<PeerId, PeerInfo>,
}


impl PeerBook {
    pub fn new() -> PeerBook {
        PeerBook::default()
    }


    pub fn insert(&mut self, info: PeerInfo) {
        match self.entries.get_mut(&info.id()) {
            Some(existing) => {
                existing.vk = info.vk;
                existing.kem_ek = info.kem_ek;
                for a in info.addrs {
                    if !existing.addrs.contains(&a) {
                        existing.addrs.push(a);
                    }
                }
            }
            None => {
                self.entries.insert(info.id(), info);
            }
        }
    }


    pub fn get(&self, id: &PeerId) -> Option<&PeerInfo> {
        self.entries.get(id)
    }


    pub fn remove(&mut self, id: &PeerId) -> Option<PeerInfo> {
        self.entries.remove(id)
    }


    pub fn contains(&self, id: &PeerId) -> bool {
        self.entries.contains_key(id)
    }


    pub fn ids(&self) -> impl Iterator<Item = PeerId> + '_ {
        self.entries.keys().copied()
    }


    pub fn len(&self) -> usize {
        self.entries.len()
    }


    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}


pub type Filter = Arc<dyn Fn(&PeerId) -> bool + Send + Sync>;


pub struct HostConfig {
    pub signing: SigningKey,
    pub static_kem: DecapsulationKey,
    pub peer_filter: Filter,
}


impl HostConfig {
    pub fn new(signing: SigningKey, static_kem: DecapsulationKey) -> HostConfig {
        HostConfig { signing, static_kem, peer_filter: Arc::new(|_| true) }
    }


    pub fn with_filter(mut self, filter: Filter) -> HostConfig {
        self.peer_filter = filter;
        self
    }
}


#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HostError {
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
    #[error("wire: {0}")]
    Wire(#[from] WireError),
    #[error("not connected to {0}")]
    NotConnected(PeerId),
    #[error("host is shut down")]
    ShutDown,
    #[error("os entropy unavailable")]
    Entropy,
    #[error("registry lock poisoned")]
    LockPoisoned,
}


#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostEvent {
    Connected(PeerId),
    Frame(PeerId, Vec<u8>),
    Disconnected(PeerId),
}


type Registry = Arc<Mutex<HashMap<PeerId, mpsc::UnboundedSender<Vec<u8>>>>>;


enum Command {
    Dial(PeerInfo),
    Shutdown,
}


struct Shared {
    signing: SigningKey,
    kem_dk: DecapsulationKey,
    filter: Filter,
}


/// The host handle: dial, send, shutdown. Cloneable; the engine task owns
/// the listener and the connection set.
#[derive(Clone)]
pub struct Host {
    cmd: mpsc::UnboundedSender<Command>,
    registry: Registry,
}


impl Host {
    pub async fn bind(
        cfg: HostConfig,
        addr: SocketAddr,
    ) -> Result<(Host, mpsc::Receiver<HostEvent>, SocketAddr), HostError> {
        let listener = TcpListener::bind(addr).await?;
        let local = listener.local_addr()?;
        let (events_tx, events_rx) = mpsc::channel(4096);
        let registry: Registry = Arc::new(Mutex::new(HashMap::new()));
        let shared = Arc::new(Shared {
            signing: cfg.signing,
            kem_dk: cfg.static_kem,
            filter: cfg.peer_filter,
        });
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        tokio::spawn(engine(listener, shared, registry.clone(), events_tx, cmd_rx));
        Ok((Host { cmd: cmd_tx, registry }, events_rx, local))
    }


    /// A single-shot dial attempt over the addresses in order (erratum
    /// 145; retry supervision is the node's).
    pub fn dial(&self, info: PeerInfo) -> Result<(), HostError> {
        self.cmd.send(Command::Dial(info)).map_err(|_| HostError::ShutDown)
    }


    /// Route a payload to a connected peer. A send racing a disconnect is
    /// lost silently; the Disconnected event is the failure surface.
    pub fn send(&self, peer: &PeerId, payload: Vec<u8>) -> Result<(), HostError> {
        let reg = self.registry.lock().map_err(|_| HostError::LockPoisoned)?;
        match reg.get(peer) {
            Some(tx) => tx.send(payload).map_err(|_| HostError::NotConnected(*peer)),
            None => Err(HostError::NotConnected(*peer)),
        }
    }


    pub fn is_connected(&self, peer: &PeerId) -> bool {
        self.registry.lock().map(|r| r.contains_key(peer)).unwrap_or(false)
    }


    pub fn shutdown(&self) {
        let _ = self.cmd.send(Command::Shutdown);
    }
}


async fn engine(
    listener: TcpListener,
    shared: Arc<Shared>,
    registry: Registry,
    events: mpsc::Sender<HostEvent>,
    mut cmd: mpsc::UnboundedReceiver<Command>,
) {
    let mut conns: JoinSet<()> = JoinSet::new();
    loop {
        tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, _addr)) => {
                    conns.spawn(serve(stream, shared.clone(), registry.clone(), events.clone()));
                }
                Err(_) => {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            },
            command = cmd.recv() => match command {
                Some(Command::Dial(info)) => {
                    conns.spawn(connect(info, shared.clone(), registry.clone(), events.clone()));
                }
                Some(Command::Shutdown) | None => break,
            },
        }
    }
    conns.abort_all();
}


fn entropy(buf: &mut [u8]) -> Result<(), HostError> {
    getrandom::getrandom(buf).map_err(|_| HostError::Entropy)
}


async fn serve(stream: TcpStream, shared: Arc<Shared>, registry: Registry, events: mpsc::Sender<HostEvent>) {
    let mut stream = stream;
    let mut enc = [0u8; 32];
    if entropy(&mut enc).is_err() {
        return;
    }
    let Ok((session, vk)) =
        accept_handshake(&mut stream, &shared.signing, &shared.kem_dk, &enc).await
    else {
        return;
    };
    let id = PeerId::of(&vk);
    if !(shared.filter)(&id) {
        return;
    }
    run_connected(&mut stream, session, &registry, &events).await;
}


async fn connect(info: PeerInfo, shared: Arc<Shared>, registry: Registry, events: mpsc::Sender<HostEvent>) {
    let mut stream = None;
    for addr in &info.addrs {
        if let Ok(s) = TcpStream::connect(addr).await {
            stream = Some(s);
            break;
        }
    }
    let mut stream = match stream {
        Some(s) => s,
        None => return,
    };
    let mut fresh = [0u8; 64];
    let mut enc = [0u8; 32];
    if entropy(&mut fresh).is_err() || entropy(&mut enc).is_err() {
        return;
    }
    let Ok((session, vk)) =
        dial_handshake(&mut stream, &shared.signing, &info.kem_ek, &fresh, &enc, Some(&info.vk))
            .await
    else {
        return;
    };
    let id = PeerId::of(&vk);
    if !(shared.filter)(&id) {
        return;
    }
    run_connected(&mut stream, session, &registry, &events).await;
}


fn register(registry: &Registry, id: PeerId) -> Option<mpsc::UnboundedReceiver<Vec<u8>>> {
    let mut reg = registry.lock().ok()?;
    if reg.contains_key(&id) {
        return None; // duplicate connection: the first wins (erratum 145)
    }
    let (tx, rx) = mpsc::unbounded_channel();
    reg.insert(id, tx);
    Some(rx)
}


fn deregister(registry: &Registry, id: &PeerId) {
    if let Ok(mut reg) = registry.lock() {
        reg.remove(id);
    }
}


async fn run_connected<S: AsyncRead + AsyncWrite + Unpin>(
    stream: &mut S,
    session: Session,
    registry: &Registry,
    events: &mpsc::Sender<HostEvent>,
) {
    let id = PeerId::of(session.peer());
    let mut session = session;
    let Some(mut outbound) = register(registry, id) else {
        return;
    };
    if events.send(HostEvent::Connected(id)).await.is_err() {
        deregister(registry, &id);
        return;
    }
    pump(stream, &mut session, &mut outbound, registry, events, id).await;
}


async fn pump<S: AsyncRead + AsyncWrite + Unpin>(
    stream: &mut S,
    session: &mut Session,
    outbound: &mut mpsc::UnboundedReceiver<Vec<u8>>,
    registry: &Registry,
    events: &mpsc::Sender<HostEvent>,
    id: PeerId,
) {
    loop {
        tokio::select! {
            out = outbound.recv() => match out {
                Some(payload) => {
                    if session.send_frame(stream, &payload).await.is_err() {
                        break;
                    }
                }
                None => break,
            },
            frame = session.recv_frame(stream) => match frame {
                Ok(payload) => {
                    if events.send(HostEvent::Frame(id, payload)).await.is_err() {
                        break;
                    }
                }
                Err(_) => break,
            },
        }
    }
    deregister(registry, &id);
    let _ = events.send(HostEvent::Disconnected(id)).await;
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::node_keys;
    use std::time::Duration;


    fn addr0() -> SocketAddr {
        "127.0.0.1:0".parse().unwrap()
    }


    fn host_setup(seed: u64, allow: bool) -> (HostConfig, VerifyingKey, EncapsulationKey) {
        let (sk, ek, dk) = node_keys(seed);
        let vk = *sk.verifying_key();
        let cfg = HostConfig::new(sk, dk).with_filter(Arc::new(move |_| allow));
        (cfg, vk, ek)
    }


    async fn next_event(rx: &mut mpsc::Receiver<HostEvent>) -> HostEvent {
        tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("event timeout")
            .expect("event channel open")
    }


    async fn no_event(rx: &mut mpsc::Receiver<HostEvent>) {
        assert!(
            tokio::time::timeout(Duration::from_millis(300), rx.recv())
                .await
                .is_err(),
            "unexpected event"
        );
    }


    #[tokio::test]
    async fn two_hosts_roundtrip_both_directions() {
        let (cfg_a, vk_a, _) = host_setup(1, true);
        let (cfg_b, vk_b, ek_b) = host_setup(2, true);
        let (h_a, mut ev_a, _) = Host::bind(cfg_a, addr0()).await.unwrap();
        let (h_b, mut ev_b, addr_b) = Host::bind(cfg_b, addr0()).await.unwrap();
        let (id_a, id_b) = (PeerId::of(&vk_a), PeerId::of(&vk_b));


        h_a.dial(PeerInfo { vk: vk_b, kem_ek: ek_b, addrs: vec![addr_b] }).unwrap();
        assert_eq!(next_event(&mut ev_a).await, HostEvent::Connected(id_b));
        assert_eq!(next_event(&mut ev_b).await, HostEvent::Connected(id_a));
        assert!(h_a.is_connected(&id_b) && h_b.is_connected(&id_a));


        h_a.send(&id_b, b"ping".to_vec()).unwrap();
        assert_eq!(next_event(&mut ev_b).await, HostEvent::Frame(id_a, b"ping".to_vec()));
        h_b.send(&id_a, b"pong".to_vec()).unwrap();
        assert_eq!(next_event(&mut ev_a).await, HostEvent::Frame(id_b, b"pong".to_vec()));


        let big = vec![0x77u8; 100_000];
        h_a.send(&id_b, big.clone()).unwrap();
        assert_eq!(next_event(&mut ev_b).await, HostEvent::Frame(id_a, big));


        // Send to an unknown peer fails.
        let (_, vk_x, _) = host_setup(9, true);
        assert!(matches!(
            h_a.send(&PeerId::of(&vk_x), vec![]),
            Err(HostError::NotConnected(_))
        ));


        // Shutdown drops the peer's connection too.
        h_a.shutdown();
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(next_event(&mut ev_b).await, HostEvent::Disconnected(id_a));
        assert!(matches!(h_a.dial(PeerInfo { vk: vk_b, kem_ek: ek_b, addrs: vec![addr_b] }),
                   Err(HostError::ShutDown)));
        assert_eq!(ev_a.recv().await, None);
    }


    #[tokio::test]
    async fn rejecting_filter_closes_the_connection() {
        let (cfg_a, vk_a, _) = host_setup(1, true);
        let (cfg_b, vk_b, ek_b) = host_setup(2, false);
        let (h_a, mut ev_a, _) = Host::bind(cfg_a, addr0()).await.unwrap();
        let (h_b, mut ev_b, addr_b) = Host::bind(cfg_b, addr0()).await.unwrap();
        let (id_a, id_b) = (PeerId::of(&vk_a), PeerId::of(&vk_b));


        h_a.dial(PeerInfo { vk: vk_b, kem_ek: ek_b, addrs: vec![addr_b] }).unwrap();
        assert_eq!(next_event(&mut ev_a).await, HostEvent::Connected(id_b));
        assert_eq!(next_event(&mut ev_a).await, HostEvent::Disconnected(id_b));
        no_event(&mut ev_b).await;
        let _ = id_a;
    }


    #[tokio::test]
    async fn unreachable_dial_is_silent() {
        let (cfg_a, _, _) = host_setup(1, true);
        let (h_a, mut ev_a, _) = Host::bind(cfg_a, addr0()).await.unwrap();
        let (_, vk_b, ek_b) = host_setup(2, true);
        let dead: SocketAddr = "127.0.0.1:1".parse().unwrap();
        h_a.dial(PeerInfo { vk: vk_b, kem_ek: ek_b, addrs: vec![dead] }).unwrap();
        no_event(&mut ev_a).await;
    }


    #[tokio::test]
    async fn both_sides_dial_simultaneously() {
        let (cfg_a, vk_a, ek_a) = host_setup(1, true);
        let (cfg_b, vk_b, ek_b) = host_setup(2, true);
        let (h_a, mut ev_a, addr_a) = Host::bind(cfg_a, addr0()).await.unwrap();
        let (h_b, mut ev_b, addr_b) = Host::bind(cfg_b, addr0()).await.unwrap();
        h_a.dial(PeerInfo { vk: vk_b, kem_ek: ek_b, addrs: vec![addr_b] }).unwrap();
        h_b.dial(PeerInfo { vk: vk_a, kem_ek: ek_a, addrs: vec![addr_a] }).unwrap();
        // Each side sees at least one Connected and working frames.
        let (id_a, id_b) = (PeerId::of(&vk_a), PeerId::of(&vk_b));
        let mut got_a = false;
        let mut got_b = false;
        for _ in 0..4 {
            match next_event(&mut ev_a).await {
                HostEvent::Connected(id) if id == id_b => got_a = true,
                _ => {}
            }
            match next_event(&mut ev_b).await {
                HostEvent::Connected(id) if id == id_a => got_b = true,
                _ => {}
            }
            if got_a && got_b {
                break;
            }
        }
        assert!(got_a && got_b);
        h_a.send(&id_b, b"x".to_vec()).unwrap();
        let _ = next_event(&mut ev_b).await;
    }


    #[test]
    fn peer_book_merge_and_lookup() {
        let (sk, ek, _) = node_keys(5);
        let vk = *sk.verifying_key();
        let a1: SocketAddr = "10.0.0.1:9000".parse().unwrap();
        let a2: SocketAddr = "10.0.0.2:9000".parse().unwrap();
        let mut book = PeerBook::new();
        assert!(book.is_empty());
        book.insert(PeerInfo { vk, kem_ek: ek.clone(), addrs: vec![a1] });
        let id = PeerId::of(&vk);
        assert_eq!(book.len(), 1);
        assert!(book.contains(&id));
        assert_eq!(book.get(&id).unwrap().addrs, vec![a1]);
        book.insert(PeerInfo { vk, kem_ek: ek, addrs: vec![a1, a2] });
        assert_eq!(book.len(), 1, "merge, not duplicate");
        assert_eq!(book.get(&id).unwrap().addrs, vec![a1, a2], "addresses deduplicated");
        assert_eq!(book.ids().collect::<Vec<_>>(), vec![id]);
        assert!(book.remove(&id).is_some());
        assert!(book.is_empty() && book.get(&id).is_none());
        assert_ne!(id, PeerId::of(&node_keys(6).0.verifying_key()));
        assert_eq!(id.to_string(), id.as_hash().to_string());
    }


    #[test]
    fn peer_id_is_the_pinned_formula() {
        let (sk, _, _) = node_keys(7);
        let id = PeerId::of(sk.verifying_key());
        let mut pre = Vec::new();
        pre.extend_from_slice(NET_PEER.as_bytes());
        pre.extend_from_slice(sk.verifying_key().as_bytes());
        assert_eq!(id.as_hash().as_bytes(), blake3::hash(&pre).as_bytes());
    }
}

/// The application-level payload framing inside an encrypted session
/// (erratum 207). The wire format is a single discriminator tag byte
/// followed by the body of the inner message; this enum is the node's
/// in-process view of "what arrived", independent of which transport
/// surface carried it.
///
/// Tags share a single 0..=8 namespace with the inner message types so
/// the wire bytes are unambiguous: 0,1,2,6,7,8 → `GossipMessage`;
/// 3 → `SubmissionMessage`; 4 → `Packet`; 5 → `FragmentFrame`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostFrame {
    Gossip(GossipMessage),
    Submission(SubmissionMessage),
    MixPacket(Packet),
    MixFragment(FragmentFrame),
}

impl HostFrame {
    /// Decode a session payload into the variant the tag selects.
    /// The caller is responsible for handing us the AEAD-opened bytes
    /// (the wire layer peels encryption before we see them).
    pub fn decode(payload: &[u8]) -> Result<HostFrame, CodecError> {
        let Some(&tag) = payload.first() else {
            return Err(CodecError::Truncated);
        };
        match tag {
            MIX_PACKET_TAG => {
                let pkt = Packet::decode(&payload[1..])?;
                Ok(HostFrame::MixPacket(pkt))
            }
            MIX_FRAGMENT_TAG => {
                let frag = FragmentFrame::decode(&payload[1..])?;
                Ok(HostFrame::MixFragment(frag))
            }
            // 3 is the SubmissionMessage tag (SUBMISSION_TAG).
            // All other tags in 0..=8 belong to GossipMessage.
            // Both inner decoders read the tag themselves, so we hand
            // them the full buffer.
            3 => {
                let s = SubmissionMessage::decode(payload)?;
                Ok(HostFrame::Submission(s))
            }
            _ => {
                let g = GossipMessage::decode(payload)?;
                Ok(HostFrame::Gossip(g))
            }
        }
    }
}
