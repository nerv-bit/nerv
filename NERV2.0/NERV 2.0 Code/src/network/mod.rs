//! P2P Networking.
//!
//! Built on rust-libp2p with custom ML-KEM-768 Noise handshakes
//! for post-quantum transport security. The mempool holds only
//! threshold-encrypted transactions — no plaintext until block execution.
//!
//! Submodules:
//! - `dht` — Distributed Hash Table (Kademlia)
//! - `gossip` — Message propagation (Gossipsub)
//! - `mempool` — Threshold-encrypted mempool
//!
//! P2P Networking — libp2p with Post-Quantum Transport.
//!
//! NERV's network layer uses libp2p with custom ML-KEM-768 Noise
//! handshakes for post-quantum transport security. The architecture
//! separates four concerns:
//!
//! | Component | Protocol | Purpose |
//! |-----------|----------|---------|
//! | DHT | Kademlia | Peer discovery, shard data location, VDW storage |
//! | Gossip | Gossipsub | Block propagation, votes, gradient submissions |
//! | Mempool | Local | Threshold-encrypted transaction pool |
//! | Transport | QUIC+Noise | PQ-secure connections between peers |
//!
//! # V2.0 Changes
//!
//! - Transport: ML-KEM-768 Noise handshake (replaces X25519)
//! - Mempool: Threshold-encrypted (no plaintext until ceremony)
//! - No TEE dependency for any network operation

pub mod dht;
pub mod gossip;
pub mod mempool;

// ─── Re-exports ─────────────────────────────────────────────────────────

pub use dht::{DhtKey, DhtRecord, DhtService};
pub use gossip::{
    GossipTopic, GossipMessage, GossipValidator,
};
pub use mempool::{
    EncryptedMempool, MempoolEntry, MempoolPriority,
};

use crate::{
    SPHINX_HOPS, BATCH_MAX_SIZE,
    ShardId, BlockHeight, ValidatorId, TxHash,
    NervError, NervResult,
};
use crate::config::NetworkConfig;
use serde::{$Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Instant;

// ─── Peer Information ────────────────────────────────────────────────────

/// Information about a connected peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    /// The peer's unique identifier (BLAKE3 of their Dilithium-3 PK).
    pub peer_id: Vec<u8>,

    /// Multiaddrs the peer is reachable at.
    pub addresses: Vec<String>,

    /// The peer's declared validator ID (if they are a validator).
    pub validator_id: Option<ValidatorId>,

    /// The shard(s) this peer is serving (if any).
    pub assigned_shards: Vec<ShardId>,

    /// Connection establishment time.
    pub connected_at: u64,

    /// Last time we received a message from this peer.
    pub last_message_at: u64,

    /// Bytes sent to this peer.
    pub bytes_sent: u64,

    /// Bytes received from this peer.
    pub bytes_received: u64,

    /// Number of protocol violations detected.
    pub violations: u32,

    /// Reputation score (0–1000, affects message prioritization).
    pub reputation: u32,

    /// Whether the connection is encrypted with PQ transport.
    pub pq_encrypted: bool,
}

impl PeerInfo {
    /// Create a new peer info entry.
    pub fn new(peer_id: Vec<u8>) -> Self {
        Self {
            peer_id,
            addresses: Vec::new(),
            validator_id: None,
            assigned_shards: Vec::new(),
            connected_at: current_epoch_millis(),
            last_message_at: current_epoch_millis(),
            bytes_sent: 0,
            bytes_received: 0,
            violations: 0,
            reputation: 500, // Default mid-range
            pq_encrypted: false,
        }
    }

    /// Record bytes sent.
    pub fn record_sent(&mut self, bytes: u64) {
        self.bytes_sent = self.bytes_sent.saturating_add(bytes);
        self.last_message_at = current_epoch_millis();
    }

    /// Record bytes received.
    pub fn record_received(&mut self, bytes: u64) {
        self.bytes_received = self.bytes_received.saturating_add(bytes);
        self.last_message_at = current_epoch_millis();
    }

    /// Record a protocol violation.
    pub fn record_violation(&mut self) {
        self.violations = self.violations.saturating_add(1);
        self.reputation = self.reputation.saturating_sub(50);
    }

    /// Check if the peer should be disconnected (too many violations).
    pub fn should_disconnect(&self, max_violations: u32) -> bool {
        self.violations >= max_violations
    }

    /// Seconds since last message.
    pub fn seconds_since_last_message(&self) -> u64 {
        let now = current_epoch_millis();
        now.saturating_sub(self.last_message_at) / 1000
    }
}

// ─── Network Events ──────────────────────────────────────────────────────

/// Events emitted by the network service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkEvent {
    /// A new peer connected.
    PeerConnected {
        /// Peer information.
        peer: PeerInfo,
    },

    /// A peer disconnected.
    PeerDisconnected {
        /// Peer identifier.
        peer_id: Vec<u8>,
        /// Reason for disconnection.
        reason: DisconnectReason,
    },

    /// A gossip message was received.
    GossipMessage {
        /// The topic.
        topic: GossipTopic,
        /// The message payload.
        message: Vec<u8>,
        /// Source peer.
        source: Vec<u8>,
    },

    /// A DHT record was retrieved.
    DhtRecordRetrieved {
        /// The key that was looked up.
        key: Vec<u8>,
        /// The record value.
        value: Vec<u8>,
    },

    /// A new encrypted transaction arrived in the mempool.
    MempoolTransaction {
        /// Transaction hash (of the encrypted blob).
        tx_hash: TxHash,
        /// Fee amount (unencrypted, for ordering).
        fee_nano: u64,
    },

    /// A peer was discovered via DHT.
    PeerDiscovered {
        /// Peer identifier.
        peer_id: Vec<u8>,
        /// Addresses.
        addresses: Vec<String>,
    },
}

/// Reason for peer disconnection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisconnectReason {
    /// Peer initiated disconnect.
    PeerInitiated,
    /// Connection timed out.
    Timeout,
    /// Too many protocol violations.
    Violations,
    /// Network shutdown.
    Shutdown,
    /// Internal error.
    Error,
}

impl std::fmt::Display for DisconnectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PeerInitiated => write!(f, "peer_initiated"),
            Self::Timeout => write!(f, "timeout"),
            Self::Violations => write!(f, "violations"),
            Self::Shutdown => write!(f, "shutdown"),
            Self::Error => write!(f, "error"),
        }
    }
}

// ─── Connection Manager ──────────────────────────────────────────────────

/// Manages peer connections and tracks peer state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionManager {
    /// Connected peers (peer_id → info).
    peers: HashMap<Vec<u8>, PeerInfo>,

    /// Maximum inbound connections.
    max_inbound: usize,

    /// Maximum outbound connections.
    max_outbound: usize,

    /// Number of inbound connections.
    inbound_count: usize,

    /// Number of outbound connections.
    outbound_count: usize,

    /// Maximum violations before disconnect.
    max_violations: u32,
}

impl ConnectionManager {
    /// Create a new connection manager.
    pub fn new(max_inbound: usize, max_outbound: usize, max_violations: u32) -> Self {
        Self {
            peers: HashMap::new(),
            max_inbound,
            max_outbound,
            inbound_count: 0,
            outbound_count: 0,
            max_violations,
        }
    }

    /// Register a connected peer.
    pub fn peer_connected(&mut self, peer: PeerInfo) -> NervResult<()> {
        if self.peers.len() >= self.max_inbound + self.max_outbound {
            return Err(NervError::Network("connection limit reached".into()));
        }
        self.peers.insert(peer.peer_id.clone(), peer);
        Ok(())
    }

    /// Record a peer disconnection.
    pub fn peer_disconnected(&mut self, peer_id: &[u8]) -> Option<PeerInfo> {
        self.peers.remove(peer_id)
    }

    /// Get peer info.
    pub fn get_peer(&self, peer_id: &[u8]) -> Option<&PeerInfo> {
        self.peers.get(peer_id)
    }

    /// Get mutable peer info.
    pub fn get_peer_mut(&mut self, peer_id: &[u8]) -> Option<&mut PeerInfo> {
        self.peers.get_mut(peer_id)
    }

    /// List all connected peer IDs.
    pub fn connected_peers(&self) -> Vec<Vec<u8>> {
        self.peers.keys().cloned().collect()
    }

    /// Number of connected peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Check if we can accept more inbound connections.
    pub fn can_accept_inbound(&self) -> bool {
        self.inbound_count < self.max_inbound
    }

    /// Check for peers that should be disconnected due to violations.
    pub fn violators(&self) -> Vec<Vec<u8>> {
        self.peers.values()
            .filter(|p| p.should_disconnect(self.max_violations))
            .map(|p| p.peer_id.clone())
            .collect()
    }

    /// Get peers serving a specific shard.
    pub fn peers_for_shard(&self, shard_id: ShardId) -> Vec<&PeerInfo> {
        self.peers.values()
            .filter(|p| p.assigned_shards.contains(&shard_id))
            .collect()
    }

    /// Total bytes sent across all peers.
    pub fn total_bytes_sent(&self) -> u64 {
        self.peers.values().map(|p| p.bytes_sent).sum()
    }

    /// Total bytes received across all peers.
    pub fn total_bytes_received(&self) -> u64 {
        self.peers.values().map(|p| p.bytes_received).sum()
    }
}

// ─── Network Service ─────────────────────────────────────────────────────

/// The top-level P2P network service.
///
/// Coordinates all networking: DHT, gossip, mempool, and
/// connection management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkService {
    /// Connection manager.
    pub connections: ConnectionManager,

    /// DHT service.
    pub dht: DhtService,

    /// Encrypted mempool.
    pub mempool: EncryptedMempool,

    /// Network configuration.
    pub config: NetworkConfig,

    /// Local peer ID.
    pub local_peer_id: Vec<u8>,

    /// Network service is running.
    pub is_running: bool,

    /// Total messages sent.
    pub messages_sent: u64,

    /// Total messages received.
    pub messages_received: u64,
}

impl NetworkService {
    /// Create a new network service.
    pub fn new(config: NetworkConfig, mempool_max_entries: usize) -> Self {
        let local_peer_id = crate::utils::random_bytes(32);
        Self {
            connections: ConnectionManager::new(
                config.max_inbound,
                config.max_outbound,
                10, // Max violations
            ),
            dht: DhtService::new(config.kad_replication_factor),
            mempool: EncryptedMempool::new(mempool_max_entries, BATCH_MAX_SIZE),
            config,
            local_peer_id,
            is_running: false,
            messages_sent: 0,
            messages_received: 0,
        }
    }

    /// Start the network service.
    pub fn start(&mut self) -> NervResult<()> {
        if self.is_running {
            return Err(NervError::Network("already running".into()));
        }
        self.is_running = true;
        Ok(())
    }

    /// Stop the network service.
    pub fn stop(&mut self) {
        self.is_running = false;
    }

    /// Broadcast a gossip message.
    pub fn gossip_broadcast(
        &mut self,
        topic: GossipTopic,
        message: Vec<u8>,
    ) -> NervResult<()> {
        if !self.is_running {
            return Err(NervError::Network("not running".into()));
        }
        self.messages_sent += 1;
        // In production: libp2p gossipsub publish
        Ok(())
    }

    /// Send a direct message to a peer.
    pub fn direct_send(
        &mut self,
        peer_id: &[u8],
        message: Vec<u8>,
    ) -> NervResult<()> {
        if !self.is_running {
            return Err(NervError::Network("not running".into()));
        }
        if let Some(peer) = self.connections.get_peer_mut(peer_id) {
            peer.record_sent(message.len() as u64);
            self.messages_sent += 1;
            Ok(())
        } else {
            Err(NervError::Network(format!(
                "peer {} not connected", hex::encode(peer_id)
            )))
        }
    }

    /// Get network stats.
    pub fn stats(&self) -> NetworkStats {
        NetworkStats {
            peer_count: self.connections.peer_count(),
            mempool_size: self.mempool.len(),
            messages_sent: self.messages_sent,
            messages_received: self.messages_received,
            bytes_sent: self.connections.total_bytes_sent(),
            bytes_received: self.connections.total_bytes_received(),
            dht_records: self.dht.record_count(),
        }
    }
}

/// Network statistics snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkStats {
    pub peer_count: usize,
    pub mempool_size: usize,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub dht_records: usize,
}

// ─── Helpers ─────────────────────────────────────────────────────────────

fn current_epoch_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_info_new() {
        let peer = PeerInfo::new(vec![1u8; 32]);
        assert_eq!(peer.violations, 0);
        assert_eq!(peer.reputation, 500);
        assert!(!peer.pq_encrypted);
    }

    #[test]
    fn test_peer_info_violation() {
        let mut peer = PeerInfo::new(vec![1u8; 32]);
        peer.record_violation();
        assert_eq!(peer.violations, 1);
        assert_eq!(peer.reputation, 450);
    }

    #[test]
    fn test_peer_info_should_disconnect() {
        let mut peer = PeerInfo::new(vec![1u8; 32]);
        for _ in 0..10 {
            peer.record_violation();
        }
        assert!(peer.should_disconnect(10));
        assert!(!peer.should_disconnect(11));
    }

    #[test]
    fn test_connection_manager_basic() {
        let mut cm = ConnectionManager::new(100, 50, 10);
        let peer = PeerInfo::new(vec![1u8; 32]);
        cm.peer_connected(peer).unwrap();
        assert_eq!(cm.peer_count(), 1);
    }

    #[test]
    fn test_connection_manager_disconnect() {
        let mut cm = ConnectionManager::new(100, 50, 10);
        let peer = PeerInfo::new(vec![1u8; 32]);
        let pid = peer.peer_id.clone();
        cm.peer_connected(peer).unwrap();
        cm.peer_disconnected(&pid);
        assert_eq!(cm.peer_count(), 0);
    }

    #[test]
    fn test_connection_manager_shard_peers() {
        let mut cm = ConnectionManager::new(100, 50, 10);
        let mut peer = PeerInfo::new(vec![1u8; 32]);
        peer.assigned_shards.push(ShardId::new(1));
        cm.peer_connected(peer).unwrap();
        assert_eq!(cm.peers_for_shard(ShardId::new(1)).len(), 1);
        assert_eq!(cm.peers_for_shard(ShardId::new(2)).len(), 0);
    }

    #[test]
    fn test_network_service_creation() {
        let config = NetworkConfig::default();
        let service = NetworkService::new(config, 500_000);
        assert!(!service.is_running);
    }

    #[test]
    fn test_network_service_start_stop() {
        let config = NetworkConfig::default();
        let mut service = NetworkService::new(config, 500_000);
        service.start().unwrap();
        assert!(service.is_running);
        assert!(service.start().is_err()); // Double start
        service.stop();
        assert!(!service.is_running);
    }

    #[test]
    fn test_disconnect_reason_display() {
        assert_eq!(DisconnectReason::Timeout.to_string(), "timeout");
    }
}

