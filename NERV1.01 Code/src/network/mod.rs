//! NERV P2P Networking Module
//!
//! This module provides the peer-to-peer networking infrastructure for the NERV blockchain.
//! It includes distributed hash table (DHT) for peer discovery, gossip protocol for
//! message propagation, and encrypted transaction pool management.
//!
//! Key Features:
//! - Kademlia DHT for efficient peer discovery and content routing
//! - Gossipsub-like protocol for efficient message propagation
//! - Encrypted mempool with TEE-protected transaction handling
//! - Multi-transport support (QUIC, WebRTC, TCP with Noise encryption)
//! - Rate limiting and DoS protection


mod dht;
mod gossip;
mod mempool;


pub use dht::{DhtManager, DhtConfig, PeerId, PeerInfo, DhtQuery, DhtError};
pub use gossip::{GossipManager, GossipConfig, MessageId, Topic, GossipError};
pub use mempool::{MempoolManager, MempoolConfig, TxPriority, MempoolError};


use crate::crypto::{CryptoProvider, Dilithium3, MlKem768, ByteSerializable};
use crate::params::{DEFAULT_P2P_PORT, DEFAULT_RPC_PORT, DEFAULT_METRICS_PORT, DHT_BUCKET_SIZE};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn, debug, error, trace};


/// Network manager coordinating all networking components
pub struct NetworkManager {
    /// DHT for peer discovery and content routing
    dht_manager: Arc<DhtManager>,
    
    /// Gossip protocol for message propagation
    gossip_manager: Arc<GossipManager>,
    
    /// Encrypted transaction pool
    mempool_manager: Arc<MempoolManager>,
    
    /// Cryptographic provider
    crypto_provider: Arc<CryptoProvider>,
    
    /// Connected peers
    peers: RwLock<HashMap<PeerId, PeerConnection>>,
    
    /// Network configuration
    config: NetworkConfig,
    
    /// Incoming message channels
    message_channels: RwLock<HashMap<Topic, Vec<mpsc::UnboundedSender<Vec<u8>>>>>,
    
    /// Metrics collector
    metrics: NetworkMetrics,
    
    /// Shutdown signal
    shutdown_signal: tokio::sync::watch::Sender<bool>,
}


impl NetworkManager {
    /// Create a new network manager
    pub async fn new(config: NetworkConfig) -> Result<Self, NetworkError> {
        info!("Initializing NERV network manager on {}:{}", 
              config.listen_addr, config.p2p_port);
        
        // Initialize cryptographic provider
        let crypto_provider = Arc::new(CryptoProvider::new()
            .map_err(|e| NetworkError::CryptoError(e.to_string()))?);
        
        // Initialize DHT
        let dht_config = DhtConfig {
            bucket_size: DHT_BUCKET_SIZE,
            replication_factor: 5,
            query_timeout_ms: 5000,
            bootstrap_nodes: config.bootstrap_nodes.clone(),
            enable_relay: true,
            max_peers: config.max_peers,
            min_peers: config.min_peers,
        };
        let dht_manager = Arc::new(DhtManager::new(dht_config, crypto_provider.clone()).await?);
        
        // Initialize gossip protocol
        let gossip_config = GossipConfig {
            fanout: config.gossip_fanout,
            heartbeat_interval_ms: config.gossip_heartbeat_ms,
            message_cache_size: 1000,
            duplicate_cache_size: 5000,
            max_message_size: 1024 * 1024, // 1MB
            validation_queue_size: 100,
        };
        let gossip_manager = Arc::new(GossipManager::new(gossip_config, crypto_provider.clone()).await?);
        
        // Initialize mempool
        let mempool_config = MempoolConfig {
            max_size_bytes: config.mempool_max_size_mb * 1024 * 1024,
            max_transactions: config.mempool_max_txs,
            priority_fee_bump_percent: config.priority_fee_bump_percent,
            eviction_batch_size: config.mempool_eviction_batch,
            ttl_seconds: config.mempool_ttl_seconds,
            encryption_required: true,
            tee_attestation_required: config.tee_attestation_required,
        };
        let mempool_manager = Arc::new(MempoolManager::new(mempool_config, crypto_provider.clone()).await?);
        
        // Create shutdown signal
        let (shutdown_sender, _) = tokio::sync::watch::channel(false);
        
        Ok(Self {
            dht_manager,
            gossip_manager,
            mempool_manager,
            crypto_provider,
            peers: RwLock::new(HashMap::new()),
            config,
            message_channels: RwLock::new(HashMap::new()),
            metrics: NetworkMetrics::new(),
            shutdown_signal: shutdown_sender,
        })
    }
    
    /// Start the network manager
    pub async fn start(&self) -> Result<(), NetworkError> {
        info!("Starting NERV network manager");
        
        // Start DHT
        self.dht_manager.start().await?;
        
        // Start gossip protocol
        self.gossip_manager.start().await?;
        
        // Start mempool
        self.mempool_manager.start().await?;
        
        // Connect to bootstrap nodes
        self.connect_to_bootstrap_nodes().await?;
        
        // Start peer management tasks
        self.start_peer_tasks().await?;
        
        // Start metrics collection
        self.start_metrics_task().await?;
        
        info!("Network manager started successfully");
        Ok(())
    }
    
    /// Stop the network manager
    pub async fn stop(&self) -> Result<(), NetworkError> {
        info!("Stopping NERV network manager");
        
        // Send shutdown signal
        self.shutdown_signal.send(true)
            .map_err(|e| NetworkError::ShutdownError(e.to_string()))?;
        
        // Stop components
        self.dht_manager.stop().await?;
        self.gossip_manager.stop().await?;
        self.mempool_manager.stop().await?;
        
        // Disconnect from all peers
        let mut peers = self.peers.write().await;
        for (peer_id, connection) in peers.drain() {
            connection.disconnect().await;
            debug!("Disconnected from peer {}", peer_id);
        }
        
        info!("Network manager stopped successfully");
        Ok(())
    }
    
    /// Broadcast a message to the network
    pub async fn broadcast(&self, topic: Topic, message: Vec<u8>) -> Result<MessageId, NetworkError> {
        let start_time = std::time::Instant::now();
        
        // Validate message
        if message.len() > self.config.max_message_size {
            return Err(NetworkError::MessageTooLarge(message.len(), self.config.max_message_size));
        }
        
        // Broadcast via gossip protocol
        let message_id = self.gossip_manager.broadcast(topic.clone(), message).await?;
        
        // Update metrics
        self.metrics.record_broadcast(start_time.elapsed(), topic.clone());
        
        debug!("Broadcast message {} to topic {}", message_id, topic);
        Ok(message_id)
    }
    
    /// Subscribe to a topic
    pub async fn subscribe(&self, topic: Topic) -> Result<mpsc::UnboundedReceiver<Vec<u8>>, NetworkError> {
        let (sender, receiver) = mpsc::unbounded_channel();
        
        let mut channels = self.message_channels.write().await;
        let topic_channels = channels.entry(topic.clone()).or_insert_with(Vec::new);
        topic_channels.push(sender);
        
        // Register with gossip manager
        self.gossip_manager.subscribe(topic).await?;
        
        info!("Subscribed to topic: {}", topic);
        Ok(receiver)
    }
    
    /// Unsubscribe from a topic
    pub async fn unsubscribe(&self, topic: Topic) -> Result<(), NetworkError> {
        let mut channels = self.message_channels.write().await;
        channels.remove(&topic);
        
        // Unregister from gossip manager
        self.gossip_manager.unsubscribe(topic.clone()).await?;
        
        info!("Unsubscribed from topic: {}", topic);
        Ok(())
    }
    
    /// Submit a transaction to the mempool
    pub async fn submit_transaction(
        &self,
        encrypted_tx: Vec<u8>,
        attestation: Vec<u8>,
        priority: TxPriority,
    ) -> Result<TxReceipt, NetworkError> {
        let start_time = std::time::Instant::now();
        
        let receipt = self.mempool_manager.submit_transaction(encrypted_tx, attestation, priority).await?;
        
        // Update metrics
        self.metrics.record_tx_submission(start_time.elapsed());
        
        // Optionally gossip to peers
        if self.config.gossip_transactions {
            let topic = Topic::from("transactions");
            let message = serde_json::to_vec(&receipt)
                .map_err(|e| NetworkError::SerializationError(e.to_string()))?;
            self.gossip_manager.broadcast(topic, message).await?;
        }
        
        Ok(receipt)
    }
    
    /// Get transactions from mempool (for validators)
    pub async fn get_transactions(&self, shard_id: u64, limit: usize) -> Result<Vec<EncryptedTransaction>, NetworkError> {
        self.mempool_manager.get_transactions(shard_id, limit).await
    }
    
    /// Lookup a peer in the DHT
    pub async fn lookup_peer(&self, peer_id: &PeerId) -> Result<Option<PeerInfo>, NetworkError> {
        self.dht_manager.lookup_peer(peer_id).await
            .map_err(NetworkError::from)
    }
    
    /// Lookup a value in the DHT
    pub async fn lookup_value(&self, key: &[u8]) -> Result<Option<Vec<u8>>, NetworkError> {
        self.dht_manager.lookup_value(key).await
            .map_err(NetworkError::from)
    }
    
    /// Store a value in the DHT
    pub async fn store_value(&self, key: Vec<u8>, value: Vec<u8>, ttl_seconds: Option<u64>) -> Result<(), NetworkError> {
        self.dht_manager.store_value(key, value, ttl_seconds).await
            .map_err(NetworkError::from)
    }
    
    /// Get network statistics
    pub async fn get_stats(&self) -> NetworkStats {
        let peers = self.peers.read().await;
        
        NetworkStats {
            total_peers: peers.len(),
            connected_peers: peers.values().filter(|p| p.is_connected()).count(),
            inbound_connections: peers.values().filter(|p| p.is_inbound()).count(),
            outbound_connections: peers.values().filter(|p| !p.is_inbound()).count(),
            bytes_sent: self.metrics.bytes_sent.load(std::sync::atomic::Ordering::Relaxed),
            bytes_received: self.metrics.bytes_received.load(std::sync::atomic::Ordering::Relaxed),
            messages_sent: self.metrics.messages_sent.load(std::sync::atomic::Ordering::Relaxed),
            messages_received: self.metrics.messages_received.load(std::sync::atomic::Ordering::Relaxed),
            dht_queries: self.dht_manager.get_query_count().await,
            gossip_messages: self.gossip_manager.get_message_count().await,
            mempool_size: self.mempool_manager.get_size().await,
        }
    }
    
    /// Handle incoming message from gossip protocol
    async fn handle_gossip_message(&self, topic: Topic, message: Vec<u8>) -> Result<(), NetworkError> {
        // Update metrics
        self.metrics.record_message_received(message.len());
        
        // Dispatch to subscribers
        let channels = self.message_channels.read().await;
        if let Some(topic_channels) = channels.get(&topic) {
            for channel in topic_channels {
                if let Err(e) = channel.send(message.clone()) {
                    warn!("Failed to send message to subscriber: {}", e);
                }
            }
        }
        
        Ok(())
    }
    
    // Internal methods
    
    async fn connect_to_bootstrap_nodes(&self) -> Result<(), NetworkError> {
        if self.config.bootstrap_nodes.is_empty() {
            warn!("No bootstrap nodes configured");
            return Ok(());
        }
        
        info!("Connecting to {} bootstrap nodes", self.config.bootstrap_nodes.len());
        
        let mut connected = 0;
        for addr in &self.config.bootstrap_nodes {
            match self.connect_to_peer(addr).await {
                Ok(_) => {
                    connected += 1;
                    info!("Connected to bootstrap node: {}", addr);
                }
                Err(e) => {
                    warn!("Failed to connect to bootstrap node {}: {}", addr, e);
                }
            }
            
            // Small delay between connections
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        
        if connected == 0 {
            warn!("Failed to connect to any bootstrap nodes");
        } else {
            info!("Connected to {}/{} bootstrap nodes", connected, self.config.bootstrap_nodes.len());
        }
        
        Ok(())
    }
    
    async fn connect_to_peer(&self, addr: &str) -> Result<PeerId, NetworkError> {
        // Parse address
        let socket_addr: SocketAddr = addr.parse()
            .map_err(|e| NetworkError::InvalidAddress(e.to_string()))?;
        
        // Create connection
        let connection = PeerConnection::connect(socket_addr, self.crypto_provider.clone()).await?;
        let peer_id = connection.peer_id();
        
        // Add to peers map
        let mut peers = self.peers.write().await;
        peers.insert(peer_id.clone(), connection);
        
        // Announce to DHT
        self.dht_manager.add_peer(peer_id.clone(), socket_addr).await?;
        
        Ok(peer_id)
    }
    
    async fn start_peer_tasks(&self) -> Result<(), NetworkError> {
        // Start peer maintenance task
        let dht_manager = self.dht_manager.clone();
        let peers = self.peers.clone();
        let metrics = self.metrics.clone();
        
        tokio::spawn(async move {
            loop {
                // Refresh DHT routing table
                if let Err(e) = dht_manager.refresh_routing_table().await {
                    warn!("Failed to refresh DHT routing table: {}", e);
                }
                
                // Clean up disconnected peers
                let mut peers_guard = peers.write().await;
                let initial_count = peers_guard.len();
                peers_guard.retain(|_, conn| conn.is_connected());
                let removed = initial_count - peers_guard.len();
                
                if removed > 0 {
                    debug!("Removed {} disconnected peers", removed);
                }
                
                // Update metrics
                metrics.update_peer_count(peers_guard.len());
                
                // Sleep for 30 seconds
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            }
        });
        
        Ok(())
    }
    
    async fn start_metrics_task(&self) -> Result<(), NetworkError> {
        let metrics = self.metrics.clone();
        
        tokio::spawn(async move {
            loop {
                // Log metrics every 60 seconds
                metrics.log_summary();
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            }
        });
        
        Ok(())
    }
}


/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Listen address for P2P connections
    pub listen_addr: String,
    
    /// P2P port
    pub p2p_port: u16,
    
    /// RPC port
    pub rpc_port: u16,
    
    /// Metrics port
    pub metrics_port: u16,
    
    /// Bootstrap nodes for peer discovery
    pub bootstrap_nodes: Vec<String>,
    
    /// Maximum number of peers
    pub max_peers: usize,
    
    /// Minimum number of peers
    pub min_peers: usize,
    
    /// Gossip fanout parameter
    pub gossip_fanout: usize,
    
    /// Gossip heartbeat interval (ms)
    pub gossip_heartbeat_ms: u64,
    
    /// Maximum message size (bytes)
    pub max_message_size: usize,
    
    /// Mempool maximum size (MB)
    pub mempool_max_size_mb: usize,
    
    /// Mempool maximum transactions
    pub mempool_max_txs: usize,
    
    /// Priority fee bump percentage
    pub priority_fee_bump_percent: f64,
    
    /// Mempool eviction batch size
    pub mempool_eviction_batch: usize,
    
    /// Mempool TTL (seconds)
    pub mempool_ttl_seconds: u64,
    
    /// Enable transaction gossip
    pub gossip_transactions: bool,
    
    /// Require TEE attestation for transactions
    pub tee_attestation_required: bool,
    
    /// Enable QUIC transport
    pub enable_quic: bool,
    
    /// Enable WebRTC transport
    pub enable_webrtc: bool,
    
    /// Enable noise encryption
    pub enable_noise: bool,
    
    /// Rate limit (requests per second)
    pub rate_limit_rps: u32,
    
    /// Connection timeout (seconds)
    pub connection_timeout_secs: u64,
}


impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0".to_string(),
            p2p_port: DEFAULT_P2P_PORT,
            rpc_port: DEFAULT_RPC_PORT,
            metrics_port: DEFAULT_METRICS_PORT,
            bootstrap_nodes: vec![],
            max_peers: 50,
            min_peers: 5,
            gossip_fanout: 6,
            gossip_heartbeat_ms: 1000,
            max_message_size: 1024 * 1024, // 1MB
            mempool_max_size_mb: 100, // 100MB
            mempool_max_txs: 100_000,
            priority_fee_bump_percent: 10.0,
            mempool_eviction_batch: 100,
            mempool_ttl_seconds: 3600, // 1 hour
            gossip_transactions: true,
            tee_attestation_required: true,
            enable_quic: true,
            enable_webrtc: false,
            enable_noise: true,
            rate_limit_rps: 100,
            connection_timeout_secs: 30,
        }
    }
}


/// Peer connection information
#[derive(Debug, Clone)]
struct PeerConnection {
    /// Peer identifier
    peer_id: PeerId,
    
    /// Socket address
    socket_addr: SocketAddr,
    
    /// Connection direction
    direction: ConnectionDirection,
    
    /// Connection state
    state: ConnectionState,
    
    /// Connection metrics
    metrics: ConnectionMetrics,
    
    /// Cryptographic session
    session: Option<CryptoSession>,
}


impl PeerConnection {
    async fn connect(addr: SocketAddr, crypto: Arc<CryptoProvider>) -> Result<Self, NetworkError> {
        // In production: actually connect to peer
        // For now, simulate connection
        
        let peer_id = PeerId::random();
        
        Ok(Self {
            peer_id,
            socket_addr: addr,
            direction: ConnectionDirection::Outbound,
            state: ConnectionState::Connected,
            metrics: ConnectionMetrics::new(),
            session: None,
        })
    }
    
    async fn disconnect(&self) {
        // In production: close connection
    }
    
    fn is_connected(&self) -> bool {
        matches!(self.state, ConnectionState::Connected)
    }
    
    fn is_inbound(&self) -> bool {
        matches!(self.direction, ConnectionDirection::Inbound)
    }
    
    fn peer_id(&self) -> PeerId {
        self.peer_id.clone()
    }
}


/// Connection direction
#[derive(Debug, Clone, Copy, PartialEq)]
enum ConnectionDirection {
    Inbound,
    Outbound,
}


/// Connection state
#[derive(Debug, Clone, Copy, PartialEq)]
enum ConnectionState {
    Connecting,
    Connected,
    Disconnecting,
    Disconnected,
    Failed,
}


/// Connection metrics
#[derive(Debug, Clone)]
struct ConnectionMetrics {
    bytes_sent: u64,
    bytes_received: u64,
    messages_sent: u64,
    messages_received: u64,
    connected_since: std::time::Instant,
    last_message_time: Option<std::time::Instant>,
}


impl ConnectionMetrics {
    fn new() -> Self {
        Self {
            bytes_sent: 0,
            bytes_received: 0,
            messages_sent: 0,
            messages_received: 0,
            connected_since: std::time::Instant::now(),
            last_message_time: None,
        }
    }
}


/// Cryptographic session for peer communication
#[derive(Debug, Clone)]
struct CryptoSession {
    /// Session ID
    session_id: [u8; 32],
    
    /// Encrypt/decrypt keys
    keys: SessionKeys,
    
    /// Session state
    state: SessionState,
}


/// Session keys
#[derive(Debug, Clone)]
struct SessionKeys {
    encryption_key: [u8; 32],
    decryption_key: [u8; 32],
    mac_key: [u8; 32],
}


/// Session state
#[derive(Debug, Clone, Copy, PartialEq)]
enum SessionState {
    Initializing,
    Handshake,
    Established,
    Expired,
    Failed,
}


/// Network metrics collector
#[derive(Debug, Clone)]
struct NetworkMetrics {
    /// Bytes sent
    bytes_sent: std::sync::atomic::AtomicU64,
    
    /// Bytes received
    bytes_received: std::sync::atomic::AtomicU64,
    
    /// Messages sent
    messages_sent: std::sync::atomic::AtomicU64,
    
    /// Messages received
    messages_received: std::sync::atomic::AtomicU64,
    
    /// Connected peers
    connected_peers: std::sync::atomic::AtomicUsize,
    
    /// Total peers
    total_peers: std::sync::atomic::AtomicUsize,
    
    /// Broadcast latency histogram
    broadcast_latency: std::sync::Mutex<Vec<u64>>,
    
    /// Message size histogram
    message_sizes: std::sync::Mutex<Vec<usize>>,
}


impl NetworkMetrics {
    fn new() -> Self {
        Self {
            bytes_sent: std::sync::atomic::AtomicU64::new(0),
            bytes_received: std::sync::atomic::AtomicU64::new(0),
            messages_sent: std::sync::atomic::AtomicU64::new(0),
            messages_received: std::sync::atomic::AtomicU64::new(0),
            connected_peers: std::sync::atomic::AtomicUsize::new(0),
            total_peers: std::sync::atomic::AtomicUsize::new(0),
            broadcast_latency: std::sync::Mutex::new(Vec::new()),
            message_sizes: std::sync::Mutex::new(Vec::new()),
        }
    }
    
    fn record_broadcast(&self, duration: std::time::Duration, _topic: Topic) {
        let latency = duration.as_millis() as u64;
        let mut latencies = self.broadcast_latency.lock().unwrap();
        latencies.push(latency);
        
        // Keep only last 1000 measurements
        if latencies.len() > 1000 {
            latencies.remove(0);
        }
    }
    
    fn record_tx_submission(&self, duration: std::time::Duration) {
        let latency = duration.as_millis() as u64;
        let mut latencies = self.broadcast_latency.lock().unwrap();
        latencies.push(latency);
        
        if latencies.len() > 1000 {
            latencies.remove(0);
        }
    }
    
    fn record_message_received(&self, size: usize) {
        self.messages_received.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.bytes_received.fetch_add(size as u64, std::sync::atomic::Ordering::Relaxed);
        
        let mut sizes = self.message_sizes.lock().unwrap();
        sizes.push(size);
        
        if sizes.len() > 1000 {
            sizes.remove(0);
        }
    }
    
    fn update_peer_count(&self, count: usize) {
        self.connected_peers.store(count, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn log_summary(&self) {
        let bytes_sent = self.bytes_sent.load(std::sync::atomic::Ordering::Relaxed);
        let bytes_received = self.bytes_received.load(std::sync::atomic::Ordering::Relaxed);
        let messages_sent = self.messages_sent.load(std::sync::atomic::Ordering::Relaxed);
        let messages_received = self.messages_received.load(std::sync::atomic::Ordering::Relaxed);
        let connected_peers = self.connected_peers.load(std::sync::atomic::Ordering::Relaxed);
        
        info!(
            "Network stats: {} peers, {} sent/{} received msgs, {} sent/{} received bytes",
            connected_peers, messages_sent, messages_received, bytes_sent, bytes_received
        );
    }
}


impl Clone for NetworkMetrics {
    fn clone(&self) -> Self {
        Self {
            bytes_sent: std::sync::atomic::AtomicU64::new(self.bytes_sent.load(std::sync::atomic::Ordering::Relaxed)),
            bytes_received: std::sync::atomic::AtomicU64::new(self.bytes_received.load(std::sync::atomic::Ordering::Relaxed)),
            messages_sent: std::sync::atomic::AtomicU64::new(self.messages_sent.load(std::sync::atomic::Ordering::Relaxed)),
            messages_received: std::sync::atomic::AtomicU64::new(self.messages_received.load(std::sync::atomic::Ordering::Relaxed)),
            connected_peers: std::sync::atomic::AtomicUsize::new(self.connected_peers.load(std::sync::atomic::Ordering::Relaxed)),
            total_peers: std::sync::atomic::AtomicUsize::new(self.total_peers.load(std::sync::atomic::Ordering::Relaxed)),
            broadcast_latency: std::sync::Mutex::new(self.broadcast_latency.lock().unwrap().clone()),
            message_sizes: std::sync::Mutex::new(self.message_sizes.lock().unwrap().clone()),
        }
    }
}


/// Network statistics
#[derive(Debug, Clone, Serialize)]
pub struct NetworkStats {
    /// Total peers known
    pub total_peers: usize,
    
    /// Currently connected peers
    pub connected_peers: usize,
    
    /// Inbound connections
    pub inbound_connections: usize,
    
    /// Outbound connections
    pub outbound_connections: usize,
    
    /// Bytes sent
    pub bytes_sent: u64,
    
    /// Bytes received
    pub bytes_received: u64,
    
    /// Messages sent
    pub messages_sent: u64,
    
    /// Messages received
    pub messages_received: u64,
    
    /// DHT queries performed
    pub dht_queries: u64,
    
    /// Gossip messages processed
    pub gossip_messages: u64,
    
    /// Mempool size in bytes
    pub mempool_size: usize,
}


/// Transaction receipt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxReceipt {
    /// Transaction hash
    pub tx_hash: String,
    
    /// Submission timestamp
    pub timestamp: std::time::SystemTime,
    
    /// Mempool status
    pub status: TxStatus,
    
    /// Estimated inclusion time
    pub estimated_inclusion_ms: Option<u64>,
    
    /// Priority fee paid
    pub priority_fee: Option<u64>,
}


/// Transaction status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TxStatus {
    Pending,
    Included,
    Rejected(String),
    Expired,
}


/// Encrypted transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedTransaction {
    /// Encrypted transaction data
    pub data: Vec<u8>,
    
    /// TEE attestation
    pub attestation: Vec<u8>,
    
    /// Shard ID
    pub shard_id: u64,
    
    /// Transaction hash
    pub hash: String,
    
    /// Submission time
    pub submission_time: std::time::SystemTime,
    
    /// Priority level
    pub priority: TxPriority,
}


/// Network errors
#[derive(Debug, thiserror::Error)]
pub enum NetworkError {
    #[error("DHT error: {0}")]
    DhtError(#[from] DhtError),
    
    #[error("Gossip error: {0}")]
    GossipError(#[from] GossipError),
    
    #[error("Mempool error: {0}")]
    MempoolError(#[from] MempoolError),
    
    #[error("Cryptographic error: {0}")]
    CryptoError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Invalid address: {0}")]
    InvalidAddress(String),
    
    #[error("Connection error: {0}")]
    ConnectionError(String),
    
    #[error("Message too large: {0} bytes, maximum {1} bytes")]
    MessageTooLarge(usize, usize),
    
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    
    #[error("Peer not found: {0}")]
    PeerNotFound(String),
    
    #[error("Topic not found: {0}")]
    TopicNotFound(String),
    
    #[error("Shutdown error: {0}")]
    ShutdownError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Timeout error: {0}")]
    TimeoutError(String),
    
    #[error("Resource error: {0}")]
    ResourceError(String),
}


// Re-export types for convenience
pub type Topic = String;
pub type MessageId = String;
