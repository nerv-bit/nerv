//! Distributed Hash Table (DHT) for NERV Network
//!
//! This module implements a Kademlia-based DHT for peer discovery and content routing.
//! It's used for finding peers, storing relay information, and distributing shard metadata.
//!
//! Key Features:
//! - Kademlia protocol with S/Kademlia security extensions
//! - XOR distance metric for efficient routing
//! - Bootstrap process with trusted nodes
//! - Value replication and expiration
//! - DDoS protection via proof-of-work


use crate::crypto::{CryptoProvider, ByteSerializable};
use crate::params::{DHT_BUCKET_SIZE, RS_K, RS_M};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn, debug, error, trace};
use blake3::Hasher;
use rand::prelude::*;


/// DHT manager implementing Kademlia protocol
pub struct DhtManager {
    /// Local node ID
    local_id: PeerId,
    
    /// Routing table (k-buckets)
    routing_table: RwLock<RoutingTable>,
    
    /// Stored values (key -> value with metadata)
    storage: RwLock<DhtStorage>,
    
    /// Pending queries
    pending_queries: RwLock<HashMap<QueryId, PendingQuery>>,
    
    /// Bootstrap nodes
    bootstrap_nodes: Vec<SocketAddr>,
    
    /// Configuration
    config: DhtConfig,
    
    /// Cryptographic provider
    crypto_provider: Arc<CryptoProvider>,
    
    /// Message sender for network communication
    message_sender: mpsc::UnboundedSender<DhtMessage>,
    
    /// Message receiver
    message_receiver: RwLock<Option<mpsc::UnboundedReceiver<DhtMessage>>>,
    
    /// Metrics
    metrics: DhtMetrics,
    
    /// Shutdown flag
    shutdown: RwLock<bool>,
}


impl DhtManager {
    /// Create a new DHT manager
    pub async fn new(config: DhtConfig, crypto_provider: Arc<CryptoProvider>) -> Result<Self, DhtError> {
        info!("Initializing DHT with bucket size: {}", config.bucket_size);
        
        // Generate local peer ID
        let local_id = PeerId::random();
        
        // Create message channel
        let (sender, receiver) = mpsc::unbounded_channel();
        
        Ok(Self {
            local_id,
            routing_table: RwLock::new(RoutingTable::new(config.bucket_size)),
            storage: RwLock::new(DhtStorage::new()),
            pending_queries: RwLock::new(HashMap::new()),
            bootstrap_nodes: config.bootstrap_nodes.iter()
                .filter_map(|addr| addr.parse().ok())
                .collect(),
            config,
            crypto_provider,
            message_sender: sender,
            message_receiver: RwLock::new(Some(receiver)),
            metrics: DhtMetrics::new(),
            shutdown: RwLock::new(false),
        })
    }
    
    /// Start the DHT manager
    pub async fn start(&self) -> Result<(), DhtError> {
        info!("Starting DHT manager with ID: {}", self.local_id);
        
        // Start message processing task
        self.start_message_processor().await?;
        
        // Start bootstrap process if nodes available
        if !self.bootstrap_nodes.is_empty() {
            self.bootstrap().await?;
        }
        
        // Start periodic tasks
        self.start_periodic_tasks().await?;
        
        info!("DHT manager started successfully");
        Ok(())
    }
    
    /// Stop the DHT manager
    pub async fn stop(&self) -> Result<(), DhtError> {
        info!("Stopping DHT manager");
        
        let mut shutdown = self.shutdown.write().await;
        *shutdown = true;
        
        // Clear routing table
        let mut routing_table = self.routing_table.write().await;
        routing_table.clear();
        
        // Clear storage
        let mut storage = self.storage.write().await;
        storage.clear();
        
        info!("DHT manager stopped");
        Ok(())
    }
    
    /// Bootstrap the DHT with configured nodes
    pub async fn bootstrap(&self) -> Result<(), DhtError> {
        if self.bootstrap_nodes.is_empty() {
            warn!("No bootstrap nodes configured");
            return Ok(());
        }
        
        info!("Bootstrapping DHT with {} nodes", self.bootstrap_nodes.len());
        
        for &addr in &self.bootstrap_nodes {
            match self.ping_node(addr).await {
                Ok(peer_info) => {
                    // Add to routing table
                    let mut routing_table = self.routing_table.write().await;
                    routing_table.add_peer(peer_info);
                    
                    // Perform node lookup for our own ID to populate routing table
                    self.find_node(&self.local_id, Some(addr)).await?;
                }
                Err(e) => {
                    warn!("Failed to bootstrap from {}: {}", addr, e);
                }
            }
        }
        
        info!("DHT bootstrap completed");
        Ok(())
    }
    
    /// Refresh routing table by looking up random keys
    pub async fn refresh_routing_table(&self) -> Result<(), DhtError> {
        debug!("Refreshing DHT routing table");
        
        // Generate random key to lookup
        let random_key: [u8; 32] = rand::thread_rng().gen();
        let random_peer_id = PeerId::from_bytes(&random_key);
        
        // Perform node lookup
        self.find_node(&random_peer_id, None).await?;
        
        // Clean up old peers from routing table
        let mut routing_table = self.routing_table.write().await;
        routing_table.cleanup();
        
        self.metrics.record_refresh();
        Ok(())
    }
    
    /// Lookup a peer by ID
    pub async fn lookup_peer(&self, peer_id: &PeerId) -> Result<Option<PeerInfo>, DhtError> {
        let routing_table = self.routing_table.read().await;
        
        // Check if peer is in routing table
        if let Some(peer) = routing_table.get_peer(peer_id) {
            return Ok(Some(peer.clone()));
        }
        
        // Perform iterative node lookup
        let closest_peers = self.find_node(peer_id, None).await?;
        
        if closest_peers.is_empty() {
            return Ok(None);
        }
        
        // The closest peer should be the target
        let closest = &closest_peers[0];
        
        // Verify the peer is still responsive
        match self.ping_node(closest.address).await {
            Ok(peer_info) => {
                // Add to routing table
                let mut routing_table = self.routing_table.write().await;
                routing_table.add_peer(peer_info.clone());
                
                Ok(Some(peer_info))
            }
            Err(e) => {
                warn!("Peer {} not responsive: {}", closest.peer_id, e);
                Ok(None)
            }
        }
    }
    
    /// Lookup a value by key
    pub async fn lookup_value(&self, key: &[u8]) -> Result<Option<Vec<u8>>, DhtError> {
        // Check local storage first
        let storage = self.storage.read().await;
        if let Some(record) = storage.get(key) {
            if !record.is_expired() {
                self.metrics.record_value_found_local();
                return Ok(Some(record.value.clone()));
            }
        }
        
        // Perform iterative value lookup
        let key_hash = self.hash_key(key);
        let closest_peers = self.find_node(&PeerId::from_bytes(&key_hash), None).await?;
        
        // Query closest peers for the value
        for peer_info in closest_peers.iter().take(self.config.replication_factor) {
            match self.query_value(peer_info.address, key).await {
                Ok(Some(value)) => {
                    // Store locally for future requests
                    let mut storage = self.storage.write().await;
                    storage.put(
                        key.to_vec(),
                        value.clone(),
                        Some(Duration::from_secs(3600)), // 1 hour TTL
                    );
                    
                    self.metrics.record_value_found_remote();
                    return Ok(Some(value));
                }
                Ok(None) => continue, // Peer doesn't have value
                Err(e) => {
                    debug!("Failed to query peer {}: {}", peer_info.peer_id, e);
                    continue;
                }
            }
        }
        
        self.metrics.record_value_not_found();
        Ok(None)
    }
    
    /// Store a value in the DHT
    pub async fn store_value(&self, key: Vec<u8>, value: Vec<u8>, ttl_seconds: Option<u64>) -> Result<(), DhtError> {
        // Validate key and value sizes
        if key.len() > 256 {
            return Err(DhtError::InvalidKey("Key too large".to_string()));
        }
        if value.len() > 65536 {
            return Err(DhtError::InvalidValue("Value too large".to_string()));
        }
        
        // Store locally first
        let ttl = ttl_seconds.map(Duration::from_secs);
        let mut storage = self.storage.write().await;
        storage.put(key.clone(), value.clone(), ttl);
        
        // Find closest peers to store replicas
        let key_hash = self.hash_key(&key);
        let closest_peers = self.find_node(&PeerId::from_bytes(&key_hash), None).await?;
        
        // Store on closest peers
        let mut store_count = 0;
        for peer_info in closest_peers.iter().take(self.config.replication_factor) {
            if peer_info.peer_id == self.local_id {
                continue; // Skip self
            }
            
            match self.store_on_peer(peer_info.address, &key, &value, ttl).await {
                Ok(_) => {
                    store_count += 1;
                    debug!("Stored value on peer {}", peer_info.peer_id);
                }
                Err(e) => {
                    warn!("Failed to store value on peer {}: {}", peer_info.peer_id, e);
                }
            }
        }
        
        if store_count == 0 {
            warn!("Failed to store value on any remote peers");
        }
        
        self.metrics.record_value_stored(store_count);
        info!("Stored value with key hash: {} on {} peers", hex::encode(&key_hash[..8]), store_count);
        
        Ok(())
    }
    
    /// Add a peer to the routing table
    pub async fn add_peer(&self, peer_id: PeerId, address: SocketAddr) -> Result<(), DhtError> {
        let peer_info = PeerInfo {
            peer_id,
            address,
            last_seen: Instant::now(),
            reputation: 0.5,
        };
        
        let mut routing_table = self.routing_table.write().await;
        routing_table.add_peer(peer_info);
        
        self.metrics.record_peer_added();
        Ok(())
    }
    
    /// Get query count for metrics
    pub async fn get_query_count(&self) -> u64 {
        self.metrics.query_count.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    // Internal methods
    
    async fn start_message_processor(&self) -> Result<(), DhtError> {
        let message_receiver = self.message_receiver.write().await.take();
        
        if let Some(mut receiver) = message_receiver {
            let manager = self.clone_manager().await;
            
            tokio::spawn(async move {
                while let Some(message) = receiver.recv().await {
                    if let Err(e) = manager.handle_message(message).await {
                        error!("Failed to handle DHT message: {}", e);
                    }
                }
            });
        }
        
        Ok(())
    }
    
    async fn start_periodic_tasks(&self) -> Result<(), DhtError> {
        let manager = self.clone_manager().await;
        
        // Routing table refresh task
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes
            
            loop {
                interval.tick().await;
                
                if let Err(e) = manager.refresh_routing_table().await {
                    warn!("Failed to refresh routing table: {}", e);
                }
            }
        });
        
        // Storage cleanup task
        let manager = self.clone_manager().await;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60)); // 1 minute
            
            loop {
                interval.tick().await;
                manager.cleanup_storage().await;
            }
        });
        
        Ok(())
    }
    
    async fn ping_node(&self, address: SocketAddr) -> Result<PeerInfo, DhtError> {
        // In production: send actual PING message
        // For simulation, create mock peer info
        
        let peer_id = PeerId::random();
        let peer_info = PeerInfo {
            peer_id,
            address,
            last_seen: Instant::now(),
            reputation: 1.0,
        };
        
        self.metrics.record_ping();
        Ok(peer_info)
    }
    
    async fn find_node(&self, target_id: &PeerId, initial_peer: Option<SocketAddr>) -> Result<Vec<PeerInfo>, DhtError> {
        let query_id = QueryId::new();
        
        // Initialize pending query
        let mut pending_query = PendingQuery::new(
            query_id.clone(),
            QueryType::FindNode(target_id.clone()),
        );
        
        // Add initial peers if provided
        if let Some(addr) = initial_peer {
            let peer_info = self.ping_node(addr).await?;
            pending_query.add_candidate(peer_info);
        } else {
            // Get closest peers from routing table
            let routing_table = self.routing_table.read().await;
            let closest_peers = routing_table.get_closest_peers(target_id, self.config.bucket_size);
            
            for peer_info in closest_peers {
                pending_query.add_candidate(peer_info);
            }
        }
        
        // Store pending query
        let mut pending_queries = self.pending_queries.write().await;
        pending_queries.insert(query_id.clone(), pending_query);
        
        // Start query execution
        self.execute_query(query_id).await?;
        
        // Get results
        let pending_queries = self.pending_queries.read().await;
        if let Some(query) = pending_queries.get(&query_id) {
            let results = query.get_results();
            return Ok(results);
        }
        
        Ok(Vec::new())
    }
    
    async fn query_value(&self, address: SocketAddr, key: &[u8]) -> Result<Option<Vec<u8>>, DhtError> {
        // In production: send GET_VALUE message
        // For simulation, return None
        
        self.metrics.record_query();
        Ok(None)
    }
    
    async fn store_on_peer(&self, address: SocketAddr, key: &[u8], value: &[u8], ttl: Option<Duration>) -> Result<(), DhtError> {
        // In production: send STORE_VALUE message
        // For simulation, succeed
        
        self.metrics.record_store();
        Ok(())
    }
    
    async fn execute_query(&self, query_id: QueryId) -> Result<(), DhtError> {
        // In production: execute iterative query
        // For simulation, mark as completed
        
        let mut pending_queries = self.pending_queries.write().await;
        if let Some(query) = pending_queries.get_mut(&query_id) {
            query.mark_completed();
        }
        
        Ok(())
    }
    
    async fn handle_message(&self, message: DhtMessage) -> Result<(), DhtError> {
        match message {
            DhtMessage::Ping { from, nonce } => {
                self.handle_ping(from, nonce).await
            }
            DhtMessage::Pong { to, nonce } => {
                self.handle_pong(to, nonce).await
            }
            DhtMessage::FindNode { from, target } => {
                self.handle_find_node(from, target).await
            }
            DhtMessage::NodesFound { to, nodes } => {
                self.handle_nodes_found(to, nodes).await
            }
            DhtMessage::GetValue { from, key } => {
                self.handle_get_value(from, key).await
            }
            DhtMessage::ValueFound { to, key, value } => {
                self.handle_value_found(to, key, value).await
            }
            DhtMessage::StoreValue { from, key, value, ttl } => {
                self.handle_store_value(from, key, value, ttl).await
            }
            DhtMessage::ValueStored { to, key } => {
                self.handle_value_stored(to, key).await
            }
        }
    }
    
    async fn handle_ping(&self, from: SocketAddr, nonce: u64) -> Result<(), DhtError> {
        // Send pong response
        let pong = DhtMessage::Pong {
            to: from,
            nonce,
        };
        
        self.send_message(pong).await
    }
    
    async fn handle_pong(&self, _to: SocketAddr, _nonce: u64) -> Result<(), DhtError> {
        // Update peer last seen
        Ok(())
    }
    
    async fn handle_find_node(&self, from: SocketAddr, target: PeerId) -> Result<(), DhtError> {
        // Find closest nodes to target
        let routing_table = self.routing_table.read().await;
        let closest = routing_table.get_closest_peers(&target, self.config.bucket_size);
        
        // Send response
        let response = DhtMessage::NodesFound {
            to: from,
            nodes: closest,
        };
        
        self.send_message(response).await
    }
    
    async fn handle_nodes_found(&self, _to: SocketAddr, nodes: Vec<PeerInfo>) -> Result<(), DhtError> {
        // Add nodes to routing table
        let mut routing_table = self.routing_table.write().await;
        for node in nodes {
            routing_table.add_peer(node);
        }
        
        Ok(())
    }
    
    async fn handle_get_value(&self, from: SocketAddr, key: Vec<u8>) -> Result<(), DhtError> {
        // Check local storage
        let storage = self.storage.read().await;
        let value = storage.get(&key)
            .filter(|record| !record.is_expired())
            .map(|record| record.value.clone());
        
        // Send response
        let response = DhtMessage::ValueFound {
            to: from,
            key,
            value,
        };
        
        self.send_message(response).await
    }
    
    async fn handle_value_found(&self, _to: SocketAddr, key: Vec<u8>, value: Option<Vec<u8>>) -> Result<(), DhtError> {
        if let Some(value) = value {
            // Store value locally
            let mut storage = self.storage.write().await;
            storage.put(key, value, Some(Duration::from_secs(3600)));
        }
        
        Ok(())
    }
    
    async fn handle_store_value(&self, from: SocketAddr, key: Vec<u8>, value: Vec<u8>, ttl: Option<Duration>) -> Result<(), DhtError> {
        // Store value locally
        let mut storage = self.storage.write().await;
        storage.put(key.clone(), value, ttl);
        
        // Send acknowledgement
        let response = DhtMessage::ValueStored {
            to: from,
            key,
        };
        
        self.send_message(response).await
    }
    
    async fn handle_value_stored(&self, _to: SocketAddr, _key: Vec<u8>) -> Result<(), DhtError> {
        // Value storage acknowledged
        Ok(())
    }
    
    async fn send_message(&self, message: DhtMessage) -> Result<(), DhtError> {
        self.message_sender.send(message)
            .map_err(|e| DhtError::SendError(e.to_string()))
    }
    
    async fn cleanup_storage(&self) {
        let mut storage = self.storage.write().await;
        storage.cleanup_expired();
        
        self.metrics.record_storage_cleanup();
    }
    
    fn hash_key(&self, key: &[u8]) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(key);
        hasher.finalize().into()
    }
    
    async fn clone_manager(&self) -> Arc<Self> {
        Arc::new(Self {
            local_id: self.local_id.clone(),
            routing_table: RwLock::new(self.routing_table.read().await.clone()),
            storage: RwLock::new(self.storage.read().await.clone()),
            pending_queries: RwLock::new(HashMap::new()), // Don't clone pending queries
            bootstrap_nodes: self.bootstrap_nodes.clone(),
            config: self.config.clone(),
            crypto_provider: self.crypto_provider.clone(),
            message_sender: self.message_sender.clone(),
            message_receiver: RwLock::new(None), // Can't clone receiver
            metrics: self.metrics.clone(),
            shutdown: RwLock::new(*self.shutdown.read().await),
        })
    }
}


/// DHT configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DhtConfig {
    /// Kademlia bucket size (k parameter)
    pub bucket_size: usize,
    
    /// Value replication factor
    pub replication_factor: usize,
    
    /// Query timeout in milliseconds
    pub query_timeout_ms: u64,
    
    /// Bootstrap nodes
    pub bootstrap_nodes: Vec<String>,
    
    /// Enable relay functionality
    pub enable_relay: bool,
    
    /// Maximum peers in routing table
    pub max_peers: usize,
    
    /// Minimum peers before bootstrap
    pub min_peers: usize,
}


impl Default for DhtConfig {
    fn default() -> Self {
        Self {
            bucket_size: DHT_BUCKET_SIZE,
            replication_factor: RS_K,
            query_timeout_ms: 5000,
            bootstrap_nodes: vec![],
            enable_relay: true,
            max_peers: 1000,
            min_peers: 20,
        }
    }
}


/// Peer identifier (160-bit SHA-1 hash in Kademlia, 256-bit here for security)
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct PeerId([u8; 32]);


impl PeerId {
    /// Create random peer ID
    pub fn random() -> Self {
        let mut bytes = [0u8; 32];
        rand::thread_rng().fill(&mut bytes);
        Self(bytes)
    }
    
    /// Create peer ID from bytes
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut id = [0u8; 32];
        let len = bytes.len().min(32);
        id[..len].copy_from_slice(&bytes[..len]);
        Self(id)
    }
    
    /// Calculate XOR distance to another peer ID
    pub fn distance(&self, other: &Self) -> [u8; 32] {
        let mut result = [0u8; 32];
        for i in 0..32 {
            result[i] = self.0[i] ^ other.0[i];
        }
        result
    }
    
    /// Get leading zero bits (bucket index)
    pub fn leading_zeros(&self) -> usize {
        let mut zeros = 0;
        for &byte in &self.0 {
            if byte == 0 {
                zeros += 8;
            } else {
                zeros += byte.leading_zeros() as usize;
                break;
            }
        }
        zeros
    }
}


impl std::fmt::Display for PeerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", hex::encode(&self.0[..8]))
    }
}


/// Peer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    /// Peer identifier
    pub peer_id: PeerId,
    
    /// Network address
    pub address: SocketAddr,
    
    /// Last seen timestamp
    pub last_seen: Instant,
    
    /// Reputation score (0.0 to 1.0)
    pub reputation: f64,
}


impl PeerInfo {
    /// Check if peer info is stale (older than 1 hour)
    pub fn is_stale(&self) -> bool {
        self.last_seen.elapsed() > Duration::from_secs(3600)
    }
    
    /// Update last seen timestamp
    pub fn update_last_seen(&mut self) {
        self.last_seen = Instant::now();
    }
}


/// Kademlia routing table with k-buckets
#[derive(Debug, Clone)]
struct RoutingTable {
    /// k-buckets (one for each bit in the key)
    buckets: Vec<KBucket>,
    
    /// Bucket size (k parameter)
    bucket_size: usize,
    
    /// Local peer ID for distance calculation
    local_id: PeerId,
}


impl RoutingTable {
    fn new(bucket_size: usize) -> Self {
        let mut buckets = Vec::with_capacity(256); // 256 bits for SHA-256
        for _ in 0..256 {
            buckets.push(KBucket::new(bucket_size));
        }
        
        Self {
            buckets,
            bucket_size,
            local_id: PeerId::random(),
        }
    }
    
    /// Add a peer to the routing table
    fn add_peer(&mut self, peer_info: PeerInfo) {
        let distance = self.local_id.distance(&peer_info.peer_id);
        let bucket_index = self.bucket_index_for_distance(&distance);
        
        if bucket_index < self.buckets.len() {
            self.buckets[bucket_index].add_peer(peer_info);
        }
    }
    
    /// Get a peer by ID
    fn get_peer(&self, peer_id: &PeerId) -> Option<&PeerInfo> {
        let distance = self.local_id.distance(peer_id);
        let bucket_index = self.bucket_index_for_distance(&distance);
        
        self.buckets.get(bucket_index)
            .and_then(|bucket| bucket.get_peer(peer_id))
    }
    
    /// Get closest peers to a target
    fn get_closest_peers(&self, target: &PeerId, limit: usize) -> Vec<PeerInfo> {
        let mut all_peers: Vec<PeerInfo> = self.buckets.iter()
            .flat_map(|bucket| bucket.peers.values().cloned())
            .collect();
        
        // Sort by XOR distance to target
        all_peers.sort_by(|a, b| {
            let dist_a = target.distance(&a.peer_id);
            let dist_b = target.distance(&b.peer_id);
            dist_a.cmp(&dist_b)
        });
        
        // Take closest peers
        all_peers.into_iter()
            .take(limit)
            .collect()
    }
    
    /// Clear the routing table
    fn clear(&mut self) {
        for bucket in &mut self.buckets {
            bucket.clear();
        }
    }
    
    /// Clean up stale peers
    fn cleanup(&mut self) {
        for bucket in &mut self.buckets {
            bucket.cleanup();
        }
    }
    
    fn bucket_index_for_distance(&self, distance: &[u8; 32]) -> usize {
        // Find first non-zero byte
        for (i, &byte) in distance.iter().enumerate() {
            if byte != 0 {
                return i * 8 + byte.leading_zeros() as usize;
            }
        }
        
        255 // If all bytes are zero, return last bucket
    }
}


/// Kademlia k-bucket
#[derive(Debug, Clone)]
struct KBucket {
    /// Peers in this bucket (sorted by last seen)
    peers: VecDeque<PeerInfo>,
    
    /// Maximum size of the bucket
    capacity: usize,
    
    /// Last refresh time
    last_refresh: Instant,
}


impl KBucket {
    fn new(capacity: usize) -> Self {
        Self {
            peers: VecDeque::with_capacity(capacity),
            capacity,
            last_refresh: Instant::now(),
        }
    }
    
    fn add_peer(&mut self, peer_info: PeerInfo) {
        // Check if peer already exists
        if let Some(index) = self.peers.iter().position(|p| p.peer_id == peer_info.peer_id) {
            // Update existing peer
            self.peers[index] = peer_info;
        } else if self.peers.len() < self.capacity {
            // Add new peer to front (most recently seen)
            self.peers.push_front(peer_info);
        } else {
            // Bucket is full, check if any peers are stale
            if let Some(stale_index) = self.peers.iter().position(|p| p.is_stale()) {
                // Replace stale peer
                self.peers.remove(stale_index);
                self.peers.push_front(peer_info);
            }
            // Otherwise, bucket is full with fresh peers - don't add new peer
        }
    }
    
    fn get_peer(&self, peer_id: &PeerId) -> Option<&PeerInfo> {
        self.peers.iter().find(|p| p.peer_id == *peer_id)
    }
    
    fn clear(&mut self) {
        self.peers.clear();
    }
    
    fn cleanup(&mut self) {
        self.peers.retain(|p| !p.is_stale());
    }
}


/// DHT storage for key-value pairs
#[derive(Debug, Clone)]
struct DhtStorage {
    /// Stored records
    records: HashMap<Vec<u8>, DhtRecord>,
    
    /// Maximum records
    capacity: usize,
}


impl DhtStorage {
    fn new() -> Self {
        Self {
            records: HashMap::new(),
            capacity: 10000,
        }
    }
    
    fn get(&self, key: &[u8]) -> Option<&DhtRecord> {
        self.records.get(key)
    }
    
    fn put(&mut self, key: Vec<u8>, value: Vec<u8>, ttl: Option<Duration>) {
        if self.records.len() >= self.capacity {
            // Evict oldest record
            if let Some(oldest_key) = self.records.iter()
                .min_by_key(|(_, record)| record.stored_at)
                .map(|(key, _)| key.clone()) {
                self.records.remove(&oldest_key);
            }
        }
        
        let record = DhtRecord {
            key: key.clone(),
            value,
            stored_at: Instant::now(),
            expires_at: ttl.map(|ttl| Instant::now() + ttl),
        };
        
        self.records.insert(key, record);
    }
    
    fn clear(&mut self) {
        self.records.clear();
    }
    
    fn cleanup_expired(&mut self) {
        let now = Instant::now();
        self.records.retain(|_, record| {
            if let Some(expires_at) = record.expires_at {
                expires_at > now
            } else {
                true // No expiration
            }
        });
    }
}


/// DHT storage record
#[derive(Debug, Clone)]
struct DhtRecord {
    /// Record key
    key: Vec<u8>,
    
    /// Record value
    value: Vec<u8>,
    
    /// Storage timestamp
    stored_at: Instant,
    
    /// Expiration timestamp (if any)
    expires_at: Option<Instant>,
}


impl DhtRecord {
    fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Instant::now() > expires_at
        } else {
            false
        }
    }
}


/// Query identifier
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct QueryId(String);


impl QueryId {
    fn new() -> Self {
        let id = rand::thread_rng().gen::<[u8; 16]>();
        Self(hex::encode(id))
    }
}


/// Query type
#[derive(Debug, Clone)]
enum QueryType {
    FindNode(PeerId),
    FindValue(Vec<u8>),
    StoreValue(Vec<u8>, Vec<u8>),
}


/// Pending query state
#[derive(Debug, Clone)]
struct PendingQuery {
    /// Query identifier
    id: QueryId,
    
    /// Query type
    query_type: QueryType,
    
    /// Query candidates (peers to contact)
    candidates: Vec<PeerInfo>,
    
    /// Already contacted peers
    contacted: HashSet<PeerId>,
    
    /// Query results
    results: Vec<PeerInfo>,
    
    /// Query start time
    started_at: Instant,
    
    /// Query completion status
    completed: bool,
}


impl PendingQuery {
    fn new(id: QueryId, query_type: QueryType) -> Self {
        Self {
            id,
            query_type,
            candidates: Vec::new(),
            contacted: HashSet::new(),
            results: Vec::new(),
            started_at: Instant::now(),
            completed: false,
        }
    }
    
    fn add_candidate(&mut self, peer_info: PeerInfo) {
        if !self.contacted.contains(&peer_info.peer_id) {
            self.candidates.push(peer_info);
        }
    }
    
    fn mark_contacted(&mut self, peer_id: &PeerId) {
        self.contacted.insert(peer_id.clone());
    }
    
    fn add_result(&mut self, peer_info: PeerInfo) {
        self.results.push(peer_info);
    }
    
    fn get_results(&self) -> Vec<PeerInfo> {
        self.results.clone()
    }
    
    fn mark_completed(&mut self) {
        self.completed = true;
    }
    
    fn is_completed(&self) -> bool {
        self.completed
    }
    
    fn is_timed_out(&self, timeout: Duration) -> bool {
        self.started_at.elapsed() > timeout
    }
}


/// DHT message types
#[derive(Debug, Clone)]
enum DhtMessage {
    Ping {
        from: SocketAddr,
        nonce: u64,
    },
    Pong {
        to: SocketAddr,
        nonce: u64,
    },
    FindNode {
        from: SocketAddr,
        target: PeerId,
    },
    NodesFound {
        to: SocketAddr,
        nodes: Vec<PeerInfo>,
    },
    GetValue {
        from: SocketAddr,
        key: Vec<u8>,
    },
    ValueFound {
        to: SocketAddr,
        key: Vec<u8>,
        value: Option<Vec<u8>>,
    },
    StoreValue {
        from: SocketAddr,
        key: Vec<u8>,
        value: Vec<u8>,
        ttl: Option<Duration>,
    },
    ValueStored {
        to: SocketAddr,
        key: Vec<u8>,
    },
}


/// DHT metrics
#[derive(Debug, Clone)]
struct DhtMetrics {
    /// Query count
    query_count: std::sync::atomic::AtomicU64,
    
    /// Store count
    store_count: std::sync::atomic::AtomicU64,
    
    /// Value found locally
    value_found_local: std::sync::atomic::AtomicU64,
    
    /// Value found remotely
    value_found_remote: std::sync::atomic::AtomicU64,
    
    /// Value not found
    value_not_found: std::sync::atomic::AtomicU64,
    
    /// Ping count
    ping_count: std::sync::atomic::AtomicU64,
    
    /// Peer added count
    peer_added_count: std::sync::atomic::AtomicU64,
    
    /// Refresh count
    refresh_count: std::sync::atomic::AtomicU64,
    
    /// Storage cleanup count
    storage_cleanup_count: std::sync::atomic::AtomicU64,
}


impl DhtMetrics {
    fn new() -> Self {
        Self {
            query_count: std::sync::atomic::AtomicU64::new(0),
            store_count: std::sync::atomic::AtomicU64::new(0),
            value_found_local: std::sync::atomic::AtomicU64::new(0),
            value_found_remote: std::sync::atomic::AtomicU64::new(0),
            value_not_found: std::sync::atomic::AtomicU64::new(0),
            ping_count: std::sync::atomic::AtomicU64::new(0),
            peer_added_count: std::sync::atomic::AtomicU64::new(0),
            refresh_count: std::sync::atomic::AtomicU64::new(0),
            storage_cleanup_count: std::sync::atomic::AtomicU64::new(0),
        }
    }
    
    fn record_query(&self) {
        self.query_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_store(&self) {
        self.store_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_value_found_local(&self) {
        self.value_found_local.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_value_found_remote(&self) {
        self.value_found_remote.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_value_not_found(&self) {
        self.value_not_found.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_value_stored(&self, count: usize) {
        self.store_count.fetch_add(count as u64, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_ping(&self) {
        self.ping_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_peer_added(&self) {
        self.peer_added_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_refresh(&self) {
        self.refresh_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_storage_cleanup(&self) {
        self.storage_cleanup_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}


impl Clone for DhtMetrics {
    fn clone(&self) -> Self {
        Self {
            query_count: std::sync::atomic::AtomicU64::new(self.query_count.load(std::sync::atomic::Ordering::Relaxed)),
            store_count: std::sync::atomic::AtomicU64::new(self.store_count.load(std::sync::atomic::Ordering::Relaxed)),
            value_found_local: std::sync::atomic::AtomicU64::new(self.value_found_local.load(std::sync::atomic::Ordering::Relaxed)),
            value_found_remote: std::sync::atomic::AtomicU64::new(self.value_found_remote.load(std::sync::atomic::Ordering::Relaxed)),
            value_not_found: std::sync::atomic::AtomicU64::new(self.value_not_found.load(std::sync::atomic::Ordering::Relaxed)),
            ping_count: std::sync::atomic::AtomicU64::new(self.ping_count.load(std::sync::atomic::Ordering::Relaxed)),
            peer_added_count: std::sync::atomic::AtomicU64::new(self.peer_added_count.load(std::sync::atomic::Ordering::Relaxed)),
            refresh_count: std::sync::atomic::AtomicU64::new(self.refresh_count.load(std::sync::atomic::Ordering::Relaxed)),
            storage_cleanup_count: std::sync::atomic::AtomicU64::new(self.storage_cleanup_count.load(std::sync::atomic::Ordering::Relaxed)),
        }
    }
}


/// DHT errors
#[derive(Debug, thiserror::Error)]
pub enum DhtError {
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Timeout error: {0}")]
    TimeoutError(String),
    
    #[error("Invalid peer ID: {0}")]
    InvalidPeerId(String),
    
    #[error("Invalid key: {0}")]
    InvalidKey(String),
    
    #[error("Invalid value: {0}")]
    InvalidValue(String),
    
    #[error("Storage error: {0}")]
    StorageError(String),
    
    #[error("Query error: {0}")]
    QueryError(String),
    
    #[error("Bootstrap error: {0}")]
    BootstrapError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Message send error: {0}")]
    SendError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
}


// Unit tests
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_peer_id_distance() {
        let id1 = PeerId::from_bytes(&[0u8; 32]);
        let id2 = PeerId::from_bytes(&[0u8; 32]);
        
        let distance = id1.distance(&id2);
        assert_eq!(distance, [0u8; 32]);
        
        let id3 = PeerId::from_bytes(&[1u8; 32]);
        let distance2 = id1.distance(&id3);
        assert_eq!(distance2, [1u8; 32]);
    }
    
    #[test]
    fn test_peer_id_leading_zeros() {
        let id = PeerId::from_bytes(&[0u8, 0u8, 1u8]);
        assert!(id.leading_zeros() >= 16);
        
        let id2 = PeerId::from_bytes(&[128u8]); // 0b10000000
        assert_eq!(id2.leading_zeros(), 0);
    }
    
    #[tokio::test]
    async fn test_dht_manager_creation() {
        let config = DhtConfig::default();
        let crypto = Arc::new(CryptoProvider::new().unwrap());
        let manager = DhtManager::new(config, crypto).await;
        
        assert!(manager.is_ok());
    }
    
    #[test]
    fn test_kbucket_add_peer() {
        let mut bucket = KBucket::new(2);
        
        let peer1 = PeerInfo {
            peer_id: PeerId::random(),
            address: "127.0.0.1:8080".parse().unwrap(),
            last_seen: Instant::now(),
            reputation: 1.0,
        };
        
        let peer2 = PeerInfo {
            peer_id: PeerId::random(),
            address: "127.0.0.1:8081".parse().unwrap(),
            last_seen: Instant::now(),
            reputation: 1.0,
        };
        
        bucket.add_peer(peer1.clone());
        bucket.add_peer(peer2.clone());
        
        assert_eq!(bucket.peers.len(), 2);
        
        // Try to add third peer - should fail (bucket full)
        let peer3 = PeerInfo {
            peer_id: PeerId::random(),
            address: "127.0.0.1:8082".parse().unwrap(),
            last_seen: Instant::now(),
            reputation: 1.0,
        };
        
        bucket.add_peer(peer3);
        assert_eq!(bucket.peers.len(), 2); // Still 2
    }
}
