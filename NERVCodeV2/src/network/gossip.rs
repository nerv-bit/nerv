//! Gossip Protocol for NERV Network
//!
//! This module implements a Gossipsub-like protocol for efficient message propagation
//! across the NERV network. It's used for broadcasting blocks, transactions, and
//! other network messages.
//!
//! Key Features:
//! - Topic-based messaging with mesh networks
//! - Heartbeat-driven message propagation
//! - Message deduplication and caching
//! - Peer scoring for spam protection
//! - Adaptive mesh formation based on peer behavior


use crate::crypto::{CryptoProvider, ByteSerializable};
use crate::params::{GOSSIP_FANOUT, GOSSIP_TTL, GOSSIP_INTERVAL_MS};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn, debug, error, trace};
use blake3::Hasher;
use rand::prelude::*;


/// Gossip manager implementing Gossipsub protocol
pub struct GossipManager {
    /// Local peer ID
    local_peer_id: String,
    
    /// Subscribed topics
    subscriptions: RwLock<HashSet<Topic>>,
    
    /// Topic meshes (peers we're connected to for each topic)
    meshes: RwLock<HashMap<Topic, HashSet<String>>>,
    
    /// Message cache for deduplication
    message_cache: RwLock<MessageCache>,
    
    /// Peer scores for spam protection
    peer_scores: RwLock<HashMap<String, PeerScore>>,
    
    /// Configuration
    config: GossipConfig,
    
    /// Cryptographic provider
    crypto_provider: Arc<CryptoProvider>,
    
    /// Incoming message channel
    incoming_rx: RwLock<Option<mpsc::UnboundedReceiver<GossipMessage>>>,
    
    /// Outgoing message channel
    outgoing_tx: mpsc::UnboundedSender<GossipMessage>,
    
    /// Metrics
    metrics: GossipMetrics,
    
    /// Shutdown flag
    shutdown: RwLock<bool>,
}


impl GossipManager {
    /// Create a new gossip manager
    pub async fn new(config: GossipConfig, crypto_provider: Arc<CryptoProvider>) -> Result<Self, GossipError> {
        info!("Initializing gossip protocol with fanout: {}", config.fanout);
        
        // Generate local peer ID
        let local_peer_id = Self::generate_peer_id();
        
        // Create message channels
        let (incoming_tx, incoming_rx) = mpsc::unbounded_channel();
        let (outgoing_tx, outgoing_rx) = mpsc::unbounded_channel();
        
        let manager = Self {
            local_peer_id,
            subscriptions: RwLock::new(HashSet::new()),
            meshes: RwLock::new(HashMap::new()),
            message_cache: RwLock::new(MessageCache::new(config.message_cache_size)),
            peer_scores: RwLock::new(HashMap::new()),
            config,
            crypto_provider,
            incoming_rx: RwLock::new(Some(incoming_rx)),
            outgoing_tx,
            metrics: GossipMetrics::new(),
            shutdown: RwLock::new(false),
        };
        
        // Start message processors
        manager.start_message_processors(incoming_tx, outgoing_rx).await?;
        
        Ok(manager)
    }
    
    /// Start the gossip manager
    pub async fn start(&self) -> Result<(), GossipError> {
        info!("Starting gossip manager");
        
        // Start heartbeat task
        self.start_heartbeat_task().await?;
        
        // Start peer scoring task
        self.start_peer_scoring_task().await?;
        
        info!("Gossip manager started successfully");
        Ok(())
    }
    
    /// Stop the gossip manager
    pub async fn stop(&self) -> Result<(), GossipError> {
        info!("Stopping gossip manager");
        
        let mut shutdown = self.shutdown.write().await;
        *shutdown = true;
        
        // Clear subscriptions
        let mut subscriptions = self.subscriptions.write().await;
        subscriptions.clear();
        
        // Clear meshes
        let mut meshes = self.meshes.write().await;
        meshes.clear();
        
        info!("Gossip manager stopped");
        Ok(())
    }
    
    /// Broadcast a message to a topic
    pub async fn broadcast(&self, topic: Topic, message: Vec<u8>) -> Result<MessageId, GossipError> {
        let start_time = Instant::now();
        
        // Validate message
        if message.len() > self.config.max_message_size {
            return Err(GossipError::MessageTooLarge(message.len(), self.config.max_message_size));
        }
        
        // Generate message ID
        let message_id = Self::generate_message_id(&topic, &message);
        
        // Check if already seen
        let mut cache = self.message_cache.write().await;
        if cache.has_message(&message_id) {
            return Err(GossipError::DuplicateMessage(message_id.clone()));
        }
        
        // Create gossip message
        let gossip_message = GossipMessage {
            message_id: message_id.clone(),
            topic: topic.clone(),
            data: message,
            source: self.local_peer_id.clone(),
            seq_no: Self::generate_sequence_number(),
            signature: Vec::new(), // Would be signed in production
            timestamp: Instant::now(),
            ttl: self.config.message_cache_size as u32,
        };
        
        // Cache message
        cache.put(gossip_message.clone());
        
        // Broadcast to mesh peers
        self.broadcast_to_mesh(&topic, &gossip_message).await?;
        
        // Update metrics
        self.metrics.record_broadcast(start_time.elapsed());
        
        info!("Broadcast message {} to topic {}", message_id, topic);
        Ok(message_id)
    }
    
    /// Subscribe to a topic
    pub async fn subscribe(&self, topic: Topic) -> Result<(), GossipError> {
        info!("Subscribing to topic: {}", topic);
        
        let mut subscriptions = self.subscriptions.write().await;
        subscriptions.insert(topic.clone());
        
        // Join mesh for this topic
        self.join_mesh(&topic).await?;
        
        Ok(())
    }
    
    /// Unsubscribe from a topic
    pub async fn unsubscribe(&self, topic: Topic) -> Result<(), GossipError> {
        info!("Unsubscribing from topic: {}", topic);
        
        let mut subscriptions = self.subscriptions.write().await;
        subscriptions.remove(&topic);
        
        // Leave mesh for this topic
        self.leave_mesh(&topic).await?;
        
        Ok(())
    }
    
    /// Get message count for metrics
    pub async fn get_message_count(&self) -> u64 {
        self.metrics.messages_received.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    // Internal methods
    
    async fn start_message_processors(
        &self,
        incoming_tx: mpsc::UnboundedSender<GossipMessage>,
        mut outgoing_rx: mpsc::UnboundedReceiver<GossipMessage>,
    ) -> Result<(), GossipError> {
        let manager = self.clone_manager().await;
        
        // Incoming message processor
        tokio::spawn(async move {
            // This would connect to network in production
            // For simulation, just pass through outgoing to incoming
            while let Some(message) = outgoing_rx.recv().await {
                if incoming_tx.send(message).is_err() {
                    break;
                }
            }
        });
        
        // Message handler
        let manager = self.clone_manager().await;
        let mut incoming_rx = self.incoming_rx.write().await.take();
        
        if let Some(mut receiver) = incoming_rx {
            tokio::spawn(async move {
                while let Some(message) = receiver.recv().await {
                    if let Err(e) = manager.handle_message(message).await {
                        error!("Failed to handle gossip message: {}", e);
                    }
                }
            });
        }
        
        Ok(())
    }
    
    async fn start_heartbeat_task(&self) -> Result<(), GossipError> {
        let manager = self.clone_manager().await;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(manager.config.heartbeat_interval_ms));
            
            loop {
                interval.tick().await;
                
                // Check shutdown flag
                if *manager.shutdown.read().await {
                    break;
                }
                
                // Run heartbeat
                if let Err(e) = manager.heartbeat().await {
                    warn!("Heartbeat failed: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    async fn start_peer_scoring_task(&self) -> Result<(), GossipError> {
        let manager = self.clone_manager().await;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                // Check shutdown flag
                if *manager.shutdown.read().await {
                    break;
                }
                
                // Update peer scores
                manager.update_peer_scores().await;
            }
        });
        
        Ok(())
    }
    
    async fn heartbeat(&self) -> Result<(), GossipError> {
        debug!("Gossip heartbeat");
        
        // Maintain mesh for each subscribed topic
        let subscriptions = self.subscriptions.read().await;
        
        for topic in subscriptions.iter() {
            // Ensure we have enough peers in mesh
            self.maintain_mesh(topic).await?;
            
            // Prune expired messages from cache
            let mut cache = self.message_cache.write().await;
            cache.prune_expired();
        }
        
        self.metrics.record_heartbeat();
        Ok(())
    }
    
    async fn join_mesh(&self, topic: &Topic) -> Result<(), GossipError> {
        let mut meshes = self.meshes.write().await;
        
        if !meshes.contains_key(topic) {
            meshes.insert(topic.clone(), HashSet::new());
        }
        
        // Add some peers to mesh (in production, would select from known peers)
        let mesh = meshes.get_mut(topic).unwrap();
        for i in 0..self.config.fanout.min(5) {
            let peer_id = format!("peer_{}_{}", topic, i);
            mesh.insert(peer_id);
        }
        
        info!("Joined mesh for topic {} with {} peers", topic, mesh.len());
        Ok(())
    }
    
    async fn leave_mesh(&self, topic: &Topic) -> Result<(), GossipError> {
        let mut meshes = self.meshes.write().await;
        meshes.remove(topic);
        
        info!("Left mesh for topic {}", topic);
        Ok(())
    }
    
    async fn maintain_mesh(&self, topic: &Topic) -> Result<(), GossipError> {
        let mut meshes = self.meshes.write().await;
        
        if let Some(mesh) = meshes.get_mut(topic) {
            let current_size = mesh.len();
            let target_size = self.config.fanout;
            
            if current_size < target_size {
                // Need more peers
                let needed = target_size - current_size;
                for i in 0..needed {
                    let peer_id = format!("new_peer_{}_{}", topic, i);
                    mesh.insert(peer_id);
                }
                
                debug!("Added {} peers to mesh for topic {}", needed, topic);
            } else if current_size > target_size * 2 {
                // Too many peers, prune some
                let excess = current_size - target_size;
                let to_remove: Vec<String> = mesh.iter()
                    .take(excess)
                    .cloned()
                    .collect();
                
                for peer_id in to_remove {
                    mesh.remove(&peer_id);
                }
                
                debug!("Removed {} peers from mesh for topic {}", excess, topic);
            }
        }
        
        Ok(())
    }
    
    async fn broadcast_to_mesh(&self, topic: &Topic, message: &GossipMessage) -> Result<(), GossipError> {
        let meshes = self.meshes.read().await;
        
        if let Some(mesh) = meshes.get(topic) {
            for peer_id in mesh {
                if peer_id != &self.local_peer_id {
                    self.send_to_peer(peer_id, message).await?;
                }
            }
        }
        
        Ok(())
    }
    
    async fn send_to_peer(&self, peer_id: &str, message: &GossipMessage) -> Result<(), GossipError> {
        // In production: send via network
        // For simulation, just forward to outgoing channel
        self.outgoing_tx.send(message.clone())
            .map_err(|e| GossipError::SendError(e.to_string()))?;
        
        self.metrics.record_message_sent();
        Ok(())
    }
    
    async fn handle_message(&self, message: GossipMessage) -> Result<(), GossipError> {
        let start_time = Instant::now();
        
        // Validate message
        self.validate_message(&message).await?;
        
        // Check cache for duplicates
        let mut cache = self.message_cache.write().await;
        if cache.has_message(&message.message_id) {
            return Ok(()); // Already processed
        }
        
        // Cache message
        cache.put(message.clone());
        
        // Update peer score (positive for valid message)
        self.update_peer_score(&message.source, 1.0).await;
        
        // Forward to mesh peers (if we're subscribed)
        let subscriptions = self.subscriptions.read().await;
        if subscriptions.contains(&message.topic) {
            self.forward_message(&message).await?;
        }
        
        // Update metrics
        self.metrics.record_message_processed(start_time.elapsed());
        
        debug!("Processed gossip message: {}", message.message_id);
        Ok(())
    }
    
    async fn validate_message(&self, message: &GossipMessage) -> Result<(), GossipError> {
        // Check TTL
        if message.ttl == 0 {
            return Err(GossipError::ExpiredMessage(message.message_id.clone()));
        }
        
        // Check size
        if message.data.len() > self.config.max_message_size {
            return Err(GossipError::MessageTooLarge(
                message.data.len(),
                self.config.max_message_size,
            ));
        }
        
        // Check signature (in production)
        // For now, just validate format
        
        // Check peer score
        let peer_scores = self.peer_scores.read().await;
        if let Some(score) = peer_scores.get(&message.source) {
            if score.current_score < -10.0 {
                return Err(GossipError::LowPeerScore(message.source.clone(), score.current_score));
            }
        }
        
        Ok(())
    }
    
    async fn forward_message(&self, message: &GossipMessage) -> Result<(), GossipError> {
        // Create forwarded message with decremented TTL
        let mut forwarded = message.clone();
        forwarded.ttl = forwarded.ttl.saturating_sub(1);
        forwarded.source = self.local_peer_id.clone();
        
        // Broadcast to mesh
        self.broadcast_to_mesh(&message.topic, &forwarded).await
    }
    
    async fn update_peer_score(&self, peer_id: &str, delta: f64) {
        let mut peer_scores = self.peer_scores.write().await;
        let score = peer_scores.entry(peer_id.to_string())
            .or_insert_with(PeerScore::new);
        
        score.update(delta);
    }
    
    async fn update_peer_scores(&self) {
        let mut peer_scores = self.peer_scores.write().await;
        
        // Decay scores over time
        for score in peer_scores.values_mut() {
            score.decay();
        }
        
        // Remove peers with very low scores
        peer_scores.retain(|_, score| score.current_score > -100.0);
        
        debug!("Updated {} peer scores", peer_scores.len());
    }
    
    fn generate_peer_id() -> String {
        let mut rng = rand::thread_rng();
        let id: [u8; 16] = rng.gen();
        hex::encode(id)
    }
    
    fn generate_message_id(topic: &str, data: &[u8]) -> String {
        let mut hasher = Hasher::new();
        hasher.update(topic.as_bytes());
        hasher.update(data);
        let hash = hasher.finalize();
        hex::encode(hash.as_bytes())
    }
    
    fn generate_sequence_number() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        COUNTER.fetch_add(1, Ordering::Relaxed)
    }
    
    async fn clone_manager(&self) -> Arc<Self> {
        Arc::new(Self {
            local_peer_id: self.local_peer_id.clone(),
            subscriptions: RwLock::new(self.subscriptions.read().await.clone()),
            meshes: RwLock::new(self.meshes.read().await.clone()),
            message_cache: RwLock::new(self.message_cache.read().await.clone()),
            peer_scores: RwLock::new(self.peer_scores.read().await.clone()),
            config: self.config.clone(),
            crypto_provider: self.crypto_provider.clone(),
            incoming_rx: RwLock::new(None), // Can't clone receiver
            outgoing_tx: self.outgoing_tx.clone(),
            metrics: self.metrics.clone(),
            shutdown: RwLock::new(*self.shutdown.read().await),
        })
    }
}


/// Gossip configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipConfig {
    /// Gossip fanout parameter
    pub fanout: usize,
    
    /// Heartbeat interval in milliseconds
    pub heartbeat_interval_ms: u64,
    
    /// Message cache size
    pub message_cache_size: usize,
    
    /// Duplicate cache size
    pub duplicate_cache_size: usize,
    
    /// Maximum message size in bytes
    pub max_message_size: usize,
    
    /// Validation queue size
    pub validation_queue_size: usize,
}


impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            fanout: GOSSIP_FANOUT,
            heartbeat_interval_ms: GOSSIP_INTERVAL_MS,
            message_cache_size: 1000,
            duplicate_cache_size: 5000,
            max_message_size: 1024 * 1024, // 1MB
            validation_queue_size: 100,
        }
    }
}


/// Gossip message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipMessage {
    /// Message identifier
    pub message_id: MessageId,
    
    /// Topic
    pub topic: Topic,
    
    /// Message data
    pub data: Vec<u8>,
    
    /// Source peer ID
    pub source: String,
    
    /// Sequence number
    pub seq_no: u64,
    
    /// Cryptographic signature
    pub signature: Vec<u8>,
    
    /// Timestamp
    pub timestamp: Instant,
    
    /// Time-to-live (hop count)
    pub ttl: u32,
}


/// Message cache for deduplication
#[derive(Debug, Clone)]
struct MessageCache {
    /// Cached messages
    messages: VecDeque<CachedMessage>,
    
    /// Maximum size
    capacity: usize,
    
    /// Duplicate cache (message ID -> timestamp)
    duplicate_cache: HashMap<MessageId, Instant>,
    
    /// Duplicate cache capacity
    duplicate_capacity: usize,
}


impl MessageCache {
    fn new(capacity: usize) -> Self {
        Self {
            messages: VecDeque::with_capacity(capacity),
            capacity,
            duplicate_cache: HashMap::new(),
            duplicate_capacity: capacity * 5,
        }
    }
    
    fn has_message(&self, message_id: &MessageId) -> bool {
        self.duplicate_cache.contains_key(message_id)
    }
    
    fn put(&mut self, message: GossipMessage) {
        // Add to duplicate cache
        self.duplicate_cache.insert(message.message_id.clone(), message.timestamp);
        
        // Prune duplicate cache if needed
        if self.duplicate_cache.len() > self.duplicate_capacity {
            let mut oldest = None;
            let mut oldest_time = Instant::now();
            
            for (id, &time) in &self.duplicate_cache {
                if time < oldest_time {
                    oldest_time = time;
                    oldest = Some(id.clone());
                }
            }
            
            if let Some(id) = oldest {
                self.duplicate_cache.remove(&id);
            }
        }
        
        // Add to message cache
        let cached = CachedMessage {
            message_id: message.message_id,
            topic: message.topic,
            timestamp: message.timestamp,
            expires_at: message.timestamp + Duration::from_secs(300), // 5 minutes
        };
        
        self.messages.push_back(cached);
        
        // Prune if needed
        if self.messages.len() > self.capacity {
            self.messages.pop_front();
        }
    }
    
    fn prune_expired(&mut self) {
        let now = Instant::now();
        
        // Prune messages
        self.messages.retain(|msg| msg.expires_at > now);
        
        // Prune duplicate cache
        self.duplicate_cache.retain(|_, &time| {
            now - time < Duration::from_secs(600) // 10 minutes
        });
    }
}


/// Cached message metadata
#[derive(Debug, Clone)]
struct CachedMessage {
    message_id: MessageId,
    topic: Topic,
    timestamp: Instant,
    expires_at: Instant,
}


/// Peer scoring for spam protection
#[derive(Debug, Clone)]
struct PeerScore {
    /// Current score
    current_score: f64,
    
    /// Score history
    history: VecDeque<f64>,
    
    /// Maximum history size
    max_history: usize,
    
    /// Last update time
    last_update: Instant,
    
    /// Decay rate per minute
    decay_rate: f64,
}


impl PeerScore {
    fn new() -> Self {
        Self {
            current_score: 0.0,
            history: VecDeque::new(),
            max_history: 100,
            last_update: Instant::now(),
            decay_rate: 0.1, // 10% decay per minute
        }
    }
    
    fn update(&mut self, delta: f64) {
        self.current_score += delta;
        self.history.push_back(self.current_score);
        
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
        
        self.last_update = Instant::now();
    }
    
    fn decay(&mut self) {
        let elapsed_minutes = self.last_update.elapsed().as_secs_f64() / 60.0;
        let decay_amount = self.current_score.abs() * self.decay_rate * elapsed_minutes;
        
        if self.current_score > 0.0 {
            self.current_score = (self.current_score - decay_amount).max(0.0);
        } else {
            self.current_score = (self.current_score + decay_amount).min(0.0);
        }
        
        self.last_update = Instant::now();
    }
}


/// Gossip metrics
#[derive(Debug, Clone)]
struct GossipMetrics {
    /// Messages sent
    messages_sent: std::sync::atomic::AtomicU64,
    
    /// Messages received
    messages_received: std::sync::atomic::AtomicU64,
    
    /// Messages processed
    messages_processed: std::sync::atomic::AtomicU64,
    
    /// Broadcast latency histogram
    broadcast_latency: std::sync::Mutex<Vec<u64>>,
    
    /// Processing latency histogram
    processing_latency: std::sync::Mutex<Vec<u64>>,
    
    /// Heartbeat count
    heartbeat_count: std::sync::atomic::AtomicU64,
    
    /// Cache hits
    cache_hits: std::sync::atomic::AtomicU64,
    
    /// Cache misses
    cache_misses: std::sync::atomic::AtomicU64,
}


impl GossipMetrics {
    fn new() -> Self {
        Self {
            messages_sent: std::sync::atomic::AtomicU64::new(0),
            messages_received: std::sync::atomic::AtomicU64::new(0),
            messages_processed: std::sync::atomic::AtomicU64::new(0),
            broadcast_latency: std::sync::Mutex::new(Vec::new()),
            processing_latency: std::sync::Mutex::new(Vec::new()),
            heartbeat_count: std::sync::atomic::AtomicU64::new(0),
            cache_hits: std::sync::atomic::AtomicU64::new(0),
            cache_misses: std::sync::atomic::AtomicU64::new(0),
        }
    }
    
    fn record_broadcast(&self, duration: std::time::Duration) {
        let latency = duration.as_millis() as u64;
        let mut latencies = self.broadcast_latency.lock().unwrap();
        latencies.push(latency);
        
        if latencies.len() > 1000 {
            latencies.remove(0);
        }
    }
    
    fn record_message_processed(&self, duration: std::time::Duration) {
        self.messages_processed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let latency = duration.as_millis() as u64;
        let mut latencies = self.processing_latency.lock().unwrap();
        latencies.push(latency);
        
        if latencies.len() > 1000 {
            latencies.remove(0);
        }
    }
    
    fn record_message_sent(&self) {
        self.messages_sent.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_heartbeat(&self) {
        self.heartbeat_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}


impl Clone for GossipMetrics {
    fn clone(&self) -> Self {
        Self {
            messages_sent: std::sync::atomic::AtomicU64::new(self.messages_sent.load(std::sync::atomic::Ordering::Relaxed)),
            messages_received: std::sync::atomic::AtomicU64::new(self.messages_received.load(std::sync::atomic::Ordering::Relaxed)),
            messages_processed: std::sync::atomic::AtomicU64::new(self.messages_processed.load(std::sync::atomic::Ordering::Relaxed)),
            broadcast_latency: std::sync::Mutex::new(self.broadcast_latency.lock().unwrap().clone()),
            processing_latency: std::sync::Mutex::new(self.processing_latency.lock().unwrap().clone()),
            heartbeat_count: std::sync::atomic::AtomicU64::new(self.heartbeat_count.load(std::sync::atomic::Ordering::Relaxed)),
            cache_hits: std::sync::atomic::AtomicU64::new(self.cache_hits.load(std::sync::atomic::Ordering::Relaxed)),
            cache_misses: std::sync::atomic::AtomicU64::new(self.cache_misses.load(std::sync::atomic::Ordering::Relaxed)),
        }
    }
}


/// Gossip errors
#[derive(Debug, thiserror::Error)]
pub enum GossipError {
    #[error("Message too large: {0} bytes, maximum {1} bytes")]
    MessageTooLarge(usize, usize),
    
    #[error("Duplicate message: {0}")]
    DuplicateMessage(String),
    
    #[error("Expired message: {0}")]
    ExpiredMessage(String),
    
    #[error("Low peer score: {0} ({1})")]
    LowPeerScore(String, f64),
    
    #[error("Invalid message: {0}")]
    InvalidMessage(String),
    
    #[error("Send error: {0}")]
    SendError(String),
    
    #[error("Topic not found: {0}")]
    TopicNotFound(String),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
}


// Re-export types
pub type MessageId = String;
pub type Topic = String;


// Unit tests
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_message_id_generation() {
        let topic = "test_topic";
        let data = b"test_data";
        
        let message_id = GossipManager::generate_message_id(topic, data);
        assert!(!message_id.is_empty());
        assert_eq!(message_id.len(), 64); // Blake3 produces 64 hex chars
    }
    
    #[test]
    fn test_peer_score_update() {
        let mut score = PeerScore::new();
        assert_eq!(score.current_score, 0.0);
        
        score.update(5.0);
        assert_eq!(score.current_score, 5.0);
        
        score.update(-3.0);
        assert_eq!(score.current_score, 2.0);
    }
    
    #[test]
    fn test_message_cache() {
        let mut cache = MessageCache::new(10);
        
        let message = GossipMessage {
            message_id: "test_id".to_string(),
            topic: "test_topic".to_string(),
            data: vec![1, 2, 3],
            source: "test_source".to_string(),
            seq_no: 1,
            signature: vec![],
            timestamp: Instant::now(),
            ttl: 10,
        };
        
        assert!(!cache.has_message(&"test_id".to_string()));
        
        cache.put(message.clone());
        assert!(cache.has_message(&"test_id".to_string()));
        
        // Test pruning
        for i in 0..20 {
            let mut msg = message.clone();
            msg.message_id = format!("test_id_{}", i);
            cache.put(msg);
        }
        
        // Should have pruned old messages
        assert!(cache.messages.len() <= 10);
    }
    
    #[tokio::test]
    async fn test_gossip_manager_creation() {
        let config = GossipConfig::default();
        let crypto = Arc::new(CryptoProvider::new().unwrap());
        let manager = GossipManager::new(config, crypto).await;
        
        assert!(manager.is_ok());
    }
}
