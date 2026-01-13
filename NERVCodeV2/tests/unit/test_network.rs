// tests/unit/test_network.rs
// ============================================================================
// NETWORK MODULE UNIT TESTS
// ============================================================================


use nerv_bit::network::*;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use rand::Rng;


#[test]
fn test_dht_creation() {
    let dht = DistributedHashTable::new(8080).unwrap();
    
    assert_eq!(dht.port, 8080);
    assert!(dht.nodes.is_empty());
}


#[test]
fn test_dht_node_management() {
    let mut dht = DistributedHashTable::new(8080).unwrap();
    
    // Add nodes
    let node1 = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8001);
    let node2 = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8002);
    
    dht.add_node(node1).unwrap();
    dht.add_node(node2).unwrap();
    
    assert_eq!(dht.node_count(), 2);
    
    // Find closest nodes
    let target_key = [0u8; 32];
    let closest = dht.find_closest_nodes(&target_key, 1).unwrap();
    
    assert_eq!(closest.len(), 1);
    
    // Remove node
    dht.remove_node(&node1).unwrap();
    assert_eq!(dht.node_count(), 1);
}


#[test]
fn test_gossip_protocol() {
    let mut gossip = GossipProtocol::new();
    
    // Create message
    let message = NetworkMessage {
        id: [1u8; 32],
        sender: [2u8; 32],
        data: b"test message".to_vec(),
        timestamp: 1234567890,
        ttl: 10,
    };
    
    // Broadcast message
    gossip.broadcast(message.clone()).unwrap();
    
    // Should have the message
    assert!(gossip.has_message(&message.id));
    
    // Get messages for peer
    let peer_id = [3u8; 32];
    let messages = gossip.get_messages_for_peer(&peer_id, 10).unwrap();
    
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].id, message.id);
    
    // Mark message as delivered
    gossip.mark_delivered(&message.id, &peer_id).unwrap();
    
    // Peer shouldn't get the message again
    let messages2 = gossip.get_messages_for_peer(&peer_id, 10).unwrap();
    assert_eq!(messages2.len(), 0);
}


#[test]
fn test_mempool_operations() {
    let mut mempool = TransactionPool::new(1000); // Max 1000 transactions
    
    // Create transactions
    let mut transactions = Vec::new();
    
    for i in 0..10 {
        let tx = PooledTransaction {
            id: [i as u8; 32],
            data: vec![i as u8; 100],
            priority: i as u64,
            timestamp: 1234567890 + i as u64,
            sender: [0u8; 32],
        };
        transactions.push(tx);
    }
    
    // Add transactions
    for tx in &transactions {
        mempool.add_transaction(tx.clone()).unwrap();
    }
    
    assert_eq!(mempool.size(), 10);
    
    // Get highest priority transactions
    let batch = mempool.get_batch(5).unwrap();
    assert_eq!(batch.len(), 5);
    
    // Check ordering by priority (descending)
    for i in 1..batch.len() {
        assert!(batch[i-1].priority >= batch[i].priority);
    }
    
    // Remove transactions
    for tx in &transactions[0..5] {
        mempool.remove_transaction(&tx.id).unwrap();
    }
    
    assert_eq!(mempool.size(), 5);
}


#[test]
fn test_network_message_serialization() {
    let message = NetworkMessage {
        id: [1u8; 32],
        sender: [2u8; 32],
        data: b"test data".to_vec(),
        timestamp: 1234567890,
        ttl: 10,
    };
    
    // Serialize
    let bytes = message.to_bytes().unwrap();
    
    // Deserialize
    let deserialized = NetworkMessage::from_bytes(&bytes).unwrap();
    
    assert_eq!(message.id, deserialized.id);
    assert_eq!(message.sender, deserialized.sender);
    assert_eq!(message.data, deserialized.data);
    assert_eq!(message.timestamp, deserialized.timestamp);
    assert_eq!(message.ttl, deserialized.ttl);
}


#[test]
fn test_peer_discovery() {
    let mut discovery = PeerDiscovery::new();
    
    // Add known peers
    let peer1 = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8001);
    let peer2 = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8002);
    
    discovery.add_peer(peer1).unwrap();
    discovery.add_peer(peer2).unwrap();
    
    assert_eq!(discovery.peer_count(), 2);
    
    // Get random peers
    let random_peers = discovery.get_random_peers(1).unwrap();
    assert_eq!(random_peers.len(), 1);
    
    // Ban peer
    discovery.ban_peer(&peer1, 3600).unwrap(); // Ban for 1 hour
    
    // Banned peer shouldn't be returned
    let available_peers = discovery.get_available_peers(10).unwrap();
    assert!(!available_peers.contains(&peer1));
    
    // Unban peer
    discovery.unban_peer(&peer1).unwrap();
    let available_peers2 = discovery.get_available_peers(10).unwrap();
    assert!(available_peers2.contains(&peer1));
}


#[test]
fn test_connection_pool() {
    let mut pool = ConnectionPool::new(5); // Max 5 connections
    
    // Create connections
    for i in 0..5 {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8000 + i as u16);
        pool.add_connection(addr).unwrap();
    }
    
    assert_eq!(pool.connection_count(), 5);
    
    // Should fail to add more connections
    let addr6 = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8005);
    assert!(pool.add_connection(addr6).is_err());
    
    // Remove connection
    let addr0 = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8000);
    pool.remove_connection(&addr0).unwrap();
    
    assert_eq!(pool.connection_count(), 4);
    
    // Now we can add a new connection
    pool.add_connection(addr6).unwrap();
    assert_eq!(pool.connection_count(), 5);
}


#[test]
fn test_message_validation() {
    // Create valid message
    let valid_message = NetworkMessage {
        id: [1u8; 32],
        sender: [2u8; 32],
        data: vec![0u8; 100], // 100 bytes
        timestamp: 1234567890,
        ttl: 10,
    };
    
    // Validate
    let is_valid = valid_message.validate().unwrap();
    assert!(is_valid);
    
    // Create invalid message (TTL = 0)
    let invalid_message = NetworkMessage {
        ttl: 0,
        ..valid_message.clone()
    };
    
    let is_valid = invalid_message.validate().unwrap();
    assert!(!is_valid);
    
    // Create invalid message (data too large)
    let large_data = vec![0u8; 10 * 1024 * 1024]; // 10MB
    let invalid_message2 = NetworkMessage {
        data: large_data,
        ..valid_message
    };
    
    let is_valid = invalid_message2.validate().unwrap();
    assert!(!is_valid);
}


#[test]
fn test_encrypted_communication() {
    let mut rng = rand::thread_rng();
    
    // Generate key pair for encryption
    let mut private_key = [0u8; 32];
    rng.fill(&mut private_key);
    
    let public_key = derive_public_key(&private_key).unwrap();
    
    // Encrypt message
    let plaintext = b"secret message";
    let (ciphertext, nonce) = encrypt_message(plaintext, &public_key).unwrap();
    
    assert_ne!(ciphertext, plaintext);
    assert_eq!(nonce.len(), 24);
    
    // Decrypt message
    let decrypted = decrypt_message(&ciphertext, &nonce, &private_key).unwrap();
    
    assert_eq!(decrypted, plaintext);
    
    // Test with wrong key
    let mut wrong_key = [0u8; 32];
    rng.fill(&mut wrong_key);
    
    let result = decrypt_message(&ciphertext, &nonce, &wrong_key);
    assert!(result.is_err());
}


// Placeholder implementations for network module


pub struct DistributedHashTable {
    port: u16,
    nodes: std::collections::HashSet<SocketAddr>,
}


impl DistributedHashTable {
    pub fn new(port: u16) -> Result<Self, NetworkError> {
        Ok(Self {
            port,
            nodes: std::collections::HashSet::new(),
        })
    }
    
    pub fn add_node(&mut self, addr: SocketAddr) -> Result<(), NetworkError> {
        self.nodes.insert(addr);
        Ok(())
    }
    
    pub fn remove_node(&mut self, addr: &SocketAddr) -> Result<(), NetworkError> {
        self.nodes.remove(addr);
        Ok(())
    }
    
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    
    pub fn find_closest_nodes(&self, target: &[u8; 32], count: usize) -> Result<Vec<SocketAddr>, NetworkError> {
        // Simplified implementation
        let mut nodes: Vec<SocketAddr> = self.nodes.iter().cloned().collect();
        nodes.truncate(count);
        Ok(nodes)
    }
}


pub struct GossipProtocol {
    messages: std::collections::HashMap<[u8; 32], NetworkMessage>,
    delivered: std::collections::HashMap<[u8; 32], std::collections::HashSet<[u8; 32]>>,
}


impl GossipProtocol {
    pub fn new() -> Self {
        Self {
            messages: std::collections::HashMap::new(),
            delivered: std::collections::HashMap::new(),
        }
    }
    
    pub fn broadcast(&mut self, message: NetworkMessage) -> Result<(), NetworkError> {
        self.messages.insert(message.id, message);
        Ok(())
    }
    
    pub fn has_message(&self, message_id: &[u8; 32]) -> bool {
        self.messages.contains_key(message_id)
    }
    
    pub fn get_messages_for_peer(&self, peer_id: &[u8; 32], limit: usize) -> Result<Vec<NetworkMessage>, NetworkError> {
        let mut messages = Vec::new();
        
        for (_, message) in &self.messages {
            // Check if peer has already received this message
            if let Some(delivered_set) = self.delivered.get(&message.id) {
                if delivered_set.contains(peer_id) {
                    continue;
                }
            }
            
            messages.push(message.clone());
            if messages.len() >= limit {
                break;
            }
        }
        
        Ok(messages)
    }
    
    pub fn mark_delivered(&mut self, message_id: &[u8; 32], peer_id: &[u8; 32]) -> Result<(), NetworkError> {
        self.delivered
            .entry(*message_id)
            .or_insert_with(std::collections::HashSet::new)
            .insert(*peer_id);
        Ok(())
    }
}


pub struct PooledTransaction {
    pub id: [u8; 32],
    pub data: Vec<u8>,
    pub priority: u64,
    pub timestamp: u64,
    pub sender: [u8; 32],
}


pub struct TransactionPool {
    transactions: std::collections::HashMap<[u8; 32], PooledTransaction>,
    max_size: usize,
}


impl TransactionPool {
    pub fn new(max_size: usize) -> Self {
        Self {
            transactions: std::collections::HashMap::new(),
            max_size,
        }
    }
    
    pub fn add_transaction(&mut self, tx: PooledTransaction) -> Result<(), NetworkError> {
        if self.transactions.len() >= self.max_size {
            return Err(NetworkError::PoolFull(self.max_size));
        }
        
        self.transactions.insert(tx.id, tx);
        Ok(())
    }
    
    pub fn remove_transaction(&mut self, tx_id: &[u8; 32]) -> Result<(), NetworkError> {
        self.transactions.remove(tx_id);
        Ok(())
    }
    
    pub fn size(&self) -> usize {
        self.transactions.len()
    }
    
    pub fn get_batch(&self, count: usize) -> Result<Vec<PooledTransaction>, NetworkError> {
        let mut transactions: Vec<PooledTransaction> = self.transactions.values().cloned().collect();
        
        // Sort by priority (descending) and timestamp (ascending)
        transactions.sort_by(|a, b| {
            b.priority.cmp(&a.priority)
                .then(a.timestamp.cmp(&b.timestamp))
        });
        
        transactions.truncate(count);
        Ok(transactions)
    }
}


#[derive(Clone)]
pub struct NetworkMessage {
    pub id: [u8; 32],
    pub sender: [u8; 32],
    pub data: Vec<u8>,
    pub timestamp: u64,
    pub ttl: u32,
}


impl NetworkMessage {
    pub fn to_bytes(&self) -> Result<Vec<u8>, NetworkError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.id);
        bytes.extend_from_slice(&self.sender);
        bytes.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.data);
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());
        bytes.extend_from_slice(&self.ttl.to_le_bytes());
        Ok(bytes)
    }
    
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NetworkError> {
        if bytes.len() < 32 + 32 + 4 + 8 + 4 {
            return Err(NetworkError::InvalidData("Message too short".to_string()));
        }
        
        let mut offset = 0;
        
        let mut id = [0u8; 32];
        id.copy_from_slice(&bytes[offset..offset+32]);
        offset += 32;
        
        let mut sender = [0u8; 32];
        sender.copy_from_slice(&bytes[offset..offset+32]);
        offset += 32;
        
        let data_len = u32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        
        if bytes.len() < offset + data_len + 8 + 4 {
            return Err(NetworkError::InvalidData("Incomplete message".to_string()));
        }
        
        let data = bytes[offset..offset+data_len].to_vec();
        offset += data_len;
        
        let timestamp = u64::from_le_bytes(bytes[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        let ttl = u32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap());
        
        Ok(Self {
            id,
            sender,
            data,
            timestamp,
            ttl,
        })
    }
    
    pub fn validate(&self) -> Result<bool, NetworkError> {
        if self.ttl == 0 {
            return Ok(false);
        }
        
        if self.data.len() > 5 * 1024 * 1024 { // 5MB max
            return Ok(false);
        }
        
        Ok(true)
    }
}


pub struct PeerDiscovery {
    peers: std::collections::HashSet<SocketAddr>,
    banned: std::collections::HashMap<SocketAddr, u64>, // addr -> ban_until timestamp
}


impl PeerDiscovery {
    pub fn new() -> Self {
        Self {
            peers: std::collections::HashSet::new(),
            banned: std::collections::HashMap::new(),
        }
    }
    
    pub fn add_peer(&mut self, addr: SocketAddr) -> Result<(), NetworkError> {
        self.peers.insert(addr);
        Ok(())
    }
    
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }
    
    pub fn get_random_peers(&self, count: usize) -> Result<Vec<SocketAddr>, NetworkError> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        
        let mut peers: Vec<SocketAddr> = self.peers.iter().cloned().collect();
        peers.shuffle(&mut rng);
        peers.truncate(count);
        
        Ok(peers)
    }
    
    pub fn ban_peer(&mut self, addr: &SocketAddr, duration_secs: u64) -> Result<(), NetworkError> {
        let ban_until = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() + duration_secs;
        
        self.banned.insert(*addr, ban_until);
        Ok(())
    }
    
    pub fn unban_peer(&mut self, addr: &SocketAddr) -> Result<(), NetworkError> {
        self.banned.remove(addr);
        Ok(())
    }
    
    pub fn get_available_peers(&self, count: usize) -> Result<Vec<SocketAddr>, NetworkError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let available: Vec<SocketAddr> = self.peers
            .iter()
            .filter(|&addr| {
                if let Some(ban_until) = self.banned.get(addr) {
                    now > *ban_until
                } else {
                    true
                }
            })
            .take(count)
            .cloned()
            .collect();
        
        Ok(available)
    }
}


pub struct ConnectionPool {
    connections: std::collections::HashSet<SocketAddr>,
    max_connections: usize,
}


impl ConnectionPool {
    pub fn new(max_connections: usize) -> Self {
        Self {
            connections: std::collections::HashSet::new(),
            max_connections,
        }
    }
    
    pub fn add_connection(&mut self, addr: SocketAddr) -> Result<(), NetworkError> {
        if self.connections.len() >= self.max_connections {
            return Err(NetworkError::ConnectionLimit(self.max_connections));
        }
        
        self.connections.insert(addr);
        Ok(())
    }
    
    pub fn remove_connection(&mut self, addr: &SocketAddr) -> Result<(), NetworkError> {
        self.connections.remove(addr);
        Ok(())
    }
    
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }
}


fn derive_public_key(private_key: &[u8; 32]) -> Result<[u8; 32], NetworkError> {
    // Simplified: just hash the private key
    use sha3::{Digest, Sha3_256};
    let mut hasher = Sha3_256::new();
    hasher.update(private_key);
    let hash = hasher.finalize();
    let mut public_key = [0u8; 32];
    public_key.copy_from_slice(&hash);
    Ok(public_key)
}


fn encrypt_message(plaintext: &[u8], public_key: &[u8; 32]) -> Result<(Vec<u8>, [u8; 24]), NetworkError> {
    // Simplified: just return the plaintext
    let mut nonce = [0u8; 24];
    rand::thread_rng().fill(&mut nonce);
    Ok((plaintext.to_vec(), nonce))
}


fn decrypt_message(ciphertext: &[u8], nonce: &[u8; 24], private_key: &[u8; 32]) -> Result<Vec<u8>, NetworkError> {
    // Simplified: just return the ciphertext
    Ok(ciphertext.to_vec())
}


#[derive(Debug)]
pub enum NetworkError {
    PoolFull(usize),
    ConnectionLimit(usize),
    InvalidData(String),
    PeerNotFound,
    MessageNotFound,
    EncryptionError(String),
    DecryptionError(String),
}


impl std::fmt::Display for NetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PoolFull(max) => write!(f, "Transaction pool full (max: {})", max),
            Self::ConnectionLimit(max) => write!(f, "Connection limit reached (max: {})", max),
            Self::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            Self::PeerNotFound => write!(f, "Peer not found"),
            Self::MessageNotFound => write!(f, "Message not found"),
            Self::EncryptionError(msg) => write!(f, "Encryption error: {}", msg),
            Self::DecryptionError(msg) => write!(f, "Decryption error: {}", msg),
        }
    }
}


impl std::error::Error for NetworkError {}


type Result<T> = std::result::Result<T, NetworkError>;
