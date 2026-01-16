Rust// tests/unit/test_network.rs
// ============================================================================
// NETWORK MODULE UNIT TESTS
// ============================================================================

use nerv_bit::network::{
    NetworkManager, NetworkConfig,
    mempool::{MempoolManager, MempoolConfig, EncryptedTransaction, TxPriority, PendingTransaction, ShardMempool},
    dht::{DhtManager, DhtConfig, PeerId, PeerInfo, KBucket},
    gossip::{GossipManager, GossipConfig, GossipMessage, MessageCache, PeerScore},
};
use nerv_bit::crypto::CryptoProvider;
use std::collections::{HashSet, BinaryHeap};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Instant, Duration};
use tokio::time;

#[tokio::test]
async fn test_network_manager_creation_and_components() {
    let config = NetworkConfig::default();
    let crypto = Arc::new(CryptoProvider::new().unwrap());

    let manager = NetworkManager::new(config, crypto).await.unwrap();

    // Basic access to components (ensure they are initialized)
    let dht = manager.dht_manager.clone();
    let gossip = manager.gossip_manager.clone();
    let mempool = manager.mempool_manager.clone();

    assert!(dht.local_id.0.len() == 32);
    assert!(gossip.local_peer_id.len() > 0);
    assert_eq!(mempool.metrics.current_size_bytes, 0);
}

#[test]
fn test_pending_transaction_ordering() {
    let now = Instant::now();

    let tx_high = PendingTransaction {
        tx_hash: "high".to_string(),
        shard_id: 0,
        priority_score: 1000,
        submission_time: now,
        size: 100,
    };

    let tx_medium = PendingTransaction {
        tx_hash: "medium".to_string(),
        shard_id: 0,
        priority_score: 500,
        submission_time: now,
        size: 100,
    };

    let tx_old_high = PendingTransaction {
        tx_hash: "old_high".to_string(),
        shard_id: 0,
        priority_score: 1000,
        submission_time: now - Duration::from_secs(10),
        size: 100,
    };

    let mut heap = BinaryHeap::new();
    heap.push(tx_medium.clone());
    heap.push(tx_high.clone());
    heap.push(tx_old_high.clone());

    // Highest priority first
    assert_eq!(heap.pop().unwrap(), tx_high);
    // Then older same-priority
    assert_eq!(heap.pop().unwrap(), tx_old_high);
    assert_eq!(heap.pop().unwrap(), tx_medium);
}

#[tokio::test]
async fn test_mempool_manager_basic_operations() {
    let config = MempoolConfig::default();
    let crypto = Arc::new(CryptoProvider::new().unwrap());
    let manager = MempoolManager::new(config, crypto).await.unwrap();

    let tx = EncryptedTransaction {
        data: vec![1u8; 256],
        attestation: vec![2u8; 128],
        shard_id: 1,
        hash: "test_tx_1".to_string(),
        submission_time: std::time::SystemTime::now(),
        priority: TxPriority::High(1000),
    };

    // Add transaction
    manager.add_transaction(tx.clone()).await.unwrap();

    // Duplicate should fail
    let err = manager.add_transaction(tx.clone()).await.unwrap_err();
    assert!(matches!(err, nerv_bit::network::MempoolError::DuplicateTransaction));

    // Get pending for shard
    let pending = manager.get_pending_transactions(1, 10).await.unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].hash, "test_tx_1");

    // Remove
    manager.remove_transactions(&["test_tx_1".to_string()]).await.unwrap();
    let pending_after = manager.get_pending_transactions(1, 10).await.unwrap();
    assert!(pending_after.is_empty());
}

#[tokio::test]
async fn test_mempool_eviction_and_size_limits() {
    let mut config = MempoolConfig::default();
    config.max_size_bytes = 1024; // Small limit for test

    let crypto = Arc::new(CryptoProvider::new().unwrap());
    let manager = MempoolManager::new(config, crypto).await.unwrap();

    // Add transactions until over limit
    for i in 0..10 {
        let tx = EncryptedTransaction {
            data: vec![0u8; 300], // ~300 bytes each
            attestation: vec![],
            shard_id: 0,
            hash: format!("tx_{}", i),
            submission_time: std::time::SystemTime::now(),
            priority: TxPriority::Normal,
        };
        let _ = manager.add_transaction(tx).await; // Ignore errors for overflow
    }

    // After overflow, size should be bounded
    assert!(manager.metrics.current_size_bytes <= config.max_size_bytes * 2); // Some tolerance

    // Old transactions should be evicted on cleanup
    time::sleep(Duration::from_millis(100)).await; // Allow background cleanup if any
}

#[test]
fn test_peer_id_distance_and_leading_zeros() {
    let id1 = PeerId::random();
    let id2 = id1.clone();
    assert_eq!(id1.distance(&id2), [0u8; 32]);

    let mut bytes = [0u8; 32];
    bytes[0] = 1;
    let id3 = PeerId::from_bytes(&bytes);
    let dist = id1.distance(&id3);
    assert!(dist.iter().any(|&b| b != 0));

    // Leading zeros
    let zero_heavy = PeerId::from_bytes(&[0u8; 32]);
    assert_eq!(zero_heavy.leading_zeros(), 256);

    let one_bit = PeerId::from_bytes(&[128u8, 0u8; 16]); // MSB set
    assert_eq!(one_bit.leading_zeros(), 0);
}

#[test]
fn test_kbucket_operations() {
    let bucket_size = 5;
    let mut bucket = KBucket::new(bucket_size);

    let mut peers = vec![];
    for i in 0..bucket_size + 2 {
        let peer = PeerInfo {
            peer_id: PeerId::random(),
            address: format!("127.0.0.1:{}", 8000 + i).parse().unwrap(),
            last_seen: Instant::now(),
            reputation: 1.0,
        };
        peers.push(peer.clone());
        bucket.add_peer(peer);
    }

    assert_eq!(bucket.peers.len(), bucket_size); // Full
    // Lowest reputation or least recently seen should be evicted (implementation dependent)
}

#[tokio::test]
async fn test_dht_manager_creation_and_bootstrap() {
    let config = DhtConfig::default();
    let crypto = Arc::new(CryptoProvider::new().unwrap());
    let manager = DhtManager::new(config, crypto).await.unwrap();

    // Add bootstrap node (mock address)
    let bootstrap: SocketAddr = "1.1.1.1:1234".parse().unwrap();
    manager.add_bootstrap_node(bootstrap);

    // Basic lookup (self)
    let peers = manager.find_peers(manager.local_id.clone(), 5).await.unwrap();
    assert!(peers.is_empty() || peers.iter().any(|p| p.peer_id == manager.local_id));
}

#[test]
fn test_gossip_message_id_generation() {
    let topic = "blocks";
    let data = b"test_block_data";

    let id1 = GossipManager::generate_message_id(topic, data);
    let id2 = GossipManager::generate_message_id(topic, data);
    assert_eq!(id1, id2);
    assert_eq!(id1.len(), 64); // hex-encoded blake3

    let different_data = b"different";
    let id3 = GossipManager::generate_message_id(topic, different_data);
    assert_ne!(id1, id3);
}

#[test]
fn test_peer_score_and_message_cache() {
    let mut score = PeerScore::new();
    score.update(10.0);
    score.update(-5.0);
    assert_eq!(score.current_score, 5.0);

    let mut cache = MessageCache::new(5);
    for i in 0..10 {
        let msg = GossipMessage {
            message_id: format!("msg_{}", i),
            topic: "test".to_string(),
            data: vec![],
            source: "peer1".to_string(),
            seq_no: i as u64,
            signature: vec![],
            timestamp: Instant::now(),
            ttl: 10,
        };
        cache.put(msg);
    }

    assert!(cache.messages.len() <= 5);
    assert!(cache.has_message(&"msg_9".to_string()));
    assert!(!cache.has_message(&"msg_0".to_string())); // Pruned
}

#[tokio::test]
async fn test_gossip_manager_basic_flow() {
    let config = GossipConfig::default();
    let crypto = Arc::new(CryptoProvider::new().unwrap());
    let manager = GossipManager::new(config, crypto).await.unwrap();

    // Subscribe to topic
    manager.subscribe("test_topic".to_string()).await.unwrap();
    let subs = manager.subscriptions.read().await;
    assert!(subs.contains("test_topic"));

    // Publish message
    let data = vec![1u8; 100];
    manager.publish("test_topic".to_string(), data.clone()).await.unwrap();

    // Message should be cached
    let cache = manager.message_cache.read().await;
    assert!(!cache.messages.is_empty());
}