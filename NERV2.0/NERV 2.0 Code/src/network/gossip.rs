//! Gossipsub Message Propagation.
//!
//! NERV uses Gossipsub for efficient broadcast of:
//! - New blocks and embedding root announcements
//! - Consensus votes and partial signatures
//! - Gradient submissions (useful-work economy)
//! - Shard split/merge events
//! - DKG threshold decryption shares
//!
//! Each message type has its own topic with specific validation
//! rules and priority levels.

use crate::{
    ShardId, BlockHeight, Epoch, ValidatorId, TxHash, EmbeddingRoot,
    BATCH_MAX_SIZE,
    NervError, NervResult,
};
use crate::economy::weight_update::GradientSubmission;
use serde::{Deserialize, Serialize};

// ─── Gossip Topic ────────────────────────────────────────────────────────

/// A gossipsub topic identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GossipTopic {
    /// New block proposals and headers.
    Blocks,

    /// Consensus votes and partial BLS signatures.
    ConsensusVotes,

    /// Gradient submissions for useful-work economy.
    GradientSubmissions,

    /// Shard reconfiguration events (split/merge).
    ShardEvents,

    /// DKG threshold decryption shares.
    DecryptionShares,

    /// Network weight update proposals.
    WeightUpdates,

    /// Transaction inclusion notifications (tx_hash only).
    TransactionNotifications,

    /// Peer discovery/identity messages.
    PeerDiscovery,
}

impl GossipTopic {
    /// Get the topic string for libp2p gossipsub.
    pub fn topic_string(&self) -> String {
        match self {
            Self::Blocks => "/nerv/blocks/v2".into(),
            Self::ConsensusVotes => "/nerv/votes/v2".into(),
            Self::GradientSubmissions => "/nerv/gradients/v2".into(),
            Self::ShardEvents => "/nerv/shards/v2".into(),
            Self::DecryptionShares => "/nerv/decryption/v2".into(),
            Self::WeightUpdates => "/nerv/weights/v2".into(),
            Self::TransactionNotifications => "/nerv/tx-notify/v2".into(),
            Self::PeerDiscovery => "/nerv/peers/v2".into(),
        }
    }

    /// Get the priority level (higher = more important).
    pub fn priority(&self) -> u8 {
        match self {
            Self::ConsensusVotes => 10, // Highest: finality critical
            Self::DecryptionShares => 9,
            Self::Blocks => 8,
            Self::ShardEvents => 7,
            Self::WeightUpdates => 6,
            Self::GradientSubmissions => 5,
            Self::TransactionNotifications => 3,
            Self::PeerDiscovery => 1,
        }
    }

    /// Maximum message size for this topic.
    pub fn max_message_size(&self) -> usize {
        match self {
            Self::Blocks => 1024 * 1024,       // 1 MB (blocks can be large)
            Self::ConsensusVotes => 512,         // Small: hash + sig
            Self::GradientSubmissions => 4096,    // Gradient + metadata
            Self::ShardEvents => 2048,           // Split/merge descriptors
            Self::DecryptionShares => 1024,      // Partial decryption share
            Self::WeightUpdates => 2048,         // Weight hash + proof
            Self::TransactionNotifications => 64, // Just a tx_hash
            Self::PeerDiscovery => 512,          // Peer multiaddr
        }
    }

    /// Whether messages on this topic require validation.
    pub fn requires_validation(&self) -> bool {
        matches!(
            self,
            Self::Blocks
            | Self::ConsensusVotes
            | Self::GradientSubmissions
            | Self::DecryptionShares
            | Self::WeightUpdates
        )
    }

    /// All topics.
    pub fn all() -> &'static [GossipTopic] {
        &[
            Self::Blocks,
            Self::ConsensusVotes,
            Self::GradientSubmissions,
            Self::ShardEvents,
            Self::DecryptionShares,
            Self::WeightUpdates,
            Self::TransactionNotifications,
            Self::PeerDiscovery,
        ]
    }
}

impl std::fmt::Display for GossipTopic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.topic_string())
    }
}

// ─── Gossip Message ──────────────────────────────────────────────────────

/// A gossip message with metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GossipMessage {
    /// The topic.
    pub topic: GossipTopic,

    /// The message payload.
    pub payload: Vec<u8>,

    /// Sender's peer ID.
    pub sender: Vec<u8>,

    /// Message sequence number (for deduplication).
    pub seq_no: u64,

    /// Timestamp (Unix epoch millis).
    pub timestamp_ms: u64,
}

impl GossipMessage {
    /// Create a new gossip message.
    pub fn new(topic: GossipTopic, payload: Vec<u8>, sender: Vec<u8>) -> Self {
        Self {
            topic,
            payload,
            sender,
            seq_no: 0,
            timestamp_ms: current_epoch_millis(),
        }
    }

    /// With a sequence number.
    pub fn with_seq_no(mut self, seq_no: u64) -> Self {
        self.seq_no = seq_no;
        self
    }

    /// Compute a unique hash for deduplication.
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.topic.topic_string().as_bytes());
        hasher.update(&self.payload);
        hasher.update(&self.sender);
        hasher.update(&self.seq_no.to_le_bytes());
        hasher.finalize().into()
    }

    /// Check if the payload exceeds the topic's max size.
    pub fn is_oversized(&self) -> bool {
        self.payload.len() > self.topic.max_message_size()
    }
}

// ─── Typed Message Payloads ──────────────────────────────────────────────

/// A block announcement message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockAnnouncement {
    /// Block height.
    pub height: BlockHeight,

    /// Epoch.
    pub epoch: Epoch,

    /// Embedding root after this block.
    pub embedding_root: EmbeddingRoot,

    /// BLAKE3 hash of the block body.
    pub block_hash: [u8; 32],

    /// Number of transactions in this block.
    pub tx_count: u32,

    /// Proposer's validator ID.
    pub proposer: ValidatorId,
}

/// A consensus vote message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsensusVote {
    /// Block height being voted on.
    pub height: BlockHeight,

    /// The predicted embedding root hash.
    pub predicted_root: EmbeddingRoot,

    /// BLS12-381 partial signature share.
    pub partial_sig: Vec<u8>,

    /// Voter's validator ID.
    pub voter: ValidatorId,

    /// Voter's stake-weighted reputation.
    pub voting_weight: u128,
}

/// A shard event message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardEventMessage {
    /// A shard split occurred.
    Split {
        /// Parent shard.
        parent: ShardId,
        /// Left child.
        left: ShardId,
        /// Right child.
        right: ShardId,
        /// Height at which split occurred.
        height: BlockHeight,
    },

    /// A shard merge occurred.
    Merge {
        /// Left sibling.
        left: ShardId,
        /// Right sibling.
        right: ShardId,
        /// New parent.
        parent: ShardId,
        /// Height at which merge occurred.
        height: BlockHeight,
    },
}

/// A weight update proposal message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightUpdateMessage {
    /// Current epoch.
    pub epoch: Epoch,

    /// Hash of the current weights.
    pub weight_hash_before: [u8; 32],

    /// Hash of the proposed new weights.
    pub weight_hash_proposed: [u8; 32],

    /// Average Huber loss before update.
    pub avg_loss_before: f64,

    /// Predicted average Huber loss after update.
    pub avg_loss_after: f64,

    /// Number of contributing validators.
    pub contributor_count: usize,

    /// Proposer's validator ID.
    pub proposer: ValidatorId,
}

// ─── Gossip Validator ────────────────────────────────────────────────────

/// Validates incoming gossip messages.
pub struct GossipValidator {
    /// Maximum accepted age of a message (millis).
    pub max_message_age_ms: u64,

    /// Known malicious peers (blacklist).
    pub blacklisted_peers: Vec<Vec<u8>>,

    /// Minimum reputation to accept messages from.
    pub min_sender_reputation: u32,
}

impl GossipValidator {
    /// Create a new validator.
    pub fn new() -> Self {
        Self {
            max_message_age_ms: 60_000, // 1 minute
            blacklisted_peers: Vec::new(),
            min_sender_reputation: 100,
        }
    }

    /// Validate a gossip message.
    pub fn validate(&self, message: &GossipMessage) -> GossipValidationResult {
        // Check blacklist
        if self.blacklisted_peers.iter().any(|p| p == &message.sender) {
            return GossipValidationResult::Rejected("sender blacklisted".into());
        }

        // Check size
        if message.is_oversized() {
            return GossipValidationResult::Rejected(format!(
                "payload {} exceeds max {} for topic {}",
                message.payload.len(),
                message.topic.max_message_size(),
                message.topic
            ));
        }

        // Check age
        let now = current_epoch_millis();
        if now > message.timestamp_ms + self.max_message_age_ms {
            return GossipValidationResult::Rejected("message too old".into());
        }

        // Future messages are also invalid
        if message.timestamp_ms > now + 30_000 {
            return GossipValidationResult::Rejected("message from the future".into());
        }

        GossipValidationResult::Accepted
    }

    /// Add a peer to the blacklist.
    pub fn blacklist(&mut self, peer_id: Vec<u8>) {
        if !self.blacklisted_peers.contains(&peer_id) {
            self.blacklisted_peers.push(peer_id);
        }
    }

    /// Remove a peer from the blacklist.
    pub fn unblacklist(&mut self, peer_id: &[u8]) {
        self.blacklisted_peers.retain(|p| p != peer_id);
    }
}

impl Default for GossipValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of gossip message validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GossipValidationResult {
    /// Message is valid and should be propagated.
    Accepted,

    /// Message is invalid; reject and optionally penalize sender.
    Rejected(String),

    /// Message is valid but should not be propagated further.
    Ignore,
}

impl GossipValidationResult {
    /// Check if accepted.
    pub fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted)
    }
}

// ─── Gossip Configuration ────────────────────────────────────────────────

/// Configuration for the gossipsub protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipConfig {
    /// Heartbeat interval in milliseconds.
    pub heartbeat_ms: u64,

    /// Fanout degree (number of peers to send to on first broadcast).
    pub fanout: usize,

    /// History length (number of gossip rounds to remember).
    pub history_length: usize,

    /// History gossip factor (fraction of peers to gossip to).
    pub history_gossip_factor: f64,

    /// D parameter (target outbound peers).
    pub d: usize,

    /// Maximum transmit size.
    pub max_transmit_size: usize,

    /// Whether to flood publish.
    pub flood_publish: bool,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            heartbeat_ms: 500,
            fanout: 6,
            history_length: 5,
            history_gossip_factor: 0.25,
            d: 6,
            max_transmit_size: 1024 * 1024, // 1 MB
            flood_publish: true,
        }
    }
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
    fn test_gossip_topic_strings() {
        assert_eq!(GossipTopic::Blocks.topic_string(), "/nerv/blocks/v2");
        assert_eq!(GossipTopic::ConsensusVotes.topic_string(), "/nerv/votes/v2");
    }

    #[test]
    fn test_gossip_topic_priorities() {
        assert!(GossipTopic::ConsensusVotes.priority() > GossipTopic::Blocks.priority());
        assert!(GossipTopic::Blocks.priority() > GossipTopic::GradientSubmissions.priority());
    }

    #[test]
    fn test_gossip_message_creation() {
        let msg = GossipMessage::new(
            GossipTopic::Blocks,
            b"block data".to_vec(),
            vec![1u8; 32],
        );
        assert!(!msg.is_oversized());
        assert_ne!(msg.hash(), [0u8; 32]);
    }

    #[test]
    fn test_gossip_message_oversized() {
        let msg = GossipMessage::new(
            GossipTopic::ConsensusVotes, // Max 512 bytes
            vec![0u8; 1024],             // 1024 bytes
            vec![1u8; 32],
        );
        assert!(msg.is_oversized());
    }

    #[test]
    fn test_gossip_validator_accept() {
        let validator = GossipValidator::new();
        let msg = GossipMessage::new(
            GossipTopic::Blocks,
            b"data".to_vec(),
            vec![1u8; 32],
        );
        assert!(validator.validate(&msg).is_accepted());
    }

    #[test]
    fn test_gossip_validator_blacklisted() {
        let mut validator = GossipValidator::new();
        let sender = vec![1u8; 32];
        validator.blacklist(sender.clone());

        let msg = GossipMessage::new(
            GossipTopic::Blocks,
            b"data".to_vec(),
            sender,
        );
        assert!(!validator.validate(&msg).is_accepted());
    }

    #[test]
    fn test_gossip_validator_oversized() {
        let validator = GossipValidator::new();
        let msg = GossipMessage::new(
            GossipTopic::ConsensusVotes,
            vec![0u8; 1024],
            vec![1u8; 32],
        );
        assert!(!validator.validate(&msg).is_accepted());
    }

    #[test]
    fn test_gossip_validator_unblacklist() {
        let mut validator = GossipValidator::new();
        let sender = vec![1u8; 32];
        validator.blacklist(sender.clone());
        validator.unblacklist(&sender);
        // After unblacklist, should not be in the list
        assert!(!validator.blacklisted_peers.contains(&sender));
    }

    #[test]
    fn test_block_announcement_serialization() {
        let announcement = BlockAnnouncement {
            height: BlockHeight::from(100),
            epoch: Epoch::from(0),
            embedding_root: EmbeddingRoot::GENESIS,
            block_hash: [1u8; 32],
            tx_count: 5000,
            proposer: ValidatorId::from_bytes([2u8; 32]),
        };
        let serialized = bincode::serialize(&announcement).unwrap();
        let deserialized: BlockAnnouncement = bincode::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.height, BlockHeight::from(100));
    }

    #[test]
    fn test_consensus_vote_serialization() {
        let vote = ConsensusVote {
            height: BlockHeight::from(100),
            predicted_root: EmbeddingRoot::GENESIS,
            partial_sig: vec![0u8; 48],
            voter: ValidatorId::from_bytes([1u8; 32]),
            voting_weight: 1000,
        };
        let serialized = bincode::serialize(&vote).unwrap();
        assert!(serialized.len() > 0);
    }

    #[test]
    fn test_shard_event_split() {
        let event = ShardEventMessage::Split {
            parent: ShardId::new(1),
            left: ShardId::new(2),
            right: ShardId::new(3),
            height: BlockHeight::from(100),
        };
        let serialized = bincode::serialize(&event).unwrap();
        let deserialized: ShardEventMessage = bincode::deserialize(&serialized).unwrap();
        assert_eq!(deserialized, event);
    }

    #[test]
    fn test_weight_update_message() {
        let msg = WeightUpdateMessage {
            epoch: Epoch::from(5),
            weight_hash_before: [1u8; 32],
            weight_hash_proposed: [2u8; 32],
            avg_loss_before: 1.0,
            avg_loss_after: 0.8,
            contributor_count: 10,
            proposer: ValidatorId::from_bytes([3u8; 32]),
        };
        let serialized = bincode::serialize(&msg).unwrap();
        assert!(serialized.len() > 0);
    }

    #[test]
    fn test_gossip_config_default() {
        let config = GossipConfig::default();
        assert_eq!(config.heartbeat_ms, 500);
        assert!(config.flood_publish);
    }
}
