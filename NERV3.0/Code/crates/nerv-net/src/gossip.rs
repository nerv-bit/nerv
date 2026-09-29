//! Mesh gossip and the DSR-8 ordering rule (WP §10.3, §6.2): headers —
//! carrying H(Δ̂_B) and H(ct_B) — propagate before any decryption partial
//! exists; partials for unknown headers are rejected, never buffered.
//!
//! The engine is pure logic; the node's event loop drives it:
//!
//! ```text
//! HostEvent::Connected(p)    => engine.on_connected(p)
//! HostEvent::Disconnected(p) => engine.on_disconnected(p)
//! HostEvent::Frame(p, b)     => match engine.receive(p, &b)? {
//!     Inbound::Accepted { forward, .. } => {
//!         for q in engine.recipients(Some(&p)) { host.send(&q, forward.clone())?; }
//!     }
//!     _ => {}
//! }
//! // local publication:
//! match engine.publish(msg) {
//!     PublishOutcome::Broadcast { frame } =>
//!         for q in engine.recipients(None) { host.send(&q, frame.clone())?; }
//!     _ => {}
//! }
//! ```
//!
//! The ordering binding carried by a Partial is (shard, height) — the
//! protocol-level reference. The cryptographic binding of a partial to
//! its block is the VPD proof's u_B context, verified by the ceremony
//! layer; gossip enforces ORDERING, the ceremony enforces BINDING.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::GOSSIP_MSG;
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::types::{Height, ShardId};
use nerv_seal::decrypt::ChunkReveal;
use nerv_seal::dkg::sigma::Proof;
use nerv_seal::ring::Poly;
use nerv_seal::vpd::PartialDecryption;
use nerv_state::ShardHeader;
use nerv_da::{CellAuth, SetCommitment};

use crate::host::PeerId;

/// Bounded dedup: accepted-message digests, FIFO eviction.
pub const DEDUP_CAPACITY: usize = 65_536;
/// Per-shard trailing header retention. Partials follow their header
/// within seconds; a partial for a pruned height targets a block whose
/// ceremony is already dead — rejection is correct (erratum 147).
pub const HEADER_WINDOW: u64 = 256;
/// Maximum BlockData payload (4 MiB, matching the host frame cap).
pub const MAX_BLOCK_DATA_LEN: usize = 4 * 1024 * 1024;


/// The dedup digest of a wire payload.
pub fn message_digest(payload: &[u8]) -> Hash256 {
    Hash256::concat(&GOSSIP_MSG, payload)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GossipMessage {
    Header { shard: ShardId, header: ShardHeader },
    Partial { shard: ShardId, height: Height, batch: u64, partial: PartialDecryption },
     Reveal { shard: ShardId, height: Height, batch: u64, reveal: ChunkReveal },
    /// A DA blob-set advertisement for `shard:height` — the per-blob-row
    /// `SetCommitment` (tag 6, gap 4). Carries no cells, only the
    /// commitment the client uses to seed its sampling positions. DA data
    /// is independent of the header/partial ordering rule — it is not
    /// gated by `known_header` (DSR-8 only pins consensus messages).
    DABlob { shard: ShardId, height: Height, set_commitment: SetCommitment },
    /// A DA sampling response — one verified `CellAuth` against the
    /// advertised `set_commitment` (tag 7, gap 4). Likewise ungated.
    DACell { shard: ShardId, height: Height, cell_auth: CellAuth },
    /// The full encoded ShardBlock for a finalized block (erratum 207).
    /// Gated by the header-before-data ordering rule (DSR-8). Moved to
    /// tag 8 to free 6/7 for the gap-4 DA topics.
    BlockData { shard: ShardId, height: Height, data: Vec<u8> },
}


impl Encode for GossipMessage {
    fn encode_into(&self, out: &mut Vec<u8>) {
        match self {
            GossipMessage::Header { shard, header } => {
                out.push(0);
                shard.encode_into(out);
                header.encode_into(out);
            }
            GossipMessage::Partial { shard, height, batch, partial } => {
                out.push(1);
                shard.encode_into(out);
                out.extend_from_slice(&height.as_u64().to_le_bytes());
                out.extend_from_slice(&batch.to_le_bytes());
                out.extend_from_slice(&partial.to_bytes());
            }
            GossipMessage::Reveal { shard, height, batch, reveal } => {
                out.push(2);
                shard.encode_into(out);
                out.extend_from_slice(&height.as_u64().to_le_bytes());
                out.extend_from_slice(&batch.to_le_bytes());
                out.extend_from_slice(&reveal.to_bytes());
            }
            GossipMessage::BlockData { shard, height, data } => {
                out.push(8);
                shard.encode_into(out);
                out.extend_from_slice(&height.as_u64().to_le_bytes());
                out.extend_from_slice(&(data.len() as u32).to_le_bytes());
                out.extend_from_slice(data);
            }
            GossipMessage::DABlob { shard, height, set_commitment } => {
                out.push(6);
                shard.encode_into(out);
                out.extend_from_slice(&height.as_u64().to_le_bytes());
                set_commitment.encode_into(out);
            }
            GossipMessage::DACell { shard, height, cell_auth } => {
                out.push(7);
                shard.encode_into(out);
                out.extend_from_slice(&height.as_u64().to_le_bytes());
                cell_auth.encode_into(out);
            }
        }
    }
    fn encoded_len(&self) -> usize {
        match self {
            GossipMessage::Header { shard, header } => {
                1 + shard.encoded_len() + header.encoded_len()
            }
            GossipMessage::Partial { shard, partial, .. } => {
                1 + shard.encoded_len() + 16 + partial.to_bytes().len()
            }
            GossipMessage::Reveal { shard, .. } => {
                1 + shard.encoded_len() + 16 + ChunkReveal::WIRE_SIZE
            }
            GossipMessage::BlockData { shard, data, .. } => {
                1 + shard.encoded_len() + 8 + 4 + data.len()
            }
            GossipMessage::DABlob { shard, set_commitment, .. } => {
                1 + shard.encoded_len() + 8 + set_commitment.encoded_len()
            }
            GossipMessage::DACell { shard, cell_auth, .. } => {
                1 + shard.encoded_len() + 8 + cell_auth.encoded_len()
            }
        }
    }

}

impl Decode for GossipMessage {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        match r.read_u8()? {
            0 => Ok(GossipMessage::Header {
                shard: ShardId::decode_from(r)?,
                header: ShardHeader::decode_from(r)?,
            }),
            1 => {
                let shard = ShardId::decode_from(r)?;
                let height = Height::from_u64(r.read_u64()?);
                let batch = r.read_u64()?;
                let bytes = r.take(r.remaining())?;
                let partial = PartialDecryption::from_bytes(bytes)
                    .map_err(|_| CodecError::InvariantViolated("malformed decryption partial"))?;
                Ok(GossipMessage::Partial { shard, height, batch, partial })
            }
            2 => {
                let shard = ShardId::decode_from(r)?;
                let height = Height::from_u64(r.read_u64()?);
                let batch = r.read_u64()?;
                let bytes = r.take(r.remaining())?;
                let reveal = ChunkReveal::from_bytes(bytes)
                    .map_err(|_| CodecError::InvariantViolated("malformed chunk reveal"))?;
                Ok(GossipMessage::Reveal { shard, height, batch, reveal })
            }
            6 => {
                let shard = ShardId::decode_from(r)?;
                let height = Height::from_u64(r.read_u64()?);
                let set_commitment = SetCommitment::decode_from(r)?;
                Ok(GossipMessage::DABlob { shard, height, set_commitment })
            }
            7 => {
                let shard = ShardId::decode_from(r)?;
                let height = Height::from_u64(r.read_u64()?);
                let cell_auth = CellAuth::decode_from(r)?;
                Ok(GossipMessage::DACell { shard, height, cell_auth })
            }
            8 => {
                let shard = ShardId::decode_from(r)?;
                let height = Height::from_u64(r.read_u64()?);
                let n = r.read_seq_len()?;
                if n > MAX_BLOCK_DATA_LEN {
                    return Err(CodecError::SeqTooLarge { count: n, max: MAX_BLOCK_DATA_LEN });
                }
                let data = r.take(n)?.to_vec();
                Ok(GossipMessage::BlockData { shard, height, data })
            }
            tag => Err(CodecError::InvalidOptionTag { tag }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejection {
    PartialBeforeHeader { shard: ShardId, height: u64 },
    BlockDataBeforeHeader { shard: ShardId, height: u64 },
    ZeroHeightHeader { shard: ShardId },
}


#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Inbound {
    Accepted { message: GossipMessage, forward: Vec<u8> },
    Duplicate,
    Rejected(Rejection),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PublishOutcome {
    Broadcast { frame: Vec<u8> },
    Duplicate,
    Rejected(Rejection),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GossipStats {
    pub accepted: u64,
    pub duplicates: u64,
    pub rejected_partials: u64,
    pub rejected_headers: u64,
    pub headers_learned: u64,
    pub pruned_headers: u64,
}

/// The flood-gossip engine: dedup, the DSR-8 ordering gate, header
/// learning with window pruning, and the connected-peer set.
#[derive(Clone, Debug)]
pub struct GossipEngine {
    peers: BTreeSet<PeerId>,
    seen: BTreeSet<Hash256>,
    seen_order: VecDeque<Hash256>,
    headers: BTreeMap<ShardId, BTreeMap<u64, Hash256>>,
    stats: GossipStats,
}

impl Default for GossipEngine {
    fn default() -> Self {
        GossipEngine::new()
    }
}

impl GossipEngine {
    pub fn new() -> GossipEngine {
        GossipEngine {
            peers: BTreeSet::new(),
            seen: BTreeSet::new(),
            seen_order: VecDeque::new(),
            headers: BTreeMap::new(),
            stats: GossipStats::default(),
        }
    }

    pub fn on_connected(&mut self, peer: PeerId) {
        self.peers.insert(peer);
    }

    pub fn on_disconnected(&mut self, peer: &PeerId) {
        self.peers.remove(peer);
    }

    pub fn peers(&self) -> Vec<PeerId> {
        self.peers.iter().copied().collect()
    }

    /// Forward targets: every connected peer, minus the source.
    pub fn recipients(&self, exclude: Option<&PeerId>) -> Vec<PeerId> {
        match exclude {
            Some(e) => self.peers.iter().filter(|&p| p != e).copied().collect(),
            None => self.peers.iter().copied().collect(),
        }
    }

    pub fn stats(&self) -> GossipStats {
        self.stats
    }

    /// The learned header at (shard, height), if any (first-wins across
    /// forks; presence is what the gate checks).
    pub fn known_header(&self, shard: &ShardId, height: u64) -> Option<Hash256> {
        self.headers.get(shard)?.get(&height).copied()
    }

    /// The inbound path: decode, dedup, gate, learn, and hand back the
    /// forward frame on acceptance.
    pub fn receive(&mut self, from: PeerId, payload: &[u8]) -> Result<Inbound, CodecError> {
        let message = GossipMessage::decode(payload)?;
        let digest = message_digest(payload);
        if self.seen.contains(&digest) {
            self.stats.duplicates += 1;
            return Ok(Inbound::Duplicate);
        }
        match self.gate_and_learn(&message) {
            Some(rejection) => {
                self.note_rejection(&rejection);
                Ok(Inbound::Rejected(rejection))
            }
            None => {
                self.insert_seen(digest);
                self.stats.accepted += 1;
                Ok(Inbound::Accepted { message, forward: payload.to_vec() })
            }
        }
    }

    /// The local publication path: identical rules — a partial published
    /// before its header is known is rejected at the origin too.
    pub fn publish(&mut self, message: GossipMessage) -> PublishOutcome {
        let payload = message.encode();
        let digest = message_digest(&payload);
        if self.seen.contains(&digest) {
            self.stats.duplicates += 1;
            return PublishOutcome::Duplicate;
        }
        match self.gate_and_learn(&message) {
            Some(rejection) => {
                self.note_rejection(&rejection);
                PublishOutcome::Rejected(rejection)
            }
            None => {
                self.insert_seen(digest);
                self.stats.accepted += 1;
                PublishOutcome::Broadcast { frame: payload }
            }
        }
    }

     fn note_rejection(&mut self, rejection: &Rejection) {
        match rejection {
            Rejection::PartialBeforeHeader { .. } | Rejection::BlockDataBeforeHeader { .. } => {
                self.stats.rejected_partials += 1
            }
            Rejection::ZeroHeightHeader { .. } => self.stats.rejected_headers += 1,
        }
    }


    fn gate_and_learn(&mut self, message: &GossipMessage) -> Option<Rejection> {
        match message {
            GossipMessage::Header { shard, header } => {
                let height = header.height.as_u64();
                if height == 0 {
                    return Some(Rejection::ZeroHeightHeader { shard: *shard });
                }
                self.learn_header(*shard, height, header.header_hash());
                None
            }
            GossipMessage::Partial { shard, height, .. } => {
                if self.known_header(shard, height.as_u64()).is_none() {
                    return Some(Rejection::PartialBeforeHeader {
                        shard: *shard,
                        height: height.as_u64(),
                    });
                }
                None
            }
            GossipMessage::BlockData { shard, height, .. } => {
                if self.known_header(shard, height.as_u64()).is_none() {
                    return Some(Rejection::BlockDataBeforeHeader {
                        shard: *shard,
                        height: height.as_u64(),
                    });
                }
                None
            }
            GossipMessage::Reveal { .. } => None,
            // DSR-8 only pins consensus messages (header/partial/blockdata).
            // DA topics are independent — sampled availability runs ahead of
            // and behind block finalization; the advertise-and-respond flow
            // is its own transport (gap 4).
            GossipMessage::DABlob { .. } | GossipMessage::DACell { .. } => None,


        }
    }

    fn learn_header(&mut self, shard: ShardId, height: u64, hash: Hash256) {
        let stats = &mut self.stats;
        let map = self.headers.entry(shard).or_default();
        map.entry(height).or_insert(hash);
        stats.headers_learned += 1;
        if let Some((&max, _)) = map.iter().next_back() {
            let floor = max.saturating_sub(HEADER_WINDOW);
            let stale: Vec<u64> = map.range(..=floor).map(|(h, _)| *h).collect();
            stats.pruned_headers += stale.len() as u64;
            for h in stale {
                map.remove(&h);
            }
        }
    }

    fn insert_seen(&mut self, digest: Hash256) {
        self.seen.insert(digest);
        self.seen_order.push_back(digest);
        while self.seen_order.len() > DEDUP_CAPACITY {
            if let Some(old) = self.seen_order.pop_front() {
                self.seen.remove(&old);
            }
        }
    }
}
