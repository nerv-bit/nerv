//! The per-shard transit log (WP §3.4, §4.5; D.3): an authenticated sorted
//! map keyed by H(txid ‖ shard — leg_index), leaves binding full entry
//! records (state included). Entries are consumed exactly once — Claim or
//! deterministic Reversion. Proofs are default-subtree-compressed (B9 shape,
//! ~1 KB at scale vs. 8 KB raw). Value never escrows here; only completion
//! conditions do (§3.4).

use std::collections::BTreeMap;
use std::fmt;
use std::sync::OnceLock;
use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::{TRANSIT, TRANSIT_EMPTY, TRANSIT_ENTRY, TRANSIT_NODE};
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::params::CUSTODY_D3_EXPIRY_GRACE_BLOCKS_L_GRACE;
use nerv_core::types::{Height, LegIndex, ShardId, TxId};

use crate::error::CustodyError;

pub const DEPTH: u16 = 256;

/// H(txid ‖ shard ‖ leg_index) — the entry's position key (WP §3.4).
pub fn transit_key(txid: &TxId, shard: &ShardId, leg: LegIndex) -> Hash256 {
    let mut msg = Vec::with_capacity(32 + 3 + 1);
    msg.extend_from_slice(txid.as_bytes());
    msg.push(shard.bits());
    msg.extend_from_slice(&shard.value().to_le_bytes());
    msg.push(leg.as_u8());
    Hash256::concat(&TRANSIT, &msg)
}

/// D.3's three states; consumed exactly once, by either path.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TransitEntryState {
    Pending,
    Claimed,
    Reverted,
}

impl TransitEntryState {
    pub const fn as_u8(self) -> u8 {
        match self {
            TransitEntryState::Pending => 0,
            TransitEntryState::Claimed => 1,
            TransitEntryState::Reverted => 2,
        }
    }

    pub fn from_u8(b: u8) -> Option<TransitEntryState> {
        match b {
            0 => Some(TransitEntryState::Pending),
            1 => Some(TransitEntryState::Claimed),
            2 => Some(TransitEntryState::Reverted),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            TransitEntryState::Pending => "Pending",
            TransitEntryState::Claimed => "Claimed",
            TransitEntryState::Reverted => "Reverted",
        }
    }
}

impl Encode for TransitEntryState {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.push(self.as_u8());
    }
    fn encoded_len(&self) -> usize {
        1
    }
}

impl Decode for TransitEntryState {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        TransitEntryState::from_u8(r.read_u8()?)
            .ok_or(CodecError::InvariantViolated("unknown transit entry state"))
    }
}

/// One transit entry: leg coordinates, lifecycle state, height of insertion
/// (Pending) or consumption (Claimed/Reverted), declared expiry height.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TransitEntry {
    pub txid: TxId,
    pub shard: ShardId,
    pub leg: LegIndex,
    pub state: TransitEntryState,
    pub height: Height,
    pub expiry: Height,
}

impl TransitEntry {
    pub fn new_pending(
        txid: TxId,
        shard: ShardId,
        leg: LegIndex,
        height: Height,
        expiry: Height,
    ) -> TransitEntry {
        TransitEntry {
            txid,
            shard,
            leg,
            state: TransitEntryState::Pending,
            height,
            expiry,
        }
    }

    pub fn key(&self) -> Hash256 {
        transit_key(&self.txid, &self.shard, self.leg)
    }

    fn leaf_hash(&self) -> Hash256 {
        let mut msg = Vec::with_capacity(32 + 3 + 1 + 1 + 8 + 8);
        msg.extend_from_slice(self.txid.as_bytes());
        msg.push(self.shard.bits());
        msg.extend_from_slice(&shard_bytes(self.shard));
        msg.push(self.leg.as_u8());
        msg.push(self.state.as_u8());
        msg.extend_from_slice(&self.height.as_u64().to_le_bytes());
        msg.extend_from_slice(&self.expiry.as_u64().to_le_bytes());
        Hash256::concat(&TRANSIT_ENTRY, &msg)
    }

    /// D.3: a Pending entry past expiry + L_grace is due for reversion.
    pub fn reversion_due(&self, current: Height) -> bool {
        self.state == TransitEntryState::Pending
            && current.as_u64() >= self.expiry.as_u64() + CUSTODY_D3_EXPIRY_GRACE_BLOCKS_L_GRACE as u64
    }
}

fn shard_bytes(s: ShardId) -> [u8; 2] {
    s.value().to_le_bytes()
}

impl Encode for TransitEntry {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.txid.encode_into(out);
        self.shard.encode_into(out);
        self.leg.encode_into(out);
        self.state.encode_into(out);
        self.height.encode_into(out);
        self.expiry.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        32 + 3 + 1 + 1 + 8 + 8
    }
}

impl Decode for TransitEntry {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let txid = TxId::decode_from(r)?;
        let shard = ShardId::decode_from(r)?;
        let leg = LegIndex::decode_from(r)?;
        let state = TransitEntryState::from_u8(r.read_u8()?)
            .ok_or(CodecError::InvariantViolated("unknown transit entry state"))?;
        let height = Height::decode_from(r)?;
        let expiry = Height::decode_from(r)?;
        Ok(TransitEntry { txid, shard, leg, state, height, expiry })
    }
}

fn node_hash(l: &Hash256, r: &Hash256) -> Hash256 {
    let mut msg = [0u8; 64];
    msg[..32].copy_from_slice(l.as_bytes());
    msg[32..].copy_from_slice(r.as_bytes());
    Hash256::concat(&TRANSIT_NODE, &msg)
}

fn default_hashes() -> &'static [Hash256; 257] {
    static TABLE: OnceLock<[Hash256; 257]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut arr = [Hash256::from_bytes([0u8; 32]); 257];
        arr[256] = Hash256::concat(&TRANSIT_EMPTY, &[0u8; 32]);
        for k in (0..256).rev() {
            arr[k] = node_hash(&arr[k + 1], &arr[k + 1]);
        }
        arr
    })
}

fn mask(key: &[u8; 32], level: u16) -> [u8; 32] {
    let mut out = *key;
    let full = (level / 8) as usize;
    for b in out.iter_mut().skip(full) {
        *b = 0;
    }
    let rem = (level % 8) as u8;
    if rem != 0 {
        out[full] &= 0xFFu8 << (8 - rem);
    }
    out
}

fn bit(key: &[u8; 32], b: u16) -> bool {
    (key[(b / 8) as usize] >> (7 - (b % 8))) & 1 == 1
}

fn sibling_prefix(key: &[u8; 32], j: u16) -> [u8; 32] {
    let mut sib = mask(key, j);
    let byte = ((j - 1) / 8) as usize;
    sib[byte] ^= 1u8 << (7 - ((j - 1) % 8));
    sib
}

fn fold(start: Hash256, key: &[u8; 32], siblings: &[(u16, Hash256)]) -> Option<Hash256> {
    let defaults = default_hashes();
    let mut map: BTreeMap<u16, Hash256> = BTreeMap::new();
    for &(level, h) in siblings {
        if level == 0 || level > DEPTH || map.insert(level, h).is_some() {
            return None;
        }
    }
    let mut node = start;
    for j in (1..=DEPTH).rev() {
        let sib = map.remove(&j).unwrap_or(defaults[j as usize]);
        node = if bit(key, j - 1) {
            node_hash(&sib, &node)
        } else {
            node_hash(&node, &sib)
        };
    }
    Some(node)
}

/// A transit-log witness: membership (entry included) or non-membership.
#[derive(Clone, PartialEq, Eq)]
pub struct TransitProof {
    pub entry: Option<TransitEntry>,
    pub siblings: Vec<(u16, Hash256)>,
}

impl fmt::Debug for TransitProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransitProof")
            .field("entry", &self.entry)
            .field("sibling_entries", &self.siblings.len())
            .finish()
    }
}

impl TransitProof {
    /// Verify that exactly this entry (state and all) is at its key.
    pub fn verify_membership(&self, root: &Hash256, entry: &TransitEntry) -> bool {
        self.entry.as_ref() == Some(entry)
            && fold(entry.leaf_hash(), entry.key().as_bytes(), &self.siblings)
                .map(|h| h == *root)
                .unwrap_or(false)
    }

    /// Verify that `key` is absent from the log (fresh transit — §4.3 rule 4).
    pub fn verify_non_membership(&self, root: &Hash256, key: &Hash256) -> bool {
        self.entry.is_none()
            && fold(default_hashes()[256], key.as_bytes(), &self.siblings)
                .map(|h| h == *root)
                .unwrap_or(false)
    }
}

impl Encode for TransitProof {
    fn encode_into(&self, out: &mut Vec<u8>) {
        match &self.entry {
            None => out.push(0),
            Some(e) => {
                out.push(1);
                e.encode_into(out);
            }
        }
        out.extend_from_slice(&(self.siblings.len() as u32).to_le_bytes());
        for (l, h) in &self.siblings {
            out.extend_from_slice(&l.to_le_bytes());
            h.encode_into(out);
        }
    }
    fn encoded_len(&self) -> usize {
        1 + self.entry.as_ref().map_or(0, |e| e.encoded_len())
            + 4 + self.siblings.len() * 34
    }
}

impl Decode for TransitProof {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let entry = match r.read_u8()? {
            0 => None,
            1 => Some(TransitEntry::decode_from(r)?),
            tag => return Err(CodecError::InvalidOptionTag { tag }),
        };
        let n = r.read_seq_len()?;
        r.enter()?;
        let mut siblings = Vec::with_capacity(n.min(1024));
        for _ in 0..n {
            let l = r.read_u16()?;
            if l == 0 || l > DEPTH {
                return Err(CodecError::InvariantViolated("sibling level out of range"));
            }
            let h = Hash256::decode_from(r)?;
            siblings.push((l, h));
        }
        r.leave();
        Ok(TransitProof { entry, siblings })
    }
}

/// The per-shard transit log: a sparse Merkle tree over 256-bit transit
/// keys, leaves binding full entry records.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TransitLog {
    nodes: BTreeMap<(u16, [u8; 32]), Hash256>,
    entries: BTreeMap<[u8; 32], TransitEntry>,
}

impl TransitLog {
    pub fn new() -> TransitLog {
        TransitLog::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn root(&self) -> Hash256 {
        self.nodes
            .get(&(0, [0u8; 32]))
            .copied()
            .unwrap_or(default_hashes()[0])
    }

    pub fn get(&self, key: &Hash256) -> Option<&TransitEntry> {
        self.entries.get(key.as_bytes())
    }

    pub fn contains_key(&self, key: &Hash256) -> bool {
        self.entries.contains_key(key.as_bytes())
    }

    fn recompute_path(&mut self, entry: &TransitEntry) {
        let key = *entry.key().as_bytes();
        let mut node = entry.leaf_hash();
        for j in (1..=DEPTH).rev() {
            self.nodes.insert((j, mask(&key, j)), node);
            let sib = self
                .nodes
                .get(&(j, sibling_prefix(&key, j)))
                .copied()
                .unwrap_or(default_hashes()[j as usize]);
            node = if bit(&key, j - 1) {
                node_hash(&sib, &node)
            } else {
                node_hash(&node, &sib)
            };
        }
        self.nodes.insert((0, [0u8; 32]), node);
    }

    /// Record a new Pending entry (settlement of an issuing leg — §4.3).
    pub fn insert_pending(&mut self, entry: TransitEntry) -> Result<(), CustodyError> {
        let key = *entry.key().as_bytes();
        if self.entries.contains_key(&key) {
            return Err(CustodyError::TransitEntryExists { key: entry.key() });
        }
        self.recompute_path(&entry);
        self.entries.insert(key, entry);
        Ok(())
    }

    fn consume(
        &mut self,
        key: &Hash256,
        next: TransitEntryState,
        height: Height,
        height_rule: Option<Height>,
    ) -> Result<TransitEntry, CustodyError> {
        let Some(mut e) = self.entries.get(key.as_bytes()).copied() else {
            return Err(CustodyError::TransitAbsent { key: *key });
        };
        if e.state != TransitEntryState::Pending {
            return Err(CustodyError::TransitEntryConsumed { state: e.state.name() });
        }
        if let Some(bound) = height_rule {
            if height.as_u64() < bound.as_u64() {
                return Err(CustodyError::ReversionTooEarly {
                    current: height,
                    earliest: bound,
                });
            }
        }
        e.state = next;
        e.height = height;
        self.recompute_path(&e);
        self.entries.insert(*key.as_bytes(), e);
        Ok(e)
    }

    /// D.3 Claim: consume a Pending entry via issue-condition settlement.
    /// The settlement-deadline check (expiry ≥ receiving shard's height) is
    /// the executor's rule 7 — enforced there, against the same parameters.
    pub fn claim(&mut self, key: &Hash256, height: Height) -> Result<TransitEntry, CustodyError> {
        self.consume(key, TransitEntryState::Claimed, height, None)
    }

    /// D.3 deterministic Reversion: consume a Pending entry at height ≥
    /// expiry + L_grace. The no-settlement precondition is proven by the
    /// caller from beacon-finalized data; this enforces the height rule
    /// (wrongful early reversion is fraud-provable from public hashes).
    pub fn revert(&mut self, key: &Hash256, height: Height) -> Result<TransitEntry, CustodyError> {
        let e = self
            .entries
            .get(key.as_bytes())
            .ok_or(CustodyError::TransitAbsent { key: *key })?;
        let earliest = Height::from_u64(
            e.expiry.as_u64() + CUSTODY_D3_EXPIRY_GRACE_BLOCKS_L_GRACE as u64,
        );
        self.consume(key, TransitEntryState::Reverted, height, Some(earliest))
    }

    fn collect_siblings(&self, key: &[u8; 32]) -> Vec<(u16, Hash256)> {
        let mut out = Vec::new();
        for j in 1..=DEPTH {
            if let Some(&h) = self.nodes.get(&(j, sibling_prefix(key, j))) {
                out.push((j, h));
            }
        }
        out
    }

    pub fn membership_proof(&self, entry: &TransitEntry) -> TransitProof {
        TransitProof {
            entry: Some(*entry),
            siblings: self.collect_siblings(entry.key().as_bytes()),
        }
    }

    pub fn non_membership_proof(&self, key: &Hash256) -> TransitProof {
        TransitProof { entry: None, siblings: self.collect_siblings(key.as_bytes()) }
    }

    /// Entries due for deterministic reversion at `current` (D.3 scan).
    pub fn reversion_due(&self, current: Height) -> Vec<TransitEntry> {
        self.entries
            .values()
            .filter(|e| e.reversion_due(current))
            .copied()
            .collect()
    }

    /// Full audit: rebuild from the entry set and compare node-for-node.
    pub fn validate_consistency(&self) -> Result<(), CustodyError> {
        let mut fresh = TransitLog::new();
        for e in self.entries.values() {
            fresh.recompute_path(e);
            fresh.entries.insert(*e.key().as_bytes(), *e);
        }
        if fresh.nodes != self.nodes {
            return Err(CustodyError::InvalidTransitLog(
                "stored nodes diverge from the entry set",
            ));
        }
        Ok(())
    }
}

impl Encode for TransitLog {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.nodes.len() as u32).to_le_bytes());
        for (&(level, prefix), h) in &self.nodes {
            out.extend_from_slice(&level.to_le_bytes());
            out.extend_from_slice(&prefix);
            h.encode_into(out);
        }
        out.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for e in self.entries.values() {
            e.encode_into(out);
        }
    }
    fn encoded_len(&self) -> usize {
        4 + self.nodes.len() * 66 + 4 + self.entries.len() * (32 + 3 + 1 + 1 + 8 + 8)
    }
}

impl Decode for TransitLog {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let n = r.read_seq_len()?;
        r.enter()?;
        let mut nodes = BTreeMap::new();
        for _ in 0..n {
            let level = r.read_u16()?;
            if level > DEPTH {
                return Err(CodecError::InvariantViolated("transit node level out of range"));
            }
            let prefix = r.take_array::<32>()?;
            if mask(&prefix, level) != prefix {
                return Err(CodecError::InvariantViolated("transit prefix not canonical"));
            }
            let h = Hash256::decode_from(r)?;
            if nodes.insert((level, prefix), h).is_some() {
                return Err(CodecError::InvariantViolated("duplicate transit node"));
            }
        }
        r.leave();
        let m = r.read_seq_len()?;
        r.enter()?;
        let mut entries = BTreeMap::new();
        for _ in 0..m {
            let e = TransitEntry::decode_from(r)?;
            if entries.insert(*e.key().as_bytes(), e).is_some() {
                return Err(CodecError::InvariantViolated("duplicate transit entry"));
            }
        }
        r.leave();
        Ok(TransitLog { nodes, entries })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;
    use nerv_core::types::ShardSet;

    fn txid(rng: &mut SplitMix64) -> TxId {
        TxId::from_hash(Hash256::from_bytes(rng.bytes32()))
    }

    fn entry(rng: &mut SplitMix64, h: u64, expiry: u64) -> TransitEntry {
        TransitEntry::new_pending(
            txid(rng),
            ShardSet::genesis().ids()[7].to_owned(),
            LegIndex::from_u8((rng.next_u64() % 4) as u8),
            Height::from_u64(h),
            Height::from_u64(expiry),
        )
    }

    #[test]
    fn transit_key_formula() {
        let t = txid(&mut SplitMix64::new(1));
        let shard = ShardSet::genesis().ids()[3];
        let leg = LegIndex::from_u8(2);
        let k = transit_key(&t, &shard, leg);
        let mut msg = Vec::new();
        msg.extend_from_slice(t.as_bytes());
        msg.push(shard.bits());
        msg.extend_from_slice(&shard.value().to_le_bytes());
        msg.push(2);
        assert_eq!(k, Hash256::concat(&TRANSIT, &msg));
        // distinct legs/shards/txids give distinct keys
        assert_ne!(k, transit_key(&t, &shard, LegIndex::from_u8(3)));
        assert_ne!(k, transit_key(&t, &ShardSet::genesis().ids()[4], leg));
        assert_ne!(k, transit_key(&txid(&mut SplitMix64::new(2)), &shard, leg));
    }

    #[test]
    fn insert_and_states_consume_once() {
        let mut rng = SplitMix64::new(3);
        let e = entry(&mut rng, 100, 1000);
        let mut log = TransitLog::new();
        assert!(log.is_empty());
        assert_eq!(log.root(), default_hashes()[0]);
        assert!(log.non_membership_proof(&e.key()).siblings.is_empty());

        log.insert_pending(e).unwrap();
        assert_eq!(log.len(), 1);
        assert!(matches!(
            log.insert_pending(e),
            Err(CustodyError::TransitEntryExists { .. })
        ));

        // non-membership for a fresh key against the non-empty root
        let fresh = entry(&mut rng, 100, 2000);
        let root = log.root();
        assert!(log
            .non_membership_proof(&fresh.key())
            .verify_non_membership(&root, &fresh.key()));

        // Claim consumes exactly once
        let claimed = log.claim(&e.key(), Height::from_u64(120)).unwrap();
        assert_eq!(claimed.state, TransitEntryState::Claimed);
        assert_eq!(claimed.height.as_u64(), 120);
        assert!(matches!(
            log.claim(&e.key(), Height::from_u64(121)),
            Err(CustodyError::TransitEntryConsumed { state: "Claimed" })
        ));
        assert!(matches!(
            log.revert(&e.key(), Height::from_u64(999_999)),
            Err(CustodyError::TransitEntryConsumed { state: "Claimed" })
        ));
        log.validate_consistency().unwrap();
    }

    #[test]
    fn reversion_rule_and_errors() {
        let mut rng = SplitMix64::new(4);
        let e = entry(&mut rng, 10, 500); // expiry 500, L_grace 10
        let mut log = TransitLog::new();
        log.insert_pending(e).unwrap();

        // before expiry + L_grace: not due, and revert() is an error
        assert!(e.reversion_due(Height::from_u64(509)).eq(&false));
        assert!(matches!(
            log.revert(&e.key(), Height::from_u64(509)),
            Err(CustodyError::ReversionTooEarly { .. })
        ));
        // at expiry + L_grace exactly: due
        assert!(e.reversion_due(Height::from_u64(510)));
        let reverted = log.revert(&e.key(), Height::from_u64(510)).unwrap();
        assert_eq!(reverted.state, TransitEntryState::Reverted);
        assert_eq!(reverted.height.as_u64(), 510);
        // consume-once after reversion
        assert!(matches!(
            log.claim(&e.key(), Height::from_u64(600)),
            Err(CustodyError::TransitEntryConsumed { state: "Reverted" })
        ));
        assert!(matches!(
            log.revert(&e.key(), Height::from_u64(511)),
            Err(CustodyError::TransitEntryConsumed { state: "Reverted" })
        ));
        // absent key
        let ghost = entry(&mut rng, 1, 2);
        assert!(matches!(
            log.claim(&ghost.key(), Height::from_u64(1)),
            Err(CustodyError::TransitAbsent { .. })
        ));
        log.validate_consistency().unwrap();
    }

    #[test]
    fn reversion_due_scan() {
        let mut rng = SplitMix64::new(5);
        let mut log = TransitLog::new();
        let e1 = entry(&mut rng, 5, 100);
        let e2 = entry(&mut rng, 5, 100);
        let e3 = entry(&mut rng, 5, 500);
        let e4 = entry(&mut rng, 5, 500);
        log.insert_pending(e1).unwrap();
        log.insert_pending(e2).unwrap();
        log.insert_pending(e3).unwrap();
        log.claim(&e1.key(), Height::from_u64(20)).unwrap();
        log.insert_pending(e4).unwrap();
        log.claim(&e3.key(), Height::from_u64(20)).unwrap();
        let due = log.reversion_due(Height::from_u64(2000));
        assert_eq!(due.len(), 2);
        assert!(due.contains(&e2));
        assert!(due.contains(&e4));
        assert!(!due.contains(&e1));
        assert!(!due.contains(&e3));
    }

    #[test]
    fn membership_proofs_bind_state() {
        let mut rng = SplitMix64::new(6);
        let mut entries = Vec::new();
        let mut log = TransitLog::new();
        for i in 0..16 {
            let e = entry(&mut rng, 10, 100 + i);
            log.insert_pending(e).unwrap();
            entries.push(e);
        }
        let root = log.root();
        for e in &entries {
            let p = log.membership_proof(e);
            assert!(p.verify_membership(&root, e));
        }
        // a stale (pre-consumption) proof fails after claim
        let target = entries[5];
        let stale = log.membership_proof(&target);
        log.claim(&target.key(), Height::from_u64(77)).unwrap();
        let root2 = log.root();
        assert!(!stale.verify_membership(&root2, &target), "old entry no longer at key");
        let consumed = log.get(&target.key()).copied().unwrap();
        let fresh = log.membership_proof(&consumed);
        assert!(fresh.verify_membership(&root2, &consumed));
        // wrong root
        assert!(!fresh.verify_membership(&root, &consumed));
        // wrong entry (state swap)
        let mut wrong = consumed;
        wrong.state = TransitEntryState::Pending;
        assert!(!fresh.verify_membership(&root2, &wrong));
        // tampered siblings
        let mut bad = fresh.clone();
        if !bad.siblings.is_empty() {
            bad.siblings[0].1 = Hash256::from_bytes(rng.bytes32());
            assert!(!bad.verify_membership(&root2, &consumed));
        }
        // omitted live sibling
        let mut missing = fresh.clone();
        if !missing.siblings.is_empty() {
            missing.siblings.remove(0);
            assert!(!missing.verify_membership(&root2, &consumed));
        }
        // non-membership still works for unclaimed keys
        let ghost = entry(&mut rng, 1, 2);
        assert!(log
            .non_membership_proof(&ghost.key())
            .verify_non_membership(&root2, &ghost.key()));
        assert!(!log
            .non_membership_proof(&target.key())
            .verify_non_membership(&root2, &target.key()));
    }

    #[test]
    fn proof_sizes() {
        let mut rng = SplitMix64::new(7);
        let mut log = TransitLog::new();
        for i in 0..4096 {
            let e = entry(&mut rng, 1, 1000);
            log.insert_pending(e).unwrap();
            assert!(e.expiry.as_u64() == 1000 || i == usize::MAX);
        }
        let root = log.root();
        let samples: Vec<&TransitEntry> = log.entries.values().step_by(64).collect();
        let mut total_bytes = 0;
        for e in samples {
            let p = log.membership_proof(e);
            assert!(p.verify_membership(&root, e));
            total_bytes += p.encode().len();
        }
        let avg = total_bytes / 64;
        assert!(avg <= 2100, "avg transit proof {avg} bytes (target ~1 KB)");
        for _ in 0..8 {
            let ghost = entry(&mut rng, 1, 2);
            let p = log.non_membership_proof(&ghost.key());
            assert!(p.siblings.len() < 32);
            assert!(p.verify_non_membership(&root, &ghost.key()));
        }
    }

    #[test]
    fn root_is_order_independent() {
        let mut rng = SplitMix64::new(8);
        let mut entries = Vec::new();
        for i in 0..25 {
            entries.push(entry(&mut rng, 10 + i, 100 + i));
        }
        let mut a = TransitLog::new();
        for e in &entries {
            a.insert_pending(*e).unwrap();
        }
        let mut shuffled = entries.clone();
        let mut r2 = SplitMix64::new(99);
        for i in (1..shuffled.len()).rev() {
            let j = (r2.next_u64() % (i as u64 + 1)) as usize;
            shuffled.swap(i, j);
        }
        let mut b = TransitLog::new();
        for e in &shuffled {
            b.insert_pending(*e).unwrap();
        }
        assert_eq!(a.root(), b.root());
        assert_eq!(a, b);
    }

    #[test]
    fn serialization_roundtrip_and_rejection() {
        let mut rng = SplitMix64::new(9);
        let mut log = TransitLog::new();
        for i in 0..10 {
            log.insert_pending(entry(&mut rng, 10, 100 + i)).unwrap();
        }
        let target = log.entries.values().next().copied().unwrap();
        log.claim(&target.key(), Height::from_u64(30)).unwrap();
        let enc = log.encode();
        assert_eq!(enc.len(), log.encoded_len());
        let d = TransitLog::decode(&enc).unwrap();
        assert_eq!(d.root(), log.root());
        assert_eq!(d.len(), 10);
        d.validate_consistency().unwrap();
        assert!(TransitLog::decode(&enc[..enc.len() - 1]).is_err());

        // tampered node hash decodes but fails audit
        let mut bad = enc.clone();
        bad[4 + 2 + 32] ^= 1;
        let t = TransitLog::decode(&bad).unwrap();
        assert!(t.validate_consistency().is_err());

        // entry codec roundtrip
        let e = log.entries.values().next().copied().unwrap();
        let eenc = e.encode();
        assert_eq!(eenc.len(), e.encoded_len());
        assert_eq!(TransitEntry::decode(&eenc).unwrap(), e);
        assert!(TransitEntry::decode(&eenc[..eenc.len() - 1]).is_err());
        // unknown state byte
        let mut bad_e = eenc.clone();
        let state_off = 32 + 3 + 1;
        bad_e[state_off] = 9;
        assert!(TransitEntry::decode(&bad_e).is_err());

        // proof codec roundtrip
        let p = log.membership_proof(&e);
        let penc = p.encode();
        assert_eq!(penc.len(), p.encoded_len());
        let p2 = TransitProof::decode(&penc).unwrap();
        assert_eq!(p2, p);
        assert!(p2.verify_membership(&log.root(), &e));
        assert!(TransitProof::decode(&penc[..penc.len() - 1]).is_err());
        let np = log.non_membership_proof(&entry(&mut rng, 1, 2).key());
        let nenc = np.encode();
        assert_eq!(TransitProof::decode(&nenc).unwrap(), np);
    }
}

