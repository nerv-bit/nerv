//! Core vocabulary types (WP §3.6, §4.2–4.3, §8.2, §11.2).
//!
//! ShardSet models the ACTIVE homing partition of §8.2 (prefix trie: no id a
//! prefix of another, union covers all bit strings). Legacy shards — split
//! parents whose notes remain spendable — stay valid `ShardId`s outside any
//! active set; ledger topology is a state-layer concern.

use std::cmp::Ordering;
use std::fmt;
use std::ops::Range;

use crate::codec::{Decode, Encode, Reader};
use crate::constants;
use crate::error::{CodecError, TypeError};
use crate::hash::Hash256;

/// Representation bound on shard-id depth (value fits u16). The protocol's
/// active-count cap (1024) is enforced by `ShardSet` (WP §8.2, App B).
pub const MAX_SHARD_BITS: u8 = 16;

/// Beacon intervals per epoch (86_400 at genesis parameters).
pub const INTERVALS_PER_EPOCH: u64 =
    crate::params::TIMING_EPOCH_SECS / crate::params::TIMING_BEACON_INTERVAL_SECS;

const _: () = assert!(crate::params::PROTOCOL_SHARD_COUNT_MAX <= 1u64 << MAX_SHARD_BITS);

// ---------------------------------------------------------------------------
// ShardId
// ---------------------------------------------------------------------------

/// A binary shard id: the `bits`-bit binary expansion of `value`.
/// Canonical order = trie DFS pre-order (padded key, then depth).
#[derive(Clone, Copy, Hash, Debug)]
pub struct ShardId {
    bits: u8,
    value: u16,
}

impl ShardId {
    /// The whole key space as a single shard (depth 0).
    pub const ROOT: ShardId = ShardId { bits: 0, value: 0 };

    pub const fn new(bits: u8, value: u16) -> Result<ShardId, TypeError> {
        if bits > MAX_SHARD_BITS {
            return Err(TypeError::ShardIdTooLong { bits, max: MAX_SHARD_BITS });
        }
        if bits < 16 {
            if value >= (1u16 << bits) {
                return Err(TypeError::OutOfRange {
                    what: "shard id value",
                    value: value as u64,
                    min: 0,
                    max: ((1u16 << bits) as u64) - 1,
                });
            }
        }
        Ok(ShardId { bits, value })
    }

    pub const fn bits(&self) -> u8 {
        self.bits
    }

    pub const fn value(&self) -> u16 {
        self.value
    }

    pub const fn is_root(&self) -> bool {
        self.bits == 0
    }

    /// Child `p‖bit` (WP §8.2 split).
    pub const fn child(&self, right: bool) -> Result<ShardId, TypeError> {
        if self.bits == MAX_SHARD_BITS {
            return Err(TypeError::ShardIdTooLong { bits: self.bits + 1, max: MAX_SHARD_BITS });
        }
        ShardId::new(self.bits + 1, self.value * 2 + if right { 1u16 } else { 0u16 })
    }

    /// Parent `p` (drops the last bit); `None` for the root.
    pub const fn parent(&self) -> Option<ShardId> {
        if self.bits == 0 {
            None
        } else {
            Some(ShardId { bits: self.bits - 1, value: self.value / 2 })
        }
    }

    /// Same-depth sibling (last bit flipped); `None` for the root.
    pub const fn sibling(&self) -> Option<ShardId> {
        if self.bits == 0 {
            None
        } else {
            Some(ShardId { bits: self.bits, value: self.value ^ 1 })
        }
    }

    /// The i-th bit of the id string, MSB-first (requires `i < bits`).
    pub const fn bit(&self, i: u8) -> bool {
        debug_assert!(i < self.bits);
        (self.value >> (self.bits - 1 - i)) & 1 == 1
    }

    /// True iff this id is a prefix of `other` (the root prefixes everything).
    pub const fn is_prefix_of(&self, other: &ShardId) -> bool {
        if self.bits == 0 {
            return true;
        }
        self.bits <= other.bits && (other.value >> (other.bits - self.bits)) == self.value
    }

    /// True iff this id is a prefix of κ (first `bits` bits, MSB-first).
    pub fn is_kappa_home(&self, kappa: &Hash256) -> bool {
        if self.bits == 0 {
            return true;
        }
        let key = kappa_key(kappa);
        (key >> (MAX_SHARD_BITS - self.bits)) == self.value as u32
    }

    /// Start of this id's interval in the 2^16 key space (DFS sort key).
    const fn padded(&self) -> u32 {
        (self.value as u32) << (MAX_SHARD_BITS - self.bits)
    }

    /// Length of this id's interval in the 2^16 key space.
    const fn span(&self) -> u32 {
        1u32 << (MAX_SHARD_BITS - self.bits)
    }
}

impl PartialEq for ShardId {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits && self.value == other.value
    }
}
impl Eq for ShardId {}
impl PartialOrd for ShardId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ShardId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.padded().cmp(&other.padded()).then(self.bits.cmp(&other.bits))
    }
}

impl fmt::Display for ShardId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.bits == 0 {
            return f.write_str("(root)");
        }
        for i in 0..self.bits {
            write!(f, "{}", u8::from(self.bit(i)))?;
        }
        Ok(())
    }
}

impl Encode for ShardId {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.push(self.bits);
        out.extend_from_slice(&self.value.to_le_bytes());
    }
    fn encoded_len(&self) -> usize {
        3
    }
}

impl Decode for ShardId {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let bits = r.read_u8()?;
        let value = r.read_u16()?;
        ShardId::new(bits, value)
            .map_err(|_| CodecError::InvariantViolated("invalid shard id on the wire"))
    }
}

/// First `MAX_SHARD_BITS` (16) bits of a BLAKE3 digest, MSB-first: the
/// top-16-bit key κ is matched against by the prefix trie.
fn kappa_key(kappa: &Hash256) -> u32 {
    let b = kappa.as_bytes();
    u16::from_be_bytes([b[0], b[1]]) as u32
}

/// κ(addr) — WP §8.2: the first 256 bits of BLAKE3("nerv.shard" ‖ addr).
/// `addr` is the address's canonical encoding (defined by nerv-custody).
pub fn kappa(addr: &[u8]) -> Hash256 {
    Hash256::concat(&constants::SHARD_HOMING, addr)
}

// ---------------------------------------------------------------------------
// ShardSet
// ---------------------------------------------------------------------------

/// The active shard set: a prefix partition of the key space (WP §8.2).
/// `ids` is kept in canonical DFS order; validity is an invariant maintained
/// by every constructor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardSet {
    ids: Vec<ShardId>,
}

impl ShardSet {
    /// Genesis topology: 64 shards with 6-bit ids 000000–111111 (WP §8.2).
    pub fn genesis() -> ShardSet {
        let ids: Vec<ShardId> = (0..64u16).map(|v| ShardId { bits: 6, value: v }).collect();
        debug_assert!(ShardSet::from_vec(ids.clone()).is_ok());
        ShardSet { ids }
    }

    /// Validate and build: non-empty, no duplicates, prefix-free, full
    /// coverage (checked exactly as interval tiling of [0, 2^16)), and
    /// active count within the protocol cap.
    pub fn from_vec(mut ids: Vec<ShardId>) -> Result<ShardSet, TypeError> {
        if ids.is_empty() {
            return Err(TypeError::InvalidShardSet { reason: "empty set" });
        }
        ids.sort();
        for w in ids.windows(2) {
            if w[0] == w[1] {
                return Err(TypeError::InvalidShardSet { reason: "duplicate shard id" });
            }
        }
        let mut expected = 0u32;
        for id in &ids {
            let start = id.padded();
            if start != expected {
                return Err(TypeError::InvalidShardSet {
                    reason: if start < expected {
                        "one shard id is a prefix of another"
                    } else {
                        "prefix trie does not cover the whole key space"
                    },
                });
            }
            expected = start + id.span();
        }
        if expected != 1u32 << MAX_SHARD_BITS {
            return Err(TypeError::InvalidShardSet {
                reason: "prefix trie does not cover the whole key space",
            });
        }
        if ids.len() as u64 > crate::params::PROTOCOL_SHARD_COUNT_MAX {
            return Err(TypeError::InvalidShardSet {
                reason: "active shard count exceeds the protocol maximum",
            });
        }
        Ok(ShardSet { ids })
    }

    pub fn from_slice(ids: &[ShardId]) -> Result<ShardSet, TypeError> {
        ShardSet::from_vec(ids.to_vec())
    }

    /// Active ids in canonical DFS order.
    pub fn ids(&self) -> &[ShardId] {
        &self.ids
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Always false: a valid partition is never empty.
    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn contains(&self, id: &ShardId) -> bool {
        self.ids.binary_search(id).is_ok()
    }

    /// Home shard of an address under the full prefix rule (WP §8.2).
    pub fn home(&self, addr: &[u8]) -> Result<ShardId, TypeError> {
        self.home_kappa(&kappa(addr))
    }

    /// Home shard for a precomputed κ: the unique active id that is a prefix
    /// of κ. Infallible on a valid set.
    pub fn home_kappa(&self, kappa: &Hash256) -> Result<ShardId, TypeError> {
        let key = kappa_key(kappa);
        let idx = self.ids.partition_point(|id| id.padded() + id.span() <= key);
        match self.ids.get(idx) {
            Some(id) if id.padded() <= key => Ok(*id),
            _ => Err(TypeError::InvalidShardSet {
                reason: "internal: home lookup failed on a validated set",
            }),
        }
    }

    /// Replace `target` with its two children (WP §8.2). Only the homing of
    /// new addresses changes; existing notes keep their shard tags.
    pub fn split(&self, target: &ShardId) -> Result<ShardSet, TypeError> {
        if !self.contains(target) {
            return Err(TypeError::InvalidShardSet { reason: "shard id is not active" });
        }
        let left = target.child(false)?;
        let right = target.child(true)?;
        let mut ids: Vec<ShardId> = Vec::with_capacity(self.ids.len() + 1);
        ids.extend(self.ids.iter().copied().filter(|id| id != target));
        ids.push(left);
        ids.push(right);
        ShardSet::from_vec(ids)
    }

    /// Replace the sibling pair containing `sibling` with their parent
    /// (WP §8.2 merge).
    pub fn merge(&self, sibling: &ShardId) -> Result<ShardSet, TypeError> {
        if !self.contains(sibling) {
            return Err(TypeError::InvalidShardSet { reason: "shard id is not active" });
        }
        let Some(parent) = sibling.parent() else {
            return Err(TypeError::InvalidShardSet {
                reason: "root shard has no parent or sibling",
            });
        };
        let Some(other) = sibling.sibling() else {
            return Err(TypeError::InvalidShardSet {
                reason: "root shard has no parent or sibling",
            });
        };
        if !self.contains(&other) {
            return Err(TypeError::InvalidShardSet {
                reason: "merge requires both siblings active",
            });
        }
        let mut ids: Vec<ShardId> = Vec::with_capacity(self.ids.len() - 1);
        ids.extend(self.ids.iter().copied().filter(|id| id != sibling && id != &other));
        ids.push(parent);
        ShardSet::from_vec(ids)
    }
}

impl Encode for ShardSet {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.ids.len() as u32).to_le_bytes());
        for id in &self.ids {
            id.encode_into(out);
        }
    }
    fn encoded_len(&self) -> usize {
        4 + 3 * self.ids.len()
    }
}

impl Decode for ShardSet {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let n = r.read_seq_len()?;
        r.enter()?;
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            ids.push(ShardId::decode_from(r)?);
        }
        r.leave();
        ShardSet::from_vec(ids)
            .map_err(|_| CodecError::InvariantViolated("invalid shard set on the wire"))
    }
}

// ---------------------------------------------------------------------------
// Height / Interval / Epoch / FeeSats
// ---------------------------------------------------------------------------

macro_rules! impl_u64_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
        pub struct $name(u64);

        impl $name {
            pub const ZERO: $name = $name(0);
            pub const fn from_u64(v: u64) -> $name {
                $name(v)
            }
            pub const fn as_u64(self) -> u64 {
                self.0
            }
        }

        impl Encode for $name {
            fn encode_into(&self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.0.to_le_bytes());
            }
            fn encoded_len(&self) -> usize {
                8
            }
        }

        impl Decode for $name {
            fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
                r.read_u64().map($name)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

impl_u64_newtype! {
    /// Block height of a shard chain (WP §4.2).
    Height
}
impl_u64_newtype! {
    /// Beacon / registry interval index; 1-second intervals (WP §5.5, §8.7).
    Interval
}
impl_u64_newtype! {
    /// Committee-rotation epoch index; 24 h at genesis parameters (WP §8.3).
    Epoch
}
impl_u64_newtype! {
    /// Declared fee per leg, in nano-NERV (WP §3.6, §4.6). Range constraints
    /// (floors, conservation) are enforced at settlement, not by the type.
    FeeSats
}

impl Interval {
    /// Interval containing wall-clock second `secs` (floor division).
    pub const fn from_secs(secs: u64) -> Interval {
        Interval(secs / crate::params::TIMING_BEACON_INTERVAL_SECS)
    }

    /// First second of the interval (saturating on overflow).
    pub const fn start_secs(self) -> u64 {
        self.0.saturating_mul(crate::params::TIMING_BEACON_INTERVAL_SECS)
    }

    /// Epoch containing this interval.
    pub const fn epoch(self) -> Epoch {
        Epoch(self.0 / INTERVALS_PER_EPOCH)
    }
}

impl Epoch {
    /// Epoch containing wall-clock second `secs` (floor division).
    pub const fn from_secs(secs: u64) -> Epoch {
        Epoch(secs / crate::params::TIMING_EPOCH_SECS)
    }

    /// First second of the epoch (saturating on overflow).
    pub const fn start_secs(self) -> u64 {
        self.0.saturating_mul(crate::params::TIMING_EPOCH_SECS)
    }

    /// First interval of the epoch (`None` on u64 overflow).
    pub const fn first_interval(self) -> Option<Interval> {
        match self.0.checked_mul(INTERVALS_PER_EPOCH) {
            Some(s) => Some(Interval(s)),
            None => None,
        }
    }

    /// Half-open interval index range of this epoch (`None` on overflow).
    pub const fn interval_span(self) -> Option<Range<u64>> {
        let s = match self.0.checked_mul(INTERVALS_PER_EPOCH) {
            Some(s) => s,
            None => return None,
        };
        match s.checked_add(INTERVALS_PER_EPOCH) {
            Some(e) => Some(s..e),
            None => None,
        }
    }
}

impl FeeSats {
    /// Checked sum of fee shares (block fee totals — WP §4.6).
    pub const fn checked_add(self, rhs: FeeSats) -> Option<FeeSats> {
        match self.0.checked_add(rhs.0) {
            Some(v) => Some(FeeSats(v)),
            None => None,
        }
    }
}

// ---------------------------------------------------------------------------
// LegIndex / TxId / LegKey
// ---------------------------------------------------------------------------

/// Index of a leg within its transaction. One byte: WP §11.2's 33-byte
/// txid+leg-index witness element bounds a transaction to at most 256 legs.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct LegIndex(u8);

impl LegIndex {
    pub const FIRST: LegIndex = LegIndex(0);
    pub const MAX: LegIndex = LegIndex(u8::MAX);

    pub const fn from_u8(v: u8) -> LegIndex {
        LegIndex(v)
    }

    pub const fn as_u8(self) -> u8 {
        self.0
    }

    pub const fn next(self) -> Option<LegIndex> {
        match self.0.checked_add(1) {
            Some(v) => Some(LegIndex(v)),
            None => None,
        }
    }
}

impl Encode for LegIndex {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.push(self.0);
    }
    fn encoded_len(&self) -> usize {
        1
    }
}

impl Decode for LegIndex {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        r.read_u8().map(LegIndex)
    }
}

impl fmt::Display for LegIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Transaction identity: `txid = BLAKE3("nerv.txid" ‖ canonical serialization
/// of all legs)` (WP §3.6). Ord is byte-lexicographic over the digest.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct TxId(Hash256);

impl TxId {
    pub const fn from_hash(h: Hash256) -> TxId {
        TxId(h)
    }

    pub const fn as_hash(&self) -> Hash256 {
        self.0
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        self.0.as_bytes()
    }

    /// Hash the canonical serialization of all legs into the txid.
    pub fn hash_canonical(legs_serialization: &[u8]) -> TxId {
        TxId(Hash256::concat(&constants::TXID, legs_serialization))
    }
}

impl Encode for TxId {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self.as_bytes());
    }
    fn encoded_len(&self) -> usize {
        32
    }
}

impl Decode for TxId {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Hash256::decode_from(r).map(TxId)
    }
}

impl fmt::Display for TxId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// Canonical settlement order key — WP §4.3: "legs are canonically ordered
/// by (txid, leg_index)". Derived Ord implements exactly that order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct LegKey {
    pub txid: TxId,
    pub leg: LegIndex,
}

impl LegKey {
    pub const fn new(txid: TxId, leg: LegIndex) -> LegKey {
        LegKey { txid, leg }
    }
}

impl Encode for LegKey {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.txid.encode_into(out);
        self.leg.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        33
    }
}

impl Decode for LegKey {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let txid = TxId::decode_from(r)?;
        let leg = LegIndex::decode_from(r)?;
        Ok(LegKey { txid, leg })
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;
    use crate::testutil::{codec_roundtrip, SplitMix64};

    fn s(bits: u8, value: u16) -> ShardId {
        ShardId::new(bits, value).unwrap()
    }

    fn kappa_of(bytes: &[u8]) -> Hash256 {
        Hash256::from_bytes(bytes[..].try_into().unwrap())
    }

    #[test]
    fn shard_id_new_bounds() {
        assert_eq!(ShardId::new(0, 0), Ok(ShardId::ROOT));
        assert_eq!(ShardId::new(1, 1), Ok(ShardId { bits: 1, value: 1 }));
        assert!(ShardId::new(16, u16::MAX).is_ok());
        assert!(matches!(ShardId::new(17, 0), Err(TypeError::ShardIdTooLong { .. })));
        assert!(matches!(ShardId::new(0, 1), Err(TypeError::OutOfRange { .. })));
        assert!(matches!(ShardId::new(6, 64), Err(TypeError::OutOfRange { .. })));
    }

    #[test]
    fn shard_id_dfs_order() {
        // "0" < "00" < "01" < "1" — trie pre-order, not (bits, value) order.
        let mut ids = vec![s(1, 1), s(2, 1), s(1, 0), s(2, 0)];
        ids.sort();
        assert_eq!(ids, vec![s(1, 0), s(2, 0), s(2, 1), s(1, 1)]);
        assert!(s(2, 1) < s(1, 1));
        assert_eq!(ShardId::ROOT.to_string(), "(root)");
        assert_eq!(s(3, 5).to_string(), "101"); // 0b101
    }

    #[test]
    fn shard_id_prefix_family() {
        // (3, 5) = "101"
        let id = s(3, 5);
        assert!(id.bit(0) && !id.bit(1) && id.bit(2));
        assert!(s(1, 0).is_prefix_of(&s(3, 2))); // "0" ⊂ "010"
        assert!(s(2, 1).is_prefix_of(&s(3, 2))); // "01" ⊂ "010"
        assert!(s(2, 1).is_prefix_of(&s(3, 3))); // "01" ⊂ "011"
        assert!(!s(1, 1).is_prefix_of(&s(2, 0))); // "1" ⊄ "00"
        assert!(ShardId::ROOT.is_prefix_of(&s(16, u16::MAX)));

        let p = s(6, 37);
        assert_eq!(p.child(false), Ok(s(7, 74)));
        assert_eq!(p.child(true), Ok(s(7, 75)));
        assert_eq!(s(7, 75).parent(), Some(p));
        assert_eq!(s(7, 74).sibling(), Some(s(7, 75)));
        assert_eq!(ShardId::ROOT.parent(), None);
        assert_eq!(ShardId::ROOT.sibling(), None);
        assert!(ShardId::ROOT.is_root());
    }

    #[test]
    fn genesis_set() {
        let g = ShardSet::genesis();
        assert_eq!(g.len(), 64);
        assert!(g.contains(&s(6, 0)));
        assert!(g.contains(&s(6, 63)));
        assert!(!g.contains(&s(5, 0)));
        let rebuilt =
            ShardSet::from_vec((0..64u16).map(|v| ShardId { bits: 6, value: v }).collect());
        assert_eq!(g, rebuilt.unwrap());
    }

    #[test]
    fn invalid_sets_rejected() {
        let reason = |ids: Vec<ShardId>| {
            ShardSet::from_vec(ids).err().map(|e| match e {
                TypeError::InvalidShardSet { reason } => reason,
                _ => "other",
            })
        };
        assert_eq!(reason(vec![]), Some("empty set"));
        assert_eq!(
            reason(vec![s(1, 0), s(2, 0), s(2, 1)]),
            Some("one shard id is a prefix of another")
        );
        assert_eq!(reason(vec![s(1, 0), s(1, 0), s(1, 1)]), Some("duplicate shard id"));
        // covers only the "0" half: trailing gap
        assert_eq!(
            reason(vec![s(2, 0), s(2, 1)]),
            Some("prefix trie does not cover the whole key space")
        );
        // "0", "11": internal gap ("10" missing)
        assert_eq!(
            reason(vec![s(1, 0), s(2, 3)]),
            Some("prefix trie does not cover the whole key space")
        );
        // valid uneven depths: {"0", "10", "11"}
        assert!(ShardSet::from_vec(vec![s(1, 0), s(2, 2), s(2, 3)]).is_ok());
        assert!(ShardSet::from_vec(vec![ShardId::ROOT]).is_ok());
        // count cap: 1023 ten-bit leaves + the two children of the missing one
        let mut many: Vec<ShardId> = (0..1024u16)
            .filter(|&v| v != 5)
            .map(|v| ShardId { bits: 10, value: v })
            .collect();
        many.push(ShardId { bits: 11, value: 10 });
        many.push(ShardId { bits: 11, value: 11 });
        assert_eq!(
            reason(many),
            Some("active shard count exceeds the protocol maximum")
        );
        let full: Vec<ShardId> = (0..1024u16).map(|v| ShardId { bits: 10, value: v }).collect();
        assert!(ShardSet::from_vec(full).is_ok());
    }

    #[test]
    fn genesis_home_matches_prefix_rule() {
        let g = ShardSet::genesis();
        let mut rng = SplitMix64::new(42);
        for _ in 0..500 {
            let k = kappa_of(&rng.bytes(32));
            let key = kappa_key(&k);
            let expected = ShardId::new(6, (key >> 10) as u16).unwrap();
            assert_eq!(g.home_kappa(&k), Ok(expected));
            assert!(expected.is_kappa_home(&k));
            let other = ShardId::new(6, (key >> 10) as u16 ^ 1).unwrap();
            assert!(!other.is_kappa_home(&k));
        }
    }

    #[test]
    fn split_then_merge_roundtrip() {
        let g = ShardSet::genesis();
        let target = s(6, 0);
        let split = g.split(&target).unwrap();
        assert_eq!(split.len(), 65);
        assert!(!split.contains(&target));
        let c0 = target.child(false).unwrap();
        let c1 = target.child(true).unwrap();
        assert!(split.contains(&c0) && split.contains(&c1));
        assert_eq!(split.merge(&c0).unwrap(), g);
        assert_eq!(split.merge(&c1).unwrap(), g);
        assert!(matches!(
            g.split(&c0),
            Err(TypeError::InvalidShardSet { reason: "shard id is not active" })
        ));
        assert!(matches!(
            g.merge(&c0),
            Err(TypeError::InvalidShardSet { reason: "shard id is not active" })
        ));
    }

    #[test]
    fn split_preserves_homing() {
        let g = ShardSet::genesis();
        let target = s(6, 9);
        let split = g.split(&target).unwrap();
        let mut rng = SplitMix64::new(7);
        for _ in 0..1000 {
            let k = kappa_of(&rng.bytes(32));
            let h0 = g.home_kappa(&k).unwrap();
            let h1 = split.home_kappa(&k).unwrap();
            if h0 == target {
                assert!(h1 == c0_of(target) || h1 == c1_of(target));
            } else {
                assert_eq!(h0, h1);
            }
        }
    }

    fn c0_of(t: ShardId) -> ShardId {
        t.child(false).unwrap()
    }
    fn c1_of(t: ShardId) -> ShardId {
        t.child(true).unwrap()
    }

    #[test]
    fn deep_split_hits_depth_cap() {
        let mut set = ShardSet::from_vec(vec![s(1, 0), s(1, 1)]).unwrap();
        let mut cur = s(1, 1);
        while cur.bits() < 16 {
            set = set.split(&cur).unwrap();
            cur = cur.child(false).unwrap();
        }
        assert_eq!(cur.bits(), 16);
        assert!(matches!(set.split(&cur), Err(TypeError::ShardIdTooLong { .. })));
    }

    #[test]
    fn merge_error_paths() {
        let root_only = ShardSet::from_vec(vec![ShardId::ROOT]).unwrap();
        assert!(matches!(
            root_only.merge(&ShardId::ROOT),
            Err(TypeError::InvalidShardSet { reason: "root shard has no parent or sibling" })
        ));
        let odd = ShardSet::from_vec(vec![s(1, 0), s(2, 2), s(2, 3)]).unwrap();
        assert!(matches!(
            odd.merge(&s(1, 0)),
            Err(TypeError::InvalidShardSet { reason: "merge requires both siblings active" })
        ));
        let two = ShardSet::from_vec(vec![s(1, 0), s(1, 1)]).unwrap();
        assert_eq!(two.merge(&s(1, 0)).unwrap(), root_only);
    }

    #[test]
    fn leg_key_orders_by_txid_then_leg() {
        let t_a = TxId::hash_canonical(b"legs-a");
        let t_b = TxId::hash_canonical(b"legs-b");
        let (first, second) = if t_a < t_b { (t_a, t_b) } else { (t_b, t_a) };
        let mut keys = vec![
            LegKey::new(second, LegIndex::from_u8(0)),
            LegKey::new(first, LegIndex::from_u8(9)),
            LegKey::new(first, LegIndex::from_u8(3)),
            LegKey::new(second, LegIndex::from_u8(255)),
        ];
        keys.sort();
        assert_eq!(
            keys,
            vec![
                LegKey::new(first, LegIndex::from_u8(3)),
                LegKey::new(first, LegIndex::from_u8(9)),
                LegKey::new(second, LegIndex::from_u8(0)),
                LegKey::new(second, LegIndex::from_u8(255)),
            ]
        );
    }

    #[test]
    fn interval_epoch_conversions() {
        assert_eq!(Interval::from_secs(0).as_u64(), 0);
        assert_eq!(Interval::from_secs(1599).as_u64(), 1599);
        assert_eq!(Interval::from_u64(86_400).epoch(), Epoch::from_u64(1));
        assert_eq!(Epoch::from_u64(1).first_interval().unwrap().as_u64(), 86_400);
        assert_eq!(Epoch::from_secs(86_399), Epoch::from_u64(0));
        assert_eq!(Epoch::from_secs(86_400), Epoch::from_u64(1));
        assert_eq!(Epoch::from_u64(2).start_secs(), 172_800);
        let span = Epoch::from_u64(3).interval_span().unwrap();
        assert_eq!((span.start, span.end), (3 * 86_400, 4 * 86_400));
        assert!(Epoch::from_u64(u64::MAX).first_interval().is_none());
        for i in [0u64, 1, 86_399, 86_400, 1_000_000] {
            let e = Interval::from_u64(i).epoch();
            let start = e.first_interval().unwrap().as_u64();
            assert!(start <= i && i < start + INTERVALS_PER_EPOCH);
        }
    }

    #[test]
    fn txid_domain_and_codec() {
        let payload = b"canonical serialization of all legs";
        assert_eq!(
            TxId::hash_canonical(payload),
            TxId::from_hash(Hash256::concat(&constants::TXID, payload))
        );
        let t = TxId::hash_canonical(payload);
        codec_roundtrip(t);
        assert_eq!(LegIndex::MAX.next(), None);
        assert_eq!(LegIndex::from_u8(0).next(), Some(LegIndex::from_u8(1)));
        assert_eq!(FeeSats::from_u64(3).checked_add(FeeSats::from_u64(4)), Some(FeeSats::from_u64(7)));
        assert_eq!(
            FeeSats::from_u64(u64::MAX).checked_add(FeeSats::from_u64(1)),
            None
        );
    }

    #[test]
    fn kappa_formula_and_home_addr() {
        let addr = b"recipient-canonical-encoding";
        assert_eq!(kappa(addr), Hash256::concat(&constants::SHARD_HOMING, addr));
        let g = ShardSet::genesis();
        assert_eq!(g.home(addr).unwrap(), g.home_kappa(&kappa(addr)).unwrap());
    }

    #[test]
    fn codec_roundtrips_and_strictness() {
        codec_roundtrip(s(6, 9));
        codec_roundtrip(s(0, 0));
        codec_roundtrip(s(16, u16::MAX));
        codec_roundtrip(LegIndex::from_u8(200));
        codec_roundtrip(LegKey::new(TxId::hash_canonical(b"x"), LegIndex::from_u8(7)));
        let set = ShardSet::genesis().split(&s(6, 9)).unwrap();
        codec_roundtrip(set.clone());
        assert_eq!(set.encoded_len(), 4 + 3 * set.len());

        // wire bytes of a gap set {("10"), ("11")} must not decode
        let mut bad = Vec::new();
        bad.extend_from_slice(&2u32.to_le_bytes());
        bad.push(2);
        bad.extend_from_slice(&2u16.to_le_bytes());
        bad.push(2);
        bad.extend_from_slice(&3u16.to_le_bytes());
        assert!(ShardSet::decode(&bad).is_err());
        assert!(ShardSet::decode(&0u32.to_le_bytes()).is_err());
    }
}
