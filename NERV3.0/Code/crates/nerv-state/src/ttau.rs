//! The T_τ inclusion tree (WP §4.3 rule 1, §5.5 tier 2; erratum 101): a
//! fixed-depth-32 frontier-fold BLAKE3 Merkle tree over one finalized
//! registry interval's deduplicated txid set, in canonical byte order.
//! The registry (chunk 14) constructs it from the dedup'd interval set;
//! the shard executor verifies legs' membership witnesses against the
//! beacon-finalized root. No G_τ verification here — that is once per
//! interval, on the beacon.

use std::sync::OnceLock;

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::{TAU_EMPTY, TAU_LEAF, TAU_NODE};
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::types::{Interval, TxId};

use crate::error::StateError;

pub const MAX_DEPTH: usize = 32;
pub const LEAF_CAPACITY: u64 = 1 << MAX_DEPTH;

pub fn leaf_digest(txid: &TxId) -> Hash256 {
    Hash256::concat(&TAU_LEAF, txid.as_bytes())
}

pub fn node_digest(left: &Hash256, right: &Hash256) -> Hash256 {
    let mut msg = [0u8; 64];
    msg[..32].copy_from_slice(left.as_bytes());
    msg[32..].copy_from_slice(right.as_bytes());
    Hash256::concat(&TAU_NODE, &msg)
}

fn empty_digests() -> &'static [Hash256; MAX_DEPTH + 1] {
    static TABLE: OnceLock<[Hash256; MAX_DEPTH + 1]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = std::array::from_fn(|_| Hash256::from_bytes([0u8; 32]));
        t[0] = Hash256::concat(&TAU_EMPTY, &[0u8; 32]);
        for k in 0..MAX_DEPTH {
            t[k + 1] = node_digest(&t[k], &t[k]);
        }
        t
    })
}

/// The root of the empty set — the genesis registry reference's T_τ root
/// (erratum 102).
pub fn empty_root() -> Hash256 {
    empty_digests()[MAX_DEPTH]
}

/// The T_τ tree. `txids` is the sorted leaf set; `levels` mirrors the NCT's
/// append structure; `count == txids.len()` (broken only by the capacity
/// guard test's direct construction).
#[derive(Clone, Default)]
pub struct TauTree {
    txids: Vec<TxId>,
    levels: Vec<Vec<Hash256>>,
    count: u64,
}

impl TauTree {
    pub fn new() -> TauTree {
        TauTree::default()
    }

    pub fn from_sorted(txids: &[TxId]) -> Result<TauTree, StateError> {
        let mut t = TauTree::new();
        for txid in txids {
            t.insert_sorted(txid)?;
        }
        Ok(t)
    }

    pub fn insert_sorted(&mut self, txid: &TxId) -> Result<u64, StateError> {
        if self.count >= LEAF_CAPACITY {
            return Err(StateError::TauFull { capacity: LEAF_CAPACITY });
        }
        if let Some(last) = self.txids.last() {
            if txid <= last {
                return Err(StateError::TauUnsorted { txid: *txid, prev: *last });
            }
        }
        let index = self.count;
        self.txids.push(*txid);
        self.count += 1;
        let mut node = leaf_digest(txid);
        let mut k = 0usize;
        loop {
            if k == self.levels.len() {
                self.levels.push(Vec::new());
            }
            let lvl = &mut self.levels[k];
            lvl.push(node);
            if lvl.len() % 2 == 1 {
                break;
            }
            let n = lvl.len();
            let (left, right) = (lvl[n - 2], lvl[n - 1]);
            node = node_digest(&left, &right);
            k += 1;
        }
        Ok(index)
    }

    pub fn len(&self) -> u64 {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn txids(&self) -> &[TxId] {
        &self.txids
    }

    pub fn position(&self, txid: &TxId) -> Option<u64> {
        self.txids.binary_search(txid).ok().map(|i| i as u64)
    }

    /// The frontier fold (erratum 101): a set bit of `count` absorbs the
    /// level's pending chunk as the left child; a clear bit pads with E_k
    /// on the right.
    pub fn root(&self) -> Hash256 {
        let empty = empty_digests();
        if self.count == 0 {
            return empty[MAX_DEPTH];
        }
        let mut r = empty[0];
        for k in 0..MAX_DEPTH {
            if (self.count >> k) & 1 == 1 {
                let left = self
                    .levels
                    .get(k)
                    .and_then(|l| l.last().copied())
                    .unwrap_or(empty[k]);
                r = node_digest(&left, &r);
            } else {
                r = node_digest(&r, &empty[k]);
            }
        }
        r
    }

    /// The membership witness for leaf `index`. Variable-length: the full
    /// 32-sibling list with the maximal E_k-valued suffix trimmed.
    pub fn witness(&self, index: u64) -> Result<TauWitness, StateError> {
        if index >= self.count {
            return Err(StateError::TauIndex { index, count: self.count });
        }
        let empty = empty_digests();
        let mut siblings = vec![Hash256::default(); MAX_DEPTH];
        let mut j = index;
        for k in 0..MAX_DEPTH {
            let sib = j ^ 1;
            siblings[k] = self
                .levels
                .get(k)
                .filter(|l| (sib as usize) < l.len())
                .and_then(|l| l.get(sib as usize))
                .copied()
                .unwrap_or(empty[k]);
            j >>= 1;
        }
        for k in (0..MAX_DEPTH).rev() {
            if siblings[k] == empty[k] {
                siblings.pop();
            } else {
                break;
            }
        }
        Ok(TauWitness { index, siblings })
    }
}

impl std::fmt::Debug for TauTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TauTree").field("count", &self.count).finish()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TauWitness {
    pub index: u64,
    pub siblings: Vec<Hash256>,
}

impl Encode for TauWitness {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.index.to_le_bytes());
        out.extend_from_slice(&(self.siblings.len() as u32).to_le_bytes());
        for s in &self.siblings {
            out.extend_from_slice(s.as_bytes());
        }
    }
    fn encoded_len(&self) -> usize {
        8 + 4 + 32 * self.siblings.len()
    }
}

impl Decode for TauWitness {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let index = r.read_u64()?;
        if index >= LEAF_CAPACITY {
            return Err(CodecError::InvariantViolated("ttau witness index exceeds capacity"));
        }
        let n = r.read_seq_len()?;
        if n > MAX_DEPTH {
            return Err(CodecError::SeqTooLarge { count: n, max: MAX_DEPTH });
        }
        let mut siblings = Vec::with_capacity(n);
        for _ in 0..n {
            siblings.push(Hash256::decode_from(r)?);
        }
        Ok(TauWitness { index, siblings })
    }
}

/// The leg-carried form: the T_τ witness plus the interval whose root it
/// verifies against (the beacon-finalized root for that interval).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistryWitness {
    pub interval: Interval,
    pub index: u64,
    pub siblings: Vec<Hash256>,
}

impl RegistryWitness {
    pub fn new(interval: Interval, witness: TauWitness) -> RegistryWitness {
        RegistryWitness {
            interval,
            index: witness.index,
            siblings: witness.siblings,
        }
    }

    pub fn verify(&self, root: &Hash256, txid: &TxId) -> bool {
        verify_tau_witness(root, self.index, txid, &self.siblings)
    }
}

impl Encode for RegistryWitness {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.interval.encode_into(out);
        out.extend_from_slice(&self.index.to_le_bytes());
        out.extend_from_slice(&(self.siblings.len() as u32).to_le_bytes());
        for s in &self.siblings {
            out.extend_from_slice(s.as_bytes());
        }
    }
    fn encoded_len(&self) -> usize {
        8 + 8 + 4 + 32 * self.siblings.len()
    }
}

impl Decode for RegistryWitness {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let interval = Interval::decode_from(r)?;
        let index = r.read_u64()?;
        if index >= LEAF_CAPACITY {
            return Err(CodecError::InvariantViolated("ttau witness index exceeds capacity"));
        }
        let n = r.read_seq_len()?;
        if n > MAX_DEPTH {
            return Err(CodecError::SeqTooLarge { count: n, max: MAX_DEPTH });
        }
        let mut siblings = Vec::with_capacity(n);
        for _ in 0..n {
            siblings.push(Hash256::decode_from(r)?);
        }
        Ok(RegistryWitness { interval, index, siblings })
    }
}

/// Verify a membership witness against a T_τ root. Missing trailing
/// siblings are the empty digests E_k — the lossless form of the trimmed
/// witness (erratum 101). Adversarial input is rejected, never panics.
pub fn verify_tau_witness(root: &Hash256, index: u64, txid: &TxId, siblings: &[Hash256]) -> bool {
    if index >= LEAF_CAPACITY {
        return false;
    }
    if siblings.len() > MAX_DEPTH {
        return false;
    }
    let empty = empty_digests();
    let mut cur = leaf_digest(txid);
    let mut j = index;
    for k in 0..MAX_DEPTH {
        let sib = if k < siblings.len() { siblings[k] } else { empty[k] };
        cur = if j & 1 == 0 {
            node_digest(&cur, &sib)
        } else {
            node_digest(&sib, &cur)
        };
        j >>= 1;
    }
    cur == *root
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;
    use proptest::prelude::*;

    fn distinct_txids(seed: u64, n: usize) -> Vec<TxId> {
        let mut rng = SplitMix64::new(seed);
        let mut v: Vec<TxId> =
            (0..n).map(|_| TxId::from_hash(Hash256::from_bytes(rng.bytes32()))).collect();
        v.sort();
        v.dedup();
        assert_eq!(v.len(), n, "256-bit id collision in the test generator");
        v
    }

    /// Independent recursive construction of the left-packed depth-32 tree —
    /// the differential reference for the frontier fold.
    fn reference_root(leaves: &[Hash256], empty: &[Hash256; MAX_DEPTH + 1]) -> Hash256 {
        fn rec(k: usize, i: u64, leaves: &[Hash256], empty: &[Hash256; MAX_DEPTH + 1]) -> Hash256 {
            if (i << k) >= leaves.len() as u64 {
                return empty[k];
            }
            if k == 0 {
                return leaves[i as usize];
            }
            let l = rec(k - 1, 2 * i, leaves, empty);
            let r = rec(k - 1, 2 * i + 1, leaves, empty);
            node_digest(&l, &r)
        }
        rec(MAX_DEPTH, 0, leaves, empty)
    }

    #[test]
    fn empty_chain_and_literal_pins() {
        let e = empty_digests();
        assert_eq!(e.len(), MAX_DEPTH + 1);
        let mut pre = Vec::new();
        pre.extend_from_slice(TAU_EMPTY.as_bytes());
        pre.extend_from_slice(&[0u8; 32]);
        assert_eq!(e[0].as_bytes(), blake3::hash(&pre).as_bytes());
        for k in 1..=MAX_DEPTH {
            assert_eq!(e[k], node_digest(&e[k - 1], &e[k - 1]));
        }
        assert_eq!(empty_root(), e[MAX_DEPTH]);

        let t = TauTree::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert_eq!(t.root(), empty_root());

        let id = TxId::from_hash(Hash256::from_bytes([3u8; 32]));
        let mut m = Vec::new();
        m.extend_from_slice(TAU_LEAF.as_bytes());
        m.extend_from_slice(id.as_bytes());
        assert_eq!(leaf_digest(&id).as_bytes(), blake3::hash(&m).as_bytes());

        let (l, r) = (Hash256::from_bytes([1u8; 32]), Hash256::from_bytes([2u8; 32]));
        let mut n = Vec::new();
        n.extend_from_slice(TAU_NODE.as_bytes());
        n.extend_from_slice(l.as_bytes());
        n.extend_from_slice(r.as_bytes());
        assert_eq!(node_digest(&l, &r).as_bytes(), blake3::hash(&n).as_bytes());
    }

    #[test]
    fn sorted_enforcement() {
        let two = distinct_txids(5, 2);
        let (a, b) = (two[0], two[1]);
        assert!(a < b);
        let mut t = TauTree::new();
        assert_eq!(t.insert_sorted(&a).unwrap(), 0);
        assert_eq!(t.insert_sorted(&b).unwrap(), 1);
        assert!(matches!(t.insert_sorted(&a), Err(StateError::TauUnsorted { .. })));
        assert!(matches!(t.insert_sorted(&b), Err(StateError::TauUnsorted { .. })));
        assert_eq!(t.len(), 2);

        let mut rev = TauTree::new();
        rev.insert_sorted(&b).unwrap();
        assert!(matches!(rev.insert_sorted(&a), Err(StateError::TauUnsorted { .. })));
        assert!(TauTree::from_sorted(&[b, a]).is_err());
        assert!(TauTree::from_sorted(&[a, a]).is_err());
        assert!(TauTree::from_sorted(&[a, b]).is_ok());
        assert!(TauTree::from_sorted(&[]).unwrap().is_empty());
    }

    #[test]
    fn sequential_roots_match_reference() {
        let ids = distinct_txids(0x7AA0, 64);
        let empty = *empty_digests();
        let mut t = TauTree::new();
        for (n, id) in ids.iter().enumerate() {
            t.insert_sorted(id).unwrap();
            let leaves: Vec<Hash256> = ids[..=n].iter().map(leaf_digest).collect();
            assert_eq!(t.root(), reference_root(&leaves, &empty), "count {}", n + 1);
        }
        for n in [100usize, 249] {
            let ids = distinct_txids(0x7AA0 + n as u64, n);
            let t = TauTree::from_sorted(&ids).unwrap();
            let leaves: Vec<Hash256> = ids.iter().map(leaf_digest).collect();
            assert_eq!(t.root(), reference_root(&leaves, &empty), "count {n}");
        }
    }

    #[test]
    fn witnesses_verify_against_tree_and_reference() {
        let empty = *empty_digests();
        for n in [1usize, 2, 3, 5, 17, 64, 100, 249] {
            let ids = distinct_txids(0x7AB0 + n as u64, n);
            let t = TauTree::from_sorted(&ids).unwrap();
            let leaves: Vec<Hash256> = ids.iter().map(leaf_digest).collect();
            let reference = reference_root(&leaves, &empty);
            let root = t.root();
            assert_eq!(root, reference, "n = {n}");
            for (i, id) in ids.iter().enumerate() {
                let i = i as u64;
                let w = t.witness(i).unwrap();
                assert!(verify_tau_witness(&root, i, id, &w.siblings), "n={n} i={i}");
                assert!(verify_tau_witness(&reference, i, id, &w.siblings), "ref n={n} i={i}");
                let other = &ids[((i as usize) + 1) % n];
                if other != id {
                    assert!(!verify_tau_witness(&root, i, other, &w.siblings));
                }
                if n > 1 {
                    let j = (i + 1) % n as u64;
                    assert!(!verify_tau_witness(&root, j, id, &w.siblings), "n={n} j={j}");
                }
                assert!(!verify_tau_witness(&Hash256::from_bytes([9u8; 32]), i, id, &w.siblings));

                let mut full = w.siblings.clone();
                while full.len() < MAX_DEPTH {
                    let k = full.len();
                    full.push(empty[k]);
                }
                assert_eq!(full.len(), MAX_DEPTH);
                assert!(verify_tau_witness(&root, i, id, &full));

                if !w.siblings.is_empty() {
                    let mut short = w.siblings.clone();
                    short.pop();
                    assert!(!verify_tau_witness(&root, i, id, &short), "pop n={n} i={i}");
                    let mut bad = w.siblings.clone();
                    let mid = bad.len() / 2;
                    let mut b = *bad[mid].as_bytes();
                    b[0] ^= 1;
                    bad[mid] = Hash256::from_bytes(b);
                    assert!(!verify_tau_witness(&root, i, id, &bad));
                }
                if let Some(&last) = w.siblings.last() {
                    assert_ne!(last, empty[w.siblings.len() - 1], "trim n={n} i={i}");
                }
            }
            assert!(matches!(t.witness(n as u64), Err(StateError::TauIndex { .. })));
        }
    }

    #[test]
    fn verify_rejects_bad_shapes() {
        let ids = distinct_txids(1, 4);
        let t = TauTree::from_sorted(&ids).unwrap();
        let root = t.root();
        let w = t.witness(2).unwrap();
        assert!(!verify_tau_witness(&root, LEAF_CAPACITY, &ids[0], &w.siblings));
        assert!(!verify_tau_witness(&root, u64::MAX, &ids[0], &w.siblings));
        let mut long = w.siblings.clone();
        while long.len() < MAX_DEPTH {
            long.push(Hash256::from_bytes([1u8; 32]));
        }
        long.push(Hash256::from_bytes([2u8; 32]));
        assert!(!verify_tau_witness(&root, 2, &ids[2], &long));
    }

    #[test]
    fn position_lookup() {
        let ids = distinct_txids(6, 40);
        let t = TauTree::from_sorted(&ids).unwrap();
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(t.position(id), Some(i as u64));
        }
        let mut rng = SplitMix64::new(0x70);
        for _ in 0..64 {
            let probe = TxId::from_hash(Hash256::from_bytes(rng.bytes32()));
            if !ids.contains(&probe) {
                assert_eq!(t.position(&probe), None);
            }
        }
        assert_eq!(t.txids(), &ids[..]);
    }

    #[test]
    fn capacity_guard() {
        let ids = distinct_txids(7, 2);
        let mut full = TauTree { txids: Vec::new(), levels: Vec::new(), count: LEAF_CAPACITY };
        assert!(matches!(
            full.insert_sorted(&ids[0]),
            Err(StateError::TauFull { capacity: LEAF_CAPACITY })
        ));
    }

    #[test]
    fn determinism_and_incremental_equivalence() {
        let ids = distinct_txids(8, 33);
        let t1 = TauTree::from_sorted(&ids).unwrap();
        let t2 = TauTree::from_sorted(&ids).unwrap();
        assert_eq!(t1.root(), t2.root());
        for i in 0..33u64 {
            assert_eq!(t1.witness(i).unwrap(), t2.witness(i).unwrap());
        }
        let mut t3 = TauTree::new();
        for id in &ids {
            t3.insert_sorted(id).unwrap();
        }
        assert_eq!(t3.root(), t1.root());
        for i in 0..33u64 {
            assert_eq!(t3.witness(i).unwrap(), t1.witness(i).unwrap());
        }
    }

    #[test]
    fn codec_roundtrips_and_validation() {
        let ids = distinct_txids(3, 9);
        let t = TauTree::from_sorted(&ids).unwrap();
        let w = t.witness(4).unwrap();
        let enc = w.encode();
        assert_eq!(enc.len(), w.encoded_len());
        let dec = TauWitness::decode(&enc).unwrap();
        assert_eq!(dec, w);
        assert!(verify_tau_witness(&t.root(), 4, &ids[4], &dec.siblings));
        for cut in 0..enc.len() {
            assert!(TauWitness::decode(&enc[..cut]).is_err(), "cut {cut}");
        }
        let mut ext = enc.clone();
        ext.push(0);
        assert!(TauWitness::decode(&ext).is_err());

        let mut bad = enc.clone();
        bad[0..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            TauWitness::decode(&bad),
            Err(CodecError::InvariantViolated(_))
        ));
        let mut bad = enc.clone();
        bad[8..12].copy_from_slice(&33u32.to_le_bytes());
        assert!(matches!(
            TauWitness::decode(&bad),
            Err(CodecError::SeqTooLarge { max: 32, .. })
        ));

        let rw = RegistryWitness::new(Interval::from_u64(7), w.clone());
        let enc2 = rw.encode();
        assert_eq!(enc2.len(), rw.encoded_len());
        assert_eq!(RegistryWitness::decode(&enc2).unwrap(), rw);
        assert!(rw.verify(&t.root(), &ids[4]));
        assert!(!rw.verify(&t.root(), &ids[5]));
        assert!(RegistryWitness::decode(&enc2[..enc2.len() - 1]).is_err());
        let mut bad_rw = enc2.clone();
        bad_rw[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            RegistryWitness::decode(&bad_rw),
            Err(CodecError::InvariantViolated(_))
        ));
    }

    proptest! {
        #[test]
        fn prop_roots_and_witnesses(n in 1usize..40, seed in any::<u64>()) {
            let mut rng = SplitMix64::new(seed ^ 0x7AA);
            let mut ids: Vec<TxId> =
                (0..n).map(|_| TxId::from_hash(Hash256::from_bytes(rng.bytes32()))).collect();
            ids.sort();
            ids.dedup();
            let n = ids.len();
            let t = TauTree::from_sorted(&ids).unwrap();
            let empty = *empty_digests();
            let leaves: Vec<Hash256> = ids.iter().map(leaf_digest).collect();
            let root = t.root();
            prop_assert_eq!(root, reference_root(&leaves, &empty));
            for i in 0..n as u64 {
                let w = t.witness(i).unwrap();
                prop_assert!(verify_tau_witness(&root, i, &ids[i as usize], &w.siblings));
                let mut full = w.siblings.clone();
                while full.len() < MAX_DEPTH {
                    let k = full.len();
                    full.push(empty[k]);
                }
                prop_assert!(verify_tau_witness(&root, i, &ids[i as usize], &full));
            }
        }
    }
}
