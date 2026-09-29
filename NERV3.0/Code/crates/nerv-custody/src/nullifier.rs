//! Nullifier derivation and the spent set (WP §3.4).
//!
//! nf = BLAKE3("nerv.nf" ‖ nk ‖ ρ). The spent set is a sparse Merkle tree of
//! depth 256 over 256-bit keys: leaves bind (nf, insertion height), internal
//! nodes commit both children, and absent subtrees are the default chain
//! (E_256 = empty leaf; E_k = H(node ‖ E_{k+1} ‖ E_{k+1})). Storage holds
//! only live path nodes — the absent/default identification IS the caching
//! of §3.4. Proofs carry only stored siblings; omitted levels fold with E_j,
//! which is sound (a lying omission changes the recomputed root) and is the
//! compression that meets B9's ≤ 1 KB at 10^9 entries.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::OnceLock;

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::{NULLIFIER, NULLIFIER_EMPTY, NULLIFIER_LEAF, NULLIFIER_NODE};
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;

use crate::error::CustodyError;

pub const DEPTH: u16 = nerv_core::params::CUSTODY_NULLIFIER_TREE_DEPTH as u16;
const _: () = assert!(DEPTH == 256);

pub fn derive_nullifier(nk: &[u8; 32], rho: &[u8; 32]) -> Hash256 {
    let mut msg = [0u8; 64];
    msg[..32].copy_from_slice(nk);
    msg[32..].copy_from_slice(rho);
    Hash256::concat(&NULLIFIER, &msg)
}

fn node_hash(l: &Hash256, r: &Hash256) -> Hash256 {
    let mut msg = [0u8; 64];
    msg[..32].copy_from_slice(l.as_bytes());
    msg[32..].copy_from_slice(r.as_bytes());
    Hash256::concat(&NULLIFIER_NODE, &msg)
}

fn leaf_hash(nf: &Hash256, height: u64) -> Hash256 {
    let mut msg = [0u8; 40];
    msg[..32].copy_from_slice(nf.as_bytes());
    msg[32..].copy_from_slice(&height.to_le_bytes());
    Hash256::concat(&NULLIFIER_LEAF, &msg)
}

fn default_hashes() -> &'static [Hash256; 257] {
    static TABLE: OnceLock<[Hash256; 257]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = std::array::from_fn(|_| Hash256::from_bytes([0u8; 32]));
        t[256] = Hash256::concat(&NULLIFIER_EMPTY, &[0u8; 32]);
        for k in (0..256).rev() {
            t[k] = node_hash(&t[k + 1], &t[k + 1]);
        }
        t
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

/// A spent-set proof: non-default siblings only, ascending by child level.
#[derive(Clone, PartialEq, Eq)]
pub struct NullifierProof {
    pub nf: Hash256,
    pub spent_height: Option<u64>,
    pub siblings: Vec<(u16, Hash256)>,
}

impl fmt::Debug for NullifierProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NullifierProof")
            .field("nf", &self.nf)
            .field("spent_height", &self.spent_height)
            .field("sibling_entries", &self.siblings.len())
            .finish()
    }
}

impl NullifierProof {
    pub fn verify_membership(&self, root: &Hash256, nf: &Hash256, spent_height: u64) -> bool {
        self.nf == *nf
            && self.spent_height == Some(spent_height)
            && fold(leaf_hash(nf, spent_height), nf, &self.siblings)
                .is_some_and(|h| h == *root)
    }

    pub fn verify_non_membership(&self, root: &Hash256, nf: &Hash256) -> bool {
        self.nf == *nf
            && self.spent_height.is_none()
            && fold(default_hashes()[256], nf, &self.siblings)
                .is_some_and(|h| h == *root)
    }
}

impl Encode for NullifierProof {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self.nf.as_bytes());
        match self.spent_height {
            None => out.push(0),
            Some(h) => {
                out.push(1);
                out.extend_from_slice(&h.to_le_bytes());
            }
        }
        out.extend_from_slice(&(self.siblings.len() as u32).to_le_bytes());
        for (l, h) in &self.siblings {
            out.extend_from_slice(&l.to_le_bytes());
            out.extend_from_slice(h.as_bytes());
        }
    }
    fn encoded_len(&self) -> usize {
        32 + 1 + self.spent_height.map_or(0, |_| 8) + 4 + self.siblings.len() * 34
    }
}

impl Decode for NullifierProof {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let nf = Hash256::decode_from(r)?;
        let spent_height = match r.read_u8()? {
            0 => None,
            1 => Some(r.read_u64()?),
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
            siblings.push((l, Hash256::decode_from(r)?));
        }
        r.leave();
        Ok(NullifierProof { nf, spent_height, siblings })
    }
}

/// Fold from `start` at leaf level up to the root along nf's bits. `None` on
/// malformed sibling lists (duplicate or out-of-range levels).
fn fold(start: Hash256, nf: &Hash256, siblings: &[(u16, Hash256)]) -> Option<Hash256> {
    let defaults = default_hashes();
    let mut map: BTreeMap<u16, Hash256> = BTreeMap::new();
    for &(level, h) in siblings {
        if level == 0 || level > DEPTH || map.insert(level, h).is_some() {
            return None;
        }
    }
    let key = nf.as_bytes();
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

/// The per-shard spent set. `nodes` holds exactly the live path nodes (keyed
/// by (child level, masked prefix)); `spent` is the authoritative (nf →
/// insertion height) table. The root is a pure function of the spent set.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NullifierSet {
    nodes: BTreeMap<(u16, [u8; 32]), Hash256>,
    spent: BTreeMap<[u8; 32], u64>,
}

impl NullifierSet {
    pub fn new() -> NullifierSet {
        NullifierSet::default()
    }

    pub fn len(&self) -> usize {
        self.spent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spent.is_empty()
    }

    pub fn contains(&self, nf: &Hash256) -> bool {
        self.spent.contains_key(nf.as_bytes())
    }

    pub fn spent_height(&self, nf: &Hash256) -> Option<u64> {
        self.spent.get(nf.as_bytes()).copied()
    }

    pub fn root(&self) -> Hash256 {
        self.nodes
            .get(&(0, [0u8; 32]))
            .copied()
            .unwrap_or(default_hashes()[0])
    }

    pub fn insert(&mut self, nf: &Hash256, height: u64) -> Result<(), CustodyError> {
        if self.spent.contains_key(nf.as_bytes()) {
            return Err(CustodyError::NullifierAlreadySpent { nf: *nf });
        }
        let key = *nf.as_bytes();
        let mut node = leaf_hash(nf, height);
        for j in (1..=DEPTH).rev() {
            self.nodes.insert((j, mask(&key, j)), node);
            let sib_prefix = sibling_prefix(&key, j);
            let sib = self
                .nodes
                .get(&(j, sib_prefix))
                .copied()
                .unwrap_or(default_hashes()[j as usize]);
            node = if bit(&key, j - 1) {
                node_hash(&sib, &node)
            } else {
                node_hash(&node, &sib)
            };
        }
        self.nodes.insert((0, [0u8; 32]), node);
        self.spent.insert(key, height);
        Ok(())
    }

    /// Atomic batch insert at one block height: rejects already-spent keys
    /// and in-batch duplicates before any mutation (an invalid block changes
    /// nothing). O(k·256) hashes — the §3.4 batch bound.
    pub fn insert_batch(&mut self, nfs: &[Hash256], height: u64) -> Result<(), CustodyError> {
        let mut batch: BTreeMap<[u8; 32], ()> = BTreeMap::new();
        for nf in nfs {
            if self.spent.contains_key(nf.as_bytes()) {
                return Err(CustodyError::NullifierAlreadySpent { nf: *nf });
            }
            if batch.insert(*nf.as_bytes(), ()).is_some() {
                return Err(CustodyError::DuplicateNullifier { nf: *nf });
            }
        }
        for nf in nfs {
            self.insert(nf, height)?;
        }
        Ok(())
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

    pub fn membership_proof(&self, nf: &Hash256) -> Result<NullifierProof, CustodyError> {
        let height = self
            .spent
            .get(nf.as_bytes())
            .copied()
            .ok_or(CustodyError::NullifierNotSpent { nf: *nf })?;
        Ok(NullifierProof {
            nf: *nf,
            spent_height: Some(height),
            siblings: self.collect_siblings(nf.as_bytes()),
        })
    }

    /// A proof that `nf` is unspent. For a spent key this returns a proof
    /// that fails verification — the verifier is the arbiter.
    pub fn non_membership_proof(&self, nf: &Hash256) -> NullifierProof {
        NullifierProof {
            nf: *nf,
            spent_height: None,
            siblings: self.collect_siblings(nf.as_bytes()),
        }
    }

    /// Full audit: rebuild from the spent table and compare node-for-node.
    /// O(n·256); for fraud audits and tests.
    pub fn validate_consistency(&self) -> Result<(), CustodyError> {
        let mut fresh = NullifierSet::new();
        for (&nf_bytes, &height) in &self.spent {
            fresh
                .insert(&Hash256::from_bytes(nf_bytes), height)
                .map_err(|_| CustodyError::InvalidNullifierSet("insert failed during rebuild"))?;
        }
        if fresh.nodes != self.nodes {
            return Err(CustodyError::InvalidNullifierSet(
                "stored nodes diverge from the spent set",
            ));
        }
        Ok(())
    }
}

impl Encode for NullifierSet {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.nodes.len() as u32).to_le_bytes());
        for (&(level, prefix), h) in &self.nodes {
            out.extend_from_slice(&level.to_le_bytes());
            out.extend_from_slice(&prefix);
            out.extend_from_slice(h.as_bytes());
        }
        out.extend_from_slice(&(self.spent.len() as u32).to_le_bytes());
        for (nf, &height) in &self.spent {
            out.extend_from_slice(nf);
            out.extend_from_slice(&height.to_le_bytes());
        }
    }
    fn encoded_len(&self) -> usize {
        4 + self.nodes.len() * 66 + 4 + self.spent.len() * 40
    }
}

impl Decode for NullifierSet {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let n = r.read_seq_len()?;
        r.enter()?;
        let mut nodes = BTreeMap::new();
        for _ in 0..n {
            let level = r.read_u16()?;
            if level > DEPTH {
                return Err(CodecError::InvariantViolated("nullifier node level out of range"));
            }
            let prefix = r.take_array::<32>()?;
            if mask(&prefix, level) != prefix {
                return Err(CodecError::InvariantViolated("nullifier prefix not canonical"));
            }
            let h = Hash256::decode_from(r)?;
            if nodes.insert((level, prefix), h).is_some() {
                return Err(CodecError::InvariantViolated("duplicate nullifier node"));
            }
        }
        r.leave();
        let m = r.read_seq_len()?;
        r.enter()?;
        let mut spent = BTreeMap::new();
        for _ in 0..m {
            let nf = r.take_array::<32>()?;
            let height = r.read_u64()?;
            if spent.insert(nf, height).is_some() {
                return Err(CodecError::InvariantViolated("duplicate spent nullifier"));
            }
        }
        r.leave();
        Ok(NullifierSet { nodes, spent })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;
    use std::collections::BTreeMap;

    fn nfs(rng: &mut SplitMix64, n: usize) -> Vec<Hash256> {
        (0..n).map(|_| Hash256::from_bytes(rng.bytes32())).collect()
    }

    /// Independent recursive construction: root of the trie over the leaf
    /// set, built top-down by partitioning (vs. the iterative path insert).
    fn reference_root(spent: &BTreeMap<[u8; 32], u64>) -> Hash256 {
        fn rec(keys: &[[u8; 32]], spent: &BTreeMap<[u8; 32], u64>, level: u16) -> Hash256 {
            if keys.is_empty() {
                return default_hashes()[level as usize];
            }
            if level == DEPTH {
                assert_eq!(keys.len(), 1);
                return leaf_hash(&Hash256::from_bytes(keys[0]), spent[&keys[0]]);
            }
            let mut left = Vec::new();
            let mut right = Vec::new();
            for &k in keys {
                if bit(&k, level) {
                    right.push(k);
                } else {
                    left.push(k);
                }
            }
            node_hash(&rec(&left, spent, level + 1), &rec(&right, spent, level + 1))
        }
        let keys: Vec<[u8; 32]> = spent.keys().copied().collect();
        rec(&keys, spent, 0)
    }

    #[test]
    fn mask_bit_sibling_invariants() {
        let mut rng = SplitMix64::new(0x9F1);
        for _ in 0..8 {
            let k = rng.bytes32();
            for level in [0u16, 1, 7, 8, 9, 100, 255, 256] {
                let m = mask(&k, level);
                assert_eq!(mask(&m, level), m);
                for b in 0..level {
                    assert_eq!(bit(&m, b), bit(&k, b), "bit {b} must survive mask {level}");
                }
                for b in level..256 {
                    assert!(!bit(&m, b), "bit {b} must be cleared by mask {level}");
                }
            }
            for j in 1..=DEPTH {
                let sib = sibling_prefix(&k, j);
                assert_eq!(mask(&sib, j - 1), mask(&k, j - 1), "level {j}");
                assert_ne!(mask(&sib, j), mask(&k, j), "level {j}");
            }
        }
    }

    #[test]
    fn nullifier_derivation_is_the_wp_formula() {
        let nk = [11u8; 32];
        let rho = [22u8; 32];
        let nf = derive_nullifier(&nk, &rho);
        assert_eq!(nf, derive_nullifier(&nk, &rho));
        assert_ne!(nf, derive_nullifier(&[33u8; 32], &rho));
        assert_ne!(nf, derive_nullifier(&nk, &[33u8; 32]));
        let mut msg = [0u8; 64];
        msg[..32].copy_from_slice(&nk);
        msg[32..].copy_from_slice(&rho);
        assert_eq!(nf, Hash256::concat(&NULLIFIER, &msg));
    }

    #[test]
    fn empty_set_behavior() {
        let s = NullifierSet::new();
        let mut rng = SplitMix64::new(1);
        let nf = Hash256::from_bytes(rng.bytes32());
        assert!(s.is_empty());
        assert!(s.contains(&nf) == false);
        assert_eq!(s.root(), default_hashes()[0]);
        let p = s.non_membership_proof(&nf);
        assert!(p.siblings.is_empty());
        assert!(p.verify_non_membership(&s.root(), &nf));
        assert!(!p.verify_membership(&s.root(), &nf, 0));
        s.validate_consistency().unwrap();
    }

    #[test]
    fn insert_contains_double_spend() {
        let mut s = NullifierSet::new();
        let mut rng = SplitMix64::new(2);
        let nf = Hash256::from_bytes(rng.bytes32());
        s.insert(&nf, 100).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s.spent_height(&nf), Some(100));
        assert!(matches!(
            s.insert(&nf, 200),
            Err(CustodyError::NullifierAlreadySpent { .. })
        ));
        assert_eq!(s.spent_height(&nf), Some(100), "failed insert changes nothing");
        let nf2 = Hash256::from_bytes(rng.bytes32());
        let r1 = s.root();
        s.insert(&nf2, 101).unwrap();
        assert_eq!(s.spent_height(&nf2), Some(101));
        assert_ne!(s.root(), r1);
        s.validate_consistency().unwrap();
    }

    #[test]
    fn batch_insert_atomic() {
        let mut rng = SplitMix64::new(3);
        let mut s = NullifierSet::new();
        let seed = nfs(&mut rng, 3);
        s.insert_batch(&seed, 10).unwrap();
        let root_before = s.root();

        let fresh = nfs(&mut rng, 2);
        let dup = Hash256::from_bytes(rng.bytes32());
        assert!(matches!(
            s.insert_batch(&[fresh[0], dup, fresh[1], dup], 11),
            Err(CustodyError::DuplicateNullifier { .. })
        ));
        assert_eq!(s.root(), root_before, "failed batch changes nothing");
        assert_eq!(s.len(), 3);

        let fresh2 = nfs(&mut rng, 1);
        assert!(matches!(
            s.insert_batch(&[seed[0], fresh2[0]], 12),
            Err(CustodyError::NullifierAlreadySpent { .. })
        ));
        assert_eq!(s.root(), root_before);

        s.insert_batch(&fresh, 13).unwrap();
        assert_eq!(s.len(), 5);
        assert_eq!(s.spent_height(&fresh[1]), Some(13));
        s.insert_batch(&[], 14).unwrap();
        assert_eq!(s.len(), 5);
        s.validate_consistency().unwrap();
    }

    #[test]
    fn root_matches_reference_construction() {
        for (seed, n) in [(4u64, 1usize), (5, 2), (6, 3), (7, 7), (8, 33), (9, 200)] {
            let mut rng = SplitMix64::new(seed);
            let keys = nfs(&mut rng, n);
            let mut s = NullifierSet::new();
            let mut spent = BTreeMap::new();
            for (i, nf) in keys.iter().enumerate() {
                s.insert(nf, i as u64 + 1).unwrap();
                spent.insert(*nf.as_bytes(), i as u64 + 1);
            }
            assert_eq!(s.root(), reference_root(&spent), "n = {n}");
            s.validate_consistency().unwrap();
        }
    }

    #[test]
    fn root_is_insertion_order_independent() {
        let mut rng = SplitMix64::new(10);
        let keys = nfs(&mut rng, 25);
        let pairs: Vec<(Hash256, u64)> =
            keys.iter().enumerate().map(|(i, k)| (*k, i as u64)).collect();
        let mut a = NullifierSet::new();
        for (nf, h) in &pairs {
            a.insert(nf, *h).unwrap();
        }
        let mut shuffled = pairs.clone();
        let mut r2 = SplitMix64::new(99);
        for i in (1..shuffled.len()).rev() {
            let j = (r2.next_u64() % (i as u64 + 1)) as usize;
            shuffled.swap(i, j);
        }
        let mut b = NullifierSet::new();
        for (nf, h) in &shuffled {
            b.insert(nf, *h).unwrap();
        }
        assert_eq!(a.root(), b.root(), "root is a function of the (nf, height) set");
    }

    #[test]
    fn membership_proofs_and_tampering() {
        let mut rng = SplitMix64::new(11);
        let keys = nfs(&mut rng, 40);
        let mut s = NullifierSet::new();
        for (i, nf) in keys.iter().enumerate() {
            s.insert(nf, 1000 + i as u64).unwrap();
        }
        let root = s.root();
        for (i, nf) in keys.iter().enumerate() {
            let h = 1000 + i as u64;
            let p = s.membership_proof(nf).unwrap();
            assert!(p.verify_membership(&root, nf, h));
            assert!(!p.verify_membership(&root, nf, h + 1), "wrong height");
            assert!(!p.verify_membership(&root, nf, h - 1));
        }
        let (nf0, h0) = (keys[0], 1000);
        let other = keys[1];
        let other_root = Hash256::from_bytes(rng.bytes32());
        let p = s.membership_proof(&nf0).unwrap();
        assert!(!p.verify_membership(&root, &other, h0), "wrong nf");
        assert!(!p.verify_membership(&other_root, &nf0, h0), "wrong root");

        let mut bad = p.clone();
        bad.siblings[0].1 = Hash256::from_bytes(rng.bytes32());
        assert!(!bad.verify_membership(&root, &nf0, h0), "tampered sibling");

        let mut missing = p.clone();
        missing.siblings.remove(0);
        assert!(!missing.verify_membership(&root, &nf0, h0), "omitted live sibling");

        let mut dup_level = p.clone();
        dup_level.siblings.push(dup_level.siblings[0]);
        assert!(!dup_level.verify_membership(&root, &nf0, h0), "duplicate level");

        let mut oor = p.clone();
        oor.siblings[0].0 = 0;
        assert!(!oor.verify_membership(&root, &nf0, h0), "level 0 invalid");

        assert!(matches!(
            s.membership_proof(&Hash256::from_bytes(rng.bytes32())),
            Err(CustodyError::NullifierNotSpent { .. })
        ));
    }

    #[test]
    fn non_membership_proofs() {
        let mut rng = SplitMix64::new(12);
        let keys = nfs(&mut rng, 30);
        let mut s = NullifierSet::new();
        for (i, nf) in keys.iter().enumerate() {
            s.insert(nf, 50 + i as u64).unwrap();
        }
        let root = s.root();
        for nf in nfs(&mut rng, 30) {
            let p = s.non_membership_proof(&nf);
            assert!(p.verify_non_membership(&root, &nf));
            assert!(!p.verify_membership(&root, &nf, 1), "unspent cannot claim membership");
        }
        // a spent key's non-membership proof fails verification
        let p = s.non_membership_proof(&keys[7]);
        assert!(!p.verify_non_membership(&root, &keys[7]));
    }

    #[test]
    fn proof_sizes_b9_shape() {
        let mut rng = SplitMix64::new(13);
        let keys = nfs(&mut rng, 4096);
        let mut s = NullifierSet::new();
        for k in &keys {
            s.insert(k, 1).unwrap();
        }
        let root = s.root();
        let mut entries = 0usize;
        let mut bytes = 0usize;
        let mut count = 0usize;
        for k in keys.iter().step_by(64) {
            let p = s.membership_proof(k).unwrap();
            assert!(p.verify_membership(&root, k, 1));
            entries += p.siblings.len();
            bytes += p.encode().len();
            count += 1;
        }
        // 4096 leaves: shared prefixes are ~log2(4096) = 12 deep; the raw
        // path is 8 KB — the compressed proof is a small fraction of it.
        assert!(entries / count <= 30, "avg sibling entries {entries}/{count}");
        assert!(bytes / count <= 2100, "avg proof {bytes}/{count} bytes");
        for k in nfs(&mut rng, 8) {
            let p = s.non_membership_proof(&k);
            assert!(p.siblings.len() < 32);
            assert!(p.verify_non_membership(&root, &k));
        }
    }

    #[test]
    fn serialization_roundtrip_and_audit() {
        let mut rng = SplitMix64::new(14);
        let keys = nfs(&mut rng, 30);
        let mut s = NullifierSet::new();
        for (i, nf) in keys.iter().enumerate() {
            s.insert(nf, i as u64 + 3).unwrap();
        }
        let enc = s.encode();
        assert_eq!(enc.len(), s.encoded_len());
        let d = NullifierSet::decode(&enc).unwrap();
        assert_eq!(d.root(), s.root());
        assert_eq!(d.len(), s.len());
        d.validate_consistency().unwrap();
        for (i, nf) in keys.iter().enumerate() {
            let p = d.membership_proof(nf).unwrap();
            assert!(p.verify_membership(&d.root(), nf, i as u64 + 3));
        }
        let p = s.membership_proof(&keys[5]).unwrap();
        let penc = p.encode();
        assert_eq!(penc.len(), p.encoded_len());
        assert_eq!(NullifierProof::decode(&penc).unwrap(), p);
        assert!(NullifierProof::decode(&penc[..penc.len() - 1]).is_err());

        // tampered node hash decodes but fails the audit
        let mut bad = enc.clone();
        bad[4 + 2 + 32] ^= 1; // first entry's hash byte
        let t = NullifierSet::decode(&bad).unwrap();
        assert!(t.validate_consistency().is_err());

        assert!(NullifierSet::decode(&enc[..enc.len() - 1]).is_err());

        let mut level_oor = enc.clone();
        level_oor[4..6].copy_from_slice(&257u16.to_le_bytes());
        assert!(NullifierSet::decode(&level_oor).is_err());

        let mut crafted = Vec::new();
        crafted.extend_from_slice(&1u32.to_le_bytes());
        crafted.extend_from_slice(&8u16.to_le_bytes());
        crafted.extend_from_slice(&[0xFFu8; 32]); // bits ≥ 8 set: non-canonical
        crafted.extend_from_slice(&[0u8; 32]);
        crafted.extend_from_slice(&0u32.to_le_bytes());
        assert!(NullifierSet::decode(&crafted).is_err());
    }
}
