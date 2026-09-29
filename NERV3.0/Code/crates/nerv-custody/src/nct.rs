//! The note commitment tree (WP §3.3): append-only, binary, depth 32,
//! 2^32-leaf capacity per shard. Leaves are BLAKE3 note commitments
//! compressed by the frozen Poseidon2 leaf function; internal nodes are
//! Poseidon2 node compressions (E-002). All real nodes are retained (the
//! full-node form — DSR-10; nerv-state persists this representation).
//! Appends amortize to ≤ 2 node hashes per leaf. Root = frontier fold over
//! the empty-subtree table. Membership witnesses: 32 sibling digests;
//! verification is the native twin of the in-circuit Merkle chip (DSR-7).

use std::fmt;
use std::sync::OnceLock;
use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::error::CodecError;
use nerv_core::field::{Goldilocks, GOLDILOCKS_PRIME};
use nerv_core::hash::Hash256;

use crate::error::CustodyError;
use crate::poseidon2;

pub const DEPTH: usize = nerv_core::params::CUSTODY_NCT_DEPTH as usize;
pub const LEAF_CAPACITY: u64 = 1u64 << DEPTH;

const _: () = assert!(DEPTH == 32);

/// A 32-byte tree digest: four Goldilocks elements, LE-encoded. Invariant:
/// every word is below the Goldilocks prime (enforced by all constructors).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct NctDigest([u8; 32]);

impl fmt::Display for NctDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl NctDigest {
    pub fn from_bytes(bytes: [u8; 32]) -> Result<NctDigest, CustodyError> {
        let d = NctDigest(bytes);
        d.to_elements()?;
        Ok(d)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn from_elements(e: &[Goldilocks; 4]) -> NctDigest {
        let mut b = [0u8; 32];
        for (i, w) in e.iter().enumerate() {
            b[i * 8..i * 8 + 8].copy_from_slice(&w.as_u64().to_le_bytes());
        }
        NctDigest(b)
    }

    pub fn to_elements(&self) -> Result<[Goldilocks; 4], CustodyError> {
        let mut out = [Goldilocks::ZERO; 4];
        for (i, e) in out.iter_mut().enumerate() {
            let mut w = [0u8; 8];
            w.copy_from_slice(&self.0[i * 8..i * 8 + 8]);
            let v = u64::from_le_bytes(w);
            *e = Goldilocks::from_valid_u64(v)
                .ok_or(CustodyError::DigestWordOutOfRange { value: v })?;
        }
        Ok(out)
    }

    pub fn as_hash256(&self) -> Hash256 {
        Hash256::from_bytes(self.0)
    }

    pub fn try_from_hash256(h: &Hash256) -> Result<NctDigest, CustodyError> {
        NctDigest::from_bytes(*h.as_bytes())
    }
}

impl Encode for NctDigest {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0);
    }
    fn encoded_len(&self) -> usize {
        32
    }
}

impl Decode for NctDigest {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let arr = r.take_array::<32>()?;
        NctDigest::from_bytes(arr)
            .map_err(|_| CodecError::InvariantViolated("digest word not below the Goldilocks prime"))
    }
}

/// Invariant-checked element decode for internal folds (constructors
/// validate; unreachable arm maps to ZERO under debug_assert).
fn elements(bytes: &[u8; 32]) -> [Goldilocks; 4] {
    let mut out = [Goldilocks::ZERO; 4];
    for (i, e) in out.iter_mut().enumerate() {
        let mut w = [0u8; 8];
        w.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
        let v = u64::from_le_bytes(w);
        debug_assert!(v < GOLDILOCKS_PRIME, "NctDigest invariant violated");
        *e = Goldilocks::from_valid_u64(v).unwrap_or(Goldilocks::ZERO);
    }
    out
}

/// Level-0 digest of a BLAKE3 note commitment: the 32 cm bytes split into
/// eight 32-bit field elements (injective — no mod-p reduction, which would
/// be lossy at ~2^-32 per word and birthday-visible at ~2^32 leaves).
pub fn leaf_digest(cm: &Hash256) -> NctDigest {
    let b = cm.as_bytes();
    let mut input = [Goldilocks::ZERO; 8];
    for w in 0..8 {
        input[w] = Goldilocks::from_u32(u32::from_le_bytes([
            b[w * 4],
            b[w * 4 + 1],
            b[w * 4 + 2],
            b[w * 4 + 3],
        ]));
    }
    NctDigest::from_elements(&poseidon2::compress(poseidon2::leaf_iv(), &input))
}

/// Internal node: H(left, right) over two 4-element digests.
pub fn node_digest(left: &NctDigest, right: &NctDigest) -> NctDigest {
    let mut input = [Goldilocks::ZERO; 8];
    input[0..4].copy_from_slice(&elements(left.as_bytes()));
    input[4..8].copy_from_slice(&elements(right.as_bytes()));
    NctDigest::from_elements(&poseidon2::compress(poseidon2::node_iv(), &input))
}

static EMPTY_DIGESTS: OnceLock<Vec<NctDigest>> = OnceLock::new();

/// E_0..=E_32: empty-subtree digests. E_0 = compress(EMPTY_IV, 0^8);
/// E_{k+1} = H(E_k, E_k).
pub fn empty_digests() -> &'static [NctDigest] {
    EMPTY_DIGESTS.get_or_init(|| {
        let mut v = Vec::with_capacity(DEPTH + 1);
        let zero = [Goldilocks::ZERO; 8];
        v.push(NctDigest::from_elements(&poseidon2::compress(
            poseidon2::empty_iv(),
            &zero,
        )));
        for _ in 1..=DEPTH {
            let prev = v[v.len() - 1];
            v.push(node_digest(&prev, &prev));
        }
        v
    })
}

/// Membership witness: leaf index + one sibling digest per level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NctWitness {
    pub leaf_index: u64,
    pub siblings: [NctDigest; DEPTH],
}

impl Encode for NctWitness {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.leaf_index.to_le_bytes());
        for s in &self.siblings {
            s.encode_into(out);
        }
    }
    fn encoded_len(&self) -> usize {
        8 + DEPTH * 32
    }
}

impl Decode for NctWitness {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let leaf_index = r.read_u64()?;
        // siblings wire-format: DEPTH consecutive NctDigest records (32 B each).
        // We decode them individually rather than via take_array<DEPTH>() so
        // the canonical Goldilocks-word sanity check fires for each digest
        // (would otherwise be silently skipped on raw-byte roundtrip).
        let mut siblings = [NctDigest([0u8; 32]); DEPTH];
        for slot in &mut siblings {
            *slot = NctDigest::decode_from(r)?;
        }
        Ok(NctWitness { leaf_index, siblings })
    }
}

/// Verify a membership witness against a root: the native twin of the
/// in-circuit Merkle chip. Rejects indices ≥ 2^32 outright.
pub fn verify_witness(
    root: &NctDigest,
    leaf_index: u64,
    leaf_cm: &Hash256,
    siblings: &[NctDigest; DEPTH],
) -> bool {
    if leaf_index >= LEAF_CAPACITY {
        return false;
    }
    let mut node = leaf_digest(leaf_cm);
    let mut j = leaf_index;
    for k in 0..DEPTH {
        node = if j & 1 == 0 {
            node_digest(&node, &siblings[k])
        } else {
            node_digest(&siblings[k], &node)
        };
        j >>= 1;
    }
    node == *root
}

/// Append-only note commitment tree.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct NoteCommitmentTree {
    levels: Vec<Vec<NctDigest>>,
    count: u64,
}

impl NoteCommitmentTree {
    pub fn new() -> NoteCommitmentTree {
        NoteCommitmentTree::default()
    }

    pub fn leaf_count(&self) -> u64 {
        self.count
    }

    pub fn is_full(&self) -> bool {
        self.count >= LEAF_CAPACITY
    }

    /// Append one commitment; returns its leaf index.
    pub fn append(&mut self, cm: &Hash256) -> Result<u64, CustodyError> {
        if self.count >= LEAF_CAPACITY {
            return Err(CustodyError::TreeFull { capacity: LEAF_CAPACITY });
        }
        let mut node = leaf_digest(cm);
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
        self.count += 1;
        Ok(self.count - 1)
    }

    /// Batch append in canonical order; returns the first leaf index.
    /// Total cost is ≤ 2k − 1 node hashes for k leaves (O(k), better than
    /// the O(k log k) bound stated in WP §3.3).
    pub fn append_batch(&mut self, cms: &[Hash256]) -> Result<u64, CustodyError> {
        if self.count + cms.len() as u64 > LEAF_CAPACITY {
            return Err(CustodyError::TreeFull { capacity: LEAF_CAPACITY });
        }
        let first = self.count;
        for cm in cms {
            self.append(cm)?;
        }
        Ok(first)
    }

    /// Tree root. Empty tree ⇒ E_32; otherwise the frontier fold
    /// r_{k+1} = H(frontier_k or E_k, r_k), r_0 = E_0, where frontier slot
    /// k is occupied iff bit k of `count` is set.
    pub fn root(&self) -> NctDigest {
        if self.count == 0 {
            return empty_digests()[DEPTH];
        }
        let empty = empty_digests();
        let mut r = empty[0];
        for k in 0..DEPTH {
            let left = if (self.count >> k) & 1 == 1 {
                self.levels[k].last().copied().unwrap_or(empty[k])
            } else {
                empty[k]
            };
            r = node_digest(&left, &r);
        }
        r
    }

    /// Membership witness for leaf `index`.
    pub fn witness(&self, index: u64) -> Result<NctWitness, CustodyError> {
        if index >= self.count {
            return Err(CustodyError::LeafIndexOutOfRange { index, count: self.count });
        }
        let empty = empty_digests();
        let mut siblings = [empty[0]; DEPTH];
        let mut j = index;
        for k in 0..DEPTH {
            let sib = j ^ 1;
            siblings[k] = match self
                .levels
                .get(k)
                .filter(|l| (sib as usize) < l.len())
                .and_then(|l| l.get(sib as usize))
            {
                Some(d) => *d,
                None => empty[k],
            };
            j >>= 1;
        }
        Ok(NctWitness { leaf_index: index, siblings })
    }

    fn check_structure(&self) -> Result<(), CustodyError> {
        if self.count > LEAF_CAPACITY {
            return Err(CustodyError::InvalidTree("count exceeds capacity"));
        }
        if self.count == 0 {
            if !self.levels.is_empty() {
                return Err(CustodyError::InvalidTree("empty tree with levels"));
            }
            return Ok(());
        }
        if self.levels.is_empty() {
            return Err(CustodyError::InvalidTree("non-empty count without levels"));
        }
        if self.levels[0].len() as u64 != self.count {
            return Err(CustodyError::InvalidTree("level-0 length != count"));
        }
        for k in 1..self.levels.len() {
            if self.levels[k].len() != self.levels[k - 1].len() / 2 {
                return Err(CustodyError::InvalidTree("level lengths violate the halving law"));
            }
        }
        if self.levels[self.levels.len() - 1].len() != 1 {
            return Err(CustodyError::InvalidTree("top level must hold exactly one node"));
        }
        Ok(())
    }

    /// Full audit: structure + every stored parent equals H(children).
    pub fn validate_consistency(&self) -> Result<(), CustodyError> {
        self.check_structure()?;
        for k in 1..self.levels.len() {
            for i in 0..self.levels[k].len() {
                let want = node_digest(&self.levels[k - 1][2 * i], &self.levels[k - 1][2 * i + 1]);
                if self.levels[k][i] != want {
                    return Err(CustodyError::InvalidTree(
                        "parent node does not match its children",
                    ));
                }
            }
        }
        Ok(())
    }
}

impl Encode for NoteCommitmentTree {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.count.to_le_bytes());
        out.extend_from_slice(&(self.levels.len() as u32).to_le_bytes());
        for lvl in &self.levels {
            out.extend_from_slice(&(lvl.len() as u32).to_le_bytes());
            for d in lvl {
                d.encode_into(out);
            }
        }
    }
    fn encoded_len(&self) -> usize {
        8 + 4 + self.levels.iter().map(|l| 4 + l.len() * 32).sum::<usize>()
    }
}

impl Decode for NoteCommitmentTree {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let count = r.read_u64()?;
        if count > LEAF_CAPACITY {
            return Err(CodecError::InvariantViolated("nct: count exceeds capacity"));
        }
        let nlevels = r.read_seq_len()?;
        r.enter()?;
        let mut levels = Vec::with_capacity(nlevels);
        for _ in 0..nlevels {
            let n = r.read_seq_len()?;
            let mut lvl = Vec::with_capacity(n);
            for _ in 0..n {
                lvl.push(NctDigest::decode_from(r)?);
            }
            levels.push(lvl);
        }
        r.leave();
        let tree = NoteCommitmentTree { levels, count };
        tree.check_structure()
            .map_err(|_| CodecError::InvariantViolated("nct: invalid level structure"))?;
        Ok(tree)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;

    fn cm(rng: &mut SplitMix64) -> Hash256 {
        Hash256::from_bytes(rng.bytes32())
    }

    fn reference_root(leaves: &[NctDigest], empty: &[NctDigest]) -> NctDigest {
        // Independent recursive construction of the depth-32 left-packed
        // tree with empty padding — the differential reference for root().
        fn rec(k: usize, i: u64, leaves: &[NctDigest], empty: &[NctDigest]) -> NctDigest {
            if (i << k) >= leaves.len() as u64 {
                return empty[k];
            }
            if k == 0 {
                return leaves[i as usize];
            }
            let l = rec(k - 1, i * 2, leaves, empty);
            let r = rec(k - 1, i * 2 + 1, leaves, empty);
            node_digest(&l, &r)
        }
        rec(DEPTH, 0, leaves, empty)
    }

    #[test]
    fn empty_table_and_empty_root() {
        let empty = empty_digests();
        assert_eq!(empty.len(), DEPTH + 1);
        for k in 1..=DEPTH {
            assert_eq!(empty[k], node_digest(&empty[k - 1], &empty[k - 1]));
        }
        let t = NoteCommitmentTree::new();
        assert_eq!(t.leaf_count(), 0);
        assert_eq!(t.root(), empty[DEPTH]);
        assert!(t.witness(0).is_err());
        t.validate_consistency().expect("empty tree consistent");
    }

    #[test]
    fn sequential_roots_match_reference() {
        let mut rng = SplitMix64::new(0x5000);
        let empty = empty_digests().to_vec();
        let mut tree = NoteCommitmentTree::new();
        let mut cms: Vec<Hash256> = Vec::new();
        for n in 1..=48u64 {
            let c = cm(&mut rng);
            cms.push(c);
            let idx = tree.append(&c).expect("append");
            assert_eq!(idx, n - 1);
            assert_eq!(tree.leaf_count(), n);
            let digests: Vec<NctDigest> = cms.iter().map(leaf_digest).collect();
            assert_eq!(tree.root(), reference_root(&digests, &empty), "count {n}");
            tree.validate_consistency().expect("consistency");
        }
    }

    #[test]
    fn witnesses_verify_across_sizes() {
        let mut rng = SplitMix64::new(0x5100);
        let mut tree = NoteCommitmentTree::new();
        let mut cms: Vec<Hash256> = Vec::new();
        for _ in 0..37 {
            let c = cm(&mut rng);
            cms.push(c);
            tree.append(&c).expect("append");
        }
        let root = tree.root();
        for (i, c) in cms.iter().enumerate() {
            let w = tree.witness(i as u64).expect("witness");
            assert!(verify_witness(&root, i as u64, c, &w.siblings), "leaf {i}");
        }
        let w5 = tree.witness(5).expect("witness");
        assert!(!verify_witness(&root, 5, &cms[6], &w5.siblings));
        assert!(!verify_witness(&root, 6, &cms[5], &w5.siblings));
        let mut tampered = w5;
        let mut b = *tampered.siblings[3].as_bytes();
        b[0] ^= 0xFF;
        tampered.siblings[3] = NctDigest::from_bytes(b).expect("valid digest");
        assert!(!verify_witness(&root, 5, &cms[5], &tampered.siblings));
        assert!(!verify_witness(&root, LEAF_CAPACITY + 5, &cms[5], &w5.siblings));
        assert!(tree.witness(37).is_err());
        assert!(tree.witness(u64::MAX).is_err());
    }

    #[test]
    fn witness_codec_roundtrip() {
        let mut rng = SplitMix64::new(0x5150);
        let mut tree = NoteCommitmentTree::new();
        let mut cms: Vec<Hash256> = Vec::new();
        for _ in 0..5 {
            let c = cm(&mut rng);
            cms.push(c);
            tree.append(&c).expect("append");
        }
        let w = tree.witness(2).expect("witness");
        let enc = w.encode();
        assert_eq!(enc.len(), 8 + DEPTH * 32);
        assert_eq!(enc.len(), w.encoded_len());
        let dec = NctWitness::decode(&enc).expect("decode");
        assert_eq!(dec, w);
        assert!(verify_witness(&tree.root(), 2, &cms[2], &dec.siblings));
        assert!(NctWitness::decode(&enc[..enc.len() - 1]).is_err());
    }

    #[test]
    fn batch_equals_sequential() {
        let mut rng = SplitMix64::new(0x5200);
        let cms: Vec<Hash256> = (0..23).map(|_| cm(&mut rng)).collect();
        let mut a = NoteCommitmentTree::new();
        let mut b = NoteCommitmentTree::new();
        for c in &cms {
            a.append(c).expect("append");
        }
        b.append_batch(&cms).expect("batch");
        assert_eq!(a, b);
        assert_eq!(a.root(), b.root());
        for i in 0..cms.len() {
            assert_eq!(a.witness(i as u64), b.witness(i as u64));
        }
        let mut empty_batch = NoteCommitmentTree::new();
        empty_batch.append_batch(&[]).expect("empty batch");
        assert_eq!(empty_batch.leaf_count(), 0);
    }

    #[test]
    fn capacity_guards() {
        let full = NoteCommitmentTree { levels: vec![], count: LEAF_CAPACITY };
        assert!(matches!(
            full.append(&Hash256::from_bytes([0u8; 32])),
            Err(CustodyError::TreeFull { .. })
        ));
        let near = NoteCommitmentTree { levels: vec![], count: LEAF_CAPACITY - 1 };
        assert!(matches!(
            near.append_batch(&[Hash256::from_bytes([1u8; 32]), Hash256::from_bytes([2u8; 32])]),
            Err(CustodyError::TreeFull { .. })
        ));
    }

    #[test]
    fn codec_roundtrip_and_structural_rejection() {
        let mut rng = SplitMix64::new(0x5300);
        let mut tree = NoteCommitmentTree::new();
        for _ in 0..9 {
            tree.append(&cm(&mut rng)).expect("append");
        }
        let enc = tree.encode();
        assert_eq!(enc.len(), tree.encoded_len());
        let decoded = NoteCommitmentTree::decode(&enc).expect("decode");
        assert_eq!(decoded, tree);
        decoded.validate_consistency().expect("decoded consistency");
        assert_eq!(decoded.root(), tree.root());
        assert!(NoteCommitmentTree::decode(&enc[..enc.len() - 1]).is_err());
        let mut ext = enc.clone();
        ext.push(0);
        assert!(NoteCommitmentTree::decode(&ext).is_err());

        let d = leaf_digest(&Hash256::from_bytes([3u8; 32]));
        let mut bad = Vec::new();
        bad.extend_from_slice(&3u64.to_le_bytes());
        bad.extend_from_slice(&1u32.to_le_bytes());
        bad.extend_from_slice(&2u32.to_le_bytes());
        d.encode_into(&mut bad);
        assert!(NoteCommitmentTree::decode(&bad).is_err());

        let mut bad2 = Vec::new();
        bad2.extend_from_slice(&0u64.to_le_bytes());
        bad2.extend_from_slice(&1u32.to_le_bytes());
        bad2.extend_from_slice(&1u32.to_le_bytes());
        d.encode_into(&mut bad2);
        assert!(NoteCommitmentTree::decode(&bad2).is_err());

        let mut good = Vec::new();
        good.extend_from_slice(&1u64.to_le_bytes());
        good.extend_from_slice(&1u32.to_le_bytes());
        good.extend_from_slice(&1u32.to_le_bytes());
        d.encode_into(&mut good);
        let t = NoteCommitmentTree::decode(&good).expect("minimal tree");
        assert_eq!(t.leaf_count(), 1);
    }

    #[test]
    fn digest_type_validation_and_conversions() {
        let mut rng = SplitMix64::new(0x5400);
        let d = leaf_digest(&cm(&mut rng));
        let enc = d.encode();
        assert_eq!(enc.len(), 32);
        assert_eq!(NctDigest::decode(&enc).expect("decode"), d);
        let mut bad = *d.as_bytes();
        bad[28] = 0xFF;
        bad[29] = 0xFF;
        bad[30] = 0xFF;
        bad[31] = 0xFF;
        assert!(NctDigest::from_bytes(bad).is_err());
        assert!(NctDigest::decode(&bad).is_err());
        let h = d.as_hash256();
        assert_eq!(NctDigest::try_from_hash256(&h).expect("valid"), d);
        assert!(NctDigest::try_from_hash256(&Hash256::from_bytes(bad)).is_err());
        let e = d.to_elements().expect("elements");
        assert_eq!(NctDigest::from_elements(&e), d);
    }
}

