//! BLAKE3 row-Merkle commitments over LDE codewords (WP §5.3, §5.7): the
//! engine's polynomial-commitment layer. A committed matrix is
//! `height × width` Goldilocks (height = domain size, a power of two);
//! the leaf is the domain-tagged hash of one ROW — all committed columns
//! at one domain position — so a query opening costs one authentication
//! path regardless of width. Extension-valued codewords are flattened by
//! the caller into pairs of base columns; this layer commits base rows
//! only.
//!
//! One tree per commitment; every root is absorbed into the Fiat–Shamir
//! transcript separately. Heights are asserted powers of two, never
//! padded — the engine commits only two-adic-sized matrices. Rayon is
//! confined to the leaf layer; every digest is a pure function of its
//! input, so commitments are byte-identical across machines and runs.

use rayon::prelude::*;
use crate::stark::domain::TWO_ADICITY;
use nerv_core::constants::{STARK_LEAF, STARK_NODE};
use nerv_core::field::Goldilocks;
use nerv_core::hash::Hash256;

/// A row-major `height × width` matrix of Goldilocks — the committed form
/// of a set of same-length codewords (one column each).
pub struct CodewordMatrix {
    values: Vec<Goldilocks>,
    width: usize,
}

impl CodewordMatrix {
    /// Assemble from column codewords (all of equal, power-of-two length).
    pub fn from_columns(columns: &[Vec<Goldilocks>]) -> CodewordMatrix {
        assert!(!columns.is_empty(), "at least one column");
        let height = columns[0].len();
        assert!(height > 0, "codewords are non-empty");
        assert!(height.is_power_of_two(), "codeword length {height} is not a power of two");
        assert!(columns.iter().all(|c| c.len() == height), "ragged columns");
        let width = columns.len();
        let mut values = vec![Goldilocks::ZERO; height * width];
        values
            .par_chunks_exact_mut(width)
            .enumerate()
            .for_each(|(r, row)| {
                for (c, col) in columns.iter().enumerate() {
                    row[c] = col[r];
                }
            });
        CodewordMatrix { values, width }
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.values.len() / self.width
    }

    /// The full row at domain position `i` (all committed columns).
    pub fn row(&self, i: usize) -> &[Goldilocks] {
        let w = self.width;
        assert!(i < self.height(), "row {i} out of range");
        &self.values[i * w..(i + 1) * w]
    }
}

impl std::fmt::Debug for CodewordMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CodewordMatrix({}x{})", self.height(), self.width())
    }
}

/// leaf = BLAKE3("nerv.stark.leaf" ‖ row), row = u64-LE words.
pub fn leaf_hash(row: &[Goldilocks]) -> Hash256 {
    let mut buf = Vec::with_capacity(8 * row.len());
    leaf_hash_into(row, &mut buf)
}

fn leaf_hash_into(row: &[Goldilocks], buf: &mut Vec<u8>) -> Hash256 {
    buf.clear();
    for v in row {
        buf.extend_from_slice(&v.as_u64().to_le_bytes());
    }
    Hash256::concat(&STARK_LEAF, buf)
}

/// node = BLAKE3("nerv.stark.node" ‖ left ‖ right) over the fixed 64-byte
/// input.
pub fn node_hash(left: &Hash256, right: &Hash256) -> Hash256 {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(left.as_bytes());
    buf[32..].copy_from_slice(right.as_bytes());
    Hash256::concat(&STARK_NODE, &buf)
}

/// A committed tree: digest layers (layer 0 = leaves, last = [root]) plus
/// the root. Prover data for authentication paths; the verifier needs only
/// the root and openings.
pub struct MerkleTree {
    digest_layers: Vec<Vec<Hash256>>,
    root: Hash256,
}

impl MerkleTree {
    /// Commit to `matrix`. Leaf layer hashing is parallel; every digest is
    /// a pure function of its input (DSR-11).
    pub fn commit(matrix: &CodewordMatrix) -> MerkleTree {
        let width = matrix.width();
        let height = matrix.height();
        assert!(width > 0, "committed matrices have at least one column");
        assert!(
            height > 0 && height.is_power_of_two(),
            "committed height is a positive power of two"
        );

        let leaves: Vec<Hash256> = matrix
            .values
            .par_chunks_exact(width)
            .map_init(|| Vec::with_capacity(8 * width), |buf, row| leaf_hash_into(row, buf))
            .collect();

        let mut layers: Vec<Vec<Hash256>> = Vec::new();
        let mut current = leaves;
        while current.len() > 1 {
            let half = current.len() / 2;
            let mut next = Vec::with_capacity(half);
            for i in 0..half {
                next.push(node_hash(&current[2 * i], &current[2 * i + 1]));
            }
            layers.push(current);
            current = next;
        }
        let root = current[0];
        layers.push(current);
        MerkleTree { digest_layers: layers, root }
    }

    pub const fn root(&self) -> Hash256 {
        self.root
    }

    /// log2 of the committed height (0 for a single-row tree).
    pub fn log_height(&self) -> usize {
        self.digest_layers.len() - 1
    }

    /// Sibling digests, leaf level to root level, for leaf `index`.
    pub fn auth_path(&self, index: usize) -> Vec<Hash256> {
        let log = self.log_height();
        assert!(index < (1usize << log), "index {index} out of range for height 2^{log}");
        let mut path = Vec::with_capacity(log);
        let mut idx = index;
        for layer in &self.digest_layers[..log] {
            path.push(layer[idx ^ 1]);
            idx >>= 1;
        }
        path
    }

    /// The SHARED authentication path for the top-bit-flip pair
    /// (index, index + height/2): the full path of `index` minus its last
    /// entry. Valid because the two indices agree on every bit below the
    /// top, so their climbs share every sibling below the top level.
    /// `index` must lie in the left half. FRI's prover-side pairing.
    pub fn auth_path_pair(&self, index: usize) -> Vec<Hash256> {
        let log = self.log_height();
        assert!(log >= 1, "a single-row tree has no pairs");
        assert!(
            index < (1usize << (log - 1)),
            "pair index {index} is not in the left half of height 2^{log}"
        );
        self.auth_path(index)[..log - 1].to_vec()
    }
}

/// One revealed row with its authentication path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowOpening {
    pub row: Vec<Goldilocks>,
    pub path: Vec<Hash256>,
}

impl RowOpening {
    /// Check this opening against a commitment root at `index`.
    pub fn verify(&self, root: &Hash256, index: usize) -> bool {
        verify_opening(root, index, &self.row, &self.path)
    }
}

/// Verify a row opening against a root: hash the row, fold with the
/// sibling digests (current node is the left child at a level iff the
/// corresponding index bit is 0), compare with the root. Rejects
/// out-of-range indices and paths longer than the engine's two-adic bound.
pub fn verify_opening(
    root: &Hash256,
    index: usize,
    row: &[Goldilocks],
    path: &[Hash256],
) -> bool {
    if path.len() > TWO_ADICITY {
        return false;
    }
    if index >> path.len() != 0 {
        return false;
    }
    let mut h = leaf_hash(row);
    for (level, sib) in path.iter().enumerate() {
        h = if (index >> level) & 1 == 1 {
            node_hash(sib, &h)
        } else {
            node_hash(&h, sib)
        };
    }
    h == *root
}

/// Climb from a leaf hash toward the root using sibling digests; the index
/// bit at each level selects the child side.
fn climb(mut h: Hash256, index: usize, path: &[Hash256]) -> Hash256 {
    for (lvl, sib) in path.iter().enumerate() {
        h = if (index >> lvl) & 1 == 1 {
            node_hash(sib, &h)
        } else {
            node_hash(&h, sib)
        };
    }
    h
}

/// Verify a top-bit-flip PAIR opening: rows at leaf indices `index` and
/// `index + 2^path.len()` differ only in the top bit, so both climbs share
/// every sibling below the top level. The shared path has length
/// `log_height − 1`; the root must be H(node_low, node_high) — the low
/// index sits in the left half by construction. FRI's per-layer query
/// authentication.
pub fn verify_pair_opening(
    root: &Hash256,
    index: usize,
    row_low: &[Goldilocks],
    row_high: &[Goldilocks],
    path: &[Hash256],
) -> bool {
    if path.len() + 1 > TWO_ADICITY {
        return false;
    }
    if index >> path.len() != 0 {
        return false;
    }
    let a = climb(leaf_hash(row_low), index, path);
    let b = climb(leaf_hash(row_high), index, path);
    *root == node_hash(&a, &b)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;
    use nerv_core::constants;

    fn gf(rng: &mut SplitMix64) -> Goldilocks {
        Goldilocks::from_u64_reduce(rng.next_u64())
    }

    struct Fixture {
        matrix: CodewordMatrix,
        columns: Vec<Vec<Goldilocks>>,
    }

    fn fixture(seed: u64, log_height: usize, width: usize) -> Fixture {
        let mut rng = SplitMix64::new(seed);
        let height = 1usize << log_height;
        let columns: Vec<Vec<Goldilocks>> = (0..width)
            .map(|_| (0..height).map(|_| gf(&mut rng)).collect())
            .collect();
        Fixture { matrix: CodewordMatrix::from_columns(&columns), columns }
    }

    #[test]
    fn leaf_and_node_hashes_are_the_literal_formulas() {
        let row = [Goldilocks::from_u32(1), Goldilocks::from_u32(2)];
        let mut pre = Vec::new();
        pre.extend_from_slice(constants::STARK_LEAF.as_bytes());
        for v in row {
            pre.extend_from_slice(&v.as_u64().to_le_bytes());
        }
        assert_eq!(leaf_hash(&row).as_bytes(), blake3::hash(&pre).as_bytes());

        let (l, r) = (Hash256::from_bytes([1u8; 32]), Hash256::from_bytes([2u8; 32]));
        let mut pre = Vec::new();
        pre.extend_from_slice(constants::STARK_NODE.as_bytes());
        pre.extend_from_slice(l.as_bytes());
        pre.extend_from_slice(r.as_bytes());
        assert_eq!(node_hash(&l, &r).as_bytes(), blake3::hash(&pre).as_bytes());
    }

    #[test]
    fn from_columns_lays_out_row_major() {
        let f = fixture(0x11, 3, 3);
        assert_eq!(f.matrix.width(), 3);
        assert_eq!(f.matrix.height(), 8);
        for r in 0..8 {
            for c in 0..3 {
                assert_eq!(f.matrix.row(r)[c], f.columns[c][r], "r={r} c={c}");
            }
        }
        assert_eq!(format!("{:?}", f.matrix), "CodewordMatrix(8x3)");
    }

    #[test]
    fn openings_verify_for_every_row() {
        for (log, width) in [(0usize, 1usize), (1, 1), (2, 3), (4, 1), (7, 2), (8, 5)] {
            let f = fixture(0xA0 + log as u64, log, width);
            let tree = MerkleTree::commit(&f.matrix);
            assert_eq!(tree.log_height(), log);
            for i in 0..(1usize << log) {
                let path = tree.auth_path(i);
                assert_eq!(path.len(), log);
                assert!(verify_opening(&tree.root(), i, f.matrix.row(i), &path), "log={log} i={i}");
            }
        }
    }

    #[test]
    fn single_row_tree_root_is_the_leaf_hash() {
        let f = fixture(0x5, 0, 2);
        let tree = MerkleTree::commit(&f.matrix);
        assert_eq!(tree.root(), leaf_hash(f.matrix.row(0)));
        assert_eq!(tree.log_height(), 0);
        assert!(tree.auth_path(0).is_empty());
        assert!(verify_opening(&tree.root(), 0, f.matrix.row(0), &[]));
        assert!(!verify_opening(&tree.root(), 1, f.matrix.row(0), &[]));
    }

    #[test]
    fn commitment_is_deterministic() {
        let f = fixture(0x0FF, 6, 4);
        assert_eq!(MerkleTree::commit(&f.matrix).root(), MerkleTree::commit(&f.matrix).root());
    }

    #[test]
    fn root_binds_content_order_and_width() {
        let a = fixture(1, 4, 2);
        let b = fixture(2, 4, 2);
        assert_ne!(MerkleTree::commit(&a.matrix).root(), MerkleTree::commit(&b.matrix).root());

        let mut cols = a.columns.clone();
        assert_ne!(cols[0][0], cols[0][1]);
        cols[0].swap(0, 1);
        let swapped = CodewordMatrix::from_columns(&cols);
        assert_ne!(MerkleTree::commit(&swapped).root(), MerkleTree::commit(&a.matrix).root());

        let one = CodewordMatrix::from_columns(&[vec![Goldilocks::from_u32(7)]]);
        let two = CodewordMatrix::from_columns(&[
            vec![Goldilocks::from_u32(7)],
            vec![Goldilocks::ZERO],
        ]);
        assert_ne!(MerkleTree::commit(&one).root(), MerkleTree::commit(&two).root());
    }

    #[test]
    fn auth_path_first_level_is_the_sibling_leaf_hash() {
        let f = fixture(0x6, 3, 2);
        let tree = MerkleTree::commit(&f.matrix);
        for i in 0..8 {
            let path = tree.auth_path(i);
            assert_eq!(path[0], leaf_hash(f.matrix.row(i ^ 1)), "i={i}");
        }
    }

    #[test]
    fn tampered_openings_are_rejected() {
        let f = fixture(0x7, 5, 3);
        let tree = MerkleTree::commit(&f.matrix);
        let root = tree.root();
        let i = 5usize;
        let path = tree.auth_path(i);
        let row = f.matrix.row(i).to_vec();

        let mut bad_row = row.clone();
        bad_row[1] = bad_row[1] + Goldilocks::ONE;
        assert!(!verify_opening(&root, i, &bad_row, &path));

        for lvl in 0..path.len() {
            let mut bad = path.clone();
            bad[lvl] = Hash256::from_bytes([9u8; 32]);
            assert!(!verify_opening(&root, i, &row, &bad), "level {lvl}");
        }

        let mut short = path.clone();
        short.pop();
        assert!(!verify_opening(&root, i, &row, &short));
        let mut long = path.clone();
        long.push(Hash256::from_bytes([9u8; 32]));
        assert!(!verify_opening(&root, i, &row, &long));

        assert!(!verify_opening(&root, i ^ 1, &row, &path));
        for j in (1usize << 5)..(1usize << 5) + 4 {
            assert!(!verify_opening(&root, j, &row, &path), "j={j}");
        }
        assert!(!verify_opening(&Hash256::from_bytes([0xAB; 32]), i, &row, &path));
    }

    #[test]
    fn verify_rejects_oversized_paths() {
        let f = fixture(0x3, 2, 1);
        let tree = MerkleTree::commit(&f.matrix);
        let mut huge = tree.auth_path(0);
        huge.resize(TWO_ADICITY + 1, Hash256::default());
        assert!(!verify_opening(&tree.root(), 0, f.matrix.row(0), &huge));
    }

    #[test]
    fn zero_matrix_commits_and_verifies() {
        let m = CodewordMatrix::from_columns(&[
            vec![Goldilocks::ZERO; 16],
            vec![Goldilocks::ZERO; 16],
        ]);
        let tree = MerkleTree::commit(&m);
        for i in 0..16 {
            assert!(verify_opening(&tree.root(), i, m.row(i), &tree.auth_path(i)));
        }
        assert!(!verify_opening(&tree.root(), 0, &[Goldilocks::ONE, Goldilocks::ZERO], &tree.auth_path(0)));
    }

    #[test]
    fn row_opening_api() {
        let f = fixture(0x9, 3, 2);
        let tree = MerkleTree::commit(&f.matrix);
        let opening = RowOpening { row: f.matrix.row(4).to_vec(), path: tree.auth_path(4) };
        assert!(opening.verify(&tree.root(), 4));
        assert!(!opening.verify(&tree.root(), 5));
    }

    #[test]
    fn large_tree_spot_check() {
        let f = fixture(0x1A2B, 12, 3);
        let tree = MerkleTree::commit(&f.matrix);
        let root = tree.root();
        let mut rng = SplitMix64::new(0xC0FFEE);
        for _ in 0..128 {
            let i = (rng.next_u64() as usize) & ((1usize << 12) - 1);
            let path = tree.auth_path(i);
            assert_eq!(path.len(), 12);
            assert!(verify_opening(&root, i, f.matrix.row(i), &path), "i={i}");
        }
    }

    #[test]
    #[should_panic(expected = "not a power of two")]
    fn from_columns_rejects_non_pow2() {
        let _ = CodewordMatrix::from_columns(&[vec![Goldilocks::ONE; 3]]);
    }

    #[test]
    #[should_panic(expected = "non-empty")]
    fn from_columns_rejects_empty_height() {
        let _ = CodewordMatrix::from_columns(&[Vec::new()]);
    }

    #[test]
    #[should_panic(expected = "at least one column")]
    fn from_columns_rejects_no_columns() {
        let _ = CodewordMatrix::from_columns(&[]);
    }

    #[test]
    #[should_panic(expected = "ragged")]
    fn from_columns_rejects_ragged() {
        let _ = CodewordMatrix::from_columns(&[vec![Goldilocks::ONE; 4], vec![Goldilocks::ONE; 8]]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn auth_path_rejects_out_of_range() {
        let f = fixture(0x2, 2, 1);
        let tree = MerkleTree::commit(&f.matrix);
        let _ = tree.auth_path(4);
    }
}
#[cfg(test)]
mod pair_tests {
    use super::*;
    use crate::testutil::SplitMix64;

    fn tree_with(log: usize, width: usize, seed: u64) -> (CodewordMatrix, MerkleTree) {
        let mut rng = SplitMix64::new(seed);
        let height = 1usize << log;
        let columns: Vec<Vec<Goldilocks>> = (0..width)
            .map(|_| (0..height).map(|_| Goldilocks::from_u64_reduce(rng.next_u64())).collect())
            .collect();
        let m = CodewordMatrix::from_columns(&columns);
        let t = MerkleTree::commit(&m);
        (m, t)
    }

    #[test]
    fn pair_paths_are_full_path_prefixes() {
        for log in 1..=6usize {
            let (m, tree) = tree_with(log, 2, 0x40 + log as u64);
            for j in 0..(1usize << (log - 1)) {
                let full = tree.auth_path(j);
                let pair = tree.auth_path_pair(j);
                assert_eq!(pair.len(), log - 1);
                assert_eq!(pair, full[..log - 1]);
                let _ = &m;
            }
        }
    }

    #[test]
    fn pair_openings_verify_across_sizes() {
        for (log, width) in [(1usize, 1usize), (2, 3), (5, 2), (8, 4)] {
            let (m, tree) = tree_with(log, width, 0x80 + log as u64);
            let root = tree.root();
            for j in 0..(1usize << (log - 1)) {
                let path = tree.auth_path_pair(j);
                assert!(
                    verify_pair_opening(&root, j, m.row(j), m.row(j + (1usize << (log - 1))), &path),
                    "log={log} j={j}"
                );
            }
        }
    }

    #[test]
    fn size_two_tree_pairs_directly() {
        let (m, tree) = tree_with(1, 2, 0x22);
        assert!(verify_pair_opening(&tree.root(), 0, m.row(0), m.row(1), &[]));
        assert!(!verify_pair_opening(&tree.root(), 0, m.row(1), m.row(0), &[]));
        assert!(!verify_pair_opening(&tree.root(), 1, m.row(0), m.row(1), &[]));
    }

    #[test]
    fn tampered_pair_openings_rejected() {
        let (m, tree) = tree_with(4, 2, 0x77);
        let root = tree.root();
        let j = 3usize;
        let path = tree.auth_path_pair(j);
        let low = m.row(j).to_vec();
        let high = m.row(j + 8).to_vec();

        let mut bad_low = low.clone();
        bad_low[0] = bad_low[0] + Goldilocks::ONE;
        assert!(!verify_pair_opening(&root, j, &bad_low, &high, &path));

        let mut bad_high = high.clone();
        bad_high[1] = bad_high[1] + Goldilocks::ONE;
        assert!(!verify_pair_opening(&root, j, &low, &bad_high, &path));

        for lvl in 0..path.len() {
            let mut bad = path.clone();
            bad[lvl] = Hash256::from_bytes([7u8; 32]);
            assert!(!verify_pair_opening(&root, j, &low, &high, &bad), "lvl={lvl}");
        }

        let mut short = path.clone();
        short.pop();
        assert!(!verify_pair_opening(&root, j, &low, &high, &short));
        let mut long = path.clone();
        long.push(Hash256::from_bytes([7u8; 32]));
        assert!(!verify_pair_opening(&root, j, &low, &high, &long));

        assert!(!verify_pair_opening(&root, j, &high, &low, &path));
        assert!(!verify_pair_opening(&root, j + 1, &low, &high, &path));
        assert!(!verify_pair_opening(&root, j + 8, &low, &high, &path));
        assert!(!verify_pair_opening(&Hash256::from_bytes([3u8; 32]), j, &low, &high, &path));
    }
}


