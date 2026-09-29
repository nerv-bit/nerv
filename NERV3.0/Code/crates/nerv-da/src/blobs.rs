//! The DA square and its commitments (WP §8.7; erratum 140).


use std::collections::BTreeMap;


use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::{DA_CELL, DA_NODE, DA_ROOT};
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::types::{Height, ShardId};


use crate::erasure;
use crate::error::DaError;


pub const CHUNK_LEN: usize = 512;
pub const MAX_K: usize = 64;
pub const MAX_BLOB_DATA: usize = MAX_K * MAX_K * CHUNK_LEN;


pub fn cell_leaf(blob: u32, row: u16, col: u16, chunk: &[u8]) -> Hash256 {
    let mut msg = Vec::with_capacity(8 + chunk.len());
    msg.extend_from_slice(&blob.to_le_bytes());
    msg.extend_from_slice(&row.to_le_bytes());
    msg.extend_from_slice(&col.to_le_bytes());
    msg.extend_from_slice(chunk);
    Hash256::concat(&DA_CELL, &msg)
}


pub fn node_hash(l: &Hash256, r: &Hash256) -> Hash256 {
    let mut msg = [0u8; 64];
    msg[..32].copy_from_slice(l.as_bytes());
    msg[32..].copy_from_slice(r.as_bytes());
    Hash256::concat(&DA_NODE, &msg)
}


/// Complete binary tree over a power-of-two leaf count.
pub fn tree_root(leaves: &[Hash256]) -> Hash256 {
    assert!(leaves.len().is_power_of_two(), "leaf count must be a power of two");
    if leaves.is_empty() {
        return Hash256::from_bytes([0u8; 32]);
    }
    let mut level: Vec<Hash256> = leaves.to_vec();
    while level.len() > 1 {
        level = level.chunks(2).map(|p| node_hash(&p[0], &p[1])).collect();
    }
    level[0]
}


pub fn tree_path(leaves: &[Hash256], index: usize) -> Vec<Hash256> {
    assert!(leaves.len().is_power_of_two());
    assert!(index < leaves.len());
    let mut level: Vec<Hash256> = leaves.to_vec();
    let mut path = Vec::new();
    let mut i = index;
    while level.len() > 1 {
        path.push(level[i ^ 1]);
        level = level.chunks(2).map(|p| node_hash(&p[0], &p[1])).collect();
        i >>= 1;
    }
    path
}


/// Adversarial input is rejected, never panics.
pub fn verify_tree_path(
    root: &Hash256,
    leaf_count: usize,
    index: usize,
    leaf: &Hash256,
    siblings: &[Hash256],
) -> bool {
    if !leaf_count.is_power_of_two() || index >= leaf_count {
        return false;
    }
    if leaf_count == 1 {
        return siblings.is_empty() && leaf == root;
    }
    let depth = leaf_count.trailing_zeros() as usize;
    if siblings.len() != depth {
        return false;
    }
    let mut cur = *leaf;
    let mut i = index;
    for s in siblings {
        cur = if i & 1 == 0 { node_hash(&cur, s) } else { node_hash(s, &cur) };
        i >>= 1;
    }
    cur == *root
}


fn k_for(len: usize) -> Result<usize, DaError> {
    let chunks = (len / CHUNK_LEN) + usize::from(len % CHUNK_LEN != 0);
    let chunks = chunks.max(1);
    let mut k = 1;
    while k * k < chunks {
        k *= 2;
    }
    if k > MAX_K {
        return Err(DaError::BlobOverflow { len, max: MAX_BLOB_DATA });
    }
    Ok(k)
}


fn check_k(k: usize) -> Result<(), DaError> {
    if k == 0 || !k.is_power_of_two() || k > MAX_K {
        return Err(DaError::BadWidth { width: 2 * k, max: 2 * MAX_K });
    }
    Ok(())
}


/// The 2k×2k extended square (erratum 140): k×k data, row parity,
/// column parity — cells stored row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Square {
    k: usize,
    cells: Vec<Vec<u8>>,
}


impl Square {
    pub fn build_with_k(data: &[u8], k: usize) -> Result<Square, DaError> {
        check_k(k)?;
        if data.len() > k * k * CHUNK_LEN {
            return Err(DaError::DataOverflow { len: data.len(), max: k * k * CHUNK_LEN });
        }
        let w = 2 * k;
        let mut cells = vec![vec![0u8; CHUNK_LEN]; w * w];
        for (idx, chunk) in data.chunks(CHUNK_LEN).enumerate() {
            let (r, c) = (idx / k, idx % k);
            cells[r * w + c][..chunk.len()].copy_from_slice(chunk);
        }
        for r in 0..k {
            let row: Vec<Vec<u8>> = (0..k).map(|c| cells[r * w + c].clone()).collect();
            let parity = erasure::encode_parity(&row, k)?;
            for c in 0..k {
                cells[r * w + k + c] = parity[c].clone();
            }
        }
        for c in 0..w {
            let col: Vec<Vec<u8>> = (0..k).map(|r| cells[r * w + c].clone()).collect();
            let parity = erasure::encode_parity(&col, k)?;
            for a in 0..k {
                cells[(k + a) * w + c] = parity[a].clone();
            }
        }
        Ok(Square { k, cells })
    }


    pub fn build(data: &[u8]) -> Result<Square, DaError> {
        Square::build_with_k(data, k_for(data.len())?)
    }


    pub fn k(&self) -> usize {
        self.k
    }


    pub fn width(&self) -> usize {
        2 * self.k
    }


    pub fn cell(&self, row: usize, col: usize) -> &[u8] {
        &self.cells[row * self.width() + col]
    }


    /// The first data_len bytes of the k×k data region.
    pub fn data(&self, data_len: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(data_len as usize);
        for idx in 0..self.k * self.k {
            let (r, c) = (idx / self.k, idx % self.k);
            let chunk = &self.cells[r * self.width() + c];
            if out.len() >= data_len as usize {
                break;
            }
            let take = ((data_len as usize) - out.len()).min(CHUNK_LEN);
            out.extend_from_slice(&chunk[..take]);
        }
        out
    }


    fn row_leaves(&self, blob: u32, row: usize) -> Vec<Hash256> {
        let w = self.width();
        (0..w).map(|c| cell_leaf(blob, row as u16, c as u16, &self.cells[row * w + c])).collect()
    }


    fn col_leaves(&self, blob: u32, col: usize) -> Vec<Hash256> {
        let w = self.width();
        (0..w).map(|r| cell_leaf(blob, r as u16, col as u16, &self.cells[r * w + col])).collect()
    }


    pub fn row_roots(&self, blob: u32) -> Vec<Hash256> {
        (0..self.width()).map(|r| tree_root(&self.row_leaves(blob, r))).collect()
    }


    pub fn col_roots(&self, blob: u32) -> Vec<Hash256> {
        (0..self.width()).map(|c| tree_root(&self.col_leaves(blob, c))).collect()
    }


    /// The blob tree commits row_roots ‖ col_roots (erratum 140).
    pub fn blob_tree_root(&self, blob: u32) -> Hash256 {
        let mut leaves = self.row_roots(blob);
        leaves.extend(self.col_roots(blob));
        tree_root(&leaves)
    }


    /// Iterated row/column decoding to fixpoint (erratum 141). Any
    /// recoverable cell set completes; unrecoverable sets return None.
    pub fn reconstruct(
        width: usize,
        cells: &BTreeMap<(u16, u16), Vec<u8>>,
    ) -> Result<Square, DaError> {
        if width == 0 || width % 2 != 0 || !width.is_power_of_two() || width > 2 * MAX_K {
            return Err(DaError::BadWidth { width, max: 2 * MAX_K });
        }
        let k = width / 2;
        let mut grid: Vec<Option<Vec<u8>>> = vec![None; width * width];
        for (&(r, c), chunk) in cells {
            let (r, c) = (r as usize, c as usize);
            if r >= width || c >= width {
                return Err(DaError::CellRange { row: r, col: c, width });
            }
            if chunk.len() != CHUNK_LEN {
                return Err(DaError::ChunkLength { len: chunk.len(), expected: CHUNK_LEN });
            }
            grid[r * width + c] = Some(chunk.clone());
        }
        loop {
            let mut progress = false;
            for r in 0..width {
                let known: Vec<usize> =
                    (0..width).filter(|&c| grid[r * width + c].is_some()).collect();
                if known.len() >= k && known.len() < width {
                    let shards: Vec<(usize, &[u8])> = known[..k]
                        .iter()
                        .map(|&c| (c, grid[r * width + c].as_ref().unwrap().as_slice()))
                        .collect();
                    let line = erasure::full_codeword(&shards, k, width)?;
                    for c in 0..width {
                        if grid[r * width + c].is_none() {
                            grid[r * width + c] = Some(line[c].clone());
                            progress = true;
                        }
                    }
                }
            }
            for c in 0..width {
                let known: Vec<usize> =
                    (0..width).filter(|&r| grid[r * width + c].is_some()).collect();
                if known.len() >= k && known.len() < width {
                    let shards: Vec<(usize, &[u8])> = known[..k]
                        .iter()
                        .map(|&r| (r, grid[r * width + c].as_ref().unwrap().as_slice()))
                        .collect();
                    let line = erasure::full_codeword(&shards, k, width)?;
                    for r in 0..width {
                        if grid[r * width + c].is_none() {
                            grid[r * width + c] = Some(line[r].clone());
                            progress = true;
                        }
                    }
                }
            }
            if !progress {
                break;
            }
        }
        let cells: Vec<Vec<u8>> = grid
            .into_iter()
            .map(|c| c.ok_or(DaError::ReconstructionFailed))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Square { k, cells })
    }
}


impl Encode for Square {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.k as u16).to_le_bytes());
        for c in &self.cells {
            out.extend_from_slice(c);
        }
    }
    fn encoded_len(&self) -> usize {
        2 + self.cells.len() * CHUNK_LEN
    }
}


impl Decode for Square {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let kb = r.read_u16()? as usize;
        check_k(kb).map_err(|_| CodecError::InvariantViolated("bad square width"))?;
        let w = 2 * kb;
        let n = r.read_seq_len()?;
        if n != w * w {
            return Err(CodecError::InvariantViolated("cell count mismatch"));
        }
        let mut cells = Vec::with_capacity(n);
        for _ in 0..n {
            cells.push(r.take_array::<CHUNK_LEN>()?.to_vec());
        }
        Ok(Square { k: kb, cells })
    }
}


/// The header-side DA commitment: one record per blob.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetCommitment {
    pub shard: ShardId,
    pub height: Height,
    pub widths: Vec<u16>,
    pub data_lens: Vec<u32>,
    pub blob_tree_roots: Vec<Hash256>,
}


impl SetCommitment {
    pub fn blob_count(&self) -> usize {
        self.blob_tree_roots.len()
    }


    pub fn width_of(&self, blob: usize) -> Result<usize, DaError> {
        let w = *self
            .widths
            .get(blob)
            .ok_or(DaError::BlobCount { found: blob + 1, expected: self.blob_count() })?
            as usize;
        if w == 0 || w % 2 != 0 || !w.is_power_of_two() || w > 2 * MAX_K {
            return Err(DaError::BadWidth { width: w, max: 2 * MAX_K });
        }
        Ok(w)
    }


    pub fn root(&self) -> Hash256 {
        let mut msg = Vec::with_capacity(11 + self.blob_count() * 38);
        msg.push(self.shard.bits());
        msg.extend_from_slice(&self.shard.value().to_le_bytes());
        msg.extend_from_slice(&self.height.as_u64().to_le_bytes());
        msg.extend_from_slice(&(self.blob_count() as u32).to_le_bytes());
        for i in 0..self.blob_count() {
            msg.extend_from_slice(&self.widths[i].to_le_bytes());
            msg.extend_from_slice(&self.data_lens[i].to_le_bytes());
            msg.extend_from_slice(self.blob_tree_roots[i].as_bytes());
        }
        Hash256::concat(&DA_ROOT, &msg)
    }
}


impl Encode for SetCommitment {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.shard.encode_into(out);
        out.extend_from_slice(&self.height.as_u64().to_le_bytes());
        out.extend_from_slice(&(self.blob_count() as u32).to_le_bytes());
        for i in 0..self.blob_count() {
            out.extend_from_slice(&self.widths[i].to_le_bytes());
            out.extend_from_slice(&self.data_lens[i].to_le_bytes());
            out.extend_from_slice(self.blob_tree_roots[i].as_bytes());
        }
    }
    fn encoded_len(&self) -> usize {
        3 + 8 + 4 + self.blob_count() * 38
    }
}


impl Decode for SetCommitment {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let shard = ShardId::decode_from(r)?;
        let height = Height::from_u64(r.read_u64()?);
        let n = r.read_seq_len()?;
        if n > 65_536 {
            return Err(CodecError::SeqTooLarge { count: n, max: 65_536 });
        }
        let mut widths = Vec::with_capacity(n);
        let mut data_lens = Vec::with_capacity(n);
        let mut roots = Vec::with_capacity(n);
        for _ in 0..n {
            widths.push(r.read_u16()?);
            data_lens.push(r.read_u32()?);
            roots.push(Hash256::decode_from(r)?);
        }
        Ok(SetCommitment { shard, height, widths, data_lens, blob_tree_roots: roots })
    }
}


/// The producer-side blob set: the block's data split into squares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlobSet {
    shard: ShardId,
    height: Height,
    squares: Vec<Square>,
    data_lens: Vec<u32>,
}


impl BlobSet {
    pub fn encode(shard: ShardId, height: Height, data: &[u8]) -> Result<BlobSet, DaError> {
        let segments: Vec<&[u8]> = if data.is_empty() {
            vec![&[][..]]
        } else {
            data.chunks(MAX_BLOB_DATA).collect()
        };
        let mut squares = Vec::with_capacity(segments.len());
        let mut data_lens = Vec::with_capacity(segments.len());
        for seg in segments {
            squares.push(Square::build(seg)?);
            data_lens.push(seg.len() as u32);
        }
        Ok(BlobSet { shard, height, squares, data_lens })
    }


    pub fn blob_count(&self) -> usize {
        self.squares.len()
    }


    pub fn square(&self, blob: usize) -> &Square {
        &self.squares[blob]
    }


    pub fn data(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for (s, &len) in self.squares.iter().zip(&self.data_lens) {
            out.extend(s.data(len));
        }
        out
    }


    pub fn commitment(&self) -> SetCommitment {
        SetCommitment {
            shard: self.shard,
            height: self.height,
            widths: self.squares.iter().map(|s| s.width() as u16).collect(),
            data_lens: self.data_lens.clone(),
            blob_tree_roots: (0..self.squares.len())
                .map(|b| self.squares[b].blob_tree_root(b as u32))
                .collect(),
        }
    }


    /// The authenticated cell (blob, row, col): the two-path proof chain
    /// of erratum 140.
    pub fn cell_auth(&self, blob: usize, row: usize, col: usize) -> crate::sampling::CellAuth {
        let s = &self.squares[blob];
        let leaves = s.row_leaves(blob as u32, row);
        let row_root = tree_root(&leaves);
        let row_path = tree_path(&leaves, col);
        let mut blob_leaves = s.row_roots(blob as u32);
        blob_leaves.extend(s.col_roots(blob as u32));
        let blob_path = tree_path(&blob_leaves, row);
        crate::sampling::CellAuth {
            blob: blob as u32,
            row: row as u16,
            col: col as u16,
            chunk: s.cell(row, col).to_vec(),
            row_root,
            row_path,
            blob_path,
        }
    }
}


impl Encode for BlobSet {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.shard.encode_into(out);
        out.extend_from_slice(&self.height.as_u64().to_le_bytes());
        out.extend_from_slice(&(self.squares.len() as u32).to_le_bytes());
        for (s, &len) in self.squares.iter().zip(&self.data_lens) {
            s.encode_into(out);
            out.extend_from_slice(&len.to_le_bytes());
        }
    }
    fn encoded_len(&self) -> usize {
        3 + 8 + 4 + self.squares.iter().map(|s| s.encoded_len() + 4).sum::<usize>()
    }
}


impl Decode for BlobSet {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let shard = ShardId::decode_from(r)?;
        let height = Height::from_u64(r.read_u64()?);
        let n = r.read_seq_len()?;
        if n > 65_536 {
            return Err(CodecError::SeqTooLarge { count: n, max: 65_536 });
        }
        let mut squares = Vec::with_capacity(n);
        let mut data_lens = Vec::with_capacity(n);
        for _ in 0..n {
            squares.push(Square::decode_from(r)?);
            data_lens.push(r.read_u32()?);
        }
        Ok(BlobSet { shard, height, squares, data_lens })
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;
    use nerv_core::types::ShardSet;


    fn shard() -> ShardId {
        ShardSet::genesis().ids()[7]
    }


    fn data(seed: u64, len: usize) -> Vec<u8> {
        SplitMix64::new(seed).bytes(len)
    }


    #[test]
    fn tree_construction_and_paths() {
        let leaves: Vec<Hash256> =
            (0..8u16).map(|i| Hash256::from_bytes([i as u8; 32])).collect();
        let root = tree_root(&leaves);
        let mut pre = Vec::new();
        pre.extend_from_slice(DA_NODE.as_bytes());
        pre.extend_from_slice(leaves[0].as_bytes());
        pre.extend_from_slice(leaves[1].as_bytes());
        assert_eq!(node_hash(&leaves[0], &leaves[1]).as_bytes(), blake3::hash(&pre).as_bytes());
        for i in 0..8 {
            let p = tree_path(&leaves, i);
            assert_eq!(p.len(), 3);
            assert!(verify_tree_path(&root, 8, i, &leaves[i], &p), "i={i}");
            assert!(!verify_tree_path(&root, 8, (i + 1) % 8, &leaves[i], &p));
            let mut bad = p.clone();
            bad[0] = Hash256::from_bytes([9u8; 32]);
            assert!(!verify_tree_path(&root, 8, i, &leaves[i], &bad));
            let mut long = p.clone();
            long.push(root);
            assert!(!verify_tree_path(&root, 8, i, &leaves[i], &long));
        }
        let one = vec![leaves[0]];
        assert_eq!(tree_root(&one), leaves[0]);
        assert!(tree_path(&one, 0).is_empty());
        assert!(verify_tree_path(&leaves[0], 1, 0, &leaves[0], &[]));
        assert!(!verify_tree_path(&root, 8, 8, &leaves[0], &tree_path(&leaves, 0)));
        assert!(!verify_tree_path(&root, 6, 0, &leaves[0], &tree_path(&leaves, 0)));
    }


    #[test]
    fn cell_leaf_is_position_bound() {
        let c = vec![7u8; CHUNK_LEN];
        let a = cell_leaf(0, 1, 2, &c);
        assert_eq!(a, cell_leaf(0, 1, 2, &c));
        assert_ne!(a, cell_leaf(1, 1, 2, &c), "blob binds");
        assert_ne!(a, cell_leaf(0, 2, 1, &c), "row/col swap binds");
        let mut c2 = c.clone();
        c2[0] ^= 1;
        assert_ne!(a, cell_leaf(0, 1, 2, &c2));
        let mut msg = Vec::new();
        msg.extend_from_slice(DA_CELL.as_bytes());
        msg.extend_from_slice(&0u32.to_le_bytes());
        msg.extend_from_slice(&1u16.to_le_bytes());
        msg.extend_from_slice(&2u16.to_le_bytes());
        msg.extend_from_slice(&c);
        assert_eq!(a.as_bytes(), blake3::hash(&msg).as_bytes());
    }


    #[test]
    fn square_build_shape_and_padding() {
        let d = data(1, 3 * CHUNK_LEN + 100);
        let s = Square::build(&d).unwrap();
        assert_eq!(s.k(), 2);
        assert_eq!(s.width(), 4);
        // k_for: 4 chunks → k=2.
        assert_eq!(k_for(0).unwrap(), 1);
        assert_eq!(k_for(1).unwrap(), 1);
        assert_eq!(k_for(CHUNK_LEN).unwrap(), 1);
        assert_eq!(k_for(CHUNK_LEN + 1).unwrap(), 2);
        assert_eq!(k_for(5 * CHUNK_LEN).unwrap(), 4);
        assert_eq!(s.data(3 * CHUNK_LEN + 100), d);
        // Data capacity and overflow.
        assert!(Square::build_with_k(&d, 1).is_err());
        assert_eq!(
            Square::build_with_k(&d, 1).unwrap_err(),
            DaError::DataOverflow { len: d.len(), max: CHUNK_LEN }
        );
        assert!(matches!(
            Square::build_with_k(&d, 3),
            Err(DaError::BadWidth { .. })
        ));
        assert!(Square::build_with_k(&d, 0).is_err());
    }


    #[test]
    fn commitments_and_set_root() {
        let d = data(2, 5 * CHUNK_LEN);
        let set = BlobSet::encode(shard(), Height::from_u64(9), &d).unwrap();
        assert_eq!(set.blob_count(), 1);
        let c = set.commitment();
        assert_eq!(c.blob_count(), 1);
        assert_eq!(c.width_of(0).unwrap(), 4);
        assert_eq!(c.data_lens, vec![d.len() as u32]);
        assert_eq!(c.root(), set.commitment().root());
        // Every cell authenticates through both paths.
        for r in 0..4 {
            for col in 0..4 {
                let auth = set.cell_auth(0, r, col);
                assert!(auth.verify(&c), "cell ({r},{col})");
            }
        }
        // The root binds every field.
        let mut m = c.clone();
        m.data_lens[0] += 1;
        assert_ne!(c.root(), m.root());
        let mut m = c.clone();
        m.blob_tree_roots[0] = Hash256::from_bytes([9u8; 32]);
        assert_ne!(c.root(), m.root());
        let mut m = c.clone();
        m.height = Height::from_u64(10);
        assert_ne!(c.root(), m.root());
        let mut m = c.clone();
        m.widths[0] = 8;
        assert_ne!(c.root(), m.root());
    }


    #[test]
    fn blob_set_splitting_and_roundtrip() {
        let d = data(3, MAX_BLOB_DATA + MAX_BLOB_DATA / 2 + 7);
        let set = BlobSet::encode(shard(), Height::from_u64(1), &d).unwrap();
        assert_eq!(set.blob_count(), 2);
        assert_eq!(set.square(0).k(), MAX_K);
        assert!(set.square(1).k() <= MAX_K);
        assert_eq!(set.data(), d);
        let c = set.commitment();
        assert_eq!(c.blob_count(), 2);
        // Cross-blob cell authentication.
        for (b, r, col) in [(0, 0, 0), (0, 127, 127), (1, 0, 0), (1, 3, 3)] {
            assert!(set.cell_auth(b, r, col).verify(&c), "({b},{r},{col})");
        }
        // A cell from blob 0 does not verify as blob 1.
        let mut forged = set.cell_auth(0, 0, 0);
        forged.blob = 1;
        assert!(!forged.verify(&c));


        // Wire roundtrip.
        let enc = set.encode();
        assert_eq!(enc.len(), set.encoded_len());
        let dec = BlobSet::decode(&enc).unwrap();
        assert_eq!(dec, set);
        assert_eq!(dec.commitment().root(), c.root());
        assert!(BlobSet::decode(&enc[..enc.len() - 1]).is_err());
        let mut ext = enc.clone();
        ext.push(0);
        assert!(BlobSet::decode(&ext).is_err());
    }


    #[test]
    fn empty_data_single_blob() {
        let set = BlobSet::encode(shard(), Height::from_u64(0), &[]).unwrap();
        assert_eq!(set.blob_count(), 1);
        assert_eq!(set.square(0).k(), 1);
        assert_eq!(set.square(0).width(), 2);
        assert!(set.data().is_empty());
        let c = set.commitment();
        assert_eq!(c.data_lens, vec![0]);
        assert!(set.cell_auth(0, 0, 0).verify(&c));
    }


    #[test]
    fn set_commitment_codec_and_validation() {
        let d = data(4, 2 * CHUNK_LEN + 5);
        let set = BlobSet::encode(shard(), Height::from_u64(77), &d).unwrap();
        let c = set.commitment();
        let enc = c.encode();
        assert_eq!(enc.len(), c.encoded_len());
        assert_eq!(SetCommitment::decode(&enc).unwrap(), c);
        assert!(SetCommitment::decode(&enc[..enc.len() - 1]).is_err());
        let mut bad = c.clone();
        bad.widths[0] = 3; // not a power of two
        assert!(bad.width_of(0).is_err());
        assert!(matches!(
            c.width_of(1),
            Err(DaError::BlobCount { expected: 1, .. })
        ));
    }


    #[test]
    fn square_codec_roundtrip() {
        let s = Square::build(&data(5, 3 * CHUNK_LEN + 9)).unwrap();
        let enc = s.encode();
        assert_eq!(enc.len(), s.encoded_len());
        assert_eq!(Square::decode(&enc).unwrap(), s);
        assert!(Square::decode(&enc[..enc.len() - 1]).is_err());
        // A corrupted k fails the width check.
        let mut bad = enc.clone();
        bad[0..2].copy_from_slice(&3u16.to_le_bytes());
        assert!(Square::decode(&bad).is_err());
    }
}
