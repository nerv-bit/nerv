//! The B7 bound and the deterministic DA fraud (WP §2.5, §8.7; errata
//! 141–142).

use std::collections::BTreeSet;

use nerv_core::codec::Encode;
use nerv_core::constants::DA_FRAUD;
use nerv_core::hash::Hash256;

use crate::blobs::{cell_leaf, tree_root, verify_tree_path, SetCommitment};
use crate::erasure;
use crate::error::AvailabilityError;
use crate::sampling::CellAuth;

pub const B7_WITHHOLD_PERMILLE: u32 = nerv_core::params::DA_B7_WITHHOLD_PERMILLE as u32;
pub const B7_DETECTION_PERMILLE: u32 = nerv_core::params::DA_B7_DETECTION_PERMILLE as u32;
/// The defensive iteration cap — unreachable (the exact requirement is at
/// most 6905 samples, at w = 1‰, p = 999‰).
const MAX_M: u32 = 8192;

/// The smallest m with ((1000−w)/1000)^m ≤ (1000−p)/1000 — the sample
/// count that detects a w-fraction withholding with probability ≥ p
/// (erratum 141). Exact u128 fixed point (64 fractional bits,
/// ceil-on-miss, floor-on-threshold): the returned m provably satisfies
/// the bound; `None` on invalid parameters.
pub fn samples_needed(withhold_permille: u32, detection_permille: u32) -> Option<u32> {
    if withhold_permille == 0 {
        return Some(0);
    }
    if withhold_permille >= 1000 {
        return Some(1);
    }
    if detection_permille == 0 || detection_permille > 999 {
        return None;
    }
    const ONE: u128 = 1u128 << 64;
    let a = (1000 - withhold_permille) as u128;
    let b: u128 = 1000;
    let t = ((1000 - detection_permille) as u128) * ONE / b;
    let mut x = ONE;
    let mut m = 0u32;
    loop {
        if x <= t {
            return Some(m);
        }
        m += 1;
        if m > MAX_M {
            return None;
        }
        let q = x / b;
        let r = x % b;
        let mut y = q * a + (r * a) / b;
        if (r * a) % b != 0 {
            y += 1;
        }
        x = y;
    }
}

/// The genesis default: 31 samples (erratum 141).
pub fn default_samples() -> u32 {
    samples_needed(B7_WITHHOLD_PERMILLE, B7_DETECTION_PERMILLE).unwrap_or(31)
}

/// BadEncoding evidence (erratum 142): k authenticated cells of one
/// committed line, the line's committed root, and its blob-tree path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BadEncodingEvidence {
    pub blob: u32,
    pub is_row: bool,
    pub line: u16,
    pub cells: Vec<CellAuth>,
    pub line_root: Hash256,
    pub line_blob_path: Vec<Hash256>,
}

impl BadEncodingEvidence {
    /// H("nerv.da.fraud" ‖ class ‖ blob ‖ line ‖ line_root ‖ cells).
    pub fn digest(&self) -> Hash256 {
        let mut buf = Vec::with_capacity(48 + self.cells.len() * (CHUNK_LEN + 4));
        buf.push(0);
        buf.extend_from_slice(&self.blob.to_le_bytes());
        buf.push(u8::from(self.is_row));
        buf.extend_from_slice(&self.line.to_le_bytes());
        buf.extend_from_slice(self.line_root.as_bytes());
        for c in &self.cells {
            buf.extend_from_slice(&c.row.to_le_bytes());
            buf.extend_from_slice(&c.col.to_le_bytes());
            buf.extend_from_slice(&c.chunk);
        }
        Hash256::concat(&DA_FRAUD, &buf)
    }

    pub fn verify(&self, set: &SetCommitment) -> Result<(), AvailabilityError> {
        let blob = self.blob as usize;
        if blob >= set.blob_count() {
            return Err(AvailabilityError::Malformed("blob outside the set"));
        }
        let width = set
            .width_of(blob)
            .map_err(|_| AvailabilityError::Malformed("bad committed width"))?;
        if self.line as usize >= width {
            return Err(AvailabilityError::Malformed("line index outside the square"));
        }
        let k = width / 2;
        if self.cells.len() != k {
            return Err(AvailabilityError::Malformed("cell count must equal k"));
        }
        let mut cross: BTreeSet<u16> = BTreeSet::new();
        for c in &self.cells {
            if !c.verify(set) {
                return Err(AvailabilityError::Malformed("cell authentication failed"));
            }
            if c.blob != self.blob {
                return Err(AvailabilityError::Malformed("cell from a foreign blob"));
            }
            let on_line = if self.is_row { c.row == self.line } else { c.col == self.line };
            if !on_line {
                return Err(AvailabilityError::Malformed("cell not on the challenged line"));
            }
            let pos = if self.is_row { c.col } else { c.row };
            if !cross.insert(pos) {
                return Err(AvailabilityError::Malformed("duplicate cross-position"));
            }
        }
        let index = if self.is_row { self.line as usize } else { width + self.line as usize };
        if !verify_tree_path(
            &set.blob_tree_roots[blob],
            2 * width,
            index,
            &self.line_root,
            &self.line_blob_path,
        ) {
            return Err(AvailabilityError::Malformed("line root authentication failed"));
        }
        let shards: Vec<(usize, &[u8])> = self
            .cells
            .iter()
            .map(|c| {
                let pos = if self.is_row { c.col } else { c.row };
                (pos as usize, c.chunk.as_slice())
            })
            .collect();
        let codeword = erasure::full_codeword(&shards, k, width)?;
        let leaves: Vec<Hash256> = (0..width)
            .map(|i| {
                if self.is_row {
                    cell_leaf(self.blob, self.line, i as u16, &codeword[i])
                } else {
                    cell_leaf(self.blob, i as u16, self.line, &codeword[i])
                }
            })
            .collect();
        if tree_root(&leaves) != self.line_root {
            Ok(())
        } else {
            Err(AvailabilityError::NotExhibited)
        }
    }
}

use crate::blobs::CHUNK_LEN;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::blobs::{tree_path, BlobSet, Square};
    use crate::testutil::SplitMix64;
    use nerv_core::types::{Height, ShardSet};

    fn shard() -> nerv_core::types::ShardId {
        ShardSet::genesis().ids()[7]
    }

    fn data(seed: u64, len: usize) -> Vec<u8> {
        SplitMix64::new(seed).bytes(len)
    }

    #[test]
    fn samples_needed_pins() {
        // Genesis parameters: 0.8^31 = 7.89e-4 ≤ 1e-3 < 0.8^30.
        assert_eq!(samples_needed(200, 999), Some(31));
        assert_eq!(default_samples(), 31);
        assert_eq!(samples_needed(500, 999), Some(10));
        assert_eq!(samples_needed(900, 999), Some(3));
        assert_eq!(samples_needed(999, 999), Some(1));
        assert_eq!(samples_needed(0, 999), Some(0));
        assert_eq!(samples_needed(1000, 999), Some(1));
        assert_eq!(samples_needed(200, 0), None);
        assert_eq!(samples_needed(200, 1000), None);
        // Wide parameters: exact values with one step of slack.
        let check = |w: u32, lo: u32, hi: u32| {
            let m = samples_needed(w, 999).unwrap();
            assert!((lo..=hi).contains(&m), "w={w}: {m} not in [{lo},{hi}]");
        };
        check(100, 66, 67);
        check(50, 135, 136);
        check(10, 688, 689);
        check(1, 6905, 6906);
        // Monotone in w (more withholding ⟹ fewer samples).
        let mut prev = u32::MAX;
        for w in [1u32, 5, 10, 50, 100, 200, 500, 900, 999] {
            let m = samples_needed(w, 999).unwrap();
            assert!(m <= prev, "monotonicity at w={w}");
            prev = m;
        }
        // The 3/4 structural bound: 31 samples beat it too.
        // 0.75^31 ≈ 1.0e-4 ≤ 1e-3 (integer check).
        let mut miss: u128 = 1;
        for _ in 0..31 {
            miss = (miss * 3) / 4; // floor — an upper bound needs no ceil here
        }
        // miss·4^31 ≤ 10^-3·4^31 ⟺ miss·4000 ≤ 4^31.
        assert!(miss * 4000 <= u128::pow(4, 31));
    }

    /// A corrupted commitment: row `bad_row` committed as a non-codeword
    /// (one cell flipped), everything else honest.
    fn corrupted(d: &[u8], bad_row: usize) -> (BlobSet, SetCommitment) {
        let set = BlobSet::encode(shard(), Height::from_u64(1), d).unwrap();
        let honest = set.commitment();
        let s = set.square(0);
        let width = s.width();

        let mut row_cells: Vec<Vec<u8>> = (0..width).map(|c| s.cell(bad_row, c).to_vec()).collect();
        row_cells[0][17] ^= 1;
        let leaves: Vec<Hash256> = (0..width)
            .map(|c| cell_leaf(0, bad_row as u16, c as u16, &row_cells[c]))
            .collect();
        let bad_root = tree_root(&leaves);

        let mut blob_leaves = s.row_roots(0);
        blob_leaves.extend(s.col_roots(0));
        blob_leaves[bad_row] = bad_root;
        let corrupted = SetCommitment {
            widths: honest.widths.clone(),
            data_lens: honest.data_lens.clone(),
            blob_tree_roots: vec![tree_root(&blob_leaves)],
            ..honest
        };
        (set, corrupted)
    }

    fn evidence_for(
        set: &BlobSet,
        bad_root: Hash256,
        blob_leaves: &[Hash256],
        width: usize,
        line: usize,
        row_cells: &[Vec<u8>],
    ) -> BadEncodingEvidence {
        let k = width / 2;
        let mut cells = Vec::new();
        for col in 0..k {
            let leaves: Vec<Hash256> = (0..width)
                .map(|c| cell_leaf(0, line as u16, c as u16, &row_cells[c]))
                .collect();
            cells.push(CellAuth {
                blob: 0,
                row: line as u16,
                col: col as u16,
                chunk: row_cells[col].clone(),
                row_root: bad_root,
                row_path: tree_path(&leaves, col),
                blob_path: tree_path(blob_leaves, line),
            });
        }
        BadEncodingEvidence {
            blob: 0,
            is_row: true,
            line: line as u16,
            cells,
            line_root: bad_root,
            line_blob_path: tree_path(blob_leaves, line),
        }
    }

    #[test]
    fn bad_encoding_fraud_proven() {
        let d = data(0xB4D, 5 * crate::blobs::CHUNK_LEN);
        let bad_row = 1usize;
        let (set, corrupted) = corrupted(&d, bad_row);
        let width = set.square(0).width();
        let s = set.square(0);
        let mut row_cells: Vec<Vec<u8>> =
            (0..width).map(|c| s.cell(bad_row, c).to_vec()).collect();
        row_cells[0][17] ^= 1;
        let leaves: Vec<Hash256> = (0..width)
            .map(|c| cell_leaf(0, bad_row as u16, c as u16, &row_cells[c]))
            .collect();
        let bad_root = tree_root(&leaves);
        let mut blob_leaves = s.row_roots(0);
        blob_leaves.extend(s.col_roots(0));
        blob_leaves[bad_row] = bad_root;

        let ev = evidence_for(&set, bad_root, &blob_leaves, width, bad_row, &row_cells);
        ev.verify(&corrupted).unwrap();
        assert_eq!(ev.digest(), ev.digest());

        // The same evidence against the HONEST commitment: the line root
        // does not authenticate (the honest root is the codeword's).
        let honest = set.commitment();
        assert!(matches!(
            ev.verify(&honest),
            Err(AvailabilityError::Malformed("line root authentication failed"))
        ));

        // Honest cells against the honest commitment: not exhibited.
        let honest_cells: Vec<Vec<u8>> = (0..width).map(|c| s.cell(bad_row, c).to_vec()).collect();
        let honest_root = s.row_roots(0)[bad_row];
        let mut honest_blob_leaves = s.row_roots(0);
        honest_blob_leaves.extend(s.col_roots(0));
        let ev_ok =
            evidence_for(&set, honest_root, &honest_blob_leaves, width, bad_row, &honest_cells);
        ev_ok.verify(&honest).unwrap_err();
        assert!(matches!(ev_ok.verify(&honest), Err(AvailabilityError::NotExhibited)));
    }

    #[test]
    fn bad_encoding_malformed_rejections() {
        let d = data(0xB4E, 5 * crate::blobs::CHUNK_LEN);
        let bad_row = 2usize;
        let (set, corrupted) = corrupted(&d, bad_row);
        let width = set.square(0).width();
        let s = set.square(0);
        let mut row_cells: Vec<Vec<u8>> = (0..width).map(|c| s.cell(bad_row, c).to_vec()).collect();
        row_cells[0][17] ^= 1;
        let leaves: Vec<Hash256> = (0..width)
            .map(|c| cell_leaf(0, bad_row as u16, c as u16, &row_cells[c]))
            .collect();
        let bad_root = tree_root(&leaves);
        let mut blob_leaves = s.row_roots(0);
        blob_leaves.extend(s.col_roots(0));
        blob_leaves[bad_row] = bad_root;
        let base = evidence_for(&set, bad_root, &blob_leaves, width, bad_row, &row_cells);

        let mut e = base.clone();
        e.cells.pop();
        assert!(matches!(e.verify(&corrupted), Err(AvailabilityError::Malformed(_))));

        let mut e = base.clone();
        e.line = width as u16;
        assert!(matches!(e.verify(&corrupted), Err(AvailabilityError::Malformed(_))));

        let mut e = base.clone();
        e.cells[1] = e.cells[0].clone();
        assert!(matches!(e.verify(&corrupted), Err(AvailabilityError::Malformed(_))));

        let mut e = base.clone();
        e.cells[0].chunk[3] ^= 1;
        assert!(matches!(e.verify(&corrupted), Err(AvailabilityError::Malformed(_))));

        let mut e = base.clone();
        e.line_blob_path = vec![];
        assert!(matches!(e.verify(&corrupted), Err(AvailabilityError::Malformed(_))));

        let mut e = base.clone();
        e.blob = 5;
        assert!(matches!(e.verify(&corrupted), Err(AvailabilityError::Malformed(_))));
    }

     #[test]
    fn column_variant() {
        // The column form: cells of one column authenticate against their
        // (corrupted) row roots; the challenged column root is corrupted.
        let d = data(0xB4F, 5 * crate::blobs::CHUNK_LEN);
        let set = BlobSet::encode(shard(), Height::from_u64(1), &d).unwrap();
        let honest = set.commitment();
        let s = set.square(0);
        let width = s.width();
        let k = s.k();
        let col = 1usize;

        // Corrupt cell (0, col) — the column's first cell.
        let mut col_cells: Vec<Vec<u8>> = (0..width).map(|r| s.cell(r, col).to_vec()).collect();
        col_cells[0][9] ^= 1;

        // Row 0 with its corrupted cell; every other row honest.
        let mut row0: Vec<Vec<u8>> = (0..width).map(|c| s.cell(0, c).to_vec()).collect();
        row0[col] = col_cells[0].clone();
        let row0_leaves: Vec<Hash256> =
            (0..width).map(|c| cell_leaf(0, 0u16, c as u16, &row0[c])).collect();
        let row0_root = tree_root(&row0_leaves);

        let col_leaves: Vec<Hash256> = (0..width)
            .map(|r| cell_leaf(0, r as u16, col as u16, &col_cells[r]))
            .collect();
        let bad_col_root = tree_root(&col_leaves);

        let mut blob_leaves = s.row_roots(0);
        blob_leaves.extend(s.col_roots(0));
        blob_leaves[0] = row0_root;
        blob_leaves[width + col] = bad_col_root;
        let corrupted = SetCommitment {
            widths: honest.widths.clone(),
            data_lens: honest.data_lens.clone(),
            blob_tree_roots: vec![tree_root(&blob_leaves)],
            ..honest
        };

        let mut cells = Vec::new();
        for r in 0..k {
            let (chunk, leaves, root) = if r == 0 {
                (row0[col].clone(), row0_leaves.clone(), row0_root)
            } else {
                let l: Vec<Hash256> = (0..width)
                    .map(|c| cell_leaf(0, r as u16, c as u16, s.cell(r, c)))
                    .collect();
                (s.cell(r, col).to_vec(), l, s.row_roots(0)[r])
            };
            cells.push(CellAuth {
                blob: 0,
                row: r as u16,
                col: col as u16,
                chunk,
                row_root: root,
                row_path: tree_path(&leaves, col),
                blob_path: tree_path(&blob_leaves, r),
            });
        }
        let ev = BadEncodingEvidence {
            blob: 0,
            is_row: false,
            line: col as u16,
            cells,
            line_root: bad_col_root,
            line_blob_path: tree_path(&blob_leaves, width + col),
        };
        for c in &ev.cells {
            assert!(c.verify(&corrupted), "cell ({},{})", c.row, c.col);
        }
        ev.verify(&corrupted).unwrap();
        // Against the honest commitment nothing verifies (roots differ).
        assert!(matches!(ev.verify(&honest), Err(AvailabilityError::Malformed(_))));
    }

}

