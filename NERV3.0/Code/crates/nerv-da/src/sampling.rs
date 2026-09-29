//! Sampled availability (WP §8.7, §2.5 B7; erratum 141): deterministic
//! position streams, cell authentication, and iterated reconstruction.

use std::collections::BTreeMap;

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::DA_SAMPLE;
use nerv_core::error::CodecError;
use nerv_core::hash::Xof;

use crate::blobs::{cell_leaf, verify_tree_path, BlobSet, SetCommitment, Square, CHUNK_LEN};
use crate::error::DaError;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SamplePos {
    pub blob: u32,
    pub row: u16,
    pub col: u16,
}

/// Deterministic per-blob position stream (erratum 141): reproducible
/// transcripts for fraud evidence.
pub fn sample_positions(set: &SetCommitment, seed: &[u8; 32], per_blob: usize) -> Vec<SamplePos> {
    let mut out = Vec::with_capacity(per_blob * set.blob_count());
    for blob in 0..set.blob_count() {
        let width = match set.width_of(blob) {
            Ok(w) => w,
            Err(_) => continue,
        };
        let shard = set.shard;
        let mut msg = Vec::with_capacity(32 + 3 + 8 + 4 + 2);
        msg.extend_from_slice(seed);
        msg.push(shard.bits());
        msg.extend_from_slice(&shard.value().to_le_bytes());
        msg.extend_from_slice(&set.height.as_u64().to_le_bytes());
        msg.extend_from_slice(&(blob as u32).to_le_bytes());
        msg.extend_from_slice(&(width as u16).to_le_bytes());
        let mut xof = Xof::new(&DA_SAMPLE, &msg);
        for _ in 0..per_blob {
            let r = (xof.next_u64() % width as u64) as u16;
            let c = (xof.next_u64() % width as u64) as u16;
            out.push(SamplePos { blob: blob as u32, row: r, col: c });
        }
    }
    out
}

/// One authenticated cell: the two-path chain of erratum 140.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellAuth {
    pub blob: u32,
    pub row: u16,
    pub col: u16,
    pub chunk: Vec<u8>,
    pub row_root: Hash256Loc,
    pub row_path: Vec<Hash256Loc>,
    pub blob_path: Vec<Hash256Loc>,
}

// A local alias to keep the struct terse; Hash256 throughout.
use nerv_core::hash::Hash256 as Hash256Loc;

impl CellAuth {
    pub fn verify(&self, set: &SetCommitment) -> bool {
        let blob = self.blob as usize;
        if blob >= set.blob_count() {
            return false;
        }
        let width = match set.width_of(blob) {
            Ok(w) => w,
            Err(_) => return false,
        };
        if self.row as usize >= width || self.col as usize >= width {
            return false;
        }
        if self.chunk.len() != CHUNK_LEN {
            return false;
        }
        let leaf = cell_leaf(self.blob, self.row, self.col, &self.chunk);
        if !verify_tree_path(&self.row_root, width, self.col as usize, &leaf, &self.row_path) {
            return false;
        }
        let blob_root = match set.blob_tree_roots.get(blob) {
            Some(r) => r,
            None => return false,
        };
        verify_tree_path(blob_root, 2 * width, self.row as usize, &self.row_root, &self.blob_path)
    }
}

impl Encode for CellAuth {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.blob.to_le_bytes());
        out.extend_from_slice(&self.row.to_le_bytes());
        out.extend_from_slice(&self.col.to_le_bytes());
        out.extend_from_slice(self.row_root.as_bytes());
        out.extend_from_slice(&(self.row_path.len() as u32).to_le_bytes());
        for h in self.row_path.iter().chain(&self.blob_path) {
            out.extend_from_slice(h.as_bytes());
        }
        out.extend_from_slice(&(self.blob_path.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.chunk);
    }
    fn encoded_len(&self) -> usize {
        8 + 32 + 4 + 32 * (self.row_path.len() + self.blob_path.len()) + 4 + CHUNK_LEN
    }
}

impl Decode for CellAuth {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let blob = r.read_u32()?;
        let row = r.read_u16()?;
        let col = r.read_u16()?;
        let row_root = Hash256Loc::decode_from(r)?;
        let n = r.read_seq_len()?;
        if n > 16 {
            return Err(CodecError::SeqTooLarge { count: n, max: 16 });
        }
        let mut row_path = Vec::with_capacity(n);
        for _ in 0..n {
            row_path.push(Hash256Loc::decode_from(r)?);
        }
        let m = r.read_seq_len()?;
        if m > 16 {
            return Err(CodecError::SeqTooLarge { count: m, max: 16 });
        }
        let mut blob_path = Vec::with_capacity(m);
        for _ in 0..m {
            blob_path.push(Hash256Loc::decode_from(r)?);
        }
        let chunk = r.take(CHUNK_LEN)?.to_vec();
        Ok(CellAuth { blob, row, col, chunk, row_root, row_path, blob_path })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SampleResult {
    Verified(CellAuth),
    Missing,
}

/// One sampling transcript: the drawn positions and the network's answers.
#[derive(Clone, Debug)]
pub struct SampleRound {
    pub positions: Vec<SamplePos>,
    pub results: Vec<SampleResult>,
}

impl SampleRound {
    pub fn run<F>(set: &SetCommitment, seed: &[u8; 32], per_blob: usize, fetch: F) -> SampleRound
    where
        F: Fn(SamplePos) -> Option<CellAuth>,
    {
        let positions = sample_positions(set, seed, per_blob);
        let results = positions
            .iter()
            .map(|&p| match fetch(p) {
                Some(a) if a.blob == p.blob
                    && a.row == p.row
                    && a.col == p.col
                    && a.verify(set) =>
                {
                    SampleResult::Verified(a)
                }
                _ => SampleResult::Missing,
            })
            .collect();
        SampleRound { positions, results }
    }

    pub fn all_verified(&self) -> bool {
        self.results.iter().all(|r| matches!(r, SampleResult::Verified(_)))
    }

    pub fn missing(&self) -> Vec<SamplePos> {
        self.positions
            .iter()
            .zip(&self.results)
            .filter(|(_, r)| matches!(r, SampleResult::Missing))
            .map(|(p, _)| *p)
            .collect()
    }

    pub fn verified_cells(&self) -> Vec<&CellAuth> {
        self.results
            .iter()
            .filter_map(|r| match r {
                SampleResult::Verified(a) => Some(a),
                SampleResult::Missing => None,
            })
            .collect()
    }
}

/// Reconstruct the full data from verified cells: every cell is
/// authenticated first, then iterated decoding runs per blob, and each
/// reconstructed square is checked against its committed blob-tree root.
pub fn reconstruct_verified(
    set: &SetCommitment,
    auths: &[CellAuth],
) -> Result<Vec<u8>, DaError> {
    let mut per_blob: Vec<BTreeMap<(u16, u16), Vec<u8>>> =
        vec![BTreeMap::new(); set.blob_count()];
    for a in auths {
        if !a.verify(set) {
            return Err(DaError::RootMismatch);
        }
        let b = a.blob as usize;
        if b >= set.blob_count() {
            return Err(DaError::BlobCount { found: b + 1, expected: set.blob_count() });
        }
        per_blob[b].insert((a.row, a.col), a.chunk.clone());
    }
    let mut out = Vec::new();
    for (b, cells) in per_blob.iter().enumerate() {
        let width = set.width_of(b)?;
        let square = Square::reconstruct(width, cells)?;
        if square.blob_tree_root(b as u32) != set.blob_tree_roots[b] {
            return Err(DaError::RootMismatch);
        }
        out.extend(square.data(set.data_lens[b]));
    }
    Ok(out)
}

/// Convenience: run a full round against a local BlobSet (producers and
/// tests; the network fetch is the client's closure in production).
pub fn round_against(
    set: &SetCommitment,
    blobs: &BlobSet,
    seed: &[u8; 32],
    per_blob: usize,
) -> SampleRound {
    SampleRound::run(set, seed, per_blob, |p| {
        if (p.blob as usize) < blobs.blob_count() {
            Some(blobs.cell_auth(p.blob as usize, p.row as usize, p.col as usize))
        } else {
            None
        }
    })
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;
    use nerv_core::types::{Height, ShardSet};

    fn shard() -> nerv_core::types::ShardId {
        ShardSet::genesis().ids()[7]
    }

    fn data(seed: u64, len: usize) -> Vec<u8> {
        SplitMix64::new(seed).bytes(len)
    }

    #[test]
    fn positions_deterministic_and_pinned() {
        let d = data(1, 3 * CHUNK_LEN);
        let set = BlobSet::encode(shard(), Height::from_u64(4), &d).unwrap();
        let c = set.commitment();
        let seed = [7u8; 32];
        let a = sample_positions(&c, &seed, 5);
        let b = sample_positions(&c, &seed, 5);
        assert_eq!(a, b);
        assert_eq!(a.len(), 5);
        assert!(a.iter().all(|p| (p.row as usize) < 4 && (p.col as usize) < 4));
        // Manual XOF replication.
        let mut msg = Vec::new();
        msg.extend_from_slice(&seed);
        msg.push(shard().bits());
        msg.extend_from_slice(&shard().value().to_le_bytes());
        msg.extend_from_slice(&4u64.to_le_bytes());
        msg.extend_from_slice(&0u32.to_le_bytes());
        msg.extend_from_slice(&4u16.to_le_bytes());
        let mut xof = Xof::new(&DA_SAMPLE, &msg);
        for p in &a {
            assert_eq!(p.row as u64, xof.next_u64() % 4);
            assert_eq!(p.col as u64, xof.next_u64() % 4);
        }
        // Seed and set sensitivity.
        assert_ne!(a, sample_positions(&c, &[8u8; 32], 5));
        let d2 = BlobSet::encode(shard(), Height::from_u64(5), &d).unwrap();
        assert_ne!(a, sample_positions(&d2.commitment(), &seed, 5));
    }

    #[test]
    fn cell_auth_verification_and_tampering() {
        let d = data(2, 5 * CHUNK_LEN);
        let set = BlobSet::encode(shard(), Height::from_u64(1), &d).unwrap();
        let c = set.commitment();
        let auth = set.cell_auth(0, 2, 3);
        assert!(auth.verify(&c));

        let mut bad = auth.clone();
        bad.chunk[0] ^= 1;
        assert!(!bad.verify(&c));

        let mut bad = auth.clone();
        bad.row_root = Hash256Loc::from_bytes([9u8; 32]);
        assert!(!bad.verify(&c));

        let mut bad = auth.clone();
        if !bad.row_path.is_empty() {
            bad.row_path[0] = Hash256Loc::from_bytes([9u8; 32]);
            assert!(!bad.verify(&c));
        }
        let mut bad = auth.clone();
        if !bad.blob_path.is_empty() {
            bad.blob_path[0] = Hash256Loc::from_bytes([9u8; 32]);
            assert!(!bad.verify(&c));
        }

        let mut bad = auth.clone();
        bad.row = 3;
        bad.col = 2;
        assert!(!bad.verify(&c), "position swap breaks the leaf binding");

        let mut forged_set = c.clone();
        forged_set.blob_tree_roots[0] = Hash256Loc::from_bytes([9u8; 32]);
        assert!(!auth.verify(&forged_set));

        let enc = auth.encode();
        assert_eq!(enc.len(), auth.encoded_len());
        assert_eq!(CellAuth::decode(&enc).unwrap(), auth);
        assert!(CellAuth::decode(&enc[..enc.len() - 1]).is_err());
    }

    fn withhold_map(
        set: &BlobSet,
        blob: usize,
        skip: &dyn Fn(usize, usize) -> bool,
    ) -> BTreeMap<(u16, u16), Vec<u8>> {
        let s = set.square(blob);
        let w = s.width();
        let mut m = BTreeMap::new();
        for r in 0..w {
            for c in 0..w {
                if !skip(r, c) {
                    m.insert((r as u16, c as u16), s.cell(r, c).to_vec());
                }
            }
        }
        m
    }

    #[test]
    fn reconstruction_recoverable_patterns() {
        let d = data(3, 5 * CHUNK_LEN + 3);
        let set = BlobSet::encode(shard(), Height::from_u64(1), &d).unwrap();
        let c = set.commitment();
        let w = set.square(0).width();
        let k = set.square(0).k();

        // All row parity withheld: rows decode from their data halves.
        let m = withhold_map(&set, 0, &|_r, col| col >= k);
        let s = Square::reconstruct(w, &m).unwrap();
        assert_eq!(s.data(d.len() as u32), d);
        assert_eq!(s.blob_tree_root(0), c.blob_tree_roots[0]);

        // All column parity withheld: columns decode.
        let m = withhold_map(&set, 0, &|row, _c| row >= k);
        let s = Square::reconstruct(w, &m).unwrap();
        assert_eq!(s.data(d.len() as u32), d);

        // Every row missing its last cell (k of 2k remain).
        let m = withhold_map(&set, 0, &|_r, c| c == w - 1);
        let s = Square::reconstruct(w, &m).unwrap();
        assert_eq!(s.data(d.len() as u32), d);

        // 30% pseudo-random withholding (k=4: every row keeps ≥ 5 of 8).
        let mut rng = SplitMix64::new(99);
        let m = withhold_map(&set, 0, &|_r, _c| rng.next_u64() % 10 < 3);
        let s = Square::reconstruct(w, &m).unwrap();
        assert_eq!(s.data(d.len() as u32), d);

        // Everything present.
        let m = withhold_map(&set, 0, &|_r, _c| false);
        let s = Square::reconstruct(w, &m).unwrap();
        assert_eq!(s, *set.square(0));
    }

    #[test]
    fn reconstruction_unrecoverable_patterns() {
        let d = data(4, 5 * CHUNK_LEN);
        let set = BlobSet::encode(shard(), Height::from_u64(1), &d).unwrap();
        let w = set.square(0).width();
        let k = set.square(0).k();

        // The (k+1)×(k+1) corner: the adversary's best (erratum 141).
        let m = withhold_map(&set, 0, &|r, c| r < k + 1 && c < k + 1);
        assert!(Square::reconstruct(w, &m).is_err());

        // A full data row plus all column parity withheld.
        let m = withhold_map(&set, 0, &|r, _c| r == 0 || r >= k);
        assert!(Square::reconstruct(w, &m).is_err());

        // Out-of-range / bad-length cells are errors, not None.
        let mut m = withhold_map(&set, 0, &|_r, _c| false);
        m.insert((w as u16, 0), vec![0u8; CHUNK_LEN]);
        assert!(matches!(
            Square::reconstruct(w, &m),
            Err(DaError::CellRange { .. })
        ));
        let mut m = withhold_map(&set, 0, &|_r, _c| false);
        m.insert((0, 0), vec![0u8; 7]);
        assert!(matches!(
            Square::reconstruct(w, &m),
            Err(DaError::ChunkLength { .. })
        ));
        assert!(matches!(Square::reconstruct(3, &Default::default()), Err(DaError::BadWidth { .. })));
    }

    #[test]
    fn reconstruct_verified_end_to_end() {
        let d = data(5, 5 * CHUNK_LEN + 11);
        let set = BlobSet::encode(shard(), Height::from_u64(2), &d).unwrap();
        let c = set.commitment();

        // Auths for a recoverable pattern (all row parity withheld).
        let k = set.square(0).k();
        let mut auths = Vec::new();
        for r in 0..set.square(0).width() {
            for col in 0..k {
                auths.push(set.cell_auth(0, r, col));
            }
        }
        assert_eq!(reconstruct_verified(&c, &auths).unwrap(), d);

        // A tampered auth fails verification.
        let mut bad = auths[0].clone();
        bad.chunk[5] ^= 1;
        assert!(matches!(
            reconstruct_verified(&c, &[bad]),
            Err(DaError::RootMismatch)
        ));

        // Insufficient cells fail reconstruction.
        assert!(reconstruct_verified(&c, &auths[..2]).is_err());
    }

    #[test]
    fn sample_round_transcript() {
        let d = data(6, 5 * CHUNK_LEN);
        let set = BlobSet::encode(shard(), Height::from_u64(3), &d).unwrap();
        let c = set.commitment();
        let seed = [3u8; 32];
        let round = round_against(&c, &set, &seed, 12);
        assert_eq!(round.positions.len(), 12);
        assert!(round.all_verified());
        assert!(round.missing().is_empty());
        assert_eq!(round.verified_cells().len(), 12);

        // A fetch that withholds everything.
        let withheld = SampleRound::run(&c, &seed, 12, |_| None);
        assert!(!withheld.all_verified());
        assert_eq!(withheld.missing().len(), 12);

        // A lying fetch (wrong chunk at every position).
        let lying = SampleRound::run(&c, &seed, 12, |p| {
            let mut a = set.cell_auth(p.blob as usize, p.row as usize, p.col as usize);
            a.chunk[0] ^= 1;
            a.row = p.row; // keep position fields aligned
            a.col = p.col;
            Some(a)
        });
        assert!(!lying.all_verified());
        assert_eq!(lying.missing().len(), 12, "tampered cells count as missing");
    }
}
