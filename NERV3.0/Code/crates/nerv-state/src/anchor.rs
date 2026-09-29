//! The anchor freshness ring (WP §4.3 rule 2; erratum 103): the last 64
//! finalized headers' NCT roots against which input-bearing legs' anchors
//! are checked. Seeded at genesis with the genesis state's empty NCT root.

use std::collections::VecDeque;
use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::error::CodecError;
use nerv_core::params::CUSTODY_ANCHOR_WINDOW_HEADERS;
use nerv_custody::NctDigest;

pub const WINDOW: usize = CUSTODY_ANCHOR_WINDOW_HEADERS as usize;

const _: () = assert!(WINDOW > 0);

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AnchorRing {
    roots: VecDeque<NctDigest>,
}

impl AnchorRing {
    /// The chain's initial ring: the genesis state's NCT root at position 0
    /// (erratum 103 — lossless; no membership witness exists against the
    /// empty NCT).
    pub fn genesis(nct_root: NctDigest) -> AnchorRing {
        let mut ring = AnchorRing::default();
        ring.push(nct_root);
        ring
    }

    fn push(&mut self, root: NctDigest) {
        self.roots.push_back(root);
        while self.roots.len() > WINDOW {
            self.roots.pop_front();
        }
    }

    /// Record a finalized header's post-block NCT root.
    pub fn push_header(&mut self, root: NctDigest) {
        self.push(root);
    }

    /// Rule 2: is this NCT root one of the last 64 finalized headers' roots?
    pub fn contains(&self, root: &NctDigest) -> bool {
        self.roots.iter().any(|r| r == root)
    }

    pub fn len(&self) -> usize {
        self.roots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    pub fn latest(&self) -> Option<&NctDigest> {
        self.roots.back()
    }

    pub fn roots(&self) -> impl Iterator<Item = &NctDigest> {
        self.roots.iter()
    }
}
impl Encode for AnchorRing {
   fn encode_into(&self, out: &mut Vec<u8>) {
       out.extend_from_slice(&(self.roots.len() as u32).to_le_bytes());
       for r in &self.roots {
           r.encode_into(out);
       }
   }
   fn encoded_len(&self) -> usize {
       4 + 32 * self.roots.len()
   }
}


impl Decode for AnchorRing {
   fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
       let n = r.read_seq_len()?;
       if n > WINDOW {
           return Err(CodecError::SeqTooLarge { count: n, max: WINDOW });
       }
       let mut roots = VecDeque::with_capacity(n);
       for _ in 0..n {
           roots.push_back(NctDigest::decode_from(r)?);
       }
       Ok(AnchorRing { roots })
   }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use nerv_core::field::Goldilocks;

    fn digest(k: u32) -> NctDigest {
        NctDigest::from_elements(&[
            Goldilocks::from_u32(k),
            Goldilocks::from_u32(k.wrapping_add(1)),
            Goldilocks::from_u32(k.wrapping_mul(3)),
            Goldilocks::from_u32(k.wrapping_mul(7)),
        ])
    }

    #[test]
    fn window_cap_seed_and_eviction() {
        let g = digest(0);
        let mut ring = AnchorRing::genesis(g);
        assert_eq!(ring.len(), 1);
        assert!(ring.contains(&g));
        assert_eq!(ring.latest(), Some(&g));

        for k in 1..64u32 {
            ring.push_header(digest(k));
        }
        assert_eq!(ring.len(), WINDOW);
        assert!(ring.contains(&g), "genesis still in the window at 64 entries");
        assert!(ring.contains(&digest(63)));

        ring.push_header(digest(64));
        assert_eq!(ring.len(), WINDOW);
        assert!(!ring.contains(&g), "genesis evicted when the 64th header lands");
        assert!(!ring.contains(&digest(0)));
        assert!(ring.contains(&digest(1)));
        assert!(ring.contains(&digest(64)));
        assert_eq!(ring.latest(), Some(&digest(64)));
        assert_eq!(ring.roots().count(), WINDOW);
    }

    #[test]
    fn empty_blocks_share_a_root() {
        // Consecutive empty blocks leave the NCT root unchanged; the ring
        // holds duplicates without complaint.
        let r = digest(9);
        let mut ring = AnchorRing::genesis(r);
        for _ in 0..10 {
            ring.push_header(r);
        }
        assert_eq!(ring.len(), 11);
        assert!(ring.contains(&r));
        assert_eq!(ring.latest(), Some(&r));
    }

       #[test]
   fn codec_roundtrip_and_window_cap() {
       let mut ring = AnchorRing::genesis(digest(0));
       for k in 1..70u32 {
           ring.push_header(digest(k));
       }
       assert_eq!(ring.len(), WINDOW);
       let enc = ring.encode();
       assert_eq!(enc.len(), ring.encoded_len());
       assert_eq!(AnchorRing::decode(&enc).unwrap(), ring);
       for cut in 0..enc.len() {
           assert!(AnchorRing::decode(&enc[..cut]).is_err(), "cut {cut}");
       }
       let mut ext = enc.clone();
       ext.push(0);
       assert!(AnchorRing::decode(&ext).is_err());
       let mut bad = vec![0u8; 4];
       bad[..4].copy_from_slice(&((WINDOW + 1) as u32).to_le_bytes());
       assert!(matches!(
           AnchorRing::decode(&bad),
           Err(CodecError::SeqTooLarge { .. })
       ));
       let g = AnchorRing::genesis(digest(3));
       assert_eq!(AnchorRing::decode(&g.encode()).unwrap(), g);
   }

    #[test]
    fn default_ring_is_empty() {
        let ring = AnchorRing::default();
        assert!(ring.is_empty());
        assert_eq!(ring.len(), 0);
        assert!(ring.latest().is_none());
        assert!(!ring.contains(&digest(1)));
    }
}
