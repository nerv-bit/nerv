//! The per-shard derived embedding (WP §7.5, §10.1; erratum 167): e_t ∈
//! (Z/2^64)^64 — the public mirror of the economy, advanced by wrapping
//! addition per revealed aggregate, never consulted by anything
//! authoritative. The full history lives in the header chain's reveal
//! stream (chunk 13's prev_reveal); the derived state keeps the
//! accumulator, the contiguity bookkeeping, and the miss log.


use nerv_codec::codec_w::Delta;
use nerv_core::constants::DERIVED_STATE;
use nerv_core::hash::Hash256;


/// A missed reveal in the derived state — the skip-and-carry log
/// (D.1(d); erratum 167). The chunk-9 ceremony's record maps here at the
/// node's wiring boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MissedReveal {
    pub height: u64,
    pub legs: u64,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum EmbedError {
    #[error("reveal height {found} does not follow {expected}")]
    NonContiguous { expected: u64, found: u64 },
}


/// e_t: O(1) updates by wrapping addition (WP §7.5). Heights are
/// contiguous per block: every height is either applied or explicitly
/// missed — a gap is an error, never a silence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Embedding {
    coord: Delta,
    next_height: u64,
    missed: Vec<MissedReveal>,
}


impl Default for Embedding {
    fn default() -> Self {
        Embedding::genesis()
    }
}


impl Embedding {
    pub fn genesis() -> Embedding {
        Embedding { coord: Delta::default(), next_height: 1, missed: Vec::new() }
    }


    pub fn coord(&self) -> &Delta {
        &self.coord
    }


    /// The next height the accumulator expects.
    pub fn next_height(&self) -> u64 {
        self.next_height
    }


    /// Blocks processed so far (applied + missed).
    pub fn blocks_processed(&self) -> u64 {
        self.next_height - 1
    }


    pub fn applied_count(&self) -> u64 {
        self.blocks_processed() - self.missed.len() as u64
    }


    pub fn misses(&self) -> &[MissedReveal] {
        &self.missed
    }


    fn advance(&mut self, height: u64) -> Result<(), EmbedError> {
        if height != self.next_height {
            return Err(EmbedError::NonContiguous { expected: self.next_height, found: height });
        }
        self.next_height += 1;
        Ok(())
    }


    /// e_{t+1} = e_t + Δ_B (mod 2^64) — the exact closed-group update.
    pub fn apply(&mut self, height: u64, delta: &Delta) -> Result<(), EmbedError> {
        self.advance(height)?;
        self.coord = self.coord.wrapping_add(delta);
        Ok(())
    }


    /// The skip-and-carry rule (erratum 167): the height advances, e and
    /// the state carry forward unchanged, the miss is recorded.
    pub fn apply_miss(&mut self, height: u64, legs: u64) -> Result<(), EmbedError> {
        self.advance(height)?;
        self.missed.push(MissedReveal { height, legs });
        Ok(())
    }
}


/// D_t = BLAKE3("nerv.derived" ‖ e_t ‖ forecaster_state_root) (WP §4.2).
/// Fixed widths (512 ‖ 32) — the concatenation is unambiguous.
pub fn derived_state_root(e: &Delta, forecaster_state_root: &[u8; 32]) -> Hash256 {
    let mut msg = Vec::with_capacity(512 + 32);
    msg.extend_from_slice(&e.canonical_bytes());
    msg.extend_from_slice(forecaster_state_root);
    Hash256::concat(&DERIVED_STATE, &msg)
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::forecaster::Forecaster;
    use crate::testutil::SplitMix64;


    #[test]
    fn genesis_state() {
        let e = Embedding::genesis();
        assert!(e.coord().is_zero());
        assert_eq!(e.next_height(), 1);
        assert_eq!(e.blocks_processed(), 0);
        assert_eq!(e.applied_count(), 0);
        assert!(e.misses().is_empty());
    }


    #[test]
    fn apply_wraps_and_misses_skip() {
        let mut e = Embedding::genesis();
        e.apply(1, &Delta([u64::MAX; 64])).unwrap();
        e.apply(2, &Delta([2; 64])).unwrap();
        assert_eq!(e.coord().0[0], 1, "MAX + 2 wraps");
        e.apply_miss(3, 128).unwrap();
        assert_eq!(e.coord().0[0], 1, "a miss leaves e unchanged");
        assert_eq!(e.misses(), &[MissedReveal { height: 3, legs: 128 }]);
        e.apply(4, &Delta([10; 64])).unwrap();
        assert_eq!(e.coord().0[0], 11);
        assert_eq!(e.next_height(), 5);
        assert_eq!(e.blocks_processed(), 4);
        assert_eq!(e.applied_count(), 3);
        assert!(matches!(
            e.apply(6, &Delta::default()),
            Err(EmbedError::NonContiguous { expected: 5, found: 6 })
        ));
        assert!(matches!(e.apply_miss(4, 1), Err(EmbedError::NonContiguous { .. })));
        assert!(matches!(e.apply(0, &Delta::default()), Err(EmbedError::NonContiguous { .. })));
    }


    #[test]
    fn long_chain_matches_a_wrapping_reference() {
        let mut rng = SplitMix64::new(0xE0B);
        let mut e = Embedding::genesis();
        let mut reference = [0u64; 64];
        for h in 1..=10_000u64 {
            if rng.next_u64() % 10 == 0 {
                e.apply_miss(h, 1).unwrap();
            } else {
                let d = Delta(std::array::from_fn(|_| rng.next_u64()));
                for (r, &v) in reference.iter_mut().zip(d.0.iter()) {
                    *r = r.wrapping_add(v);
                }
                e.apply(h, &d).unwrap();
            }
        }
        assert_eq!(e.coord().0, reference);
        assert_eq!(e.applied_count() + e.misses().len() as u64, 10_000);
        assert!(!e.misses().is_empty());
    }


    #[test]
    fn derived_root_is_the_literal_formula() {
        let fs = [7u8; 32];
        let e = Delta([42u64; 64]);
        let d = derived_state_root(&e, &fs);
        let mut pre = Vec::new();
        pre.extend_from_slice(DERIVED_STATE.as_bytes());
        pre.extend_from_slice(&e.canonical_bytes());
        pre.extend_from_slice(&fs);
        assert_eq!(d.as_bytes(), blake3::hash(&pre).as_bytes());
        assert_eq!(d, derived_state_root(&e, &fs));
        assert_ne!(d, derived_state_root(&Delta([43u64; 64]), &fs));
        assert_ne!(d, derived_state_root(&e, &[8u8; 32]));
        assert_ne!(derived_state_root(&Delta::default(), &fs), d);
    }


    #[test]
    fn genesis_derived_state_composition() {
        let e = Embedding::genesis();
        let f = Forecaster::reference();
        let fs = f.state_root();
        let d0 = derived_state_root(e.coord(), fs.as_bytes());
        // Determinism across fresh constructions.
        assert_eq!(
            d0,
            derived_state_root(
                Embedding::genesis().coord(),
                Forecaster::reference().state_root().as_bytes()
            )
        );
        // Either component moves it.
        let mut e2 = Embedding::genesis();
        e2.apply(1, &Delta([1; 64])).unwrap();
        assert_ne!(derived_state_root(e2.coord(), fs.as_bytes()), d0);
        let mut f2 = Forecaster::reference();
        f2.moments_mut().step = 1;
        assert_ne!(derived_state_root(e.coord(), f2.state_root().as_bytes()), d0);
    }
}
