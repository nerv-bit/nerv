//! Finality (WP §4.8 T4, §5.5; errata 127, 132): the beacon's interval
//! watermark and the per-shard finalized line; no reorg past a finalized
//! interval.
//!
//! The two distinct notions live side-by-side:
//! * [`ShardFinality`] is the per-shard chain guard — the witness of the
//!   one line the shard is on, with `chain` (the hash of every height),
//!   `links` (the parent recorded at each height) and `alts` (conflicts
//!   recorded at heights below the line).
//! * [`IntervalFinality`] is the beacon's interval watermark — the
//!   monotonically advancing last-finalized interval, with no reorg past
//!   it. The chain is contiguous: finalize must always be `last + 1`.
//!
//! Both surface a small, frozen error type and three observe outcomes.

use std::collections::BTreeMap;

use nerv_core::hash::Hash256;
use nerv_core::types::{Interval, ShardId};


/// Errors raised by finality operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum FinalityError {
    #[error("detached header at height {height}, chain tip is {tip}")]
    Detached { height: u64, tip: u64 },
    #[error("unknown height {height}, chain tip is {tip}")]
    UnknownHeight { height: u64, tip: u64 },
    #[error("interval gap: expected {expected}, found {found}")]
    IntervalGap { expected: u64, found: u64 },
}


/// The outcome of observing a header against the per-shard guard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObserveOutcome {
    /// The header extended the active line.
    Attached,
    /// The header displaced the current tip via a competing, lower-hash
    /// fork (the canonical reorg rule).
    Reorganized,
    /// The header was recorded as an alternate at its height, but did
    /// not become the line (either an off-line conflict or a heavier
    /// fork at a non-extending height).
    ForkRecorded,
}


/// A recorded conflict between the finalized line and an alternate fork.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalityViolation {
    pub height: u64,
    pub finalized_hash: Hash256,
    pub conflicting_hash: Hash256,
}


/// The per-shard chain guard: one shard's contiguous `chain`, with its
/// parent links and any recorded alts below the line.
#[derive(Debug, Clone)]
pub struct ShardFinality {
    pub shard: ShardId,
    genesis: Hash256,
    chain: Vec<Hash256>,
    links: Vec<Hash256>,
    alts: BTreeMap<u64, Vec<Hash256>>,
    finalized: u64,
}


impl ShardFinality {
    pub fn new(shard: ShardId, c0: Hash256) -> ShardFinality {
        ShardFinality {
            shard,
            genesis: c0,
            chain: vec![c0],
            links: Vec::new(),
            alts: BTreeMap::new(),
            finalized: 0,
        }
    }


    /// The shard's tip (height, header hash) or None for an empty guard.
    pub fn tip(&self) -> Option<(u64, Hash256)> {
        if self.chain.is_empty() {
            None
        } else {
            let h = self.chain.len() as u64 - 1;
            Some((h, self.chain[h as usize]))
        }
    }


    /// The recorded alternate hashes by height (read-only).
    pub fn alts(&self) -> &BTreeMap<u64, Vec<Hash256>> {
        &self.alts
    }


    /// Observe a QC-validated header at `height` with hash `hash`,
    /// linking back to `link`. Returns [`ObserveOutcome::Attached`]
    /// when the header extends the line, [`ObserveOutcome::Reorganized`]
    /// when a lower-hash competing fork displaces the tip, and
    /// [`ObserveOutcome::ForkRecorded`] when the header is an alternate
    /// that does not become the line.
    pub fn observe(
        &mut self,
        height: u64,
        hash: Hash256,
        link: Hash256,
    ) -> Result<ObserveOutcome, FinalityError> {
        if height == self.chain.len() as u64 {
            let expected = self.links.last().copied().unwrap_or(self.genesis);
            if link == expected {
                self.chain.push(hash);
                self.links.push(link);
                return Ok(ObserveOutcome::Attached);
            }
            if link < expected {
                self.links.push(link);
                return Ok(ObserveOutcome::Reorganized);
            }
            self.alts.entry(height).or_default().push(hash);
            return Ok(ObserveOutcome::ForkRecorded);
        }
        if height == self.chain.len() as u64 + 1 {
            let expected = self.links.last().copied().unwrap_or(self.genesis);
            if link == expected {
                self.chain.push(hash);
                self.links.push(link);
                return Ok(ObserveOutcome::Attached);
            }
            self.alts.entry(height).or_default().push(hash);
            return Ok(ObserveOutcome::ForkRecorded);
        }
        Err(FinalityError::Detached { height, tip: self.chain.len() as u64 })
    }


    /// Finalize through `height`: flushes every recorded alt at or below
    /// it as a violation, and freezes the line. Idempotent.
    pub fn finalize(&mut self, height: u64) -> Result<Vec<FinalityViolation>, FinalityError> {
        if height > self.chain.len() as u64 {
            return Err(FinalityError::UnknownHeight { height, tip: self.chain.len() as u64 });
        }
        if height <= self.finalized {
            return Ok(Vec::new());
        }
        let mut violations = Vec::new();
        for h in (self.finalized + 1)..=height {
            let line = self.chain[(h - 1) as usize];
            if let Some(alts_at_h) = self.alts.remove(&h) {
                for alt in alts_at_h {
                    violations.push(FinalityViolation {
                        height: h,
                        finalized_hash: line,
                        conflicting_hash: alt,
                    });
                }
            }
        }
        self.finalized = height;
        Ok(violations)
    }
}


/// The beacon's interval watermark: monotonically advancing, with no
/// reorg past the last finalized interval.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IntervalFinality {
    last: Option<Interval>,
}


impl IntervalFinality {
    pub fn new() -> IntervalFinality {
        IntervalFinality { last: None }
    }


    /// The last finalized interval, or None for an empty watermark.
    pub fn last(&self) -> Option<Interval> {
        self.last
    }


    /// True iff `interval` is at or below the watermark.
    pub fn is_finalized(&self, interval: Interval) -> bool {
        match self.last {
            None => false,
            Some(l) => interval.as_u64() <= l.as_u64(),
        }
    }


    /// Advance the watermark to `interval`. The call must be exactly
    /// `last + 1` (or any interval from genesis). Returns
    /// [`FinalityError::IntervalGap`] otherwise.
    pub fn finalize(&mut self, interval: Interval) -> Result<(), FinalityError> {
        let next = match self.last {
            None => 0,
            Some(l) => l.as_u64() + 1,
        };
        if interval.as_u64() != next {
            return Err(FinalityError::IntervalGap {
                expected: next,
                found: interval.as_u64(),
            });
        }
        self.last = Some(interval);
        Ok(())
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
   use super::*;
   use crate::testutil::SplitMix64;


   fn h(seed: u64) -> Hash256 {
       Hash256::from_bytes(SplitMix64::new(seed).bytes32())
   }


   fn shard() -> ShardId {
       nerv_core::types::ShardSet::genesis().ids()[7]
   }


   #[test]
   fn interval_watermark() {
       let mut f = IntervalFinality::new();
       assert_eq!(f.last(), None);
       assert!(!f.is_finalized(Interval::from_u64(0)));
       f.finalize(Interval::from_u64(5)).unwrap();
       assert!(f.is_finalized(Interval::from_u64(5)));
       assert!(f.is_finalized(Interval::from_u64(0)));
       assert!(!f.is_finalized(Interval::from_u64(6)));
       f.finalize(Interval::from_u64(6)).unwrap();
       assert!(matches!(
           f.finalize(Interval::from_u64(8)),
           Err(FinalityError::IntervalGap { expected: 7, found: 8 })
       ));
       assert_eq!(f.last(), Some(Interval::from_u64(6)));
   }
}