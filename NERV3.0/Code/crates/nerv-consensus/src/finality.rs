//! Finality (WP §4.8 T4, §5.5; errata 127, 132): the beacon's interval
               self.links.push(link);
               return Ok(ObserveOutcome::Reorganized);
           }
           self.alts.entry(height).or_default().insert(hash);
           return Ok(ObserveOutcome::ForkRecorded);
       }
       if h == self.chain.len() + 1 {
           let expected = self.links.last().copied().unwrap_or(self.genesis);
           if prev == expected {
               self.chain.push(hash);
               self.links.push(link);
               return Ok(ObserveOutcome::Attached);
           }
           self.alts.entry(height).or_default().insert(hash);
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
