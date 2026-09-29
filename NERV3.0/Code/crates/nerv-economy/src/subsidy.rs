
//! The validator subsidy stream (WP §12.3; erratum 162): the per-epoch
//! split of the schedule's validator-subsidy bucket across the active
//! shard set.

use nerv_core::types::{Epoch, ShardId};

use crate::emission::LedgerError;
use crate::schedule::EmissionSchedule;

pub const SUBSIDY_BUCKET: &str = "validator-subsidy";

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SubsidyError {
    #[error("the schedule has no `{bucket}` bucket")]
    NoBucket { bucket: &'static str },
    #[error(transparent)]
    Ledger(#[from] LedgerError),
}

/// The epoch's whole-bucket emission in nano (one day = one epoch).
pub fn epoch_subsidy_nano(schedule: &EmissionSchedule, epoch: Epoch) -> Result<u128, SubsidyError> {
    let b = schedule
        .bucket(SUBSIDY_BUCKET)
        .ok_or(SubsidyError::NoBucket { bucket: SUBSIDY_BUCKET })?;
    Ok(u128::from(b.day_emission(epoch.as_u64())) * 1_000_000_000u128)
}

/// The even split with canonical-order remainder distribution (erratum
/// 162): exact, lossless, deterministic.
pub fn split_subsidy(amount_nano: u128, shards: &[ShardId]) -> Vec<(ShardId, u64)> {
    let mut canonical = shards.to_vec();
    canonical.sort_unstable();
    let n = canonical.len().max(1) as u128;
    let base = amount_nano / n;
    let rem = (amount_nano % n) as usize;
    canonical
        .into_iter()
        .enumerate()
        .map(|(i, shard)| {
            let extra = u64::from(i < rem);
            (shard, base as u64 + extra)
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::schedule::EmissionSchedule;
    use std::collections::BTreeSet;

    fn set(n: u16) -> Vec<ShardId> {
        (0..n).map(|v| ShardId::new(6, v).unwrap()).collect()
    }

    #[test]
    fn genesis_stream_shape() {
        let s = EmissionSchedule::genesis();
        // Day 0: nothing.
        assert_eq!(epoch_subsidy_nano(&s, Epoch::from_u64(0)).unwrap(), 0);
        // Day 1: the constant year-one rate.
        let d1 = epoch_subsidy_nano(&s, Epoch::from_u64(1)).unwrap();
        assert_eq!(d1, u128::from(s.bucket(SUBSIDY_BUCKET).unwrap().day_emission(1)) * 10u128.pow(9));
        // The constant rate through year one; decline after.
        let d360 = epoch_subsidy_nano(&s, Epoch::from_u64(360)).unwrap();
        let d361 = epoch_subsidy_nano(&s, Epoch::from_u64(361)).unwrap();
        let d3599 = epoch_subsidy_nano(&s, Epoch::from_u64(3599)).unwrap();
        assert!(d1 > 0);
        assert_eq!(d360, epoch_subsidy_nano(&s, Epoch::from_u64(200)).unwrap(),
                   "constant within year one");
        assert!(d361 < d360, "the decline begins at year two");
        assert!(d3599 < d361, "the decline continues");
        assert_eq!(epoch_subsidy_nano(&s, Epoch::from_u64(3600)).unwrap(), 3_000_000_000u128 * 10u128.pow(9));
        assert_eq!(epoch_subsidy_nano(&s, Epoch::from_u64(4000)).unwrap(),
                   epoch_subsidy_nano(&s, Epoch::from_u64(3600)).unwrap());
    }

    #[test]
    fn split_exact_and_canonical_remainder() {
        let shards = set(64);
        let split = split_subsidy(1_000_000_000_000_000_001, &shards);
        assert_eq!(split.len(), 64);
        let sum: u128 = split.iter().map(|(_, a)| u128::from(*a)).sum();
        assert_eq!(sum, 1_000_000_000_000_000_001);
        // base = .../64 with remainder 1: shard 0 carries the extra nano.
        let base = 1_000_000_000_000_000_001 / 64;
        assert_eq!(split[0].1 as u128, base + 1);
        assert_eq!(split[1].1 as u128, base);
        assert_eq!(split[63].1 as u128, base);
        // The canonical order is the sorted order.
        let ids: Vec<ShardId> = split.iter().map(|(s, _)| *s).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);

        // Remainder spread: 7 nano over 3 shards → 3, 2, 2.
        let three = set(3);
        let sp = split_subsidy(7, &three);
        assert_eq!(sp.iter().map(|(_, a)| *a).collect::<Vec<_>>(), vec![3, 2, 2]);

        // Order-independence of the input.
        let mut shuffled = shards.clone();
        shuffled.reverse();
        assert_eq!(split_subsidy(999, &shards), split_subsidy(999, &shuffled));

        // Zero and one.
        assert!(split_subsidy(0, &shards).iter().all(|(_, a)| *a == 0));
        let one = split_subsidy(1, &shards);
        assert_eq!(one[0].1, 1);
        assert!(one[1..].iter().all(|(_, a)| *a == 0));
        // Empty shard set: nothing to split (no panic).
        assert!(split_subsidy(5, &[]).is_empty());

        // Distinctness preserved.
        let uniq: BTreeSet<ShardId> = split.iter().map(|(s, _)| *s).collect();
        assert_eq!(uniq.len(), 64);
    }

    #[test]
    fn full_epoch_replay_is_lossless() {
        let s = EmissionSchedule::genesis();
        let shards = set(64);
        for day in [1u64, 2, 100, 360, 361, 1000, 3599] {
            let total = epoch_subsidy_nano(&s, Epoch::from_u64(day)).unwrap();
            let split = split_subsidy(total, &shards);
            let sum: u128 = split.iter().map(|(_, a)| u128::from(*a)).sum();
            assert_eq!(sum, total, "day {day}");
        }
    }
}
