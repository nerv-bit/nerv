//! The supply ledger (WP §12.3 M1, §12.6; erratum 164): the per-epoch
//! emission/burn accumulation and the supply identity.

use std::collections::{BTreeMap, BTreeSet};

use nerv_core::types::Epoch;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BurnCategory {
    TransparentExit,
    UnclaimedWindow,
    AbandonedIssue,
}

impl BurnCategory {
    pub fn name(self) -> &'static str {
        match self {
            BurnCategory::TransparentExit => "transparent-exit",
            BurnCategory::UnclaimedWindow => "unclaimed-window",
            BurnCategory::AbandonedIssue => "abandoned-issue",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BurnRecord {
    pub category: BurnCategory,
    pub amount_nano: u64,
    pub epoch: Epoch,
    /// The category's native digest (the burn commitment / the account /
    /// the transit key) — replay dedup.
    pub reference: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SupplyError {
    #[error("burn {reference:?} is already recorded")]
    DuplicateBurn { reference: [u8; 32] },
    #[error("the burn would drive supply negative — an accounting bug")]
    NegativeSupply,
    #[error("recorded emission {found} does not match the schedule's {expected}")]
    EmissionMismatch { found: u128, expected: u128 },
}

/// One epoch's publication (§12.6's per-epoch supply ledger).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SupplyPublication {
    pub epoch: Epoch,
    pub emitted_nano: u128,
    pub burned_exit_nano: u128,
    pub burned_window_nano: u128,
    pub burned_abandoned_nano: u128,
    pub supply_nano: u128,
}

impl SupplyPublication {
    pub fn burned_total_nano(&self) -> u128 {
        self.burned_exit_nano + self.burned_window_nano + self.burned_abandoned_nano
    }
}

#[derive(Clone, Debug, Default)]
pub struct SupplyLedger {
    epoch: Epoch,
    emitted: u128,
    burns: Vec<BurnRecord>,
    per_category: BTreeMap<BurnCategory, u128>,
    seen_references: BTreeSet<[u8; 32]>,
}

impl SupplyLedger {
    pub fn new() -> SupplyLedger {
        SupplyLedger::default()
    }

    pub fn epoch(&self) -> Epoch {
        self.epoch
    }

    pub fn set_epoch(&mut self, epoch: Epoch) {
        self.epoch = epoch;
    }

    pub fn record_emission(&mut self, amount_nano: u64) {
        self.emitted += u128::from(amount_nano);
    }

    pub fn record_burn(&mut self, record: BurnRecord) -> Result<(), SupplyError> {
        if !self.seen_references.insert(record.reference) {
            return Err(SupplyError::DuplicateBurn { reference: record.reference });
        }
        let new_supply = self
            .supply_nano()
            .checked_sub(u128::from(record.amount_nano))
            .ok_or(SupplyError::NegativeSupply)?;
        let _ = new_supply;
        *self.per_category.entry(record.category).or_insert(0) += u128::from(record.amount_nano);
        self.burns.push(record);
        Ok(())
    }

    pub fn emitted_nano(&self) -> u128 {
        self.emitted
    }

    pub fn burned_nano(&self, category: BurnCategory) -> u128 {
        self.per_category.get(&category).copied().unwrap_or(0)
    }

    pub fn burns(&self) -> &[BurnRecord] {
        &self.burns
    }

    /// supply = emitted − burned (M1's identity; erratum 164).
    pub fn supply_nano(&self) -> u128 {
        self.emitted - self.per_category.values().sum::<u128>()
    }

    pub fn publish(&self) -> SupplyPublication {
        SupplyPublication {
            epoch: self.epoch,
            emitted_nano: self.emitted,
            burned_exit_nano: self.burned_nano(BurnCategory::TransparentExit),
            burned_window_nano: self.burned_nano(BurnCategory::UnclaimedWindow),
            burned_abandoned_nano: self.burned_nano(BurnCategory::AbandonedIssue),
            supply_nano: self.supply_nano(),
        }
    }

    /// The M1 cross-check: recorded emissions equal the schedule's
    /// time-driven replay plus event-driven grants.
    pub fn audit_emission(
        &self,
        expected_time_driven_nano: u128,
        expected_event_driven_nano: u128,
    ) -> Result<(), SupplyError> {
        let expected = expected_time_driven_nano + expected_event_driven_nano;
        if self.emitted != expected {
            return Err(SupplyError::EmissionMismatch { found: self.emitted, expected });
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn rec(category: BurnCategory, amount: u64, epoch: u64, seed: u8) -> BurnRecord {
        BurnRecord {
            category,
            amount_nano: amount,
            epoch: Epoch::from_u64(epoch),
            reference: [seed; 32],
        }
    }

    #[test]
    fn identity_and_categories() {
        let mut l = SupplyLedger::new();
        l.set_epoch(Epoch::from_u64(10));
        assert_eq!(l.supply_nano(), 0);
        l.record_emission(1_000);
        assert_eq!(l.supply_nano(), 1_000);
        l.record_burn(rec(BurnCategory::TransparentExit, 100, 10, 1)).unwrap();
        l.record_burn(rec(BurnCategory::UnclaimedWindow, 200, 10, 2)).unwrap();
        l.record_burn(rec(BurnCategory::AbandonedIssue, 50, 10, 3)).unwrap();
        assert_eq!(l.supply_nano(), 650);
        let p = l.publish();
        assert_eq!(p.epoch, Epoch::from_u64(10));
        assert_eq!(p.emitted_nano, 1_000);
        assert_eq!(p.burned_exit_nano, 100);
        assert_eq!(p.burned_window_nano, 200);
        assert_eq!(p.burned_abandoned_nano, 50);
        assert_eq!(p.burned_total_nano(), 350);
        assert_eq!(p.supply_nano, 650);
        assert_eq!(BurnCategory::TransparentExit.name(), "transparent-exit");
    }

    #[test]
    fn replay_dedup_and_negative_rejected() {
        let mut l = SupplyLedger::new();
        l.record_emission(100);
        l.record_burn(rec(BurnCategory::TransparentExit, 100, 1, 7)).unwrap();
        assert!(matches!(
            l.record_burn(rec(BurnCategory::TransparentExit, 1, 1, 7)),
            Err(SupplyError::DuplicateBurn { reference }) if reference == [7u8; 32]
        ));
        // Same reference, different category: still a replay.
        assert!(matches!(
            l.record_burn(rec(BurnCategory::AbandonedIssue, 1, 1, 7)),
            Err(SupplyError::DuplicateBurn { .. })
        ));
        assert_eq!(l.burns().len(), 1);
        // Negative supply is an error at record time.
        assert!(matches!(
            l.record_burn(rec(BurnCategory::TransparentExit, 1, 1, 8)),
            Err(SupplyError::NegativeSupply)
        ));
        // After more emission, the same burn succeeds.
        l.record_emission(50);
        l.record_burn(rec(BurnCategory::TransparentExit, 1, 1, 8)).unwrap();
        assert_eq!(l.supply_nano(), 49);
    }

    #[test]
    fn audit_and_category_isolation() {
        let mut l = SupplyLedger::new();
        l.record_emission(900);
        l.audit_emission(700, 200).unwrap();
        assert!(matches!(
            l.audit_emission(700, 201),
            Err(SupplyError::EmissionMismatch { found: 900, expected: 901 })
        ));
        // Categories accumulate independently.
        l.record_burn(rec(BurnCategory::UnclaimedWindow, 10, 1, 1)).unwrap();
        l.record_burn(rec(BurnCategory::UnclaimedWindow, 20, 2, 2)).unwrap();
        assert_eq!(l.burned_nano(BurnCategory::UnclaimedWindow), 30);
        assert_eq!(l.burned_nano(BurnCategory::TransparentExit), 0);
        assert_eq!(l.burned_nano(BurnCategory::AbandonedIssue), 0);
    }
}

