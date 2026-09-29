//! # nerv-economy — emission, fees, staking (WP §12)
//!
//! * `schedule`       — the genesis-committed emission schedule: the pure
//!   (day, params) → NERV function theorem M1 audits against.
//! * `emission`       — the emission ledger: signed accounts,
//!   commitment-notes, beacon-credential credits bounded per bucket-total,
//!   `emission_root`.
//! * `claim`          — the claim rail: EMISSION-tree witnesses, claim
//!   nullifiers, claim-leg validation.
//! * `fees`           — the exact four-way split and the D.4 admission
//!   floor (reveal-driven, bounded, self-reverting).
//! * `subsidy`        — the per-epoch validator-subsidy shard split.
//! * `staking`        — the transparent stake ledger and slash table.
//! * `supply_ledger`  — the M1 supply identity and per-epoch publication.
//!
//! No floating point (P5).

#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic, clippy::float_cmp)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod claim;
pub mod emission;
pub mod error;
pub mod fees;
pub mod schedule;
pub mod staking;
pub mod subsidy;
pub mod supply_ledger;

#[cfg(test)]
mod testutil;

pub use claim::{
    verify_claim_leg, ClaimError, ClaimLegParts, ClaimWitness, EmissionTree, EMISSION_TREE_DEPTH,
};
pub use emission::{
    claim_nullifier, derive_claim_key, derive_lek, eligibility_digest, note_commitment,
    signed_account_id, AccountEntry, AccountHolder, AccountId, BurnEvent, EmissionCredential,
    EmissionLedger, LedgerError, NoteAccount, SignedAccount,
};
pub use error::ScheduleError;
pub use fees::{
    reveal_statistic, AdmissionFloor, FeeSplit, BASE_FLOOR_NANO, M_MAX, TRIGGER_MULT, WINDOW,
};
pub use schedule::{
    AccountKind, BucketSchedule, Curve, EmissionSchedule, DAY_SECS, NANO_PER_NERV, SUPPLY_NERV,
};
pub use staking::{SlashClass, SlashRecord, StakeError, StakeLedger};
pub use subsidy::{epoch_subsidy_nano, split_subsidy, SubsidyError, SUBSIDY_BUCKET};
pub use supply_ledger::{
    BurnCategory, BurnRecord, SupplyError, SupplyLedger, SupplyPublication,
};
