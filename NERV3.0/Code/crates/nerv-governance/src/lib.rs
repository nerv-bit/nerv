//! # nerv-governance — the two chambers (WP §12.8, App C)
//!
//! * `chambers`  — the note-holder (shielded) and validator (transparent)
//!   chambers; the bootstrap rule.
//! * `ballot`    — the shielded ballot: voting nullifier, Ajtai weight
//!   commitment, sigma proof; the transparent validator vote.
//! * `tally`     — the aggregate-only tally: two-phase (vote → reveal).
//! * `referenda` — the lifecycle: draft → active → closed; the three
//!   tiers (parameter, W-epoch, constitutional); the two-epoch rule.
//! * `emergency` — the enumerated, time-boxed powers: kill-switch,
//!   hash widening, accelerated split (§C.2).


#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic, clippy::float_cmp)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]


pub mod ballot;
pub mod chambers;
pub mod emergency;
pub mod error;
pub mod referenda;
pub mod tally;


pub use ballot::{
    ballot_matrix, commit_weight, derive_voting_nullifier, open_weight, prove_weight,
    verify_weight, Choice, ShieldedBallot, ValidatorVote, WeightOpening,
    BLINDING_BOUND, LIMB_BITS, LIMB_BOUND, LIMBS,
};
pub use chambers::{BootstrapPhase, Chamber, ChamberTally, ChamberVote};
pub use emergency::{
    EmergencyDeclaration, EmergencyKind, EmergencyLedger, EmergencyState,
    KILL_SWITCH_EPOCHS, ACCELERATED_SPLIT_EPOCHS, HASH_WIDENING_EPOCHS,
};
pub use error::EmergencyError;
pub use error::{BallotError, ChamberError, ReferendumError, TallyError};
pub use referenda::{
    ConstitutionalPair, Referendum, ReferendumId, ReferendumSubject, Tier, WEpochEvidence,
};
pub use tally::{TallyResult, PARTICIPATION_FLOOR_PERMILLE};
