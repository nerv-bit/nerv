
pub use crate::claim::ClaimError;
pub use crate::emission::{AccountId, LedgerError};
// ScheduleError is defined here directly because `schedule.rs` is
// infrastructure (parser + verification) and does not own the public error
// type — keeping it here means downstream `pub use crate::error::*` covers
// every variant without an extra re-export chain.


#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ScheduleError {
    #[error("bucket `{bucket}`: unknown kind `{kind}`")]
    UnknownKind { bucket: &'static str, kind: &'static str },
    #[error("bucket `{bucket}`: unknown account `{account}`")]
    UnknownAccount { bucket: &'static str, account: &'static str },
    #[error("bucket `{bucket}`: missing required field `{field}`")]
    MissingField { bucket: &'static str, field: &'static str },
    #[error("bucket `{bucket}`: {detail}")]
    Inconsistent { bucket: &'static str, detail: &'static str },
    #[error("bucket `{bucket}`: geometric parameters overflow")]
    Overflow { bucket: &'static str },
    #[error("duplicate bucket name `{name}`")]
    DuplicateBucket { name: &'static str },
    #[error("share permilles sum to {found}, expected 1000")]
    ShareSum { found: u64 },
    #[error("bucket totals sum to {found} NERV, expected {expected}")]
    SupplySum { found: u128, expected: u64 },
}
