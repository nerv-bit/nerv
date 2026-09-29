//! # nerv-wallet-core — the platform-agnostic wallet state machine
//!
//! Every UI (terminal, desktop, web, mobile) renders the same
//! `WalletState` and dispatches the same `WalletAction` through the
//! same pure `update` function. No platform types, no async, no I/O
//! — the state machine is the single place where wallet business
//! rules are enforced.
//!
//! ```text
//! UI (any platform) → WalletAction → update() → WalletState' + events
//!                                        ↑
//!                              nerv-wallet (domain)
//! ```

#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic, clippy::float_cmp)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod action;
pub mod claim_submit;
pub mod producer_submit;
pub mod state;
pub mod storage;
pub mod theme;
pub mod update;


pub use action::WalletAction;
pub use claim_submit::{build_claim_parts, submit_claim_leg, ClaimSubmitError, ClaimSubmitResult};
pub use producer_submit::{
    register_producer_stake, ProducerSubmitError, ProducerSubmitResult,
};
pub use state::{
    ClaimBucket, ClaimDraft, ClaimError, ClaimStage, ClaimValidation, ProducerDraft, ProducerError,
    ProducerState, ProducerValidation, Screen, SyncStatus, TransactionDraft, WalletState,
};
pub use theme::Theme;
pub use storage::{
    EncryptedSeed, FileStorage, MemoryStorage, StorageError, WalletStorage,
};
pub use update::update;


