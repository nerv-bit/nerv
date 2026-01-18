// src/lib.rs
// NERV Wallet - Private, Post-Quantum Light Wallet
// Complete implementation of all Epics (1-8) from the Wallet Protocol Specification
// Version: Comprehensive Production Edition

pub mod keys;
pub mod balance;
pub mod tx;
pub mod sync;
pub mod vdw;
pub mod history;
pub mod backup;
pub mod sdk;
pub mod ui;
pub mod types;
pub mod utils;
pub mod error;

// Re-exports for easy access
pub use keys::{HdWallet, Account, AccountKeys};
pub use balance::BalanceTracker;
pub use tx::{TransactionBuilder, PrivateTransaction};
pub use sync::LightClient;
pub use vdw::{VDWManager, VerifiedProof, VDW};
pub use history::{HistoryManager, TransactionHistoryEntry};
pub use backup::{BackupManager, SelectiveProof, EncryptedBackup};
pub use sdk::NervWallet;
pub use types::*;
pub use error::WalletError;

// Feature flags for platform-specific optimizations
#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "uniffi")]
pub mod ffi;