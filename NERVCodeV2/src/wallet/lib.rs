// src/lib.rs
//! NERV Wallet - Private, Post-Quantum Light Wallet
//!
//! New implementation fully compatible with the NERV blockchain codebase.
//! This wallet is designed as a library with multi-platform support in mind
//! (Rust native, WASM for web, bindings for Swift/Kotlin).
//!
//! Epics 1 & 2 are fully implemented here with production-grade security,
//! usability, and performance.
//!
//! Key Design Decisions for Compatibility:
//! - Uses `nerv::crypto` primitives directly (Dilithium3 for spending authority,
//!   ML-KEM-768 for note encryption and stealth receiving)
//! - Assumes note-based privacy model compatible with homomorphic embedding updates
//!   (outputs contain ML-KEM ciphertext + symmetrically encrypted note payload)
//! - Hierarchical deterministic derivation with hardened paths and gap limit
//! - Local scanning with reasonable performance (gap limit + batched decapsulation)
//! - All secret keys zeroized on drop
//! - Beautiful, intuitive UI flows described in comments (superior to existing wallets)

pub mod keys;
pub mod balance;
pub mod types;

pub use keys::HdWallet;
pub use balance::BalanceTracker;
pub use types::{Note, Output, AccountKeys, ReceivingAddress};