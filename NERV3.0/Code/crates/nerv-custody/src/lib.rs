//! # nerv-custody — the cryptographic ledger of record (WP §3)
//!
//! Depends on nerv-core and nerv-crypto — nothing else (DSR policy).
//! Notes and commitments (§3.2–3.3), the key hierarchy and diversified
//! delivery keys, note sealing and trial-decryption, and the Poseidon2
//! note commitment tree (E-002's dual-hash assignment).

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod address;
pub mod burn;
pub mod commitment;
pub mod error;
pub mod nct;
pub mod note;
pub mod nullifier;
pub mod poseidon2;
pub mod transit;
pub mod tx;

#[cfg(test)]
mod testutil;

pub use address::{Address, DeliveryKeyPair, DetectionSeed, MasterSeed, WalletKeys};
pub use burn::BurnCommitment;
pub use commitment::{note_commitment, NoteOpening, BLINDING_LEN, DELIVERY_KEY_LEN, NONCE_LEN};
pub use error::{CustodyError, NoteError};
pub use note::{seal_note, trial_decrypt, Note, NotePlaintext, SealedNote};
pub use nct::{
    empty_digests, leaf_digest, node_digest, verify_witness, NctDigest, NctWitness,
    NoteCommitmentTree, DEPTH as NCT_DEPTH, LEAF_CAPACITY,
};
pub use nullifier::{derive_nullifier, NullifierProof, NullifierSet};
pub use poseidon2::{compress, domain_iv, permute};
pub use transit::{transit_key, TransitEntry, TransitEntryState, TransitLog, TransitProof};
pub use tx::{InputSet, LegShell, Output, TransactionShell};

