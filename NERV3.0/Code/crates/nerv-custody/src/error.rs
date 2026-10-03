//! Error taxonomy for nerv-custody.


use nerv_core::hash::Hash256;
use nerv_core::types::{Height, ShardId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CustodyError {
    #[error("note value {value} outside [{min}, {max}] nano-NERV (§3.2)")]
    ValueOutOfRange { value: u64, min: u64, max: u64 },
    #[error("digest word {value} is not below the Goldilocks prime")]
    DigestWordOutOfRange { value: u64 },
    #[error("note commitment tree is full ({capacity} leaves)")]
    TreeFull { capacity: u64 },
    #[error("leaf index {index} outside the tree ({count} leaves)")]
    LeafIndexOutOfRange { index: u64, count: u64 },
    #[error("tree structure invalid: {0}")]
    InvalidTree(&'static str),
    #[error("shard tag is not a prefix of the address homing key (forged or corrupt address)")]
    InvalidShardTag,
    #[error("address derivation failed: {0}")]
    Derivation(&'static str),
    #[error("memo length {len} exceeds the {max}-byte maximum")]
    MemoTooLarge { len: usize, max: usize },
    #[error("nullifier {nf} is already spent — double-spend (§3.4)")]
    NullifierAlreadySpent { nf: Hash256 },
    #[error("duplicate nullifier {nf} within the batch (in-block double-spend, §3.4)")]
    DuplicateNullifier { nf: Hash256 },
    #[error("nullifier {nf} is not in the spent set")]
    NullifierNotSpent { nf: Hash256 },
    #[error("nullifier set failed its consistency audit: {0}")]
    InvalidNullifierSet(&'static str),
    #[error("transit entry already exists: {key}")]
    TransitEntryExists { key: Hash256 },
    #[error("transit log consistency audit failed: {0}")]
    InvalidTransitLog(&'static str),
    #[error("no transit entry at key {key}")]
    TransitAbsent { key: Hash256 },
    #[error("reversion at {current} is before expiry + L_grace ({earliest}) — wrongful early reversion is fraud-provable (D.3)")]
    ReversionTooEarly { current: Height, earliest: Height },
  #[error("transit entry is {state}, not Pending — consumed once, by either path (D.3)")]
    TransitEntryConsumed { state: &'static str },
  #[error("leg has no inputs and no outputs")]
    EmptyLeg,
    #[error("{found} inputs exceed the leg maximum {max}")]
    TooManyInputs { found: usize, max: usize },
    #[error("{found} outputs exceed the leg maximum {max}")]
    TooManyOutputs { found: usize, max: usize },
    #[error("duplicate nullifier within a leg")]
    DuplicateNullifierInLeg,
    #[error("conditional cross-shard output {index} is missing its revert pairing (D.3)")]
    MissingRevertOutput { index: usize },
    #[error("unconditional output {index} declares a revert pairing (D.3)")]
    UnexpectedRevertOutput { index: usize },
    #[error("output {index} is invalid")]
    InvalidOutput { index: usize },
    #[error("transaction has no legs")]
    NoLegs,
    #[error("{found} legs exceed the maximum {max}")]
    TooManyLegs { found: usize, max: usize },
    #[error("duplicate leg for shard {shard}")]
    DuplicateShardLeg { shard: ShardId },
    #[error("declared fees overflow")]
    FeeOverflow,
    #[error("expiry {expiry} outside [{lo}, {hi}] for inclusion height {include} (D.3)")]
    ExpiryOutOfBounds { expiry: Height, include: Height, lo: u64, hi: u64 },
    #[error("leg sealed-delta ciphertext {len} B exceeds the {max}-B ceiling")]
    CtTooLarge { len: usize, max: usize },
    #[error("{found} burn commitments exceed the leg maximum {max}")]
    TooManyBurns { found: usize, max: usize },
    #[error("crypto: {0}")]
    Crypto(#[from] nerv_crypto::CryptoError),
}

/// Sealing / trial-decryption failures (WP §3.2 note encryption).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NoteError {
    #[error(transparent)]
    Custody(#[from] CustodyError),
    #[error(transparent)]
    Crypto(#[from] nerv_crypto::CryptoError),
    #[error("sealed-note authentication failed (wrong delivery key or tampered data)")]
    DecryptionFailed,
    #[error("sealed-note plaintext is not canonical")]
    MalformedPlaintext,
    #[error("recovered note does not match the published commitment")]
    CommitmentMismatch,
}
