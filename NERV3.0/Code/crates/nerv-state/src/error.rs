//! nerv-state error taxonomy.
//!
//! - `StateError`     — T_τ-tree local invariants (rule 1).
//! - `BlockError`     — block body / wire-format failures (errata 105–108).
//! - `ExecutorError`  — rule 1–7 validation, state-application, and root
//!                      reconciliation errors (errata 107–114). The single
//!                      cross-cutting type the executor, fraud prover, and
//!                      `propose` path surface.
//! - `FraudError`     — fraud-proof verification failures (erratum 116).
//! - `StoreError`     — node-store failures (DSR-10; erratum 117).
//! - `RebuildError`   — rebuild-from-DA failures (erratum 117).

use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::types::{LegKey, ShardId, TxId};
use nerv_custody::CustodyError;
use nerv_seal::SealError;


#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum StateError {
    #[error("T_τ leaves must be strictly ascending: {txid} does not follow {prev}")]
    TauUnsorted { txid: TxId, prev: TxId },
    #[error("T_τ tree is full ({capacity} leaves)")]
    TauFull { capacity: u64 },
    #[error("T_τ witness index {index} outside the {count}-leaf tree")]
    TauIndex { index: u64, count: u64 },
}


/// Block-body construction and resolution failures (WP §4.3, §11.2;
/// errata 105–108). Surfaced by `apply_block` / `propose` only as
/// `ExecutorError::Resolve(BlockError)`; this enum is the inner type.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum BlockError {
    #[error("block legs must be strictly ascending by (txid, leg): {key:?} does not follow {prev:?}")]
    LegUnsorted { key: LegKey, prev: LegKey },
    #[error("block has {count} legs, exceeding the {max}-leg cap")]
    LegCapExceeded { count: usize, max: usize },
    #[error("leg tree is full ({capacity} leaves)")]
    LegTreeFull { capacity: u64 },
    #[error("leg-tree witness index {index} outside the {count}-leaf tree")]
    LegTreeIndex { index: u64, count: u64 },
    #[error("settled leg {leg}: shell index {index} outside the {len}-leg canonical shell")]
    LegIndex { leg: usize, index: usize, len: usize },
    #[error("settled leg {leg}: shell shard {found} does not match the block's {expected}")]
    WrongShard { leg: usize, found: ShardId, expected: ShardId },
    #[error("settled leg {leg}: malformed shell: {source}")]
    BadShell { leg: usize, source: CustodyError },
    #[error("settled leg {leg}: malformed seal ciphertext: {source}")]
    BadCt { leg: usize, source: SealError },
    #[error("header fee total {found} does not match the legs' {expected}")]
    FeeTotalMismatch { expected: u64, found: u64 },
    #[error("declared leg fees overflow the fee total")]
    FeeOverflow,
    #[error("leg {leg}: declared fee {fee_nano} is below the D.4 admission floor {floor_nano}")]
    FeeBelowFloor { leg: usize, fee_nano: u64, floor_nano: u64 },
    #[error("header ct_batch_hash does not match H(ct_B) of the settled legs")]
    CtBatchHashMismatch,
    #[error("recomputed nct_root does not match the header")]
    NctRootMismatch,
    #[error("recomputed nullifier_root does not match the header")]
    NullifierRootMismatch,
    #[error("recomputed transit_root does not match the header")]
    TransitRootMismatch,
    #[error("leg {leg}: conditional output on an input-bearing leg (D.3)")]
    ShellConditionalOnSpendLeg { leg: usize },
    #[error("leg {leg}: shell with conditional outputs has {spend_legs} input-bearing legs, not 1")]
    ShellNotSingleSpend { leg: usize, spend_legs: usize },
    #[error("leg {leg}: shell with conditional outputs has unequal leg expiries")]
    ShellExpiryMismatch { leg: usize },
    #[error("leg {leg}: expiry {expiry} outside [{lo}, {hi}] for the including height")]
    ExpiryBounds { leg: usize, expiry: u64, lo: u64, hi: u64 },
    #[error("leg {leg}: T_τ membership witness does not verify")]
    TauWitness { leg: usize },
    #[error("leg {leg}: anchor is not one of the last 64 finalized NCT roots")]
    AnchorStale { leg: usize },
    #[error("leg {leg}: nullifier {nf} is already spent or an in-block duplicate")]
    NullifierSpent { leg: usize, nf: Hash256 },
    #[error("leg {leg}: transit entry already exists (replay)")]
    TransitSpent { leg: usize },
    #[error("leg {leg}: settlement deadline — expiry {expiry} below the receiving height {height}")]
    SettlementDeadline { leg: usize, expiry: u64, height: u64 },
    #[error("leg {leg}: {found} sibling evidences for {expected} spend legs")]
    SiblingCount { leg: usize, expected: usize, found: usize },
    #[error("leg {leg}: sibling evidence {sibling} does not verify")]
    SiblingEvidence { leg: usize, sibling: usize },
    #[error("record {record}: escrow entry not found")]
    RecordEntryMissing { record: usize },
    #[error("record {record}: escrow entry is {state}, not Pending")]
    RecordEntryNotPending { record: usize, state: &'static str },
    #[error("record {record}: escrow already consumed by an earlier record in this block")]
    RecordDoubleConsume { record: usize },
    #[error("record {record}: escrow target is born in this block")]
    RecordTargetBornThisBlock { record: usize },
    #[error("record {record}: escrow shell unavailable from the chain")]
    RecordShellUnavailable { record: usize },
    #[error("record {record}: recovered shell does not hash to the escrow txid")]
    RecordShellMismatch { record: usize },
    #[error("record {record}: reversion before the grace boundary (expiry {expiry}, height {height})")]
    ReversionNotDue { record: usize, expiry: u64, height: u64 },
    #[error("record {record}: {found} evidence entries for {expected} issue legs")]
    ReversionEvidenceCount { record: usize, expected: usize, found: usize },
    #[error("record {record}: settled-evidence for issue leg {issue} does not verify")]
    ReversionEvidence { record: usize, issue: usize },
    #[error("record {record}: issue leg {issue}'s expiry-height transit root is not beacon-finalized")]
    ReversionUnsettledRootUnavailable { record: usize, issue: usize },
    #[error("record {record}: unsettled-evidence for issue leg {issue} does not verify")]
    ReversionBadUnsettled { record: usize, issue: usize },
    #[error("record {record}: {found} evidence entries for {expected} issue legs")]
    ClaimEvidenceCount { record: usize, expected: usize, found: usize },
    #[error("record {record}: evidence for issue leg {issue} does not verify")]
    ClaimEvidence { record: usize, issue: usize },
    #[error("omitted due reversion: escrow {txid} is due and triggerable but unresolved")]
    OmittedDueReversion { txid: TxId },
    #[error("executor internal invariant violated: {0}")]
    Internal(&'static str),
}


/// Rule 1–7 validation, state-application, and root-reconciliation errors
/// (WP §4.3, §11.2; errata 107–114). The single surface for the executor,
/// fraud prover, and `propose` path. `Resolve(BlockError)` is the bridge
/// from the block-body constructor (above) into the executor's pipeline.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ExecutorError {
    #[error("block's shard {block} does not match the state shard {state}")]
    WrongShard { block: ShardId, state: ShardId },
    #[error("header prev {found:?} does not match the state's commitment {expected:?}")]
    PrevMismatch { expected: Hash256, found: Hash256 },
    #[error("header height {found} does not match the expected next height {expected}")]
    HeightMismatch { expected: u64, found: u64 },
    #[error("header params_root {found:?} does not match the state's {expected:?}")]
    ParamsMismatch { expected: Hash256, found: Hash256 },
    #[error("registry interval {interval} is not beacon-finalized")]
    RegistryNotFinalized { interval: u64 },
    #[error("registry interval {interval}: T_τ root {found:?} does not match the view's {expected:?}")]
    RegistryRootMismatch { interval: u64, expected: Hash256, found: Hash256 },
    #[error("header qc_hash does not match the block's QC certificate")]
    QcHashMismatch,
    #[error("leg {leg}: T_τ membership witness does not verify")]
    TauWitness { leg: usize },
    #[error("leg {leg}: declared fee {fee_nano} is below the D.4 admission floor {floor_nano}")]
    FeeBelowFloor { leg: usize, fee_nano: u64, floor_nano: u64 },
    #[error("leg {leg}: anchor is not one of the last 64 finalized NCT roots")]
    AnchorStale { leg: usize },
    #[error("leg {leg}: nullifier {nf} is already spent or an in-block duplicate")]
    NullifierSpent { leg: usize, nf: Hash256 },
    #[error("leg {leg}: transit entry already exists (replay)")]
    TransitSpent { leg: usize },
    #[error("leg {leg}: settlement deadline — expiry {expiry} below the receiving height {height}")]
    SettlementDeadline { leg: usize, expiry: u64, height: u64 },
    #[error("leg {leg}: {found} sibling evidences for {expected} spend legs")]
    SiblingCount { leg: usize, expected: usize, found: usize },
    #[error("leg {leg}: sibling evidence {sibling} does not verify")]
    SiblingEvidence { leg: usize, sibling: usize },
    #[error("declared leg fees overflow the fee total")]
    FeeOverflow,
    #[error("header ct_batch_hash does not match H(ct_B) of the settled legs")]
    CtBatchHashMismatch,
    #[error("header fee total does not match the legs' total")]
    FeeTotalMismatch,
    #[error("leg {leg}: conditional output on an input-bearing leg (D.3)")]
    ShellConditionalOnSpendLeg { leg: usize },
    #[error("leg {leg}: shell with conditional outputs has {spend_legs} input-bearing legs, not 1")]
    ShellNotSingleSpend { leg: usize, spend_legs: usize },
    #[error("leg {leg}: shell with conditional outputs has unequal leg expiries")]
    ShellExpiryMismatch { leg: usize },
    #[error("leg {leg}: expiry {expiry} outside [{lo}, {hi}] for the including height")]
    ExpiryBounds { leg: usize, expiry: u64, lo: u64, hi: u64 },
    #[error("recomputed nct_root does not match the header")]
    NctRootMismatch,
    #[error("recomputed nullifier_root does not match the header")]
    NullifierRootMismatch,
    #[error("recomputed transit_root does not match the header")]
    TransitRootMismatch,
    #[error("record {record}: escrow entry not found")]
    RecordEntryMissing { record: usize },
    #[error("record {record}: escrow entry is {state}, not Pending")]
    RecordEntryNotPending { record: usize, state: &'static str },
    #[error("record {record}: escrow already consumed by an earlier record in this block")]
    RecordDoubleConsume { record: usize },
    #[error("record {record}: escrow target is born in this block")]
    RecordTargetBornThisBlock { record: usize },
    #[error("record {record}: escrow shell unavailable from the chain")]
    RecordShellUnavailable { record: usize },
    #[error("record {record}: recovered shell does not hash to the escrow txid")]
    RecordShellMismatch { record: usize },
    #[error("record {record}: reversion before the grace boundary (expiry {expiry}, height {height})")]
    ReversionNotDue { record: usize, expiry: u64, height: u64 },
    #[error("record {record}: {found} evidence entries for {expected} issue legs")]
    ReversionEvidenceCount { record: usize, expected: usize, found: usize },
    #[error("record {record}: settled-evidence for issue leg {issue} does not verify")]
    ReversionEvidence { record: usize, issue: usize },
    #[error("record {record}: issue leg {issue}'s expiry-height transit root is not beacon-finalized")]
    ReversionUnsettledRootUnavailable { record: usize, issue: usize },
    #[error("record {record}: unsettled-evidence for issue leg {issue} does not verify")]
    ReversionBadUnsettled { record: usize, issue: usize },
    #[error("record {record}: {found} evidence entries for {expected} issue legs")]
    ClaimEvidenceCount { record: usize, expected: usize, found: usize },
    #[error("record {record}: evidence for issue leg {issue} does not verify")]
    ClaimEvidence { record: usize, issue: usize },
    #[error("omitted due reversion: escrow {txid} is due and triggerable but unresolved")]
    OmittedDueReversion { txid: TxId },
    #[error("executor internal invariant violated: {0}")]
    Internal(&'static str),
    /// A block-body construction / resolution failure raised from
    /// `resolve_legs`, `ct_sum`, `BlockLegTree::from_resolved`, or any other
    /// helper that returns `BlockError`. The `apply_block` pipeline lifts it
    /// into this enum via the `?` operator.
    #[error("block construction failed: {0}")]
    Resolve(#[from] BlockError),
}


/// Fraud-evidence verification failures (WP §4.3, §11.4; erratum 116).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum FraudError {
    #[error("claimed beacon facts do not match the view")]
    FactsRejected,
    #[error("the block is not invalid in the claimed way")]
    ConditionNotExhibited,
    #[error("malformed fraud claim: {0}")]
    Malformed(&'static str),
}


/// Node-store failures (DSR-10; erratum 117).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum StoreError {
    #[error("store backend: {0}")]
    Backend(String),
    #[error("decode of {what} failed: {source}")]
    Decode { what: &'static str, source: CodecError },
    #[error("store schema version {found}, expected {expected}")]
    Version { found: u32, expected: u32 },
    #[error("height u64::MAX is reserved for the tip marker")]
    ReservedHeight,
    #[error("store lock poisoned")]
    LockPoisoned,
}


/// Rebuild-from-DA failures (erratum 117).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum RebuildError {
    #[error(transparent)]
    Executor(#[from] ExecutorError),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("archive gap: block {height} missing below the tip")]
    Gap { height: u64 },
}