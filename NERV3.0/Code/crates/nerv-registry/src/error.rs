use nerv_core::hash::Hash256;
use nerv_core::types::TxId;
use nerv_crypto::{CryptoError, QcError};
use nerv_custody::CustodyError;
use nerv_proofs::{DedupError, FsError, TxError};
use nerv_state::StateError;


/// The proof-verification gate's failures (WP §5.5; erratum 119).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum VerificationError {
    #[error(transparent)]
    Shell(#[from] CustodyError),
    #[error(transparent)]
    Fs(#[from] FsError),
    #[error(transparent)]
    Tx(#[from] TxError),
    #[error("proof rejected: verification failed against the anchored public inputs")]
    ProofRejected,
    #[error("shell weight version {shell} does not match the codec's {codec}")]
    WeightVersion { shell: u64, codec: u64 },
}


#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MempoolError {
    #[error(transparent)]
    Verification(#[from] VerificationError),
    #[error("mempool is full ({count}/{capacity} entries)")]
    Full { count: usize, capacity: usize },
}


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum BundleError {
    #[error(transparent)]
    Shell(#[from] CustodyError),
    #[error(transparent)]
    Tau(#[from] nerv_state::StateError),
    #[error(transparent)]
    Crypto(#[from] CryptoError),
    #[error("transaction {index} failed verification: {source}")]
    Verification { index: usize, source: VerificationError },
    #[error("bundle is empty")]
    Empty,
    #[error("bundle carries {count} transactions, exceeding the {max} maximum")]
    TooLarge { count: usize, max: usize },
    #[error("attestation counts {count} txids for {transactions} transactions")]
    CountMismatch { count: usize, transactions: usize },
    #[error("txid {txid} appears twice in the bundle — malformed (the commitment is over a set)")]
    DuplicateTxid { txid: TxId },
    #[error("transaction at index {index} does not hash to its declared txid {txid}")]
    TxidMismatch { index: usize, txid: TxId },
    #[error("attestation root {found} does not match the recomputed {expected}")]
    RootMismatch { expected: Hash256, found: Hash256 },
    #[error("aggregator attestation signature failed verification")]
    BadSignature,
}


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum IntervalError {
    #[error(transparent)]
    Bundle(#[from] BundleError),
    #[error(transparent)]
    Dedup(#[from] DedupError),
    #[error(transparent)]
    Tau(#[from] StateError),
    #[error("txid {txid} is not in the interval set")]
    TxidAbsent { txid: TxId },
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DegradedError {
    #[error(transparent)]
    Qc(#[from] QcError),
    #[error(transparent)]
    Crypto(#[from] CryptoError),
    #[error("QC subject does not match the interval commit digest")]
    SubjectMismatch,
}


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ChallengeError {
    #[error("witness interval {found} does not match the challenged {expected}")]
    IntervalMismatch { expected: u64, found: u64 },
    #[error(transparent)]
    Shell(#[from] CustodyError),
    #[error("T_τ witness does not verify against the attested root")]
    WitnessRejected,
    #[error("challenge context mismatch: shell weight version {shell}, codec {codec}")]
    ContextVersion { shell: u64, codec: u64 },
}

