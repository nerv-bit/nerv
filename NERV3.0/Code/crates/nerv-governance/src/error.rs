
use crate::emergency::EmergencyKind;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum BallotError {
    #[error("voting nullifier {nf:?} is already cast — double vote")]
    DoubleVote { nf: nerv_core::hash::Hash256 },
    #[error("weight {weight} exceeds the encodable maximum {max}")]
    WeightTooLarge { weight: u64, max: u64 },
    #[error("weight proof failed")]
    ProofFailed,
    #[error("weight opening failed — the commitment does not open to the claimed limbs")]
    OpeningFailed,
    #[error("limb {index} value {value} exceeds the bound {bound}")]
    LimbOutOfRange { index: usize, value: u64, bound: u64 },
    #[error("crypto: {0}")]
    Crypto(#[from] nerv_crypto::CryptoError),
    #[error("seal: {0}")]
    Seal(#[from] nerv_seal::SealError),
    /// The weight σ-proof failed to generate: the seal-side FS-with-aborts
    /// engine rejected the witness. Maps every `nerv_seal::SigmaError`
    /// variant (P-curve, ZK, sampling, …) to a single ballot-level
    /// outcome — the ballot is malformed and must be discarded.
    #[error("weight σ-proof failed: {0}")]
    Sigma(#[from] nerv_seal::SigmaError),
}


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ChamberError {
    #[error("the validator {vk:?} has already voted")]
    DoubleVote { vk: nerv_crypto::mldsa::VerifyingKey },
    #[error("the validator {vk:?} has no stake")]
    NoStake { vk: nerv_crypto::mldsa::VerifyingKey },
    #[error("vote signature failed")]
    BadSignature,
}


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TallyError {
    #[error("ballot {index}: {source}")]
    Ballot { index: usize, source: BallotError },
    #[error("opening {index}: {source}")]
    Opening { index: usize, source: BallotError },
    #[error("vote {index}: {source}")]
    Vote { index: usize, source: ChamberError },
    #[error("participation {cast} below the floor {required}")]
    BelowFloor { cast: u128, required: u128 },
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ReferendumError {
    #[error("the referendum is not in the active phase (currently {state})")]
    NotActive { state: &'static str },
    #[error("the W-epoch gate requires complete machine-check evidence")]
    WEpochGateIncomplete,
    #[error("the constitutional confirmation must be for epoch {expected}, found {found}")]
    ConfirmationEpoch { expected: u64, found: u64 },
    #[error("the constitutional confirmation's subject {found} does not match {expected}")]
    ConfirmationSubject { expected: nerv_core::hash::Hash256, found: nerv_core::hash::Hash256 },
}


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum EmergencyError {
    #[error("emergency power {kind} is already active")]
    AlreadyActive { kind: EmergencyKind },
    #[error("emergency power {kind} has expired")]
    Expired { kind: EmergencyKind },
    #[error("hash widening from {from} to {to} bits is not the 256→384 upgrade")]
    BadWidening { from: u16, to: u16 },
    #[error("kill-switch renewal at epoch {at} is not after the current expiry {expiry}")]
    BadRenewal { at: u64, expiry: u64 },
}


