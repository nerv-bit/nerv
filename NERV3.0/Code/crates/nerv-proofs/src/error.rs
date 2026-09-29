//! Error taxonomy for nerv-proofs. `FsError` is the Fiat–Shamir layer's
//! surface; WitnessError is for custody-witness construction and wallet-side binding;
//! statement/AIR and folding variants land with their parts.

use nerv_custody::error::CustodyError;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum FsError {
    #[error(transparent)]
    Shell(#[from] CustodyError),
    #[error("anchor count {found} differs from the shell's {expected} input-bearing legs")]
    AnchorCount { expected: usize, found: usize },
    #[error("anchor {index}: (shard, root) pair differs from the shell's input leg")]
    AnchorMismatch { index: usize },
    #[error("public expiry {found} is not the shell's earliest leg expiry {expected}")]
    ExpiryMismatch { expected: u64, found: u64 },
    #[error("leg declares encoder version {leg}; public inputs declare {tx}")]
    WeightVersionMismatch { leg: u64, tx: u64 },
    #[error("public shell digest does not match the canonical shell")]
    ShellDigestMismatch,
}

/// Custody-witness construction and wallet-side binding failures.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum WitnessError {
    #[error(transparent)]
    Shell(#[from] CustodyError),
    #[error("witness has no inputs — the custody AIR covers spend transactions (claim legs are a later statement set)")]
    NoInputs,
    #[error("fee {fee} for leg {leg} is not below 2^63 (field-representable)")]
    FeeTooLarge { leg: usize, fee: u64 },
    #[error("input {input} has {found} sibling entries, expected {expected}")]
    SiblingCount { input: usize, expected: usize, found: usize },
    #[error("shell and witness disagree on input/output counts ({found} of {expected})")]
    InputCount { expected: usize, found: usize },
    #[error("input {index}: derived nullifier does not match the shell's leg nullifier")]
    NullifierMismatch { index: usize },
    #[error("input {index}: anchor does not match its leg's anchored root")]
    AnchorMismatch { index: usize },
    #[error("output {index}: commitment or value does not match the shell's output")]
    OutputMismatch { index: usize },
    #[error("leg {leg}: fee does not match the shell's declared fee")]
    FeeMismatch { leg: usize },
    #[error("input {index}: nullifier_key does not hash to the opening's pk_n (erratum 72)")]
    NullifierKeyMismatch { index: usize },
      #[error("{found} revert witnesses for {expected} conditional outputs (D.3)")]
    RevertCount { expected: usize, found: usize },
    #[error("revert {index}: commitment does not match the shell's revert_cm (D.3)")]
    RevertCmMismatch { index: usize },
    #[error("revert {index}: value does not equal its conditional output's (D.3 revert equation)")]
    RevertValueMismatch { index: usize },
    #[error("burn {index}: commitment does not match H(burn ‖ txid ‖ leg ‖ value)")]
    BurnMismatch { index: usize },

}
    /// Transaction-witness generation and validation failures.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum WitnessGenError {
    #[error(transparent)]
    Witness(#[from] WitnessError),
    #[error(transparent)]
    Custody(#[from] CustodyError),
    #[error(transparent)]
    Feature(#[from] nerv_codec::features::FeatureError),
    #[error(transparent)]
    Inadmissible(#[from] nerv_codec::features::AdmissibilityViolation),
    #[error(transparent)]
    Seal(#[from] nerv_seal::SealError),
    #[error(transparent)]
    Stmt(#[from] nerv_seal::CircuitStmtError),
    #[error("leg {leg}: delta is zero — statement 8 violated")]
    ZeroDelta { leg: usize },
    #[error("shell declares encoder version {shell}; the codec is {codec}")]
    WeightVersion { shell: u64, codec: u64 },
    #[error("{found} noise seeds for {expected} legs")]
    SeedCount { expected: usize, found: usize },
    #[error("epoch key identifier mismatch")]
    EpochKeyId,
    #[error("witness leg index {found}, expected {expected}")]
    LegIndex { found: usize, expected: usize },
    #[error("leg {leg}: movement does not match the custody data")]
    MovementMismatch { leg: usize },
    #[error("leg {leg}: feature vector is not the movement's canonical build")]
    FeaturesMismatch { leg: usize },
    #[error("leg {leg}: delta is not W·ΔS of the features")]
    DeltaMismatch { leg: usize },
    #[error("leg {leg}: seal triplet does not derive from the noise seed")]
    TripletMismatch { leg: usize },
    #[error("leg {leg}: plaintext is not the delta's digitization")]
    PlaintextMismatch { leg: usize },
    #[error("leg {leg}: expected ciphertext does not match the statement-10 native")]
    CiphertextMismatch { leg: usize },
}



