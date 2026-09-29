//! Error taxonomy for nerv-seal. Ring arithmetic, carry resolution, and
//! sigma verification equations are total or exactly checkable; parsing
//! and validation are fallible. `RevealError`: invalid reveals (WP §6.3.4).
//! `SigmaError`: FS proofs. `DkgError`: the DKG ceremony. `VpdError`:
//! partial decryption. `EpochError`: key lifecycle. `CircuitStmtError`:
//! the native statement-9/10 twins.


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SealError {
    #[error("serialized length {len}, expected {expected}")]
    BadLength { len: usize, expected: usize },
    #[error("coefficient {value} is not reduced below the ring modulus Q")]
    UnreducedCoefficient { value: u64 },
    #[error("sampling exhausted {attempts} attempts — unreachable at frozen parameters (P9)")]
    SamplingExhausted { attempts: u64 },
    #[error("slot {slot} value {value} exceeds the 8-bit digit maximum 255")]
    DigitOutOfRange { slot: usize, value: u64 },
}


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum RevealError {
    #[error("a reveal of 0 legs is not a chunk")]
    EmptyChunk,
    #[error("chunk of {legs} legs exceeds the maximum {max}")]
    ChunkTooLarge { legs: u64, max: u64 },
    #[error("serialized length {len}, expected {expected}")]
    BadLength { len: usize, expected: usize },
    #[error("slot {slot} decoded to {value} — below the digit floor")]
    NegativeDigit { slot: usize, value: i64 },
    #[error("slot {slot} digit sum {value} exceeds the envelope {envelope}")]
    DigitSumAboveEnvelope { slot: usize, value: u64, envelope: u64 },
}


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SigmaError {
    #[error("serialized length {len}, expected {expected}")]
    BadLength { len: usize, expected: usize },
    #[error("block {block} bound {bound} infeasible against the uniformity range")]
    InfeasibleBound { block: usize, bound: u64 },
    #[error("row {row} has a duplicate entry for block {block}")]
    DuplicateEntry { row: usize, block: usize },
    #[error("row {row} is empty")]
    EmptyRow { row: usize },
    #[error("row {row} references block {block} outside {blocks} witness blocks")]
    EntryOutOfRange { row: usize, block: usize, blocks: usize },
    #[error("proof shape mismatch: h {h_len}, z {z_len}, rows {rows}, blocks {blocks}")]
    StructuralMismatch { h_len: usize, z_len: usize, rows: usize, blocks: usize },
    #[error("equation {row} failed")]
    EquationFailed { row: usize },
    #[error("response block {block} norm {value} exceeds {bound}")]
    NormExceeded { block: usize, value: u64, bound: u64 },
    #[error("witness block {block} violates its published bound")]
    WitnessViolatesBound { block: usize },
    #[error("proof generation exhausted {attempts} attempts")]
    GenExhausted { attempts: u64 },
    #[error(transparent)]
    Seal(#[from] SealError),
}


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DkgError {
    #[error("committee size {n} outside [1, 255]")]
    BadCommittee { n: usize },
    #[error("threshold {t} outside [1, {n}]")]
    BadThreshold { t: usize, n: usize },
    #[error("member {member} outside [1, {n}]")]
    BadMember { member: u8, n: usize },
    #[error("share set has {len} members, expected {expected}")]
    BadShareSet { len: usize, expected: usize },
    #[error("duplicate member {member} in share set")]
    DuplicateMember { member: u8 },
    #[error("Lagrange denominator not invertible for member {member}")]
    NotInvertible { member: u8 },
    #[error("interpolation identity check failed — internal invariant violated")]
    InterpolationCheckFailed,
    #[error("fragment opening from {from} to {to} does not match its commitment")]
    FragmentMismatch { from: u8, to: u8 },
    #[error(transparent)]
    Sigma(#[from] SigmaError),
    #[error(transparent)]
    Seal(#[from] SealError),
}


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum VpdError {
    #[error("partial set has {len} entries, expected {expected}")]
    BadSubset { len: usize, expected: usize },
    #[error("partial {index} is from member {partial}, subset expects {subset}")]
    MemberMismatch { index: usize, partial: u8, subset: u8 },
    #[error("member {member} is not in the subset")]
    MemberNotInSubset { member: u8 },
    #[error("serialized length {len}, expected {expected}")]
    BadLength { len: usize, expected: usize },
    #[error(transparent)]
    Sigma(#[from] SigmaError),
    #[error(transparent)]
    Seal(#[from] SealError),
    #[error(transparent)]
    Dkg(#[from] DkgError),
}


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum EpochError {
    #[error("serialized length {len}, expected {expected}")]
    BadLength { len: usize, expected: usize },
    #[error("handoff window is closed")]
    WindowClosed,
    #[error("unknown batch {batch}")]
    UnknownBatch { batch: u64 },
    #[error("duplicate batch {batch}")]
    DuplicateBatch { batch: u64 },
    #[error("reveal cap {cap} reached for the key ({reveals} reveals recorded)")]
    OverRevealCap { reveals: u64, cap: u64 },
    #[error("batch of {legs} legs is not sub-minimum (need 1..{min})")]
    PadRange { legs: u64, min: u64 },
    #[error("backup payload member {member} outside [1, {n}]")]
    BadPayloadMember { member: u8, n: usize },
    #[error("backup share bounds violated (share {share}, rho {rho})")]
    PayloadBound { share: u64, rho: u64 },
    #[error(transparent)]
    Dkg(#[from] DkgError),
    #[error(transparent)]
    Seal(#[from] SealError),
}


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CircuitStmtError {
    #[error("statement-10 relation row {row} failed (rows 0–7: u, 8–9: v)")]
    RelationFailed { row: usize },
    #[error("witness block {block} (r ‖ e₁ ‖ e₂) has coefficient {value}, bound {bound}")]
    NoiseBound { block: usize, value: u64, bound: u64 },
    #[error("plaintext slot {slot} value {value} exceeds the digit bound")]
    DigitBound { slot: usize, value: u64 },
    #[error(transparent)]
    Seal(#[from] SealError),
}
