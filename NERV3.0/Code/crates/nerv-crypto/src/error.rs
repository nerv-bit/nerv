//! Error taxonomy for nerv-crypto.

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CryptoError {
    #[error("invalid length for {what}: expected {expected}, found {found}")]
    InvalidLength { what: &'static str, expected: usize, found: usize },
    #[error("external primitive rejected input: {0}")]
    Provider(&'static str),
    #[error("AEAD authentication failed")]
    AeadAuthentication,
    #[error("HKDF output length {len} exceeds the maximum {max}")]
    KdfOutputTooLong { len: usize, max: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum QcError {
    #[error("committee size {size} exceeds the bitmap limit {max}")]
    CommitteeTooLarge { size: usize, max: usize },
    #[error("member index {member} outside committee of {committee}")]
    MemberIndexOutOfRange { member: usize, committee: usize },
    #[error("signature from member {member} failed verification")]
    InvalidVote { member: usize },
    #[error("quorum not met: {have} of {need}")]
    QuorumNotMet { have: usize, need: usize },
    #[error("malformed certificate: {0}")]
    MalformedCertificate(&'static str),
}

