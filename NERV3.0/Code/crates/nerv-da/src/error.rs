
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ErasureError {
    #[error("at least one data shard is required")]
    NoData,
    #[error("parity count must be at least 1")]
    NoParity,
    #[error("total shard count {n} exceeds the field's 256 elements")]
    TooManyShards { n: usize },
    #[error("shard lengths differ ({a} vs {b})")]
    LengthMismatch { a: usize, b: usize },
    #[error("reconstruction requires exactly {expected} shards, found {found}")]
    ShardCount { expected: usize, found: usize },
    #[error("shard indices must be distinct")]
    DuplicateIndex,
    #[error("shard index {index} outside the codeword of {n}")]
    IndexRange { index: usize, n: usize },
    #[error("the k×k system is singular — not a valid shard selection")]
    Singular,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DaError {
    #[error(transparent)]
    Erasure(#[from] ErasureError),
    #[error("serialized length {len}, expected {expected}")]
    BadLength { len: usize, expected: usize },
    #[error("square width {width} is not an even power of two in [2, {max}]")]
    BadWidth { width: usize, max: usize },
    #[error("data length {len} exceeds the {max}-byte blob capacity")]
    BlobOverflow { len: usize, max: usize },
    #[error("blob set carries {found} blobs, expected {expected}")]
    BlobCount { found: usize, expected: usize },
    #[error("cell ({row}, {col}) outside the {width}×{width} square")]
    CellRange { row: usize, col: usize, width: usize },
    #[error("cell chunk length {len}, expected {expected}")]
    ChunkLength { len: usize, expected: usize },
    #[error("reconstruction failed — insufficient verified cells")]
    ReconstructionFailed,
    #[error("reconstructed square does not match the committed roots")]
    RootMismatch,
    #[error("merkle path of {len} siblings does not fit {count} leaves")]
    PathLength { len: usize, count: usize },
    #[error("data length {len} exceeds the square's {max} capacity")]
    DataOverflow { len: usize, max: usize },
}


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AvailabilityError {
    #[error("malformed evidence: {0}")]
    Malformed(&'static str),
    #[error("the committed line is a valid codeword — condition not exhibited")]
    NotExhibited,
    #[error(transparent)]
    Erasure(#[from] ErasureError),
}
