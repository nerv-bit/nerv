//! The nerv-core error taxonomy ([BUILD], thiserror).
//!
//! One error type per module family; `Error` is the umbrella. Error payloads
//! carry no heap allocations (fixed-size fields and `&'static str`) — rare
//! paths, but the consensus path still avoids gratuitous allocation.

use thiserror::Error;

/// Canonical-codec failures. A `CodecError` always means "these bytes are not
/// a canonical encoding of the target type" — never "wrong value".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CodecError {
    #[error("truncated input")]
    Truncated,
    #[error("{excess} trailing byte(s) after the canonical value")]
    TrailingBytes { excess: usize },
    #[error("invalid bool byte 0x{byte:02x}")]
    InvalidBool { byte: u8 },
    #[error("invalid Option tag 0x{tag:02x}")]
    InvalidOptionTag { tag: u8 },
    #[error(
        "sequence length {count} exceeds remaining input ({remaining} bytes) — non-canonical or corrupt"
    )]
    SeqLenOverrun { count: usize, remaining: usize },
    #[error("sequence length {count} exceeds the parser cap {max}")]
    SeqTooLarge { count: usize, max: usize },
    #[error("string byte-length {len} exceeds the cap {max}")]
    StringTooLong { len: usize, max: usize },
    #[error("string is not valid UTF-8")]
    InvalidUtf8,
    #[error("nesting depth exceeds {max}")]
    DepthLimit { max: u32 },
    #[error("codec invariant violated: {0}")]
    InvariantViolated(&'static str),
    #[error("field element {value} is not below the Goldilocks prime")]
    FieldElementOutOfRange { value: u64 },

}

/// Type-construction failures for core vocabulary types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum TypeError {
    #[error("shard id is {bits} bits; the prefix trie allows at most {max} bits")]
    ShardIdTooLong { bits: u8, max: u8 },
    #[error("{what} = {value} is outside [{min}, {max}]")]
    OutOfRange { what: &'static str, value: u64, min: u64, max: u64 },
    #[error("invalid shard set: {reason}")]
    InvalidShardSet { reason: &'static str },
    
   #[error("transit entry is {state}, not Pending — consumed once, by either path (D.3)")]
    TransitEntryConsumed { state: &'static str },

}

/// Fixed-point arithmetic failures (WP §7.2 exact integer arithmetic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum FixedPointError {
    #[error("fixed-point division by zero")]
    DivisionByZero,
    #[error("fixed-point overflow in {op}")]
    Overflow { op: &'static str },
    #[error("value unrepresentable in the target format ({op})")]
    Unrepresentable { op: &'static str },
}

/// Umbrella error for the core crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum Error {
    #[error(transparent)]
    Codec(#[from] CodecError),
    #[error(transparent)]
    Type(#[from] TypeError),
    #[error(transparent)]
    FixedPoint(#[from] FixedPointError),
}

