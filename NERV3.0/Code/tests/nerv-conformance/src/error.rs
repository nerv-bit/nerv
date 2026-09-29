//! Error taxonomy for the conformance registry.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum SpecError {
    #[error("I/O error reading {path}: {source}")]
    Io { path: PathBuf, source: std::io::Error },
    #[error("TOML parse error: {0}")]
    Parse(toml::de::Error),
    #[error("TOML deserialization (typed spec) error: {0}")]
    Deserialize(toml::de::Error),
    #[error("P5 violation — forbidden value types in the spec tree (no floats, no datetimes): {paths:?}")]
    Forbidden { paths: Vec<String> },
    #[error("spec validation failed with {count} violation(s):\n  - {report}")]
    Validation { count: usize, report: String },
    #[error("internal error: {0}")]
    Internal(&'static str),
}

#[derive(Debug, thiserror::Error)]
pub enum EncError {
    #[error("truncated canonical input")]
    Truncated,
    #[error("{excess} trailing byte(s) after the canonical value")]
    Trailing { excess: usize },
    #[error("unknown canonical tag 0x{tag:02x}")]
    UnknownTag { tag: u8 },
    #[error("invalid bool byte 0x{byte:02x}")]
    InvalidBool { byte: u8 },
    #[error("duplicate map key in canonical input")]
    DuplicateKey,
    #[error("invalid UTF-8 in canonical string")]
    InvalidUtf8,
    #[error("nesting depth exceeds {0}")]
    DepthLimit(u32),
    #[error("type mismatch: wanted {wanted}, got {got}")]
    TypeMismatch { wanted: &'static str, got: &'static str },
    #[error("missing map key `{key}`")]
    MissingKey { key: String },
    #[error("negative integer {value} — canonical ints are unsigned")]
    NegativeInt { value: i64 },
    #[error("forbidden TOML value type (float/datetime) — P5: the spec is integer-exact")]
    ForbiddenTomlType,
}

#[derive(Debug, thiserror::Error)]
pub enum ScheduleError {
    #[error("bucket `{bucket}`: unknown kind `{kind}`")]
    UnknownKind { bucket: String, kind: String },
    #[error("bucket `{bucket}`: {reason}")]
    Malformed { bucket: String, reason: &'static str },
    #[error("schedule arithmetic failed: {op}")]
    Arithmetic { op: &'static str },
    #[error("bucket totals sum to {sum} nano, expected {expected} (supply identity, §12.1)")]
    SupplyMismatch { sum: u128, expected: u128 },
}

#[derive(Debug, thiserror::Error)]
pub enum RegError {
    #[error("I/O error on {path}: {source}")]
    Io { path: PathBuf, source: std::io::Error },
    #[error("conformance encoding error: {0}")]
    Encoding(#[from] EncError),
    #[error("schedule reference error: {0}")]
    Schedule(#[from] ScheduleError),
    #[error("{0}")]
    Generator(String),
    #[error("manifest format version {found} != expected {expected}")]
    FormatVersion { found: u64, expected: u64 },
    #[error("SPEC DRIFT: frozen vectors pin spec-hash {frozen}, current spec hashes {current} — parameters changed since freeze; deliberate re-freeze required (make freeze-vectors)")]
    SpecDrift { frozen: String, current: String },
    #[error("SCHEMA DRIFT: family `{family}` schema hash changed since freeze — record structure changed; deliberate re-freeze required")]
    SchemaDrift { family: String },
    #[error("TAMPER: family `{family}` data file hash does not match the frozen manifest")]
    Tamper { family: String },
    #[error("MISSING DATA: family `{family}` has a manifest entry but no data file")]
    MissingData { family: String },
    #[error("family `{family}` record count {found} != frozen {expected}")]
    RecordCount { family: String, found: u64, expected: u64 },
    #[error("GENERATOR DRIFT: family `{family}` re-derivation differs from frozen records (first differing record: {first_diff}) — implementation changed since freeze")]
    GeneratorDrift { family: String, first_diff: u64 },
    #[error("family `{family}` is frozen but no generator is registered in this build — the owning chunk's feature is missing")]
    MissingGenerator { family: String },
    #[error("family `{family}` is schema-defined (deferred) but a data file exists — not frozen through the registry")]
    UnexpectedData { family: String },
    #[error("unknown family `{family}` in manifest (renamed or removed without a deliberate re-freeze)")]
    UnknownFamily { family: String },
    #[error("family `{family}` is generated in this build but absent from the frozen manifest")]
    MissingFamily { family: String },
    #[error("vectors directory has no manifest — run `make bootstrap` (or `cargo xtask conformance-freeze`) and commit specs/vectors/")]
    NoManifest,
    #[error("frozen manifest already exists — re-freezing is deliberate; pass --force (make freeze-vectors) and review the diff")]
    AlreadyFrozen,
}
