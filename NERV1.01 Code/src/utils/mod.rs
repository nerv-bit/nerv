// src/utils/mod.rs
// ============================================================================
// NERV UTILITIES MODULE
// ============================================================================
// This module provides shared utilities used across the NERV codebase:
// 1. Serialization (Protobuf for deterministic encoding)
// 2. Logging and tracing setup
// 3. Metrics collection and reporting
// 4. Common helper functions and error handling
// ============================================================================

// Re-export commonly used utilities
pub use serialization::{ProtobufEncoder, ProtobufDecoder, SerializeError};
pub use logging::{init_logging, LogLevel, setup_tracing};
pub use metrics::{MetricsCollector, Counter, Gauge, Histogram};
pub use errors::{ErrorExt, ContextExt, ensure};

// Module declarations
pub mod serialization;
pub mod logging;
pub mod metrics;
pub mod errors;
pub mod time;
pub mod validation;

// ============================================================================
// COMMON TYPE ALIASES AND CONSTANTS
// ============================================================================

/// Result type for utility functions
pub type UtilResult<T> = std::result::Result<T, UtilError>;

/// Error type for utility functions
#[derive(thiserror::Error, Debug)]
pub enum UtilError {
    #[error("Serialization error: {0}")]
    Serialization(#[from] SerializeError),
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Validation error: {0}")]
    Validation(String),
}

/// Convert bytes to hex string
pub fn bytes_to_hex(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

/// Convert hex string to bytes
pub fn hex_to_bytes(hex_str: &str) -> UtilResult<Vec<u8>> {
    hex::decode(hex_str).map_err(|e| UtilError::Validation(e.to_string()))
}

/// Calculate BLAKE3 hash of data
pub fn blake3_hash(data: &[u8]) -> [u8; 32] {
    use blake3::Hasher;
    let mut hasher = Hasher::new();
    hasher.update(data);
    let hash = hasher.finalize();
    *hash.as_bytes()
}

/// Compare two fixed-size arrays in constant time (for cryptographic safety)
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    use subtle::ConstantTimeEq;
    a.ct_eq(b).into()
}

/// Generate cryptographically secure random bytes
pub fn secure_random_bytes(len: usize) -> Vec<u8> {
    use rand::RngCore;
    let mut rng = rand::thread_rng();
    let mut bytes = vec![0u8; len];
    rng.fill_bytes(&mut bytes);
    bytes
}

/// Generate a nonce for cryptographic operations
pub fn generate_nonce() -> [u8; 12] {
    let mut nonce = [0u8; 12];
    nonce.copy_from_slice(&secure_random_bytes(12)[..12]);
    nonce
}

/// Calculate the current timestamp in milliseconds since Unix epoch
pub fn current_timestamp_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Format a duration for human readability
pub fn format_duration(duration_ms: u64) -> String {
    if duration_ms < 1000 {
        format!("{}ms", duration_ms)
    } else if duration_ms < 60_000 {
        format!("{:.2}s", duration_ms as f64 / 1000.0)
    } else if duration_ms < 3_600_000 {
        format!("{:.2}m", duration_ms as f64 / 60_000.0)
    } else {
        format!("{:.2}h", duration_ms as f64 / 3_600_000.0)
    }
}

// ============================================================================
// MACROS FOR COMMON PATTERNS
// ============================================================================

/// Macro for logging debug messages only in debug builds
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {
        if cfg!(debug_assertions) {
            tracing::debug!($($arg)*);
        }
    };
}

/// Macro for timing a block of code and logging the duration
#[macro_export]
macro_rules! timed {
    ($name:expr, $code:block) => {{
        let start = std::time::Instant::now();
        let result = $code;
        let duration = start.elapsed();
        tracing::debug!("{} took {:?}", $name, duration);
        result
    }};
}

/// Macro for ensuring a condition is met, returning an error if not
#[macro_export]
macro_rules! ensure {
    ($condition:expr, $error:expr) => {
        if !$condition {
            return Err($error.into());
        }
    };
    ($condition:expr, $fmt:expr, $($arg:tt)*) => {
        if !$condition {
            return Err(format!($fmt, $($arg)*).into());
        }
    };
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_hex() {
        let bytes = [0xDE, 0xAD, 0xBE, 0xEF];
        let hex = bytes_to_hex(&bytes);
        assert_eq!(hex, "deadbeef");
    }

    #[test]
    fn test_hex_to_bytes() {
        let hex = "deadbeef";
        let bytes = hex_to_bytes(hex).unwrap();
        assert_eq!(bytes, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_blake3_hash() {
        let data = b"hello world";
        let hash = blake3_hash(data);
        assert_eq!(hash.len(), 32);
        
        // Different input should produce different hash
        let hash2 = blake3_hash(b"hello world!");
        assert_ne!(hash, hash2);
    }

    #[test]
    fn test_constant_time_eq() {
        let a = [1, 2, 3];
        let b = [1, 2, 3];
        let c = [1, 2, 4];
        
        assert!(constant_time_eq(&a, &b));
        assert!(!constant_time_eq(&a, &c));
    }

    #[test]
    fn test_secure_random_bytes() {
        let bytes1 = secure_random_bytes(32);
        let bytes2 = secure_random_bytes(32);
        
        assert_eq!(bytes1.len(), 32);
        assert_eq!(bytes2.len(), 32);
        // Very low probability of collision, but possible
        // In practice, this test is fine
    }

    #[test]
    fn test_generate_nonce() {
        let nonce = generate_nonce();
        assert_eq!(nonce.len(), 12);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(500), "500ms");
        assert_eq!(format_duration(1500), "1.50s");
        assert_eq!(format_duration(90000), "1.50m");
        assert_eq!(format_duration(7200000), "2.00h");
    }
}
