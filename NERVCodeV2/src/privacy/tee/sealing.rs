// src/privacy/tee/sealing.rs
// ============================================================================
// COMMON TEE STATE SEALING WRAPPER
// ============================================================================
// Defines SealedState struct used across TEE runtimes.
// Includes TEE type tag for cross-platform unsealing (if supported).
// ============================================================================

use crate::privacy::tee::TEEType;
use serde::{Serialize, Deserialize};

/// Sealed state blob returned by TEERuntime::seal_state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SealedState {
    /// TEE type that created this sealed blob
    pub tee_type: TEEType,
    
    /// Opaque sealed data (encrypted + MAC)
    pub blob: Vec<u8>,
    
    /// Optional version or policy info
    pub version: u32,
}

impl SealedState {
    /// Create new sealed state
    pub fn new(tee_type: TEEType, blob: Vec<u8>) -> Self {
        Self {
            tee_type,
            blob,
            version: 1,
        }
    }
    
    /// Check if this sealed state is for a specific TEE type
    pub fn is_for_tee(&self, tee_type: TEEType) -> bool {
        self.tee_type == tee_type
    }
}

/// Sealing errors (re-exported in tee mod)
#[derive(Debug, thiserror::Error)]
pub enum SealingError {
    #[error("TEE type mismatch for unsealing")]
    TeeTypeMismatch,
    
    #[error("Sealing failed: {0}")]
    SealFailed(String),
    
    #[error("Unsealing failed: {0}")]
    UnsealFailed(String),
    
    #[error("Integrity check failed")]
    IntegrityCheckFailed,
}
