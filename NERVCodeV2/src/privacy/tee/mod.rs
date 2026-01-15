// src/privacy/tee/mod.rs
// ============================================================================
// TEE (Trusted Execution Environment) MODULE - Reconciled Comprehensive Version
// ============================================================================
// This reconciled implementation combines all functionality from both versions:
// - Multi-vendor runtime trait (SGX/SEV) with async methods (new)
// - Arbitrary enclave code execution (old)
// - Blind validation, sealing/unsealing, measurement (new)
// - Full TEE manager with configuration, fallback to non-TEE (old)
// - Remote attestation service with URL and reqwest client (old)
// - Transaction-specific attestation and remote verification (old)
// - Factory, availability checks, and None variant (merged)
// ============================================================================

pub mod attestation;
pub mod sgx;
pub mod sev;
pub mod sealing;

pub use attestation::{TEEAttestation, AttestationError, AttestationService};
pub use sgx::SGXRuntime;
pub use sev::SEVRuntime;
pub use sealing::{SealedState, SealingError};

use crate::{Result, NervError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::sync::Arc;
use reqwest::Client;
use crate::embedding::circuit::Fp;  // pallas::Base

/// Supported TEE types (merged - added None from old)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TEEType {
    SGX,
    SEV,
    None, // Fallback when no hardware TEE
    // Future: ARM CCA, etc.
}

/// Comprehensive TEE configuration (merged)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TEEConfig {
    /// Selected TEE type (None for fallback)
    pub tee_type: TEEType,
    
    /// Path to signed enclave binary (if required)
    pub enclave_path: Option<String>,
    
    /// Expected measurement for verification
    pub expected_measurement: Option<[u8; 32]>,
    
    /// Enable remote attestation
    pub enable_attestation: bool,
    
    /// Remote attestation service URL (DCAP/AMD KDS)
    pub remote_attestation_url: Option<String>,
}

impl Default for TEEConfig {
    fn default() -> Self {
        Self {
            tee_type: TEEType::None,
            enclave_path: None,
            expected_measurement: None,
            enable_attestation: false,
            remote_attestation_url: None,
        }
    }
}

/// Common trait for all TEE runtimes (extended with old features)
#[async_trait]
pub trait TEERuntime: Send + Sync + Debug {
    /// Check if this TEE type is available on hardware
    fn is_available() -> bool;
    
    /// Perform remote attestation and return proof
    async fn perform_attestation(&self) -> Result<TEEAttestation, NervError>;
    
    /// Execute blind validation of embedding delta
    async fn execute_blind_validation(&self, delta_data: &[u8]) -> Result<bool, NervError>;
    
    /// Execute arbitrary code inside enclave (from old)
    async fn execute(&self, code: &[u8], data: &[u8]) -> Result<Vec<u8>, NervError>;
    
    /// Seal sensitive state
    async fn seal_state(&self, data: &[u8]) -> Result<SealedState, NervError>;
    
    /// Unseal state
    async fn unseal_state(&self, sealed: &SealedState) -> Result<Vec<u8>, NervError>;
    
    /// Get TEE type
    fn tee_type(&self) -> TEEType;
    
    /// Get hardware measurement
    fn measurement(&self) -> [u8; 32];
    
    /// Generate local attestation (from old trait)
    async fn local_attest(&self, data: &[u8]) -> Result<Vec<u8>, NervError>;

    /// Prove a homomorphic embedding update inside the TEE
    /// 
    /// Inside the TEE:
    /// - Unseal the previous private embedding (stored separately by the node)
    /// - Compute new_embedding = prev_embedding + summed_delta + error (error optional/bounded)
    /// - Synthesize the LatentLedgerCircuit witness
    /// - Generate step proof (mock for now; real Halo2/Nova later)
    /// - Compute new_embedding_hash
    /// 
    /// Returns: (proof_bytes, attestation_report)
    /// The attestation covers the code measurement and optional report data (e.g., new_hash)
    async fn prove_embedding_update(
        &self,
        sealed_prev_embedding: &[u8],     // Provided by host, unsealed inside TEE
        summed_delta: Vec<Fp>,             // Public or from batch, passed in clear
        expected_new_hash: [u8; 32],       // For verification (blake3 or poseidon)
    ) -> Result<(Vec<u8>, Vec<u8>), NervError>;  // proof, attestation_report

    /// NEW: Execute DP-SGD aggregation for federated learning
    async fn execute_dp_sgd_aggregation(
        &self,
        gradients: Vec<GradientUpdate>,
        dp_params: DpParams,
    ) -> Result<(GradientUpdate, Vec<u8>), NervError>;

}

/// Main TEE manager (from old, adapted to new trait)
pub struct TEEManager {
    runtime: Option<Arc<dyn TEERuntime>>,
    config: TEEConfig,
    attestation_service: Option<AttestationService>,
}

impl TEEManager {
    /// Create manager with config and auto-detection/fallback
    pub async fn new(config: TEEConfig) -> Result<Self, NervError> {
        let runtime = match config.tee_type {
            TEEType::SGX => {
                if SGXRuntime::is_available() {
                    Some(Arc::new(SGXRuntime::new()?))
                } else {
                    None
                }
            }
            TEEType::SEV => {
                if SEVRuntime::is_available() {
                    Some(Arc::new(SEVRuntime::new()?))
                } else {
                    None
                }
            }
            TEEType::None => None,
        };
        
        let attestation_service = if config.enable_attestation {
            Some(AttestationService::new(&config)?)
        } else {
            None
        };
        
        // Perform initial attestation if runtime available and enabled
        if let Some(rt) = &runtime {
            if config.enable_attestation {
                let att = rt.perform_attestation().await?;
                att.verify(&config)?;
            }
        }
        
        Ok(Self {
            runtime,
            config,
            attestation_service,
        })
    }
    
    /// Execute arbitrary code in enclave (fallback to empty if no TEE)
    pub async fn execute_in_enclave(&self, code: &[u8], data: &[u8]) -> Result<Vec<u8>, NervError> {
        if let Some(rt) = &self.runtime {
            rt.execute(code, data).await
        } else {
            Ok(Vec::new()) // Fallback
        }
    }
    
    /// Generate attestation for a transaction
    pub async fn attest_transaction(&self, tx: &[u8]) -> Result<TEEAttestation, NervError> {
        if let Some(service) = &self.attestation_service {
            service.attest(tx).await
        } else if let Some(rt) = &self.runtime {
            rt.perform_attestation().await
        } else {
            Ok(TEEAttestation::dummy())
        }
    }
    
    /// Verify remote attestation
    pub fn verify_attestation(
        &self,
        attestation: &TEEAttestation,
    ) -> Result<bool, NervError> {
        if let Some(service) = &self.attestation_service {
            service.verify(attestation, self.config.expected_measurement)
        } else {
            Ok(true) // Accept if disabled
        }
    }
    
    /// Check if TEE is available
    pub fn is_available(&self) -> bool {
        self.runtime.is_some()
    }
    
    /// Access underlying runtime (for PrivacyManager integration)
    pub fn runtime(&self) -> Option<&Arc<dyn TEERuntime>> {
        self.runtime.as_ref()
    }
}

/// Factory for standalone runtime creation
impl dyn TEERuntime {
    pub fn new(tee_type: TEEType) -> Result<Arc<dyn TEERuntime + Send + Sync>, NervError> {
        match tee_type {
            TEEType::SGX => {
                if SGXRuntime::is_available() {
                    Ok(Arc::new(SGXRuntime::new()?))
                } else {
                    Err(NervError::TEEAttestation("SGX not available".into()))
                }
            }
            TEEType::SEV => {
                if SEVRuntime::is_available() {
                    Ok(Arc::new(SEVRuntime::new()?))
                } else {
                    Err(NervError::TEEAttestation("SEV-SNP not available".into()))
                }
            }
            TEEType::None => Err(NervError::TEEAttestation("None selected".into())),
        }
    }
}
