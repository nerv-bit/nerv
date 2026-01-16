// tests/unit/test_privacy.rs
// ============================================================================
// PRIVACY MODULE UNIT TESTS
// ============================================================================

use nerv_bit::privacy::{
    PrivacyManager, PrivacyConfig,
    tee::{
        TEERuntime, TEEType, TEEAttestation, AttestationError,
        attestation::{AttestationService, verify_attestation_report},
        sealing::{SealedState, SealingError},
    },
    mixer::{Mixer, MixConfig, OnionLayer, EncryptedPayload},
    vdw::{VDWGenerator, VDWVerifier, VDW, DelayParameters, VDWError},
};
use nerv_bit::crypto::CryptoProvider;
use nerv_bit::embedding::EmbeddingVec;
use rand::{thread_rng, Rng};
use std::sync::Arc;
use std::time::{SystemTime, Duration};
use tokio::time;

#[test]
fn test_privacy_config_defaults() {
    let config = PrivacyConfig::default();

    assert!(config.enable_tee);
    assert!(config.enable_mixer);
    assert!(config.enable_vdw);
    assert_eq!(config.mix_hops, 5);
    assert!(config.enable_blind_validation);
    assert!(config.cover_ratio > 1.0);
}

#[tokio::test]
async fn test_privacy_manager_creation() {
    let crypto = Arc::new(CryptoProvider::new().unwrap());
    let config = PrivacyConfig::default();
    let manager = PrivacyManager::new(config, crypto).await.unwrap();

    assert!(manager.is_tee_available());
    assert!(manager.mixer.is_some());
    assert!(manager.vdw_generator.is_some());
}

#[test]
fn test_sealed_state_basics() {
    let blob = vec![1u8; 256];
    let sealed = SealedState::new(TEEType::SGX, blob.clone());

    assert_eq!(sealed.tee_type, TEEType::SGX);
    assert_eq!(sealed.blob, blob);
    assert_eq!(sealed.version, 1);
    assert!(sealed.is_for_tee(TEEType::SGX));
    assert!(!sealed.is_for_tee(TEEType::SEV));
}

#[tokio::test]
async fn test_tee_attestation_verification_mock() {
    // Use mock report (simulation mode)
    let mock_report = vec![0u8; 512];
    let attestation = TEEAttestation::Remote {
        quote: mock_report.clone(),
        signature: vec![],
        cert_chain: vec![],
        report_data: vec![],
        timestamp: SystemTime::now(),
    };

    // Verification should pass in simulation or with disabled checks
    let result = verify_attestation_report(&attestation, None);
    assert!(result.is_ok());

    // Invalid type
    let invalid = TEEAttestation::Dummy;
    let result_invalid = verify_attestation_report(&invalid, None);
    assert!(matches!(result_invalid, Err(AttestationError::InvalidType)));
}

#[tokio::test]
async fn test_mixer_onion_layer_building() {
    let crypto = Arc::new(CryptoProvider::new().unwrap());
    let config = MixConfig::default();
    let mut mixer = Mixer::new(config, crypto);

    let payload = vec![42u8; 128];
    let hops = vec![
        [1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32], [5u8; 32],
    ];

    let onion = mixer.build_onion(payload.clone(), &hops).await.unwrap();

    assert_eq!(onion.layers.len(), 5);
    assert!(onion.encrypted_payload.len() > payload.len());

    // Process first layer (mock next hop)
    let (decrypted, next_hop) = mixer.process_layer(&onion).await.unwrap();
    assert_eq!(next_hop, hops[4]); // Innermost hop revealed last
}

#[tokio::test]
async fn test_mixer_cover_traffic_generation() {
    let crypto = Arc::new(CryptoProvider::new().unwrap());
    let mut config = MixConfig::default();
    config.cover_ratio = 2.0; // High for test
    config.probabilistic_cover_ratio = 1.0; // Always trigger

    let mut mixer = Mixer::new(config, crypto);

    // Trigger probabilistic cover
    mixer.maybe_generate_cover_traffic().await;

    // Background task (run briefly)
    let handle = tokio::spawn(async move {
        mixer.generate_background_cover_traffic().await;
    });

    time::sleep(Duration::from_millis(50)).await;
    handle.abort(); // Stop background task
}

#[tokio::test]
async fn test_mixer_symmetric_encryption_round_trip() {
    let key = [0x42u8; 32];
    let plaintext = b"secret privacy payload";
    let nonce = [0u8; 12];

    let ciphertext = Mixer::symmetric_encrypt(&key, plaintext, &nonce).unwrap();
    let decrypted = Mixer::symmetric_decrypt(&key, &ciphertext).unwrap();

    assert_eq!(decrypted, plaintext);

    // Tampered ciphertext
    let mut tampered = ciphertext.clone();
    tampered[20] ^= 0xff;
    assert!(Mixer::symmetric_decrypt(&key, &tampered).is_err());
}

#[test]
fn test_vdw_delay_parameters() {
    let params = DelayParameters {
        steps: 1_000_000,
        difficulty: 28,
        delay_blocks: 1000,
    };

    assert_eq!(params.estimated_delay_seconds(), 1000 * 10); // Assuming ~10s block time
}

#[tokio::test]
async fn test_vdw_generation_and_verification_mock() {
    let crypto = Arc::new(CryptoProvider::new().unwrap());
    let generator = VDWGenerator::new(crypto);

    let proof = vec![1u8; 256];
    let embedding_hash = [2u8; 32];
    let height = 1000;

    // Generate mock VDW
    let vdw_id = generator.generate_and_store(proof, embedding_hash, height).await.unwrap();

    // Verify (in mock, should pass basic checks)
    let verified = VDWVerifier::verify(&vdw_id).await.unwrap();
    assert!(verified); // Mock always true or based on simple checks

    // Delay not satisfied
    let early_vdw = VDW {
        embedding_hash,
        recursive_proof: vec![],
        block_height: height,
        params: DelayParameters {
            steps: 100,
            difficulty: 10,
            delay_blocks: 10000, // Large delay
        },
        vdf_output: vec![],
        vdf_proof: vec![],
        timestamp: SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs(),
        signature: vec![],
        storage_id: "mock".to_string(),
    };

    let early_result = VDWVerifier::verify_vdw(&early_vdw, height + 100);
    assert!(matches!(early_result, Err(VDWError::DelayNotSatisfied)));
}

#[tokio::test]
async fn test_tee_runtime_mock_operations() {
    // Use factory with None or simulation mode
    let runtime_result = <dyn TEERuntime>::new(TEEType::SGX);
    if runtime_result.is_err() {
        // Expected in non-hardware environment
        assert!(matches!(runtime_result.unwrap_err(), NervError::TEEAttestation(_)));
        return;
    }

    let runtime = runtime_result.unwrap();

    // Local attestation
    let data = b"test_report_data";
    let report = runtime.local_attest(data).await.unwrap();
    assert!(!report.is_empty());

    // Seal/unseal round-trip (mock)
    let state = vec![9u8; 512];
    let sealed = runtime.seal_state(&state).await.unwrap();
    let unsealed = runtime.unseal_state(&sealed.blob).await.unwrap();
    assert_eq!(unsealed, state);

    // Blind validation (mock passes)
    let delta = vec![0.1f64; 512];
    let valid = runtime.execute_blind_validation(&delta).await.unwrap();
    assert!(valid);
}