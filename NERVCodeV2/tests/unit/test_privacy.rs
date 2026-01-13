// tests/unit/test_privacy.rs
// ============================================================================
// PRIVACY MODULE UNIT TESTS
// ============================================================================


use nerv_bit::privacy::mixer::*;
use nerv_bit::privacy::tee::attestation::*;
use rand::Rng;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};


#[test]
fn test_onion_mixer_creation() {
    let config = MixerConfig::default();
    let mixer = OnionMixer::new(config);
    
    assert_eq!(mixer.config.hop_count, 5);
    assert_eq!(mixer.config.cover_traffic_enabled, true);
    assert_eq!(mixer.config.cover_traffic_ratio, 0.3);
}


#[test]
fn test_onion_layer_creation() {
    let payload = b"test payload";
    let next_hop = b"127.0.0.1:8080";
    let recipient_pk = [0u8; 32];
    
    let layer = OnionLayer::new(payload, next_hop, 0, &recipient_pk).unwrap();
    
    assert_eq!(layer.layer_index, 0);
    assert_eq!(layer.next_hop, next_hop);
    assert_eq!(layer.encrypted_payload.0.len() > 0, true);
}


#[test]
fn test_onion_route_creation() {
    let config = MixerConfig {
        hop_count: 3,
        relay_nodes: vec![
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8001),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8002),
        ],
        ..Default::default()
    };
    
    let destination = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 9000);
    let relay_pks = vec![[1u8; 32], [2u8; 32]];
    
    let route = OnionRoute::new(&config, destination, relay_pks).unwrap();
    
    assert_eq!(route.route.len(), 3);
    assert_eq!(route.session_keys.len(), 3);
}


#[test]
fn test_encrypted_payload() {
    let data = b"secret message";
    let recipient_pk = [0u8; 32];
    
    let payload = EncryptedPayload::new(data, &recipient_pk).unwrap();
    
    assert_eq!(payload.iv.len(), 12);
    assert_eq!(payload.tag.len(), 16);
    assert_eq!(payload.recipient_id.len(), 32);
    assert!(payload.ciphertext.len() > 0);
}


#[test]
fn test_tee_attestation_creation() {
    let data = b"attestation data";
    
    // Test SGX attestation
    let sgx_report = generate_sgx_report(data).unwrap();
    let sgx_attestation = TeeAttestation::Sgx(sgx_report);
    
    assert_eq!(sgx_attestation.get_type(), "SGX");
    
    // Test SEV attestation
    let sev_report = generate_sev_report(data).unwrap();
    let sev_attestation = TeeAttestation::Sev(sev_report);
    
    assert_eq!(sev_attestation.get_type(), "SEV");
    
    // Test dummy attestation
    let dummy = TeeAttestation::dummy();
    assert_eq!(dummy.get_type(), "Dummy");
}


#[test]
fn test_attestation_report() {
    let data = b"test data for attestation";
    let signer_key = [0u8; 32];
    
    // Create dummy attestation
    let tee_attestation = TeeAttestation::dummy();
    
    let report = AttestationReport::new(tee_attestation, data, &signer_key).unwrap();
    
    assert_eq!(report.measurement.len(), 32);
    assert_eq!(report.data_hash.len(), 32);
    assert!(report.timestamp > 0);
    
    // Verify report
    let data_hash = blake3::hash(data).into();
    let is_valid = report.verify(&data_hash).unwrap();
    
    assert!(is_valid, "Attestation report should be valid");
}


#[test]
fn test_attestation_verification() {
    let data = b"verification test";
    
    // Create SGX attestation
    let sgx_report = generate_sgx_report(data).unwrap();
    let attestation = TeeAttestation::Sgx(sgx_report);
    
    // Verify with correct data
    let is_valid = attestation.verify(data).unwrap();
    assert!(is_valid, "SGX attestation should verify");
    
    // Verify with wrong data
    let wrong_data = b"wrong data";
    let is_valid = attestation.verify(wrong_data).unwrap();
    assert!(!is_valid, "Should fail with wrong data");
}


#[test]
fn test_remote_attestation() {
    let remote = RemoteAttestation {
        quote: vec![1u8; 100],
        signature: vec![2u8; 64],
        signing_cert: vec![3u8; 200],
        timestamp: 1234567890,
    };
    
    let attestation = TeeAttestation::Remote(remote);
    
    assert_eq!(attestation.get_type(), "Remote");
    
    // Test verification (will fail due to timestamp)
    let is_valid = attestation.verify(b"test").unwrap();
    assert!(!is_valid, "Remote attestation should fail due to timestamp");
}


#[test]
fn test_mixer_error_handling() {
    let config = MixerConfig::default();
    let mixer = OnionMixer::new(config);
    
    // Test invalid hop count
    let payload = EncryptedPayload::new(b"test", &[0u8; 32]).unwrap();
    
    let result = mixer.encrypt_layers(payload, 3); // Should fail, config has 5 hops
    assert!(result.is_err());
    
    // Test layer processing with invalid data
    let invalid_layer = OnionLayer {
        encrypted_payload: MlKemCiphertext::from(vec![0u8; 100]),
        next_hop: vec![0u8; 32],
        layer_index: 0,
        auth_tag: [0u8; 32],
    };
    
    let result = mixer.process_layer(invalid_layer);
    assert!(result.is_err(), "Should fail with invalid layer");
}
