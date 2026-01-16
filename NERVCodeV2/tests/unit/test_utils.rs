Rust// tests/unit/test_utils.rs
// ============================================================================
// UTILS MODULE UNIT TESTS
// ============================================================================

use nerv_bit::utils::{
    serialization::{ProtobufEncoder, ProtobufDecoder, SerializeError, NeuralVoteProto},
    logging::{init_logging, LogLevel, LogFormat, LoggingConfig, log_transaction, log_batch_update, log_tee_attestation, log_consensus_event, log_shard_event, log_performance_metric, record_counter, record_gauge, record_histogram},
    metrics::{MetricsCollector, MetricsConfig, Timer},
    errors::{ensure, ErrorExt, ContextExt},
    time::{format_duration, current_timestamp},
    validation::{validate_embedding_size, validate_proof_size, ValidationError},
    blake3_hash, constant_time_eq, secure_random_bytes, generate_nonce,
};
use std::time::Duration;
use tokio::time::sleep;

#[test]
fn test_blake3_hash_deterministic_and_collision_resistant() {
    let data1 = b"NERV utils test data";
    let data2 = b"NERV utils test data - modified";

    let hash1_a = blake3_hash(data1);
    let hash1_b = blake3_hash(data1);
    let hash2 = blake3_hash(data2);

    assert_eq!(hash1_a, hash1_b);
    assert_ne!(hash1_a, hash2);
}

#[test]
fn test_constant_time_eq() {
    let a = [1u8; 32];
    let b = [1u8; 32];
    let c = {
        let mut tmp = [1u8; 32];
        tmp[31] = 2;
        tmp
    };

    assert!(constant_time_eq(&a, &b));
    assert!(!constant_time_eq(&a, &c));
    assert!(!constant_time_eq(&a, &[0u8; 31])); // Length mismatch
}

#[test]
fn test_secure_random_bytes() {
    let bytes1 = secure_random_bytes(32);
    let bytes2 = secure_random_bytes(32);

    assert_eq!(bytes1.len(), 32);
    assert_eq!(bytes2.len(), 32);
    // Extremely low probability of collision
    assert_ne!(bytes1, bytes2);
}

#[test]
fn test_generate_nonce() {
    let nonce1 = generate_nonce();
    let nonce2 = generate_nonce();

    assert_eq!(nonce1.len(), 12);
    assert_eq!(nonce2.len(), 12);
    assert_ne!(nonce1, nonce2);
}

#[test]
fn test_format_duration() {
    assert_eq!(format_duration(500), "500ms");
    assert_eq!(format_duration(1500), "1.50s");
    assert_eq!(format_duration(90000), "1.50m");
    assert_eq!(format_duration(7200000), "2.00h");
    assert_eq!(format_duration(86400000), "24.00h");
}

#[test]
fn test_current_timestamp() {
    let ts1 = current_timestamp();
    sleep(Duration::from_millis(10)).unwrap(); // Sync sleep for test
    let ts2 = current_timestamp();

    assert!(ts2 > ts1);
}

#[test]
fn test_protobuf_serialization_roundtrip() {
    let original = NeuralVoteProto {
        predicted_hash: vec![1u8; 32],
        partial_signature: vec![2u8; 48],
        reputation_score: 0.95,
        tee_attestation: vec![3u8; 64],
        timestamp: 1234567890,
    };

    let bytes = original.to_protobuf().unwrap();
    assert!(!bytes.is_empty());

    let restored = NeuralVoteProto::from_protobuf(&bytes).unwrap();

    assert_eq!(original.predicted_hash, restored.predicted_hash);
    assert_eq!(original.partial_signature, restored.partial_signature);
    assert!((original.reputation_score - restored.reputation_score).abs() < 1e-6);
    assert_eq!(original.tee_attestation, restored.tee_attestation);
    assert_eq!(original.timestamp, restored.timestamp);
}

#[test]
fn test_protobuf_error_cases() {
    // Truncated data
    let bad = vec![0u8; 10];
    assert!(matches!(
        NeuralVoteProto::from_protobuf(&bad),
        Err(SerializeError::Decode(_))
    ));
}

#[tokio::test]
async fn test_metrics_collector_basic_operations() {
    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config).unwrap();

    collector.initialize_metrics().await.unwrap();

    // Record performance metrics
    collector.record_performance_metrics(1000.0, 150.0, 99.9, 500).await.unwrap();

    let json = collector.get_metrics_json().await.unwrap();
    assert!(json.contains("nerv_tps_current"));
    assert!(json.contains("nerv_latency_p99"));
    assert!(json.contains("nerv_uptime_percent"));

    // Timer
    let timer = collector.start_timer("test_timer").await;
    sleep(Duration::from_millis(50)).await;
    timer.record().await.unwrap();

    let json = collector.get_metrics_json().await.unwrap();
    assert!(json.contains("test_timer_histogram"));
}

#[tokio::test]
async fn test_metrics_economic_recording() {
    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config).unwrap();

    collector.initialize_metrics().await.unwrap();

    collector.record_economic_metrics(
        1_000_000.0,
        50_000.0,
        0.75,
        1000,
        Some(1),
    ).await.unwrap();

    let json = collector.get_metrics_json().await.unwrap();
    assert!(json.contains("nerv_stake_total"));
    assert!(json.contains("nerv_rewards_distributed_total"));
    assert!(json.contains("nerv_shapley_values"));
    assert!(json.contains("nerv_gradient_contributions_total"));
}

#[test]
fn test_logging_initialization_and_helpers() {
    let config = LoggingConfig {
        level: LogLevel::Debug,
        format: LogFormat::Json,
        enable_file_logging: false,
        enable_stdout: true,
        enable_tracing: false,
        log_file_path: None,
    };

    // Should not panic
    let _guard = init_logging(&config).unwrap();

    // Structured logging helpers (should not panic)
    log_transaction(&[1u8; 32], 100, "transfer");
    log_batch_update(1, 256, &[0u8; 32]);
    log_tee_attestation("sgx", &[1u8; 32], true);
    log_consensus_event(100, 1, "vote", "67%");
    log_shard_event(1, &[2u8, 3u8], "split", 3400);
    log_performance_metric("tps", 1000.0, "tx/s");

    // Prometheus-style recording (should not panic)
    record_counter("test_counter", 100, &[("type", "test")]);
    record_gauge("test_gauge", 1024.0, &[("node", "1")]);
    record_histogram("test_hist", 150.0, &[("endpoint", "/test")]);
}

#[test]
fn test_error_extensions() {
    let base_err = std::io::Error::new(std::io::ErrorKind::Other, "base error");

    // ContextExt
    let with_context = base_err.context("additional context");
    assert!(format!("{}", with_context).contains("additional context"));

    // ErrorExt (ensure macro)
    let ensured = ensure!(1 + 1 == 2, "math broken");
    assert!(ensured.is_ok());

    let failed = ensure!(1 + 1 == 3, "intentional failure");
    assert!(failed.is_err());
}

#[test]
fn test_validation_helpers() {
    // Valid embedding
    let valid_emb = vec![0.0f64; 512];
    assert!(validate_embedding_size(&valid_emb).is_ok());

    // Invalid size
    let invalid = vec![0.0f64; 511];
    assert!(matches!(
        validate_embedding_size(&invalid),
        Err(ValidationError::InvalidEmbeddingSize)
    ));

    // Valid proof size (<2KB)
    let small_proof = vec![0u8; 1024];
    assert!(validate_proof_size(&small_proof).is_ok());

    // Invalid proof size (>2KB)
    let large_proof = vec![0u8; 2049];
    assert!(matches!(
        validate_proof_size(&large_proof),
        Err(ValidationError::ProofTooLarge)
    ));
}