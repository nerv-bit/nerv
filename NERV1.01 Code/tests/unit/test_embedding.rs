// tests/unit/test_embedding.rs
// ============================================================================
// EMBEDDING MODULE UNIT TESTS
// ============================================================================

use crate::embedding::{
    encoder::{NeuralEncoder, EncoderConfig, QuantizedTensor},
    homomorphism::{TransferDelta, TransferTransaction, HomomorphismVerifier, BatchDelta},
    TransactionBatch, StateUpdate, StateSnapshot, StateMetadata,
    embedding_distance, embedding_similarity, normalize_embedding,
    interpolate_embeddings, add_fixed_embeddings, embedding_to_fixed,
    fixed_to_embedding, EmbeddingVec, HOMO_ERROR_BOUND,
};
use crate::embedding::circuit::FixedPoint32_16;
use rand::{RngCore, thread_rng};
use std::path::PathBuf;
use tempfile::NamedTempFile;

#[test]
fn test_neural_encoder_creation() {
    let config = EncoderConfig::default();
    let encoder = NeuralEncoder::new(config).unwrap();

    assert_eq!(encoder.config.embedding_dim, 512);
    assert_eq!(encoder.layers.len(), 24);
    assert_eq!(encoder.epoch, 0);

    let param_count = encoder.parameter_count();
    assert!(param_count > 0);
    println!("Encoder parameters: {}", param_count);

    let size_mb = encoder.size_mb();
    assert!(size_mb > 0.0);
    assert!(size_mb < 150.0, "Encoder too large: {:.2} MB", size_mb);
}

#[test]
fn test_quantized_tensor_round_trip() {
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0, 10.5, -15.7];
    let quantized = QuantizedTensor::quantize(&data, 8);

    assert_eq!(quantized.bits, 8);
    assert_eq!(quantized.data.len(), data.len());

    let dequantized = quantized.dequantize();
    for i in 0..data.len() {
        // 8-bit quantization error bound ≈ 1/127 ≈ 0.008 of full range
        assert!((data[i] - dequantized[i]).abs() < 0.2);
    }
}

#[test]
fn test_encoder_serialization_round_trip() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_path_buf();

    let encoder = NeuralEncoder::new(EncoderConfig::default()).unwrap();

    // Save
    encoder.save(&path).unwrap();
    assert!(path.exists());

    // Load
    let loaded = NeuralEncoder::load(&path).unwrap();

    // Compare key properties
    assert_eq!(encoder.config, loaded.config);
    assert_eq!(encoder.layers.len(), loaded.layers.len());
    assert_eq!(encoder.epoch, loaded.epoch);
    assert_eq!(encoder.weight_hash, loaded.weight_hash);
}

#[test]
fn test_transfer_delta_computation_and_verification() {
    let mut rng = thread_rng();

    let mut sender_embedding = vec![];
    let mut receiver_embedding = vec![];
    let mut bias_terms = vec![];

    for _ in 0..512 {
        sender_embedding.push(FixedPoint32_16::from_float(rng.gen_range(-1.0..1.0)));
        receiver_embedding.push(FixedPoint32_16::from_float(rng.gen_range(-1.0..1.0)));
        bias_terms.push(FixedPoint32_16::from_float(rng.gen_range(-0.01..0.01)));
    }

    let transaction = TransferTransaction {
        sender: [1u8; 32],
        receiver: [2u8; 32],
        amount: FixedPoint32_16::from_float(100.0),
        nonce: 1,
        timestamp: 1234567890,
        balance_proof: None,
    };

    let delta = TransferDelta::new(
        transaction.clone(),
        &sender_embedding,
        &receiver_embedding,
        &bias_terms,
    )
    .unwrap();

    assert_eq!(delta.delta.len(), 512);

    let verified = delta
        .verify(&sender_embedding, &receiver_embedding, &bias_terms)
        .unwrap();
    assert!(verified);
}

#[test]
fn test_homomorphism_verifier_single_transfer() {
    let mut verifier = HomomorphismVerifier::new();

    let mut old_embedding = vec![];
    let mut new_embedding = vec![];
    let delta_values = vec![FixedPoint32_16::from_float(1.0); 512];

    for i in 0..512 {
        let old_val = FixedPoint32_16::from_float(i as f64);
        let new_val = FixedPoint32_16::from_float(i as f64 + 1.0);
        old_embedding.push(old_val);
        new_embedding.push(new_val);
    }

    let transaction = TransferTransaction {
        sender: [1u8; 32],
        receiver: [2u8; 32],
        amount: FixedPoint32_16::from_float(1.0),
        nonce: 1,
        timestamp: 1234567890,
        balance_proof: None,
    };

    let delta = TransferDelta {
        delta: delta_values,
        transaction,
        error_bound: FixedPoint32_16::from_float(1e-9),
        signature: None,
    };

    let ok = verifier
        .verify_single_transfer(&old_embedding, &delta, &new_embedding)
        .unwrap();
    assert!(ok);
    assert_eq!(verifier.stats().successful_verifications, 1);
}

#[test]
fn test_batch_delta_aggregation() {
    let mut rng = thread_rng();

    let mut deltas = vec![];

    for i in 0..10 {
        let mut sender = vec![];
        let mut receiver = vec![];
        let bias = vec![FixedPoint32_16::from_float(0.0); 512];

        for _ in 0..512 {
            sender.push(FixedPoint32_16::from_float(rng.gen_range(-1.0..1.0)));
            receiver.push(FixedPoint32_16::from_float(rng.gen_range(-1.0..1.0)));
        }

        let tx = TransferTransaction {
            sender: [i as u8; 32],
            receiver: [(i + 1) as u8; 32],
            amount: FixedPoint32_16::from_float(rng.gen_range(1.0..100.0)),
            nonce: i as u64,
            timestamp: 1234567890 + i as u64,
            balance_proof: None,
        };

        let delta = TransferDelta::new(tx, &sender, &receiver, &bias).unwrap();
        deltas.push(delta);
    }

    let batch = BatchDelta::new(deltas).unwrap();

    assert_eq!(batch.transactions.len(), 10);
    assert_eq!(batch.aggregated_delta.len(), 512);

    let compression = batch.compression_ratio(100_000);
    assert!(compression > 100.0, "Got compression {}", compression);
}

#[test]
fn test_transaction_batch_lifecycle() {
    let mut batch = TransactionBatch::new(1, 1);

    assert!(batch.is_empty());
    assert!(!batch.is_full());

    let delta = [0.1f64; 512];
    batch.add_delta(delta).unwrap();

    assert_eq!(batch.size(), 1);
    assert_eq!(batch.aggregated_delta[0], 0.1);

    batch.clear();
    assert!(batch.is_empty());
    assert_eq!(batch.aggregated_delta[0], 0.0);
}

#[test]
fn test_state_update_error_validation() {
    let old = StateSnapshot {
        height: 0,
        shard_id: 1,
        embedding: EmbeddingVec::zeros(),
        embedding_hash: [0u8; 32],
        metadata: StateMetadata {
            epoch: 0,
            timestamp: 0,
            account_count: 100,
            total_balance: 1000,
            proof: None,
        },
    };

    let new = old.clone(); // same embedding → error = 0

    let batch = TransactionBatch::new(1, 1);

    let update = StateUpdate {
        old_state: old,
        new_state: new,
        batch,
        homomorphism_error: 0.0,
        update_proof: vec![],
    };

    assert!(update.validate_error_bound().is_ok());

    // Too large error
    let mut bad = update.clone();
    bad.homomorphism_error = HOMO_ERROR_BOUND * 10.0;
    assert!(bad.validate_error_bound().is_err());
}

#[test]
fn test_embedding_vector_operations() {
    let zeros = EmbeddingVec::zeros();

    let mut values = [0.0; 512];
    for i in 0..512 {
        values[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let alt = EmbeddingVec::new(values.map(|x| x as f64));

    // Addition/subtraction
    let sum = zeros.add(&alt);
    assert!(sum.approx_eq(&alt, 1e-6));

    let diff = sum.sub(&alt);
    assert!(diff.approx_eq(&zeros, 1e-6));

    // Distance & similarity
    assert!(embedding_distance(&zeros, &alt) > 0.0);
    assert!(embedding_similarity(&zeros, &alt).abs() < 1e-6);

    // Normalization
    let mut to_norm = alt.clone();
    normalize_embedding(&mut to_norm);
    let norm = (to_norm.0.iter().map(|v| v * v).sum::<f64>()).sqrt();
    assert!((norm - 1.0).abs() < 1e-6);

    // Interpolation
    let interp = interpolate_embeddings(&zeros, &alt, 0.5);
    for i in 0..512 {
        let expected = if i % 2 == 0 { 0.5 } else { -0.5 };
        assert!((interp.0[i] - expected).abs() < 1e-6);
    }
}

#[test]
fn test_fixed_point_conversion_round_trip() {
    let mut values = [0.0; 512];
    values[0] = 1.5;
    values[1] = -2.3;
    values[2] = 42.12345;

    let embedding = EmbeddingVec::new(values);
    let fixed = embedding_to_fixed(&embedding);
    let restored = fixed_to_embedding(&fixed);

    assert!(embedding.approx_eq(&restored, 1e-4));
}