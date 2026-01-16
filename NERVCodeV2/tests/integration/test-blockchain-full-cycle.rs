Rust// tests/integration/test_blockchain_full_cycle.rs
// ============================================================================
// COMPREHENSIVE FULL-CYCLE BLOCKCHAIN INTEGRATION TEST
// ============================================================================
// This integration test simulates a complete end-to-end blockchain cycle covering:
// 1. Crypto: Post-quantum keygen, signing (Dilithium), encryption (ML-KEM)
// 2. Privacy: Transaction mixing (5-hop onion), TEE attestation generation & verification, VDW issuance & verification
// 3. Network: Encrypted mempool submission, gossip broadcast (mocked)
// 4. Consensus: Block proposal, neural predictor (fallback), voting, commit
// 5. Embedding: Homomorphic delta application and state update
// 6. Sharding: Load metrics recording and prediction cycle
// 
// Special focus: Explicit TEE attestation check and VDW issuance/verification
// Uses simulation modes, fallbacks, and mocks for fast execution without real hardware/models.
// ============================================================================

use nerv_bit::{
    crypto::{CryptoProvider, Dilithium3, MlKem768},
    privacy::{
        PrivacyManager, PrivacyConfig,
        tee::attestation::verify_attestation_report,
        vdw::{VDWGenerator, VDWVerifier},
    },
    network::{NetworkManager, NetworkConfig, EncryptedTransaction, TxPriority},
    consensus::{ConsensusEngine, ConsensusConfig, BlockProposal, Vote},
    embedding::{
        encoder::{NeuralEncoder, EncoderConfig},
        homomorphism::{TransferTransaction, TransferDelta, FixedPoint32_16},
        EmbeddingVec,
    },
    sharding::{ShardingManager, ShardingConfig, ShardLoadMetrics},
};
use rand::{thread_rng, Rng};
use std::sync::Arc;
use blake3::hash;

#[tokio::test]
async fn test_blockchain_full_cycle() {
    // ========================================================================
    // 1. Shared component setup
    // ========================================================================
    let crypto = Arc::new(CryptoProvider::new().unwrap());

    // Privacy manager with TEE simulation enabled
    let mut privacy_cfg = PrivacyConfig::default();
    privacy_cfg.enable_tee = true; // Simulation mode
    privacy_cfg.enable_vdw = true;
    let privacy = PrivacyManager::new(privacy_cfg, crypto.clone()).await.unwrap();

    // Network
    let net_cfg = NetworkConfig::default();
    let network = NetworkManager::new(net_cfg, crypto.clone()).await.unwrap();

    // Consensus (fallback predictor)
    let mut cons_cfg = ConsensusConfig::default();
    cons_cfg.predictor_model_path = "nonexistent.pt".to_string();
    let consensus = ConsensusEngine::new(cons_cfg, crypto.clone()).await.unwrap();

    // Sharding
    let shard_cfg = ShardingConfig::default();
    let sharding = ShardingManager::new(shard_cfg).await.unwrap();

    // Encoder
    let encoder_cfg = EncoderConfig::default();
    let encoder = NeuralEncoder::new(encoder_cfg).unwrap();

    // ========================================================================
    // 2. Crypto: Keygen and transaction signing
    // ========================================================================
    let mut rng = thread_rng();
    let (sig_pk, sig_sk) = Dilithium3::keypair(&mut rng);

    let tx = TransferTransaction {
        sender: [1u8; 32],
        receiver: [2u8; 32],
        amount: FixedPoint32_16::from_float(1000.0),
        nonce: 1,
        timestamp: 1735689600,
        balance_proof: None,
    };

    let tx_bytes = bincode::serialize(&tx).unwrap();
    let signature = Dilithium3::sign(&sig_sk, &tx_bytes).unwrap();

    // ========================================================================
    // 3. Privacy: Mixing + TEE attestation generation
    // ========================================================================
    let mock_hops = vec![[10u8; 32]; 5];
    let onion = privacy.mixer.as_ref().unwrap()
        .build_onion(tx_bytes.clone(), &mock_hops)
        .await
        .unwrap();

    let mut encrypted_tx = EncryptedTransaction {
        data: bincode::serialize(&onion).unwrap(),
        attestation: vec![],
        shard_id: 0,
        hash: hash(&tx_bytes).to_hex().to_string(),
        submission_time: std::time::SystemTime::now(),
        priority: TxPriority::Normal,
    };

    // Generate local TEE attestation for the transaction
    let tee_runtime = privacy.tee_runtime.as_ref().unwrap();
    let attestation_report = tee_runtime.local_attest(&tx_bytes).await.unwrap();
    encrypted_tx.attestation = attestation_report.clone();

    // ========================================================================
    // 4. Privacy: Explicit TEE attestation verification
    // ========================================================================
    let attestation = bincode::deserialize::<nerv_bit::privacy::tee::TEEAttestation>(&attestation_report)
        .expect("Valid attestation format");
    let verify_result = verify_attestation_report(&attestation, Some(&hash(&tx_bytes).into()));
    assert!(verify_result.is_ok(), "TEE attestation verification failed: {:?}", verify_result.err());

    // ========================================================================
    // 5. Network: Mempool submission + gossip
    // ========================================================================
    network.mempool_manager.add_transaction(encrypted_tx.clone()).await.unwrap();

    let pending = network.mempool_manager.get_pending_transactions(0, 10).await.unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].hash, encrypted_tx.hash);

    network.gossip_manager.publish("transactions".to_string(), tx_bytes.clone()).await.unwrap();

    // ========================================================================
    // 6. Consensus: Block proposal + voting + commit
    // ========================================================================
    let proposal = BlockProposal {
        height: 1,
        shard_id: 0,
        transactions: vec![encrypted_tx],
        previous_hash: [0u8; 32],
        proposer: sig_pk.clone(),
        timestamp: 1735689600,
        neural_prediction: None,
    };

    consensus.receive_proposal(proposal.clone()).await.unwrap();

    let vote = Vote {
        voter: sig_pk,
        proposal_hash: hash(&bincode::serialize(&proposal).unwrap()).into(),
        vote_yes: true,
        signature: signature.clone(),
        reputation: 0.95,
    };

    consensus.receive_vote(vote).await.unwrap();
    let committed = consensus.try_commit_block(&proposal).await;
    assert!(committed.is_ok());

    // ========================================================================
    // 7. Embedding: Homomorphic update
    // ========================================================================
    let mut current_embedding = EmbeddingVec::zeros();

    let sender_emb = vec![FixedPoint32_16::from_float(-0.5); 512];
    let receiver_emb = vec![FixedPoint32_16::from_float(0.5); 512];
    let bias = vec![FixedPoint32_16::from_float(0.0); 512];

    let delta = TransferDelta::new(tx.clone(), &sender_emb, &receiver_emb, &bias).unwrap();
    current_embedding = current_embedding.add(&delta.delta.into());

    let new_embedding_hash = hash(&current_embedding.to_bytes());

    // ========================================================================
    // 8. Privacy: VDW issuance for new state
    // ========================================================================
    let mock_recursive_proof = vec![0u8; 256]; // In real: ZK proof of correct encoding
    let current_height = 1;

    let vdw_id = privacy.vdw_generator.as_ref().unwrap()
        .generate_and_store(mock_recursive_proof.clone(), new_embedding_hash.into(), current_height)
        .await
        .unwrap();

    // ========================================================================
    // 9. Privacy: VDW verification (mock delay satisfied)
    // ========================================================================
    let vdw_verified = VDWVerifier::verify(&vdw_id).await.unwrap();
    assert!(vdw_verified, "VDW verification failed for issued witness");

    // ========================================================================
    // 10. Sharding: Record load metrics
    // ========================================================================
    let load_metrics = ShardLoadMetrics {
        shard_id: 0,
        tx_count: 1,
        embedding_count: 1,
        storage_bytes: 4096,
        cpu_load: 0.3,
        timestamp: std::time::SystemTime::now(),
    };

    sharding.record_load_metrics(load_metrics).await.unwrap();
    sharding.run_prediction_cycle().await.unwrap();

    // No split expected with low load
    let pending_ops = sharding.get_pending_operations().await;
    assert!(pending_ops.is_empty());

    // ========================================================================
    // Final success
    // ========================================================================
    println!("Full-cycle blockchain integration test with TEE attestation and VDW checks passed!");
}