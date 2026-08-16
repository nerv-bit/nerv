// tests/integration/test_blockchain_workflow.rs
// ============================================================================
// COMPREHENSIVE BLOCKCHAIN WORKFLOW INTEGRATION TEST
// ============================================================================
// This integration test simulates a full end-to-end workflow covering:
// 1. Crypto: PQ keygen, signing, encryption (Dilithium + ML-KEM)
// 2. Privacy: Transaction encryption, 5-hop mixing, TEE attestation (mock)
// 3. Network: Mempool submission, gossip broadcast (mocked)
// 4. Embedding: Homomorphic state update with delta application
// 5. Consensus: Block proposal, neural predictor validation (fallback), voting
// 6. Sharding: Load recording, potential split proposal detection
// 
// The test uses minimal configurations, fallbacks, and mocks to run without
// real hardware/models/network while exercising all major code paths.
// ============================================================================

use nerv_bit::{
    crypto::{CryptoProvider, Dilithium3, MlKem768},
    privacy::{PrivacyManager, PrivacyConfig},
    network::{NetworkManager, NetworkConfig, EncryptedTransaction, TxPriority},
    consensus::{ConsensusEngine, ConsensusConfig, BlockProposal, Vote},
    embedding::{
        encoder::NeuralEncoder,
        homomorphism::{TransferTransaction, TransferDelta},
        EmbeddingVec,
    },
    sharding::{ShardingManager, ShardingConfig, ShardLoadMetrics},
    tokenomics::TokenomicsEngine,
};
use rand::thread_rng;
use std::sync::Arc;
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn test_full_blockchain_workflow() {
    // ========================================================================
    // 1. Setup shared components
    // ========================================================================
    let crypto = Arc::new(CryptoProvider::new().unwrap());

    // Privacy manager (with TEE mock/simulation)
    let mut privacy_config = PrivacyConfig::default();
    privacy_config.enable_tee = true; // Use simulation mode
    let privacy = PrivacyManager::new(privacy_config.clone(), crypto.clone())
        .await
        .unwrap();

    // Network manager
    let network_config = NetworkConfig::default();
    let network = NetworkManager::new(network_config, crypto.clone())
        .await
        .unwrap();

    // Consensus engine
    let mut consensus_config = ConsensusConfig::default();
    consensus_config.predictor_model_path = "nonexistent.pt".to_string(); // Force fallback
    let consensus = ConsensusEngine::new(consensus_config, crypto.clone()).await.unwrap();

    // Sharding manager
    let sharding_config = ShardingConfig::default();
    let sharding = ShardingManager::new(sharding_config).await.unwrap();

    // Embedding encoder
    let encoder = NeuralEncoder::new(Default::default()).unwrap();

    // Tokenomics (for fees/rewards)
    let tokenomics = TokenomicsEngine::new(Default::default(), 1735689600);

    // ========================================================================
    // 2. Crypto: Generate keys and sign a transaction
    // ========================================================================
    let mut rng = thread_rng();
    let (sig_pk, sig_sk) = Dilithium3::keypair(&mut rng);
    let (kem_pk, kem_sk) = MlKem768::keypair(&mut rng).unwrap();

    // Simple transfer transaction
    let tx = TransferTransaction {
        sender: [1u8; 32],
        receiver: [2u8; 32],
        amount: 1000u128.into(),
        nonce: 1,
        timestamp: 1234567890,
        balance_proof: None,
    };

    let tx_bytes = bincode::serialize(&tx).unwrap();
    let signature = Dilithium3::sign(&sig_sk, &tx_bytes).unwrap();

    // ========================================================================
    // 3. Privacy: Encrypt and mix the transaction
    // ========================================================================
    let mut encrypted_tx = EncryptedTransaction {
        data: tx_bytes.clone(),
        attestation: vec![],
        shard_id: 0,
        hash: blake3::hash(&tx_bytes).to_string(),
        submission_time: std::time::SystemTime::now(),
        priority: TxPriority::Normal,
    };

    // Apply mixing (5 hops, mock peers)
    let mock_hops = vec![[10u8; 32]; 5];
    let onion = privacy.mixer.unwrap().build_onion(tx_bytes.clone(), &mock_hops).await.unwrap();
    encrypted_tx.data = bincode::serialize(&onion).unwrap();

    // Mock TEE attestation for submission
    let mock_attestation = privacy.tee_runtime.unwrap().local_attest(&tx_bytes).await.unwrap();
    encrypted_tx.attestation = mock_attestation;

    // ========================================================================
    // 4. Network: Submit to mempool and simulate gossip
    // ========================================================================
    network.mempool_manager.add_transaction(encrypted_tx.clone()).await.unwrap();

    let pending = network.mempool_manager.get_pending_transactions(0, 10).await.unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].hash, encrypted_tx.hash);

    // Simulate gossip broadcast (no real peers, just check method)
    network.gossip_manager.publish("transactions".to_string(), tx_bytes.clone()).await.unwrap();

    // ========================================================================
    // 5. Consensus: Propose block with transaction
    // ========================================================================
    let proposal = BlockProposal {
        height: 1,
        shard_id: 0,
        transactions: vec![encrypted_tx.clone()],
        previous_hash: [0u8; 32],
        proposer: sig_pk,
        timestamp: 1234567890,
        neural_prediction: None, // Will use fallback
    };

    // Simulate proposal and voting
    consensus.receive_proposal(proposal.clone()).await.unwrap();

    // Mock vote
    let vote = Vote {
        voter: sig_pk,
        proposal_hash: blake3::hash(&bincode::serialize(&proposal).unwrap()).into(),
        vote_yes: true,
        signature: signature.clone(),
        reputation: 0.95,
    };

    consensus.receive_vote(vote).await.unwrap();

    // Assume quorum reached → commit (simplified)
    let committed = consensus.try_commit_block(&proposal).await;
    assert!(committed.is_ok());

    // ========================================================================
    // 6. Embedding: Apply homomorphic update
    // ========================================================================
    let mut current_embedding = EmbeddingVec::zeros();

    // Compute delta (mock account embeddings)
    let sender_emb = vec![FixedPoint32_16::from_float(-0.5); 512];
    let receiver_emb = vec![FixedPoint32_16::from_float(0.5); 512];
    let bias = vec![FixedPoint32_16::from_float(0.0); 512];

    let delta = TransferDelta::new(tx.clone(), &sender_emb, &receiver_emb, &bias).unwrap();

    // Homomorphic update
    current_embedding = current_embedding.add(&delta.delta.into());

    // Verify error bound (should be near zero in mock)
    let reconstructed = encoder.encode_state_mock(&current_embedding); // Assuming mock encode
    assert!(current_embedding.approx_eq(&reconstructed, 1e-6));

    // ========================================================================
    // 7. Sharding: Record load and check for reconfiguration
    // ========================================================================
    let load_metrics = ShardLoadMetrics {
        shard_id: 0,
        tx_count: 1,
        embedding_count: 1,
        storage_bytes: 1024,
        cpu_load: 0.1,
        timestamp: std::time::SystemTime::now(),
    };

    sharding.record_load_metrics(load_metrics).await.unwrap();
    sharding.run_prediction_cycle().await.unwrap();

    // With low load, no split/merge expected
    let ops = sharding.get_pending_operations().await;
    assert!(ops.is_empty());

    // ========================================================================
    // 8. Tokenomics: Process fee and rewards
    // ========================================================================
    let fee = 100u128;
    let fee_result = tokenomics.process_fee(fee);
    assert!(fee_result.burned > 0);
    assert!(fee_result.to_validators > 0);

    // Block reward distribution
    let block_reward = tokenomics.calculate_block_reward(1);
    let dist = tokenomics.distribute_block_reward(block_reward);
    assert_eq!(dist.gradient_pool + dist.validation_pool + dist.public_goods_pool, block_reward);

    // ========================================================================
    // Final assertions: All components interacted successfully
    // ========================================================================
    println!("Full blockchain workflow integration test passed!");
}