Rust// tests/integration/test_economy_ecosystem.rs
// ============================================================================
// COMPREHENSIVE ECONOMY ECOSYSTEM INTEGRATION TEST
// ============================================================================
// This integration test simulates a full end-to-end workflow covering:
// 1. Crypto: PQ keygen, signing, encryption
// 2. Privacy: Transaction encryption, mixing, TEE attestation (mock)
// 3. Network: Mempool submission, gossip
// 4. Consensus: Block proposal, neural predictor (fallback), voting/commit
// 5. Embedding: Homomorphic state update
// 6. Sharding: Load recording, prediction cycle
// 7. Economy: Gradient contribution, FL aggregation, Shapley calculation, rewards
// 8. Tokenomics: Fee processing, block reward distribution, vesting claim
// 
// Uses minimal configs, fallbacks, and mocks for fast execution without
// real hardware/models/network.
// ============================================================================

use nerv_bit::{
    crypto::{CryptoProvider, Dilithium3, MlKem768},
    privacy::{PrivacyManager, PrivacyConfig},
    network::{NetworkManager, NetworkConfig, EncryptedTransaction, TxPriority},
    consensus::{ConsensusEngine, ConsensusConfig, BlockProposal, Vote},
    embedding::{
        encoder::{NeuralEncoder, EncoderConfig},
        homomorphism::{TransferTransaction, TransferDelta},
        EmbeddingVec, FixedPoint32_16,
    },
    sharding::{ShardingManager, ShardingConfig, ShardLoadMetrics},
    economy::{
        EconomyManager, EconomyConfig,
        fl_aggregation::{GradientUpdate, FLAggregator},
        shapley::ShapleyComputer,
        rewards::{RewardDistributor, IndividualReward},
    },
    tokenomics::{TokenomicsEngine, TokenomicsConfig, VestingManager, VestingType},
};
use rand::{thread_rng, Rng};
use std::sync::Arc;
use tokio::time::{sleep, Duration};

const GENESIS_TIME: u64 = 1735689600;

#[tokio::test]
async fn test_economy_ecosystem_workflow() {
    // ========================================================================
    // 1. Shared setup
    // ========================================================================
    let crypto = Arc::new(CryptoProvider::new().unwrap());

    // Privacy (simulation TEE)
    let mut privacy_cfg = PrivacyConfig::default();
    privacy_cfg.enable_tee = true;
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

    // Embedding encoder
    let encoder_cfg = EncoderConfig::default();
    let encoder = NeuralEncoder::new(encoder_cfg).unwrap();

    // Economy
    let mut econ_cfg = EconomyConfig::default();
    econ_cfg.rewards_enabled = true;
    let economy = EconomyManager::new(econ_cfg).await.unwrap();

    // Tokenomics
    let token_cfg = TokenomicsConfig::default();
    let tokenomics = TokenomicsEngine::new(token_cfg, GENESIS_TIME);

    // ========================================================================
    // 2. Crypto + Transaction creation
    // ========================================================================
    let mut rng = thread_rng();
    let (sig_pk, sig_sk) = Dilithium3::keypair(&mut rng);

    let tx = TransferTransaction {
        sender: [1u8; 32],
        receiver: [2u8; 32],
        amount: FixedPoint32_16::from_float(1000.0),
        nonce: 1,
        timestamp: GENESIS_TIME,
        balance_proof: None,
    };

    let tx_bytes = bincode::serialize(&tx).unwrap();
    let signature = Dilithium3::sign(&sig_sk, &tx_bytes).unwrap();

    // ========================================================================
    // 3. Privacy: Mix + TEE attest
    // ========================================================================
    let mock_hops = vec![[10u8; 32]; 5];
    let onion = privacy.mixer.as_ref().unwrap().build_onion(tx_bytes.clone(), &mock_hops).await.unwrap();

    let mut encrypted_tx = EncryptedTransaction {
        data: bincode::serialize(&onion).unwrap(),
        attestation: vec![],
        shard_id: 0,
        hash: blake3::hash(&tx_bytes).to_string(),
        submission_time: std::time::SystemTime::now(),
        priority: TxPriority::Normal,
    };

    let mock_attest = privacy.tee_runtime.as_ref().unwrap().local_attest(&tx_bytes).await.unwrap();
    encrypted_tx.attestation = mock_attest;

    // ========================================================================
    // 4. Network: Mempool + gossip
    // ========================================================================
    network.mempool_manager.add_transaction(encrypted_tx.clone()).await.unwrap();
    let pending = network.mempool_manager.get_pending_transactions(0, 10).await.unwrap();
    assert_eq!(pending.len(), 1);

    network.gossip_manager.publish("transactions".to_string(), tx_bytes.clone()).await.unwrap();

    // ========================================================================
    // 5. Consensus: Propose + vote + commit
    // ========================================================================
    let proposal = BlockProposal {
        height: 1,
        shard_id: 0,
        transactions: vec![encrypted_tx],
        previous_hash: [0u8; 32],
        proposer: sig_pk.clone(),
        timestamp: GENESIS_TIME,
        neural_prediction: None,
    };

    consensus.receive_proposal(proposal.clone()).await.unwrap();

    let vote = Vote {
        voter: sig_pk,
        proposal_hash: blake3::hash(&bincode::serialize(&proposal).unwrap()).into(),
        vote_yes: true,
        signature: signature.clone(),
        reputation: 0.95,
    };

    consensus.receive_vote(vote).await.unwrap();
    let committed = consensus.try_commit_block(&proposal).await;
    assert!(committed.is_ok());

    // ========================================================================
    // 6. Embedding: Homomorphic update
    // ========================================================================
    let mut embedding = EmbeddingVec::zeros();

    let sender_emb = vec![FixedPoint32_16::from_float(-0.5); 512];
    let receiver_emb = vec![FixedPoint32_16::from_float(0.5); 512];
    let bias = vec![FixedPoint32_16::from_float(0.0); 512];

    let delta = TransferDelta::new(tx.clone(), &sender_emb, &receiver_emb, &bias).unwrap();
    embedding = embedding.add(&delta.delta.into());

    // ========================================================================
    // 7. Sharding: Record load
    // ========================================================================
    let load = ShardLoadMetrics {
        shard_id: 0,
        tx_count: 1,
        embedding_count: 1,
        storage_bytes: 2048,
        cpu_load: 0.2,
        timestamp: std::time::SystemTime::now(),
    };
    sharding.record_load_metrics(load).await.unwrap();
    sharding.run_prediction_cycle().await.unwrap();

    // ========================================================================
    // 8. Economy: Gradient contribution + aggregation + Shapley + rewards
    // ========================================================================
    let node_id = "contributor_1".to_string();

    // Submit gradient
    let gradient = GradientUpdate::new(vec![0.01f32; 512], vec![]);
    economy.fl_aggregator.submit_gradient(node_id.clone(), gradient.clone()).await.unwrap();

    // Trigger aggregation (manual for test)
    let global_gradient = economy.fl_aggregator.aggregate_gradients().await.unwrap();

    // Apply to encoder (mock update)
    let new_hash = encoder.apply_gradient_update_mock(&global_gradient);

    // Shapley (single contributor → full share)
    let mut contributions = std::collections::HashMap::new();
    contributions.insert(node_id.clone(), global_gradient.clone().into());
    let shapley = economy.shapley_computer.compute_shapley(&contributions).await.unwrap();

    assert_eq!(shapley.len(), 1);
    assert!((shapley[0].value - 1.0).abs() < 0.1); // ~1.0 for single

    // Distribute rewards
    let block_reward = tokenomics.calculate_block_reward(1);
    let dist = tokenomics.distribute_block_reward(block_reward);

    let rewards = economy.reward_distributor.calculate_rewards(
        shapley.into_iter().map(|v| (v.node_id, v.value)).collect(),
        dist.gradient_pool as f64,
    ).await.unwrap();

    assert_eq!(rewards.len(), 1);
    assert!(rewards[0].final_amount > 0);

    // ========================================================================
    // 9. Tokenomics: Fee + vesting
    // ========================================================================
    let fee = 100u128;
    let fee_result = tokenomics.process_fee(fee);
    assert!(fee_result.burned > 0);

    // Simulate vesting claim (e.g., treasury allocation)
    let vesting_id = tokenomics.vesting_manager.create_schedule(
        [9u8; 20],
        1_000_000_000u128,
        GENESIS_TIME,
        VestingType::Linear,
        Some(4 * 365 * 24 * 60 * 60),
        false,
        None,
        None,
    ).unwrap();

    let claim_time = GENESIS_TIME + 365 * 24 * 60 * 60; // 1 year
    let claimable = tokenomics.vesting_manager.claim_vested(vesting_id, claim_time).unwrap();
    assert!(claimable > 0);

    println!("Full economy ecosystem integration test passed!");
}