//! Reward Distribution for NERV Useful-Work Economy
//! 
//! This module implements the distribution of NERV tokens as rewards for
//! useful-work contributions in the federated learning process. Rewards are
//! distributed based on Shapley values computed from gradient contributions.
//! 
//! Key Features:
//! - Fair distribution based on marginal contributions (Shapley values)
//! - Reputation-weighted rewards to incentivize consistent quality
//! - On-chain distribution with verifiable proofs
//! - Slashing for malicious behavior
//! - Vesting schedules for large rewards
//! 
//! Reward Formula:
//! Reward_i = Total_Reward_Pool × (φ_i × R_i) / Σ(φ_j × R_j)
//! Where φ_i is Shapley value and R_i is reputation score.


use crate::crypto::{CryptoProvider, Dilithium3, ByteSerializable};
use crate::params::{TOTAL_SUPPLY, REWARD_GRADIENT_PERCENT, REWARD_VALIDATION_PERCENT};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug, error};
use chrono::{DateTime, Utc};


/// Reward distributor for NERV tokens
pub struct RewardDistributor {
    /// Configuration parameters
    config: RewardConfig,
    
    /// Cryptographic provider for signatures
    crypto_provider: Arc<CryptoProvider>,
    
    /// Reward records database
    reward_records: RwLock<HashMap<String, Vec<RewardRecord>>>,
    
    /// Vesting schedules
    vesting_schedules: RwLock<HashMap<String, Vec<VestingSchedule>>>,
    
    /// Slashing records
    slashing_records: RwLock<HashMap<String, Vec<SlashingRecord>>>,
    
    /// Distribution queue
    distribution_queue: RwLock<Vec<DistributionTask>>,
    
    /// Metrics collector
    metrics: RewardMetrics,
    
    /// On-chain integration interface
    chain_interface: Option<ChainInterface>,
}


impl RewardDistributor {
    /// Create a new reward distributor
    pub fn new(config: RewardConfig) -> Self {
        let crypto_provider = Arc::new(CryptoProvider::new()
            .expect("Failed to create crypto provider"));
        
        Self {
            config,
            crypto_provider,
            reward_records: RwLock::new(HashMap::new()),
            vesting_schedules: RwLock::new(HashMap::new()),
            slashing_records: RwLock::new(HashMap::new()),
            distribution_queue: RwLock::new(Vec::new()),
            metrics: RewardMetrics::new(),
            chain_interface: None,
        }
    }
    
    /// Distribute rewards for an epoch based on Shapley values
    pub async fn distribute(
        &self,
        reward_allocations: &[RewardAllocation],
        total_reward_pool: u64,
    ) -> Result<DistributionResult, DistributionError> {
        let start_time = std::time::Instant::now();
        
        if reward_allocations.is_empty() {
            return Err(DistributionError::EmptyAllocation);
        }
        
        if total_reward_pool == 0 {
            return Err(DistributionError::ZeroRewardPool);
        }
        
        info!("Distributing {} NERV to {} nodes based on Shapley values",
              total_reward_pool, reward_allocations.len());
        
        // Calculate individual rewards
        let individual_rewards = self.calculate_individual_rewards(
            reward_allocations,
            total_reward_pool,
        ).await?;
        
        // Apply reputation multipliers
        let reputation_adjusted = self.apply_reputation_multipliers(
            &individual_rewards,
        ).await?;
        
        // Apply slashing deductions if any
        let final_rewards = self.apply_slashing_deductions(
            &reputation_adjusted,
        ).await?;
        
        // Create distribution tasks
        let distribution_tasks = self.create_distribution_tasks(
            &final_rewards,
        ).await?;
        
        // Queue distribution tasks
        self.queue_distribution_tasks(distribution_tasks.clone()).await?;
        
        // Execute distribution (on-chain or simulated)
        let distribution_result = self.execute_distribution(
            &distribution_tasks,
        ).await?;
        
        // Create and store reward records
        self.create_reward_records(
            &distribution_result,
            reward_allocations,
        ).await?;
        
        // Update vesting schedules for large rewards
        self.update_vesting_schedules(
            &distribution_result,
        ).await?;
        
        // Update metrics
        self.metrics.record_distribution(
            start_time.elapsed(),
            distribution_result.total_distributed,
            distribution_result.recipients.len(),
        );
        
        info!("Reward distribution completed: {} NERV to {} nodes",
              distribution_result.total_distributed,
              distribution_result.recipients.len());
        
        Ok(distribution_result)
    }
    
    /// Get reward history for a node
    pub async fn get_reward_history(
        &self,
        node_id: &str,
        limit: Option<usize>,
    ) -> Result<Vec<RewardRecord>, DistributionError> {
        let records = self.reward_records.read().await;
        
        if let Some(node_records) = records.get(node_id) {
            let mut history = node_records.clone();
            
            // Sort by timestamp (newest first)
            history.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
            
            // Apply limit
            if let Some(limit) = limit {
                Ok(history.into_iter().take(limit).collect())
            } else {
                Ok(history)
            }
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Get pending rewards for a node (vesting and unvested)
    pub async fn get_pending_rewards(&self, node_id: &str) -> Result<PendingRewards, DistributionError> {
        let mut pending = PendingRewards {
            node_id: node_id.to_string(),
            immediately_available: 0,
            vesting: Vec::new(),
            total_vesting: 0,
            next_vesting_date: None,
        };
        
        // Check reward records for unclaimed rewards
        let records = self.reward_records.read().await;
        if let Some(node_records) = records.get(node_id) {
            for record in node_records {
                if !record.claimed {
                    pending.immediately_available += record.amount;
                }
            }
        }
        
        // Check vesting schedules
        let vesting = self.vesting_schedules.read().await;
        if let Some(node_vesting) = vesting.get(node_id) {
            let now = Utc::now();
            
            for schedule in node_vesting {
                if schedule.end_date > now {
                    // Still vesting
                    let vested_so_far = schedule.vested_amount(now);
                    let remaining = schedule.total_amount - vested_so_far;
                    
                    if remaining > 0 {
                        pending.vesting.push(schedule.clone());
                        pending.total_vesting += remaining;
                        
                        // Update next vesting date
                        if pending.next_vesting_date.is_none() 
                            || schedule.next_vesting_date(now) < pending.next_vesting_date.unwrap() {
                            pending.next_vesting_date = Some(schedule.next_vesting_date(now));
                        }
                    }
                }
            }
        }
        
        Ok(pending)
    }
    
    /// Claim rewards for a node
    pub async fn claim_rewards(
        &self,
        node_id: &str,
        amount: Option<u64>,
        destination_address: &str,
    ) -> Result<ClaimResult, DistributionError> {
        let start_time = std::time::Instant::now();
        
        // Get pending rewards
        let pending = self.get_pending_rewards(node_id).await?;
        
        // Determine claimable amount
        let claimable = pending.immediately_available;
        let mut claim_amount = amount.unwrap_or(claimable);
        
        if claim_amount == 0 {
            return Err(DistributionError::NoRewardsToClaim(node_id.to_string()));
        }
        
        if claim_amount > claimable {
            warn!("Node {} attempting to claim {} NERV but only {} available",
                  node_id, claim_amount, claimable);
            claim_amount = claimable;
        }
        
        // Update reward records
        let mut records = self.reward_records.write().await;
        if let Some(node_records) = records.get_mut(node_id) {
            let mut remaining = claim_amount;
            
            for record in node_records.iter_mut() {
                if remaining == 0 {
                    break;
                }
                
                if !record.claimed {
                    let claim_from_record = record.amount.min(remaining);
                    record.claimed_amount += claim_from_record;
                    record.claimed = record.claimed_amount >= record.amount;
                    remaining -= claim_from_record;
                }
            }
        }
        
        // Create claim transaction
        let claim_tx = self.create_claim_transaction(
            node_id,
            claim_amount,
            destination_address,
        ).await?;
        
        // Sign transaction
        let signed_tx = self.sign_claim_transaction(claim_tx).await?;
        
        // Broadcast to network
        let tx_hash = self.broadcast_claim_transaction(signed_tx).await?;
        
        // Update metrics
        self.metrics.record_claim(start_time.elapsed(), claim_amount);
        
        info!("Node {} claimed {} NERV to address {}, tx: {}",
              node_id, claim_amount, destination_address, tx_hash);
        
        Ok(ClaimResult {
            node_id: node_id.to_string(),
            amount_claimed: claim_amount,
            destination_address: destination_address.to_string(),
            transaction_hash: tx_hash,
            timestamp: Utc::now(),
            pending_vesting: pending.total_vesting,
        })
    }
    
    /// Apply slashing penalty to a node
    pub async fn slash_node(
        &self,
        node_id: &str,
        amount: u64,
        reason: SlashingReason,
        evidence: Vec<u8>,
    ) -> Result<SlashingResult, DistributionError> {
        info!("Slashing node {}: {} NERV for {:?}", node_id, amount, reason);
        
        // Create slashing record
        let record = SlashingRecord {
            node_id: node_id.to_string(),
            amount,
            reason: reason.clone(),
            evidence,
            timestamp: Utc::now(),
            applied: false,
        };
        
        // Store record
        let mut slashing_records = self.slashing_records.write().await;
        let node_records = slashing_records.entry(node_id.to_string())
            .or_insert_with(Vec::new);
        node_records.push(record.clone());
        
        // Apply slashing to pending rewards
        let pending = self.get_pending_rewards(node_id).await?;
        let slashable = pending.immediately_available;
        
        if slashable >= amount {
            // Can slash from immediately available rewards
            self.apply_immediate_slashing(node_id, amount).await?;
            record.applied = true;
        } else {
            // Need to slash from future rewards
            self.apply_future_slashing(node_id, amount - slashable).await?;
            record.applied = true;
        }
        
        // Update reputation (severe penalty for slashing)
        self.update_reputation_after_slashing(node_id, &reason).await?;
        
        // Emit slashing event
        self.emit_slashing_event(&record).await?;
        
        // Update metrics
        self.metrics.record_slashing(amount);
        
        Ok(SlashingResult {
            node_id: node_id.to_string(),
            amount_slashed: amount,
            reason,
            timestamp: Utc::now(),
            evidence_hash: self.hash_evidence(&record.evidence).await?,
        })
    }
    
    /// Get distribution statistics
    pub async fn get_statistics(&self) -> DistributionStatistics {
        self.metrics.get_statistics()
    }
    
    // Internal implementation methods
    
    async fn calculate_individual_rewards(
        &self,
        allocations: &[RewardAllocation],
        total_pool: u64,
    ) -> Result<Vec<IndividualReward>, DistributionError> {
        let mut rewards = Vec::with_capacity(allocations.len());
        
        // Calculate sum of weighted Shapley values
        let total_weight: f64 = allocations.iter()
            .map(|a| a.reward_share)
            .sum();
        
        if total_weight <= 0.0 {
            return Err(DistributionError::InvalidAllocation(
                "Total reward share weight is zero or negative".to_string()
            ));
        }
        
        // Calculate individual rewards
        for allocation in allocations {
            let share = allocation.reward_share / total_weight;
            let amount = (total_pool as f64 * share) as u64;
            
            // Apply minimum reward threshold
            let final_amount = if amount < self.config.min_reward_amount {
                if self.config.enforce_min_reward {
                    0
                } else {
                    self.config.min_reward_amount
                }
            } else {
                amount
            };
            
            if final_amount > 0 {
                rewards.push(IndividualReward {
                    node_id: allocation.node_id.clone(),
                    shapley_value: allocation.shapley_value,
                    reputation_score: allocation.reputation_score,
                    raw_amount: amount,
                    final_amount,
                    share_percentage: share * 100.0,
                });
            }
        }
        
        // Sort by amount (descending)
        rewards.sort_by(|a, b| b.final_amount.cmp(&a.final_amount));
        
        Ok(rewards)
    }
    
    async fn apply_reputation_multipliers(
        &self,
        rewards: &[IndividualReward],
    ) -> Result<Vec<IndividualReward>, DistributionError> {
        let mut adjusted = Vec::with_capacity(rewards.len());
        
        for reward in rewards {
            // Reputation multiplier: 0.5 + reputation_score
            // This gives range [0.5, 1.5] based on reputation
            let rep_multiplier = 0.5 + reward.reputation_score.clamp(0.0, 1.0);
            
            // Apply multiplier with smoothing
            let adjusted_amount = (reward.final_amount as f64 * rep_multiplier) as u64;
            
            adjusted.push(IndividualReward {
                final_amount: adjusted_amount,
                ..reward.clone()
            });
        }
        
        Ok(adjusted)
    }
    
    async fn apply_slashing_deductions(
        &self,
        rewards: &[IndividualReward],
    ) -> Result<Vec<IndividualReward>, DistributionError> {
        let mut deducted = Vec::with_capacity(rewards.len());
        
        for reward in rewards {
            // Check for pending slashing
            let slashing_records = self.slashing_records.read().await;
            let total_slashing: u64 = slashing_records.get(&reward.node_id)
                .map(|records| records.iter()
                    .filter(|r| !r.applied)
                    .map(|r| r.amount)
                    .sum())
                .unwrap_or(0);
            
            let final_amount = if total_slashing > 0 {
                reward.final_amount.saturating_sub(total_slashing)
            } else {
                reward.final_amount
            };
            
            if final_amount > 0 {
                deducted.push(IndividualReward {
                    final_amount,
                    ..reward.clone()
                });
            }
        }
        
        Ok(deducted)
    }
    
    async fn create_distribution_tasks(
        &self,
        rewards: &[IndividualReward],
    ) -> Result<Vec<DistributionTask>, DistributionError> {
        let mut tasks = Vec::with_capacity(rewards.len());
        
        for reward in rewards {
            // Check if reward needs vesting
            let needs_vesting = reward.final_amount >= self.config.vesting_threshold;
            
            if needs_vesting {
                // Create vesting schedule
                let vesting_task = self.create_vesting_task(reward).await?;
                tasks.push(vesting_task);
            } else {
                // Immediate distribution
                let immediate_task = DistributionTask {
                    node_id: reward.node_id.clone(),
                    amount: reward.final_amount,
                    distribution_type: DistributionType::Immediate,
                    status: DistributionStatus::Pending,
                    created_at: Utc::now(),
                };
                tasks.push(immediate_task);
            }
        }
        
        Ok(tasks)
    }
    
    async fn create_vesting_task(
        &self,
        reward: &IndividualReward,
    ) -> Result<DistributionTask, DistributionError> {
        // Calculate vesting schedule
        let vesting_months = if reward.final_amount >= self.config.long_vesting_threshold {
            self.config.long_vesting_months
        } else {
            self.config.standard_vesting_months
        };
        
        let cliff_months = self.config.cliff_months;
        
        Ok(DistributionTask {
            node_id: reward.node_id.clone(),
            amount: reward.final_amount,
            distribution_type: DistributionType::Vested {
                cliff_months,
                vesting_months,
                start_date: Utc::now(),
            },
            status: DistributionStatus::Pending,
            created_at: Utc::now(),
        })
    }
    
    async fn queue_distribution_tasks(
        &self,
        tasks: Vec<DistributionTask>,
    ) -> Result<(), DistributionError> {
        let mut queue = self.distribution_queue.write().await;
        queue.extend(tasks);
        
        // Sort by amount (largest first for visibility)
        queue.sort_by(|a, b| b.amount.cmp(&a.amount));
        
        Ok(())
    }
    
    async fn execute_distribution(
        &self,
        tasks: &[DistributionTask],
    ) -> Result<DistributionResult, DistributionError> {
        let mut recipients = Vec::with_capacity(tasks.len());
        let mut total_distributed = 0u64;
        
        for task in tasks {
            match &task.distribution_type {
                DistributionType::Immediate => {
                    // Distribute immediately
                    let recipient = self.distribute_immediate(task).await?;
                    recipients.push(recipient);
                    total_distributed += task.amount;
                }
                DistributionType::Vested { cliff_months, vesting_months, start_date } => {
                    // Create vesting schedule
                    let recipient = self.create_vesting_schedule(task, *cliff_months, *vesting_months, *start_date).await?;
                    recipients.push(recipient);
                    total_distributed += task.amount;
                }
            }
        }
        
        // Sort recipients by amount (descending)
        recipients.sort_by(|a, b| b.amount_nerv.cmp(&a.amount_nerv));
        
        Ok(DistributionResult {
            epoch: self.get_current_epoch().await,
            total_distributed,
            timestamp: Utc::now(),
            recipients,
        })
    }
    
    async fn distribute_immediate(
        &self,
        task: &DistributionTask,
    ) -> Result<RecipientInfo, DistributionError> {
        // Get node's default address
        let address = self.get_node_address(&task.node_id).await?;
        
        // Create on-chain transaction
        let tx_hash = if let Some(chain) = &self.chain_interface {
            chain.transfer(&task.node_id, &address, task.amount).await?
        } else {
            // Simulated transaction
            format!("SIM_{}_{}", task.node_id, Utc::now().timestamp())
        };
        
        Ok(RecipientInfo {
            node_id: task.node_id.clone(),
            amount_nerv: task.amount,
            share_percent: 0.0, // Will be calculated later
            reputation_score: self.get_node_reputation(&task.node_id).await.unwrap_or(0.5),
            tx_hash: Some(tx_hash),
        })
    }
    
    async fn create_vesting_schedule(
        &self,
        task: &DistributionTask,
        cliff_months: u32,
        vesting_months: u32,
        start_date: DateTime<Utc>,
    ) -> Result<RecipientInfo, DistributionError> {
        // Create vesting schedule
        let schedule = VestingSchedule {
            node_id: task.node_id.clone(),
            total_amount: task.amount,
            start_date,
            cliff_months,
            vesting_months,
            claimed_amount: 0,
            created_at: Utc::now(),
        };
        
        // Store schedule
        let mut vesting_schedules = self.vesting_schedules.write().await;
        let node_schedules = vesting_schedules.entry(task.node_id.clone())
            .or_insert_with(Vec::new);
        node_schedules.push(schedule);
        
        // Immediate cliff amount (if any)
        let cliff_amount = if cliff_months == 0 {
            // No cliff, immediate vesting
            (task.amount as f64 / vesting_months as f64) as u64
        } else {
            0
        };
        
        // Distribute cliff amount if any
        let tx_hash = if cliff_amount > 0 {
            let address = self.get_node_address(&task.node_id).await?;
            if let Some(chain) = &self.chain_interface {
                Some(chain.transfer(&task.node_id, &address, cliff_amount).await?)
            } else {
                Some(format!("VEST_CLIFF_{}_{}", task.node_id, Utc::now().timestamp()))
            }
        } else {
            None
        };
        
        Ok(RecipientInfo {
            node_id: task.node_id.clone(),
            amount_nerv: task.amount,
            share_percent: 0.0,
            reputation_score: self.get_node_reputation(&task.node_id).await.unwrap_or(0.5),
            tx_hash,
        })
    }
    
    async fn create_reward_records(
        &self,
        distribution: &DistributionResult,
        allocations: &[RewardAllocation],
    ) -> Result<(), DistributionError> {
        let mut records = self.reward_records.write().await;
        
        for recipient in &distribution.recipients {
            // Find corresponding allocation for Shapley value
            let shapley_value = allocations.iter()
                .find(|a| a.node_id == recipient.node_id)
                .map(|a| a.shapley_value)
                .unwrap_or(0.0);
            
            let record = RewardRecord {
                node_id: recipient.node_id.clone(),
                amount: recipient.amount_nerv,
                shapley_value,
                reputation_score: recipient.reputation_score,
                epoch: distribution.epoch,
                timestamp: distribution.timestamp,
                transaction_hash: recipient.tx_hash.clone(),
                claimed: false,
                claimed_amount: 0,
            };
            
            let node_records = records.entry(recipient.node_id.clone())
                .or_insert_with(Vec::new);
            node_records.push(record);
        }
        
        Ok(())
    }
    
    async fn update_vesting_schedules(
        &self,
        distribution: &DistributionResult,
    ) -> Result<(), DistributionError> {
        // Update any vesting schedules that have vested
        let now = Utc::now();
        let mut vesting_schedules = self.vesting_schedules.write().await;
        
        for schedules in vesting_schedules.values_mut() {
            for schedule in schedules.iter_mut() {
                if schedule.is_vested(now) && schedule.claimed_amount < schedule.total_amount {
                    // Calculate vested amount
                    let vested = schedule.vested_amount(now);
                    let claimable = vested - schedule.claimed_amount;
                    
                    if claimable > 0 {
                        // Auto-claim vested amount
                        schedule.claimed_amount = vested;
                        
                        // Create reward record for claimed amount
                        let mut records = self.reward_records.write().await;
                        let node_records = records.entry(schedule.node_id.clone())
                            .or_insert_with(Vec::new);
                        
                        node_records.push(RewardRecord {
                            node_id: schedule.node_id.clone(),
                            amount: claimable,
                            shapley_value: 0.0, // Unknown for vesting
                            reputation_score: 0.5,
                            epoch: distribution.epoch,
                            timestamp: now,
                            transaction_hash: Some(format!("VEST_CLAIM_{}", now.timestamp())),
                            claimed: false,
                            claimed_amount: 0,
                        });
                    }
                }
            }
        }
        
        Ok(())
    }
    
    async fn create_claim_transaction(
        &self,
        node_id: &str,
        amount: u64,
        destination: &str,
    ) -> Result<ClaimTransaction, DistributionError> {
        Ok(ClaimTransaction {
            node_id: node_id.to_string(),
            amount,
            destination_address: destination.to_string(),
            nonce: self.generate_nonce().await?,
            timestamp: Utc::now(),
            fee: self.calculate_fee(amount).await?,
        })
    }
    
    async fn sign_claim_transaction(
        &self,
        tx: ClaimTransaction,
    ) -> Result<SignedTransaction, DistributionError> {
        // Serialize transaction
        let tx_bytes = bincode::serialize(&tx)
            .map_err(|e| DistributionError::SerializationError(e.to_string()))?;
        
        // Sign with node's key (in production, node would sign)
        // For simulation, use a mock signature
        let signature = vec![0u8; 64]; // Mock signature
        
        Ok(SignedTransaction {
            transaction: tx,
            signature,
            public_key: vec![], // Would be node's public key
        })
    }
    
    async fn broadcast_claim_transaction(
        &self,
        tx: SignedTransaction,
    ) -> Result<String, DistributionError> {
        if let Some(chain) = &self.chain_interface {
            chain.broadcast_transaction(tx).await
        } else {
            // Simulated transaction hash
            let mut hasher = blake3::Hasher::new();
            hasher.update(&bincode::serialize(&tx.transaction)
                .map_err(|e| DistributionError::SerializationError(e.to_string()))?);
            hasher.update(&tx.signature);
            
            let hash = hasher.finalize();
            Ok(hex::encode(hash.as_bytes()))
        }
    }
    
    async fn apply_immediate_slashing(
        &self,
        node_id: &str,
        amount: u64,
    ) -> Result<(), DistributionError> {
        // Deduct from unclaimed rewards
        let mut records = self.reward_records.write().await;
        
        if let Some(node_records) = records.get_mut(node_id) {
            let mut remaining = amount;
            
            for record in node_records.iter_mut() {
                if remaining == 0 {
                    break;
                }
                
                if !record.claimed {
                    let unclaimed = record.amount - record.claimed_amount;
                    let slash_from_record = unclaimed.min(remaining);
                    
                    record.amount -= slash_from_record;
                    remaining -= slash_from_record;
                    
                    if record.amount == 0 {
                        record.claimed = true;
                    }
                }
            }
            
            if remaining > 0 {
                warn!("Could not fully slash node {}: {} NERV remaining",
                      node_id, remaining);
            }
        }
        
        Ok(())
    }
    
    async fn apply_future_slashing(
        &self,
        node_id: &str,
        amount: u64,
    ) -> Result<(), DistributionError> {
        // Future slashing will be applied to next rewards
        // This is already handled by apply_slashing_deductions
        
        // For now, just log
        warn!("Node {} has future slashing liability of {} NERV",
              node_id, amount);
        
        Ok(())
    }
    
    async fn update_reputation_after_slashing(
        &self,
        node_id: &str,
        reason: &SlashingReason,
    ) -> Result<(), DistributionError> {
        // Severe reputation penalty for slashing
        // In production: update in reputation system
        // For now, just log
        
        let penalty_severity = match reason {
            SlashingReason::ByzantineBehavior => 0.8, // 80% penalty
            SlashingReason::DoubleSigning => 1.0,     // 100% penalty
            SlashingReason::InvalidGradient => 0.5,   // 50% penalty
            SlashingReason::Other(_) => 0.3,          // 30% penalty
        };
        
        warn!("Node {} reputation penalized by {:.0}% for {:?}",
              node_id, penalty_severity * 100.0, reason);
        
        Ok(())
    }
    
    async fn emit_slashing_event(
        &self,
        record: &SlashingRecord,
    ) -> Result<(), DistributionError> {
        // Emit on-chain event
        // In production: emit smart contract event
        
        info!("Slashing event: node {} slashed {} NERV for {:?}",
              record.node_id, record.amount, record.reason);
        
        Ok(())
    }
    
    async fn hash_evidence(&self, evidence: &[u8]) -> Result<String, DistributionError> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(evidence);
        let hash = hasher.finalize();
        
        Ok(hex::encode(hash.as_bytes()))
    }
    
    // Helper methods
    
    async fn get_current_epoch(&self) -> u64 {
        // In production: query from blockchain
        // For simulation: use timestamp-based epoch
        Utc::now().timestamp() as u64 / (30 * 24 * 60 * 60) // 30-day epochs
    }
    
    async fn get_node_address(&self, node_id: &str) -> Result<String, DistributionError> {
        // In production: query from node registry
        // For simulation: generate deterministic address
        Ok(format!("nerv_addr_{}", node_id))
    }
    
    async fn get_node_reputation(&self, node_id: &str) -> Result<f64, DistributionError> {
        // In production: query from reputation system
        // For simulation: return default
        Ok(0.5)
    }
    
    async fn generate_nonce(&self) -> Result<u64, DistributionError> {
        // In production: use monotonic counter
        // For simulation: use timestamp
        Ok(Utc::now().timestamp_millis() as u64)
    }
    
    async fn calculate_fee(&self, amount: u64) -> Result<u64, DistributionError> {
        // Fixed fee or percentage
        let fee = (amount as f64 * 0.001).max(1.0) as u64; // 0.1% with minimum 1 NERV
        Ok(fee)
    }
}


/// Reward configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardConfig {
    /// Percentage of rewards for gradient contributions (from whitepaper: 60%)
    pub gradient_percent: f64,
    
    /// Percentage for honest validation (from whitepaper: 30%)
    pub validation_percent: f64,
    
    /// Percentage for public goods grants (from whitepaper: 10%)
    pub public_goods_percent: f64,
    
    /// Epoch duration in days (from whitepaper: 30 days)
    pub epoch_duration_days: u32,
    
    /// Minimum reward amount (to avoid dust)
    pub min_reward_amount: u64,
    
    /// Enforce minimum reward (if false, give min_reward_amount even if calculated less)
    pub enforce_min_reward: bool,
    
    /// Reputation decay factor (0.0 to 1.0)
    pub reputation_decay_factor: f64,
    
    /// Vesting threshold (rewards above this get vested)
    pub vesting_threshold: u64,
    
    /// Long vesting threshold (rewards above this get longer vesting)
    pub long_vesting_threshold: u64,
    
    /// Standard vesting period in months
    pub standard_vesting_months: u32,
    
    /// Long vesting period in months
    pub long_vesting_months: u32,
    
    /// Cliff period in months (no vesting during cliff)
    pub cliff_months: u32,
    
    /// Maximum slashing percentage per incident (0.0 to 1.0)
    pub max_slashing_percentage: f64,
}


impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            gradient_percent: REWARD_GRADIENT_PERCENT,
            validation_percent: REWARD_VALIDATION_PERCENT,
            public_goods_percent: 0.10,
            epoch_duration_days: 30,
            min_reward_amount: 1, // 1 NERV minimum
            enforce_min_reward: true,
            reputation_decay_factor: 0.95,
            vesting_threshold: 1_000_000, // 1M NERV threshold for vesting
            long_vesting_threshold: 10_000_000, // 10M NERV threshold for long vesting
            standard_vesting_months: 12, // 1 year vesting
            long_vesting_months: 48, // 4 years vesting
            cliff_months: 3, // 3 month cliff
            max_slashing_percentage: 0.05, // 5% maximum slashing
        }
    }
}


/// Reward record for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardRecord {
    /// Node identifier
    pub node_id: String,
    
    /// Reward amount in NERV
    pub amount: u64,
    
    /// Shapley value for this reward
    pub shapley_value: f64,
    
    /// Reputation score at time of reward
    pub reputation_score: f64,
    
    /// Epoch number
    pub epoch: u64,
    
    /// Timestamp of reward
    pub timestamp: DateTime<Utc>,
    
    /// Transaction hash (if on-chain)
    pub transaction_hash: Option<String>,
    
    /// Whether reward has been claimed
    pub claimed: bool,
    
    /// Amount already claimed (for partial claims)
    pub claimed_amount: u64,
}


/// Vesting schedule for large rewards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VestingSchedule {
    /// Node identifier
    pub node_id: String,
    
    /// Total vesting amount
    pub total_amount: u64,
    
    /// Start date of vesting
    pub start_date: DateTime<Utc>,
    
    /// Cliff period in months
    pub cliff_months: u32,
    
    /// Vesting period in months
    pub vesting_months: u32,
    
    /// Amount already claimed
    pub claimed_amount: u64,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}


impl VestingSchedule {
    /// Check if schedule has vested at given time
    pub fn is_vested(&self, now: DateTime<Utc>) -> bool {
        let months_since_start = ((now - self.start_date).num_days() as f64 / 30.44) as u32;
        months_since_start >= self.cliff_months
    }
    
    /// Calculate vested amount at given time
    pub fn vested_amount(&self, now: DateTime<Utc>) -> u64 {
        if !self.is_vested(now) {
            return 0;
        }
        
        let months_since_start = ((now - self.start_date).num_days() as f64 / 30.44) as u32;
        let months_vested = months_since_start.saturating_sub(self.cliff_months);
        let total_vesting_months = self.vesting_months.saturating_sub(self.cliff_months);
        
        if total_vesting_months == 0 {
            return self.total_amount;
        }
        
        let vested_fraction = months_vested as f64 / total_vesting_months as f64;
        let vested = (self.total_amount as f64 * vested_fraction.min(1.0)) as u64;
        
        vested
    }
    
    /// Get next vesting date
    pub fn next_vesting_date(&self, now: DateTime<Utc>) -> DateTime<Utc> {
        if !self.is_vested(now) {
            // Next date is end of cliff
            return self.start_date + chrono::Duration::days(self.cliff_months as i64 * 30);
        }
        
        let months_since_start = ((now - self.start_date).num_days() as f64 / 30.44) as u32;
        let months_vested = months_since_start.saturating_sub(self.cliff_months);
        
        if months_vested >= self.vesting_months {
            // Fully vested
            return self.start_date + chrono::Duration::days(self.vesting_months as i64 * 30);
        }
        
        // Next vesting is at the next month boundary
        let next_month = months_since_start + 1;
        self.start_date + chrono::Duration::days(next_month as i64 * 30)
    }
    
    /// Calculate end date of vesting
    pub fn end_date(&self) -> DateTime<Utc> {
        self.start_date + chrono::Duration::days(self.vesting_months as i64 * 30)
    }
}


/// Slashing record for penalized nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashingRecord {
    /// Node identifier
    pub node_id: String,
    
    /// Slashing amount in NERV
    pub amount: u64,
    
    /// Reason for slashing
    pub reason: SlashingReason,
    
    /// Evidence of violation
    pub evidence: Vec<u8>,
    
    /// Timestamp of slashing
    pub timestamp: DateTime<Utc>,
    
    /// Whether slashing has been applied
    pub applied: bool,
}


/// Slashing reason
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlashingReason {
    /// Byzantine behavior (malicious gradient submission)
    ByzantineBehavior,
    
    /// Double signing in consensus
    DoubleSigning,
    
    /// Invalid gradient submission
    InvalidGradient,
    
    /// Other reasons
    Other(String),
}


/// Distribution task for reward processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionTask {
    /// Node identifier
    pub node_id: String,
    
    /// Amount to distribute
    pub amount: u64,
    
    /// Distribution type
    pub distribution_type: DistributionType,
    
    /// Current status
    pub status: DistributionStatus,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}


/// Distribution type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionType {
    /// Immediate distribution
    Immediate,
    
    /// Vested distribution with cliff
    Vested {
        cliff_months: u32,
        vesting_months: u32,
        start_date: DateTime<Utc>,
    },
}


/// Distribution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionStatus {
    /// Pending distribution
    Pending,
    
    /// Distribution completed
    Completed,
    
    /// Distribution failed
    Failed(String),
}


/// Individual reward calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndividualReward {
    /// Node identifier
    pub node_id: String,
    
    /// Shapley value
    pub shapley_value: f64,
    
    /// Reputation score
    pub reputation_score: f64,
    
    /// Raw calculated amount
    pub raw_amount: u64,
    
    /// Final amount after adjustments
    pub final_amount: u64,
    
    /// Share percentage
    pub share_percentage: f64,
}


/// Distribution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionResult {
    /// Epoch number
    pub epoch: u64,
    
    /// Total NERV distributed
    pub total_distributed: u64,
    
    /// Distribution timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Individual recipient information
    pub recipients: Vec<RecipientInfo>,
}


/// Recipient information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipientInfo {
    /// Node identifier
    pub node_id: String,
    
    /// NERV amount received
    pub amount_nerv: u64,
    
    /// Reward share percentage
    pub share_percent: f64,
    
    /// Reputation score at distribution
    pub reputation_score: f64,
    
    /// Transaction hash (if on-chain)
    pub tx_hash: Option<String>,
}


/// Pending rewards for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingRewards {
    /// Node identifier
    pub node_id: String,
    
    /// Immediately available rewards
    pub immediately_available: u64,
    
    /// Vesting schedules
    pub vesting: Vec<VestingSchedule>,
    
    /// Total vesting amount
    pub total_vesting: u64,
    
    /// Next vesting date
    pub next_vesting_date: Option<DateTime<Utc>>,
}


/// Claim transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClaimTransaction {
    /// Node identifier
    pub node_id: String,
    
    /// Claim amount
    pub amount: u64,
    
    /// Destination address
    pub destination_address: String,
    
    /// Transaction nonce
    pub nonce: u64,
    
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Transaction fee
    pub fee: u64,
}


/// Signed transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SignedTransaction {
    /// Transaction data
    pub transaction: ClaimTransaction,
    
    /// Cryptographic signature
    pub signature: Vec<u8>,
    
    /// Public key
    pub public_key: Vec<u8>,
}


/// Claim result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimResult {
    /// Node identifier
    pub node_id: String,
    
    /// Amount claimed
    pub amount_claimed: u64,
    
    /// Destination address
    pub destination_address: String,
    
    /// Transaction hash
    pub transaction_hash: String,
    
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Pending vesting amount
    pub pending_vesting: u64,
}


/// Slashing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashingResult {
    /// Node identifier
    pub node_id: String,
    
    /// Amount slashed
    pub amount_slashed: u64,
    
    /// Slashing reason
    pub reason: SlashingReason,
    
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Evidence hash
    pub evidence_hash: String,
}


/// Distribution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionStatistics {
    /// Total NERV distributed
    pub total_distributed: u64,
    
    /// Total claims processed
    pub total_claims: u64,
    
    /// Total slashing applied
    pub total_slashing: u64,
    
    /// Average distribution time (ms)
    pub avg_distribution_time_ms: f64,
    
    /// Average claim time (ms)
    pub avg_claim_time_ms: f64,
    
    /// Active nodes with rewards
    pub active_nodes: usize,
    
    /// Total vesting schedules
    pub total_vesting_schedules: usize,
}


/// Chain interface for on-chain operations
trait ChainInterface {
    /// Transfer NERV tokens
    async fn transfer(&self, from: &str, to: &str, amount: u64) -> Result<String, DistributionError>;
    
    /// Broadcast transaction
    async fn broadcast_transaction(&self, tx: SignedTransaction) -> Result<String, DistributionError>;
}


/// Reward allocation for distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardAllocation {
    /// Node identifier
    pub node_id: String,
    
    /// Computed Shapley value
    pub shapley_value: f64,
    
    /// Reputation score at distribution time
    pub reputation_score: f64,
    
    /// Final reward share (0.0 to 1.0)
    pub reward_share: f64,
    
    /// Estimated NERV amount (before final calculation)
    pub estimated_nerv: u64,
}


/// Reward metrics collector
#[derive(Debug, Clone, Default)]
struct RewardMetrics {
    /// Total NERV distributed
    total_distributed: u64,
    
    /// Total claims processed
    total_claims: u64,
    
    /// Total slashing applied
    total_slashing: u64,
    
    /// Average distribution time (ms)
    avg_distribution_time_ms: f64,
    
    /// Average claim time (ms)
    avg_claim_time_ms: f64,
    
    /// Distribution count
    distribution_count: u64,
    
    /// Claim count
    claim_count: u64,
}


impl RewardMetrics {
    fn new() -> Self {
        Self::default()
    }
    
    fn record_distribution(&mut self, duration: std::time::Duration, amount: u64, recipients: usize) {
        self.total_distributed += amount;
        self.distribution_count += 1;
        
        // Update moving average
        let new_time = duration.as_millis() as f64;
        self.avg_distribution_time_ms = 0.9 * self.avg_distribution_time_ms + 0.1 * new_time;
    }
    
    fn record_claim(&mut self, duration: std::time::Duration, amount: u64) {
        self.total_claims += amount;
        self.claim_count += 1;
        
        // Update moving average
        let new_time = duration.as_millis() as f64;
        self.avg_claim_time_ms = 0.9 * self.avg_claim_time_ms + 0.1 * new_time;
    }
    
    fn record_slashing(&mut self, amount: u64) {
        self.total_slashing += amount;
    }
    
    fn get_statistics(&self) -> DistributionStatistics {
        DistributionStatistics {
            total_distributed: self.total_distributed,
            total_claims: self.total_claims,
            total_slashing: self.total_slashing,
            avg_distribution_time_ms: self.avg_distribution_time_ms,
            avg_claim_time_ms: self.avg_claim_time_ms,
            active_nodes: 0, // Would be calculated from records
            total_vesting_schedules: 0, // Would be calculated from schedules
        }
    }
}


/// Distribution errors
#[derive(Debug, thiserror::Error)]
pub enum DistributionError {
    #[error("Empty reward allocation")]
    EmptyAllocation,
    
    #[error("Zero reward pool")]
    ZeroRewardPool,
    
    #[error("Invalid allocation: {0}")]
    InvalidAllocation(String),
    
    #[error("No rewards to claim for node {0}")]
    NoRewardsToClaim(String),
    
    #[error("Cryptographic error: {0}")]
    CryptoError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Chain interface error: {0}")]
    ChainInterfaceError(String),
    
    #[error("Vesting error: {0}")]
    VestingError(String),
    
    #[error("Slashing error: {0}")]
    SlashingError(String),
    
    #[error("Insufficient funds: {0}")]
    InsufficientFunds(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Node not found: {0}")]
    NodeNotFound(String),
}


// Unit tests
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_reward_config_default() {
        let config = RewardConfig::default();
        
        assert_eq!(config.gradient_percent, REWARD_GRADIENT_PERCENT);
        assert_eq!(config.validation_percent, REWARD_VALIDATION_PERCENT);
        assert_eq!(config.public_goods_percent, 0.10);
        assert_eq!(config.epoch_duration_days, 30);
        assert_eq!(config.min_reward_amount, 1);
        assert!(config.enforce_min_reward);
        assert_eq!(config.reputation_decay_factor, 0.95);
        assert_eq!(config.vesting_threshold, 1_000_000);
        assert_eq!(config.long_vesting_threshold, 10_000_000);
        assert_eq!(config.standard_vesting_months, 12);
        assert_eq!(config.long_vesting_months, 48);
        assert_eq!(config.cliff_months, 3);
        assert_eq!(config.max_slashing_percentage, 0.05);
    }
    
    #[test]
    fn test_vesting_schedule_calculations() {
        let start_date = Utc::now();
        let schedule = VestingSchedule {
            node_id: "test_node".to_string(),
            total_amount: 1_000_000,
            start_date,
            cliff_months: 3,
            vesting_months: 12,
            claimed_amount: 0,
            created_at: Utc::now(),
        };
        
        // Before cliff
        let before_cliff = start_date + chrono::Duration::days(60); // 2 months
        assert!(!schedule.is_vested(before_cliff));
        assert_eq!(schedule.vested_amount(before_cliff), 0);
        
        // After cliff, before full vesting
        let after_cliff = start_date + chrono::Duration::days(150); // 5 months
        assert!(schedule.is_vested(after_cliff));
        let vested = schedule.vested_amount(after_cliff);
        // 5 months total, 3 month cliff = 2 months vested out of 9 vesting months
        assert!(vested > 0 && vested < 1_000_000);
        
        // After full vesting
        let after_full = start_date + chrono::Duration::days(400); // ~13 months
        assert_eq!(schedule.vested_amount(after_full), 1_000_000);
    }
    
    #[tokio::test]
    async fn test_reward_distributor_creation() {
        let config = RewardConfig::default();
        let distributor = RewardDistributor::new(config);
        
        assert!(distributor.get_statistics().await.total_distributed == 0);
    }
    
    #[test]
    fn test_individual_reward_calculation() {
        // Test that IndividualReward struct can be created
        let reward = IndividualReward {
            node_id: "test_node".to_string(),
            shapley_value: 0.5,
            reputation_score: 0.8,
            raw_amount: 1000,
            final_amount: 1200,
            share_percentage: 10.0,
        };
        
        assert_eq!(reward.node_id, "test_node");
        assert_eq!(reward.shapley_value, 0.5);
        assert_eq!(reward.reputation_score, 0.8);
        assert_eq!(reward.raw_amount, 1000);
        assert_eq!(reward.final_amount, 1200);
        assert_eq!(reward.share_percentage, 10.0);
    }
}
