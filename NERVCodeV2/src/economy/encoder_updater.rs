//! Encoder Updater - Closes the perpetual self-improvement loop
//!
//! This module runs a background task that:
//! 1. Waits for new aggregated global gradients (from FL aggregation rounds)
//! 2. Applies them to the running NeuralEncoder
//! 3. Computes new weight hash
//! 4. Proposes encoder update via consensus (epoch transition or governance)
//! 
//! Matches whitepaper: perpetual improvement via useful-work FL contributions

use crate::{
    embedding::encoder::NeuralEncoder,
    economy::fl_aggregation::{FLAggregator, GlobalGradient},
    consensus::{ConsensusEngine, ParameterProposal},  // Or your consensus trait
    Result, NervError,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Encoder updater with shared access to components
pub struct EncoderUpdater {
    encoder: Arc<RwLock<NeuralEncoder>>,
    aggregator: Arc<FLAggregator>,
    consensus: Arc<ConsensusEngine>,  // For proposing new params
    update_interval_blocks: u64,     // e.g., every 1000 blocks/epoch
}

impl EncoderUpdater {
    pub fn new(
        encoder: Arc<RwLock<NeuralEncoder>>,
        aggregator: Arc<FLAggregator>,
        consensus: Arc<ConsensusEngine>,
        update_interval_blocks: u64,
    ) -> Self {
        Self {
            encoder,
            aggregator,
            consensus,
            update_interval_blocks,
        }
    }

    /// Background task: run forever, check for new gradients periodically
    pub async fn run(mut self, current_height: u64) {
        let mut last_update_height = current_height;

        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;  // Or block-based trigger

            let current_height = self.consensus.current_height().await;  // Or from node state
            if current_height - last_update_height < self.update_interval_blocks {
                continue;
            }

            // Fetch latest aggregated gradient (if available since last update)
            if let Some(global_gradient) = self.aggregator.get_latest_gradient().await {
                let mut encoder = self.encoder.write().await;

                match encoder.apply_gradient_update(&global_gradient, None) {
                    Ok(new_hash) => {
                        info!("Applied FL gradient update at height {} - new encoder hash: {:x?}", 
                              current_height, new_hash);

                        // Propose new encoder version to consensus
                        let proposal = ParameterProposal::EncoderUpdate {
                            new_weight_hash: new_hash,
                            epoch: encoder.epoch,
                        };
                        let _ = self.consensus.propose_parameter_change(proposal).await;

                        last_update_height = current_height;
                    }
                    Err(e) => {
                        warn!("Failed to apply gradient update: {}", e);
                        // Optionally slash malicious contributors
                    }
                }
            }
        }
    }
}
