//! Node configuration (design doc: [WRAP figment]; erratum 195).


use anyhow::{Context, Result};
use std::net::SocketAddr;


#[derive(Clone, Debug)]
pub struct NodeConfig {
    pub role: String,
    pub listen: String,
    pub metrics: Option<String>,
    pub data_dir: String,
    pub peers: Vec<String>,
    pub log_level: String,
    pub shard: Option<u16>,
    /// Producer master seed (32 bytes, hex-encoded). Only honored
    /// when `role == "producer"`. When `None`, the node generates a
    /// fresh seed from OS entropy — fine for the testnet/dev, never
    /// for production (operators should pin the seed to a known
    /// secret so payouts are recoverable).
    pub producer_seed: Option<[u8; 32]>,
    /// Producer stake bond in nano-NERV. `None` → 0 (no stake
    /// recorded). Production operators set this to the bond they
    /// want committed via `StakeLedger::stake`.
    pub producer_stake_nano: Option<u64>,
}


impl NodeConfig {
    pub fn from_cli(cli: &crate::Cli) -> Result<NodeConfig> {
        let mut config = NodeConfig {
            role: cli.role.clone(),
            listen: cli.listen.clone(),
            metrics: cli.metrics.clone(),
            data_dir: cli.data_dir.clone(),
            peers: Vec::new(),
            log_level: cli.log_level.clone(),
            shard: None,
            producer_seed: None,
            producer_stake_nano: None,
        };


        // Parse the peers list.
        if let Some(peers) = &cli.peers {
            config.peers = peers.split(',').map(|s| s.trim().to_string()).collect();
        }


        // Parse the config file if provided (simple key=value pairs).
        if let Some(path) = &cli.config {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("read config file {path}"))?;
            for line in text.lines() {
                if let Some((k, v)) = line.split_once('=') {
                    let k = k.trim();
                    let v = v.trim().trim_matches('"');
                    match k {
                        "shard" => {
                            config.shard = Some(v.parse().context("shard must be a number")?);
                        }
                        "listen" => config.listen = v.to_string(),
                        "data_dir" => config.data_dir = v.to_string(),
                        "producer_seed" => {
                            let trimmed = v.trim_start_matches("0x");
                            if trimmed.len() != 64 {
                                anyhow::bail!("producer_seed must be 64 hex chars (32 bytes)");
                            }
                            let mut out = [0u8; 32];
                            for i in 0..32 {
                                out[i] = u8::from_str_radix(&trimmed[2 * i..2 * i + 2], 16)
                                    .with_context(|| format!("invalid hex at byte {i}"))?;
                            }
                            config.producer_seed = Some(out);
                        }
                        "producer_stake_nano" => {
                            config.producer_stake_nano = Some(
                                v.parse().context("producer_stake_nano must be a u64")?,
                            );
                        }
                        _ => {}
                    }
                }
            }
        }


        Ok(config)
    }


    pub fn listen_addr(&self) -> Result<SocketAddr> {
        self.listen
            .parse()
            .with_context(|| format!("invalid listen address: {}", self.listen))
    }
}
