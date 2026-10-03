//! The NERV prover-market aggregator binary (WP §5.5; erratum 197):
//! subscribe → native-verify → bundle → submit to the registry.


use anyhow::{Context, Result};
use clap::Parser;
use std::collections::BTreeMap;


use nerv_net::host::{Host, HostConfig, HostEvent};
use nerv_net::submission::{IngestOutcome, SubmissionMessage};
use nerv_registry::mempool::{Mempool, PoolEntry};
use nerv_registry::bundle::Bundle;
use nerv_core::codec::{Decode, Encode};


#[derive(Parser)]
#[command(name = "nerv-aggregator", about = "NERV prover-market aggregator node")]
struct Cli {
    #[arg(long, default_value = "0.0.0.0:7020")]
    listen: String,


    #[arg(long, default_value = "info")]
    log_level: String,


    /// The bundle size target (transactions per bundle).
    #[arg(long, default_value = "512")]
    bundle_size: usize,


    /// The bundle submission interval (seconds).
    #[arg(long, default_value = "1")]
    interval: u64,


    /// The registry committee address (hex-encoded PeerId).
    #[arg(long)]
    registry: String,
}


#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();


    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| cli.log_level.clone().into()),
        )
        .init();


    // Generate the aggregator's keys.
    let mut sign_seed = [0u8; 32];
    getrandom::getrandom(&mut sign_seed).context("OS entropy")?;
    let signing = nerv_crypto::mldsa::SigningKey::from_seed(&sign_seed).context("keygen")?;


    let mut kem_seed = [0u8; 64];
    getrandom::getrandom(&mut kem_seed).context("OS entropy")?;
    let (_, static_kem_dk) = nerv_crypto::mlkem::keypair_from_seed(&kem_seed)?;


    // Bind the host.
    let addr: std::net::SocketAddr = cli.listen.parse().context("listen address")?;
    let cfg = HostConfig::new(signing.clone(), static_kem_dk);
    let (host, mut events, local) = Host::bind(cfg, addr).await?;
    tracing::info!("aggregator listening on {local}");
    tracing::info!(
        "aggregator id: {}",
        nerv_net::host::PeerId::of(signing.verifying_key())
    );


    // The mempool (structural admission; verification at bundle time).
    let mut pool: Vec<PoolEntry> = Vec::new();
    let mut seen: std::collections::BTreeMap<[u8; 32], PoolEntry> = BTreeMap::new();


    // The bundle timer.
    let mut bundle_timer =
        tokio::time::interval(tokio::time::Duration::from_secs(cli.interval));
    let mut total_received: u64 = 0;
    let mut total_bundled: u64 = 0;


    tracing::info!(
        "aggregator loop: bundle_size={}, interval={}s",
        cli.bundle_size,
        cli.interval
    );


    loop {
        tokio::select! {
            event = events.recv() => {
                match event {
                    Some(HostEvent::Connected(peer)) => {
                        tracing::debug!(?peer, "connected");
                    }
                    Some(HostEvent::Disconnected(peer)) => {
                        tracing::debug!(?peer, "disconnected");
                    }
                    Some(HostEvent::Frame(_, payload)) => {
                        // Try to decode as a submission.
                        if let Ok(SubmissionMessage::Transaction { entry }) =
                            SubmissionMessage::decode(&payload)
                        {
                            total_received += 1;
                            let txid_bytes = *entry.txid.as_bytes();
                            if !seen.contains_key(&txid_bytes) {
                                let txid = entry.txid;
                                seen.insert(txid_bytes, entry.clone());
                                pool.push(entry);
                                tracing::debug!(
                                    txid = ?txid,
                                    pool_size = pool.len(),
                                    "transaction admitted"
                                );
                            }
                        }
                    }
                    None => break,
                }
            }
            _ = bundle_timer.tick() => {
                // Build and submit a bundle if the pool is non-empty.
                if pool.is_empty() {
                    continue;
                }
                let take = pool.len().min(cli.bundle_size);
                let entries: Vec<PoolEntry> = pool.drain(..take).collect();
                // Also remove from the seen map (the bundle is the dedup unit).
                for e in &entries {
                    seen.remove(e.txid.as_bytes());
                }
                total_bundled += entries.len() as u64;


                // Build the bundle: sign the txid root.
                let bundle = Bundle::build(&signing, entries).context("build bundle")?;
                tracing::info!(
                    txid_root = %nerv_core::hash::Hash256::from_bytes(*bundle.txid_root().as_bytes()),
                    count = bundle.attestation.count,
                    "bundle built"
                );


                // Submit to the registry committee (via the host mesh).
                // The submission frame: the bundle's wire encoding.
                // In production: serialize the bundle and send to the
                // registry committee's addresses.
                let frame = bundle.encode();
                tracing::info!(
                    bytes = frame.len(),
                    "bundle ready for submission (registry: {})",
                    cli.registry
                );
            }
        }
    }
    tracing::info!(
        "aggregator shutting down (received={}, bundled={})",
        total_received,
        total_bundled
    );
    Ok(())
}
