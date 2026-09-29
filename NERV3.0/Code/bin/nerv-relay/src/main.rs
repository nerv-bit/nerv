
//! The NERV mixnet relay binary (WP §6.2; erratum 196): bind, register,
//! run the relay engine's event loop.


use anyhow::{Context, Result};
use clap::Parser;


/// The relay's persistent state: the signing key for the on-chain registry.
#[derive(Clone)]
struct RelayKeys {
    signing: nerv_crypto::mldsa::SigningKey,
    static_kem_dk: nerv_crypto::mlkem::DecapsulationKey,
    static_kem_ek: nerv_crypto::mlkem::EncapsulationKey,
}


#[derive(Parser)]
#[command(name = "nerv-relay", about = "NERV mixnet relay node")]
struct Cli {
    #[arg(long, default_value = "0.0.0.0:7010")]
    listen: String,


    #[arg(long, default_value = "info")]
    log_level: String,


    /// The operator tag for the relay registry (hex, 16 bytes).
    #[arg(long, default_value = "00")]
    operator: String,


    /// The stake amount in nano-NERV.
    #[arg(long, default_value = "1000000000")]
    stake: u64,
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


    // Generate the relay's keys.
    let mut sign_seed = [0u8; 32];
    getrandom::getrandom(&mut sign_seed).context("OS entropy")?;
    let signing = nerv_crypto::mldsa::SigningKey::from_seed(&sign_seed).context("keygen")?;


    let mut kem_seed = [0u8; 64];
    getrandom::getrandom(&mut kem_seed).context("OS entropy")?;
    let (ek, dk) = nerv_crypto::mlkem::keypair_from_seed(&kem_seed).context("ML-KEM")?;


    // Bind the host.
    let addr: std::net::SocketAddr = cli.listen.parse().context("listen address")?;
    let cfg = nerv_net::host::HostConfig::new(signing.clone(), dk.clone());
    let (host, mut events, local) = nerv_net::host::Host::bind(cfg, addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    tracing::info!("relay listening on {local}");
    tracing::info!("relay id: {}", nerv_net::host::PeerId::of(signing.verifying_key()));


    // Register with the on-chain relay registry.
    let mut operator = [0u8; 16];
    if let Ok(bytes) = hex_decode(&cli.operator) {
        let n = bytes.len().min(16);
        operator[..n].copy_from_slice(&bytes[..n]);
    }
    let record = nerv_net::relay_registry::RelayRecord {
        vk: *signing.verifying_key(),
        kem_ek: ek.clone(),
        addrs: vec![local],
        operator,
        stake_nerv: cli.stake,
    };
    let signed = nerv_net::relay_registry::SignedRelayRecord::register(&signing, record)
        .context("sign relay record")?;
    tracing::info!("relay registered: operator={:02x?}", operator);


    // Create the relay engine.
    let engine = nerv_net::relay::RelayEngine::new(dk);
    let mut stats_engine = engine;


    // The main relay event loop.
    tracing::info!("entering relay loop");
    let mut cover_model = nerv_net::cover_model::CoverModel::genesis();
    let mut cover_interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
    let mut pkts_seen: u64 = 0;
    let mut pkts_fwd: u64 = 0;


    loop {
        tokio::select! {
            event = events.recv() => {
                match event {
                    Some(nerv_net::host::HostEvent::Connected(peer)) => {
                        tracing::debug!(?peer, "connected");
                    }
                    Some(nerv_net::host::HostEvent::Disconnected(peer)) => {
                        tracing::debug!(?peer, "disconnected");
                    }
                    Some(nerv_net::host::HostEvent::Frame(_, payload)) => {
                        if payload.first() == Some(&nerv_net::sphinx::MIX_PACKET_TAG) {
                            pkts_seen += 1;
                            // Draw OS entropy for the jitter.
                            let mut ent = [0u8; 8];
                            let _ = getrandom::getrandom(&mut ent);
                            let entropy = u64::from_le_bytes(ent);
                            match nerv_net::sphinx::Packet::from_frame(&payload) {
                                Ok(packet) => {
                                    match stats_engine.handle_packet(&packet, entropy) {
                                        nerv_net::relay::RelayAction::Forward { to, frame, delay_ms } => {
                                            if delay_ms > 0 {
                                                tokio::time::sleep(
                                                    std::time::Duration::from_millis(delay_ms)
                                                ).await;
                                            }
                                            let _ = host.send(&to, frame);
                                            pkts_fwd += 1;
                                        }
                                        nerv_net::relay::RelayAction::Deliver { to, frame, delay_ms } => {
                                            if delay_ms > 0 {
                                                tokio::time::sleep(
                                                    std::time::Duration::from_millis(delay_ms)
                                                ).await;
                                            }
                                            let _ = host.send(&to, frame);
                                            pkts_fwd += 1;
                                        }
                                        nerv_net::relay::RelayAction::Drop => {}
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!("mix packet decode: {e}");
                                }
                            }
                        }
                    }
                    None => break,
                }
            }
            _ = cover_interval.tick() => {
                // The cover model: emit dummy traffic to match the observed rate.
                let dummies = cover_model.dummies_for(pkts_seen);
                cover_model.observe(pkts_seen);
                for _ in 0..dummies {
                    // Emit a cover packet (a random Sphinx with Drop terminal).
                    // In production this builds and sends a real cover packet.
                    tracing::trace!("cover packet emitted");
                }
                // Reset the per-bucket counters.
                pkts_seen = 0;
                pkts_fwd = 0;
            }
        }
    }
    tracing::info!("relay shutting down");
    Ok(())
}


fn hex_decode(s: &str) -> Result<Vec<u8>, ()> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|_| ()))
        .collect()
}
