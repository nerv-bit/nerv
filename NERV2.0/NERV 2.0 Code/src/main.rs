//! NERV v2.0 Node Entry Point.
//!
//! Starts the full node with all subsystems:
//! - RocksDB storage backend
//! - P2P networking (libp2p with ML-KEM Noise handshakes)
//! - PQ-Sphinx privacy layer
//! - DKG threshold mempool
//! - NWO embedding engine with continuous Adam backprop
//! - AI-native consensus
//! - Dynamic neural sharding
//! - RPC / VDW server
//! - Metrics exporter

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::broadcast;
use tracing::{error, info, warn};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use nerv::{NodeConfig, EMBEDDING_DIM, EMBEDDING_BYTES, BATCH_MAX_SIZE, SPHINX_HOPS};

// ─── CLI ─────────────────────────────────────────────────────────────────

/// NERV v2.0 — The Self-Evolving Blockchain Node
#[derive(Parser, Debug)]
#[command(
    name = "nerv",
    version = "2.0.0",
    about = "NERV v2.0: Private, Post-Quantum, Infinitely Scalable Blockchain via NWO Paradigm",
    long_about = None,
    propagate_version = true,
    arg_required_else_help = false
)]
struct Cli {
    /// Path to the TOML configuration file.
    #[arg(short, long, env = "NERV_CONFIG", default_value = "nerv.toml")]
    config: PathBuf,

    /// Data directory (overrides config file).
    #[arg(short, long, env = "NERV_DATA_DIR")]
    data_dir: Option<PathBuf>,

    /// Log level filter (overrides config file).
    /// Examples: "info", "debug", "nerv=trace,nerv_network=warn"
    #[arg(short, long, env = "NERV_LOG_LEVEL")]
    log_level: Option<String>,

    /// Generate a default configuration file and exit.
    #[arg(long)]
    generate_config: bool,

    /// Run in "validator" mode (participate in consensus + useful-work).
    #[arg(long)]
    validator: bool,

    /// Run as a Sphinx mixnet relay.
    #[arg(long)]
    relay: bool,

    /// Seed phrase for the validator/relay key (if not provided, loads from keystore).
    #[arg(long, env = "NERV_SEED_PHRASE")]
    seed_phrase: Option<String>,

    /// P2P listen address override (e.g., "/ip4/0.0.0.0/tcp/41000").
    #[arg(long)]
    listen: Option<String>,

    /// Bootnode multiaddr for initial peer discovery (can be repeated).
    #[arg(long)]
    bootnode: Option<String>,

    /// Disable the RPC/API server.
    #[arg(long)]
    no_rpc: bool,

    /// Disable metrics collection.
    #[arg(long)]
    no_metrics: bool,

    /// Print the NERV banner and exit.
    #[arg(long)]
    banner: bool,
}

// ─── NERV ASCII Banner ──────────────────────────────────────────────────

fn print_banner() {
    const BANNER: &str = r#"
  ███╗   ██╗███████╗██████╗  ██╗   ██╗
  ████╗  ██║██╔════╝██╔══██╗██║   ██║
  ██╔██╗ ██║█████╗  ██████╔╝██║   ██║
  ██║╚██╗██║██╔══╝  ██╔══██╗╚██╗ ██╔╝
  ██║ ╚████║███████╗██║  ██║ ╚████╔╝
  ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝  ╚═══╝

  NERV v2.0 — The Self-Evolving Blockchain
  NWO Paradigm • PQ-Sphinx Privacy • LatentLedger Lite
  Private • Post-Quantum • Infinitely Scalable
"#;
    println!("{BANNER}");
}

// ─── Main ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // Parse CLI arguments
    let cli = Cli::parse();

    // Handle banner-only mode
    if cli.banner {
        print_banner();
        return Ok(());
    }

    // Handle config generation mode
    if cli.generate_config {
        let default_toml = nerv::config::generate_default_config()
            .context("failed to generate default config")?;
        std::fs::write(&cli.config, &default_toml)
            .with_context(|| format!("failed to write config to {:?}", cli.config))?;
        println!("Default configuration written to {:?}", cli.config);
        return Ok(());
    }

    // Load configuration
    let mut config = if cli.config.exists() {
        NodeConfig::load_from_file(&cli.config)
            .with_context(|| format!("failed to load config from {:?}", cli.config))?
    } else {
        warn!("Config file {:?} not found, using defaults", cli.config);
        let mut cfg = NodeConfig::default();
        cfg.validate()
            .context("default config validation failed")?;
        cfg
    };

    // Apply CLI overrides
    if let Some(data_dir) = cli.data_dir {
        config.data_dir = data_dir;
    }
    if let Some(log_level) = &cli.log_level {
        config.log_level = log_level.clone();
    }
    if let Some(listen) = &cli.listen {
        config.network.listen_addrs = vec![listen.clone()];
    }
    if let Some(bootnode) = &cli.bootnode {
        config.network.bootnodes.push(bootnode.clone());
    }
    if cli.no_rpc {
        config.rpc.listen_addr = "127.0.0.1:0".parse().unwrap();
    }
    if cli.no_metrics {
        config.metrics.enabled = false;
    }

    // Initialize tracing/logging
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&config.log_level));
    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true).with_thread_ids(true).compact())
        .with(env_filter)
        .init();

    // Print startup banner
    print_banner();
    info!(
        node_name = %config.node_name,
        data_dir = ?config.data_dir,
        embedding_dim = EMBEDDING_DIM,
        embedding_bytes = EMBEDDING_BYTES,
        batch_size = BATCH_MAX_SIZE,
        sphinx_hops = SPHINX_HOPS,
        validator = cli.validator,
        relay = cli.relay,
        "NERV v2.0 node starting"
    );

    // Ensure data directory exists
    config.ensure_data_dir()
        .context("failed to create data directory")?;

    // Initialize the RocksDB storage backend
    let db_path = config.data_dir.join("rocksdb");
    let db_opts = {
        let mut opts = rocksdb::Options::default();
        opts.set_max_open_files(512);
        opts.set_use_fsync(false);
        opts.set_bytes_per_sync(1048576);
        opts.set_max_background_jobs(6);
        opts.set_compaction_style(rocksdb::DBCompactionStyle::Universal);
        opts.set_create_if_missing(true);
        opts.set_create_missing_column_families(true);
        opts
    };

    // Define column families for organized storage
    let cf_names = vec![
        "embedding_roots",    // Canonical embedding root hashes
        "embeddings",         // Full 512-byte embedding vectors
        "vdws",               // Verifiable Delay Witnesses
        "blocks",             // Block headers and bodies
        "transactions",       // Transaction metadata (hash → location)
        "validators",         // Validator registry (id → stake + reputation)
        "weights",            // NWO Perceptron weights W and bias b
        "adam_state",         // Adam optimizer state (m, v moments)
        "dkg_shares",         // DKG share storage
        "mempool",            // Threshold-encrypted mempool entries
        "shard_metadata",     // Shard IDs and their state
        "staking",            // Staking ledger
        "tokenomics",         // Vesting schedules and allocations
        "peers",              // Peer reputation and metadata
    ];

    let cfs: Vec<rocksdb::ColumnFamilyDescriptor> = cf_names
        .iter()
        .map(|name| {
            let mut cf_opts = rocksdb::Options::default();
            cf_opts.set_compression(rocksdb::DBCompressionType::Lz4);
            cf_opts.set_block_size(16 * 1024);
            rocksdb::ColumnFamilyDescriptor::new(*name, cf_opts)
        })
        .collect();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cfs)
        .with_context(|| format!("failed to open RocksDB at {:?}", db_path))?;
    let db = Arc::new(db);
    info!(path = ?db_path, "RocksDB storage initialized");

    // Create shutdown signal channel
    let (shutdown_tx, _) = broadcast::channel::<()>(1);

    // Determine node role
    let role = if cli.validator {
        "validator"
    } else if cli.relay {
        "relay"
    } else {
        "full"
    };
    info!(role = role, "Node role determined");

    // ── Initialize subsystems ────────────────────────────────────────

    // Each subsystem receives:
    //   - A clone of the Arc<DB> for persistent storage
    //   - The relevant sub-configuration
    //   - A clone of the shutdown receiver for graceful termination
    //
    // Subsystems are spawned as independent tokio tasks and
    // communicate via async channels (mpsc, broadcast, oneshot).

    // (1) Post-Quantum Cryptography Backend
    // Initialize OQS (Open Quantum Safe) — ensures Dilithium-3 and ML-KEM-768
    // are available. This is a no-op in most builds but validates the backend.
    {
        use oqs::sig::Dilithium3;
        use oqs::kem::MlKem768;
        // Verify PQ algorithms are available by attempting keygen
        let _dilithium_check = Dilithium3::keypair()
            .context("Dilithium-3 key generation failed — is liboqs installed?")?;
        let _mlkem_check = MlKem768::keypair()
            .context("ML-KEM-768 key generation failed — is liboqs installed?")?;
        info!("Post-quantum crypto backend initialized (Dilithium-3 + ML-KEM-768)");
    }

    // (2) NWO Embedding Engine
    // The Perceptron weights W and Adam state are loaded from DB
    // or initialized with weight_init_scale if this is a fresh node.
    info!(
        dim = config.embedding.dimension,
        lr = config.embedding.learning_rate,
        beta1 = config.embedding.adam_beta1,
        beta2 = config.embedding.adam_beta2,
        huber_delta = config.embedding.huber_delta,
        continuous = config.embedding.continuous_updates,
        "NWO embedding engine configured"
    );

    // (3) Privacy Layer (PQ-Sphinx + DKG Mempool)
    info!(
        hops = config.privacy.sphinx_hops,
        dkg_participants = config.privacy.dkg_participants,
        dkg_threshold = config.privacy.dkg_threshold,
        mempool_capacity = config.privacy.mempool_max_encrypted_txs,
        batch_size = config.privacy.mempool_batch_size,
        "Privacy layer configured (pure-cryptographic, no TEEs)"
    );

    // (4) Consensus Engine
    info!(
        quorum = format!("{}%", config.consensus.quorum_numerator * 100 / config.consensus.quorum_denominator),
        vote_window_ms = config.consensus.vote_window_ms,
        max_reorg = config.consensus.max_reorg_depth,
        "AI-native consensus configured"
    );

    // (5) Sharding
    info!(
        split_threshold = config.sharding.split_threshold,
        merge_threshold = config.sharding.merge_threshold,
        erasure = format!("({},{})", config.sharding.erasure_k, config.sharding.erasure_m),
        "Dynamic neural sharding configured"
    );

    // (6) P2P Networking
    info!(
        listen = ?config.network.listen_addrs,
        bootnodes = config.network.bootnodes.len(),
        max_peers = config.network.max_inbound + config.network.max_outbound,
        "P2P networking configured"
    );

    // (7) RPC / API Server
    if !cli.no_rpc {
        info!(
            addr = %config.rpc.listen_addr,
            vdw_endpoint = config.rpc.enable_vdw_endpoint,
            max_connections = config.rpc.max_connections,
            "RPC/API server configured"
        );
    }

    // (8) Metrics
    if config.metrics.enabled && !cli.no_metrics {
        info!(
            addr = %config.metrics.listen_addr,
            "Prometheus metrics exporter configured"
        );
    }

    // ── Start subsystem tasks ────────────────────────────────────────
    //
    // Each subsystem is spawned as a tokio task. They communicate
    // via channels defined in the network/ and consensus/ modules.
    // The full wiring will be completed as modules are implemented
    // in subsequent chunks (2–15).

    // Spawn metrics exporter if enabled
    if config.metrics.enabled && !cli.no_metrics {
        let metrics_addr = config.metrics.listen_addr;
        tokio::spawn(async move {
            if let Err(e) = start_metrics_server(metrics_addr).await {
                error!(error = %e, "Metrics server failed");
            }
        });
    }

    // ── Graceful Shutdown ────────────────────────────────────────────

    info!("NERV v2.0 node is running. Press Ctrl+C to shut down gracefully.");

    // Wait for shutdown signal (Ctrl+C or SIGTERM)
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .context("failed to install Ctrl+C handler")?;
        Ok::<(), anyhow::Error>(())
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())?
            .recv()
            .await;
        Ok::<(), anyhow::Error>(())
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<Result<(), anyhow::Error>>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C, initiating graceful shutdown...");
        }
        _ = terminate => {
            info!("Received SIGTERM, initiating graceful shutdown...");
        }
    }

    // Send shutdown signal to all subsystems
    let _ = shutdown_tx.send(());
    info!("Shutdown signal sent to all subsystems");

    // Give subsystems a moment to flush and close
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    // Flush and close RocksDB
    {
        let db = Arc::try_unwrap(db);
        match db {
            Ok(db) => {
                drop(db);
                info!("RocksDB closed cleanly");
            }
            Err(arc_db) => {
                // Other tasks still hold references — try flushing
                if let Err(e) = arc_db.flush() {
                    warn!(error = %e, "RocksDB flush on shutdown failed");
                } else {
                    info!("RocksDB flushed (references still held by subsystems)");
                }
            }
        }
    }

    info!("NERV v2.0 node shut down successfully.");
    Ok(())
}

// ─── Metrics Server ─────────────────────────────────────────────────────

async fn start_metrics_server(addr: std::net::SocketAddr) -> Result<()> {
    use axum::Router;
    use axum::routing::get;
    use tower_http::cors::CorsLayer;

    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind(addr).await
        .with_context(|| format!("failed to bind metrics listener on {addr}"))?;

    info!(addr = %addr, "Metrics server listening");
    axum::serve(listener, app).await
        .context("metrics server error")?;

    Ok(())
}

async fn metrics_handler() -> String {
    use prometheus::TextEncoder;
    let encoder = TextEncoder::new();
    let metric_families = prometheus::default_registry().gather();
    encoder.encode_to_string(&metric_families).unwrap_or_default()
}
