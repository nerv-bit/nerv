//! src/main.rs
//! # NERV Node Entrypoint
//!
//! This is the binary entrypoint for the NERV blockchain node.
//! It handles:
//! - CLI argument parsing (clap)
//! - Configuration loading (from TOML/JSON/YAML file or defaults)
//! - Logging and metrics initialization (tracing + Prometheus)
//! - NervNode instantiation and full lifecycle management
//! - Graceful shutdown on signals (SIGINT/SIGTERM)
//! - Subcommands: keygen, config dump, benchmark stub
//!
//! Supports all node roles: full, light, relay, archive
//! Integrates all subsystems: privacy (mixer/TEE/VDW), recursive proving, economy, etc.

use clap::{Parser, Subcommand};
use nerv::{
    Config, NervNode, NodeRole, Result, NervError,
    utils::metrics::{self, MetricsCollector},
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use tracing::{info, error, warn, debug};

/// NERV Blockchain Node CLI
#[derive(Parser, Debug)]
#[command(name = "nerv", version = env!("CARGO_PKG_VERSION"), about = "NERV - Private, Post-Quantum, Infinitely Scalable Blockchain Node")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Path to configuration file (TOML, JSON, or YAML)
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Network ID (mainnet, testnet, devnet)
    #[arg(short, long, default_value = "devnet")]
    network: String,

    /// Node role (full, light, relay, archive)
    #[arg(short, long, default_value = "full")]
    role: NodeRole,

    /// Data/storage directory
    #[arg(short, long, default_value = "./nerv_data")]
    data_dir: PathBuf,

    /// Enable verbose/debug logging
    #[arg(short, long)]
    verbose: bool,

    /// P2P bind address
    #[arg(long, default_value = "0.0.0.0")]
    bind_addr: String,

    /// P2P port
    #[arg(long, default_value = "30333")]
    p2p_port: u16,

    /// Enable Prometheus metrics exporter
    #[arg(long)]
    metrics: bool,

    /// Prometheus exporter port
    #[arg(long, default_value = "9100")]
    metrics_port: u16,

    /// Enable privacy mixer (relay nodes)
    #[arg(long)]
    enable_mixer: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate a new keypair (Dilithium-3 primary)
    Keygen,

    /// Print current configuration
    Config,

    /// Run internal benchmarks
    Bench {
        #[arg(short, long, default_value = "10000")]
        tx_count: u64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let filter = if cli.verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("info")
    };

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();

    info!("Starting NERV node v{} - Role: {:?} - Network: {}", 
          env!("CARGO_PKG_VERSION"), cli.role, cli.network);

    // Load or create config
    let mut config = match cli.config {
        Some(path) => Config::load_from_file(&path)?,
        None => Config::default(),
    };

    // Apply CLI overrides
    config.network_id = cli.network;
    config.node_role = cli.role;
    config.storage_path = cli.data_dir;
    config.network.p2p_bind_addr = cli.bind_addr.parse().map_err(|_| NervError::Other("Invalid bind address".into()))?;
    config.network.p2p_port = cli.p2p_port;
    config.privacy.mixer_hops = if cli.enable_mixer { 5 } else { 0 }; // Relay nodes enable mixer
    config.enable_metrics = cli.metrics;

    // Handle subcommands
    match cli.command {
        Some(Commands::Keygen) => {
            let crypto = nerv::crypto::CryptoProvider::new()
                .map_err(|e| NervError::Crypto(e.to_string()))?;
            let kp = crypto.generate_signature_keypair()
                .map_err(|e| NervError::Crypto(e.to_string()))?;
            println!("Dilithium-3 Public Key:  {}", hex::encode(&kp.public_key));
            println!("Dilithium-3 Secret Key: {}", hex::encode(&kp.secret_key));
            println!("(Store secret key securely - used for node identity and signing)");
            return Ok(());
        }
        Some(Commands::Config) => {
            println!("{:#?}", config);
            return Ok(());
        }
        Some(Commands::Bench { tx_count }) => {
            info!("Running benchmark with {} synthetic transactions...", tx_count);
            // Placeholder - integrate real benchmark suite later
            println!("Benchmark stub: {} tx processed (unimplemented)", tx_count);
            return Ok(());
        }
        None => {}
    }

    // Initialize and start Prometheus exporter if enabled
    let metrics_handle = if config.enable_metrics {
        let collector = Arc::new(MetricsCollector::new(config.clone().into())?);
        metrics::start_prometheus_exporter(cli.metrics_port, collector.clone()).await?;
        info!("Prometheus metrics exporter started on http://0.0.0.0:{}", cli.metrics_port);
        Some(collector)
    } else {
        None
    };

    // Create and start the node
    let mut node = NervNode::new(config.clone()).await?;

    // Inject metrics collector if enabled (for internal use)
    if let Some(collector) = metrics_handle {
        // In real impl: pass via node or global
        debug!("Metrics collector injected");
    }

    node.start().await?;

    info!("NERV node fully operational - Privacy: {} hops, Role: {:?}", 
          config.privacy.mixer_hops, config.node_role);
    
          let updater = EncoderUpdater::new(
    Arc::clone(&node.encoder),
    Arc::clone(&node.fl_aggregator),
    Arc::clone(&node.consensus),
    10_000,  // Update every ~10k blocks
);
tokio::spawn(updater.run(current_height));

    // Wait for shutdown signal
    let shutdown = handle_signals().await;
    info!("Shutdown signal received: {:?}", shutdown);

    // Graceful shutdown
    info!("Initiating graceful shutdown...");
    node.shutdown().await?;
    info!("NERV node stopped cleanly");

    Ok(())
}

/// Handle OS signals for graceful shutdown
async fn handle_signals() {
    let mut sigint = signal::unix::signal(signal::unix::SignalKind::interrupt()).unwrap();
    let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate()).unwrap();

    tokio::select! {
        _ = sigint.recv() => "SIGINT",
        _ = sigterm.recv() => "SIGTERM",
    };
    info!("Received shutdown signal");
}

impl Config {
    /// Load config from file (supports TOML, JSON, YAML via serde)
    pub fn load_from_file(path: &PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| NervError::Io(e.to_string()))?;

        let ext = path.extension()
            .and_then(|s| s.to_str())
            .unwrap_or("toml");

        let config: Self = match ext {
            "toml" => toml::from_str(&content),
            "json" => serde_json::from_str(&content),
            "yaml" | "yml" => serde_yaml::from_str(&content),
            _ => return Err(NervError::Other("Unsupported config format".into())),
        };

        config.map_err(|e| NervError::Serialization(e.to_string()))
    }
}
