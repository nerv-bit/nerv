//! The NERV full-node binary (WP App A; design doc; erratum 195).


use anyhow::Result;
use clap::{Parser, Subcommand};

mod chain;
mod config;
mod producer;
mod roles;
mod wiring;


#[derive(Parser)]
#[command(name = "nerv-node", about = "NERV full node")]
struct Cli {
    /// The node's role.
    #[arg(long, default_value = "validator")]
    role: String,


    /// The listen address for the P2P mesh.
    #[arg(long, default_value = "0.0.0.0:7000")]
    listen: String,


    /// The metrics (Prometheus) listen address.
    #[arg(long)]
    metrics: Option<String>,


    /// The data directory (RocksDB).
    #[arg(long, default_value = "./data")]
    data_dir: String,


    /// The config file (TOML).
    #[arg(long)]
    config: Option<String>,


    /// Peer addresses to connect to at startup (comma-separated).
    #[arg(long)]
    peers: Option<String>,


    /// The log level (trace, debug, info, warn, error).
    #[arg(long, default_value = "info")]
    log_level: String,
}


#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();


    // Initialize tracing.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| cli.log_level.clone().into()),
        )
        .init();


    // Load the config.
    let config = config::NodeConfig::from_cli(&cli)?;


    // Parse the role.
    let role = roles::parse_role(&config.role)?;
    tracing::info!("role: {:?}", role);
    tracing::info!("listen: {}", config.listen);
    tracing::info!("data_dir: {}", config.data_dir);


    // Run the node.
    wiring::run_node(config, role).await?;


    Ok(())
}
