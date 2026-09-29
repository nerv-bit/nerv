//! The NERV CLI (design doc; erratum 191).

use anyhow::Result;
use clap::{Parser, Subcommand};

mod commands;

#[derive(Parser)]
#[command(name = "nerv-cli", about = "NERV wallet operations and protocol tooling")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Wallet key management.
    Keys {
        #[command(subcommand)]
        action: KeysAction,
    },
    /// Transaction construction and proving.
    Tx {
        #[command(subcommand)]
        action: TxAction,
    },
    /// Inclusion witness regeneration.
    Witness {
        #[command(subcommand)]
        action: WitnessAction,
    },
    /// Epoch attestation chain verification.
    Cert {
        #[command(subcommand)]
        action: CertAction,
    },
    /// Supply identity audit.
    Supply {
        #[command(subcommand)]
        action: SupplyAction,
    },
    /// Conformance vector generation.
    Vectors {
        #[command(subcommand)]
        action: VectorsAction,
    },
}

#[derive(Subcommand)]
enum KeysAction {
    /// Generate a new wallet from OS entropy.
    Generate {
        #[arg(long)]
        output: Option<String>,
        #[arg(long, default_value = "8")]
        coverage: usize,
    },
    /// Display the diversified address set for a seed.
    Addresses {
        #[arg(long)]
        seed: String,
        #[arg(long, default_value = "8")]
        coverage: usize,
    },
}

#[derive(Subcommand)]
enum TxAction {
    /// Construct a transaction (requires the node's wallet RPC).
    Build {
        #[arg(long)]
        seed: String,
        #[arg(long)]
        recipient: String,
        #[arg(long)]
        amount: u64,
        #[arg(long, default_value = "700")]
        fee: u64,
    },
    /// Prove a constructed transaction (requires the node's wallet RPC).
    Prove {
        #[arg(long)]
        seed: String,
        #[arg(long)]
        transaction: String,
    },
}

#[derive(Subcommand)]
enum WitnessAction {
    /// Regenerate a witness from an archival block file.
    Regen {
        #[arg(long)]
        block: String,
        #[arg(long)]
        txid: String,
        #[arg(long)]
        leg: u8,
    },
}

#[derive(Subcommand)]
enum CertAction {
    /// Verify an epoch attestation chain.
    Verify {
        #[arg(long)]
        chain: String,
    },
}

#[derive(Subcommand)]
enum SupplyAction {
    /// Audit the emission schedule.
    Audit,
}

#[derive(Subcommand)]
enum VectorsAction {
    /// Generate the emission-schedule conformance vectors.
    EmissionSchedule {
        #[arg(long)]
        output: Option<String>,
    },
    /// Generate the seal-decode conformance vectors.
    SealDecode {
        #[arg(long, default_value = "00")]
        seed: String,
        #[arg(long, default_value = "64")]
        count: usize,
        #[arg(long)]
        output: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Keys { action } => commands::keys(action),
        Commands::Tx { action } => commands::tx(action),
        Commands::Witness { action } => commands::witness(action),
        Commands::Cert { action } => commands::cert(action),
        Commands::Supply { action } => commands::supply(action),
        Commands::Vectors { action } => commands::vectors(action),
    }
}
