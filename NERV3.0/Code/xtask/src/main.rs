//! xtask — workspace policy enforcement binary (chunk 1 CI machinery).
//!
//! Implements the checks the design document demands as build rules, not
//! review guidelines: the DSR-1 dependency firewall, the P5 float-ban scan,
//! and the spec/conformance gates backed by tests/nerv-conformance.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use nerv_conformance::spec::LoadedSpec;
use nerv_conformance::util::locate;

mod firewall;
mod float_scan;
mod policy;

#[derive(Parser)]
#[command(name = "xtask", about = "NERV workspace policy enforcement")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// DSR-1: dependency-direction firewall (cargo metadata graph vs policy).
    Firewall,
    /// P5: float-ban source scan (types and literals; comments/strings aware).
    FloatScan,
    /// Validate specs/params.toml; print spec-hash, parameter count, errata.
    SpecCheck,
    /// Freeze conformance vectors (deliberate act; --force to overwrite).
    ConformanceFreeze {
        /// Overwrite an existing frozen manifest.
        #[arg(long)]
        force: bool,
    },
    /// Verify frozen vectors against the current spec, code, and schemas.
    ConformanceVerify,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("xtask: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Firewall => firewall::run(),
        Cmd::FloatScan => float_scan::run(),
        Cmd::SpecCheck => spec_check(),
        Cmd::ConformanceFreeze { force } => {
            let spec_path = locate("specs/params.toml").context("run from the workspace root")?;
            let out = locate("specs/vectors")
                .unwrap_or_else(|| spec_path.parent().expect("non-root").join("vectors"));
            let loaded = LoadedSpec::load(&spec_path)?;
            let manifest = nerv_conformance::registry::freeze(&loaded, &out, force)?;
            println!(
                "froze {} families to {} (spec-hash {})",
                manifest.families.len(),
                out.display(),
                nerv_conformance::util::hex(&manifest.spec_hash)
            );
            Ok(())
        }
        Cmd::ConformanceVerify => {
            let spec_path = locate("specs/params.toml").context("run from the workspace root")?;
            let dir = locate("specs/vectors")
                .unwrap_or_else(|| spec_path.parent().expect("non-root").join("vectors"));
            let loaded = LoadedSpec::load(&spec_path)?;
            let report = nerv_conformance::registry::verify(&loaded, &dir)?;
            print!("{report}");
            println!(
                "conformance: PASS ({} families, spec-hash {})",
                report.families.len(),
                nerv_conformance::util::hex(&report.spec_hash)
            );
            Ok(())
        }
    }
}

fn spec_check() -> Result<()> {
    let path = locate("specs/params.toml").context("run from the workspace root")?;
    let loaded = LoadedSpec::load(&path)?;
    let meta = &loaded.spec.meta;
    println!(
        "spec: {} v{} — validated ({} leaves, freeze at {})",
        meta.spec_name,
        meta.spec_version,
        nerv_conformance::util::count_leaves(&loaded.raw),
        meta.freeze_milestone
    );
    println!(
        "committees (size/quorum/derived-tolerance): {}",
        loaded.spec
            .committee_summary()
            .into_iter()
            .map(|(n, s, q, f)| format!("{n} {s}/{q}/f={f}"))
            .collect::<Vec<_>>()
            .join("  ")
    );
    println!("errata applied: {}", loaded.spec.errata.len());
    for e in &loaded.spec.errata {
        println!("  {} [{}] {}", e.id, e.refs, e.resolution);
    }
    println!("spec-hash: {}", nerv_conformance::util::hex(&loaded.hash));
    Ok(())
}

