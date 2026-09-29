//! Command implementations (erratum 191).

use anyhow::{bail, Context, Result};

use crate::{CertAction, KeysAction, SupplyAction, TxAction, VectorsAction, WitnessAction};

fn parse_seed(hex: &str) -> Result<[u8; 32]> {
    let bytes = hex_decode(hex)?;
    bytes
        .try_into()
        .map_err(|v: Vec<u8>| anyhow::anyhow!("seed must be 32 bytes, got {}", v.len()))
}

fn parse_hash(hex: &str) -> Result<[u8; 32]> {
    let bytes = hex_decode(hex)?;
    bytes
        .try_into()
        .map_err(|v: Vec<u8>| anyhow::anyhow!("hash must be 32 bytes, got {}", v.len()))
}


fn hex_decode(s: &str) -> Result<Vec<u8>, anyhow::Error> {
    if s.len() % 2 != 0 {
        bail!("odd-length hex string");
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    for i in (0..s.len()).step_by(2) {
        let byte = u8::from_str_radix(&s[i..i + 2], 16)
            .with_context(|| format!("invalid hex at position {i}"))?;
        out.push(byte);
    }
    Ok(out)
}



fn hex_encode(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn wallet_from_seed(seed: [u8; 32]) -> nerv_custody::WalletKeys {
    nerv_custody::WalletKeys::from_master(&nerv_custody::MasterSeed::from_bytes(seed))
}

// ---------------------------------------------------------------------------
// keys
// ---------------------------------------------------------------------------

pub fn keys(action: KeysAction) -> Result<()> {
    match action {
        KeysAction::Generate { output, coverage } => {
            let mut seed = [0u8; 32];
            getrandom::getrandom(&mut seed).context("OS entropy unavailable")?;
            let hex_seed = hex_encode(&seed);
            println!("master_seed: {hex_seed}");

            let keys = wallet_from_seed(seed);
            let active = nerv_core::types::ShardSet::genesis();
            let addrs =
                nerv_wallet::AddressSet::generate_with_coverage(&keys, &active, coverage)?;
            println!("coverage: {}/{} shards", addrs.coverage(), coverage);
            println!("addresses: {}", addrs.len());
            for addr in addrs.addresses() {
                println!(
                    "  {:>3}  shard={:<4} ek={}",
                    addr.index,
                    format!("{:?}", addr.shard()),
                    &hex_encode(addr.ek.as_bytes())
                );
            }
            if let Some(path) = output {
                std::fs::write(&path, &hex_seed)
                    .with_context(|| format!("write seed to {path}"))?;
                eprintln!("seed written to {path}");
            }
            Ok(())
        }
        KeysAction::Addresses { seed, coverage } => {
            let seed = parse_seed(&seed)?;
            let keys = wallet_from_seed(seed);
            let active = nerv_core::types::ShardSet::genesis();
            let addrs =
                nerv_wallet::AddressSet::generate_with_coverage(&keys, &active, coverage)?;
            println!("coverage: {}/{} shards ({} addresses)", addrs.coverage(), coverage, addrs.len());
            for addr in addrs.addresses() {
                println!(
                    "  {:>3}  shard={:<4} ek={}",
                    addr.index,
                    format!("{:?}", addr.shard()),
                    &hex_encode(addr.ek.as_bytes())
                );
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// tx
// ---------------------------------------------------------------------------

pub fn tx(action: TxAction) -> Result<()> {
    match action {
        TxAction::Build { .. } => {
            bail!(
                "tx build requires the wallet's scanned-note state (NCT positions, \
                 nullifiers) — use the node's wallet RPC (chunk 20). The offline \
                 pipeline (construct → prove) is fully implemented in nerv_wallet::construct \
                 and nerv_wallet::prove."
            )
        }
        TxAction::Prove { .. } => {
            bail!(
                "tx prove requires the constructed transaction plus the codec and epoch \
                 key context — use the node's wallet RPC (chunk 20)."
            )
        }
    }
}

// ---------------------------------------------------------------------------
// witness
// ---------------------------------------------------------------------------

pub fn witness(action: WitnessAction) -> Result<()> {
    match action {
        WitnessAction::Regen { block, txid, leg } => {
            let bytes = std::fs::read(&block)
                .with_context(|| format!("read block file {block}"))?;
            let txid_bytes = parse_hash(&txid)?;
            let txid = nerv_core::types::TxId::from_hash(
                nerv_core::hash::Hash256::from_bytes(txid_bytes),
            );
            let block: nerv_state::block::ShardBlock =
                nerv_state::block::ShardBlock::decode(&bytes).context("decode block")?;
            let w = nerv_witness::regenerate(
                &block,
                &txid,
                nerv_core::types::LegIndex::from_u8(leg),
            )?;
            println!("txid: {}", hex_encode(w.txid.as_bytes()));
            println!("leg: {}", w.leg.as_u8());
            println!("leaf_index: {}", w.leaf_index);
            println!("siblings: {}", w.siblings.len());
            println!("shard: {:?}", w.shard);
            println!("height: {}", w.height);
            println!("interval: {}", w.interval);
            println!("header_hash: {}", hex_encode(w.header_hash.as_bytes()));
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// cert
// ---------------------------------------------------------------------------

pub fn cert(action: CertAction) -> Result<()> {
    match action {
        CertAction::Verify { chain } => {
            let bytes = std::fs::read(&chain)
                .with_context(|| format!("read chain file {chain}"))?;
            let mut attestations = Vec::new();
            let mut offset = 0;
            while offset < bytes.len() {
                match nerv_consensus::attestation::EpochAttestation::decode(&bytes[offset..]) {
                    Ok(att) => {
                        offset += att.encoded_len();
                        attestations.push(att);
                    }
                    Err(_) => break,
                }
            }
            if attestations.is_empty() {
                bail!("no epoch attestations found in {chain}");
            }
            println!("epoch_attestations: {}", attestations.len());
            for (i, att) in attestations.iter().enumerate() {
                println!(
                    "  [{i}] epoch={} digest={}",
                    att.epoch.as_u64(),
                    &hex_encode(att.digest().as_bytes())
                );
            }
            for w in attestations.windows(2) {
                if w[1].prev != w[0].digest() {
                    bail!(
                        "chain broken at epoch {}: prev {} ≠ digest {}",
                        w[1].epoch.as_u64(),
                        &hex_encode(w[1].prev.as_bytes()),
                        &hex_encode(w[0].digest().as_bytes())
                    );
                }
            }
            println!("chain_links: OK");
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// supply
// ---------------------------------------------------------------------------

pub fn supply(action: SupplyAction) -> Result<()> {
    match action {
        SupplyAction::Audit => {
            let schedule = nerv_economy::EmissionSchedule::genesis();
            schedule.validate_allocation()?;
            println!(
                "allocation: VALID ({} buckets, 1000 permille, 10B NERV)",
                schedule.buckets().len()
            );
            for b in schedule.buckets() {
                println!(
                    "  {:<20} total={:>13} account={:<18}",
                    b.name(),
                    b.total_nerv(),
                    format!("{:?}", b.account()),
                );
            }
            for day in [0u64, 1, 180, 360, 361, 720, 1440, 1800, 3600, 3650] {
                println!(
                    "  day={day:>5} released={:>13} upper_bound={}",
                    schedule.released_by_day(day),
                    schedule.upper_bound_by_day(day),
                );
            }
            println!("supply_identity: emitted − burned − abandoned = supply (M1)");
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// vectors
// ---------------------------------------------------------------------------

struct SplitMix {
    state: u64,
}

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

pub fn vectors(action: VectorsAction) -> Result<()> {
    match action {
        VectorsAction::EmissionSchedule { output } => {
            let schedule = nerv_economy::EmissionSchedule::genesis();
            schedule.validate_allocation()?;

            let mut boundaries = vec![0u64, 1];
            for b in schedule.buckets() {
                boundaries.push(b.term_days());
                if let nerv_economy::Curve::Linear { cliff_days, .. } = b.curve() {
                    boundaries.push(cliff_days);
                    boundaries.push(cliff_days + 1);
                }
            }
            boundaries.sort();
            boundaries.dedup();
            boundaries.push(boundaries.last().unwrap_or(&0) + 1);

            let mut out = Vec::new();
            for b in schedule.buckets() {
                for &day in &boundaries {
                    let released = b.released_by_day(day);
                    out.extend_from_slice(&(day as u32).to_le_bytes());
                    out.extend_from_slice(&released.to_le_bytes());
                }
            }

            let digest = blake3::hash(&out);
            println!("family: emission_schedule");
            println!("buckets: {}", schedule.buckets().len());
            println!("boundaries: {}", boundaries.len());
            println!("bytes: {}", out.len());
            println!("blake3: {}", hex_encode(digest.as_bytes()));

            if let Some(path) = output {
                std::fs::write(&path, &out)
                    .with_context(|| format!("write vectors to {path}"))?;
                eprintln!("vectors written to {path}");
            }
            Ok(())
        }
        VectorsAction::SealDecode { seed, count, output } => {
            let seed_bytes = parse_seed(&seed).unwrap_or([0u8; 32]);
            let state = u64::from_le_bytes(
                seed_bytes[..8].try_into().unwrap_or([1, 0, 0, 0, 0, 0, 0, 0]),
            ) | 1;
            let mut rng = SplitMix { state };

            let mut out = Vec::new();
            for i in 0..count {
                let delta: [u64; 64] =
                    core::array::from_fn(|j| rng.next_u64() ^ ((i as u64) << 32) ^ (j as u64));
                let pt = nerv_seal::digitize::digitize(&delta);
                let sums = pt.slot_values();
                let resolved = nerv_seal::digitize::resolve(&sums);
                for d in &delta {
                    out.extend_from_slice(&d.to_le_bytes());
                }
                for d in &resolved {
                    out.extend_from_slice(&d.to_le_bytes());
                }
            }

            let digest = blake3::hash(&out);
            println!("family: seal_decode");
            println!("pairs: {}", count);
            println!("bytes: {}", out.len());
            println!("blake3: {}", hex_encode(digest.as_bytes()));

            if let Some(path) = output {
                std::fs::write(&path, &out)
                    .with_context(|| format!("write vectors to {path}"))?;
                eprintln!("vectors written to {path}");
            }
            Ok(())
        }
    }
}

