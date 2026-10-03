//! # Cross-crate integration smoke test
//!
//! Exercises the public API surface of the wallet → proof → state → codec
//! stack end-to-end.
//!
//!   * **Step 1** — `construct_payment` (nerv-wallet) builds a
//!     `ConstructedTx`. `TransactionWitness::generate` (nerv-proofs)
//!     binds the witness to the shell.
//!   * **Step 2** — round-trip the `TransactionWitness` and
//!     `TransactionShell` through `nerv_core::codec::{Encode, Decode}`.
//!   * **Step 3** — build a minimal `ShardBlock` (empty block + a stub
//!     QC) and round-trip it through the codec.
//!
//! ## What this does NOT exercise
//!
//! `nerv_state::apply_block` is also a target. The executor's test
//! scaffolding (`TestWorld`, `seal()`, `register_tau()`) lives behind
//! `#[cfg(test)] mod tests` in `nerv_state::executor` and is not
//! re-exported. A full `apply_block` smoke would require either:
//!   * lifting those helpers into a `pub` test-support module, or
//!   * re-implementing ~200 lines of executor-test scaffolding.
//!
//! The executor's existing `mod tests` already exercises `apply_block`
//! end-to-end (single_shard_spend_settles_and_chains, empty_block_advances,
//! etc.) so the *path itself* is verified — this smoke test instead
//! proves the public cross-crate API surface stays connected.
//!
//! Run with:
//!     cargo run -p nerv-wallet --example smoke_cross_crate --release

use nerv_codec::codec_w::{CodecW, WeightVersion};
use nerv_codec::weight_gen::{certify, expand, BeaconRandomness, CertConfig};
use nerv_core::codec::{Decode, Encode};
use nerv_core::field::Goldilocks;
use nerv_core::hash::Hash256;
use nerv_core::types::{FeeSats, Height, Interval, ShardId, ShardSet};
use nerv_crypto::sigaggr::QuorumCertificate;
use nerv_crypto::mldsa::SigningKey as MlDsaSigningKey;
use nerv_custody::nct::NctDigest;
use nerv_custody::{Address, MasterSeed, WalletKeys};
use nerv_proofs::TransactionWitness;
use nerv_seal::encrypt::PublicKey;
use nerv_seal::sampling::ASeed;
use nerv_state::{RegistryRef, ShardBlock, ShardHeader};

use nerv_custody::tx::TransactionShell;

// ---- Helpers -------------------------------------------------------------

fn splitmix64(seed: u64) -> impl FnMut() -> [u8; 32] {
    let mut state = seed;
    move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let mut out = [0u8; 32];
        for chunk in out.chunks_exact_mut(8) {
            chunk.copy_from_slice(&(z ^ (z >> 31)).to_le_bytes());
        }
        out
    }
}

fn codec() -> CodecW {
    let w = expand(
        &BeaconRandomness::from_bytes([0x57; 32]),
        WeightVersion(1),
    );
    certify(
        &w,
        CertConfig {
            spark_samples_per_size: 2,
            column_rail_samples: 1,
        },
    )
    .unwrap();
    w
}

fn epoch_pk() -> PublicKey {
    // A deterministic, well-formed epoch public key: an ASeed plus a zero
    // T matrix. The seal flow only needs `pk.a_seed()` for `epoch_key_id`,
    // so a zero T is fine for the smoke path.
    let a_seed = ASeed::from_bytes([0x5E; 32]);
    PublicKey::new(a_seed, nerv_seal::ring::Mat2x8::default())
}

fn wallet_keys(seed_byte: u8) -> WalletKeys {
    let mut b = [0u8; 32];
    b[0] = seed_byte;
    let master = MasterSeed::from_bytes(b);
    WalletKeys::from_master(&master)
}

fn funded_wallet(
    seed: u64,
    n: usize,
    value: u64,
    shard: ShardId,
) -> Result<(WalletKeys, nerv_wallet::keys::AddressSet, nerv_wallet::scan::WalletNoteSet, u64), Box<dyn std::error::Error>> {
    let w = wallet_keys(seed as u8);
    let active = ShardSet::genesis();
    let addrs = nerv_wallet::keys::AddressSet::generate(&w, &active)?;
    let addr = addrs
        .addresses_for_shard(&shard)
        .into_iter()
        .next()
        .unwrap();

    let mut wallet_ns = nerv_wallet::scan::WalletNoteSet::default();
    let mut tree = nerv_custody::nct::NoteCommitmentTree::new();
    let mut rng = splitmix64(seed ^ 0xF00D);
    let anchor = tree.root();
    let mut total = 0u64;
    for i in 0..n {
        let pt = nerv_custody::note::NotePlaintext::new(value, rng(), rng(), None)?;
        let kem_randomness = rng();
        let (cm, sealed) = nerv_custody::seal_note(&pt, &addr.address, &kem_randomness)?;
        let idx = tree.append(&cm)?;
        let scanned = nerv_wallet::scan::scan_sealed_note(
            addrs.addresses(),
            &sealed.encode(),
            &cm,
            shard,
        )
        .ok_or("scan_sealed_note returned None")?;
        wallet_ns.insert(scanned);
        let wit = tree.witness(idx)?;
        // NotePosition stores siblings as `Vec<Goldilocks>` (flattened);
        // the witness's underlying `NctWitness.siblings` is `[NctDigest; DEPTH]`,
        // where each digest exposes `[Goldilocks; 4]` via `to_elements`.
        let siblings: Vec<Goldilocks> = wit
            .siblings
            .iter()
            .flat_map(|d| d.to_elements().unwrap())
            .collect();
        wallet_ns.set_position(
            *cm.as_bytes(),
            nerv_wallet::scan::NotePosition {
                leaf_index: idx,
                anchor,
                siblings,
                shard,
                received_height: 1,
            },
        );
        let _ = i;
        total += value;
    }
    Ok((w, addrs, wallet_ns, total))
}

fn dummy_qc() -> QuorumCertificate {
    QuorumCertificate {
        epoch: nerv_core::types::Epoch::from_u64(0),
        subject: Hash256::from_bytes([0u8; 32]),
        signers: 0,
        signatures: Vec::new(),
    }
}

fn minimal_block(shard: ShardId) -> ShardBlock {
    ShardBlock {
        shard,
        header: ShardHeader {
            prev: Hash256::from_bytes([0u8; 32]),
            height: Height::ZERO,
            nct_root: NctDigest::from_elements(&[Goldilocks::ZERO; 4]),
            nullifier_root: Hash256::from_bytes([0u8; 32]),
            transit_root: Hash256::from_bytes([0u8; 32]),
            params_root: Hash256::from_bytes([0x9A; 32]),
            derived: [0u8; 32],
            ct_batch_hash: Hash256::from_bytes([0u8; 32]),
            prev_reveal: None,
            registry: RegistryRef {
                interval: Interval::from_u64(0),
                root: Hash256::from_bytes([0u8; 32]),
            },
            fee_total: FeeSats::ZERO,
            producer_payout: Address::new(
                nerv_crypto::mlkem::EncapsulationKey::from_bytes([0u8; 1184]),
                ShardId::new(6, 0).unwrap(),
                [0u8; 32],
            )
            .unwrap(),
            qc_hash: Hash256::from_bytes([0u8; 32]),
        },
        legs: vec![],
        reversions: vec![],
        claims: vec![],
        qc: dummy_qc(),
    }
}

// ---- Driver --------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("NERV 3.0 cross-crate smoke test");
    println!("=================================");

    // ----- Step 1: construct_payment + TransactionWitness::generate ----
    println!("\n== Step 1: TransactionWitness::generate ==");
    let active = ShardSet::genesis();
    let spend_shard = active.ids()[7];
    let recv_shard = active.ids()[40];

    let (sender_keys, sender_addrs, notes, _total) = funded_wallet(0xC0, 3, 10_000_000_000, spend_shard)?;
    let (recipient_keys, _recipient_addrs) = {
        let w = wallet_keys(0xC1);
        let a = nerv_wallet::keys::AddressSet::generate(&w, &active)?;
        (w, a)
    };
    let _ = recipient_keys;
    let recipient_addr = sender_addrs.addresses_for_shard(&recv_shard).into_iter().next().unwrap_or_else(|| {
        // Fallback: build a recipient on the same shard if the recv shard has no wallet coverage.
        sender_addrs.addresses_for_shard(&spend_shard).into_iter().next().unwrap()
    });

    let spec = nerv_wallet::construct::PaymentSpec {
        recipient: recipient_addr.address.clone(),
        amount_nano: 15_000_000_000,
        fee_nano: 1_000,
        expiry_height: 5_000,
    };

    let c = codec();
    let pk = epoch_pk();
    let mut entropy: Box<dyn FnMut() -> [u8; 32]> = Box::new(splitmix64(0xC2));

    let constructed = nerv_wallet::construct::construct_payment(
        &spec,
        &notes,
        &sender_addrs,
        &sender_keys,
        &active,
        &c,
        &pk,
        100,
        &mut entropy,
    )?;

    println!(
        "  constructed txid = {} (legs = {}, noise_seeds = {})",
        hex32(constructed.txid.as_bytes()),
        constructed.shell.legs.len(),
        constructed.noise_seeds.len()
    );

    let witness = TransactionWitness::generate(
        &constructed.shell,
        constructed.custody_witness.clone(),
        &c,
        &pk,
        &constructed.noise_seeds,
    )?;

    println!(
        "  witness: custody.inputs = {}, custody.outputs = {}, custody.reverts = {}, seal.legs = {}",
        witness.custody.inputs.len(),
        witness.custody.outputs.len(),
        witness.custody.reverts.len(),
        witness.seal.len()
    );

    // ----- Step 2: encode / decode round-trip ----------------------------
    println!("\n== Step 2: TransactionWitness + TransactionShell round-trip via nerv-codec ==");
    // Round-trip the TransactionWitness itself — its witness bytes are
    // now canonical-encodable (added in nerv-proofs).
    let witness_bytes = witness.encode();
    println!("  TransactionWitness.encode -> {} bytes", witness_bytes.len());
    let witness_round = TransactionWitness::decode(&witness_bytes)?;
    assert_eq!(witness_round.custody.inputs.len(), witness.custody.inputs.len());
    assert_eq!(witness_round.custody.outputs.len(), witness.custody.outputs.len());
    assert_eq!(witness_round.custody.reverts.len(), witness.custody.reverts.len());
    assert_eq!(witness_round.seal.len(), witness.seal.len());
    assert_eq!(witness_round.epoch_key_id, witness.epoch_key_id);
    // Also deep-assert the input-witness siblings encode/decode preserves
    // both length and value (most error-prone type).
    for (a, b) in witness.custody.inputs.iter().zip(witness_round.custody.inputs.iter()) {
        assert_eq!(a.siblings.len(), b.siblings.len());
        for (sa, sb) in a.siblings.iter().zip(b.siblings.iter()) {
            assert_eq!(sa, sb);
        }
        assert_eq!(a.nullifier_key, b.nullifier_key);
        assert_eq!(a.leaf_index, b.leaf_index);
    }
    println!(
        "  TransactionWitness round-trip OK ({} bytes, {} seal legs, {} input siblings verified)",
        witness_bytes.len(),
        witness_round.seal.len(),
        witness_round.custody.inputs.iter().map(|i| i.siblings.len()).sum::<usize>()
    );

    let shell_bytes = constructed.shell.encode();
    let shell_round = TransactionShell::decode(&shell_bytes)?;
    assert_eq!(shell_round.legs.len(), constructed.shell.legs.len());
    assert_eq!(shell_round.txid()?, constructed.txid);
    println!(
        "  TransactionShell round-trip OK: {} legs, {} bytes",
        shell_round.legs.len(),
        shell_bytes.len()
    );

    // ----- Step 3: ShardBlock round-trip via nerv-codec -------------------
    println!("\n== Step 3: ShardBlock round-trip via nerv-codec ==");
    let block = minimal_block(spend_shard);
    let block_bytes = block.encode();
    println!("  ShardBlock.encode -> {} bytes", block_bytes.len());
    let block_round = ShardBlock::decode(&block_bytes)?;
    assert_eq!(block_round.shard, block.shard);
    assert_eq!(block_round.header.height, block.header.height);
    assert_eq!(block_round.header.prev, block.header.prev);
    assert_eq!(block_round.header.params_root, block.header.params_root);
    assert_eq!(block_round.header.fee_total, block.header.fee_total);
    assert_eq!(block_round.legs.len(), block.legs.len());
    println!(
        "  ShardBlock round-trip OK: shard = {:?}, height = {}, fee_total = {}, legs = {}",
        block_round.shard,
        block_round.header.height.as_u64(),
        block_round.header.fee_total.as_u64(),
        block_round.legs.len()
    );

    println!("\nAll smoke-test steps PASSED.");
    Ok(())
}

// Tiny hex helper (avoid pulling a hex crate).
fn hex32(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        s.push_str(&format!("{:02x}", byte));
    }
    s
}

// Quiet unused-import warnings on the unused types we brought in for
// readability; these would be used in the full apply_block step.
#[allow(dead_code)]
fn _unused_import_quiet(
    _: MlDsaSigningKey,
) {
}
