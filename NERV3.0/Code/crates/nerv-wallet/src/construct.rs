//! Transaction construction (WP §3.6, App A; erratum 186): note
//! selection, output creation, leg assembly, delta computation, sealing.

use std::collections::BTreeSet;

use nerv_codec::codec_w::CodecW;
use nerv_codec::features::{build_leg_features, LegKind, LegMovement};
use nerv_core::hash::Hash256;
use nerv_core::types::{FeeSats, Height, ShardId, ShardSet, TxId};
use nerv_custody::nct::NctDigest;
use nerv_custody::note::{seal_note, NotePlaintext};
use nerv_custody::nullifier::derive_nullifier;
use nerv_custody::tx::{InputSet, LegShell, Output, TransactionShell};
use nerv_custody::{Address, NoteOpening, WalletKeys};
use nerv_proofs::custody_air::{CustodyWitness, InputWitness, OutputWitness, RevertWitness};
use nerv_seal::digitize::digitize;
use nerv_seal::encrypt::{Ciphertext, PublicKey};
use nerv_seal::sampling::NoiseSeed;

use crate::keys::AddressSet;
use crate::scan::{ScannedNote, WalletNoteSet};

/// The wallet's fresh-randomness source: one 32-byte draw per call.
pub type WalletEntropy<'a> = dyn FnMut() -> [u8; 32] + 'a;

/// The payment request.
#[derive(Clone, Debug)]
pub struct PaymentSpec {
    pub recipient: Address,
    pub amount_nano: u64,
    pub fee_nano: u64,
    pub expiry_height: u64,
}

/// The constructed transaction: everything the prover needs.
#[derive(Clone, Debug)]
pub struct ConstructedTx {
    pub shell: TransactionShell,
    pub txid: TxId,
    pub custody_witness: CustodyWitness,
    pub noise_seeds: Vec<NoiseSeed>,
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ConstructError {
    #[error("insufficient funds: {available} available, {needed} needed")]
    InsufficientFunds { available: u64, needed: u64 },
    #[error("no unspent notes to spend")]
    NoNotes,
    #[error("inputs span multiple shards — split into separate transactions")]
    MultiShardInputs,
    #[error("no wallet address on the change shard {shard}")]
    NoChangeAddress { shard: ShardId },
    #[error("delta is zero for leg {leg}")]
    ZeroDelta { leg: usize },
    #[error("inadmissible features for leg {leg}: {source:?}")]
    Inadmissible { leg: usize, source: nerv_codec::features::AdmissibilityViolation },

    #[error("custody: {0}")]
    Custody(#[from] nerv_custody::CustodyError),
    #[error("seal: {0}")]
    Seal(#[from] nerv_seal::SealError),
    #[error("proofs: {0}")]
    Proofs(#[from] nerv_proofs::WitnessGenError),
    #[error("prover: {0}")]
    Prover(#[from] nerv_proofs::TxError),
}

/// OS entropy for production use.
pub fn os_entropy() -> [u8; 32] {
    let mut b = [0u8; 32];
    getrandom::getrandom(&mut b).expect("OS entropy unavailable");
    b
}

/// Select the smallest sufficient set of unspent notes.
fn select_notes(
    wallet: &WalletNoteSet,
    needed: u64,
) -> Result<(Vec<(&[u8; 32], &ScannedNote)>, u64), ConstructError> {
    let mut unspent: Vec<(&[u8; 32], &ScannedNote)> = wallet.unspent();
    if unspent.is_empty() {
        return Err(ConstructError::NoNotes);
    }
    unspent.sort_by_key(|(_, n)| n.opening.value);
    let mut selected = Vec::new();
    let mut total = 0u64;
    for (cm, note) in unspent {
        selected.push((cm, note));
        total += note.opening.value;
        if total >= needed {
            return Ok((selected, total));
        }
    }
    Err(ConstructError::InsufficientFunds { available: total, needed })
}

/// Create one output note (fresh randomness from the entropy source).
fn make_output(
    value: u64,
    recipient: &Address,
    entropy: &mut WalletEntropy<'_>,
) -> Result<(Output, NoteOpening), ConstructError> {
    let rho = entropy();
    let blinding = entropy();
    let kem_rand = entropy();
    let pt = NotePlaintext::new(value, rho, blinding, None)?;
    let (cm, sealed) = seal_note(&pt, recipient, &kem_rand)?;
    let opening = NoteOpening {
        value,
        rho,
        delivery: *recipient.delivery().as_bytes(),
        blinding,
        pk_n: *recipient.pk_n(),
    };
    Ok((
        Output { cm, sealed_note: sealed.encode(), value, conditional: false, revert_cm: None },
        opening,
    ))
}

/// Create the revert note for a cross-shard output (D.3): same value,
/// homed to the spend shard, spendable by the sender.
fn make_revert(
    value: u64,
    sender_address: &Address,
    entropy: &mut WalletEntropy<'_>,
) -> Result<(Hash256, NoteOpening), ConstructError> {
    let rho = entropy();
    let blinding = entropy();
    let kem_rand = entropy();
    let pt = NotePlaintext::new(value, rho, blinding, None)?;
    let (cm, _) = seal_note(&pt, sender_address, &kem_rand)?;
    let opening = NoteOpening {
        value,
        rho,
        delivery: *sender_address.delivery().as_bytes(),
        blinding,
        pk_n: *sender_address.pk_n(),
    };
    Ok((cm, opening))
}

fn map_admissibility(
    e: nerv_codec::features::AdmissibilityViolation,
) -> nerv_codec::features::FeatureError {
    nerv_codec::features::FeatureError::ValueOutOfRange { value: 0 }
}




/// Compute the leg's delta and seal it, returning the ct bytes.
fn seal_leg_delta(
    codec: &CodecW,
    movement: &LegMovement,
    epoch_pk: &PublicKey,
    entropy: &mut WalletEntropy<'_>,
) -> Result<(Vec<u8>, NoiseSeed), ConstructError> {
    let features = build_leg_features(movement)
        .map_err(|e| ConstructError::Inadmissible { leg: 0, source: e })?;
       features
        .check_admissible()
        .map_err(|e| ConstructError::Inadmissible { leg: 0, source: map_admissibility(e) })?;

    let delta = codec.apply(&features);
    if delta.is_zero() {
        return Err(ConstructError::ZeroDelta { leg: 0 });
    }
    let noise_seed = NoiseSeed::from_bytes(entropy());
    let plaintext = digitize(&delta.0);
    let ct = Ciphertext::encrypt(epoch_pk, &noise_seed, &plaintext)?;
    Ok((ct.to_bytes().to_vec(), noise_seed))
}

/// Construct a payment (erratum 186). Handles single-shard and
/// two-leg cross-shard cases.
#[allow(clippy::too_many_arguments)]
pub fn construct_payment(
    spec: &PaymentSpec,
    wallet: &WalletNoteSet,
    addresses: &AddressSet,
    keys: &WalletKeys,
    active: &ShardSet,
    codec: &CodecW,
    epoch_pk: &PublicKey,
    current_height: u64,
    entropy: &mut WalletEntropy<'_>,
) -> Result<ConstructedTx, ConstructError> {
    let needed = spec.amount_nano + spec.fee_nano;
    let (selected, total_value) = select_notes(wallet, needed)?;
    let change = total_value - needed;

    // All inputs must be on the same shard.
    let input_shards: BTreeSet<ShardId> = selected
        .iter()
        .filter_map(|(_, n)| {
            wallet
                .position(n.opening.commitment().ok()?.as_bytes())
                .map(|p| p.shard)
        })
        .collect();
    if input_shards.len() > 1 {
        return Err(ConstructError::MultiShardInputs);
    }
    let input_shard = *input_shards.iter().next().ok_or(ConstructError::NoNotes)?;
    let recipient_shard = active
        .home_kappa(&nerv_core::types::kappa(spec.recipient.delivery().as_bytes()))
        .map_err(|e| ConstructError::Custody(e))?;

    // The change address: the wallet's first address on the input shard.
    let change_addr = addresses
        .addresses_for_shard(&input_shard)
        .into_iter()
        .next()
        .ok_or(ConstructError::NoChangeAddress { shard: input_shard })?;

    // Nullifiers (precomputed from the scan).
    let nullifiers: Vec<Hash256> = selected
        .iter()
        .map(|(_, n)| n.nullifier)
        .collect();

    // The anchor: the most recent NCT root from any input's position.
    let anchor = selected
        .iter()
        .filter_map(|(_, n)| wallet.position(n.opening.commitment().ok()?.as_bytes()))
        .next()
        .map(|p| p.anchor)
        .ok_or(ConstructError::NoNotes)?;

    let weight_version = codec.version().0;
    let expiry = Height::from_u64(spec.expiry_height);
    let is_cross_shard = recipient_shard != input_shard;

    // Build the legs and outputs.
    let (legs, output_openings, revert_openings, noise_seeds) = if is_cross_shard {
        build_cross_shard(
            spec, &selected, wallet, change_addr, codec, epoch_pk, input_shard,
            recipient_shard, anchor, weight_version, expiry, entropy,
        )?
    } else {
        build_single_shard(
            spec, &selected, wallet, change_addr, change, codec, epoch_pk,
            input_shard, anchor, weight_version, expiry, entropy,
        )?
    };

    // Build the shell.
    let shell = TransactionShell { legs };

    // Canonicalize and compute the txid.
    let canon = shell.canonicalize()?;
    let txid = TxId::from_hash(Hash256::concat(
        &nerv_core::constants::Domain::new("nerv.txid"),
        &{
            let mut buf = Vec::new();
            for l in &canon.legs {
                l.encode_into(&mut buf);
            }
            buf
        },
    ));

    // Build the custody witness.
    let mut inputs = Vec::new();
    for (_, note) in &selected {
        let pos = wallet
            .position(note.opening.commitment().ok()?.as_bytes())
            .ok_or(ConstructError::NoNotes)?;
        inputs.push(InputWitness {
            opening: note.opening.clone(),
            nullifier_key: note.nullifier_key,
            leaf_index: pos.leaf_index,
            siblings: pos.siblings.clone(),
            anchor: pos.anchor,
        });
    }

    let fees: Vec<u64> = canon.legs.iter().map(|l| l.fee.as_u64()).collect();
    let custody_witness = CustodyWitness {
        inputs,
        outputs: output_openings.iter().map(|o| OutputWitness { opening: o.clone() }).collect(),
        fees,
        reverts: revert_openings.iter().map(|o| RevertWitness { opening: o.clone() }).collect(),
        burns: vec![],
    };

    Ok(ConstructedTx { shell, txid, custody_witness, noise_seeds })
}

type LegParts = (
    Vec<LegShell>,
    Vec<NoteOpening>,
    Vec<NoteOpening>,
    Vec<NoiseSeed>,
);

#[allow(clippy::too_many_arguments)]
fn build_single_shard(
    spec: &PaymentSpec,
    selected: &[(&[u8; 32], &ScannedNote)],
    _wallet: &WalletNoteSet,
    change_addr: &crate::keys::WalletAddress,
    change: u64,
    codec: &CodecW,
    epoch_pk: &PublicKey,
    input_shard: ShardId,
    anchor: NctDigest,
    weight_version: u64,
    expiry: Height,
    entropy: &mut WalletEntropy<'_>,
) -> Result<LegParts, ConstructError> {
    // One leg: inputs + outputs on the same shard.
    let nullifiers: Vec<Hash256> = selected.iter().map(|(_, n)| n.nullifier).collect();

    // The recipient output.
    let (recipient_output, recipient_opening) =
        make_output(spec.amount_nano, &spec.recipient, entropy)?;

    // The change output (if any).
    let mut outputs = vec![recipient_output];
    let mut output_openings = vec![recipient_opening];
    if change > 0 {
        let (change_output, change_opening) =
            make_output(change, &change_addr.address, entropy)?;
        outputs.push(change_output);
        output_openings.push(change_opening);
    }

    // The movement for the feature vector.
    let movement = LegMovement {
        inputs: selected
            .iter()
            .map(|(_, n)| {
                (n.opening.delivery.to_vec(), n.opening.value)
            })
            .collect(),
        outputs: outputs
            .iter()
            .map(|o| (recipient_opening.delivery.to_vec(), o.value))
            .collect(),
        fee_nano: spec.fee_nano,
        kind: LegKind::SingleShard,
        expiry_height: expiry.as_u64(),
        epoch_length_blocks: nerv_core::types::INTERVALS_PER_EPOCH,
    };

    let (ct_bytes, noise_seed) = seal_leg_delta(codec, &movement, epoch_pk, entropy)?;

    let leg = LegShell {
        shard: input_shard,
        inputs: InputSet::new(nullifiers),
        outputs,
        fee: FeeSats::from_u64(spec.fee_nano),
        anchor: Hash256::from_bytes(*anchor.as_bytes()),
        expiry,
        weight_version,
        ct: ct_bytes,
        burns: vec![],
    };

    Ok((vec![leg], output_openings, vec![], vec![noise_seed]))
}

#[allow(clippy::too_many_arguments)]
fn build_cross_shard(
    spec: &PaymentSpec,
    selected: &[(&[u8; 32], &ScannedNote)],
    wallet: &WalletNoteSet,
    change_addr: &crate::keys::WalletAddress,
    codec: &CodecW,
    epoch_pk: &PublicKey,
    input_shard: ShardId,
    recipient_shard: ShardId,
    anchor: NctDigest,
    weight_version: u64,
    expiry: Height,
    entropy: &mut WalletEntropy<'_>,
) -> Result<LegParts, ConstructError> {
    let nullifiers: Vec<Hash256> = selected.iter().map(|(_, n)| n.nullifier).collect();
    let total_value: u64 = selected.iter().map(|(_, n)| n.opening.value).sum();
    let change = total_value - spec.amount_nano - spec.fee_nano;

    // The recipient output: conditional, on the recipient's shard.
    let (recipient_output, recipient_opening) =
        make_output(spec.amount_nano, &spec.recipient, entropy)?;
    let mut recipient_output = recipient_output;
    recipient_output.conditional = true;

    // The revert note: same value, on the spend shard, spendable by the sender.
    let (revert_cm, revert_opening) =
        make_revert(spec.amount_nano, &change_addr.address, entropy)?;
    recipient_output.revert_cm = Some(revert_cm);

    // The change output (on the spend shard).
    let mut spend_outputs = Vec::new();
    let mut output_openings = vec![recipient_opening];
    if change > 0 {
        let (change_output, change_opening) =
            make_output(change, &change_addr.address, entropy)?;
        spend_outputs.push(change_output);
        output_openings.push(change_opening);
    }

    // Fee split: evenly across the two legs.
    let fee_spend = spec.fee_nano / 2;
    let fee_issue = spec.fee_nano - fee_spend;

    // The spend leg's movement (inputs + change outputs).
    let spend_movement = LegMovement {
        inputs: selected
            .iter()
            .map(|(_, n)| (n.opening.delivery.to_vec(), n.opening.value))
            .collect(),
        outputs: spend_outputs
            .iter()
            .map(|o| (change_addr.ek.as_bytes().to_vec(), o.value))
            .collect(),
        fee_nano: fee_spend,
        kind: LegKind::CrossShardSpend,
        expiry_height: expiry.as_u64(),
        epoch_length_blocks: nerv_core::types::INTERVALS_PER_EPOCH,
    };
    let (ct_spend, seed_spend) = seal_leg_delta(codec, &spend_movement, epoch_pk, entropy)?;

    // The issue leg's movement (recipient output only).
    let issue_movement = LegMovement {
        inputs: vec![],
        outputs: vec![(spec.recipient.delivery().as_bytes().to_vec(), spec.amount_nano)],
        fee_nano: fee_issue,
        kind: LegKind::CrossShardIssue,
        expiry_height: expiry.as_u64(),
        epoch_length_blocks: nerv_core::types::INTERVALS_PER_EPOCH,
    };
    let (ct_issue, seed_issue) = seal_leg_delta(codec, &issue_movement, epoch_pk, entropy)?;

    let spend_leg = LegShell {
        shard: input_shard,
        inputs: InputSet::new(nullifiers),
        outputs: spend_outputs,
        fee: FeeSats::from_u64(fee_spend),
        anchor: Hash256::from_bytes(*anchor.as_bytes()),
        expiry,
        weight_version,
        ct: ct_spend,
        burns: vec![],
    };
    let issue_leg = LegShell {
        shard: recipient_shard,
        inputs: InputSet::new(vec![]),
        outputs: vec![recipient_output],
        fee: FeeSats::from_u64(fee_issue),
        anchor: Hash256::from_bytes([0u8; 32]),
        expiry,
        weight_version,
        ct: ct_issue,
        burns: vec![],
    };

    Ok((
        vec![spend_leg, issue_leg],
        output_openings,
        vec![revert_opening],
        vec![seed_spend, seed_issue],
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::keys::AddressSet;
    use crate::scan::{scan_sealed_note, NotePosition, WalletNoteSet};
    use crate::testutil::SplitMix64;
    use nerv_custody::nct::NoteCommitmentTree;
    use nerv_custody::{MasterSeed, WalletKeys};
    use nerv_seal::encrypt::derive_reference_keypair;

    fn wallet(seed: u64) -> WalletKeys {
        WalletKeys::from_master(&MasterSeed::from_bytes(SplitMix64::new(seed).bytes32()))
    }

    fn epoch_pk() -> PublicKey {
        derive_reference_keypair(&[0xE0; 32]).unwrap().0
    }

    fn codec() -> CodecW {
        use nerv_codec::codec_w::WeightVersion;
        use nerv_codec::weight_gen::{certify, expand, BeaconRandomness, CertConfig};
        let w = expand(&BeaconRandomness::from_bytes([0x57; 32]), WeightVersion(1));
        certify(&w, CertConfig { spark_samples_per_size: 2, column_rail_samples: 1 }).unwrap();
        w
    }

    /// Build a wallet with `n` notes of `value` each on `shard`.
    fn funded_wallet(
        seed: u64,
        n: usize,
        value: u64,
        shard: ShardId,
    ) -> (WalletKeys, AddressSet, WalletNoteSet, NctDigest, u64) {
        let w = wallet(seed);
        let active = ShardSet::genesis();
        let addrs = AddressSet::generate(&w, &active).unwrap();
        let addr = addrs.addresses_for_shard(&shard).into_iter().next().unwrap();

        let mut wallet_ns = WalletNoteSet::new();
        let mut tree = NoteCommitmentTree::new();
        let mut rng = SplitMix64::new(seed ^ 0xF00D);
        let anchor = tree.root();
        for i in 0..n {
            let pt = NotePlaintext::new(value, rng.bytes32(), rng.bytes32(), None).unwrap();
            let (cm, sealed) = seal_note(&pt, &addr.address, &rng.bytes32()).unwrap();
            let idx = tree.append(&cm).unwrap();
            let scanned = scan_sealed_note(
                addrs.addresses(), &sealed.encode(), &cm, shard,
            ).unwrap();
            wallet_ns.insert(scanned);
            let wit = tree.witness(idx).unwrap();
            wallet_ns.set_position(
                *cm.as_bytes(),
                NotePosition {
                    leaf_index: idx,
                    anchor,
                    siblings: wit.siblings.iter().map(|d| d.to_elements().unwrap()).collect(),
                    shard,
                    received_height: 1,
                },
            );
        }
        (w, addrs, wallet_ns, anchor, (n as u64) * value)
    }

    fn test_entropy(seed: u64) -> impl FnMut() -> [u8; 32] {
        let mut rng = SplitMix64::new(seed);
        move || rng.bytes32()
    }

    #[test]
    fn single_shard_construction() {
        let shard = ShardSet::genesis().ids()[7];
        let (w, addrs, notes, anchor, total) =
            funded_wallet(0xC0, 3, 10_000_000_000, shard);
        let active = ShardSet::genesis();
        let c = codec();
        let pk = epoch_pk();

        // A recipient on the same shard.
        let recipient_wallet = wallet(0xC1);
        let recipient_addrs = AddressSet::generate(&recipient_wallet, &active).unwrap();
        let recipient = recipient_addrs
            .addresses_for_shard(&shard)
            .into_iter()
           .next()
            .unwrap();

        let spec = PaymentSpec {
            recipient: recipient.address.clone(),
            amount_nano: 15_000_000_000,
            fee_nano: 1000,
            expiry_height: 5_000,
        };

        let mut entropy = test_entropy(0xC2);
        let ctx = construct_payment(
            &spec, &notes, &addrs, &w, &active, &c, &pk, 100, &mut entropy,
        ).unwrap();

        // The shell canonicalizes.
        let canon = ctx.shell.canonicalize().unwrap();
        assert_eq!(canon.legs.len(), 1);
        assert_eq!(canon.legs[0].shard, shard);

        // The txid is deterministic.
        assert_eq!(ctx.txid, TxId::from_hash(Hash256::concat(
            &nerv_core::constants::Domain::new("nerv.txid"),
            &{
                let mut buf = Vec::new();
                for l in &canon.legs { l.encode_into(&mut buf); }
                buf
            },
        )));

        // The custody witness has the right input count.
        assert_eq!(ctx.custody_witness.inputs.len(), 2); // 15G needs 2 notes of 10G
        assert!(ctx.custody_witness.outputs.len() >= 1);
        assert_eq!(ctx.custody_witness.reverts.len(), 0);
        assert_eq!(ctx.noise_seeds.len(), 1);

        // The ct bytes parse as a valid Ciphertext.
        let ct_len = nerv_seal::encrypt::Ciphertext::WIRE_SIZE;
        assert_eq!(canon.legs[0].ct.len(), ct_len);

        // Change: 30G - 15G - 700 = ~15G.
        let total_out: u64 = canon.legs[0].outputs.iter().map(|o| o.value).sum();
        let fee = canon.legs[0].fee.as_u64();
        let total_in: u64 = ctx.custody_witness.inputs.iter().map(|i| i.opening.value).sum();
        assert_eq!(total_in, total_out + fee);
    }

    #[test]
    fn cross_shard_construction() {
        let spend_shard = ShardSet::genesis().ids()[7];
        let recv_shard = ShardSet::genesis().ids()[40];
        let (w, addrs, notes, _anchor, _total) =
            funded_wallet(0xC3, 2, 50_000_000_000, spend_shard);
        let active = ShardSet::genesis();
        let c = codec();
        let pk = epoch_pk();

        // A recipient on a different shard.
        let recipient_wallet = wallet(0xC4);
        let recipient_addrs = AddressSet::generate(&recipient_wallet, &active).unwrap();
        let recipient = recipient_addrs
            .addresses_for_shard(&recv_shard)
            .into_iter()
            .next()
            .unwrap();

        let spec = PaymentSpec {
            recipient: recipient.address.clone(),
            amount_nano: 50_000_000_000,
            fee_nano: 1_000,
            expiry_height: 5_000,
        };

        let mut entropy = test_entropy(0xC5);
        let ctx = construct_payment(
            &spec, &notes, &addrs, &w, &active, &c, &pk, 100, &mut entropy,
        ).unwrap();

        let canon = ctx.shell.canonicalize().unwrap();
        assert_eq!(canon.legs.len(), 2);
        assert_eq!(canon.legs[0].shard, spend_shard);
        assert_eq!(canon.legs[1].shard, recv_shard);

        // The spend leg has inputs; the issue leg does not.
        assert!(!canon.legs[0].inputs.nullifiers.is_empty());
        assert!(canon.legs[1].inputs.nullifiers.is_empty());

        // The issue leg's output is conditional with a revert_cm.
        assert!(canon.legs[1].outputs[0].conditional);
        assert!(canon.legs[1].outputs[0].revert_cm.is_some());

        // The custody witness has a revert witness.
        assert_eq!(ctx.custody_witness.reverts.len(), 1);
        assert_eq!(ctx.custody_witness.reverts[0].opening.value, 50_000_000_000);

        // Two noise seeds (one per leg).
        assert_eq!(ctx.noise_seeds.len(), 2);
    }

    #[test]
    fn insufficient_funds() {
        let shard = ShardSet::genesis().ids()[7];
        let (w, addrs, notes, _, total) =
            funded_wallet(0xC6, 1, 1_000, shard);
        let active = ShardSet::genesis();
        let c = codec();
        let pk = epoch_pk();
        let recipient = addrs.addresses()[0].address.clone();

        let spec = PaymentSpec {
            recipient,
            amount_nano: total + 1,
            fee_nano: 100,
            expiry_height: 5_000,
        };
        let mut entropy = test_entropy(0xC7);
        assert!(matches!(
            construct_payment(&spec, &notes, &addrs, &w, &active, &c, &pk, 100, &mut entropy),
            Err(ConstructError::InsufficientFunds { .. })
        ));
    }

    #[test]
    fn deterministic_construction() {
        let shard = ShardSet::genesis().ids()[7];
        let (w, addrs, notes, _, _) =
            funded_wallet(0xC8, 2, 5_000_000_000, shard);
        let active = ShardSet::genesis();
        let c = codec();
        let pk = epoch_pk();
        let recipient = addrs.addresses()[0].address.clone();

        let spec = PaymentSpec {
            recipient,
            amount_nano: 3_000_000_000,
            fee_nano: 500,
            expiry_height: 5_000,
        };

        let mut e1 = test_entropy(0xC9);
        let mut e2 = test_entropy(0xC9);
        let r1 = construct_payment(&spec, &notes, &addrs, &w, &active, &c, &pk, 100, &mut e1).unwrap();
        let r2 = construct_payment(&spec, &notes, &addrs, &w, &active, &c, &pk, 100, &mut e2).unwrap();
        assert_eq!(r1.txid, r2.txid);
        assert_eq!(r1.shell, r2.shell);
    }
}
