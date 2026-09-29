//! Self-spend shard migration (WP §8.5; erratum 190).

use std::collections::BTreeSet;

use nerv_codec::codec_w::CodecW;
use nerv_core::types::{ShardId, ShardSet};
use nerv_custody::WalletKeys;
use nerv_seal::encrypt::PublicKey;

use crate::construct::{construct_payment, ConstructedTx, ConstructError, PaymentSpec, WalletEntropy};
use crate::keys::AddressSet;
use crate::scan::WalletNoteSet;

/// The result of a rehome: the new transaction and the recipient info.
#[derive(Clone, Debug)]
pub struct RehomeResult {
    pub constructed: ConstructedTx,
    pub from_shard: ShardId,
    pub to_shard: ShardId,
    pub amount_nano: u64,
}

/// Rehome notes from one shard to another (erratum 190): a self-spend
/// where the sender is the recipient on a different shard. The wallet's
/// own addresses are both the sender (change) and the recipient.
#[allow(clippy::too_many_arguments)]
pub fn rehome_notes(
    notes: &WalletNoteSet,
    addresses: &mut AddressSet,
    keys: &WalletKeys,
    active: &ShardSet,
    from_shard: ShardId,
    to_shard: ShardId,
    amount_nano: u64,
    fee_nano: u64,
    codec: &CodecW,
    epoch_pk: &PublicKey,
    current_height: u64,
    entropy: &mut WalletEntropy<'_>,
) -> Result<RehomeResult, ConstructError> {
    // Ensure the wallet has an address on the target shard.
    addresses.ensure_shard(keys, active, to_shard)?;

    // Find the wallet's address on the target shard (the recipient).
    let recipient = addresses
        .addresses_for_shard(&to_shard)
        .into_iter()
        .next()
        .ok_or(ConstructError::NoChangeAddress { shard: to_shard })?;

    // The spec: a "payment" to self on the target shard.
    let spec = PaymentSpec {
        recipient: recipient.address.clone(),
        amount_nano,
        fee_nano,
        expiry_height: current_height + 500,
    };

    // Construct the cross-shard self-spend.
    let constructed = construct_payment(
        &spec, notes, addresses, keys, active, codec, epoch_pk,
        current_height, entropy,
    )?;

    Ok(RehomeResult {
        constructed,
        from_shard,
        to_shard,
        amount_nano,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::keys::AddressSet;
    use crate::scan::{scan_sealed_note, NotePosition, WalletNoteSet};
    use crate::testutil::SplitMix64;
    use nerv_codec::codec_w::WeightVersion;
    use nerv_codec::weight_gen::{certify, expand, BeaconRandomness, CertConfig};
    use nerv_custody::nct::NoteCommitmentTree;
    use nerv_custody::note::{seal_note, NotePlaintext};
    use nerv_custody::MasterSeed;
    use nerv_seal::encrypt::derive_reference_keypair;

    fn wallet(seed: u64) -> WalletKeys {
        WalletKeys::from_master(&MasterSeed::from_bytes(SplitMix64::new(seed).bytes32()))
    }

    fn codec() -> CodecW {
        let w = expand(&BeaconRandomness::from_bytes([0x57; 32]), WeightVersion(1));
        certify(&w, CertConfig { spark_samples_per_size: 2, column_rail_samples: 1 }).unwrap();
        w
    }

    fn epoch_pk() -> PublicKey {
        derive_reference_keypair(&[0xE0; 32]).unwrap().0
    }

    #[test]
    fn rehome_builds_a_cross_shard_self_spend() {
        let from = ShardSet::genesis().ids()[7];
        let to = ShardSet::genesis().ids()[40];
        let active = ShardSet::genesis();

        let w = wallet(0x3E);
        let mut addrs = AddressSet::generate_with_coverage(&w, &active, 4).unwrap();
        let addr = addrs.addresses_for_shard(&from).into_iter().next().unwrap();

        // Fund the wallet on `from`.
        let mut ns = WalletNoteSet::new();
        let mut tree = NoteCommitmentTree::new();
        let mut rng = SplitMix64::new(0x3F);
        let anchor = tree.root();
        for _ in 0..2 {
            let pt = NotePlaintext::new(
                10_000_000_000, rng.bytes32(), rng.bytes32(), None,
            ).unwrap();
            let (cm, sealed) = seal_note(&pt, &addr.address, &rng.bytes32()).unwrap();
            let idx = tree.append(&cm).unwrap();
            let scanned = scan_sealed_note(addrs.addresses(), &sealed.encode(), &cm, from).unwrap();
            ns.insert(scanned);
            let wit = tree.witness(idx).unwrap();
            ns.set_position(*cm.as_bytes(), NotePosition {
                leaf_index: idx, anchor,
                siblings: wit.siblings.iter().map(|d| d.to_elements().unwrap()).collect(),
                shard: from, received_height: 1,
            });
        }

        let mut entropy = SplitMix64::new(0x40);
        let mut ent = move || entropy.bytes32();
        let c = codec();
        let pk = epoch_pk();

        let result = rehome_notes(
            &ns, &mut addrs, &w, &active, from, to, 15_000_000_000, 1000,
            &c, &pk, 100, &mut ent,
        ).unwrap();

        assert_eq!(result.from_shard, from);
        assert_eq!(result.to_shard, to);

        // The constructed transaction is cross-shard.
        let canon = result.constructed.shell.canonicalize().unwrap();
        assert_eq!(canon.legs.len(), 2);
        assert_eq!(canon.legs[0].shard, from);
        assert_eq!(canon.legs[1].shard, to);
        // The issue leg is conditional (cross-shard).
        assert!(canon.legs[1].outputs[0].conditional);
    }
}
