//! The prover driver (WP §5.4, App A; erratum 187): drives the STARK
//! proving as a synchronous function the caller spawns as a background
//! job, overlapped with mixnet transit.

use nerv_codec::codec_w::CodecW;
use nerv_core::types::TxId;
use nerv_proofs::air::fs::{bind_transaction, canonical_nullifiers, shell_digest, TxPublicInputs};
use nerv_proofs::air::tx_air::prove_transaction;
use nerv_proofs::{FriShape, TransactionProof, TransactionWitness};
use nerv_seal::circuit_stmt::epoch_key_identifier;
use nerv_seal::encrypt::PublicKey;

use crate::construct::ConstructedTx;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ProveError {
    #[error("proofs: {0}")]
    Proofs(#[from] nerv_proofs::WitnessGenError),
    #[error("prover: {0}")]
    Prover(#[from] nerv_proofs::TxError),
    #[error("fs: {0}")]
    Fs(#[from] nerv_proofs::FsError),
    #[error("custody: {0}")]
    Custody(#[from] nerv_custody::CustodyError),
}

/// The proved transaction: the shell + proof + txid, ready for submission.
#[derive(Clone, Debug)]
pub struct ProvedTx {
    pub shell: nerv_custody::tx::TransactionShell,
    pub proof: TransactionProof,
    pub txid: TxId,
}

/// Prove a constructed transaction (erratum 187). Synchronous; the caller
/// spawns this as a tokio task for the background-job property.
pub fn prove(
    constructed: &ConstructedTx,
    codec: &CodecW,
    epoch_pk: &PublicKey,
    fri: &FriShape,
) -> Result<ProvedTx, ProveError> {
    let canon = constructed.shell.canonicalize()?;

    // The statement-11 binding.
    let epoch_key_id = nerv_core::hash::Hash256::from_bytes(
        epoch_key_identifier(epoch_pk.a_seed(), epoch_pk.t()),
    );
    let public = TxPublicInputs::derive(&canon, epoch_key_id)?;
    let txid = canon.txid()?;
    let sd = shell_digest(&canon)?;
    let nfs = canonical_nullifiers(&canon)?;
    let (mut transcript, _, _) = bind_transaction(&nfs, &txid, &sd, &public);

    // Generate the full witness.
    let witness = TransactionWitness::generate(
        &canon,
        constructed.custody_witness.clone(),
        codec,
        epoch_pk,
        &constructed.noise_seeds,
    )?;

    // Prove.
    let proof = prove_transaction(fri, &witness, &canon, codec, epoch_pk, &mut transcript)?;

    Ok(ProvedTx { shell: canon, proof, txid })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::construct::{construct_payment, PaymentSpec};
    use crate::keys::AddressSet;
    use crate::scan::{scan_sealed_note, NotePosition, WalletNoteSet};
    use crate::testutil::SplitMix64;
    use nerv_codec::codec_w::WeightVersion;
    use nerv_codec::weight_gen::{certify, expand, BeaconRandomness, CertConfig};
    use nerv_custody::nct::NoteCommitmentTree;
    use nerv_custody::note::{seal_note, NotePlaintext};
    use nerv_custody::{MasterSeed, WalletKeys};
    use nerv_core::types::ShardSet;
    use nerv_proofs::verify_transaction;
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

    fn fri() -> FriShape {
        FriShape {
            log_blowup: 4,
            num_queries: 56,
            log_final_poly_len: 1,
            max_log_arity: FriShape::DEFAULT_ARITY,
            commit_pow_bits: 0,
            query_pow_bits: 0,
        }
    }

    #[test]
    #[ignore = "full STARK proving: minutes in debug; `cargo test -p nerv-wallet --release -- --ignored`"]
    fn prove_and_verify_single_shard() {
        let shard = ShardSet::genesis().ids()[7];
        let w = wallet(0xE1);
        let active = ShardSet::genesis();
        let addrs = AddressSet::generate(&w, &active).unwrap();
        let addr = addrs.addresses_for_shard(&shard).into_iter().next().unwrap();

        // Fund the wallet.
        let mut wallet_ns = WalletNoteSet::new();
        let mut tree = NoteCommitmentTree::new();
        let mut rng = SplitMix64::new(0xE2);
        let anchor = tree.root();
        for _ in 0..2 {
            let pt = NotePlaintext::new(
                10_000_000_000, rng.bytes32(), rng.bytes32(), None,
            ).unwrap();
            let (cm, sealed) = seal_note(&pt, &addr.address, &rng.bytes32()).unwrap();
            let idx = tree.append(&cm).unwrap();
            let scanned = scan_sealed_note(addrs.addresses(), &sealed.encode(), &cm, shard).unwrap();
            wallet_ns.insert(scanned);
            let wit = tree.witness(idx).unwrap();
            wallet_ns.set_position(*cm.as_bytes(), NotePosition {
                leaf_index: idx, anchor,
                siblings: wit.siblings.iter().map(|d| d.to_elements().unwrap()).collect(),
                shard, received_height: 1,
            });
        }

        // A recipient on the same shard.
        let rw = wallet(0xE3);
        let raddrs = AddressSet::generate(&rw, &active).unwrap();
        let recipient = raddrs.addresses_for_shard(&shard).into_iter().next().unwrap();

        let spec = PaymentSpec {
            recipient: recipient.address.clone(),
            amount_nano: 15_000_000_000,
            fee_nano: 1000,
            expiry_height: 5_000,
        };

        let mut entropy = SplitMix64::new(0xE4);
        let mut ent = move || entropy.bytes32();
        let c = codec();
        let pk = epoch_pk();
        let constructed = construct_payment(
            &spec, &wallet_ns, &addrs, &w, &active, &c, &pk, 100, &mut ent,
        ).unwrap();

        // Prove.
        let proved = prove(&constructed, &c, &pk, &fri()).unwrap();

        // Verify: the round-trip.
        let canon = proved.shell.canonicalize().unwrap();
        let epoch_key_id = nerv_core::hash::Hash256::from_bytes(
            epoch_key_identifier(pk.a_seed(), pk.t()),
        );
        let public = TxPublicInputs::derive(&canon, epoch_key_id).unwrap();
        let sd = shell_digest(&canon).unwrap();
        let nfs = canonical_nullifiers(&canon).unwrap();
        let txid = canon.txid().unwrap();
        let (mut vt, _, _) = bind_transaction(&nfs, &txid, &sd, &public);
        let ok = verify_transaction(&fri(), &canon, &c, &pk, &proved.proof, &mut vt).unwrap();
        assert!(ok, "the proof must verify");
    }
}
