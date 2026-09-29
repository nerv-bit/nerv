//! Archival regeneration (WP §11.6; erratum 177): any finalized witness
//! from the DA-published block, forever. The 𝔾 path is the caller's.


use nerv_core::types::LegKey;
use nerv_state::block::{BlockLegTree, ShardBlock};


use crate::error::WitnessError;
use crate::witness::InclusionWitness;


/// Regenerate the inclusion witness for (txid, leg) from the archival
/// block. The 𝔾 path is the caller's (it requires the anchor context).
pub fn regenerate(
    block: &ShardBlock,
    txid: &nerv_core::types::TxId,
    leg: nerv_core::types::LegIndex,
) -> Result<InclusionWitness, WitnessError> {
    let resolved = block
        .resolve_legs()
        .map_err(|e| WitnessError::BlockUnresolvable(format!("{e}")))?;
    let tree = BlockLegTree::from_resolved(&resolved)
        .map_err(|e| WitnessError::BlockUnresolvable(format!("{e}")))?;
    let key = LegKey::new(*txid, leg);
    let index = tree
        .position(&key)
        .ok_or(WitnessError::LegNotFound { key })?;
    let w = tree
        .witness(index)
        .map_err(|e| WitnessError::BlockUnresolvable(format!("{e}")))?;
    Ok(InclusionWitness {
        txid: *txid,
        leg,
        leaf_index: w.index,
        siblings: w.siblings,
        shard: block.shard,
        height: block.header.height.as_u64(),
        interval: block.header.registry.interval.as_u64(),
        header_hash: block.header.header_hash(),
        g_path: None,
    })
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::witness::{verify, VerifyContext};
    use nerv_core::hash::Hash256;
    use nerv_core::types::{Epoch, FeeSats, Height, Interval, ShardSet, TxId};
    use nerv_core::field::Goldilocks;
    use nerv_crypto::mldsa::SigningKey;
    use nerv_crypto::sigaggr::{vote_bytes, VoteCollector};
    use nerv_custody::nct::NctDigest;
    use nerv_custody::tx::{InputSet, LegShell, Output, TransactionShell};
    use nerv_custody::{Address, NoteOpening, WalletKeys};
    use nerv_state::block::SettledLeg;
    use nerv_state::header::{RegistryRef, ShardHeader};
    use nerv_state::ttau::TauTree;


    fn payout() -> Address {
        let mut b = [0u8; 32];
        b[0] = 0xFA;
        let (sk, _, _) = {
            let mut b2 = [0u8; 32];
            b2.copy_from_slice(&b);
            let mut sign_seed = [0u8; 32];
            sign_seed.copy_from_slice(&b2);
            let mut kem_seed = [0u8; 64];
            kem_seed[..32].copy_from_slice(&b2);
            let (ek, dk) = nerv_crypto::mlkem::keypair_from_seed(&kem_seed).unwrap();
            let sk = SigningKey::from_seed(&sign_seed).unwrap();
            (sk, ek, dk)
        };
        let _ = sk;
        let mut d = [0u8; 1184];
        // Use a deterministic ML-KEM ek for the payout.
        let mut kem_seed = [0u8; 64];
        kem_seed[..32].copy_from_slice(&b);
        let (ek, _) = nerv_crypto::mlkem::keypair_from_seed(&kem_seed).unwrap();
        d.copy_from_slice(ek.as_bytes());
        let ek = nerv_crypto::mlkem::EncapsulationKey::from_bytes(d);
        let set = ShardSet::genesis();
        let tag = set
            .home_kappa(&nerv_core::types::kappa(ek.as_bytes()))
            .unwrap();
        Address::new(ek, tag, b).unwrap()
    }


    fn header(shard: nerv_core::types::ShardId, height: u64) -> ShardHeader {
        ShardHeader {
            prev: Hash256::from_bytes([1u8; 32]),
            height: Height::from_u64(height),
            nct_root: NctDigest::from_elements(&[Goldilocks::ONE; 4]),
            nullifier_root: Hash256::from_bytes([2u8; 32]),
            transit_root: Hash256::from_bytes([3u8; 32]),
            params_root: Hash256::from_bytes([4u8; 32]),
            derived: [0u8; 32],
            ct_batch_hash: Hash256::from_bytes([5u8; 32]),
            prev_reveal: None,
            registry: RegistryRef {
                interval: Interval::from_u64(86_400),
                root: Hash256::from_bytes([6u8; 32]),
            },
            fee_total: FeeSats::from_u64(1000),
            producer_payout: payout(),
            qc_hash: Hash256::from_bytes([7u8; 32]),
        }
    }


    fn block(shard: nerv_core::types::ShardId, n_legs: usize) -> (ShardBlock, Vec<TxId>) {
        let set = ShardSet::genesis();
        let mut txids = Vec::new();
        let mut legs = Vec::new();
        let mut all: Vec<TransactionShell> = Vec::new();
        for i in 0..n_legs {
            let shell = TransactionShell {
                legs: vec![LegShell {
                    shard,
                    inputs: InputSet::new(vec![Hash256::from_bytes([i as u8; 32])]),
                    outputs: vec![Output {
                        cm: Hash256::from_bytes([(i + 1) as u8; 32]),
                        sealed_note: vec![0xA5; 48],
                        value: 1_000_000_000,
                        conditional: false,
                        revert_cm: None,
                    }],
                    fee: FeeSats::from_u64(1000),
                    anchor: Hash256::from_bytes([9u8; 32]),
                    expiry: Height::from_u64(5_000),
                    weight_version: 1,
                    ct: vec![0u8; nerv_seal::encrypt::Ciphertext::WIRE_SIZE],
                    burns: vec![],
                }],
            };
            let txid = shell.txid().unwrap();
            txids.push(txid);
            all.push(shell);
        }
        all.sort_by(|a, b| {
            let ta = nerv_state::canonical_txid(&a.canonicalize().unwrap());
            let tb = nerv_state::canonical_txid(&b.canonicalize().unwrap());
            ta.cmp(&tb)
        });
        for shell in &all {
            let canon = shell.canonicalize().unwrap();
            let txid = nerv_state::canonical_txid(&canon);
            let tau = TauTree::from_sorted(&[txid]).unwrap();
            legs.push(SettledLeg {
                shell: canon,
                leg: nerv_core::types::LegIndex::FIRST,
                tau: tau.witness(0).unwrap(),
                siblings: vec![],
            });
        }
        let hdr = header(shard, 1);
        let qc = {
            let mut b = [0u8; 32];
            b[0] = 0xB0;
            let sk = SigningKey::from_seed(&b).unwrap();
            let epoch = Epoch::from_u64(1);
            let subject = hdr.header_hash();
            let roster = vec![*sk.verifying_key()];
            let mut vc = VoteCollector::new(epoch, subject);
            vc.add(0, sk.sign(&vote_bytes(epoch, &subject)).unwrap(), &roster).unwrap();
            vc.assemble(1).unwrap()
        };
        let block = ShardBlock {
            shard,
            header: hdr,
            legs,
            reversions: vec![],
            claims: vec![],
            qc,
        };
        (block, txids)
    }


    #[test]
    fn regenerate_and_verify() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let (blk, txids) = block(shard, 20);
        let resolved = blk.resolve_legs().unwrap();
        let tree = BlockLegTree::from_resolved(&resolved).unwrap();
        let root = tree.root();


        for (i, &txid) in txids.iter().enumerate() {
            let wit = regenerate(&blk, &txid, nerv_core::types::LegIndex::FIRST).unwrap();
            assert_eq!(wit.shard, shard);
            assert_eq!(wit.height, 1);
            assert_eq!(wit.header_hash, blk.header.header_hash());
            let ctx = VerifyContext { leg_tree_root: &root, g: None };
            assert!(verify(&wit, &ctx), "leg {i}");
        }


        // A missing leg.
        let ghost = TxId::from_hash(Hash256::from_bytes([0xEE; 32]));
        assert!(matches!(
            regenerate(&blk, &ghost, nerv_core::types::LegIndex::FIRST),
            Err(WitnessError::LegNotFound { .. })
        ));


        // Determinism: regenerating twice gives the same witness.
        let w1 = regenerate(&blk, &txids[5], nerv_core::types::LegIndex::FIRST).unwrap();
        let w2 = regenerate(&blk, &txids[5], nerv_core::types::LegIndex::FIRST).unwrap();
        assert_eq!(w1, w2);
    }
}
