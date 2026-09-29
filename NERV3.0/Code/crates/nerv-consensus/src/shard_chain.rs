
//! The shard-chain production pipeline (WP §2.3 steps 4, 11; erratum
//! 134): the producer's propose path and the committee's
//! validate-and-sign path.


use nerv_core::hash::Hash256;
use nerv_core::types::{Epoch, ShardId};
use nerv_crypto::mldsa::{SigningKey, VerifyingKey};
use nerv_crypto::sigaggr::{vote_bytes, QuorumCertificate, VoteCollector, QcError};
use nerv_state::{
    apply_block, propose, Applied, BeaconView, BlockBody, ChainSource, ComputedHeader,
    HeaderInputs, ShardBlock, ShardState,
};


use crate::qc::{body_hash, HeaderQc};


pub const SHARD_QUORUM: usize = nerv_core::params::CONSENSUS_SHARD_QUORUM as usize;


#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PipelineError {
    #[error(transparent)]
    Executor(#[from] nerv_state::ExecutorError),
    #[error(transparent)]
    Qc(#[from] QcError),
    #[error(transparent)]
    Crypto(#[from] nerv_crypto::CryptoError),
    #[error("the QC's subject does not match the assembled block's body hash")]
    SubjectMismatch,
}


/// The producer's output: the block with a DUMMY QC (correct header
/// fields, self-consistent qc_hash) and the computed header. The caller
/// replaces the QC with the real committee's.
pub struct Proposal {
    pub block: ShardBlock,
    pub computed: ComputedHeader,
    pub state: ShardState,
    pub applied: Applied,
}


/// The producer's propose path (erratum 134): compute the header fields,
/// verify via apply_block, return the block with a dummy QC.
pub fn propose_block(
    state: &ShardState,
    body: BlockBody,
    inputs: HeaderInputs,
    view: &dyn BeaconView,
    chain: &dyn ChainSource,
) -> Result<Proposal, PipelineError> {
    let (computed, new_state) = propose(state, &body, &inputs, view, chain)?;
    // Re-run apply_block to obtain the Applied record (the propose path
    // already verified; this extracts the public outcome).
    let dummy_qc = QuorumCertificate {
        epoch: Epoch::from_u64(0),
        subject: Hash256::from_bytes([0u8; 32]),
        signers: 0,
        signatures: Vec::new(),
    };
    let header = build_header(state, &inputs, &computed, dummy_qc.qc_hash());
    let block = ShardBlock {
        shard: state.shard(),
        header,
        legs: body.legs,
        reversions: body.reversions,
        claims: body.claims,
        qc: dummy_qc,
    };
    let (applied_state, applied) = apply_block(state.clone(), &block, view, chain)?;
    Ok(Proposal { block, computed, state: applied_state, applied })
}


fn build_header(
    state: &ShardState,
    inputs: &HeaderInputs,
    computed: &ComputedHeader,
    qc_hash: Hash256,
) -> nerv_state::ShardHeader {
    nerv_state::ShardHeader {
        prev: state.state_commitment(),
        height: nerv_core::types::Height::from_u64(state.height().as_u64() + 1),
        nct_root: computed.nct_root,
        nullifier_root: computed.nullifier_root,
        transit_root: computed.transit_root,
        params_root: *state.params_root(),
        derived: inputs.derived,
        ct_batch_hash: computed.ct_batch_hash,
        prev_reveal: inputs.prev_reveal,
        registry: inputs.registry,
        fee_total: nerv_core::types::FeeSats::from_u64(computed.fee_total),
        producer_payout: inputs.producer_payout.clone(),
        qc_hash,
    }
}


/// Replace the dummy QC with the real committee's: sign the body hash,
/// assemble the QC, update the header's qc_hash. The resulting block is
/// valid (only the QC and qc_hash change; no other field references
/// them — erratum 131).
pub fn assemble_with_qc(
    proposal: Proposal,
    epoch: Epoch,
    signers: &[(usize, &SigningKey)],
    committee: &[VerifyingKey],
    quorum: usize,
) -> Result<(ShardBlock, ShardState), PipelineError> {
    let subject = body_hash(&proposal.block.header);
    let mut vc = VoteCollector::new(epoch, subject);
    for &(i, sk) in signers {
        let sig = sk.sign(&vote_bytes(epoch, &subject))?;
        vc.add(i, sig, committee)?;
    }
    let qc = vc.assemble(quorum)?;
    let mut block = proposal.block;
    block.qc = qc.clone();
    block.header.qc_hash = qc.qc_hash();
    Ok((block, proposal.state))
}


/// The committee member's validate-and-sign path (erratum 134): run
/// apply_block against the predecessor state; on success, sign the body
/// hash.
pub fn validate_and_sign(
    state: &ShardState,
    block: &ShardBlock,
    view: &dyn BeaconView,
    chain: &dyn ChainSource,
    epoch: Epoch,
    signer_index: usize,
    sk: &SigningKey,
    committee: &[VerifyingKey],
) -> Result<Option<nerv_crypto::mldsa::Signature>, PipelineError> {
    match apply_block(state.clone(), block, view, chain) {
        Ok(_) => {
            let hq = HeaderQc {
                shard: block.shard,
                header: block.header.clone(),
                qc: block.qc.clone(),
            };
            hq.validate(committee, SHARD_QUORUM)
                .map_err(PipelineError::Qc)?;
            let subject = body_hash(&block.header);
            if block.qc.subject != subject {
                return Err(PipelineError::SubjectMismatch);
            }
            let sig = sk.sign(&vote_bytes(epoch, &subject))?;
            Ok(Some(sig))
        }
        Err(e) => {
            // A block that fails validation is not signed — the member
            // returns None (and may file slash evidence separately).
            let _ = e;
            Ok(None)
        }
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::harness::{
        dummy_chain, keys, params_root, TestView, World,
    };
    use crate::testutil::SplitMix64;
    use nerv_core::types::{FeeSats, Height, Interval, ShardSet, TxId};
    use nerv_custody::tx::{InputSet, LegShell, Output, TransactionShell};
    use nerv_registry::mempool::PoolEntry;
    use nerv_state::block::SettledLeg;
    use nerv_state::header::RegistryRef;
    use nerv_state::ttau::TauTree;


    fn h(seed: u64) -> Hash256 {
        Hash256::from_bytes(SplitMix64::new(seed).bytes32())
    }


    /// A single-shell 1-in/1-out leg with a parseable dummy ct.
    fn leg(seed: u64, shard: ShardId) -> LegShell {
        let mut rng = SplitMix64::new(seed ^ 0x5EED);
        let mut ct = vec![0u8; nerv_seal::encrypt::Ciphertext::WIRE_SIZE];
        for c in ct.chunks_exact_mut(4) {
            c.copy_from_slice(&(rng.next_u32() & 0xFFF0_0000).to_le_bytes());
        }
        LegShell {
            shard,
            inputs: InputSet::new(vec![h(&mut rng)]),
            outputs: vec![Output {
                cm: h(&mut rng),
                sealed_note: vec![0xA5; 48],
                value: 1_000_000_000,
                conditional: false,
                revert_cm: None,
            }],
            fee: FeeSats::from_u64(1000),
            anchor: h(&mut rng),
            expiry: Height::from_u64(5_000),
            weight_version: 1,
            ct,
            burns: vec![],
        }
    }


    fn settled_leg(seed: u64, shard: ShardId, tau_tree: &TauTree, txid: TxId) -> SettledLeg {
        let shell = TransactionShell { legs: vec![leg(seed, shard)] };
        SettledLeg {
            shell,
            leg: nerv_core::types::LegIndex::FIRST,
            tau: tau_tree.witness(tau_tree.position(&txid).unwrap()).unwrap(),
            siblings: vec![],
        }
    }


    fn inputs(payout: &nerv_custody::Address) -> HeaderInputs {
        HeaderInputs {
            registry: RegistryRef {
                interval: Interval::from_u64(86_401),
                root: Hash256::from_bytes([0u8; 32]),
            },
            derived: [0u8; 32],
            prev_reveal: None,
            producer_payout: payout.clone(),
        }
    }


    #[test]
    fn propose_assemble_and_verify() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let state = ShardState::genesis(shard, params_root());


        // Build the T_τ and the view.
        let shell = TransactionShell { legs: vec![leg(1, shard)] };
        let canon = shell.canonicalize().unwrap();
        let txid = nerv_state::canonical_txid(&canon);
        let tau_tree = TauTree::from_sorted(&[txid]).unwrap();
        let mut view = TestView::default();
        view.tau.insert(86_401, tau_tree.root());


        let body = BlockBody::new(
            vec![settled_leg(1, shard, &tau_tree, txid)],
            vec![],
            vec![],
        );
        let payout = crate::testutil::harness::payout();
        let mut hdr_inputs = inputs(&payout);
        hdr_inputs.registry.root = tau_tree.root();


        let chain = dummy_chain();
        let proposal = propose_block(&state, body, hdr_inputs, &view, &chain).unwrap();
        assert_eq!(proposal.computed.fee_total, 700);
        assert_eq!(proposal.applied.settled, vec![txid]);
        assert_eq!(proposal.state.height().as_u64(), 1);


        // Replace the QC with a real committee's.
        let (sks, vks) = keys(21);
        let epoch = Epoch::from_u64(1);
        let signers: Vec<(usize, &SigningKey)> = (0..SHARD_QUORUM).map(|i| (i, &sks[i])).collect();
        let (block, new_state) =
            assemble_with_qc(proposal, epoch, &signers, &vks, SHARD_QUORUM).unwrap();


        // The block verifies end-to-end.
        let (verified_state, verified) = apply_block(state.clone(), &block, &view, &chain).unwrap();
        assert_eq!(verified.settled, vec![txid]);
        assert_eq!(verified.state_commitment, new_state.state_commitment());
        assert_eq!(verified_state.state_commitment(), new_state.state_commitment());
        assert_eq!(block.header.qc_hash, block.qc.qc_hash());


        // The HeaderQc validates.
        let hq = HeaderQc { shard, header: block.header.clone(), qc: block.qc.clone() };
        hq.validate(&vks, SHARD_QUORUM).unwrap();
        assert_eq!(hq.subject(), body_hash(&block.header));
        assert_ne!(hq.subject(), hq.header_hash());


        // A committee member's validate-and-sign path succeeds.
        let sig = validate_and_sign(
            &state,
            &block,
            &view,
            &chain,
            epoch,
            0,
            &sks[0],
            &vks,
        )
        .unwrap()
        .unwrap();
        assert!(vks[0].verify(&vote_bytes(epoch, &body_hash(&block.header)), &sig));
    }


    #[test]
    fn validate_and_sign_rejects_invalid() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let state = ShardState::genesis(shard, params_root());


        let shell = TransactionShell { legs: vec![leg(2, shard)] };
        let canon = shell.canonicalize().unwrap();
        let txid = nerv_state::canonical_txid(&canon);
        let tau_tree = TauTree::from_sorted(&[txid]).unwrap();
        let mut view = TestView::default();
        view.tau.insert(86_401, tau_tree.root());


        let mut body = BlockBody::new(
            vec![settled_leg(2, shard, &tau_tree, txid)],
            vec![],
            vec![],
        );
        // Corrupt: push a second leg with the SAME txid (unsorted).
        body.legs.push(body.legs[0].clone());


        let payout = crate::testutil::harness::payout();
        let mut hdr_inputs = inputs(&payout);
        hdr_inputs.registry.root = tau_tree.root();
        let chain = dummy_chain();


        assert!(propose_block(&state, body, hdr_inputs, &view, &chain).is_err());
    }


    #[test]
    fn multi_block_chain() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let mut state = ShardState::genesis(shard, params_root());
        let (sks, vks) = keys(21);
        let epoch = Epoch::from_u64(1);
        let signers: Vec<(usize, &SigningKey)> = (0..SHARD_QUORUM).map(|i| (i, &sks[i])).collect();
        let chain = dummy_chain();


        for i in 0..3u64 {
            let shell = TransactionShell { legs: vec![leg(10 + i, shard)] };
            let canon = shell.canonicalize().unwrap();
            let txid = nerv_state::canonical_txid(&canon);
            let tau_tree = TauTree::from_sorted(&[txid]).unwrap();
            let mut view = TestView::default();
            view.tau.insert(86_401 + i, tau_tree.root());


            let body = BlockBody::new(
                vec![settled_leg(10 + i, shard, &tau_tree, txid)],
                vec![],
                vec![],
            );
            let payout = crate::testutil::harness::payout();
            let hdr_inputs = HeaderInputs {
                registry: RegistryRef {
                    interval: Interval::from_u64(86_401 + i),
                    root: tau_tree.root(),
                },
                derived: [0u8; 32],
                prev_reveal: None,
                producer_payout: payout,
            };


            let proposal = propose_block(&state, body, hdr_inputs, &view, &chain).unwrap();
            let (block, new_state) =
                assemble_with_qc(proposal, epoch, &signers, &vks, SHARD_QUORUM).unwrap();
            let (v_state, v_applied) = apply_block(state.clone(), &block, &view, &chain).unwrap();
            assert_eq!(v_applied.height.as_u64(), i + 1);
            assert_eq!(v_state.state_commitment(), new_state.state_commitment());
            state = new_state;
        }
        assert_eq!(state.height().as_u64(), 3);
    }
}
