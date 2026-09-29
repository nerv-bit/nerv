//! The deterministic executor (WP §4.3 rules 1–7, App D.3; errata
//! 107–114). `apply_block` is the pure Update: validate everything, then
//! mutate — by value, so an invalid block never damages the caller's
//! state.

use std::collections::HashSet;
use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::CT_BATCH;
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::params::{
    CUSTODY_D3_EXPIRY_GRACE_BLOCKS_L_GRACE, CUSTODY_D3_EXPIRY_MAX_BLOCKS_T_MAX,
    CUSTODY_D3_EXPIRY_MIN_BLOCKS_T_MIN,
};
use nerv_core::types::{Epoch, Height, Interval, LegIndex, LegKey, ShardId, TxId};
use nerv_custody::{
    transit_key, Address, NctDigest, NoteCommitmentTree, NullifierSet, TransitEntry,
    TransitEntryState, TransitLog,
};
use nerv_core::types::FeeSats;
use nerv_crypto::sigaggr::QuorumCertificate;
use nerv_seal::encrypt::Ciphertext;
use crate::anchor::AnchorRing;
use crate::block::{ct_sum, BlockLegTree, LegEvidence, ResolvedLeg, ShardBlock, TransitEvidence};
use crate::commitment::state_commitment;
use crate::error::ExecutorError;
use crate::header::{RegistryRef, ShardHeader, REVEAL_BYTES};
use crate::ttau::verify_tau_witness;
use nerv_economy::fees::AdmissionFloor;

/// The beacon's finalized facts the executor consumes: the T_τ root for a
/// registry interval (rule 1) and per-shard finalized transit roots at
/// heights (rule 5, D.3 evidence and triggers).
pub trait BeaconView {
    fn tau_root(&self, interval: Interval) -> Option<Hash256>;
    fn transit_root(&self, shard: ShardId, height: Height) -> Option<Hash256>;
}

/// The settling chain's own history: the resolved leg that settled at a
/// height (escrow-shell recovery, D.3(d)). `None` = data unavailable
/// (conservative skip, erratum 109).
pub trait ChainSource {
    fn settled_leg(&self, height: Height, key: &LegKey) -> Option<ResolvedLeg>;
}

/// One shard's authoritative state: the §4.2 tuple components plus the
/// validator-maintained anchor ring plus the D.4 admission floor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardState {
    shard: ShardId,
    nct: NoteCommitmentTree,
    nullifiers: NullifierSet,
    transit: TransitLog,
    anchors: AnchorRing,
    params_root: Hash256,
    prev: Hash256,
    height: Height,
    /// The D.4 admission floor (erratum 161; WP App D.4): the per-leg fee
    /// minimum `m·BASE_FLOOR_NANO`. `m` starts at 1 (genesis) and
    /// escalates/decays one-way per interval from the revealed Δ_B
    /// statistic (state machine lives in `nerv-economy`).
    fee_floor: AdmissionFloor,
}

impl ShardState {
    pub fn genesis(shard: ShardId, params_root: Hash256) -> ShardState {
        ShardState {
            shard,
            nct: NoteCommitmentTree::new(),
            nullifiers: NullifierSet::new(),
            transit: TransitLog::new(),
            anchors: AnchorRing::genesis(NoteCommitmentTree::new().root()),
            params_root,
            prev: Hash256::from_bytes([0u8; 32]),
            height: Height::ZERO,
            fee_floor: AdmissionFloor::genesis(),
        }
    }

    /// C_t = Commit(S_t) — the sole source of truth.
    pub fn state_commitment(&self) -> Hash256 {
        state_commitment(
            &self.nct.root(),
            &self.nullifiers.root(),
            &self.transit.root(),
            &self.params_root,
            &self.prev,
            self.height,
        )
    }

    pub fn shard(&self) -> ShardId {
        self.shard
    }

    pub fn height(&self) -> Height {
        self.height
    }

    /// S_t's `prev` field (C_{t−1}), not the current C_t.
    pub fn prev(&self) -> &Hash256 {
        &self.prev
    }

    pub fn params_root(&self) -> &Hash256 {
        &self.params_root
    }

    pub fn nct(&self) -> &NoteCommitmentTree {
        &self.nct
    }

    pub fn nullifiers(&self) -> &NullifierSet {
        &self.nullifiers
    }

    pub fn transit(&self) -> &TransitLog {
        &self.transit
    }

    pub fn anchors(&self) -> &AnchorRing {
        &self.anchors
    }

    /// The D.4 admission floor (erratum 161): the per-leg fee minimum.
    /// Read-only from outside the executor; mutated only by
    /// [`apply_block`] (rule 1.5 checks it; pass 2 observes the reveal).
    pub fn fee_floor(&self) -> &AdmissionFloor {
        &self.fee_floor
    }
}

impl Encode for ShardState {
   fn encode_into(&self, out: &mut Vec<u8>) {
       self.shard.encode_into(out);
       self.nct.encode_into(out);
       self.nullifiers.encode_into(out);
       self.transit.encode_into(out);
       self.anchors.encode_into(out);
       self.params_root.encode_into(out);
       self.prev.encode_into(out);
       self.height.encode_into(out);
       self.fee_floor.encode_into(out);
   }
   fn encoded_len(&self) -> usize {
       self.shard.encoded_len()
           + self.nct.encoded_len()
           + self.nullifiers.encoded_len()
           + self.transit.encoded_len()
           + self.anchors.encoded_len()
           + self.params_root.encoded_len()
           + self.prev.encoded_len()
           + self.height.encoded_len()
           + self.fee_floor.encoded_len()
   }
}


impl Decode for ShardState {
   fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
       Ok(ShardState {
           shard: ShardId::decode_from(r)?,
           nct: NoteCommitmentTree::decode_from(r)?,
           nullifiers: NullifierSet::decode_from(r)?,
           transit: TransitLog::decode_from(r)?,
           anchors: AnchorRing::decode_from(r)?,
           params_root: Hash256::decode_from(r)?,
           prev: Hash256::decode_from(r)?,
           height: Height::decode_from(r)?,
           fee_floor: AdmissionFloor::decode_from(r)?,
       })
   }
}


/// The public outcome of one applied block.
#[derive(Clone, Debug, PartialEq)]
pub struct Applied {

    pub height: Height,
    pub state_commitment: Hash256,
    pub header_hash: Hash256,
    pub leg_tree_root: Hash256,
    pub nct_root: NctDigest,
    pub nullifier_root: Hash256,
    pub transit_root: Hash256,
    pub ct_batch: Ciphertext,
    pub settled: Vec<TxId>,
    pub escrows_opened: Vec<TxId>,
    pub escrows_claimed: Vec<TxId>,
    pub escrows_reverted: Vec<TxId>,
    pub reverted_note_count: usize,
}

pub(crate) fn transit_evidence_ok(
    ev: &TransitEvidence,
    view: &dyn BeaconView,
    txid: &TxId,
    shard: ShardId,
    leg: LegIndex,
) -> bool {
    let Some(entry) = ev.proof.entry.as_ref() else { return false };
    if entry.txid != *txid || entry.shard != shard || entry.leg != leg {
        return false;
    }
    let Some(root) = view.transit_root(shard, ev.root_height) else { return false };
    ev.proof.verify_membership(&root, entry)
}

/// The D.3 shell laws visible at settled-leg index `leg` (errata
/// 107(b),(c)); the executor's and the fraud verifier's shared predicate.
pub(crate) fn shell_law_error(
   leg: usize,
   shell: &nerv_custody::tx::TransactionShell,
   settle: &nerv_custody::tx::LegShell,
   height: u64,
) -> Option<ExecutorError> {
   for l in &shell.legs {
       if !l.inputs.nullifiers.is_empty() && l.outputs.iter().any(|o| o.conditional) {
           return Some(ExecutorError::ShellConditionalOnSpendLeg { leg });
       }
   }
   let has_cond = shell.legs.iter().any(|l| l.outputs.iter().any(|o| o.conditional));
   if has_cond {
       let spend_legs = shell.legs.iter().filter(|l| !l.inputs.nullifiers.is_empty()).count();
       if spend_legs != 1 {
           return Some(ExecutorError::ShellNotSingleSpend { leg, spend_legs });
       }
       let e0 = shell.legs[0].expiry;
       if shell.legs.iter().any(|l| l.expiry != e0) {
           return Some(ExecutorError::ShellExpiryMismatch { leg });
       }
   }
   if has_cond && !settle.inputs.nullifiers.is_empty() {
       let lo = height + CUSTODY_D3_EXPIRY_MIN_BLOCKS_T_MIN;
       let hi = height + CUSTODY_D3_EXPIRY_MAX_BLOCKS_T_MAX;
       let e = settle.expiry.as_u64();
       if e < lo || e > hi {
           return Some(ExecutorError::ExpiryBounds { leg, expiry: e, lo, hi });
       }
   }
   None
}

fn issue_leg_indices(r: &ResolvedLeg) -> Vec<usize> {
    r.canon
        .legs
        .iter()
        .enumerate()
        .filter(|(_, l)| l.inputs.nullifiers.is_empty())
        .map(|(li, _)| li)
        .collect()
}

#[allow(clippy::too_many_lines)]
fn recover_escrow(
    st: &ShardState,
    chain: &dyn ChainSource,
    txid: &TxId,
    spend_leg: LegIndex,
    spent_keys: &HashSet<[u8; 32]>,
    consumed: &mut HashSet<[u8; 32]>,
    record: usize,
) -> Result<(Hash256, TransitEntry, ResolvedLeg), ExecutorError> {
    let key = transit_key(txid, &st.shard, spend_leg);
    let Some(entry) = st.transit.get(&key).cloned() else {
        return Err(ExecutorError::RecordEntryMissing { record });
    };
    if entry.state != TransitEntryState::Pending {
        return Err(ExecutorError::RecordEntryNotPending { record, state: entry.state.name() });
    }
    if !consumed.insert(*key.as_bytes()) {
        return Err(ExecutorError::RecordDoubleConsume { record });
    }
    if spent_keys.contains(key.as_bytes()) {
        return Err(ExecutorError::RecordTargetBornThisBlock { record });
    }
    let Some(r) = chain.settled_leg(entry.height, &LegKey::new(*txid, spend_leg)) else {
        return Err(ExecutorError::RecordShellUnavailable { record });
    };
    if r.txid != *txid || r.leg_shell().inputs.nullifiers.is_empty() {
        return Err(ExecutorError::RecordShellMismatch { record });
    }
    Ok((key, entry, r))
}

/// Update(C_t, B): validate rules 1–7 over the whole block, then apply.
/// By value — an invalid block drops the moved state, leaving the caller's
/// untouched (erratum 112).
#[allow(clippy::too_many_lines)]
pub fn apply_block(
    state: ShardState,
    block: &ShardBlock,
    view: &dyn BeaconView,
    chain: &dyn ChainSource,
) -> Result<(ShardState, Applied), ExecutorError> {
    let hdr = &block.header;
    let c_prev = state.state_commitment();
    let mut out = validate_and_apply(state, block, view, chain)?;
    let st = &mut out.post_state;

    // Rule 6 (post-application): the three committing roots in `hdr` must
    // equal the freshly-computed roots; otherwise the block has lied about
    // the post-state it builds on (errata 107, 110).
    if st.nct.root() != hdr.nct_root {
        return Err(ExecutorError::NctRootMismatch);
    }
    if st.nullifiers.root() != hdr.nullifier_root {
        return Err(ExecutorError::NullifierRootMismatch);
    }
    if st.transit.root() != hdr.transit_root {
        return Err(ExecutorError::TransitRootMismatch);
    }

    st.anchors.push_header(st.nct.root());
    st.prev = c_prev;
    st.height = hdr.height;

    // D.4 admission floor — observation step (erratum 161, gap 3). After
    // the reveal is processed (validated by the caller above via the
    // header's `prev_reveal`), feed its statistic to the floor so the
    // multiplier escalates on anomaly or decays on a quiet interval. The
    // next block's rule 1.5 sees the updated floor. `prev_reveal = None`
    // records a missed reveal ceremony (D.1(d)) — the skip-and-carry rule
    // does NOT push a zero statistic; the floor simply rests on its prior
    // state.
    if let Some(reveal) = hdr.prev_reveal {
        if reveal.len() != REVEAL_BYTES {
            return Err(ExecutorError::Internal("reveal length"));
        }
        st.fee_floor.observe_reveal(&reveal);
    }

    let PassOutput {
        post_state,
        leg_tree_root,
        ct_batch,
        fee_total: _,
        settled,
        escrows_opened,
        escrows_claimed,
        escrows_reverted,
        reverted_note_count,
    } = out;

    let applied = Applied {
        height: post_state.height,
        state_commitment: post_state.state_commitment(),
        header_hash: hdr.header_hash(),
        leg_tree_root,
        nct_root: post_state.nct.root(),
        nullifier_root: post_state.nullifiers.root(),
        transit_root: post_state.transit.root(),
        ct_batch,
        settled,
        escrows_opened,
        escrows_claimed,
        escrows_reverted,
        reverted_note_count,
    };
    Ok((post_state, applied))
}


/// The shared payload of `validate_and_apply`: the post-state with
/// `nct / nullifiers / transit` already updated, plus every public-facing
/// record `apply_block` packs into the `Applied` value and `propose` folds
/// into the `ComputedHeader`. `prev / height / anchors / fee_floor` are
/// intentionally still pre-block — those are finalized by the caller once
/// the root-reconciliation step succeeds.
struct PassOutput {
    post_state: ShardState,
    leg_tree_root: Hash256,
    ct_batch: Ciphertext,
    fee_total: u64,
    settled: Vec<TxId>,
    escrows_opened: Vec<TxId>,
    escrows_claimed: Vec<TxId>,
    escrows_reverted: Vec<TxId>,
    reverted_note_count: usize,
}


/// Rules 1–7 of `apply_block` minus the final root reconciliation: validates
/// the block and applies the state changes, returning the post-state
/// together with the per-record book-keeping. Used by `apply_block` (which
/// adds the root check and finalizes `prev / height / anchors / fee_floor`)
/// and by `propose` (which reads the freshly-applied roots to build the
/// header it returns).
#[allow(clippy::too_many_lines)]
fn validate_and_apply(
    state: ShardState,
    block: &ShardBlock,
    view: &dyn BeaconView,
    chain: &dyn ChainSource,
) -> Result<PassOutput, ExecutorError> {
    if block.shard != state.shard {
        return Err(ExecutorError::WrongShard { block: block.shard, state: state.shard });
    }
    let hdr = &block.header;
    let c_prev = state.state_commitment();
    // `c_prev` is captured here for `apply_block`'s `prev` finalization; not
    // consulted in pass 1 / pass 2 below.
    let _ = c_prev;
    if hdr.prev != c_prev {
        return Err(ExecutorError::PrevMismatch { expected: c_prev, found: hdr.prev });
    }
    let expected_height = state.height.as_u64() + 1;
    if hdr.height.as_u64() != expected_height {
        return Err(ExecutorError::HeightMismatch { expected: expected_height, found: hdr.height.as_u64() });
    }
    if hdr.params_root != state.params_root {
        return Err(ExecutorError::ParamsMismatch { expected: state.params_root, found: hdr.params_root });
    }
    let Some(tau_root) = view.tau_root(hdr.registry.interval) else {
        return Err(ExecutorError::RegistryNotFinalized { interval: hdr.registry.interval.as_u64() });
    };
    if tau_root != hdr.registry.root {
        return Err(ExecutorError::RegistryRootMismatch {
            interval: hdr.registry.interval.as_u64(),
            expected: tau_root,
            found: hdr.registry.root,
        });
    }
    if hdr.qc_hash != block.qc.qc_hash() {
        return Err(ExecutorError::QcHashMismatch);
    }

    let resolved = block.resolve_legs()?;
    let leg_tree = BlockLegTree::from_resolved(&resolved)?;

    // -- pass 1a: per-leg rules --
    let mut spent_nfs: HashSet<[u8; 32]> = HashSet::new();
    let mut spent_keys: HashSet<[u8; 32]> = HashSet::new();
    let mut fee: u64 = 0;
    for (i, (sl, r)) in block.legs.iter().zip(resolved.iter()).enumerate() {
        let leg = r.leg_shell();
        let shell = &r.canon;

        if let Some(e) = shell_law_error(i, shell, leg, hdr.height.as_u64()) {
            return Err(e);
        }
        let input_bearing = !leg.inputs.nullifiers.is_empty();


        // Rule 1.
        if !verify_tau_witness(&tau_root, sl.tau.index, &r.txid, &sl.tau.siblings) {
            return Err(ExecutorError::TauWitness { leg: i });
        }

        // Rule 1.5 — D.4 admission floor (erratum 161; WP App D.4). The
        // dynamic floor escalates on anomaly and decays otherwise; the
        // leg's declared fee must clear the current minimum. This sits
        // between rules 1 and 2 so a sub-floor leg is rejected before the
        // anchor-check (rule 2) even runs.
        let floor_nano = state.fee_floor.floor_nano();
        let leg_fee_nano = leg.fee.as_u64();
        if leg_fee_nano < floor_nano {
            return Err(ExecutorError::FeeBelowFloor {
                leg: i,
                fee_nano: leg_fee_nano,
                floor_nano,
            });
        }

        // Rule 2 (input-bearing legs only, erratum 103).
        if input_bearing {
            let fresh = NctDigest::try_from_hash256(&leg.anchor)
                .map(|d| state.anchors.contains(&d))
                .unwrap_or(false);
            if !fresh {
                return Err(ExecutorError::AnchorStale { leg: i });
            }
        }

        // Rule 3.
        for nf in &leg.inputs.nullifiers {
            if state.nullifiers.contains(nf) || !spent_nfs.insert(*nf.as_bytes()) {
                return Err(ExecutorError::NullifierSpent { leg: i, nf: *nf });
            }
        }

        // Rule 4.
        let key = transit_key(&r.txid, &state.shard, sl.leg);
        if state.transit.contains_key(&key) || !spent_keys.insert(*key.as_bytes()) {
            return Err(ExecutorError::TransitSpent { leg: i });
        }

        // Rule 5 (issue legs) + the D.3 settlement deadline.
        if !input_bearing {
            if leg.expiry < hdr.height {
                return Err(ExecutorError::SettlementDeadline {
                    leg: i,
                    expiry: leg.expiry.as_u64(),
                    height: hdr.height.as_u64(),
                });
            }
            let spends: Vec<usize> = shell
                .legs
                .iter()
                .enumerate()
                .filter(|(_, l)| !l.inputs.nullifiers.is_empty())
                .map(|(li, _)| li)
                .collect();
            if sl.siblings.len() != spends.len() {
                return Err(ExecutorError::SiblingCount {
                    leg: i,
                    expected: spends.len(),
                    found: sl.siblings.len(),
                });
            }
            for (j, &li) in spends.iter().enumerate() {
                let sib = &shell.legs[li];
                if !transit_evidence_ok(
                    &sl.siblings[j],
                    view,
                    &r.txid,
                    sib.shard,
                    LegIndex::from_u8(li as u8),
                ) {
                    return Err(ExecutorError::SiblingEvidence { leg: i, sibling: j });
                }
            }
        }

        fee = fee.checked_add(leg.fee.as_u64()).ok_or(ExecutorError::FeeOverflow)?;
    }

    let ct_batch = ct_sum(&resolved)?;
    if Hash256::concat(&CT_BATCH, &ct_batch.to_bytes()) != hdr.ct_batch_hash {
        return Err(ExecutorError::CtBatchHashMismatch);
    }
    if hdr.fee_total.as_u64() != fee {
        return Err(ExecutorError::FeeTotalMismatch);
    }

    // -- pass 1b: records --
    let mut consumed: HashSet<[u8; 32]> = HashSet::new();
    let mut reversion_mints: Vec<Vec<Hash256>> = Vec::with_capacity(block.reversions.len());
    for (ri, rec) in block.reversions.iter().enumerate() {
        let (_key, entry, r) =
            recover_escrow(&state, chain, &rec.txid, rec.spend_leg, &spent_keys, &mut consumed, ri)?;
        if hdr.height.as_u64() < entry.expiry.as_u64() + CUSTODY_D3_EXPIRY_GRACE_BLOCKS_L_GRACE {
            return Err(ExecutorError::ReversionNotDue {
                record: ri,
                expiry: entry.expiry.as_u64(),
                height: hdr.height.as_u64(),
            });
        }
        let issue = issue_leg_indices(&r);
        if rec.evidence.len() != issue.len() {
            return Err(ExecutorError::ReversionEvidenceCount {
                record: ri,
                expected: issue.len(),
                found: rec.evidence.len(),
            });
        }
        let mut mints = Vec::new();
        for (j, &li) in issue.iter().enumerate() {
            let ileg = &r.canon.legs[li];
            match &rec.evidence[j] {
                LegEvidence::Settled(ev) => {
                    if !transit_evidence_ok(ev, view, &rec.txid, ileg.shard, LegIndex::from_u8(li as u8)) {
                        return Err(ExecutorError::ReversionEvidence { record: ri, issue: j });
                    }
                }
                LegEvidence::Unsettled { proof } => {
                    let Some(root) = view.transit_root(ileg.shard, entry.expiry) else {
                        return Err(ExecutorError::ReversionUnsettledRootUnavailable { record: ri, issue: j });
                    };
                    let ikey = transit_key(&rec.txid, &ileg.shard, LegIndex::from_u8(li as u8));
                    if !proof.verify_non_membership(&root, &ikey) {
                        return Err(ExecutorError::ReversionBadUnsettled { record: ri, issue: j });
                    }
                    for out in &ileg.outputs {
                        if out.conditional {
                            match out.revert_cm {
                                Some(cm) => mints.push(cm),
                                None => return Err(ExecutorError::RecordShellMismatch { record: ri }),
                            }
                        }
                    }
                }
            }
        }
        reversion_mints.push(mints);
    }

    for (ri, rec) in block.claims.iter().enumerate() {
        let (_key, _entry, r) =
            recover_escrow(&state, chain, &rec.txid, rec.spend_leg, &spent_keys, &mut consumed, ri)?;
        let issue = issue_leg_indices(&r);
        if rec.evidence.len() != issue.len() {
            return Err(ExecutorError::ClaimEvidenceCount {
                record: ri,
                expected: issue.len(),
                found: rec.evidence.len(),
            });
        }
        for (j, &li) in issue.iter().enumerate() {
            let ileg = &r.canon.legs[li];
            if !transit_evidence_ok(&rec.evidence[j], view, &rec.txid, ileg.shard, LegIndex::from_u8(li as u8)) {
                return Err(ExecutorError::ClaimEvidence { record: ri, issue: j });
            }
        }
    }

    // -- rule 7: the omission scan (erratum 114) --
    for entry in state.transit.reversion_due(hdr.height) {
        if consumed.contains(entry.key().as_bytes()) {
            continue;
        }
        let Some(r) = chain.settled_leg(entry.height, &LegKey::new(entry.txid, entry.leg)) else {
            continue;
        };
        if r.txid != entry.txid {
            continue;
        }
        let triggerable = r
            .canon
            .legs
            .iter()
            .filter(|l| l.inputs.nullifiers.is_empty())
            .all(|l| view.transit_root(l.shard, entry.expiry).is_some());
        if triggerable {
            return Err(ExecutorError::OmittedDueReversion { txid: entry.txid });
        }
    }

    // -- pass 2: apply --
    let mut st = state;
    let mut settled = Vec::with_capacity(resolved.len());
    let mut escrows_opened = Vec::new();
    for (sl, r) in block.legs.iter().zip(resolved.iter()) {
        let leg = r.leg_shell();
        let has_cond = r.canon.legs.iter().any(|l| l.outputs.iter().any(|o| o.conditional));
        let input_bearing = !leg.inputs.nullifiers.is_empty();
        for out in &leg.outputs {
            st.nct.append(&out.cm).map_err(|_| ExecutorError::Internal("nct append"))?;
        }
        for nf in &leg.inputs.nullifiers {
            st.nullifiers
                .insert(nf, hdr.height.as_u64())
                .map_err(|_| ExecutorError::Internal("nullifier insert"))?;
        }
        let estate = if input_bearing && has_cond {
            escrows_opened.push(r.txid);
            TransitEntryState::Pending
        } else {
            TransitEntryState::Claimed
        };
        st.transit
            .insert_pending(TransitEntry {
                txid: r.txid,
                shard: st.shard,
                leg: sl.leg,
                state: estate,
                height: hdr.height,
                expiry: leg.expiry,
            })
            .map_err(|_| ExecutorError::Internal("transit insert"))?;
        settled.push(r.txid);
    }
    let mut escrows_reverted = Vec::new();
    let mut reverted_note_count = 0usize;
    for (rec, mints) in block.reversions.iter().zip(&reversion_mints) {
        let key = transit_key(&rec.txid, &st.shard, rec.spend_leg);
        st.transit
            .revert(&key, hdr.height)
            .map_err(|_| ExecutorError::Internal("transit revert"))?;
        for cm in mints {
            st.nct.append(cm).map_err(|_| ExecutorError::Internal("nct mint"))?;
            reverted_note_count += 1;
        }
        escrows_reverted.push(rec.txid);
    }
    let mut escrows_claimed = Vec::new();
    for rec in &block.claims {
        let key = transit_key(&rec.txid, &st.shard, rec.spend_leg);
        st.transit
            .claim(&key, hdr.height)
            .map_err(|_| ExecutorError::Internal("transit claim"))?;
        escrows_claimed.push(rec.txid);
    }

    // Root reconciliation is performed by the caller (`apply_block`); here
    // we just return the freshly-applied state and the per-record book-
    // keeping it produced.
    Ok(PassOutput {
        post_state: st,
        leg_tree_root: leg_tree.root(),
        ct_batch,
        fee_total: fee,
        settled,
        escrows_opened,
        escrows_claimed,
        escrows_reverted,
        reverted_note_count,
    })
}


// ---------------------------------------------------------------------------
// The producer's propose path (erratum 134): assemble the header fields and
// verify the body's self-consistency by re-running `apply_block` on a
// self-built block (a dummy QC stands in for the eventual committee
// certificate — `shard_chain::propose_block` substitutes the real QC before
// publishing).
// ---------------------------------------------------------------------------

/// The fields a producer computes and bundles into a proposed header before
/// the QC is attached.
#[derive(Clone, Debug, PartialEq)]
pub struct ComputedHeader {
    pub nct_root: NctDigest,
    pub nullifier_root: Hash256,
    pub transit_root: Hash256,
    pub fee_total: u64,
    pub ct_batch_hash: Hash256,
    pub ct_batch: Ciphertext,
}

/// The body the producer submits to the proposer (rule 1–7 still apply at
/// `apply_block`; this struct just separates the producer-supplied legs and
/// records from the QC and the header).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockBody {
    pub legs: Vec<crate::block::SettledLeg>,
    pub reversions: Vec<crate::block::ReversionRecord>,
    pub claims: Vec<crate::block::ClaimRecord>,
}

impl BlockBody {
    pub fn new(
        legs: Vec<crate::block::SettledLeg>,
        reversions: Vec<crate::block::ReversionRecord>,
        claims: Vec<crate::block::ClaimRecord>,
    ) -> BlockBody {
        BlockBody { legs, reversions, claims }
    }
}

/// The producer-supplied inputs that do not derive from the body itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeaderInputs {
    pub registry: RegistryRef,
    pub derived: [u8; 32],
    pub prev_reveal: Option<[u8; REVEAL_BYTES]>,
    pub producer_payout: Address,
}

/// Update(C_t, B): validate rules 1–7 over the proposed body, derive the
/// header fields, and return the canonical post-state. The body is checked
/// end-to-end via `validate_and_apply` (which runs the same rule-7 pipeline
/// `apply_block` uses, minus the final root reconciliation — that step
/// would require a header whose roots we haven't computed yet, so
/// `apply_block`'s caller re-runs the full pipeline once the QC is in place;
/// `shard_chain::propose_block` is the canonical consumer). The dummy QC
/// here stands in for the eventual committee certificate.
#[allow(clippy::too_many_lines)]
pub fn propose(
    state: ShardState,
    body: &BlockBody,
    inputs: &HeaderInputs,
    view: &dyn BeaconView,
    chain: &dyn ChainSource,
) -> Result<(ComputedHeader, ShardState), ExecutorError> {
    // -- Assemble a header with everything derivable up front. The three
    //    committing roots (`nct_root`, `nullifier_root`, `transit_root`) are
    //    placeholders: `validate_and_apply` doesn't consult them and the
    //    real values are taken from the post-state we return. --
    let dummy_qc = QuorumCertificate {
        epoch: Epoch::from_u64(0),
        subject: Hash256::from_bytes([0u8; 32]),
        signers: 0,
        signatures: Vec::new(),
    };
    let mut hdr = ShardHeader {
        prev: state.state_commitment(),
        height: Height::from_u64(state.height().as_u64() + 1),
        // Placeholders — overwritten with the post-state roots below.
        nct_root: NctDigest::from_bytes([0u8; 32])
            .map_err(|_| ExecutorError::Internal("propose: placeholder NctDigest"))?,
        nullifier_root: Hash256::from_bytes([0u8; 32]),
        transit_root: Hash256::from_bytes([0u8; 32]),
        params_root: *state.params_root(),
        derived: inputs.derived,
        // Placeholder — overwritten with the canonical H(ct_B) below.
        ct_batch_hash: Hash256::from_bytes([0u8; 32]),
        prev_reveal: inputs.prev_reveal,
        registry: inputs.registry,
        // Placeholder — overwritten with the canonical fee sum below.
        fee_total: FeeSats::from_u64(0),
        producer_payout: inputs.producer_payout.clone(),
        qc_hash: dummy_qc.qc_hash(),
    };
    let block = ShardBlock {
        shard: state.shard(),
        header: hdr.clone(),
        legs: body.legs.clone(),
        reversions: body.reversions.clone(),
        claims: body.claims.clone(),
        qc: dummy_qc,
    };

    // -- Validate the body, apply the state, and extract the canonical
    //    post-state. `validate_and_apply` returns the summed fee, the
    //    ct_batch, and every rule-7 record; we use the first two to
    //    complete the header. --
    let PassOutput {
        post_state: new_state,
        ct_batch,
        fee_total,
        ..
    } = validate_and_apply(state, &block, view, chain)?;

    // -- Adopt the freshly-applied roots and the canonical ct_batch /
    //    fee_total into both the assembled header (for the caller's
    //    downstream use) and the returned ComputedHeader. --
    let ct_batch_hash = Hash256::concat(&CT_BATCH, &ct_batch.to_bytes());
    hdr.nct_root = new_state.nct.root();
    hdr.nullifier_root = new_state.nullifiers.root();
    hdr.transit_root = new_state.transit.root();
    hdr.ct_batch_hash = ct_batch_hash;
    hdr.fee_total = FeeSats::from_u64(fee_total);

    let computed = ComputedHeader {
        nct_root: new_state.nct.root(),
        nullifier_root: new_state.nullifiers.root(),
        transit_root: new_state.transit.root(),
        fee_total,
        ct_batch_hash,
        ct_batch,
    };
    Ok((computed, new_state))
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::block::canonical_txid;
    use crate::commitment::genesis_commitment;
    use crate::header::RegistryRef;
    use crate::ttau::TauTree;
    use crate::testutil::SplitMix64;
    use nerv_core::field::Goldilocks;
    use nerv_core::types::{Epoch, FeeSats, Interval, ShardSet, kappa};
    use nerv_crypto::mlkem::{EncapsulationKey, SigningKey, VerifyingKey};
    use nerv_crypto::sigaggr::{vote_bytes, VoteCollector};
    use nerv_custody::tx::{InputSet, LegShell, Output, TransactionShell};
    use nerv_custody::Address;
    use std::collections::BTreeMap;
    use std::sync::OnceLock;

    const PARAMS: Hash256 = Hash256::from_bytes([0x9A; 32]);
    const TAU0: u64 = 86_400;

    fn h(rng: &mut SplitMix64) -> Hash256 {
        Hash256::from_bytes(rng.bytes32())
    }

    fn dummy_ct(rng: &mut SplitMix64) -> Vec<u8> {
        let mut out = vec![0u8; Ciphertext::WIRE_SIZE];
        for c in out.chunks_exact_mut(4) {
            c.copy_from_slice(&(rng.next_u32() & 0xFFF0_0000).to_le_bytes());
        }
        out
    }

    fn leg(
        rng: &mut SplitMix64,
        shard: ShardId,
        inputs: Vec<Hash256>,
        outputs: Vec<(u64, bool)>,
        fee: u64,
        expiry: u64,
        anchor: Hash256,
    ) -> LegShell {
        let outs = outputs
            .into_iter()
            .map(|(v, cond)| Output {
                cm: h(rng),
                sealed_note: vec![0xA5; 48],
                value: v,
                conditional: cond,
                revert_cm: cond.then(|| h(rng)),
            })
            .collect();
        LegShell {
            shard,
            inputs: InputSet::new(inputs),
            outputs: outs,
            fee: FeeSats::from_u64(fee),
            anchor,
            expiry: Height::from_u64(expiry),
            weight_version: 1,
            ct: dummy_ct(rng),
            burns: vec![],
        }
    }

    fn payout() -> Address {
        static PAYOUT: OnceLock<Address> = OnceLock::new();
        PAYOUT
            .get_or_init(|| {
                let mut rng = SplitMix64::new(0xFA0);
                let set = ShardSet::genesis();
                let mut d = [0u8; 1184];
                for c in d.chunks_exact_mut(8) {
                    c.copy_from_slice(&rng.next_u64().to_le_bytes());
                }
                let ek = EncapsulationKey::from_bytes(d);
                let tag = set.home_kappa(&kappa(ek.as_bytes())).unwrap();
                Address::new(ek, tag, rng.bytes32()).unwrap()
            })
            .clone()
    }

    use nerv_crypto::sigaggr::QuorumCertificate;
    fn shared_qc() -> QuorumCertificate {
        static QC: OnceLock<QuorumCertificate> = OnceLock::new();
        QC.get_or_init(|| {
            let mut rng = SplitMix64::new(0xB0C);
            let keys: Vec<SigningKey> = (0..21u64)
                .map(|i| {
                    let mut b = [0u8; 32];
                    b[..8].copy_from_slice(&rng.next_u64().to_le_bytes());
                    b[24..32].copy_from_slice(&i.to_le_bytes());
                    SigningKey::from_seed(&b).unwrap()
                })
                .collect();
            let roster: Vec<VerifyingKey> = keys.iter().map(|k| *k.verifying_key()).collect();
            let epoch = Epoch::from_u64(3);
            let subject = h(&mut rng);
            let mut vc = VoteCollector::new(epoch, subject);
            for (i, k) in keys.iter().enumerate().take(15) {
                vc.add(i, k.sign(&vote_bytes(epoch, &subject)).unwrap(), &roster).unwrap();
            }
            vc.assemble(15).unwrap()
        })
        .clone()
    }


    #[derive(Default)]
    struct TestWorld {
        tau: BTreeMap<u64, Hash256>,
        transit_roots: BTreeMap<(ShardId, u64), Hash256>,
        chain: BTreeMap<(u64, LegKey), ResolvedLeg>,
    }

    impl BeaconView for TestWorld {
        fn tau_root(&self, interval: Interval) -> Option<Hash256> {
            self.tau.get(&interval.as_u64()).copied()
        }
        fn transit_root(&self, shard: ShardId, height: Height) -> Option<Hash256> {
            self.transit_roots.get(&(shard, height.as_u64())).copied()
        }
    }

    impl ChainSource for TestWorld {
        fn settled_leg(&self, height: Height, key: &LegKey) -> Option<ResolvedLeg> {
            self.chain.get(&(height.as_u64(), *key)).cloned()
        }
    }

    use crate::block::{ClaimRecord, ReversionRecord, SettledLeg};
    use crate::ttau::TauWitness;

    fn block(
        shard: ShardId,
        interval: u64,
        legs: Vec<SettledLeg>,
        reversions: Vec<ReversionRecord>,
        claims: Vec<ClaimRecord>,
    ) -> ShardBlock {
        ShardBlock {
            shard,
            header: ShardHeader {
                prev: Hash256::from_bytes([0u8; 32]),
                height: Height::ZERO,
                nct_root: NctDigest::from_elements(&[Goldilocks::ZERO; 4]),
                nullifier_root: Hash256::from_bytes([0u8; 32]),
                transit_root: Hash256::from_bytes([0u8; 32]),
                params_root: PARAMS,
                derived: [0u8; 32],
                ct_batch_hash: Hash256::from_bytes([0u8; 32]),
                prev_reveal: None,
                registry: RegistryRef {
                    interval: Interval::from_u64(interval),
                    root: Hash256::from_bytes([0u8; 32]),
                },
                fee_total: FeeSats::ZERO,
                producer_payout: payout(),
                qc_hash: Hash256::from_bytes([0u8; 32]),
            },
            legs,
            reversions,
            claims,
            qc: shared_qc(),
        }
    }

    fn register_tau(world: &mut TestWorld, interval: u64, txids: &[TxId]) -> TauTree {
        let mut v = txids.to_vec();
        v.sort();
        let tree = TauTree::from_sorted(&v).unwrap();
        world.tau.insert(interval, tree.root());
        tree
    }

    /// The independent header completion: computes every header field from
    /// the state and the block's effects using custody directly (the
    /// differential the executor must agree with on valid blocks). Effect
    /// failures are ignored — invalid blocks never reach the post-root
    /// checks, so their sealed roots are irrelevant.
    fn seal(st: &ShardState, b: &mut ShardBlock, world: &mut TestWorld, mints: &[Vec<Hash256>]) {
        let h = Height::from_u64(st.height().as_u64() + 1);
        let resolved = b.resolve_legs().ok();
        let interval = b.header.registry.interval.as_u64();
        if !world.tau.contains_key(&interval) {
            let txids: Vec<TxId> = match &resolved {
                Some(rs) => {
                    let mut v: Vec<TxId> = rs.iter().map(|r| r.txid).collect();
                    v.sort();
                    v
                }
                None => vec![],
            };
            let tree = TauTree::from_sorted(&txids).unwrap();
            world.tau.insert(interval, tree.root());
        }
        b.header.prev = st.state_commitment();
        b.header.height = h;
        b.header.registry.root = world.tau[&interval];
        b.header.qc_hash = b.qc.qc_hash();
        let Some(resolved) = resolved else { return };

        let mut nct = st.nct().clone();
        let mut nfs = st.nullifiers().clone();
        let mut log = st.transit().clone();
        let mut fee = 0u64;
        for (sl, r) in b.legs.iter().zip(resolved.iter()) {
            let l = r.leg_shell();
            for out in &l.outputs {
                let _ = nct.append(&out.cm);
            }
            for nf in &l.inputs.nullifiers {
                let _ = nfs.insert(nf, h.as_u64());
            }
            let has_cond = r.canon.legs.iter().any(|x| x.outputs.iter().any(|o| o.conditional));
            let estate = if !l.inputs.nullifiers.is_empty() && has_cond {
                TransitEntryState::Pending
            } else {
                TransitEntryState::Claimed
            };
            let _ = log.insert_pending(TransitEntry {
                txid: r.txid,
                shard: st.shard(),
                leg: sl.leg,
                state: estate,
                height: h,
                expiry: l.expiry,
            });
            fee += l.fee.as_u64();
        }
        for (rec, mint) in b.reversions.iter().zip(mints.iter()) {
            let _ = log.revert(&transit_key(&rec.txid, &st.shard(), rec.spend_leg), h);
            for cm in mint {
                let _ = nct.append(cm);
            }
        }
        for rec in &b.claims {
            let _ = log.claim(&transit_key(&rec.txid, &st.shard(), rec.spend_leg), h);
        }
        b.header.nct_root = nct.root();
        b.header.nullifier_root = nfs.root();
        b.header.transit_root = log.root();
        b.header.fee_total = FeeSats::from_u64(fee);
        b.header.ct_batch_hash = Hash256::concat(&CT_BATCH, &ct_sum(&resolved).unwrap().to_bytes());
        for r in &resolved {
            world.chain.insert((h.as_u64(), r.key), r.clone());
        }
    }

    fn advance(mut st: ShardState, world: &mut TestWorld, n: u64) -> ShardState {
        for _ in 0..n {
            let h = st.height().as_u64() + 1;
            let mut b = block(st.shard(), TAU0 + h, vec![], vec![], vec![]);
            seal(&st, &mut b, world, &[]);
            let (s, _) = apply_block(st, &b, world, world).unwrap();
            st = s;
        }
        st
    }

    fn genesis(shard: ShardId) -> ShardState {
        ShardState::genesis(shard, PARAMS)
    }

    fn empty_anchor() -> Hash256 {
        NoteCommitmentTree::new().root().as_hash256()
    }

    fn one_leg_block(
        rng: &mut SplitMix64,
        shard: ShardId,
        st: &ShardState,
        world: &mut TestWorld,
        inputs: Vec<Hash256>,
        outputs: Vec<(u64, bool)>,
    ) -> (ShardBlock, TxId, Hash256) {
        let nf = inputs.first().copied().unwrap_or_else(|| h(rng));
        let l = leg(rng, shard, if inputs.is_empty() { vec![nf] } else { inputs.clone() }, outputs.clone(), 1000, 5_000, empty_anchor());
        let shell = TransactionShell { legs: vec![l] };
        let txid = shell.txid().unwrap();
        let interval = TAU0 + st.height().as_u64() + 1;
        let tau = register_tau(world, interval, &[txid]);
        let sl = SettledLeg {
            shell,
            leg: LegIndex::FIRST,
            tau: tau.witness(tau.position(&txid).unwrap()).unwrap(),
            siblings: vec![],
        };
        let mut b = block(shard, interval, vec![sl], vec![], vec![]);
        seal(st, &mut b, world, &[]);
        (b, txid, nf)
    }

    #[test]
    fn genesis_state_matches_c0() {
        let set = ShardSet::genesis();
        let st = genesis(set.ids()[7]);
        assert_eq!(st.height(), Height::ZERO);
        assert_eq!(st.state_commitment(), genesis_commitment(&PARAMS));
        assert_eq!(st.nct().leaf_count(), 0);
        assert!(st.transit().is_empty());
        assert_eq!(st.anchors().len(), 1);
    }

    #[test]
    fn single_shard_spend_settles_and_chains() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let mut rng = SplitMix64::new(0x5EED);
        let mut world = TestWorld::default();
        let st = genesis(shard);
        let (b, txid, nf) = one_leg_block(&mut rng, shard, &st, &mut world, vec![h(&mut rng)], vec![(1_000_000_000, false), (2_000_000_000, false)]);
        let (st1, app) = apply_block(st, &b, &world, &world).unwrap();
        assert_eq!(st1.height().as_u64(), 1);
        assert_eq!(app.settled, vec![txid]);
        assert_eq!(st1.nct().leaf_count(), 2);
        assert!(st1.nullifiers().contains(&nf));
        assert_eq!(st1.nct().root(), b.header.nct_root);
        assert_eq!(st1.nullifiers().root(), b.header.nullifier_root);
        assert_eq!(st1.transit().root(), b.header.transit_root);
        assert_eq!(app.state_commitment, st1.state_commitment());
        assert_eq!(app.header_hash, b.header.header_hash());
        assert_eq!(app.nct_root, b.header.nct_root);
        let key = transit_key(&txid, shard, LegIndex::FIRST);
        assert_eq!(st1.transit().get(&key).unwrap().state, TransitEntryState::Claimed);
        assert!(app.escrows_opened.is_empty());
        assert_eq!(app.reverted_note_count, 0);

        // The anchor ring advanced: block 2's leg anchors at block 1's root.
        let (b2, _, _) = one_leg_block(&mut rng, shard, &st1, &mut world, vec![h(&mut rng)], vec![(500, false)]);
        let (st2, _) = apply_block(st1, &b2, &world, &world).unwrap();
        assert_eq!(st2.height().as_u64(), 2);
        assert_eq!(st2.nct().leaf_count(), 3);
    }

    #[test]
    fn empty_block_advances() {
        let set = ShardSet::genesis();
        let shard = set.ids()[3];
        let mut world = TestWorld::default();
        let st = genesis(shard);
        let mut b = block(shard, TAU0 + 1, vec![], vec![], vec![]);
        seal(&st, &mut b, &mut world, &[]);
        let (st1, app) = apply_block(st, &b, &world, &world).unwrap();
        assert_eq!(st1.height().as_u64(), 1);
        assert_eq!(st1.nct().leaf_count(), 0);
        assert_eq!(app.ct_batch.to_bytes(), Ciphertext::zero().to_bytes());
        assert_eq!(app.state_commitment, st1.state_commitment());
        assert_ne!(app.state_commitment, genesis_commitment(&PARAMS));
    }

    #[test]
    fn wrong_shard_rejected() {
        let set = ShardSet::genesis();
        let mut world = TestWorld::default();
        let st = genesis(set.ids()[7]);
        let mut b = block(set.ids()[8], TAU0 + 1, vec![], vec![], vec![]);
        seal(&st, &mut b, &mut world, &[]);
        assert!(matches!(
            apply_block(st, &b, &world, &world),
            Err(ExecutorError::WrongShard { .. })
        ));
    }

    #[test]
    fn header_rejections() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let mut rng = SplitMix64::new(0x4EA0);
        let mut world = TestWorld::default();
        let st = genesis(shard);
        let (b, _, _) = one_leg_block(&mut rng, shard, &st, &mut world, vec![h(&mut rng)], vec![(1_000_000_000, false)]);

        let mut x = b.clone();
        x.header.prev = Hash256::from_bytes([1u8; 32]);
        assert!(matches!(
            apply_block(st.clone(), &x, &world, &world),
            Err(ExecutorError::PrevMismatch { .. })
        ));

        let mut x = b.clone();
        x.header.height = Height::from_u64(2);
        assert!(matches!(
            apply_block(st.clone(), &x, &world, &world),
            Err(ExecutorError::HeightMismatch { .. })
        ));

        let mut x = b.clone();
        x.header.params_root = Hash256::from_bytes([2u8; 32]);
        assert!(matches!(
            apply_block(st.clone(), &x, &world, &world),
            Err(ExecutorError::ParamsMismatch { .. })
        ));

        let mut x = b.clone();
        x.header.registry.interval = Interval::from_u64(TAU0 + 99);
        assert!(matches!(
            apply_block(st.clone(), &x, &world, &world),
            Err(ExecutorError::RegistryNotFinalized { .. })
        ));

        let mut x = b.clone();
        x.header.registry.root = Hash256::from_bytes([3u8; 32]);
        assert!(matches!(
            apply_block(st.clone(), &x, &world, &world),
            Err(ExecutorError::RegistryRootMismatch { .. })
        ));

        let mut x = b.clone();
        x.header.qc_hash = Hash256::from_bytes([4u8; 32]);
        assert!(matches!(
            apply_block(st.clone(), &x, &world, &world),
            Err(ExecutorError::QcHashMismatch)
        ));

        let mut x = b.clone();
        x.header.fee_total = FeeSats::from_u64(x.header.fee_total.as_u64() + 1);
        assert!(matches!(
            apply_block(st.clone(), &x, &world, &world),
            Err(ExecutorError::FeeTotalMismatch { .. })
        ));

        let mut x = b.clone();
        x.header.ct_batch_hash = Hash256::from_bytes([5u8; 32]);
        assert!(matches!(
            apply_block(st.clone(), &x, &world, &world),
            Err(ExecutorError::CtBatchHashMismatch)
        ));
    }

    #[test]
    fn post_root_mismatches_and_atomicity() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let mut rng = SplitMix64::new(0x4EA1);
        let mut world = TestWorld::default();
        let st = genesis(shard);
        let (b, _, _) = one_leg_block(&mut rng, shard, &st, &mut world, vec![h(&mut rng)], vec![(1_000_000_000, false)]);

        let mut x = b.clone();
        x.header.nct_root = NctDigest::from_elements(&[Goldilocks::ONE, Goldilocks::ZERO, Goldilocks::ZERO, Goldilocks::ZERO]);
        assert!(matches!(
            apply_block(st.clone(), &x, &world, &world),
            Err(ExecutorError::NctRootMismatch)
        ));

        let mut x = b.clone();
        x.header.nullifier_root = Hash256::from_bytes([6u8; 32]);
        assert!(matches!(
            apply_block(st.clone(), &x, &world, &world),
            Err(ExecutorError::NullifierRootMismatch)
        ));

        let mut x = b.clone();
        x.header.transit_root = Hash256::from_bytes([7u8; 32]);
        assert!(matches!(
            apply_block(st.clone(), &x, &world, &world),
            Err(ExecutorError::TransitRootMismatch)
        ));

        // Atomicity: the failures consumed clones; the original still applies.
        let (st1, _) = apply_block(st, &b, &world, &world).unwrap();
        assert_eq!(st1.height().as_u64(), 1);
    }

    #[test]
    fn stale_anchor_rejected() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let mut rng = SplitMix64::new(0x4EA2);
        let mut world = TestWorld::default();
        let st = genesis(shard);
        let (b, _, _) = one_leg_block(&mut rng, shard, &st, &mut world, vec![h(&mut rng)], vec![(1_000_000_000, false)]);
        let mut bad = b.clone();
        bad.legs[0].shell.legs[0].anchor = Hash256::from_bytes([8u8; 32]);
        assert!(matches!(
            apply_block(st.clone(), &bad, &world, &world),
            Err(ExecutorError::AnchorStale { leg: 0 })
        ));

        // After 64 finalized blocks the genesis seed is evicted.
        let st64 = advance(st, &mut world, 64);
        let (b65, _, _) = one_leg_block(&mut rng, shard, &st64, &mut world, vec![h(&mut rng)], vec![(1_000_000_000, false)]);
        let mut stale = b65.clone();
        stale.legs[0].shell.legs[0].anchor = empty_anchor();
        assert!(matches!(
            apply_block(st64.clone(), &stale, &world, &world),
            Err(ExecutorError::AnchorStale { leg: 0 })
        ));
        // The latest root is fresh.
        apply_block(st64, &b65, &world, &world).unwrap();
    }

    #[test]
    fn tau_witness_rejected() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let mut rng = SplitMix64::new(0x4EA3);
        let mut world = TestWorld::default();
        let st = genesis(shard);
        let (b, _, _) = one_leg_block(&mut rng, shard, &st, &mut world, vec![h(&mut rng)], vec![(1_000_000_000, false)]);
        let mut bad = b.clone();
        bad.legs[0].tau.siblings.push(Hash256::from_bytes([9u8; 32]));
        assert!(matches!(
            apply_block(st, &bad, &world, &world),
            Err(ExecutorError::TauWitness { leg: 0 })
        ));
    }

    #[test]
    fn nullifier_conflicts_rejected() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let mut rng = SplitMix64::new(0x4EA4);
        let mut world = TestWorld::default();
        let st = genesis(shard);
        let (b1, _, _) = one_leg_block(&mut rng, shard, &st, &mut world, vec![nf0(&mut rng)], vec![(1_000_000_000, false)]);
        let (_, st1) = {
            let r = apply_block(st.clone(), &b1, &world, &world).unwrap();
            ((), r.0)
        };


        // Pre-block: a different txid carrying the same nullifier.
        let (b2, _, _) = one_leg_block(&mut rng, shard, &st, &mut world, vec![b1.legs[0].shell.legs[0].inputs.nullifiers[0]], vec![(2_000_000_000, false)]);
        assert!(matches!(
            apply_block(st1.clone(), &b2, &world, &world),
            Err(ExecutorError::NullifierSpent { leg: 0, .. })
        ));

        // In-block: two legs, same nullifier, ascending order.
        let mut la = leg(&mut rng, shard, vec![h(&mut rng)], vec![(1, false)], 1000, 5_000, empty_anchor());
        let mut lb = leg(&mut rng, shard, vec![h(&mut rng)], vec![(1, false)], 1000, 5_000, empty_anchor());
        let shared = h(&mut rng);
        la.inputs.nullifiers[0] = shared;
        lb.inputs.nullifiers[0] = shared;
        let sa = TransactionShell { legs: vec![la] };
        let sb = TransactionShell { legs: vec![lb] };
       let (ta, tb) = (sa.txid().unwrap(), sb.txid().unwrap());
        let mk = |shell: TransactionShell| SettledLeg {
            shell,
            leg: LegIndex::FIRST,
            tau: TauWitness { index: 0, siblings: vec![] },
            siblings: vec![],
        };
        let (fa, fb) = (mk(sa), mk(sb));
        let interval = TAU0 + 1;
        let tree = register_tau(&mut world, interval, &[ta, tb]);
        let mut legs = if ta < tb { vec![fa, fb] } else { vec![fb, fa] };
        for l in &mut legs {
            let t = l.shell.txid().unwrap();
            l.tau = tree.witness(tree.position(&t).unwrap()).unwrap();
        }

        let mut b = block(shard, interval, legs, vec![], vec![]);
        seal(&st, &mut b, &mut world, &[]);
        assert!(matches!(
            apply_block(st, &b, &world, &world),
            Err(ExecutorError::NullifierSpent { leg: 1, .. })
        ));
    }

    fn nf0(rng: &mut SplitMix64) -> Hash256 {
        h(rng)
    }

    #[test]
    fn shell_rule_rejections() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let other = set.ids()[40];
        let third = set.ids()[9];
        let mut rng = SplitMix64::new(0x4EA5);
        let mut world = TestWorld::default();
        let st = genesis(shard);

        let mk = |rng: &mut SplitMix64, legs: Vec<LegShell>| -> (ShardBlock, TxId) {
            let shell = TransactionShell { legs };
            let txid = shell.txid().unwrap();
            let interval = TAU0 + 1;
            let tree = register_tau(&mut world, interval, &[txid]);
            let sl = SettledLeg {
                shell,
                leg: LegIndex::FIRST,
                tau: tree.witness(0).unwrap(),
                siblings: vec![],
            };
            let mut b = block(shard, interval, vec![sl], vec![], vec![]);
            seal(&st, &mut b, &mut world, &[]);
            (b, txid)
        };

        // Conditional output on an input-bearing leg.
        let (b, _) = mk(&mut rng, vec![leg(&mut rng, shard, vec![h(&mut rng)], vec![(1_000, true)], 1000, 5_000, empty_anchor())]);
        assert!(matches!(
            apply_block(st.clone(), &b, &world, &world),
            Err(ExecutorError::ShellConditionalOnSpendLeg { leg: 0 })
        ));

        // Two spend legs with a conditional issue leg.
        let (b, _) = mk(&mut rng, vec![
            leg(&mut rng, shard, vec![h(&mut rng)], vec![], 1000, 5_000, empty_anchor()),
            leg(&mut rng, other, vec![h(&mut rng)], vec![], 1000, 5_000, empty_anchor()),
            leg(&mut rng, third, vec![], vec![(1_000, true)], 1000, 5_000, empty_anchor()),
        ]);
        assert!(matches!(
            apply_block(st.clone(), &b, &world, &world),
            Err(ExecutorError::ShellNotSingleSpend { leg: 0, spend_legs: 2 })
        ));

        // Unequal expiries.
        let (b, _) = mk(&mut rng, vec![
            leg(&mut rng, shard, vec![h(&mut rng)], vec![], 1000, 61, empty_anchor()),
            leg(&mut rng, other, vec![], vec![(1_000, true)], 1000, 62, empty_anchor()),
        ]);
        assert!(matches!(
            apply_block(st.clone(), &b, &world, &world),
            Err(ExecutorError::ShellExpiryMismatch { leg: 0 })
        ));

        // Expiry below T_min.
        let (b, _) = mk(&mut rng, vec![
            leg(&mut rng, shard, vec![h(&mut rng)], vec![], 1000, 60, empty_anchor()),
            leg(&mut rng, other, vec![], vec![(1_000, true)], 1000, 60, empty_anchor()),
        ]);
        assert!(matches!(
            apply_block(st.clone(), &b, &world, &world),
            Err(ExecutorError::ExpiryBounds { leg: 0, expiry: 60, lo: 61, hi: 14401 })
        ));

        // Expiry above T_max.
        let (b, _) = mk(&mut rng, vec![
            leg(&mut rng, shard, vec![h(&mut rng)], vec![], 1000, 14402, empty_anchor()),
            leg(&mut rng, other, vec![], vec![(1_000, true)], 1000, 14402, empty_anchor()),
        ]);
        assert!(matches!(
            apply_block(st.clone(), &b, &world, &world),
            Err(ExecutorError::ExpiryBounds { leg: 0, expiry: 14402, .. })
        ));
    }

    #[test]
    fn issue_leg_rules() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let other = set.ids()[40];
        let mut rng = SplitMix64::new(0x4EA6);
        let mut world = TestWorld::default();
        let st = genesis(shard);

        // Sibling count: an issue leg with no evidence for its one spend leg.
        let shell = TransactionShell {
            legs: vec![
                leg(&mut rng, shard, vec![h(&mut rng)], vec![], 1000, 5_000, empty_anchor()),
                leg(&mut rng, other, vec![], vec![(1_000, false)], 1000, 5_000, empty_anchor()),
            ],
        };
        let txid = shell.txid().unwrap();
        let interval = TAU0 + 1;
        let tree = register_tau(&mut world, interval, &[txid]);
        let sl = SettledLeg {
            shell,
            leg: LegIndex::from_u8(1),
            tau: tree.witness(0).unwrap(),
            siblings: vec![],
        };
        let mut b = block(other, interval, vec![sl], vec![], vec![]);
        let st40 = genesis(other);
        seal(&st40, &mut b, &mut world, &[]);
        assert!(matches!(
            apply_block(st40.clone(), &b, &world, &world),
            Err(ExecutorError::SiblingCount { leg: 0, expected: 1, found: 0 })
        ));

        // Deadline: expiry below the receiving height.
        let st40 = advance(st40, &mut world, 61);
        assert_eq!(st40.height().as_u64(), 61);

        let shell = TransactionShell {
            legs: vec![
                leg(&mut rng, shard, vec![h(&mut rng)], vec![], 1000, 61, empty_anchor()),
                leg(&mut rng, other, vec![], vec![(1_000, false)], 1000, 61, empty_anchor()),
            ],
        };
        let txid = shell.txid().unwrap();
        let interval = TAU0 + 100;
        let tree = register_tau(&mut world, interval, &[txid]);
        let sl = SettledLeg {
            shell,
            leg: LegIndex::from_u8(1),
            tau: tree.witness(0).unwrap(),
            siblings: vec![],
        };
        let mut b = block(other, interval, vec![sl], vec![], vec![]);
        seal(&st40, &mut b, &mut world, &[]);
        assert!(matches!(
            apply_block(st40, &b, &world, &world),
            Err(ExecutorError::SettlementDeadline { leg: 0, expiry: 61, height: 62 })
        ));
    }

    /// The D.3 fixture: a conditional shell (spend@7, issue1@40, issue2@41,
    /// expiry 61), the spend settled (escrow Pending), issue1 settled in
    /// shard 40, and shard-7 advanced to height 69 — block 70 is pre-grace,
    /// block 71 is due.
    struct RevFix {
        st: ShardState,
        world: TestWorld,
        txid: TxId,
        key7: Hash256,
        cm2: Hash256,
        nm2: crate::custody_transit::TransitProof,
        ev1: TransitEvidence,
    }

    fn reversion_fixture(seed: u64) -> RevFix {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let s40 = set.ids()[40];
        let s41 = set.ids()[41];
        let mut rng = SplitMix64::new(seed);
        let mut world = TestWorld::default();

        let spend = leg(&mut rng, shard, vec![h(&mut rng)], vec![], 1000, 61, empty_anchor());
        let issue1 = leg(&mut rng, s40, vec![], vec![(1_000_000_000, true)], 1000, 61, h(&mut rng));
        let issue2 = leg(&mut rng, s41, vec![], vec![(2_000_000_000, true)], 400, 61, h(&mut rng));
        let cm2 = issue2.outputs[0].revert_cm.unwrap();
        let shell = TransactionShell { legs: vec![spend, issue1, issue2] };
        let txid = shell.txid().unwrap();

        // Block 1 (shard 7): the spend leg settles; escrow born Pending.
        let st = genesis(shard);
        let tree = register_tau(&mut world, TAU0 + 1, &[txid]);
        let sl = SettledLeg {
            shell: shell.clone(),
            leg: LegIndex::FIRST,
            tau: tree.witness(tree.position(&txid).unwrap()).unwrap(),
            siblings: vec![],
        };
        let mut b1 = block(shard, TAU0 + 1, vec![sl], vec![], vec![]);
        seal(&st, &mut b1, &mut world, &[]);
        let (st1, app1) = apply_block(st, &b1, &world, &world).unwrap();
        let key7 = transit_key(&txid, shard, LegIndex::FIRST);
        assert_eq!(st1.transit().get(&key7).unwrap().state, TransitEntryState::Pending);
        assert_eq!(app1.escrows_opened, vec![txid]);
        assert_eq!(st1.nct().leaf_count(), 0);
        world.transit_roots.insert((shard, 1), st1.transit().root());

        // Shard 40, block 1: issue1 settles with sibling evidence.
        let st40 = genesis(s40);
        let spend_entry = st1.transit().get(&key7).unwrap().clone();
        let ev = TransitEvidence {
            root_height: Height::from_u64(1),
            proof: st1.transit().membership_proof(&spend_entry),
        };
        let sl40 = SettledLeg {
            shell: shell.clone(),
            leg: LegIndex::from_u8(1),
            tau: tree.witness(tree.position(&txid).unwrap()).unwrap(),
            siblings: vec![ev],
        };
        let mut b40 = block(s40, TAU0 + 1, vec![sl40], vec![], vec![]);
        seal(&st40, &mut b40, &mut world, &[]);
        let (st40_1, _) = apply_block(st40, &b40, &world, &world).unwrap();
        let key40 = transit_key(&txid, s40, LegIndex::from_u8(1));
        assert_eq!(st40_1.transit().get(&key40).unwrap().state, TransitEntryState::Claimed);
        assert_eq!(st40_1.nct().leaf_count(), 1);
        world.transit_roots.insert((s40, 1), st40_1.transit().root());
        let issue1_entry = st40_1.transit().get(&key40).unwrap().clone();
        let ev1 = TransitEvidence {
            root_height: Height::from_u64(1),
            proof: st40_1.transit().membership_proof(&issue1_entry),
        };

        // Shard 41's finalized transit root at the expiry height (61),
        // without issue2's entry.
        let log41 = TransitLog::new();
        world.transit_roots.insert((s41, 61), log41.root());
        let ikey2 = transit_key(&txid, s41, LegIndex::from_u8(2));
        let nm2 = log41.non_membership_proof(&ikey2);

        let st69 = advance(st1, &mut world, 68);
        assert_eq!(st69.height().as_u64(), 69);
        RevFix { st: st69, world, txid, key7, cm2, nm2, ev1 }
    }

    fn mixed_reversion(fix: &RevFix) -> ReversionRecord {
        ReversionRecord {
            txid: fix.txid,
            spend_leg: LegIndex::FIRST,
            evidence: vec![
                LegEvidence::Settled(fix.ev1.clone()),
                LegEvidence::Unsettled { proof: fix.nm2.clone() },
            ],
        }
    }

    #[test]
    fn reversion_lifecycle_mints_only_unsettled_legs() {
        let mut fix = reversion_fixture(0x0E5A);
        let st70 = {
            let mut w = std::mem::take(&mut fix.world);
            let s = advance(fix.st.clone(), &mut w, 1);
            fix.world = w;
            s
        };
        assert_eq!(st70.height().as_u64(), 70);
        let rec = mixed_reversion(&fix);
        let mut b = block(st70.shard(), TAU0 + 71, vec![], vec![rec], vec![]);
        {
            let mut w = std::mem::take(&mut fix.world);
            seal(&st70, &mut b, &mut w, &[vec![fix.cm2]]);
            fix.world = w;
        }
        let (st71, app) = apply_block(st70, &b, &fix.world, &fix.world).unwrap();
        assert_eq!(st71.height().as_u64(), 71);
        assert_eq!(st71.transit().get(&fix.key7).unwrap().state, TransitEntryState::Reverted);
        assert_eq!(app.escrows_reverted, vec![fix.txid]);
        assert_eq!(app.reverted_note_count, 1);
        assert_eq!(st71.nct().leaf_count(), 1);
        assert_eq!(st71.nct().root(), b.header.nct_root);
    }

    #[test]
    fn omitted_due_reversion_invalid() {
        let mut fix = reversion_fixture(0x0E5B);
        let st70 = {
            let mut w = std::mem::take(&mut fix.world);
            let s = advance(fix.st.clone(), &mut w, 1);
            fix.world = w;
            s
        };
        let mut b = block(st70.shard(), TAU0 + 71, vec![], vec![], vec![]);
        {
            let mut w = std::mem::take(&mut fix.world);
            seal(&st70, &mut b, &mut w, &[]);
            fix.world = w;
        }
        assert!(matches!(
            apply_block(st70, &b, &fix.world, &fix.world),
            Err(ExecutorError::OmittedDueReversion { txid }) if txid == fix.txid
        ));
    }

    #[test]
    fn early_reversion_rejected() {
        let mut fix = reversion_fixture(0x0E5C);
        let rec = mixed_reversion(&fix);
        let mut b = block(fix.st.shard(), TAU0 + 70, vec![], vec![rec], vec![]);
        {
            let mut w = std::mem::take(&mut fix.world);
            seal(&fix.st, &mut b, &mut w, &[vec![fix.cm2]]);
            fix.world = w;
        }
        assert!(matches!(
            apply_block(fix.st, &b, &fix.world, &fix.world),
            Err(ExecutorError::ReversionNotDue { expiry: 61, height: 70, .. })
        ));
    }

    #[test]
    fn record_rejections() {
        let mut fix = reversion_fixture(0x0E5D);
        let st70 = {
            let mut w = std::mem::take(&mut fix.world);
            let s = advance(fix.st.clone(), &mut w, 1);
            fix.world = w;
            s
        };
        let shard = st70.shard();

        // Missing entry.
        let rec = ReversionRecord {
            txid: TxId::from_hash(Hash256::from_bytes([0xEE; 32])),
            spend_leg: LegIndex::FIRST,
            evidence: vec![],
        };
        let mut b = block(shard, TAU0 + 71, vec![], vec![rec], vec![]);
        seal(&st70, &mut b, &mut fix.world, &[]);
        assert!(matches!(
            apply_block(st70.clone(), &b, &fix.world, &fix.world),
            Err(ExecutorError::RecordEntryMissing { record: 0 })
        ));

        // Evidence count.
        let rec = ReversionRecord {
            txid: fix.txid,
            spend_leg: LegIndex::FIRST,
            evidence: vec![LegEvidence::Unsettled { proof: fix.nm2.clone() }],
        };
        let mut b = block(shard, TAU0 + 71, vec![], vec![rec], vec![]);
        seal(&st70, &mut b, &mut fix.world, &[]);
        assert!(matches!(
            apply_block(st70.clone(), &b, &fix.world, &fix.world),
            Err(ExecutorError::ReversionEvidenceCount { expected: 2, found: 1 })
        ));

        // Bad settled-evidence (tampered sibling).
        let mut ev1 = fix.ev1.clone();
        if ev1.proof.siblings.is_empty() {
            ev1.proof.siblings.push((1u16, Hash256::from_bytes([0xEE; 32])));
        } else {
            ev1.proof.siblings[0].1 = Hash256::from_bytes([0xEE; 32]);
        }
        let rec = ReversionRecord {
            txid: fix.txid,
            spend_leg: LegIndex::FIRST,
            evidence: vec![
                LegEvidence::Settled(ev1),
                LegEvidence::Unsettled { proof: fix.nm2.clone() },
            ],
        };
        let mut b = block(shard, TAU0 + 71, vec![], vec![rec], vec![]);
        seal(&st70, &mut b, &mut fix.world, &[]);
        assert!(matches!(
            apply_block(st70.clone(), &b, &fix.world, &fix.world),
            Err(ExecutorError::ReversionEvidence { issue: 0, .. })
        ));

        // Unsettled root unavailable: drop shard-41's expiry-height root.
        let set = ShardSet::genesis();
        fix.world.transit_roots.remove(&(set.ids()[41], 61));
        let rec = mixed_reversion(&fix);
        let mut b = block(shard, TAU0 + 71, vec![], vec![rec], vec![]);
        seal(&st70, &mut fix.world, &[]);
        assert!(matches!(
            apply_block(st70.clone(), &b, &fix.world, &fix.world),
            Err(ExecutorError::ReversionUnsettledRootUnavailable { issue: 1 })
        ));
        fix.world.transit_roots.insert((set.ids()[41], 61), TransitLog::new().root());

        // Bad unsettled-evidence: proof generated against a different root.
        let mut log41 = TransitLog::new();
        let unrelated = TransitEntry::new_pending(
            TxId::from_hash(Hash256::from_bytes([0xAB; 32])),
            set.ids()[41],
            LegIndex::FIRST,
            Height::from_u64(1),
            Height::from_u64(500),
        );
        log41.insert_pending(unrelated).unwrap();
        let ikey2 = transit_key(&fix.txid, set.ids()[41], LegIndex::from_u8(2));
        let bad_nm = log41.non_membership_proof(&ikey2);
        let rec = ReversionRecord {
            txid: fix.txid,
            spend_leg: LegIndex::FIRST,
            evidence: vec![
                LegEvidence::Settled(fix.ev1.clone()),
                LegEvidence::Unsettled { proof: bad_nm },
            ],
        };
        let mut b = block(shard, TAU0 + 71, vec![], vec![rec], vec![]);
        seal(&st70, &mut b, &mut fix.world, &[]);
        assert!(matches!(
            apply_block(st70.clone(), &b, &fix.world, &fix.world),
            Err(ExecutorError::ReversionBadUnsettled { issue: 1 })
        ));

        // Double consumption in one block.
        let rec = mixed_reversion(&fix);
        let mut b = block(shard, TAU0 + 71, vec![], vec![rec.clone(), rec], vec![]);
        seal(&st70, &mut b, &mut fix.world, &[]);
        assert!(matches!(
            apply_block(st70.clone(), &b, &fix.world, &fix.world),
            Err(ExecutorError::RecordDoubleConsume { record: 1 })
        ));

        // Shell unavailable: the chain entry is missing.
        fix.world.chain.remove(&(1, LegKey::new(fix.txid, LegIndex::FIRST)));
        let rec = mixed_reversion(&fix);
        let mut b = block(shard, TAU0 + 71, vec![], vec![rec], vec![]);
        seal(&st70, &mut b, &mut fix.world, &[]);
        assert!(matches!(
            apply_block(st70.clone(), &b, &fix.world, &fix.world),
            Err(ExecutorError::RecordShellUnavailable { record: 0 })
        ));

        // Shell mismatch: a lying chain serves a foreign shell.
        let mut rng = SplitMix64::new(0x0E5E);
        let foreign_shell = TransactionShell {
            legs: vec![leg(&mut rng, shard, vec![h(&mut rng)], vec![(1, false)], 0, 5_000, empty_anchor())],
        };
        let canon = foreign_shell.canonicalize().unwrap();
        let ftxid = canonical_txid(&canon);
        let foreign = ResolvedLeg {
            txid: ftxid,
            key: LegKey::new(ftxid, LegIndex::FIRST),
            canon,
            leg: 0,
        };
        fix.world.chain.insert((1, LegKey::new(fix.txid, LegIndex::FIRST)), foreign);
        let rec = mixed_reversion(&fix);
        let mut b = block(shard, TAU0 + 71, vec![], vec![rec], vec![]);
        seal(&st70, &mut b, &mut fix.world, &[]);
        assert!(matches!(
            apply_block(st70.clone(), &b, &fix.world, &fix.world),
            Err(ExecutorError::RecordShellMismatch { record: 0 })
        ));

        // Not pending after resolution: a reversion, then another.
        let set2 = ShardSet::genesis();
        let mut w2 = TestWorld::default();
        let mut fix2 = reversion_fixture(0x0E5F);
        let st70b = {
            let mut w = std::mem::take(&mut fix2.world);
            let s = advance(fix2.st.clone(), &mut w, 1);
            fix2.world = w;
            s
        };
        let rec = mixed_reversion(&fix2);
        let mut b71 = block(st70b.shard(), TAU0 + 71, vec![], vec![rec], vec![]);
        {
            let mut w = std::mem::take(&mut fix2.world);
            seal(&st70b, &mut b71, &mut w, &[vec![fix2.cm2]]);
            fix2.world = w;
        }
        let (st71, _) = apply_block(st70b, &b71, &fix2.world, &fix2.world).unwrap();
        let rec = mixed_reversion(&fix2);
        let mut b72 = block(st71.shard(), TAU0 + 72, vec![], vec![rec], vec![]);
        seal(&st71, &mut b72, &mut fix2.world, &[]);
        assert!(matches!(
            apply_block(st71, &b72, &fix2.world, &fix2.world),
            Err(ExecutorError::RecordEntryNotPending { state: "Reverted", .. })
        ));
        let _ = (set2, w2);

        // Born-this-block target: a fresh conditional spend plus a claim
        // on its own escrow in the same block.
        let mut fix3 = reversion_fixture(0x0E60);
        let st70c = {
            let mut w = std::mem::take(&mut fix3.world);
            let s = advance(fix3.st.clone(), &mut w, 1);
            fix3.world = w;
            s
        };
        let mut rng = SplitMix64::new(0x0E61);
        let shell2 = TransactionShell {
            legs: vec![
                leg(&mut rng, shard, vec![h(&mut rng)], vec![], 1000, 5000, empty_anchor()),
                leg(&mut rng, set.ids()[40], vec![], vec![(1_000, true)], 100, 5000, h(&mut rng)),
            ],
        };
        let txid2 = shell2.txid().unwrap();
        let tree = register_tau(&mut fix3.world, TAU0 + 71, &[txid2]);
        let sl2 = SettledLeg {
            shell: shell2,
            leg: LegIndex::FIRST,
            tau: tree.witness(0).unwrap(),
            siblings: vec![],
        };
        let claim = ClaimRecord { txid: txid2, spend_leg: LegIndex::FIRST, evidence: vec![] };
        let mut b = block(shard, TAU0 + 71, vec![sl2], vec![], vec![claim]);
        seal(&st70c, &mut b, &mut fix3.world, &[]);
        assert!(matches!(
            apply_block(st70c, &b, &fix3.world, &fix3.world),
            Err(ExecutorError::RecordTargetBornThisBlock { record: 0 })
        ));
    }

    #[test]
    fn claim_lifecycle_and_issue_leg_settlement() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let s40 = set.ids()[40];
        let mut rng = SplitMix64::new(0xC1A1);
        let mut world = TestWorld::default();

        let spend = leg(&mut rng, shard, vec![h(&mut rng)], vec![], 1000, 5_000, empty_anchor());
        let issue = leg(&mut rng, s40, vec![], vec![(1_000_000_000, true)], 1000, 5_000, h(&mut rng));
        let shell = TransactionShell { legs: vec![spend, issue] };
        let txid = shell.txid().unwrap();

        let st = genesis(shard);
        let tree = register_tau(&mut world, TAU0 + 1, &[txid]);
        let sl = SettledLeg {
            shell: shell.clone(),
            leg: LegIndex::FIRST,
            tau: tree.witness(0).unwrap(),
            siblings: vec![],
        };
        let mut b1 = block(shard, TAU0 + 1, vec![sl], vec![], vec![]);
        seal(&st, &mut b1, &mut world, &[]);
        let (st1, _) = apply_block(st, &b1, &world, &world).unwrap();
        let key7 = transit_key(&txid, shard, LegIndex::FIRST);
        assert_eq!(st1.transit().get(&key7).unwrap().state, TransitEntryState::Pending);
        world.transit_roots.insert((shard, 1), st1.transit().root());

        // The issue leg settles in shard 40 (rule 5 evidence).
        let st40 = genesis(s40);
        let spend_entry = st1.transit().get(&key7).unwrap().clone();
        let ev = TransitEvidence {
            root_height: Height::from_u64(1),
            proof: st1.transit().membership_proof(&spend_entry),
        };
        let sl40 = SettledLeg {
            shell: shell.clone(),
            leg: LegIndex::from_u8(1),
            tau: tree.witness(0).unwrap(),
            siblings: vec![ev],
        };
        let mut b40 = block(s40, TAU0 + 1, vec![sl40], vec![], vec![]);
        seal(&st40, &mut b40, &mut world, &[]);
        let (st40_1, app40) = apply_block(st40, &b40, &world, &world).unwrap();
        let key40 = transit_key(&txid, s40, LegIndex::from_u8(1));
        assert_eq!(st40_1.transit().get(&key40).unwrap().state, TransitEntryState::Claimed);
        assert_eq!(st40_1.nct().leaf_count(), 1);
        assert_eq!(app40.settled, vec![txid]);
        world.transit_roots.insert((s40, 1), st40_1.transit().root());

        // Replay of the issue leg: rule 4 (the nullifiers are absent).
        let mut b40b = block(s40, TAU0 + 2, vec![sl40], vec![], vec![]);
        seal(&st40_1, &mut b40b, &mut world, &[]);
        assert!(matches!(
            apply_block(st40_1.clone(), &b40b, &world, &world),
            Err(ExecutorError::TransitSpent { leg: 0 })
        ));

        // The claim (before due — optional hygiene, D.3(e)).
        let issue_entry = st40_1.transit().get(&key40).unwrap().clone();
        let claim = ClaimRecord {
            txid,
            spend_leg: LegIndex::FIRST,
            evidence: vec![TransitEvidence {
                root_height: Height::from_u64(1),
                proof: st40_1.transit().membership_proof(&issue_entry),
            }],
        };
        let mut b2 = block(shard, TAU0 + 2, vec![], vec![], vec![claim]);
        seal(&st1, &mut b2, &mut world, &[]);
        let (st2, app2) = apply_block(st1, &b2, &world, &world).unwrap();
        assert_eq!(st2.transit().get(&key7).unwrap().state, TransitEntryState::Claimed);
        assert_eq!(app2.escrows_claimed, vec![txid]);
        assert_eq!(st2.nct().leaf_count(), 0);
        assert!(app2.settled.is_empty());

        // A second claim: not pending.
        let mut b3 = block(shard, TAU0 + 3, vec![], vec![], vec![claim.clone()]);
        seal(&st2, &mut b3, &mut world, &[]);
        assert!(matches!(
            apply_block(st2.clone(), &b3, &world, &world),
            Err(ExecutorError::RecordEntryNotPending { state: "Claimed", .. })
        ));

        // Claim evidence tamper (on the pre-claim state).
        let mut bad_ev = TransitEvidence {
            root_height: Height::from_u64(1),
            proof: st40_1.transit().membership_proof(&issue_entry),
        };
        if bad_ev.proof.siblings.is_empty() {
            bad_ev.proof.siblings.push((1u16, Hash256::from_bytes([0xCC; 32])));
        } else {
            bad_ev.proof.siblings[0].1 = Hash256::from_bytes([0xCC; 32]);
        }
        let bad_claim = ClaimRecord {
            txid,
            spend_leg: LegIndex::FIRST,
            evidence: vec![bad_ev],
        };
        let mut b2b = block(shard, TAU0 + 2, vec![], vec![], vec![bad_claim]);
        seal(&st1.clone(), &mut b2b, &mut world, &[]);
        assert!(matches!(
            apply_block(st1.clone(), &b2b, &world, &world),
            Err(ExecutorError::ClaimEvidence { issue: 0, .. })
        ));

        // Claim evidence count.
        let empty_claim = ClaimRecord { txid, spend_leg: LegIndex::FIRST, evidence: vec![] };
        let mut b2c = block(shard, TAU0 + 2, vec![], vec![], vec![empty_claim]);
        seal(&st1, &mut b2c, &mut world, &[]);
        assert!(matches!(
            apply_block(st1, &b2c, &world, &world),
            Err(ExecutorError::ClaimEvidenceCount { expected: 1, found: 0 })
        ));
    }

    #[test]
    fn nonconditional_cross_shard_has_no_escrow() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let s40 = set.ids()[40];
        let mut rng = SplitMix64::new(0xC1A2);
        let mut world = TestWorld::default();

        let spend = leg(&mut rng, shard, vec![h(&mut rng)], vec![], 1000, 5_000, empty_anchor());
        let issue = leg(&mut rng, s40, vec![], vec![(1_000_000_000, false)], 1000, 5_000, h(&mut rng));
        let shell = TransactionShell { legs: vec![spend, issue] };
        let txid = shell.txid().unwrap();

        let st = genesis(shard);
        let tree = register_tau(&mut world, TAU0 + 1, &[txid]);
        let sl = SettledLeg {
            shell: shell.clone(),
            leg: LegIndex::FIRST,
            tau: tree.witness(0).unwrap(),
            siblings: vec![],
        };
        let mut b1 = block(shard, TAU0 + 1, vec![sl], vec![], vec![]);
        seal(&st, &mut b1, &mut world, &[]);
        let (st1, app1) = apply_block(st, &b1, &world, &world).unwrap();
        let key7 = transit_key(&txid, shard, LegIndex::FIRST);
        assert_eq!(st1.transit().get(&key7).unwrap().state, TransitEntryState::Claimed);
        assert!(app1.escrows_opened.is_empty());
        world.transit_roots.insert((shard, 1), st1.transit().root());

        // The issue leg settles; no claim record is ever required.
        let st40 = genesis(s40);
        let spend_entry = st1.transit().get(&key7).unwrap().clone();
        let ev = TransitEvidence {
            root_height: Height::from_u64(1),
            proof: st1.transit().membership_proof(&spend_entry),
        };
        let sl40 = SettledLeg {
            shell,
            leg: LegIndex::from_u8(1),
            tau: tree.witness(0).unwrap(),
            siblings: vec![ev],
        };
        let mut b40 = block(s40, TAU0 + 1, vec![sl40], vec![], vec![]);
        seal(&st40, &mut b40, &mut world, &[]);
        let (_, _) = apply_block(st40, &b40, &world, &world).unwrap();
    }

    #[test]
    fn multi_leg_block_and_replay_rejection() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let mut rng = SplitMix64::new(0x41BA);
        let mut world = TestWorld::default();
        let st = genesis(shard);

        let mut shells = Vec::new();
        for _ in 0..8 {
            let l = leg(&mut rng, shard, vec![h(&mut rng)], vec![(1_000_000_000, false), (2_000_000_000, false)], 1000, 5_000, empty_anchor());
            shells.push(TransactionShell { legs: vec![l] });
        }
        let mut txids: Vec<TxId> = shells.iter().map(|s| s.txid().unwrap()).collect();
        txids.sort();
        let tree = register_tau(&mut world, TAU0 + 1, &txids);
        let mut legs = Vec::new();
        for s in &shells {
            let t = s.txid().unwrap();
            legs.push(SettledLeg {
                shell: s.clone(),
                leg: LegIndex::FIRST,
                tau: tree.witness(tree.position(&t).unwrap()).unwrap(),
                siblings: vec![],
            });
        }
        legs.sort_by_key(|sl| sl.shell.txid().unwrap());
        let mut b = block(shard, TAU0 + 1, legs, vec![], vec![]);
        seal(&st, &mut b, &mut world, &[]);
        let (st1, app) = apply_block(st, &b, &world, &world).unwrap();
        assert_eq!(app.settled.len(), 8);
        assert_eq!(st1.nct().leaf_count(), 16);
        assert_eq!(app.settled, txids);

        // Replaying any leg of the settled set is rejected.
        let mut b2 = block(shard, TAU0 + 2, vec![b.legs[3].clone()], vec![], vec![]);
        seal(&st1, &mut b2, &mut world, &[]);
        assert!(matches!(
            apply_block(st1, &b2, &world, &world),
            Err(ExecutorError::NullifierSpent { .. }) | Err(ExecutorError::TransitSpent { .. })
        ));
    }

    // ---- Gap 3: D.4 admission floor ----

    /// Helper: build a valid one-leg block whose fee is below a target
    /// floor. Used to drive the rule-1.5 reject path.
    fn subfloor_block(
        st: &ShardState,
        world: &mut World,
        fee: u64,
    ) -> ShardBlock {
        let mut rng = SplitMix64::new(0xFEE1_5EED);
        let shard = st.shard();
        let shell = TransactionShell {
            legs: vec![leg(
                &mut rng,
                shard,
                vec![h(&mut rng)],
                vec![(1_000_000_000, false), (2_000_000_000, false)],
                fee,
                5_000,
                empty_anchor(),
            )],
        };
        let txid = shell.txid().unwrap();
        let interval = TAU0 + st.height().as_u64() + 1;
        let tree = register_tau(world, interval, &[txid]);
        let sl = SettledLeg {
            shell,
            leg: LegIndex::FIRST,
            tau: tree.witness(tree.position(&txid).unwrap()).unwrap(),
            siblings: vec![],
        };
        let mut b = block(shard, interval, vec![sl]);
        seal(st, &mut b, world);
        b
    }

    #[test]
    fn genesis_floor_is_one_nano_block() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let st = genesis(shard);

        // The genesis AdmissionFloor is m = 1 (BASE_FLOOR_NANO = 1000).
        assert_eq!(st.fee_floor().multiplier(), 1);
        assert_eq!(st.fee_floor().floor_nano(), 1000);
    }

    #[test]
    fn fee_below_floor_rejected_with_diag() {
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let st = genesis(shard);
        let mut world = World::default();

        // Fee 999 < floor 1000 → rule 1.5 fires; the diagnostic carries
        // the leg index and the two values.
        let b = subfloor_block(&st, &mut world, 999);
        match apply_block(st.clone(), &b, &world, &world) {
            Err(ExecutorError::FeeBelowFloor { leg, fee_nano, floor_nano }) => {
                assert_eq!(leg, 0);
                assert_eq!(fee_nano, 999);
                assert_eq!(floor_nano, 1000);
            }
            other => panic!("expected FeeBelowFloor, got {other:?}"),
        }
    }

    #[test]
    fn fee_at_floor_admitted() {
        // Boundary: leg fee == floor (1000) is admitted; fees below are
        // not. The genesis floor is exactly BASE_FLOOR_NANO.
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let st = genesis(shard);
        let mut world = World::default();

        // 1000 → admitted.
        let b_ok = subfloor_block(&st, &mut world, 1000);
        assert!(apply_block(st.clone(), &b_ok, &world, &world).is_ok());
    }

    #[test]
    fn fee_below_floor_short_circuits_before_anchor_check() {
        // A sub-floor leg paired with a *stale* anchor — only the floor
        // error should come back, not `AnchorStale`. This pins the
        // ordering rule 1 → 1.5 → 2.
        let set = ShardSet::genesis();
        let shard = set.ids()[7];
        let mut world = World::default();
        let st = genesis(shard);

        let mut rng = SplitMix64::new(0xAABB);
        let stale_anchor = Hash256::from_bytes(rng.bytes32()); // not in the ring
        let shell = TransactionShell {
            legs: vec![leg(
                &mut rng,
                shard,
                vec![h(&mut rng)],
                vec![(1_000_000_000, false)],
                500, // below the floor
                5_000,
                stale_anchor,
            )],
        };
        let txid = shell.txid().unwrap();
        let interval = TAU0 + st.height().as_u64() + 1;
        let tree = register_tau(&mut world, interval, &[txid]);
        let sl = SettledLeg {
            shell,
            leg: LegIndex::FIRST,
            tau: tree.witness(tree.position(&txid).unwrap()).unwrap(),
            siblings: vec![],
        };
        let mut b = block(shard, interval, vec![sl]);
        seal(&st, &mut b, &mut world);

        match apply_block(st, &b, &world, &world) {
            Err(ExecutorError::FeeBelowFloor { fee_nano: 500, floor_nano: 1000, .. }) => {}
            other => panic!("expected FeeBelowFloor (not AnchorStale), got {other:?}"),
        }
    }
}

