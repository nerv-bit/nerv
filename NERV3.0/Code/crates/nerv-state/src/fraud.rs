//! Fraud evidence (WP §4.3, §5.5, §11.4; errata 115–116): public-hash
//! proofs that a block is settlement-invalid.
//!
//! * The minimal class — block + declared beacon facts; verifiable by
//!   anyone holding the beacon. Facts are confirmed against the view,
//!   then the condition is evaluated inside the claims' closure.
//! * The reexecution class — the block alone; verified by a holder of
//!   the predecessor state (the full node, §4.3's native path) running
//!   Update. Any ExecutorError is the proven condition.

use std::collections::HashSet;

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::{CT_BATCH, FRAUD};
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::types::{Height, Interval, LegIndex, ShardId};

use crate::block::{ct_sum, ResolvedLeg, ShardBlock};
use crate::error::{ExecutorError, FraudError};
use crate::executor::{
    apply_block, shell_law_error, transit_evidence_ok, BeaconView, ChainSource, ShardState,
};
use crate::ttau::verify_tau_witness;

// ---------------------------------------------------------------------------
// Beacon facts
// ---------------------------------------------------------------------------

/// The beacon reads a proof depends on, as claims the verifier confirms
/// before any check. `None` claims absence (a root the beacon does not
/// finalize); the claims are the proof's complete read set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BeaconFacts {
    pub tau_interval: u64,
    pub tau_root: Option<Hash256>,
    pub transit_roots: Vec<(ShardId, u64, Option<Hash256>)>,
}

impl BeaconFacts {
    /// The facts every minimal condition over `block` may read: the
    /// header's registry root and each issue leg's sibling-evidence
    /// (shard, root height) pairs. Deterministic in the block.
    pub fn for_block(block: &ShardBlock, view: &dyn BeaconView) -> BeaconFacts {
        let mut facts = BeaconFacts {
            tau_interval: block.header.registry.interval.as_u64(),
            tau_root: view.tau_root(block.header.registry.interval),
            transit_roots: Vec::new(),
        };
        if let Ok(resolved) = block.resolve_legs() {
            for (sl, r) in block.legs.iter().zip(resolved.iter()) {
                if !r.leg_shell().inputs.nullifiers.is_empty() {
                    continue;
                }
                let spends: Vec<ShardId> = r
                    .canon
                    .legs
                    .iter()
                    .filter(|l| !l.inputs.nullifiers.is_empty())
                    .map(|l| l.shard)
                    .collect();
                for (j, ev) in sl.siblings.iter().enumerate() {
                    if let Some(&shard) = spends.get(j) {
                        facts.transit_roots.push((
                            shard,
                            ev.root_height.as_u64(),
                            view.transit_root(shard, ev.root_height),
                        ));
                    }
                }
            }
        }
        facts
    }

    /// Every claimed fact must match the view, absence included.
    pub fn confirm(&self, view: &dyn BeaconView) -> bool {
        if self.tau_root != view.tau_root(Interval::from_u64(self.tau_interval)) {
            return false;
        }
        self.transit_roots.iter().all(|(shard, height, root)| {
            *root == view.transit_root(*shard, Height::from_u64(*height))
        })
    }
}

impl Encode for BeaconFacts {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.tau_interval.to_le_bytes());
        match &self.tau_root {
            None => out.push(0),
            Some(r) => {
                out.push(1);
                out.extend_from_slice(r.as_bytes());
            }
        }
        out.extend_from_slice(&(self.transit_roots.len() as u32).to_le_bytes());
        for (shard, height, root) in &self.transit_roots {
            shard.encode_into(out);
            out.extend_from_slice(&height.to_le_bytes());
            match root {
                None => out.push(0),
                Some(r) => {
                    out.push(1);
                    out.extend_from_slice(r.as_bytes());
                }
            }
        }
    }
    fn encoded_len(&self) -> usize {
        9 + if self.tau_root.is_some() { 32 } else { 0 }
            + 4
            + self
                .transit_roots
                .iter()
                .map(|(s, _, r)| s.encoded_len() + 9 + if r.is_some() { 32 } else { 0 })
                .sum::<usize>()
    }
}

impl Decode for BeaconFacts {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let tau_interval = r.read_u64()?;
        let tau_root = match r.read_u8()? {
            0 => None,
            1 => Some(Hash256::decode_from(r)?),
            tag => return Err(CodecError::InvalidOptionTag { tag }),
        };
        let n = r.read_seq_len()?;
        if n > 65_536 {
            return Err(CodecError::SeqTooLarge { count: n, max: 65_536 });
        }
        let mut transit_roots = Vec::with_capacity(n);
        for _ in 0..n {
            let shard = ShardId::decode_from(r)?;
            let height = r.read_u64()?;
            let root = match r.read_u8()? {
                0 => None,
                1 => Some(Hash256::decode_from(r)?),
                tag => return Err(CodecError::InvalidOptionTag { tag }),
            };
            transit_roots.push((shard, height, root));
        }
        Ok(BeaconFacts { tau_interval, tau_root, transit_roots })
    }
}

/// The proof's read closure: serves only claimed facts; an unclaimed read
/// fails (the condition can never "guess" from outside its claims).
struct FactsView<'a> {
    facts: &'a BeaconFacts,
}

impl BeaconView for FactsView<'_> {
    fn tau_root(&self, interval: Interval) -> Option<Hash256> {
        if interval.as_u64() != self.facts.tau_interval {
            return None;
        }
        self.facts.tau_root
    }
    fn transit_root(&self, shard: ShardId, height: Height) -> Option<Hash256> {
        self.facts
            .transit_roots
            .iter()
            .find(|(s, hh, _)| *s == shard && *hh == height.as_u64())
            .and_then(|(_, _, r)| *r)
    }
}

// ---------------------------------------------------------------------------
// The minimal proof
// ---------------------------------------------------------------------------

/// A settlement-invalidity condition verifiable from the block plus the
/// declared beacon facts alone (erratum 116).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FraudCondition {
    /// The block's legs do not resolve (canonicality, order, cap, shard,
    /// index, or an unparseable ct).
    Unresolvable,
    ConditionalOnSpendLeg { leg: usize },
    NotSingleSpend { leg: usize },
    LegExpiryMismatch { leg: usize },
    ExpiryOutOfBounds { leg: usize },
    /// Rule 1: the leg's T_τ witness fails against the finalized root.
    TauWitness { leg: usize },
    /// The header's fee total disagrees with the legs' declared fees.
    FeeTotalMismatch,
    /// The header's H(ct_B) disagrees, or a leg's ct does not parse.
    CtBatchMismatch,
    /// The first in-block nullifier conflict occurs at `leg`.
    InBlockNullifier { leg: usize },
    /// An issue leg settling past its expiry (the D.3 deadline).
    SettlementDeadline { leg: usize },
    SiblingCount { leg: usize },
    SiblingEvidence { leg: usize, sibling: usize },
}

impl Encode for FraudCondition {
    fn encode_into(&self, out: &mut Vec<u8>) {
        let leg = |out: &mut Vec<u8>, l: usize| out.extend_from_slice(&(l as u32).to_le_bytes());
        match *self {
            FraudCondition::Unresolvable => out.push(0),
            FraudCondition::ConditionalOnSpendLeg { leg: l } => {
                out.push(1);
                leg(out, l);
            }
            FraudCondition::NotSingleSpend { leg: l } => {
                out.push(2);
                leg(out, l);
            }
            FraudCondition::LegExpiryMismatch { leg: l } => {
                out.push(3);
                leg(out, l);
            }
            FraudCondition::ExpiryOutOfBounds { leg: l } => {
                out.push(4);
                leg(out, l);
            }
            FraudCondition::TauWitness { leg: l } => {
                out.push(5);
                leg(out, l);
            }
            FraudCondition::FeeTotalMismatch => out.push(6),
            FraudCondition::CtBatchMismatch => out.push(7),
            FraudCondition::InBlockNullifier { leg: l } => {
                out.push(8);
                leg(out, l);
            }
            FraudCondition::SettlementDeadline { leg: l } => {
                out.push(9);
                leg(out, l);
            }
            FraudCondition::SiblingCount { leg: l } => {
                out.push(10);
                leg(out, l);
            }
            FraudCondition::SiblingEvidence { leg: l, sibling } => {
                out.push(11);
                leg(out, l);
                out.extend_from_slice(&(sibling as u32).to_le_bytes());
            }
        }
    }
    fn encoded_len(&self) -> usize {
        match self {
            FraudCondition::SiblingEvidence { .. } => 9,
            FraudCondition::Unresolvable
            | FraudCondition::FeeTotalMismatch
            | FraudCondition::CtBatchMismatch => 1,
            _ => 5,
        }
    }
}

impl Decode for FraudCondition {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let leg = |r: &mut Reader<'_>| r.read_u32().map(|v| v as usize);
        Ok(match r.read_u8()? {
            0 => FraudCondition::Unresolvable,
            1 => FraudCondition::ConditionalOnSpendLeg { leg: leg(r)? },
            2 => FraudCondition::NotSingleSpend { leg: leg(r)? },
            3 => FraudCondition::LegExpiryMismatch { leg: leg(r)? },
            4 => FraudCondition::ExpiryOutOfBounds { leg: leg(r)? },
            5 => FraudCondition::TauWitness { leg: leg(r)? },
            6 => FraudCondition::FeeTotalMismatch,
            7 => FraudCondition::CtBatchMismatch,
            8 => FraudCondition::InBlockNullifier { leg: leg(r)? },
            9 => FraudCondition::SettlementDeadline { leg: leg(r)? },
            10 => FraudCondition::SiblingCount { leg: leg(r)? },
            11 => {
                let l = leg(r)?;
                let sibling = r.read_u32()? as usize;
                FraudCondition::SiblingEvidence { leg: l, sibling }
            }
            tag => return Err(CodecError::InvalidOptionTag { tag }),
        })
    }
}

fn first_nullifier_conflict(resolved: &[ResolvedLeg]) -> Option<usize> {
    let mut seen = HashSet::new();
    for (i, r) in resolved.iter().enumerate() {
        for nf in &r.leg_shell().inputs.nullifiers {
            if !seen.insert(*nf.as_bytes()) {
                return Some(i);
            }
        }
    }
    None
}

/// The minimal fraud proof: the block (DA data), the beacon facts it
/// depends on, and the exhibited condition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FraudProof {
    pub block: ShardBlock,
    pub facts: BeaconFacts,
    pub condition: FraudCondition,
}

impl FraudProof {
    /// H("nerv.fraud" ‖ 0 ‖ canonical encoding) — the slashable-evidence
    /// binding (the slashing machinery is chunks 14/17's).
    pub fn digest(&self) -> Hash256 {
        let mut buf = Vec::with_capacity(1 + self.encoded_len());
        buf.push(0);
        self.encode_into(&mut buf);
        Hash256::concat(&FRAUD, &buf)
    }

    pub fn verify(&self, view: &dyn BeaconView) -> Result<(), FraudError> {
        if !self.facts.confirm(view) {
            return Err(FraudError::FactsRejected);
        }
        let fv = FactsView { facts: &self.facts };
        if self.condition_holds(&fv)? {
            Ok(())
        } else {
            Err(FraudError::ConditionNotExhibited)
        }
    }

    fn condition_holds(&self, view: &dyn BeaconView) -> Result<bool, FraudError> {
        let resolved = match self.block.resolve_legs() {
            Ok(r) => r,
            Err(_) => return Ok(matches!(self.condition, FraudCondition::Unresolvable)),
        };
        let leg_of = |i: usize| -> Result<&ResolvedLeg, FraudError> {
            resolved
                .get(i)
                .ok_or(FraudError::Malformed("leg index outside the block"))
        };
        let height = self.block.header.height;
        let shell_law = |leg: usize| {
            let r = leg_of(leg)?;
            Ok::<_, FraudError>(shell_law_error(
                leg,
                &r.canon,
                r.leg_shell(),
                height.as_u64(),
            ))
        };
        match self.condition {
            FraudCondition::Unresolvable => Ok(false),
            FraudCondition::ConditionalOnSpendLeg { leg } => Ok(matches!(
                shell_law(leg)?,
                Some(ExecutorError::ShellConditionalOnSpendLeg { .. })
            )),
            FraudCondition::NotSingleSpend { leg } => Ok(matches!(
                shell_law(leg)?,
                Some(ExecutorError::ShellNotSingleSpend { .. })
            )),
            FraudCondition::LegExpiryMismatch { leg } => Ok(matches!(
                shell_law(leg)?,
                Some(ExecutorError::ShellExpiryMismatch { .. })
            )),
            FraudCondition::ExpiryOutOfBounds { leg } => Ok(matches!(
                shell_law(leg)?,
                Some(ExecutorError::ExpiryBounds { .. })
            )),
            FraudCondition::TauWitness { leg } => {
                if self.facts.tau_interval != self.block.header.registry.interval.as_u64() {
                    return Err(FraudError::Malformed(
                        "facts interval does not match the header's registry",
                    ));
                }
                let Some(root) = self.facts.tau_root else {
                    return Err(FraudError::Malformed("facts lack the registry root"));
                };
                let r = leg_of(leg)?;
                let sl = self
                    .block
                    .legs
                    .get(leg)
                    .ok_or(FraudError::Malformed("leg index outside the block"))?;
                Ok(!verify_tau_witness(&root, sl.tau.index, &r.txid, &sl.tau.siblings))
            }
            FraudCondition::FeeTotalMismatch => {
                let mut fee = 0u64;
                for r in &resolved {
                    match fee.checked_add(r.leg_shell().fee.as_u64()) {
                        Some(f) => fee = f,
                        None => return Ok(true),
                    }
                }
                Ok(fee != self.block.header.fee_total.as_u64())
            }
            FraudCondition::CtBatchMismatch => match ct_sum(&resolved) {
                Err(_) => Ok(true),
                Ok(ct) => {
                    Ok(Hash256::concat(&CT_BATCH, &ct.to_bytes())
                        != self.block.header.ct_batch_hash)
                }
            },
            FraudCondition::InBlockNullifier { leg } => {
                Ok(first_nullifier_conflict(&resolved) == Some(leg))
            }
            FraudCondition::SettlementDeadline { leg } => {
                let r = leg_of(leg)?;
                let l = r.leg_shell();
                Ok(l.inputs.nullifiers.is_empty() && l.expiry < height)
            }
            FraudCondition::SiblingCount { leg } => {
                let r = leg_of(leg)?;
                if !r.leg_shell().inputs.nullifiers.is_empty() {
                    return Ok(false);
                }
                let spends = r.canon.legs.iter().filter(|l| !l.inputs.nullifiers.is_empty()).count();
                Ok(self
                    .block
                    .legs
                    .get(leg)
                    .ok_or(FraudError::Malformed("leg index outside the block"))?
                    .siblings
                    .len()
                    != spends)
            }
            FraudCondition::SiblingEvidence { leg, sibling } => {
                let r = leg_of(leg)?;
                if !r.leg_shell().inputs.nullifiers.is_empty() {
                    return Ok(false);
                }
                let spends: Vec<(usize, ShardId)> = r
                    .canon
                    .legs
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| !l.inputs.nullifiers.is_empty())
                    .map(|(li, l)| (li, l.shard))
                    .collect();
                let (li, shard) = spends
                    .get(sibling)
                    .ok_or(FraudError::Malformed("sibling index outside the spend set"))?;
                let ev = self
                    .block
                    .legs
                    .get(leg)
                    .and_then(|sl| sl.siblings.get(sibling))
                    .ok_or(FraudError::Malformed("sibling evidence outside the carried set"))?;
                if view.transit_root(*shard, ev.root_height).is_none() {
                    return Err(FraudError::Malformed(
                        "facts lack the sibling's transit root",
                    ));
                }
                Ok(!transit_evidence_ok(
                    ev,
                    view,
                    &r.txid,
                    *shard,
                    LegIndex::from_u8(*li as u8),
                ))
            }
        }
    }
}

impl Encode for FraudProof {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.block.encode_into(out);
        self.facts.encode_into(out);
        self.condition.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        self.block.encoded_len() + self.facts.encoded_len() + self.condition.encoded_len()
    }
}

impl Decode for FraudProof {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(FraudProof {
            block: ShardBlock::decode_from(r)?,
            facts: BeaconFacts::decode_from(r)?,
            condition: FraudCondition::decode_from(r)?,
        })
    }
}

// ---------------------------------------------------------------------------
// The reexecution proof
// ---------------------------------------------------------------------------

/// Wrong-root and every other state-dependent condition (erratum 116b):
/// the block alone, verified by running Update against the predecessor —
/// the full node's native path. Any `ExecutorError` is the proven
/// condition; a valid block rejects (the false-claim slash is later
/// chunks' machinery).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReexecutionFraud {
    pub block: ShardBlock,
}

impl ReexecutionFraud {
    /// H("nerv.fraud" ‖ 1 ‖ canonical encoding).
    pub fn digest(&self) -> Hash256 {
        let mut buf = Vec::with_capacity(1 + self.block.encoded_len());
        buf.push(1);
        self.block.encode_into(&mut buf);
        Hash256::concat(&FRAUD, &buf)
    }

    pub fn verify(
        &self,
        predecessor: &ShardState,
        view: &dyn BeaconView,
        chain: &dyn ChainSource,
    ) -> Result<ExecutorError, FraudError> {
        match apply_block(predecessor.clone(), &self.block, view, chain) {
            Err(e) => Ok(e),
            Ok(_) => Err(FraudError::ConditionNotExhibited),
        }
    }
}

impl Encode for ReexecutionFraud {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.block.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        self.block.encoded_len()
    }
}

impl Decode for ReexecutionFraud {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(ReexecutionFraud { block: ShardBlock::decode_from(r)? })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::block::{SettledLeg, TransitEvidence};
    use crate::testutil::harness::{
        block, empty_anchor, h, leg, params, register_tau, seal, valid_block, World, TAU0,
    };
    use crate::testutil::SplitMix64;
    use nerv_core::types::{FeeSats, ShardSet};
    use nerv_custody::tx::{LegShell, TransactionShell};
    use nerv_custody::transit_key;

    fn genesis_state(shard: nerv_core::types::ShardId) -> ShardState {
        ShardState::genesis(shard, params())
    }

    fn settle_shell(
        st: &ShardState,
        world: &mut World,
        legs: Vec<LegShell>,
        leg_index: u8,
    ) -> ShardBlock {
        let shell = TransactionShell { legs };
        let txid = shell.txid().unwrap();
        let interval = TAU0 + st.height().as_u64() + 1;
        let tree = register_tau(world, interval, &[txid]);
        let sl = SettledLeg {
            shell,
            leg: LegIndex::from_u8(leg_index),
            tau: tree.witness(0).unwrap(),
            siblings: vec![],
        };
        let mut b = block(st.shard(), interval, vec![sl]);
        seal(st, &mut b, world);
        b
    }

    /// [spend@7, issue@40] with a shared expiry; the spend settled at
    /// shard-7 height 1 (its transit root registered); the issue block
    /// for shard 40 at height 1, returned un-applied.
    fn issue_fixture(seed: u64, expiry: u64) -> (World, ShardBlock, ShardState) {
        let set = ShardSet::genesis();
        let s7 = set.ids()[7];
        let s40 = set.ids()[40];
        let mut rng = SplitMix64::new(seed);
        let mut world = World::default();
        let spend = leg(&mut rng, s7, vec![h(&mut rng)], vec![], 1000, expiry, empty_anchor());
        let issue = leg(&mut rng, s40, vec![], vec![(1_000_000_000, false)], 1000, expiry, h(&mut rng));
        let shell = TransactionShell { legs: vec![spend, issue] };
        let txid = shell.txid().unwrap();
        let st7 = genesis_state(s7);
        let interval = TAU0 + 1;
        let tree = register_tau(&mut world, interval, &[txid]);
        let sl = SettledLeg {
            shell: shell.clone(),
            leg: LegIndex::FIRST,
            tau: tree.witness(0).unwrap(),
            siblings: vec![],
        };
        let mut b1 = block(s7, interval, vec![sl]);
        seal(&st7, &mut b1, &mut world);
        let (st7_1, _) = apply_block(st7, &b1, &world, &world).unwrap();
        world.transit_roots.insert((s7, 1), st7_1.transit().root());

        let key7 = transit_key(&txid, s7, LegIndex::FIRST);
        let entry = st7_1.transit().get(&key7).cloned().unwrap();
        let ev = TransitEvidence {
            root_height: Height::from_u64(1),
            proof: st7_1.transit().membership_proof(&entry),
        };
        let sl40 = SettledLeg {
            shell,
            leg: LegIndex::from_u8(1),
            tau: tree.witness(0).unwrap(),
            siblings: vec![ev],
        };
        let st40 = genesis_state(s40);
        let mut b2 = block(s40, interval, vec![sl40]);
        seal(&st40, &mut b2, &mut world);
        (world, b2, st40)
    }

    #[test]
    fn honest_block_rejects_every_condition() {
        let set = ShardSet::genesis();
        let st = genesis_state(set.ids()[7]);
        let mut world = World::default();
        let (b, _) = valid_block(&st, &mut world, 0xF001);
        let facts = BeaconFacts::for_block(&b, &world);
        let conditions = [
            FraudCondition::Unresolvable,
            FraudCondition::ConditionalOnSpendLeg { leg: 0 },
            FraudCondition::NotSingleSpend { leg: 0 },
            FraudCondition::LegExpiryMismatch { leg: 0 },
            FraudCondition::ExpiryOutOfBounds { leg: 0 },
            FraudCondition::TauWitness { leg: 0 },
            FraudCondition::FeeTotalMismatch,
            FraudCondition::CtBatchMismatch,
            FraudCondition::InBlockNullifier { leg: 0 },
            FraudCondition::SettlementDeadline { leg: 0 },
            FraudCondition::SiblingCount { leg: 0 },
            FraudCondition::SiblingEvidence { leg: 0, sibling: 0 },
        ];
        for condition in conditions {
            let p = FraudProof { block: b.clone(), facts: facts.clone(), condition };
            assert!(
                matches!(p.verify(&world), Err(FraudError::ConditionNotExhibited)),
                "{condition:?}"
            );
        }
        apply_block(st, &b, &world, &world).unwrap();
    }

    #[test]
    fn tau_witness_fraud() {
        let set = ShardSet::genesis();
        let st = genesis_state(set.ids()[7]);
        let mut world = World::default();
        let (b, _) = valid_block(&st, &mut world, 0xF002);
        let mut bad = b.clone();
        bad.legs[0].tau.siblings.push(Hash256::from_bytes([0xEE; 32]));
        let facts = BeaconFacts::for_block(&bad, &world);
        let p = FraudProof { block: bad.clone(), facts, condition: FraudCondition::TauWitness { leg: 0 } };
        p.verify(&world).unwrap();
        assert!(matches!(
            apply_block(st.clone(), &bad, &world, &world),
            Err(ExecutorError::TauWitness { leg: 0 })
        ));
        // A different tau interval in the facts is rejected before any check.
        let mut wrong = BeaconFacts::for_block(&b, &world);
        wrong.tau_interval += 1;
        let p = FraudProof { block: b.clone(), facts: wrong, condition: FraudCondition::TauWitness { leg: 0 } };
        assert!(matches!(p.verify(&world), Err(FraudError::FactsRejected)));
    }

    #[test]
    fn fee_and_ct_batch_fraud() {
        let set = ShardSet::genesis();
        let st = genesis_state(set.ids()[7]);
        let mut world = World::default();
        let (b, _) = valid_block(&st, &mut world, 0xF003);

        let mut bad = b.clone();
        bad.header.fee_total = FeeSats::from_u64(bad.header.fee_total.as_u64() + 1);
        let facts = BeaconFacts::for_block(&bad, &world);
        let p = FraudProof { block: bad.clone(), facts, condition: FraudCondition::FeeTotalMismatch };
        p.verify(&world).unwrap();
        assert!(matches!(
            apply_block(st.clone(), &bad, &world, &world),
            Err(ExecutorError::FeeTotalMismatch)
        ));

        let mut bad = b.clone();
        bad.header.ct_batch_hash = Hash256::from_bytes([0xEE; 32]);
        let facts = BeaconFacts::for_block(&bad, &world);
        let p = FraudProof { block: bad.clone(), facts, condition: FraudCondition::CtBatchMismatch };
        p.verify(&world).unwrap();
        assert!(matches!(
            apply_block(st.clone(), &bad, &world, &world),
            Err(ExecutorError::CtBatchHashMismatch)
        ));

        // An unparseable ct exhibits the same condition.
        let mut bad = b.clone();
        bad.legs[0].shell.legs[0].ct = vec![0xA5; 10];
        let facts = BeaconFacts::for_block(&bad, &world);
        let p = FraudProof { block: bad, facts, condition: FraudCondition::CtBatchMismatch };
        p.verify(&world).unwrap();
    }

    #[test]
    fn in_block_nullifier_fraud() {
        let set = ShardSet::genesis();
        let s7 = set.ids()[7];
        let mut rng = SplitMix64::new(0xF004);
        let mut world = World::default();
        let shared = h(&mut rng);
        let mk = |rng: &mut SplitMix64| {
            TransactionShell {
                legs: vec![leg(rng, s7, vec![shared], vec![(1, false)], 1000, 5_000, empty_anchor())],
            }
        };
        let (sa, sb) = (mk(&mut rng), mk(&mut rng));
        let (ta, tb) = (sa.txid().unwrap(), sb.txid().unwrap());
        let interval = TAU0 + 1;
        let tree = register_tau(&mut world, interval, &[ta, tb]);
        let make = |shell: TransactionShell| SettledLeg {
            shell,
            leg: LegIndex::FIRST,
            tau: tree.witness(0).unwrap(),
            siblings: vec![],
        };
        let (fa, fb) = (make(sa), make(sb));
        let mut legs = if ta < tb { vec![fa, fb] } else { vec![fb, fa] };
        let second = 1usize;
        let second_txid = legs[second].shell.txid().unwrap();
        for l in &mut legs {
            let t = l.shell.txid().unwrap();
            l.tau = tree.witness(tree.position(&t).unwrap()).unwrap();
        }
        let _ = second_txid;
        let st = genesis_state(s7);
        let mut b = block(s7, interval, legs);
        seal(&st, &mut b, &mut world);

        let facts = BeaconFacts::for_block(&b, &world);
        let p = FraudProof {
            block: b.clone(),
            facts: facts.clone(),
            condition: FraudCondition::InBlockNullifier { leg: second },
        };
        p.verify(&world).unwrap();
        assert!(matches!(
            apply_block(st, &b, &world, &world),
            Err(ExecutorError::NullifierSpent { leg: 1, .. })
        ));
        let p = FraudProof { block: b, facts, condition: FraudCondition::InBlockNullifier { leg: 0 } };
        assert!(matches!(p.verify(&world), Err(FraudError::ConditionNotExhibited)));
    }

    #[test]
    fn settlement_deadline_fraud() {
        let (world, b, st40) = issue_fixture(0xF005, 0);
        let facts = BeaconFacts::for_block(&b, &world);
        let p = FraudProof {
            block: b.clone(),
            facts,
            condition: FraudCondition::SettlementDeadline { leg: 0 },
        };
        p.verify(&world).unwrap();
        assert!(matches!(
            apply_block(st40, &b, &world, &world),
            Err(ExecutorError::SettlementDeadline { leg: 0, expiry: 0, height: 1 })
        ));
    }

    #[test]
    fn sibling_count_and_evidence_fraud() {
        let (world, b, st40) = issue_fixture(0xF006, 5_000);
        apply_block(st40.clone(), &b, &world, &world).unwrap(); // control: valid

        let mut bad = b.clone();
        bad.legs[0].siblings.clear();
        let facts = BeaconFacts::for_block(&bad, &world);
        let p = FraudProof { block: bad.clone(), facts, condition: FraudCondition::SiblingCount { leg: 0 } };
        p.verify(&world).unwrap();
        assert!(matches!(
            apply_block(st40.clone(), &bad, &world, &world),
            Err(ExecutorError::SiblingCount { leg: 0, .. })
        ));

        let mut bad = b.clone();
        {
            let sib = &mut bad.legs[0].siblings[0];
            if sib.proof.siblings.is_empty() {
                sib.proof.siblings.push((1u16, Hash256::from_bytes([0xEE; 32])));
            } else {
                sib.proof.siblings[0].1 = Hash256::from_bytes([0xEE; 32]);
            }
        }
        let facts = BeaconFacts::for_block(&bad, &world);
        let p = FraudProof {
            block: bad.clone(),
            facts: facts.clone(),
            condition: FraudCondition::SiblingEvidence { leg: 0, sibling: 0 },
        };
        p.verify(&world).unwrap();
        assert!(matches!(
            apply_block(st40.clone(), &bad, &world, &world),
            Err(ExecutorError::SiblingEvidence { leg: 0, sibling: 0 })
        ));
        // Out-of-range sibling indices are malformed claims.
        let p = FraudProof {
            block: bad.clone(),
            facts,
            condition: FraudCondition::SiblingEvidence { leg: 0, sibling: 9 },
        };
        assert!(matches!(p.verify(&world), Err(FraudError::Malformed(_))));
    }

    #[test]
    fn facts_closure_is_exact() {
        // A proof whose facts omit the sibling root cannot exhibit
        // SiblingEvidence — the closure never guesses from the live view.
        let (world, b, _) = issue_fixture(0xF007, 5_000);
        let mut bad = b.clone();
        {
            let sib = &mut bad.legs[0].siblings[0];
            if sib.proof.siblings.is_empty() {
                sib.proof.siblings.push((1u16, Hash256::from_bytes([0xEE; 32])));
            } else {
                sib.proof.siblings[0].1 = Hash256::from_bytes([0xEE; 32]);
            }
        }
        let mut facts = BeaconFacts::for_block(&bad, &world);
        facts.transit_roots.clear(); // claims nothing
        let p = FraudProof {
            block: bad,
            facts,
            condition: FraudCondition::SiblingEvidence { leg: 0, sibling: 0 },
        };
        assert!(matches!(p.verify(&world), Err(FraudError::Malformed(_))));
    }

    #[test]
    fn shell_law_frauds() {
        let set = ShardSet::genesis();
        let s7 = set.ids()[7];
        let s40 = set.ids()[40];
        let s9 = set.ids()[9];

        // Conditional output on an input-bearing leg.
        let mut world = World::default();
        let st = genesis_state(s7);
        let mut rng = SplitMix64::new(0xF008);
        let l = leg(&mut rng, s7, vec![h(&mut rng)], vec![(1_000, true)], 1000, 5_000, empty_anchor());
        let b = settle_shell(&st, &mut world, vec![l], 0);
        let facts = BeaconFacts::for_block(&b, &world);
        FraudProof { block: b.clone(), facts, condition: FraudCondition::ConditionalOnSpendLeg { leg: 0 } }
            .verify(&world)
            .unwrap();
        assert!(matches!(
            apply_block(st, &b, &world, &world),
            Err(ExecutorError::ShellConditionalOnSpendLeg { leg: 0 })
        ));

        // Two spend legs around a conditional issue leg.
        let mut world = World::default();
        let st = genesis_state(s7);
        let mut rng = SplitMix64::new(0xF009);
        let legs = vec![
            leg(&mut rng, s7, vec![h(&mut rng)], vec![], 1000, 5_000, empty_anchor()),
            leg(&mut rng, s40, vec![h(&mut rng)], vec![], 1000, 5_000, h(&mut rng)),
            leg(&mut rng, s9, vec![], vec![(1_000, true)], 1000, 5_000, h(&mut rng)),
        ];
        let b = settle_shell(&st, &mut world, legs, 0);
        let facts = BeaconFacts::for_block(&b, &world);
        FraudProof { block: b.clone(), facts, condition: FraudCondition::NotSingleSpend { leg: 0 } }
            .verify(&world)
            .unwrap();
        assert!(matches!(
            apply_block(st, &b, &world, &world),
            Err(ExecutorError::ShellNotSingleSpend { leg: 0, .. })
        ));

        // Unequal expiries across a conditional shell.
        let mut world = World::default();
        let st = genesis_state(s7);
        let mut rng = SplitMix64::new(0xF00A);
        let legs = vec![
            leg(&mut rng, s7, vec![h(&mut rng)], vec![], 1000, 61, empty_anchor()),
            leg(&mut rng, s40, vec![], vec![(1_000, true)], 1000, 62, h(&mut rng)),
        ];
        let b = settle_shell(&st, &mut world, legs, 0);
        let facts = BeaconFacts::for_block(&b, &world);
        FraudProof { block: b.clone(), facts, condition: FraudCondition::LegExpiryMismatch { leg: 0 } }
            .verify(&world)
            .unwrap();
        assert!(matches!(
            apply_block(st, &b, &world, &world),
            Err(ExecutorError::ShellExpiryMismatch { leg: 0 })
        ));

        // Expiry below T_min at the including height.
        let mut world = World::default();
        let st = genesis_state(s7);
        let mut rng = SplitMix64::new(0xF00B);
        let legs = vec![
            leg(&mut rng, s7, vec![h(&mut rng)], vec![], 1000, 60, empty_anchor()),
            leg(&mut rng, s40, vec![], vec![(1_000, true)], 1000, 60, h(&mut rng)),
        ];
        let b = settle_shell(&st, &mut world, legs, 0);
        let facts = BeaconFacts::for_block(&b, &world);
        FraudProof { block: b.clone(), facts, condition: FraudCondition::ExpiryOutOfBounds { leg: 0 } }
            .verify(&world)
            .unwrap();
        assert!(matches!(
            apply_block(st, &b, &world, &world),
            Err(ExecutorError::ExpiryBounds { leg: 0, expiry: 60, lo: 61, hi: 14_401 })
        ));
    }

    #[test]
    fn unresolvable_fraud() {
        let set = ShardSet::genesis();
        let s7 = set.ids()[7];
        let mut rng = SplitMix64::new(0xF00C);
        let mut world = World::default();
        let mk = |rng: &mut SplitMix64| {
            TransactionShell {
                legs: vec![leg(rng, s7, vec![h(rng)], vec![(1, false)], 1000, 5_000, empty_anchor())],
            }
        };
        let (sa, sb) = (mk(&mut rng), mk(&mut rng));
        let (ta, tb) = (sa.txid().unwrap(), sb.txid().unwrap());
        let interval = TAU0 + 1;
        let tree = register_tau(&mut world, interval, &[ta, tb]);
        let make = |shell: TransactionShell| SettledLeg {
            shell,
            leg: LegIndex::FIRST,
            tau: tree.witness(0).unwrap(),
            siblings: vec![],
        };
        let (fa, fb) = (make(sa), make(sb));
        let mut legs = if ta < tb { vec![fb, fa] } else { vec![fa, fb] }; // descending: unresolvable
        for l in &mut legs {
            let t = l.shell.txid().unwrap();
            l.tau = tree.witness(tree.position(&t).unwrap()).unwrap();
        }
        let st = genesis_state(s7);
        let mut b = block(s7, interval, legs);
        seal(&st, &mut b, &mut world);
        let facts = BeaconFacts::for_block(&b, &world);
        let p = FraudProof { block: b.clone(), facts, condition: FraudCondition::Unresolvable };
        p.verify(&world).unwrap();
        assert!(matches!(
            apply_block(st, &b, &world, &world),
            Err(ExecutorError::Resolve(_))
        ));
    }

    #[test]
    fn facts_and_malformed_rejections() {
        let set = ShardSet::genesis();
        let st = genesis_state(set.ids()[7]);
        let mut world = World::default();
        let (b, _) = valid_block(&st, &mut world, 0xF00D);

        let mut wrong = BeaconFacts::for_block(&b, &world);
        wrong.tau_root = Some(Hash256::from_bytes([0xEE; 32]));
        let p = FraudProof { block: b.clone(), facts: wrong, condition: FraudCondition::TauWitness { leg: 0 } };
        assert!(matches!(p.verify(&world), Err(FraudError::FactsRejected)));

        let facts = BeaconFacts::for_block(&b, &world);
        for condition in [
            FraudCondition::TauWitness { leg: 99 },
            FraudCondition::SettlementDeadline { leg: 99 },
            FraudCondition::InBlockNullifier { leg: 99 },
        ] {
            let p = FraudProof { block: b.clone(), facts: facts.clone(), condition };
            assert!(matches!(p.verify(&world), Err(FraudError::Malformed(_))), "{condition:?}");
        }

        let mut absent = facts.clone();
        absent.tau_root = None;
        let p = FraudProof { block: b.clone(), facts: absent, condition: FraudCondition::TauWitness { leg: 0 } };
        assert!(matches!(p.verify(&world), Err(FraudError::FactsRejected)));
    }

    #[test]
    fn reexecution_class() {
        let set = ShardSet::genesis();
        let st = genesis_state(set.ids()[7]);
        let mut world = World::default();
        let (b, _) = valid_block(&st, &mut world, 0xF00E);

        assert!(matches!(
            ReexecutionFraud { block: b.clone() }.verify(&st, &world, &world),
            Err(FraudError::ConditionNotExhibited)
        ));

        let mut bad = b.clone();
        bad.header.nullifier_root = Hash256::from_bytes([0xEE; 32]);
        let e = ReexecutionFraud { block: bad }.verify(&st, &world, &world).unwrap();
        assert!(matches!(e, ExecutorError::NullifierRootMismatch));

        let mut bad = b.clone();
        bad.header.fee_total = FeeSats::from_u64(bad.header.fee_total.as_u64() + 1);
        let e = ReexecutionFraud { block: bad }.verify(&st, &world, &world).unwrap();
        assert!(matches!(e, ExecutorError::FeeTotalMismatch));

        assert_eq!(
            ReexecutionFraud { block: b.clone() }.digest(),
            ReexecutionFraud { block: b.clone() }.digest()
        );
    }

    #[test]
    fn digest_and_codec_roundtrips() {
        let set = ShardSet::genesis();
        let st = genesis_state(set.ids()[7]);
        let mut world = World::default();
        let (b, _) = valid_block(&st, &mut world, 0xF00F);
        let facts = BeaconFacts::for_block(&b, &world);

        let p = FraudProof {
            block: b.clone(),
            facts: facts.clone(),
            condition: FraudCondition::TauWitness { leg: 0 },
        };
        assert_eq!(p.digest(), p.digest());
        let enc = p.encode();
        assert_eq!(enc.len(), p.encoded_len());
        assert_eq!(FraudProof::decode(&enc).unwrap(), p);
        for cut in 0..enc.len() {
            assert!(FraudProof::decode(&enc[..cut]).is_err(), "cut {cut}");
        }
        let mut ext = enc.clone();
        ext.push(0);
        assert!(FraudProof::decode(&ext).is_err());

        let q = FraudProof {
            block: b.clone(),
            facts: facts.clone(),
            condition: FraudCondition::FeeTotalMismatch,
        };
        assert_ne!(p.digest(), q.digest());

        let fe = facts.encode();
        assert_eq!(fe.len(), facts.encoded_len());
        assert_eq!(BeaconFacts::decode(&fe).unwrap(), facts);
        assert!(BeaconFacts::decode(&fe[..fe.len() - 1]).is_err());

        for c in [
            FraudCondition::Unresolvable,
            FraudCondition::FeeTotalMismatch,
            FraudCondition::TauWitness { leg: 3 },
            FraudCondition::SiblingEvidence { leg: 1, sibling: 2 },
        ] {
            assert_eq!(FraudCondition::decode(&c.encode()).unwrap(), c);
        }

        let r = ReexecutionFraud { block: b.clone() };
        assert_eq!(ReexecutionFraud::decode(&r.encode()).unwrap(), r);
        assert_ne!(r.digest(), p.digest()); // class tags differ
    }
}

