//! The block body (WP §4.3, §11.2; errata 105–111): settled-leg records
//! with the full-shell txid binding, D.3 reversion and claim records,
//! the quorum certificate, the §11.2 leg tree, and ct_B summation.

use std::sync::OnceLock;

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::{LEG_TREE_EMPTY, LEG_TREE_LEAF, LEG_TREE_NODE};
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::params::SHARDING_BLOCK_LEG_CAP;
use nerv_core::types::{Height, LegIndex, LegKey, ShardId, TxId};
use nerv_crypto::sigaggr::QuorumCertificate;
use nerv_custody::tx::TransactionShell;
use nerv_custody::TransitProof;
use nerv_seal::encrypt::Ciphertext;

use crate::error::BlockError;
use crate::header::ShardHeader;
use crate::ttau::TauWitness;

/// §11.2: depth ≤ 14 at the 10,000-leg block cap.
pub const LEG_TREE_DEPTH: usize = 14;
pub const LEG_TREE_CAPACITY: u64 = 1 << LEG_TREE_DEPTH;
/// The block leg cap (params' sharding.block_leg_cap).
pub const MAX_LEGS: usize = SHARDING_BLOCK_LEG_CAP as usize;

const _: () = assert!((1usize << LEG_TREE_DEPTH) >= MAX_LEGS);

const RECORD_CAP: usize = 65_536;

// ---------------------------------------------------------------------------
// The leg tree (§11.2; erratum 106)
// ---------------------------------------------------------------------------

pub fn leg_leaf_digest(txid: &TxId, leg: LegIndex) -> Hash256 {
    let mut msg = [0u8; 33];
    msg[..32].copy_from_slice(txid.as_bytes());
    msg[32] = leg.as_u8();
    Hash256::concat(&LEG_TREE_LEAF, &msg)
}

pub fn leg_node_digest(left: &Hash256, right: &Hash256) -> Hash256 {
    let mut msg = [0u8; 64];
    msg[..32].copy_from_slice(left.as_bytes());
    msg[32..].copy_from_slice(right.as_bytes());
    Hash256::concat(&LEG_TREE_NODE, &msg)
}

fn empty_digests() -> &'static [Hash256; LEG_TREE_DEPTH + 1] {
    static TABLE: OnceLock<[Hash256; LEG_TREE_DEPTH + 1]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = std::array::from_fn(|_| Hash256::from_bytes([0u8; 32]));
        t[0] = Hash256::concat(&LEG_TREE_EMPTY, &[0u8; 32]);
        for k in 0..LEG_TREE_DEPTH {
            t[k + 1] = leg_node_digest(&t[k], &t[k]);
        }
        t
    })
}

/// The frontier-fold tree over the block's legs in canonical order
/// (erratum 101's law at depth 14; leaves strictly ascending by LegKey).
#[derive(Clone, Default)]
pub struct BlockLegTree {
    keys: Vec<LegKey>,
    levels: Vec<Vec<Hash256>>,
    count: u64,
}

impl BlockLegTree {
    pub fn new() -> BlockLegTree {
        BlockLegTree::default()
    }

    pub fn from_resolved(resolved: &[ResolvedLeg]) -> Result<BlockLegTree, BlockError> {
        if resolved.len() > MAX_LEGS {
            return Err(BlockError::LegCapExceeded { count: resolved.len(), max: MAX_LEGS });
        }
        let mut t = BlockLegTree::new();
        for r in resolved {
            t.insert(r.key)?;
        }
        Ok(t)
    }

    pub fn insert(&mut self, key: LegKey) -> Result<u64, BlockError> {
        if self.count >= LEG_TREE_CAPACITY {
            return Err(BlockError::LegTreeFull { capacity: LEG_TREE_CAPACITY });
        }
        if let Some(last) = self.keys.last() {
            if key <= *last {
                return Err(BlockError::LegUnsorted { key, prev: *last });
            }
        }
        let index = self.count;
        self.keys.push(key);
        self.count += 1;
        let mut node = leg_leaf_digest(&key.txid, key.leg);
        let mut k = 0usize;
        loop {
            if k == self.levels.len() {
                self.levels.push(Vec::new());
            }
            let lvl = &mut self.levels[k];
            lvl.push(node);
            if lvl.len() % 2 == 1 {
                break;
            }
            let n = lvl.len();
            let (left, right) = (lvl[n - 2], lvl[n - 1]);
            node = leg_node_digest(&left, &right);
            k += 1;
        }
        Ok(index)
    }

    pub fn len(&self) -> u64 {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn keys(&self) -> &[LegKey] {
        &self.keys
    }

    pub fn position(&self, key: &LegKey) -> Option<u64> {
        self.keys.binary_search(key).ok().map(|i| i as u64)
    }

    pub fn root(&self) -> Hash256 {
        let empty = empty_digests();
        if self.count == 0 {
            return empty[LEG_TREE_DEPTH];
        }
        let mut r = empty[0];
        for k in 0..LEG_TREE_DEPTH {
            if (self.count >> k) & 1 == 1 {
                let left = self
                    .levels
                    .get(k)
                    .and_then(|l| l.last().copied())
                    .unwrap_or(empty[k]);
                r = leg_node_digest(&left, &r);
            } else {
                r = leg_node_digest(&r, &empty[k]);
            }
        }
        r
    }

    pub fn witness(&self, index: u64) -> Result<LegTreeWitness, BlockError> {
        if index >= self.count {
            return Err(BlockError::LegTreeIndex { index, count: self.count });
        }
        let empty = empty_digests();
        let mut siblings = vec![Hash256::default(); LEG_TREE_DEPTH];
        let mut j = index;
        for k in 0..LEG_TREE_DEPTH {
            let sib = j ^ 1;
            siblings[k] = self
                .levels
                .get(k)
                .filter(|l| (sib as usize) < l.len())
                .and_then(|l| l.get(sib as usize))
                .copied()
                .unwrap_or(empty[k]);
            j >>= 1;
        }
        for k in (0..LEG_TREE_DEPTH).rev() {
            if siblings[k] == empty[k] {
                siblings.pop();
            } else {
                break;
            }
        }
        Ok(LegTreeWitness { index, siblings })
    }
}

impl std::fmt::Debug for BlockLegTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockLegTree").field("count", &self.count).finish()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegTreeWitness {
    pub index: u64,
    pub siblings: Vec<Hash256>,
}

impl Encode for LegTreeWitness {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.index.to_le_bytes());
        out.extend_from_slice(&(self.siblings.len() as u32).to_le_bytes());
        for s in &self.siblings {
            out.extend_from_slice(s.as_bytes());
        }
    }
    fn encoded_len(&self) -> usize {
        8 + 4 + 32 * self.siblings.len()
    }
}

impl Decode for LegTreeWitness {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let index = r.read_u64()?;
        if index >= LEG_TREE_CAPACITY {
            return Err(CodecError::InvariantViolated("leg-tree witness index exceeds capacity"));
        }
        let n = r.read_seq_len()?;
        if n > LEG_TREE_DEPTH {
            return Err(CodecError::SeqTooLarge { count: n, max: LEG_TREE_DEPTH });
        }
        let mut siblings = Vec::with_capacity(n);
        for _ in 0..n {
            siblings.push(Hash256::decode_from(r)?);
        }
        Ok(LegTreeWitness { index, siblings })
    }
}

/// Verify a leg-tree witness. Missing trailing siblings are the empty
/// digests (erratum 106). Adversarial input is rejected, never panics.
pub fn verify_leg_witness(
    root: &Hash256,
    index: u64,
    txid: &TxId,
    leg: LegIndex,
    siblings: &[Hash256],
) -> bool {
    if index >= LEG_TREE_CAPACITY {
        return false;
    }
    if siblings.len() > LEG_TREE_DEPTH {
        return false;
    }
    let empty = empty_digests();
    let mut cur = leg_leaf_digest(txid, leg);
    let mut j = index;
    for k in 0..LEG_TREE_DEPTH {
        let sib = if k < siblings.len() { siblings[k] } else { empty[k] };
        cur = if j & 1 == 0 {
            leg_node_digest(&cur, &sib)
        } else {
            leg_node_digest(&sib, &cur)
        };
        j >>= 1;
    }
    cur == *root
}

// ---------------------------------------------------------------------------
// Shell resolution (erratum 105)
// ---------------------------------------------------------------------------

/// The txid of an already-canonical shell — custody's `txid()` body over
/// the caller's canonical form (one canonicalize per shell, not two).
pub fn canonical_txid(canon: &TransactionShell) -> TxId {
    let mut buf = Vec::new();
    for l in &canon.legs {
        l.encode_into(&mut buf);
    }
    TxId::hash_canonical(&buf)
}

/// One block leg, resolved: the shell canonicalized, the txid derived, the
/// target leg identified and shard-checked, the settlement key fixed.
#[derive(Clone, Debug)]
pub struct ResolvedLeg {
    pub txid: TxId,
    pub key: LegKey,
    pub canon: TransactionShell,
    pub leg: usize,
}

impl ResolvedLeg {
    pub fn leg_shell(&self) -> &nerv_custody::tx::LegShell {
        &self.canon.legs[self.leg]
    }
}

// ---------------------------------------------------------------------------
// Records (D.3; erratum 107)
// ---------------------------------------------------------------------------

/// A transit-entry proof against a beacon-finalized transit root: the
/// membership evidence for rule 5 (sibling spend legs), claim records, and
/// the settled arm of reversion evidence.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TransitEvidence {
    pub root_height: Height,
    pub proof: TransitProof,
}

impl Encode for TransitEvidence {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.root_height.encode_into(out);
        self.proof.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        8 + self.proof.encoded_len()
    }
}

impl Decode for TransitEvidence {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let root_height = Height::decode_from(r)?;
        let proof = TransitProof::decode_from(r)?;
        Ok(TransitEvidence { root_height, proof })
    }
}

/// One issue leg's disposition inside a reversion record (erratum 107d).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum LegEvidence {
    /// The leg settled: membership of its transit entry against the
    /// receiving shard's finalized transit root at `root_height`.
    Settled(TransitEvidence),
    /// The leg did not settle through the escrow's expiry: non-membership
    /// against the receiving shard's finalized transit root at height
    /// `expiry` (the root the executor derives from the escrow entry).
    Unsettled { proof: TransitProof },
}

impl Encode for LegEvidence {
    fn encode_into(&self, out: &mut Vec<u8>) {
        match self {
            LegEvidence::Settled(ev) => {
                out.push(0);
                ev.encode_into(out);
            }
            LegEvidence::Unsettled { proof } => {
                out.push(1);
                proof.encode_into(out);
            }
        }
    }
    fn encoded_len(&self) -> usize {
        match self {
            LegEvidence::Settled(ev) => 1 + ev.encoded_len(),
            LegEvidence::Unsettled { proof } => 1 + proof.encoded_len(),
        }
    }
}

impl Decode for LegEvidence {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        match r.read_u8()? {
            0 => Ok(LegEvidence::Settled(TransitEvidence::decode_from(r)?)),
            1 => {
                let proof = TransitProof::decode_from(r)?;
                Ok(LegEvidence::Unsettled { proof })
            }
            tag => Err(CodecError::InvalidOptionTag { tag }),
        }
    }
}

/// The D.3 reversion record: consumes the escrow's Pending transit entry
/// and mints the unsettled issue legs' revert commitments.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ReversionRecord {
    pub txid: TxId,
    pub spend_leg: LegIndex,
    /// One entry per issue leg of the shell, canonical leg order.
    pub evidence: Vec<LegEvidence>,
}

/// The D.3 claim record: consumes the escrow's Pending transit entry on
/// evidence that every issue leg settled.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ClaimRecord {
    pub txid: TxId,
    pub spend_leg: LegIndex,
    /// One entry per issue leg of the shell, canonical leg order.
    pub evidence: Vec<TransitEvidence>,
}

fn encode_leg_index_seq(indices: &[LegIndex], out: &mut Vec<u8>) {
    out.extend_from_slice(&(indices.len() as u32).to_le_bytes());
    for i in indices {
        out.push(i.as_u8());
    }
}

fn read_capped<T: Decode>(r: &mut Reader<'_>, cap: usize) -> Result<Vec<T>, CodecError> {
    let n = r.read_seq_len()?;
    if n > cap {
        return Err(CodecError::SeqTooLarge { count: n, max: cap });
    }
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(T::decode_from(r)?);
    }
    Ok(v)
}

impl Encode for ReversionRecord {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.txid.encode_into(out);
        out.push(self.spend_leg.as_u8());
        out.extend_from_slice(&(self.evidence.len() as u32).to_le_bytes());
        for e in &self.evidence {
            e.encode_into(out);
        }
    }
    fn encoded_len(&self) -> usize {
        32 + 1 + 4 + self.evidence.iter().map(|e| e.encoded_len()).sum::<usize>()
    }
}

impl Decode for ReversionRecord {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let txid = TxId::decode_from(r)?;
        let spend_leg = LegIndex::from_u8(r.read_u8()?);
        let evidence = read_capped::<LegEvidence>(r, 255)?;
        Ok(ReversionRecord { txid, spend_leg, evidence })
    }
}

impl Encode for ClaimRecord {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.txid.encode_into(out);
        out.push(self.spend_leg.as_u8());
        out.extend_from_slice(&(self.evidence.len() as u32).to_le_bytes());
        for e in &self.evidence {
            e.encode_into(out);
        }
    }
    fn encoded_len(&self) -> usize {
        32 + 1 + 4 + self.evidence.iter().map(|e| e.encoded_len()).sum::<usize>()
    }
}

impl Decode for ClaimRecord {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let txid = TxId::decode_from(r)?;
        let spend_leg = LegIndex::from_u8(r.read_u8()?);
        let evidence = read_capped::<TransitEvidence>(r, 255)?;
        Ok(ClaimRecord { txid, spend_leg, evidence })
    }
}

// ---------------------------------------------------------------------------
// The settled leg (erratum 105)
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SettledLeg {
    /// The full transaction shell; the txid is re-derived at resolution.
    pub shell: TransactionShell,
    /// The settling leg's index into the canonical shell.
    pub leg: LegIndex,
    /// Rule 1: T_τ membership (the root is the header's registry ref).
    pub tau: TauWitness,
    /// Rule 5 (issue legs only): one entry per sibling spend leg,
    /// canonical leg order.
    pub siblings: Vec<TransitEvidence>,
}

impl Encode for SettledLeg {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.shell.encode_into(out);
        out.push(self.leg.as_u8());
        self.tau.encode_into(out);
        out.extend_from_slice(&(self.siblings.len() as u32).to_le_bytes());
        for s in &self.siblings {
            s.encode_into(out);
        }
    }
    fn encoded_len(&self) -> usize {
        self.shell.encoded_len() + 1 + self.tau.encoded_len() + 4
            + self.siblings.iter().map(|s| s.encoded_len()).sum::<usize>()
    }
}

impl Decode for SettledLeg {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let shell = TransactionShell::decode_from(r)?;
        let leg = LegIndex::from_u8(r.read_u8()?);
        let tau = TauWitness::decode_from(r)?;
        let siblings = read_capped::<TransitEvidence>(r, 255)?;
        Ok(SettledLeg { shell, leg, tau, siblings })
    }
}

// ---------------------------------------------------------------------------
// The block (§4.3, §4.6)
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ShardBlock {
    pub shard: ShardId,
    pub header: ShardHeader,
    /// Canonically ordered by (txid, leg) — strictly ascending.
    pub legs: Vec<SettledLeg>,
    pub reversions: Vec<ReversionRecord>,
    pub claims: Vec<ClaimRecord>,
    /// The full ML-DSA certificate (§4.6); the header commits its hash.
    pub qc: QuorumCertificate,
}

impl ShardBlock {
    /// Resolves every leg: canonicalize, derive the txid, check the leg
    /// index and the shard, enforce the cap and strict canonical order
    /// (errata 105, 108).
    pub fn resolve_legs(&self) -> Result<Vec<ResolvedLeg>, BlockError> {
        if self.legs.len() > MAX_LEGS {
            return Err(BlockError::LegCapExceeded { count: self.legs.len(), max: MAX_LEGS });
        }
        let mut out: Vec<ResolvedLeg> = Vec::with_capacity(self.legs.len());
        for (i, sl) in self.legs.iter().enumerate() {
            let canon = sl
                .shell
                .canonicalize()
                .map_err(|source| BlockError::BadShell { leg: i, source })?;
            let txid = canonical_txid(&canon);
            let li = usize::from(sl.leg.as_u8());
            if li >= canon.legs.len() {
                return Err(BlockError::LegIndex { leg: i, index: li, len: canon.legs.len() });
            }
            if canon.legs[li].shard != self.shard {
                return Err(BlockError::WrongShard {
                    leg: i,
                    found: canon.legs[li].shard,
                    expected: self.shard,
                });
            }
            let key = LegKey::new(txid, sl.leg);
            if let Some(prev) = out.last() {
                if key <= prev.key {
                    return Err(BlockError::LegUnsorted { key, prev: prev.key });
                }
            }
            out.push(ResolvedLeg { txid, key, canon, leg: li });
        }
        Ok(out)
    }

    pub fn leg_tree(&self) -> Result<BlockLegTree, BlockError> {
        BlockLegTree::from_resolved(&self.resolve_legs()?)
    }
}

impl Encode for ShardBlock {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.shard.encode_into(out);
        self.header.encode_into(out);
        out.extend_from_slice(&(self.legs.len() as u32).to_le_bytes());
        for l in &self.legs {
            l.encode_into(out);
        }
        out.extend_from_slice(&(self.reversions.len() as u32).to_le_bytes());
        for r in &self.reversions {
            r.encode_into(out);
        }
        out.extend_from_slice(&(self.claims.len() as u32).to_le_bytes());
        for c in &self.claims {
            c.encode_into(out);
        }
        self.qc.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        3 + self.header.encoded_len() + 4 + self.legs.iter().map(|l| l.encoded_len()).sum::<usize>()
            + 4 + self.reversions.iter().map(|r| r.encoded_len()).sum::<usize>()
            + 4 + self.claims.iter().map(|c| c.encoded_len()).sum::<usize>()
            + self.qc.encoded_len()
    }
}

impl Decode for ShardBlock {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let shard = ShardId::decode_from(r)?;
        let header = ShardHeader::decode_from(r)?;
        let legs = read_capped::<SettledLeg>(r, MAX_LEGS)?;
        let reversions = read_capped::<ReversionRecord>(r, RECORD_CAP)?;
        let claims = read_capped::<ClaimRecord>(r, RECORD_CAP)?;
        let qc = QuorumCertificate::decode_from(r)?;
        Ok(ShardBlock { shard, header, legs, reversions, claims, qc })
    }
}

/// ct_B = Σ ct_leg mod q (rule 6; erratum 110) over resolved legs.
pub fn ct_sum(resolved: &[ResolvedLeg]) -> Result<Ciphertext, BlockError> {
    let mut sum = Ciphertext::zero();
    for (i, r) in resolved.iter().enumerate() {
        let ct = Ciphertext::from_bytes(&r.leg_shell().ct)
            .map_err(|source| BlockError::BadCt { leg: i, source })?;
        sum = sum.add(&ct);
    }
    Ok(sum)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::header::RegistryRef;
    use crate::testutil::SplitMix64;
    use nerv_core::field::Goldilocks;
    use nerv_core::types::{Epoch, FeeSats, Interval, ShardSet};
    use nerv_crypto::mldsa::{SigningKey, VerifyingKey};
    use nerv_crypto::sigaggr::{vote_bytes, VoteCollector};
    use nerv_custody::nct::NctDigest;
    use nerv_custody::tx::{InputSet, LegShell, Output};
    use nerv_custody::{Address, MasterSeed, NoteOpening, TransitEntry, TransitLog, WalletKeys};
    use nerv_seal::digitize::{digitize, COORDS};
    use nerv_seal::encrypt::derive_reference_keypair;
    use nerv_seal::sampling::NoiseSeed;
    use proptest::prelude::*;

    struct Fix {
        wk: WalletKeys,
        set: ShardSet,
        rng: SplitMix64,
        addr_idx: u64,
    }

    impl Fix {
        fn new(seed: u64) -> Fix {
            let mut rng = SplitMix64::new(seed);
            let wk = WalletKeys::from_master(&MasterSeed::from_bytes(rng.bytes32()));
            Fix { wk, set: ShardSet::genesis(), rng, addr_idx: 0 }
        }

        fn opening(&mut self, v: u64) -> NoteOpening {
            let i = self.addr_idx;
            self.addr_idx += 1;
            let addr =
                Address::generate(self.wk.viewing(), self.wk.nullifier_key(), i, &self.set)
                    .unwrap();
            NoteOpening {
                value: v,
                rho: self.rng.bytes32(),
                delivery: *addr.delivery().as_bytes(),
                blinding: self.rng.bytes32(),
                pk_n: *addr.pk_n(),
            }
        }

        /// Parseable dummy ct: every u32 coefficient < Q.
        fn ct(&mut self) -> Vec<u8> {
            let mut out = Vec::with_capacity(Ciphertext::WIRE_SIZE);
            while out.len() < Ciphertext::WIRE_SIZE {
                out.extend_from_slice(&(self.rng.next_u32() & 0xFFF0_0000).to_le_bytes());
            }
            out
        }
    }

    fn leg_of(fix: &mut Fix, shard: nerv_core::types::ShardId, n_in: usize, n_out: usize) -> LegShell {
        let inputs: Vec<nerv_core::hash::Hash256> =
            (0..n_in).map(|_| nerv_core::hash::Hash256::from_bytes(fix.rng.bytes32())).collect();
        let outputs: Vec<Output> = (0..n_out)
            .map(|_| {
                let o = fix.opening(1_000_000_000);
                Output {
                    cm: o.commitment().unwrap(),
                    sealed_note: vec![0xA5; 48],
                    value: o.value,
                    conditional: false,
                    revert_cm: None,
                }
            })
            .collect();
        LegShell {
            shard,
            inputs: InputSet::new(inputs),
            outputs,
            fee: FeeSats::from_u64(1000),
            anchor: nerv_core::hash::Hash256::from_bytes(fix.rng.bytes32()),
            expiry: nerv_core::types::Height::from_u64(5_000),
            weight_version: 1,
            ct: fix.ct(),
            burns: vec![],
        }
    }

    fn digest(k: u32) -> NctDigest {
        NctDigest::from_elements(&[
            Goldilocks::from_u32(k),
            Goldilocks::from_u32(k.wrapping_add(1)),
            Goldilocks::from_u32(k.wrapping_mul(3)),
            Goldilocks::from_u32(k.wrapping_mul(7)),
        ])
    }

    fn header(fix: &mut Fix, height: u64) -> ShardHeader {
        ShardHeader {
            prev: nerv_core::hash::Hash256::from_bytes(fix.rng.bytes32()),
            height: nerv_core::types::Height::from_u64(height),
            nct_root: digest(7),
            nullifier_root: nerv_core::hash::Hash256::from_bytes(fix.rng.bytes32()),
            transit_root: nerv_core::hash::Hash256::from_bytes(fix.rng.bytes32()),
            params_root: nerv_core::hash::Hash256::from_bytes(fix.rng.bytes32()),
            derived: fix.rng.bytes32(),
            ct_batch_hash: nerv_core::hash::Hash256::from_bytes(fix.rng.bytes32()),
            prev_reveal: None,
            registry: RegistryRef {
                interval: Interval::from_u64(86_400),
                root: nerv_core::hash::Hash256::from_bytes(fix.rng.bytes32()),
            },
            fee_total: FeeSats::from_u64(1_400),
            producer_payout: Address::generate(
                fix.wk.viewing(),
                fix.wk.nullifier_key(),
                100,
                &fix.set,
            )
            .unwrap(),
            qc_hash: nerv_core::hash::Hash256::from_bytes(fix.rng.bytes32()),
        }
    }

    fn committee_qc(subject: nerv_core::hash::Hash256) -> QuorumCertificate {
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
        let mut vc = VoteCollector::new(epoch, subject);
        for (i, k) in keys.iter().enumerate().take(15) {
            vc.add(i, k.sign(&vote_bytes(epoch, &subject)).unwrap(), &roster).unwrap();
        }
        vc.assemble(15).unwrap()
    }

    fn transit_proof(
        txid: TxId,
        shard: nerv_core::types::ShardId,
        leg: LegIndex,
    ) -> (TransitEntry, TransitProof) {
        let mut log = TransitLog::new();
        let e = TransitEntry::new_pending(
            txid,
            shard,
            leg,
            nerv_core::types::Height::from_u64(400),
            nerv_core::types::Height::from_u64(5_000),
        );
        log.insert_pending(e).unwrap();
        let p = log.membership_proof(&e);
        (e, p)
    }

    // -- the leg tree ---------------------------------------------------------

    fn reference_root(leaves: &[Hash256], empty: &[Hash256; LEG_TREE_DEPTH + 1]) -> Hash256 {
        fn rec(k: usize, i: u64, leaves: &[Hash256], empty: &[Hash256; LEG_TREE_DEPTH + 1]) -> Hash256 {
            if (i << k) >= leaves.len() as u64 {
                return empty[k];
            }
            if k == 0 {
                return leaves[i as usize];
            }
            let l = rec(k - 1, 2 * i, leaves, empty);
            let r = rec(k - 1, 2 * i + 1, leaves, empty);
            leg_node_digest(&l, &r)
        }
        rec(LEG_TREE_DEPTH, 0, leaves, empty)
    }

    fn sorted_keys(seed: u64, n: usize) -> Vec<LegKey> {
        let mut rng = SplitMix64::new(seed);
        let mut txids: Vec<TxId> =
            (0..n).map(|_| TxId::from_hash(Hash256::from_bytes(rng.bytes32()))).collect();
        txids.sort();
        txids.dedup();
        let mut out = Vec::with_capacity(n);
        let mut i = 0usize;
        while out.len() < n {
            let leg = if i == 0 { 0 } else { (i % 3) as u8 };
            let key = LegKey::new(txids[i % txids.len().max(1)], LegIndex::from_u8(leg));
            if out.last().map_or(true, |l| *l < key) {
                out.push(key);
            }
            i += 1;
        }
        out
    }

    #[test]
    fn leg_tree_literal_pins() {
        assert_eq!(LEG_TREE_DEPTH, 14);
        assert_eq!(LEG_TREE_CAPACITY, 16_384);
        assert!(MAX_LEGS <= LEG_TREE_CAPACITY as usize);
        let e = empty_digests();
        assert_eq!(e.len(), LEG_TREE_DEPTH + 1);
        let mut pre = Vec::new();
        pre.extend_from_slice(LEG_TREE_EMPTY.as_bytes());
        pre.extend_from_slice(&[0u8; 32]);
        assert_eq!(e[0].as_bytes(), blake3::hash(&pre).as_bytes());
        for k in 1..=LEG_TREE_DEPTH {
            assert_eq!(e[k], leg_node_digest(&e[k - 1], &e[k - 1]));
        }

        let t = BlockLegTree::new();
        assert!(t.is_empty());
        assert_eq!(t.root(), e[LEG_TREE_DEPTH]);

        let id = TxId::from_hash(Hash256::from_bytes([3u8; 32]));
        let leg = LegIndex::from_u8(9);
        let mut m = [0u8; 33];
        m[..32].copy_from_slice(id.as_bytes());
        m[32] = 9;
        let mut pre = Vec::new();
        pre.extend_from_slice(LEG_TREE_LEAF.as_bytes());
        pre.extend_from_slice(&m);
        assert_eq!(leg_leaf_digest(&id, leg).as_bytes(), blake3::hash(&pre).as_bytes());

        let (l, r) = (Hash256::from_bytes([1u8; 32]), Hash256::from_bytes([2u8; 32]));
        let mut n = Vec::new();
        n.extend_from_slice(LEG_TREE_NODE.as_bytes());
        n.extend_from_slice(l.as_bytes());
        n.extend_from_slice(r.as_bytes());
        assert_eq!(leg_node_digest(&l, &r).as_bytes(), blake3::hash(&n).as_bytes());
    }

    #[test]
    fn leg_tree_matches_reference_and_witnesses_verify() {
        let empty = *empty_digests();
        for (seed, n) in [(0x1A1u64, 1usize), (0x1A2, 2), (0x1A3, 17), (0x1A4, 100)] {
            let keys = sorted_keys(seed, n);
            let mut t = BlockLegTree::new();
            for k in &keys {
                t.insert(*k).unwrap();
            }
            let leaves: Vec<Hash256> = keys.iter().map(|k| leg_leaf_digest(&k.txid, k.leg)).collect();
            let root = t.root();
            assert_eq!(root, reference_root(&leaves, &empty), "n = {n}");

            for (i, k) in keys.iter().enumerate() {
                let i = i as u64;
                let w = t.witness(i).unwrap();
                assert!(verify_leg_witness(&root, i, &k.txid, k.leg, &w.siblings), "n={n} i={i}");

                let mut full = w.siblings.clone();
                while full.len() < LEG_TREE_DEPTH {
                    let d = full.len();
                    full.push(empty[d]);
                }
                assert!(verify_leg_witness(&root, i, &k.txid, k.leg, &full));

                let other = &keys[(i as usize + 1) % n];
                if other.txid != k.txid || other.leg != k.leg {
                    assert!(!verify_leg_witness(&root, i, &other.txid, other.leg, &w.siblings));
                }
                assert!(!verify_leg_witness(
                    &Hash256::from_bytes([9u8; 32]),
                    i,
                    &k.txid,
                    k.leg,
                    &w.siblings
                ));

                if !w.siblings.is_empty() {
                    let mut bad = w.siblings.clone();
                    let mid = bad.len() / 2;
                    let mut b = *bad[mid].as_bytes();
                    b[0] ^= 1;
                    bad[mid] = Hash256::from_bytes(b);
                    assert!(!verify_leg_witness(&root, i, &k.txid, k.leg, &bad));
                    let mut short = w.siblings.clone();
                    short.pop();
                    assert!(!verify_leg_witness(&root, i, &k.txid, k.leg, &short));
                    let last = *w.siblings.last().unwrap();
                    assert_ne!(last, empty[w.siblings.len() - 1]);
                }
            }
            assert!(matches!(t.witness(n as u64), Err(BlockError::LegTreeIndex { .. })));
        }
    }

    #[test]
    fn leg_tree_ascending_capacity_and_determinism() {
        let mut rng = SplitMix64::new(0x1A5);
        let a = TxId::from_hash(Hash256::from_bytes(rng.bytes32()));
        let b = TxId::from_hash(Hash256::from_bytes(rng.bytes32()));
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let k0 = LegKey::new(lo, LegIndex::FIRST);
        let k1 = LegKey::new(hi, LegIndex::FIRST);

        let mut t = BlockLegTree::new();
        assert_eq!(t.insert(k0).unwrap(), 0);
        assert_eq!(t.insert(k1).unwrap(), 1);
        assert!(matches!(t.insert(k0), Err(BlockError::LegUnsorted { .. })));
        assert!(matches!(t.insert(k1), Err(BlockError::LegUnsorted { .. })));
        assert_eq!(t.len(), 2);
        assert_eq!(t.position(&k1), Some(1));
        assert_eq!(t.position(&k0), Some(0));

        let same = LegKey::new(hi, LegIndex::from_u8(1));
        assert!(matches!(t.insert(same), Err(BlockError::LegUnsorted { .. })));
        let next = LegKey::new(hi, LegIndex::from_u8(2));
        t.insert(next).unwrap();

        let mut full = BlockLegTree::new();
        let keys = sorted_keys(0x1A6, 200);
        for k in &keys {
            full.insert(*k).unwrap();
        }
        assert_eq!(full.len(), 200);

        let t2 = {
            let mut t2 = BlockLegTree::new();
            for k in &keys {
                t2.insert(*k).unwrap();
            }
            t2
        };
        assert_eq!(t2.root(), full.root());
        for i in 0..200u64 {
            assert_eq!(t2.witness(i).unwrap(), full.witness(i).unwrap());
        }
        assert_eq!(full.keys(), &keys[..]);
    }

    #[test]
    fn leg_tree_witness_codec() {
        let keys = sorted_keys(0x1A7, 9);
        let mut t = BlockLegTree::new();
        for k in &keys {
            t.insert(*k).unwrap();
        }
        let w = t.witness(3).unwrap();
        let enc = w.encode();
        assert_eq!(enc.len(), w.encoded_len());
        assert_eq!(LegTreeWitness::decode(&enc).unwrap(), w);
        assert!(verify_leg_witness(&t.root(), 3, &keys[3].txid, keys[3].leg, &w.siblings));
        for cut in 0..enc.len() {
            assert!(LegTreeWitness::decode(&enc[..cut]).is_err(), "cut {cut}");
        }
        let mut ext = enc.clone();
        ext.push(0);
        assert!(LegTreeWitness::decode(&ext).is_err());
        let mut bad = enc.clone();
        bad[0..8].copy_from_slice(&LEG_TREE_CAPACITY.to_le_bytes());
        assert!(matches!(
            LegTreeWitness::decode(&bad),
            Err(CodecError::InvariantViolated(_))
        ));
        let mut bad = enc.clone();
        bad[8..12].copy_from_slice(&15u32.to_le_bytes());
        assert!(matches!(
            LegTreeWitness::decode(&bad),
            Err(CodecError::SeqTooLarge { max: 14, .. })
        ));
    }

    // -- shell resolution -----------------------------------------------------

    #[test]
    fn canonical_txid_matches_custody() {
        let mut fix = Fix::new(0x2B1);
        let shell = TransactionShell {
            legs: vec![
                leg_of(&mut fix, fix.set.ids()[7], 2, 2),
                leg_of(&mut fix, fix.set.ids()[40], 0, 1),
            ],
        };
        let canon = shell.canonicalize().unwrap();
        assert_eq!(canonical_txid(&canon), shell.txid().unwrap());
        assert_eq!(canonical_txid(&canon), canon.txid().unwrap());
    }

    fn block_with_legs(fix: &mut Fix, legs: Vec<SettledLeg>) -> ShardBlock {
        let qc_subject = Hash256::from_bytes(fix.rng.bytes32());
        let qc = committee_qc(qc_subject);
        let mut hdr = header(fix, 1);
        hdr.qc_hash = qc.qc_hash();
        ShardBlock {
            shard: fix.set.ids()[7],
            header: hdr,
            legs,
            reversions: vec![],
            claims: vec![],
            qc,
        }
    }

    fn settled(fix: &mut Fix, n_in: usize, n_out: usize) -> SettledLeg {
        let leg = leg_of(fix, fix.set.ids()[7], n_in, n_out);
        SettledLeg {
            shell: TransactionShell { legs: vec![leg] },
            leg: LegIndex::FIRST,
            tau: crate::ttau::TauWitness { index: 0, siblings: vec![] },
            siblings: vec![],
        }
    }

    #[test]
    fn resolve_legs_validates() {
        let mut fix = Fix::new(0x2B2);
        let a = settled(&mut fix, 1, 1);
        let b = settled(&mut fix, 2, 1);
        let mut block = block_with_legs(&mut fix, vec![a.clone(), b.clone()]);
        let resolved = block.resolve_legs().unwrap();
        assert_eq!(resolved.len(), 2);
        assert!(resolved[0].key < resolved[1].key);
        assert_eq!(resolved[0].txid, resolved[0].key.txid);
        assert_eq!(resolved[0].leg_shell().shard, block.shard);
        assert_eq!(resolved[0].leg_shell().ct, resolved[0].canon.legs[0].ct);
        assert_eq!(canonical_txid(&resolved[0].canon), a.shell.txid().unwrap());

        // Out of order: invalid.
        let mut bad = block_with_legs(&mut fix, vec![b.clone(), a.clone()]);
        assert!(matches!(bad.resolve_legs(), Err(BlockError::LegUnsorted { .. })));

        // Leg index outside the shell.
        let mut oob = a.clone();
        oob.leg = LegIndex::from_u8(1);
        let mut bad = block_with_legs(&mut fix, vec![oob]);
        assert!(matches!(bad.resolve_legs(), Err(BlockError::LegIndex { index: 1, len: 1, .. })));

        // Wrong shard: the leg homed to shard 40 in this shard's block.
        let mut foreign = SettledLeg {
            shell: TransactionShell { legs: vec![leg_of(&mut fix, fix.set.ids()[40], 1, 1)] },
            leg: LegIndex::FIRST,
            tau: crate::ttau::TauWitness { index: 0, siblings: vec![] },
            siblings: vec![],
        };
        foreign.shell.canonicalize().unwrap();
        let mut bad = block_with_legs(&mut fix, vec![foreign]);
        assert!(matches!(bad.resolve_legs(), Err(BlockError::WrongShard { .. })));

        // Malformed shell: duplicate shard legs.
        let mut dup = SettledLeg {
            shell: TransactionShell {
                legs: vec![
                    leg_of(&mut fix, fix.set.ids()[7], 1, 1),
                    leg_of(&mut fix, fix.set.ids()[7], 1, 0),
                ],
            },
            leg: LegIndex::FIRST,
            tau: crate::ttau::TauWitness { index: 0, siblings: vec![] },
            siblings: vec![],
        };
        dup.shell.canonicalize().unwrap_err();
        let mut bad = block_with_legs(&mut fix, vec![dup]);
        assert!(matches!(bad.resolve_legs(), Err(BlockError::BadShell { .. })));

        // Over the cap.
        let mut many = Vec::new();
        for _ in 0..(MAX_LEGS + 1) {
            many.push(settled(&mut fix, 1, 1));
        }
        let mut bad = block_with_legs(&mut fix, many);
        assert!(matches!(
            bad.resolve_legs(),
            Err(BlockError::LegCapExceeded { count, max }) if count == MAX_LEGS + 1 && max == MAX_LEGS
        ));
    }

    #[test]
    fn leg_tree_builds_from_resolved() {
        let mut fix = Fix::new(0x2B3);
        let legs: Vec<SettledLeg> =
            (0..37).map(|_| settled(&mut fix, 1, 1)).collect();
        let block = block_with_legs(&mut fix, legs);
        let resolved = block.resolve_legs().unwrap();
        let tree = BlockLegTree::from_resolved(&resolved).unwrap();
        assert_eq!(tree.len(), 37);
        let root = tree.root();
        for (i, r) in resolved.iter().enumerate() {
            let w = tree.witness(i as u64).unwrap();
            assert!(verify_leg_witness(&root, i as u64, &r.txid, r.key.leg, &w.siblings));
        }
        assert_eq!(block.leg_tree().unwrap().root(), root);
    }

    // -- ct_B ------------------------------------------------------------------

    #[test]
    fn ct_sum_sums_real_ciphertexts_mod_q() {
        let (pk, _) = derive_reference_keypair(&[0xB7; 32]).unwrap();
        let mut st = 0x2B4u64;
        let mut coords = || {
            let mut c = [0u64; COORDS];
            for v in c.iter_mut() {
                let s = SplitMix64::new(st);
                st = s.state;
                st = st.wrapping_add(0x9E37_79B9_7F4A_7C15);
                *v = SplitMix64::new(st).next_u64();
            }
            c
        };
        let mk = |s: &mut u64| -> Vec<u8> {
            let mut b = [0u8; 32];
            b[..8].copy_from_slice(&SplitMix64::new(*s).next_u64().to_le_bytes());
            *s = *s ^ 0x2B5;
            Ciphertext::encrypt(&pk, &NoiseSeed::from_bytes(b), &digitize(&coords())).unwrap()
        };
        let ct1 = mk(&mut st);
        let ct2 = mk(&mut st);
        let ct3 = mk(&mut st);

        let mut fix = Fix::new(0x2B6);
        let mk_leg = |fix: &mut Fix, ct: Vec<u8>| {
            let mut l = leg_of(fix, fix.set.ids()[7], 1, 1);
            l.ct = ct;
            SettledLeg {
                shell: TransactionShell { legs: vec![l] },
                leg: LegIndex::FIRST,
                tau: crate::ttau::TauWitness { index: 0, siblings: vec![] },
                siblings: vec![],
            }
        };
        let block = block_with_legs(
            &mut fix,
            vec![mk_leg(&mut fix, ct1.clone()), mk_leg(&mut fix, ct2.clone()), mk_leg(&mut fix, ct3.clone())],
        );
        let resolved = block.resolve_legs().unwrap();
        let sum = ct_sum(&resolved).unwrap();
        let manual = Ciphertext::from_bytes(&ct1).unwrap();
        let manual = manual.add(&Ciphertext::from_bytes(&ct2).unwrap());
        let manual = manual.add(&Ciphertext::from_bytes(&ct3).unwrap());
        assert_eq!(sum.to_bytes(), manual.to_bytes());
        assert_eq!(ct_sum(&[]).unwrap().to_bytes(), Ciphertext::zero().to_bytes());
    }

    #[test]
    fn ct_sum_rejects_malformed() {
        let mut fix = Fix::new(0x2B7);
        let mut short = settled(&mut fix, 1, 1);
        short.shell.canonicalize().unwrap();
        {
            let mut l = short.shell.legs[0].clone();
            l.ct = vec![0xA5; 10];
            short.shell = TransactionShell { legs: vec![l] };
        }
        let block = block_with_legs(&mut fix, vec![short]);
        let resolved = block.resolve_legs().unwrap();
        assert!(matches!(ct_sum(&resolved), Err(BlockError::BadCt { leg: 0, .. })));

        // Unreduced coefficient: u32 ≥ Q at the first word.
        let mut bad = settled(&mut fix, 1, 1);
        {
            let mut l = bad.shell.legs[0].clone();
            l.ct = vec![0u8; Ciphertext::WIRE_SIZE];
            l.ct[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
            bad.shell = TransactionShell { legs: vec![l] };
        }
        let block = block_with_legs(&mut fix, vec![bad]);
        let resolved = block.resolve_legs().unwrap();
        assert!(matches!(ct_sum(&resolved), Err(BlockError::BadCt { leg: 0, .. })));
    }

    // -- codecs ----------------------------------------------------------------

    fn full_block(seed: u64) -> ShardBlock {
        let mut fix = Fix::new(seed);
        let leg = leg_of(&mut fix, fix.set.ids()[7], 2, 2);
        let txid = TransactionShell { legs: vec![leg.clone()] }.txid().unwrap();

        let (_, proof) = transit_proof(txid, fix.set.ids()[7], LegIndex::FIRST);
        let settled_leg = SettledLeg {
            shell: TransactionShell { legs: vec![leg] },
            leg: LegIndex::FIRST,
            tau: crate::ttau::TauWitness { index: 5, siblings: vec![] },
            siblings: vec![TransitEvidence {
                root_height: nerv_core::types::Height::from_u64(432),
                proof: proof.clone(),
            }],
        };

        let (_, proof2) = transit_proof(txid, fix.set.ids()[9], LegIndex::from_u8(2));
        let rev = ReversionRecord {
            txid,
            spend_leg: LegIndex::FIRST,
            evidence: vec![
                LegEvidence::Settled(TransitEvidence {
                    root_height: nerv_core::types::Height::from_u64(500),
                    proof: proof2,
                }),
                LegEvidence::Unsettled { proof },
            ],
        };
        let claim = ClaimRecord {
            txid,
            spend_leg: LegIndex::FIRST,
            evidence: vec![TransitEvidence {
                root_height: nerv_core::types::Height::from_u64(600),
                proof: TransitLog::new().non_membership_proof(&txid.as_hash()),
            }],
        };
        block_with_legs(&mut fix, vec![settled_leg])
            .with_records(vec![rev], vec![claim])
    }

    impl ShardBlock {
        fn with_records(
            mut self,
            reversions: Vec<ReversionRecord>,
            claims: Vec<ClaimRecord>,
        ) -> ShardBlock {
            self.reversions = reversions;
            self.claims = claims;
            self
        }
    }

    #[test]
    fn codec_roundtrips_and_strictness() {
        let block = full_block(0x2B8);
        block.resolve_legs().unwrap();

        let enc = block.encode();
        assert_eq!(enc.len(), block.encoded_len());
        let dec = ShardBlock::decode(&enc).unwrap();
        assert_eq!(dec, block);
        for cut in 0..enc.len() {
            assert!(ShardBlock::decode(&enc[..cut]).is_err(), "cut {cut}");
        }
        let mut ext = enc.clone();
        ext.push(0);
        assert!(ShardBlock::decode(&ext).is_err());

        // Component roundtrips.
        let sl = &block.legs[0];
        let slc = sl.encode();
        assert_eq!(slc.len(), sl.encoded_len());
        assert_eq!(SettledLeg::decode(&slc).unwrap(), *sl);
        let rc = &block.reversions[0];
        assert_eq!(ReversionRecord::decode(&rc.encode()).unwrap(), *rc);
        assert_eq!(rc.encode().len(), rc.encoded_len());
        let cl = &block.claims[0];
        assert_eq!(ClaimRecord::decode(&cl.encode()).unwrap(), *cl);
        assert_eq!(cl.encode().len(), cl.encoded_len());

        // Tag discipline.
        let mut bad = rc.encode();
        let last = bad.len() - 1;
        bad[last] = 2;
        assert!(ReversionRecord::decode(&bad).is_err() || rc.evidence.len() == 0);
    }

    #[test]
    fn codec_rejects_absurd_counts() {
        let huge = 0xFFFF_FFFFu32.to_le_bytes().to_vec();
        assert!(matches!(
            SettledLeg::decode(&huge),
            Err(CodecError::SeqLenOverrun { .. })
        ));
    }

    proptest! {
        #[test]
        fn prop_leg_tree_witnesses(n in 1usize..40, seed in any::<u64>()) {
            let mut rng = SplitMix64::new(seed ^ 0x1AB);
            let mut txids: Vec<TxId> =
                (0..n).map(|_| TxId::from_hash(Hash256::from_bytes(rng.bytes32()))).collect();
            txids.sort();
            txids.dedup();
            let n = txids.len();
            let mut t = BlockLegTree::new();
            for id in &txids {
                t.insert(LegKey::new(*id, LegIndex::FIRST)).unwrap();
            }
            let empty = *empty_digests();
            let leaves: Vec<Hash256> =
                txids.iter().map(|id| leg_leaf_digest(id, LegIndex::FIRST)).collect();
            let root = t.root();
            prop_assert_eq!(root, reference_root(&leaves, &empty));
            for i in 0..n as u64 {
                let w = t.witness(i).unwrap();
                prop_assert!(verify_leg_witness(&root, i, &txids[i as usize], LegIndex::FIRST, &w.siblings));
            }
        }
    }
}

