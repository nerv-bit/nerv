//! The inclusion witness (WP §11.2; erratum 175): the ~600 B client-held
//! proof that a leg settled in a finalized block.


use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::types::{LegIndex, ShardId, TxId};
use nerv_state::block::LEG_TREE_DEPTH;


pub const MAX_LEG_SIBLINGS: usize = LEG_TREE_DEPTH;
pub const MAX_G_SIBLINGS: usize = 10;


/// The cold-verifier's 𝔾 path: the shard's position and siblings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GPath {
    pub shard_index: usize,
    pub siblings: Vec<Hash256>,
}


impl Encode for GPath {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.shard_index as u16).to_le_bytes());
        out.extend_from_slice(&(self.siblings.len() as u32).to_le_bytes());
        for s in &self.siblings {
            out.extend_from_slice(s.as_bytes());
        }
    }
    fn encoded_len(&self) -> usize {
        2 + 4 + 32 * self.siblings.len()
    }
}


impl Decode for GPath {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let shard_index = r.read_u16()? as usize;
        let n = r.read_seq_len()?;
        if n > MAX_G_SIBLINGS {
            return Err(CodecError::SeqTooLarge { count: n, max: MAX_G_SIBLINGS });
        }
        let mut siblings = Vec::with_capacity(n);
        for _ in 0..n {
            siblings.push(Hash256::decode_from(r)?);
        }
        Ok(GPath { shard_index, siblings })
    }
}


/// The per-leg portion: the leaf identity and its leg-tree path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegWitness {
    pub txid: TxId,
    pub leg: LegIndex,
    pub leaf_index: u64,
    pub siblings: Vec<Hash256>,
}


impl Encode for LegWitness {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.txid.encode_into(out);
        out.push(self.leg.as_u8());
        out.extend_from_slice(&self.leaf_index.to_le_bytes());
        out.extend_from_slice(&(self.siblings.len() as u32).to_le_bytes());
        for s in &self.siblings {
            out.extend_from_slice(s.as_bytes());
        }
    }
    fn encoded_len(&self) -> usize {
        32 + 1 + 8 + 4 + 32 * self.siblings.len()
    }
}


impl Decode for LegWitness {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let txid = TxId::decode_from(r)?;
        let leg = LegIndex::from_u8(r.read_u8()?);
        let leaf_index = r.read_u64()?;
        let n = r.read_seq_len()?;
        if n > MAX_LEG_SIBLINGS {
            return Err(CodecError::SeqTooLarge { count: n, max: MAX_LEG_SIBLINGS });
        }
        let mut siblings = Vec::with_capacity(n);
        for _ in 0..n {
            siblings.push(Hash256::decode_from(r)?);
        }
        Ok(LegWitness { txid, leg, leaf_index, siblings })
    }
}


/// The ~600 B inclusion witness (erratum 175). The standard form omits
/// the 𝔾 path (the verifier tracks the shard); the cold form carries it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InclusionWitness {
    pub txid: TxId,
    pub leg: LegIndex,
    pub leaf_index: u64,
    pub siblings: Vec<Hash256>,
    pub shard: ShardId,
    pub height: u64,
    pub interval: u64,
    pub header_hash: Hash256,
    pub g_path: Option<GPath>,
}


impl InclusionWitness {
    pub fn serialized_len(&self) -> usize {
        let base = 32 + 1 + 8 + 4 + 32 * self.siblings.len() + 3 + 8 + 8 + 32 + 1;
        base + self.g_path.as_ref().map_or(0, |g| g.encoded_len())
    }


    pub fn is_cold(&self) -> bool {
        self.g_path.is_some()
    }


    /// Build the cold form by attaching the 𝔾 path.
    pub fn with_g_path(mut self, g: GPath) -> InclusionWitness {
        self.g_path = Some(g);
        self
    }
}


impl Encode for InclusionWitness {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.txid.encode_into(out);
        out.push(self.leg.as_u8());
        out.extend_from_slice(&self.leaf_index.to_le_bytes());
        out.extend_from_slice(&(self.siblings.len() as u32).to_le_bytes());
        for s in &self.siblings {
            out.extend_from_slice(s.as_bytes());
        }
        self.shard.encode_into(out);
        out.extend_from_slice(&self.height.to_le_bytes());
        out.extend_from_slice(&self.interval.to_le_bytes());
        out.extend_from_slice(self.header_hash.as_bytes());
        match &self.g_path {
            None => out.push(0),
            Some(g) => {
                out.push(1);
                g.encode_into(out);
            }
        }
    }
    fn encoded_len(&self) -> usize {
        self.serialized_len()
    }
}


impl Decode for InclusionWitness {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let txid = TxId::decode_from(r)?;
        let leg = LegIndex::from_u8(r.read_u8()?);
        let leaf_index = r.read_u64()?;
        let n = r.read_seq_len()?;
        if n > MAX_LEG_SIBLINGS {
            return Err(CodecError::SeqTooLarge { count: n, max: MAX_LEG_SIBLINGS });
        }
        let mut siblings = Vec::with_capacity(n);
        for _ in 0..n {
            siblings.push(Hash256::decode_from(r)?);
        }
        let shard = ShardId::decode_from(r)?;
        let height = r.read_u64()?;
        let interval = r.read_u64()?;
        let header_hash = Hash256::decode_from(r)?;
        let g_path = match r.read_u8()? {
            0 => None,
            1 => Some(GPath::decode_from(r)?),
            tag => return Err(CodecError::InvalidOptionTag { tag }),
        };
        Ok(InclusionWitness {
            txid, leg, leaf_index, siblings, shard, height, interval, header_hash, g_path,
        })
    }
}


/// The verification context: what the verifier supplies beyond the
/// witness itself. The leg-tree root comes from the block data or a
/// trusted full node (erratum 175); the 𝔾 root comes from the anchor.
#[derive(Clone, Debug)]
pub struct VerifyContext<'a> {
    pub leg_tree_root: &'a Hash256,
    /// (g_root, shard_count) — required iff the witness is cold.
    pub g: Option<(&'a Hash256, usize)>,
}


/// Verify a witness: the leg-tree path against the block's root, and
/// (if cold) the header against the 𝔾 root. Returns false on any
/// failure — adversarial input is rejected, never panics.
pub fn verify(w: &InclusionWitness, ctx: &VerifyContext<'_>) -> bool {
    if !nerv_state::verify_leg_witness(
        ctx.leg_tree_root,
        w.leaf_index,
        &w.txid,
        w.leg,
        &w.siblings,
    ) {
        return false;
    }
    if let (Some(gp), Some((g_root, shard_count))) = (&w.g_path, ctx.g) {
        if gp.shard_index >= shard_count {
            return false;
        }
        if !nerv_consensus::verify_g_witness(
            g_root,
            shard_count,
            gp.shard_index,
            &w.header_hash,
            &gp.siblings,
        ) {
            return false;
        }
    } else if w.g_path.is_some() && ctx.g.is_none() {
        return false; // a cold witness requires the 𝔾 context
    }
    true
}


/// The portfolio witness (§11.6): k legs of one block share the locator
/// and header digest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortfolioWitness {
    pub shard: ShardId,
    pub height: u64,
    pub interval: u64,
    pub header_hash: Hash256,
    pub g_path: Option<GPath>,
    pub legs: Vec<LegWitness>,
}


impl PortfolioWitness {
    pub fn serialized_len(&self) -> usize {
        let base = 3 + 8 + 8 + 32 + 1;
        let g = self.g_path.as_ref().map_or(0, |p| p.encoded_len());
        let legs = 4 + self.legs.iter().map(|l| l.encoded_len()).sum::<usize>();
        base + g + legs
    }


    /// Decompose into per-leg inclusion witnesses (sharing the context).
    pub fn to_inclusion_witnesses(&self) -> Vec<InclusionWitness> {
        self.legs
            .iter()
            .map(|l| InclusionWitness {
                txid: l.txid,
                leg: l.leg,
                leaf_index: l.leaf_index,
                siblings: l.siblings.clone(),
                shard: self.shard,
                height: self.height,
                interval: self.interval,
                header_hash: self.header_hash,
                g_path: self.g_path.clone(),
            })
            .collect()
    }
}


impl Encode for PortfolioWitness {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.shard.encode_into(out);
        out.extend_from_slice(&self.height.to_le_bytes());
        out.extend_from_slice(&self.interval.to_le_bytes());
        out.extend_from_slice(self.header_hash.as_bytes());
        match &self.g_path {
            None => out.push(0),
            Some(g) => {
                out.push(1);
                g.encode_into(out);
            }
        }
        out.extend_from_slice(&(self.legs.len() as u32).to_le_bytes());
        for l in &self.legs {
            l.encode_into(out);
        }
    }
    fn encoded_len(&self) -> usize {
        self.serialized_len()
    }
}


impl Decode for PortfolioWitness {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let shard = ShardId::decode_from(r)?;
        let height = r.read_u64()?;
        let interval = r.read_u64()?;
        let header_hash = Hash256::decode_from(r)?;
        let g_path = match r.read_u8()? {
            0 => None,
            1 => Some(GPath::decode_from(r)?),
            tag => return Err(CodecError::InvalidOptionTag { tag }),
        };
        let n = r.read_seq_len()?;
        if n > 65_536 {
            return Err(CodecError::SeqTooLarge { count: n, max: 65_536 });
        }
        let mut legs = Vec::with_capacity(n);
        for _ in 0..n {
            legs.push(LegWitness::decode_from(r)?);
        }
        Ok(PortfolioWitness { shard, height, interval, header_hash, g_path, legs })
    }
}


/// Verify a portfolio: every leg against the block's root.
pub fn verify_portfolio(p: &PortfolioWitness, ctx: &VerifyContext<'_>) -> bool {
    if p.legs.is_empty() {
        return false;
    }
    if let (Some(gp), Some((g_root, shard_count))) = (&p.g_path, ctx.g) {
        if gp.shard_index >= shard_count {
            return false;
        }
        if !nerv_consensus::verify_g_witness(
            g_root, shard_count, gp.shard_index, &p.header_hash, &gp.siblings,
        ) {
            return false;
        }
    } else if p.g_path.is_some() && ctx.g.is_none() {
        return false;
    }
    p.legs.iter().all(|l| {
        nerv_state::verify_leg_witness(ctx.leg_tree_root, l.leaf_index, &l.txid, l.leg, &l.siblings)
    })
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use nerv_core::types::{LegKey, ShardSet};
    use nerv_state::block::BlockLegTree;


    fn shard() -> ShardId {
        ShardSet::genesis().ids()[7]
    }


    fn h(seed: u64) -> Hash256 {
        Hash256::from_bytes([seed as u8; 32])
    }


    fn keys(seed: u64, n: usize) -> Vec<LegKey> {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(LegKey::new(h(seed + i as u64), nerv_core::types::LegIndex::from_u8(0)));
        }
        out.sort();
        out.dedup();
        out
    }


    fn tree_and_witness(seed: u64, n: usize, target: usize) -> (BlockLegTree, InclusionWitness) {
        let keys = keys(seed, n);
        let mut tree = BlockLegTree::new();
        for k in &keys {
            tree.insert(*k).unwrap();
        }
        let key = keys[target];
        let w = tree.witness(target as u64).unwrap();
        let wit = InclusionWitness {
            txid: key.txid,
            leg: key.leg,
            leaf_index: w.index,
            siblings: w.siblings,
            shard: shard(),
            height: 42,
            interval: 86_400,
            header_hash: h(999),
            g_path: None,
        };
        (tree, wit)
    }


    #[test]
    fn verify_standard_and_cold() {
        let (tree, wit) = tree_and_witness(1, 100, 50);
        let root = tree.root();
        let ctx = VerifyContext { leg_tree_root: &root, g: None };
        assert!(verify(&wit, &ctx));


        // Tampered siblings.
        let mut bad = wit.clone();
        if !bad.siblings.is_empty() {
            bad.siblings[0] = h(888);
            assert!(!verify(&bad, &ctx));
        }
        // Wrong root.
        let wrong = VerifyContext { leg_tree_root: &h(777), g: None };
        assert!(!verify(&wit, &wrong));
        // Wrong txid.
        let mut bad = wit.clone();
        bad.txid = h(555);
        assert!(!verify(&bad, &ctx));
        // Out-of-range leaf index.
        let mut bad = wit.clone();
        bad.leaf_index = u64::MAX;
        assert!(!verify(&bad, &ctx));


        // Cold form: the g_path must verify against the g_root.
        let (g_root, shard_count) = (h(100), 64);
        let mut cold = wit.clone();
        let g_siblings = vec![h(201), h(202), h(203), h(204), h(205), h(206)];
        cold.g_path = Some(GPath { shard_index: 7, siblings: g_siblings.clone() });
        // Without the g context: rejected.
        assert!(!verify(&cold, &ctx));
        // With a g context: the path must actually verify (we need a real
        // g tree for that — the path check is against the root).
        let cold_ctx = VerifyContext {
            leg_tree_root: &root,
            g: Some((&g_root, shard_count)),
        };
        // The g_path we fabricated won't verify against g_root (it's not
        // a real path), so this fails:
        assert!(!verify(&cold, &cold_ctx));
    }


    #[test]
    fn cold_g_path_verification_with_a_real_tree() {
        // Build a real 𝔾 tree and verify a cold witness against it.
        let g = nerv_consensus::GTree::new(64);
        // The empty tree's root: all leaves are the sentinel.
        let g_root = g.root();
        let (tree, mut wit) = tree_and_witness(2, 50, 25);
        let root = tree.root();


        // The shard's header hash is a leaf in the 𝔾 tree at index 7.
        // For the empty tree, the leaf is the sentinel. Let's set it.
        let mut g2 = nerv_consensus::GTree::new(64);
        g2.set_tip(7, wit.header_hash);
        let g2_root = g2.root();
        let g_witness = g2.witness(7).unwrap();


        wit.g_path = Some(GPath { shard_index: 7, siblings: g_witness });
        let ctx = VerifyContext { leg_tree_root: &root, g: Some((&g2_root, 64)) };
        assert!(verify(&wit, &ctx));


        // Wrong g root.
        let bad_ctx = VerifyContext { leg_tree_root: &root, g: Some((&g_root, 64)) };
        assert!(!verify(&wit, &bad_ctx));
        // Wrong shard count.
        let bad_ctx = VerifyContext { leg_tree_root: &root, g: Some((&g2_root, 65)) };
        assert!(!verify(&wit, &bad_ctx));
        // Out-of-range shard index.
        let mut bad = wit.clone();
        if let Some(gp) = &mut bad.g_path {
            gp.shard_index = 64;
        }
        assert!(!verify(&bad, &ctx));
    }


    #[test]
    fn serialized_sizes_match_the_wp_budget() {
        let (_, wit) = tree_and_witness(3, 10_000, 5_000);
        // Standard: ~530 B (the WP's figure at the 10,000-leg cap).
        assert!(wit.serialized_len() <= 600, "standard: {}", wit.serialized_len());
        assert!(wit.serialized_len() >= 500, "not too small: {}", wit.serialized_len());


        // Cold: +320 B for the 𝔾 path at 1,024 shards (depth 10).
        let mut cold = wit.clone();
        cold.g_path = Some(GPath {
            shard_index: 7,
            siblings: vec![h(1); 10],
        });
        assert!(cold.serialized_len() <= 900, "cold: {}", cold.serialized_len());
        assert!(cold.serialized_len() >= 800, "cold not too small: {}", cold.serialized_len());
    }


    #[test]
    fn codec_roundtrips() {
        let (_, wit) = tree_and_witness(4, 100, 50);
        let enc = wit.encode();
        assert_eq!(enc.len(), wit.encoded_len());
        assert_eq!(InclusionWitness::decode(&enc).unwrap(), wit);
        assert!(InclusionWitness::decode(&enc[..enc.len() - 1]).is_err());


        let mut cold = wit.clone();
        cold.g_path = Some(GPath { shard_index: 7, siblings: vec![h(1); 6] });
        let enc = cold.encode();
        assert_eq!(InclusionWitness::decode(&enc).unwrap(), cold);
        assert!(InclusionWitness::decode(&enc[..enc.len() - 1]).is_err());
        // Bad g_path tag.
        let mut bad = enc.clone();
        let tag_off = enc.len() - 1 - cold.g_path.as_ref().unwrap().encoded_len();
        bad[tag_off] = 9;
        assert!(InclusionWitness::decode(&bad).is_err());
    }


    #[test]
    fn portfolio_roundtrip_and_verify() {
        let (tree, wit) = tree_and_witness(5, 200, 100);
        let root = tree.root();


        // Build a portfolio of 3 legs from the same block.
        let keys = keys(5, 200);
        let mut legs = Vec::new();
        for &i in &[50, 100, 150] {
            let w = tree.witness(i).unwrap();
            legs.push(LegWitness {
                txid: keys[i].txid,
                leg: keys[i].leg,
                leaf_index: w.index,
                siblings: w.siblings,
            });
        }
        let p = PortfolioWitness {
            shard: wit.shard,
            height: wit.height,
            interval: wit.interval,
            header_hash: wit.header_hash,
            g_path: None,
            legs,
        };
        let ctx = VerifyContext { leg_tree_root: &root, g: None };
        assert!(verify_portfolio(&p, &ctx));
        assert_eq!(p.legs.len(), 3);


        // One bad leg fails the portfolio.
        let mut bad = p.clone();
        bad.legs[1].txid = h(999);
        assert!(!verify_portfolio(&bad, &ctx));


        // Empty portfolio fails.
        let empty = PortfolioWitness {
            shard: wit.shard, height: wit.height, interval: wit.interval,
            header_hash: wit.header_hash, g_path: None, legs: vec![],
        };
        assert!(!verify_portfolio(&empty, &ctx));


        // Roundtrip.
        let enc = p.encode();
        assert_eq!(enc.len(), p.encoded_len());
        assert_eq!(PortfolioWitness::decode(&enc).unwrap(), p);
        assert!(PortfolioWitness::decode(&enc[..enc.len() - 1]).is_err());


        // Decomposition: each inclusion witness verifies individually.
        for iw in p.to_inclusion_witnesses() {
            assert!(verify(&iw, &ctx));
        }


        // The portfolio shares the locator: smaller than 3 separate witnesses.
        let separate: usize = p.to_inclusion_witnesses().iter().map(|w| w.serialized_len()).sum();
        assert!(p.serialized_len() < separate, "portfolio {} vs separate {}", p.serialized_len(), separate);
    }


    #[test]
    fn cross_shard_receipt_is_two_witnesses_bound_by_txid() {
        // A two-leg cross-shard payment: two inclusion witnesses sharing
        // the txid, ~1.1 KB total (§11.6).
        let (tree_a, wit_a) = tree_and_witness(6, 100, 50);
        let (tree_b, wit_b) = tree_and_witness(7, 100, 50);
        // Bind by txid.
        let mut wit_b = wit_b;
        wit_b.txid = wit_a.txid;
        wit_b.shard = ShardSet::genesis().ids()[40];
        let total = wit_a.serialized_len() + wit_b.serialized_len();
        assert!(total <= 1200, "two-leg receipt: {total}");
        assert!(total >= 1000, "not too small: {total}");
        assert_eq!(wit_a.txid, wit_b.txid);
        let _ = (tree_a.root(), tree_b.root());
    }
}
