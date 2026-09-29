//! Leg and shell types with static well-formedness (WP §3.6; D.3). The
//! shell checks are the public-input pre-images of the §5.1 statements;
//! the circuit proves the secret counterparts.

use std::collections::BTreeSet;
use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::params::{
    CUSTODY_D3_EXPIRY_MAX_BLOCKS_T_MAX, CUSTODY_D3_EXPIRY_MIN_BLOCKS_T_MIN,
    CUSTODY_VALUE_MAX_NANO, CUSTODY_VALUE_MIN_NANO,
};
use nerv_core::types::{FeeSats, Height, ShardId, TxId};

use crate::error::CustodyError;

pub const MAX_LEGS: usize = 256;
pub const MAX_INPUTS: usize = 8;
pub const MAX_OUTPUTS: usize = 8;
pub const MAX_CT_BYTES: usize = 12_288;   // genesis seal ct 10,240 + margin
pub const MAX_BURNS_PER_LEG: usize = 8;


/// The nullifiers of input notes homed to this shard (§3.6).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct InputSet {
    pub nullifiers: Vec<Hash256>,
}

impl InputSet {
    pub fn new(nullifiers: Vec<Hash256>) -> InputSet {
        InputSet { nullifiers }
    }

    fn check(&self) -> Result<(), CustodyError> {
        if self.nullifiers.len() > MAX_INPUTS {
            return Err(CustodyError::TooManyInputs { found: self.nullifiers.len(), max: MAX_INPUTS });
        }
        let set: BTreeSet<[u8; 32]> = self.nullifiers.iter().map(|n| *n.as_bytes()).collect();
        if set.len() != self.nullifiers.len() {
            return Err(CustodyError::DuplicateNullifierInLeg);
        }
        Ok(())
    }
}

/// One output: commitment, sealed note blob, declared value, and the D.3
/// conditional-cross-shard marker with its paired revert commitment.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Output {
    pub cm: Hash256,
    pub sealed_note: Vec<u8>,
    pub value: u64,
    pub conditional: bool,
    pub revert_cm: Option<Hash256>,
}

impl Output {
    fn check(&self, index: usize) -> Result<(), CustodyError> {
        if self.value < CUSTODY_VALUE_MIN_NANO || self.value > CUSTODY_VALUE_MAX_NANO {
            return Err(CustodyError::ValueOutOfRange {
                value: self.value,
                min: CUSTODY_VALUE_MIN_NANO,
                max: CUSTODY_VALUE_MAX_NANO,
            });
        }
        if self.conditional && self.revert_cm.is_none() {
            return Err(CustodyError::MissingRevertOutput { index });
        }
        if !self.conditional && self.revert_cm.is_some() {
            return Err(CustodyError::UnexpectedRevertOutput { index });
        }
        Ok(())
    }
}

/// One leg's public shell (§3.6).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LegShell {
    pub shard: ShardId,
    pub inputs: InputSet,
    pub outputs: Vec<Output>,
    pub fee: FeeSats,
    pub anchor: Hash256,
    pub expiry: Height,
    pub weight_version: u64,
    /// The leg's published sealed-delta ciphertext (the (u, v) wire bytes,
    /// erratum 76/E.3: ~10,240 B genesis). In-circuit binding to the
    /// statement-10 relations is the seal chip's (statement 10).
    pub ct: Vec<u8>,
    /// Public burn commitments declared on this leg (§3.8, erratum 76).
    pub burns: Vec<Hash256>,

}

impl LegShell {
    pub fn check(&self) -> Result<(), CustodyError> {
        if self.inputs.nullifiers.is_empty() && self.outputs.is_empty() {
            return Err(CustodyError::EmptyLeg);
        }
        self.inputs.check()?;
        if self.outputs.len() > MAX_OUTPUTS {
            return Err(CustodyError::TooManyOutputs { found: self.outputs.len(), max: MAX_OUTPUTS });
        }
        for (i, o) in self.outputs.iter().enumerate() {
            o.check(i).map_err(|_| CustodyError::InvalidOutput { index: i })?;
        }
         if self.ct.len() > MAX_CT_BYTES {
            return Err(CustodyError::CtTooLarge { len: self.ct.len(), max: MAX_CT_BYTES });
        }
        if self.burns.len() > MAX_BURNS_PER_LEG {
            return Err(CustodyError::TooManyBurns { found: self.burns.len(), max: MAX_BURNS_PER_LEG });
        }

        Ok(())
    }
}

/// The whole transaction's public shell: all legs, canonicalized.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TransactionShell {
    pub legs: Vec<LegShell>,
}

impl TransactionShell {
    /// Canonical form: legs sorted by shard then wire bytes; duplicate
    /// shards rejected (one leg per shard — §3.6).
    pub fn canonicalize(&self) -> Result<TransactionShell, CustodyError> {
        if self.legs.is_empty() {
            return Err(CustodyError::NoLegs);
        }
        if self.legs.len() > MAX_LEGS {
            return Err(CustodyError::TooManyLegs { found: self.legs.len(), max: MAX_LEGS });
        }
        let mut seen = BTreeSet::new();
        let mut legs = self.legs.clone();
        legs.sort_by(|a, b| a.shard.cmp(&b.shard).then_with(|| a.encode().cmp(&b.encode())));
        for l in &legs {
            if !seen.insert(l.shard) {
                return Err(CustodyError::DuplicateShardLeg { shard: l.shard });
            }
            l.check()?;
        }
        Ok(TransactionShell { legs })
    }

    /// txid = BLAKE3("nerv.txid" ‖ canonical serialization of all legs).
    pub fn txid(&self) -> Result<TxId, CustodyError> {
        let c = self.canonicalize()?;
        let mut buf = Vec::new();
        for l in &c.legs {
            l.encode_into(&mut buf);
        }
        Ok(TxId::hash_canonical(&buf))
    }

    /// D.3 relation (b), public side: every conditional output declares its
    /// revert pairing (value equality is proven in-circuit together with
    /// conservation).
    pub fn check_revert_pairing(&self) -> Result<(), CustodyError> {
        let c = self.canonicalize()?;
        for l in &c.legs {
            for (i, o) in l.outputs.iter().enumerate() {
                if o.conditional && o.revert_cm.is_none() {
                    return Err(CustodyError::MissingRevertOutput { index: i });
                }
            }
        }
        Ok(())
    }

    /// Total declared fee (transparent; §4.6).
    pub fn total_fee(&self) -> Result<FeeSats, CustodyError> {
        let c = self.canonicalize()?;
        let mut acc = FeeSats::from_u64(0);
        for l in &c.legs {
            acc = acc.checked_add(l.fee).ok_or(CustodyError::FeeOverflow)?;
        }
        Ok(acc)
    }

    /// D.3 expiry ∈ [h_include + T_min, h_include + T_max] per leg.
    pub fn check_expiry_bounds(&self, include_height: Height) -> Result<(), CustodyError> {
        let c = self.canonicalize()?;
        let lo = include_height.as_u64().saturating_add(CUSTODY_D3_EXPIRY_MIN_BLOCKS_T_MIN as u64);
        let hi = include_height.as_u64().saturating_add(CUSTODY_D3_EXPIRY_MAX_BLOCKS_T_MAX as u64);
        for l in &c.legs {
            let e = l.expiry.as_u64();
            if e < lo || e > hi {
                return Err(CustodyError::ExpiryOutOfBounds {
                    expiry: l.expiry,
                    include: include_height,
                    lo,
                    hi,
                });
            }
        }
        Ok(())
    }
}

impl Encode for InputSet {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.nullifiers.len() as u32).to_le_bytes());
        for n in &self.nullifiers {
            out.extend_from_slice(n.as_bytes());
        }
    }
    fn encoded_len(&self) -> usize {
        4 + self.nullifiers.len() * 32
    }
}

impl Decode for InputSet {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let n = r.read_seq_len()?;
        r.enter()?;
        let mut nullifiers = Vec::with_capacity(n.min(MAX_INPUTS * 2));
        for _ in 0..n {
            nullifiers.push(Hash256::decode_from(r)?);
        }
        r.leave();
        Ok(InputSet { nullifiers })
    }
}

impl Encode for Output {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self.cm.as_bytes());
        out.extend_from_slice(&(self.sealed_note.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.sealed_note);
        out.extend_from_slice(&self.value.to_le_bytes());
        out.push(u8::from(self.conditional));
        match &self.revert_cm {
            None => out.push(0),
            Some(r) => {
                out.push(1);
                out.extend_from_slice(r.as_bytes());
            }
        }
    }
    fn encoded_len(&self) -> usize {
        32 + 4 + self.sealed_note.len() + 8 + 1 + 1 + self.revert_cm.map_or(0, |_| 32)
    }
}

impl Decode for Output {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let cm = Hash256::decode_from(r)?;
        let n = r.read_seq_len()?;
        r.enter()?;
        let sealed_note = r.take(n)?.to_vec();
        r.leave();
        let value = r.read_u64()?;
        let conditional = match r.read_u8()? {
            0 => false,
            1 => true,
            tag => return Err(CodecError::InvalidOptionTag { tag }),
        };
        let revert_cm = match r.read_u8()? {
            0 => None,
            1 => Some(Hash256::decode_from(r)?),
            tag => return Err(CodecError::InvalidOptionTag { tag }),
        };
        Ok(Output { cm, sealed_note, value, conditional, revert_cm })
    }
}

impl Encode for LegShell {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.shard.encode_into(out);
        self.inputs.encode_into(out);
        out.extend_from_slice(&(self.outputs.len() as u32).to_le_bytes());
        for o in &self.outputs {
            o.encode_into(out);
        }
        self.fee.encode_into(out);
        self.anchor.encode_into(out);
        self.expiry.encode_into(out);
        out.extend_from_slice(&self.weight_version.to_le_bytes());
        // ct and burns are part of the leg's public wire (D.3, E.3).
        out.extend_from_slice(&(self.ct.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.ct);
        out.extend_from_slice(&(self.burns.len() as u32).to_le_bytes());
        for b in &self.burns {
            b.encode_into(out);
        }
    }
    fn encoded_len(&self) -> usize {
        3 + self.inputs.encoded_len() + 4 + self.outputs.iter().map(|o| o.encoded_len()).sum::<usize>()
            + 8 + 32 + 8 + 8 + 4 + self.ct.len() + 4
            + self.burns.iter().map(|b| b.encoded_len()).sum::<usize>()
    }
}

impl Decode for LegShell {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let shard = ShardId::decode_from(r)?;
        let inputs = InputSet::decode_from(r)?;
        let n = r.read_seq_len()?;
        r.enter()?;
        let mut outputs = Vec::with_capacity(n.min(MAX_OUTPUTS * 2));
        for _ in 0..n {
            outputs.push(Output::decode_from(r)?);
        }
        r.leave();
        let fee = FeeSats::decode_from(r)?;
        let anchor = Hash256::decode_from(r)?;
        let expiry = Height::decode_from(r)?;
        let weight_version = r.read_u64()?;
        // ct and burns round-trip symmetrically with the Encode side above.
        let ct_len = r.read_u32()? as usize;
        if ct_len > MAX_CT_BYTES {
            return Err(CodecError::InvariantViolated("LegShell.ct exceeds MAX_CT_BYTES"));
        }
        let mut ct = vec![0u8; ct_len];
        if ct_len > 0 {
            ct.copy_from_slice(r.take(ct_len)?);
        }
        let b_len = r.read_u32()? as usize;
        if b_len > MAX_BURNS_PER_LEG {
            return Err(CodecError::SeqTooLarge { count: b_len, max: MAX_BURNS_PER_LEG });
        }
        let mut burns = Vec::with_capacity(b_len);
        for _ in 0..b_len {
            burns.push(Hash256::decode_from(r)?);
        }
        Ok(LegShell { shard, inputs, outputs, fee, anchor, expiry, weight_version, ct, burns })
    }
}

impl Encode for TransactionShell {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.legs.len() as u32).to_le_bytes());
        for l in &self.legs {
            l.encode_into(out);
        }
    }
    fn encoded_len(&self) -> usize {
        4 + self.legs.iter().map(|l| l.encoded_len()).sum::<usize>()
    }
}

impl Decode for TransactionShell {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let n = r.read_seq_len()?;
        r.enter()?;
        let mut legs = Vec::with_capacity(n.min(MAX_LEGS * 2));
        for _ in 0..n {
            legs.push(LegShell::decode_from(r)?);
        }
        r.leave();
        Ok(TransactionShell { legs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;
    use nerv_core::types::ShardSet;

    fn rng() -> SplitMix64 {
        SplitMix64::new(0x7EE5)
    }

    fn leg(seed: &mut SplitMix64, shard: ShardId, n_in: usize, n_out: usize) -> LegShell {
        LegShell {
            shard,
            inputs: InputSet::new((0..n_in).map(|_| Hash256::from_bytes(seed.bytes32())).collect()),
            outputs: (0..n_out)
                .map(|_| Output {
                    cm: Hash256::from_bytes(seed.bytes32()),
                    sealed_note: vec![0xA5; 64],
                    value: 100 + seed.next_u64() % 1000,
                    conditional: false,
                    revert_cm: None,
                })
                .collect(),
            fee: FeeSats::from_u64(1000 + seed.next_u64() % 1000),
            anchor: Hash256::from_bytes(seed.bytes32()),
            expiry: Height::from_u64(5000),
            weight_version: 1,
        }
    }

    #[test]
    fn canonicalize_orders_and_validates() {
        let mut seed = rng();
        let g = ShardSet::genesis();
        let mut tx = TransactionShell {
            legs: vec![
                leg(&mut seed, g.ids()[9], 2, 2),
                leg(&mut seed, g.ids()[2], 1, 1),
                leg(&mut seed, g.ids()[30], 3, 1),
            ],
        };
        let c = tx.canonicalize().unwrap();
        let shards: Vec<ShardId> = c.legs.iter().map(|l| l.shard).collect();
        assert_eq!(shards, vec![g.ids()[2], g.ids()[9], g.ids()[30]]);
        assert_eq!(c.legs.len(), 3);
        let tx2 = tx.clone();
        tx.legs.reverse();
        assert_eq!(tx.canonicalize().unwrap(), c, "canonicalization is order-invariant");
        assert_eq!(tx2.txid().unwrap(), tx.txid().unwrap());
    }

    #[test]
    fn duplicate_shard_rejected() {
        let mut seed = rng();
        let g = ShardSet::genesis();
        let tx = TransactionShell {
            legs: vec![leg(&mut seed, g.ids()[5], 1, 1), leg(&mut seed, g.ids()[5], 1, 0)],
        };
        assert!(matches!(
            tx.canonicalize(),
            Err(CustodyError::DuplicateShardLeg { .. })
        ));
    }

    #[test]
    fn leg_limits_and_value_ranges() {
        let mut seed = rng();
        let g = ShardSet::genesis();
        let mut l = leg(&mut seed, g.ids()[0], MAX_INPUTS, 1);
        assert!(l.check().is_ok());
        l.inputs.nullifiers.push(Hash256::from_bytes(seed.bytes32()));
        assert!(matches!(l.check(), Err(CustodyError::TooManyInputs { .. })));
        let mut l2 = leg(&mut seed, g.ids()[1], 1, MAX_OUTPUTS);
        assert!(l2.check().is_ok());
        l2.outputs.push(Output {
            cm: Hash256::from_bytes(seed.bytes32()),
            sealed_note: vec![0; 64],
            value: 5,
            conditional: false,
            revert_cm: None,
        });
        assert!(matches!(l2.check(), Err(CustodyError::TooManyOutputs { .. })));
        let mut l3 = leg(&mut seed, g.ids()[2], 1, 1);
        l3.outputs[0].value = 0;
        assert!(matches!(l3.check(), Err(CustodyError::InvalidOutput { index: 0 })));
        let mut l4 = leg(&mut seed, g.ids()[3], 1, 1);
        l4.outputs[0].value = CUSTODY_VALUE_MAX_NANO + 1;
        assert!(matches!(l4.check(), Err(CustodyError::InvalidOutput { index: 0 })));
        let mut empty = leg(&mut seed, g.ids()[4], 0, 0);
        assert!(matches!(empty.check(), Err(CustodyError::EmptyLeg)));
        // duplicate nullifier within a leg
        let mut l5 = leg(&mut seed, g.ids()[5], 2, 0);
        l5.inputs.nullifiers[1] = l5.inputs.nullifiers[0];
        assert!(matches!(l5.check(), Err(CustodyError::DuplicateNullifierInLeg)));
    }

    #[test]
    fn revert_pairing_rules() {
        let mut seed = rng();
        let g = ShardSet::genesis();
        let mut l = leg(&mut seed, g.ids()[6], 2, 2);
        l.outputs[1].conditional = true;
        assert!(matches!(l.check(), Err(CustodyError::InvalidOutput { index: 1 })));
        l.outputs[1].revert_cm = Some(Hash256::from_bytes(seed.bytes32()));
        assert!(l.check().is_ok());
        l.outputs[0].revert_cm = Some(Hash256::from_bytes(seed.bytes32()));
        assert!(matches!(l.check(), Err(CustodyError::InvalidOutput { index: 0 })));
    }

    #[test]
    fn expiry_bounds_d3() {
        let mut seed = rng();
        let g = ShardSet::genesis();
        let tx = TransactionShell { legs: vec![leg(&mut seed, g.ids()[7], 1, 1)] };
        // genesis T_min = 60, T_max = 14400 (blocks; D.6: 60s/24h/10s at
        // 1-second target blocks)
        assert!(tx.check_expiry_bounds(Height::from_u64(1000)).is_err(), "5000 < 1000+60");
        let mut ok_tx = TransactionShell { legs: vec![leg(&mut seed, g.ids()[8], 1, 1)] };
        ok_tx.legs[0].expiry = Height::from_u64(1000 + D3_EXPIRY_MIN_BLOCKS_T_MIN as u64);
        assert!(ok_tx.check_expiry_bounds(Height::from_u64(1000)).is_ok());
        ok_tx.legs[0].expiry = Height::from_u64(1000 + D3_EXPIRY_MAX_BLOCKS_T_MAX as u64);
        assert!(ok_tx.check_expiry_bounds(Height::from_u64(1000)).is_ok());
        ok_tx.legs[0].expiry = Height::from_u64(1000 + D3_EXPIRY_MAX_BLOCKS_T_MAX as u64 + 1);
        assert!(matches!(
            ok_tx.check_expiry_bounds(Height::from_u64(1000)),
            Err(CustodyError::ExpiryOutOfBounds { .. })
        ));
        ok_tx.legs[0].expiry = Height::from_u64(1000 + D3_EXPIRY_MIN_BLOCKS_T_MIN as u64 - 1);
        assert!(matches!(
            ok_tx.check_expiry_bounds(Height::from_u64(1000)),
            Err(CustodyError::ExpiryOutOfBounds { .. })
        ));
    }

    #[test]
    fn total_fee_sums() {
        let mut seed = rng();
        let g = ShardSet::genesis();
        let mut tx = TransactionShell {
            legs: vec![leg(&mut seed, g.ids()[10], 1, 1), leg(&mut seed, g.ids()[11], 1, 1)],
        };
        let f1 = tx.legs[0].fee;
        let f2 = tx.legs[1].fee;
        assert_eq!(tx.total_fee().unwrap().as_u64(), f1.as_u64() + f2.as_u64());
        tx.legs[0].fee = FeeSats::from_u64(u64::MAX);
        tx.legs[1].fee = FeeSats::from_u64(1);
        assert!(matches!(tx.total_fee(), Err(CustodyError::FeeOverflow)));
    }

    #[test]
    fn codec_roundtrips() {
        let mut seed = rng();
        let g = ShardSet::genesis();
        let tx = TransactionShell {
            legs: vec![leg(&mut seed, g.ids()[12], 2, 2), leg(&mut seed, g.ids()[13], 1, 1)],
        };
        let c = tx.canonicalize().unwrap();
        let enc = c.encode();
        assert_eq!(enc.len(), c.encoded_len());
        let d = TransactionShell::decode(&enc).unwrap();
        assert_eq!(d, c);
        assert_eq!(d.txid().unwrap(), c.txid().unwrap());
        assert!(TransactionShell::decode(&enc[..enc.len() - 1]).is_err());
        let mut ext = enc.clone();
        ext.push(0);
        assert!(TransactionShell::decode(&ext).is_err());
        // conditional output roundtrip
        let mut cond = leg(&mut seed, g.ids()[14], 1, 1);
        cond.outputs[0].conditional = true;
        cond.outputs[0].revert_cm = Some(Hash256::from_bytes(seed.bytes32()));
        let cond_enc = cond.encode();
        let cond_d = LegShell::decode(&cond_enc).unwrap();
        assert_eq!(cond_d, cond);
        assert!(cond_d.outputs[0].conditional);
        assert!(cond_d.outputs[0].revert_cm.is_some());
    }

    #[test]
    fn txid_deterministic_and_sensitive() {
        let mut seed = rng();
        let g = ShardSet::genesis();
        let tx = TransactionShell { legs: vec![leg(&mut seed, g.ids()[15], 1, 1)] };
        let id1 = tx.txid().unwrap();
        assert_eq!(id1, tx.txid().unwrap());
        let mut other = tx.clone();
        other.legs[0].fee = FeeSats::from_u64(other.legs[0].fee.as_u64() + 1);
        assert_ne!(id1, other.txid().unwrap());
        let mut shuffled = tx.clone();
        let mut t2 = TransactionShell { legs: shuffled.legs.clone() };
        t2.legs.reverse();
        assert_eq!(id1, t2.txid().unwrap(), "leg order does not change the canonical txid");
    }
}
