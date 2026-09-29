//! The shard header (WP §4.3; erratum 104). C_t's six fields plus the
//! header-only commitments; `header_hash` is the QC subject and the 𝔾_t
//! leaf. D_t and prev_reveal are opaque bytes through this crate (DSR-4).

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::HEADER;
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::types::{FeeSats, Height, Interval};
use nerv_custody::{Address, NctDigest};

use crate::commitment::state_commitment;
use crate::ttau;

/// The previous block's revealed Δ_B: 64 coordinates × 8 bytes, verbatim.
pub const REVEAL_BYTES: usize = 512;

/// The registry reference: the finalized interval whose T_τ root this block
/// builds against (rule 1) and the anchor of the block's build position.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RegistryRef {
    pub interval: Interval,
    pub root: Hash256,
}

impl RegistryRef {
    /// (interval 0, the empty T_τ root) — finalized by definition; no
    /// txid can exhibit membership in the empty tree (erratum 102).
    pub fn genesis() -> RegistryRef {
        RegistryRef { interval: Interval::from_u64(0), root: ttau::empty_root() }
    }
}

impl Encode for RegistryRef {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.interval.encode_into(out);
        out.extend_from_slice(self.root.as_bytes());
    }
    fn encoded_len(&self) -> usize {
        40
    }
}

impl Decode for RegistryRef {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let interval = Interval::decode_from(r)?;
        let root = Hash256::decode_from(r)?;
        Ok(RegistryRef { interval, root })
    }
}

/// One shard block's header. Field order is frozen; the canonical encoding
/// is the header hash's preimage.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ShardHeader {
    /// The predecessor's C_t (C₀ for height 1 — there is no genesis header).
    pub prev: Hash256,
    pub height: Height,
    pub nct_root: NctDigest,
    pub nullifier_root: Hash256,
    pub transit_root: Hash256,
    pub params_root: Hash256,
    /// D_t — 32 opaque bytes (DSR-4).
    pub derived: [u8; 32],
    /// H(ct_B) over the block's summed sealed-delta ciphertext.
    pub ct_batch_hash: Hash256,
    /// The previous block's revealed Δ_B; `None` records a missed reveal
    /// ceremony (D.1(d) — the knowledge layer's skip-and-carry rule).
    pub prev_reveal: Option<[u8; REVEAL_BYTES]>,
    pub registry: RegistryRef,
    pub fee_total: FeeSats,
    pub producer_payout: Address,
    pub qc_hash: Hash256,
}

impl ShardHeader {
    /// C_t over the six S_t fields (§4.2).
    pub fn state_commitment(&self) -> Hash256 {
        state_commitment(
            &self.nct_root,
            &self.nullifier_root,
            &self.transit_root,
            &self.params_root,
            &self.prev,
            self.height,
        )
    }

    /// BLAKE3("nerv.hdr" ‖ canonical encoding) — the 𝔾_t leaf, the
   /// chaining identity, and the fraud digest preimage.
   pub fn header_hash(&self) -> Hash256 {
       Hash256::concat(&HEADER, &self.encode())
   }


   /// The QC's subject (erratum 131): the canonical encoding with
   /// qc_hash zeroed — the certificate cannot sign its own compression.
   pub fn signing_hash(&self) -> Hash256 {
       let zeroed =
           ShardHeader { qc_hash: Hash256::from_bytes([0u8; 32]), ..self.clone() };
       Hash256::concat(&HEADER, &zeroed.encode())
   }

}

impl Encode for ShardHeader {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self.prev.as_bytes());
        out.extend_from_slice(&self.height.as_u64().to_le_bytes());
        out.extend_from_slice(self.nct_root.as_bytes());
        out.extend_from_slice(self.nullifier_root.as_bytes());
        out.extend_from_slice(self.transit_root.as_bytes());
        out.extend_from_slice(self.params_root.as_bytes());
        out.extend_from_slice(&self.derived);
        out.extend_from_slice(self.ct_batch_hash.as_bytes());
        match &self.prev_reveal {
            None => out.push(0),
            Some(r) => {
                out.push(1);
                out.extend_from_slice(r);
            }
        }
        self.registry.encode_into(out);
        out.extend_from_slice(&self.fee_total.as_u64().to_le_bytes());
        self.producer_payout.encode_into(out);
        out.extend_from_slice(self.qc_hash.as_bytes());
    }
  fn encoded_len(&self) -> usize {
        8 * 32 // prev, nct, nullifier, transit, params, derived, ct_batch, qc
            + 8 // height
            + 1
            + REVEAL_BYTES * usize::from(self.prev_reveal.is_some())
            + 40 // registry: interval ‖ root
            + 8 // fee_total
            + self.producer_payout.encoded_len()
    }

}

impl Decode for ShardHeader {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let prev = Hash256::decode_from(r)?;
        let height = Height::decode_from(r)?;
        let nct_root = NctDigest::decode_from(r)?;
        let nullifier_root = Hash256::decode_from(r)?;
        let transit_root = Hash256::decode_from(r)?;
        let params_root = Hash256::decode_from(r)?;
        let derived = r.take_array::<32>()?;
        let ct_batch_hash = Hash256::decode_from(r)?;
        let prev_reveal = match r.read_u8()? {
            0 => None,
            1 => Some(r.take_array::<REVEAL_BYTES>()?),
            tag => return Err(CodecError::InvalidOptionTag { tag }),
        };
        let registry = RegistryRef::decode_from(r)?;
        let fee_total = FeeSats::decode_from(r)?;
        let producer_payout = Address::decode_from(r)?;
        let qc_hash = Hash256::decode_from(r)?;
        Ok(ShardHeader {
            prev,
            height,
            nct_root,
            nullifier_root,
            transit_root,
            params_root,
            derived,
            ct_batch_hash,
            prev_reveal,
            registry,
            fee_total,
            producer_payout,
            qc_hash,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;
    use nerv_core::field::Goldilocks;
    use nerv_core::types::ShardSet;
    use nerv_custody::{MasterSeed, WalletKeys, DELIVERY_KEY_LEN};

    fn payout(seed: u64) -> Address {
        let mut rng = SplitMix64::new(seed);
        let wk = WalletKeys::from_master(&MasterSeed::from_bytes(rng.bytes32()));
        Address::generate(wk.viewing(), wk.nullifier_key(), 0, &ShardSet::genesis()).unwrap()
    }

    fn digest(k: u32) -> NctDigest {
        NctDigest::from_elements(&[
            Goldilocks::from_u32(k),
            Goldilocks::from_u32(k.wrapping_add(1)),
            Goldilocks::from_u32(k.wrapping_mul(3)),
            Goldilocks::from_u32(k.wrapping_mul(7)),
        ])
    }

    fn header(seed: u64, revealed: bool) -> ShardHeader {
        let mut rng = SplitMix64::new(seed);
        ShardHeader {
            prev: Hash256::from_bytes(rng.bytes32()),
            height: Height::from_u64(1_000),
            nct_root: digest(7),
            nullifier_root: Hash256::from_bytes(rng.bytes32()),
            transit_root: Hash256::from_bytes(rng.bytes32()),
            params_root: Hash256::from_bytes(rng.bytes32()),
            derived: rng.bytes32(),
            ct_batch_hash: Hash256::from_bytes(rng.bytes32()),
            prev_reveal: revealed.then(|| {
                let mut r = [0u8; REVEAL_BYTES];
                for chunk in r.chunks_exact_mut(8) {
                    chunk.copy_from_slice(&rng.next_u64().to_le_bytes());
                }
                r
            }),
            registry: RegistryRef {
                interval: Interval::from_u64(86_400),
                root: Hash256::from_bytes(rng.bytes32()),
            },
            fee_total: FeeSats::from_u64(1_234_567),
            producer_payout: payout(seed),
            qc_hash: Hash256::from_bytes(rng.bytes32()),
        }
    }

    #[test]
    fn c_t_is_exactly_the_six_fields() {
        let h = header(0x5D01, true);
        let c = h.state_commitment();
        assert_eq!(
            c,
            state_commitment(
                &h.nct_root,
                &h.nullifier_root,
                &h.transit_root,
                &h.params_root,
                &h.prev,
                h.height,
            )
        );

        // Header-only commitments: C_t untouched, header hash moved.
        let mut m = h.clone();
        m.derived = [9u8; 32];
        assert_eq!(m.state_commitment(), c);
        assert_ne!(m.header_hash(), h.header_hash());

        let mut m = h.clone();
        m.ct_batch_hash = Hash256::from_bytes([9u8; 32]);
        assert_eq!(m.state_commitment(), c);
        assert_ne!(m.header_hash(), h.header_hash());

        let mut m = h.clone();
        m.prev_reveal = None;
        assert_eq!(m.state_commitment(), c);
        assert_ne!(m.header_hash(), h.header_hash());

        let mut m = h.clone();
        m.prev_reveal = Some([0u8; REVEAL_BYTES]);
        assert_eq!(m.state_commitment(), c);
        assert_ne!(m.header_hash(), h.header_hash());

        let mut m = h.clone();
        m.registry = RegistryRef::genesis();
        assert_eq!(m.state_commitment(), c);
        assert_ne!(m.header_hash(), h.header_hash());

        let mut m = h.clone();
        m.fee_total = FeeSats::from_u64(h.fee_total.as_u64() + 1);
        assert_eq!(m.state_commitment(), c);
        assert_ne!(m.header_hash(), h.header_hash());

        let mut m = h.clone();
        m.qc_hash = Hash256::from_bytes([9u8; 32]);
        assert_eq!(m.state_commitment(), c);
        assert_ne!(m.header_hash(), h.header_hash());

        let mut m = h.clone();
        m.producer_payout = payout(0x5D02);
        assert_eq!(m.state_commitment(), c);
        assert_ne!(m.header_hash(), h.header_hash());

        // Every S_t field moves both C_t and the header hash.
        let mutators: [fn(&mut ShardHeader); 6] = [
            |m| m.prev = Hash256::from_bytes([9u8; 32]),
            |m| m.height = Height::from_u64(m.height.as_u64() + 1),
            |m| m.nct_root = digest(8),
            |m| m.nullifier_root = Hash256::from_bytes([9u8; 32]),
            |m| m.transit_root = Hash256::from_bytes([9u8; 32]),
            |m| m.params_root = Hash256::from_bytes([9u8; 32]),
        ];
        for mutate in mutators {
            let mut m = h.clone();
            mutate(&mut m);
            assert_ne!(m.state_commitment(), c);
            assert_ne!(m.header_hash(), h.header_hash());
        }
    }

    #[test]
    fn header_hash_is_the_literal_formula() {
        for revealed in [false, true] {
            let h = header(0x5D03, revealed);
            let mut pre = Vec::with_capacity(HEADER.as_bytes().len() + h.encoded_len());
            pre.extend_from_slice(HEADER.as_bytes());
            pre.extend_from_slice(&h.encode());
            assert_eq!(h.header_hash().as_bytes(), blake3::hash(&pre).as_bytes());
            assert_eq!(h.header_hash(), h.header_hash());
        }
    }

    fn roundtrip(h: &ShardHeader) {
        let enc = h.encode();
        assert_eq!(enc.len(), h.encoded_len());
        assert_eq!(&ShardHeader::decode(&enc).unwrap(), h);
        for cut in 0..enc.len() {
            assert!(ShardHeader::decode(&enc[..cut]).is_err(), "cut {cut}");
        }
        let mut ext = enc.clone();
        ext.push(0);
        assert!(ShardHeader::decode(&ext).is_err());
    }

    #[test]
    fn codec_roundtrips_both_reveal_forms() {
        roundtrip(&header(0x5D04, false));
        roundtrip(&header(0x5D05, true));
    }

    #[test]
    fn wire_size_pins() {
        assert_eq!(header(0x5D06, false).encode().len(), 1532);
        assert_eq!(header(0x5D07, true).encode().len(), 1532 + REVEAL_BYTES);
    }

    #[test]
    fn decode_validates_reveal_tag_and_payout_address() {
        let h = header(0x5D08, false);
        let enc = h.encode();

        // The reveal tag follows the eight 32-byte fields and the height.
        let mut bad = enc.clone();
        bad[8 * 32 + 8] = 2;
        assert!(matches!(
            ShardHeader::decode(&bad),
            Err(CodecError::InvalidOptionTag { tag: 2 })
        ));

        // A forged payout shard tag (the homing sibling — never also a
        // kappa prefix) fails Address decode inside the header decode.
        let payout_enc = h.producer_payout.encode();
        let at = enc.len() - 32 - payout_enc.len();
        let mut forged = payout_enc.clone();
        forged[DELIVERY_KEY_LEN + 1] ^= 1;
        let mut bad = enc.clone();
        bad[at..at + payout_enc.len()].copy_from_slice(&forged);
        assert!(matches!(
            ShardHeader::decode(&bad),
            Err(CodecError::InvariantViolated(_))
        ));

        // Control: re-splicing the honest payout decodes to the same header.
        let mut good = enc.clone();
        good[at..at + payout_enc.len()].copy_from_slice(&payout_enc);
        assert_eq!(ShardHeader::decode(&good).unwrap(), h);
    }

    #[test]
    fn registry_ref_genesis_and_roundtrip() {
        let g = RegistryRef::genesis();
        assert_eq!(g.interval, Interval::from_u64(0));
        assert_eq!(g.root, ttau::empty_root());
        let enc = g.encode();
        assert_eq!(enc.len(), 40);
        assert_eq!(g.encoded_len(), 40);
        assert_eq!(RegistryRef::decode(&enc).unwrap(), g);
        assert!(RegistryRef::decode(&enc[..39]).is_err());
        let mut ext = enc.clone();
        ext.push(0);
        assert!(RegistryRef::decode(&ext).is_err());

        let other =
            RegistryRef { interval: Interval::from_u64(5), root: Hash256::from_bytes([1u8; 32]) };
        assert_eq!(RegistryRef::decode(&other.encode()).unwrap(), other);
        assert_ne!(g.encode(), other.encode());
    }
}
