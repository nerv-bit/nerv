//! Public burn commitments (WP §3.8): a shielded→transparent exit spends a
//! note with no output note and publishes burn = BLAKE3("nerv.burn" ‖ txid ‖
//! leg ‖ value). Anyone recomputes it from public leg data; the
//! whole-transaction STARK (chunk 10) proves the burned value entered the
//! conservation equation. Burns feed the supply identity
//! emitted − burned − abandoned (§12.3, M1).

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::BURN;
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::params::{CUSTODY_VALUE_MAX_NANO, CUSTODY_VALUE_MIN_NANO};
use nerv_core::types::{LegIndex, TxId};

use crate::error::CustodyError;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BurnCommitment(Hash256);

fn burn_message(txid: &TxId, leg: LegIndex, value: u64) -> [u8; 41] {
    let mut msg = [0u8; 41];
    msg[..32].copy_from_slice(txid.as_bytes());
    msg[32] = leg.as_u8();
    msg[33..].copy_from_slice(&value.to_le_bytes());
    msg
}

impl BurnCommitment {
    pub fn new(txid: &TxId, leg: LegIndex, value: u64) -> Result<BurnCommitment, CustodyError> {
        if value < CUSTODY_VALUE_MIN_NANO || value > CUSTODY_VALUE_MAX_NANO {
            return Err(CustodyError::ValueOutOfRange {
                value,
                min: CUSTODY_VALUE_MIN_NANO,
                max: CUSTODY_VALUE_MAX_NANO,
            });
        }
        Ok(BurnCommitment(Hash256::concat(&BURN, &burn_message(txid, leg, value))))
    }

    pub const fn as_hash(&self) -> &Hash256 {
        &self.0
    }

    /// Out-of-range values never verify (defense in depth: the executor
    /// range-checks independently).
    pub fn verify(&self, txid: &TxId, leg: LegIndex, value: u64) -> bool {
        (value >= CUSTODY_VALUE_MIN_NANO && value <= CUSTODY_VALUE_MAX_NANO)
            && self.0 == Hash256::concat(&BURN, &burn_message(txid, leg, value))
    }
}

impl Encode for BurnCommitment {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self.0.as_bytes());
    }
    fn encoded_len(&self) -> usize {
        32
    }
}

impl Decode for BurnCommitment {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(BurnCommitment(Hash256::decode_from(r)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;

    fn txid(rng: &mut SplitMix64) -> TxId {
        TxId::from_hash(Hash256::from_bytes(rng.bytes32()))
    }

    #[test]
    fn commitment_is_the_wp_shape() {
        let mut rng = SplitMix64::new(0xB04);
        let t = txid(&mut rng);
        let leg = LegIndex::from_u8(3);
        let v = 1_500_000_000u64;
        let b = BurnCommitment::new(&t, leg, v).unwrap();
        let mut msg = [0u8; 41];
        msg[..32].copy_from_slice(t.as_bytes());
        msg[32] = 3;
        msg[33..].copy_from_slice(&v.to_le_bytes());
        assert_eq!(*b.as_hash(), Hash256::concat(&BURN, &msg));
    }

    #[test]
    fn verify_binds_all_fields() {
        let mut rng = SplitMix64::new(0xB05);
        let t = txid(&mut rng);
        let leg = LegIndex::from_u8(1);
        let v = 999u64;
        let b = BurnCommitment::new(&t, leg, v).unwrap();
        assert!(b.verify(&t, leg, v));
        assert!(!b.verify(&txid(&mut rng), leg, v));
        assert!(!b.verify(&t, LegIndex::from_u8(2), v));
        assert!(!b.verify(&t, leg, v + 1));
        assert!(BurnCommitment::new(&t, leg, v).unwrap() == b);
    }

    #[test]
    fn value_range_enforced() {
        let mut rng = SplitMix64::new(0xB06);
        let t = txid(&mut rng);
        let leg = LegIndex::from_u8(0);
        assert!(matches!(
            BurnCommitment::new(&t, leg, 0),
            Err(CustodyError::ValueOutOfRange { value: 0, .. })
        ));
        assert!(matches!(
            BurnCommitment::new(&t, leg, CUSTODY_VALUE_MAX_NANO + 1),
            Err(CustodyError::ValueOutOfRange { .. })
        ));
        let max = BurnCommitment::new(&t, leg, CUSTODY_VALUE_MAX_NANO).unwrap();
        assert!(max.verify(&t, leg, CUSTODY_VALUE_MAX_NANO));
        let handcrafted = BurnCommitment::from_bytes(*Hash256::concat(
            &BURN,
            &burn_message(&t, leg, 0),
        ).as_bytes());
        assert!(!handcrafted.verify(&t, leg, 0), "out-of-range value never verifies");
    }

    #[test]
    fn codec_roundtrip() {
        let mut rng = SplitMix64::new(0xB07);
        let b = BurnCommitment::new(&txid(&mut rng), LegIndex::from_u8(9), 42).unwrap();
        let enc = b.encode();
        assert_eq!(enc.len(), 32);
        assert_eq!(BurnCommitment::decode(&enc).unwrap(), b);
        assert!(BurnCommitment::decode(&enc[..31]).is_err());
        let mut ext = enc.clone();
        ext.push(0);
        assert!(BurnCommitment::decode(&ext).is_err());
    }
}

