//! Note commitments (WP §3.3 as amended by erratum 72):
//! cm = BLAKE3("nerv.cm" ‖ v ‖ ρ ‖ d ‖ r ‖ pk_n) — the per-address
//! nullifier-key commitment pk_n = H("nerv.nf.pk" ‖ nk_j) is the final
//! field; it is what pins the nullifier key (and therefore the nullifier)
//! to the note.

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::{NOTE_COMMITMENT, NULLIFIER_PK};
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::params::{CUSTODY_VALUE_MAX_NANO, CUSTODY_VALUE_MIN_NANO};

use crate::error::CustodyError;
use nerv_crypto::mlkem::EK_LEN;

pub const DELIVERY_KEY_LEN: usize = EK_LEN;
pub const NONCE_LEN: usize = 32;
pub const BLINDING_LEN: usize = 32;
pub const PK_N_LEN: usize = 32;

pub fn nullifier_pk(nk: &[u8; 32]) -> [u8; 32] {
    *Hash256::concat(&NULLIFIER_PK, nk).as_bytes()
}

pub fn note_commitment(
    value: u64,
    rho: &[u8; NONCE_LEN],
    delivery: &[u8; DELIVERY_KEY_LEN],
    blinding: &[u8; BLINDING_LEN],
    pk_n: &[u8; PK_N_LEN],
) -> Hash256 {
    let mut msg = Vec::with_capacity(8 + NONCE_LEN + DELIVERY_KEY_LEN + BLINDING_LEN + PK_N_LEN);
    msg.extend_from_slice(&value.to_le_bytes());
    msg.extend_from_slice(rho);
    msg.extend_from_slice(delivery);
    msg.extend_from_slice(blinding);
    msg.extend_from_slice(pk_n);
    Hash256::concat(&NOTE_COMMITMENT, &msg)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoteOpening {
    pub value: u64,
    pub rho: [u8; NONCE_LEN],
    pub delivery: [u8; DELIVERY_KEY_LEN],
    pub blinding: [u8; BLINDING_LEN],
    pub pk_n: [u8; PK_N_LEN],
}

impl NoteOpening {
    pub fn validate(&self) -> Result<(), CustodyError> {
        if self.value < CUSTODY_VALUE_MIN_NANO || self.value > CUSTODY_VALUE_MAX_NANO {
            return Err(CustodyError::ValueOutOfRange {
                value: self.value,
                min: CUSTODY_VALUE_MIN_NANO,
                max: CUSTODY_VALUE_MAX_NANO,
            });
        }
        Ok(())
    }

    pub fn commitment(&self) -> Result<Hash256, CustodyError> {
        self.validate()?;
        Ok(note_commitment(self.value, &self.rho, &self.delivery, &self.blinding, &self.pk_n))
    }
}

impl Encode for NoteOpening {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.value.to_le_bytes());
        out.extend_from_slice(&self.rho);
        out.extend_from_slice(&self.delivery);
        out.extend_from_slice(&self.blinding);
        out.extend_from_slice(&self.pk_n);
    }
    fn encoded_len(&self) -> usize {
        8 + NONCE_LEN + DELIVERY_KEY_LEN + BLINDING_LEN + PK_N_LEN
    }
}

impl Decode for NoteOpening {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let value = r.read_u64()?;
        if value < CUSTODY_VALUE_MIN_NANO || value > CUSTODY_VALUE_MAX_NANO {
            return Err(CodecError::InvariantViolated("note value outside (0, 2^60]"));
        }
        Ok(NoteOpening {
            value,
            rho: r.take_array::<NONCE_LEN>()?,
            delivery: r.take_array::<DELIVERY_KEY_LEN>()?,
            blinding: r.take_array::<BLINDING_LEN>()?,
            pk_n: r.take_array::<PK_N_LEN>()?,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;

    fn delivery(rng: &mut SplitMix64) -> [u8; DELIVERY_KEY_LEN] {
        let mut d = [0u8; DELIVERY_KEY_LEN];
        let mut i = 0;
        while i < DELIVERY_KEY_LEN {
            let w = rng.next_u64().to_le_bytes();
            let take = (DELIVERY_KEY_LEN - i).min(8);
            d[i..i + take].copy_from_slice(&w[..take]);
            i += take;
        }
        d
    }

    fn opening(rng: &mut SplitMix64) -> NoteOpening {
        NoteOpening {
            value: 1 + rng.next_u64() % CUSTODY_VALUE_MAX_NANO,
            rho: rng.bytes32(),
            delivery: delivery(rng),
            blinding: rng.bytes32(),
            pk_n: rng.bytes32(),
        }
    }

    #[test]
    fn commitment_message_layout() {
        let mut rng = SplitMix64::new(0xC0A1);
        let o = opening(&mut rng);
        let cm = o.commitment().unwrap();
        let mut msg = Vec::new();
        msg.extend_from_slice(&o.value.to_le_bytes());
        msg.extend_from_slice(&o.rho);
        msg.extend_from_slice(&o.delivery);
        msg.extend_from_slice(&o.blinding);
        msg.extend_from_slice(&o.pk_n);
        assert_eq!(msg.len(), 1288);
        assert_eq!(cm, Hash256::concat(&NOTE_COMMITMENT, &msg));
    }

    #[test]
    fn every_field_changes_the_commitment() {
        let mut rng = SplitMix64::new(0xC0A2);
        let o = opening(&mut rng);
        let cm = o.commitment().unwrap();
        let t = |m: fn(&mut NoteOpening), o: &NoteOpening| {
            let mut c = o.clone();
            m(&mut c);
            c.commitment().unwrap()
        };
        assert_ne!(t(|c| c.value += 1, &o), cm);
        assert_ne!(t(|c| c.rho[0] ^= 1, &o), cm);
        assert_ne!(t(|c| c.delivery[100] ^= 1, &o), cm);
        assert_ne!(t(|c| c.blinding[31] ^= 1, &o), cm);
        assert_ne!(t(|c| c.pk_n[7] ^= 1, &o), cm, "pk_n must bind");
        assert_eq!(o.clone().commitment().unwrap(), cm);
    }

    #[test]
    fn nullifier_pk_is_the_domain_formula() {
        let nk = [0x5Eu8; 32];
        assert_eq!(nullifier_pk(&nk), *Hash256::concat(&NULLIFIER_PK, &nk).as_bytes());
        assert_ne!(nullifier_pk(&nk), nullifier_pk(&[0x5F; 32]));
    }

    #[test]
    fn value_validation_and_codec() {
        let mut rng = SplitMix64::new(0xC0A3);
        let mut o = opening(&mut rng);
        o.value = 0;
        assert!(matches!(o.validate(), Err(CustodyError::ValueOutOfRange { value: 0, .. })));
        o.value = CUSTODY_VALUE_MAX_NANO + 1;
        assert!(o.validate().is_err());
        o.value = CUSTODY_VALUE_MAX_NANO;
        assert!(o.validate().is_ok());

        let enc = o.encode();
        assert_eq!(enc.len(), 1288);
        assert_eq!(NoteOpening::decode(&enc).unwrap(), o);
        assert!(NoteOpening::decode(&enc[..1287]).is_err());
        let mut bad = enc.clone();
        bad[0..8].copy_from_slice(&0u64.to_le_bytes());
        assert!(NoteOpening::decode(&bad).is_err());
    }
}

