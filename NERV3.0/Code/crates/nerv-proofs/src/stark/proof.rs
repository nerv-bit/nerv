//! Canonical wire format for composed proofs (nerv-core codec laws; WP
//! §5.4's proof-size surface). Every container length is capped far above
//! any honest bound, so decoding adversarial bytes allocates
//! proportionally to input size and never panics; semantic shape checks
//! live in `compose_verify`, which sees the decoded object.

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::error::CodecError;
use nerv_core::field::Goldilocks;
use nerv_core::hash::Hash256;

use crate::stark::compose::{ComposedProof, OuterOpenings};
use crate::stark::ext_field::ExtF;
use crate::stark::fri::{FriProof, FriQuery, PairOpening};
use crate::stark::merkle::RowOpening;

/// Auth-path digests per opening: honest ≤ 32 (two-adicity).
const MAX_PATH: usize = 64;
/// Committed FRI layers / pair openings per query: honest ≤ 32.
const MAX_LAYERS: usize = 64;
/// Queries per proof: honest ≤ 512.
const MAX_QUERIES: usize = 4096;
/// Row / value / coefficient lengths: honest ≤ the widest AIR's columns.
const MAX_VALUES: usize = 1 << 20;

fn read_capped<T: Decode>(r: &mut Reader<'_>, cap: usize) -> Result<Vec<T>, CodecError> {
    let n = r.read_u32()? as usize;
    if n > cap {
        return Err(CodecError::SeqTooLarge { count: n, max: cap });
    }
    if n > r.remaining() {
        return Err(CodecError::SeqLenOverrun { count: n, remaining: r.remaining() });
    }
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(T::decode_from(r)?);
    }
    Ok(v)
}

impl Encode for RowOpening {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.row.encode_into(out);
        self.path.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        self.row.encoded_len() + self.path.encoded_len()
    }
}

impl Decode for RowOpening {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(RowOpening {
            row: read_capped::<Goldilocks>(r, MAX_VALUES)?,
            path: read_capped::<Hash256>(r, MAX_PATH)?,
        })
    }
}

impl Encode for PairOpening {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.low.encode_into(out);
        self.high.encode_into(out);
        self.path.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        16 + 16 + self.path.encoded_len()
    }
}

impl Decode for PairOpening {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(PairOpening {
            low: ExtF::decode_from(r)?,
            high: ExtF::decode_from(r)?,
            path: read_capped::<Hash256>(r, MAX_PATH)?,
        })
    }
}

impl Encode for FriQuery {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.openings.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        self.openings.encoded_len()
    }
}

impl Decode for FriQuery {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(FriQuery { openings: read_capped::<PairOpening>(r, MAX_LAYERS)? })
    }
}

impl Encode for FriProof {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.roots.encode_into(out);
        self.final_coeffs.encode_into(out);
        out.extend_from_slice(&self.pow_nonce.to_le_bytes());
        self.queries.encode_into(out);
        self.positions.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        self.roots.encoded_len()
            + self.final_coeffs.encoded_len()
            + 8
            + self.queries.encoded_len()
            + self.positions.encoded_len()
    }
}

impl Decode for FriProof {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(FriProof {
            roots: read_capped::<Hash256>(r, MAX_LAYERS)?,
            final_coeffs: read_capped::<ExtF>(r, MAX_VALUES)?,
            pow_nonce: r.read_u64()?,
            queries: read_capped::<FriQuery>(r, MAX_QUERIES)?,
            positions: read_capped::<u64>(r, MAX_QUERIES)?,
        })
    }
}

impl Encode for OuterOpenings {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.trace_low.encode_into(out);
        self.trace_next.encode_into(out);
        self.quotient.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        self.trace_low.encoded_len() + self.trace_next.encoded_len() + self.quotient.encoded_len()
    }
}

impl Decode for OuterOpenings {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(OuterOpenings {
            trace_low: RowOpening::decode_from(r)?,
            trace_next: RowOpening::decode_from(r)?,
            quotient: RowOpening::decode_from(r)?,
        })
    }
}

impl Encode for ComposedProof {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.trace_root.encode_into(out);
        self.quotient_root.encode_into(out);
        self.trace_zeta.encode_into(out);
        self.trace_zeta_next.encode_into(out);
        self.quotient_zeta.encode_into(out);
        out.extend_from_slice(&self.num_assertions.to_le_bytes());
        self.fri.encode_into(out);
        self.outer.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        32 + 32
            + self.trace_zeta.encoded_len()
            + self.trace_zeta_next.encoded_len()
            + 16
            + 8
            + self.fri.encoded_len()
            + self.outer.encoded_len()
    }
}

impl Decode for ComposedProof {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(ComposedProof {
            trace_root: Hash256::decode_from(r)?,
            quotient_root: Hash256::decode_from(r)?,
            trace_zeta: read_capped::<ExtF>(r, MAX_VALUES)?,
            trace_zeta_next: read_capped::<ExtF>(r, MAX_VALUES)?,
            quotient_zeta: ExtF::decode_from(r)?,
            num_assertions: r.read_u64()?,
            fri: FriProof::decode_from(r)?,
            outer: read_capped::<OuterOpenings>(r, MAX_QUERIES)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::chips::range::{gen_range_witness, RangeChip};
    use crate::air::fs::FsTranscript;
    use crate::security::FriShape;
    use crate::stark::collector::measure_true;
    use crate::stark::compose::{compose_prove, Plan};

    fn small_fri() -> FriShape {
        FriShape {
            log_blowup: 1,
            num_queries: 2,
            log_final_poly_len: 1,
            max_log_arity: FriShape::DEFAULT_ARITY,
            commit_pow_bits: 0,
            query_pow_bits: 0,
        }
    }

    fn sample_proof() -> ComposedProof {
        let rows: Vec<Vec<Goldilocks>> =
            (0..4u64).map(|i| gen_range_witness(0x30 + i, 32)).collect();
        let chip = RangeChip::new(0, 1, 32);
        let td = measure_true(&chip);
        let plan = Plan::new(2, &small_fri(), &td).unwrap();
        let mut t = FsTranscript::new();
        compose_prove(&plan, &chip, &rows, &[], &[], &mut t).unwrap()
    }

    #[test]
    fn roundtrip_and_exact_length() {
        let proof = sample_proof();
        let enc = proof.encode();
        assert_eq!(enc.len(), proof.encoded_len());
        assert_eq!(ComposedProof::decode(&enc), Ok(proof.clone()));
        assert_eq!(ComposedProof::decode(&proof.encode()), Ok(proof));
    }

    #[test]
    fn strictness_every_prefix_and_trailing() {
        let enc = sample_proof().encode();
        for cut in 0..enc.len() {
            assert!(ComposedProof::decode(&enc[..cut]).is_err(), "cut={cut}");
        }
        let mut ext = enc.clone();
        ext.push(0);
        assert!(ComposedProof::decode(&ext).is_err());
    }

    #[test]
    fn tampered_bytes_never_decode_to_the_same_proof() {
        let enc = sample_proof().encode();
        for pos in [0usize, 1, enc.len() / 2, enc.len() - 1] {
            let mut bad = enc.clone();
            bad[pos] ^= 1;
            let ok = ComposedProof::decode(&bad).map(|p| p == ComposedProof::decode(&enc).unwrap());
            assert_ne!(ok, Ok(true), "pos={pos}");
        }
    }

    #[test]
    fn container_caps_reject_absurd_counts() {
        let huge = 0xFFFF_FFFFu32.to_le_bytes().to_vec();
        assert!(matches!(
            FriProof::decode(&huge),
            Err(CodecError::SeqTooLarge { .. })
        ));
        assert!(matches!(
            RowOpening::decode(&huge),
            Err(CodecError::SeqTooLarge { .. })
        ));
        assert!(matches!(
            FriQuery::decode(&huge),
            Err(CodecError::SeqTooLarge { .. })
        ));
        // Just under the cap but past the input: overrun, not allocation.
        let over = 65u32.to_le_bytes().to_vec();
        assert!(matches!(
            FriProof::decode(&over),
            Err(CodecError::SeqLenOverrun { .. })
        ));
    }
}

