//! C_t — the composite state commitment (WP §4.2) and C₀ (WP §12.1 as
//! amended by E-008). Exactly six fields; everything else a header carries
//! (D_t, H(ct_B), the reveal, the registry reference, fees, the payout,
//! QC_hash) is header-committed via `nerv.hdr`, never C_t-committed.

use nerv_core::constants::STATE_COMMITMENT;
use nerv_core::hash::Hash256;
use nerv_core::types::Height;
use nerv_custody::{NoteCommitmentTree, NullifierSet, NctDigest, TransitLog};

/// `C_t = BLAKE3("nerv.state" ‖ nct_root ‖ nullifier_root ‖ transit_root ‖
/// params_root ‖ prev ‖ height)` — the sole source of truth. Fixed-width
/// fields throughout (5 × 32 + 8 bytes): the concatenation is unambiguous.
pub fn state_commitment(
    nct_root: &NctDigest,
    nullifier_root: &Hash256,
    transit_root: &Hash256,
    params_root: &Hash256,
    prev: &Hash256,
    height: Height,
) -> Hash256 {
    let mut msg = [0u8; 5 * 32 + 8];
    msg[..32].copy_from_slice(nct_root.as_bytes());
    msg[32..64].copy_from_slice(nullifier_root.as_bytes());
    msg[64..96].copy_from_slice(transit_root.as_bytes());
    msg[96..128].copy_from_slice(params_root.as_bytes());
    msg[128..160].copy_from_slice(prev.as_bytes());
    msg[160..168].copy_from_slice(&height.as_u64().to_le_bytes());
    Hash256::concat(&STATE_COMMITMENT, &msg)
}

/// C₀: empty component trees, zero predecessor, height 0 — the published
/// genesis commitment (WP §13.3). The emission ledger is beacon-side state
/// and is not part of any shard's C_t.
pub fn genesis_commitment(params_root: &Hash256) -> Hash256 {
    state_commitment(
        &NoteCommitmentTree::new().root(),
        &NullifierSet::new().root(),
        &TransitLog::new().root(),
        params_root,
        &Hash256::from_bytes([0u8; 32]),
        Height::ZERO,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use nerv_core::field::Goldilocks;
    use nerv_custody::{empty_digests, NCT_DEPTH};

    fn digest(k: u32) -> NctDigest {
        NctDigest::from_elements(&[
            Goldilocks::from_u32(k),
            Goldilocks::from_u32(k.wrapping_add(1)),
            Goldilocks::from_u32(k.wrapping_mul(3)),
            Goldilocks::from_u32(k.wrapping_mul(7)),
        ])
    }

    #[test]
    fn literal_formula_and_field_sensitivity() {
        let nct = digest(11);
        let null = Hash256::from_bytes([1u8; 32]);
        let transit = Hash256::from_bytes([2u8; 32]);
        let params = Hash256::from_bytes([3u8; 32]);
        let prev = Hash256::from_bytes([4u8; 32]);
        let h = Height::from_u64(9);
        let c = state_commitment(&nct, &null, &transit, &params, &prev, h);

        let mut msg = Vec::new();
        msg.extend_from_slice(STATE_COMMITMENT.as_bytes());
        msg.extend_from_slice(nct.as_bytes());
        msg.extend_from_slice(null.as_bytes());
        msg.extend_from_slice(transit.as_bytes());
        msg.extend_from_slice(params.as_bytes());
        msg.extend_from_slice(prev.as_bytes());
        msg.extend_from_slice(&h.as_u64().to_le_bytes());
        assert_eq!(msg.len(), STATE_COMMITMENT.as_bytes().len() + 168);
        assert_eq!(c.as_bytes(), blake3::hash(&msg).as_bytes());

        assert_eq!(c, state_commitment(&nct, &null, &transit, &params, &prev, h));
        assert_ne!(c, state_commitment(&digest(12), &null, &transit, &params, &prev, h));
        assert_ne!(
            c,
            state_commitment(&nct, &Hash256::from_bytes([9u8; 32]), &transit, &params, &prev, h)
        );
        assert_ne!(
            c,
            state_commitment(&nct, &null, &Hash256::from_bytes([9u8; 32]), &params, &prev, h)
        );
        assert_ne!(
            c,
            state_commitment(&nct, &null, &transit, &Hash256::from_bytes([9u8; 32]), &prev, h)
        );
        assert_ne!(
            c,
            state_commitment(&nct, &null, &transit, &params, &Hash256::from_bytes([9u8; 32]), h)
        );
        assert_ne!(c, state_commitment(&nct, &null, &transit, &params, &prev, Height::from_u64(10)));
    }

    #[test]
    fn genesis_commitment_is_the_empty_state() {
        let p = Hash256::from_bytes([7u8; 32]);
        let g = genesis_commitment(&p);
        assert_eq!(g, genesis_commitment(&p));
        assert_ne!(g, genesis_commitment(&Hash256::from_bytes([8u8; 32])));

        let empty_nct = NoteCommitmentTree::new().root();
        assert_eq!(empty_nct, empty_digests()[NCT_DEPTH]);
        assert_eq!(
            g,
            state_commitment(
                &empty_nct,
                &NullifierSet::new().root(),
                &TransitLog::new().root(),
                &p,
                &Hash256::from_bytes([0u8; 32]),
                Height::ZERO,
            )
        );
    }
}
