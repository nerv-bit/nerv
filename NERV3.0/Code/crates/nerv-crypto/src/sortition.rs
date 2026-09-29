//! Beacon-seeded hash sortition (E-007 / DSR-5): committee membership is the
//! canonical ranking of BLAKE3("nerv.sortition" ‖ randomness ‖ pubkey ‖
//! epoch). No lattice-VRF exists in production; the protocol's actual
//! requirements — unpredictability-before-finality (the randomness is
//! supplied by the beacon hash chain), public verifiability, and determinism
//! — are all delivered by this construction. Ties (a BLAKE3 collision) break
//! by canonical pubkey order, so the ranking is a total order.

use nerv_core::constants::SORTITION;
use nerv_core::hash::Hash256;
use nerv_core::types::Epoch;
use crate::mldsa::VerifyingKey;

pub fn sortition_key(randomness: &Hash256, pubkey: &VerifyingKey, epoch: Epoch) -> Hash256 {
    let epoch_le = epoch.as_u64().to_le_bytes();
    let mut msg = Vec::with_capacity(32 + 1952 + 8);
    msg.extend_from_slice(randomness.as_bytes());
    msg.extend_from_slice(pubkey.as_bytes());
    msg.extend_from_slice(&epoch_le);
    Hash256::concat(&SORTITION, &msg)
}

/// Select `size` members from `candidates`, returning indices into
/// `candidates` in canonical rank order — that order IS the committee
/// roster, so member indices in quorum certificates refer to these
/// positions. `size ≥ candidates.len()` selects everyone; `size = 0`
/// selects nobody.
pub fn select_committee(
    randomness: &Hash256,
    candidates: &[VerifyingKey],
    epoch: Epoch,
    size: usize,
) -> Vec<usize> {
    let mut ranked: Vec<(Hash256, usize)> = candidates
        .iter()
        .enumerate()
        .map(|(i, vk)| (sortition_key(randomness, vk, epoch), i))
        .collect();
    ranked.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| candidates[a.1].as_bytes().cmp(candidates[b.1].as_bytes()))
    });
    ranked.into_iter().map(|(_, i)| i).take(size).collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::mldsa::SigningKey;
    use crate::testutil::DetRng;

    fn keys(n: u64) -> Vec<VerifyingKey> {
        (0..n)
            .map(|i| {
                let mut rng = DetRng::new(1000 + i);
                *SigningKey::from_seed(&rng.bytes32()).unwrap().verifying_key()
            })
            .collect()
    }

    #[test]
    fn deterministic_and_seed_sensitive() {
        let cands = keys(10);
        let r = Hash256::from_bytes(DetRng::new(1).bytes32());
        let e = Epoch::from_u64(4);
        assert_eq!(
            select_committee(&r, &cands, e, 5),
            select_committee(&r, &cands, e, 5)
        );
        let r2 = Hash256::from_bytes(DetRng::new(2).bytes32());
        assert_ne!(select_committee(&r, &cands, e, 5), select_committee(&r2, &cands, e, 5));
        assert_ne!(
            select_committee(&r, &cands, e, 5),
            select_committee(&r, &cands, Epoch::from_u64(5), 5)
        );
    }

    #[test]
    fn size_clamping() {
        let cands = keys(7);
        let r = Hash256::from_bytes(DetRng::new(3).bytes32());
        let e = Epoch::from_u64(1);
        assert_eq!(select_committee(&r, &cands, e, 0).len(), 0);
        assert_eq!(select_committee(&r, &cands, e, 7).len(), 7);
        let all = select_committee(&r, &cands, e, 99);
        assert_eq!(all.len(), 7);
        assert_eq!(BTreeSet::from_iter(all.iter().copied()).len(), 7);
    }

    #[test]
    fn selection_is_the_rank_prefix() {
        let cands = keys(12);
        let r = Hash256::from_bytes(DetRng::new(4).bytes32());
        let e = Epoch::from_u64(2);
        let chosen = select_committee(&r, &cands, e, 5);
        assert_eq!(chosen.len(), 5);
        let in_set: BTreeSet<usize> = chosen.iter().copied().collect();
        assert_eq!(in_set.len(), 5);
        for &i in &chosen {
            for j in 0..cands.len() {
                if in_set.contains(&j) {
                    continue;
                }
                let hi = sortition_key(&r, &cands[i], e);
                let hj = sortition_key(&r, &cands[j], e);
                assert!(
                    hi < hj || (hi == hj && cands[i].as_bytes() < cands[j].as_bytes()),
                    "chosen member {i} does not rank below non-member {j}"
                );
            }
        }
    }

    #[test]
    fn spread_is_flat_under_uniform_randomness() {
        let cands = keys(20);
        let e = Epoch::from_u64(1);
        let mut counts = [0u32; 20];
        let mut rng = DetRng::new(99);
        for _ in 0..1500 {
            let r = Hash256::from_bytes(rng.bytes32());
            for i in select_committee(&r, &cands, e, 5) {
                counts[i] += 1;
            }
        }
        for (i, c) in counts.iter().enumerate() {
            assert!((150..600).contains(c), "member {i} count {c} outside tolerance");
        }
    }
}

