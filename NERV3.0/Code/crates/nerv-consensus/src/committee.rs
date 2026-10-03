//! Epoch sortition (WP §8.3; DSR-5/E-007; erratum 125): beacon-seeded hash
//! ranking over the validator roster, role-separated by derived
//! randomness, with the beacon committee staggered in thirds.

use std::collections::{BTreeMap, BTreeSet};

use nerv_core::constants::SORTITION_DERIVE;
use nerv_core::hash::Hash256;
use nerv_core::params::{
    CONSENSUS_ATTESTATION_QUORUM, CONSENSUS_ATTESTATION_SIGNERS,
    CONSENSUS_BEACON_COMMITTEE_SIZE, CONSENSUS_REGISTRY_COMMITTEE_SIZE,
    CONSENSUS_SHARD_COMMITTEE_SIZE, SEAL_COMMITTEE_N,
};
use nerv_core::types::{Epoch, Interval, ShardId, ShardSet};
use nerv_crypto::mldsa::{VerifyingKey, PK_LEN};
use nerv_crypto::sortition::select_committee;

pub const SHARD_COMMITTEE_SIZE: usize = CONSENSUS_SHARD_COMMITTEE_SIZE as usize;
pub const BEACON_COMMITTEE_SIZE: usize = CONSENSUS_BEACON_COMMITTEE_SIZE as usize;
pub const REGISTRY_COMMITTEE_SIZE: usize = CONSENSUS_REGISTRY_COMMITTEE_SIZE as usize;
pub const ATTESTATION_SIGNERS: usize = CONSENSUS_ATTESTATION_SIGNERS as usize;
pub const ATTESTATION_QUORUM: usize = CONSENSUS_ATTESTATION_QUORUM as usize;
/// E-003: the decryption committee — 10 of the 21 shard members.
pub const DECRYPTION_COMMITTEE_SIZE: usize = SEAL_COMMITTEE_N as usize;

/// Beacon cohort sizes by selection-epoch mod 3 (erratum 125d).
pub const BEACON_COHORTS: [usize; 3] = [11, 10, 10];
/// Staggering begins once three selection epochs exist; epochs 0–2 are
/// the bootstrap window (erratum 125d).
pub const STAGGER_START: u64 = 3;

const _: () = assert!(BEACON_COHORTS[0] + BEACON_COHORTS[1] + BEACON_COHORTS[2] == BEACON_COMMITTEE_SIZE);
const _: () = assert!(SHARD_COMMITTEE_SIZE <= BEACON_COMMITTEE_SIZE);

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CommitteeError {
    #[error("randomness for epoch {epoch} is unavailable")]
    MissingRandomness { epoch: u64 },
    #[error("roster of {found} validators is below the {min} required")]
    RosterTooSmall { found: usize, min: usize },
}

/// Role- and instance-separated randomness derivation (erratum 125b).
pub fn derive_randomness(base: &Hash256, label: &[u8], extra: &[&[u8]]) -> Hash256 {
    let mut parts: Vec<&[u8]> = Vec::with_capacity(2 + extra.len());
    parts.push(base.as_bytes());
    parts.push(label);
    parts.extend_from_slice(extra);
    Hash256::framed(&SORTITION_DERIVE, &parts)
}

/// R_e (erratum 125a): derived from epoch e−1's epoch-attestation digest.
pub fn epoch_randomness(prev_epoch_attestation: &Hash256, for_epoch: Epoch) -> Hash256 {
    derive_randomness(
        prev_epoch_attestation,
        b"epoch-randomness",
        &[&for_epoch.as_u64().to_le_bytes()],
    )
}

/// R_0 — the bootstrap value (erratum 125a; genesis-config).
pub fn genesis_randomness() -> Hash256 {
    derive_randomness(
        &Hash256::from_bytes([0u8; 32]),
        b"genesis-randomness",
        &[&0u64.to_le_bytes()],
    )
}

fn shard_bytes(s: &ShardId) -> [u8; 3] {
    let mut b = [0u8; 3];
    b[0] = s.bits();
    b[1..3].copy_from_slice(&s.value().to_le_bytes());
    b
}

fn ranked(
    randomness: &Hash256,
    candidates: &[VerifyingKey],
    epoch: Epoch,
    size: usize,
) -> Vec<VerifyingKey> {
    select_committee(randomness, candidates, epoch, size)
        .into_iter()
        .map(|i| candidates[i])
        .collect()
}

fn canonical_roster(roster: &[VerifyingKey]) -> Vec<VerifyingKey> {
    let mut seen: BTreeSet<[u8; PK_LEN]> = BTreeSet::new();
    let mut out: Vec<VerifyingKey> = Vec::with_capacity(roster.len());
    for vk in roster {
        if seen.insert(*vk.as_bytes()) {
            out.push(*vk);
        }
    }
    out.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    out
}

/// The epoch's committees: one 21-member set per active shard (plus its
/// 10-member decryption subset), the 31-member staggered beacon
/// committee, and the 21-member registry committee.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Committees {
    pub epoch: Epoch,
    pub shard: BTreeMap<ShardId, Vec<VerifyingKey>>,
    pub decryption: BTreeMap<ShardId, Vec<VerifyingKey>>,
    pub beacon: Vec<VerifyingKey>,
    pub registry: Vec<VerifyingKey>,
}

pub fn select(
    epoch: Epoch,
    roster: &[VerifyingKey],
    active: &ShardSet,
    randomness: &dyn Fn(Epoch) -> Option<Hash256>,
) -> Result<Committees, CommitteeError> {
    let roster = canonical_roster(roster);
    if roster.len() < BEACON_COMMITTEE_SIZE {
        return Err(CommitteeError::RosterTooSmall {
            found: roster.len(),
            min: BEACON_COMMITTEE_SIZE,
        });
    }
    let r = |e: Epoch| {
        randomness(e).ok_or(CommitteeError::MissingRandomness { epoch: e.as_u64() })
    };

    let mut shard = BTreeMap::new();
    let mut decryption = BTreeMap::new();
    let re = r(epoch)?;
    for id in active.ids() {
        let sb = shard_bytes(id);
        let sr = derive_randomness(&re, b"shard", &[&sb]);
        let committee = ranked(&sr, &roster, epoch, SHARD_COMMITTEE_SIZE);
        let dr = derive_randomness(&re, b"decryption", &[&sb]);
        let dec = ranked(&dr, &committee, epoch, DECRYPTION_COMMITTEE_SIZE);
        shard.insert(*id, committee);
        decryption.insert(*id, dec);
    }

    let beacon = beacon_committee(epoch, &roster, randomness)?;
    let rr = derive_randomness(&re, b"registry", &[]);
    let registry = ranked(&rr, &roster, epoch, REGISTRY_COMMITTEE_SIZE);

    Ok(Committees { epoch, shard, decryption, beacon, registry })
}

/// The cohort selected at `sel_epoch` (size by sel_epoch mod 3; erratum
/// 125d). Meaningful for sel_epoch ≥ STAGGER_START; the bootstrap window
/// uses [`beacon_committee`] directly.
pub fn beacon_cohort(
    sel_epoch: Epoch,
    roster: &[VerifyingKey],
    randomness: &dyn Fn(Epoch) -> Option<Hash256>,
) -> Result<Vec<VerifyingKey>, CommitteeError> {
    let r = randomness(sel_epoch)
        .ok_or(CommitteeError::MissingRandomness { epoch: sel_epoch.as_u64() })?;
    let cohort = (sel_epoch.as_u64() % 3) as usize;
    Ok(ranked(&r, roster, sel_epoch, BEACON_COHORTS[cohort]))
}

/// The 31-member beacon committee: three staggered cohorts (erratum
/// 125d); a single selection during the bootstrap window.
pub fn beacon_committee(
    epoch: Epoch,
    roster: &[VerifyingKey],
    randomness: &dyn Fn(Epoch) -> Option<Hash256>,
) -> Result<Vec<VerifyingKey>, CommitteeError> {
    let e = epoch.as_u64();
    if e < STAGGER_START {
        let r = randomness(epoch)
            .ok_or(CommitteeError::MissingRandomness { epoch: e })?;
        return Ok(ranked(&r, roster, epoch, BEACON_COMMITTEE_SIZE));
    }
    let mut out = Vec::with_capacity(BEACON_COMMITTEE_SIZE);
    for back in 0..3u64 {
        out.extend(beacon_cohort(Epoch::from_u64(e - back), roster, randomness)?);
    }
    Ok(out)
}

/// E-001 (erratum 125e): the interval's 21-member signer subset of the
/// epoch's beacon committee.
pub fn attestation_signers(
    beacon: &[VerifyingKey],
    epoch_randomness_value: &Hash256,
    interval: Interval,
) -> Vec<VerifyingKey> {
    let ir = derive_randomness(
        epoch_randomness_value,
        b"attestation",
        &[&interval.as_u64().to_le_bytes()],
    );
    ranked(&ir, beacon, interval.epoch(), ATTESTATION_SIGNERS)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::harness::{keys, vks_of, RandomnessMap};
    use crate::testutil::SplitMix64;

   fn rmap(seed: u64, n: u64) -> RandomnessMap {
       let mut rng = SplitMix64::new(seed);
       RandomnessMap::with((0..n).map(|e| (e, h(&mut rng))).collect())
   }


   fn rand_of(r: &RandomnessMap) -> impl Fn(Epoch) -> Option<Hash256> + '_ {
       move |e| r.get(e)
   }


    #[test]
    fn sizes_roles_and_determinism() {
        let set = ShardSet::genesis();
        let (keys, roster) = keys(48);
        let _ = keys;
        let r = rmap(1, 8);
       let (ra, rb) = (rand_of(&r), rand_of(&r));
       let a = select(Epoch::from_u64(4), &roster, &set, &ra).unwrap();
       let b = select(Epoch::from_u64(4), &roster, &set, &rb).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.beacon.len(), BEACON_COMMITTEE_SIZE);
        assert_eq!(a.registry.len(), REGISTRY_COMMITTEE_SIZE);
        assert_eq!(a.shard.len(), 64);
        assert_eq!(a.decryption.len(), 64);
        for (shard, committee) in &a.shard {
            assert_eq!(committee.len(), SHARD_COMMITTEE_SIZE);
            let dec = &a.decryption[shard];
            assert_eq!(dec.len(), DECRYPTION_COMMITTEE_SIZE);
            for m in dec {
                assert!(committee.contains(m), "decryption member not in the shard committee");
            }
        }
        // Role separation: shard committees differ, none equals the beacon.
        assert_ne!(a.shard[&set.ids()[7]], a.shard[&set.ids()[8]]);
        assert_ne!(a.shard[&set.ids()[7]], a.beacon);
        assert_ne!(a.beacon, a.registry);
    }

    #[test]
    fn randomness_and_roster_sensitivity() {
        let set = ShardSet::genesis();
        let (_, roster) = keys(48);
        let ra = rmap(2, 8);
       let rb = rmap(3, 8);
       let (fa, fb) = (rand_of(&ra), rand_of(&rb));
       let a = select(Epoch::from_u64(4), &roster, &set, &fa).unwrap();
       let b = select(Epoch::from_u64(4), &roster, &set, &fb).unwrap();
       assert_ne!(a.beacon, b.beacon);
       assert_ne!(a.shard[&set.ids()[7]], b.shard[&set.ids()[7]]);


       // Epoch sensitivity: fresh per-epoch shard committees.
       let c = select(Epoch::from_u64(5), &roster, &set, &fa).unwrap();
       assert_ne!(a.shard[&set.ids()[7]], c.shard[&set.ids()[7]]);


       // Roster sensitivity.
       let (_, other) = keys(48 + 5);
       let d = select(Epoch::from_u64(4), &other, &set, &fa).unwrap();
       assert_ne!(a.beacon, d.beacon);
    }

    #[test]
    fn roster_order_independence_and_dedup() {
        let set = ShardSet::genesis();
        let (_, roster) = keys(48);
        let r = rmap(4, 8);
        let f = rand_of(&r);
       let a = select(Epoch::from_u64(6), &roster, &set, &f).unwrap();
        let mut shuffled = roster.clone();
        let mut rng = SplitMix64::new(99);
        for i in (1..shuffled.len()).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            shuffled.swap(i, j);
        }
        let b = select(Epoch::from_u64(6), &shuffled, &set, &f).unwrap();
        assert_eq!(a, b);

        // Duplicate pubkeys collapse.
        let mut duped = roster.clone();
        duped.extend_from_slice(&roster[..10]);
        let c = select(Epoch::from_u64(6), &duped, &set, &f).unwrap();
        assert_eq!(a, c);
    }

    #[test]
    fn bootstrap_then_staggered_thirds() {
        let (_, roster) = keys(60);
        let r = rmap(5, 10);
       let f = rand_of(&r);
       for e in 0..STAGGER_START {
           let b = beacon_committee(Epoch::from_u64(e), &roster, &f).unwrap();
           assert_eq!(b.len(), BEACON_COMMITTEE_SIZE);
       }
       // The bootstrap selection is the plain R_e ranking of 31.
       let b0 = beacon_committee(Epoch::from_u64(0), &roster, &f).unwrap();

       assert_eq!(
           b0,
           ranked(&r.map[&0], &roster, Epoch::from_u64(0), BEACON_COMMITTEE_SIZE)
       );

        // Staggered: beacon(e+1) retains the cohorts selected at e and e−1.
        let b3 = beacon_committee(Epoch::from_u64(3), &roster, &f).unwrap();
       let b4 = beacon_committee(Epoch::from_u64(4), &roster, &f).unwrap();
        assert_eq!(b3.len(), BEACON_COMMITTEE_SIZE);
        assert_eq!(b4.len(), BEACON_COMMITTEE_SIZE);
        let c3 = beacon_cohort(Epoch::from_u64(3), &roster, &f).unwrap();
        let c2 = beacon_cohort(Epoch::from_u64(2), &roster, &f).unwrap();
        assert_eq!(c3.len(), BEACON_COHORTS[0]);
        assert_eq!(c2.len(), BEACON_COHORTS[2]);
        for m in c3.iter().chain(c2.iter()) {
            assert!(b3.contains(m) && b4.contains(m), "retained cohort member missing");
        }
        let i34 = b3.iter().filter(|m| b4.contains(m)).count();
        assert!(i34 >= BEACON_COHORTS[0] + BEACON_COHORTS[2], "intersection {i34}");
        assert!(i34 <= BEACON_COMMITTEE_SIZE);
    }

    #[test]
    fn attestation_signers_subset() {
        let (_, roster) = keys(40);
        let r = rmap(6, 4);
       let f = rand_of(&r);
       let e = Epoch::from_u64(3);
       let beacon = beacon_committee(e, &roster, &f).unwrap();
        let re = r.map[&e.as_u64()];
        let i1 = Interval::from_u64(3 * 86_400 + 10);
        let i2 = Interval::from_u64(3 * 86_400 + 11);
        let s1 = attestation_signers(&beacon, &re, i1);
        let s1b = attestation_signers(&beacon, &re, i1);
        assert_eq!(s1, s1b);
        assert_eq!(s1.len(), ATTESTATION_SIGNERS);
        for m in &s1 {
            assert!(beacon.contains(m));
        }
        let s2 = attestation_signers(&beacon, &re, i2);
        assert_eq!(s2.len(), ATTESTATION_SIGNERS);
        assert_ne!(s1, s2, "the signer set rotates per interval");
    }

    #[test]
    fn randomness_derivations_are_pinned() {
        let d = Hash256::from_bytes([7u8; 32]);
        let a = derive_randomness(&d, b"shard", &[&[1, 2, 3]]);
        assert_eq!(a, derive_randomness(&d, b"shard", &[&[1, 2, 3]]));
        assert_ne!(a, derive_randomness(&d, b"registry", &[]));
        assert_ne!(a, derive_randomness(&Hash256::from_bytes([8u8; 32]), b"shard", &[&[1, 2, 3]]));

        let e5 = epoch_randomness(&d, Epoch::from_u64(5));
        assert_eq!(e5, epoch_randomness(&d, Epoch::from_u64(5)));
        assert_ne!(e5, epoch_randomness(&d, Epoch::from_u64(6)));
        assert_ne!(e5, epoch_randomness(&Hash256::from_bytes([9u8; 32]), Epoch::from_u64(5)));
        assert_eq!(genesis_randomness(), genesis_randomness());
        assert_ne!(genesis_randomness(), e5);
    }

    #[test]
    fn error_paths() {
        let (_, roster) = keys(48);
        let set = ShardSet::genesis();
        let partial = RandomnessMap::with(vec![(4u64, Hash256::from_bytes([1u8; 32]))]);
       let fp = rand_of(&partial);
       // Staggering (epoch 4 ≥ 3) needs R_4, R_3, R_2.
       assert!(matches!(
           select(Epoch::from_u64(4), &roster, &set, &fp),
           Err(CommitteeError::MissingRandomness { epoch: 3 })
       ));
       let (_, small) = keys(10);
       let full = rmap(7, 8);
       let ff = rand_of(&full);
       assert!(matches!(
           select(Epoch::from_u64(4), &small, &set, &ff),
           Err(CommitteeError::RosterTooSmall { found: 10, min: 31 })
       ));

        let (_, tiny) = keys(2);
        let _ = vks_of(&tiny);
    }
}
