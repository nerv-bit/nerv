//! Transparent staking (design doc staking.rs; WP §2.5, §11.4; erratum
//! 163): the stake ledger, the slash table, epoch-boundary withdrawals.
//! Evidence verification is the node's (nerv-consensus); this ledger
//! consumes the verified (class, digest, offender) triple.

use std::collections::{BTreeMap, BTreeSet};

use nerv_core::hash::Hash256;
use nerv_core::types::Epoch;
use nerv_crypto::mldsa::VerifyingKey;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlashClass {
    DoubleSign,
    InvalidBlock,
    InvalidInclusion,
    InvalidBundle,
}

impl SlashClass {
    /// The genesis-config slash table (permille of stake; erratum 163).
    pub fn slash_permille(self) -> u64 {
        match self {
            SlashClass::DoubleSign => 1000,
            SlashClass::InvalidBlock => 1000,
            SlashClass::InvalidInclusion => 250,
            SlashClass::InvalidBundle => 100,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlashRecord {
    pub offender: VerifyingKey,
    pub class: SlashClass,
    pub evidence_digest: Hash256,
    pub amount_nano: u64,
    pub epoch: Epoch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum StakeError {
    #[error("validator {offender:?} has no stake")]
    NotStaked { offender: VerifyingKey },
    #[error("withdrawal of {amount} exceeds the stake {stake}")]
    OverWithdrawal { amount: u64, stake: u64 },
    #[error("slash evidence {digest} is already consumed")]
    DuplicateEvidence { digest: Hash256 },
}

/// The map key: BLAKE3 of the verifying key — collision-free identity.
fn vk_key(vk: &VerifyingKey) -> [u8; 32] {
    *Hash256::concat(&nerv_core::constants::SLASH, vk.as_bytes()).as_bytes()
}

/// The stake ledger: bonded amounts, pending epoch-boundary withdrawals,
/// the slash log. Slash consumes stake first, then pending (erratum 163).
#[derive(Clone, Debug, Default)]
pub struct StakeLedger {
    entries: BTreeMap<[u8; 32], (VerifyingKey, u64)>,
    pending: BTreeMap<[u8; 32], Vec<(Epoch, u64)>>,
    slash_log: Vec<SlashRecord>,
    consumed_evidence: BTreeSet<[u8; 32]>,
}

impl StakeLedger {
    pub fn new() -> StakeLedger {
        StakeLedger::default()
    }

    pub fn stake(&mut self, vk: VerifyingKey, amount_nano: u64) {
        self.entries.entry(vk_key(&vk)).or_insert((vk, 0)).1 += amount_nano;
    }

    pub fn stake_of(&self, vk: &VerifyingKey) -> u64 {
        self.entries.get(&vk_key(vk)).map(|(_, s)| *s).unwrap_or(0)
    }

    pub fn total_stake(&self) -> u128 {
        self.entries.values().map(|(_, s)| u128::from(*s)).sum()
    }

    pub fn request_withdrawal(
        &mut self,
        vk: &VerifyingKey,
        amount_nano: u64,
        at_epoch: Epoch,
    ) -> Result<Epoch, StakeError> {
        let k = vk_key(vk);
        let Some((_, stake)) = self.entries.get_mut(&k) else {
            return Err(StakeError::NotStaked { offender: *vk });
        };
        if *stake < amount_nano {
            return Err(StakeError::OverWithdrawal { amount: amount_nano, stake: *stake });
        }
        *stake -= amount_nano;
        let effective = Epoch::from_u64(at_epoch.as_u64().saturating_add(1));
        self.pending.entry(k).or_default().push((effective, amount_nano));
        Ok(effective)
    }

    pub fn finalize_epoch(&mut self, epoch: Epoch) -> Vec<(VerifyingKey, u64)> {
        let mut out = Vec::new();
        for (k, queue) in self.pending.iter_mut() {
            let Some((vk, _)) = self.entries.get(k) else { continue };
            let mut paid = 0u64;
            queue.retain(|(eff, amt)| {
                if *eff <= epoch {
                    paid += amt;
                    false
                } else {
                    true
                }
            });
            if paid > 0 {
                out.push((*vk, paid));
            }
        }
        out
    }

    fn pending_of(&self, k: &[u8; 32]) -> u64 {
        self.pending.get(k).map(|q| q.iter().map(|(_, a)| *a).sum()).unwrap_or(0)
    }

    /// Slash: fraction × (stake + pending), from stake first, then
    /// pending. Evidence digests are single-use.
    pub fn slash(
        &mut self,
        vk: &VerifyingKey,
        class: SlashClass,
        evidence_digest: Hash256,
        epoch: Epoch,
    ) -> Result<u64, StakeError> {
        if !self.consumed_evidence.insert(*evidence_digest.as_bytes()) {
            return Err(StakeError::DuplicateEvidence { digest: evidence_digest });
        }
        let k = vk_key(vk);
        // Materialize the pending-total before taking the mutable borrow on
        // `entries`: `self.pending_of` takes `&self`, which would otherwise
        // alias the mutable reborrow below (Rust's two-phase borrow checker
        // rejects the overlap even though `entries` and `pending` are
        // disjoint fields, because the method receiver is `&self` as a whole).
        let pending_total = self.pending_of(&k);
        let Some((_, stake)) = self.entries.get_mut(&k) else {
            return Err(StakeError::NotStaked { offender: *vk });
        };
        let bonded = u128::from(*stake + pending_total);
        let target = ((bonded * u128::from(class.slash_permille())) / 1000) as u64;
        let mut from_stake = target.min(*stake);
        *stake -= from_stake;
        let mut from_pending = target - from_stake;
        if from_pending > 0 {
            if let Some(queue) = self.pending.get_mut(&k) {
                for (_, amt) in queue.iter_mut() {
                    if from_pending == 0 {
                        break;
                    }
                    let d = from_pending.min(*amt);
                    *amt -= d;
                    from_pending -= d;
                }
                queue.retain(|(_, a)| *a > 0);
            }
        }
        let slashed = target - from_pending;
        self.slash_log.push(SlashRecord {
            offender: *vk,
            class,
            evidence_digest,
            amount_nano: slashed,
            epoch,
        });
        Ok(slashed)
    }

    pub fn slash_log(&self) -> &[SlashRecord] {
        &self.slash_log
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use nerv_crypto::mldsa::SigningKey;

    fn key(seed: u64) -> SigningKey {
        let mut b = [0u8; 32];
        b[..8].copy_from_slice(&seed.to_le_bytes());
        SigningKey::from_seed(&b).unwrap()
    }

    fn dig(seed: u64) -> Hash256 {
        Hash256::from_bytes([seed as u8; 32])
    }

    #[test]
    fn slash_table_pins() {
        assert_eq!(SlashClass::DoubleSign.slash_permille(), 1000);
        assert_eq!(SlashClass::InvalidBlock.slash_permille(), 1000);
        assert_eq!(SlashClass::InvalidInclusion.slash_permille(), 250);
        assert_eq!(SlashClass::InvalidBundle.slash_permille(), 100);
    }

    #[test]
    fn stake_withdraw_boundaries() {
        let mut l = StakeLedger::new();
        let (a, b) = (key(1), key(2));
        let avk = *a.verifying_key();
        l.stake(avk, 1000);
        l.stake(avk, 500);
        l.stake(*b.verifying_key(), 700);
        assert_eq!(l.stake_of(&avk), 1500);
        assert_eq!(l.stake_of(b.verifying_key()), 700);
        assert_eq!(l.total_stake(), 2200);
        assert_eq!(l.stake_of(&key(99).verifying_key()), 0);

        // Over-withdrawal rejected.
        assert!(matches!(
            l.request_withdrawal(&avk, 1501, Epoch::from_u64(5)),
            Err(StakeError::OverWithdrawal { amount: 1501, stake: 1500 })
        ));
        assert!(matches!(
            l.request_withdrawal(&key(99).verifying_key(), 1, Epoch::from_u64(5)),
            Err(StakeError::NotStaked { .. })
        ));

        // Boundary: request at 5 → effective 6.
        let eff = l.request_withdrawal(&avk, 1000, Epoch::from_u64(5)).unwrap();
        assert_eq!(eff, Epoch::from_u64(6));
        assert_eq!(l.stake_of(&avk), 500, "moved out of stake into pending");
        assert!(l.finalize_epoch(Epoch::from_u64(5)).is_empty(), "not yet due");
        let paid = l.finalize_epoch(Epoch::from_u64(6));
        assert_eq!(paid, vec![(avk, 1000)]);
        assert!(l.finalize_epoch(Epoch::from_u64(7)).is_empty(), "paid once");
        assert_eq!(l.stake_of(&avk), 500, "the remaining stake persists");
    }

    #[test]
    fn slash_classes_and_pending_interaction() {
        // Full slashes.
        for class in [SlashClass::DoubleSign, SlashClass::InvalidBlock] {
            let mut l = StakeLedger::new();
            let vk = *key(1).verifying_key();
            l.stake(vk, 1000);
            let slashed = l.slash(&vk, class, dig(1), Epoch::from_u64(1)).unwrap();
            assert_eq!(slashed, 1000);
            assert_eq!(l.stake_of(&vk), 0);
        }
        // Partial.
        let mut l = StakeLedger::new();
        let vk = *key(2).verifying_key();
        l.stake(vk, 1000);
        let slashed = l.slash(&vk, SlashClass::InvalidInclusion, dig(2), Epoch::from_u64(1)).unwrap();
        assert_eq!(slashed, 250);
        assert_eq!(l.stake_of(&vk), 750);
        let slashed = l.slash(&vk, SlashClass::InvalidBundle, dig(3), Epoch::from_u64(2)).unwrap();
        assert_eq!(slashed, 75, "10% of the remaining 750");
        assert_eq!(l.stake_of(&vk), 675);

        // Slash with a pending withdrawal: 1000 stake + 1000 pending.
        let mut l = StakeLedger::new();
        let vk = *key(3).verifying_key();
        l.stake(vk, 2000);
        l.request_withdrawal(&vk, 1000, Epoch::from_u64(9)).unwrap();
        assert_eq!(l.stake_of(&vk), 1000);
        let slashed = l.slash(&vk, SlashClass::InvalidInclusion, dig(4), Epoch::from_u64(9)).unwrap();
        assert_eq!(slashed, 500, "25% of (1000 + 1000)");
        assert_eq!(l.stake_of(&vk), 500);
        let paid = l.finalize_epoch(Epoch::from_u64(10));
        assert_eq!(paid, vec![(vk, 500)], "the pending withdrawal was reduced");

        // Evidence replay rejected; unknown offender rejected.
        let mut l = StakeLedger::new();
        let vk = *key(4).verifying_key();
        l.stake(vk, 100);
        l.slash(&vk, SlashClass::DoubleSign, dig(5), Epoch::from_u64(1)).unwrap();
        assert!(matches!(
            l.slash(&vk, SlashClass::DoubleSign, dig(5), Epoch::from_u64(2)),
            Err(StakeError::DuplicateEvidence { .. })
        ));
        let other = *key(5).verifying_key();
        assert!(matches!(
            l.slash(&other, SlashClass::DoubleSign, dig(6), Epoch::from_u64(1)),
            Err(StakeError::NotStaked { .. })
        ));

        // The log binds everything.
        let log = l.slash_log();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].amount_nano, 100);
        assert_eq!(log[0].class, SlashClass::DoubleSign);
        assert_eq!(log[0].evidence_digest, dig(5));
        assert_eq!(log[0].epoch, Epoch::from_u64(1));
    }
}


