//! The emission ledger (WP §12.3; errata 158–159).

use std::collections::BTreeMap;

use nerv_core::constants::{
    CLAIM_COMMIT, CLAIM_ELIG, EMISSION, EMISSION_CRED, EMISSION_ROOT,
};
use nerv_core::hash::Hash256;
use nerv_core::types::Epoch;
use nerv_crypto::mldsa::{Signature, SigningKey, VerifyingKey};

use crate::schedule::{AccountKind, EmissionSchedule};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum LedgerError {
    #[error("account {account:?} not found")]
    AccountNotFound { account: AccountId },
    #[error("amount {amount} exceeds the available {spendable}")]
    Insufficient { amount: u64, spendable: u64 },
    #[error("credential epoch {credential} does not match the ledger's {ledger}")]
    EpochMismatch { credential: u64, ledger: u64 },
    #[error("unknown bucket `{bucket}`")]
    BucketMismatch { bucket: &'static str },
    #[error("crediting {amount} would exceed the day's scheduled {scheduled}")]
    OverEmission { amount: u64, scheduled: u64 },
    #[error("account already credited for epoch {epoch}")]
    AlreadyCredited { epoch: u64 },
    #[error("credential signature failed")]
    BadSignature,
    #[error("the account is not a commitment-note account")]
    NotANote,
    #[error("the account is not a signed account")]
    NotSigned,
    #[error("claim nullifier {nullifier:?} is already spent")]
    NullifierSpent { nullifier: [u8; 32] },
    #[error("burn exceeds the account's committed remainder")]
    BurnExceedsCommitted,
}

/// A ledger account's identity: H("nerv.emission" ‖ key-material ‖ bucket).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct AccountId([u8; 32]);

impl AccountId {
    pub fn from_bytes(b: [u8; 32]) -> AccountId {
        AccountId(b)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn as_hash(&self) -> Hash256 {
        Hash256::from_bytes(self.0)
    }
}

pub fn derive_lek(seed: &[u8; 32]) -> [u8; 32] {
    *Hash256::concat(&nerv_core::constants::EMISSION_LEK, seed).as_bytes()
}

pub fn derive_claim_key(seed: &[u8; 32]) -> [u8; 32] {
    *Hash256::concat(&nerv_core::constants::CLAIM_KEY, seed).as_bytes()
}

pub fn signed_account_id(lek: &[u8; 32], bucket: &str) -> AccountId {
    let mut msg = Vec::with_capacity(64 + bucket.len());
    msg.extend_from_slice(lek);
    msg.extend_from_slice(bucket.as_bytes());
    AccountId(*Hash256::concat(&EMISSION, &msg).as_bytes())
}

pub fn note_commitment(
    ck: &[u8; 32],
    bucket: &str,
    amount_nano: u64,
    blinding: &[u8; 32],
) -> [u8; 32] {
    let mut msg = Vec::with_capacity(72 + bucket.len());
    msg.extend_from_slice(ck);
    msg.extend_from_slice(bucket.as_bytes());
    msg.extend_from_slice(&amount_nano.to_le_bytes());
    msg.extend_from_slice(blinding);
    *Hash256::concat(&CLAIM_COMMIT, &msg).as_bytes()
}

pub fn claim_nullifier(ck: &[u8; 32], bucket: &str) -> [u8; 32] {
    let mut msg = Vec::with_capacity(32 + bucket.len());
    msg.extend_from_slice(ck);
    msg.extend_from_slice(bucket.as_bytes());
    *Hash256::concat(&nerv_core::constants::CLAIM_NULL, &msg).as_bytes()
}

pub fn eligibility_digest(ck: &[u8; 32], bucket: &str, amount_nano: u64) -> [u8; 32] {
    let mut msg = Vec::with_capacity(40 + bucket.len());
    msg.extend_from_slice(ck);
    msg.extend_from_slice(bucket.as_bytes());
    msg.extend_from_slice(&amount_nano.to_le_bytes());
    *Hash256::concat(&CLAIM_ELIG, &msg).as_bytes()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccountEntry {
    pub committed_nano: u64,
    pub spendable_nano: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignedAccount {
    pub lek: [u8; 32],
    pub vk: VerifyingKey,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoteAccount {
    pub commitment: [u8; 32],
    pub eligibility: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AccountHolder {
    Signed(SignedAccount),
    Note(NoteAccount),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmissionCredential {
    pub lek: [u8; 32],
    pub bucket: &'static str,
    pub amount_nano: u64,
    pub epoch: Epoch,
    pub signature: Signature,
}

impl EmissionCredential {
    pub fn message(&self) -> Vec<u8> {
        let mut m = Vec::with_capacity(EMISSION_CRED.as_bytes().len() + 32 + 16 + 8 + 8);
        m.extend_from_slice(EMISSION_CRED.as_bytes());
        m.extend_from_slice(&self.lek);
        m.extend_from_slice(self.bucket.as_bytes());
        m.extend_from_slice(&self.amount_nano.to_le_bytes());
        m.extend_from_slice(&self.epoch.as_u64().to_le_bytes());
        m
    }

    pub fn build(
        beacon: &SigningKey,
        lek: [u8; 32],
        bucket: &'static str,
        amount_nano: u64,
        epoch: Epoch,
    ) -> Result<EmissionCredential, nerv_crypto::CryptoError> {
        let probe = EmissionCredential {
            lek,
            bucket,
            amount_nano,
            epoch,
            signature: Signature::from_bytes([0u8; nerv_crypto::mldsa::SIG_LEN]),
        };
        let signature = beacon.sign(&probe.message())?;
        Ok(EmissionCredential { lek, bucket, amount_nano, epoch, signature })
    }

    pub fn verify(&self, beacon_vk: &VerifyingKey) -> bool {
        beacon_vk.verify(&self.message(), &self.signature)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BurnEvent {
    pub account: AccountId,
    pub amount_nano: u64,
    pub epoch: Epoch,
}

#[derive(Clone, Debug, Default)]
pub struct EmissionLedger {
    epoch: Epoch,
    accounts: BTreeMap<AccountId, (AccountHolder, AccountEntry)>,
    nullifiers: std::collections::BTreeSet<[u8; 32]>,
    burns: Vec<BurnEvent>,
    /// (bucket, day) → nano credited — the bucket-total bound (erratum 165).
    credited: BTreeMap<(&'static str, u64), u128>,
    /// account → last credited epoch (per-account freshness).
    last_credited: BTreeMap<AccountId, u64>,
}

impl EmissionLedger {
    pub fn new() -> EmissionLedger {
        EmissionLedger::default()
    }

    pub fn epoch(&self) -> Epoch {
        self.epoch
    }

    pub fn set_epoch(&mut self, epoch: Epoch) {
        self.epoch = epoch;
    }

    pub fn accounts(&self) -> impl Iterator<Item = (&AccountId, &AccountHolder, &AccountEntry)> {
        self.accounts.iter().map(|(id, (h, e))| (id, h, e))
    }

    pub fn account(&self, id: &AccountId) -> Option<(&AccountHolder, &AccountEntry)> {
        self.accounts.get(id).map(|(h, e)| (h, e))
    }

    pub fn len(&self) -> usize {
        self.accounts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.accounts.is_empty()
    }

    pub fn nullifiers(&self) -> &std::collections::BTreeSet<[u8; 32]> {
        &self.nullifiers
    }

    pub fn burns(&self) -> &[BurnEvent] {
        &self.burns
    }

    pub fn add_signed(
        &mut self,
        lek: [u8; 32],
        bucket: &'static str,
        vk: VerifyingKey,
    ) -> AccountId {
        let id = signed_account_id(&lek, bucket);
        self.accounts.entry(id).or_insert_with(|| {
            (
                AccountHolder::Signed(SignedAccount { lek, vk }),
                AccountEntry { committed_nano: 0, spendable_nano: 0 },
            )
        });
        id
    }

    pub fn add_note(&mut self, note: NoteAccount, committed_nano: u64) -> AccountId {
        let id = AccountId(note.commitment);
        let entry = self.accounts.entry(id).or_insert_with(|| {
            (AccountHolder::Note(note), AccountEntry { committed_nano: 0, spendable_nano: 0 })
        });
        entry.1.committed_nano = entry.1.committed_nano.saturating_add(committed_nano);
        id
    }

    pub fn credit(
        &mut self,
        schedule: &EmissionSchedule,
        beacon_vk: &VerifyingKey,
        credential: &EmissionCredential,
    ) -> Result<(), LedgerError> {
        if !credential.verify(beacon_vk) {
            return Err(LedgerError::BadSignature);
        }
        if credential.epoch != self.epoch {
            return Err(LedgerError::EpochMismatch {
                credential: credential.epoch.as_u64(),
                ledger: self.epoch.as_u64(),
            });
        }
        let bucket = schedule
            .bucket(credential.bucket)
            .ok_or(LedgerError::BucketMismatch { bucket: credential.bucket })?;
        let id = signed_account_id(&credential.lek, credential.bucket);
        let Some((holder, entry)) = self.accounts.get_mut(&id) else {
            return Err(LedgerError::AccountNotFound { account: id });
        };
        if !matches!(holder, AccountHolder::Signed(_)) {
            return Err(LedgerError::NotSigned);
        }
        if let Some(&last) = self.last_credited.get(&id) {
            if last >= credential.epoch.as_u64() {
                return Err(LedgerError::AlreadyCredited { epoch: last });
            }
        }
        let day = credential.epoch.as_u64();
        let cap = u128::from(bucket.day_emission(day)) * 1_000_000_000u128;
        let key = (credential.bucket, day);
        let acc = *self.credited.get(&key).unwrap_or(&0);
        if acc + u128::from(credential.amount_nano) > cap {
            return Err(LedgerError::OverEmission {
                amount: credential.amount_nano,
                scheduled: cap as u64,
            });
        }
        self.credited.insert(key, acc + u128::from(credential.amount_nano));
        self.last_credited.insert(id, day);
        entry.committed_nano += credential.amount_nano;
        entry.spendable_nano += credential.amount_nano;
        Ok(())
    }

    /// The nano credited to `bucket` on `day` (the bucket-total tracker).
    pub fn credited_nano(&self, bucket: &'static str, day: u64) -> u128 {
        self.credited.get(&(bucket, day)).copied().unwrap_or(0)
    }

    /// The M1 schedule-replay audit (erratum 165): per time-driven
    /// bucket, the cumulative credited total equals released_by_day.
    pub fn audit_day(&self, day: u64, schedule: &EmissionSchedule) -> Result<(), LedgerError> {
        for b in schedule.buckets() {
            if !b.is_time_driven() {
                continue;
            }
            let mut sum = 0u128;
            for d in 1..=day {
                sum += self.credited_nano(b.name(), d);
            }
            let want = u128::from(b.released_by_day(day)) * 1_000_000_000u128;
            if sum != want {
                return Err(LedgerError::OverEmission {
                    amount: sum as u64,
                    scheduled: want as u64,
                });
            }
        }
        Ok(())
    }
    /// Grant a note's spendable against its committed: the ledger side of
    /// the claim rail (claim.rs validates the leg; this marks it spent).
    pub fn grant_claim(
        &mut self,
        commitment: &[u8; 32],
        amount_nano: u64,
        nullifier: &[u8; 32],
        eligibility: &[u8; 32],
    ) -> Result<AccountId, LedgerError> {
        let id = AccountId(*commitment);
        let Some((holder, entry)) = self.accounts.get_mut(&id) else {
            return Err(LedgerError::AccountNotFound { account: id });
        };
        let AccountHolder::Note(note) = holder else {
            return Err(LedgerError::NotANote);
        };
        if &note.eligibility != eligibility {
            return Err(LedgerError::BadSignature);
        }
        if !self.nullifiers.insert(*nullifier) {
            return Err(LedgerError::NullifierSpent { nullifier: *nullifier });
        }
        if entry.spendable_nano + amount_nano > entry.committed_nano {
            return Err(LedgerError::Insufficient {
                amount: amount_nano,
                spendable: entry.committed_nano - entry.spendable_nano,
            });
        }
        entry.spendable_nano += amount_nano;
        Ok(id)
    }

    pub fn spend(&mut self, id: &AccountId, amount_nano: u64) -> Result<(), LedgerError> {
        let Some((_, entry)) = self.accounts.get_mut(id) else {
            return Err(LedgerError::AccountNotFound { account: *id });
        };
        if entry.spendable_nano < amount_nano {
            return Err(LedgerError::Insufficient {
                amount: amount_nano,
                spendable: entry.spendable_nano,
            });
        }
        entry.spendable_nano -= amount_nano;
        Ok(())
    }

    pub fn burn_unclaimed(&mut self, id: &AccountId, epoch: Epoch) -> Result<u64, LedgerError> {
        let Some((_, entry)) = self.accounts.get_mut(id) else {
            return Err(LedgerError::AccountNotFound { account: *id });
        };
        let remainder = entry
            .committed_nano
            .checked_sub(entry.spendable_nano)
            .ok_or(LedgerError::BurnExceedsCommitted)?;
        if remainder == 0 {
            return Ok(0);
        }
        entry.committed_nano = entry.spendable_nano;
        self.burns.push(BurnEvent { account: *id, amount_nano: remainder, epoch });
        Ok(remainder)
    }

    pub fn root(&self) -> [u8; 32] {
        let mut msg = Vec::with_capacity(8 + self.accounts.len() * 80);
        msg.extend_from_slice(&self.epoch.as_u64().to_le_bytes());
        for (id, (_, entry)) in &self.accounts {
            msg.extend_from_slice(id.as_bytes());
            msg.extend_from_slice(&entry.committed_nano.to_le_bytes());
            msg.extend_from_slice(&entry.spendable_nano.to_le_bytes());
        }
        *Hash256::concat(&EMISSION_ROOT, &msg).as_bytes()
    }

    /// The M1 audit per bucket: Σ committed over the bucket's accounts
    /// equals the schedule's released-by-day total.
    pub fn audit_bucket(
        &self,
        lek: &[u8; 32],
        bucket: &'static str,
        schedule_day: u64,
        schedule: &EmissionSchedule,
    ) -> Result<(), LedgerError> {
        let id = signed_account_id(lek, bucket);
        let Some((_, entry)) = self.accounts.get(&id) else {
            return Err(LedgerError::AccountNotFound { account: id });
        };
        let b = schedule
            .bucket(bucket)
            .ok_or(LedgerError::BucketMismatch { bucket })?;
        let want = u128::from(b.released_by_day(schedule_day)) * 1_000_000_000u128;
        if u128::from(entry.committed_nano) != want {
            return Err(LedgerError::OverEmission {
                amount: entry.committed_nano,
                scheduled: want as u64,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn test_raw(name: &'static str, kind: &'static str) -> nerv_core::params::EconomyBucket {
    nerv_core::params::EconomyBucket {
        name,
        kind,
        total_nerv: 0,
        share_permille: 0,
        account: "signed",
        term_days: 0,
        cliff_days: None,
        linear_days: None,
        year_one_days: None,
        quarters: None,
        quarter_days: None,
        ratio_num: None,
        ratio_den: None,
        window_days: None,
        burn_unclaimed: None,
    }
}

#[cfg(test)]
pub(crate) fn test_vk(seed: u64) -> VerifyingKey {
    let mut b = [0u8; 32];
    b[..8].copy_from_slice(&seed.to_le_bytes());
    *SigningKey::from_seed(&b).unwrap().verifying_key()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::claim::EmissionTree;
    use crate::schedule::EmissionSchedule;

    fn key(seed: u64) -> SigningKey {
        let mut b = [0u8; 32];
        b[..8].copy_from_slice(&seed.to_le_bytes());
        SigningKey::from_seed(&b).unwrap()
    }

    fn beacon() -> SigningKey {
        key(0xBEAC0N)
    }

    fn lek(seed: u64) -> [u8; 32] {
        let mut b = [0u8; 32];
        b[..8].copy_from_slice(&seed.to_le_bytes());
        derive_lek(&b)
    }

    #[test]
    fn derivations_are_pinned() {
        let seed = [7u8; 32];
        let lek = derive_lek(&seed);
        let ck = derive_claim_key(&seed);
        assert_eq!(lek, derive_lek(&seed));
        assert_ne!(lek, ck);
        // The literal formulas.
        let mut m = Vec::new();
        m.extend_from_slice(nerv_core::constants::EMISSION_LEK.as_bytes());
        m.extend_from_slice(&seed);
        assert_eq!(lek.as_slice(), blake3::hash(&m).as_slice());
        let mut m = Vec::new();
        m.extend_from_slice(nerv_core::constants::CLAIM_KEY.as_bytes());
        m.extend_from_slice(&seed);
        assert_eq!(ck.as_slice(), blake3::hash(&m).as_slice());

        let id = signed_account_id(&lek, "founder");
        assert_eq!(id, signed_account_id(&lek, "founder"));
        assert_ne!(id.as_bytes(), signed_account_id(&lek, "useful-work").as_bytes());
        assert_ne!(id.as_bytes(), signed_account_id(&ck, "founder").as_bytes());
        assert_eq!(AccountId::from_bytes(*id.as_bytes()), id);
        assert_eq!(id.as_hash().as_bytes(), id.as_bytes());

        let cm = note_commitment(&ck, "community", 500, &[9u8; 32]);
        assert_eq!(cm, note_commitment(&ck, "community", 500, &[9u8; 32]));
        assert_ne!(cm, note_commitment(&ck, "community", 501, &[9u8; 32]));
        assert_ne!(cm, note_commitment(&ck, "ecosystem", 500, &[9u8; 32]));
        assert_ne!(cm, note_commitment(&lek, "community", 500, &[9u8; 32]));

        let nf = claim_nullifier(&ck, "community");
        assert_eq!(nf, claim_nullifier(&ck, "community"));
        assert_ne!(nf, claim_nullifier(&ck, "ecosystem"));
        assert_ne!(nf, claim_nullifier(&lek, "community"));

        let el = eligibility_digest(&ck, "community", 500);
        assert_ne!(el, eligibility_digest(&ck, "community", 501));
        assert_ne!(el, nf);
    }

    #[test]
    fn credential_build_verify_and_tamper() {
        let b = beacon();
        let c = EmissionCredential::build(&b, lek(1), "founder", 1000, Epoch::from_u64(5)).unwrap();
        assert!(c.verify(b.verifying_key()));
        assert_eq!(c.message().len() > 32, true);

        let mut bad = c.clone();
        bad.amount_nano = 1001;
        assert!(!bad.verify(b.verifying_key()));
        let mut bad = c.clone();
        bad.epoch = Epoch::from_u64(6);
        assert!(!bad.verify(b.verifying_key()));
        let mut bad = c.clone();
        bad.lek = lek(2);
        assert!(!bad.verify(b.verifying_key()));
        let mut bad = c.clone();
        let mut sb = *bad.signature.as_bytes();
        sb[50] ^= 1;
        bad.signature = Signature::from_bytes(sb);
        assert!(!bad.verify(b.verifying_key()));

        let other = key(99);
        assert!(!c.verify(other.verifying_key()));
    }

    fn setup(schedule: &EmissionSchedule, epoch: u64) -> (EmissionLedger, SigningKey) {
        let mut ledger = EmissionLedger::new();
        ledger.set_epoch(Epoch::from_u64(epoch));
        (ledger, beacon())
    }

    #[test]
    fn credit_multi_account_bucket_totals() {
        let schedule = EmissionSchedule::genesis();
        let (mut ledger, b) = setup(&schedule, 1);
        let day1 = schedule.bucket("founder").unwrap().day_emission(1); // 0 (cliff)
        assert_eq!(day1, 0);

        // Day 361: the founder bucket emits; two holders split it 60/40.
        ledger.set_epoch(Epoch::from_u64(361));
        let total = schedule.bucket("founder").unwrap().day_emission(361) * 1_000_000_000u64;
        assert!(total > 0);
        let (a60, a40) = (total * 6 / 10, total - total * 6 / 10);
        let (l1, l2) = (lek(10), lek(11));
        ledger.add_signed(l1, "founder", *key(10).verifying_key());
        ledger.add_signed(l2, "founder", *key(11).verifying_key());
        let c1 = EmissionCredential::build(&b, l1, "founder", a60, Epoch::from_u64(361)).unwrap();
        let c2 = EmissionCredential::build(&b, l2, "founder", a40, Epoch::from_u64(361)).unwrap();
        ledger.credit(&schedule, b.verifying_key(), &c1).unwrap();
        ledger.credit(&schedule, b.verifying_key(), &c2).unwrap();
        assert_eq!(ledger.credited_nano("founder", 361), u128::from(total));
        assert_eq!(
            ledger.account(&signed_account_id(&l1, "founder")).unwrap().1.spendable_nano,
            a60
        );
        assert_eq!(
            ledger.account(&signed_account_id(&l2, "founder")).unwrap().1.spendable_nano,
            a40
        );

        // A third credential — any amount — exceeds the day total.
        let c3 = EmissionCredential::build(&b, lek(12), "founder", 1, Epoch::from_u64(361)).unwrap();
        // The account must exist for the credit path; register then over-emit.
        ledger.add_signed(lek(12), "founder", *key(12).verifying_key());
        assert!(matches!(
            ledger.credit(&schedule, b.verifying_key(), &c3),
            Err(LedgerError::OverEmission { .. })
        ));

        // The audit passes at 361 only when every emitting day is covered.
        assert!(ledger.audit_day(361, &schedule).is_err(), "days 1..360 uncredited");
        for d in 1..=361u64 {
            // Non-emitting days credit nothing (cliff) — the audit requires
            // 0 == 0, which holds; the failure above is day 361's partial
            // state only before c1/c2... recheck: c1+c2 completed 361.
            let _ = d;
        }
        // Days 1..=360 emit 0 and were never credited (0 == 0 ✓); day 361
        // is fully credited — so the audit passes:
        ledger.audit_day(361, &schedule).unwrap();
    }

    #[test]
    fn credit_rejections() {
        let schedule = EmissionSchedule::genesis();
        let (mut ledger, b) = setup(&schedule, 361);
        let l = lek(20);
        let id = ledger.add_signed(l, "founder", *key(20).verifying_key());
        let amt = schedule.bucket("founder").unwrap().day_emission(361) * 1_000_000_000u64;
        let c = EmissionCredential::build(&b, l, "founder", amt, Epoch::from_u64(361)).unwrap();

        // Wrong epoch.
        let (mut l2, _) = setup(&schedule, 300);
        l2.add_signed(l, "founder", *key(20).verifying_key());
        assert!(matches!(
            l2.credit(&schedule, b.verifying_key(), &c),
            Err(LedgerError::EpochMismatch { credential: 361, ledger: 300 })
        ));

        // Bad signature.
        let forged = EmissionCredential {
            signature: Signature::from_bytes(*Signature::from_bytes([0u8; nerv_crypto::mldsa::SIG_LEN]).as_bytes()),
            ..c.clone()
        };
        let _ = forged;
        let mut bad = c.clone();
        bad.bucket = "nonexistent-bucket";
        // The bucket lookup fails before the signature matters only after
        // verify — order: signature first. Forge properly:
        let impostor = key(77);
        let c_bad = EmissionCredential::build(&impostor, l, "founder", amt, Epoch::from_u64(361))
            .unwrap();
        assert!(matches!(
            ledger.credit(&schedule, b.verifying_key(), &c_bad),
            Err(LedgerError::BadSignature)
        ));

        // Unknown bucket (beacon-signed, but the schedule has no such bucket).
        let c_unk = EmissionCredential::build(&b, l, "no-such", amt, Epoch::from_u64(361)).unwrap();
        assert!(matches!(
            ledger.credit(&schedule, b.verifying_key(), &c_unk),
            Err(LedgerError::BucketMismatch { bucket: "no-such" })
        ));

        // Unknown account (valid credential, never registered).
        let c_ghost = EmissionCredential::build(&b, lek(99), "founder", 1, Epoch::from_u64(361))
            .unwrap();
        assert!(matches!(
            ledger.credit(&schedule, b.verifying_key(), &c_ghost),
            Err(LedgerError::AccountNotFound { .. })
        ));

        // Happy path, then the same account re-credited in the epoch.
        ledger.credit(&schedule, b.verifying_key(), &c).unwrap();
        let c_again = EmissionCredential::build(&b, l, "founder", 1, Epoch::from_u64(361)).unwrap();
        assert!(matches!(
            ledger.credit(&schedule, b.verifying_key(), &c_again),
            Err(LedgerError::AlreadyCredited { epoch: 361 })
        ));

        // A note account cannot take a signed credential.
        let ck = derive_claim_key(&[3u8; 32]);
        let cm = note_commitment(&ck, "community", 100, &[0u8; 32]);
        let note_id = ledger.add_note(
            crate::emission::NoteAccount {
                commitment: cm,
                eligibility: eligibility_digest(&ck, "community", 100),
            },
            100,
        );
        let _ = note_id;
        let c_note = EmissionCredential::build(&b, ck, "founder", 1, Epoch::from_u64(361)).unwrap();
        assert!(matches!(
            ledger.credit(&schedule, b.verifying_key(), &c_note),
            Err(LedgerError::AccountNotFound { .. })
        ));

        // Cliff days reject any amount (cap 0).
        let (mut l3, b3) = setup(&schedule, 100);
        let lc = lek(21);
        l3.add_signed(lc, "founder", *key(21).verifying_key());
        let c_cliff = EmissionCredential::build(&b3, lc, "founder", 1, Epoch::from_u64(100))
            .unwrap();
        assert!(matches!(
            l3.credit(&schedule, b3.verifying_key(), &c_cliff),
            Err(LedgerError::OverEmission { .. })
        ));
    }

    #[test]
    fn note_grant_spend_burn_and_root() {
        let mut ledger = EmissionLedger::new();
        ledger.set_epoch(Epoch::from_u64(50));
        let ck = derive_claim_key(&[5u8; 32]);
        let amount = 700u64;
        let eligibility = eligibility_digest(&ck, "community", amount);
        let commitment = note_commitment(&ck, "community", amount, &[1u8; 32]);
        let id = ledger.add_note(
            crate::emission::NoteAccount { commitment, eligibility },
            amount,
        );
        assert_eq!(ledger.account(&id).unwrap().1.committed_nano, amount);
        assert_eq!(ledger.account(&id).unwrap().1.spendable_nano, 0);

        let nullifier = claim_nullifier(&ck, "community");
        // Over-committed grant.
        assert!(matches!(
            ledger.grant_claim(&commitment, amount + 1, &nullifier, &eligibility),
            Err(LedgerError::Insufficient { .. })
        ));
        // Wrong eligibility.
        assert!(matches!(
            ledger.grant_claim(&commitment, 1, &nullifier, &[9u8; 32]),
            Err(LedgerError::BadSignature)
        ));
        // Unknown commitment.
        assert!(matches!(
            ledger.grant_claim(&[8u8; 32], 1, &nullifier, &eligibility),
            Err(LedgerError::AccountNotFound { .. })
        ));
        // Valid.
        ledger.grant_claim(&commitment, 300, &nullifier, &eligibility).unwrap();
        assert_eq!(ledger.account(&id).unwrap().1.spendable_nano, 300);
        // Nullifier replay.
        assert!(matches!(
            ledger.grant_claim(&commitment, 100, &nullifier, &eligibility),
            Err(LedgerError::NullifierSpent { .. })
        ));
        // A different nullifier can grant the remainder.
        let ck2 = derive_claim_key(&[6u8; 32]);
        let _ = ck2;
        // (Same note, same ck — the nullifier is per (ck, bucket); a second
        // grant of the same note is impossible by construction. The
        // remainder stays committed until the window burn.)

        // Spend.
        ledger.spend(&id, 300).unwrap();
        assert_eq!(ledger.account(&id).unwrap().1.spendable_nano, 0);
        assert!(matches!(
            ledger.spend(&id, 1),
            Err(LedgerError::Insufficient { amount: 1, spendable: 0 })
        ));
        assert!(matches!(
            ledger.spend(&AccountId::from_bytes([7u8; 32]), 1),
            Err(LedgerError::AccountNotFound { .. })
        ));

        // Window close: the unclaimed remainder burns.
        let burned = ledger.burn_unclaimed(&id, Epoch::from_u64(720)).unwrap();
        assert_eq!(burned, amount);
        assert_eq!(ledger.account(&id).unwrap().1.committed_nano, 0);
        assert_eq!(ledger.burns().len(), 1);
        assert_eq!(ledger.burns()[0].amount_nano, amount);
        assert_eq!(ledger.burn_unclaimed(&id, Epoch::from_u64(721)).unwrap(), 0);
        assert!(matches!(
            ledger.burn_unclaimed(&AccountId::from_bytes([7u8; 32]), Epoch::from_u64(1)),
            Err(LedgerError::AccountNotFound { .. })
        ));

        // Root: determinism and field sensitivity.
        let r = ledger.root();
        assert_eq!(r, ledger.root());
        let mut l2 = ledger.clone();
        l2.set_epoch(Epoch::from_u64(51));
        assert_ne!(r, l2.root());
        let mut l3 = ledger.clone();
        let lk = lek(30);
        l3.add_signed(lk, "founder", *key(30).verifying_key());
        assert_ne!(r, l3.root());
        // The empty ledger's root is the epoch framing alone.
        let mut empty = EmissionLedger::new();
        empty.set_epoch(Epoch::from_u64(50));
        let mut msg = Vec::new();
        msg.extend_from_slice(&50u64.to_le_bytes());
        assert_eq!(
            empty.root().as_slice(),
            blake3::hash(&[
                nerv_core::constants::EMISSION_ROOT.as_bytes().as_slice(),
                msg.as_slice()
            ].concat()).as_slice()
        );
    }

    #[test]
    fn audit_day_full_replay() {
        let schedule = EmissionSchedule::genesis();
        let (mut ledger, b) = setup(&schedule, 0);
        // Replay three days of the founder bucket across two holders.
        let (l1, l2) = (lek(40), lek(41));
        ledger.add_signed(l1, "founder", *key(40).verifying_key());
        ledger.add_signed(l2, "founder", *key(41).verifying_key());
        for day in 361..=363u64 {
            ledger.set_epoch(Epoch::from_u64(day));
            let total =
                schedule.bucket("founder").unwrap().day_emission(day) * 1_000_000_000u64;
            let (h1, h2) = (total / 2, total - total / 2);
            let c1 = EmissionCredential::build(&b, l1, "founder", h1, Epoch::from_u64(day))
                .unwrap();
            let c2 = EmissionCredential::build(&b, l2, "founder", h2, Epoch::from_u64(day))
                .unwrap();
            ledger.credit(&schedule, b.verifying_key(), &c1).unwrap();
            ledger.credit(&schedule, b.verifying_key(), &c2).unwrap();
        }
        // The audit demands EVERY time-driven bucket's full replay — the
        // others are uncredited, so this fails for them.
        assert!(ledger.audit_day(363, &schedule).is_err());
        // The founder bucket alone is consistent:
        let mut sum = 0u128;
        for d in 1..=363u64 {
            sum += ledger.credited_nano("founder", d);
        }
        assert_eq!(
            sum,
            u128::from(schedule.bucket("founder").unwrap().released_by_day(363)) * 1_000_000_000u128
        );
        // A synthetic two-bucket schedule audits clean end-to-end.
        let mk = |name: &'static str| nerv_core::params::EconomyBucket {
            share_permille: 500,
            cliff_days: Some(0),
            linear_days: Some(10),
            term_days: 10,
            total_nerv: 500,
            ..crate::test_raw(name, "linear-vesting")
        };
        let s2 = EmissionSchedule::from_params(&[mk("a"), mk("b")]).unwrap();
        let (mut l, bb) = setup(&s2, 0);
        for day in 1..=10u64 {
            l.set_epoch(Epoch::from_u64(day));
            for name in ["a", "b"] {
                let amt = s2.bucket(name).unwrap().day_emission(day) * 1_000_000_000u64;
                if amt == 0 {
                    continue;
                }
                let lk = lek(if name == "a" { 50 } else { 51 });
                l.add_signed(lek(50), "a", *key(50).verifying_key());
                l.add_signed(lek(51), "b", *key(51).verifying_key());
                let c = EmissionCredential::build(&bb, lk, name, amt, Epoch::from_u64(day))
                    .unwrap();
                l.credit(&s2, bb.verifying_key(), &c).unwrap();
            }
        }
        l.audit_day(10, &s2).unwrap();
        assert!(l.audit_day(9, &s2).is_err(), "day 10 was credited after");
    }
}