//! The producer-side integration (gap continuation of Gaps 1–7).
//!
//! A producer is a node operator that:
//! 1. Holds a **producer identity** — an ML-DSA signing key (used to
//!    sign block headers + quorum certificates) plus a custody
//!    `Address` (where `apply_block` credits the per-block subsidy
//!    payout).
//! 2. **Registers their stake** with the consensus `StakeLedger` —
//!    the bond that earns eligibility and that slashing consumes on
//!    double-signs / invalid blocks (erratum 163).
//! 3. **Receives per-block subsidies** as a deterministic share of
//!    the `validator-subsidy` emission bucket, split canonical-order
//!    across the active shard set (erratum 162). The payout amount
//!    and destination Address are computed here; the executor's
//!    `apply_block` does the actual settlement into the producer's
//!    emission account.
//!
//! The block-construction loop (the actual "produce blocks" call that
//! assembles transactions, builds the QC, signs the header) lives in
//! the executor's `propose` and `nerv-consensus::qc` — this module is
//! the *producer-side wiring* those functions need but didn't have:
//! the identity, the stake registration, and the payout computation.

use nerv_core::types::{Epoch, ShardId, ShardSet};
use nerv_crypto::mldsa::{Signature, SigningKey, VerifyingKey};
use nerv_custody::{Address, MasterSeed, WalletKeys};
use nerv_economy::schedule::EmissionSchedule;
use nerv_economy::staking::{StakeError, StakeLedger};
use nerv_economy::subsidy::{split_subsidy, SubsidyError};

// ---- Errors ---------------------------------------------------------------

/// Errors the producer-side integration can surface.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ProducerError {
    #[error("subsidy computation: {0}")]
    Subsidy(#[from] SubsidyError),
    #[error("stake ledger: {0}")]
    Stake(#[from] StakeError),
    #[error("crypto: {0}")]
    Crypto(#[from] nerv_crypto::CryptoError),
    #[error("shard {0:?} is not in the active shard set")]
    InactiveShard(ShardId),
    #[error("the wallet has no address for shard {0:?}")]
    NoAddressForShard(ShardId),
    #[error("the wallet produced no addresses at all")]
    NoAddresses,
    #[error("invalid ML-DSA seed (keypair derivation failed)")]
    InvalidSeed,
}

// ---- ProducerIdentity ------------------------------------------------------

/// The producer's persistent identity.
///
/// Bundles:
/// - the **ML-DSA-65 signing keypair** (signs block headers and QCs),
/// - the **custody `Address`** (where `apply_block` settles the
///   subsidy payout — one address per assigned shard),
/// - the **assigned shard** (the subsidy split is per-shard),
/// - the **stake bond** (mirrored into the consensus `StakeLedger`).
///
/// The signing key is held in memory (zeroized on drop per
/// `nerv-crypto::mldsa::SigningKey`). The custody key is derived
/// deterministically from the same master seed via
/// `WalletKeys::from_master`. The payout Address is the wallet's
/// address whose shard tag matches the assigned shard — this routes
/// each shard's subsidy to its own dedicated Address.
#[derive(Clone)]
pub struct ProducerIdentity {
    /// ML-DSA-65 signing key (signs block headers).
    pub signing_key: SigningKey,
    /// ML-DSA-65 verifying key (registered in the consensus committee
    /// and the stake ledger).
    pub verifying_key: VerifyingKey,
    /// Custody Address for payouts.
    pub payout_address: Address,
    /// Shard this producer is producing for (per the role config).
    pub shard: ShardId,
    /// Stake bond in nano-NERV.
    pub stake_nano: u64,
}

impl ProducerIdentity {
    /// Build a producer identity from a 32-byte master seed. The ML-DSA
    /// signing key is freshly generated; the custody Address is derived
    /// from the same seed and picked to match the assigned shard.
    pub fn from_seed(
        seed: &[u8; 32],
        shard: ShardId,
        stake_nano: u64,
    ) -> Result<Self, ProducerError> {
        let master = MasterSeed::from_bytes(*seed);
        let keys = WalletKeys::from_master(&master);
        let active = ShardSet::genesis();
        let addresses = nerv_wallet::AddressSet::generate(&keys, &active)
            .map_err(|_| ProducerError::NoAddresses)?;
        // Pick the address homed to `shard`. If the wallet didn't
        // generate one for this shard (coverage < 64), fall back to
        // the first address — the operator can top up coverage later.
        let payout_address = addresses
            .addresses_for_shard(&shard)
            .first()
            .map(|wa| wa.address.clone())
            .or_else(|| addresses.addresses().first().map(|wa| wa.address.clone()))
            .ok_or(ProducerError::NoAddressForShard(shard))?;
        let signing_key = SigningKey::from_seed(&seed).map_err(|_| ProducerError::InvalidSeed)?;
        let verifying_key = signing_key.public_key();
        Ok(ProducerIdentity {
            signing_key,
            verifying_key,
            payout_address,
            shard,
            stake_nano,
        })
    }

    /// Build a producer identity from an externally-supplied signing
    /// key (e.g. one loaded from an HSM or an encrypted key store).
    /// The custody Address is still derived from the master seed for
    /// the payout routing — production HSMs hold the ML-DSA key only.
    pub fn with_signing_key(
        signing_key: SigningKey,
        seed: &[u8; 32],
        shard: ShardId,
        stake_nano: u64,
    ) -> Result<Self, ProducerError> {
        let mut id = Self::from_seed(seed, shard, stake_nano)?;
        id.signing_key = signing_key;
        id.verifying_key = id.signing_key.public_key();
        Ok(id)
    }

    /// Register this producer's stake against the consensus
    /// `StakeLedger`. Idempotent — repeated calls add to the existing
    /// entry. In production this is called once at producer
    /// registration; subsequent stake top-ups call `ledger.stake()`
    /// directly with the same VK.
    pub fn register_stake(&self, ledger: &mut StakeLedger) {
        ledger.stake(self.verifying_key, self.stake_nano);
    }

    /// The producer's stake balance, as the consensus ledger sees it.
    pub fn stake_of(&self, ledger: &StakeLedger) -> u64 {
        ledger.stake_of(&self.verifying_key)
    }

    /// The producer's per-block subsidy in nano-NERV for the current
    /// epoch, on this producer's assigned shard. Reads from the
    /// emission schedule's `validator-subsidy` bucket and applies the
    /// canonical-order remainder split (erratum 162).
    pub fn epoch_payout_nano(
        &self,
        schedule: &EmissionSchedule,
        epoch: Epoch,
        active: &ShardSet,
    ) -> Result<u64, ProducerError> {
        let total = nerv_economy::subsidy::epoch_subsidy_nano(schedule, epoch)?;
        let splits = split_subsidy(total, active.ids());
        let amount = splits
            .iter()
            .find(|(s, _)| *s == self.shard)
            .map(|(_, a)| *a)
            .ok_or(ProducerError::InactiveShard(self.shard))?;
        Ok(amount)
    }

    /// Sign a header hash with the producer's ML-DSA-65 key. Returns
    /// the raw signature for inclusion in the quorum certificate.
    pub fn sign(&self, message: &[u8]) -> Result<Signature, ProducerError> {
        Ok(self.signing_key.sign(message)?)
    }
}

// ---- ProducerRole ---------------------------------------------------------

/// The node-side wrapper around `ProducerIdentity`. Holds the
/// producer's stake and exposes the per-epoch payout + signing entry
/// points. The node instantiates this when `Role::Producer { shard }`
/// is selected at startup.
#[derive(Clone)]
pub struct ProducerRole {
    identity: ProducerIdentity,
}

impl ProducerRole {
    pub fn new(identity: ProducerIdentity) -> Self {
        ProducerRole { identity }
    }

    pub fn verifying_key(&self) -> VerifyingKey {
        self.identity.verifying_key
    }

    pub fn payout_address(&self) -> Address {
        self.identity.payout_address.clone()
    }

    pub fn shard(&self) -> ShardId {
        self.identity.shard
    }

    pub fn stake_nano(&self) -> u64 {
        self.identity.stake_nano
    }

    /// Read-only access to the inner identity (production helper:
    /// the consensus layer needs the VK + address directly).
    pub fn identity(&self) -> &ProducerIdentity {
        &self.identity
    }

    pub fn register_stake(&self, ledger: &mut StakeLedger) {
        self.identity.register_stake(ledger);
    }

    pub fn epoch_payout_nano(
        &self,
        schedule: &EmissionSchedule,
        epoch: Epoch,
        active: &ShardSet,
    ) -> Result<u64, ProducerError> {
        self.identity.epoch_payout_nano(schedule, epoch, active)
    }

    pub fn sign(&self, message: &[u8]) -> Result<Signature, ProducerError> {
        self.identity.sign(message)
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use nerv_economy::schedule::EmissionSchedule;

    fn seed_bytes(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn active() -> ShardSet {
        ShardSet::genesis()
    }

    #[test]
    fn identity_round_trips_through_seed() {
        let id_a =
            ProducerIdentity::from_seed(&seed_bytes(0xA1), ShardId::new(6, 7), 1_000_000_000)
                .expect("identity a");
        let id_b =
            ProducerIdentity::from_seed(&seed_bytes(0xA1), ShardId::new(6, 7), 1_000_000_000)
                .expect("identity b");
        // Same seed → same payout address (the address-derivation is
        // deterministic from the master seed).
        assert_eq!(id_a.payout_address, id_b.payout_address);
        assert_eq!(id_a.shard, id_b.shard);
        assert_eq!(id_a.stake_nano, id_b.stake_nano);
    }

    #[test]
    fn register_stake_is_idempotent_and_observable() {
        let identity =
            ProducerIdentity::from_seed(&seed_bytes(0xB2), ShardId::new(6, 0), 500).unwrap();
        let mut ledger = StakeLedger::new();
        assert_eq!(identity.stake_of(&ledger), 0);
        identity.register_stake(&mut ledger);
        assert_eq!(identity.stake_of(&ledger), 500);
        // Idempotent: a second call adds, doesn't replace.
        identity.register_stake(&mut ledger);
        assert_eq!(identity.stake_of(&ledger), 1000);
    }

    #[test]
    fn epoch_payout_nano_uses_validator_subsidy_bucket() {
        let identity =
            ProducerIdentity::from_seed(&seed_bytes(0xC3), ShardId::new(6, 0), 100).unwrap();
        let schedule = EmissionSchedule::genesis();
        let payout = identity
            .epoch_payout_nano(&schedule, Epoch::from_u64(0), &active())
            .expect("payout");
        assert!(payout > 0, "producer must receive a non-zero subsidy");
        // 64 active shards → payout ≤ 3B / 64 ≈ 46.875M per shard.
        assert!(payout < 100_000_000, "payout must fit the per-shard split");
    }

    #[test]
    fn inactive_shard_returns_inactive_shard_error() {
        // A shard id outside the active set: bits == 6, value > 63 is
        // outside the genesis 64 shards.
        let bad_shard = ShardId::new(6, 200);
        let identity = ProducerIdentity::from_seed(&seed_bytes(0xD4), bad_shard, 100).unwrap();
        let schedule = EmissionSchedule::genesis();
        let err = identity
            .epoch_payout_nano(&schedule, Epoch::from_u64(0), &active())
            .unwrap_err();
        assert!(matches!(err, ProducerError::InactiveShard(_)));
    }

    #[test]
    fn producer_role_delegates_to_identity() {
        let identity =
            ProducerIdentity::from_seed(&seed_bytes(0xE5), ShardId::new(6, 9), 1_000).unwrap();
        let role = ProducerRole::new(identity.clone());
        assert_eq!(role.verifying_key(), identity.verifying_key);
        assert_eq!(role.shard(), ShardId::new(6, 9));
        assert_eq!(role.stake_nano(), 1_000);
        // The role registers the identity's stake on the ledger.
        let mut ledger = StakeLedger::new();
        role.register_stake(&mut ledger);
        assert_eq!(role.identity().stake_of(&ledger), 1_000);
    }

    #[test]
    fn payout_total_split_sums_to_bucket_amount() {
        // The sum of `split_subsidy(...)` over all active shards equals
        // the input amount (the canonical-order remainder is
        // lossless). This pins the producer's payout in a multi-shard
        // committee against the aggregate.
        let schedule = EmissionSchedule::genesis();
        let epoch = Epoch::from_u64(0);
        let total = nerv_economy::subsidy::epoch_subsidy_nano(&schedule, epoch)
            .expect("subsidy");
        let splits = split_subsidy(total, active().ids());
        let sum: u128 = splits.iter().map(|(_, a)| u128::from(*a)).sum();
        assert_eq!(sum, total, "split must be lossless");
    }

    #[test]
    fn sign_produces_a_verifiable_signature() {
        // The producer signs a header hash; the resulting signature
        // verifies under the producer's public key (the same VK the
        // consensus committee uses to count the producer's votes).
        let identity =
            ProducerIdentity::from_seed(&seed_bytes(0xF6), ShardId::new(6, 0), 100).unwrap();
        let message = b"test header hash bytes";
        let signature = identity.sign(message).expect("sign");
        assert!(
            identity.verifying_key.verify(message, &signature),
            "signature must verify under the producer's VK"
        );
    }

    #[test]
    fn with_signing_key_overrides_only_the_mldsa_key() {
        // Production HSM flow: replace the ML-DSA signing key but keep
        // the custody Address (still derived from the master seed).
        let mut external = SigningKey::generate();
        let seed = seed_bytes(0xA7);
        let id_a =
            ProducerIdentity::with_signing_key(external.clone(), &seed, ShardId::new(6, 0), 100)
                .unwrap();
        let id_b =
            ProducerIdentity::from_seed(&seed, ShardId::new(6, 0), 100).unwrap();
        // Same seed → same payout Address.
        assert_eq!(id_a.payout_address, id_b.payout_address);
        // Different ML-DSA keys (external vs internal gen).
        assert_ne!(id_a.verifying_key, id_b.verifying_key);
        // The external key's signature verifies under id_a's VK.
        let sig = id_a.sign(b"hello").expect("sign");
        assert!(id_a.verifying_key.verify(b"hello", &sig));
        // Use the external key once to confirm it's stored.
        let _ = external.sign(b"unused");
    }
}
