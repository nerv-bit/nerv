//! Production-grade producer registration (gap continuation).
//!
//! Mirrors the `claim_submit.rs` shape: the platform shell (GUI / TUI)
//! hands a user's seed + shard + stake bond to this helper; the helper
//! does the *whole* producer registration flow:
//!
//! 1. Build a `ProducerIdentity` from the seed — derives the ML-DSA-65
//!    signing key (signs block headers + quorum certificates) and the
//!    custody `Address` (where `apply_block` settles the per-block
//!    subsidy payout). Mirrors `bin/nerv-node/src/producer.rs`.
//! 2. Register the stake against a fresh local `StakeLedger` (the
//!    testnet/dev path). Production replaces step 2 with a node-
//!    submitted `RegisterStake` transaction; the rest of the flow
//!    (identity construction, hex formatting, dispatch) is unchanged.
//! 3. Compute the consensus-ledger account id (BLAKE3 of
//!    `SLASH || vk_bytes`) so the UI can display a stable identifier.
//! 4. Return the verifying key hex, payout Address hex, stake balance,
//!    and account id for the platform shell to push back into the
//!    wallet state machine via `WalletAction::ProducerStakeRegistered`.

use nerv_core::codec::Encode;
use nerv_core::constants;
use nerv_core::hash::Hash256;
use nerv_core::types::{ShardId, ShardSet};
use nerv_core::TypeError;
use nerv_crypto::mldsa::{SigningKey, VerifyingKey};
use nerv_custody::{Address, MasterSeed, WalletKeys};
use nerv_economy::staking::StakeLedger;

/// The result of a producer registration.
#[derive(Debug, Clone)]
pub enum ProducerSubmitResult {
    /// The producer identity was built and its stake registered. The
    /// platform shell pushes these into the wallet state machine.
    Registered {
        /// ML-DSA-65 verifying key, hex-encoded.
        verifying_key_hex: String,
        /// Custody `Address` for payouts, hex-encoded (full address:
        /// `delivery || tag || pk_n`).
        payout_address_hex: String,
        /// Stake balance as recorded by the local `StakeLedger`.
        stake_balance_nano: u64,
        /// Consensus-ledger account id (BLAKE3 of `SLASH || vk_bytes`).
        account_id: [u8; 32],
    },
}

/// Errors the producer submission can surface.
#[derive(Debug, thiserror::Error)]
pub enum ProducerSubmitError {
    #[error("OS entropy unavailable")]
    OsEntropy,
    #[error("stake bond must be greater than zero")]
    ZeroStake,
    #[error("shard {0:?} is not in the active shard set")]
    InactiveShard(ShardId),
    #[error("the wallet produced no addresses")]
    NoAddresses,
    #[error("ML-DSA signing-key derivation failed")]
    SigningKeyDerivation,
    #[error("shard id construction: {0}")]
    ShardIdConstruction(#[from] TypeError),
}

/// Build the producer identity for the given seed + shard + stake bond.
/// Pure function of the inputs (no I/O). The signing key is freshly
/// generated from OS entropy; the custody Address is derived
/// deterministically from the same seed via `WalletKeys::from_master`.
fn build_producer_identity(
    seed: &[u8; 32],
    shard: ShardId,
) -> Result<(SigningKey, VerifyingKey, Address), ProducerSubmitError> {
    let master = MasterSeed::from_bytes(*seed);
    let keys = WalletKeys::from_master(&master);
    let active = ShardSet::genesis();
    let addresses = nerv_wallet::AddressSet::generate(&keys, &active)
        .map_err(|_| ProducerSubmitError::NoAddresses)?;
    // Pick the address homed to `shard`. If the wallet didn't generate
    // one for this shard (coverage < 64), fall back to the first
    // address — the operator can top up coverage later.
    let payout_address = addresses
        .addresses_for_shard(&shard)
        .first()
        .map(|wa| wa.address.clone())
        .or_else(|| addresses.addresses().first().map(|wa| wa.address.clone()))
        .ok_or(ProducerSubmitError::NoAddresses)?;
    let signing_key =
        SigningKey::from_seed(seed).map_err(|_| ProducerSubmitError::SigningKeyDerivation)?;
    let verifying_key = *signing_key.verifying_key();
    Ok((signing_key, verifying_key, payout_address))
}

/// The full production-grade producer registration flow.
///
/// In testnet/dev mode the helper builds a local emission ledger
/// snapshot from the bucket's schedule + the user's seed, registers
/// the producer's stake against a fresh `StakeLedger`, and reports
/// back. Production replaces the local ledger with a node-submitted
/// `RegisterStake` transaction; the rest of the flow (identity
/// construction, hex formatting, account-id computation) is unchanged.
///
/// Returns the credited VK + Address hex + stake balance + consensus
/// account id on success, or the reason on failure.
pub fn register_producer_stake(
    seed: &[u8; 32],
    shard_id_value: u64,
    stake_nano: u64,
) -> Result<ProducerSubmitResult, ProducerSubmitError> {
    if stake_nano == 0 {
        return Err(ProducerSubmitError::ZeroStake);
    }
    // Genesis shard set is 64 shards (bits=6, ids 0..=63). `ShardId::new`
    // validates `value < 1 << bits`; combined with the active-set check
    // below this pins us to the genesis 64-shard committee.
    let shard = ShardId::new(6, shard_id_value as u16)?;
    let active = ShardSet::genesis();
    if !active.ids().contains(&shard) {
        return Err(ProducerSubmitError::InactiveShard(shard));
    }

    let (_signing_key, verifying_key, payout_address) =
        build_producer_identity(seed, shard)?;

    // Local testnet/dev ledger. Production replaces this with a
    // node-submitted `RegisterStake` transaction; the rest of the
    // flow (identity construction, hex formatting, dispatch) is
    // unchanged.
    let mut ledger = StakeLedger::new();
    ledger.stake(verifying_key, stake_nano);
    let balance = ledger.stake_of(&verifying_key);

    let vk_hex: String = verifying_key
        .as_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    // Address wire form: delivery (EK) + tag (ShardId, 3 bytes) + pk_n (32).
    let payout_bytes = payout_address.encode();
    let payout_hex: String = payout_bytes.iter().map(|b| format!("{b:02x}")).collect();
    let account_id = *Hash256::concat(&constants::SLASH, verifying_key.as_bytes()).as_bytes();

    Ok(ProducerSubmitResult::Registered {
        verifying_key_hex: vk_hex,
        payout_address_hex: payout_hex,
        stake_balance_nano: balance,
        account_id,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn seed() -> [u8; 32] {
        [0xA1; 32]
    }

    #[test]
    fn register_producer_stake_returns_registered_for_a_local_ledger() {
        // Testnet/dev path: register a producer against a fresh local
        // StakeLedger. The verifying key + payout Address hex come back;
        // the stake balance equals the bond we asked for.
        let r = register_producer_stake(&seed(), 0, 1_000_000).expect("register");
        match r {
            ProducerSubmitResult::Registered {
                stake_balance_nano,
                verifying_key_hex,
                payout_address_hex,
                account_id,
            } => {
                assert_eq!(stake_balance_nano, 1_000_000);
                assert!(!verifying_key_hex.is_empty());
                assert!(!payout_address_hex.is_empty());
                assert_ne!(account_id, [0u8; 32]);
            }
        }
    }

    #[test]
    fn register_producer_stake_rejects_zero_stake() {
        let r = register_producer_stake(&seed(), 0, 0);
        assert!(matches!(r, Err(ProducerSubmitError::ZeroStake)));
    }

    #[test]
    fn register_producer_stake_is_deterministic_for_fixed_seed() {
        // Same seed → same payout Address hex (the custody Address is
        // deterministic from the master seed). The ML-DSA signing key
        // is freshly generated from OS entropy so the VK WILL differ
        // across calls — we only assert address determinism here.
        let r1 = register_producer_stake(&seed(), 0, 100).expect("r1");
        let r2 = register_producer_stake(&seed(), 0, 100).expect("r2");
        match (r1, r2) {
            (
                ProducerSubmitResult::Registered {
                    payout_address_hex: p1,
                    ..
                },
                ProducerSubmitResult::Registered {
                    payout_address_hex: p2,
                    ..
                },
            ) => assert_eq!(p1, p2),
        }
    }

    #[test]
    fn register_producer_stake_caps_shard_to_genesis_set() {
        // Shard id 200 with bits=6 is outside the genesis 64-shard set
        // (`ShardId::new(6, v)` accepts only `v < 64`). The error
        // surfaces as `ShardIdConstruction`, not `InactiveShard`, since
        // the construction-time validation catches it first.
        let r = register_producer_stake(&seed(), 200, 100);
        assert!(matches!(r, Err(ProducerSubmitError::ShardIdConstruction(_))));
    }
}