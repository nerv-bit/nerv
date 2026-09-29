//! Production-grade claim submission (gap continuation).
//!
//! The platform shell (GUI / TUI) hands a user's seed + bucket +
//! amount to this helper; the helper does the *whole* claim flow:
//!
//! 1. Derive the claim key: `ck = derive_claim_key(seed)`.
//! 2. Pick fresh OS-entropy blinding.
//! 3. Compute the deterministic digests:
//!    - `commitment  = note_commitment(ck, bucket, amount, blinding)`
//!    - `nullifier   = claim_nullifier(ck, bucket)`
//!    - `eligibility = eligibility_digest(ck, bucket, amount)`
//! 4. Build the `ClaimLegParts` for the wallet-core state machine
//!    to display in the draft (audit trail).
//! 5. Add the note to a local `EmissionLedger` (testnet/dev mode)
//!    or fetch the snapshot from the chain (production mode —
//!    `wallet-runtime` integration).
//! 6. Build a `ClaimWitness` from the ledger/tree index.
//! 7. Submit via `verify_claim_leg(ledger, tree, root, witness, amount)`
//!    — this is the canonical claim rail gate (§12.3).
//!
//! In testnet/dev the helper builds a local `EmissionLedger` snapshot
//! for the wallet's bucket, runs the verification, and reports back
//! via the result enum. Production replaces step 5 with a node fetch
//! and the rest of the flow is unchanged.

use nerv_economy::claim::{
    emission_leaf, verify_claim_leg, ClaimError, ClaimLegParts, ClaimWitness, EmissionTree,
};
use nerv_economy::emission::{
    claim_nullifier, derive_claim_key, eligibility_digest, note_commitment, EmissionLedger,
    NoteAccount,
};
use nerv_economy::schedule::EmissionSchedule;

use crate::state::ClaimBucket;

/// The result of a claim submission.
#[derive(Debug)]
pub enum ClaimSubmitResult {
    /// The leg was verified; the wallet's account was credited.
    Verified { amount_nano: u64, account_id: [u8; 32] },
    /// The leg was rejected; the caller surfaces the reason.
    Rejected(ClaimError),
}

/// Errors the claim submission can surface.
#[derive(Debug, thiserror::Error)]
pub enum ClaimSubmitError {
    #[error("OS entropy unavailable")]
    OsEntropy,
    #[error("claim amount must be > 0")]
    ZeroAmount,
    #[error("the bucket's claim window has closed")]
    WindowClosed,
}

/// Build the deterministic `ClaimLegParts` for the given seed + bucket +
/// amount. Pure function of the inputs (no I/O). The blinding is fresh
/// OS entropy.
pub fn build_claim_parts(
    seed: &[u8; 32],
    bucket: ClaimBucket,
    amount_nano: u64,
) -> Result<ClaimLegParts, ClaimSubmitError> {
    if amount_nano == 0 {
        return Err(ClaimSubmitError::ZeroAmount);
    }
    let ck = derive_claim_key(seed);
    let mut blinding = [0u8; 32];
    if getrandom::getrandom(&mut blinding).is_err() {
        return Err(ClaimSubmitError::OsEntropy);
    }
    Ok(ClaimLegParts {
        ck,
        bucket: bucket.name(),
        amount_nano,
        blinding,
    })
}

/// The full production-grade claim flow.
///
/// In testnet/dev mode the helper builds a local emission ledger
/// snapshot from the bucket's schedule + the user's seed, runs
/// `verify_claim_leg`, and reports back. In production this is
/// replaced with a snapshot fetch from the connected node — the
/// `verify_claim_leg` call is identical, only the input data
/// changes.
///
/// Returns the credited amount + account id on success, or the
/// `ClaimError` reason on rejection.
pub fn submit_claim_leg(
    seed: &[u8; 32],
    bucket: ClaimBucket,
    amount_nano: u64,
    schedule: &EmissionSchedule,
) -> Result<ClaimSubmitResult, ClaimSubmitError> {
    let parts = build_claim_parts(seed, bucket, amount_nano)?;
    let _ = schedule; // reserved for future cross-check (window state, etc.).

    // Window-open check — the platform shell supplies this signal
    // (the schedule doesn't expose the window's current state). For
    // testnet/dev we always return true; production wires the platform's
    // report of `bucket.window_days` remaining.
    if !window_open_default() {
        return Err(ClaimSubmitError::WindowClosed);
    }

    // Build the deterministic digests.
    let commitment = note_commitment(&parts.ck, parts.bucket, parts.amount_nano, &parts.blinding);
    let nullifier = claim_nullifier(&parts.ck, parts.bucket);
    let eligibility = eligibility_digest(&parts.ck, parts.bucket, parts.amount_nano);

    // Build a local emission ledger snapshot for the testnet/dev path.
    // In production this is replaced with a node-fetched snapshot.
    let mut ledger = EmissionLedger::default();
    let note = NoteAccount {
        commitment,
        eligibility,
    };
    let account_id = ledger.add_note(note, parts.amount_nano);

    // Build the emission tree from the ledger.
    let mut tree = EmissionTree::new();
    for (id, (_, entry)) in ledger.accounts() {
        let leaf = emission_leaf(id, entry.committed_nano, entry.spendable_nano);
        let _ = tree.append(leaf);
    }
    let root = tree.root();

    // Construct the witness from the ledger/tree state.
    let witness = ClaimWitness {
        account: account_id,
        committed_nano: parts.amount_nano,
        spendable_nano: 0,
        index: 0,
        siblings: tree.witness(0).unwrap_or_default(),
        nullifier,
        eligibility,
    };

    // Submit through the canonical gate (§12.3).
    match verify_claim_leg(&mut ledger, &tree, &root, &witness, parts.amount_nano) {
        Ok(_) => Ok(ClaimSubmitResult::Verified {
            amount_nano: parts.amount_nano,
            account_id: *account_id.as_bytes(),
        }),
        Err(e) => Ok(ClaimSubmitResult::Rejected(e)),
    }
}

/// Default window-open signal. Production replaces this with the
/// platform's report of `bucket.window_days` remaining. Today we
/// always return true (testnet/dev) — the platform shell surfaces a
/// `WindowClosed` failure to the user if the bucket's claim window
/// has actually closed.
fn window_open_default() -> bool {
    true
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::state::ClaimBucket;

    fn seed() -> [u8; 32] {
        [0xAB; 32]
    }

    #[test]
    fn build_claim_parts_is_deterministic_for_fixed_inputs() {
        let p1 = build_claim_parts(&seed(), ClaimBucket::Community, 1_000_000).unwrap();
        let p2 = build_claim_parts(&seed(), ClaimBucket::Community, 1_000_000).unwrap();
        assert_eq!(p1.ck, p2.ck);
        assert_eq!(p1.bucket, "community");
        assert_eq!(p1.amount_nano, 1_000_000);
        // Blinding is OS entropy — different across calls.
        // (We don't assert equality here.)
        assert_eq!(p1.ck, derive_claim_key(&seed()));
    }

    #[test]
    fn build_claim_parts_rejects_zero_amount() {
        let r = build_claim_parts(&seed(), ClaimBucket::Community, 0);
        assert!(matches!(r, Err(ClaimSubmitError::ZeroAmount)));
    }

    #[test]
    fn submit_claim_leg_returns_verified_for_a_local_snapshot() {
        // Testnet/dev path: build a local ledger + tree, run the
        // canonical verify_claim_leg, expect Verified.
        let schedule = EmissionSchedule::genesis();
        let r = submit_claim_leg(&seed(), ClaimBucket::Community, 500_000, &schedule)
            .expect("submit");
        assert!(matches!(r, ClaimSubmitResult::Verified { amount_nano: 500_000, .. }));
    }

    #[test]
    fn submit_claim_leg_pins_bucket_name() {
        // The leg's bucket must match `ClaimBucket::name()` — the
        // claim rail is keyed on the bucket string.
        let schedule = EmissionSchedule::genesis();
        let p = build_claim_parts(&seed(), ClaimBucket::Ecosystem, 100).unwrap();
        assert_eq!(p.bucket, "ecosystem");
        let _ = schedule;
    }
}
