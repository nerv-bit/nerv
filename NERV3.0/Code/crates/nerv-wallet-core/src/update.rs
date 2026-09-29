//! The pure update function (erratum 200): (state, action) →
//! (state', events). This is the single place where wallet business
//! rules are enforced. The same function runs on every platform.

use crate::action::{SendStage, WalletAction};
use crate::state::{
    ClaimBucket, ClaimDraft, ClaimError, ClaimStage, Direction, DraftError, HistoryEntry,
    NotificationLevel, ProducerDraft, ProducerError, ProducerState, Screen, SyncStatus,
    WalletState,
};

/// Events emitted by the update (side-effect requests for the platform shell).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WalletEvent {
    /// The platform shell should generate a new wallet from OS entropy
    /// and call back with `ImportSeed`.
    RequestNewWallet,
    /// The platform shell should copy the given text to the clipboard.
    CopyToClipboard(String),
    /// The platform shell should start the send pipeline with the
    /// given draft parameters.
    StartSendPipeline { recipient_hex: String, amount_nano: u64, fee_nano: u64 },
    /// The platform shell should derive the `ClaimLegParts` from the
    /// user's seed, fetch the current emission-tree witness for the
    /// bucket, and submit the leg via `verify_claim_leg`. The shell
    /// owns the seed and the wallet does not touch it directly.
    StartClaimPipeline {
        bucket: ClaimBucket,
        amount_nano: u64,
    },
    /// The state changed and the UI should re-render.
    Redraw,
    /// The platform shell should build a `ProducerIdentity` from the
    /// given seed + shard + stake bond, register the stake against the
    /// consensus `StakeLedger`, and report back via
    /// `WalletAction::ProducerStakeRegistered` /
    /// `ProducerRegistrationFailed`.
    RegisterProducerStake {
        seed: [u8; 32],
        shard: u64,
        stake_nano: u64,
    },
}

/// The pure update function. Returns the new state and the events
/// the platform shell should act on.
pub fn update(state: &mut WalletState, action: WalletAction) -> Vec<WalletEvent> {
    let mut events = Vec::new();

    match action {
        // === Lifecycle ===
        WalletAction::GenerateWallet => {
            events.push(WalletEvent::RequestNewWallet);
        }
        WalletAction::ImportSeed(seed_hex) => {
            match parse_seed(&seed_hex) {
                Ok(seed) => {
                    let keys = nerv_custody::WalletKeys::from_master(
                        &nerv_custody::MasterSeed::from_bytes(seed),
                    );
                    let active = nerv_core::types::ShardSet::genesis();
                    match nerv_wallet::AddressSet::generate(&keys, &active) {
                        Ok(addrs) => {
                            state.seed = Some(seed);
                            state.keys = Some(keys);
                            state.addresses = Some(addrs);
                            state.sync = SyncStatus::Scanning { progress_permille: 0 };
                            state.screen = Screen::Dashboard;
                            state.notify(NotificationLevel::Success, "Wallet loaded".into());
                            events.push(WalletEvent::Redraw);
                        }
                        Err(e) => {
                            state.notify(NotificationLevel::Error, format!("Failed to generate addresses: {e}"));
                        }
                    }
                }
                Err(msg) => {
                    state.notify(NotificationLevel::Error, msg);
                }
            }
        }
        WalletAction::Lock => {
            state.seed = None;
            state.keys = None;
            state.addresses = None;
            state.notes = nerv_wallet::scan::WalletNoteSet::new();
            state.history.clear();
            state.draft = None;
            state.claim_draft = None;
            state.producer_draft = None;
            // The producer state is wiped on Lock — the live state is
            // tied to the wallet's seed, so once the seed is dropped
            // there is no way to reauthenticate as the producer.
            state.producer_state = ProducerState::default();
            state.sync = SyncStatus::Locked;
            state.screen = Screen::Dashboard;
            state.notify(NotificationLevel::Info, "Wallet locked".into());
            events.push(WalletEvent::Redraw);
        }
        WalletAction::Quit => {
            state.quitting = true;
            events.push(WalletEvent::Redraw);
        }

        // === Navigation ===
        WalletAction::Navigate(screen) => {
            if state.screen != screen {
                state.screen = screen;
                if screen == Screen::Send && state.draft.is_none() {
                    state.draft = Some(crate::state::TransactionDraft {
                        fee_nano: 1000,
                        expiry_height: state.chain_height.saturating_add(500),
                        ..Default::default()
                    });
                }
                events.push(WalletEvent::Redraw);
            }
        }
        WalletAction::NextTab => {
            let next = Screen::from_index(state.screen.index() + 1);
            state.screen = next;
            events.push(WalletEvent::Redraw);
        }
        WalletAction::PrevTab => {
            let prev = Screen::from_index(state.screen.index() + Screen::ALL.len() - 1);
            state.screen = prev;
            events.push(WalletEvent::Redraw);
        }

        // === Send flow ===
        WalletAction::StartSend => {
            state.screen = Screen::Send;
            state.draft = Some(crate::state::TransactionDraft {
                fee_nano: 1000,
                expiry_height: state.chain_height.saturating_add(500),
                ..Default::default()
            });
            events.push(WalletEvent::Redraw);
        }
        WalletAction::SetRecipient(hex) => {
            if let Some(draft) = &mut state.draft {
                draft.recipient_hex = hex;
                let balance = state.balance_nano();
                let synced = matches!(state.sync, SyncStatus::Synced { .. });
                draft.validate(balance, synced);
            }
            events.push(WalletEvent::Redraw);
        }
        WalletAction::SetAmount(amount) => {
            if let Some(draft) = &mut state.draft {
                draft.amount_nano = amount;
                let balance = state.balance_nano();
                let synced = matches!(state.sync, SyncStatus::Synced { .. });
                draft.validate(balance, synced);
            }
            events.push(WalletEvent::Redraw);
        }
        WalletAction::SetFee(fee) => {
            if let Some(draft) = &mut state.draft {
                draft.fee_nano = fee;
                let balance = state.balance_nano();
                let synced = matches!(state.sync, SyncStatus::Synced { .. });
                draft.validate(balance, synced);
            }
            events.push(WalletEvent::Redraw);
        }
        WalletAction::ConfirmSend => {
            if let Some(draft) = &mut state.draft {
                let balance = state.balance_nano();
                let synced = matches!(state.sync, SyncStatus::Synced { .. });
                draft.validate(balance, synced);
                if draft.is_ready() {
                    state.notify(NotificationLevel::Info, "Ready to sign".into());
                }
            }
            events.push(WalletEvent::Redraw);
        }
        WalletAction::SignAndSend => {
            if let Some(draft) = state.draft.take() {
                if draft.is_ready() {
                    state.notify(NotificationLevel::Info, SendStage::Constructing.label().into());
                    events.push(WalletEvent::StartSendPipeline {
                        recipient_hex: draft.recipient_hex,
                        amount_nano: draft.amount_nano,
                        fee_nano: draft.fee_nano,
                    });
                }
            }
            events.push(WalletEvent::Redraw);
        }
        WalletAction::CancelSend => {
            state.draft = None;
            state.screen = Screen::Dashboard;
            events.push(WalletEvent::Redraw);
        }

        // === Claim flow ===
        WalletAction::StartClaim => {
            state.screen = Screen::Claim;
            state.claim_draft = Some(ClaimDraft::default());
            state.notify(
                NotificationLevel::Info,
                "Pick a claim bucket and amount".into(),
            );
            events.push(WalletEvent::Redraw);
        }
        WalletAction::SetClaimBucket(bucket) => {
            if let Some(draft) = &mut state.claim_draft {
                draft.bucket = Some(bucket);
                let window_open = matches!(
                    state.sync,
                    SyncStatus::Synced { .. } | SyncStatus::Scanning { .. }
                );
                draft.validate(window_open);
            }
            events.push(WalletEvent::Redraw);
        }
        WalletAction::SetClaimAmount(amount) => {
            if let Some(draft) = &mut state.claim_draft {
                draft.amount_nano = amount;
                let window_open = matches!(
                    state.sync,
                    SyncStatus::Synced { .. } | SyncStatus::Scanning { .. }
                );
                draft.validate(window_open);
            }
            events.push(WalletEvent::Redraw);
        }
        WalletAction::SignAndClaim => {
            if let Some(draft) = state.claim_draft.take() {
                if let Some(bucket) = draft.bucket {
                    if draft.is_ready() {
                        state.notify(
                            NotificationLevel::Info,
                            ClaimStage::Deriving.label().into(),
                        );
                        events.push(WalletEvent::StartClaimPipeline {
                            bucket,
                            amount_nano: draft.amount_nano,
                        });
                    } else {
                        // Validation failed: bounce the draft back to the
                        // UI so the user sees the error.
                        state.claim_draft = Some(draft);
                    }
                } else {
                    // No bucket picked — bounce the draft back.
                    state.claim_draft = Some(draft);
                    state.notify(
                        NotificationLevel::Warning,
                        ClaimError::NoBucket.message(),
                    );
                }
            }
            events.push(WalletEvent::Redraw);
        }
        WalletAction::CancelClaim => {
            state.claim_draft = None;
            state.screen = Screen::Dashboard;
            events.push(WalletEvent::Redraw);
        }

        // === Producer flow ===
        WalletAction::StartProducer => {
            state.screen = Screen::Producer;
            let mut draft = state
                .producer_draft
                .take()
                .unwrap_or_default();
            // Re-validate so the validation reflects the draft's
            // current state (the previous validation may be stale if
            // the user filled some fields, navigated away, and came
            // back).
            draft.validate(None);
            state.producer_draft = Some(draft);
            state.notify(
                NotificationLevel::Info,
                "Pick a seed, shard, and stake bond".into(),
            );
            events.push(WalletEvent::Redraw);
        }
        WalletAction::SetProducerSeed(hex) => {
            if let Some(draft) = &mut state.producer_draft {
                match parse_seed(&hex) {
                    Ok(seed) => {
                        draft.seed = Some(seed);
                    }
                    Err(_) => {
                        // Invalid hex — clear any previous seed, leave
                        // the validation to surface `SeedInvalid` below.
                        draft.seed = None;
                    }
                }
                draft.validate(Some(&hex));
            }
            events.push(WalletEvent::Redraw);
        }
        WalletAction::SetProducerStake(stake) => {
            if let Some(draft) = &mut state.producer_draft {
                draft.stake_nano = stake;
                draft.validate(None);
            }
            events.push(WalletEvent::Redraw);
        }
        WalletAction::SetProducerShard(shard) => {
            if let Some(draft) = &mut state.producer_draft {
                // Cap at the genesis shard count (64 shards, ids 0..=63).
                if shard < 64 {
                    draft.shard = Some(shard);
                } else {
                    draft.shard = None;
                }
                draft.validate(None);
            }
            events.push(WalletEvent::Redraw);
        }
        WalletAction::RegisterProducerStake => {
            if let Some(draft) = state.producer_draft.take() {
                if !draft.is_ready() {
                    // Validation failed — bounce the draft back to the
                    // UI so the user sees the error.
                    state.producer_draft = Some(draft);
                    state.notify(
                        NotificationLevel::Warning,
                        draft
                            .validation
                            .error
                            .as_ref()
                            .map(ProducerError::message)
                            .unwrap_or_else(|| "Producer draft is not ready".into()),
                    );
                    events.push(WalletEvent::Redraw);
                    return events;
                }
                // Resolve the seed: explicit producer seed wins;
                // otherwise fall back to the wallet's unlocked seed.
                let resolved_seed = draft.seed.or(state.seed);
                let Some(seed) = resolved_seed else {
                    // Wallet is locked AND no explicit seed — bounce.
                    state.producer_draft = Some(draft);
                    state.notify(
                        NotificationLevel::Warning,
                        ProducerError::NoSeed.message(),
                    );
                    events.push(WalletEvent::Redraw);
                    return events;
                };
                let shard = draft.shard.unwrap_or(0);
                state.notify(
                    NotificationLevel::Info,
                    format!(
                        "Registering producer stake: {} nano on shard {}",
                        draft.stake_nano, shard
                    ),
                );
                events.push(WalletEvent::RegisterProducerStake {
                    seed,
                    shard,
                    stake_nano: draft.stake_nano,
                });
            }
            events.push(WalletEvent::Redraw);
        }
        WalletAction::CancelProducer => {
            state.producer_draft = None;
            state.screen = Screen::Dashboard;
            events.push(WalletEvent::Redraw);
        }
        WalletAction::ProducerStakeRegistered {
            verifying_key_hex,
            payout_address_hex,
            stake_balance_nano,
            account_id: _account_id,
        } => {
            // Populate the producer-state snapshot. The node's role
            // string + epoch payout are filled in by future
            // `ProducerStateRefreshed` events as the node reports
            // them; we initialize the basics here.
            state.producer_state = ProducerState {
                verifying_key_hex: verifying_key_hex.clone(),
                payout_address_hex: payout_address_hex.clone(),
                stake_balance_nano,
                total_payouts_nano: 0,
                epoch_payout_nano: 0,
                assigned_shard: state
                    .producer_draft
                    .as_ref()
                    .and_then(|d| d.shard)
                    .unwrap_or(0),
                node_role: "Producer".into(),
                stake_registered: true,
            };
            state.producer_draft = None;
            state.screen = Screen::Dashboard;
            state.notify(
                NotificationLevel::Success,
                format!(
                    "Producer registered: {} NERV bonded",
                    stake_balance_nano as f64 / 1e9
                ),
            );
            events.push(WalletEvent::Redraw);
        }
        WalletAction::ProducerRegistrationFailed(reason) => {
            state.producer_draft = None;
            state.screen = Screen::Dashboard;
            state.notify(
                NotificationLevel::Error,
                format!("Producer registration failed: {reason}"),
            );
            events.push(WalletEvent::Redraw);
        }
        WalletAction::ProducerStateRefreshed(snapshot) => {
            state.producer_state = snapshot;
            events.push(WalletEvent::Redraw);
        }

        // === Receive ===
        WalletAction::CopyAddress => {
            if let Some(hex) = state.primary_address_hex() {
                events.push(WalletEvent::CopyToClipboard(hex.clone()));
                state.notify(NotificationLevel::Success, "Address copied".into());
            }
        }
        WalletAction::NewAddress => {
            // In production: ensure a fresh address on a different shard.
            state.notify(NotificationLevel::Info, "Address generated".into());
            events.push(WalletEvent::Redraw);
        }

        // === History ===
        WalletAction::SelectHistoryEntry(idx) => {
            let _ = idx; // The UI handles selection rendering.
            events.push(WalletEvent::Redraw);
        }
        WalletAction::FilterHistory(dir) => {
            let _ = dir; // The UI handles filtering.
            events.push(WalletEvent::Redraw);
        }

        // === Settings ===
        WalletAction::SetCoverage(coverage) => {
            let _ = coverage;
            state.notify(NotificationLevel::Info, "Coverage updated".into());
            events.push(WalletEvent::Redraw);
        }
        WalletAction::ToggleFeeBucketing => {
            state.notify(NotificationLevel::Info, "Fee bucketing toggled".into());
            events.push(WalletEvent::Redraw);
        }

        // === Notifications ===
        WalletAction::DismissNotification => {
            state.dismiss_notification();
            events.push(WalletEvent::Redraw);
        }

        // === External events ===
        WalletAction::ChainHeightAdvanced(new_height) => {
            if new_height > state.chain_height {
                state.chain_height = new_height;
                if matches!(state.sync, SyncStatus::Scanning { .. }) {
                    state.sync = SyncStatus::Synced { height: new_height };
                }
                // Re-validate the draft (the balance may have changed).
                if let Some(draft) = &mut state.draft {
                    let balance = state.balance_nano();
                    let synced = matches!(state.sync, SyncStatus::Synced { .. });
                    draft.validate(balance, synced);
                }
                events.push(WalletEvent::Redraw);
            }
        }
        WalletAction::PeerCountChanged(count) => {
            state.peer_count = count;
            if count == 0 && matches!(state.sync, SyncStatus::Synced { .. }) {
                state.sync = SyncStatus::Disconnected;
                state.notify(NotificationLevel::Warning, "Disconnected from peers".into());
            } else if count > 0 && matches!(state.sync, SyncStatus::Disconnected) {
                state.sync = SyncStatus::Synced { height: state.chain_height };
            }
            events.push(WalletEvent::Redraw);
        }
        WalletAction::NoteReceived(note) => {
            let value = note.opening.value;
            let height = state.chain_height;
            state.notes.insert(note);
            state.history.push_front(HistoryEntry {
                txid: nerv_core::types::TxId::from_hash(
                    nerv_core::hash::Hash256::from_bytes(note.nullifier.as_bytes().to_owned().try_into().unwrap_or([0u8; 32])),
                ),
                direction: Direction::Incoming,
                amount_nano: value,
                fee_nano: 0,
                height,
                confirmations: 1,
            });
            state.notify(NotificationLevel::Success, format!("Received {value} nano"));
            events.push(WalletEvent::Redraw);
        }
        WalletAction::TransactionConfirmed { txid, height } => {
            for entry in state.history.iter_mut() {
                if entry.txid == txid {
                    entry.confirmations = state.chain_height.saturating_sub(height).max(1);
                }
            }
            events.push(WalletEvent::Redraw);
        }
        WalletAction::SendProgress(stage) => {
            state.notify(NotificationLevel::Info, stage.label().into());
            events.push(WalletEvent::Redraw);
        }
        WalletAction::SendFailed(reason) => {
            state.notify(NotificationLevel::Error, format!("Send failed: {reason}"));
            events.push(WalletEvent::Redraw);
        }
        WalletAction::SendSucceeded(txid) => {
            state.history.push_front(HistoryEntry {
                txid,
                direction: Direction::Outgoing,
                amount_nano: state.draft.as_ref().map_or(0, |d| d.amount_nano),
                fee_nano: state.draft.as_ref().map_or(0, |d| d.fee_nano),
                height: state.chain_height,
                confirmations: 0,
            });
            state.draft = None;
            state.screen = Screen::Dashboard;
            state.notify(NotificationLevel::Success, "Transaction sent".into());
            events.push(WalletEvent::Redraw);
        }
        WalletAction::ClaimProgress(stage) => {
            state.notify(NotificationLevel::Info, stage.label().into());
            events.push(WalletEvent::Redraw);
        }
        WalletAction::ClaimFailed(reason) => {
            state.claim_draft = None;
            state.screen = Screen::Dashboard;
            state.notify(
                NotificationLevel::Error,
                format!("Claim failed: {reason}"),
            );
            events.push(WalletEvent::Redraw);
        }
        WalletAction::ClaimSucceeded { bucket, amount_nano } => {
            // Push a synthetic history entry. The emission rail doesn't
            // emit a txid (the nullifier is the dedup key, but the
            // emission-side receipt carries a derivation tag instead).
            // We use `Hash256::from_bytes(bucket.as_bytes() ++ amount)`
            // as a stable receipt id for the UI; production wires a real
            // receipt id from the registry's `verify_claim_leg` return.
            let mut receipt_bytes = [0u8; 32];
            let bucket_name = bucket.name().as_bytes();
            let copy_len = bucket_name.len().min(16);
            receipt_bytes[..copy_len].copy_from_slice(&bucket_name[..copy_len]);
            receipt_bytes[16..24].copy_from_slice(&amount_nano.to_le_bytes());
            state.history.push_front(HistoryEntry {
                txid: nerv_core::types::TxId::from_hash(
                    nerv_core::hash::Hash256::from_bytes(receipt_bytes),
                ),
                direction: Direction::Incoming,
                amount_nano,
                fee_nano: 0,
                height: state.chain_height,
                confirmations: 1,
            });
            state.claim_draft = None;
            state.screen = Screen::Dashboard;
            state.notify(
                NotificationLevel::Success,
                format!("Claimed {} NERV from {}", amount_nano, bucket.label()),
            );
            events.push(WalletEvent::Redraw);
        }
    }

    events
}

fn parse_seed(hex: &str) -> Result<[u8; 32], String> {
    if hex.len() != 64 {
        return Err(format!("Seed must be 64 hex chars, got {}", hex.len()));
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)
            .map_err(|_| format!("Invalid hex at position {i}"))?;
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn navigation_cycles() {
        let mut s = WalletState::new();
        assert_eq!(s.screen, Screen::Dashboard);
        update(&mut s, WalletAction::NextTab);
        assert_eq!(s.screen, Screen::Send);
        update(&mut s, WalletAction::NextTab);
        assert_eq!(s.screen, Screen::Receive);
        update(&mut s, WalletAction::PrevTab);
        assert_eq!(s.screen, Screen::Send);
    }

    #[test]
    fn draft_validation_flow() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartSend);
        assert!(s.draft.is_some());

        // Empty recipient: error.
        update(&mut s, WalletAction::ConfirmSend);
        assert!(matches!(
            s.draft.as_ref().unwrap().validation.error,
            Some(DraftError::EmptyRecipient)
        ));

        // Invalid recipient: error.
        update(&mut s, WalletAction::SetRecipient("zzz".into()));
        assert!(matches!(
            s.draft.as_ref().unwrap().validation.error,
            Some(DraftError::InvalidRecipientHex)
        ));

        // Valid recipient but zero amount: error.
        let valid_hex = "ab".repeat(1184);
        update(&mut s, WalletAction::SetRecipient(valid_hex));
        assert!(matches!(
            s.draft.as_ref().unwrap().validation.error,
            Some(DraftError::AmountZero)
        ));

        // Amount but insufficient funds: error.
        update(&mut s, WalletAction::SetAmount(1000));
        assert!(matches!(
            s.draft.as_ref().unwrap().validation.error,
            Some(DraftError::InsufficientFunds { .. })
        ));

        // Cancel: back to dashboard, draft cleared.
        update(&mut s, WalletAction::CancelSend);
        assert!(s.draft.is_none());
        assert_eq!(s.screen, Screen::Dashboard);
    }

    #[test]
    fn lock_clears_everything() {
        let mut s = WalletState::new();
        s.seed = Some([1u8; 32]);
        s.chain_height = 100;
        update(&mut s, WalletAction::Lock);
        assert!(s.seed.is_none());
        assert_eq!(s.chain_height, 0);
        assert!(matches!(s.sync, SyncStatus::Locked));
    }

    #[test]
    fn quit_sets_flag() {
        let mut s = WalletState::new();
        assert!(!s.quitting);
        update(&mut s, WalletAction::Quit);
        assert!(s.quitting);
    }

    #[test]
    fn notifications_capped() {
        let mut s = WalletState::new();
        for i in 0..20 {
            update(&mut s, WalletAction::DismissNotification);
            s.notify(NotificationLevel::Info, format!("msg {i}"));
        }
        assert!(s.notifications.len() <= 10);
    }

    // === Claim flow tests ===

    use crate::state::{ClaimBucket, ClaimDraft, ClaimError};

    #[test]
    fn start_claim_initializes_draft() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartClaim);
        assert_eq!(s.screen, Screen::Claim);
        assert!(s.claim_draft.is_some());
        assert!(matches!(
            s.claim_draft.as_ref().unwrap().validation.error,
            Some(ClaimError::NoBucket)
        ));
    }

    #[test]
    fn set_claim_bucket_validates() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartClaim);
        update(&mut s, WalletAction::SetClaimBucket(ClaimBucket::Community));
        let draft = s.claim_draft.as_ref().unwrap();
        assert_eq!(draft.bucket, Some(ClaimBucket::Community));
        assert!(draft.validation.bucket_ok);
        assert!(matches!(
            draft.validation.error,
            Some(ClaimError::AmountZero)
        ));
    }

    #[test]
    fn set_claim_amount_validates() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartClaim);
        update(&mut s, WalletAction::SetClaimBucket(ClaimBucket::Community));
        update(&mut s, WalletAction::SetClaimAmount(0));
        assert!(matches!(
            s.claim_draft.as_ref().unwrap().validation.error,
            Some(ClaimError::AmountZero)
        ));
        update(&mut s, WalletAction::SetClaimAmount(50_000));
        assert!(s.claim_draft.as_ref().unwrap().is_ready());
    }

    #[test]
    fn sign_and_claim_emits_pipeline_event() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartClaim);
        update(&mut s, WalletAction::SetClaimBucket(ClaimBucket::Community));
        update(&mut s, WalletAction::SetClaimAmount(50_000));

        let events = update(&mut s, WalletAction::SignAndClaim);
        // Must contain StartClaimPipeline with the bucket + amount.
        assert!(events.iter().any(|e| matches!(
            e,
            WalletEvent::StartClaimPipeline {
                bucket: ClaimBucket::Community,
                amount_nano: 50_000,
            }
        )));
        // The draft is consumed on success.
        assert!(s.claim_draft.is_none());
    }

    #[test]
    fn sign_and_claim_without_bucket_does_not_emit_pipeline() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartClaim);
        update(&mut s, WalletAction::SetClaimAmount(50_000));
        // No bucket set: draft should bounce back, no StartClaimPipeline.
        let events = update(&mut s, WalletAction::SignAndClaim);
        assert!(!events.iter().any(|matches!(
            e, WalletEvent::StartClaimPipeline { .. }
        )));
        assert!(s.claim_draft.is_some());
    }

    #[test]
    fn sign_and_claim_with_closed_window_does_not_emit_pipeline() {
        let mut s = WalletState::new();
        // Sync status: Locked → window considered closed.
        update(&mut s, WalletAction::StartClaim);
        update(&mut s, WalletAction::SetClaimBucket(ClaimBucket::Community));
        update(&mut s, WalletAction::SetClaimAmount(50_000));
        let events = update(&mut s, WalletAction::SignAndClaim);
        assert!(!events.iter().any(|e| matches!(e, WalletEvent::StartClaimPipeline { .. })));
        assert!(s.claim_draft.is_some());
        assert!(matches!(
            s.claim_draft.as_ref().unwrap().validation.error,
            Some(ClaimError::WindowClosed)
        ));
    }

    #[test]
    fn claim_succeeded_pushes_incoming_history_entry() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartClaim);
        update(&mut s, WalletAction::SetClaimBucket(ClaimBucket::Community));
        update(&mut s, WalletAction::SetClaimAmount(50_000));
        update(&mut s, WalletAction::SignAndClaim);

        update(
            &mut s,
            WalletAction::ClaimSucceeded {
                bucket: ClaimBucket::Community,
                amount_nano: 50_000,
            },
        );

        assert_eq!(s.history.len(), 1);
        let entry = &s.history[0];
        assert_eq!(entry.direction, Direction::Incoming);
        assert_eq!(entry.amount_nano, 50_000);
        assert_eq!(entry.fee_nano, 0);
        assert_eq!(s.screen, Screen::Dashboard);
        assert!(s.claim_draft.is_none());
    }

    #[test]
    fn claim_failed_clears_draft_and_notifies() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartClaim);
        update(&mut s, WalletAction::SetClaimBucket(ClaimBucket::Community));
        update(&mut s, WalletAction::SetClaimAmount(50_000));
        update(&mut s, WalletAction::SignAndClaim);
        update(&mut s, WalletAction::ClaimFailed("nullifier already spent".into()));
        assert!(s.claim_draft.is_none());
        assert_eq!(s.screen, Screen::Dashboard);
        // Last notification carries the failure reason.
        assert!(s
            .notifications
            .iter()
            .rev()
            .any(|n| n.message.contains("nullifier already spent")));
    }

    #[test]
    fn cancel_claim_returns_to_dashboard() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartClaim);
        update(&mut s, WalletAction::CancelClaim);
        assert!(s.claim_draft.is_none());
        assert_eq!(s.screen, Screen::Dashboard);
    }

    #[test]
    fn lock_clears_claim_draft() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartClaim);
        assert!(s.claim_draft.is_some());
        update(&mut s, WalletAction::Lock);
        assert!(s.claim_draft.is_none());
    }

    #[test]
    fn screen_indices_include_claim() {
        // The Claim screen's index is 3 — the tab order is:
        // Dashboard (0), Send (1), Receive (2), Claim (3), ...
        assert_eq!(Screen::Claim.index(), 3);
        assert_eq!(Screen::from_index(3), Screen::Claim);
        assert_eq!(Screen::Claim.title(), "Claim");
    }

    #[test]
    fn claim_bucket_name_matches_params_toml() {
        // The `bucket.name()` strings must match `[[economy.buckets]]`
        // names in `specs/params.toml` — the platform shell uses these
        // to route the leg to the right emission tree.
        assert_eq!(ClaimBucket::Community.name(), "community");
        assert_eq!(ClaimBucket::Ecosystem.name(), "ecosystem");
        assert_eq!(ClaimBucket::Foundation.name(), "foundation");
        assert_eq!(ClaimBucket::Founder.name(), "founder");
    }

    #[test]
    fn user_buckets_excludes_founder() {
        // Founder claims go through a separate authenticated path; the
        // wallet's claim screen must not offer them to end users.
        let names: Vec<&'static str> = ClaimBucket::USER_BUCKETS
            .iter()
            .map(|b| b.name())
            .collect();
        assert!(!names.contains(&"founder"));
        assert_eq!(names, vec!["community", "ecosystem", "foundation"]);
    }

    // === Producer-flow state-machine tests (gap continuation) ===

    fn hex64(byte: u8) -> String {
        // 64-hex-char string ("ab..." * 32).
        format!("{:02x}", byte).repeat(32)
    }

    #[test]
    fn start_producer_initializes_draft_and_switches_screen() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartProducer);
        assert_eq!(s.screen, Screen::Producer);
        assert!(s.producer_draft.is_some());
        assert!(!s.producer_draft.as_ref().unwrap().is_ready());
    }

    #[test]
    fn set_producer_seed_parses_64_hex_chars() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartProducer);
        update(
            &mut s,
            WalletAction::SetProducerSeed(hex64(0xA1)),
        );
        let draft = s.producer_draft.as_ref().unwrap();
        assert_eq!(draft.seed, Some([0xA1; 32]));
        assert!(draft.validation.seed_ok);
    }

    #[test]
    fn set_producer_seed_rejects_wrong_length() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartProducer);
        update(&mut s, WalletAction::SetProducerSeed("aabb".into()));
        let draft = s.producer_draft.as_ref().unwrap();
        // 4 chars is not a valid 32-byte seed.
        assert_eq!(draft.seed, None);
        assert!(!draft.validation.seed_ok);
    }

    #[test]
    fn set_producer_seed_rejects_non_hex() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartProducer);
        // 64 chars but contains 'zz' — `parse_seed` only accepts hex
        // digits and so fails. The draft's seed stays None and
        // validation flips `seed_ok = false`.
        let bad = "zz".repeat(32);
        update(&mut s, WalletAction::SetProducerSeed(bad));
        let draft = s.producer_draft.as_ref().unwrap();
        assert_eq!(draft.seed, None);
        assert!(!draft.validation.seed_ok);
    }

    #[test]
    fn set_producer_shard_caps_at_genesis_set() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartProducer);
        update(&mut s, WalletAction::SetProducerShard(42));
        assert_eq!(
            s.producer_draft.as_ref().unwrap().shard,
            Some(42)
        );
        // 200 is outside the 0..=63 range; the handler caps it and
        // clears the draft's shard.
        update(&mut s, WalletAction::SetProducerShard(200));
        assert_eq!(s.producer_draft.as_ref().unwrap().shard, None);
    }

    #[test]
    fn set_producer_stake_zero_marks_validation_error() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartProducer);
        update(&mut s, WalletAction::SetProducerStake(0));
        let draft = s.producer_draft.as_ref().unwrap();
        assert_eq!(draft.stake_nano, 0);
        assert!(!draft.validation.stake_ok);
    }

    #[test]
    fn register_producer_emits_register_event_when_ready() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartProducer);
        update(
            &mut s,
            WalletAction::SetProducerSeed(hex64(0x5C)),
        );
        update(&mut s, WalletAction::SetProducerShard(7));
        update(
            &mut s,
            WalletAction::SetProducerStake(1_500_000_000),
        );
        let events = update(&mut s, WalletAction::RegisterProducerStake);
        // The draft is taken on RegisterProducerStake; a
        // RegisterProducerStake event is emitted for the platform
        // shell to run the actual identity construction.
        assert!(s.producer_draft.is_none());
        let mut found = false;
        for ev in &events {
            if let WalletEvent::RegisterProducerStake { shard, stake_nano, .. } = ev {
                assert_eq!(*shard, 7);
                assert_eq!(*stake_nano, 1_500_000_000);
                found = true;
            }
        }
        assert!(found, "expected a RegisterProducerStake event");
    }

    #[test]
    fn register_producer_bounces_when_draft_not_ready() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartProducer);
        // No seed, no shard, stake=0 — draft is not ready.
        let events = update(&mut s, WalletAction::RegisterProducerStake);
        // Draft is preserved (not consumed) and a warning fires.
        assert!(s.producer_draft.is_some());
        for ev in &events {
            assert!(matches!(ev, WalletEvent::Redraw));
        }
    }

    #[test]
    fn producer_stake_registered_populates_state_and_returns_to_dashboard() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartProducer);
        update(
            &mut s,
            WalletAction::SetProducerSeed(hex64(0x42)),
        );
        update(&mut s, WalletAction::SetProducerShard(3));
        update(&mut s, WalletAction::SetProducerStake(900_000_000));
        let _ = update(&mut s, WalletAction::RegisterProducerStake);

        let vk = hex64(0x99);
        let addr = hex64(0x33);
        let account_id = [0xAAu8; 32];
        update(
            &mut s,
            WalletAction::ProducerStakeRegistered {
                verifying_key_hex: vk.clone(),
                payout_address_hex: addr.clone(),
                stake_balance_nano: 900_000_000,
                account_id,
            },
        );
        assert_eq!(s.producer_state.verifying_key_hex, vk);
        assert_eq!(s.producer_state.payout_address_hex, addr);
        assert_eq!(s.producer_state.stake_balance_nano, 900_000_000);
        assert!(s.producer_state.stake_registered);
        assert_eq!(s.producer_state.assigned_shard, 3);
        assert_eq!(s.screen, Screen::Dashboard);
        assert!(s.producer_draft.is_none());
    }

    #[test]
    fn producer_registration_failed_clears_draft() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartProducer);
        update(
            &mut s,
            WalletAction::SetProducerSeed(hex64(0x11)),
        );
        update(&mut s, WalletAction::SetProducerShard(0));
        update(&mut s, WalletAction::SetProducerStake(100));
        let _ = update(&mut s, WalletAction::RegisterProducerStake);
        assert!(s.producer_draft.is_none());

        update(
            &mut s,
            WalletAction::ProducerRegistrationFailed("shard inactive".into()),
        );
        assert!(s.producer_draft.is_none());
        assert_eq!(s.screen, Screen::Dashboard);
        assert!(!s.producer_state.stake_registered);
    }

    #[test]
    fn producer_state_refreshed_overwrites_state() {
        let mut s = WalletState::new();
        let snapshot = ProducerState {
            verifying_key_hex: hex64(0x77),
            payout_address_hex: hex64(0x22),
            stake_balance_nano: 5_000_000_000,
            total_payouts_nano: 42_000,
            epoch_payout_nano: 7_000,
            assigned_shard: 11,
            node_role: "Producer".into(),
            stake_registered: true,
        };
        update(&mut s, WalletAction::ProducerStateRefreshed(snapshot.clone()));
        assert_eq!(s.producer_state, snapshot);
    }

    #[test]
    fn cancel_producer_returns_to_dashboard() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartProducer);
        assert!(s.producer_draft.is_some());
        update(&mut s, WalletAction::CancelProducer);
        assert!(s.producer_draft.is_none());
        assert_eq!(s.screen, Screen::Dashboard);
    }

    #[test]
    fn lock_clears_producer_draft_and_state() {
        let mut s = WalletState::new();
        update(&mut s, WalletAction::StartProducer);
        update(
            &mut s,
            WalletAction::SetProducerSeed(hex64(0x12)),
        );
        update(&mut s, WalletAction::SetProducerShard(0));
        update(&mut s, WalletAction::SetProducerStake(100));
        let _ = update(&mut s, WalletAction::RegisterProducerStake);
        update(
            &mut s,
            WalletAction::ProducerStakeRegistered {
                verifying_key_hex: hex64(0xAA),
                payout_address_hex: hex64(0xBB),
                stake_balance_nano: 100,
                account_id: [0xCC; 32],
            },
        );
        assert!(s.producer_state.stake_registered);
        update(&mut s, WalletAction::Lock);
        assert!(s.producer_draft.is_none());
        assert!(!s.producer_state.stake_registered);
    }

    #[test]
    fn screen_indices_include_producer() {
        // Producer sits between Claim (3) and History (5).
        assert_eq!(Screen::Producer.index(), 4);
        assert_eq!(Screen::from_index(4), Screen::Producer);
        assert_eq!(Screen::Producer.title(), "Producer");
    }
}
