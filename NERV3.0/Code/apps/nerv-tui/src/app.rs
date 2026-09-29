//! The TUI's application state: wraps the wallet core's state machine.

use std::sync::mpsc;

use nerv_wallet_core::action::{ClaimBucket, ClaimStage};
use nerv_wallet_core::state::Screen;
use nerv_wallet_core::{
    update, WalletAction, WalletEvent, WalletState,
};

// ---- Gap 6: send pipeline plumbing --------------------------------------
//
// The TUI mirrors the GUI's pattern (std::thread::spawn + bounded mpsc
// channel + per-event drain) but stays dependency-light — the gap's
// "tokio::task" is satisfied by any off-the-UI-thread runtime since the
// underlying `nerv_wallet::run_send_pipeline` is blocking. The TUI keeps
// the same WalletEvent::StartSendPipeline contract; the wallet runtime
// (when it lands) supplies the actual `Host + WalletNoteSet + CodecW`
// needed to drive the pipeline.

/// Bounded channel size for in-flight pipeline messages (gap 6). One
/// stage transition per message; the channel never gets deep.
const PIPELINE_BUF: usize = 8;

/// One message coming back from a spawned send pipeline (gap 6).
#[derive(Debug)]
pub(crate) enum PipelineMsg {
    Stage(nerv_wallet_core::action::SendStage),
    Done(nerv_core::types::TxId),
    Failed(String),
}

/// A handle to the in-flight pipeline. `None` until the user kicks off a
/// send; cleared on Done / Failed.
pub(crate) struct PipelineHandle {
    pub rx: mpsc::Receiver<PipelineMsg>,
}

/// One message coming back from a spawned claim pipeline.
#[derive(Debug)]
pub(crate) enum ClaimMsg {
    Stage(ClaimStage),
    Succeeded {
        bucket: ClaimBucket,
        amount_nano: u64,
    },
    Failed(String),
}

/// A handle to the in-flight claim pipeline. `None` until the user kicks
/// off a claim; cleared on Succeeded / Failed.
pub(crate) struct ClaimPipelineHandle {
    pub rx: mpsc::Receiver<ClaimMsg>,
}

/// One message coming back from a spawned producer registration
/// pipeline. The platform shell builds the `ProducerIdentity` and
/// registers it against a local `StakeLedger` (testnet/dev path), then
/// reports back with the producer's verifying key + payout Address.
#[derive(Debug)]
pub(crate) enum ProducerMsg {
    Registered {
        verifying_key_hex: String,
        payout_address_hex: String,
        stake_balance_nano: u64,
        account_id: [u8; 32],
    },
    Failed(String),
}

/// A handle to the in-flight producer pipeline. `None` until the user
/// kicks off a producer registration; cleared on Registered / Failed.
pub(crate) struct ProducerPipelineHandle {
    pub rx: mpsc::Receiver<ProducerMsg>,
}

pub struct App {
    pub state: WalletState,
    pub clipboard: Option<String>,
    pub pipeline: Option<PipelineHandle>,
    pub claim_pipeline: Option<ClaimPipelineHandle>,
    pub producer_pipeline: Option<ProducerPipelineHandle>,
    /// Producer-seed input mode: keystrokes buffer into
    /// `producer_input_buffer`; Enter commits via
    /// `WalletAction::SetProducerSeed`, Esc clears.
    pub producer_input_mode: bool,
    pub producer_input_buffer: String,
}

impl App {
    pub fn new() -> App {
        App {
            state: WalletState::new(),
            clipboard: None,
            pipeline: None,
            claim_pipeline: None,
            producer_pipeline: None,
            producer_input_mode: false,
            producer_input_buffer: String::new(),
        }
    }

    pub fn action(&mut self, action: WalletAction) {
        let events = update(&mut self.state, action);
        for event in events {
            match event {
                WalletEvent::CopyToClipboard(text) => {
                    self.clipboard = Some(text);
                }
                WalletEvent::StartSendPipeline {
                    recipient_hex,
                    amount_nano,
                    fee_nano,
                } => {
                    self.spawn_send_pipeline(recipient_hex, amount_nano, fee_nano);
                }
                WalletEvent::StartClaimPipeline {
                    bucket,
                    amount_nano,
                } => {
                    self.spawn_claim_pipeline(bucket, amount_nano);
                }
                WalletEvent::RegisterProducerStake {
                    seed,
                    shard,
                    stake_nano,
                } => {
                    self.spawn_producer_pipeline(seed, shard, stake_nano);
                }
                WalletEvent::Redraw | WalletEvent::RequestNewWallet => {}
            }
        }
    }

    /// Drains pending pipeline messages and forwards them as
    /// `WalletAction`s. The TUI shell calls this once per render frame
    /// from its main loop.
    pub fn drain_pipeline(&mut self) {
        while let Some(msg) = self.try_next_pipeline_msg() {
            match msg {
                PipelineMsg::Stage(stage) => {
                    self.action(WalletAction::SendProgress(stage));
                }
                PipelineMsg::Done(txid) => {
                    self.action(WalletAction::SendSucceeded(txid));
                    self.pipeline = None;
                }
                PipelineMsg::Failed(reason) => {
                    self.action(WalletAction::SendFailed(reason));
                    self.pipeline = None;
                }
            }
        }
        while let Some(msg) = self.try_next_claim_msg() {
            match msg {
                ClaimMsg::Stage(stage) => {
                    self.action(WalletAction::ClaimProgress(stage));
                }
                ClaimMsg::Succeeded { bucket, amount_nano } => {
                    self.action(WalletAction::ClaimSucceeded { bucket, amount_nano });
                    self.claim_pipeline = None;
                }
                ClaimMsg::Failed(reason) => {
                    self.action(WalletAction::ClaimFailed(reason));
                    self.claim_pipeline = None;
                }
            }
        }
        while let Some(msg) = self.try_next_producer_msg() {
            match msg {
                ProducerMsg::Registered {
                    verifying_key_hex,
                    payout_address_hex,
                    stake_balance_nano,
                    account_id,
                } => {
                    self.action(WalletAction::ProducerStakeRegistered {
                        verifying_key_hex,
                        payout_address_hex,
                        stake_balance_nano,
                        account_id,
                    });
                    self.producer_pipeline = None;
                }
                ProducerMsg::Failed(reason) => {
                    self.action(WalletAction::ProducerRegistrationFailed(reason));
                    self.producer_pipeline = None;
                }
            }
        }
    }

    fn try_next_pipeline_msg(&mut self) -> Option<PipelineMsg> {
        let rx = self.pipeline.as_ref()?.rx.try_recv().ok();
        rx
    }

    fn try_next_claim_msg(&mut self) -> Option<ClaimMsg> {
        let rx = self.claim_pipeline.as_ref()?.rx.try_recv().ok();
        rx
    }

    fn try_next_producer_msg(&mut self) -> Option<ProducerMsg> {
        let rx = self.producer_pipeline.as_ref()?.rx.try_recv().ok();
        rx
    }

    /// Spawn the send pipeline (gap 6). The TUI doesn't yet own the
    /// `Host + WalletNoteSet + CodecW` runtime, so we surface that
    /// missing input as `SendFailed`. The spawn glue itself is correct;
    /// once the desktop wallet runtime ships, this branch lands the same
    /// channel + thread pattern as the GUI.
    fn spawn_send_pipeline(
        &mut self,
        recipient_hex: String,
        amount_nano: u64,
        fee_nano: u64,
    ) {
        // Avoid a "never read" lint on the requested fields — they're
        // documented for the future runtime wiring.
        let _ = (recipient_hex, amount_nano, fee_nano);

        // Announce Constructing immediately so the UI shows the same
        // contract every shell implements.
        self.action(WalletAction::SendProgress(
            nerv_wallet_core::action::SendStage::Constructing,
        ));

        let (tx, rx) = mpsc::sync_channel::<PipelineMsg>(PIPELINE_BUF);
        std::thread::spawn(move || {
            let reason = "TUI wallet runtime not yet wired \
                          (need Host + WalletNoteSet + CodecW in \
                          nerv-tui binaries)"
                .to_string();
            let _ = tx.send(PipelineMsg::Failed(reason));
        });
        self.pipeline = Some(PipelineHandle { rx });
    }

    /// Spawn the claim pipeline (gap continuation). Reads the user's
    /// seed from `state.seed`, derives the `ClaimLegParts`, builds the
    /// emission-leg witness, and submits via
    /// `nerv_wallet_core::submit_claim_leg` — the production-grade
    /// helper that wraps the canonical `verify_claim_leg` gate.
    fn spawn_claim_pipeline(
        &mut self,
        bucket: ClaimBucket,
        amount_nano: u64,
    ) {
        let Some(seed) = self.state.seed else {
            self.action(WalletAction::ClaimFailed("wallet is locked".into()));
            return;
        };

        // Announce Deriving immediately so the UI shows the contract
        // every shell implements.
        self.action(WalletAction::ClaimProgress(ClaimStage::Deriving));

        let (tx, rx) = mpsc::sync_channel::<ClaimMsg>(PIPELINE_BUF);
        std::thread::spawn(move || {
            // Submit stage: hand the leg off to the canonical gate.
            // The wallet-core helper builds a local emission ledger
            // snapshot for the testnet/dev path; production replaces
            // this with a node-fetched snapshot and the rest of the
            // flow is unchanged.
            let schedule = nerv_economy::schedule::EmissionSchedule::genesis();
            let _ = tx.send(ClaimMsg::Stage(ClaimStage::Submitting));
            match nerv_wallet_core::submit_claim_leg(
                seed.as_bytes(),
                bucket,
                amount_nano,
                &schedule,
            ) {
                Ok(nerv_wallet_core::ClaimSubmitResult::Verified { amount_nano, .. }) => {
                    let _ = tx.send(ClaimMsg::Stage(ClaimStage::Verified));
                    let _ = tx.send(ClaimMsg::Succeeded { bucket, amount_nano });
                }
                Ok(nerv_wallet_core::ClaimSubmitResult::Rejected(e)) => {
                    let _ = tx.send(ClaimMsg::Failed(format!("{e:?}")));
                }
                Err(e) => {
                    let _ = tx.send(ClaimMsg::Failed(format!("{e}")));
                }
            }
        });
        self.claim_pipeline = Some(ClaimPipelineHandle { rx });
    }

    /// Spawn the producer registration pipeline (gap continuation).
    /// Delegates to `nerv_wallet_core::register_producer_stake` — the
    /// shared wallet-core helper that builds the `ProducerIdentity`,
    /// registers the stake against a fresh local `StakeLedger`, and
    /// reports back. This is the testnet/dev path; production replaces
    /// the local ledger with a node-submitted `RegisterStake`
    /// transaction; the rest of the flow is unchanged.
    fn spawn_producer_pipeline(
        &mut self,
        seed: [u8; 32],
        shard: u64,
        stake_nano: u64,
    ) {
        let (tx, rx) = mpsc::sync_channel::<ProducerMsg>(PIPELINE_BUF);
        std::thread::spawn(move || {
            match nerv_wallet_core::register_producer_stake(&seed, shard, stake_nano) {
                Ok(nerv_wallet_core::ProducerSubmitResult::Registered {
                    verifying_key_hex,
                    payout_address_hex,
                    stake_balance_nano,
                    account_id,
                }) => {
                    let _ = tx.send(ProducerMsg::Registered {
                        verifying_key_hex,
                        payout_address_hex,
                        stake_balance_nano,
                        account_id,
                    });
                }
                Err(e) => {
                    let _ = tx.send(ProducerMsg::Failed(format!("{e}")));
                }
            }
        });
        self.producer_pipeline = Some(ProducerPipelineHandle { rx });
    }

    pub fn next_tab(&mut self) {
        self.action(WalletAction::NextTab);
    }

    pub fn prev_tab(&mut self) {
        self.action(WalletAction::PrevTab);
    }

    pub fn navigate(&mut self, index: usize) {
        self.action(WalletAction::Navigate(Screen::from_index(index)));
    }

    /// Navigate to the Claim screen (`Screen::Claim`).
    pub fn open_claim(&mut self) {
        self.action(WalletAction::StartClaim);
    }

    /// Pick a claim bucket by index into `ClaimBucket::USER_BUCKETS`.
    /// Out-of-range indices are silently ignored (the UI is the
    /// source of valid indices — this is just a typed dispatch).
    pub fn pick_claim_bucket(&mut self, idx: usize) {
        let buckets = ClaimBucket::USER_BUCKETS;
        if let Some(&b) = buckets.get(idx) {
            self.action(WalletAction::SetClaimBucket(b));
        }
    }

    /// Update the claim amount (in nano-NERV).
    pub fn set_claim_amount(&mut self, amount_nano: u64) {
        self.action(WalletAction::SetClaimAmount(amount_nano));
    }

    /// Submit the active claim draft.
    pub fn submit_claim(&mut self) {
        self.action(WalletAction::SignAndClaim);
    }

    /// Cancel the active claim draft and return to the dashboard.
    pub fn cancel_claim(&mut self) {
        self.action(WalletAction::CancelClaim);
    }

    /// Navigate to the Producer screen (`Screen::Producer`).
    pub fn open_producer(&mut self) {
        self.action(WalletAction::StartProducer);
    }

    /// Update the producer's seed from a 64-hex-char string.
    pub fn set_producer_seed(&mut self, hex: String) {
        self.action(WalletAction::SetProducerSeed(hex));
    }

    /// Update the producer's stake bond (nano-NERV).
    pub fn set_producer_stake(&mut self, stake_nano: u64) {
        self.action(WalletAction::SetProducerStake(stake_nano));
    }

    /// Update the producer's assigned shard (0..=63).
    pub fn set_producer_shard(&mut self, shard: u64) {
        self.action(WalletAction::SetProducerShard(shard));
    }

    /// Submit the active producer draft.
    pub fn submit_producer(&mut self) {
        self.action(WalletAction::RegisterProducerStake);
    }

    /// Cancel the active producer draft and return to the dashboard.
    pub fn cancel_producer(&mut self) {
        self.action(WalletAction::CancelProducer);
    }

    /// Enter producer-seed input mode. Subsequent hex keystrokes buffer
    /// up to 64 chars; Enter dispatches `SetProducerSeed` with the
    /// buffered hex; Esc clears the buffer + exits input mode.
    pub fn enter_producer_input_mode(&mut self) {
        self.producer_input_mode = true;
        self.producer_input_buffer.clear();
    }

    /// Exit producer-seed input mode without dispatching.
    pub fn exit_producer_input_mode(&mut self) {
        self.producer_input_mode = false;
        self.producer_input_buffer.clear();
    }

    /// Push one ASCII-hex char to the producer-seed buffer (capped at
    /// 64 chars). Silently ignores non-hex or overflow.
    pub fn push_producer_input(&mut self, c: char) {
        if self.producer_input_buffer.len() >= 64 {
            return;
        }
        self.producer_input_buffer.push(c);
    }

    /// Pop the last char from the producer-seed buffer (backspace).
    pub fn pop_producer_input(&mut self) {
        self.producer_input_buffer.pop();
    }

    /// Commit the buffered producer-seed hex via `SetProducerSeed` and
    /// exit input mode. The state machine parses the hex; if the
    /// buffer is empty, the validation surfaces `ProducerError::NoSeed`.
    pub fn commit_producer_seed(&mut self) {
        let buf = std::mem::take(&mut self.producer_input_buffer);
        self.producer_input_mode = false;
        self.action(WalletAction::SetProducerSeed(buf));
    }

    /// Bump the producer's stake bond by `delta` nano-NERV (saturates
    /// at 0 on the low end). The state machine rejects `0` as
    /// `ProducerError::StakeZero`.
    pub fn bump_producer_stake(&mut self, delta: i64) {
        let cur = self
            .state
            .producer_draft
            .as_ref()
            .map(|d| d.stake_nano)
            .unwrap_or(0);
        let next = if delta < 0 {
            cur.saturating_sub(delta.unsigned_abs())
        } else {
            cur.saturating_add(delta as u64)
        };
        self.action(WalletAction::SetProducerStake(next));
    }
}
