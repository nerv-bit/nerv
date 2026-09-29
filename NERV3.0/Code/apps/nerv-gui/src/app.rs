//! The eframe App implementation (erratum 202–203): the entire wallet
//! GUI, rendering the `nerv-wallet-core` state machine.

use egui::{Align, Color32, Layout, RichText, Sense, Vec2};
use nerv_wallet_core::state::{
    Direction, DraftError, HistoryEntry, NotificationLevel, Screen, SyncStatus,
};
use nerv_wallet_core::{update, WalletAction, WalletEvent, WalletState};

use crate::theme;

// ---- Gap 6: send pipeline plumbing --------------------------------------

/// A message coming back from the spawned send pipeline (gap 6). The
/// pipeline runs on a background thread (native) or the microtask queue
/// (wasm); the GUI's `update()` drains these once per frame and forwards
/// them to the state machine.
#[derive(Clone, Debug)]
pub(crate) enum PipelineMsg {
    /// A stage transition (Constructing / Proving / …). Maps to
    /// `WalletAction::SendProgress(stage)`.
    Stage(nerv_wallet::PipelineStage),
    /// The pipeline finished successfully — the txid.
    Done(nerv_core::types::TxId),
    /// The pipeline failed — human-readable reason.
    Failed(String),
}

/// One message coming back from a spawned claim pipeline (gap
/// continuation). The platform shell derives the `ClaimLegParts` from
/// the user's seed, builds the emission-leg witness, and submits via
/// `nerv_wallet_core::submit_claim_leg`.
#[derive(Clone, Debug)]
pub(crate) enum ClaimMsg {
    Stage(nerv_wallet_core::action::ClaimStage),
    Succeeded {
        bucket: nerv_wallet_core::action::ClaimBucket,
        amount_nano: u64,
    },
    Failed(String),
}

/// One message coming back from a spawned producer registration
/// pipeline (gap continuation). The platform shell builds the
/// `ProducerIdentity`, registers it against a local `StakeLedger`
/// (testnet/dev path), and reports back.
#[derive(Clone, Debug)]
pub(crate) enum ProducerMsg {
    Registered {
        verifying_key_hex: String,
        payout_address_hex: String,
        stake_balance_nano: u64,
        account_id: [u8; 32],
    },
    Failed(String),
}

/// A handle to a spawned pipeline. The GUI owns this; when the user kicks
/// off a send, we install one. When the pipeline finishes (Done or Failed
/// terminal) we drop it.
pub(crate) enum PipelineHandle {
    /// Native (desktop) — pulls from a bounded mpsc channel via
    /// `std::sync::mpsc`. The pipeline thread is joined by dropping the
    /// sender; we don't store the JoinHandle because the GUI doesn't
    /// expose cancellation today.
    #[cfg(not(target_arch = "wasm32"))]
    Channel(std::sync::mpsc::Receiver<PipelineMsg>),
    /// WASM — the wallet's proving is delegated to a connected node
    /// (see `Cargo.toml`); the GUI just logs the event for now.
    #[cfg(target_arch = "wasm32")]
    WasmStub,
}

/// How many in-flight stage messages the GUI tolerates before
/// backpressuring the pipeline. The pipeline runs synchronously and only
/// sends one stage transition at a time, so the channel never needs to be
/// deep.
#[cfg(not(target_arch = "wasm32"))]
const PIPELINE_BUF: usize = 8;

/// The NERV wallet GUI application.
pub struct NervApp {
    state: WalletState,
    seed_input: String,
    send_recipient: String,
    send_amount: String,
    send_fee: String,
    clipboard: Option<String>,
    nav_selected: usize,
    frame_count: u64,
    /// Active pipeline (gap 6). `None` until a `StartSendPipeline` event
    /// is dispatched. Multiple sends in flight at once are out of scope
    /// today (the state machine takes the draft on Send).
    pipeline: Option<PipelineHandle>,
    /// Claim-flow draft state (gap continuation). Drives the Claim screen.
    claim_bucket_index: usize,
    claim_amount: String,
    /// Active claim pipeline. `None` until a `StartClaimPipeline` event
    /// is dispatched. Mirrors `pipeline` (gap 6) for the claim path.
    #[cfg(not(target_arch = "wasm32"))]
    claim_pipeline: Option<std::sync::mpsc::Receiver<ClaimMsg>>,
    /// Active producer pipeline. `None` until a
    /// `RegisterProducerStake` event is dispatched.
    #[cfg(not(target_arch = "wasm32"))]
    producer_pipeline: Option<std::sync::mpsc::Receiver<ProducerMsg>>,
    /// Producer-flow input fields (gap continuation). Mirrors the
    /// claim-screen text inputs (`claim_amount`).
    producer_seed_input: String,
    producer_stake_input: String,
    producer_shard_input: String,
}

impl NervApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> NervApp {
        theme::install(&cc.egui_ctx);
        NervApp {
            state: WalletState::new(),
            seed_input: String::new(),
            send_recipient: String::new(),
            send_amount: String::new(),
            send_fee: "700".to_string(),
            clipboard: None,
            nav_selected: 0,
            frame_count: 0,
            pipeline: None,
            claim_bucket_index: 0,
            claim_amount: String::new(),
            #[cfg(not(target_arch = "wasm32"))]
            claim_pipeline: None,
            #[cfg(not(target_arch = "wasm32"))]
            producer_pipeline: None,
            producer_seed_input: String::new(),
            producer_stake_input: String::new(),
            producer_shard_input: String::new(),
        }
    }

    fn dispatch(&mut self, action: WalletAction) {
        let events = update(&mut self.state, action);
        for ev in events {
            match ev {
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
                _ => {}
            }
        }
    }

    /// WASM ⇄ JS bridge: drain any actions queued by the JS shim via
    /// `nerv_navigate` / `nerv_lock`. Each frame the queue is drained
    /// once; if multiple nav events landed in the same frame, the
    /// latest one wins (the queue is `Option<String>` — `take()`-
    /// semantics).
    #[cfg(target_arch = "wasm32")]
    fn drain_bridge_queues(&mut self) {
        if let Some(screen_name) = crate::take_pending_nav() {
            if let Some(screen) = screen_from_str(&screen_name) {
                self.dispatch(WalletAction::Navigate(screen));
            }
        }
        if crate::take_pending_lock() {
            self.dispatch(WalletAction::Lock);
        }
    }

    /// Drains pending `PipelineMsg` events from any in-flight send
    /// pipeline and turns them into `WalletAction`s. Called from
    /// `update()` so the UI repaints on every stage transition.
    fn drain_pipeline(&mut self) {
        while let Some(msg) = self.next_pipeline_msg() {
            match msg {
                PipelineMsg::Stage(stage) => {
                    let label = stage.label();
                    let core_stage = match stage {
                        nerv_wallet::PipelineStage::Constructing => {
                            nerv_wallet_core::action::SendStage::Constructing
                        }
                        nerv_wallet::PipelineStage::Proving => {
                            nerv_wallet_core::action::SendStage::Proving
                        }
                        nerv_wallet::PipelineStage::Sealing => {
                            nerv_wallet_core::action::SendStage::Sealing
                        }
                        nerv_wallet::PipelineStage::Routing => {
                            nerv_wallet_core::action::SendStage::Routing
                        }
                        nerv_wallet::PipelineStage::Submitted => {
                            nerv_wallet_core::action::SendStage::Submitted
                        }
                    };
                    // Notify via the state machine.
                    self.dispatch(WalletAction::SendProgress(core_stage));
                    // Mirror as a transient notification for the top bar.
                    self.state.notify(
                        nerv_wallet_core::state::NotificationLevel::Info,
                        label.to_string(),
                    );
                }
                PipelineMsg::Done(txid) => {
                    self.dispatch(WalletAction::SendSucceeded(txid));
                    self.pipeline = None;
                }
                PipelineMsg::Failed(reason) => {
                    self.dispatch(WalletAction::SendFailed(reason));
                    self.pipeline = None;
                }
            }
        }
        // Drain the in-flight claim pipeline (if any) and forward its
        // messages as `WalletAction::ClaimProgress` / `ClaimSucceeded` /
        // `ClaimFailed` — the same forward path as the send pipeline.
        while let Some(msg) = self.next_claim_msg() {
            match msg {
                ClaimMsg::Stage(stage) => {
                    let label = stage.label();
                    self.dispatch(WalletAction::ClaimProgress(stage));
                    self.state.notify(
                        nerv_wallet_core::state::NotificationLevel::Info,
                        label.to_string(),
                    );
                }
                ClaimMsg::Succeeded { bucket, amount_nano } => {
                    self.dispatch(WalletAction::ClaimSucceeded { bucket, amount_nano });
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        self.claim_pipeline = None;
                    }
                }
                ClaimMsg::Failed(reason) => {
                    self.dispatch(WalletAction::ClaimFailed(reason));
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        self.claim_pipeline = None;
                    }
                }
            }
        }
        // Drain the in-flight producer pipeline (if any) and forward
        // its messages as `WalletAction::ProducerStakeRegistered` /
        // `ProducerRegistrationFailed` — the same forward path as the
        // claim pipeline.
        #[cfg(not(target_arch = "wasm32"))]
        while let Some(msg) = self.next_producer_msg() {
            match msg {
                ProducerMsg::Registered {
                    verifying_key_hex,
                    payout_address_hex,
                    stake_balance_nano,
                    account_id,
                } => {
                    self.dispatch(WalletAction::ProducerStakeRegistered {
                        verifying_key_hex,
                        payout_address_hex,
                        stake_balance_nano,
                        account_id,
                    });
                    self.producer_pipeline = None;
                }
                ProducerMsg::Failed(reason) => {
                    self.dispatch(WalletAction::ProducerRegistrationFailed(reason));
                    self.producer_pipeline = None;
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn next_producer_msg(&mut self) -> Option<ProducerMsg> {
        let rx = self.producer_pipeline.as_mut()?;
        rx.try_recv().ok()
    }

    #[cfg(target_arch = "wasm32")]
    fn next_producer_msg(&mut self) -> Option<ProducerMsg> {
        None
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn next_pipeline_msg(&mut self) -> Option<PipelineMsg> {
        let rx = match self.pipeline.as_mut() {
            Some(PipelineHandle::Channel(rx)) => rx,
            _ => return None,
        };
        // Try to drain quickly without blocking — the pipeline runs as
        // fast as it can; we never want to stall the UI thread.
        rx.try_recv().ok()
    }

    #[cfg(target_arch = "wasm32")]
    fn next_pipeline_msg(&mut self) -> Option<PipelineMsg> {
        None
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn next_claim_msg(&mut self) -> Option<ClaimMsg> {
        let rx = self.claim_pipeline.as_mut()?;
        rx.try_recv().ok()
    }

    #[cfg(target_arch = "wasm32")]
    fn next_claim_msg(&mut self) -> Option<ClaimMsg> {
        None
    }

    /// Spawn the send pipeline (gap 6). The state machine already
    /// validated the draft; we only need the raw fields plus the wallet
    /// context that's already in `self.state`.
    fn spawn_send_pipeline(
        &mut self,
        recipient_hex: String,
        amount_nano: u64,
        fee_nano: u64,
    ) {
        // Pull what we need from the wallet state. The state machine
        // took the draft on Send, but `state.keys`, `state.balance_nano`,
        // and the discovered notes survive Send — they're owned by the
        // wallet, not the draft.
        let Some(_seed) = self.state.seed else {
            self.dispatch(WalletAction::SendFailed(
                "wallet is locked".to_string(),
            ));
            return;
        };
        // The GUI does not yet own a Host / wallet-note-set / codec / epoch
        // pk (the wallet currently lives in `nerv-wallet-core::state` as
        // a balance + draft; the actual `WalletNoteSet`, `AddressSet`,
        // `CodecW`, and `Host` are owned by the future desktop wallet
        // runtime). Until that runtime lands, we surface a clear error
        // rather than spinning a thread that can't actually execute.
        //
        // The thread-and-channel glue below stays — when the runtime
        // ships, this branch turns into "instantly spawnable". The
        // companion `WalletAction::SendProgress(SendStage::Constructing)`
        // emission path is the contract every platform shell implements
        // (the TUI does the same), so this stays in lock-step.
        let _ = (recipient_hex, amount_nano, fee_nano);

        #[cfg(target_arch = "wasm32")]
        {
            // Wallet proving on wasm is delegated to a connected node
            // (see Cargo.toml); mark it as a no-op for now.
            self.pipeline = Some(PipelineHandle::WasmStub);
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Stage 1: announce Constructing immediately, even though we
            // can't actually run the pipeline yet — this is the contract
            // the state machine expects. Until the desktop wallet runtime
            // provides a `Host + notes + codec`, we report a one-line
            // failure explaining the missing inputs.
            self.dispatch(WalletAction::SendProgress(
                nerv_wallet_core::action::SendStage::Constructing,
            ));
            let (tx, rx) = std::sync::mpsc::sync_channel::<PipelineMsg>(PIPELINE_BUF);
            std::thread::spawn(move || {
                let reason = "desktop wallet runtime not yet wired \
                              (need Host + WalletNoteSet + CodecW in \
                              nerv-gui desktop binaries)"
                    .to_string();
                let _ = tx.send(PipelineMsg::Failed(reason));
            });
            self.pipeline = Some(PipelineHandle::Channel(rx));
        }
    }

    /// Spawn the claim pipeline (gap continuation). Mirrors
    /// `spawn_send_pipeline`'s thread-and-channel pattern.
    ///
    /// Production flow:
    /// 1. Derive `ck = nerv_economy::claim::derive_claim_key(seed)`.
    /// 2. `bucket = ClaimBucket::name()`, `amount_nano` from the draft.
    /// 3. Fresh OS-entropy `blinding = [u8; 32]`.
    /// 4. The commitment/nullifier/eligibility digests fall out of the
    ///    `ClaimLegParts` via `note_commitment`, `claim_nullifier`,
    ///    `eligibility_digest`.
    /// 5. Submit via the wallet-core `submit_claim_leg` helper, which
    ///    calls the canonical `verify_claim_leg` gate.
    fn spawn_claim_pipeline(
        &mut self,
        bucket: nerv_wallet_core::action::ClaimBucket,
        amount_nano: u64,
    ) {
        // Wallet must be unlocked to read the seed.
        let Some(seed) = self.state.seed else {
            self.dispatch(WalletAction::ClaimFailed(
                "wallet is locked".to_string(),
            ));
            return;
        };

        // Announce Deriving immediately so the UI shows the contract.
        self.dispatch(WalletAction::ClaimProgress(
            nerv_wallet_core::action::ClaimStage::Deriving,
        ));

        #[cfg(target_arch = "wasm32")]
        {
            // WASM: claim submission is delegated to the connected
            // node (see Cargo.toml); we don't spawn a thread.
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let (tx, rx) = std::sync::mpsc::sync_channel::<ClaimMsg>(PIPELINE_BUF);
            std::thread::spawn(move || {
                let schedule = nerv_economy::schedule::EmissionSchedule::genesis();
                let _ = tx.send(ClaimMsg::Stage(
                    nerv_wallet_core::action::ClaimStage::Submitting,
                ));
                match nerv_wallet_core::submit_claim_leg(
                    seed.as_bytes(),
                    bucket,
                    amount_nano,
                    &schedule,
                ) {
                    Ok(nerv_wallet_core::ClaimSubmitResult::Verified {
                        amount_nano, ..
                    }) => {
                        let _ = tx.send(ClaimMsg::Stage(
                            nerv_wallet_core::action::ClaimStage::Verified,
                        ));
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
            self.claim_pipeline = Some(rx);
        }
    }

    /// Spawn the producer registration pipeline (gap continuation).
    /// Delegates to `nerv_wallet_core::register_producer_stake` —
    /// the shared wallet-core helper that builds the
    /// `ProducerIdentity`, registers the stake against a fresh local
    /// `StakeLedger`, and reports back. This is the testnet/dev path;
    /// production replaces the local ledger with a node-submitted
    /// `RegisterStake` transaction; the rest of the flow is unchanged.
    fn spawn_producer_pipeline(
        &mut self,
        seed: [u8; 32],
        shard: u64,
        stake_nano: u64,
    ) {
        #[cfg(target_arch = "wasm32")]
        {
            // WASM: producer registration is delegated to the connected
            // node (see Cargo.toml); we don't spawn a thread.
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (tx, rx) = std::sync::mpsc::sync_channel::<ProducerMsg>(PIPELINE_BUF);
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
            self.producer_pipeline = Some(rx);
        }
    }

    fn synced(&self) -> bool {
        matches!(self.state.sync, SyncStatus::Synced { .. })
    }
}

impl eframe::App for NervApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // WASM ⇄ JS bridge: drain any pending actions queued by the
        // JS shim (calls into `nerv_navigate` / `nerv_lock` from
        // `bridge.js`). On native builds, this is a no-op.
        #[cfg(target_arch = "wasm32")]
        self.drain_bridge_queues();
        self.frame_count += 1;

        // Gap 6: drain any pending pipeline messages (Constructing /
        // Proving / Sealing / Routing / Submitted / Done / Failed) and
        // forward them through `dispatch`. Must happen before the UI
        // renders so the progress notification updates the same frame
        // the spawn fires.
        self.drain_pipeline();

        // Top bar.
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.add_space(8.0);
                ui.heading(
                    RichText::new("NERV")
                        .color(theme::color(ctx, "nerv_accent"))
                        .strong()
                        .size(20.0),
                );
                ui.label(
                    RichText::new("Wallet")
                        .color(theme::color(ctx, "nerv_text"))
                        .size(20.0),
                );
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    self.sync_badge(ui);
                });
            });
        });

        // Bottom status bar.
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.add_space(8.0);
                let muted = theme::color(ctx, "nerv_muted");
                let s = format!(
                    "  {} peers  ·  block {}  ·  {}",
                    self.state.peer_count,
                    self.state.chain_height,
                    if self.state.is_unlocked() { "unlocked" } else { "locked" },
                );
                ui.label(RichText::new(&s).color(muted).size(11.0));
                if self.state.notifications.len() > 0 {
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui
                            .small_button("✕ dismiss")
                            .on_hover_text("Dismiss notification (n)")
                            .clicked()
                        {
                            self.dispatch(WalletAction::DismissNotification);
                        }
                    });
                }
            });
        });

        // Left navigation rail.
        egui::SidePanel::left("nav").show(ctx, |ui| {
            ui.add_space(12.0);
            let nav_items: Vec<(Screen, &str, &str)> = vec![
                (Screen::Dashboard, "◧", "Dashboard"),
                (Screen::Send, "→", "Send"),
                (Screen::Receive, "←", "Receive"),
                (Screen::Claim, "◉", "Claim"),
                (Screen::Producer, "⛏", "Producer"),
                (Screen::History, "≡", "History"),
                (Screen::Settings, "⚙", "Settings"),
                (Screen::Help, "?", "Help"),
            ];
            let accent = theme::color(ctx, "nerv_accent");
            let muted = theme::color(ctx, "nerv_muted");
            for (screen, icon, label) in nav_items {
                let selected = self.state.screen == screen;
                let color = if selected { accent } else { muted };
                let btn = egui::Button::new(
                    RichText::new(format!("{icon}  {label}")).color(color).size(14.0),
                )
                .fill(if selected {
                    Color32::from_rgba_unmultiplied(0x58, 0xA6, 0xFF, 0x1A)
                } else {
                    Color32::TRANSPARENT
                });
                if ui.add(btn).clicked() {
                    self.dispatch(WalletAction::Navigate(screen));
                }
                ui.add_space(4.0);
            }

            // Lock button at the bottom.
            ui.with_layout(Layout::bottom_up(Align::LEFT), |ui| {
                if self.state.is_unlocked() && ui.button("🔒 Lock").clicked() {
                    self.dispatch(WalletAction::Lock);
                }
            });
        });

        // Central content.
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.state.screen {
                Screen::Dashboard => self.dashboard(ui, ctx),
                Screen::Send => self.send(ui, ctx),
                Screen::Receive => self.receive(ui, ctx),
                Screen::Claim => self.claim(ui, ctx),
                Screen::Producer => self.producer(ui, ctx),
                Screen::History => self.history(ui, ctx),
                Screen::Settings => self.settings(ui, ctx),
                Screen::Help => self.help(ui, ctx),
            }
        });

        // Notification overlay.
        if let Some(note) = self.state.notifications.back().cloned() {
            let level_color = match note.level {
                NotificationLevel::Info => theme::color(ctx, "nerv_accent"),
                NotificationLevel::Success => theme::color(ctx, "nerv_success"),
                NotificationLevel::Warning => theme::color(ctx, "nerv_warning"),
                NotificationLevel::Error => theme::color(ctx, "nerv_error"),
            };
            egui::Area::new(egui::Id::("notification"))
                .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-8.0, -8.0))
                .show(ctx, |ui| {
                    egui::Frame::default()
                        .fill(theme::color(ctx, "nerv_surface"))
                        .stroke(egui::Stroke::new(1.0, level_color))
                        .corner_radius(6.0)
                        .inner_margin(egui::Margin::same(10))
                        .show(ui, |ui| {
                            ui.set_min_width(240.0);
                            ui.horizontal(|ui| {
                                ui.colored_label(level_color, "●");
                                ui.label(
                                    RichText::new(&note.message)
                                        .color(theme::color(ctx, "nerv_text"))
                                        .size(13.0),
                                );
                            });
                        });
                });
        }

        // Request repaint for notifications to auto-fade.
        if !self.state.notifications.is_empty() {
            ctx.request_repaint_after(std::time::Duration::from_millis(500));
        }
    }
}

// ---------------------------------------------------------------------------
// Screen implementations
// ---------------------------------------------------------------------------

impl NervApp {
    fn sync_badge(&mut self, ui: &mut egui::Ui) {
        let (label, color) = match self.state.sync {
            SyncStatus::Locked => ("🔒 LOCKED", theme::color(ui.ctx(), "nerv_muted")),
            SyncStatus::Scanning { .. } => ("⟳ SYNCING", theme::color(ui.ctx(), "nerv_warning")),
            SyncStatus::Synced { .. } => ("● SYNCED", theme::color(ui.ctx(), "nerv_success")),
            SyncStatus::Disconnected => ("✕ OFFLINE", theme::color(ui.ctx(), "nerv_error")),
        };
        ui.add_space(4.0);
        ui.label(RichText::new(label).color(color).size(12.0).strong());
        ui.add_space(8.0);
    }

    fn dashboard(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if !self.state.is_unlocked() {
            self.locked_screen(ui, ctx);
            return;
        }

        ui.add_space(16.0);

        // Balance hero card.
        egui::Frame::default()
            .fill(theme::color(ctx, "nerv_surface"))
            .corner_radius(8.0)
            .inner_margin(egui::Margin::same(24))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(
                    RichText::new("Total Balance")
                        .color(theme::color(ctx, "nerv_muted"))
                        .size(13.0),
                );
                ui.add_space(4.0);
                let balance_nerv = self.state.balance_nano() as f64 / 1e9;
                ui.label(
                    RichText::new(format!("{balance_nerv:.9} NERV"))
                        .color(theme::color(ctx, "nerv_text"))
                        .size(32.0)
                        .strong(),
                );
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    let addr_count = self.state.address_count();
                    let note_count = self.state.notes.len();
                    ui.label(
                        RichText::new(format!("{addr_count} addresses  ·  {note_count} notes"))
                            .color(theme::color(ctx, "nerv_muted"))
                            .size(12.0),
                    );
                });
            });

        ui.add_space(16.0);

        // Quick actions.
        ui.horizontal(|ui| {
            if ui
                .add_sized([120.0, 36.0], egui::Button::new("→  Send"))
                .clicked()
            {
                self.dispatch(WalletAction::StartSend);
            }
            if ui
                .add_sized([120.0, 36.0], egui::Button::new("←  Receive"))
                .clicked()
            {
                self.dispatch(WalletAction::Navigate(Screen::Receive));
            }
        });

        ui.add_space(16.0);

        // Recent activity.
        ui.label(
            RichText::new("Recent Activity")
                .color(theme::color(ctx, "nerv_text"))
                .size(16.0)
                .strong(),
        );
        ui.add_space(8.0);

        if self.state.history.is_empty() {
            ui.label(
                RichText::new("No transactions yet")
                    .color(theme::color(ctx, "nerv_muted"))
                    .size(13.0),
            );
        } else {
            egui::Frame::default()
                .fill(theme::color(ctx, "nerv_surface"))
                .corner_radius(8.0)
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    for entry in self.state.history.iter().take(8) {
                        self.history_row(ui, ctx, entry);
                        ui.add_space(2.0);
                    }
                });
        }
    }

    fn locked_screen(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.add_space(48.0);
        ui.vertical_centered(|ui| {
            ui.label(
                RichText::new("NERV")
                    .color(theme::color(ctx, "nerv_accent"))
                    .size(48.0)
                    .strong(),
            );
            ui.label(
                RichText::new("Post-Quantum Privacy Wallet")
                    .color(theme::color(ctx, "nerv_muted"))
                    .size(14.0),
            );
        });

        ui.add_space(32.0);

        ui.vertical_centered(|ui| {
            ui.set_max_width(360.0);

            if ui
                .add_sized([240.0, 40.0], egui::Button::new(
                    RichText::new("Create New Wallet").size(15.0).strong()
                ))
                .clicked()
            {
                self.dispatch(WalletAction::GenerateWallet);
            }

            ui.add_space(12.0);
            ui.label(
                RichText::new("— or —").color(theme::color(ctx, "nerv_muted")).size(12.0),
            );
            ui.add_space(12.0);

            ui.label(
                RichText::new("Import seed (64 hex chars):")
                    .color(theme::color(ctx, "nerv_text"))
                    .size(13.0),
            );
            ui.add_space(4.0);
            let response = ui.add(
                egui::TextEdit::singleline(&mut self.seed_input)
                    .hint_text("a1b2c3d4e5f6…")
                    .desired_width(320.0)
                    .font(egui::TextStyle::Monospace),
            );
            if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                self.dispatch(WalletAction::ImportSeed(self.seed_input.clone()));
            }
            ui.add_space(8.0);
            if ui.add_sized([120.0, 32.0], egui::Button::new("Import")).clicked() {
                self.dispatch(WalletAction::ImportSeed(self.seed_input.clone()));
            }
        });
    }

    fn send(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if !self.state.is_unlocked() {
            self.locked_screen(ui, ctx);
            return;
        }

        ui.add_space(16.0);
        ui.label(
            RichText::new("Send NERV")
                .color(theme::color(ctx, "nerv_text"))
                .size(20.0)
                .strong(),
        );
        ui.add_space(16.0);

        // Sync the text fields with the state's draft.
        let has_draft = self.state.draft.is_some();
        if !has_draft {
            self.dispatch(WalletAction::StartSend);
        }

        let accent = theme::color(ctx, "nerv_accent");
        let muted = theme::color(ctx, "nerv_muted");
        let text = theme::color(ctx, "nerv_text");
        let error = theme::color(ctx, "nerv_error");

        let max_width = 480.0;

        // Recipient.
        ui.set_max_width(max_width);
        ui.label(RichText::new("Recipient address").color(muted).size(13.0));
        let recipient_resp = ui.add(
            egui::TextEdit::singleline(&mut self.send_recipient)
                .hint_text("Paste the recipient's address (hex)…")
                .desired_width(max_width)
                .font(egui::TextStyle::Monospace),
        );
        if recipient_resp.changed() {
            self.dispatch(WalletAction::SetRecipient(self.send_recipient.clone()));
        }

        ui.add_space(12.0);

        // Amount.
        ui.label(RichText::new("Amount (nano-NERV)").color(muted).size(13.0));
        let amount_resp = ui.add(
            egui::TextEdit::singleline(&mut self.send_amount)
                .hint_text("0")
                .desired_width(200.0)
                .font(egui::TextStyle::Monospace),
        );
        if amount_resp.changed() {
            let amt: u64 = self.send_amount.parse().unwrap_or(0);
            self.dispatch(WalletAction::SetAmount(amt));
        }

        ui.add_space(12.0);

        // Fee.
        ui.label(RichText::new("Fee (nano-NERV)").color(muted).size(13.0));
        let fee_resp = ui.add(
            egui::TextEdit::singleline(&mut self.send_fee)
                .desired_width(200.0)
                .font(egui::TextStyle::Monospace),
        );
        if fee_resp.changed() {
            let fee: u64 = self.send_fee.parse().unwrap_or(0);
            self.dispatch(WalletAction::SetFee(fee));
        }

        ui.add_space(16.0);

        // Validation feedback.
        if let Some(draft) = &self.state.draft {
            if let Some(err) = &draft.validation.error {
                ui.label(
                    RichText::new(format!("⚠ {}", err.message()))
                        .color(theme::color(ctx, "nerv_warning"))
                        .size(13.0),
                );
            } else {
                let total = draft.amount_nano.saturating_add(draft.fee_nano);
                let total_nerv = total as f64 / 1e9;
                ui.label(
                    RichText::new(format!("✓ Ready — total: {total_nerv:.9} NERV (incl. fee)"))
                        .color(theme::color(ctx, "nerv_success"))
                        .size(13.0),
                );
            }
        }

        ui.add_space(16.0);

        // Buttons.
        ui.horizontal(|ui| {
            let is_ready = self
                .state
                .draft
                .as_ref()
                .map(|d| d.is_ready())
                .unwrap_or(false);

            let send_btn = egui::Button::new(
                RichText::new("Sign & Send").size(15.0).strong(),
            );
            let send_btn = if is_ready {
                send_btn.fill(accent).text_color(Color32::WHITE)
            } else {
                send_btn
            };
            if ui.add_sized([140.0, 38.0], send_btn).clicked() && is_ready {
                self.dispatch(WalletAction::SignAndSend);
            }

            ui.add_space(8.0);

            if ui.add_sized([100.0, 38.0], egui::Button::new("Cancel")).clicked() {
                self.send_recipient.clear();
                self.send_amount.clear();
                self.send_fee = "700".into();
                self.dispatch(WalletAction::CancelSend);
            }
        });
    }

    fn receive(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if !self.state.is_unlocked() {
            self.locked_screen(ui, ctx);
            return;
        }

        ui.add_space(16.0);
        ui.label(
            RichText::new("Receive NERV")
                .color(theme::color(ctx, "nerv_text"))
                .size(20.0)
                .strong(),
        );
        ui.add_space(16.0);

        if let Some(addr_hex) = self.state.primary_address_hex() {
            let accent = theme::color(ctx, "nerv_accent");
            let muted = theme::color(ctx, "nerv_muted");

            egui::Frame::default()
                .fill(theme::color(ctx, "nerv_surface"))
                .corner_radius(8.0)
                .inner_margin(egui::Margin::same(20))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(
                        RichText::new("Your primary address")
                            .color(muted)
                            .size(13.0),
                    );
                    ui.add_space(8.0);

                    // Truncated display.
                    let display = if addr_hex.len() > 72 {
                        format!("{}…{}", &addr_hex[..36], &addr_hex[addr_hex.len()-36..])
                    } else {
                        addr_hex.clone()
                    };
                    let copy_text = addr_hex.clone();
                    if ui
                        .add(
                            egui::Label::new(
                                RichText::new(display)
                                    .color(accent)
                                    .size(14.0)
                                    .monospace(),
                            )
                            .sense(Sense::click())
                            .selectable(true),
                        )
                        .on_hover_text("Click to select · Use the copy button below")
                        .clicked()
                    {
                        self.dispatch(WalletAction::CopyAddress);
                    }

                    ui.add_space(12.0);
                    ui.horizontal(|ui| {
                        if ui
                            .add_sized([110.0, 32.0], egui::Button::new("📋 Copy"))
                            .clicked()
                        {
                            self.dispatch(WalletAction::CopyAddress);
                        }
                        if let Some(clip) = &self.clipboard {
                            ui.label(
                                RichText::new("✓ copied")
                                    .color(theme::color(ctx, "nerv_success"))
                                    .size(12.0),
                            );
                        }
                    });
                });

            ui.add_space(16.0);

            // All addresses.
            ui.label(
                RichText::new("All Addresses")
                    .color(theme::color(ctx, "nerv_text"))
                    .size(16.0)
                    .strong(),
            );
            ui.add_space(8.0);

            if let Some(addrs) = &self.state.addresses {
                egui::Frame::default()
                    .fill(theme::color(ctx, "nerv_surface"))
                    .corner_radius(8.0)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        for addr in addrs.addresses() {
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new(format!("#{:>3}", addr.index))
                                        .color(muted)
                                        .size(12.0),
                                );
                                ui.label(
                                    RichText::new(format!("{:?}", addr.shard()))
                                        .color(accent)
                                        .size(12.0),
                                );
                                let ek_hex: String = addr
                                    .ek
                                    .as_bytes()
                                    .iter()
                                    .map(|b| format!("{b:02x}"))
                                    .collect();
                                let short = if ek_hex.len() > 24 {
                                    format!("{}…", &ek_hex[..24])
                                } else {
                                    ek_hex
                                };
                                ui.label(
                                    RichText::new(short)
                                        .color(theme::color(ctx, "nerv_text"))
                                        .size(11.0)
                                        .monospace(),
                                );
                            });
                            ui.add_space(2.0);
                        }
                    });
            }
        }
    }

    /// The Claim screen — pick a public claim bucket, enter an amount,
    /// and submit. The state machine owns the draft; the platform shell
    /// only renders and dispatches. The actual `verify_claim_leg`
    /// submission happens on a background thread spawned by
    /// `spawn_claim_pipeline`.
    fn claim(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if !self.state.is_unlocked() {
            self.locked_screen(ui, ctx);
            return;
        }
        ui.add_space(16.0);
        ui.heading("Claim NERV");
        ui.add_space(8.0);
        ui.label(
            "Pick a public claim bucket (1–3) and enter the amount in nano-NERV. \
             The Founder bucket is reserved for the platform shell's \
             authenticated path and is not exposed here.",
        );
        ui.add_space(12.0);

        // Ensure the draft exists when the screen opens (the state
        // machine creates it on `StartClaim`, but a deep-link to
        // `Screen::Claim` would otherwise have no draft).
        if self.state.claim_draft.is_none() {
            self.dispatch(WalletAction::StartClaim);
            return;
        }

        // Bucket picker (filtered to USER_BUCKETS — excludes Founder).
        let user_buckets: &[nerv_wallet_core::action::ClaimBucket] =
            &nerv_wallet_core::action::ClaimBucket::USER_BUCKETS;
        ui.horizontal(|ui| {
            ui.label("Bucket:");
            for (idx, bucket) in user_buckets.iter().enumerate() {
                let selected = self
                    .state
                    .claim_draft
                    .as_ref()
                    .and_then(|d| d.bucket)
                    == Some(*bucket);
                if ui
                    .selectable_label(selected, format!("{}. {}", idx + 1, bucket.label()))
                    .clicked()
                {
                    self.claim_bucket_index = idx;
                    self.dispatch(WalletAction::SetClaimBucket(*bucket));
                }
            }
        });

        ui.add_space(8.0);

        // Amount input (nano-NERV). Mirror the send screen's input
        // shape — numeric, with a clear error on non-numeric input.
        ui.horizontal(|ui| {
            ui.label("Amount (nano-NERV):");
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.claim_amount)
                    .hint_text("e.g. 1000000000")
                    .desired_width(240.0),
            );
            if resp.lost_focus()
                && let Ok(n) = self.claim_amount.trim().parse::<u64>()
            {
                self.dispatch(WalletAction::SetClaimAmount(n));
            }
        });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("NERV:");
            let nano_nerv = self
                .claim_amount
                .trim()
                .parse::<u64>()
                .map(|n| n as f64 / 1_000_000_000.0)
                .unwrap_or(0.0);
            ui.label(format!("{:.9}", nano_nerv));
        });

        ui.add_space(8.0);

        // Validation readout — mirror the draft validation state so
        // the user sees why the button is disabled.
        if let Some(draft) = &self.state.claim_draft {
            if let Some(err) = &draft.validation.error {
                ui.colored_label(egui::Color32::YELLOW, err.message());
            } else if draft.is_ready() {
                ui.colored_label(egui::Color32::GREEN, "Ready to claim");
            }
        }

        ui.add_space(12.0);

        // Submit button — only enabled when the draft is ready.
        let ready = self
            .state
            .claim_draft
            .as_ref()
            .map(|d| d.is_ready())
            .unwrap_or(false);
        if ui
            .add_enabled(ready, egui::Button::new("Claim"))
            .clicked()
        {
            self.dispatch(WalletAction::SignAndClaim);
        }

        ui.add_space(16.0);
        if ui.button("←  Back to Dashboard").clicked() {
            self.dispatch(WalletAction::CancelClaim);
        }
    }

    /// Producer screen (gap continuation). Mirrors `fn claim` —
    /// the state machine owns the draft; the GUI only renders +
    /// dispatches. The actual `ProducerIdentity` construction +
    /// `StakeLedger` registration happen on a background thread
    /// spawned by `spawn_producer_pipeline`.
    fn producer(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if !self.state.is_unlocked() {
            self.locked_screen(ui, ctx);
            return;
        }
        ui.add_space(16.0);
        ui.heading("Register a Producer");
        ui.add_space(8.0);
        ui.label(
            "Bond a stake, pick a shard, and register a producer identity. \
             The producer's signing key signs block headers + quorum certificates; \
             the custody Address receives the per-block subsidy payouts. The \
             testnet/dev path registers against a local StakeLedger; production \
             replaces this with a node-submitted RegisterStake transaction.",
        );
        ui.add_space(12.0);

        // Ensure the draft exists when the screen opens.
        if self.state.producer_draft.is_none() {
            self.dispatch(WalletAction::StartProducer);
            return;
        }

        let text = theme::color(ctx, "nerv_text");
        let muted = theme::color(ctx, "nerv_muted");
        let accent = theme::color(ctx, "nerv_accent");

        // === Seed input (hex). 64 hex chars = 32-byte seed.
        ui.horizontal(|ui| {
            ui.label("Seed (64 hex chars):");
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.producer_seed_input)
                    .hint_text("a1b2c3d4…")
                    .desired_width(420.0)
                    .font(egui::TextStyle::Monospace),
            );
            if resp.lost_focus() {
                self.dispatch(WalletAction::SetProducerSeed(
                    self.producer_seed_input.clone(),
                ));
            }
        });
        ui.add_space(8.0);

        // === Shard picker (0..=63). Slider for the first 64 shards.
        ui.horizontal(|ui| {
            ui.label("Shard:");
            // We use a numeric TextEdit so the operator can type any
            // value 0..=63; the state machine rejects out-of-range
            // values.
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.producer_shard_input)
                    .hint_text("0..=63")
                    .desired_width(60.0),
            );
            if resp.lost_focus()
                && let Ok(n) = self.producer_shard_input.trim().parse::<u64>()
            {
                self.dispatch(WalletAction::SetProducerShard(n));
            }
            ui.label(
                RichText::new("(genesis 64-shard committee)")
                    .color(muted)
                    .size(11.0),
            );
        });
        ui.add_space(8.0);

        // === Stake bond input (nano-NERV).
        ui.horizontal(|ui| {
            ui.label("Stake bond (nano-NERV):");
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.producer_stake_input)
                    .hint_text("e.g. 1000000000")
                    .desired_width(240.0),
            );
            if resp.lost_focus()
                && let Ok(n) = self.producer_stake_input.trim().parse::<u64>()
            {
                self.dispatch(WalletAction::SetProducerStake(n));
            }
        });
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("NERV:");
            let nano_nerv = self
                .producer_stake_input
                .trim()
                .parse::<u64>()
                .map(|n| n as f64 / 1_000_000_000.0)
                .unwrap_or(0.0);
            ui.label(format!("{:.9}", nano_nerv));
        });
        ui.add_space(8.0);

        // === Validation readout.
        if let Some(draft) = &self.state.producer_draft {
            if let Some(err) = &draft.validation.error {
                ui.colored_label(egui::Color32::YELLOW, err.message());
            } else if draft.is_ready() {
                ui.colored_label(
                    egui::Color32::GREEN,
                    "Ready to register the producer",
                );
            }
        }

        ui.add_space(12.0);

        // === Submit / Cancel.
        let ready = self
            .state
            .producer_draft
            .as_ref()
            .map(|d| d.is_ready())
            .unwrap_or(false);
        ui.horizontal(|ui| {
            if ui
                .add_enabled(ready, egui::Button::new("Register Producer"))
                .clicked()
            {
                self.dispatch(WalletAction::RegisterProducerStake);
            }
            if ui.button("Cancel").clicked() {
                self.dispatch(WalletAction::CancelProducer);
            }
        });

        ui.add_space(16.0);

        // === Live producer-state card (if registered).
        if self.state.producer_state.stake_registered {
            egui::Frame::default()
                .fill(theme::color(ctx, "nerv_surface"))
                .corner_radius(8.0)
                .inner_margin(egui::Margin::same(16))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(
                        RichText::new("● Producer Registered")
                            .color(egui::Color32::GREEN)
                            .size(14.0)
                            .strong(),
                    );
                    ui.add_space(6.0);
                    let s = &self.state.producer_state;
                    ui.label(format!(
                        "Shard: {}   Stake: {} NERV",
                        s.assigned_shard,
                        s.stake_balance_nano as f64 / 1_000_000_000.0,
                    ));
                    ui.add_space(2.0);
                    let vk_short = if s.verifying_key_hex.len() > 24 {
                        format!("{}…", &s.verifying_key_hex[..24])
                    } else {
                        s.verifying_key_hex.clone()
                    };
                    ui.label(
                        RichText::new(format!("VK: {vk_short}"))
                            .color(text)
                            .size(11.0)
                            .font(egui::TextStyle::Monospace),
                    );
                    ui.label(
                        RichText::new(format!("Payout: {}", &s.payout_address_hex))
                            .color(muted)
                            .size(11.0)
                            .font(egui::TextStyle::Monospace),
                    );
                    if s.total_payouts_nano > 0 {
                        ui.label(format!(
                            "Total payouts: {} NERV   (epoch: {} NERV)",
                            s.total_payouts_nano as f64 / 1_000_000_000.0,
                            s.epoch_payout_nano as f64 / 1_000_000_000.0,
                        ));
                    }
                });
            ui.add_space(8.0);
            ui.label(
                RichText::new(
                    "Producer is now a registered validator — blocks it signs are \
                     counted toward the per-shard quorum.",
                )
                .color(accent)
                .size(11.0),
            );
        }

        ui.add_space(16.0);
        if ui.button("←  Back to Dashboard").clicked() {
            self.dispatch(WalletAction::CancelProducer);
        }
    }

    fn history(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if !self.state.is_unlocked() {
            self.locked_screen(ui, ctx);
            return;
        }

        ui.add_space(16.0);
        ui.label(
            RichText::new("Transaction History")
                .color(theme::color(ctx, "nerv_text"))
                .size(20.0)
                .strong(),
        );
        ui.add_space(16.0);

        if self.state.history.is_empty() {
            ui.label(
                RichText::new("No transactions yet")
                    .color(theme::color(ctx, "nerv_muted"))
                    .size(14.0),
            );
            return;
        }

        egui::Frame::default()
            .fill(theme::color(ctx, "nerv_surface"))
            .corner_radius(8.0)
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                for entry in self.state.history.iter() {
                    self.history_row(ui, ctx, entry);
                    ui.add_space(4.0);
                }
            });
    }

    fn history_row(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, entry: &HistoryEntry) {
        let (arrow, color) = match entry.direction {
            Direction::Incoming => ("↓", theme::color(ctx, "nerv_success")),
            Direction::Outgoing => ("↑", theme::color(ctx, "nerv_error")),
        };
        let muted = theme::color(ctx, "nerv_muted");

        ui.horizontal(|ui| {
            ui.label(RichText::new(arrow).color(color).size(16.0));
            ui.add_space(8.0);

            let amount = entry.amount_nano as f64 / 1e9;
            ui.label(
                RichText::new(format!("{amount:.9} NERV"))
                    .color(color)
                    .size(14.0),
            );

            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let conf = if entry.confirmations > 0 {
                    format!("✓{}", entry.confirmations)
                } else {
                    "⋯".to_string()
                };
                ui.label(RichText::new(&conf).color(muted).size(11.0));
                ui.add_space(8.0);
                ui.label(
                    RichText::new(format!("block {}", entry.height))
                        .color(muted)
                        .size(11.0),
                );
            });
        });
    }

    fn settings(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if !self.state.is_unlocked() {
            self.locked_screen(ui, ctx);
            return;
        }

        ui.add_space(16.0);
        ui.label(
            RichText::new("Settings")
                .color(theme::color(ctx, "nerv_text"))
                .size(20.0)
                .strong(),
        );
        ui.add_space(16.0);

        let muted = theme::color(ctx, "nerv_muted");
        let text = theme::color(ctx, "nerv_text");

        ui.set_max_width(480.0);

        ui.label(RichText::new("Network").color(muted).size(13.0));
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Peers:").color(muted).size(13.0));
            ui.label(
                RichText::new(self.state.peer_count.to_string())
                    .color(text)
                    .size(14.0),
            );
            ui.add_space(16.0);
            ui.label(RichText::new("Chain height:").color(muted).size(13.0));
            ui.label(
                RichText::new(self.state.chain_height.to_string())
                    .color(text)
                    .size(14.0),
            );
        });

        ui.add_space(20.0);

        ui.label(RichText::new("Wallet").color(muted).size(13.0));
        ui.add_space(4.0);
        ui.label(
            RichText::new(format!("Addresses: {}", self.state.address_count()))
                .color(text)
                .size(14.0),
        );
        ui.label(
            RichText::new(format!("Notes: {}", self.state.notes.len()))
                .color(text)
                .size(14.0),
        );

        ui.add_space(20.0);

        if ui
            .add_sized([160.0, 36.0], egui::Button::new(
                RichText::new("🔒 Lock Wallet").size(14.0),
            ))
            .clicked()
        {
            self.dispatch(WalletAction::Lock);
        }
    }

    fn help(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.add_space(16.0);
        ui.label(
            RichText::new("Help & Shortcuts")
                .color(theme::color(ctx, "nerv_text"))
                .size(20.0)
                .strong(),
        );
        ui.add_space(16.0);

        let muted = theme::color(ctx, "nerv_muted");
        let text = theme::color(ctx, "nerv_text");

        ui.set_max_width(520.0);

        let shortcuts = [
            ("Navigate", "Click the sidebar or use the number keys 1-6"),
            ("Send", "Fill the form, then click Sign & Send"),
            ("Copy address", "Click the address or the Copy button on the Receive screen"),
            ("Dismiss notification", "Click the ✕ or press N"),
            ("Lock wallet", "Settings → Lock Wallet, or the sidebar's lock button"),
        ];

        for (action, how) in shortcuts {
            ui.horizontal(|ui| {
                ui.label(RichText::new(action).color(text).size(13.0).strong());
                ui.label(RichText::new(format!("— {how}")).color(muted).size(13.0));
            });
            ui.add_space(4.0);
        }

        ui.add_space(20.0);
        ui.separator();
        ui.add_space(8.0);

        ui.label(
            RichText::new("About")
                .color(text)
                .size(16.0)
                .strong(),
        );
        ui.add_space(4.0);
        ui.label(
            RichText::new(
                "NERV is a post-quantum, privacy-first, sharded blockchain.\n\
                 This wallet uses ML-DSA-65 signatures, ML-KEM-768 key exchange,\n\
                 and STARK proofs — no elliptic curves anywhere.",
            )
            .color(muted)
            .size(13.0),
        );
        ui.add_space(8.0);
        ui.label(
            RichText::new("Version 0.1.0  ·  MIT / Apache-2.0")
                .color(muted)
                .size(11.0),
        );
    }
}

// ───────────────────────────────────────────────────────────────────────
// WASM ⇄ JS bridge helper (erratum 205)
//
// The native chrome (Web bottom-nav, Android Kotlin bottom-nav, iOS SwiftUI
// bottom-nav) sends screen names as strings to the wasm-exposed
// `nerv_navigate(&str)` function. The names match the `Screen::title()`
// strings. This helper maps them back to `Screen` variants for
// `WalletAction::Navigate`. Lives outside `NervApp` so it can be
// unit-tested without spinning up the eframe runtime.
// ───────────────────────────────────────────────────────────────────────

/// Map a JS-side screen name (matching `Screen::title()` on the
/// desktop GUI side) to a `Screen` variant. Returns `None` for
/// unknown names; the caller (the bridge-queue drain) ignores
/// `None` and leaves the wallet UI on its current screen.
pub(crate) fn screen_from_str(name: &str) -> Option<Screen> {
    Some(match name {
        "Dashboard" => Screen::Dashboard,
        "Send" => Screen::Send,
        "Receive" => Screen::Receive,
        "Claim" => Screen::Claim,
        "Producer" => Screen::Producer,
        "History" => Screen::History,
        "Settings" => Screen::Settings,
        "Help" => Screen::Help,
        _ => return None,
    })
}

#[cfg(test)]
mod bridge_tests {
    use super::*;
    use nerv_wallet_core::update;
    use nerv_wallet_core::WalletAction;

    /// Every `Screen` variant must round-trip through `screen_from_str`
    /// using the JS-side name (`Screen::title()`). If the desktop GUI
    /// renames a screen, this test catches the divergence.
    #[test]
    fn screen_from_str_round_trips_every_variant() {
        for &screen in Screen::ALL.iter() {
            let name = screen.title();
            assert_eq!(
                screen_from_str(name),
                Some(screen),
                "screen_from_str failed to round-trip {name:?}"
            );
        }
    }

    #[test]
    fn screen_from_str_rejects_unknown_names() {
        assert_eq!(screen_from_str("Bogus"), None);
        assert_eq!(screen_from_str(""), None);
        assert_eq!(screen_from_str("dashboard"), None); // case-sensitive
    }

    /// Dispatching `WalletAction::Navigate(screen)` for a screen name
    /// resolved by `screen_from_str` updates `state.screen`. This is
    /// what the JS bridge does each frame after `take_pending_nav`.
    #[test]
    fn dispatch_navigate_changes_screen() {
        let mut state = nerv_wallet_core::WalletState::new();
        assert_eq!(state.screen, Screen::Dashboard);
        let screen = screen_from_str("Send").unwrap();
        let _events = update(&mut state, WalletAction::Navigate(screen));
        assert_eq!(state.screen, Screen::Send);
        // The Send handler in `update.rs` auto-creates a draft.
        assert!(state.draft.is_some());
    }
}
