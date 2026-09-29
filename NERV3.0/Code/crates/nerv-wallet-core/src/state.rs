
//! The observable wallet state (erratum 200): plain data, no platform
//! types, no I/O. Every UI renders this.

use std::collections::VecDeque;

use nerv_core::hash::Hash256;
use nerv_core::types::{ShardId, TxId};
use nerv_custody::WalletKeys;
use nerv_wallet::keys::AddressSet;
use nerv_wallet::scan::{ScannedNote, WalletNoteSet};

/// Which screen the UI is showing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Screen {
    Dashboard,
    Send,
    Receive,
    Claim,
    Producer,
    History,
    Settings,
    Help,
}

impl Screen {
    pub const ALL: [Screen; 8] = [
        Screen::Dashboard,
        Screen::Send,
        Screen::Receive,
        Screen::Claim,
        Screen::Producer,
        Screen::History,
        Screen::Settings,
        Screen::Help,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Screen::Dashboard => "Dashboard",
            Screen::Send => "Send",
            Screen::Receive => "Receive",
            Screen::Claim => "Claim",
            Screen::Producer => "Producer",
            Screen::History => "History",
            Screen::Settings => "Settings",
            Screen::Help => "Help",
        }
    }

    pub fn index(self) -> usize {
        match self {
            Screen::Dashboard => 0,
            Screen::Send => 1,
            Screen::Receive => 2,
            Screen::Claim => 3,
            Screen::Producer => 4,
            Screen::History => 5,
            Screen::Settings => 6,
            Screen::Help => 7,
        }
    }

    pub fn from_index(i: usize) -> Screen {
        Screen::ALL[i % Screen::ALL.len()]
    }
}

/// The sync status of the wallet's view of the chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncStatus {
    /// No wallet loaded — the app just opened.
    Locked,
    /// Wallet loaded, scanning the chain for notes.
    Scanning { progress_permille: u64 },
    /// Fully synced to the given height.
    Synced { height: u64 },
    /// Connection lost.
    Disconnected,
}

/// The send flow's in-progress draft with inline validation state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TransactionDraft {
    pub recipient_hex: String,
    pub amount_nano: u64,
    pub fee_nano: u64,
    pub expiry_height: u64,
    /// The last validation result (for inline UI errors).
    pub validation: DraftValidation,
}

/// The emission buckets the wallet UI exposes for the claim rail. Each
/// maps to a `[[economy.buckets]]` entry in `specs/params.toml` (WP §12.2).
/// Adding a new bucket requires a parallel enum variant AND a
/// `bucket_name()` arm so the platform shell can route the leg to the
/// correct emission tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClaimBucket {
    /// Public shielded-claim window, 24 months, burn-unclaimed (§12.2).
    Community,
    /// Governance-signed milestone grants (§12.2).
    Ecosystem,
    /// Foundation operational budget (§12.2).
    Foundation,
    /// Founder allocation — sigmoid/vested (§12.2). The wallet does not
    /// expose this to end users; it is reserved for the platform shell's
    /// founder-recovery flow.
    Founder,
}

impl ClaimBucket {
    /// The canonical `[[economy.buckets]]` name (matches `name = "..."`
    /// in `specs/params.toml`). The platform shell passes this to the
    /// emission tree's lookup.
    pub fn name(self) -> &'static str {
        match self {
            ClaimBucket::Community => "community",
            ClaimBucket::Ecosystem => "ecosystem",
            ClaimBucket::Foundation => "foundation",
            ClaimBucket::Founder => "founder",
        }
    }

    /// User-facing label.
    pub fn label(self) -> &'static str {
        match self {
            ClaimBucket::Community => "Community (claim window, 24 months)",
            ClaimBucket::Ecosystem => "Ecosystem (milestone grants)",
            ClaimBucket::Foundation => "Foundation (operational)",
            ClaimBucket::Founder => "Founder (vested)",
        }
    }

    /// The buckets available to end users via the public claim rail. The
    /// `Founder` bucket is intentionally excluded here — the platform
    /// shell drives founder claims through a separate authenticated path.
    pub const USER_BUCKETS: [ClaimBucket; 3] = [
        ClaimBucket::Community,
        ClaimBucket::Ecosystem,
        ClaimBucket::Foundation,
    ];
}

/// The claim pipeline's stages (mirror of `SendStage`). Drives the UI
/// progress display and the platform shell's status notifications.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClaimStage {
    /// Deriving the `ClaimLegParts` (commitment / nullifier / eligibility)
    /// from the user's seed and OS entropy.
    Deriving,
    /// Submitting the leg to the registry's emission tree.
    Submitting,
    /// Verified — nullifier spent, account credited, balance updated.
    Verified,
}

impl ClaimStage {
    pub fn label(self) -> &'static str {
        match self {
            ClaimStage::Deriving => "Deriving claim commitment…",
            ClaimStage::Submitting => "Submitting claim leg…",
            ClaimStage::Verified => "Claim verified, balance credited",
        }
    }
}

/// The claim flow's in-progress draft. Mirrors `TransactionDraft` for
/// the send flow; validation here is lighter (amount > 0 and the bucket's
/// window still open).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClaimDraft {
    pub bucket: Option<ClaimBucket>,
    pub amount_nano: u64,
    /// The last validation result (for inline UI errors).
    pub validation: ClaimValidation,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClaimValidation {
    pub bucket_ok: bool,
    pub amount_ok: bool,
    pub window_open: bool,
    /// The first error to display (None = ready to claim).
    pub error: Option<ClaimError>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClaimError {
    NoBucket,
    AmountZero,
    WindowClosed,
}

impl ClaimError {
    pub fn message(&self) -> String {
        match self {
            ClaimError::NoBucket => "Pick a claim bucket".into(),
            ClaimError::AmountZero => "Claim amount must be greater than zero".into(),
            ClaimError::WindowClosed => "The bucket's claim window has closed".into(),
        }
    }
}

impl ClaimDraft {
    pub fn is_ready(&self) -> bool {
        self.validation.error.is_none()
    }

    /// Validate against the platform-reported window-open status. The
    /// window is open iff the bucket is currently claimable per the
    /// emission schedule (`specs/params.toml::[economy.buckets].*`).
    /// `window_open` is the platform shell's report of that fact.
    pub fn validate(&mut self, window_open: bool) {
        let mut v = ClaimValidation {
            bucket_ok: self.bucket.is_some(),
            amount_ok: self.amount_nano > 0,
            window_open,
            error: None,
        };
        if self.bucket.is_none() {
            v.error = Some(ClaimError::NoBucket);
        } else if self.amount_nano == 0 {
            v.error = Some(ClaimError::AmountZero);
        } else if !window_open {
            v.error = Some(ClaimError::WindowClosed);
        }
        self.validation = v;
    }
}

// ---- Producer state types (gap continuation) -----------------------------

/// The producer setup form (in-progress). Created by `StartProducer`,
/// updated by the operator, and submitted via `RegisterProducerStake`.
/// The wallet-core state machine owns the draft; the platform shell
/// owns the actual ML-DSA key + payout Address construction (which
/// requires the user's seed — the wallet-core never touches the seed
/// directly).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProducerDraft {
    /// 32-byte master seed (hex-encoded in the UI). `None` means the
    /// operator hasn't picked a seed yet.
    pub seed: Option<[u8; 32]>,
    /// The producer's stake bond in nano-NERV.
    pub stake_nano: u64,
    /// The producer's assigned shard (the subsidy split is per-shard).
    pub shard: Option<u64>,
    /// The draft's last validation result.
    pub validation: ProducerValidation,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProducerValidation {
    pub seed_ok: bool,
    pub stake_ok: bool,
    pub shard_ok: bool,
    /// The first error to display (None = ready to register).
    pub error: Option<ProducerError>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProducerError {
    NoSeed,
    NoShard,
    StakeZero,
    SeedInvalid,
}

impl ProducerError {
    pub fn message(&self) -> String {
        match self {
            ProducerError::NoSeed => "Enter a 32-byte (64 hex char) seed".into(),
            ProducerError::NoShard => "Pick the shard you produce for".into(),
            ProducerError::StakeZero => "Stake bond must be greater than zero".into(),
            ProducerError::SeedInvalid => "Seed must be 64 hex chars (32 bytes)".into(),
        }
    }
}

impl ProducerDraft {
    pub fn is_ready(&self) -> bool {
        self.validation.error.is_none()
    }

    /// Validate the draft. `seed_hex` is the raw text from the UI's
    /// seed field — the platform shell passes it in after parsing.
    pub fn validate(&mut self, seed_hex: Option<&str>) {
        let seed_ok = self.seed.is_some()
            || seed_hex
                .map(|s| {
                    let trimmed = s.trim();
                    trimmed.len() == 64 && trimmed.chars().all(|c| c.is_ascii_hexdigit())
                })
                .unwrap_or(false);
        let stake_ok = self.stake_nano > 0;
        let shard_ok = self.shard.is_some();
        let mut v = ProducerValidation {
            seed_ok,
            stake_ok,
            shard_ok,
            error: None,
        };
        if !seed_ok {
            v.error = Some(match seed_hex {
                Some(s) if !is_valid_seed_hex(s) => ProducerError::SeedInvalid,
                _ => ProducerError::NoSeed,
            });
        } else if !shard_ok {
            v.error = Some(ProducerError::NoShard);
        } else if !stake_ok {
            v.error = Some(ProducerError::StakeZero);
        }
        self.validation = v;
    }
}

/// The live producer state, as reported by the connected node /
/// consensus layer. The wallet-core treats this as read-only — the
/// operator inspects it on the Producer screen and the platform shell
/// updates it via `WalletAction::ProducerStateRefreshed`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProducerState {
    /// ML-DSA-65 verifying key (32-byte hash, hex-encoded in the UI).
    pub verifying_key_hex: String,
    /// Custody Address for payouts (hex-encoded).
    pub payout_address_hex: String,
    /// Stake balance in nano-NERV (read from the consensus `StakeLedger`).
    pub stake_balance_nano: u64,
    /// Total subsidies earned, in nano-NERV (cumulative).
    pub total_payouts_nano: u64,
    /// Current epoch's per-block subsidy, in nano-NERV.
    pub epoch_payout_nano: u64,
    /// The producer's assigned shard.
    pub assigned_shard: u64,
    /// The producer's role at the connected node (e.g. "Validator",
    /// "Producer", "Committee"). Empty if the node isn't running a
    /// producer role.
    pub node_role: String,
    /// Whether the stake is currently registered with the consensus
    /// ledger (true = registered, false = pending).
    pub stake_registered: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DraftValidation {
    pub recipient_ok: bool,
    pub amount_ok: bool,
    pub fee_ok: bool,
    /// The first error message to display (None = ready to confirm).
    pub error: Option<DraftError>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DraftError {
    EmptyRecipient,
    InvalidRecipientHex,
    AmountZero,
    InsufficientFunds { available: u64, needed: u64 },
    FeeTooLow { minimum: u64 },
    NotSynced,
}

impl DraftError {
    pub fn message(&self) -> String {
        match self {
            DraftError::EmptyRecipient => "Enter a recipient address".into(),
            DraftError::InvalidRecipientHex => "Invalid address (expected hex-encoded ML-KEM key)".into(),
            DraftError::AmountZero => "Amount must be greater than zero".into(),
            DraftError::InsufficientFunds { available, needed } => {
                format!("Insufficient funds: {available} available, {needed} needed")
            }
            DraftError::FeeTooLow { minimum } => {
                format!("Fee below the network minimum ({minimum} nano)")
            }
            DraftError::NotSynced => "Wallet is still syncing".into(),
        }
    }
}

impl TransactionDraft {
    pub fn is_ready(&self) -> bool {
        self.validation.error.is_none()
    }

    /// Validate the draft against the wallet's current state.
    pub fn validate(&mut self, available_nano: u64, synced: bool) {
        let mut v = DraftValidation {
            recipient_ok: !self.recipient_hex.is_empty(),
            amount_ok: self.amount_nano > 0,
            fee_ok: self.fee_nano >= 1000,
            error: None,
        };
        if self.recipient_hex.is_empty() {
            v.error = Some(DraftError::EmptyRecipient);
        } else if self.recipient_hex.len() != 2368 || !self.recipient_hex.chars().all(|c| c.is_ascii_hexdigit()) {
            v.recipient_ok = false;
            v.error = Some(DraftError::InvalidRecipientHex);
        } else if self.amount_nano == 0 {
            v.error = Some(DraftError::AmountZero);
        } else if self.fee_nano < 700 {
            v.error = Some(DraftError::FeeTooLow { minimum: 700 });
        } else if !synced {
            v.error = Some(DraftError::NotSynced);
        } else {
            let needed = self.amount_nano.saturating_add(self.fee_nano);
            if available_nano < needed {
                v.error = Some(DraftError::InsufficientFunds { available: available_nano, needed });
            }
        }
        self.validation = v;
    }
}

/// One entry in the transaction history (derived from scan + sent records).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryEntry {
    pub txid: TxId,
    pub direction: Direction,
    pub amount_nano: u64,
    pub fee_nano: u64,
    pub height: u64,
    pub confirmations: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Incoming,
    Outgoing,
}

/// The full wallet state — everything any UI needs to render.
#[derive(Clone, Debug)]
pub struct WalletState {
    /// The master seed (None = locked/no wallet).
    pub seed: Option<[u8; 32]>,
    /// The derived wallet keys.
    pub keys: Option<WalletKeys>,
    /// The diversified address set.
    pub addresses: Option<AddressSet>,
    /// The wallet's note set (scanned + spent tracking).
    pub notes: WalletNoteSet,
    /// The current screen.
    pub screen: Screen,
    /// The send-flow draft (present when on the Send screen).
    pub draft: Option<TransactionDraft>,
    /// The claim-flow draft (present when on the Claim screen).
    pub claim_draft: Option<ClaimDraft>,
    /// The producer-flow draft (present when on the Producer screen).
    pub producer_draft: Option<ProducerDraft>,
    /// The live producer state, refreshed from the connected node /
    /// consensus layer via `WalletAction::ProducerStateRefreshed`. Empty
    /// until the platform shell has reported the first observation.
    pub producer_state: ProducerState,
    /// The chain height the wallet has scanned to.
    pub chain_height: u64,
    /// Sync status.
    pub sync: SyncStatus,
    /// Transaction history (most recent first).
    pub history: VecDeque<HistoryEntry>,
    /// The number of connected peers (from the host).
    pub peer_count: usize,
    /// Notifications the UI should display (toast/banner).
    pub notifications: VecDeque<Notification>,
    /// Whether the wallet is exiting.
    pub quitting: bool,
}

/// A UI notification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Notification {
    pub level: NotificationLevel,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotificationLevel {
    Info,
    Success,
    Warning,
    Error,
}

impl Default for WalletState {
    fn default() -> Self {
        WalletState::new()
    }
}

impl WalletState {
    pub fn new() -> WalletState {
        WalletState {
            seed: None,
            keys: None,
            addresses: None,
            notes: WalletNoteSet::new(),
            screen: Screen::Dashboard,
            draft: None,
            claim_draft: None,
            producer_draft: None,
            producer_state: ProducerState::default(),
            chain_height: 0,
            sync: SyncStatus::Locked,
            history: VecDeque::new(),
            peer_count: 0,
            notifications: VecDeque::new(),
            quitting: false,
        }
    }

    /// The total unspent balance in nano-NERV.
    pub fn balance_nano(&self) -> u64 {
        self.notes.unspent_value()
    }

    /// Whether a wallet is loaded and unlocked.
    pub fn is_unlocked(&self) -> bool {
        self.seed.is_some()
    }

    /// The first address (for the Receive screen).
    pub fn primary_address_hex(&self) -> Option<String> {
        self.addresses
            .as_ref()
            .and_then(|a| a.addresses().first())
            .map(|addr| {
                addr.ek.as_bytes().iter().map(|b| format!("{b:02x}")).collect()
            })
    }

    /// The number of shard-covered addresses.
    pub fn address_count(&self) -> usize {
        self.addresses.as_ref().map_or(0, |a| a.len())
    }

    /// Push a notification (capped at 10).
    pub fn notify(&mut self, level: NotificationLevel, message: String) {
        self.notifications.push_back(Notification { level, message });
        while self.notifications.len() > 10 {
            self.notifications.pop_front();
        }
    }

    /// Dismiss the oldest notification.
    pub fn dismiss_notification(&mut self) {
        self.notifications.pop_front();
    }
}

/// Returns `true` iff `s` is a valid 64-hex-char string (with optional
/// surrounding whitespace). Used by `ProducerDraft::validate` to
/// distinguish "no seed entered" (`NoSeed`) from "garbage entered"
/// (`SeedInvalid`).
fn is_valid_seed_hex(s: &str) -> bool {
    let trimmed = s.trim();
    trimmed.len() == 64 && trimmed.chars().all(|c| c.is_ascii_hexdigit())
}
