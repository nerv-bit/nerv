//! User intents (erratum 200): every possible action a user can take,
//! from any platform's UI. The update function is the single handler.

use crate::state::{ClaimBucket, ClaimStage, ProducerDraft, ProducerError, ProducerState, Screen};

/// Every user intent in the wallet. Each platform's UI translates its
/// native input events into these actions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WalletAction {
    // === Lifecycle ===
    GenerateWallet,
    ImportSeed(String),
    Lock,
    Quit,

    // === Navigation ===
    Navigate(Screen),
    NextTab,
    PrevTab,

    // === Send flow ===
    /// Start composing a payment.
    StartSend,
    /// Update the recipient address field.
    SetRecipient(String),
    /// Update the amount field.
    SetAmount(u64),
    /// Update the fee field.
    SetFee(u64),
    /// The user pressed "confirm" — validate and prepare for signing.
    ConfirmSend,
    /// The user pressed "send" — the platform shell initiates the
    /// construct→prove→send pipeline.
    SignAndSend,
    /// Cancel the draft.
    CancelSend,

    // === Claim flow ===
    /// Start composing a claim (errata 200/161).
    StartClaim,
    /// Switch to a different claim bucket.
    SetClaimBucket(ClaimBucket),
    /// Update the claim-amount field on the active draft.
    SetClaimAmount(u64),
    /// The user pressed "claim" — the platform shell derives the
    /// `ClaimLegParts` from the user's seed and submits the leg to the
    /// registry's claim rail.
    SignAndClaim,
    /// Cancel the claim draft.
    CancelClaim,

    // === Producer flow (gap continuation) ===
    /// Start composing a producer registration — pick a seed, shard,
    /// and stake bond. The state machine owns the draft; the platform
    /// shell owns the actual `ProducerIdentity` construction (which
    /// needs the user's seed).
    StartProducer,
    /// Update the producer's seed from a 64-hex-char string. The state
    /// machine parses the hex; on parse failure the draft's validation
    /// surfaces `ProducerError::SeedInvalid`.
    SetProducerSeed(String),
    /// Update the producer's stake bond in nano-NERV.
    SetProducerStake(u64),
    /// Update the producer's assigned shard (0..=63).
    SetProducerShard(u64),
    /// The user pressed "register" — the state machine validates the
    /// draft and emits `WalletEvent::RegisterProducerStake` for the
    /// platform shell, which builds the `ProducerIdentity` and
    /// registers the stake against the consensus `StakeLedger`.
    RegisterProducerStake,
    /// Cancel the producer draft.
    CancelProducer,

    // === Receive ===
    /// Copy the primary address to the platform clipboard.
    CopyAddress,
    /// Generate a fresh address (for one-time use).
    NewAddress,

    // === History ===
    SelectHistoryEntry(usize),
    /// Filter history by direction.
    FilterHistory(Option<crate::state::Direction>),

    // === Settings ===
    /// Update the shard coverage target.
    SetCoverage(usize),
    /// Toggle fee bucketing.
    ToggleFeeBucketing,

    // === Notifications ===
    DismissNotification,

    // === External events (the platform shell reports these) ===
    /// The chain height advanced (from a connected node).
    ChainHeightAdvanced(u64),
    /// The peer count changed.
    PeerCountChanged(usize),
    /// A note was received (from scanning).
    NoteReceived(ScannedNote),
    /// A transaction was confirmed on-chain.
    TransactionConfirmed { txid: TxId, height: u64 },
    /// The send pipeline's stages (for progress display).
    SendProgress(SendStage),
    /// The send pipeline failed.
    SendFailed(String),
    /// The send pipeline succeeded.
    SendSucceeded(TxId),
    /// The claim pipeline reported a stage transition.
    ClaimProgress(ClaimStage),
    /// The claim pipeline failed (rejection, double-claim, etc.).
    ClaimFailed(String),
    /// The claim pipeline succeeded — the wallet's emission account was
    /// credited. The platform shell surfaces this as a notification.
    ClaimSucceeded {
        bucket: ClaimBucket,
        amount_nano: u64,
    },

    /// The producer registration succeeded — the platform shell reports
    /// the producer's verifying key (hex) and payout Address (hex) plus
    /// the consensus ledger's recorded balance for this producer. The
    /// state machine stores these in `state.producer_state` and clears
    /// the draft; the UI shows the producer card.
    ProducerStakeRegistered {
        verifying_key_hex: String,
        payout_address_hex: String,
        stake_balance_nano: u64,
        account_id: [u8; 32],
    },
    /// The producer registration failed (stake rejected, signing-key
    /// derivation error, etc.). The state machine surfaces the reason
    /// and returns the user to the dashboard.
    ProducerRegistrationFailed(String),
    /// The platform shell pushed a fresh producer-state snapshot
    /// (periodically or after an epoch boundary). The state machine
    /// overwrites `state.producer_state` with the new values.
    ProducerStateRefreshed(ProducerState),
}

/// The send pipeline's stages for progress display.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SendStage {
    Constructing,
    Proving,
    Sealing,
    Routing,
    Submitted,
}

impl SendStage {
    pub fn label(self) -> &'static str {
        match self {
            SendStage::Constructing => "Constructing transaction…",
            SendStage::Proving => "Generating proof (0.8–2 s)…",
            SendStage::Sealing => "Sealing delta…",
            SendStage::Routing => "Routing through mixnet…",
            SendStage::Submitted => "Submitted to 3 aggregators",
        }
    }
}
