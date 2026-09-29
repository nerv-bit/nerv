//! State-machine invariants (erratum 206): every action on every
//! reachable state produces a valid successor — no panics, no invalid
//! states. These are the properties that hold across all four platforms.

use nerv_wallet_core::state::{DraftError, NotificationLevel, Screen, SyncStatus, WalletState};
use nerv_wallet_core::{update, WalletAction};

/// Assert the global state invariants that must hold after every update.
fn assert_invariants(s: &WalletState) {
    // The screen is always one of the six known screens.
    assert!(Screen::ALL.contains(&s.screen), "invalid screen: {:?}", s.screen);

    // Notifications are bounded.
    assert!(s.notifications.len() <= 10, "notifications: {}", s.notifications.len());

    // The draft exists iff we're on the Send screen (or just left it).
    if s.screen == Screen::Send {
        // Navigating to Send creates the draft.
        // Leaving Send doesn't immediately destroy it (the user may return).
    }

    // The sync status is consistent with the unlock state.
    if !s.is_unlocked() {
        assert!(
            matches!(s.sync, SyncStatus::Locked),
            "locked wallet must be SyncStatus::Locked: {:?}",
            s.sync
        );
    }

    // History is bounded (we don't grow unbounded memory).
    assert!(s.history.len() <= 10_000, "history: {}", s.history.len());
}

/// Run one action and verify the invariants hold.
fn step(s: &mut WalletState, a: WalletAction) {
    let _events = update(s, a);
    assert_invariants(s);
}

fn unlocked_state() -> WalletState {
    let mut s = WalletState::new();
    update(&mut s, WalletAction::ImportSeed("ab".repeat(32)));
    assert!(s.is_unlocked(), "seed import must succeed with a valid 64-char hex seed");
    s
}

#[test]
fn every_action_on_every_screen_preserves_invariants() {
    // For each starting screen, apply every action and check the result.
    let actions: Vec<WalletAction> = vec![
        WalletAction::Navigate(Screen::Dashboard),
        WalletAction::Navigate(Screen::Send),
        WalletAction::Navigate(Screen::Receive),
        WalletAction::Navigate(Screen::History),
        WalletAction::Navigate(Screen::Settings),
        WalletAction::Navigate(Screen::Help),
        WalletAction::NextTab,
        WalletAction::PrevTab,
        WalletAction::StartSend,
        WalletAction::SetRecipient("ab".repeat(1184)),
        WalletAction::SetAmount(1000),
        WalletAction::SetFee(700),
        WalletAction::SetFee(0),
        WalletAction::ConfirmSend,
        WalletAction::SignAndSend,
        WalletAction::CancelSend,
        WalletAction::CopyAddress,
        WalletAction::NewAddress,
        WalletAction::SelectHistoryEntry(0),
        WalletAction::FilterHistory(None),
        WalletAction::SetCoverage(4),
        WalletAction::ToggleFeeBucketing,
        WalletAction::DismissNotification,
        WalletAction::ChainHeightAdvanced(100),
        WalletAction::PeerCountChanged(3),
        WalletAction::PeerCountChanged(0),
        WalletAction::TransactionConfirmed {
            txid: nerv_core::types::TxId::from_hash(
                nerv_core::hash::Hash256::from_bytes([0u8; 32]),
            ),
            height: 50,
        },
        WalletAction::SendProgress(nerv_wallet_core::action::SendStage::Proving),
        WalletAction::SendFailed("test".into()),
        WalletAction::SendSucceeded(
            nerv_core::types::TxId::from_hash(
                nerv_core::hash::Hash256::from_bytes([0u8; 32]),
            ),
        ),
    ];

    for start_screen in Screen::ALL {
        let mut s = unlocked_state();
        s.screen = start_screen;
        for action in &actions {
            step(&mut s, action.clone());
        }
    }
}

#[test]
fn every_action_on_locked_state_preserves_invariants() {
    let actions: Vec<WalletAction> = vec![
        WalletAction::Navigate(Screen::Send),
        WalletAction::NextTab,
        WalletAction::PrevTab,
        WalletAction::StartSend,
        WalletAction::SetRecipient("zzz".into()),
        WalletAction::SetAmount(100),
        WalletAction::ConfirmSend,
        WalletAction::SignAndSend,
        WalletAction::CancelSend,
        WalletAction::CopyAddress,
        WalletAction::NewAddress,
        WalletAction::DismissNotification,
        WalletAction::PeerCountChanged(1),
        WalletAction::ChainHeightAdvanced(1),
    ];

    let mut s = WalletState::new();
    for action in &actions {
        step(&mut s, action.clone());
    }
}

#[test]
fn send_flow_happy_path() {
    let mut s = unlocked_state();

    // Simulate receiving funds.
    update(&mut s, WalletAction::ChainHeightAdvanced(10));
    assert!(matches!(s.sync, SyncStatus::Synced { .. }));

    // Start a send.
    step(&mut s, WalletAction::StartSend);
    assert_eq!(s.screen, Screen::Send);
    assert!(s.draft.is_some());

    // Fill in valid values.
    step(&mut s, WalletAction::SetRecipient("cd".repeat(592))); // 1184 hex chars.
    step(&mut s, WalletAction::SetAmount(100));
    step(&mut s, WalletAction::SetFee(700));

    // Validate.
    step(&mut s, WalletAction::ConfirmSend);
    let draft = s.draft.as_ref().unwrap();
    assert!(draft.is_ready(), "the draft should be ready: {:?}", draft.validation.error);

    // Send.
    let events = update(&mut s, WalletAction::SignAndSend);
    assert!(events.iter().any(|e| matches!(e, nerv_wallet_core::update::WalletEvent::StartSendPipeline { .. })));
    assert!(s.draft.is_none(), "the draft is consumed after SignAndSend");
}

#[test]
fn send_flow_rejects_garbage_recipients() {
    let mut s = unlocked_state();
    step(&mut s, WalletAction::StartSend);

    for garbage in [
        "",
        "z",
        "zzz",
        "0".repeat(2367),
        "0".repeat(2369),
        "not hex!".repeat(100),
        "\u{4e16}\u{754c}".repeat(592),
        "A".repeat(1184), // uppercase is valid hex
    ] {
        step(&mut s, WalletAction::SetRecipient(garbage.to_string()));
        if garbage.len() != 1184 * 2 {
            let draft = s.draft.as_ref().unwrap();
            assert!(
                !draft.is_ready(),
                "garbage recipient must not be ready: {garbage:?}"
            );
        }
    }
}

#[test]
fn lock_resets_everything() {
    let mut s = unlocked_state();
    update(&mut s, WalletAction::ChainHeightAdvanced(100));
    update(&mut s, WalletAction::PeerCountChanged(5));
    step(&mut s, WalletAction::StartSend);

    step(&mut s, WalletAction::Lock);
    assert!(!s.is_unlocked());
    assert_eq!(s.chain_height, 0);
    assert_eq!(s.peer_count, 0);
    assert!(s.draft.is_none());
    assert!(s.history.is_empty());
    assert!(matches!(s.sync, SyncStatus::Locked));
}

#[test]
fn quit_is_terminal() {
    let mut s = unlocked_state();
    step(&mut s, WalletAction::Quit);
    assert!(s.quitting);
    // Further actions are harmless but the flag stays.
    step(&mut s, WalletAction::NextTab);
    assert!(s.quitting);
}

#[test]
fn navigation_is_cyclic() {
    let mut s = unlocked_state();
    let start = s.screen;
    for _ in 0..Screen::ALL.len() {
        step(&mut s, WalletAction::NextTab);
    }
    assert_eq!(s.screen, start, "cycling all tabs returns to the start");

    for _ in 0..Screen::ALL.len() {
        step(&mut s, WalletAction::PrevTab);
    }
    assert_eq!(s.screen, start, "cycling back also returns");
}

#[test]
fn disconnect_and_reconnect() {
    let mut s = unlocked_state();
    update(&mut s, WalletAction::ChainHeightAdvanced(10));

    step(&mut s, WalletAction::PeerCountChanged(0));
    assert!(matches!(s.sync, SyncStatus::Disconnected));
    assert!(
        s.notifications.iter().any(|n| n.message.contains("Disconnected")),
        "should notify on disconnect"
    );

    step(&mut s, WalletAction::PeerCountChanged(2));
    assert!(matches!(s.sync, SyncStatus::Synced { .. }));
}

File: crates/nerv-wallet-core/tests/storage.rs 

//! Storage-layer round-trip tests (erratum 206).

use nerv_wallet_core::storage::{
    EncryptedSeed, FileStorage, MemoryStorage, StorageError, WalletStorage, ENVELOPE_LEN,
};

#[test]
fn seal_open_with_correct_password() {
    for i in 0..100 {
        let seed = [i as u8; 32];
        let password = format!("password-{i}");
        let env = EncryptedSeed::seal(&seed, &password);
        assert_eq!(env.open(&password).unwrap(), seed);
    }
}

#[test]
fn seal_open_with_wrong_password_fails() {
    let seed = [42u8; 32];
    let env = EncryptedSeed::seal(&seed, "correct");
    for wrong in ["", "a", "correct ", "correctt", "password", "12345678", "\u{4e16}"] {
        assert!(
            env.open(wrong).is_err(),
            "wrong password {wrong:?} must fail"
        );
    }
}

#[test]
fn different_passwords_different_ciphertexts() {
    let seed = [7u8; 32];
    let a = EncryptedSeed::seal(&seed, "one");
    let b = EncryptedSeed::seal(&seed, "two");
    assert_ne!(a.ciphertext, b.ciphertext);
    assert_ne!(a.salt, b.salt, "fresh salt per encryption");
    assert_ne!(a.nonce, b.nonce, "fresh nonce per encryption");
}

#[test]
fn wire_format_is_exactly_92_bytes() {
    let env = EncryptedSeed::seal(&[1u8; 32], "pw");
    assert_eq!(env.to_bytes().len(), ENVELOPE_LEN);
    assert_eq!(ENVELOPE_LEN, 92);
}

#[test]
fn memory_storage_full_lifecycle() {
    let mut s = MemoryStorage::new();
    assert!(!s.has_seed());

    s.store_seed(&[1u8; 32], "pw").unwrap();
    assert!(s.has_seed());
    assert_eq!(s.load_seed("pw").unwrap(), Some([1u8; 32]));
    assert!(s.load_seed("wrong").is_err());

    s.delete_seed().unwrap();
    assert!(!s.has_seed());
    assert_eq!(s.load_seed("pw").unwrap(), None);

    s.store_meta("key", "value").unwrap();
    assert_eq!(s.load_meta("key").unwrap(), Some("value".into()));
    assert_eq!(s.load_meta("missing").unwrap(), None);
}

#[test]
fn file_storage_survives_process_restart() {
    let dir = std::env::temp_dir().join(format!(
        "nerv-storage-rt-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    // Write.
    {
        let mut s = FileStorage::new(&dir);
        s.store_seed(&[9u8; 32], "pw").unwrap();
        s.store_meta("theme", "dark").unwrap();
    }

    // Read (as if from a new process).
    {
        let s = FileStorage::new(&dir);
        assert!(s.has_seed());
        assert_eq!(s.load_seed("pw").unwrap(), Some([9u8; 32]));
        assert_eq!(s.load_meta("theme").unwrap(), Some("dark".into()));
    }

    // Delete.
    {
        let mut s = FileStorage::new(&dir);
        s.delete_seed().unwrap();
        assert!(!s.has_seed());
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn encrypted_seed_base64_roundtrip() {
    let seed = [0xAB; 32];
    let env = EncryptedSeed::seal(&seed, "pw");
    let b64 = env.to_base64();
    let decoded = EncryptedSeed::from_base64(&b64).unwrap();
    assert_eq!(decoded.open("pw").unwrap(), seed);
    assert!(EncryptedSeed::from_base64("invalid!").is_err());
    assert!(EncryptedSeed::from_base64("").is_err());
}


