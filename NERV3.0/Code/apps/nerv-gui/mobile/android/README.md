# NERV Wallet — Android Shell

Native Android shell for the NERV wallet. The shell hosts the
desktop eframe UI (compiled to WASM) inside an Android WebView and
adds a polished native chrome on top:

- **Splash screen** — NERV logo + tagline + brand wordmark with a
  breathing-pulse animation.
- **Biometric lock** — `androidx.biometric.BiometricPrompt` (strong +
  device credential fallback). The wallet's seed is wiped on Lock.
- **Native top bar** — small NERV logo, sync status pill
  (Synced/Syncing/Offline/Locked), lock button.
- **Native bottom nav** — 8 tabs mirroring the desktop GUI's
  `Screen` enum (Dashboard, Send, Receive, Claim, Producer, History,
  Settings, Help).
- **Snackbar host** — coloured by notification level (info /
  success / warning / error).
- **JS bridge** — Kotlin ⇄ JS for chrome-level concerns: clipboard,
  share sheet, haptics, biometric re-prompt, navigation events,
  sync-state updates, toasts.

## Why a Compose shell + WebView (not pure native Kotlin)?

The NERV wallet's logic — the wallet-core state machine, the eframe
GUI, the post-quantum cryptography — lives in Rust. The same
`apps/nerv-gui` crate compiles to native (desktop), WASM (web / PWA
/ Android / iOS), and would compile to a Rust-mobile binary if we
later want one. The Android shell keeps the Rust ↔ Kotlin boundary
clean: Rust owns wallet state; Kotlin owns chrome.

If we wanted a pure-native Kotlin UI in the future, the wallet-core
state machine is pure (no I/O), so a Kotlin port would be a
mechanical translation. The Compose shell today is the cheapest path
to a polished Android experience.

## Build

```bash
cd apps/nerv-gui/mobile/android
./build_android.sh
```

Prereqs (one-time):

```bash
rustup default 1.85.0
cargo install wasm-pack
# Android SDK + platform-tools + build-tools 34
# JDK 17
```

The output APK lands at:

```
app/build/outputs/apk/release/app-release.apk
```

Install on a connected device:

```bash
adb install -r app/build/outputs/apk/release/app-release.apk
adb shell am start -n org.nerv.wallet/.MainActivity
```

## Project layout

```
app/src/main/
├── AndroidManifest.xml          # manifest; declares the activity,
│                                # biometric + INTERNET permissions,
│                                # anti-backup rules (seed never
│                                # leaves device)
├── assets/www/                  # WASM bundle + index.html (built by
│                                # build_android.sh)
├── java/org/nerv/wallet/
│   ├── MainActivity.kt          # entry point; Splash → Lock → Wallet
│   ├── NervApplication.kt       # Application class; holds prefs handle
│   ├── data/
│   │   ├── BiometricAuth.kt     # BiometricPrompt wrapper
│   │   └── NervPreferences.kt   # EncryptedSharedPreferences
│   └── ui/
│       ├── bridge/
│       │   └── WebViewBridge.kt # Kotlin ⇄ JS bridge
│       ├── components/
│       │   ├── NervTopBar.kt
│       │   ├── NervBottomNav.kt
│       │   └── Snackbar.kt
│       ├── screens/
│       │   ├── SplashScreen.kt
│       │   ├── LockScreen.kt
│       │   └── WalletScreen.kt
│       └── theme/
│           ├── Color.kt         # NERV palette tokens
│           ├── Type.kt          # Material 3 type scale
│           └── Theme.kt         # NervTheme entry point
└── res/
    ├── drawable/
    │   ├── ic_nerv_logo.png         # supplied NERV logo
    │   └── ic_launcher_foreground.png
    ├── mipmap-anydpi-v26/
    │   ├── ic_launcher.xml          # adaptive icon
    │   └── ic_launcher_round.xml
    ├── values/
    │   ├── colors.xml               # XML palette tokens
    │   ├── strings.xml
    │   └── themes.xml
    ├── values-night/
    │   └── themes.xml
    └── xml/
        ├── backup_rules.xml
        └── data_extraction_rules.xml
```

## JS bridge contract

The eframe UI inside the WebView calls native APIs via
`window.NervBridge.*`:

| JS call                                    | Native behaviour                                 |
| ------------------------------------------ | ------------------------------------------------ |
| `NervBridge.postEvent('ready', '')`        | marks the WebView as ready; splash overlay fades |
| `NervBridge.postEvent('nav', 'Send')`      | updates native bottom-nav active state           |
| `NervBridge.postEvent('sync', 'synced')`   | updates the top-bar sync badge                   |
| `NervBridge.postEvent('toast', msg)`       | shows a snackbar                                 |
| `NervBridge.postEvent('lock', '')`         | locks the wallet (returns to LockScreen)         |
| `NervBridge.vibrate(20)`                   | 20 ms haptic tick                                 |
| `NervBridge.copyToClipboard(text)`         | copies text to system clipboard                  |
| `NervBridge.shareText(text, title)`        | opens the share sheet                             |

Native → JS:

| Native call                                | JS handler                                       |
| ------------------------------------------ | ------------------------------------------------ |
| `webView.postNativeEvent('nav', {to:...})` | `nervBridgeFromNative('nav', json)` → navigates  |
| `webView.postNativeEvent('sync', '...')`   | updates the eframe UI's sync badge               |

The eframe UI binds these in a small JS shim — see
`apps/nerv-gui/src/lib.rs` (the `bind_bridge` shim) for the
counterpart that exposes a `navigate(screen)` function so the native
bottom-nav can drive wallet actions.

## Security notes

- The wallet's seed is held inside the WebView's IndexedDB. It is
  wiped on `WalletAction::Lock` (the eframe UI handles this).
- The native shell never sees the seed or any cryptographic
  material — only the chrome-level concerns above.
- Backup rules explicitly exclude the wallet's data so the seed
  cannot leak to Google's auto-backup.
- The release keystore is provisioned separately; the default
  `signingConfig = signingConfigs.debug` line lets the APK build
  for local testing. Switch to a release keystore before
  publishing.
- Biometric prompts are *gates*, not *storage*: even if the
  biometric check passes, the wallet still must be unlocked in the
  eframe UI (via `WalletAction::Unlock` / seed import).

## Known limitations

- The WebView-based host adds a small rendering overhead vs. a
  pure-native Kotlin UI. Acceptable for a wallet where the dominant
  action is "look at the balance / send once every few days".
- Voice-Over / TalkBack labels are wired up for the chrome (see
  `strings.xml`'s `cd_*` entries) but the eframe UI inside the
  WebView is not exposed to TalkBack — a future enhancement.
- The release APK uses the debug keystore for local builds; see
  the "Security notes" above.