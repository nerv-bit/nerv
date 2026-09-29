# NERV Wallet — iOS Shell

Native iOS shell for the NERV wallet. The shell hosts the desktop
eframe UI (compiled to WASM) inside a `WKWebView` and adds a
polished native SwiftUI chrome on top:

- **Splash screen** — NERV logo + tagline + brand wordmark with a
  breathing-pulse animation.
- **Biometric lock** — `LocalAuthentication.LAContext` (Face ID /
  Touch ID / device passcode fallback). The wallet's seed is wiped
  on Lock.
- **Native top bar** — small NERV logo, sync status pill
  (Synced/Syncing/Offline/Locked), lock button.
- **Native bottom nav** — 8 tabs mirroring the desktop GUI's
  `Screen` enum (Dashboard, Send, Receive, Claim, Producer, History,
  Settings, Help) with SF Symbols.
- **Snackbar host** — coloured by notification level (info /
  success / warning / error), auto-dismisses after 4 s.
- **JS bridge** — Swift ⇄ JavaScript for chrome-level concerns:
  haptics, clipboard, share sheet, biometric re-prompt, navigation
  events, sync-state updates, toasts.

## Why a SwiftUI shell + WKWebView (not pure native Swift)?

The NERV wallet's logic — the wallet-core state machine, the eframe
GUI, the post-quantum cryptography — lives in Rust. The same
`apps/nerv-gui` crate compiles to native (desktop), WASM (web / PWA
/ Android / iOS), and would compile to a Rust-mobile binary if we
later want one. The iOS shell keeps the Rust ↔ Swift boundary
clean: Rust owns wallet state; Swift owns chrome.

This mirrors the Android shell's architecture — the two shells
share the same Rust ↔ JS contract surface (`NervBridge` on Android,
`nervBridge` on iOS) and the same WASM bundle.

## Build

```bash
cd apps/nerv-gui/mobile/ios
./build_ios.sh
```

Prereqs (one-time, on macOS):

```bash
rustup default 1.85.0
cargo install wasm-pack
brew install xcodegen
# Xcode 15 + iOS SDK 17
```

The output `.app` lands at:

```
build/Build/Products/Release-iphoneos/Nerv.app
```

Install on a connected device or simulator:

```bash
xcrun simctl install booted build/Build/Products/Release-iphoneos/Nerv.app
xcrun simctl launch booted org.nerv.wallet
```

## Project layout

```
ios/
├── App.swift                       # entry point; phase machine (Splash → Lock → Wallet)
├── Info.plist                      # permissions (Face ID, camera), privacy manifest
├── project.yml                     # XcodeGen spec
├── build_ios.sh                    # one-shot WASM + .app build
├── README.md                       # this file
├── NervWallet/
│   ├── Bridge/
│   │   ├── BiometricAuth.swift     # LocalAuthentication wrapper
│   │   └── WebViewBridge.swift     # Swift ⇄ JS bridge + haptics
│   ├── Components/
│   │   ├── NervBottomNav.swift     # 8-tab native bottom nav
│   │   ├── NervTopBar.swift        # brand + sync pill + lock
│   │   └── SnackbarController.swift# auto-dismissing toast host
│   ├── Models/
│   │   └── WalletModels.swift      # WalletScreen / SyncState / SnackbarLevel + Color extensions
│   ├── Screens/
│   │   ├── LockScreen.swift        # biometric gate
│   │   ├── SplashScreen.swift      # NERV logo + pulse
│   │   └── WalletScreen.swift      # WKWebView host + chrome + bridge
│   ├── Theme/
│   │   ├── NervAssetCatalog.swift  # asset-catalog wrapper
│   │   └── NervTheme.swift         # NERV palette tokens + typography
│   └── Resources/
│       ├── Assets.xcassets/        # AppIcon, AccentColor, nerv_logo
│       └── www/                    # WASM bundle + index.html (built by build_ios.sh)
└── Resources/                      # raw PNGs + a duplicate copy for the asset catalog
```

## JS bridge contract

The eframe UI inside the WKWebView calls native APIs via
`window.webkit.messageHandlers.nervBridge.postMessage({type, payload})`:

| JS postMessage | Native behaviour |
| -------------- | ---------------- |
| `{type:'ready', payload:''}` | Marks the WebView as ready; first-load shimmer fades |
| `{type:'nav', payload:'Send'}` | Updates native bottom-nav active state |
| `{type:'sync', payload:'syncing'}` | Updates the top-bar sync pill |
| `{type:'toast', payload:msg}` | Shows a snackbar (level inferred from message prefix: `✓`/`⚠`/`✗`) |
| `{type:'lock', payload:''}` | Locks the wallet (returns to LockScreen) |
| `{type:'haptic', payload:'success'\|'warning'\|'error'\|'tap'}` | Fires the matching UIKit haptic |

Native → JS:

| Swift call | JS handler |
| ---------- | ---------- |
| `bridge.postNativeEvent(type: 'nav', payload: ['to':'Send'])` | `nervBridgeFromNative('nav', json)` → eframe dispatches `WalletAction::Navigate(Screen::Send)` |

The eframe UI binds `nervBridgeFromNative` in a small JS shim —
see `apps/nerv-gui/src/lib.rs` for the counterpart. Without the
shim, the native chrome (top bar, bottom nav, snackbars) still
works; tapping a bottom-nav tab updates the chrome but doesn't
move the WebView's screen until the JS shim is in place.

## Security notes

- The wallet's seed is held inside the WKWebView's IndexedDB. It is
  wiped on `WalletAction::Lock` (the eframe UI handles this).
- The native shell never sees the seed or any cryptographic
  material — only the chrome-level concerns above.
- The Info.plist explicitly opts out of iCloud Keychain, app
  backup, and cross-device tracking (`NSPrivacyTracking=false`).
- Biometric prompts are *gates*, not *storage*: even if the
  biometric check passes, the wallet still must be unlocked in the
  eframe UI (via `WalletAction::Unlock` / seed import).
- Code-signing is **disabled for local builds** (`CODE_SIGNING_ALLOWED=NO`).
  Before publishing to TestFlight / App Store, set
  `DEVELOPMENT_TEAM` in `project.yml` and re-enable signing.
- `LSRequiresIPhoneOS=true` + portrait-only orientation keeps the
  UI focused on the wallet experience.

## Parity with the Android shell

| Surface | Android (Kotlin/Compose) | iOS (SwiftUI) |
| ------- | ------------------------ | ------------- |
| Splash | `SplashScreen.kt` | `SplashScreen.swift` |
| Lock | `LockScreen.kt` + `BiometricAuth.kt` | `LockScreen.swift` + `BiometricAuth.swift` |
| Wallet host | `WalletScreen.kt` (WebView) | `WalletScreen.swift` (WKWebView) |
| Top bar | `NervTopBar.kt` | `NervTopBar.swift` |
| Bottom nav | `NervBottomNav.kt` | `NervBottomNav.swift` |
| Snackbar | Material 3 `SnackbarHost` | `SnackbarController.swift` (custom) |
| Bridge | `WebViewBridge.kt` (Kotlin ⇄ JS) | `WebViewBridge.swift` (Swift ⇄ JS) |
| NERV logo | `ic_nerv_logo.png` | `nerv_logo.png` (in asset catalog) |
| Haptics | `Vibrator` via `VibrationEffect` | UIKit `UIImpactFeedbackGenerator` / `UINotificationFeedbackGenerator` |
| Biometric API | `androidx.biometric.BiometricPrompt` | `LocalAuthentication.LAContext` |
| Encrypted prefs | `EncryptedSharedPreferences` | `UserDefaults` (chrome-level prefs only; no crypto material) |

## Known limitations

- The WKWebView-based host adds a small rendering overhead vs. a
  pure-native SwiftUI UI. Acceptable for a wallet where the
  dominant action is "look at the balance / send once every few
  days".
- VoiceOver / TalkBack labels are wired up for the chrome (the SF
  Symbols have native a11y; custom controls carry
  `accessibilityLabel`) but the eframe UI inside the WKWebView is
  not exposed to VoiceOver — a future enhancement.
- The release `.app` is unsigned; see "Security notes" above.