// App.swift (erratum 205; iOS shell entry point — iOS counterpart
// to `apps/nerv-gui/mobile/android/.../MainActivity.kt`).
//
// Wires the three top-level Compose screens into a state machine:
//
//   Splash ──(min duration + WASM ready)──▶ LockScreen
//   LockScreen ──(biometric success)──▶ WalletScreen
//   WalletScreen ──(lock button)──▶ LockScreen
//
// All wallet logic lives in Rust (compiled to WASM and rendered by
// eframe inside the WKWebView in `WalletScreen`). The SwiftUI shell
// here is purely a native chrome (splash, biometric lock, top/bottom
// navigation, snackbars) plus the Swift ⇄ JS bridge.

import SwiftUI

@main
struct NervApp: App {

    /// One source of truth for which phase of the app the user is in.
    /// We advance forward when the WebView fires `ready`; we always
    /// advance to `Lock` first (biometric gate) before the wallet UI.
    @State private var phase: AppPhase = .splash

    var body: some Scene {
        WindowGroup {
            ZStack {
                NervTheme.bg.ignoresSafeArea()
                switch phase {
                case .splash:
                    SplashScreen(onReady: { phase = .lock })
                        .transition(.opacity)
                case .lock:
                    LockScreen(onUnlocked: { phase = .wallet })
                        .transition(.opacity)
                case .wallet:
                    WalletScreen(onLock: { phase = .lock })
                        .transition(.opacity)
                }
            }
            .animation(.easeInOut(duration: 0.35), value: phase)
            .preferredColorScheme(.dark)
            // Force the status bar to stay light against the dark
            // chrome regardless of the SwiftUI sub-screen.
            .statusBarHidden(false)
        }
    }
}

enum AppPhase: Equatable {
    case splash
    case lock
    case wallet
}