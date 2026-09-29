// WalletModels.swift (erratum 205; iOS shell).
//
// Shared data types used across the NERV iOS shell. Mirrors the
// Kotlin counterparts in `apps/nerv-gui/mobile/android/app/src/main/
// java/org/nerv/wallet/`.

import Foundation

/// The wallet's 8 top-level screens. Mirrors the desktop GUI's
/// `Screen` enum so the bottom nav can drive
/// `WalletAction::Navigate(screen)` via the JS bridge.
public enum WalletScreen: Int, CaseIterable, Identifiable {
    case dashboard = 0
    case send      = 1
    case receive   = 2
    case claim     = 3
    case producer  = 4
    case history   = 5
    case settings  = 6
    case help      = 7

    public var id: Int { rawValue }

    /// The matching `Screen` name in the eframe UI's JS bridge
    /// (must match the desktop GUI's `Screen::title()` strings).
    public var jsName: String {
        switch self {
        case .dashboard: return "Dashboard"
        case .send:      return "Send"
        case .receive:   return "Receive"
        case .claim:     return "Claim"
        case .producer:  return "Producer"
        case .history:   return "History"
        case .settings:  return "Settings"
        case .help:      return "Help"
        }
    }

    /// Human-readable title for the bottom-nav label.
    public var title: String {
        switch self {
        case .dashboard: return "Dashboard"
        case .send:      return "Send"
        case .receive:   return "Receive"
        case .claim:     return "Claim"
        case .producer:  return "Producer"
        case .history:   return "History"
        case .settings:  return "Settings"
        case .help:      return "Help"
        }
    }
}

/// The chain-sync state surfaced in the top bar.
public enum SyncState: String {
    case synced
    case syncing
    case offline
    case locked

    /// Token for the badge colour.
    public var color: Color {
        switch self {
        case .synced:  return .nervSuccess
        case .syncing: return .nervWarning
        case .offline: return .nervError
        case .locked:  return .nervTextMuted
        }
    }

    public var label: String {
        switch self {
        case .synced:  return "Synced"
        case .syncing: return "Syncing"
        case .offline: return "Offline"
        case .locked:  return "Locked"
        }
    }
}

/// One bridge event coming back from the WebView. The eframe UI
/// inside the WKWebView calls `window.webkit.messageHandlers.
/// nervBridge.postMessage({type, payload})` (mirroring the Kotlin
/// `@JavascriptInterface` methods).
public struct BridgeEventPayload: Equatable {
    public let type: String
    public let payload: String

    public init(type: String, payload: String) {
        self.type = type
        self.payload = payload
    }
}

/// Snackbar level (mirrors `SnackbarLevel.kt` on Android).
public enum SnackbarLevel {
    case info
    case success
    case warning
    case error
}

// MARK: - Color helpers

extension Color {
    static var nervBg:           Color { NervTheme.bg }
    static var nervBgElev:       Color { NervTheme.bgElev }
    static var nervSurface:      Color { NervTheme.surface }
    static var nervSurfaceVar:   Color { NervTheme.surfaceVar }
    static var nervText:         Color { NervTheme.text }
    static var nervTextMuted:    Color { NervTheme.textMuted }
    static var nervTextSubtle:   Color { NervTheme.textSubtle }
    static var nervAccent:       Color { NervTheme.accent }
    static var nervAccentHi:     Color { NervTheme.accentHi }
    static var nervAccentLo:     Color { NervTheme.accentLo }
    static var nervGold:         Color { NervTheme.gold }
    static var nervGoldHi:       Color { NervTheme.goldHi }
    static var nervGoldLo:       Color { NervTheme.goldLo }
    static var nervSuccess:      Color { NervTheme.success }
    static var nervWarning:      Color { NervTheme.warning }
    static var nervError:        Color { NervTheme.error }
    static var nervBorder:       Color { NervTheme.border }
    static var nervBorderSubtle: Color { NervTheme.borderSubtle }
    static var nervSplashBg:     Color { NervTheme.splashBg }
}