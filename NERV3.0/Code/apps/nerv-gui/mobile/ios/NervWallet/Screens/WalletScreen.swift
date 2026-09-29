// WalletScreen.swift (erratum 205; iOS counterpart to
// `apps/nerv-gui/mobile/android/.../ui/screens/WalletScreen.kt`).
//
// The main screen of the NERV mobile app. Hosts the WKWebView that
// runs the WASM-built eframe UI (all 8 desktop screens render
// inside it: Dashboard, Send, Receive, Claim, Producer, History,
// Settings, Help) and overlays a native SwiftUI chrome on top:
//
//   ┌──────────────────────────────────────┐
//   │  ⊛ NERV                ● SYNCED   🔒 │  ← native top bar
//   ├──────────────────────────────────────┤
//   │                                      │
//   │           WKWebView (eframe)         │  ← 8 screens render here
//   │                                      │
//   │                                      │
//   ├──────────────────────────────────────┤
//   │  ◧   ↗   ↙   👛   🧠   ⏱   ⚙   ?    │  ← native bottom nav
//   └──────────────────────────────────────┘
//
// The bridge lets the WebView call native iOS APIs (haptics,
// clipboard, share sheet) and lets the bottom nav drive wallet-core
// `Navigate(screen)` actions via a small JS shim in the eframe UI.

import SwiftUI
import WebKit

struct WalletScreen: View {

    let onLock: () -> Void

    @StateObject private var bridge = WebViewBridge()
    @State private var activeScreen: WalletScreen = {
        let saved = UserDefaults.standard.integer(forKey: "nerv.lastScreenIndex")
        return WalletScreen(rawValue: saved) ?? .dashboard
    }()
    @State private var syncState: SyncState = .synced
    @State private var webReady = false
    @State private var snackbarItem: SnackbarItem?
    @State private var snackbarDismissTask: Task<Void, Never>?

    var body: some View {
        ZStack(alignment: .top) {
            NervTheme.bg.ignoresSafeArea()

            VStack(spacing: 0) {
                NervTopBar(syncState: syncState, onLock: handleLockTap)
                WebViewHost(bridge: bridge, onReady: { webReady = true })
                    .opacity(webReady ? 1 : 0.001) // keep it in the hierarchy
                NervBottomNav(active: activeScreen, onSelect: handleTabTap)
            }

            // First-load overlay: shimmer + brand wordmark. Fades out
            // once the eframe UI fires its "ready" event.
            if !webReady {
                FirstLoadOverlay()
                    .transition(.opacity)
            }

            // Snackbar overlay.
            SnackbarHost(item: $snackbarItem)
        }
        .preferredColorScheme(.dark)
        .ignoresSafeArea(edges: .bottom)
        .onAppear {
            // Forward bridge events into SwiftUI state.
            // `bridge.lastEvent` is `@Published`, so binding a
            // `.onChange` here would also work; we use a small
            // task loop instead so we don't drop the very first
            // event before the view is in the hierarchy.
            Task { @MainActor in
                while !Task.isCancelled {
                    if let event = bridge.lastEvent {
                        handleBridgeEvent(event)
                        // Clear so we don't reprocess.
                        bridge.lastEvent = nil
                    }
                    try? await Task.sleep(nanoseconds: 100_000_000)
                }
            }
        }
        .onChange(of: activeScreen) { _, newValue in
            UserDefaults.standard.set(newValue.rawValue, forKey: "nerv.lastScreenIndex")
        }
    }

    // MARK: - Handlers

    private func handleTabTap(_ screen: WalletScreen) {
        activeScreen = screen
        bridge.postNativeEvent(type: "nav", payload: ["to": screen.jsName])
        Haptics.tap()
    }

    private func handleLockTap() {
        activeScreen = .dashboard
        bridge.postNativeEvent(type: "nav", payload: ["to": "Dashboard"])
        onLock()
    }

    private func handleBridgeEvent(_ event: BridgeEventPayload) {
        switch event.type {
        case "ready":
            webReady = true
            // Surface the active screen so the WebView's eframe UI
            // starts on the same tab the bottom-nav was on before
            // the cold-start.
            bridge.postNativeEvent(type: "nav", payload: ["to": activeScreen.jsName])
        case "nav":
            if let target = WalletScreen.allCases.first(where: {
                $0.jsName.caseInsensitiveCompare(event.payload) == .orderedSame
            }) {
                activeScreen = target
            }
        case "sync":
            if let s = SyncState(rawValue: event.payload.lowercased()) {
                syncState = s
            }
        case "toast":
            let level = inferLevel(from: event.payload)
            showSnackbar(message: event.payload, level: level)
        case "lock":
            onLock()
        case "haptic":
            switch event.payload {
            case "success": Haptics.success()
            case "warning": Haptics.warning()
            case "error":   Haptics.error()
            default:        Haptics.tap()
            }
        default:
            break
        }
    }

    /// The eframe UI prefixes its notification messages with a
    /// marker (`✓` for success, `⚠` for warning, `✗` for error);
    /// we infer the level from that prefix so the snackbar colours
    /// match the desktop GUI's notification levels.
    private func inferLevel(from message: String) -> SnackbarLevel {
        if message.hasPrefix("✓") { return .success }
        if message.hasPrefix("⚠") { return .warning }
        if message.hasPrefix("✗") { return .error }
        return .info
    }

    private func showSnackbar(message: String, level: SnackbarLevel) {
        snackbarDismissTask?.cancel()
        snackbarItem = SnackbarItem(message: message, level: level)
        // Auto-dismiss after 4 s.
        snackbarDismissTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 4_000_000_000)
            if !Task.isCancelled {
                snackbarItem = nil
            }
        }
    }
}

// MARK: - WKWebView host

private struct WebViewHost: UIViewRepresentable {
    let bridge: WebViewBridge
    let onReady: () -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(bridge: bridge)
    }

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()

        // Bridge handler.
        let userContent = WKUserContentController()
        userContent.add(context.coordinator, name: "nervBridge")
        config.userContentController = userContent

        // The eframe UI is WebGL; these settings keep the canvas
        // crisp on Retina and let JS use the IndexedDB-backed
        // session.
        config.preferences.setValue(true, forKey: "allowFileAccessFromFileURLs")
        let webpagePrefs = WKWebpagePreferences()
        webpagePrefs.allowsContentJavaScript = true
        config.defaultWebpagePreferences = webpagePrefs

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.isOpaque = false
        webView.backgroundColor = UIColor(
            red: 0x0D/255, green: 0x11/255, blue: 0x17/255, alpha: 1
        )
        webView.scrollView.backgroundColor = UIColor(
            red: 0x0D/255, green: 0x11/255, blue: 0x17/255, alpha: 1
        )
        webView.scrollView.bounces = false
        webView.scrollView.showsVerticalScrollIndicator = false
        webView.allowsBackForwardNavigationGestures = false
        webView.navigationDelegate = context.coordinator

        bridge.attach(to: webView)

        // Load the bundled WASM app from the app bundle.
        if let url = Bundle.main.url(forResource: "index", withExtension: "html") {
            webView.loadFileURL(
                url,
                allowingReadAccessTo: url.deletingLastPathComponent()
            )
        }

        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        // No-op: state changes go through `bridge.postNativeEvent`.
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        let bridge: WebViewBridge

        init(bridge: WebViewBridge) {
            self.bridge = bridge
        }

        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction,
            decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
        ) {
            // External URLs open in the system browser instead of
            // replacing the wallet UI.
            if let url = navigationAction.request.url,
               let scheme = url.scheme,
               scheme == "https" || scheme == "http" {
                #if canImport(UIKit)
                UIApplication.shared.open(url)
                #endif
                decisionHandler(.cancel)
                return
            }
            decisionHandler(.allow)
        }
    }
}

// MARK: - First-load overlay

private struct FirstLoadOverlay: View {
    @State private var pulse: CGFloat = 0.95

    var body: some View {
        ZStack {
            NervTheme.bg.ignoresSafeArea()
            VStack(spacing: NervSpacing.m) {
                Image("nerv_logo")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 120, height: 120)
                    .scaleEffect(pulse)
                Text("Loading NERV…")
                    .font(NervTypography.labelMedium())
                    .foregroundStyle(NervTheme.textMuted)
            }
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 1.2).repeatForever(autoreverses: true)) {
                pulse = 1.05
            }
        }
    }
}