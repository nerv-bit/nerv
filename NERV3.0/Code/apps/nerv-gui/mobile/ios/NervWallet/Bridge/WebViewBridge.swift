// WebViewBridge.swift (erratum 205; iOS shell).
//
// Swift ⇄ JavaScript bridge. The eframe UI inside the WKWebView can
// call out to native iOS APIs (haptics, share sheet, copy to
// clipboard) via `window.webkit.messageHandlers.nervBridge.post
// Message({type, payload})`. SwiftUI pushes events into the WebView
// via `WebViewStore.evaluateJavaScript("nervBridgeFromNative(...)")`.
//
// The contract surface is intentionally small — wallet actions stay
// inside the wallet-core state machine (Rust). The bridge is for
// *chrome*-level concerns only.

import Foundation
import SwiftUI
import UIKit
import WebKit

/// A bridge message received from the WebView. Maps directly to
/// `BridgeEvent(type, payload)` on the Kotlin side.
public struct BridgeMessage: Codable {
    public let type: String
    public let payload: String?
}

/// The Swift bridge handler. Conforms to `WKScriptMessageHandler` so
/// the WKWebView can route `nervBridge` postMessage events here, and
/// exposes `@MainActor` methods for SwiftUI to call.
@MainActor
public final class WebViewBridge: NSObject, ObservableObject, WKScriptMessageHandler {

    /// The SwiftUI-observable sink for events coming from the
    /// WebView. The screen binds this and dispatches on `type`.
    @Published public var lastEvent: BridgeEventPayload?

    private weak var webView: WKWebView?

    public override init() {
        super.init()
    }

    public func attach(to webView: WKWebView) {
        self.webView = webView
    }

    // MARK: - WKScriptMessageHandler

    public func userContentController(
        _ userContentController: WKUserContentController,
        didReceive message: WKScriptMessage,
    ) {
        guard message.name == "nervBridge" else { return }
        guard let body = message.body as? [String: Any] else { return }
        let type = body["type"] as? String ?? ""
        let payload = body["payload"] as? String ?? ""
        Task { @MainActor in
            self.lastEvent = BridgeEventPayload(type: type, payload: payload)
        }
    }

    // MARK: - Native → JS

    /// Push a JSON-encoded event into the WebView. The eframe UI's
    /// JS shim (`nervBridgeFromNative(type, json)`) receives it and
    /// dispatches the corresponding wallet-core action.
    public func postNativeEvent(type: String, payload: String) {
        let safeType = type.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "'", with: "\\'")
        let safePayload = payload.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "'", with: "\\'")
        let js = "nervBridgeFromNative('\(safeType)', '\(safePayload)');"
        webView?.evaluateJavaScript(js, completionHandler: nil)
    }

    /// Convenience overload: encode a `[String: String]` payload as
    /// a tiny JSON object before dispatching.
    public func postNativeEvent(type: String, payload: [String: String]) {
        let parts = payload.map { key, value in
            let escaped = value.replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
            return "\"\(key)\":\"\(escaped)\""
        }
        let json = "{" + parts.joined(separator: ",") + "}"
        postNativeEvent(type: type, payload: json)
    }
}

/// Tiny UIKit haptics bridge. iOS provides three feedback generators
/// — `.light()` for taps, `.medium()` for confirmations, `.heavy()`
/// for errors — wrapped here so SwiftUI screens can fire them.
@MainActor
public enum Haptics {
    public static func tap() {
        let g = UIImpactFeedbackGenerator(style: .light)
        g.prepare()
        g.impactOccurred()
    }

    public static func success() {
        let g = UINotificationFeedbackGenerator()
        g.prepare()
        g.notificationOccurred(.success)
    }

    public static func warning() {
        let g = UINotificationFeedbackGenerator()
        g.prepare()
        g.notificationOccurred(.warning)
    }

    public static func error() {
        let g = UINotificationFeedbackGenerator()
        g.prepare()
        g.notificationOccurred(.error)
    }
}