// WebViewBridge (erratum 205).
//
// Kotlin <-> JavaScript bridge. The eframe UI inside the WebView can
// call out to native Android APIs (haptics, clipboard, share sheet,
// biometric re-prompt) via `window.NervBridge.*`; Kotlin calls into
// the WebView via `evaluateJavascript("nervBridgeFromNative(...)")`.
//
// The contract surface is intentionally small — wallet actions stay
// inside the wallet-core state machine (Rust). The bridge is for
// *chrome*-level concerns only.

package org.nerv.wallet.ui.bridge

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.webkit.JavascriptInterface
import android.webkit.WebView
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.runtime.staticCompositionLocalOf

/** The callback a screen registers to receive native-side events. */
data class BridgeEvent(
    val type: String,
    val payload: String,
)

val LocalBridge = compositionLocalOf<WebViewBridge> {
    error("WebViewBridge not provided")
}

/**
 * The JavaScript interface injected into the WebView. Exposed at
 * `window.NervBridge.*` from the JS side.
 *
 * Every method is documented with its corresponding
 * `nervBridgeFromNative(...)` JS-callable name so the two sides stay
 * in lock-step.
 */
class WebViewBridge(
    private val context: Context,
    private val onEvent: (BridgeEvent) -> Unit,
) {

    @JavascriptInterface
    fun postEvent(type: String, payload: String) {
        onEvent(BridgeEvent(type, payload))
    }

    @JavascriptInterface
    fun vibrate(durationMs: Long) {
        val v = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            (context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE)
                as VibratorManager).defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }
        v.vibrate(VibrationEffect.createOneShot(durationMs, 80))
    }

    @JavascriptInterface
    fun copyToClipboard(text: String) {
        val cm = context.getSystemService(Context.CLIPBOARD_SERVICE)
            as ClipboardManager
        cm.setPrimaryClip(ClipData.newPlainText("NERV", text))
    }

    @JavascriptInterface
    fun shareText(text: String, title: String) {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_SUBJECT, title)
            putExtra(Intent.EXTRA_TEXT, text)
        }
        val chooser = Intent.createChooser(intent, title).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        context.startActivity(chooser)
    }
}

/**
 * Dispatch a JSON event into the WebView's JS context. The payload
 * is JSON-encoded so the JS side can parse it with
 * `JSON.parse(...)`.
 */
fun WebView.postNativeEvent(type: String, payload: String) {
    val escapedType = type.replace("'", "\\'")
    val escapedPayload = payload.replace("'", "\\'")
    evaluateJavascript(
        "nervBridgeFromNative('$escapedType', '$escapedPayload');",
        null,
    )
}

/**
 * Convenience wrapper that JSON-encodes a [Map] payload before
 * dispatching to the WebView.
 */
fun WebView.postNativeEvent(type: String, payload: Map<String, Any?>) {
    val json = payload.entries.joinToString(
        prefix = "{",
        postfix = "}",
        separator = ",",
    ) { (k, v) ->
        val jsVal: String = when (v) {
            null -> "null"
            is Boolean -> v.toString()
            is Number -> v.toString()
            is String -> "'${v.replace("'", "\\'")}'"
            else -> "'${v.toString().replace("'", "\\'")}'"
        }
        "\"$k\":$jsVal"
    }
    val escapedType = type.replace("'", "\\'")
    evaluateJavascript(
        "nervBridgeFromNative('$escapedType', '$json');",
        null,
    )
}