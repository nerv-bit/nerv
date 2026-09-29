// WalletScreen (erratum 205).
//
// The main screen of the NERV mobile app. Hosts the WebView that
// runs the WASM-built eframe UI (all 8 desktop screens render
// inside it: Dashboard, Send, Receive, Claim, Producer, History,
// Settings, Help) and overlays a native chrome on top:
//
//   ┌──────────────────────────────────────┐
//   │  ⊛ NERV                🔒 ● SYNCED   │  ← native top bar (logo + status)
//   ├──────────────────────────────────────┤
//   │                                      │
//   │           WebView (eframe)           │  ← 8 screens render here
//   │                                      │
//   │                                      │
//   ├──────────────────────────────────────┤
//   │  ◧   →   ←   ◉   ⛏   ≡   ⚙   ?      │  ← native bottom nav (8 tabs)
//   └──────────────────────────────────────┘
//
// The bridge lets the WebView call native Android APIs (haptics,
// clipboard, share sheet) and lets the bottom nav drive wallet-core
// `Navigate(screen)` actions via a small JS shim in the eframe UI.

package org.nerv.wallet.ui.screens

import android.content.Intent
import android.net.Uri
import android.view.ViewGroup
import android.webkit.WebView
import android.webkit.WebSettings
import android.webkit.WebViewClient
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.nerv.wallet.R
import org.nerv.wallet.data.NervPreferences
import org.nerv.wallet.ui.bridge.BridgeEvent
import org.nerv.wallet.ui.bridge.WebViewBridge
import org.nerv.wallet.ui.bridge.postNativeEvent
import org.nerv.wallet.ui.components.NervBottomNav
import org.nerv.wallet.ui.components.NervTopBar
import org.nerv.wallet.ui.components.SnackbarLevel
import org.nerv.wallet.ui.theme.NervColors

/** The wallet's 8 top-level screens. Mirrors the desktop GUI's
 *  `Screen` enum so the bottom nav can drive
 *  `WalletAction::Navigate(screen)` via the JS bridge. */
enum class WalletScreen(
    val index: Int,
    val jsName: String,
    val titleRes: Int,
    val icon: androidx.compose.ui.graphics.vector.ImageVector,
) {
    Dashboard(0, "Dashboard",      R.string.nav_dashboard, Icons.Rounded.Dashboard),
    Send     (1, "Send",           R.string.nav_send,      Icons.Rounded.Send),
    Receive  (2, "Receive",        R.string.nav_receive,   Icons.Rounded.CallReceived),
    Claim    (3, "Claim",          R.string.nav_claim,     Icons.Rounded.AccountBalanceWallet),
    Producer (4, "Producer",       R.string.nav_producer,  Icons.Rounded.Memory),
    History  (5, "History",        R.string.nav_history,   Icons.Rounded.History),
    Settings (6, "Settings",       R.string.nav_settings,  Icons.Rounded.Settings),
    Help     (7, "Help",           R.string.nav_help,      Icons.Rounded.Help);

    companion object {
        fun fromIndex(i: Int): WalletScreen =
            entries[i.mod(entries.size)]
    }
}

enum class SyncState { Synced, Syncing, Offline, Locked }

@Composable
fun WalletScreen(
    onLock: () -> Unit,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val snackbarHostState = remember { SnackbarHostState() }

    var activeScreen by remember {
        mutableStateOf(WalletScreen.fromIndex(NervPreferences.lastScreenIndex))
    }
    var syncState by remember { mutableStateOf(SyncState.Synced) }
    var webReady by remember { mutableStateOf(false) }
    var webView by remember { mutableStateOf<WebView?>(null) }

    // Persistence: write the active screen so cold-starts restore it.
    LaunchedEffect(activeScreen) {
        NervPreferences.lastScreenIndex = activeScreen.index
    }

    // The bridge is created once per screen composition; it hands
    // events back into our Compose state.
    val bridge = remember(context) {
        WebViewBridge(context) { event ->
            when (event.type) {
                "ready" -> webReady = true
                "nav"   -> {
                    val target = WalletScreen.entries.firstOrNull {
                        it.jsName.equals(event.payload, ignoreCase = true)
                    }
                    if (target != null) activeScreen = target
                }
                "sync" -> {
                    syncState = when (event.payload.lowercase()) {
                        "synced"  -> SyncState.Synced
                        "syncing" -> SyncState.Syncing
                        "offline" -> SyncState.Offline
                        "locked"  -> SyncState.Locked
                        else      -> SyncState.Synced
                    }
                }
                "toast" -> scope.launch {
                    snackbarHostState.showSnackbar(
                        message = event.payload,
                        withDismissAction = true,
                    )
                }
                "lock" -> onLock()
                else -> { /* unknown — ignore */ }
            }
        }
    }

    Scaffold(
        topBar = {
            NervTopBar(
                syncState = syncState,
                onLock = {
                    webView?.postNativeEvent("nav", mapOf("to" to "Dashboard"))
                    activeScreen = WalletScreen.Dashboard
                    onLock()
                },
            )
        },
        bottomBar = {
            NervBottomNav(
                active = activeScreen,
                onSelect = { screen ->
                    activeScreen = screen
                    webView?.postNativeEvent(
                        "nav",
                        mapOf("to" to screen.jsName),
                    )
                },
            )
        },
        snackbarHost = {
            SnackbarHost(hostState = snackbarHostState) { data ->
                val level = when {
                    data.visuals.message.startsWith("✗") -> SnackbarLevel.Error
                    data.visuals.message.startsWith("⚠") -> SnackbarLevel.Warning
                    data.visuals.message.startsWith("✓") -> SnackbarLevel.Success
                    else -> SnackbarLevel.Info
                }
                androidx.compose.material3.Snackbar(
                    snackbarData = data,
                    containerColor = when (level) {
                        SnackbarLevel.Info    -> NervColors.BgElev
                        SnackbarLevel.Success -> NervColors.Success
                        SnackbarLevel.Warning -> NervColors.Warning
                        SnackbarLevel.Error   -> NervColors.Error
                    },
                    contentColor = when (level) {
                        SnackbarLevel.Info    -> NervColors.Text
                        SnackbarLevel.Success -> NervColors.Bg
                        SnackbarLevel.Warning -> NervColors.Bg
                        SnackbarLevel.Error   -> NervColors.Text
                    },
                )
            }
        },
        containerColor = NervColors.Bg,
    ) { padding ->
        Box(
            modifier = Modifier
                .padding(padding)
                .fillMaxSize()
                .background(NervColors.Bg),
        ) {
            // The WebView — hosts the eframe UI built from Rust.
            AndroidView(
                factory = { ctx ->
                    WebView(ctx).apply {
                        layoutParams = ViewGroup.LayoutParams(
                            ViewGroup.LayoutParams.MATCH_PARENT,
                            ViewGroup.LayoutParams.MATCH_PARENT,
                        )
                        settings.apply {
                            javaScriptEnabled = true
                            domStorageEnabled = true
                            allowFileAccess = true
                            cacheMode = WebSettings.LOAD_DEFAULT
                            // Mobile rendering surface — keep on
                            // canvas as the eframe UI is WebGL.
                            mediaPlaybackRequiresUserGesture = false
                            useWideViewPort = true
                            loadWithOverviewMode = true
                            setSupportZoom(false)
                            builtInZoomControls = false
                            displayZoomControls = false
                        }
                        setBackgroundColor(android.graphics.Color.parseColor("#0D1117"))
                        addJavascriptInterface(
                            bridge,
                            "NervBridge",
                        )
                        webViewClient = object : WebViewClient() {
                            override fun shouldOverrideUrlLoading(
                                view: WebView,
                                url: String,
                            ): Boolean {
                                // External URLs (https://...) — open
                                // in the system browser instead of
                                // replacing the wallet UI.
                                if (url.startsWith("https://") ||
                                    url.startsWith("http://")
                                ) {
                                    val intent = Intent(
                                        Intent.ACTION_VIEW,
                                        Uri.parse(url),
                                    ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                                    ContextCompat.startActivity(ctx, intent, null)
                                    return true
                                }
                                return false
                            }
                        }
                        loadUrl("file:///android_asset/www/index.html")
                        webView = this
                    }
                },
                modifier = Modifier.fillMaxSize(),
            )

            // First-load overlay: shimmer + brand wordmark. Fades out
            // once the eframe UI fires its "ready" event.
            AnimatedVisibility(
                visible = !webReady,
                enter = fadeIn(),
                exit = fadeOut(),
                modifier = Modifier.fillMaxSize(),
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(NervColors.Bg),
                    contentAlignment = Alignment.Center,
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        androidx.compose.foundation.Image(
                            painter = painterResource(R.drawable.ic_nerv_logo),
                            contentDescription = null,
                            modifier = Modifier.size(120.dp),
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = stringResource(R.string.splash_loading),
                            style = MaterialTheme.typography.labelMedium,
                            color = NervColors.TextMuted,
                        )
                    }
                }
            }
        }
    }
}