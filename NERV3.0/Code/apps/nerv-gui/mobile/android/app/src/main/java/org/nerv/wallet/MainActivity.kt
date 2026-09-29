// MainActivity (erratum 205).
//
// The single Activity hosting the NERV mobile shell. Wires the
// three top-level Compose screens into a state machine:
//
//   Splash ──(min duration + WASM ready)──▶ LockScreen
//   LockScreen ──(biometric success)──▶ WalletScreen
//   WalletScreen ──(lock button)──▶ LockScreen
//
// All wallet logic lives in Rust (compiled to WASM and rendered by
// eframe inside the WebView in `WalletScreen`). The Compose shell
// here is purely a native chrome (splash, biometric lock, top/bottom
// navigation, snackbars) plus the Kotlin <-> JS bridge.

package org.nerv.wallet

import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.activity.SystemBarStyle
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.fragment.app.FragmentActivity
import org.nerv.wallet.ui.screens.LockScreen
import org.nerv.wallet.ui.screens.SplashScreen
import org.nerv.wallet.ui.screens.WalletScreen
import org.nerv.wallet.ui.theme.NervColors
import org.nerv.wallet.ui.theme.NervTheme

class MainActivity : FragmentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        // Install the Android 12+ SplashScreen API first so the
        // initial frame paints with the NERV logo + brand background
        // (the post-splash Compose overlay is `SplashScreen.kt`).
        val splash = installSplashScreen()
        // Keep the splash visible until our Compose state says
        // "ready". The minimum duration is enforced in
        // `SplashScreen.kt` via a LaunchedEffect delay.
        var keepSplash = true
        splash.setKeepOnScreenCondition { keepSplash }

        super.onCreate(savedInstanceState)

        // Edge-to-edge: paint the system bars in the NERV palette so
        // the chrome blends with the WebView below.
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(
                android.graphics.Color.parseColor("#0D1117"),
            ),
            navigationBarStyle = SystemBarStyle.dark(
                android.graphics.Color.parseColor("#0D1117"),
            ),
        )

        setContent {
            NervTheme(darkTheme = true) {
                App(keepSplash = { keepSplash = it })
            }
        }
    }
}

@Composable
private fun App(keepSplash: (Boolean) -> Unit) {
    var phase by remember { mutableStateOf(AppPhase.Splash) }

    androidx.compose.foundation.layout.Box(
        modifier = Modifier
            .fillMaxSize()
            .background(NervColors.Bg),
    ) {
        when (phase) {
            AppPhase.Splash -> SplashScreen(
                onReady = {
                    keepSplash(false)
                    phase = AppPhase.Lock
                },
            )
            AppPhase.Lock -> LockScreen(
                onUnlocked = { phase = AppPhase.Wallet },
            )
            AppPhase.Wallet -> WalletScreen(
                onLock = { phase = AppPhase.Lock },
            )
        }
    }
}

private enum class AppPhase { Splash, Lock, Wallet }