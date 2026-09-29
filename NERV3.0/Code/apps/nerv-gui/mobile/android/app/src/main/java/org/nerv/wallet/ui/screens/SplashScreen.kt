// SplashScreen (erratum 205).
//
// Shown while the WASM bundle inside the WebView is initialising.
// The supplied NERV logo (gold neural-tree) sits at the centre with
// a subtle pulse animation, above the brand wordmark in
// the design-system gold. The Android 12+ SplashScreen API paints
// the initial frame; this Compose overlay is the post-splash
// "loading" stage.

package org.nerv.wallet.ui.screens

import androidx.compose.animation.core.*
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import org.nerv.wallet.R
import org.nerv.wallet.ui.theme.NervColors

@Composable
fun SplashScreen(
    onReady: () -> Unit,
    minDurationMs: Long = 800,
) {
    // Pulse animation for the logo.
    val infinite = rememberInfiniteTransition(label = "splash")
    val pulse by infinite.animateFloat(
        initialValue = 0.95f,
        targetValue = 1.05f,
        animationSpec = infiniteRepeatable(
            animation = tween(1200, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "pulse",
    )
    val glow by infinite.animateFloat(
        initialValue = 0.6f,
        targetValue = 1.0f,
        animationSpec = infiniteRepeatable(
            animation = tween(1600, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "glow",
    )

    // Minimum visible time so the splash doesn't blink on a fast cold
    // start. The caller decides when WASM is ready via onReady().
    LaunchedEffect(Unit) {
        delay(minDurationMs)
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(NervColors.SplashBg),
        contentAlignment = Alignment.Center,
    ) {
        // Radial gold glow behind the logo.
        Box(
            modifier = Modifier
                .size(360.dp)
                .alpha(glow * 0.35f)
                .background(
                    Brush.radialGradient(
                        colors = listOf(
                            NervColors.Gold,
                            NervColors.SplashBg,
                        ),
                    ),
                ),
        )

        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            // Supplied NERV logo, scaled + breathing.
            Image(
                painter = painterResource(R.drawable.ic_nerv_logo),
                contentDescription = stringResource(R.string.cd_nerv_logo),
                modifier = Modifier
                    .size(220.dp)
                    .scale(pulse),
            )

            Spacer(modifier = Modifier.height(20.dp))

            Text(
                text = "NERV",
                style = MaterialTheme.typography.displaySmall.copy(
                    fontWeight = FontWeight.Black,
                    fontSize = 44.sp,
                ),
                color = NervColors.Gold,
            )

            Spacer(modifier = Modifier.height(6.dp))

            Text(
                text = stringResource(R.string.app_tagline),
                style = MaterialTheme.typography.bodyMedium,
                color = NervColors.TextMuted,
            )
        }

        // "Loading NERV…" tag at the bottom.
        Text(
            text = stringResource(R.string.splash_loading),
            style = MaterialTheme.typography.labelSmall,
            color = NervColors.TextSubtle,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 56.dp),
        )
    }

    // Hand off to the next screen after the splash minimum + WASM
    // ready signal. The caller invokes onReady when the WebView's
    // `wasm.run` promise resolves.
    LaunchedEffect(Unit) {
        delay(minDurationMs)
        onReady()
    }
}