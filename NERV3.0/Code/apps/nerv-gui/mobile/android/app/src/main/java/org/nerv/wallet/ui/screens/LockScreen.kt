// LockScreen (erratum 205).
//
// Biometric / device-credential lock gate. Mirrors the desktop GUI's
// locked state with a Compose-native experience: large NERV logo,
// "Tap to unlock" button, fallback error messaging, and
// authentication via AndroidX BiometricPrompt.
//
// The screen kicks off the biometric prompt on first composition;
// if the device has no biometric (cheap tablet, fresh install) the
// user sees the "biometric unavailable" state and can still unlock
// by setting up device-credential.

package org.nerv.wallet.ui.screens

import android.app.Activity
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Fingerprint
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.fragment.app.FragmentActivity
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.nerv.wallet.R
import org.nerv.wallet.data.BiometricAuth
import org.nerv.wallet.data.BiometricResult
import org.nerv.wallet.ui.theme.NervColors

@Composable
fun LockScreen(
    onUnlocked: () -> Unit,
) {
    val context = LocalContext.current
    val activity = context as? FragmentActivity
    val scope = rememberCoroutineScope()

    var failedMessage by remember { mutableStateOf<String?>(null) }
    var attempted by remember { mutableStateOf(false) }
    var authenticating by remember { mutableStateOf(false) }
    val biometricAvailable = remember(activity) {
        activity != null && BiometricAuth.isAvailable(activity)
    }

    // Kick off the prompt on first composition if biometric is
    // available. Wrapped in a key-less LaunchedEffect so it fires
    // exactly once.
    LaunchedEffect(Unit) {
        if (biometricAvailable && activity != null && !attempted) {
            attempted = true
            runAuth(
                activity = activity,
                scope = scope,
                setAuthenticating = { authenticating = it },
                onSuccess = onUnlocked,
                onFailed = { failedMessage = it },
            )
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(NervColors.Bg),
        contentAlignment = Alignment.Center,
    ) {
        // Soft gold glow centred behind the logo.
        Box(
            modifier = Modifier
                .size(420.dp)
                .background(
                    Brush.radialGradient(
                        colors = listOf(
                            NervColors.Gold.copy(alpha = 0.18f),
                            NervColors.Bg,
                        ),
                    ),
                ),
        )

        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
            modifier = Modifier.padding(horizontal = 32.dp),
        ) {
            // NERV logo.
            Image(
                painter = painterResource(R.drawable.ic_nerv_logo),
                contentDescription = stringResource(R.string.cd_nerv_logo),
                modifier = Modifier
                    .size(180.dp)
                    .clip(CircleShape),
            )

            Spacer(modifier = Modifier.height(24.dp))

            Text(
                text = "NERV",
                style = MaterialTheme.typography.displaySmall.copy(
                    fontWeight = FontWeight.Black,
                    fontSize = 40.sp,
                ),
                color = NervColors.Gold,
            )

            Spacer(modifier = Modifier.height(4.dp))

            Text(
                text = stringResource(R.string.app_tagline),
                style = MaterialTheme.typography.bodyMedium,
                color = NervColors.TextMuted,
                textAlign = TextAlign.Center,
            )

            Spacer(modifier = Modifier.height(56.dp))

            if (biometricAvailable) {
                FilledTonalIconButton(
                    onClick = {
                        if (!authenticating && activity != null) {
                            attempted = true
                            runAuth(
                                activity = activity,
                                scope = scope,
                                setAuthenticating = { authenticating = it },
                                onSuccess = onUnlocked,
                                onFailed = { failedMessage = it },
                            )
                        }
                    },
                    modifier = Modifier.size(80.dp),
                    colors = IconButtonDefaults.filledTonalIconButtonColors(
                        containerColor = NervColors.Gold.copy(alpha = 0.16f),
                        contentColor = NervColors.Gold,
                    ),
                ) {
                    Icon(
                        imageVector = Icons.Rounded.Fingerprint,
                        contentDescription = stringResource(
                            R.string.biometric_title
                        ),
                        modifier = Modifier.size(40.dp),
                    )
                }

                Spacer(modifier = Modifier.height(20.dp))

                Text(
                    text = stringResource(R.string.lock_unlock_button),
                    style = MaterialTheme.typography.titleMedium,
                    color = NervColors.Text,
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = stringResource(R.string.lock_help),
                    style = MaterialTheme.typography.bodySmall,
                    color = NervColors.TextMuted,
                    textAlign = TextAlign.Center,
                )

                AnimatedVisibility(visible = failedMessage != null) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Spacer(modifier = Modifier.height(12.dp))
                        Surface(
                            color = NervColors.Error.copy(alpha = 0.12f),
                            shape = RoundedCornerShape(8.dp),
                        ) {
                            Text(
                                text = failedMessage
                                    ?: stringResource(R.string.lock_failed),
                                style = MaterialTheme.typography.labelMedium,
                                color = NervColors.Error,
                                modifier = Modifier.padding(
                                    horizontal = 12.dp,
                                    vertical = 6.dp,
                                ),
                            )
                        }
                    }
                }
            } else {
                Text(
                    text = stringResource(R.string.lock_biometric_unavailable),
                    style = MaterialTheme.typography.bodyMedium,
                    color = NervColors.TextMuted,
                    textAlign = TextAlign.Center,
                )
                Spacer(modifier = Modifier.height(20.dp))
                FilledTonalButton(
                    onClick = onUnlocked,
                    colors = ButtonDefaults.filledTonalButtonColors(
                        containerColor = NervColors.Gold,
                        contentColor = NervColors.Bg,
                    ),
                ) {
                    Text(
                        text = stringResource(R.string.lock_unlock_button),
                        style = MaterialTheme.typography.titleSmall.copy(
                            fontWeight = FontWeight.SemiBold,
                        ),
                    )
                }
            }
        }
    }
}

private fun runAuth(
    activity: FragmentActivity,
    scope: kotlinx.coroutines.CoroutineScope,
    setAuthenticating: (Boolean) -> Unit,
    onSuccess: () -> Unit,
    onFailed: (String) -> Unit,
) {
    setAuthenticating(true)
    scope.launch {
        val result = BiometricAuth.authenticate(
            activity = activity,
            title = activity.getString(R.string.biometric_title),
            subtitle = activity.getString(R.string.biometric_subtitle),
            description = activity.getString(R.string.biometric_description),
            negativeLabel = activity.getString(R.string.biometric_negative),
        )
        setAuthenticating(false)
        when (result) {
            BiometricResult.Success -> onSuccess()
            BiometricResult.NotAvailable -> onFailed(
                activity.getString(R.string.lock_biometric_unavailable)
            )
            BiometricResult.Cancelled -> {
                // user explicitly cancelled — leave failed message null
            }
            is BiometricResult.Failed -> onFailed(result.message)
        }
        // Brief delay so the error surface has time to animate in
        // before the user can re-tap.
        delay(150)
    }
}