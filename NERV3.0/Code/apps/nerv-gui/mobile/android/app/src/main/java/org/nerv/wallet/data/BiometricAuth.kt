// BiometricAuth (erratum 205).
//
// AndroidX BiometricPrompt wrapper. The NERV wallet is unlocked via
// fingerprint / face / device PIN. The wallet's seed never leaves
// the device; the biometric check only gates the WebView that holds
// the eframe UI session.

package org.nerv.wallet.data

import android.content.Context
import android.os.Build
import androidx.biometric.BiometricManager
import androidx.biometric.BiometricPrompt
import androidx.core.content.ContextCompat
import androidx.fragment.app.FragmentActivity
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume

/**
 * Outcome of an authentication attempt. Distinguishing
 * [Result.NotAvailable] (no biometric / no enrolled credential) from
 * [Result.Failed] (the prompt fired but the user failed to
 * authenticate) lets the caller decide whether to fall back to
 * device-credential PIN.
 */
sealed class BiometricResult {
    object Success : BiometricResult()
    data class Failed(val message: String) : BiometricResult()
    object NotAvailable : BiometricResult()
    object Cancelled : BiometricResult()
}

object BiometricAuth {

    /** Strong biometric, falling back to device PIN/pattern/password. */
    private const val ALLOWED_AUTH =
        BiometricManager.Authenticators.BIOMETRIC_STRONG or
            BiometricManager.Authenticators.DEVICE_CREDENTIAL

    /** Pure-biometric (no device credential). Used when API < 30. */
    private const val ALLOWED_BIOMETRIC_ONLY =
        BiometricManager.Authenticators.BIOMETRIC_STRONG or
            BiometricManager.Authenticators.BIOMETRIC_WEAK

    /**
     * Check whether the device can perform the prompt right now.
     * Returns false on:
     *  - hardware absent (cheap Android Go tablet, etc.)
     *  - no enrolled fingerprint/face
     *  - device-credential not set up
     */
    fun isAvailable(context: Context): Boolean {
        val manager = BiometricManager.from(context)
        val authenticators =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R)
                ALLOWED_AUTH
            else
                ALLOWED_BIOMETRIC_ONLY
        return manager.canAuthenticate(authenticators) ==
            BiometricManager.BIOMETRIC_SUCCESS
    }

    /**
     * Show the biometric prompt and suspend until the user
     * authenticates (or cancels / fails).
     */
    suspend fun authenticate(
        activity: FragmentActivity,
        title: String,
        subtitle: String,
        description: String,
        negativeLabel: String,
    ): BiometricResult {
        val manager = BiometricManager.from(activity)
        val authenticators =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R)
                ALLOWED_AUTH
            else
                ALLOWED_BIOMETRIC_ONLY
        if (manager.canAuthenticate(authenticators) !=
            BiometricManager.BIOMETRIC_SUCCESS
        ) {
            return BiometricResult.NotAvailable
        }

        return suspendCancellableCoroutine { cont ->
            val executor = ContextCompat.getMainExecutor(activity)
            val callback = object : BiometricPrompt.AuthenticationCallback() {
                override fun onAuthenticationSucceeded(
                    result: BiometricPrompt.AuthenticationResult,
                ) {
                    if (cont.isActive) cont.resume(BiometricResult.Success)
                }

                override fun onAuthenticationError(
                    errorCode: Int,
                    errString: CharSequence,
                ) {
                    if (!cont.isActive) return
                    when (errorCode) {
                        BiometricPrompt.ERROR_USER_CANCELED,
                        BiometricPrompt.ERROR_NEGATIVE_BUTTON,
                        BiometricPrompt.ERROR_CANCELED ->
                            cont.resume(BiometricResult.Cancelled)
                        else ->
                            cont.resume(
                                BiometricResult.Failed(errString.toString())
                            )
                    }
                }

                override fun onAuthenticationFailed() {
                    // No-op: BiometricPrompt will retry automatically
                    // until ERROR_MAX_ATTEMPTS or the user cancels.
                }
            }

            val prompt = BiometricPrompt(activity, executor, callback)
            val info = BiometricPrompt.PromptInfo.Builder()
                .setTitle(title)
                .setSubtitle(subtitle)
                .setDescription(description)
                .setAllowedAuthenticators(authenticators)
                .build()
            // Pre-Android 30 with `DEVICE_CREDENTIAL` cannot combine
            // with a negative button; with biometric-only the negative
            // label is required.
            if (authenticators == ALLOWED_BIOMETRIC_ONLY) {
                BiometricPrompt.PromptInfo.Builder()
                    .setNegativeButtonText(negativeLabel)
                    .build()
            }
            prompt.authenticate(info)
        }
    }
}