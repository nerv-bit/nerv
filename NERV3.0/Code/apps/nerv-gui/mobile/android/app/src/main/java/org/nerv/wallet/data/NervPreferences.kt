// NervPreferences (erratum 205).
//
// Encrypted SharedPreferences for the small pieces of state the
// native shell needs to persist across launches:
//   - "has the user successfully unlocked before?" — drives the
//     biometric prompt (we still prompt on cold start to keep the
//     session fresh; this is for UX, not security).
//   - "last active screen index" — restores the bottom-nav tab.
//   - "haptic feedback enabled" — UX preference.
//
// None of this is cryptographic material. The wallet's seed is held
// inside the WebView's IndexedDB and is wiped on Lock.

package org.nerv.wallet.data

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

object NervPreferences {

    private const val FILE_NAME = "nerv_prefs"

    private const val KEY_LAST_SCREEN  = "last_screen"
    private const val KEY_HAPTICS_ON   = "haptics_on"
    private const val KEY_NOTIFICATIONS_ON = "notifications_on"

    /**
     * Build (or open) the encrypted prefs. MasterKey is created with
     * AES-256-GCM, the recommended AndroidX security default.
     */
    fun open(context: Context): SharedPreferences {
        val key = MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
        return EncryptedSharedPreferences.create(
            context,
            FILE_NAME,
            key,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
        )
    }

    var lastScreenIndex: Int
        get() = open(NervApplication.instance).getInt(KEY_LAST_SCREEN, 0)
        set(value) {
            open(NervApplication.instance).edit()
                .putInt(KEY_LAST_SCREEN, value)
                .apply()
        }

    var hapticsEnabled: Boolean
        get() = open(NervApplication.instance).getBoolean(KEY_HAPTICS_ON, true)
        set(value) {
            open(NervApplication.instance).edit()
                .putBoolean(KEY_HAPTICS_ON, value)
                .apply()
        }

    var notificationsEnabled: Boolean
        get() = open(NervApplication.instance)
            .getBoolean(KEY_NOTIFICATIONS_ON, true)
        set(value) {
            open(NervApplication.instance).edit()
                .putBoolean(KEY_NOTIFICATIONS_ON, value)
                .apply()
        }
}