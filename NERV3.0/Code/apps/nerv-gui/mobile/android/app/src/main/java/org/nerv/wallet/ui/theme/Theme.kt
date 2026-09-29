// NERV theme (erratum 205).
//
// Material 3 theme with the NERV palette wired in. The colour scheme
// matches the desktop GUI's eframe theme so the WebView canvas and
// the native chrome blend seamlessly.

package org.nerv.wallet.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val NervDarkScheme = darkColorScheme(
    primary           = NervColors.Accent,
    onPrimary         = NervColors.Text,
    primaryContainer  = NervColors.AccentLo,
    onPrimaryContainer = NervColors.Text,

    secondary         = NervColors.Gold,
    onSecondary       = NervColors.Bg,
    secondaryContainer = NervColors.GoldLo,
    onSecondaryContainer = NervColors.Text,

    tertiary          = NervColors.Success,
    onTertiary        = NervColors.Bg,

    background        = NervColors.Bg,
    onBackground      = NervColors.Text,

    surface           = NervColors.BgElev,
    onSurface         = NervColors.Text,
    surfaceVariant    = NervColors.SurfaceVariant,
    onSurfaceVariant  = NervColors.TextMuted,

    error             = NervColors.Error,
    onError           = NervColors.Text,

    outline           = NervColors.Border,
    outlineVariant    = NervColors.BorderSubtle,

    scrim             = NervColors.Scrim,
)

// NERV is dark-first; light is offered for accessibility but the
// primary brand is the dark scheme (the supplied NERV logo is on a
// black background).
private val NervLightScheme = lightColorScheme(
    primary           = NervColors.AccentLo,
    onPrimary         = NervColors.Text,
    primaryContainer  = NervColors.Accent,
    onPrimaryContainer = NervColors.Bg,

    secondary         = NervColors.GoldLo,
    onSecondary       = NervColors.Bg,
    secondaryContainer = NervColors.Gold,
    onSecondaryContainer = NervColors.Bg,

    tertiary          = NervColors.Success,
    onTertiary        = NervColors.Bg,

    background        = Color(0xFFF6F8FA),
    onBackground      = Color(0xFF0D1117),

    surface           = Color(0xFFFFFFFF),
    onSurface         = Color(0xFF0D1117),
    surfaceVariant    = Color(0xFFE6EDF3),
    onSurfaceVariant  = Color(0xFF57606A),

    error             = NervColors.Error,
    onError           = NervColors.Text,

    outline           = Color(0xFFD0D7DE),
    outlineVariant    = Color(0xFFAFB8C1),

    scrim             = NervColors.Scrim,
)

/**
 * The NERV theme entry point. The [darkTheme] flag defaults to the
 * system setting; we expect most users to use NERV in dark mode (the
 * brand and supplied NERV logo are dark-first).
 *
 * On Android 12+ we also paint the system bars to match the theme
 * background so the chrome is flush with the WebView below it.
 */
@Composable
fun NervTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    val colorScheme = if (darkTheme) NervDarkScheme else NervLightScheme

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.background.toArgb()
            window.navigationBarColor = colorScheme.background.toArgb()
            val controller = WindowCompat.getInsetsController(window, view)
            controller.isAppearanceLightStatusBars = !darkTheme
            controller.isAppearanceLightNavigationBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = NervTypography,
        content = content,
    )
}