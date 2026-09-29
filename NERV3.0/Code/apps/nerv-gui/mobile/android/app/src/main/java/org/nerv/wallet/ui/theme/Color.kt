// NERV design tokens (erratum 205).
//
// The colour palette mirrors the desktop GUI's eframe theme so the
// native chrome (top bar, bottom nav, splash) and the WebView canvas
// feel like one continuous app. The accent gold is sampled from the
// supplied NERV logo; the cool blue accent matches the desktop GUI's
// primary colour so cross-platform users get the same brand.

package org.nerv.wallet.ui.theme

import androidx.compose.ui.graphics.Color

// Surface tokens (dark-first; light variants declared for completeness).
object NervColors {
    // Backgrounds.
    val Bg          = Color(0xFF0D1117)
    val BgElev      = Color(0xFF161B22)
    val Surface     = Color(0xFF1C2128)
    val SurfaceVar  = Color(0xFF21262D)

    // Text.
    val Text        = Color(0xFFE6EDF3)
    val TextMuted   = Color(0xFF8B949E)
    val TextSubtle  = Color(0xFF6E7681)

    // Accent — primary brand colour (matches desktop GUI).
    val Accent      = Color(0xFF58A6FF)
    val AccentHi    = Color(0xFF79B8FF)
    val AccentLo    = Color(0xFF388BFD)

    // Brand gold (sampled from the supplied NERV logo).
    val Gold        = Color(0xFFD4AF37)
    val GoldHi      = Color(0xFFE9C75A)
    val GoldLo      = Color(0xFF8C7220)

    // Semantic.
    val Success     = Color(0xFF3FB950)
    val Warning     = Color(0xFFD29922)
    val Error       = Color(0xFFF85149)
    val Info        = Color(0xFF58A6FF)

    // Borders.
    val Border      = Color(0xFF30363D)
    val BorderSubtle = Color(0xFF21262D)

    // Splash.
    val SplashBg    = Color(0xFF0D1117)

    // Scrim (for status bar / nav bar blending).
    val Scrim       = Color(0xCC0D1117)
}