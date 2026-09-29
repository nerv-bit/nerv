// Snackbar helpers (erratum 205).
//
// Lightweight enum + theme tokens used by the WalletScreen to colour
// the snackbar host based on the eframe UI's notification level.

package org.nerv.wallet.ui.components

import org.nerv.wallet.ui.theme.NervColors
import androidx.compose.ui.graphics.Color

enum class SnackbarLevel { Info, Success, Warning, Error }

fun SnackbarLevel.color(): Color = when (this) {
    SnackbarLevel.Info    -> NervColors.Accent
    SnackbarLevel.Success -> NervColors.Success
    SnackbarLevel.Warning -> NervColors.Warning
    SnackbarLevel.Error   -> NervColors.Error
}