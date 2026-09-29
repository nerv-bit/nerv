// NervBottomNav (erratum 205).
//
// Native bottom navigation with the 8 desktop screens. The active
// screen is highlighted with the brand gold; tapping a tab fires
// the JS bridge `nav` event so the WebView's eframe UI navigates to
// the matching `Screen` via `WalletAction::Navigate`.

package org.nerv.wallet.ui.components

import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.nerv.wallet.ui.screens.WalletScreen
import org.nerv.wallet.ui.theme.NervColors

@Composable
fun NervBottomNav(
    active: WalletScreen,
    onSelect: (WalletScreen) -> Unit,
) {
    Surface(
        color = NervColors.BgElev,
        contentColor = NervColors.Text,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.fillMaxWidth()) {
            // Top border so the bar reads against the WebView above.
            Spacer(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(1.dp)
                    .background(NervColors.BorderSubtle),
            )

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .navigationBarsPadding()
                    .padding(horizontal = 4.dp, vertical = 6.dp),
                horizontalArrangement = Arrangement.SpaceAround,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                WalletScreen.entries.forEach { screen ->
                    BottomNavItem(
                        screen = screen,
                        selected = screen == active,
                        onClick = { onSelect(screen) },
                    )
                }
            }
        }
    }
}

@Composable
private fun BottomNavItem(
    screen: WalletScreen,
    selected: Boolean,
    onClick: () -> Unit,
) {
    val targetColor by animateColorAsState(
        targetValue = if (selected) NervColors.Gold else NervColors.TextMuted,
        label = "tabColor",
    )
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
        modifier = Modifier
            .clip(RoundedCornerShape(12.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 6.dp, vertical = 6.dp),
    ) {
        Box(
            modifier = Modifier
                .size(width = 36.dp, height = 28.dp)
                .clip(RoundedCornerShape(8.dp))
                .background(
                    if (selected)
                        NervColors.Gold.copy(alpha = 0.16f)
                    else
                        Color.Transparent
                ),
            contentAlignment = Alignment.Center,
        ) {
            Icon(
                imageVector = screen.icon,
                contentDescription = stringResource(
                    when (screen) {
                        WalletScreen.Dashboard -> org.nerv.wallet.R.string.cd_nav_dashboard
                        WalletScreen.Send      -> org.nerv.wallet.R.string.cd_nav_send
                        WalletScreen.Receive   -> org.nerv.wallet.R.string.cd_nav_receive
                        WalletScreen.Claim     -> org.nerv.wallet.R.string.cd_nav_claim
                        WalletScreen.Producer  -> org.nerv.wallet.R.string.cd_nav_producer
                        WalletScreen.History   -> org.nerv.wallet.R.string.cd_nav_history
                        WalletScreen.Settings  -> org.nerv.wallet.R.string.cd_nav_settings
                        WalletScreen.Help      -> org.nerv.wallet.R.string.cd_nav_help
                    }
                ),
                tint = targetColor,
                modifier = Modifier.size(20.dp),
            )
        }
        Spacer(modifier = Modifier.height(2.dp))
        Text(
            text = stringResource(screen.titleRes),
            style = MaterialTheme.typography.labelSmall.copy(
                fontSize = 9.sp,
                fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
            ),
            color = targetColor,
            maxLines = 1,
        )
    }
}