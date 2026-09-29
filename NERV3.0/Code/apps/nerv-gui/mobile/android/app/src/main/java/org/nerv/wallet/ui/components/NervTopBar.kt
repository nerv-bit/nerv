// NervTopBar (erratum 205).
//
// Native top bar: brand logo (small) on the left, chain sync status
// in the centre, lock button on the right. Painted with a subtle
// bottom border so it reads as a "bar" against the WebView below.

package org.nerv.wallet.ui.components

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Lock
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.nerv.wallet.R
import org.nerv.wallet.ui.screens.SyncState
import org.nerv.wallet.ui.theme.NervColors

@Composable
fun NervTopBar(
    syncState: SyncState,
    onLock: () -> Unit,
) {
    Surface(
        color = NervColors.BgElev,
        contentColor = NervColors.Text,
        tonalElevation = 0.dp,
        shadowElevation = 0.dp,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.fillMaxWidth()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .statusBarsPadding()
                    .padding(horizontal = 16.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                // Brand: small logo + wordmark.
                Image(
                    painter = painterResource(R.drawable.ic_nerv_logo),
                    contentDescription = stringResource(R.string.cd_nerv_logo),
                    modifier = Modifier
                        .size(36.dp)
                        .clip(CircleShape),
                )
                Spacer(modifier = Modifier.width(10.dp))
                Text(
                    text = "NERV",
                    style = MaterialTheme.typography.titleLarge.copy(
                        fontWeight = FontWeight.Black,
                        fontSize = 22.sp,
                    ),
                    color = NervColors.Gold,
                )

                Spacer(modifier = Modifier.weight(1f))

                // Sync status pill.
                SyncBadge(syncState)

                Spacer(modifier = Modifier.width(8.dp))

                // Lock button.
                FilledTonalIconButton(
                    onClick = onLock,
                    colors = IconButtonDefaults.filledTonalIconButtonColors(
                        containerColor = NervColors.Bg,
                        contentColor = NervColors.TextMuted,
                    ),
                ) {
                    Icon(
                        imageVector = Icons.Rounded.Lock,
                        contentDescription = stringResource(
                            R.string.cd_lock_button
                        ),
                        modifier = Modifier.size(18.dp),
                    )
                }
            }
            // Bottom border to read as a bar.
            Spacer(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(1.dp)
                    .background(NervColors.BorderSubtle),
            )
        }
    }
}

@Composable
private fun SyncBadge(syncState: SyncState) {
    val (label, color, labelRes) = when (syncState) {
        SyncState.Synced  -> Triple(
            "Synced", NervColors.Success, R.string.status_synced
        )
        SyncState.Syncing -> Triple(
            "Syncing", NervColors.Warning, R.string.status_syncing
        )
        SyncState.Offline -> Triple(
            "Offline", NervColors.Error, R.string.status_offline
        )
        SyncState.Locked  -> Triple(
            "Locked", NervColors.TextMuted, R.string.status_locked
        )
    }
    Surface(
        color = color.copy(alpha = 0.16f),
        contentColor = color,
        shape = androidx.compose.foundation.shape.RoundedCornerShape(50),
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(
                horizontal = 10.dp,
                vertical = 4.dp,
            ),
        ) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .clip(CircleShape)
                    .background(color),
            )
            Spacer(modifier = Modifier.width(6.dp))
            Text(
                text = stringResource(labelRes),
                style = MaterialTheme.typography.labelSmall.copy(
                    fontWeight = FontWeight.SemiBold,
                ),
                color = color,
            )
        }
    }
}