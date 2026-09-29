// NervTopBar.swift (erratum 205; iOS counterpart to
// `apps/nerv-gui/mobile/android/.../ui/components/NervTopBar.kt`).
//
// Native top bar: brand logo (small) on the left, chain sync status
// in the centre-right, lock button on the far right. Painted with a
// subtle bottom border so it reads as a "bar" against the WebView
// below.

import SwiftUI

struct NervTopBar: View {
    let syncState: SyncState
    let onLock: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: NervSpacing.s + 2) {
                // Brand: small logo + wordmark.
                Image("nerv_logo")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 32, height: 32)
                    .clipShape(Circle())
                    .accessibilityLabel(Text("NERV logo"))

                Text("NERV")
                    .font(.system(size: 20, weight: .black))
                    .foregroundStyle(NervTheme.gold)

                Spacer(minLength: 0)

                SyncBadge(state: syncState)

                Button(action: onLock) {
                    Image(systemName: "lock.fill")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(NervTheme.textMuted)
                        .padding(8)
                        .background(NervTheme.bg)
                        .clipShape(Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(Text("Lock wallet"))
            }
            .padding(.horizontal, NervSpacing.m)
            .padding(.top, NervSpacing.s)
            .padding(.bottom, NervSpacing.s)

            // Bottom border.
            Rectangle()
                .fill(NervTheme.borderSubtle)
                .frame(height: 1)
        }
        .background(NervTheme.bgElev)
    }
}

private struct SyncBadge: View {
    let state: SyncState

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(state.color)
                .frame(width: 8, height: 8)
            Text(state.label)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(state.color)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .background(state.color.opacity(0.16))
        .clipShape(Capsule())
    }
}

#Preview {
    VStack(spacing: 0) {
        NervTopBar(syncState: .synced, onLock: {})
        Spacer()
    }
    .background(NervTheme.bg)
    .preferredColorScheme(.dark)
}