// NervBottomNav.swift (erratum 205; iOS counterpart to
// `apps/nerv-gui/mobile/android/.../ui/components/NervBottomNav.kt`).
//
// Native bottom navigation with the 8 desktop screens. The active
// screen is highlighted with the brand gold; tapping a tab fires
// the JS bridge `nav` event so the WKWebView's eframe UI navigates
// to the matching `Screen` via `WalletAction::Navigate`.
//
// iOS-native custom tab bar (not the system `TabView` — that's more
// like a top-level navigation pattern and doesn't fit the chrome
// metaphor we want here).

import SwiftUI

struct NervBottomNav: View {
    let active: WalletScreen
    let onSelect: (WalletScreen) -> Void

    var body: some View {
        VStack(spacing: 0) {
            // Top border so the bar reads against the WebView above.
            Rectangle()
                .fill(NervTheme.borderSubtle)
                .frame(height: 1)

            HStack(spacing: 0) {
                ForEach(WalletScreen.allCases) { screen in
                    BottomNavItem(
                        screen: screen,
                        selected: screen == active,
                        onClick: { onSelect(screen) },
                    )
                    .frame(maxWidth: .infinity)
                }
            }
            .padding(.horizontal, 4)
            .padding(.top, NervSpacing.xs + 2)
            .padding(.bottom, NervSpacing.xs + 2)
            // Respect the home-indicator on Face-ID devices.
            .background(
                NervTheme.bgElev.ignoresSafeArea(edges: .bottom),
            )
        }
        .background(NervTheme.bgElev)
    }
}

private struct BottomNavItem: View {
    let screen: WalletScreen
    let selected: Bool
    let onClick: () -> Void

    var body: some View {
        Button(action: onClick) {
            VStack(spacing: 2) {
                ZStack {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(selected ? NervTheme.gold.opacity(0.16) : .clear)
                        .frame(width: 36, height: 28)
                    Image(systemName: iconName(for: screen))
                        .font(.system(size: 16, weight: selected ? .semibold : .regular))
                        .foregroundStyle(selected ? NervTheme.gold : NervTheme.textMuted)
                }
                Text(screen.title)
                    .font(.system(size: 9, weight: selected ? .semibold : .regular))
                    .foregroundStyle(selected ? NervTheme.gold : NervTheme.textMuted)
                    .lineLimit(1)
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 6)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(Text(screen.title))
    }

    /// SF Symbol name per screen. The NERV wallet uses the system
    /// icons so the chrome feels native to iOS while staying
    /// recognisable across screens.
    private func iconName(for screen: WalletScreen) -> String {
        switch screen {
        case .dashboard: return "square.grid.2x2"
        case .send:      return "arrow.up.right"
        case .receive:   return "arrow.down.left"
        case .claim:     return "wallet.pass"
        case .producer:  return "cpu"
        case .history:   return "clock.arrow.circlepath"
        case .settings:  return "gearshape"
        case .help:      return "questionmark.circle"
        }
    }
}

#Preview {
    VStack(spacing: 0) {
        Spacer()
        NervBottomNav(active: .dashboard, onSelect: { _ in })
    }
    .background(NervTheme.bg)
    .preferredColorScheme(.dark)
}