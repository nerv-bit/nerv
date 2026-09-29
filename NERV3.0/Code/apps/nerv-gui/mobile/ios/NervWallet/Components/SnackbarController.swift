// SnackbarController.swift (erratum 205; iOS counterpart to
// `apps/nerv-gui/mobile/android/.../ui/components/Snackbar.kt`).
//
// A tiny overlay-snackbar host. iOS doesn't ship a "snackbar" the
// way Material 3 does, so this is a `ZStack`-aligned card that
// auto-dismisses after a few seconds. Coloured by the
// notification's `level` (info / success / warning / error).

import SwiftUI

public struct SnackbarItem: Identifiable, Equatable {
    public let id = UUID()
    public let message: String
    public let level: SnackbarLevel

    public init(message: String, level: SnackbarLevel) {
        self.message = message
        self.level = level
    }
}

public struct SnackbarHost: View {
    @Binding public var item: SnackbarItem?

    public init(item: Binding<SnackbarItem?>) {
        self._item = item
    }

    public var body: some View {
        ZStack {
            if let item = item {
                VStack {
                    Spacer()
                    SnackbarCard(message: item.message, level: item.level)
                        .padding(.horizontal, NervSpacing.m)
                        .padding(.bottom, 88) // sits above the bottom nav
                        .transition(
                            .move(edge: .bottom).combined(with: .opacity)
                        )
                }
            }
        }
        .animation(.easeInOut(duration: 0.25), value: item)
    }
}

private struct SnackbarCard: View {
    let message: String
    let level: SnackbarLevel

    var body: some View {
        HStack(spacing: NervSpacing.s) {
            Image(systemName: iconName)
                .foregroundStyle(textColor)
            Text(message)
                .font(NervTypography.bodyMedium())
                .foregroundStyle(textColor)
                .lineLimit(3)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, NervSpacing.m)
        .padding(.vertical, NervSpacing.s + 4)
        .background(backgroundColor)
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(borderColor, lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.35), radius: 8, x: 0, y: 4)
    }

    private var iconName: String {
        switch level {
        case .info:    return "info.circle.fill"
        case .success: return "checkmark.circle.fill"
        case .warning: return "exclamationmark.triangle.fill"
        case .error:   return "xmark.octagon.fill"
        }
    }

    private var backgroundColor: Color {
        switch level {
        case .info:    return NervTheme.bgElev
        case .success: return NervTheme.success.opacity(0.18)
        case .warning: return NervTheme.warning.opacity(0.22)
        case .error:   return NervTheme.error.opacity(0.22)
        }
    }

    private var borderColor: Color {
        switch level {
        case .info:    return NervTheme.border
        case .success: return NervTheme.success
        case .warning: return NervTheme.warning
        case .error:   return NervTheme.error
        }
    }

    private var textColor: Color {
        switch level {
        case .info:    return NervTheme.text
        case .success: return NervTheme.success
        case .warning: return NervTheme.warning
        case .error:   return NervTheme.error
        }
    }
}