// NERVTheme.swift (erratum 205; iOS counterpart to the Android shell).
//
// NERV design tokens for iOS. The palette mirrors the desktop GUI's
// eframe theme so the WebView canvas (running the WASM-built NERV
// wallet) and the native SwiftUI chrome blend seamlessly. The accent
// gold is sampled from the supplied NERV logo.
//
// Apple platform notes:
//   - All tokens are `Color` values, not `UIColor` — SwiftUI handles
//     dark/light automatically when wired through the asset catalog.
//   - The NERV palette is dark-first (the brand logo sits on a black
//     background); the app forces `.preferredColorScheme(.dark)` so
//     light-mode tokens are for accessibility only.
//   - Hex literals use the `Color(red:green:blue:opacity:)` form to
//     stay independent of any future asset catalog rename.

import SwiftUI

public enum NervTheme {

    // Backgrounds.
    public static let bg          = Color(red: 0x0D/255, green: 0x11/255, blue: 0x17/255)
    public static let bgElev      = Color(red: 0x16/255, green: 0x1B/255, blue: 0x22/255)
    public static let surface     = Color(red: 0x1C/255, green: 0x21/255, blue: 0x28/255)
    public static let surfaceVar  = Color(red: 0x21/255, green: 0x26/255, blue: 0x2D/255)

    // Text.
    public static let text        = Color(red: 0xE6/255, green: 0xED/255, blue: 0xF3/255)
    public static let textMuted   = Color(red: 0x8B/255, green: 0x94/255, blue: 0x9E/255)
    public static let textSubtle  = Color(red: 0x6E/255, green: 0x76/255, blue: 0x81/255)

    // Accent — primary brand colour (matches desktop GUI).
    public static let accent      = Color(red: 0x58/255, green: 0xA6/255, blue: 0xFF/255)
    public static let accentHi    = Color(red: 0x79/255, green: 0xB8/255, blue: 0xFF/255)
    public static let accentLo    = Color(red: 0x38/255, green: 0x8B/255, blue: 0xFD/255)

    // Brand gold (sampled from the supplied NERV logo).
    public static let gold        = Color(red: 0xD4/255, green: 0xAF/255, blue: 0x37/255)
    public static let goldHi      = Color(red: 0xE9/255, green: 0xC7/255, blue: 0x5A/255)
    public static let goldLo      = Color(red: 0x8C/255, green: 0x72/255, blue: 0x20/255)

    // Semantic.
    public static let success     = Color(red: 0x3F/255, green: 0xB9/255, blue: 0x50/255)
    public static let warning     = Color(red: 0xD2/255, green: 0x99/255, blue: 0x22/255)
    public static let error       = Color(red: 0xF8/255, green: 0x51/255, blue: 0x49/255)
    public static let info        = Color(red: 0x58/255, green: 0xA6/255, blue: 0xFF/255)

    // Borders.
    public static let border      = Color(red: 0x30/255, green: 0x36/255, blue: 0x3D/255)
    public static let borderSubtle = Color(red: 0x21/255, green: 0x26/255, blue: 0x2D/255)

    // Splash.
    public static let splashBg    = Color(red: 0x0D/255, green: 0x11/255, blue: 0x17/255)

    // Scrim.
    public static let scrim       = Color(red: 0x0D/255, green: 0x11/255, blue: 0x17/255).opacity(0.8)
}

// Typography tokens (mirror the desktop GUI's monospace + sans pairing).
public enum NervTypography {

    public static func displayLarge() -> Font {
        .system(size: 48, weight: .black, design: .default)
    }
    public static func displayMedium() -> Font {
        .system(size: 36, weight: .heavy, design: .default)
    }
    public static func displaySmall() -> Font {
        .system(size: 28, weight: .heavy, design: .default)
    }
    public static func headlineLarge() -> Font {
        .system(size: 24, weight: .semibold, design: .default)
    }
    public static func headlineMedium() -> Font {
        .system(size: 20, weight: .semibold, design: .default)
    }
    public static func headlineSmall() -> Font {
        .system(size: 18, weight: .semibold, design: .default)
    }
    public static func titleLarge() -> Font {
        .system(size: 16, weight: .medium, design: .default)
    }
    public static func titleMedium() -> Font {
        .system(size: 14, weight: .medium, design: .default)
    }
    public static func bodyLarge() -> Font {
        .system(size: 16, weight: .regular, design: .default)
    }
    public static func bodyMedium() -> Font {
        .system(size: 14, weight: .regular, design: .default)
    }
    public static func bodySmall() -> Font {
        .system(size: 12, weight: .regular, design: .default)
    }
    public static func labelLarge() -> Font {
        .system(size: 14, weight: .medium, design: .default)
    }
    public static func labelMedium() -> Font {
        .system(size: 12, weight: .medium, design: .default)
    }
    public static func labelSmall() -> Font {
        .system(size: 10, weight: .medium, design: .default)
    }
    /// Monospace — used for hex seeds, addresses, txids.
    public static func mono(size: CGFloat = 14, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }
}

// Spacing tokens (8-pt grid).
public enum NervSpacing {
    public static let xs: CGFloat  = 4
    public static let s:  CGFloat  = 8
    public static let m:  CGFloat  = 16
    public static let l:  CGFloat  = 24
    public static let xl: CGFloat  = 32
    public static let xxl:CGFloat  = 48
}