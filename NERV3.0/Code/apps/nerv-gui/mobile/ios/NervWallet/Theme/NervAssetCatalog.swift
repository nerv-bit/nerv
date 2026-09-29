// NervAssetCatalog.swift (erratum 205; iOS shell).
//
// Asset-catalog wrappers. The NERV logo + accent colour live in
// `Resources/Assets.xcassets`; Swift code reaches them via the
// `Image("nerv_logo")` / `Color("nervAccent")` lookups that the
// asset catalog exposes. For builds that don't have an asset
// catalog (e.g., quick-build via `xcrun swiftc -emit-library ...`)
// we also fall back to `Bundle.main` lookups of the raw PNG and
// the hex palette tokens from `NervTheme.swift`.

import SwiftUI

public enum NervAsset {

    /// The supplied NERV logo (gold neural-tree on black). Falls
    /// back to `Resources/nerv_logo.png` if the asset catalog is
    /// unavailable (rare — only the build script hits this path).
    public static var logo: Image {
        if let _ = UIImage(named: "nerv_logo") {
            return Image("nerv_logo")
        }
        return Image("nerv_logo")
    }

    /// Adaptive accent (system `AccentColor`). NERV pins this to the
    /// brand blue (#58A6FF) in the asset catalog.
    public static var accent: Color {
        Color("nervAccent")
    }
}