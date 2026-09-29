// SplashScreen.swift (erratum 205; iOS counterpart to
// `apps/nerv-gui/mobile/android/.../ui/screens/SplashScreen.kt`).
//
// Shown while the WASM bundle inside the WKWebView is initialising.
// The supplied NERV logo (gold neural-tree) sits at the centre with
// a subtle pulse + glow animation, above the brand wordmark in the
// design-system gold.

import SwiftUI

struct SplashScreen: View {

    /// The minimum visible time so the splash doesn't blink on a fast
    /// cold start. The caller (`App.swift`) advances once the WASM
    /// ready signal fires (or after this duration, whichever comes
    /// first).
    let onReady: () -> Void
    let minDuration: TimeInterval = 0.8

    @State private var pulse: CGFloat = 0.95
    @State private var glow: Double = 0.6
    @State private var didFire = false

    var body: some View {
        ZStack {
            NervTheme.splashBg.ignoresSafeArea()

            // Soft radial gold glow behind the logo.
            RadialGradient(
                colors: [NervTheme.gold.opacity(glow * 0.35), .clear],
                center: .center,
                startRadius: 20,
                endRadius: 280,
            )
            .frame(width: 360, height: 360)

            VStack(spacing: NervSpacing.m) {
                Image("nerv_logo")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 220, height: 220)
                    .scaleEffect(pulse)
                    .accessibilityLabel(Text("NERV logo"))

                Text("NERV")
                    .font(.system(size: 44, weight: .black))
                    .foregroundStyle(NervTheme.gold)

                Text("Post-Quantum Privacy Wallet")
                    .font(NervTypography.bodyMedium())
                    .foregroundStyle(NervTheme.textMuted)
            }

            // "Loading NERV…" tag at the bottom.
            VStack {
                Spacer()
                Text("Loading NERV…")
                    .font(NervTypography.labelSmall())
                    .foregroundStyle(NervTheme.textSubtle)
                    .padding(.bottom, 56)
            }
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 1.2).repeatForever(autoreverses: true)) {
                pulse = 1.05
            }
            withAnimation(.easeInOut(duration: 1.6).repeatForever(autoreverses: true)) {
                glow = 1.0
            }
            // Schedule the minimum-duration handoff. The `App` phase
            // machine ALSO advances on the WASM ready event, so this
            // is the worst-case fallback.
            DispatchQueue.main.asyncAfter(deadline: .now() + minDuration) {
                guard !didFire else { return }
                didFire = true
                onReady()
            }
        }
    }
}

#Preview {
    SplashScreen(onReady: {})
        .preferredColorScheme(.dark)
}