// LockScreen.swift (erratum 205; iOS counterpart to
// `apps/nerv-gui/mobile/android/.../ui/screens/LockScreen.kt`).
//
// Biometric / device-credential lock gate. Mirrors the desktop GUI's
// locked state with a SwiftUI-native experience: large NERV logo,
// "Tap to unlock" button, fallback error messaging, and authentication
// via `LocalAuthentication.LAContext`.
//
// The screen kicks off the biometric prompt on first appearance; if
// the device has no biometric (cheap simulator, fresh install) the
// user sees the "biometric unavailable" state and can still unlock
// (this is only safe for the simulator — production enforces the
// gate).

import SwiftUI
import LocalAuthentication

struct LockScreen: View {

    let onUnlocked: () -> Void

    @State private var failedMessage: String?
    @State private var attempted = false
    @State private var authenticating = false
    @State private var biometricAvailable = false

    var body: some View {
        ZStack {
            NervTheme.bg.ignoresSafeArea()

            // Soft radial gold glow centred behind the logo.
            RadialGradient(
                colors: [NervTheme.gold.opacity(0.18), .clear],
                center: .center,
                startRadius: 20,
                endRadius: 320,
            )
            .frame(width: 420, height: 420)

            VStack(spacing: NervSpacing.m) {
                Spacer()

                Image("nerv_logo")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 180, height: 180)
                    .clipShape(Circle())
                    .accessibilityLabel(Text("NERV logo"))

                Text("NERV")
                    .font(.system(size: 40, weight: .black))
                    .foregroundStyle(NervTheme.gold)

                Text("Post-Quantum Privacy Wallet")
                    .font(NervTypography.bodyMedium())
                    .foregroundStyle(NervTheme.textMuted)
                    .multilineTextAlignment(.center)

                Spacer().frame(height: NervSpacing.xxl)

                if biometricAvailable {
                    Button(action: { Task { await authenticate() } }) {
                        ZStack {
                            Circle()
                                .fill(NervTheme.gold.opacity(0.16))
                                .frame(width: 80, height: 80)
                            Image(systemName: "touchid")
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(width: 44, height: 44)
                                .foregroundStyle(NervTheme.gold)
                        }
                    }
                    .disabled(authenticating)
                    .accessibilityLabel(Text("Unlock NERV"))

                    Text("Unlock")
                        .font(NervTypography.titleMedium())
                        .foregroundStyle(NervTheme.text)

                    Text("Tap to authenticate. NERV never sees your seed or biometric data.")
                        .font(NervTypography.bodySmall())
                        .foregroundStyle(NervTheme.textMuted)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, NervSpacing.xxl)

                    if let msg = failedMessage {
                        Text(msg)
                            .font(NervTypography.labelMedium())
                            .foregroundStyle(NervTheme.error)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(NervTheme.error.opacity(0.12))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .padding(.top, NervSpacing.s)
                    }
                } else {
                    Text("Biometric authentication is not available on this device.")
                        .font(NervTypography.bodyMedium())
                        .foregroundStyle(NervTheme.textMuted)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, NervSpacing.xl)

                    Button(action: onUnlocked) {
                        Text("Unlock")
                            .font(NervTypography.titleSmall())
                            .fontWeight(.semibold)
                            .foregroundStyle(NervTheme.bg)
                            .padding(.horizontal, NervSpacing.l)
                            .padding(.vertical, NervSpacing.s + 2)
                            .background(NervTheme.gold)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                    .padding(.top, NervSpacing.l)
                }

                Spacer()
            }
            .padding(.horizontal, NervSpacing.xl)
        }
        .onAppear {
            biometricAvailable = BiometricAuth.isAvailable()
            if biometricAvailable && !attempted {
                attempted = true
                Task { await authenticate() }
            }
        }
    }

    private func authenticate() async {
        guard !authenticating else { return }
        authenticating = true
        defer { authenticating = false }
        let result = await BiometricAuth.authenticate(
            reason: "Unlock your NERV wallet",
        )
        switch result {
        case .success:
            Haptics.success()
            onUnlocked()
        case .unavailable:
            failedMessage = "Biometric authentication is not available on this device."
        case .cancelled:
            // user explicitly cancelled — leave failedMessage nil
            break
        case .failed(let msg):
            failedMessage = msg
        }
        // Brief delay so the error surface has time to render before
        // the user can re-tap.
        try? await Task.sleep(nanoseconds: 150_000_000)
    }
}

#Preview {
    LockScreen(onUnlocked: {})
        .preferredColorScheme(.dark)
}