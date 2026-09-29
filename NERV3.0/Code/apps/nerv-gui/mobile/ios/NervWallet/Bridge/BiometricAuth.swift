// BiometricAuth.swift (erratum 205; iOS counterpart to
// `apps/nerv-gui/mobile/android/.../data/BiometricAuth.kt`).
//
// LocalAuthentication wrapper. The NERV wallet is unlocked via
// Touch ID / Face ID / device passcode. The wallet's seed never
// leaves the device; the biometric check only gates the WKWebView
// that hosts the eframe UI session.

import Foundation
import LocalAuthentication

public enum BiometricResult {
    case success
    case failed(String)
    case unavailable
    case cancelled
}

public enum BiometricAuth {

    /// Strong biometric with device-credential fallback. On older
    /// devices or simulators, `LAPolicy.deviceOwnerAuthentication`
    /// is the right call.
    private static let policy: LAPolicy = .deviceOwnerAuthentication

    /// Whether the device can perform the prompt right now. Returns
    /// `false` on simulators without an enrolled fingerprint/face
    /// or with no device passcode set.
    public static func isAvailable() -> Bool {
        let ctx = LAContext()
        var error: NSError?
        return ctx.canEvaluatePolicy(policy, error: &error)
    }

    /// Show the biometric prompt and suspend until the user
    /// authenticates (or cancels / fails).
    @MainActor
    public static func authenticate(reason: String) async -> BiometricResult {
        let ctx = LAContext()
        var error: NSError?
        guard ctx.canEvaluatePolicy(policy, error: &error) else {
            return .unavailable
        }
        do {
            try await ctx.evaluatePolicy(policy, localizedReason: reason)
            return .success
        } catch let err as LAError {
            switch err.code {
            case .userCancel, .appCancel, .systemCancel:
                return .cancelled
            case .biometryNotAvailable, .biometryNotEnrolled, .passcodeNotSet:
                return .unavailable
            default:
                return .failed(err.localizedDescription)
            }
        } catch {
            return .failed(error.localizedDescription)
        }
    }
}