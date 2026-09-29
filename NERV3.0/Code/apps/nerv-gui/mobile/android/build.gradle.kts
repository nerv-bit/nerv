// Top-level build file (erratum 205; mobile shell).
// The NERV Android shell wraps the same Rust wallet-core that the
// desktop GUI uses — the wallet's logic lives in Rust (compiled to
// WASM) and runs inside an Android WebView. The Compose shell in this
// project is purely a thin native chrome (splash, biometric lock,
// top/bottom navigation, status badges, snackbars) plus a JS bridge
// so the native chrome can drive wallet actions dispatched by the
// eframe UI inside the WebView.

plugins {
    id("com.android.application") version "8.2.2" apply false
    id("org.jetbrains.kotlin.android") version "1.9.22" apply false
}