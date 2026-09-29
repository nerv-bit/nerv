#!/usr/bin/env bash
# Build the NERV wallet for Android (erratum 205).
#
# Pipeline:
#   1. wasm-pack builds the Rust wallet-core + eframe shell into WASM.
#   2. The bundle (`pkg/`) plus `index.html` + `manifest.json` + `sw.js`
#      are copied into `app/src/main/assets/www/`.
#   3. Gradle assembles the release APK, bundling the WASM into the
#      APK's assets so the WebView can load it offline.
#
# Prerequisites (one-time setup):
#   - Rust toolchain (>=1.85): `rustup default 1.85.0`
#   - Android SDK + platform-tools + build-tools 34
#   - JDK 17
#   - `cargo install wasm-pack`
#
# The release keystore is provisioned separately by the testnet
# operator; the default `signingConfig = signingConfigs.debug` line in
# `app/build.gradle.kts` lets the APK build for local testing. Switch
# to a release keystore before publishing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WALLET_DIR="$SCRIPT_DIR/../.."
ASSETS_DIR="$SCRIPT_DIR/app/src/main/assets/www"

echo "Building NERV Wallet for Android (WASM → PWA → WebView)…"

# 1. Build the WASM bundle.
cd "$WALLET_DIR"
rustup target add wasm32-unknown-unknown >/dev/null 2>&1 || true
if ! command -v wasm-pack &> /dev/null; then
    cargo install wasm-pack
fi
wasm-pack build \
    --target web \
    --release \
    --out-dir pkg \
    -- --features "default"

# 2. Assemble the assets.
rm -rf "$ASSETS_DIR"
mkdir -p "$ASSETS_DIR"
cp index.html  "$ASSETS_DIR/"
cp manifest.json "$ASSETS_DIR/" 2>/dev/null || true
cp sw.js      "$ASSETS_DIR/" 2>/dev/null || true
cp -r pkg     "$ASSETS_DIR/"

# 3. Build the APK.
cd "$SCRIPT_DIR"
./gradlew assembleRelease

echo
echo "Done. APK: $SCRIPT_DIR/app/build/outputs/apk/release/app-release.apk"
echo "Install: adb install -r \$(_)"
echo "Run on device: adb shell am start -n org.nerv.wallet/.MainActivity"