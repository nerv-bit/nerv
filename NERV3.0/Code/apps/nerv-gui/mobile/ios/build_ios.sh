#!/usr/bin/env bash
# Build the NERV wallet for iOS (erratum 205).
#
# Pipeline:
#   1. wasm-pack builds the Rust wallet-core + eframe shell into WASM.
#   2. The bundle (`pkg/`) plus `index.html` + `manifest.json` + `sw.js`
#      are copied into `Resources/www/` (referenced by the Xcode
#      project as a folder reference, so the bundle ships inside the
#      .ipa's bundle).
#   3. XcodeGen generates `Nerv.xcodeproj` from `project.yml`.
#   4. xcodebuild assembles the release .app.
#
# Prerequisites (one-time setup on macOS):
#   - Rust toolchain (>=1.85): `rustup default 1.85.0`
#   - Xcode 15 + iOS SDK 17
#   - `cargo install wasm-pack`
#   - `brew install xcodegen`
#
# Code-signing is disabled for local builds (see `project.yml`). For
# TestFlight / App Store submission, set `DEVELOPMENT_TEAM` in
# `project.yml` and re-enable code signing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WALLET_DIR="$SCRIPT_DIR/../.."
WWW_DIR="$SCRIPT_DIR/NervWallet/Resources/www"
APP_RESOURCES_DIR="$SCRIPT_DIR/NervWallet/Resources"

echo "Building NERV Wallet for iOS (WASM → PWA → WKWebView)…"

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

# 2. Stage the assets into the iOS bundle's Resources/www/.
rm -rf "$WWW_DIR"
mkdir -p "$WWW_DIR"
cp index.html  "$WWW_DIR/"
cp manifest.json "$WWW_DIR/" 2>/dev/null || true
cp sw.js      "$WWW_DIR/" 2>/dev/null || true
cp -r pkg     "$WWW_DIR/"

# 3. Generate the Xcode project from `project.yml` if XcodeGen is
# available, or use an existing `Nerv.xcodeproj` if the operator
# already created one in Xcode.
if command -v xcodegen &> /dev/null; then
    echo "Generating Xcode project from project.yml…"
    cd "$SCRIPT_DIR"
    xcodegen generate
else
    echo "XcodeGen not installed — assuming Nerv.xcodeproj already exists."
    echo "(Install via: brew install xcodegen)"
fi

# 4. Build the .app.
cd "$SCRIPT_DIR"
xcodebuild \
    -project Nerv.xcodeproj \
    -scheme Nerv \
    -configuration Release \
    -destination 'generic/platform=iOS' \
    -derivedDataPath build \
    CODE_SIGNING_ALLOWED=NO \
    CODE_SIGN_IDENTITY="" \
    clean build

echo
echo "Done. App bundle: $SCRIPT_DIR/build/Build/Products/Release-iphoneos/Nerv.app"
echo "Run on simulator: xcrun simctl install booted \$(_)"
echo "Run on device:    xcrun devicectl device install app \$(_)"