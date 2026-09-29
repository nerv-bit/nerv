#!/usr/bin/env bash
set -euo pipefail

# Build the NERV Wallet web bundle (erratum 205).
#
# Pipeline:
#   1. wasm-pack builds the Rust wallet-core + eframe shell into WASM.
#   2. The bundle (`pkg/`) plus the web wallet chrome
#      (`index.html`, `styles.css`, `bridge.js`, `manifest.json`,
#      `sw.js`, `assets/`) are copied into
#      `web/wallet/dist/` for `vercel deploy` to consume.
#
# Output: `web/wallet/dist/` — a deployable Vercel-ready directory.
#
# Prerequisites (one-time):
#   - Rust toolchain (>=1.85): `rustup default 1.85.0`
#   - `cargo install wasm-pack`

TARGET_DIR="web/wallet/dist"
PKG_DIR="pkg"

echo "Building NERV Wallet for web (WASM)…"

# Install the WASM target if not present.
rustup target add wasm32-unknown-unknown >/dev/null 2>&1 || true

# Install wasm-pack if not present.
if ! command -v wasm-pack &> /dev/null; then
    cargo install wasm-pack
fi

# Build.
wasm-pack build \
    --target web \
    --release \
    --out-dir "$PKG_DIR" \
    -- --features "default"

# Assemble the dist directory.
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"
cp web/wallet/index.html      "$TARGET_DIR/"
cp web/wallet/styles.css      "$TARGET_DIR/"
cp web/wallet/bridge.js       "$TARGET_DIR/"
cp web/wallet/manifest.json   "$TARGET_DIR/"
cp web/wallet/sw.js           "$TARGET_DIR/"
cp -r web/wallet/assets       "$TARGET_DIR/"
cp -r "$PKG_DIR"              "$TARGET_DIR/"

echo
echo "Done. Deploy with:"
echo "  cd web/wallet && vercel --prod"
echo "Or serve locally with:"
echo "  cd $TARGET_DIR && python3 -m http.server 8080"
echo "Then open: http://localhost:8080"