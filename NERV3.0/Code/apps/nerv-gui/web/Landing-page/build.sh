#!/usr/bin/env bash
# build.sh — NERV monorepo build orchestrator (erratum 205).
#
# This script lives at the REPO ROOT on GitHub (`nerv-bit/nerv/`,
# alongside `app/` and `NERV3.0/`). Vercel invokes it via the
# `buildCommand` declared in `app/vercel.json`.
#
# Pipeline:
#   1. Build the WASM bundle from `NERV3.0/apps/nerv-gui/`.
#   2. Stage the wallet's chrome + WASM bundle into
#      `app/public/wallet/` so Next.js's static-asset serving
#      picks it up automatically at `/wallet/*`.
#   3. Vercel then runs `next build` automatically (because
#      `framework: "nextjs"` is set in `app/vercel.json`).
#
# Prereqs on Vercel:
#   - Rust toolchain (auto-installed via the `rust-toolchain.toml`
#     at `NERV3.0/apps/nerv-gui/rust-toolchain.toml`).
#   - `cargo install wasm-pack` (run once during the first build —
#     cached for subsequent builds).
#   - `app/node_modules/` is installed by Vercel's standard
#     Next.js framework handler (after this buildCommand exits).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
APP_DIR="$SCRIPT_DIR/app"
WALLET_SRC="$SCRIPT_DIR/NERV3.0/apps/nerv-gui"
WALLET_PUBLIC="$APP_DIR/public/wallet"

echo "╔════════════════════════════════════════════════════════╗"
echo "║  NERV monorepo build (landing page + wallet)         ║"
echo "╚════════════════════════════════════════════════════════╝"

# ───────────────────────────────────────────────────────────────
# 1. Build the WASM bundle from NERV3.0/apps/nerv-gui.
# ───────────────────────────────────────────────────────────────
echo
echo "[1/3] Building WASM wallet from NERV3.0/apps/nerv-gui/..."
echo "      workspace root: $SCRIPT_DIR"
echo "      wallet source:  $WALLET_SRC"
echo "      wallet target:  $WALLET_PUBLIC"

if [ ! -d "$WALLET_SRC" ]; then
    echo "ERROR: wallet source not found at $WALLET_SRC" >&2
    echo "       The NERV3.0 codebase must live at NERV3.0/ in the repo root." >&2
    exit 1
fi

cd "$WALLET_SRC"

# Ensure the WASM target is installed.
rustup target add wasm32-unknown-unknown >/dev/null 2>&1 || true

# Ensure wasm-pack is installed.
if ! command -v wasm-pack >/dev/null 2>&1; then
    echo "      installing wasm-pack (one-time)..."
    cargo install wasm-pack
fi

# Build the bundle.
wasm-pack build \
    --target web \
    --release \
    --out-dir pkg \
    -- --features "default"

# ───────────────────────────────────────────────────────────────
# 2. Stage the wallet's chrome + WASM into app/public/wallet/.
# ───────────────────────────────────────────────────────────────
echo
echo "[2/3] Staging wallet into $WALLET_PUBLIC/"

rm -rf "$WALLET_PUBLIC"
mkdir -p "$WALLET_PUBLIC"

cp web/wallet/index.html      "$WALLET_PUBLIC/"
cp web/wallet/styles.css      "$WALLET_PUBLIC/"
cp web/wallet/bridge.js       "$WALLET_PUBLIC/"
cp web/wallet/manifest.json   "$WALLET_PUBLIC/"
cp web/wallet/sw.js           "$WALLET_PUBLIC/"
cp -r web/wallet/assets       "$WALLET_PUBLIC/"
cp -r pkg                     "$WALLET_PUBLIC/"

# ───────────────────────────────────────────────────────────────
# 3. Vercel's Next.js framework handler runs `next build` next.
# ───────────────────────────────────────────────────────────────
echo
echo "[3/3] Vercel will now run 'next build' for the landing page..."
echo
echo "✓ Build staging complete. Handing off to Vercel's Next.js framework handler."