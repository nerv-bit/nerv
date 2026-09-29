# NERV — Web (erratum 205)

This folder hosts two distinct web deployments:

```
apps/nerv-gui/web/
├── NERV landing page/      <- existing Next.js landing page (lives at
│                              https://nerv-3w4y.vercel.app/)
└── wallet/                 <- the new NERV Wallet web app
                              (mounted at /wallet/ on the same domain)
```

## How the landing page + wallet share one Vercel project

The landing page is a Next.js app; the wallet is a static HTML/JS/CSS
shell hosting the same WASM bundle that the Android + iOS shells
embed. They share a single Vercel project (`nerv-3w4y.vercel.app`):

- `/` → Next.js landing page (the existing marketing site).
- `/wallet/` → the static WASM wallet (the chrome + canvas).

The wallet's static files live in `app/public/wallet/` (Next.js's
static-asset directory) so Vercel serves them automatically at
`/wallet/*`. The build orchestrates this via `vercel.json` +
`build.sh` — see the [wallet README](./wallet/README.md#deploy-to-vercel)
for the full deployment pipeline.

## GitHub repo layout (target)

After the NERV3.0 codebase moves to GitHub at
`https://github.com/nerv-bit/nerv/tree/main/NERV3.0`, the structure is:

```
nerv-bit/nerv/                    <- repo root
├── app/                          <- landing page (Vercel Root Directory)
│   ├── page.tsx
│   ├── layout.tsx
│   ├── vercel.json               <- orchestrates the build
│   ├── package.json
│   └── public/
│       ├── NERV Logo.png         <- landing-page hero image
│       └── wallet/               <- staged at build time
└── NERV3.0/                      <- NERV codebase
    └── apps/nerv-gui/
        ├── Cargo.toml
        ├── rust-toolchain.toml   <- pins 1.85 for wasm-pack
        └── web/wallet/
            ├── index.html
            ├── styles.css
            ├── bridge.js
            ├── manifest.json
            ├── sw.js
            └── assets/nerv_logo.png

build.sh                          <- orchestrator (lives at the repo root)
```

The local workspace structure mirrors this layout — `apps/nerv-gui/`
in the workspace corresponds to `NERV3.0/apps/nerv-gui/` on GitHub.

## What to copy to GitHub when moving

| Local path | GitHub path |
| ---------- | ----------- |
| `apps/nerv-gui/web/NERV landing page/vercel.json` | `app/vercel.json` |
| `apps/nerv-gui/web/NERV landing page/build.sh` | `build.sh` (repo root) |
| `apps/nerv-gui/rust-toolchain.toml` | `NERV3.0/apps/nerv-gui/rust-toolchain.toml` |
| `apps/nerv-gui/web/wallet/*` | `NERV3.0/apps/nerv-gui/web/wallet/*` |

The landing-page `page.tsx` already has the "Open NERV Wallet →" CTA
added — copy it to `app/page.tsx` on GitHub to keep the hero in sync.

## Build pipeline summary

1. **`wasm-pack build --target web --release`** from
   `NERV3.0/apps/nerv-gui/`. `rust-toolchain.toml` pins Rust to 1.85.
2. **Stage the wallet into `app/public/wallet/`** — copies
   `index.html`, `styles.css`, `bridge.js`, `manifest.json`,
   `sw.js`, `assets/`, plus the wasm-pack `pkg/` output.
3. **Vercel runs `next build`** for the landing page.
4. **Output**:
   - Landing page at `/`.
   - Wallet at `/wallet/`.
   - WASM bundle at `/wallet/pkg/nerv_gui_bg.wasm`.

## URL routing (erratum 205)

| Path | Handler |
| ---- | ------- |
| `/` | Next.js landing page (existing) |
| `/wallet` | Rewrite → `/wallet/index.html` (NERV Wallet chrome) |
| `/wallet/` | Rewrite → `/wallet/index.html` |
| `/wallet/{anything}` | Served from `app/public/wallet/` (Next.js static) |
| `/wallet/pkg/{wasm,js}` | Served from `app/public/wallet/pkg/` (COEP/COOP set) |
| `/wallet/sw.js` | Service worker (no cache, Service-Worker-Allowed: /) |

## Known limitations

- **First build is slow.** `wasm-pack` installs once per cold Vercel
  build (~1–2 min) and the WASM compile itself takes 1–2 min. After
  the first build, Vercel's build cache speeds subsequent builds.
- **`framework: "nextjs"` is set explicitly.** This overrides Vercel's
  auto-detection — important because the project root is `app/` but
  the build needs to escape to `../build.sh`. Without the explicit
  framework setting, Vercel might treat the buildCommand output
  differently.
- **rust-toolchain.toml scope.** The local workspace root pins to
  Rust 1.83 (per the user's current preference). The
  `apps/nerv-gui/rust-toolchain.toml` pins to 1.85 specifically for
  the wallet's wasm-pack build, without affecting the rest of the
  NERV3.0 codebase.