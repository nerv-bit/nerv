# NERV Wallet — Web (PWA)

Web counterpart to the Android + iOS shells. The same WASM-built
eframe UI that runs inside the mobile shells runs here inside a
plain HTML5 canvas — wrapped in a native HTML/CSS chrome that
mirrors the mobile experience:

```
Splash ─▶ LockScreen ─▶ Wallet (top bar + WASM canvas + bottom nav)
```

## Architecture

The web wallet is the **same** `apps/nerv-gui` Rust crate compiled
to `wasm32-unknown-unknown` and hosted on Vercel. There is **no
separate web wallet codebase** — the mobile shells (Android + iOS)
also embed the same WASM bundle inside their `WebView` / `WKWebView`
hosts.

The web wallet's `index.html` adds the native chrome on top:

- **Splash** — NERV logo + radial gold glow + breathing pulse +
  brand wordmark.
- **Lock screen** — gold-glow NERV logo + large unlock button (Web
  builds use a tap-to-unlock gate; production biometric checks go
  through the platform shells).
- **Top bar** — small NERV logo + brand wordmark + sync badge
  (Synced / Syncing / Offline / Locked) + lock button.
- **Main canvas** — the WASM eframe UI.
- **Bottom nav** — 8 tabs mirroring the desktop `Screen` enum
  (Dashboard, Send, Receive, Claim, Producer, History, Settings,
  Help) with inline SVG icons.
- **Snackbar host** — level-coloured, auto-dismisses after 4 s.

## Build

```bash
cd apps/nerv-gui
./build_web.sh
```

Prereqs (one-time):

```bash
rustup default 1.85.0
cargo install wasm-pack
```

The output lands at:

```
apps/nerv-gui/web/wallet/dist/
├── index.html
├── styles.css
├── bridge.js
├── manifest.json
├── sw.js
├── assets/
│   └── nerv_logo.png
└── pkg/                # wasm-pack output
    ├── nerv_gui.js
    └── nerv_gui_bg.wasm
```

## Deploy to Vercel

The web wallet is designed to live **on the same Vercel project as
the landing page** so the entire NERV experience is reachable from
`https://nerv-3w4y.vercel.app/`. The wallet is mounted at
`/wallet/` via Vercel rewrites.

### GitHub repo layout (target)

After the NERV3.0 codebase moves to GitHub, the structure is:

```
nerv-bit/nerv/                    <- repo root
├── app/                          <- landing page (Next.js)
│   ├── page.tsx
│   ├── layout.tsx
│   ├── vercel.json               <- lives here (orchestrates the build)
│   ├── package.json
│   ├── public/
│   │   ├── NERV Logo.png         <- landing-page hero image
│   │   └── wallet/               <- staged here at build time
│   └── ...
├── NERV3.0/                      <- the NERV codebase (after the move)
│   ├── Cargo.toml
│   ├── apps/nerv-gui/
│   │   ├── Cargo.toml
│   │   ├── rust-toolchain.toml   <- pins to 1.85 for wasm-pack
│   │   └── web/wallet/
│   │       ├── index.html
│   │       ├── styles.css
│   │       ├── bridge.js
│   │       ├── manifest.json
│   │       ├── sw.js
│   │       └── assets/nerv_logo.png
│   └── ...
└── build.sh                      <- orchestrator (runs from buildCommand)
```

### How the build works (erratum 205)

Vercel's `app/vercel.json` declares:

```json
{
  "buildCommand": "bash ../build.sh",
  "framework": "nextjs",
  "outputDirectory": ".next",
  "rewrites": [
    { "source": "/wallet",  "destination": "/wallet/index.html" },
    { "source": "/wallet/", "destination": "/wallet/index.html" }
  ]
}
```

`build.sh` (at the repo root) does:

1. **`wasm-pack build --target web --release`** from
   `NERV3.0/apps/nerv-gui/`. The `rust-toolchain.toml` next to
   `Cargo.toml` pins Rust to 1.85 (required for `wasm-pack` with
   `edition2024`).
2. **Stage the wallet into `app/public/wallet/`** — copies
   `index.html`, `styles.css`, `bridge.js`, `manifest.json`,
   `sw.js`, `assets/`, plus the wasm-pack `pkg/` output.
3. **Hand off to Vercel's Next.js handler**, which runs
   `npm install` + `next build` and produces `.next/`.

The wallet is then reachable at:

- `https://nerv-3w4y.vercel.app/wallet/` — the full WASM-backed
  web wallet.
- `https://nerv-3w4y.vercel.app/wallet/styles.css` — wallet assets
  served directly by Next.js's static-asset handler.
- `https://nerv-3w4y.vercel.app/wallet/pkg/nerv_gui_bg.wasm` —
  the WASM bundle.

The landing-page hero CTA already links to `/wallet/`:

```tsx
<a href="/wallet/" className="...">Open NERV Wallet →</a>
```

### What to copy to GitHub

The local workspace has these files at the equivalent locations;
the user copies them into the GitHub repo after the move:

| Local path | GitHub path |
| ---------- | ----------- |
| `apps/nerv-gui/web/NERV landing page/vercel.json` | `app/vercel.json` |
| `apps/nerv-gui/web/NERV landing page/build.sh` | `build.sh` (repo root) |
| `apps/nerv-gui/rust-toolchain.toml` | `NERV3.0/apps/nerv-gui/rust-toolchain.toml` |
| `apps/nerv-gui/web/wallet/index.html` | `NERV3.0/apps/nerv-gui/web/wallet/index.html` |
| `apps/nerv-gui/web/wallet/styles.css` | `NERV3.0/apps/nerv-gui/web/wallet/styles.css` |
| `apps/nerv-gui/web/wallet/bridge.js` | `NERV3.0/apps/nerv-gui/web/wallet/bridge.js` |
| `apps/nerv-gui/web/wallet/manifest.json` | `NERV3.0/apps/nerv-gui/web/wallet/manifest.json` |
| `apps/nerv-gui/web/wallet/sw.js` | `NERV3.0/apps/nerv-gui/web/wallet/sw.js` |
| `apps/nerv-gui/web/wallet/assets/nerv_logo.png` | `NERV3.0/apps/nerv-gui/web/wallet/assets/nerv_logo.png` |
| `apps/nerv-gui/web/NERV landing page/page.tsx` (updated) | `app/page.tsx` |

### Standalone deploy (alternative)

If you ever want to deploy the wallet to a separate Vercel project
(e.g., a staging environment), the wallet directory is
self-contained:

```bash
cd apps/nerv-gui
./build_web.sh
# Output: web/wallet/dist/

cd web/wallet
vercel --prod
```

The standalone deploy uses `vercel.json` (security headers,
COEP/COOP, SW cache, content-type overrides) already in this
folder.

## Project layout

```
apps/nerv-gui/web/wallet/
├── index.html              # Entry point with the chrome
├── styles.css              # NERV palette + chrome styles
├── bridge.js               # JS bridge (mirror of Kotlin/Swift)
├── manifest.json           # PWA manifest (installable, dark theme)
├── sw.js                   # Service worker (offline-first cache)
├── assets/
│   └── nerv_logo.png       # The supplied NERV logo
├── pkg/                    # wasm-pack output (built by build_web.sh)
├── vercel.json             # Vercel deployment config
└── README.md               # This file
```

## JS bridge contract (mirrors Android + iOS)

### Eframe → chrome (WASM → DOM)

The eframe UI inside the canvas calls native APIs via
`nervBridgeFromNative(type, payload)`:

| JS call | Native behaviour |
| ------- | ---------------- |
| `nervBridgeFromNative('ready', '')` | Hides the lock screen; pushes the active tab back to the canvas |
| `nervBridgeFromNative('nav', '{"to":"Send"}')` | Updates native bottom-nav active state |
| `nervBridgeFromNative('sync', 'syncing')` | Updates the top-bar sync badge |
| `nervBridgeFromNative('toast', msg)` | Shows a snackbar (level inferred from prefix: `✓`/`⚠`/`✗`) |
| `nervBridgeFromNative('lock', '')` | Returns to lock screen |
| `nervBridgeFromNative('haptic', 'success'\|'warning'\|'error'\|'tap')` | Fires the Vibration API |

### Chrome → eframe (DOM → WASM)

The native chrome dispatches `nerv:native-event` CustomEvents. For
`nav` and `lock` events, `bridge.js` forwards into the eframe UI
via the wasm-bindgen exports (defined in `apps/nerv-gui/src/lib.rs`):

| Event | Bridge JS shim calls | Effect on eframe UI |
| ----- | -------------------- | ------------------- |
| `nerv:native-event` with `type:'nav', payload:'{"to":"Send"}'` | `window.nerv_navigate('Send')` | `WalletAction::Navigate(Screen::Send)` dispatched next frame |
| `nerv:native-event` with `type:'lock', payload:''` | `window.nerv_lock()` | `WalletAction::Lock` dispatched next frame |

The bridge exports are exposed on `window` by `index.html`:

```js
import init, { nerv_navigate, nerv_lock, nerv_active_screen, nerv_set_active_screen } from './pkg/nerv_gui.js';

window.nerv_navigate = nerv_navigate;
window.nerv_lock = nerv_lock;
window.nerv_active_screen = nerv_active_screen;
window.nerv_set_active_screen = nerv_set_active_screen;
```

The Rust side stores the screen name / lock flag in a thread-local
queue (`bridge::PENDING_NAV`, `bridge::PENDING_LOCK`); the eframe
UI's `update()` drains the queues on the next frame and dispatches
the matching `WalletAction`. This indirection (queue + drain on next
frame) avoids borrowing `NervApp`'s state from JS while eframe is
mutating it.

### Lower-level helpers (chrome → WASM)

| Call | Effect |
| ---- | ------ |
| `NervBridge.postNativeEvent('nav', {to:'Send'})` | Bottom-nav tab updates; eframe canvas receives the matching `nerv:native-event` |
| `NervBridge.haptic('success')` | Vibration API fires |
| `NervBridge.copyToClipboard(text)` / `shareText(text, title)` | Clipboard API / Web Share API |

### Update path summary

```
bottom-nav tap (DOM)
       │
       ▼
NervBridge.setActiveScreen('Send')            ← updates chrome state
       │
       ▼
postNativeEvent('nav', {to:'Send'})           ← fires CustomEvent
       │
       ▼
bridge.js listener (nerv:native-event)
       │
       ▼
window.nerv_navigate('Send')                  ← wasm-bindgen export
       │
       ▼
PENDING_NAV queue (Rust thread-local)
       │
       ▼
NervApp::update() drains queue                ← next frame
       │
       ▼
self.dispatch(WalletAction::Navigate(Send))   ← state machine
       │
       ▼
state.screen = Screen::Send                   ← next render
```

## Security notes

- **No wallet keys touch this layer.** The seed is held inside the
  WASM canvas's IndexedDB; it's wiped on `WalletAction::Lock`.
- **`Cross-Origin-Embedder-Policy: require-corp`** on the WASM
  bundle enables SharedArrayBuffer + threading in the WASM
  runtime (where available).
- **Referrer-Policy: no-referrer** prevents the wallet's URL from
  leaking through outbound links.
- **Permissions-Policy** disables camera / microphone /
  geolocation; the wallet doesn't need them.
- **Manifest sets `theme_color` + `background_color`** so the
  installed PWA matches the chrome.

## Known limitations

- The Vibration API is best-effort — iOS Safari ignores it
  entirely; Chrome on Android supports it but only for short
  pulses.
- The Web Share API requires HTTPS; clipboard falls back
  gracefully if it's unavailable.
- Service-worker cache is invalidated when `CACHE_VERSION` in
  `sw.js` is bumped. Bump it on every release that touches
  `index.html` / `bridge.js` / `styles.css`.
- The eframe ⇄ JS shim (`nerv:native-event` handler inside the
  WASM bundle) is still TODO. The chrome updates without it; the
  WASM canvas won't react to bottom-nav taps until it's in place.