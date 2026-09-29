// bridge.js (erratum 205; web counterpart to the Kotlin + Swift bridges).
//
// JS ⇄ "Native" bridge. On Android the native side is Kotlin
// (NervBridge), on iOS it's Swift (nervBridge message handler).
// On the Web there's no separate native process — the "native"
// chrome is plain HTML/CSS/JS in the same context as the WASM
// bundle, so the bridge collapses into a tiny message bus.
//
// The contract surface is intentionally small — wallet actions
// stay inside the wallet-core state machine (Rust). The bridge
// is for *chrome*-level concerns only: navigation events from
// the bottom nav, sync-state updates, snackbar toasts, haptics
// (via the Vibration API where supported), clipboard, and share.

(function () {
    'use strict';

    // ───────────────────────────────────────────────────────────
    // Constants — the 8 wallet screens (mirror WalletScreen.kt /
    // WalletScreen.swift / Screen::ALL on the desktop).
    // ───────────────────────────────────────────────────────────
    const SCREENS = [
        { id: 'Dashboard', title: 'Dashboard', icon: 'M3 3h7v7H3V3zm0 11h7v7H3v-7zm11-11h7v7h-7V3zm0 11h7v7h-7v-7z' },
        { id: 'Send',      title: 'Send',      icon: 'M5 19L19 5M19 5H9m10 0v10' },
        { id: 'Receive',   title: 'Receive',   icon: 'M19 5L5 19M5 19v-10m0 10h10' },
        { id: 'Claim',     title: 'Claim',     icon: 'M3 7h18v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7zm0 0V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v2' },
        { id: 'Producer',  title: 'Producer',  icon: 'M9 3v2m6-2v2M9 19v2m6-2v2M3 9h2m-2 6h2m14-6h2m-2 6h2M7 7h10v10H7V7z' },
        { id: 'History',   title: 'History',   icon: 'M12 8v4l3 3M3 12a9 9 0 1 0 18 0 9 9 0 0 0-18 0z' },
        { id: 'Settings',  title: 'Settings',  icon: 'M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM19 12c0 .5-.05.98-.13 1.46l2.07 1.62a.5.5 0 0 1 .12.64l-1.96 3.4a.5.5 0 0 1-.6.22l-2.45-.98a7 7 0 0 1-2.54 1.47l-.37 2.6a.5.5 0 0 1-.5.42h-3.92a.5.5 0 0 1-.5-.42l-.37-2.6a7 7 0 0 1-2.54-1.47l-2.45.98a.5.5 0 0 1-.6-.22L2.13 15.72a.5.5 0 0 1 .12-.64l2.07-1.62A7 7 0 0 1 4.2 12c0-.5.05-.98.13-1.46L2.26 8.92a.5.5 0 0 1-.12-.64L4.1 4.88a.5.5 0 0 1 .6-.22l2.45.98a7 7 0 0 1 2.54-1.47l.37-2.6A.5.5 0 0 1 10.56 1h3.92c.25 0 .46.18.5.42l.37 2.6a7 7 0 0 1 2.54 1.47l2.45-.98a.5.5 0 0 1 .6.22l1.96 3.4a.5.5 0 0 1-.12.64l-2.07 1.62c.08.48.13.96.13 1.46z' },
        { id: 'Help',      title: 'Help',      icon: 'M9.5 9a2.5 2.5 0 1 1 4.5 1.5c-1 .83-2 1.5-2 2.5v.5M12 17h.01M12 22a10 10 0 1 1 0-20 10 10 0 0 1 0 20z' },
    ];

    // ───────────────────────────────────────────────────────────
    // Active-screen state (mirrored across top bar, bottom nav,
    // and the WASM canvas).
    // ───────────────────────────────────────────────────────────
    let activeScreen = (function () {
        try {
            const saved = parseInt(localStorage.getItem('nerv.lastScreenIndex') || '0', 10);
            return SCREENS[saved] ? SCREENS[saved].id : SCREENS[0].id;
        } catch (_) {
            return SCREENS[0].id;
        }
    })();

    function setActiveScreen(id, persist) {
        if (!SCREENS.some((s) => s.id === id)) return;
        activeScreen = id;
        if (persist) {
            try {
                const idx = SCREENS.findIndex((s) => s.id === id);
                localStorage.setItem('nerv.lastScreenIndex', String(idx));
            } catch (_) { /* storage may be unavailable */ }
        }
        renderBottomNav();
    }

    // ───────────────────────────────────────────────────────────
    // Bottom-nav rendering.
    // ───────────────────────────────────────────────────────────
    function renderBottomNav() {
        const container = document.getElementById('bottomnav-items');
        if (!container) return;
        container.innerHTML = '';
        SCREENS.forEach((screen) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className =
                'bottomnav-item' +
                (screen.id === activeScreen ? ' bottomnav-item--active' : '');
            btn.setAttribute('aria-label', screen.title);
            btn.setAttribute('aria-current', screen.id === activeScreen ? 'page' : 'false');
            btn.innerHTML =
                '<span class="bottomnav-icon-wrap">' +
                    '<svg class="bottomnav-icon" viewBox="0 0 24 24" fill="none" ' +
                        'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
                        'stroke-linejoin="round">' +
                        '<path d="' + screen.icon + '" />' +
                    '</svg>' +
                '</span>' +
                '<span class="bottomnav-label">' + screen.title + '</span>';
            btn.addEventListener('click', () => {
                setActiveScreen(screen.id, true);
                // Push the navigation into the WASM canvas so the
                // eframe UI dispatches `WalletAction::Navigate`.
                postNativeEvent('nav', { to: screen.id });
                if (typeof navigator !== 'undefined' && navigator.vibrate) {
                    navigator.vibrate(15);
                }
            });
            container.appendChild(btn);
        });
    }

    // ───────────────────────────────────────────────────────────
    // Sync badge.
    // ───────────────────────────────────────────────────────────
    function setSyncState(state) {
        const badge = document.getElementById('sync-badge');
        if (!badge) return;
        const labels = { synced: 'Synced', syncing: 'Syncing', offline: 'Offline', locked: 'Locked' };
        badge.className = 'sync-badge sync-badge--' + state;
        const labelEl = badge.querySelector('.sync-label');
        if (labelEl) labelEl.textContent = labels[state] || 'Synced';
    }

    // ───────────────────────────────────────────────────────────
    // Snackbar host.
    // ───────────────────────────────────────────────────────────
    let snackbarTimer = null;
    function showSnackbar(message, level) {
        const el = document.getElementById('snackbar');
        if (!el) return;
        el.textContent = message;
        el.className = 'snackbar';
        if (level && level !== 'info') el.classList.add('snackbar--' + level);
        el.classList.add('snackbar--visible');
        if (snackbarTimer) clearTimeout(snackbarTimer);
        snackbarTimer = setTimeout(() => {
            el.classList.remove('snackbar--visible');
        }, 4000);
    }

    /** Infer the level from the eframe UI's notification prefix. */
    function inferLevel(message) {
        if (!message) return 'info';
        if (message.indexOf('✓') === 0) return 'success';
        if (message.indexOf('⚠') === 0) return 'warning';
        if (message.indexOf('✗') === 0) return 'error';
        return 'info';
    }

    // ───────────────────────────────────────────────────────────
    // Haptics (Vibration API).
    // ───────────────────────────────────────────────────────────
    function haptic(kind) {
        if (typeof navigator === 'undefined' || !navigator.vibrate) return;
        switch (kind) {
            case 'success': navigator.vibrate([10, 30, 20]); break;
            case 'warning': navigator.vibrate(40); break;
            case 'error':   navigator.vibrate([60, 40, 60]); break;
            default:        navigator.vibrate(15);
        }
    }

    // ───────────────────────────────────────────────────────────
    // Clipboard + share helpers (mirrors Android/iOS bridge methods).
    // ───────────────────────────────────────────────────────────
    function copyToClipboard(text) {
        if (typeof navigator !== 'undefined' && navigator.clipboard) {
            navigator.clipboard.writeText(text).catch(() => {});
        }
    }

    function shareText(text, title) {
        if (typeof navigator !== 'undefined' && navigator.share) {
            navigator.share({ title: title || 'NERV', text: text }).catch(() => {});
        } else {
            copyToClipboard(text);
        }
    }

    // ───────────────────────────────────────────────────────────
    // Event routing from the WASM canvas.
    //
    // The eframe UI calls these globals:
    //   nervBridgeFromNative('ready', '')
    //   nervBridgeFromNative('nav', '{"to":"Send"}')
    //   nervBridgeFromNative('sync', 'syncing')
    //   nervBridgeFromNative('toast', 'Sent 1 NERV')
    //   nervBridgeFromNative('lock', '')
    //   nervBridgeFromNative('haptic', 'success')
    // ───────────────────────────────────────────────────────────
    window.nervBridgeFromNative = function (type, payload) {
        switch (type) {
            case 'ready':
                // The eframe UI is up. Activate the screen the
                // bottom-nav was on so the Web wallet starts on the
                // same tab the user last used.
                postNativeEvent('nav', { to: activeScreen });
                break;
            case 'nav':
                try {
                    const parsed = JSON.parse(payload);
                    if (parsed && parsed.to) setActiveScreen(parsed.to, false);
                } catch (_) { /* ignore malformed payload */ }
                break;
            case 'sync':
                setSyncState(String(payload).toLowerCase());
                break;
            case 'toast':
                showSnackbar(String(payload), inferLevel(String(payload)));
                break;
            case 'lock':
                // Triggered by the eframe UI when the user pressed
                // the in-app lock button. The top-bar lock button
                // is handled directly in index.html; this is a
                // redundant safety net.
                break;
            case 'haptic':
                haptic(String(payload));
                break;
            default:
                // Unknown event — ignore.
                break;
        }
    };

    // ───────────────────────────────────────────────────────────
    // Outgoing helper: push an event into the WASM canvas.
    //
    // The eframe UI is expected to define a JS handler
    // (typically in a `bind_bridge` shim) that dispatches the
    // matching `WalletAction::*`. Without the shim, the chrome
    // still updates; the WASM canvas just doesn't react.
    // ───────────────────────────────────────────────────────────
    function postNativeEvent(type, payload) {
        const payloadStr = (typeof payload === 'string')
            ? payload
            : JSON.stringify(payload || {});
        // Custom event so a shim can listen without globals.
        window.dispatchEvent(new CustomEvent('nerv:native-event', {
            detail: { type: type, payload: payloadStr },
        }));
    }

    // ───────────────────────────────────────────────────────────
    // Bridge surface — exposed for the eframe UI + index.html.
    // ───────────────────────────────────────────────────────────
    window.NervBridge = {
        activeScreen: function () { return activeScreen; },
        setActiveScreen: setActiveScreen,
        postNativeEvent: postNativeEvent,
        haptic: haptic,
        copyToClipboard: copyToClipboard,
        shareText: shareText,
        showSnackbar: showSnackbar,
        setSyncState: setSyncState,
        SCREENS: SCREENS,
    };

    // Initial paint.
    renderBottomNav();
    setSyncState('synced');

    // ───────────────────────────────────────────────────────────
    // Chrome → eframe dispatch (erratum 205).
    //
    // The native chrome (top bar / bottom nav / lock button) pushes
    // events into the WASM canvas via `nerv:native-event`
    // CustomEvents. For `nav` and `lock` events, we forward into the
    // eframe UI by calling the `wasm_bindgen`-exported functions
    // exposed on `window` by `index.html` (`window.nerv_navigate`,
    // `window.nerv_lock`). For `sync` / `toast` / `haptic` events,
    // the chrome is the only consumer — no eframe dispatch needed.
    //
    // The eframe UI's `update()` drains the matching queues on the
    // next frame and dispatches `WalletAction::Navigate(screen)` /
    // `WalletAction::Lock` through `nerv_wallet_core::update()`.
    // ───────────────────────────────────────────────────────────
    window.addEventListener('nerv:native-event', function (ev) {
        const detail = ev && ev.detail;
        if (!detail) return;
        const type = detail.type;
        const payload = detail.payload || '';

        if (type === 'nav') {
            let target = null;
            try {
                const parsed = JSON.parse(payload);
                if (parsed && typeof parsed.to === 'string') {
                    target = parsed.to;
                }
            } catch (_) { /* malformed payload — ignore */ }
            if (target && typeof window.nerv_navigate === 'function') {
                window.nerv_navigate(target);
            }
        } else if (type === 'lock') {
            if (typeof window.nerv_lock === 'function') {
                window.nerv_lock();
            }
        }
        // sync / toast / haptic are chrome-only — no eframe dispatch.
    });

    // Push the active screen into the eframe UI on cold-start so
    // the WASM canvas renders the right tab on the first frame.
    if (typeof window.nerv_set_active_screen === 'function') {
        window.nerv_set_active_screen(activeScreen);
    }
})();