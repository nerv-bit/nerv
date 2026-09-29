// sw.js — NERV Wallet service worker (erratum 205).
//
// Cache-first strategy for the WASM bundle + index.html so the
// wallet is usable offline (after the first successful load). The
// service worker is registered from index.html.
//
// Cache versioning: bump `CACHE_VERSION` to force-reload.

const CACHE_VERSION = 'nerv-wallet-v1';
const ASSETS = [
    './',
    './index.html',
    './styles.css',
    './bridge.js',
    './manifest.json',
    './assets/nerv_logo.png',
    './pkg/nerv_gui.js',
    './pkg/nerv_gui_bg.wasm',
    './pkg/snippets/',
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_VERSION).then((cache) => {
            // Pre-cache what we can; the WASM bundle can be large,
            // so we add each item individually and tolerate misses.
            return Promise.all(
                ASSETS.map((url) =>
                    cache.add(url).catch(() => {
                        // Some assets (notably the pkg/ directory
                        // contents) may not exist yet on first
                        // install — they're added to the cache on
                        // first fetch below.
                    })
                )
            );
        })
    );
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) =>
            Promise.all(
                keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k))
            )
        )
    );
    self.clients.claim();
});

self.addEventListener('fetch', (event) => {
    // Only handle GET; let everything else through.
    if (event.request.method !== 'GET') return;

    const url = new URL(event.request.url);
    // Same-origin only.
    if (url.origin !== self.location.origin) return;

    event.respondWith(
        caches.match(event.request).then((cached) => {
            if (cached) return cached;

            return fetch(event.request)
                .then((response) => {
                    // Cache successful, basic responses for next time.
                    if (response && response.ok && response.type === 'basic') {
                        const clone = response.clone();
                        caches.open(CACHE_VERSION).then((cache) => {
                            cache.put(event.request, clone).catch(() => {});
                        });
                    }
                    return response;
                })
                .catch(() => {
                    // Network failed and not in cache — return a
                    // minimal offline shell so the user sees the
                    // wallet brand even on a cold offline boot.
                    if (event.request.mode === 'navigate') {
                        return caches.match('./index.html');
                    }
                });
        })
    );
});