// NERV Wallet service worker (PWA offline support).
const CACHE_NAME = 'nerv-wallet-v1';
const ASSETS = [
    '/',
    '/index.html',
    '/manifest.json',
    '/pkg/nerv_gui.js',
    '/pkg/nerv_gui_bg.wasm',
];

self.addEventListener('install', (e) => {
    e.waitUntil(
        caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
    );
});

self.addEventListener('fetch', (e) => {
    e.respondWith(
        caches.match(e.request).then((cached) => {
            return cached || fetch(e.request);
        })
    );
});

self.addEventListener('activate', (e) => {
    e.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(
                keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
            );
        })
    );
});
