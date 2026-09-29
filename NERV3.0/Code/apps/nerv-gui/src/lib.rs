//! The NERV wallet GUI (erratum 202–203): one Rust codebase, all platforms.
//!
//! The `NervApp` struct implements `eframe::App` and wraps the
//! `nerv-wallet-core` state machine. The desktop entry calls
//! `eframe::run_native`; the web entry calls `eframe::WebRunner`.
//!
//! WASM ⇄ JS bridge (erratum 205): on the web/PWA surface, the
//! `nerv_navigate` / `nerv_lock` / `nerv_active_screen` /
//! `nerv_set_active_screen` functions are `#[wasm_bindgen]`-exported
//! so the JS shim can dispatch `WalletAction`s into the wallet-core
//! state machine from outside the eframe loop. The `update()` method
//! drains the queues on each frame; see `take_pending_nav` /
//! `take_pending_lock` below.
//!
//! The matching JS shim lives in
//! `apps/nerv-gui/web/wallet/bridge.js` — it listens for the
//! `nerv:native-event` CustomEvent (dispatched by the native chrome's
//! bottom-nav / sync / etc.) and calls the wasm-exported functions
//! below. The eframe bundle's `update()` drains the queues on the
//! next frame.

#![forbid(unsafe_code)]

pub mod app;
pub mod theme;

pub use app::NervApp;

// ---------------------------------------------------------------------------
// WASM entry point (web/PWA)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
pub async fn start_web(
    canvas: web_sys::HtmlCanvasElement,
) -> Result<(), eframe::wasm_bindgen::JsValue> {
    let options = eframe::WebOptions::default();
    eframe::WebRunner::new()
        .start(canvas, options, Box::new(|cc| Box::new(NervApp::new(cc))))
        .await
}

// ---------------------------------------------------------------------------
// WASM ⇄ JS bridge (erratum 205)
// ---------------------------------------------------------------------------
//
// The native chrome (Web bottom-nav, Android Kotlin bottom-nav, iOS SwiftUI
// bottom-nav) pushes events into the eframe bundle via:
//   - Web:    `window.dispatchEvent(new CustomEvent('nerv:native-event',
//                 {detail:{type:'nav', payload:'{"to":"Send"}'}}))`
//   - Android/iOS: equivalent postMessage variants defined in the bridge.
//
// The JS shim inside the eframe bundle listens for those events and calls
// the `#[wasm_bindgen]`-exported functions below to queue a corresponding
// wallet-core action. `NervApp::update()` drains the queues on the next
// frame and dispatches the actions through `update(&mut state, action)`.
//
// This indirection (queue + drain on next frame) avoids borrowing the
// `NervApp` state from JS while eframe is mutating it.

#[cfg(target_arch = "wasm32")]
mod bridge {
    use std::cell::RefCell;
    use wasm_bindgen::prelude::*;

    thread_local! {
        /// Screen name queued by the JS shim (consumed by
        /// `NervApp::update()` and dispatched as
        /// `WalletAction::Navigate(screen)`).
        static PENDING_NAV: RefCell<Option<String>> = const { RefCell::new(None) };
        /// `WalletAction::Lock` requested by the JS shim. Coalesces to
        /// a single boolean — multiple `lock` events in one frame are
        /// idempotent.
        static PENDING_LOCK: RefCell<bool> = const { RefCell::new(false) };
        /// The most-recent screen the chrome asked us to navigate to.
        /// Persisted into the JS-side localStorage by the JS shim so
        /// cold-starts restore the right tab.
        static ACTIVE_SCREEN: RefCell<String> = const { RefCell::new(String::from("Dashboard")) };
    }

    /// JS → Rust: queue a `WalletAction::Navigate(screen)`. The
    /// screen name must match one of the variants of
    /// `nerv_wallet_core::Screen` (the JS shim validates this;
    /// unknown names are ignored at dispatch time).
    #[wasm_bindgen]
    pub fn nerv_navigate(screen: &str) {
        if !screen.is_empty() {
            PENDING_NAV.with(|cell| {
                *cell.borrow_mut() = Some(screen.to_string());
            });
            ACTIVE_SCREEN.with(|cell| {
                *cell.borrow_mut() = screen.to_string();
            });
        }
    }

    /// JS → Rust: queue a `WalletAction::Lock`. The state machine
    /// wipes the seed + draft + history on lock.
    #[wasm_bindgen]
    pub fn nerv_lock() {
        PENDING_LOCK.with(|cell| {
            *cell.borrow_mut() = true;
        });
    }

    /// JS → Rust: the eframe UI's `update()` drains the nav queue.
    /// Returns `Some(screen)` exactly once per `nerv_navigate` call.
    pub(crate) fn take_pending_nav() -> Option<String> {
        PENDING_NAV.with(|cell| cell.borrow_mut().take())
    }

    /// JS → Rust: the eframe UI's `update()` drains the lock flag.
    /// Returns `true` exactly once per `nerv_lock` call.
    pub(crate) fn take_pending_lock() -> bool {
        PENDING_LOCK.with(|cell| {
            let mut flag = cell.borrow_mut();
            let was = *flag;
            *flag = false;
            was
        })
    }

    /// JS → Rust: read the most-recent active screen (used by
    /// `NervApp::update()` to push the active screen into the eframe
    /// UI before the first render so the canvas starts on the right
    /// tab after a cold-start).
    pub(crate) fn active_screen() -> String {
        ACTIVE_SCREEN.with(|cell| cell.borrow().clone())
    }

    /// JS → Rust: explicitly set the active screen (e.g., when the
    /// eframe UI itself navigates and the chrome should follow).
    /// The chrome's JS shim reads this back via `nerv_active_screen`.
    #[wasm_bindgen]
    pub fn nerv_set_active_screen(screen: &str) {
        if !screen.is_empty() {
            ACTIVE_SCREEN.with(|cell| {
                *cell.borrow_mut() = screen.to_string();
            });
        }
    }

    /// JS → Rust: the chrome's JS shim calls this to fetch the
    /// current active screen on cold-start.
    #[wasm_bindgen]
    pub fn nerv_active_screen() -> String {
        ACTIVE_SCREEN.with(|cell| cell.borrow().clone())
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use bridge::{
    active_screen, nerv_active_screen, nerv_lock, nerv_navigate,
    nerv_set_active_screen, take_pending_lock, take_pending_nav,
};