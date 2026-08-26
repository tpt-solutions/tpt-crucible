//! # tpt-crucible-observer-web
//!
//! **The Dashboard frontend** (Phase 4): a pure-Rust Wasm (Leptos CSR) view
//! over the Observer WebSocket stream — live tokens/sec, memory bandwidth,
//! thermal drift, and node latency for Alloy swarms, Fusion overlays, and
//! Element arrays (`spec2.txt` §3.5), styled as an industrial-blueprint dark
//! theme.
//!
//! ## Wire contract
//!
//! Frames are the Observer backend's [`ServerEvent`] JSON objects
//! (`{"kind":"telemetry",…}` / `{"kind":"preflight",…}`). This crate declares
//! its own mirror of that schema ([`frames`]) instead of depending on the
//! backend crate: the backend pulls in axum/tokio, which cannot compile for
//! `wasm32-unknown-unknown`. The two sides are pinned together by shared
//! tests upstream; if the wire schema changes, both change.
//!
//! ## Building
//!
//! ```bash
//! cargo install trunk        # once
//! cd crates/tpt-crucible-observer-web
//! trunk serve                # dev server with hot reload
//! trunk build --release      # dist/ is fully static - any file server works
//! ```
//!
//! [`ServerEvent`]: https://docs.rs/tpt-crucible-observer/latest/tpt_crucible_observer/enum.ServerEvent.html

pub mod app;
pub mod frames;
pub mod ws;

use leptos::prelude::*;

/// Mount the dashboard into the DOM (`index.html` provides `<div id="app">`);
/// called from `src/main.rs` when trunk builds the binary.
pub fn mount() {
    console_error_panic_hook::set_once();
    mount_to_body(move || {
        view! { <app::App/> }
    });
}
