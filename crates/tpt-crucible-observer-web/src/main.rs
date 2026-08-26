//! Wasm entry point; `index.html` + trunk drive the build.
//!
//! ```bash
//! cd crates/tpt-crucible-observer-web && trunk serve
//! ```

fn main() {
    tpt_crucible_observer_web::mount();
}
