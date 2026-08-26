//! WebSocket glue: connect to the Observer `/ws` stream and route frames
//! into Leptos signals. DOM-touching code lives here so [`crate::frames`]
//! stays host-testable.

use leptos::*;
use wasm_bindgen::prelude::*;
use web_sys::js_sys::Function;
use web_sys::{MessageEvent, WebSocket};

use crate::frames::{parse_frame, Frame};

/// Open `url`, routing every frame through `on_frame`.
///
/// Reconnect is deliberately out of scope for the scaffold: the backend
/// broadcast channel has no replay yet, so a reconnect would silently miss
/// events; it lands alongside snapshot support on the server.
pub fn connect(url: &str, on_frame: impl Fn(Frame) + 'static) -> Option<WebSocket> {
    let ws = WebSocket::new(url).ok()?;
    let onmessage = Closure::wrap(Box::new(move |e: MessageEvent| {
        // Text frames carry the JSON envelope; binary frames never occur.
        if let Some(text) = e.data().as_string() {
            on_frame(parse_frame(&text));
        }
    }) as Box<dyn FnMut(MessageEvent)>);
    ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref::<Function>()));
    // The socket owns the closure for its lifetime.
    onmessage.forget();
    Some(ws)
}

/// Dashboard state shared between the socket and the view.
#[derive(Default, Clone)]
pub struct Dashboard {
    pub telemetry: Vec<crate::frames::TelemetryRow>,
    pub preflight_total: u64,
    pub preflight_blockers: Vec<String>,
}

impl Dashboard {
    /// Fold one frame into the running dashboard state (capped history).
    pub fn push(&mut self, frame: Frame, history_cap: usize) {
        match frame {
            Frame::Telemetry(row) => {
                self.telemetry.insert(0, row);
                self.telemetry.truncate(history_cap);
            }
            Frame::Preflight(p) => {
                self.preflight_total += 1;
                if p.verdict == "unsupported" {
                    let label = format!("{} ({})", p.op, p.family);
                    if !self.preflight_blockers.contains(&label) {
                        self.preflight_blockers.push(label);
                    }
                }
            }
            Frame::Unknown => {}
        }
    }
}
