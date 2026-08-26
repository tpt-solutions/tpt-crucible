//! The live dashboard: telemetry table, pre-flight blocker banner, and the
//! WebSocket feed wiring. Styling is plain CSS (`assets/style.css`) in the
//! industrial-blueprint dark theme; Tailwind-via-build-step stays an option
//! tracked in todo.md.

use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::frames::{Frame, TelemetryRow};
use crate::ws::{self, Dashboard};

/// Default Observer address; overridable via `?obs=host:port` later.
const DEFAULT_OBSERVER_WS: &str = "ws://127.0.0.1:8787/ws";

/// History length for the live telemetry table (rows are newest-first).
const HISTORY_CAP: usize = 50;

#[component]
pub fn App() -> impl IntoView {
    let (connected, set_connected) = signal(false);
    let (dashboard, set_dashboard) = signal(Dashboard::default());

    // Connect once on mount; every frame folds into dashboard state.
    spawn_local(async move {
        let socket = ws::connect(DEFAULT_OBSERVER_WS, move |frame: Frame| {
            set_dashboard.update(|d| d.push(frame, HISTORY_CAP));
            set_connected.set(true);
        });
        if socket.is_none() {
            set_connected.set(false);
        }
    });

    // Newest-first telemetry rows rendered straight from state (a plain map
    // keeps us independent of <For> API churn across Leptos versions).
    let rows = move || {
        dashboard
            .get()
            .telemetry
            .iter()
            .map(|row: &TelemetryRow| {
                view! {
                    <tr>
                        <td>{row.node_id.clone()}</td>
                        <td>{row.hw_type.clone()}</td>
                        <td>{row.tokens_per_sec.map(|v| format!("{v:.1}"))}</td>
                        <td>{row.mem_bandwidth_gbps.map(|v| format!("{v:.1}"))}</td>
                        <td>{row.thermal_drift_c.map(|v| format!("{v:+.2}"))}</td>
                        <td>{row.latency_ms.map(|v| format!("{v:.2}"))}</td>
                    </tr>
                }
            })
            .collect::<Vec<_>>()
    };

    let blockers = move || {
        dashboard
            .get()
            .preflight_blockers
            .iter()
            .map(|b| view! { <li>{b.clone()}</li> })
            .collect::<Vec<_>>()
    };

    let blocker_summary = move || {
        if dashboard.get().preflight_blockers.is_empty() {
            "none - all families clear".to_owned()
        } else {
            String::new()
        }
    };

    view! {
        <div class="board">
            <header class="masthead">
                <h1>"TPT CRUCIBLE // OBSERVER"</h1>
                <span class="status" class:live=move || connected.get()>
                    {move || if connected.get() { "LINK LIVE" } else { "LINK IDLE" }}
                </span>
            </header>

            <section class="panel preflight">
                <h2>"PRE-FLIGHT BLOCKERS"</h2>
                <p class="ok">{blocker_summary}</p>
                <ul>{blockers}</ul>
            </section>

            <section class="panel telemetry">
                <h2>"LIVE TELEMETRY"</h2>
                <table>
                    <thead>
                        <tr>
                            <th>"NODE"</th><th>"FAMILY"</th><th>"TOK/S"</th>
                            <th>"MEM GB/S"</th><th>"DRIFT °C"</th><th>"LAT MS"</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </section>
        </div>
    }
}
