# tpt-crucible-observer

Telemetry backend for TPT Crucible (Phase 4). Will aggregate live tokens/sec,
memory bandwidth, thermal drift, and node latency from Alloy swarms, Fusion
overlays, and Element simulations over an axum + WebSocket API, consumed by the
Leptos-based dashboard frontend.

Status today: the unified `TelemetryEvent` schema is defined so other crates can
can emit against a stable shape; the server itself lands in Phase 4. See the
[workspace todo.md](https://github.com/tpt-solutions/tpt-crucible/blob/master/todo.md).