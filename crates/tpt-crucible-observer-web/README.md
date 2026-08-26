# tpt-crucible-observer-web

The **Observer dashboard frontend**: a pure-Rust Wasm app (Leptos CSR)
rendering the live telemetry stream — tokens/sec, memory bandwidth, thermal
drift, node latency — plus streaming pre-flight blockers, styled as an
industrial-blueprint dark theme.

## Wire contract

Connects to the Observer backend's `GET /ws` stream and parses its JSON
envelopes (`{"kind":"telemetry",…}`, `{"kind":"preflight",…}`). The schema is
mirrored locally (`src/frames.rs`) because the backend crate cannot target
wasm; the pair is regression-tested on both sides.

## Build & run

```bash
cargo install trunk          # once
cd crates/tpt-crucible-observer-web
trunk serve                  # http://127.0.0.1:8080 - hot reload
```

Point it at a running Observer:

```bash
tpt preflight model.tptir --serve 127.0.0.1:8787   # streams pre-flight events
# or any emitter against TelemetryServer::emit
```

`trunk build --release` produces a fully static `dist/` directory.

## Status

Scaffold complete: WS feed, live telemetry table, blocker banner, theme.
Next (see workspace todo.md): 3D swarm topology + PCB visualization via wgpu,
historical charts.
