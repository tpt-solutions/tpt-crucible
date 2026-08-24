# TPT Crucible - Rust Rewrite TODO

Tracks the full rewrite of TPT Crucible as a pure-Rust Cargo workspace, per
`spec2.txt`. Organized by roadmap phase (`spec2.txt` section 5); each phase is
broken down per-crate using the key features from `spec2.txt` section 3.

## Phase 0: Workspace & Repo Setup

- [x] Cargo workspace scaffold (7 crates, stub `lib.rs`/`main.rs`, crates.io metadata)
- [x] Dual license (MIT OR Apache-2.0): `LICENSE-MIT`, `LICENSE-APACHE`
- [x] `README.md` rewritten for the new 7-crate scope
- [x] `todo.md` (this file)
- [x] Recreate `CONTRIBUTING.md` / `SECURITY.md` / `CHANGELOG.md`
- [x] CI: fmt + clippy + test + build workflow (`.github/workflows/ci.yml`)
- [x] CI: `cargo publish` workflow (needs `CARGO_REGISTRY_TOKEN` secret)
- [x] Verify all 7 crate names are available on crates.io
  (all `tpt-crucible*` names were unregistered as of 2026-08-25)

## Phase 1: The Catalyst & The Swarm (Months 1-6)

### tpt-crucible-common

- [x] TPT-IR core types (graph, ops, tensors, dtypes incl. GGML block quants,
      structural validation, Graphviz export)
- [x] Shared error handling (`thiserror`-based error enum)
- [x] JSON + binary (de)serialization for TPT-IR
      (JSON via serde_json; binary = `TPTIR` magic header + bincode)

### tpt-crucible-catalyst

- [x] SafeTensors ingestion (native parser + encoder, no heavy deps)
- [x] GGUF ingestion (native v2/v3 parser: metadata tree, tensor directory,
      legacy block-quant dtypes; candle-core integration optional/later)
- [ ] ONNX ingestion
- [ ] PyTorch ingestion
- [ ] TensorFlow SavedModel ingestion
- [ ] TFLite ingestion
- [ ] AWQ/GPTQ ingestion
- [ ] EXL2 ingestion
- [ ] JAX/Flax ingestion
- [ ] Llamafile ingestion
- [ ] Keras ingestion
- [ ] HuggingFace Hub fetch integration (format layer recognizes HF artifacts; network fetch pending)
- [ ] Operator fusion via `egg` e-graphs
- [ ] Quantization auto-search (`--accuracy-budget` flag; INT4 with INT8 promotion on fragile layers, validated against SiL pass / `.tptprofile` sensitivity data)
- [ ] Streaming pre-flight: operator compatibility analysis streamed to Observer over WebSockets
- [x] `tpt-doctor` toolchain verifier subcommand (scans external tools, checks versions, runs smoke test)
- [ ] Custom MLIR dialect for TPT-IR (`mlir-sys` / `llvm-sys`)
- [x] TPT-IR output serializable to JSON/Binary

### tpt-crucible-alloy

- [x] TPT-IR -> node partitioning via `petgraph`
- [x] Topology-aware partitioning (minimizes data travel time)
- [x] Physical topology auto-discovery from a node-reported latency matrix
- [x] Attention-head parallel partitioning, resuming layer-serial partitioning for FFN sublayers
      (v1 emits head-slice groups per attention op; graph-level head split lands with the executor)
- [x] KV cache distribution across nodes (prevents OOM on memory-constrained nodes;
      plans fail loudly naming the tight node)
- [ ] Fault-tolerant execution: node heartbeats, dead-node bypass
      (heartbeat codec + failure detector done; execution engine pending)
- [ ] Rolling pipeline parallelism (eliminates inference stalls)
- [x] Node-specific firmware generation via `askama` templating (Rust/C++, memory-safe)
      (string-built templates today; askama migration pending)
- [x] Master flashing script generation
- [x] Heartbeat protocol implementation
- [x] Wasm compilation target (`wasm32-unknown-unknown`) for the browser SiL demo
      (`cargo check --target wasm32-unknown-unknown` passes for common+alloy)

### tpt-crucible-cli

- [x] `tpt` binary skeleton + subcommand routing
- [x] `ingest` / `compile` commands wired to Catalyst + Alloy
- [x] `--features fpga,swarm` cargo feature gating (per spec2.txt section 4.7;
      `swarm` is default-on, compiled-out targets fail with a rebuild hint)
- [ ] **Milestone:** Load TinyLlama, partition via Alloy, flash to 16 ESP32s

## Phase 2: The Silicon Canvas (Months 6-12)

### tpt-crucible-fusion

- [ ] Hardware description via `rust-hdl`
- [ ] HBM auto-router (wires compute arrays to HBM pins via pre-verified memory controllers)
- [ ] FPGA overlay architecture (writes weight data + datapath config into a pre-synthesized overlay instead of triggering full resynthesis; ~10s per-model compile)
- [ ] Yosys/Nextpnr wrappers (`std::process::Command` / FFI)
- [ ] LiteX/LiteDRAM integration via generated Verilog wrappers
- [ ] Output: synthesizable RTL, memory initialization files, `.fusecfg` overlay configuration files
- [ ] **Milestone:** Select a Xilinx Alveo FPGA board, output a ready-to-flash bitstream using HBM

## Phase 3: The Physics Engine (Year 2)

### tpt-crucible-element

- [ ] Xyce/ngspice FFI bindings
- [ ] SPICE netlist generation from TPT-IR weights (floating-point weights -> physical electrical components)
- [ ] "Reality Check" engine: injects simulated thermal noise, voltage drift, component tolerance errors
- [ ] Hardware mitigation suggestions
- [ ] `ort`-based ML model to predict drift instantly
- [ ] PCB layout recommendation output
- [ ] Confidence score output
- [ ] **Milestone:** Design a 3-layer analog NN, simulate thermal drift, output a KiCad PCB

## Phase 4: The Observer (Year 2+)

### tpt-crucible-observer

- [x] Unified telemetry schema: tokens/sec, memory bandwidth, thermal drift, node latency
      (`TelemetryEvent` defined so other crates can emit against it early)
- [ ] `axum` + `tokio-tungstenite` WebSocket telemetry backend

### tpt-crucible-observer-web (8th workspace crate, pure Rust frontend)

- [ ] Scaffold as a Leptos app (`crates/tpt-crucible-observer-web`), built via `cargo-leptos`, targeting `wasm32-unknown-unknown`
- [ ] Cyberpunk-industrial UI, reactive telemetry views wired to the Observer WebSocket backend
- [ ] 3D swarm topology + PCB visualization via `wgpu` directly (no Three.js/React Three Fiber -- hand-rolled scene rendering, runs via WebGPU with WebGL fallback in-browser)
- [ ] "Industrial blueprint" dark-mode theme (styling approach TBD -- Leptos supports plain CSS/Tailwind-via-build-step; revisit when this phase starts)
- [ ] **Milestone:** Dashboard unifying telemetry across Alloy/Fusion/Element hardware types

## Accessibility & Democratization (cross-cutting, spec2.txt section 4.7)

- [ ] Zero-install browser demo: Catalyst + Alloy compiled to Wasm, SiL emulator driving the `tpt-crucible-observer-web` frontend via Web Workers
- [ ] Pre-compiled package marketplace: community registry of `.tptpkg` files
- [ ] Open hardware reference board: KiCad PCB for an ESP32 swarm carrier board (`hardware/reference-designs/alloy-carrier/`)
- [ ] One-line bootstrap verified end-to-end: `cargo install tpt-crucible-cli` (and `--features fpga,swarm`)

## crates.io Release Readiness (ongoing, all crates)

- [x] Per-crate `description`/`keywords`/`categories` finalized
- [x] Per-crate `README.md` written (crates.io renders these)
- [ ] docs.rs builds cleanly for every crate (check feature-gated code paths)
- [ ] Semver policy documented; 0.1.0 -> 1.0.0 criteria defined
- [ ] Publish order respected: `common` -> `catalyst` -> {`fusion`, `element`, `alloy`} -> `cli`; `observer` and `observer-web` optional
      (publish workflow encodes this order; needs a real registry run to verify)
- [ ] Decide whether `tpt-crucible-observer-web` is published to crates.io at all (frontend wasm binaries are unusual crates.io citizens) or just built/deployed from the workspace without publishing
- [ ] Add `version` to internal path dependencies once the first crate is published
- [ ] `cargo publish --dry-run` passes for every crate, in dependency order
