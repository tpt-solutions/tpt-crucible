# TPT Crucible

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Version](https://img.shields.io/badge/version-0.1.0-cyan)](CHANGELOG.md)

**Hardware-agnostic AI compiler suite, written in pure Rust.** TPT Crucible bypasses the traditional GPU/AI-hardware monopoly: it lets engineers, researchers, and DIY hardware builders compile, simulate, and deploy standard AI models onto non-traditional, custom-built hardware — FPGAs, analog compute-in-memory circuits, and distributed microcontroller swarms.

> Instead of forcing your AI to fit a commercial GPU, TPT Crucible adapts the AI to fit the physical reality of your hardware.

---

## Quick Start

```bash
# Install the full toolchain (installs the `tpt` CLI)
cargo install tpt-crucible-cli

# ...or opt into specific hardware targets
cargo install tpt-crucible-cli --features fpga,swarm

# Ingest a GGUF model and compile it for an ESP32 swarm
tpt ingest models/tinyllama.gguf --target alloy --output dist/tinyllama.tptpkg
```

No hardware required to get started — Alloy and Catalyst compile to WebAssembly for a zero-install, Software-in-the-Loop browser demo.

---

## Architecture

The suite is a "Core and Modules" architecture managed as a single Rust Cargo workspace. The Core translates an AI model into a raw, hardware-agnostic mathematical format (TPT-IR); each Module translates that IR into physical hardware instructions for one class of target.

```
AI Model (.gguf / .safetensors / .onnx / .pt / .tflite / ...)
        |
   tpt-crucible-catalyst  -->  TPT-IR
        |
   +----+----+----+
 alloy fusion element
   |     |     |
  MCU   RTL  SPICE
        |
   tpt-crucible-observer  (live telemetry dashboard)
```

## Modules

| Crate | Purpose |
|---|---|
| [`tpt-crucible-common`](crates/tpt-crucible-common) | TPT-IR definitions and shared error handling used by every other crate |
| [`tpt-crucible-catalyst`](crates/tpt-crucible-catalyst) | Model ingestion into TPT-IR - SafeTensors, GGUF, ONNX, Llamafile, AWQ/GPTQ implemented natively; PyTorch/TF/TFLite/EXL2/JAX/Keras on the roadmap |
| [`tpt-crucible-fusion`](crates/tpt-crucible-fusion) | FPGA module — high-bandwidth logic synthesis for HBM-backed MAC arrays |
| [`tpt-crucible-element`](crates/tpt-crucible-element) | Analog module — physics-to-weight mapping and thermal/noise circuit simulation |
| [`tpt-crucible-alloy`](crates/tpt-crucible-alloy) | Swarm module — distributed graph partitioning and firmware generation for microcontroller swarms (ESP32, RP2040, RISC-V) |
| [`tpt-crucible-observer`](crates/tpt-crucible-observer) | Real-time telemetry and hardware monitoring dashboard backend |
| [`tpt-crucible-observer-web`](crates/tpt-crucible-observer-web) | Observer dashboard frontend — a Leptos (Rust/Wasm) app with `wgpu`-rendered 3D swarm topology and PCB views |
| [`tpt-crucible-cli`](crates/tpt-crucible-cli) | The unified `tpt` binary entrypoint |

The entire suite, frontend included, is pure Rust — the Observer dashboard compiles to WebAssembly via Leptos/`cargo-leptos` and talks to `tpt-crucible-observer`'s WebSocket API, keeping one unified toolchain from compiler backend to generated firmware to UI.

## Key Features

- **Pure Rust platform** — memory safety, zero-cost abstractions, and seamless WebAssembly compilation for a browser-based demo, with one unified toolchain for the compiler backend and generated firmware.
- **Universal model ingestion** — GGUF, SafeTensors, HuggingFace Hub, ONNX, PyTorch, TensorFlow SavedModel, TFLite, AWQ/GPTQ, EXL2, JAX/Flax, Llamafile, Keras.
- **Operator fusion** via Rust-based e-graphs (`egg`), and quantization auto-search against an accuracy budget.
- **FPGA overlay architecture** — cuts per-model FPGA compile time from hours to ~10 seconds.
- **Transformer-native swarm partitioning** — attention-head parallel + layer-serial hybrid partitioning, with KV-cache-aware memory planning and fault-tolerant execution.
- **"Reality Check" analog simulation** — injects thermal noise, voltage drift, and component tolerance errors, with an ML-predicted confidence score.
- **Zero-install browser demo** — Catalyst and Alloy compile to `wasm32-unknown-unknown` for a Software-in-the-Loop emulator that runs entirely client-side.

## Development Roadmap

- **Phase 1 (Months 1-6): The Catalyst & The Swarm** — Build `tpt-crucible-catalyst` and `tpt-crucible-alloy`. *Milestone: load TinyLlama, partition via Alloy, flash to 16 ESP32s.*
- **Phase 2 (Months 6-12): The Silicon Canvas** — Build `tpt-crucible-fusion`. *Milestone: select a Xilinx Alveo FPGA board, output a ready-to-flash HBM bitstream.*
- **Phase 3 (Year 2): The Physics Engine** — Build `tpt-crucible-element`. *Milestone: design a 3-layer analog NN, simulate thermal drift, output a KiCad PCB.*
- **Phase 4 (Year 2+): The Observer** — Build the `tpt-crucible-observer` dashboard to unify telemetry across all hardware types.

See [todo.md](todo.md) for the full task-level checklist.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and PRs are welcome.

## Security

See [SECURITY.md](SECURITY.md) for how to report vulnerabilities.

## License

Dual-licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option. Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project shall be dual-licensed as above, without any additional terms or conditions.

Copyright 2026 TPT Solutions.