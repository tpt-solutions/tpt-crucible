# Contributing to TPT Crucible

Thanks for helping build hardware-agnostic AI tooling! This repo is the pure-Rust
rewrite of TPT Crucible tracked in [todo.md](todo.md) against
[spec2.txt](spec2.txt).

## Development setup

1. Install Rust via [rustup](https://rustup.rs). The `rust-toolchain.toml`
   file pins stable plus `rustfmt`, `clippy`, and the `wasm32-unknown-unknown`
   target - they install automatically on your first cargo command.
2. Clone and build:

```bash
git clone https://github.com/tpt-solutions/tpt-crucible
cd tpt-crucible
cargo build --workspace
cargo test --workspace
```

## Workspace layout

| Crate | Purpose |
|---|---|
| `tpt-crucible-common` | TPT-IR types (graph/ops/tensors/dtypes), shared errors, JSON+binary serialization |
| `tpt-crucible-catalyst` | Model ingestion (SafeTensors, GGUF today; more formats on the roadmap), `tpt-doctor` |
| `tpt-crucible-alloy` | Swarm partitioning, KV-cache planning, heartbeat protocol, firmware generation |
| `tpt-crucible-fusion` | FPGA overlay backend (Phase 2) |
| `tpt-crucible-element` | Analog simulation backend (Phase 3) |
| `tpt-crucible-observer` | Telemetry schema/backend (Phase 4) |
| `tpt-crucible-cli` | The unified `tpt` binary |

Dependency direction is one-way: everything may depend on `common`; backends
must not depend on each other; only `cli` wires backends together (behind the
`swarm` / `fpga` features).

## Ground rules

* **No unsafe code.** The workspace forbids it (`unsafe_code = "forbid"`).
* **Pure Rust, wasm-friendly.** Anything in `common`, `catalyst`, or `alloy`
  must compile for `wasm32-unknown-unknown`.
* **MSRV:** declared in `[workspace.package]` (`rust-version`). Do not raise it
  without discussion.
* **Errors:** use the shared `common::Error` enum; add variants rather than
  introducing new error types per crate.
* **Automation-first:** if a failure can be detected and surfaced as a
  structured message, do that instead of returning raw process output.

## Before you open a PR

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

CI runs exactly these three on every push and PR, plus a wasm check for the
browser-demo crates. Keep PRs focused; reference the `todo.md` item your work
implements. Conventional-commit-style messages (`feat:`, `fix:`, `docs:`,
`chore:`) keep changelog generation simple.

## Picking up work

Grab any unchecked item from [todo.md](todo.md). Phase ordering matters:
Phase 1 (Catalyst + Alloy) before Phase 2/3 backends. If an item needs a design
decision (e.g. quantization auto-search heuristics), open an issue first with
your proposal so the roadmap stays coherent.

## Licensing

By contributing you agree your contributions are dual-licensed under
MIT OR Apache-2.0, per the headers in `LICENSE-MIT` and `LICENSE-APACHE`.