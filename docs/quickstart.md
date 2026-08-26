# TPT Crucible Quickstart

Task-oriented walkthroughs for the three things people do first. For the
big picture see the [README](../README.md); for the full task checklist,
[todo.md](../todo.md).

---

## 1. Ingest a model and inspect it

Everything starts by lowering a model file into **TPT-IR**, the
hardware-agnostic intermediate representation.

```bash
# Install the CLI (or `cargo install tpt-crucible-cli --features fpga,swarm`)
cargo install --path crates/tpt-crucible-cli

# GGUF works out of the box; so do .safetensors, .onnx, .pt, .keras, ...
tpt ingest models/tinyllama.gguf --output models/tinyllama.tptir

# Look inside: node count, op histogram, metadata, optional Graphviz dump
tpt info models/tinyllama.tptir
tpt info models/tinyllama.tptir --dot > graph.dot
```

No model handy? The repo ships a ~1.7 KB synthetic-but-valid fixture:

```bash
tpt ingest examples/models/tiny-llama-block.gguf --output /tmp/tiny.tptir
```

Ingestion validates as it goes: structural/SSA invariants *and*
shape/dtype consistency between connected ops. A malformed or unsupported
file fails with a structured message naming what to do — never a stack trace.

## 2. Run the browser demo (zero-install Software-in-the-Loop)

Catalyst and Alloy compile to WebAssembly, and the Observer dashboard is a
pure-Rust Leptos app served from static files.

```bash
cargo install trunk          # once
cd crates/tpt-crucible-observer-web
trunk serve                  # http://127.0.0.1:8080
```

Then start an emitter so the dashboard has something to show:

```bash
tpt preflight models/tinyllama.tptir --serve 127.0.0.1:8787
```

The dashboard connects to `ws://127.0.0.1:8787/ws` and renders: live
telemetry rows across all hardware families, streaming pre-flight blockers,
and the SVG swarm-topology map. No GPU, no hardware, no JavaScript
toolchain — the whole frontend is Rust compiled to `wasm32-unknown-unknown`.

## 3. Add a new ingestion format

All ingestion lives in [`crates/tpt-crucible-catalyst/src`](../crates/tpt-crucible-catalyst/src).
Each format is one module behind a shared entrypoint; nothing outside
Catalyst knows formats exist.

1. **Read how a sibling does it.** `safetensors.rs` is the smallest native
   parser; `gguf.rs` shows metadata trees + quantized dtypes; `onnx.rs`
   shows protobuf decoding and building compute graphs (not just weights).

2. **Write your module** (`src/myformat.rs`) exposing

   ```rust
   pub fn ingest(path: &Path) -> Result<Graph> {
       // Parse bytes natively — Catalyst stays dependency-light.
       // Build tpt_crucible_common::Graph nodes topologically (SSA order):
       //   Op::Input(TensorDesc) / Op::Constant { tensor } sources,
       //   compute ops referencing earlier NodeIds,
       //   Op::Output markers + graph.mark_input/mark_output.
       // Set metadata: source_format plus any hyperparameters you want
       // downstream passes (KV planning!) to find.
   }
   ```

3. **Register it** in `src/format.rs` (detection + enum entry) and
   `src/ingest.rs` (dispatch), plus the CLI's `FormatArg` if it needs an
   explicit override flag.

4. **Test like the existing suites do**: synthesize the container bytes in
   memory, assert on the resulting `Graph` (see the tests at the bottom of
   `gguf.rs`). Malformed input must surface structured errors
   (`Error::ParseFormat{..}`) — fuzzing-friendly, panic-free.

5. Ground rules that will fail CI otherwise: no unsafe, no new heavy
   dependencies, wasm-compatible, MSRV respected. See
   [CONTRIBUTING](../CONTRIBUTING.md); run `just pre-pr` before pushing.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `feature 'fpga' not enabled` | Rebuild with `--features fpga` (same for `swarm`, `hub`) |
| `symbolic dimensions are not supported` (ONNX) | Export the model with static shapes |
| `tpt-uir-serde` build error | It is a sibling checkout (`../../../tpt-uir`); see todo.md's uir-adapter section |
| Browser demo shows LINK IDLE | Nothing is emitting; start `tpt preflight --serve 127.0.0.1:8787` |
