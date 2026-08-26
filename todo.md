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
- [x] ONNX ingestion (native protobuf reader; core-op subset: MatMul/Gemm/elementwise/Softmax/LayerNorm/Reshape/Transpose/Concat/Cast/Gather)
- [x] PyTorch ingestion (torch.save zip format: STORED-member zip reader + restricted pickle VM over data.pkl; deflated/legacy-tar/non-contiguous surface clear errors)
- [x] TensorFlow SavedModel ingestion (frozen-graph `saved_model.pb`: native
      protobuf reader over the shared `pb` decoder; Placeholder/Const/Identity,
      MatMul/BatchMatMul (transpose attrs → Transpose), elementwise binaries +
      unaries, Softmax/Reshape/Transpose/Concat/Cast; outputs from
      `signature_def` (preferring `serving_default`) else dangling producers;
      checkpoint-variable graphs rejected with frozen-graph guidance)
- [x] TFLite ingestion (native read-only FlatBuffers cursor `flatbuf.rs`;
      first subgraph of the "TFL3" model: FULLY_CONNECTED → weight-transpose +
      MatMul (+bias), BATCH_MATMUL with adjoints, ADD/SUB/MUL/DIV,
      CONCATENATION/RESHAPE/TRANSPOSE/CAST, SOFTMAX/LOGISTIC/TANH/RELU +
      unaries; fused RELU/TANH expanded, others rejected; CUSTOM ops named;
      quantized tensors ingested at stored dtype)
- [x] AWQ/GPTQ ingestion (SafeTensors containers + quant-name tagging; dequant kernels pending)
- [x] EXL2 ingestion (HF-style dirs via config.json `quant_method=exl2` sniff
      or `.q_weight` tensor markers; multi-shard SafeTensors merge into one
      weight-only graph; tagged `quant_format=exl2`, `quant_bits` from the
      config's average bitrate else `mixed`; dequant kernels pending)
- [x] JAX/Flax ingestion (native msgpack decoder `msgpack.rs`; Flax pytrees
      via `flax.rs`: ndarray ext-1 leaves decode `[shape, numpy-dtype-name,
      row-major bytes]` incl. bfloat16, ext-3 scalars, bin leaves as U8,
      `__msgpack_chunked_array__` wrappers concatenated; weight-only graph
      named by dotted tree paths; complex dtypes rejected)
- [x] Llamafile ingestion (embedded-GGUF extraction feeding the native GGUF parser)
- [x] Keras ingestion (Keras v3 `.keras` zip archives: config.json signature check, nested `states.npz` stores via native NPY parser; legacy `.h5` rejected with guidance)
- [x] HuggingFace directory-layout ingestion (`tpt ingest <hf-repo-dir>/`:
      config.json + `*.safetensors` shards detected as SafeTensors and merged
      into one weight-only graph, duplicate tensor names rejected across
      shards; EXL2 dirs keep their dedicated detection/ingestor)
- [x] HuggingFace Hub network fetch (download a repo id into the local cache;
      needs an HTTP client dependency — deferred)
      (done as feature-gated `hub` module + `tpt ingest <org/name> --hub`:
      `ureq`/rustls stays opt-in so the wasm SiL build is untouched; lists
      repo siblings via the Hub API, downloads `config.json` + every
      `*.safetensors` shard with `.part` atomic renames and size-skip resume,
      then feeds the existing HF directory-layout ingestion; verified live
      against `hf-internal-testing/tiny-random-LlamaForCausalLM`)
- [ ] Operator fusion via `egg` e-graphs
      (deferred: pulls a heavyweight e-graph dependency into a deliberately
      dep-light crate; revisit alongside the executor so fused groups have
      something to lower into)
- [x] Quantization auto-search (`--accuracy-budget` flag; INT4 with INT8
      promotion on fragile layers, validated against SiL pass / `.tptprofile`
      sensitivity data)
      (`autosearch`: root-sum-square error model, most-fragile-first
      promotion to Q8_0; `.tptprofile` sidecar mode + graph-shape heuristic
      fallback; serializable `QuantPlan`, `tpt quantize` exits non-zero when
      the budget is unreachable; live SiL validation hooks in once the
      executor exists)
- [x] Streaming pre-flight: operator compatibility analysis streamed to Observer over WebSockets
      (`preflight`: per-node events emitted mid-traversal against a static v1
      capability matrix for alloy/fusion/emulated/unsupported verdicts;
      `tpt preflight --serve <addr>` binds the Observer telemetry server and
      fans events to `/ws` clients as `{"kind":"preflight",…}` frames)
- [x] `tpt-doctor` toolchain verifier subcommand (scans external tools, checks versions, runs smoke test)
- [ ] Custom MLIR dialect for TPT-IR (`mlir-sys` / `llvm-sys`)
      (deferred: requires the LLVM/MLIR native toolchain on every dev/CI
      machine, contradicting the pure-Rust no-heavy-deps posture; revisit if
      a consumer actually needs MLIR interop)
- [x] TPT-IR output serializable to JSON/Binary

### tpt-crucible-uir-adapter (TPT-UIR compatibility)

- [x] `Graph` -> `Region` conversion (Crucible dialect, single block, no
      block arguments — inputs modeled as `tpt_crucible.input` ops so node
      names survive the round trip)
- [x] `Region` -> `Graph` conversion (inverse; round-trip tested against a
      Llama-block-shaped graph)
- [x] Crucible dialect extended in `tpt-uir` with ~27 `tpt_crucible.*`
      compute op names (matmul/attention/rms_norm/softmax/rope/etc.),
      previously only had the 3 hardware-placement ops
      (`map_flash`/`route_fpga`/`analog_conv`)
- [x] `AttributeValue::Bytes` added to `tpt-uir-core` for embedded constant
      tensors (postcard + hand-extended FlatBuffers union support)
- [x] `ScalarType::Q5_0`/`Q5_1` added to `tpt-uir-core` to match TPT-IR's
      `DType` (postcard, text, FlatBuffers)
- [x] `tpt-uir-text`'s parser doesn't yet round-trip `bytes<...>` attributes
      (fixed in the sibling `tpt-uir` checkout: the lexer now emits a raw
      `Tok::Hex` for `bytes<...>` payloads so all-digit hex no longer gets
      mangled into integers, and the parser decodes them into
      `AttributeValue::Bytes`; round-trip covered by
      `tpt-uir-text::tests::bytes_attribute_roundtrips_through_text`)
- [x] Wire `tpt-crucible-catalyst` output through the adapter +
      `tpt-uir-serde`: `tpt ingest --uir <file>` emits a postcard-encoded
      Crucible-dialect region alongside the `.tptir`; end-to-end round trip
      covered by an integration test ingesting a synthetic GGUF v3 fixture
      (`crates/tpt-crucible-uir-adapter/tests/ingest_roundtrip.rs`), and the
      emitted artifact was validated live with `tpt-uir-cli validate --dialect crucible`
- [ ] `tpt-uir` is currently a sibling-checkout path dependency
      (`../../../tpt-uir/crates/...`); revisit once/if `tpt-uir` publishes to
      crates.io

### tpt-crucible-alloy

- [x] TPT-IR -> node partitioning via `petgraph`
- [x] Topology-aware partitioning (minimizes data travel time)
- [x] Physical topology auto-discovery from a node-reported latency matrix
- [x] Attention-head parallel partitioning, resuming layer-serial partitioning for FFN sublayers
      (v1 emits head-slice groups per attention op; graph-level head split lands with the executor)
- [x] KV cache distribution across nodes (prevents OOM on memory-constrained nodes;
      plans fail loudly naming the tight node)
- [x] Dead-node bypass (`recovery::bypass_dead_nodes`: prunes dead topology
      ids, re-slices the latency/bandwidth matrices over survivors, re-runs
      the planner, and reports displaced IR nodes plus a new→old survivor
      map; composes with `FailureDetector` output, tolerates duplicate/
      out-of-range ids, rejects all-dead fleets)
- [x] Runtime execution engine (distributes shards, drives heartbeats,
      triggers recovery on failure)
      (`runtime::ExecutionEngine`: bincode `ShardDeployment`s keyed by stable
      topology ids, injected-time `FailureDetector` + automatic
      `recovery::bypass_dead_nodes` on silence, generation counter and
      `NodeLost`/`Recovered`/`FleetLost` lifecycle events; transport-free by
      design — the coordinator wires it to its own sockets)
- [x] Rolling pipeline parallelism (eliminates inference stalls)
      (`pipeline`: exact greedy flow-shop simulation of micro-batches across
      shard stages with measured inter-stage transfer; makespan vs.
      layer-serial baseline, speedup, steady-state bottleneck period,
      per-slot timelines; `PipelineModel::from_plan` estimates stage costs
      from shard weight bytes)
- [x] Node-specific firmware generation via `askama` templating (Rust/C++, memory-safe)
      (four compile-time checked askama templates under
      `crates/tpt-crucible-alloy/templates/` — Rust/C++ firmware sources plus
      bash/PowerShell master flash scripts; broken templates fail the build,
      not a node's first boot)
- [x] Master flashing script generation
- [x] Heartbeat protocol implementation
- [x] Wasm compilation target (`wasm32-unknown-unknown`) for the browser SiL demo
      (`cargo check --target wasm32-unknown-unknown` passes for common+alloy)
- [x] Hybrid silicon+FPGA node representation (`topology::FpgaProfile`,
      `SwarmNode.fpga`) — descriptive metadata only; not yet read by the
      partitioner or firmware generator
- [x] FPGA-aware partitioning (`PartitionOptions::fpga_offload`: GEMM-class
      segments — MatMul/Attention — prefer opening on hybrid boards whose
      fabric can stage the segment's weights in block RAM; shards report
      `fpga_ops`/`fpga_offload()`, stats count routed ops; silicon fallback
      when the BRAM cannot fit; default off keeps placement unchanged)
- [x] Runtime-adaptive capability (`SwarmNode::report_fpga` /
      `Topology::report_fpga`: hybrid boards re-report their `FpgaProfile`
      after an overlay load/teardown and the next partition run plans against
      the new block-RAM capability — covered by a fallback→offload→revert test)

### tpt-crucible-cli

- [x] `tpt` binary skeleton + subcommand routing
- [x] `ingest` / `compile` commands wired to Catalyst + Alloy
- [x] `--features fpga,swarm` cargo feature gating (per spec2.txt section 4.7;
      `swarm` is default-on, compiled-out targets fail with a rebuild hint)
- [ ] **Milestone:** Load TinyLlama, partition via Alloy, flash to 16 ESP32s
      (SiL dry run of the full pipeline is proven end-to-end in
      `crates/tpt-crucible-alloy/tests/swarm_end_to_end.rs`: real GGUF
      ingestion → 16-node hybrid partitioning + KV planning → firmware/flash
      generation → engine deploy → mid-flight node death → automatic
      recovery → rolling-pipeline scheduling.
      **Physical leg started on real hardware**: an ESP32-C3 (rev v0.4,
      4 MB XMC flash, MAC `e8:3d:c1:83:72:d8`, probed via esptool on COM5)
      now runs `no_std` esp-hal firmware (`hardware/esp32c3-heartbeat/`)
      broadcasting protocol-conformant heartbeat records at 1 Hz over its
      USB-Serial/JTAG port; captured records decode through the production
      codec and drive the production `FailureDetector`
      (`alloy/tests/heartbeat_firmware_conformance.rs`). Remaining for the
      full milestone: 15 more boards, WiFi transport replacing the serial
      link, and the executor to run real shards.)

## Phase 2: The Silicon Canvas (Months 6-12)

### tpt-crucible-fusion

- [ ] Hardware description via `rust-hdl`
      (v1 emits hand-written parameterized Verilog-2001 instead — see
      `rtl.rs`; migrating generation to rust-hdl stays open as the tracked
      follow-up)
- [ ] HBM auto-router (wires compute arrays to HBM pins via pre-verified memory controllers)
      (`overlay`: v1 assigns banks → HBM pseudo-channels round-robin in
      `.fusecfg`; pin-level routing lands with the real bitstream backend)
- [x] FPGA overlay architecture (writes weight data + datapath config into a pre-synthesized overlay instead of triggering full resynthesis; ~10s per-model compile)
      (`overlay::plan_overlay`: BRAM-bank staging per GEMM layer + DSP
      accounting + HBM channel assignment, emitted as versioned `.fusecfg`;
      pure planning, microsecond-scale, no synthesis in the loop)
- [x] Yosys/Nextpnr wrappers (`std::process::Command` / FFI)
      (`tools`: silent subprocess wrappers behind an injectable `ToolRunner`;
      missing binaries surface as structured `Error::ExternalTool` values
      with install hints; v1 covers elaboration + place-and-route entry
      points)
- [ ] LiteX/LiteDRAM integration via generated Verilog wrappers
- [x] Output: synthesizable RTL, memory initialization files, `.fusecfg` overlay configuration files
      (`compile` produces an `OverlayBundle`: `overlay.fusecfg`,
      `rtl/mac_array.v`, per-layer `$readmemh` meminit files;
      `tpt compile --target fusion --out-dir` writes it)
- [ ] **Milestone:** Select a Xilinx Alveo FPGA board, output a ready-to-flash bitstream using HBM
- [x] `alloy`<->`fusion` bridge for hybrid boards: `fusion::compile` targets
      `tpt_crucible_alloy::topology::FpgaProfile` as its output contract, once
      it's more than a stub (see matching items under `tpt-crucible-alloy`)
      (fusion depends on alloy and converts its `FpgaProfile` directly;
      hybrid-board partitioning already plans against the same profile)

## Phase 3: The Physics Engine (Year 2)

### tpt-crucible-element

- [x] Xyce/ngspice FFI bindings
      (v1 drives ngspice in batch mode via `std::process::Command` —
      `simulator::run_ngspice` runs a rendered netlist (opt-in
      `.control`/`wrdata` export block) and parses the operating point back;
      missing binaries surface as structured `Error::ExternalTool` values;
      in-process shared-library FFI and the Xyce adapter remain open)
- [x] SPICE netlist generation from TPT-IR weights (floating-point weights -> physical electrical components)
      (`netlist`: float MatMul layers extracted as crossbar MAC arrays — VCCS
      `G`-element transconductance bank per weight + load resistors so
      `V(out_i) = R·Σ W[i][j]·V(in_j)`; `.temp`/`.op`/`.end` deck renders for
      ngspice/Xyce; quantized payloads rejected with dequantize guidance)
- [x] "Reality Check" engine: injects simulated thermal noise, voltage drift, component tolerance errors
      (`reality`: seeded xorshift* Monte-Carlo over the three factors,
      Irwin–Hall normals, evenly spaced row sampling with caps; reports mean/
      max deviation, confidence score = fraction of outputs inside tolerance,
      and dominant-factor attribution via single-factor isolation runs)
- [x] Hardware mitigation suggestions
      (rule-based on the dominant factor: calibration DACs / precision parts
      for tolerance, dedicated LDO + chopper auto-zero for drift, impedance /
      averaging for thermal; plus confidence-threshold derating advice)
- [ ] `ort`-based ML model to predict drift instantly
- [x] PCB layout recommendation output
      (`pcb`: universal analog hygiene — star ground, guard rings, decoupling
      — plus geometry-driven advice (input-bus segmentation, output mux
      placement) and physics-driven blocks keyed to the dominant factor)
- [x] Confidence score output
      (`RealityReport.confidence`, surfaced by `tpt compile --target element`
      alongside `netlist.sp` and `reality_report.json` artifacts)
- [ ] **Milestone:** Design a 3-layer analog NN, simulate thermal drift, output a KiCad PCB

## Phase 4: The Observer (Year 2+)

### tpt-crucible-observer

- [x] Unified telemetry schema: tokens/sec, memory bandwidth, thermal drift, node latency
      (`TelemetryEvent` defined so other crates can emit against it early)
- [x] `axum` + `tokio-tungstenite` WebSocket telemetry backend
      (`TelemetryServer::bind("ip:port")` spawns the axum app; emitters push
      `TelemetryEvent`s via `emit`, fanned out through a broadcast channel to
      every connected `/ws` client as JSON text frames — lagging clients get
      a `{"lagged":n}` note instead of stalling the swarm; in-process taps via
      `subscribe()`)

### tpt-crucible-observer-web (8th workspace crate, pure Rust frontend)

- [x] Scaffold as a Leptos app (`crates/tpt-crucible-observer-web`), built via `cargo-leptos`, targeting `wasm32-unknown-unknown`
      (Leptos 0.8 CSR scaffold compiles for wasm32-unknown-unknown; build/run
      via trunk (`index.html`, `trunk serve`) — cargo-leptos not required at
      this size; WebSocket client wired to the Observer `/ws` stream with a
      locally-mirrored, regression-tested frame schema)
- [x] Cyberpunk-industrial UI, reactive telemetry views wired to the Observer WebSocket backend
      (v1: reactive live-telemetry table across all hardware families +
      pre-flight blocker banner + link-status pill; deeper visualization
      split into the wgpu item below)
- [ ] 3D swarm topology + PCB visualization via `wgpu` directly (no Three.js/React Three Fiber -- hand-rolled scene rendering, runs via WebGPU with WebGL fallback in-browser)
- [x] "Industrial blueprint" dark-mode theme (styling approach TBD -- Leptos supports plain CSS/Tailwind-via-build-step; revisit when this phase started)
      (decision made: plain CSS in `assets/style.css` — grid-paper backdrop,
      neon-cyan accents, monospace; Tailwind stays available if the UI grows)
- [ ] **Milestone:** Dashboard unifying telemetry across Alloy/Fusion/Element hardware types
      (UI renders all families from one schema; milestone needs live
      multi-source validation against real emitters)

## Accessibility & Democratization (cross-cutting, spec2.txt section 4.7)

- [ ] Zero-install browser demo: Catalyst + Alloy compiled to Wasm, SiL emulator driving the `tpt-crucible-observer-web` frontend via Web Workers
- [ ] Pre-compiled package marketplace: community registry of `.tptpkg` files
- [ ] Open hardware reference board: KiCad PCB for an ESP32 swarm carrier board (`hardware/reference-designs/alloy-carrier/`)
- [ ] One-line bootstrap verified end-to-end: `cargo install tpt-crucible-cli` (and `--features fpga,swarm`)

## crates.io Release Readiness (ongoing, all crates)

- [x] Per-crate `description`/`keywords`/`categories` finalized
- [x] Per-crate `README.md` written (crates.io renders these)
- [x] docs.rs builds cleanly for every crate (verified locally: `cargo doc
      --workspace --no-deps` is warning-free, including the `fpga,swarm`
      feature-gated code paths; re-check on real docs.rs once the first
      crate is published)
- [x] Semver policy documented; 0.1.0 → 1.0.0 criteria defined
      ([`VERSIONING.md`](VERSIONING.md): 0.x breakage = minor bump, data-format
      stability contract for TPT-IR JSON/binary + heartbeat frames, additive-
      only enum growth, two clean 0.y releases + real usage to graduate,
      MSRV/deprecation/yanking rules)
- [ ] Publish order respected: `common` -> `catalyst` -> {`fusion`, `element`, `alloy`} -> `cli`; `observer` and `observer-web` optional
      (publish workflow encodes this order; needs a real registry run to verify)
      (probed locally 2026-08-26: `cargo publish --dry-run -p
      tpt-crucible-common` passes cleanly; every downstream crate currently
      aborts because internal path deps carry no version yet — see next item)
- [ ] Decide whether `tpt-crucible-observer-web` is published to crates.io at all (frontend wasm binaries are unusual crates.io citizens) or just built/deployed from the workspace without publishing
- [x] Add `version` to internal path dependencies once the first crate is published
      (done proactively: `{ path, version = "0.1.0" }` in
      `[workspace.dependencies]`; harmless pre-publish since `path` wins for
      local builds, and it clears the packaging-time "dependency needs a
      version" error for every internal crate)
- [ ] `cargo publish --dry-run` passes for every crate, in dependency order
      (`tpt-crucible-common` passes fully. Downstream crates now package
      cleanly but their dependency resolution targets the registry, so their
      dry-runs only complete after the first real publish cascades;
      `uir-adapter`/`cli` additionally wait on `tpt-uir` publishing)
