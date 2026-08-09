# TPT Crucible — Master Task Checklist

---

## Phase 1: The Catalyst & The Swarm (Months 1–6)

### TPT Catalyst — Core IR Compiler
- [x] Set up mono-repo project structure (Rust workspace + Python packages)
- [x] Integrate MLIR toolchain and define custom TPT-IR MLIR dialect
- [x] Implement PyTorch model ingestion (.pt / TorchScript)
- [x] Implement ONNX model ingestion
- [x] Implement TensorFlow/SavedModel ingestion
- [x] Implement GGUF model ingestion (llama.cpp format, quantization-preserving)
- [x] Preserve GGUF quantization metadata (Q4_K, Q8_0, etc.) through TPT-IR
- [x] Integrate Apache TVM for initial graph optimization pass
- [x] Implement operator fusion (fuse sequential ops like MatMul + activation into single blocks)
- [x] Define and implement TPT-IR serialization format (JSON + binary)
- [x] Write unit tests for ingestion and IR correctness
- [x] CLI: `tpt-catalyst ingest <model>` → outputs `.tptir` file

### TPT Catalyst — Extended Input Formats
- [x] Implement SafeTensors ingestion (`safetensors` library; memory-mapped, dtype/shape metadata preserved)
- [x] Implement HuggingFace model directory ingestion: auto-detect config.json + weights file, load tokenizer metadata
- [x] CLI: `tpt-catalyst ingest <hf-model-dir>` and `tpt-catalyst ingest --hf-repo org/model-name`
- [x] Implement HuggingFace Hub pull: download model to local cache if not present, then ingest
- [x] Implement TFLite ingestion (`.tflite`): parse FlatBuffer schema, map pre-quantized ops to TPT-IR, preserve quantization params
- [x] Implement AWQ/GPTQ ingestion: read `quantize_config.json` from HF repo, extract per-layer bit-width assignments, pass to TPT-IR quantization metadata
- [x] Implement EXL2 ingestion (`.exl2`): extract per-layer quantization scale/zero tables into TPT-IR
- [x] Implement JAX/Flax orbax checkpoint ingestion: load parameter tree, convert to float32 weight tensors, map to TPT-IR ops via model config
- [x] Implement Llamafile header-strip: detect `.llamafile` magic bytes, skip executable prefix, route to GGUF ingestion
- [x] Implement Keras `.h5` ingestion: convert via `tf.keras.models.load_model` → route to TF SavedModel path
- [x] Auto-detect input format from file extension + magic bytes (no `--format` flag required)
- [x] Unit tests: ingest one model per format, verify TPT-IR output is structurally equivalent to PyTorch baseline

### TPT Catalyst — Pre-flight Compatibility Analyzer
- [x] Define operator support matrix per hardware target (FPGA / Swarm / Analog)
- [x] Implement graph scan pass that flags unsupported operators for a given target
- [x] Suggest operator substitutions where possible (e.g., Flash Attention → standard MHA for analog)
- [x] Output structured compatibility report with pass/warn/fail per hardware type + readiness score
- [x] Expose via CLI: `tpt-catalyst check <model.tptir> --target alloy`
- [x] Surface warnings inline in Visual IR Graph Editor

### TPT Catalyst — Auto-Quantization Advisor
- [x] Define per-hardware quantization profiles (FPGA: INT8/INT4, Swarm: INT8, Analog: float/unquantized)
- [x] Implement quantization advisor pass: recommend scheme + estimated accuracy loss vs. resource tradeoff
- [x] Implement auto-apply quantization pass (rewrite TPT-IR weights to target dtype)
- [x] CLI flag: `tpt-catalyst ingest <model> --quantize auto --target fusion`
- [x] UI toggle in Observer compilation panel

### TPT Alloy — Swarm / Microcontroller Module
- [x] Integrate METIS or KaFFPa C++ graph partitioning library
- [x] Build Python bindings for the partitioning library
- [x] Implement TPT-IR → neural network graph conversion for partitioning
- [x] Implement topology-aware partitioning (accept 2D grid / star / custom wiring layout as input)
- [x] Build Rust-based per-node firmware code generator (C++/Rust output)
- [x] Integrate PlatformIO build system for targeting ESP32 / RP2040 / RISC-V
- [x] Optionally integrate Zephyr RTOS support for custom RISC-V targets
- [x] Generate master flashing script (flash all N nodes from one command)
- [x] Test partition + firmware gen against TinyLlama on 16x ESP32 swarm
- [x] CLI: `tpt-alloy partition <model.tptir> --topology grid2d --nodes 16`

### TPT Alloy — KV Cache Distribution
- [x] Design KV cache sharding scheme: each node owns KV heads for its assigned attention layers
- [x] Implement second-pass KV allocation in partition planner after layer assignment
- [x] Add per-node memory budget enforcement: block generation if KV + activations exceed node PSRAM
- [x] Stream only query/key vectors between nodes (not full cache dumps) in inter-node protocol
- [x] Add KV allocation report to pre-flight output
- [x] Test: verify no OOM on TinyLlama 16× ESP32 across a 128-token generation

### TPT Alloy — Fault-Tolerant Execution
- [x] Define heartbeat protocol: each node sends a keepalive packet every N ms to coordinator
- [x] Coordinator firmware: detect node timeout, trigger layer reassignment to neighbors
- [x] Implement degraded-mode rerouting: redistribute dead node's layers across k nearest nodes
- [x] Add `fault_tolerance` field to `topology.json` (enabled/disabled + timeout threshold)
- [x] Observer UI: dead-node heatmap — nodes color-coded green/amber/red by heartbeat status
- [x] Auto-recover: when a dead node responds again, re-integrate it and rebalance partitioning
- [x] CLI: `tpt-alloy partition ... --fault-tolerance enabled`

### TPT Alloy — Attention-Head Parallel Partitioning
- [x] Detect transformer attention layers in TPT-IR during partitioning analysis
- [x] Implement head-parallel partitioning strategy in `crates/tpt-alloy/src/partition.rs`
- [x] Implement hybrid mode: head-parallel for attention sublayers, layer-serial for FFN sublayers
- [x] Add sum-reduce handshake to firmware inter-node protocol for head aggregation
- [x] Topology Advisor: auto-recommend head-parallel strategy for transformer models
- [x] Add `--partition-strategy layer|head-parallel|hybrid` flag to Alloy CLI
- [ ] Benchmark: compare layer-wise vs. head-parallel throughput on TinyLlama swarm SiL

### TPT Alloy — Physical Topology Auto-Discovery (`alloy/auto-discovery/`)
- [x] Design broadcast protocol: each node pings all others, measures round-trip time
- [x] Nodes aggregate RTT matrix and report to coordinator node over WiFi
- [x] Alloy Python: reconstruct graph from RTT matrix (minimum spanning tree inference)
- [x] Present inferred topology to user for confirmation before partitioning begins
- [x] Observer UI: show auto-discovered topology in 3D viewer before compile
- [x] Fallback: if auto-discovery fails, fall through to manual topology input
- [x] CLI: `tpt-alloy discover --nodes 16 --timeout 30s` → outputs `topology.json`

### TPT Alloy — Pipeline Parallelism
- [x] Design pipeline scheduler: rolling token window across the node chain
- [x] Implement pipeline depth configuration in firmware generator (depth 1 = sequential)
- [x] Nodes buffer in-flight KV state for pipeline_depth tokens simultaneously
- [x] Add `pipeline_depth` field to `topology.json`; default to `min(node_count, 4)`
- [ ] Benchmark pipeline depth vs. throughput vs. PSRAM usage on SiL
- [x] Observer UI: show pipeline utilization chart (pipeline bubble %) alongside tokens/sec

### Phase 1 Milestone
- [ ] **DEMO:** Load TinyLlama → TPT Catalyst → TPT Alloy → flash 16x ESP32 → run basic chat interface

---

## Phase 2: The Silicon Canvas (Months 6–12)

### TPT Fusion — FPGA / HBM Module
- [x] Set up Amaranth HDL Python environment
- [x] Implement TPT-IR → MAC array hardware description in Amaranth HDL
- [x] Integrate Yosys for open-source logic synthesis (wrap CLI, no raw shell exposure to user)
- [x] Integrate Nextpnr for place-and-route (wrap CLI)
- [x] Build FPGA board library (starting with Xilinx Alveo)
- [x] Integrate LiteX + LiteDRAM for HBM controller generation
- [x] Implement HBM Auto-Router: auto-wire compute arrays to HBM pins given a board selection
- [x] Handle complex HBM timing constraint injection automatically
- [x] Generate synthesizable RTL (Verilog/VHDL output)
- [x] Generate memory initialization files
- [x] UI: board selector → triggers Fusion pipeline → outputs ready-to-flash bitstream
- [ ] Test end-to-end on Xilinx Alveo with a quantized AI model

### TPT Fusion — FPGA Overlay Architecture (`fusion/overlay/`)
- [x] Design overlay bitstream spec: parameterized MAC array + weight BRAM banks + HBM controller as fixed bitstream
- [x] Define `.fusecfg` config file format: datapath width, layer count, weight loading addresses
- [x] Implement overlay compiler: TPT-IR → `.fusecfg` + weight binary (no Yosys/Nextpnr invoked)
- [x] Build reference Alveo overlay bitstream (shipped with Fusion; covers INT8 + INT4 MAC configurations)
- [x] Add overlay-vs-resynthesis decision in Fusion pipeline: use overlay if target board has a pre-built overlay, else fall through to full synthesis
- [x] CLI: `tpt-fusion compile <model.tptir> --board alveo --mode overlay|full`
- [ ] Benchmark: measure overlay compile time vs. full resynthesis on same model
- [x] Observer UI: show compile mode (overlay / full synthesis) and estimated time before starting

### TPT Fusion — Fast Model Switching (Overlay Hot-Swap)
- [x] Implement hot-swap protocol: load new `.fusecfg` + weight binary into running overlay without re-flashing the bitstream
- [x] DMA weight loader: stream model weights from host NVMe → FPGA HBM over PCIe at full bandwidth; target <60s switch time for any model that fits the overlay config
- [x] Implement model config cache in HBM: reserve a fixed HBM region per slot (configurable N slots); track slot occupancy + LRU eviction when all slots full
- [x] HBM cache sizing: auto-calculate max slots from available HBM minus model weight size; expose `--cache-slots N` override in CLI
- [x] CLI: `tpt-fusion load <model.fusecfg>` — hot-load weights to a running overlay; no bitstream flash required
- [x] CLI: `tpt-fusion cache list` — show HBM slot occupancy, model name, size, last-used timestamp
- [x] CLI: `tpt-fusion cache evict <model>` — manually free an HBM slot
- [x] Observer UI: model switcher panel — list of loaded configs in HBM cache with slot indicators; one-click switch; estimated load time shown for uncached models
- [x] Define overlay compatibility check: verify `.fusecfg` datapath width and layer count match the installed overlay before attempting load; clear error if incompatible
- [ ] Benchmark: measure end-to-end model switch time (unload → transfer → load → first token) on Alveo U250

### TPT Fusion — Multi-Overlay Management
- [x] Define overlay manifest format: each overlay bitstream tagged with supported datapath (dense/MoE), precision (INT4/INT8), max layer count, max model size
- [x] Ship at least two reference overlays for Alveo U250: `dense-int4` (covers standard transformer models) and `moe-int4` (covers MoE routing with expert gating)
- [x] Implement overlay switcher: detect which overlay is currently flashed; if incoming `.fusecfg` requires a different overlay type, auto-flash the correct one (~5–10 min, user notified)
- [x] Cache current overlay type in a local state file so `tpt-fusion load` can skip the compatibility check on repeat invocations
- [x] Observer UI: show active overlay type + compatible model families in the sidebar; warn before auto-switching overlay
- [x] CLI: `tpt-fusion overlay list` — show available overlays for the connected board
- [x] CLI: `tpt-fusion overlay flash <overlay-name>` — manually flash a specific overlay

### Phase 2 Milestone
- [ ] **DEMO:** Select Xilinx Alveo in UI → TPT Fusion outputs bitstream → flash board → runs quantized AI model using HBM

---

## Automation & Pipeline Intelligence

### Toolchain Error Handling
- [x] Build error interception layer in Fusion tool wrappers (Yosys, Nextpnr): parse stderr, match against pattern catalog, emit structured error type
- [x] Build error catalog: ~30 common Yosys/Nextpnr failure patterns (timing closure, missing module, resource overflow) → plain-English messages + suggested action
- [x] Build equivalent error catalog for Xyce/ngspice (Element) and PlatformIO (Alloy)
- [x] Observer UI: show structured error message + suggested action; "raw output" toggle for advanced users
- [x] LLM fallback: if error doesn't match catalog and LLM provider is configured, send full error + context → show "AI Diagnosis" panel in Observer
- [x] CLI: structured errors also printed to stderr in JSON format for CI/CD consumption

### `tpt-doctor` Toolchain Verifier (`python/tpt_catalyst/doctor.py`)
- [x] Detect and version-check all required external tools: Yosys, Nextpnr, PlatformIO, Xyce/ngspice, Verilator
- [x] Report: installed / wrong version / missing, with platform-specific install instructions per tool
- [x] Run end-to-end smoke test: compile a minimal 2-layer model through each available hardware path
- [x] Output: green/amber/red per tool; overall readiness score
- [x] CLI: `tpt-doctor` (check all) and `tpt-doctor --target alloy|fusion|element`
- [x] Integrate into first-run wizard: auto-run `tpt-doctor` on first Observer launch, show results inline

### Quantization Auto-Search Loop (Catalyst)
- [x] Implement `--quantize auto --accuracy-budget <float>` flag in Catalyst ingest CLI
- [x] Search strategy: start all layers at INT4 → run fast SiL accuracy check or `.tptprofile` sensitivity delta → promote fragile layers to INT8 until budget is met
- [x] Add mixed-precision mode: `--quantize mixed-precision` — uses per-layer sensitivity from `.tptprofile` to assign bit-widths independently
- [x] Emit per-layer quantization decision log into `.tptpkg` for auditability
- [x] Observer UI: show per-layer quantization map (INT4 = green, INT8 = amber, float = red) after search completes

### Streaming Pre-flight + One-Click Auto-Fix
- [x] Refactor pre-flight graph scan to emit results as a stream (channel/iterator) rather than a blocking report
- [x] Go backend: expose pre-flight stream as a WebSocket event feed
- [x] Observer UI: operators flash pass/warn/fail in the Visual IR Graph Editor as they're checked
- [x] Add `fix` action to each pre-flight warning: deterministic substitutions (Flash Attention → MHA, SwiGLU → GELU, RMSNorm → LayerNorm) applied on click
- [x] Show diff preview for ambiguous substitutions; require confirmation before apply
- [x] Export fixed IR back to `.tptir` automatically after applying fixes

### Automatic Accuracy Regression on Recompile
- [x] On every `tpt-catalyst pack` invocation, check for existing `.tptpkg` for same model+target in working directory
- [x] If prior package found: auto-launch validator in background (SiL), diff accuracy vs. prior package
- [x] Observer UI: show regression badge (▲ improved / ▼ regressed / = unchanged) on new package card
- [x] Block flash (with override option) if regression exceeds configurable threshold (default 2%)
- [x] CLI: `--no-regression-check` flag to skip for CI environments that run it separately

### Parallel Firmware Flashing (Alloy OTA enhancement)
- [x] Detect WiFi-capable nodes (ESP32); default to parallel OTA broadcast when WiFi configured
- [x] USB fallback: detect all connected USB hubs, distribute node flashing across ports in parallel
- [x] Flash progress: per-node status in Observer OTA heatmap (pending → flashing → done → failed)
- [x] Estimate total flash time before starting; show progress bar with ETA
- [x] CLI: `tpt-alloy flash --parallel` (USB) / `tpt-alloy flash --ota` (WiFi)

### SiL Communication Parameter Auto-Tuner (Alloy)
- [x] Define tunable parameters: WiFi message size, batch size, retry count, UART baud rate
- [x] Implement SiL latency sweep: iterate parameter combinations, measure p99 inter-node latency in virtual swarm
- [x] Select parameter set that minimizes p99 latency within firmware memory budget
- [x] Bake tuned parameters into generated firmware as compile-time constants
- [x] Report: show tuned parameters vs. defaults and expected latency improvement in Observer
- [x] CLI: `tpt-alloy tune <model.tptpkg> --topology topology.json` → outputs `tuned_params.json`

### SPICE Dataset Auto-Generation Pipeline (Element)
- [x] Define parametric sweep space: component tolerance (±1/5/10%), temperature range (-20°C to 85°C), supply voltage variance (±10%), circuit topology variants
- [x] Implement sweep orchestrator: enumerate parameter combinations, generate SPICE netlist per combo, submit to synthesis worker queue
- [x] Worker integration: Xyce jobs run on synthesis worker cluster (same Docker infra as FPGA synthesis)
- [x] Collect results: capture output voltages, power dissipation, failure modes per run
- [x] Auto-train Reality Check ML model when dataset reaches threshold (e.g., 5,000 runs)
- [x] Ship pre-trained Reality Check checkpoint with the repo; auto-update when new dataset is larger
- [x] CLI: `tpt-element generate-dataset --runs 10000 --worker-url <url>`

### HuggingFace Model Search in Wizard (`python/tpt_catalyst/hf_search.py`)
- [x] Integrate `huggingface_hub` search API: search by name/tag, filter by model size, quantization type, task
- [x] Display results in Observer wizard step 1: model cards with size, quant type, license, download size
- [x] One-click download: fetch GGUF or SafeTensors to local model directory (shared with Spark layout)
- [x] Show download progress in wizard; auto-advance to step 2 when complete
- [x] Cache model index for offline use; refresh on demand

---

## AI Acceleration Features

### LLM Error Diagnosis (Observer + tool wrappers)
- [x] Add LLM fallback hook to error interception layer: triggered when structured catalog returns no match
- [x] Build context bundle: full stderr + model size/operator count + target board + synthesis flags
- [x] Send context to pluggable LLM provider; render response as "AI Diagnosis" panel in Observer
- [x] "Apply suggestion" button if LLM response includes a specific actionable fix
- [x] Log LLM diagnoses locally for future catalog expansion
- [x] Hidden entirely when no LLM provider is configured

### Mixed-Precision Quantization Search (Catalyst AI pass)
- [x] Extend auto-search loop to support per-layer bit-width assignment (not just uniform INT4/INT8)
- [x] Use `.tptprofile` activation sensitivity scores to rank layers by quantization fragility
- [x] Implement greedy promotion: sort layers by sensitivity, promote until accuracy budget is met
- [x] Fallback when `.tptprofile` absent: use gradient-free sensitivity estimation (small perturbation test per layer)
- [ ] Benchmark: compare uniform INT4 vs. mixed-precision on TinyLlama accuracy + compression ratio

### Synthesis Constraint Auto-Tuner (Fusion AI pass)
- [x] Define tunable synthesis parameter space: Yosys strategy (area/speed/balanced), effort level, ABC passes; Nextpnr routing seed, timing margin
- [x] Log synthesis job outcomes to synthesis worker: (parameters + model shape) → (timing slack, LUT utilization, duration)
- [x] Train regression model on accumulated job logs to predict optimal parameters given model shape + board
- [x] Integrate predictor as pre-synthesis pass: set Yosys/Nextpnr parameters automatically before invoking tools
- [x] Fallback: use sensible defaults when predictor hasn't accumulated enough data yet
- [x] Observer UI: show predicted vs. actual timing slack after each synthesis run to validate predictor

### Predictive Compile Time Estimator (Fusion + Observer)
- [x] Log synthesis job durations to synthesis worker alongside model shape metadata
- [x] Train regression model: (operator count, tensor shapes, board, synthesis mode) → predicted time (minutes)
- [x] Observer UI: show "Estimated compile time: ~X–Y min" with confidence interval before user confirms compile
- [x] Update estimate live as synthesis progresses (remaining time = predicted − elapsed)
- [x] Show historical compile times for same model+board in a tooltip

### AI-Generated Validation Prompt Suite (Validator)
- [x] Extract model domain hint from GGUF model card metadata (description, tags, model_type)
- [x] LLM prompt: "Given a model trained for [domain], generate 50 diverse test prompts that stress-test domain-specific vocabulary and edge cases"
- [x] Combine LLM-generated prompts with fixed 20-prompt standard suite
- [x] Cache generated suite per model ID to avoid re-generation on every validation run
- [x] Hidden when no LLM provider configured; falls back to standard 20-prompt suite only

### Bandwidth-Weighted Partition Graph (Alloy)
- [x] Add communication volume estimator pass before METIS: traverse TPT-IR edges, compute byte volume per edge (tensor dtype × shape × batch size)
- [x] Annotate partition graph edge weights with byte volumes
- [x] Pass weighted graph to METIS (already supports edge weights via `adjwgt` parameter)
- [x] Benchmark: compare unweighted vs. bandwidth-weighted partitioning on inter-node traffic in SiL

---

## Phase 3: The Physics Engine (Year 2)

### TPT Element — Analog Compute Module
- [x] Set up Xyce SPICE simulator (preferred) with PySpice Python wrapper
- [x] Add ngspice fallback support
- [x] Implement floating-point weight → physical component mapping (resistors, memristors, op-amp gains)
- [x] Build SPICE netlist generator from TPT-IR weights
- [x] Implement thermal noise injection in simulations
- [x] Implement voltage drift simulation
- [x] Implement component tolerance error simulation (1%, 5%, 10% tolerance profiles)
- [x] Build "Reality Check" dataset: run thousands of SPICE simulations, capture failure modes
- [x] Train lightweight PyTorch ML model on SPICE dataset to predict thermal/noise drift instantly
- [x] Integrate Reality Check model as fast inference pass (replacing slow brute-force SPICE for routine checks)
- [x] Generate human-readable mitigation suggestions (e.g., "Add heatsink to Node 4", "Use 1% resistors here")
- [x] Generate SPICE netlist output files
- [x] Generate PCB layout recommendations
- [x] Integrate KiCad file export for PCB manufacturing
- [x] Compute and output analog "Confidence Score" for each design
- [x] Test on a 3-layer analog neural network design

### Phase 3 Milestone
- [x] **DEMO:** Design 3-layer analog NN → TPT Element simulates thermal drift → outputs KiCad PCB file ready for manufacturing

---

## Phase 4: The Observer (Year 2+)

### TPT Observer — Unified Dashboard
- [x] Set up Go (Golang) backend service
- [x] Implement WebSocket server for real-time hardware telemetry streaming
- [x] Define unified telemetry data schema (tokens/s, memory bandwidth, thermal drift, node latency)
- [x] Implement FPGA telemetry adapter (memory bandwidth utilization)
- [x] Implement Analog telemetry adapter (thermal drift over time)
- [x] Implement Swarm telemetry adapter (per-node latency)
- [x] Set up React + Next.js frontend
- [x] Implement Tailwind CSS "industrial blueprint" dark theme (dark grays, neon cyan/amber, monospaced data fonts)
- [x] Build unified telemetry dashboard view (all hardware types in one UI)
- [x] Integrate Three.js / React Three Fiber for 3D swarm topology visualizer (nodes + wires)
- [x] Integrate Three.js / React Three Fiber for PCB layout visualizer (Analog module)
- [x] Build tokens-per-second live chart
- [x] Build memory bandwidth utilization live chart (FPGA)
- [x] Build thermal drift live chart (Analog)
- [x] Build node-latency heatmap (Swarm)

### Phase 4 Milestone
- [ ] **DEMO:** Single Observer dashboard shows live telemetry from FPGA, Analog, and Swarm hardware simultaneously

---

## Software-in-the-Loop (SiL) Emulator (`emulator/`)

- [x] Define unified emulator interface (same telemetry schema as real hardware)
- [x] **Alloy SiL**: Implement virtual N-node swarm simulator (message-passing, inter-node latency model)
- [x] **Fusion SiL**: Wrap Verilator for cycle-accurate RTL simulation of generated Verilog
- [x] **Element SiL**: Wire Xyce/ngspice (already in spec) as the analog emulator backend
- [x] Emulator output streams to Observer via same WebSocket telemetry path as real hardware
- [x] CLI: `tpt-emulate <compiled-output> --hardware alloy|fusion|element`

---

## Hybrid Cross-Hardware Deployment (`mosaic/`)

- [x] Define layer annotation format in TPT-IR (tag each layer with target: fpga/swarm/analog)
- [x] Implement `mosaic/` orchestrator: reads annotations, calls relevant module per partition
- [x] Define inter-hardware communication protocol (USB/UART/Ethernet bridge spec)
- [x] Generate inter-hardware glue code (data serialization between segments)
- [x] Observer: unified pipeline view showing per-segment latency across hardware types
- [x] UI: drag-and-drop layer-to-hardware assignment in Visual IR Graph Editor

---

## TPT Observer — Visual TPT-IR Graph Editor

- [x] Integrate React Flow into Observer Next.js frontend
- [x] Render TPT-IR as interactive DAG (nodes = operators, edges = tensor shapes/dtypes)
- [x] Show pre-flight compatibility warnings as inline node badges
- [x] Allow operator swap (right-click → substitute with compatible op)
- [x] Allow quantization pass insertion between nodes
- [x] Export modified IR back to `.tptir` file

---

## TPT Observer — Telemetry Replay & Time-Travel Debug

- [x] Define `.tptlog` binary format (timestamped telemetry stream + inference metadata)
- [x] Go backend: record all active telemetry streams to `.tptlog` on user request
- [x] Observer UI: replay scrub bar with per-token step navigation (pause/play/step)
- [x] Overlay mode: compare two `.tptlog` files side-by-side (e.g., before/after firmware update)
- [x] Emit replay telemetry through same Observer chart components as live data

---

## TPT Spark Integration

- [x] Detect TPT Spark model directory at startup; expose as model source in Catalyst UI
- [x] Catalyst CLI: accept Spark model ID as input (`tpt-catalyst ingest --spark-model llama3-8b`)
- [x] Define IPC/file-based protocol for Spark → Crucible model handoff
- [x] Observer: pull Spark tokens/sec baseline from local JSON conversation history
- [x] Observer: display side-by-side benchmark — Spark (GPU/CPU) vs. Crucible (custom hardware)
- [x] SiL emulator: accept Spark conversation JSON as prompt replay input for regression benchmarking
- [x] (Spark-side) Add "Compile for Custom Hardware" button to Spark sidebar — exports model to Crucible

---

## TPT Package Format (`.tptpkg`)

- [x] Define `.tptpkg` container spec and `manifest.json` schema (format version, model name, source SHA-256, targets list, per-node firmware checksums)
- [x] Implement `.tptpkg` writer in Catalyst (Rust) — ZIP container produced after compilation
- [x] Implement `.tptpkg` reader — used by each module to load its compiled artifacts at flash/deploy time
- [x] Implement SHA-256 checksumming for source model, TPT-IR, and all per-node firmware files
- [x] CLI: `tpt-catalyst pack <model.tptir> --targets alloy,fusion` → `model.tptpkg`
- [x] CLI: `tpt-catalyst unpack <model.tptpkg>` → inspect/extract contents
- [x] Update TPT Alloy to write firmware + topology.json + flash.sh into `.tptpkg` structure
- [x] Update TPT Fusion to write bitstream + mem_init + board.json into `.tptpkg` structure
- [x] Update TPT Element to write netlist.spice + pcb.kicad_pcb + confidence.json into `.tptpkg` structure
- [x] Write pre-flight report into `compat/preflight.json` in the package
- [x] Write quantization profile into `quant/quant_profile.json` in the package
- [x] Write Mosaic partition plan into `mosaic/partition.json` in the package
- [x] Observer: display `.tptpkg` manifest metadata (model name, targets, checksums, readiness score) in UI

---

## TPT Drivers — Hardware Driver SDK & Registry (`drivers/`)

- [x] Define hardware driver interface spec (Rust trait + Python protocol): board identity, pin/resource map, synthesis constraints, telemetry adapter, flash protocol
- [x] Implement driver loader: resolve drivers by name/version from local cache or registry
- [x] Build driver types: FPGA board profiles, MCU variants (ESP32, RP2040, RISC-V), analog component libraries
- [x] Migrate existing Xilinx Alveo board profile to driver format as reference implementation
- [x] Build community registry index format (TOML manifest, versioned, signed)
- [x] CLI: `tpt-drivers install <driver-name>` — downloads and caches driver from registry
- [x] CLI: `tpt-drivers list` — show installed drivers; `tpt-drivers search <query>`
- [x] Implement recipe system: `tpt-drivers install tinyllama-esp32-16node` pulls a verified topology + driver bundle
- [x] Write driver authoring docs + driver SDK template repo

### Hardware Auto-Detection (`drivers/probe/`)
- [x] Implement USB/serial device probing (udev on Linux, WMI on Windows, IOKit on macOS)
- [x] Build VID/PID → driver registry lookup
- [x] Observer UI: auto-populate board profile on device connect; prompt user to confirm
- [x] Fallback: guided manual board selection wizard if device isn't in registry

---

## Natural Language Hardware Config

- [x] Define pluggable LLM provider interface (OpenRouter, Anthropic, Ollama/OpenAI-compatible, TPT Spark IPC)
- [x] Implement LLM provider config in user settings (API key, model selection, endpoint URL)
- [x] Hide NL config feature entirely when no LLM provider is configured
- [x] Implement structured topology JSON generation prompt + schema validation
- [x] Observer UI: free-text input field → confirmed topology preview in Visual IR Graph Editor
- [x] Save natural language description in `.tptpkg` alongside `topology.json` (for reproducibility)
- [x] Test against: OpenRouter (cloud), Ollama (local), TPT Spark (local GGUF via IPC)

---

## Hot Recompilation (TPT Catalyst)

- [x] Implement content-addressed cache: per-operator hash keyed on TPT-IR subgraph
- [x] Store cache in `.tpt-cache/` directory adjacent to working `.tptpkg`
- [x] CLI flag: `tpt-catalyst pack --incremental` - skip operators whose hash matches cache
- [x] Observer UI: show per-layer cache hit/miss indicators during compilation
- [x] Cache invalidation: bust cache for a layer when its inputs, weights, or target hardware change

---

## Hardware-Aware Training Hooks (`tpt-train/`)

- [x] Implement `TPTProbeCallback` for PyTorch: attaches to all layers, records min/max activations, weight histograms, gradient norms per epoch
- [x] Implement equivalent JAX/Flax hook
- [x] Output: `model.tptprofile` JSON (per-layer activation stats + weight distributions)
- [x] Catalyst integration: if `.tptprofile` exists alongside model, use it to set per-layer quantization clamps
- [x] Auto-Quantization Advisor: prefer `.tptprofile` data over static weight analysis when available
- [x] Add `.tptprofile` reference to `.tptpkg` manifest if profile was used during compilation
- [x] Publish `tpt-train` as a standalone pip package (`pip install tpt-train`)

---

## Cloud Synthesis Worker — Self-Hostable (`cloud/synthesis-worker/`)

- [x] Go worker service: accept `.tptpkg` upload, run Yosys + Nextpnr, return updated `.tptpkg` with bitstream
- [x] Redis-based job queue (stateless workers, horizontally scalable)
- [x] Observer UI: "Offload synthesis to worker" toggle (shown only when a worker URL is configured in settings)
- [x] Dockerfile + Docker Compose for single-node deployment
- [x] Worker deployment docs

---

## TPT Crucible Cloud — Self-Hostable Full Pipeline (`cloud/crucible-cloud/`) *(optional/bonus)*

- [x] Go API server: model upload, compilation job management, `.tptpkg` download endpoints
- [x] Containerized Catalyst + module workers (one image per hardware target type)
- [x] Minimal Next.js web UI: upload model → select target → track job → download `.tptpkg`
- [x] Docker Compose stack for self-hosted single-server deployment
- [x] Helm chart for Kubernetes deployment
- [x] Deployment and configuration docs

---

## AI Driver Generator (`drivers/ai-gen/`)

- [x] Implement PDF/URL datasheet text extractor (PyMuPDF for PDF, BeautifulSoup for URLs)
- [x] Define structured LLM extraction prompt: extract pinout, memory map, peripheral specs, flash protocol, clock/timing
- [x] LLM output → driver manifest TOML + Rust trait skeleton + synthesis constraints + flash protocol stub
- [x] Observer UI: diff-style preview of generated driver; user edits and approves
- [x] Run SDK schema validator on generated driver before allowing publish to registry
- [x] "Publish to Registry" flow from the review screen
- [x] LLM backend: uses same pluggable provider interface as NL Hardware Config

---

## AI Swarm Topology Advisor (`alloy/ai-topology/`)

- [x] Define input schema: TPT-IR profile (layer count, bandwidth matrix) + user constraints (node count, latency budget, power budget, form factor)
- [x] Implement LLM-based topology recommendation (initial approach)
- [x] Define training data schema: (model profile + constraints + topology) → measured SiL performance
- [x] Accumulate SiL run results as training data automatically
- [x] Train ML model on accumulated SiL data when dataset is large enough; swap in as default
- [x] Output: ranked topology recommendations (ring/mesh/star/tree/hybrid) with predicted latency + power
- [x] Observer UI: 3D preview of each recommended topology; one-click "use this" to feed into Alloy partitioner

---

## AI RTL Assistant (`fusion/ai-rtl/`)

- [x] Implement compute pattern extractor from TPT-IR (layer types, tensor shapes, dtypes, repetition count)
- [x] Build LLM prompt template: compute pattern + board constraints → candidate Verilog MAC array
- [x] Implement static timing pre-check on generated Verilog (wrap OpenTimer or `yosys stat`)
- [x] Flag timing violations with suggested datapath modifications before full synthesis
- [x] Observer UI: generated RTL shown in Visual IR Editor for review before entering Fusion pipeline
- [x] Fallback: if no LLM configured, Fusion uses existing Amaranth HDL template generation (no regression)

---

## AI Analog Circuit Designer (`element/ai-circuit/`)

- [x] Define training data schema: (target op spec: matrix size, precision, activation) ↔ (netlist + SPICE-measured accuracy)
- [x] Build dataset generation pipeline: enumerate known analog implementations, run SPICE, record results
- [x] Implement retrieval-augmented circuit generator (Phase 1): find closest circuit in dataset, adapt component values
- [x] Integrate Reality Check model as fast validation pass on generated candidates
- [x] Implement iterative refinement loop: generate → validate → perturb if low confidence → repeat
- [x] Output: top-N ranked candidate circuits with confidence scores and predicted failure modes
- [x] Observer UI: side-by-side candidate circuit viewer with confidence scores
- [ ] Phase 2 (later): train generative model (VAE or diffusion over circuit graph space) to replace retrieval approach

---

## Bill of Materials (BOM) Generator

- [x] Add `[bom]` section to driver manifest spec: part numbers, quantities, supplier SKUs (DigiKey, Mouser, LCSC)
- [x] Populate `[bom]` for all first-party drivers (ESP32, RP2040, Xilinx Alveo, etc.)
- [x] Compilation pipeline: extract BOM from all used drivers, write `bom/parts.json` + `bom/parts.csv` into `.tptpkg`
- [x] Element BOM: extract component list from generated SPICE netlist (resistor values, tolerances, memristor specs)
- [x] Observer UI: "BOM" tab with parts list and supplier links; one-click CSV export
- [x] JLCPCB integration: generate JLCPCB-compatible BOM + CPL from Element's KiCad output for automated PCB assembly quotes

---

## Hardware Cost Estimator

- [x] Add `[pricing]` section to driver manifest: typical unit price range, supplier URL
- [x] Observer UI: cost estimate shown on target selection screen before compilation
- [x] Estimate breakdown: component cost × node count + PCB fab estimate + board cost
- [x] "Cheap / Medium / Expensive" tier badge on each hardware target card
- [x] Link to actual supplier pages; display caveat that prices are estimates

---

## First-Run Guided Wizard (Observer UI)

- [x] Implement 5-step wizard flow in Observer shown on first launch
- [x] Step 1: model picker (file, Spark model selector, HuggingFace URL)
- [x] Step 2: auto-run pre-flight check; show traffic-light results per hardware type
- [x] Step 3: hardware picker with cost estimates + BOM previews; "no hardware yet" path → recommend ESP32 swarm + SiL
- [x] Step 4: compilation with auto-quantization on by default
- [x] Step 5: "Flash or Emulate?" — offer flash if USB device detected, otherwise launch SiL
- [x] Save wizard state; make it skippable and re-launchable from help menu

---

## Model Accuracy Validator (`validator/`)

- [x] Define standardized prompt suite for accuracy testing (diverse token types, edge cases)
- [x] Implement reference backend connector: Spark IPC or local CPU inference
- [x] Implement hardware output connector: reads inference results from live deployment or SiL
- [x] Compute token-level similarity + perplexity delta between hardware and reference outputs
- [x] For analog: report per-layer output voltage vs. SPICE-expected value
- [x] Observer UI: accuracy dashboard tab with per-layer green/amber/red indicators
- [x] CLI: `tpt-validate <model.tptpkg> --reference spark --hardware alloy`

---

## OTA Update System (`alloy/ota/`)

- [x] Implement per-node firmware binary diff between new and previous `.tptpkg`
- [x] Generate patch manifest: list of node IDs with changed binaries
- [x] OTA flashing: push firmware only to changed nodes; unchanged nodes remain live during update
- [x] Store previous firmware in `targets/alloy/firmware/prev/` inside `.tptpkg` for rollback
- [x] Observer UI: OTA progress heatmap — per-node status (pending / flashing / done / failed)
- [x] One-click rollback from Observer UI
- [x] CLI: `tpt-alloy ota --pkg new.tptpkg --prev old.tptpkg --topology topology.json`

---

## Power Consumption Estimator + Monitor

- [x] Add `[power]` section to driver manifest: idle mW, active mW/MHz, peak mW
- [x] Pre-flight: compute total estimated power draw (active mW × node count + overhead)
- [x] Include power estimate in pre-flight report and BOM output
- [x] Observer telemetry: add optional power monitoring channel (INA219 or similar; shown only if hardware supports it)
- [x] Element: extract power estimate from Xyce SPICE simulation results (already available)

---

## Hardware Diagnostics Mode

- [x] Implement diagnostic test pattern runner (known-good inputs + expected outputs per hardware type)
- [x] **Alloy diagnostics**: ping each node, measure RTT latency, check firmware version, report CPU temp
- [x] **Fusion diagnostics**: run small test inference through FPGA; verify output vs. golden reference; check HBM bandwidth
- [x] **Element diagnostics**: inject low-amplitude test signal; compare output to SPICE-predicted response; flag components outside tolerance
- [x] Observer UI: hardware health heatmap — node grid (swarm), block diagram (FPGA), circuit diagram (analog), color-coded by health
- [x] CLI: `tpt-diagnose <model.tptpkg> --hardware alloy|fusion|element`

---

## Hardware-in-the-Loop Training (`tpt-train/hardware_aware.py`)

- [x] Design `TPTHardwareAwareCallback`: accepts hardware telemetry + CPU reference outputs, computes per-layer deviation profile
- [x] Implement deviation-to-loss conversion: map per-layer output errors to a regularization term
- [x] Wire callback into PyTorch training loop (runs after N steps when hardware telemetry is available)
- [x] Validator integration: `tpt-validate` outputs a `.tptdeviation` file consumable by the callback
- [x] Document workflow: train → deploy to SiL/hardware → collect deviations → fine-tune → recompile
- [ ] Test: measure accuracy delta before/after hardware-aware fine-tuning on TinyLlama + 16× ESP32 SiL

---

## Accessibility & Democratization

### WASM Browser Demo
- [x] Configure Cargo targets to compile `tpt-catalyst` and `tpt-alloy` to WASM (`wasm32-unknown-unknown`)
- [x] Expose WASM API: `compile(modelBytes, targetConfig)` → `.tptpkg` binary
- [x] Wrap WASM in a Web Worker (non-blocking UI)
- [x] Wire WASM SiL output to Observer frontend via in-browser WebSocket mock
- [x] Build hosted demo page: drag-and-drop model file → compile → watch SiL swarm visualization
- [x] Target: <60 seconds from page load to running SiL visualization for a quantized TinyLlama

### Pre-compiled Package Marketplace (`tpt-packages/`)
- [x] Define package registry manifest format (JSON index: model ID, hardware target, accuracy delta, SHA-256, download URL)
- [x] CLI: `tpt get <package-name>` — downloads and verifies `.tptpkg` from registry
- [x] CLI: `tpt packages list` — browse available pre-compiled packages
- [x] Bootstrap registry with: TinyLlama Q4 × 16× ESP32, TinyLlama Q8 × Alveo
- [x] Observer UI: "Get pre-compiled package" option on target selection screen
- [x] Publish registry as static files on GitHub Releases (no server required)

### Open Hardware Reference Board (`hardware/reference-designs/alloy-carrier/`)
- [x] Design 8-node ESP32 carrier board in KiCad: standardized bus connectors, shared 5V rail, UART headers, per-node LEDs
- [x] Generate JLCPCB-compatible BOM + CPL (target: < $15 assembled per board)
- [x] Write assembly guide (step-by-step, photos, ESP32 pinout reference)
- [x] Validate board design against auto-discovery protocol and heartbeat firmware
- [x] Publish KiCad source + Gerbers in repo; link from Observer first-run wizard

### One-Line Bootstrap / Developer Experience
- [x] Restructure Python packaging: `pip install tpt-crucible` (base, SiL-only) + `[fpga]` and `[swarm]` extras
- [x] First-run detection: if `[fpga]` extra not installed but FPGA target selected, show install instructions + SiL fallback offer
- [x] Bundle `tpt-catalyst` and `tpt-alloy` WASM binaries in the pip package for offline browser demo
- [x] Write 5-minute quickstart: `pip install tpt-crucible` → download TinyLlama → SiL run → see tokens/sec
- [x] Test install experience on clean Windows, macOS, and Ubuntu VMs

### Docker-First Distribution (Setup Friction Reduction)
- [x] Build `tpt-crucible/synthesis` Docker image: Yosys + Nextpnr + MLIR + full toolchain pre-installed; no host toolchain required
- [x] Build `tpt-crucible/runtime` Docker image: minimal XRT + Crucible runtime for machines with a physical FPGA card
- [x] Write XRT one-line kernel module installer script (`install-xrt.sh`): handles Ubuntu version detection, kernel header install, XRT deb install, card detection verify; target <15 min on a clean Ubuntu LTS
- [x] Publish both images to Docker Hub; version-pin to matching XRT + Yosys releases to eliminate version mismatch errors
- [x] Update `tpt-doctor` to detect Docker-based toolchain as valid alternative to host-installed tools
- [x] Observer UI: "Run synthesis in Docker" toggle — shown when Docker is detected but host toolchain is absent; routes synthesis jobs through the container transparently
- [x] Document Docker path as the recommended setup route in quickstart guide

### Windows-Native Support
- [x] Bundle pre-compiled Yosys and Nextpnr Windows binaries in the `[fpga]` pip extra — eliminate build-from-source requirement on Windows
- [x] Implement WSL2 auto-setup helper: detect WSL2 availability, install Ubuntu 22.04 distro, configure XRT inside WSL2, verify card passthrough; one command from PowerShell
- [x] Test and document Alveo U250 XRT passthrough on Windows 11 via WSL2
- [x] Write Windows-specific quickstart guide covering Docker Desktop path as primary and WSL2 as alternative
- [x] CI: add Windows install smoke test (pip install + tpt-doctor) to CI pipeline

---

## Infrastructure & Cross-Cutting

- [x] Initialize mono-repo (Rust workspace + Python packages + Go service + Next.js app)
- [x] Set up CI/CD pipeline (lint, test, build for all components)
- [x] Write developer onboarding docs / CONTRIBUTING.md
- [x] Set up docs site (architecture overview, module API references)
- [x] Define versioning and release strategy (open-core vs. proprietary layers)
- [x] License: choose open-core licensing (e.g., Apache 2.0 for compilers, commercial for optimization layers)

---

## Gap Features — Competitive Differentiation

### Gap 1: TPT Silicon — Compute-in-Memory Backend (`python/tpt_silicon/`)

No general-purpose compiler exists for arbitrary model → CIM. Targets Taalas HC1, Axelera Europa, D-Matrix DIMC.

- [x] Create `python/tpt_silicon/` module (pyproject.toml + package scaffold)
- [x] `weight_packer.py` — `CimWeightPacker`: quantize + tile weight tensors into memory array row format (row = single MAC unit); `PackedArray` dataclass; `serialize_array()` → bytes
- [x] `array_layout.py` — `CimArrayLayout`: map TPT-IR ops to array dimensions; tile large matmuls across multiple physical arrays; `LayoutConfig` dataclass (array_rows, array_cols, bit_precision)
- [x] `bitline.py` — `BitlineOpGenerator`: emit bitline read/accumulate/ADC-skip op sequences as low-level op list; handle partial-row masking for sparse tiles
- [x] `package_writer.py` — `write_silicon_artifacts()`: write `targets/silicon/` into `.tptpkg` (weight_arrays.bin, layout.json, config.json)
- [x] `cli.py` — `tpt-silicon compile <model.tptir> --board <name> [--precision 4|8]`
- [x] Add `HardwareTarget.CIM` to `python/tpt_mosaic/tpt_mosaic/partition.py`
- [x] Dispatch CIM target in `python/tpt_mosaic/tpt_mosaic/orchestrator.py`
- [x] Add CIM column to operator support matrix in `python/tpt_catalyst/tpt_catalyst/compat.py`
- [x] Add `"cim"` hardware_type + `CimArraySpec` dataclass to `python/tpt_drivers/tpt_drivers/driver.py`
- [x] Add `"silicon"` to target name set in `crates/tpt-catalyst/src/package.rs`
- [x] Observer UI: CIM target card in hardware picker with array utilization metric

### Gap 2: TPT Pulse — Neuromorphic / ANN→SNN Compiler (`python/tpt_pulse/`)

No production open-source ANN→SNN compiler exists. Targets Intel Loihi (lava-nc) and BrainScaleS (PyNN).

- [x] Create `python/tpt_pulse/` module scaffold
- [x] `lif_node.py` — `LifNeuron` dataclass (threshold, decay, reset_mode: subtract|zero); `SnnGraph` (nodes + spike_edges)
- [x] `converter.py` — `SnnConverter`: walk TPT-IR ops; replace ReLU activations with LIF nodes; normalize thresholds via `.tptprofile` max-activation if available, else weight-norm heuristic
- [x] `sim_export.py` — `SimExporter`: pure-Python LIF simulation for SiL testing without neuromorphic hardware
- [x] `package_writer.py` — write `targets/pulse/` (snn_graph.json + backend export + accuracy_estimate.json)
- [x] `cli.py` — `tpt-pulse convert <model.tptir> --target loihi|brainscales|sim`
- [x] Add `HardwareTarget.NEUROMORPHIC` to partition.py; dispatch in orchestrator.py
- [x] Add `"neuromorphic"` hardware_type + `NeuromorphicSpec` to driver.py
- [x] Add neuromorphic column to compat.py operator support matrix
- [x] Observer UI: neuromorphic target card; show spike rate + accuracy estimate metrics

### Gap 3: Carbon-Aware Compilation

No compiler offers a `--optimize carbon` flag. Power data already in driver manifests; adds grid carbon intensity + cost function.

- [x] `python/tpt_catalyst/tpt_catalyst/carbon.py` — `GRID_INTENSITY_GCO2_PER_KWH` region map; `CarbonEstimate` dataclass; `estimate_carbon(target, driver, inference_time_s, region)`; `select_lowest_carbon_target(estimates)`
- [x] Add `--optimize carbon` and `--carbon-region <region>` flags to `tpt-catalyst ingest` and `tpt-catalyst check` CLI
- [x] Include `CarbonEstimate` per target in `CompatibilityReport` output
- [x] Add optional `carbon_overhead_gco2: float` to `PowerProfile` in driver.py (embodied carbon amortisation)
- [x] Add `carbon_profile: Option<CarbonProfile>` to `PackageManifest` in `crates/tpt-catalyst/src/package.rs`
- [x] Observer UI: carbon cost column in hardware picker table; "lowest carbon" badge on recommended target
- [x] CLI: `tpt-catalyst check model.tptir --optimize carbon --carbon-region eu` prints ranked carbon table

### Gap 4: Cross-Hardware Speculative Decoding (`python/tpt_mosaic/tpt_mosaic/speculative.py`)

Draft model on Alloy swarm + verify model on Fusion FPGA. Standard speculative decoding loop across physically separate hardware — native to Mosaic, impossible to replicate on single-hardware stacks.

- [x] `speculative.py` — `SpeculativeConfig` dataclass (draft_pkg, verify_pkg, gamma=4, acceptance_threshold=0.8); `SpeculativeOrchestrator.run(prompt_tokens)` async loop; `get_metrics()` → `SpeculativeMetrics`
- [x] Implement standard token acceptance criterion (probability ratio reject/accept) over Mosaic bridge
- [x] Add `run_speculative(config)` method to `MosaicOrchestrator`
- [x] Add `tpt-mosaic speculative --draft alloy.tptpkg --verify fusion.tptpkg --gamma 4` CLI command
- [x] Add `SpeculativeMetrics` struct to Observer telemetry schema (`services/tpt-observer/internal/telemetry/schema.go`)
- [x] Observer UI: speculative decoding dashboard panel — acceptance rate gauge, draft vs. effective TPS comparison

### Gap 5: Hardware-Locked Model IP Protection

Cryptographically bind a `.tptpkg` to specific hardware serial numbers. Package refuses to load on mismatched hardware.

- [x] `python/tpt_catalyst/tpt_catalyst/ip_lock.py` — `HardwareLock` dataclass (fingerprint_sha256, lock_type, locked_at, issuer); `create_lock(hardware_ids)`; `verify_lock(lock, present_ids)`
- [x] `crates/tpt-catalyst/src/ip_lock.rs` — `HardwareLock` struct; `verify_lock()` called in `PackageReader::open()` before returning contents
- [x] Add `hardware_lock: Option<HardwareLock>` to `PackageManifest` in package.rs
- [x] Add `--lock-to-hardware <id1,id2,...>` flag to `tpt-catalyst pack` CLI
- [x] Alloy firmware generator: embed fingerprint verification at boot before inference (reads node serial from efuse/OTP)
- [x] Observer UI: lock icon + hardware IDs in package manifest view; "Locked" badge on locked packages
- [x] CLI: `tpt-catalyst unpack` prints lock status; errors clearly on fingerprint mismatch

### Gap 6: Structured Sparsity Exploitation

2:4 sparsity (≥2 zeros per 4 weights) lets FPGA MAC arrays skip zero multiplications, halving compute. Source: `.tptprofile` activation_sparsity per layer.

- [x] `python/tpt_catalyst/tpt_catalyst/sparsity.py` — `SparsityMode` enum (NONE, TWO_FOUR, FOUR_EIGHT, AUTO); `SparsityPattern` dataclass; `SparsityAnalyzer.analyze(ir, profile)`; `enforce_2_4(weights)` pruning function
- [x] `python/tpt_fusion/tpt_fusion/sparse_mac.py` — `SparseMacArray`: Amaranth HDL with skip-zero gating; compressed index sidecar for routing non-zero values
- [x] Add `sparsity_map: dict[str, SparsityPattern]` to `QuantizationProfile` in `python/tpt_catalyst/tpt_catalyst/quantize.py`
- [x] Extend `MacConfig` with `sparsity_mode: SparsityMode`; dispatch to `SparseMacArray` in `mac_array.py`
- [x] Add `--sparsity auto|2:4|4:8|none` flag to `tpt-catalyst ingest` CLI
- [x] Observer UI: per-layer sparsity map in IR Graph Editor (density heatmap overlay)

### Gap 7: Community Compilation Cache

FPGA synthesis takes hours. A content-addressed public registry turns repeated compilations of common model+board combos from hours to seconds.

- [x] `python/tpt_catalyst/tpt_catalyst/community_cache.py` — `CommunityCacheClient.lookup(model_sha256, board, synthesis_flags)`; `publish(tptpkg_path, ...)`; local index cache with 1h TTL
- [x] `cloud/synthesis-broker/internal/cache/cache.go` — `CacheClient` struct; `Lookup(modelSHA, board, flagsHash)` called before synthesis job dispatch in broker.go
- [x] Define `community_cache_index.json` schema: array of `{model_sha256, board, flags_hash, download_url, verified_at, accuracy_delta}`
- [x] Add `--community-cache` / `--no-community-cache` flags to `tpt-catalyst pack` CLI
- [x] Integrate `CacheClient.Lookup()` into `cloud/synthesis-broker/internal/broker/broker.go` dispatch path
- [x] Bootstrap index with TinyLlama Q4 × Alveo U250 entry on GitHub Releases
- [x] Observer UI: "Using community build (saved ~4h synthesis)" notification on cache hit

### Gap 8: RISC-V Custom ML ISA Generation (`python/tpt_alloy/tpt_alloy/riscv_isa.py`)

Generate per-model RISC-V custom instructions (using RISC-V custom opcode spaces) as Chisel source, then synthesize to soft-core RISC-V FPGA — bridges Alloy and Fusion.

- [x] `riscv_isa.py` — `RiscVCustomOp` dataclass (mnemonic, opcode_space, funct3, funct7, latency_cycles); `RiscVExtensionGenerator.analyze(ir)` profiles op frequency; `generate_chisel(ops)` emits Chisel3 VexRiscV plugin source; `generate_gnu_binutils_patch(ops)` emits GAS assembler extension
- [x] Add `tpt-alloy riscv-isa <model.tptir> [--top-n 8] --output <dir>` subcommand to Alloy CLI
- [x] Add `--riscv-ext <custom_ext.scala>` flag to `tpt-fusion generate` CLI; include extension in soft-core synthesis when provided
- [x] Observer UI: RISC-V ISA panel showing generated instructions with estimated speedup per op

### Gap 9: TPT Photon — Photonic Backend Stub (`python/tpt_photon/`)

Experimental forward-looking backend for silicon photonics inference (MZI mesh). No open-source compiler exists. Targets Lightmatter Passage, LightOn OPUs. Marked EXPERIMENTAL throughout.

- [x] Create `python/tpt_photon/` module scaffold
- [x] `mzi_mesh.py` — `MziMeshGenerator`: SVD decompose weight matrix (Clements decomposition); `phase_encode(U)` → MZI phase angles (radians); `MziConfig` dataclass (mesh_size, phase_angles, nonlinearity)
- [x] `package_writer.py` — write `targets/photon/` (phase_config.json, mzi_layout.json, EXPERIMENTAL marker)
- [x] `cli.py` — `tpt-photon compile <model.tptir>` (prints EXPERIMENTAL warning prominently)
- [x] Add `HardwareTarget.PHOTONIC` (experimental flag) to partition.py; dispatch in orchestrator.py
- [x] Add `"photonic"` hardware_type + `PhotonicSpec` (mesh_size, wavelength_nm, modulation) to driver.py
- [x] Observer UI: photonic target card with EXPERIMENTAL badge; phase angle visualizer

### Gap 10: Intermittent / Energy-Harvesting Computing Support

Batteryless sensors lose power mid-inference. Checkpoint ops in TPT-IR let firmware save and resume state across power cycles.

- [x] `python/tpt_catalyst/tpt_catalyst/intermittent.py` — `CheckpointGranularity` enum (LAYER, BLOCK, OPERATOR); `IntermittentProfile` dataclass; `CheckpointPlanner.insert_checkpoints(ir, profile)`; `estimate_energy_per_layer(ir, driver)`; `validate_budget(ir, profile)` → budget warnings
- [x] Add `--intermittent`, `--checkpoint-granularity layer|block|operator`, `--energy-budget-mj <float>` flags to `tpt-catalyst ingest` CLI
- [x] Alloy firmware generator: recognise `tpt.checkpoint` ops; emit EEPROM write/read around boundaries; add power-monitor ISR hook (GPIO voltage-drop interrupt)
- [x] Add `checkpoint_storage: str | None` and `power_monitor_pin: str | None` to `DriverManifest` in driver.py
- [x] Observer UI: energy budget progress bar; checkpoint marker overlay on IR graph timeline; estimated inferences per harvest cycle

### Gap Features — Cross-Cutting Changes

- [x] `python/tpt_mosaic/tpt_mosaic/partition.py` — extend `HardwareTarget` enum: add CIM, NEUROMORPHIC, PHOTONIC
- [x] `python/tpt_drivers/tpt_drivers/driver.py` — add `CimArraySpec`, `NeuromorphicSpec`, `PhotonicSpec` dataclasses; extend hardware_type literals; add `carbon_overhead_gco2` to `PowerProfile`; add `checkpoint_storage` + `power_monitor_pin` to `DriverManifest`
- [x] `crates/tpt-catalyst/src/package.rs` — add `hardware_lock: Option<HardwareLock>` and `carbon_profile: Option<CarbonProfile>` to `PackageManifest`
- [x] `services/tpt-observer/internal/telemetry/schema.go` — add `SpeculativeMetrics` and `CarbonMetrics` structs

---

## Release Readiness — Stub Fixes, Frontend Integration & Security

### Backend Stub Fixes

- [x] **A1** `crates/tpt-catalyst/src/wasm_api.rs` — fix `check_compatibility()`: parse `ir_json` as `TptIr`, call real compat logic, remove hardcoded stub return; added per-target unsupported op lists and 50 MB input cap
- [x] **A2** `crates/tpt-catalyst/src/package.rs` — fix `PackageBuilder::build()`: ZIP-compress staging directory into `.tptpkg` using `zip` crate; path-traversal guard rejects `..` and absolute paths
- [x] **A3** `crates/tpt-alloy/src/firmware.rs` — replaced `// inference loop placeholder` in ESP32/RP2040/RISC-V generators with real per-layer dispatch loop + `tpt_sync_neighbors()`
- [x] **A4** `python/tpt_catalyst/tpt_catalyst/optimizer.py` — implemented `_tvm_optimize()`: `FuseOps` + `EliminateCommonSubexpr` + `SimplifyInference` + `FoldConstant` via `tvm.transform.Sequential`; falls back to builtin optimizer on any exception
- [x] **A5** `python/tpt_alloy/tpt_alloy/cli.py` — removed hardcoded `100` layer count; added `_layer_count_from_ir()` reading actual node count from `.tptir` JSON
- [x] **A6** `python/tpt_alloy/tpt_alloy/firmware.py` — implemented eFuse/OTP hardware-lock reads: RP2040 (`flash_get_unique_id`), RISC-V (SiFive OTP MMIO), Zephyr (`hwinfo_get_device_id`); all generators emit inference dispatch loop
- [x] **A7** `cloud/synthesis-worker/internal/job/job.go` — replaced `echo` stub in `runAlloy()` with real `platformio run` invocation with separate arg slices; validates `firmware.bin` artifact exists after compilation
- [x] **A8** `cloud/crucible-cloud/internal/jobs/manager.go` — fixed `SaveManifest()` to populate `model_name`, `targets`, and `created_at` from the `Job` struct

### Frontend — Backend Integration

- [x] **B1** `frontend/next.config.ts` + `frontend/.env.local.example` — added `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_WS_URL` env vars; replaced all hardcoded `localhost:8080` references
- [x] **B2** `frontend/src/contexts/TelemetryContext.tsx` — created telemetry context with WebSocket connection, auto-reconnect, full message handler for all telemetry types; wired into `app/page.tsx`
- [x] **B3** `TelemetryCharts` → `TelemetryContext` (no more `Math.random()`); `DownloadPanel` → `GET /api/packages/{id}/artifacts`; static fallbacks for remaining sample-data components
- [x] **B4** IR Graph Editor: Export `.tptir` (`POST /api/ir/export`), Save Changes (`PUT /api/ir/current`); Cloud page: Start Compilation (`POST /api/jobs`) with real polling
- [x] **B5** `frontend/src/components/ErrorBoundary.tsx` — React error boundary with blueprint-styled fallback and "Reload component" button
- [x] **B6** `frontend/src/app/not-found.tsx` — blueprint-styled 404; `frontend/src/app/settings/page.tsx` — LLM provider config with Zod-validated localStorage
- [x] **B7** `frontend/src/components/Toast.tsx` — notification queue with 4s auto-dismiss; dark/light mode toggle in `Sidebar.tsx` with localStorage persistence; `frontend/src/app/jobs/page.tsx` — job history with status badges + download links; `/jobs` and `/settings` links added to Sidebar

### Security Fixes

- [x] **C1** `TelemetryContext.tsx` — WS URL from `NEXT_PUBLIC_WS_URL` env var (no hardcoded `localhost`); uses `wss://` when appropriate
- [x] **C2** `frontend/src/components/SetupWizard.tsx` — added `WizardStateSchema` (Zod); `loadWizardState()` validates parsed JSON and calls `localStorage.removeItem` on failure
- [x] **C3** `frontend/src/app/cloud/page.tsx` — file type allowlist (`.gguf,.pt,.onnx,.tflite,.safetensors`) + 10 GB size guard; inline error shown on rejection
- [x] **C4** `cloud/synthesis-worker/internal/job/job.go` — `runAlloy()` uses separate arg slices (`exec.Command("platformio", "run", "--project-dir", ...)`) matching the safe pattern in `runFusion()`
- [x] **C5** `services/tpt-observer/internal/ws/hub.go` — `allowedOrigin()` replaces `return true`; allows `localhost` in dev, requires `ALLOWED_ORIGIN` env var in prod
- [x] **C6** `services/tpt-observer/internal/ws/hub.go` — per-IP connection limiter via `sync.Map` + `atomic.AddInt32`; rejects upgrades beyond 5 concurrent per IP with HTTP 429
- [x] **C7** `crates/tpt-catalyst/src/wasm_api.rs` — 50 MB input size cap + successful deserialization check before processing; structured error JSON returned on failure
- [x] **C8** `crates/tpt-catalyst/src/package.rs` — `write_dir_to_zip()` rejects entries with `..` or leading `/`/`\`

### GitHub Release Preparation

- [x] **C9** `LICENSE` — copyright updated from `2024 TPT Crucible Contributors` → `2026 TPT Solutions`
- [x] **D1** `CHANGELOG.md` — created; Keep a Changelog format; `[0.1.0] — 2026-06-29` section covering all modules
- [x] **D2** `SECURITY.md` — created; supported versions, `security@tpt.solutions` reporting, 90-day coordinated disclosure, known mitigations
- [x] **D3** `CONTRIBUTING.md` — updated GitHub org URL to `tpt-solutions`; removed stale `.github/workflows` reference
- [x] **D4** `README.md` — created; Apache 2.0 + version badges; 5-minute quickstart above the fold; module table, architecture diagram, Spark integration
- [x] **D5** `Cargo.toml` — added `authors = ["TPT Solutions <contact@tpt.solutions>"]`; updated `repository` to `tpt-solutions/tpt-crucible`; `quickstart.py` URL updated

### Innovation / UX

- [x] Toast notification system (`Toast.tsx`) for background operation feedback (OTA, synthesis, flashing)
- [x] Job history page (`/jobs`) listing past compilations with status badges and re-download links
- [x] Dark/light mode toggle in `Sidebar.tsx` with `data-theme` attribute and localStorage persistence
- [x] `/jobs`, `/settings`, `/compare`, `/tournament`, and `/provenance` navigation links added to Sidebar
- [ ] Command palette (`Ctrl+K`) — deferred; depends on `cmdk` package addition

---

## Innovation Pipeline — Phase 2

### Feature 1 — Cross-Hardware Benchmark Comparison Report

- [x] `python/tpt_catalyst/tpt_catalyst/compare.py` — `ComparisonConfig` dataclass (targets list, constraint budget); `ComparisonRunner.run()` orchestrates SiL runs per target via `tpt-emulate`; `ComparisonReport` dataclass (per-target: tokens_sec, power_mw, cost_per_inference, carbon_gco2, accuracy_delta); `select_recommended()` picks best fit given constraints
- [x] Add `compare/report.json` as optional artifact written into `.tptpkg` by `PackageBuilder`
- [x] CLI: `tpt-catalyst compare <model.tptir> --targets all|alloy,fusion,element --max-latency Xms --max-power XW`
- [x] `frontend/src/app/compare/page.tsx` — comparison page: interactive Pareto scatter (canvas-based, TPS × latency); "Recommended" banner; per-target detail cards; "Compile" link to `/cloud?target=X`
- [ ] Add "Compare All Targets" button to Observer dashboard and cloud compilation flow
- [ ] Community cache integration: call `CommunityCache.lookup()` before each SiL run to skip redundant synthesis

### Feature 2 — Compilation Tournament / Pareto Optimizer

- [x] `python/tpt_catalyst/tpt_catalyst/tournament.py` — `TournamentConfig`; `TournamentRunner.run()` sweeps quant × target × synthesis × node_count; `ParetoPoint` dataclass; Pareto front computation; recommended config selection
- [x] CLI: `tpt-catalyst tournament <model.tptir> --max-latency 50ms --max-power 5W --min-accuracy 0.90`
- [x] `frontend/src/app/tournament/page.tsx` — interactive Pareto scatter; target/quant/synth toggle chips; recommended config banner; "Compile with this config →" link pre-fills cloud form
- [ ] Persist tournament results to `tournament/results.json` in `.tptpkg`

### Feature 3 — Telemetry Anomaly Detection + Predictive Maintenance

- [x] `python/tpt_drivers/tpt_drivers/anomaly.py` — `AnomalyDetector` with sliding-window analysis; `AnomalyAlert` dataclass (node_id, metric, severity: warn|critical, predicted_tta_minutes, suggested_action); thermal slope regression for TTA estimation; per-metric threshold checks (thermal, latency, bandwidth, analog drift)
- [ ] `services/tpt-observer/internal/anomaly/` — Go subscriber (pending)
- [ ] Observer UI: amber/red pulsing overlay on affected nodes (pending)
- [ ] Settings page: anomaly detection toggle (pending)

### Feature 4 — Live Adaptive Recompilation ("Self-Healing" Deployments)

- [x] `python/tpt_catalyst/tpt_catalyst/adaptive.py` — `AdaptiveThresholds`; `AdaptiveRecompiler`: background thread watches telemetry, identifies affected layers from partition.json, triggers `tpt-catalyst pack --incremental`, calls `tpt-alloy ota`; `HealingEvent` log with status tracking
- [ ] `services/tpt-observer/internal/adaptive/` — Go dispatcher (pending)
- [ ] Observer UI: "Healing" badge (pending)

### Feature 5 — Hardware REPL (`python/tpt_shell/`)

- [x] `python/tpt_shell/` package scaffold (pyproject.toml + `tpt-shell` entry point)
- [x] `python/tpt_shell/tpt_shell/session.py` — `ShellSession`: WebSocket connection; `run_layer`, `inspect`, `telemetry_snapshot`, `diff` async methods; layer ID discovery from `.tptpkg/ir/model.tptir`
- [x] `python/tpt_shell/tpt_shell/repl.py` — `prompt_toolkit` REPL loop; tab completion for layer IDs; all commands implemented
- [x] CLI: `tpt-shell <model.tptpkg> [--hardware alloy|fusion|element|sil] [--node <ip>] [--layer <id>]`
- [ ] Alloy firmware: `TPT_DEBUG_MODE` flag (pending firmware generator update)
- [ ] Observer UI: node-click → embedded shell panel (pending)

### Feature 6 — Model Lineage & Provenance Graph

- [x] `python/tpt_catalyst/tpt_catalyst/provenance.py` — `ProvenanceNode`; `ProvenanceGraph` DAG with `append_step()`, `to_json()`, `from_file()`, `print_tree()`, `diff()`; serialized to `provenance/lineage.json`; `graph_for_model()` factory; `StepType` enum covering all pipeline stages
- [x] CLI: `tpt-catalyst provenance <model.tptpkg>` — prints lineage tree + diff against another package
- [ ] Wire `append_step()` calls into ingest, optimize, quantize, pack pipeline stages
- [x] `frontend/src/app/provenance/page.tsx` — compilation timeline with expandable step cards; step-type color legend; accuracy-delta bar chart; diff mode (added/removed steps highlighted); two-package path inputs
- [ ] Add `provenance_root` to `PackageManifest` in `crates/tpt-catalyst/src/package.rs`

### Feature 7 — Federated Learning Orchestration (`python/tpt_fl/`)

- [x] `python/tpt_fl/` package scaffold (pyproject.toml + `tpt-fl` entry point)
- [x] `python/tpt_fl/tpt_fl/config.py` — `FederatedConfig` with strategy, rounds, compression, min_participants, local_epochs, lr, batch_size, recompile_after_rounds
- [x] `python/tpt_fl/tpt_fl/orchestrator.py` — `FLOrchestrator.run()`: per-round gradient collection, FedAvg/FedProx aggregation, global weight update, push to nodes, incremental recompile + OTA every N rounds; `RoundMetrics` + `FLSession` dataclasses
- [x] `python/tpt_fl/tpt_fl/compression.py` — `GradientCompressor`: top-K sparsification with error-feedback residual accumulation; `CompressedGradient` with `decompress()`
- [x] CLI: `tpt-fl train <model.tptpkg> --data-sources 192.168.1.10,192.168.1.11 --rounds 10 --strategy fedavg`
- [ ] Alloy firmware: `TPT_FL_MODE` build flag (pending firmware generator update)
- [ ] Observer UI: FL round progress panel (pending)

---

## Platform Improvements — Usability & Automation

### Developer Tooling

- [x] Add `Makefile` at repo root with `build` (cargo + maturin + pip), `test` (cargo + pytest + go), `dev` (observer + frontend concurrently), `doctor` (runs `tpt-doctor`), `lint` (cargo fmt --check + ruff + go vet) targets
- [x] Add root `pyproject.toml` workspace: single `pip install -e .[all]` installs all 13 Python packages; define `[fpga]`, `[swarm]`, `[analog]` extras; eliminates the hardcoded PYTHONPATH string used in testing
- [x] Add `conftest.py` at repo root that appends all `python/*/` paths to `sys.path` so pytest discovers packages without manual PYTHONPATH export
- [x] Add `.pre-commit-config.yaml` with hooks: `cargo fmt --check`, `ruff check`, `go fmt ./...`, `npm run lint` (frontend)
- [x] Automate PyO3 binding build order: wire `maturin develop` into `make build` so `crates/tpt-catalyst-python` and `crates/tpt-alloy-python` rebuild automatically when Rust source changes; document in CONTRIBUTING.md
- [ ] Consolidate `cloud/` and `services/` directories — `services/` is canonical; delete `cloud/synthesis-worker`, `cloud/synthesis-broker`, `cloud/crucible-cloud` and update any references (confirm before executing)

### Observer Backend Config

- [x] `services/tpt-observer/cmd/observer/main.go` — read `PORT` env var, default `"8080"`; replace hardcoded `:8080` string
- [x] `services/tpt-observer/cmd/observer/main.go` — make simulation TPS baseline configurable via `TPT_SIM_TPS` env var instead of hardcoded `120.5`

### Frontend — Silent Failure Fixes

- [x] `frontend/src/app/cloud/page.tsx` — add toast notification on job submission failure (backend offline or non-ok response); also added success toast on job created
- [x] `frontend/src/app/editor/page.tsx` — toast on save failure (backend unreachable or HTTP error); status message remains in header for inline feedback
- [x] `frontend/src/components/DownloadPanel.tsx` — show amber "Offline — showing sample data" banner when falling back to mock artifact list; `offline` state tracks whether API was reached

### Frontend — SetupWizard & Dashboard Realism

- [x] `frontend/src/components/SetupWizard.tsx` — replaced hardcoded fixture tools with real `POST /api/doctor` call on mount; added loading state; "offline" status shown when backend unreachable; step 2 pre-flight calls `POST /api/preflight` with fallback to fixture
- [x] `frontend/src/components/Dashboard.tsx` — added `HardwareStatusCard` component using `useTelemetry()`; Observer shows ONLINE/OFFLINE from WS; other modules show ACTIVE (have telemetry)/IDLE/OFFLINE based on live data

### Frontend — Polish

- [x] Add per-page error boundaries: `cloud/page.tsx`, `editor/page.tsx`, `compare/page.tsx`, `tournament/page.tsx`, `jobs/page.tsx` all wrapped in `<ErrorBoundary>`; `Providers.tsx` adds `ToastProvider` to root layout
- [x] Add loading skeleton states: `jobs/page.tsx` shows 5 animated skeleton rows while fetching; `SetupWizard.tsx` shows skeleton grid while pre-flight loads
- [x] Add pagination to `frontend/src/app/jobs/page.tsx` — 20 jobs/page with prev/next controls and "X–Y of N" counter
- [x] Add "Export as CSV/JSON" buttons to compare and tournament results pages

### Cross-Platform Config Fix

- [x] `drivers/certification/certify.py` — replaced `Path("/tmp/tpt-cert-registry")` with `Path(tempfile.gettempdir()) / "tpt-cert-registry"` to fix Windows compatibility

### Integration Testing

- [x] Add `tests/integration/test_pipeline.py`: synthetic TptIr → pre-flight check → pack into `.tptpkg` → AlloySil emulator → assert telemetry schema; run with `pytest -m integration`

---

## Cross-Repo Suite Integration (tpt-gpu & tpt-spark)

These items came out of analysing the three-repo TPT AI compute suite (tpt-gpu, tpt-spark, tpt-crucible) for cross-repo synergies. None are required for tpt-crucible to work standalone — they are optional improvements that strengthen the suite as a whole.

### Adopt tpt-gpu's TPTIR as Catalyst's output dialect (depends on tpt-gpu item 1)

tpt-gpu defines TPTIR as its MLIR-based intermediate representation (Layer 3). Crucible's Catalyst module generates its own TPT-IR from GGUF/ONNX/PyTorch inputs. If both dialects are the same (or if Crucible adopts tpt-gpu's published spec), a model compiled once to TPTIR can route to GPU (tpt-gpu runtime), FPGA (Fusion), MCU swarm (Alloy), or analog (Element) — single IR, all targets.

- [ ] Wait for tpt-gpu to publish `tptir-spec` (see tpt-gpu `todo1.md` item 1)
- [ ] Compare tpt-gpu's TPTIR dialect with Catalyst's current TPT-IR output format (op names, type system, attributes, text-format serialisation)
- [ ] If the dialects are compatible: add `tptir-spec` as a dependency to Catalyst and align Catalyst's output to the published spec
- [ ] If there are gaps: open a discussion issue in tpt-gpu listing the ops Crucible needs that TPTIR doesn't cover yet
- [ ] Once aligned, Catalyst's output becomes a strict subset of tpt-gpu's TPTIR — Alloy/Fusion/Element consume the same IR that tpt-gpu's runtime does

### Adopt Spark as the default local LLM backend (depends on tpt-spark item 2)

Crucible currently lists Spark IPC as one of several optional LLM backends (alongside OpenRouter, Anthropic API, Ollama). Spark is the only one that is fully offline, privacy-preserving, and part of the same open-source suite. It should be the default when available, with cloud providers as explicit opt-in fallbacks.

- [ ] Wait for tpt-spark to publish its `HEADLESS_API.md` spec (see tpt-spark `todo1.md` item 2)
- [x] Implement Spark auto-detection: on startup, check for `$XDG_RUNTIME_DIR/tpt-spark.sock` (Linux), `\\.\pipe\tpt-spark` (Windows), or `/tmp/tpt-spark.sock` (macOS) — `python/tpt_catalyst/tpt_catalyst/spark_autodetect.py`, exposed via Observer's `/api/spark/status`
- [x] If Spark is detected, set it as the default LLM backend in Observer's settings panel — `frontend/src/app/settings/page.tsx` auto-sets `llmProvider: "spark"` when detected and no provider chosen yet
- [x] If Spark is not detected, show an "Install TPT Spark for offline AI assistance" prompt with a link, then fall back to cloud providers — same settings page, "TPT Spark not running" banner with install link
- [ ] Update `LLM_BACKENDS.md` (or equivalent standalone config doc) to document the priority order: Spark IPC → Ollama-compatible → OpenRouter → Anthropic API (currently only documented in CLAUDE.md and inline in the settings UI)

### Adopt shared model registry (`~/.tpt/models/`)

tpt-spark downloads and manages GGUF models in `~/.tpt/models/`. Crucible's Catalyst ingests the same GGUF models as compilation inputs. Without a shared convention, users download models twice.

- [x] Update Catalyst's model loading code to scan `~/.tpt/models/` as the primary GGUF source — `python/tpt_catalyst/tpt_catalyst/spark_integration.py::SparkDetector` (`SHARED_REGISTRY`, checked before legacy `.tpt-spark` directories)
- [x] Read `~/.tpt/models/models.json` (see tpt-gpu `todo1.md` item 2 for the manifest spec) to parse a model list without filesystem scanning — `SparkDetector.read_models_manifest()` / `_models_from_manifest()`
- [x] If `~/.tpt/models/` doesn't exist, fall back to the current behaviour (legacy Spark dirs, then explicit file path) — `SparkDetector._find_spark_dir()`
- [ ] Wire `SparkDetector.list_models()` into an Observer API endpoint so the wizard's model selector is actually populated (currently only a static "Spark Model Library" option in `SetupWizard.tsx` with no backend call)
- [ ] Show a "No models found — open TPT Spark to download models" message in Observer when `SparkDetector.registry_empty` is true (property exists in the backend; not yet surfaced in the frontend)

### Read Spark benchmark baselines for emulator validation (depends on tpt-spark item 4)

Crucible's emulator validates compiled edge targets against expected performance bounds. Spark writes GPU benchmark records to `~/.tpt/benchmarks/spark-{date}.json`. Reading these gives the emulator a real-world GPU reference baseline automatically, without the user manually entering numbers.

- [x] On emulator startup, scan `~/.tpt/benchmarks/spark-*.json` and load the most recent record for the current model (matched by model name) — `python/tpt_catalyst/tpt_catalyst/spark_benchmark.py::SparkBenchmark.load_all_baselines()`, wired into `python/tpt_emulator/tpt_emulator/interface.py::EmulatorInterface._load_spark_baseline()`
- [x] Use `tokens_per_second` and `time_to_first_token_ms` as the GPU reference baseline in emulator comparison reports — `SparkBaselineRef` attached to every `EmulatorResult`
- [x] If no Spark benchmark exists for the model, fall back gracefully (returns `None`; emulator continues without a baseline) — no hardcoded reference values were needed since the field is optional
- [ ] Surface `EmulatorResult.spark_baseline` / `display_label()` in the Observer frontend benchmark panel (backend produces the label string; no frontend consumer found yet)

### TPT Script as a Catalyst input format (deferred — depends on items above and tpt-gpu item 3)

Once TPTIR is unified (item 1) and tpt-gpu's TPT Script compiler frontend is stable, add TPT Script (`.tpt` files) as an optional Catalyst input format. This would mean a single language — TPT Script — can target GPU (via tpt-gpu), FPGA (via Fusion), MCU swarm (via Alloy), and analog (via Element). No Crucible implementation work needed until TPTIR unification is confirmed; this is a note to revisit at that point.

- [ ] Revisit once TPTIR unification (item 1) is confirmed and tpt-gpu's TPT Script frontend is stable
