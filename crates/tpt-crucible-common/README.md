# tpt-crucible-common

Shared **TPT-IR** (TPT Intermediate Representation) types and error handling for
the TPT Crucible hardware-agnostic AI compiler suite.

* `Graph` / `GraphNode` / `NodeId` - SSA-ordered computation graph with validation
* `Op` - strongly-typed operator set (transformer-native: attention, RoPE,
  RMSNorm, SiLU, plus elementwise/linalg/layout primitives)
* `Tensor` / `TensorDesc` - shape + dtype metadata with little-endian payloads;
  dtypes include f16/bf16 and GGML block quants (`q4_0`..`q8_0`)
* `Error` - one shared error enum for every crate
* Serialization: JSON (`.tptir.json`) and compact binary with a magic header
  (`.tptir`), sniffed automatically by `Graph::save`/`Graph::load`

Pure Rust, no unsafe, wasm-compatible.

See the [workspace README](https://github.com/tpt-solutions/tpt-crucible).