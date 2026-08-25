# Changelog

All notable changes to TPT Crucible are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versioning follows [Semantic Versioning](https://semver.org/) until 1.0,
with the caveat that pre-1.0 minor versions may break APIs (see the semver
policy note in todo.md).

## [Unreleased]

### Added

- Cargo workspace scaffold with seven crates: `tpt-crucible-common`,
  `-catalyst`, `-alloy`, `-fusion`, `-element`, `-observer`, and `-cli`.
- **TPT-IR** in `tpt-crucible-common`: strongly-typed computation graph
  (`Graph`/`Op`/`TensorDesc`), dtypes including GGML block quants
  (`Q4_0`..`Q8_0`), structural validation, Graphviz export, and JSON + binary
  (magic-headered bincode) serialization.
- **Catalyst**: SafeTensors ingestion (native parser + encoder) and GGUF v2/v3
  ingestion (native parser: metadata tree, tensor directory, block-quant dtype
  mapping, Llama hyperparameter extraction); format detection for all twelve
  roadmap formats; `Ingestor` registry.
- **Catalyst**: ONNX ingestion via a native protobuf wire-format reader
   (MatMul/Gemm lowering, elementwise, Softmax, LayerNorm, Reshape,
   Transpose, Concat, Cast, Gather); Llamafile ingestion (embedded-GGUF
   extraction); AWQ/GPTQ quantized SafeTensors containers tagged with
   ``quant_format`` metadata.
- **Catalyst**: PyTorch checkpoint ingestion (``.pt``/``.pth`` torch.save zip
  format): STORED-only zip reader plus a restricted pickle-stack interpreter
  for ``data.pkl`` (whitelisted REDUCE targets, storage persistent-IDs);
  nested state-dicts flatten to dotted IR names.
- **Catalyst**: Keras v3 ``.keras`` archive ingestion: ``config.json``
  signature detection, per-saveable ``states.npz`` weight stores read via a
  native NPY parser (C-order, little-endian dtypes); legacy HDF5 weights
  surface as structured unsupported errors.
- **TPT-UIR adapter** (`tpt-crucible-uir-adapter`, new crate): lossless
  TPT-IR `Graph` <-> Crucible-dialect `Region` conversion (~27
  `tpt_crucible.*` compute ops; constant tensors embedded as
  `AttributeValue::Bytes`; graph name/metadata carried by a trailing
  `graph_info` op); wired into the CLI via `tpt ingest --uir <file>`, which
  emits postcard-encoded regions consumable by `tpt-uir-cli`.
- **Catalyst**: `tpt-doctor` toolchain verifier scanning python/esptool/yosys/
  nextpnr/kicad-cli.
- **Alloy**: topology auto-discovery from node-reported latency/bandwidth
  matrices; topology-aware partitioner (layer-serial, head-parallel, and
  hybrid strategies); KV-cache planning that fails loudly instead of OOMing;
  heartbeat protocol codec and failure detector; per-node Rust/C++ firmware
  skeletons, JSON shard manifests, and parallel esptool flash scripts.
- **CLI** (`tpt`): `ingest`, `info`, `compile --target alloy|fusion|element`,
  and `doctor` subcommands with `swarm` (default) and `fpga` cargo features.
- CI workflow (fmt, clippy `-D warnings`, tests on Linux+Windows, wasm32 check)
  and publish-order-aware `cargo publish` workflow.
- Dual MIT OR Apache-2.0 licensing, CONTRIBUTING.md, SECURITY.md.

### Fixed

- Repaired repository files corrupted during the platform migration
  (stray markup-junk lines from a corrupted earlier write: manifests, licenses,
  docs, toolchain file).