# tpt-crucible-catalyst

The universal model translator for TPT Crucible: ingests standard AI model
formats and lowers them into hardware-agnostic TPT-IR.

Implemented today:

* **SafeTensors** ingestion (native parser + encoder)
* **GGUF v2/v3** ingestion (native parser: metadata tree, tensor directory,
  block-quant dtype mapping, Llama hyperparameter extraction)
* Format detection for all twelve roadmap formats (ONNX, PyTorch, TensorFlow,
  TFLite, AWQ/GPTQ, EXL2, JAX/Flax, Llamafile, Keras recognized; ingestion on
  the roadmap)
* `tpt-doctor`: external toolchain discovery and verification

Pure Rust, no heavy dependencies, wasm-compatible.

See the [workspace README](https://github.com/tpt-solutions/tpt-crucible) and
todo.md for the remaining Phase 1 roadmap (operator fusion via e-graphs,
quantization auto-search, streaming pre-flight).