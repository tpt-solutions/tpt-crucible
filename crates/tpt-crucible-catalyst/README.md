# tpt-crucible-catalyst

The universal model translator for TPT Crucible: ingests standard AI model
formats and lowers them into hardware-agnostic TPT-IR.

Implemented today:

* **SafeTensors** ingestion (native parser + encoder)
* **GGUF v2/v3** ingestion (native parser: metadata tree, tensor directory,
  block-quant dtype mapping, Llama hyperparameter extraction)
* **ONNX** ingestion via a native protobuf wire-format reader (MatMul/Gemm
  lowering, elementwise ops, Softmax, LayerNorm, Reshape, Transpose, Concat,
  Cast, Gather)
* **PyTorch** `.pt`/`.pth` ingestion (torch.save zip format: STORED zip
  reader + restricted pickle VM for `data.pkl`; state-dicts flatten to dotted
  names)
* **Keras v3** `.keras` archives (`config.json` detection + nested`n  `states.npz` stores via a native NPY parser; legacy `.h5` rejected)
* **Llamafile** ingestion (embedded-GGUF extraction)
* **AWQ / GPTQ** quantized SafeTensors containers (tagged with `quant_format`
  metadata; dequantization kernels are a later pass)
* Format detection for all twelve roadmap formats (TensorFlow SavedModel,
  TFLite, EXL2, JAX/Flax, Keras recognized; ingestion on the roadmap)
* `tpt-doctor`: external toolchain discovery and verification

Pure Rust, no heavy dependencies, wasm-compatible.

See the [workspace README](https://github.com/tpt-solutions/tpt-crucible) and
todo.md for the remaining Phase 1 roadmap (operator fusion via e-graphs,
quantization auto-search, streaming pre-flight).