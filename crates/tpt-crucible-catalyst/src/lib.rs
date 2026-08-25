//! # tpt-crucible-catalyst
//!
//! **The Core / universal translator** (`spec2.txt` section 3.1): ingests
//! standard AI model formats, strips GPU-specific assumptions, and lowers
//! everything into the hardware-agnostic TPT-IR graph defined in
//! [`tpt_crucible_common`].
//!
//! ## Implemented today
//!
//! * [`format::ModelFormat`] — identifies all 12 target container formats.
//! * [`ingest`] - dispatches to format ingestors. Native parsers:
//!   **SafeTensors** (single files and HuggingFace multi-shard directories),
//!   **GGUF (v2/v3)**, **ONNX** (core-op subset),
//!   **PyTorch** (torch.save zip format via a restricted pickle VM),
//!   **Keras v3** (.keras archives), **TensorFlow SavedModel** (frozen-graph
//!   `saved_model.pb` via a native protobuf reader), **TFLite** (`.tflite`
//!   flatbuffers via a hand-rolled reader; core-op subset), **Llamafile**
//!   (embedded-GGUF extraction), **AWQ/GPTQ/EXL2** (quantized checkpoints
//!   tagged with `quant_format` metadata; EXL2 directories with shard
//!   merging) and **JAX/Flax** (msgpack parameter pytrees via a native
//!   decoder).
//! * [`doctor`] - `tpt-doctor` toolchain verifier.
//!
//! ```no_run
//! use tpt_crucible_catalyst::ingest_path;
//!
//! let graph = ingest_path("models/tinyllama.gguf")?;
//! graph.validate()?;
//! # Ok::<(), tpt_crucible_common::Error>(())
//! ```
//!
//! ## Roadmap (tracked in todo.md)
//!
//! Egg-based operator fusion, quantization auto-search against an accuracy
//! budget, streaming pre-flight over WebSockets, and a custom MLIR dialect.

pub mod doctor;
pub mod flatbuf;
pub mod flax;
pub mod format;
pub mod gguf;
pub mod ingest;
pub mod keras;
pub mod llamafile;
pub mod msgpack;
pub mod npy;
pub mod onnx;
/// Shared protobuf wire-format reader (used by ONNX + TF SavedModel).
pub(crate) mod pb;
pub mod pickle;
pub mod pkzip;
pub mod pytorch;
pub mod quant;
pub mod safetensors;
pub mod tensorflow;
pub mod tflite;

pub use format::ModelFormat;
pub use ingest::{ingest_path, ingest_with_format, Ingestor};

/// Re-export of the shared IR crate for downstream convenience.
pub use tpt_crucible_common as common;

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_crucible_common::Graph;

    #[test]
    fn reexported_graph_roundtrips() {
        let mut g = Graph::new("smoke");
        g.push(
            "x",
            common::Op::Input(common::TensorDesc::new(vec![1], common::DType::F32)),
            Vec::<common::NodeId>::new(),
        );
        let json = g.to_json_string().unwrap();
        assert_eq!(Graph::from_json_str(&json).unwrap(), g);
    }
}
