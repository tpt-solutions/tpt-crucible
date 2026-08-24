//! # tpt-crucible-catalyst
//!
//! **The Core / universal translator** (`spec2.txt` §3.1): ingests standard AI
//! model formats, strips GPU-specific assumptions, and lowers everything into
//! the hardware-agnostic TPT-IR graph defined in [`tpt_crucible_common`].
//!
//! ## Today
//!
//! * `format` — identifies all 12 target container formats.

//! * [`ingest`] — dispatches to format ingestors; **SafeTensors** and
//!   **GGUF (v2/v3)** are fully implemented natively (no heavy deps,
//!   wasm-ready).
//! * [`doctor`] — `tpt-doctor` toolchain verifier.
//!
//! ```no_run
//! use tpt_crucible_catalyst::ingest_path;
//!
//! let graph = ingest_path("models/tinyllama.gguf")?;
//! graph.validate()?;
//! # Ok::<(), tpt_crucible_common::Error>(())
//! ```
//!
//! ## Roadmap (tracked in `todo.md`)
//!
//! ONNX/PyTorch/TensorFlow/TFLite/AWQ/GPTQ/EXL2/JAX/Keras ingestion,
//! `egg`-based operator fusion, quantization auto-search against an accuracy
//! budget, streaming pre-flight over WebSockets, and a custom MLIR dialect.

pub mod doctor;
pub mod format;
pub mod gguf;
pub mod ingest;
pub mod safetensors;

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
