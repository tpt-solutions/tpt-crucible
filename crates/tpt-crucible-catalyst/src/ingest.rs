//! The ingestion pipeline: format detection → ingestor dispatch.
//!
//! ```no_run
//! use tpt_crucible_catalyst::ingest_path;
//!
//! let graph = ingest_path("models/tinyllama.gguf").unwrap();
//! println!("{} nodes", graph.len());
//! ```

use std::path::Path;

use tpt_crucible_common::error::{Error, Result};
use tpt_crucible_common::Graph;

use crate::format::{self, ModelFormat};

/// A format-specific model loader.
pub trait Ingestor {
    /// Which format this ingestor handles.
    fn format(&self) -> ModelFormat;
    /// Load `path` into TPT-IR.
    fn ingest(&self, path: &Path) -> Result<Graph>;
}

struct SafeTensorsIngestor;

impl Ingestor for SafeTensorsIngestor {
    fn format(&self) -> ModelFormat {
        ModelFormat::SafeTensors
    }

    fn ingest(&self, path: &Path) -> Result<Graph> {
        crate::safetensors::ingest(path)
    }
}

struct GgufIngestor;

impl Ingestor for GgufIngestor {
    fn format(&self) -> ModelFormat {
        ModelFormat::Gguf
    }

    fn ingest(&self, path: &Path) -> Result<Graph> {
        crate::gguf::ingest(path)
    }
}

struct PyTorchIngestor;

impl Ingestor for PyTorchIngestor {
    fn format(&self) -> ModelFormat {
        ModelFormat::PyTorch
    }

    fn ingest(&self, path: &Path) -> Result<Graph> {
        crate::pytorch::ingest(path)
    }
}

struct KerasIngestor;

impl Ingestor for KerasIngestor {
    fn format(&self) -> ModelFormat {
        ModelFormat::Keras
    }

    fn ingest(&self, path: &Path) -> Result<Graph> {
        crate::keras::ingest(path)
    }
}

struct OnnxIngestor;

impl Ingestor for OnnxIngestor {
    fn format(&self) -> ModelFormat {
        ModelFormat::Onnx
    }

    fn ingest(&self, path: &Path) -> Result<Graph> {
        crate::onnx::ingest(path)
    }
}

struct TensorFlowIngestor;

impl Ingestor for TensorFlowIngestor {
    fn format(&self) -> ModelFormat {
        ModelFormat::TensorFlowSavedModel
    }

    fn ingest(&self, path: &Path) -> Result<Graph> {
        crate::tensorflow::ingest(path)
    }
}

struct TfliteIngestor;

impl Ingestor for TfliteIngestor {
    fn format(&self) -> ModelFormat {
        ModelFormat::TensorFlowLite
    }

    fn ingest(&self, path: &Path) -> Result<Graph> {
        crate::tflite::ingest(path)
    }
}

struct LlamafileIngestor;

impl Ingestor for LlamafileIngestor {
    fn format(&self) -> ModelFormat {
        ModelFormat::Llamafile
    }

    fn ingest(&self, path: &Path) -> Result<Graph> {
        crate::llamafile::ingest(path)
    }
}

struct AwqIngestor;

impl Ingestor for AwqIngestor {
    fn format(&self) -> ModelFormat {
        ModelFormat::Awq
    }

    fn ingest(&self, path: &Path) -> Result<Graph> {
        crate::quant::ingest_awq(path)
    }
}

struct GptqIngestor;

impl Ingestor for GptqIngestor {
    fn format(&self) -> ModelFormat {
        ModelFormat::Gptq
    }

    fn ingest(&self, path: &Path) -> Result<Graph> {
        crate::quant::ingest_gptq(path)
    }
}

struct Exl2Ingestor;

impl Ingestor for Exl2Ingestor {
    fn format(&self) -> ModelFormat {
        ModelFormat::Exl2
    }

    fn ingest(&self, path: &Path) -> Result<Graph> {
        crate::quant::ingest_exl2(path)
    }
}

struct JaxFlaxIngestor;

impl Ingestor for JaxFlaxIngestor {
    fn format(&self) -> ModelFormat {
        ModelFormat::JaxFlax
    }

    fn ingest(&self, path: &Path) -> Result<Graph> {
        crate::flax::ingest(path)
    }
}

/// Ingestor for a format. Every format [`ModelFormat`] names has a native
/// ingestor today.
pub fn ingestor_for(format: ModelFormat) -> Option<Box<dyn Ingestor>> {
    match format {
        ModelFormat::SafeTensors => Some(Box::new(SafeTensorsIngestor)),
        ModelFormat::Gguf => Some(Box::new(GgufIngestor)),
        ModelFormat::Onnx => Some(Box::new(OnnxIngestor)),
        ModelFormat::PyTorch => Some(Box::new(PyTorchIngestor)),
        ModelFormat::Keras => Some(Box::new(KerasIngestor)),
        ModelFormat::Llamafile => Some(Box::new(LlamafileIngestor)),
        ModelFormat::Awq => Some(Box::new(AwqIngestor)),
        ModelFormat::Gptq => Some(Box::new(GptqIngestor)),
        ModelFormat::Exl2 => Some(Box::new(Exl2Ingestor)),
        ModelFormat::JaxFlax => Some(Box::new(JaxFlaxIngestor)),
        ModelFormat::TensorFlowSavedModel => Some(Box::new(TensorFlowIngestor)),
        ModelFormat::TensorFlowLite => Some(Box::new(TfliteIngestor)),
    }
}

/// True when [`ingest_path`] can handle this format today.
pub fn is_supported(format: ModelFormat) -> bool {
    matches!(
        format,
        ModelFormat::SafeTensors
            | ModelFormat::Gguf
            | ModelFormat::Onnx
            | ModelFormat::Llamafile
            | ModelFormat::PyTorch
            | ModelFormat::Keras
            | ModelFormat::Awq
            | ModelFormat::Gptq
            | ModelFormat::Exl2
            | ModelFormat::JaxFlax
            | ModelFormat::TensorFlowSavedModel
            | ModelFormat::TensorFlowLite
    )
}

/// Detect the format of `path` and load it into TPT-IR.
///
/// Override detection with [`ingest_with_format`] when extensions lie.
pub fn ingest_path(path: impl AsRef<Path>) -> Result<Graph> {
    let path = path.as_ref();
    let format = format::detect(path)?;
    ingest_with_format(path, format)
}

/// Load `path` with an explicit [`ModelFormat`] (skips detection).
pub fn ingest_with_format(path: &Path, format: ModelFormat) -> Result<Graph> {
    match ingestor_for(format) {
        Some(ingestor) => ingestor.ingest(path),
        None => Err(Error::NotImplemented {
            feature: format!("ingestion of {format} models"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors;
    use std::collections::BTreeMap;

    #[test]
    fn jax_flax_end_to_end_via_registry() {
        let dir = std::env::temp_dir().join("catalyst-ingest-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("params.msgpack");
        std::fs::write(&p, crate::flax::tests::fixture_bundle()).unwrap();

        let g = ingest_path(&p).unwrap();
        assert_eq!(g.metadata("source_format"), Some("jax-flax"));
        g.validate().unwrap();
        assert!(is_supported(ModelFormat::JaxFlax));
    }

    #[test]
    fn safetensors_end_to_end() {
        let dir = std::env::temp_dir().join("catalyst-ingest-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("model.safetensors");

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "w".to_string(),
            tpt_crucible_common::Tensor::from_f32(vec![2], &[1.0, 2.0]),
        );
        std::fs::write(&p, safetensors::encode(&tensors).unwrap()).unwrap();

        let g = ingest_path(&p).unwrap();
        assert_eq!(g.metadata("source_format"), Some("safetensors"));
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn onnx_end_to_end_via_registry() {
        let dir = std::env::temp_dir().join("catalyst-ingest-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("model.onnx");
        std::fs::write(&p, crate::onnx::tests::fixture_matmul_add_softmax()).unwrap();

        let g = ingest_path(&p).unwrap();
        assert_eq!(g.metadata("source_format"), Some("onnx"));
        g.validate().unwrap();
        assert!(is_supported(ModelFormat::Onnx));
    }

    #[test]
    fn tensorflow_savedmodel_end_to_end_via_registry() {
        // Directory layout (saved_model.pb inside a folder), as exported by
        // `tf.saved_model.save`.
        let dir = std::env::temp_dir().join("catalyst-ingest-tests/savedmodel");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("saved_model.pb"),
            crate::tensorflow::tests::fixture_serving_minimal(),
        )
        .unwrap();

        let g = ingest_path(&dir).unwrap();
        assert_eq!(g.metadata("source_format"), Some("tf-savedmodel"));
        g.validate().unwrap();
        assert_eq!(g.inputs.len(), 1);
        assert_eq!(g.outputs.len(), 1);
        assert!(is_supported(ModelFormat::TensorFlowSavedModel));
    }

    #[test]
    fn tflite_end_to_end_via_registry() {
        let dir = std::env::temp_dir().join("catalyst-ingest-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("model.tflite");
        std::fs::write(&p, crate::tflite::tests::fixture_fc_softmax()).unwrap();

        let g = ingest_path(&p).unwrap();
        assert_eq!(g.metadata("source_format"), Some("tflite"));
        g.validate().unwrap();
        assert!(is_supported(ModelFormat::TensorFlowLite));
    }

    #[test]
    fn every_format_reports_support_status() {
        // Implemented formats must resolve to an ingestor; the rest must be
        // honest NotImplemented errors.
        for f in ModelFormat::ALL {
            let supported = is_supported(*f);
            assert_eq!(ingestor_for(*f).is_some(), supported, "{f}");
        }
    }
}
