//! Model-format identification.

use std::path::Path;

use tpt_crucible_common::error::{Error, Result};

/// Every model container Catalyst knows how to name.
///
/// Detection is best-effort: extension first, then magic bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    /// HuggingFace `.safetensors`.
    SafeTensors,
    /// llama.cpp `.gguf`.
    Gguf,
    /// Open Neural Network Exchange `.onnx`.
    Onnx,
    /// PyTorch `.pt` / `.pth` / `.ckpt` pickles.
    PyTorch,
    /// TensorFlow `saved_model.pb` directory or file.
    TensorFlowSavedModel,
    /// TensorFlow Lite `.tflite` flatbuffers.
    TensorFlowLite,
    /// AWQ 4-bit quantized checkpoints.
    Awq,
    /// GPTQ 4-bit quantized checkpoints.
    Gptq,
    /// ExLlamaV2 `.exl2` archives.
    Exl2,
    /// JAX/Flax `.msgpack` parameter bundles.
    JaxFlax,
    /// Llamafile single-file executables embedding GGUF weights.
    Llamafile,
    /// Keras `.keras` / legacy `.h5` archives.
    Keras,
}

impl ModelFormat {
    /// All formats in stable display order.
    pub const ALL: &'static [ModelFormat] = &[
        Self::SafeTensors,
        Self::Gguf,
        Self::Onnx,
        Self::PyTorch,
        Self::TensorFlowSavedModel,
        Self::TensorFlowLite,
        Self::Awq,
        Self::Gptq,
        Self::Exl2,
        Self::JaxFlax,
        Self::Llamafile,
        Self::Keras,
    ];

    /// Canonical lowercase name.
    pub fn name(self) -> &'static str {
        match self {
            Self::SafeTensors => "safetensors",
            Self::Gguf => "gguf",
            Self::Onnx => "onnx",
            Self::PyTorch => "pytorch",
            Self::TensorFlowSavedModel => "tf-savedmodel",
            Self::TensorFlowLite => "tflite",
            Self::Awq => "awq",
            Self::Gptq => "gptq",
            Self::Exl2 => "exl2",
            Self::JaxFlax => "jax-flax",
            Self::Llamafile => "llamafile",
            Self::Keras => "keras",
        }
    }
}

impl core::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// Identify the format of `path`.
///
/// Strategy: extension lookup first; when unknown, sniff magic bytes
/// (`GGUF`, ELF for Llamafiles, Zip/PK for Keras-v3 containers).
/// Directory inputs for SavedModel/ExL2 layouts are recognized too.
pub fn detect(path: &Path) -> Result<ModelFormat> {
    if path.is_dir() {
        return detect_directory(path);
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());

    if let Some(f) = match ext.as_deref() {
        Some("safetensors") => Some(ModelFormat::SafeTensors),
        Some("gguf") => Some(ModelFormat::Gguf),
        Some("onnx") => Some(ModelFormat::Onnx),
        Some("pt") | Some("pth") | Some("ckpt") => Some(ModelFormat::PyTorch),
        Some("pb") => Some(ModelFormat::TensorFlowSavedModel),
        Some("tflite") => Some(ModelFormat::TensorFlowLite),
        Some("awq") => Some(ModelFormat::Awq),
        Some("gptq") => Some(ModelFormat::Gptq),
        Some("exl2") => Some(ModelFormat::Exl2),
        Some("msgpack") => Some(ModelFormat::JaxFlax),
        Some("llamafile") => Some(ModelFormat::Llamafile),
        Some("keras") | Some("h5") | Some("hdf5") => Some(ModelFormat::Keras),
        _ => None,
    } {
        return Ok(f);
    }

    detect_by_magic(path)
}

fn detect_directory(path: &Path) -> Result<ModelFormat> {
    if path.join("saved_model.pb").exists() {
        Ok(ModelFormat::TensorFlowSavedModel)
    } else if path.extension().and_then(|e| e.to_str()) == Some("exl2") {
        Ok(ModelFormat::Exl2)
    } else {
        Err(Error::UnknownFormat {
            path: path.display().to_string(),
        })
    }
}

fn detect_by_magic(path: &Path) -> Result<ModelFormat> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut magic = [0u8; 4];
    let n = file.read(&mut magic)?;
    match (n, &magic) {
        (4, b"GGUF") => Ok(ModelFormat::Gguf),
        // ELF executables with embedded weights == Llamafile convention.
        (4, b"\x7fELF") => Ok(ModelFormat::Llamafile),
        // Zip container: Keras v3 or zipped TF SavedModel.
        (4, b"PK\x03\x04") => Ok(ModelFormat::Keras),
        _ => Err(Error::UnknownFormat {
            path: path.display().to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn tmp_file(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("catalyst-detect-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(bytes).unwrap();
        p
    }

    #[test]
    fn detects_by_extension() {
        assert_eq!(
            detect(Path::new("m/model.safetensors")).unwrap(),
            ModelFormat::SafeTensors
        );
        assert_eq!(detect(Path::new("m/x.GGUF")).unwrap(), ModelFormat::Gguf);
        assert_eq!(detect(Path::new("m/net.onnx")).unwrap(), ModelFormat::Onnx);
        assert_eq!(detect(Path::new("m/w.pt")).unwrap(), ModelFormat::PyTorch);
        assert_eq!(
            detect(Path::new("m/y.tflite")).unwrap(),
            ModelFormat::TensorFlowLite
        );
        assert_eq!(detect(Path::new("m/z.keras")).unwrap(), ModelFormat::Keras);
        assert_eq!(
            detect(Path::new("m/j.msgpack")).unwrap(),
            ModelFormat::JaxFlax
        );
    }

    #[test]
    fn detects_by_magic_bytes() {
        assert_eq!(
            detect(&tmp_file("noext-gguf", b"GGUF\x03\x00\x00\x00")).unwrap(),
            ModelFormat::Gguf
        );
        assert_eq!(
            detect(&tmp_file("run.bin", b"\x7fELF\x02\x01\x01\x00")).unwrap(),
            ModelFormat::Llamafile
        );
    }

    #[test]
    fn unknown_format_errors() {
        let err = detect(&tmp_file("mystery.xyz", b"\x00\x00\x00\x00")).unwrap_err();
        assert!(matches!(err, Error::UnknownFormat { .. }));
    }

    #[test]
    fn all_formats_have_names() {
        assert_eq!(ModelFormat::ALL.len(), 12);
        for f in ModelFormat::ALL {
            assert!(!f.name().is_empty());
        }
    }
}
