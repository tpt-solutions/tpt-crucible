//! AWQ / GPTQ quantized-checkpoint ingestion.
//!
//! AWQ and GPTQ checkpoints ship in standard SafeTensors (or PyTorch pickle)
//! containers; the quantization is expressed through *tensor naming* and
//! sidecar configs, not a new file format. This module reuses the SafeTensors
//! reader and tags the resulting graph with the detected scheme:
//!
//! * `qweight` / `qzeros` / `scales` / `g_idx` tensor suffixes → GPTQ-style
//!   4-bit packing,
//! * AWQ's `qweight`/`qzeros`/`scales` layout with per-channel scales → AWQ.
//!
//! Weights land as their raw packed dtypes; dequantization kernels are a
//! separate pass (tracked in todo.md).

use std::path::Path;

use tpt_crucible_common::error::{Error, Result};
use tpt_crucible_common::Graph;

/// Tensor-name markers for quantized linears.
const QUANT_MARKERS: &[&str] = &["qweight", "qzeros", "scales", "g_idx"];

/// Tensor-suffix markers specific to ExLlamaV2's measured variable-bit
/// packing (`*.q_weight` INT32 + `*.q_scale`/`*.q_invmax` F16 sidecars).
const EXL2_MARKERS: &[&str] = &["q_weight", "q_scale", "q_invmax"];

/// True when the graph carries tensors laid out like a quantized checkpoint.
fn looks_quantized(graph: &Graph) -> bool {
    graph.nodes.iter().any(|n| {
        QUANT_MARKERS
            .iter()
            .any(|m| n.name.ends_with(m) || n.name.ends_with(&format!(".{m}")))
    })
}

/// True when the graph carries ExLlamaV2-packed tensors.
fn looks_exl2(graph: &Graph) -> bool {
    graph.nodes.iter().any(|n| {
        EXL2_MARKERS
            .iter()
            .any(|m| n.name.ends_with(m) || n.name.ends_with(&format!(".{m}")))
    })
}

fn ingest_quantized(path: &Path, scheme: &'static str) -> Result<Graph> {
    let mut graph = crate::safetensors::ingest(path)?;
    if !looks_quantized(&graph) {
        return Err(Error::ParseFormat {
            path: path.display().to_string(),
            format: scheme.into(),
            reason: "checkpoint has no quantized tensor names (qweight/qzeros/scales)".into(),
        });
    }
    graph.set_metadata("quant_format", scheme);
    graph.set_metadata("quant_bits", "4");
    Ok(graph)
}

/// Cheap textual sniff of `config.json` for the exl2 quant method.
fn has_exl2_config_marker(path: &Path) -> bool {
    std::fs::read_to_string(path.join("config.json"))
        .map(|raw| raw.contains("exl2"))
        .unwrap_or(false)
}

/// Format an average-bits value without a trailing `.0`.
fn fmt_bits(bits: f64) -> String {
    if (bits - bits.trunc()).abs() < f64::EPSILON {
        format!("{}", bits as i64)
    } else {
        format!("{bits}")
    }
}

/// Read the average bit-rate from an ExLlamaV2 `config.json`, if present.
///
/// Accepts both `quantization_config.bits` and a top-level `bits`
/// (HuggingFace convention); returns `None` when absent or unparsable.
fn exl2_config_bits(path: &Path) -> Result<Option<String>> {
    let cfg_path = path.join("config.json");
    if !cfg_path.is_file() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(cfg_path)?;
    let v: serde_json::Value = serde_json::from_str(&raw).map_err(|e| Error::ParseFormat {
        path: path.display().to_string(),
        format: "exl2".into(),
        reason: format!("invalid config.json: {e}"),
    })?;
    let bits = v
        .get("quantization_config")
        .and_then(|q| q.get("bits"))
        .or_else(|| v.get("bits"))
        .and_then(|b| b.as_f64());
    Ok(bits.map(fmt_bits))
}

/// Ingest an ExLlamaV2 (EXL2) checkpoint.
///
/// Accepts either an HF-style directory (`config.json` + `*.safetensors`
/// shards) or a bare SafeTensors file. Weights are stored measured /
/// variable-bit packed (`*.q_weight` INT32 with F16 scale sidecars) and land
/// at their stored dtypes; dequantization kernels are tracked separately in
/// todo.md. The graph is tagged `quant_format=exl2` and `quant_bits` from
/// the config's average bitrate (`mixed` when unknown).
pub fn ingest_exl2(path: &Path) -> Result<Graph> {
    // Gather shard files.
    let (shards, source_name) = if path.is_dir() {
        let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(path)?
            .filter_map(std::result::Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
            })
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(Error::ParseFormat {
                path: path.display().to_string(),
                format: "exl2".into(),
                reason: "directory contains no *.safetensors shards".into(),
            });
        }
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();
        (shards, name)
    } else {
        (vec![path.to_path_buf()], String::new())
    };
    let config_bits = exl2_config_bits(path)?;

    // Ingest and merge shards (weight-only graphs: every node is a constant,
    // so merging is a straight re-push under one graph).
    let mut merged = Graph::new(if source_name.is_empty() {
        "model"
    } else {
        source_name.as_str()
    });
    merged.set_metadata("source_format", "exl2");
    let mut shard_count = 0usize;
    let mut any_exl2_names = false;
    for shard in &shards {
        let bytes = std::fs::read(shard)?;
        let stem = shard
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("shard")
            .to_string();
        let g = crate::safetensors::parse_bytes(&bytes, &stem)?;
        any_exl2_names |= looks_exl2(&g);
        for n in &g.nodes {
            merged.push(&n.name, n.op.clone(), Vec::<_>::new());
        }
        shard_count += 1;
    }

    // Validate EXL2-ness: config.json marker or packed-tensor naming.
    let has_config_marker = config_bits.is_some() || has_exl2_config_marker(path);
    if !any_exl2_names && !has_config_marker {
        return Err(Error::ParseFormat {
            path: path.display().to_string(),
            format: "exl2".into(),
            reason: "no EXL2 markers found (expected config.json quantization_config \
                     or *.q_weight tensors)"
                .into(),
        });
    }

    merged.set_metadata("quant_format", "exl2");
    let bits = config_bits.unwrap_or_else(|| "mixed".to_string());
    merged.set_metadata("quant_bits", &bits);
    merged.set_metadata("exl2_shards", shard_count.to_string());
    merged.validate()?;
    Ok(merged)
}

/// Ingest an AWQ checkpoint (SafeTensors container).
pub fn ingest_awq(path: &Path) -> Result<Graph> {
    ingest_quantized(path, "awq")
}

/// Ingest a GPTQ checkpoint (SafeTensors container).
pub fn ingest_gptq(path: &Path) -> Result<Graph> {
    ingest_quantized(path, "gptq")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::io::Write as _;
    use tpt_crucible_common::{DType, Tensor};

    fn write_checkpoint(name: &str, tensors: BTreeMap<String, Tensor>) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("catalyst-quant-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(&crate::safetensors::encode(&tensors).unwrap())
            .unwrap();
        p
    }

    fn gptq_tensors() -> BTreeMap<String, Tensor> {
        let mut m = BTreeMap::new();
        // Packed 4-bit weights: stored as u8 payload bytes here.
        m.insert(
            "blk.0.attn_q.qweight".into(),
            Tensor {
                desc: tpt_crucible_common::TensorDesc::new(vec![4, 2], DType::U32),
                data: tpt_crucible_common::TensorData::from_vec(vec![0xAA; 32]),
            },
        );
        m.insert(
            "blk.0.attn_q.scales".into(),
            Tensor::from_f32(vec![4], &[0.1; 4]),
        );
        m.insert(
            "blk.0.attn_q.qzeros".into(),
            Tensor {
                desc: tpt_crucible_common::TensorDesc::new(vec![4], DType::U32),
                data: tpt_crucible_common::TensorData::from_vec(vec![0; 16]),
            },
        );
        m.insert(
            "blk.0.ln.weight".into(),
            Tensor::from_f32(vec![8], &[0.5; 8]),
        );
        m
    }

    #[test]
    fn gptq_checkpoint_tagged() {
        let p = write_checkpoint("model.gptq", gptq_tensors());
        let g = ingest_gptq(&p).unwrap();
        assert_eq!(g.metadata("quant_format"), Some("gptq"));
        assert_eq!(g.metadata("quant_bits"), Some("4"));
        assert_eq!(g.len(), 4);
    }

    #[test]
    fn plain_checkpoint_rejected_for_gptq() {
        let mut m = BTreeMap::new();
        m.insert("w".to_string(), Tensor::from_f32(vec![2], &[1.0, 2.0]));
        let p = write_checkpoint("plain.safetensors", m);
        let err = ingest_awq(&p).unwrap_err();
        assert!(err.to_string().contains("no quantized tensor names"));
    }

    // ---- EXL2 ----------------------------------------------------------------

    use crate::format::ModelFormat;
    use crate::{ingest_path, safetensors};

    fn exl2_tensors() -> BTreeMap<String, Tensor> {
        let mut m = BTreeMap::new();
        // Measured variable-bit packing: INT32 payload + F16 scale sidecar.
        m.insert(
            "blk.0.attn_q.q_weight".into(),
            Tensor {
                desc: tpt_crucible_common::TensorDesc::new(vec![4, 2], DType::U32),
                data: tpt_crucible_common::TensorData::from_vec(vec![0x55; 32]),
            },
        );
        m.insert(
            "blk.0.attn_q.q_scale".into(),
            Tensor {
                desc: tpt_crucible_common::TensorDesc::new(vec![4], DType::F16),
                data: tpt_crucible_common::TensorData::from_vec(vec![0x3C; 8]),
            },
        );
        m.insert(
            "blk.0.ln.weight".into(),
            Tensor::from_f32(vec![8], &[0.5; 8]),
        );
        m
    }

    #[test]
    fn exl2_directory_with_config() {
        let dir = std::env::temp_dir().join("catalyst-quant-tests/exl2dir");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"quantization_config":{"quant_method":"exl2","bits":3.5}}"#,
        )
        .unwrap();
        let shard = dir.join("model.safetensors");
        let mut f = std::fs::File::create(&shard).unwrap();
        f.write_all(&safetensors::encode(&exl2_tensors()).unwrap())
            .unwrap();

        let g = ingest_exl2(&dir).unwrap();
        assert_eq!(g.metadata("quant_format"), Some("exl2"));
        assert_eq!(g.metadata("quant_bits"), Some("3.5"));
        assert_eq!(g.metadata("exl2_shards"), Some("1"));
        assert_eq!(g.len(), 3);
    }

    #[test]
    fn exl2_directory_detected_and_ingested_via_registry() {
        let dir = std::env::temp_dir().join("catalyst-quant-tests/exl2reg");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"quantization_config":{"quant_method":"exl2","bits":4}}"#,
        )
        .unwrap();
        let shard = dir.join("model.safetensors");
        let mut f = std::fs::File::create(&shard).unwrap();
        f.write_all(&safetensors::encode(&exl2_tensors()).unwrap())
            .unwrap();

        assert_eq!(crate::format::detect(&dir).unwrap(), ModelFormat::Exl2);
        let g = ingest_path(&dir).unwrap();
        assert_eq!(g.metadata("source_format"), Some("exl2"));
        assert_eq!(g.metadata("quant_bits"), Some("4"));
        g.validate().unwrap();
    }

    #[test]
    fn exl2_single_file_by_qweight_marker() {
        let p = write_checkpoint("exl2-shard.bin", exl2_tensors());
        let g = ingest_exl2(&p).unwrap();
        assert_eq!(g.metadata("quant_format"), Some("exl2"));
        assert_eq!(g.metadata("quant_bits"), Some("mixed"));
        assert_eq!(g.len(), 3);
    }

    #[test]
    fn non_exl2_checkpoint_rejected() {
        let mut m = BTreeMap::new();
        m.insert("w".to_string(), Tensor::from_f32(vec![2], &[1.0, 2.0]));
        let p = write_checkpoint("plain2.safetensors", m);
        let err = ingest_exl2(&p).unwrap_err();
        assert!(err.to_string().contains("no EXL2 markers"), "{err}");
    }
}
