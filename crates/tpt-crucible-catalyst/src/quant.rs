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

/// True when the graph carries tensors laid out like a quantized checkpoint.
fn looks_quantized(graph: &Graph) -> bool {
    graph.nodes.iter().any(|n| {
        QUANT_MARKERS
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
}
