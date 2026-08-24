//! SafeTensors ingestion — HuggingFace's zero-copy weight container.
//!
//! The format is deliberately simple: an 8-byte little-endian header length,
//! a JSON header mapping tensor names to `{dtype, shape, data_offsets}`, then
//! a contiguous little-endian data region. Parsed here without any external
//! crate so it works natively (and under wasm).

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;
use tpt_crucible_common::dtype::DType;
use tpt_crucible_common::error::{Error, Result};
use tpt_crucible_common::ops::Op;
use tpt_crucible_common::{Graph, Tensor, TensorData, TensorDesc};

/// One header entry describing a single tensor.
#[derive(Debug, Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

/// Full JSON header: `__metadata__` plus one entry per tensor.
#[derive(Debug, Deserialize)]
struct Header {
    #[serde(rename = "__metadata__")]
    metadata: Option<BTreeMap<String, String>>,
    #[serde(flatten)]
    entries: BTreeMap<String, Entry>,
}

fn map_dtype(name: &str) -> Result<DType> {
    Ok(match name {
        "BOOL" => DType::Bool,
        "U8" => DType::U8,
        "U16" => DType::U16,
        "U32" => DType::U32,
        "U64" => DType::U64,
        "I8" => DType::I8,
        "I16" => DType::I16,
        "I32" => DType::I32,
        "I64" => DType::I64,
        "F16" => DType::F16,
        "BF16" => DType::Bf16,
        "F32" => DType::F32,
        "F64" => DType::F64,
        other => return Err(Error::UnsupportedDType(other.to_string())),
    })
}

/// Parse SafeTensors bytes into a weight-only TPT-IR [`Graph`].
///
/// The resulting graph contains one `constant` node per tensor (sorted by
/// name) and no outputs — SafeTensors carries weights, not compute. Attach a
/// compute graph (ONNX/PyTorch ingestion, planned) to get an executable IR.
pub fn parse_bytes(bytes: &[u8], source_name: &str) -> Result<Graph> {
    const HEADER_LEN_BYTES: usize = 8;

    if bytes.len() < HEADER_LEN_BYTES {
        return Err(Error::ParseFormat {
            path: source_name.into(),
            format: "safetensors".into(),
            reason: "file shorter than the 8-byte header length".into(),
        });
    }
    let header_len = usize::try_from(u64::from_le_bytes(
        bytes[..HEADER_LEN_BYTES].try_into().unwrap(),
    ))
    .map_err(|_| Error::ParseFormat {
        path: source_name.into(),
        format: "safetensors".into(),
        reason: "header length overflows usize".into(),
    })?;
    let data_start =
        HEADER_LEN_BYTES
            .checked_add(header_len)
            .ok_or_else(|| Error::ParseFormat {
                path: source_name.into(),
                format: "safetensors".into(),
                reason: "header length overflow".into(),
            })?;
    if data_start > bytes.len() {
        return Err(Error::ParseFormat {
            path: source_name.into(),
            format: "safetensors".into(),
            reason: format!(
                "header length {header_len} exceeds file size {}",
                bytes.len()
            ),
        });
    }

    let header: Header =
        serde_json::from_slice(&bytes[HEADER_LEN_BYTES..data_start]).map_err(|e| {
            Error::ParseFormat {
                path: source_name.into(),
                format: "safetensors".into(),
                reason: format!("invalid header json: {e}"),
            }
        })?;

    let mut graph = Graph::new(source_name);
    graph.set_metadata("source_format", "safetensors");
    if let Some(meta) = &header.metadata {
        for (k, v) in meta {
            graph.set_metadata(format!("st.{k}"), v.clone());
        }
    }

    for (name, entry) in &header.entries {
        let dtype = map_dtype(&entry.dtype)?;
        let (begin, end) = entry.data_offsets;
        if begin > end || end > bytes.len() - data_start {
            return Err(Error::ParseFormat {
                path: source_name.into(),
                format: "safetensors".into(),
                reason: format!("tensor `{name}` offsets [{begin}, {end}) out of bounds"),
            });
        }
        let desc = TensorDesc::new(entry.shape.clone(), dtype);
        let expected = desc.byte_size();
        let actual = end - begin;
        if actual != expected {
            return Err(Error::TensorSizeMismatch {
                name: name.clone(),
                expected,
                actual,
            });
        }
        let tensor = Tensor {
            desc,
            data: TensorData::from_vec(bytes[data_start + begin..data_start + end].to_vec()),
        };
        graph.push(name.clone(), Op::Constant { tensor }, Vec::<_>::new());
    }

    graph.validate()?;
    Ok(graph)
}

/// Ingest a `.safetensors` file.
pub fn ingest(path: &Path) -> Result<Graph> {
    let bytes = std::fs::read(path)?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    parse_bytes(&bytes, &name)
}

/// Encode tensors into valid SafeTensors bytes.
///
/// Used by round-trip tests and (later) by `.tptpkg` export; kept public
/// because it doubles as the reference implementation of the container.
pub fn encode(tensors: &BTreeMap<String, Tensor>) -> Result<Vec<u8>> {
    #[derive(serde::Serialize)]
    struct OutEntry<'a> {
        dtype: &'static str,
        shape: &'a [usize],
        data_offsets: (usize, usize),
    }

    fn dtype_tag(dtype: DType) -> Result<&'static str> {
        Ok(match dtype {
            DType::Bool => "BOOL",
            DType::U8 => "U8",
            DType::U16 => "U16",
            DType::U32 => "U32",
            DType::U64 => "U64",
            DType::I8 => "I8",
            DType::I16 => "I16",
            DType::I32 => "I32",
            DType::I64 => "I64",
            DType::F16 => "F16",
            DType::Bf16 => "BF16",
            DType::F32 => "F32",
            DType::F64 => "F64",
            other => {
                return Err(Error::UnsupportedDType(other.to_string()));
            }
        })
    }

    let mut entries = BTreeMap::new();
    let mut body = Vec::new();
    for (name, t) in tensors {
        let start = body.len();
        body.extend_from_slice(t.data.as_slice());
        entries.insert(
            name.clone(),
            OutEntry {
                dtype: dtype_tag(t.desc.dtype)?,
                shape: &t.desc.shape,
                data_offsets: (start, body.len()),
            },
        );
    }

    let mut header_json = serde_json::to_vec(&entries)?;
    // Pad header so the data region starts 8-byte aligned (spec convention).
    while (header_json.len() + 8) % 8 != 0 {
        header_json.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + header_json.len() + body.len());
    out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_json);
    out.extend_from_slice(&body);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_crucible_common::NodeId;

    fn sample() -> BTreeMap<String, Tensor> {
        let mut m = BTreeMap::new();
        m.insert(
            "layer.0.weight".to_string(),
            Tensor::from_f32(vec![2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.5, -6.25]),
        );
        m.insert(
            "bias".to_string(),
            Tensor::from_f32(vec![3], &[7.0, 8.0, 9.0]),
        );
        m
    }

    #[test]
    fn encode_parse_roundtrip() {
        let tensors = sample();
        let bytes = encode(&tensors).unwrap();
        let g = parse_bytes(&bytes, "sample").unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(g.name, "sample");
        g.validate().unwrap();

        // BTreeMap order: `bias` sorts before `layer.0.weight`.
        let w = g.get_node(NodeId(1)).unwrap();
        assert_eq!(w.name, "layer.0.weight");
        match &w.op {
            Op::Constant { tensor } => {
                assert_eq!(tensor.desc.shape, vec![2, 3]);
                let vals = tensor.data.to_f32(&tensor.desc).unwrap();
                assert_eq!(&vals[4..6], &[5.5, -6.25]);
            }
            other => panic!("expected constant, got {other:?}"),
        }
    }

    #[test]
    fn truncated_file_rejected() {
        assert!(parse_bytes(b"\x01\x02", "tiny").is_err());
    }

    #[test]
    fn corrupt_header_rejected() {
        let mut bytes = vec![0u8; 8];
        bytes[..8].copy_from_slice(&64u64.to_le_bytes());
        bytes.extend_from_slice(b"not json at all");
        assert!(parse_bytes(&bytes, "bad").is_err());
    }

    #[test]
    fn out_of_bounds_offsets_rejected() {
        let header = br#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,9999]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0u8; 16]);
        let err = parse_bytes(&bytes, "oob").unwrap_err();
        assert!(matches!(err, Error::ParseFormat { .. }));
    }

    #[test]
    fn unsupported_dtype_reports_name() {
        let header = br#"{"a":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.push(0);
        let err = parse_bytes(&bytes, "f8").unwrap_err();
        assert!(err.to_string().contains("F8_E4M3"));
    }
}
