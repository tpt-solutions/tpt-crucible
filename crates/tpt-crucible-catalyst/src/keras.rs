//! Keras v3 (`.keras`) checkpoint ingestion.
//!
//! A `.keras` file is a zip archive containing `config.json`,
//! `metadata.json`, and one or more NPZ weight stores — either a single
//! `model.weights.npz` at the root or per-saveable `<path>/states.npz`
//! files written by newer Keras versions. NPZ members are STORED
//! (numpy `np.savez`); DEFLATED members are rejected with guidance.
//!
//! Weight names: array keys inside each NPZ are used verbatim, prefixed by
//! the store's directory path (`model/dense/states.npz` + key `kernel` →
//! `model.dense.kernel`). One leading underscore is stripped, mirroring the
//! Keras name generator. Legacy `.h5` weights surface as an explicit
//! unsupported error.

use std::path::Path;

use tpt_crucible_common::error::{Error, Result};
use tpt_crucible_common::ops::Op;
use tpt_crucible_common::{Graph, Tensor, TensorData, TensorDesc};

fn bad(reason: impl Into<String>) -> Error {
    Error::ParseFormat {
        path: "<keras>".into(),
        format: "keras".into(),
        reason: reason.into(),
    }
}

/// Ingest a `.keras` archive into a weight-only TPT-IR graph.
pub fn ingest(path: &Path) -> Result<Graph> {
    let data = std::fs::read(path)?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    build_graph(&data, &name)
}

/// Build a weight-only TPT-IR graph from `.keras` zip bytes.
pub fn build_graph(data: &[u8], source_name: &str) -> Result<Graph> {
    let entries = crate::pkzip::read_entries(data)?;

    // config.json is the v3-format signature.
    let has_config = entries.iter().any(|e| e.name == "config.json");
    if !has_config {
        return Err(bad(
            "no config.json — this is not a Keras v3 `.keras` archive",
        ));
    }

    // Collect every NPZ weight store: root-level `*.weights.npz` and
    // per-saveable `<dir>/states.npz`.
    let npz_entries: Vec<&crate::pkzip::ZipEntry> = entries
        .iter()
        .filter(|e| e.name.ends_with(".npz") || e.name == "model.weights.npz")
        .collect();

    if npz_entries.is_empty() {
        let h5_only = entries.iter().any(|e| e.name.ends_with(".h5"));
        return Err(if h5_only {
            Error::UnsupportedOperation(
                "legacy HDF5 keras weights are not supported; re-export the \
                 model in the native `.keras` format"
                    .into(),
            )
        } else {
            bad("archive contains no .npz weight stores")
        });
    }

    let mut tensors: Vec<(String, Tensor)> = Vec::new();
    for npz in &npz_entries {
        // Directory path of the store becomes the name prefix.
        let dir = match npz.name.rfind('/') {
            Some(i) => npz.name[..i].replace('/', "."),
            None => String::new(),
        };
        let inner = crate::pkzip::read_entries(&npz.data)?;
        for member in &inner {
            if !member.name.ends_with(".npy") {
                continue;
            }
            let key = member.name.trim_end_matches(".npy");
            let key = key.strip_prefix('_').unwrap_or(key);
            let full_name = if dir.is_empty() {
                key.to_string()
            } else {
                format!("{dir}.{key}")
            };
            let arr = crate::npy::parse_npy(&member.data)?;
            tensors.push((
                full_name,
                Tensor {
                    desc: TensorDesc::new(arr.shape.clone(), arr.dtype),
                    data: TensorData::from_vec(arr.data.clone()),
                },
            ));
        }
    }

    if tensors.is_empty() {
        return Err(bad("archive contained no weight arrays"));
    }
    tensors.sort_by(|a, b| a.0.cmp(&b.0));

    let mut graph = Graph::new(source_name);
    graph.set_metadata("source_format", "keras");
    for (name, tensor) in tensors {
        graph.push(name, Op::Constant { tensor }, Vec::<_>::new());
    }
    graph.validate()?;
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an .npy payload for f32 data.
    fn npy_f32(shape: &[usize], data: &[f32]) -> Vec<u8> {
        let shape_s: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        let mut h = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, ), }}",
            shape_s.join(", ")
        )
        .into_bytes();
        h.push(b'\n');
        while (h.len() + 10) % 64 != 0 {
            h.insert(h.len() - 1, b' ');
        }

        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(h.len() as u16).to_le_bytes());
        out.extend_from_slice(&h);
        for v in data {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn end_to_end_nested_states_npz() {
        // Inner NPZ: model/dense/states.npz {kernel:[2,3], bias:[2]}
        let inner = crate::pkzip::build_zip(&[
            (
                "_kernel.npy",
                npy_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ),
            ("_bias.npy", npy_f32(&[2], &[7.0, 8.0])),
        ]);
        // Outer .keras archive.
        let outer = crate::pkzip::build_zip(&[
            ("config.json", b"{}".to_vec()),
            ("metadata.json", b"{}".to_vec()),
            ("model/dense/states.npz", inner),
        ]);

        let g = build_graph(&outer, "m").unwrap();
        g.validate().unwrap();
        assert_eq!(g.metadata("source_format"), Some("keras"));
        assert_eq!(g.len(), 2);

        let k = g
            .nodes
            .iter()
            .find(|n| n.name == "model.dense.kernel")
            .unwrap();
        match &k.op {
            Op::Constant { tensor } => {
                assert_eq!(tensor.desc.shape, vec![2, 3]);
                let vals = tensor.data.to_f32(&tensor.desc).unwrap();
                assert_eq!(vals[5], 6.0);
            }
            _ => panic!("expected constant"),
        }
    }

    #[test]
    fn h5_only_rejected_with_guidance() {
        let outer = crate::pkzip::build_zip(&[
            ("config.json", b"{}".to_vec()),
            ("model.weights.h5", vec![1, 2, 3]),
        ]);
        let err = build_graph(&outer, "m").unwrap_err();
        assert!(err.to_string().contains("legacy HDF5"));
    }
}
