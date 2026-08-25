//! PyTorch checkpoint ingestion (`torch.save` zip format, protocol 2+).
//!
//! Layout of a modern checkpoint: an uncompressed zip holding `data.pkl`
//! (a pickle stream) and one raw blob per storage under `<prefix>/data/<key>`.
//! This module runs a restricted pickle interpreter over `data.pkl`, collects
//! storage persistent-IDs, materializes every `_rebuild_tensor_v2` tensor from
//! its storage slice, and flattens nested state-dicts into dotted IR names.
//!
//! Limitations (surfacing as structured errors): deflated members, legacy
//! pre-1.6 tar checkpoints, non-contiguous tensors, and any pickle `REDUCE`
//! outside the torch rebuild/OrderedDict whitelist.

use std::path::Path;

use tpt_crucible_common::dtype::DType;
use tpt_crucible_common::error::{Error, Result};
use tpt_crucible_common::ops::Op;
use tpt_crucible_common::{Graph, Tensor, TensorData, TensorDesc};

/// Map a torch storage class name onto a TPT-IR dtype.
fn storage_dtype(type_name: &str) -> Result<DType> {
    Ok(match type_name.rsplit('.').next().unwrap_or(type_name) {
        "FloatStorage" => DType::F32,
        "DoubleStorage" => DType::F64,
        "HalfStorage" => DType::F16,
        "BFloat16Storage" => DType::Bf16,
        "IntStorage" => DType::I32,
        "LongStorage" => DType::I64,
        "ShortStorage" => DType::I16,
        "CharStorage" => DType::I8,
        "ByteStorage" => DType::U8,
        "BoolStorage" => DType::Bool,
        other => {
            return Err(Error::UnsupportedDType(format!("torch storage `{other}`")));
        }
    })
}

fn bad(reason: impl Into<String>) -> Error {
    Error::ParseFormat {
        path: "<pytorch>".into(),
        format: "pytorch".into(),
        reason: reason.into(),
    }
}

/// Find the storage payload for `key` (`data/<key>` under any prefix).
fn find_storage<'e>(entries: &'e [crate::pkzip::ZipEntry], key: &str) -> Result<&'e [u8]> {
    let direct = format!("data/{key}");
    for e in entries {
        if e.name == direct || e.name.ends_with(&format!("/{direct}")) {
            return Ok(&e.data);
        }
    }
    Err(bad(format!("storage blob `{key}` missing from archive")))
}

/// Reject non-contiguous layouts up front (we only materialize dense rows).
fn ensure_contiguous(size: &[i64], stride: &[i64]) -> Result<()> {
    let mut expected = 1i64;
    let mut ok = size.len() == stride.len();
    if ok {
        for i in (0..size.len()).rev() {
            ok &= stride[i] == expected;
            expected *= size[i].max(0);
        }
    }
    if !ok {
        return Err(Error::UnsupportedOperation(
            "non-contiguous tensors are not supported yet".into(),
        ));
    }
    Ok(())
}

/// Flatten a decoded pickle value into `(dotted name, tensor meta)` pairs.
fn flatten(prefix: &str, v: &Value, out: &mut Vec<(String, TensorMeta)>) {
    match v {
        Value::Dict(pairs) => {
            for (k, val) in pairs {
                if let Some(k) = as_str(k) {
                    let child = if prefix.is_empty() {
                        k.to_string()
                    } else {
                        format!("{prefix}.{k}")
                    };
                    flatten(&child, val, out);
                }
            }
        }
        Value::List(items) => {
            for (i, val) in items.iter().enumerate() {
                let child = if prefix.is_empty() {
                    format!("{i}")
                } else {
                    format!("{prefix}.{i}")
                };
                flatten(&child, val, out);
            }
        }
        Value::Tensor(meta) => out.push((prefix.to_string(), meta.clone())),
        _ => {}
    }
}

use crate::pickle::{TensorMeta, Value};

fn as_str(v: &Value) -> Option<&str> {
    match v {
        Value::Str(s) => Some(s),
        _ => None,
    }
}

/// Build a weight-only TPT-IR graph from torch.save zip bytes.
pub fn build_graph(data: &[u8], source_name: &str) -> Result<Graph> {
    let entries = crate::pkzip::read_entries(data)?;

    let pkl = entries
        .iter()
        .find(|e| e.name == "data.pkl" || e.name.ends_with("/data.pkl"))
        .ok_or_else(|| bad("archive has no data.pkl"))?;

    let (root, storages) = crate::pickle::Vm::new(&pkl.data).run()?;
    let mut flat = Vec::new();
    flatten("", &root, &mut flat);
    if flat.is_empty() {
        return Err(bad("checkpoint contains no tensors"));
    }

    let mut graph = Graph::new(source_name);
    graph.set_metadata("source_format", "pytorch");

    // Sort for deterministic node order regardless of dict ordering.
    flat.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, meta) in &flat {
        let (type_name, _numel) = storages.get(&meta.storage_key).ok_or_else(|| {
            bad(format!(
                "tensor `{name}` references unknown storage `{}`",
                meta.storage_key
            ))
        })?;
        let dtype = storage_dtype(type_name)?;
        ensure_contiguous(&meta.size, &meta.stride)?;

        let item = dtype.item_size().unwrap_or(1);
        let offset_bytes = meta.offset.max(0) as usize * item;
        let byte_len = meta.size.iter().product::<i64>().max(0) as usize * item;
        let blob = find_storage(&entries, &meta.storage_key)?;
        let end = offset_bytes
            .checked_add(byte_len)
            .ok_or_else(|| bad("tensor extent overflow"))?;
        if end > blob.len() {
            return Err(bad(format!(
                "tensor `{name}` needs bytes [{offset_bytes}, {end}) but its \
                 storage holds {}",
                blob.len()
            )));
        }

        let desc = TensorDesc::new(
            meta.size
                .iter()
                .map(|&d| d.max(0) as usize)
                .collect::<Vec<_>>(),
            dtype,
        );
        let tensor = Tensor {
            desc,
            data: TensorData::from_vec(blob[offset_bytes..end].to_vec()),
        };
        graph.push(name.clone(), Op::Constant { tensor }, Vec::<_>::new());
    }

    graph.validate()?;
    Ok(graph)
}

/// Ingest a `.pt` / `.pth` checkpoint file.
pub fn ingest(path: &Path) -> Result<Graph> {
    let bytes = std::fs::read(path)?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    build_graph(&bytes, &name)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- minimal pickle encoders (test fixtures) ----

    fn proto(out: &mut Vec<u8>) {
        out.push(0x80);
        out.push(2);
    }

    fn short_str(out: &mut Vec<u8>, s: &str) {
        out.push(0x8c);
        out.push(s.len() as u8);
        out.extend_from_slice(s.as_bytes());
    }

    fn binint1(out: &mut Vec<u8>, v: i64) {
        out.push(b'K');
        out.push(v as u8);
    }

    fn stack_global(out: &mut Vec<u8>, module: &str, name: &str) {
        short_str(out, module); // module below
        short_str(out, name); // name on top
        out.push(0x93); // STACK_GLOBAL
    }

    /// Storage persistent-ID tuple contents **inside an existing MARK**:
    /// `('storage', <ty global>, <key>, 'cpu', <numel>)` + BINPERSID,
    /// leaving a tensor placeholder above the outer mark.
    fn persid_storage(out: &mut Vec<u8>, ty: &str, key: &str, numel: i64) {
        out.push(0x28); // MARK — nested pid tuple start
        short_str(out, "storage"); // pid tag
        stack_global(out, "torch", ty);
        short_str(out, key);
        short_str(out, "cpu");
        binint1(out, numel);
        out.push(b't'); // TUPLE (closes this nested mark)
        out.push(b'Q'); // BINPERSID
    }

    /// Full tensor emission:
    /// `_rebuild_tensor_v2(storage, offset, size, stride, False, ())`.
    fn emit_tensor(
        out: &mut Vec<u8>,
        ty: &str,
        key: &str,
        numel: i64,
        offset: i64,
        size: &[i64],
        stride: &[i64],
    ) {
        stack_global(out, "torch._utils", "_rebuild_tensor_v2"); // callable
        out.push(0x28); // MARK — args start
        persid_storage(out, ty, key, numel);
        binint1(out, offset);
        out.push(0x5d); // EMPTY_LIST — size
        for &d in size {
            binint1(out, d);
            out.push(b'a');
        }
        out.push(0x5d); // EMPTY_LIST — stride
        for &d in stride {
            binint1(out, d);
            out.push(b'a');
        }
        out.push(0x89); // NEWFALSE — requires_grad
        out.push(b')'); // EMPTY_TUPLE — hooks
        out.push(b't'); // TUPLE(args)
        out.push(b'R'); // REDUCE
    }

    /// Fixture: {"model": {"weight": [2,3] f32, "bias": [2] f32}}
    fn state_dict_pkl() -> Vec<u8> {
        let mut p = Vec::new();
        proto(&mut p);
        p.push(b'}'); // outer dict
        p.push(0x28); // MARK — outer pairs
        short_str(&mut p, "model");

        p.push(b'}'); // inner dict
        p.push(0x28); // MARK — inner pairs
        short_str(&mut p, "weight");
        emit_tensor(&mut p, "FloatStorage", "0", 6, 0, &[2, 3], &[3, 1]);
        short_str(&mut p, "bias");
        emit_tensor(&mut p, "FloatStorage", "1", 2, 0, &[2], &[1]);
        p.push(b'u'); // SETITEMS inner

        p.push(b'u'); // SETITEMS outer
        stop(&mut p);
        p
    }

    fn stop(out: &mut Vec<u8>) {
        out.push(b'.');
    }

    fn zip_fixture() -> Vec<u8> {
        let mut weight = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            weight.extend_from_slice(&v.to_le_bytes());
        }
        let mut bias = Vec::new();
        for v in [7.0f32, 8.0] {
            bias.extend_from_slice(&v.to_le_bytes());
        }
        crate::pkzip::build_zip(&[
            ("archive/data.pkl", state_dict_pkl()),
            ("archive/data/0", weight),
            ("archive/data/1", bias),
            ("archive/byteorder", b"little".to_vec()),
        ])
    }

    #[test]
    fn end_to_end_state_dict() {
        let z = zip_fixture();
        let g = build_graph(&z, "ckpt").unwrap();
        g.validate().unwrap();

        assert_eq!(g.len(), 2);

        let w = g.nodes.iter().find(|n| n.name == "model.weight").unwrap();
        match &w.op {
            Op::Constant { tensor } => {
                assert_eq!(tensor.desc.shape, vec![2, 3]);
                assert_eq!(tensor.desc.dtype, DType::F32);
                let vals = tensor.data.to_f32(&tensor.desc).unwrap();
                assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            }
            other => panic!("expected constant, got {other:?}"),
        }

        let b = g.nodes.iter().find(|n| n.name == "model.bias").unwrap();
        match &b.op {
            Op::Constant { tensor } => {
                let vals = tensor.data.to_f32(&tensor.desc).unwrap();
                assert_eq!(vals, vec![7.0, 8.0]);
            }
            _ => panic!("expected constant"),
        }

        // JSON roundtrip for good measure.
        let json = g.to_json_string().unwrap();
        assert_eq!(Graph::from_json_str(&json).unwrap(), g);
    }

    #[test]
    fn non_contiguous_rejected() {
        let mut p = Vec::new();
        proto(&mut p);
        p.push(b'}');
        short_str(&mut p, "w");
        emit_tensor(&mut p, "FloatStorage", "0", 4, 0, &[2, 2], &[4, 1]); // stride[0] must be 2
        p.push(b's');
        stop(&mut p);

        let z =
            crate::pkzip::build_zip(&[("archive/data.pkl", p), ("archive/data/0", vec![0u8; 16])]);
        let err = build_graph(&z, "nc").unwrap_err();
        assert!(err.to_string().contains("non-contiguous"));
    }

    #[test]
    fn unknown_reduce_target_names_itself() {
        let mut p = Vec::new();
        proto(&mut p);
        p.push(b'}');
        short_str(&mut p, "k");
        stack_global(&mut p, "copyreg", "_reconstructor");
        p.push(b')'); // EMPTY_TUPLE args
        p.push(b'R'); // REDUCE
        p.push(b's');
        stop(&mut p);

        let z =
            crate::pkzip::build_zip(&[("archive/data.pkl", p), ("archive/data/0", vec![0u8; 4])]);
        let err = build_graph(&z, "bad").unwrap_err();
        assert!(err.to_string().contains("_reconstructor"));
    }

    #[test]
    fn missing_data_pkl_errors() {
        let z = crate::pkzip::build_zip(&[("other/file", vec![1])]);
        let err = build_graph(&z, "m").unwrap_err();
        assert!(err.to_string().contains("data.pkl"));
    }

    #[test]
    fn ingests_from_disk_via_full_detection_path() {
        let dir = std::env::temp_dir().join("catalyst-pt-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let pt = dir.join("model.pt");
        std::fs::write(&pt, zip_fixture()).unwrap();

        // Extension detection -> registry -> pickle/zip parser.
        let g = crate::ingest_path(&pt).unwrap();
        assert_eq!(g.metadata("source_format"), Some("pytorch"));
        assert_eq!(g.len(), 2);
    }
}
