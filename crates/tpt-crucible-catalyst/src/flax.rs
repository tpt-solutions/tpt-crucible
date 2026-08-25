//! JAX/Flax `.msgpack` parameter-bundle ingestion.
//!
//! Flax checkpoints (`flax.serialization.msgpack_serialize`, used by
//! `orbax`-free flows and `save_args`) are plain MessagePack pytrees whose
//! leaves carry ndarrays as **extension type 1**; each payload is *itself*
//! msgpack: `[shape_tuple, numpy_dtype_name, row-major bytes]`. Scalars ride
//! the same layout under extension type 3, complex numbers under type 2, and
//! arrays above 1 GiB are wrapped in a `__msgpack_chunked_array__` dict of
//! concatenated chunks.
//!
//! This module decodes the bundle into a weight-only TPT-IR graph: every
//! array leaf becomes a `Constant` node named by its `.`-joined tree path
//! (e.g. `params.decoder.layers.0.self_attn.query.kernel`). Raw `bin`
//! leaves become U8 constants; unsupported dtypes (complex, object) are
//! rejected naming the NumPy dtype.

use std::path::Path;

use tpt_crucible_common::dtype::DType;
use tpt_crucible_common::error::{Error, Result};
use tpt_crucible_common::ops::Op;
use tpt_crucible_common::{Graph, Tensor, TensorData, TensorDesc};

use crate::msgpack::{self, Mp};

const FORMAT: &str = "jax-flax";

/// `_MsgpackExtType.ndarray`
const EXT_NDARRAY: i8 = 1;
/// `_MsgpackExtType.native_complex`
#[allow(dead_code)]
const EXT_COMPLEX: i8 = 2;
/// `_MsgpackExtType.npscalar`
const EXT_NPSCALAR: i8 = 3;
/// Chunked-array wrapper marker key (>1 GiB leaves).
const CHUNKED_MARKER: &str = "__msgpack_chunked_array__";

fn malformed(reason: impl Into<String>) -> Error {
    Error::ParseFormat {
        path: "<jax-flax>".into(),
        format: FORMAT.into(),
        reason: reason.into(),
    }
}

/// Map a NumPy dtype name onto a TPT-IR dtype.
///
/// Covers every fixed-size numeric name `np.dtype` produces that TPT-IR can
/// represent, plus JAX's `bfloat16`.
fn dtype_from_name(name: &str) -> Result<DType> {
    Ok(match name {
        "bool" => DType::Bool,
        "uint8" => DType::U8,
        "uint16" => DType::U16,
        "uint32" => DType::U32,
        "uint64" => DType::U64,
        "int8" => DType::I8,
        "int16" => DType::I16,
        "int32" => DType::I32,
        "int64" => DType::I64,
        "float16" => DType::F16,
        "bfloat16" => DType::Bf16,
        "float32" => DType::F32,
        "float64" => DType::F64,
        other => {
            return Err(Error::UnsupportedDType(match other {
                "complex64" | "complex128" => other.to_string(),
                _ => format!("numpy dtype `{other}`"),
            }))
        }
    })
}

/// Decode one ndarray leaf payload:
/// msgpack array `[shape(ints), dtype_name(str), data(bin)]`.
fn decode_ndarray(_code: i8, payload: &[u8], leaf_name: &str) -> Result<Tensor> {
    let decoded = msgpack::parse(payload)
        .map_err(|e| malformed(format!("ndarray leaf `{leaf_name}`: {e}")))?;
    let mut parts = match decoded {
        Mp::Array(parts) if parts.len() == 3 => parts,
        other => {
            return Err(malformed(format!(
                "ndarray leaf `{leaf_name}` payload is {:?}, expected [shape, dtype, data]",
                kind_of(&other)
            )))
        }
    };
    let data_mp = parts.pop().expect("len checked");
    let dtype_mp = parts.pop().expect("len checked");
    let shape_mp = parts.pop().expect("len checked");

    let shape: Vec<usize> = match shape_mp {
        Mp::Array(dims) => dims
            .into_iter()
            .map(|d| {
                d.as_int()
                    .and_then(|v| usize::try_from(v.max(0)).ok())
                    .ok_or_else(|| {
                        malformed(format!(
                            "ndarray leaf `{leaf_name}` has a non-integer or negative dim"
                        ))
                    })
            })
            .collect::<Result<_>>()?,
        _ => {
            return Err(malformed(format!(
                "ndarray leaf `{leaf_name}` shape is not an array"
            )))
        }
    };
    let dtype_name = match dtype_mp {
        Mp::Str(s) => s,
        _ => {
            return Err(malformed(format!(
                "ndarray leaf `{leaf_name}` dtype is not a string"
            )))
        }
    };
    let data = match data_mp {
        Mp::Bin(b) => b,
        _ => {
            return Err(malformed(format!(
                "ndarray leaf `{leaf_name}` data is not binary"
            )))
        }
    };

    let dtype = dtype_from_name(&dtype_name)?;
    let desc = TensorDesc::new(shape, dtype);
    let expected = desc.byte_size();
    if !data.is_empty() && data.len() != expected {
        return Err(Error::TensorSizeMismatch {
            name: leaf_name.to_string(),
            expected,
            actual: data.len(),
        });
    }
    Ok(Tensor {
        desc,
        data: TensorData::from_vec(data),
    })
}

/// Short human label for a value (used in diagnostics).
fn kind_of(v: &Mp) -> &'static str {
    match v {
        Mp::Nil => "nil",
        Mp::Bool(_) => "bool",
        Mp::Int(_) | Mp::F64(_) => "number",
        Mp::Str(_) => "string",
        Mp::Bin(_) => "binary",
        Mp::Array(_) => "array",
        Mp::Map(_) => "map",
        Mp::Ext(..) => "ext",
    }
}

// ---- pytree walk -------------------------------------------------------------

/// Walk the decoded pytree, emitting one `Constant` node per array/binary
/// leaf named by its `.`-joined path.
fn walk(b: &mut Graph, v: &Mp, path: &str) -> Result<()> {
    // Chunked arrays (>1 GiB leaves) arrive as marker dicts; decode before
    // treating the map as tree structure.
    if let Some(tensor) = unchunk(v)? {
        b.push(path, Op::Constant { tensor }, Vec::<_>::new());
        return Ok(());
    }

    match v {
        Mp::Map(entries) => {
            for (k, val) in entries {
                let key = k
                    .as_str()
                    .ok_or_else(|| malformed(format!("map key under `{path}` is not a string")))?;
                let child = if path.is_empty() {
                    key.to_string()
                } else {
                    format!("{path}.{key}")
                };
                walk(b, val, &child)?;
            }
            Ok(())
        }
        // Flax tuples restore as lists; index them like dict keys.
        Mp::Array(items) => {
            for (i, val) in items.iter().enumerate() {
                walk(b, val, &format!("{path}.{i}"))?;
            }
            Ok(())
        }
        Mp::Ext(code @ (EXT_NDARRAY | EXT_NPSCALAR), payload) => {
            let tensor = decode_ndarray(*code, payload, path)?;
            b.push(path, Op::Constant { tensor }, Vec::<_>::new());
            Ok(())
        }
        Mp::Bin(bytes) => {
            let tensor = Tensor {
                desc: TensorDesc::new(vec![bytes.len()], DType::U8),
                data: TensorData::from_vec(bytes.clone()),
            };
            b.push(path, Op::Constant { tensor }, Vec::<_>::new());
            Ok(())
        }
        // Native complex numbers have no TPT-IR dtype.
        Mp::Ext(EXT_COMPLEX, _) => Err(Error::UnsupportedDType("complex".into())),
        Mp::Ext(other, _) => Err(malformed(format!(
            "unknown extension type {other} at `{path}`"
        ))),
        // Scalars (version markers, step counters, …) carry no weights.
        Mp::Nil | Mp::Bool(_) | Mp::Int(_) | Mp::F64(_) | Mp::Str(_) => Ok(()),
    }
}

/// Decode a `{"__msgpack_chunked_array__": true, "shape": {...},
/// "chunks": {"0": ext1, ...}}` wrapper into one tensor; `Ok(None)` when `v`
/// is not such a wrapper.
fn unchunk(v: &Mp) -> Result<Option<Tensor>> {
    let Some(entries) = v.as_map() else {
        return Ok(None);
    };
    if !entries
        .iter()
        .any(|(k, _)| matches!(k.as_str(), Some(CHUNKED_MARKER)))
    {
        return Ok(None);
    }

    let mut dims: Vec<usize> = Vec::new();
    let mut chunks: Vec<(usize, Tensor)> = Vec::new();
    for (k, val) in entries {
        match k.as_str() {
            Some("shape") => {
                let Some(shape_entries) = val.as_map() else {
                    return Err(malformed("chunked array shape is not a map"));
                };
                // Keys are stringified indices (`_tuple_to_dict`).
                let mut kv: Vec<(usize, i64)> = shape_entries
                    .iter()
                    .map(|(ik, iv)| {
                        let idx: usize = ik
                            .as_str()
                            .and_then(|s| s.parse().ok())
                            .ok_or_else(|| malformed("chunk shape key"))?;
                        let dim = iv.as_int().ok_or_else(|| malformed("chunk shape dim"))?;
                        Ok((idx, dim))
                    })
                    .collect::<Result<_>>()?;
                kv.sort_by_key(|(i, _)| *i);
                dims.extend(kv.into_iter().map(|(_, d)| d.max(0) as usize));
            }
            Some("chunks") => {
                let Some(chunk_entries) = val.as_map() else {
                    return Err(malformed("chunked array chunks is not a map"));
                };
                for (ck, cv) in chunk_entries {
                    let idx: usize = ck
                        .as_str()
                        .and_then(|s| s.parse().ok())
                        .ok_or_else(|| malformed("chunk key"))?;
                    let Mp::Ext(EXT_NDARRAY, payload) = cv else {
                        return Err(malformed("chunk payload is not an ndarray ext"));
                    };
                    let t = decode_ndarray(EXT_NDARRAY, payload, CHUNKED_MARKER)?;
                    chunks.push((idx, t));
                }
            }
            _ => {}
        }
    }
    if chunks.is_empty() {
        return Err(malformed("chunked array carries no chunks"));
    }
    chunks.sort_by_key(|(i, _)| *i);

    // Chunks must share one dtype; the leaf is their concatenation reshaped
    // to the declared shape.
    let dtype = chunks[0].1.desc.dtype;
    if chunks.iter().any(|(_, t)| t.desc.dtype != dtype) {
        return Err(malformed("chunked array chunks have mixed dtypes"));
    }
    let data_len: usize = chunks.iter().map(|(_, t)| t.data.as_slice().len()).sum();
    let desc = TensorDesc::new(dims, dtype);
    if data_len != desc.byte_size() {
        return Err(Error::TensorSizeMismatch {
            name: CHUNKED_MARKER.to_string(),
            expected: desc.byte_size(),
            actual: data_len,
        });
    }
    let mut data = Vec::with_capacity(data_len);
    for (_, t) in &chunks {
        data.extend_from_slice(t.data.as_slice());
    }
    Ok(Some(Tensor {
        desc,
        data: TensorData::from_vec(data),
    }))
}

// ---- entry points --------------------------------------------------------------

/// Parse a Flax `.msgpack` parameter bundle into a weight-only TPT-IR graph.
pub fn parse_bytes(bytes: &[u8], source_name: &str) -> Result<Graph> {
    let tree = msgpack::parse(bytes)?;
    let mut g = Graph::new(source_name);
    g.set_metadata("source_format", FORMAT);
    walk(&mut g, &tree, "")?;

    if g.is_empty() {
        return Err(malformed(
            "bundle contains no array leaves (not a Flax parameter pytree?)",
        ));
    }
    g.validate()?;
    Ok(g)
}

/// Ingest a `.msgpack` Flax parameter bundle.
pub fn ingest(path: &Path) -> Result<Graph> {
    let bytes = std::fs::read(path)?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    parse_bytes(&bytes, &name)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    // ---- minimal msgpack writer (test fixtures) --------------------------

    fn w_marker(out: &mut Vec<u8>, n: usize, fix_base: u8, m16: u8, m32: u8) {
        // MessagePack multi-byte ints are big-endian.
        if n < 16 {
            out.push(fix_base | n as u8);
        } else if n < 65536 {
            out.push(m16);
            out.extend_from_slice(&(n as u16).to_be_bytes());
        } else {
            out.push(m32);
            out.extend_from_slice(&(n as u32).to_be_bytes());
        }
    }

    fn w_str(out: &mut Vec<u8>, s: &str) {
        w_marker(out, s.len(), 0xa0, 0xda, 0xdb);
        out.extend_from_slice(s.as_bytes());
    }

    fn w_bin(out: &mut Vec<u8>, b: &[u8]) {
        // `bin` has no fix-form; always emit an explicit-length marker.
        if b.len() < 256 {
            out.push(0xc4);
            out.push(b.len() as u8);
        } else if b.len() < 65536 {
            out.push(0xc5);
            out.extend_from_slice(&(b.len() as u16).to_be_bytes());
        } else {
            out.push(0xc6);
            out.extend_from_slice(&(b.len() as u32).to_be_bytes());
        }
        out.extend_from_slice(b);
    }

    fn w_uint(out: &mut Vec<u8>, v: u64) {
        if v < 128 {
            out.push(v as u8);
        } else {
            out.push(0xcf);
            out.extend_from_slice(&v.to_be_bytes());
        }
    }

    /// `ExtType(1, packb([shape, dtype_name, data]))` — a Flax ndarray leaf.
    fn w_ndarray(out: &mut Vec<u8>, dtype_name: &str, shape: &[usize], data: &[u8]) {
        let mut payload = Vec::new();
        w_marker(&mut payload, 3, 0x90, 0xdc, 0xdd);
        w_marker(&mut payload, shape.len(), 0x90, 0xdc, 0xdd);
        for &d in shape {
            w_uint(&mut payload, d as u64);
        }
        w_str(&mut payload, dtype_name);
        w_bin(&mut payload, data);
        assert!(payload.len() < 256, "fixture payloads stay in ext8 range");
        out.push(0xc7);
        out.push(payload.len() as u8);
        out.push(EXT_NDARRAY as u8);
        out.extend_from_slice(&payload);
    }

    fn f32_data(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// `{"params": {"dense": {"kernel": f32[2,3], "bias": f32[2]}},
    ///   "step": 7, "sidecar": bin[3]}` — the canonical bundle shape.
    pub(crate) fn fixture_bundle() -> Vec<u8> {
        let mut out = Vec::new();
        w_marker(&mut out, 3, 0x80, 0xde, 0xdf);

        w_str(&mut out, "params");
        w_marker(&mut out, 1, 0x80, 0xde, 0xdf);
        w_str(&mut out, "dense");
        w_marker(&mut out, 2, 0x80, 0xde, 0xdf);
        w_str(&mut out, "kernel");
        w_ndarray(
            &mut out,
            "float32",
            &[2, 3],
            &f32_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        w_str(&mut out, "bias");
        w_ndarray(&mut out, "bfloat16", &[2], &[0x3F, 0x80, 0x40, 0x00]);

        w_str(&mut out, "step");
        w_uint(&mut out, 7);

        w_str(&mut out, "sidecar");
        w_bin(&mut out, &[9, 8, 7]);
        out
    }

    // ---- tests -------------------------------------------------------------

    #[test]
    fn ingests_nested_pytree() {
        let g = parse_bytes(&fixture_bundle(), "bundle").unwrap();
        g.validate().unwrap();
        assert_eq!(g.metadata("source_format"), Some("jax-flax"));

        // kernel + bias + sidecar (step scalar is ignored).
        assert_eq!(g.len(), 3);

        let kernel = &g.nodes[0];
        assert_eq!(kernel.name, "params.dense.kernel");
        match &kernel.op {
            Op::Constant { tensor } => {
                assert_eq!(tensor.desc.shape, vec![2, 3]);
                assert_eq!(tensor.desc.dtype, DType::F32);
                assert_eq!(
                    tensor.data.as_slice(),
                    f32_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).as_slice()
                );
            }
            other => panic!("expected constant: {other:?}"),
        }

        let bias = &g.nodes[1];
        match &bias.op {
            Op::Constant { tensor } => {
                assert_eq!(bias.name, "params.dense.bias");
                assert_eq!(tensor.desc.dtype, DType::Bf16);
            }
            other => panic!("expected constant: {other:?}"),
        }

        let sidecar = &g.nodes[2];
        assert_eq!(sidecar.name, "sidecar");
        assert!(matches!(&sidecar.op, Op::Constant { tensor } if tensor.desc.dtype == DType::U8));
    }

    #[test]
    fn chunked_array_leaf_concatenates() {
        // {"w": {"__msgpack_chunked_array__": true, "shape": {"0": 4},
        //        "chunks": {"0": f32[2], "1": f32[2]}}}
        let bytes = assemble(|out| {
            w_marker(out, 1, 0x80, 0xde, 0xdf);
            w_str(out, "w");
            w_marker(out, 3, 0x80, 0xde, 0xdf);
            w_str(out, CHUNKED_MARKER);
            out.push(0xc3); // true
            w_str(out, "shape");
            w_marker(out, 1, 0x80, 0xde, 0xdf);
            w_str(out, "0");
            w_uint(out, 4);
            w_str(out, "chunks");
            w_marker(out, 2, 0x80, 0xde, 0xdf);
            w_str(out, "0");
            w_ndarray(out, "float32", &[2], &f32_data(&[1.0, 2.0]));
            w_str(out, "1");
            w_ndarray(out, "float32", &[2], &f32_data(&[3.0, 4.0]));
        });
        let g = parse_bytes(&bytes, "m").unwrap();
        assert_eq!(g.len(), 1);
        match &g.nodes[0].op {
            Op::Constant { tensor } => {
                assert_eq!(tensor.desc.shape, vec![4]);
                assert_eq!(
                    tensor.data.as_slice(),
                    f32_data(&[1.0, 2.0, 3.0, 4.0]).as_slice()
                );
            }
            other => panic!("expected constant: {other:?}"),
        }
    }

    /// Assemble a bundle from a single callback over the writer.
    fn assemble(f: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
        let mut out = Vec::new();
        f(&mut out);
        out
    }

    #[test]
    fn unsupported_dtype_named() {
        let bytes = assemble(|out| {
            w_marker(out, 1, 0x80, 0xde, 0xdf);
            w_str(out, "c");
            w_ndarray(out, "complex64", &[1], &[0; 8]);
        });
        let err = parse_bytes(&bytes, "m").unwrap_err();
        assert!(err.to_string().contains("complex64"), "{err}");
    }

    #[test]
    fn truncated_and_empty_bundles_rejected() {
        let full = fixture_bundle();
        let err = parse_bytes(&full[..full.len() - 4], "m").unwrap_err();
        assert!(err.to_string().contains("truncated message"), "{err}");

        // A bundle with no array leaves is not a parameter file.
        let empty = assemble(|out| {
            w_marker(out, 1, 0x80, 0xde, 0xdf);
            w_str(out, "step");
            w_uint(out, 3);
        });
        let err = parse_bytes(&empty, "m").unwrap_err();
        assert!(err.to_string().contains("no array leaves"), "{err}");
    }
}
