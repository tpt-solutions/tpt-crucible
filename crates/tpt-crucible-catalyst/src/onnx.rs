//! ONNX ingestion — Open Neural Network Exchange models.
//!
//! A native, dependency-free reader: ONNX is protobuf, so this module carries
//! a miniature wire-format decoder (varints, length-delimited fields, packed
//! repeated scalars) and maps the subset of `ModelProto` we care about onto
//! TPT-IR.
//!
//! Supported ops (v1): `MatMul`, `Gemm` (lowered to Transpose/MatMul/Add),
//! elementwise binaries and unaries (`Add Sub Mul Div Neg Exp Log Sqrt Sin Cos
//! Tanh Erf Relu Sigmoid Gelu`), `Softmax`, `LayerNormalization`,
//! `Reshape`, `Transpose`, `Concat`, `Cast`, `Identity`, and `Gather`
//! (lowered to [`Op::Embedding`] — the standard transformer use).
//! Anything else surfaces as [`Error::UnsupportedOperation`] naming the op.
//!
//! Initializers become `constant` nodes; graph inputs with static shapes
//! become `input` nodes; graph outputs are wrapped in `output` nodes.

use std::collections::HashMap;

use tpt_crucible_common::dtype::DType;
use tpt_crucible_common::error::{Error, Result};
use tpt_crucible_common::ops::{LayerNormAttrs, Op, ReshapeAttrs, SoftmaxAttrs, TransposeAttrs};
use tpt_crucible_common::{Graph, Tensor, TensorData, TensorDesc};

// ---- miniature protobuf wire-format decoder -------------------------------
//
// Shared with `tensorflow.rs` (`crate::pb`); this module binds it to ONNX's
// `<onnx>` diagnostic context.

use crate::pb::{repeated_f32, repeated_i64, str_field, Rd};

fn malformed(reason: &str) -> Error {
    Error::ParseFormat {
        path: "<onnx>".into(),
        format: "onnx".into(),
        reason: reason.into(),
    }
}

/// Walk every field of a protobuf message, dispatching on `(field, wire_type)`.
///
/// Handlers consume their own payloads from the cursor.
fn for_fields<F>(buf: &[u8], on: F) -> Result<()>
where
    F: FnMut(u32, u8, &mut Rd) -> Result<()>,
{
    crate::pb::for_fields(buf, "<onnx>", "onnx", on)
}

// ---- ONNX message model ----------------------------------------------------

/// Parsed `AttributeProto` (only scalar/`Tensor` attributes we consume).
#[derive(Debug, Default)]
struct OnnxAttr {
    name: String,
    i: i64,
    f: f32,
    ints: Vec<i64>,
    floats: Vec<f32>,
    tensor: Option<Vec<u8>>,
}

impl OnnxAttr {
    fn get<'m>(attrs: &'m [OnnxAttr], name: &str) -> Option<&'m OnnxAttr> {
        attrs.iter().find(|a| a.name == name)
    }

    fn i_or(attrs: &[OnnxAttr], name: &str, default: i64) -> Result<i64> {
        Ok(Self::get(attrs, name).map(|a| a.i).unwrap_or(default))
    }

    fn f_or(attrs: &[OnnxAttr], name: &str, default: f32) -> Result<f32> {
        Ok(Self::get(attrs, name).map(|a| a.f).unwrap_or(default))
    }
}

fn parse_attr(buf: &[u8]) -> Result<OnnxAttr> {
    let mut a = OnnxAttr::default();
    for_fields(buf, |field, wt, r| {
        match field {
            1 => a.name = str_field(r)?,
            2 => a.f = r.fixed32()?,
            3 => a.i = r.varint()? as i64,
            4 => a.name.push_str(&String::from_utf8_lossy(r.bytes()?)), // `s`
            5 => a.tensor = Some(r.bytes()?.to_vec()),
            7 => repeated_f32(wt, r, &mut a.floats)?,
            8 => repeated_i64(wt, r, &mut a.ints)?,
            20 => {
                let _kind = r.varint()?;
            }
            _ => r.skip(wt)?,
        }
        Ok(())
    })?;
    Ok(a)
}

/// Parsed `TensorProto`.
#[derive(Debug, Default)]
struct OnnxTensor {
    name: String,
    dtype: i64,
    dims: Vec<i64>,
    raw_data: Option<Vec<u8>>,
    float_data: Vec<f32>,
    int64_data: Vec<i64>,
}

fn parse_tensor(buf: &[u8]) -> Result<OnnxTensor> {
    let mut t = OnnxTensor::default();
    for_fields(buf, |field, wt, r| {
        match field {
            1 => repeated_i64(wt, r, &mut t.dims)?,
            2 => t.dtype = r.varint()? as i64,
            4 => repeated_f32(wt, r, &mut t.float_data)?,
            7 => repeated_i64(wt, r, &mut t.int64_data)?,
            8 => t.name = str_field(r)?,
            9 => t.raw_data = Some(r.bytes()?.to_vec()),
            _ => r.skip(wt)?,
        }
        Ok(())
    })?;
    Ok(t)
}

/// Parsed `ValueInfoProto` with a fully static shape.
#[derive(Debug)]
struct ValueInfo {
    name: String,
    dtype: i64,
    shape: Vec<i64>,
}

fn parse_value_info(buf: &[u8]) -> Result<ValueInfo> {
    let mut vi = ValueInfo {
        name: String::new(),
        dtype: 0,
        shape: Vec::new(),
    };
    for_fields(buf, |field, wt, r| {
        match field {
            1 => vi.name = str_field(r)?,
            2 => {
                let (dtype, shape) = parse_type_proto(r.bytes()?)?;
                vi.dtype = dtype;
                vi.shape = shape;
            }
            _ => r.skip(wt)?,
        }
        Ok(())
    })?;
    Ok(vi)
}

/// Parse `TypeProto` → `(elem_type, static_shape)`.
fn parse_type_proto(buf: &[u8]) -> Result<(i64, Vec<i64>)> {
    let mut out = (0i64, Vec::new());
    for_fields(buf, |field, wt, r| {
        match field {
            1 => {
                // Tensor { elem_type(1), shape(2) }
                let tensor = r.bytes()?;
                let mut elem = 0i64;
                let mut dims = Vec::new();
                for_fields(tensor, |f2, w2, r2| {
                    match f2 {
                        1 => elem = r2.varint()? as i64,
                        2 => dims = parse_shape(r2.bytes()?)?,
                        _ => r2.skip(w2)?,
                    }
                    Ok(())
                })?;
                out = (elem, dims);
            }
            _ => r.skip(wt)?,
        }
        Ok(())
    })?;
    Ok(out)
}

/// Parse `TensorShapeProto` → dims (symbolic dims unsupported).
fn parse_shape(buf: &[u8]) -> Result<Vec<i64>> {
    let mut dims = Vec::new();
    for_fields(buf, |field, wt, r| {
        match field {
            1 => {
                // Dimension { dim_value(1), dim_param(3) }
                let dim = r.bytes()?;
                for_fields(dim, |f2, w2, r2| {
                    match f2 {
                        1 => dims.push(r2.varint()? as i64),
                        3 => {
                            return Err(malformed(
                                "symbolic dimensions are not supported; models need static shapes",
                            ));
                        }
                        _ => r2.skip(w2)?,
                    }
                    Ok(())
                })?;
            }
            _ => r.skip(wt)?,
        }
        Ok(())
    })?;
    Ok(dims)
}

/// Parsed `NodeProto`.
#[derive(Debug, Default)]
struct OnnxNode {
    inputs: Vec<String>,
    outputs: Vec<String>,
    name: String,
    op_type: String,
    attrs: Vec<OnnxAttr>,
}

fn parse_node(buf: &[u8]) -> Result<OnnxNode> {
    let mut n = OnnxNode::default();
    for_fields(buf, |field, wt, r| {
        match field {
            1 => n.inputs.push(str_field(r)?),
            2 => n.outputs.push(str_field(r)?),
            3 => n.name = str_field(r)?,
            4 => n.op_type = str_field(r)?,
            5 => n.attrs.push(parse_attr(r.bytes()?)?),
            _ => r.skip(wt)?,
        }
        Ok(())
    })?;
    Ok(n)
}

/// Parsed `GraphProto`.
#[derive(Debug, Default)]
struct OnnxGraph {
    name: String,
    nodes: Vec<OnnxNode>,
    initializers: Vec<OnnxTensor>,
    inputs: Vec<ValueInfo>,
    outputs: Vec<ValueInfo>,
}

fn parse_graph(buf: &[u8]) -> Result<OnnxGraph> {
    let mut g = OnnxGraph::default();
    for_fields(buf, |field, wt, r| {
        match field {
            1 => g.nodes.push(parse_node(r.bytes()?)?),
            2 => g.name = str_field(r)?,
            5 => g.initializers.push(parse_tensor(r.bytes()?)?),
            11 => g.inputs.push(parse_value_info(r.bytes()?)?),
            12 => g.outputs.push(parse_value_info(r.bytes()?)?),
            _ => r.skip(wt)?,
        }
        Ok(())
    })?;
    Ok(g)
}

// ---- conversion to TPT-IR ---------------------------------------------------

use tpt_crucible_common::NodeId;

/// Map an ONNX `TensorProto.DataType` code onto a TPT-IR dtype.
fn onnx_dtype(code: i64) -> Result<DType> {
    Ok(match code {
        1 => DType::F32,
        2 => DType::U8,
        3 => DType::I8,
        4 => DType::U16,
        5 => DType::I16,
        6 => DType::I32,
        7 => DType::I64,
        9 => DType::Bool,
        10 => DType::F16,
        11 => DType::F64,
        12 => DType::U32,
        13 => DType::U64,
        16 => DType::Bf16,
        other => {
            return Err(Error::UnsupportedDType(format!("onnx elem_type {other}")));
        }
    })
}

/// Materialize an ONNX tensor into an IR `Tensor` (raw_data preferred).
fn tensor_from_onnx(t: &OnnxTensor) -> Result<Tensor> {
    let desc = TensorDesc::new(
        t.dims
            .iter()
            .map(|&d| d.max(0) as usize)
            .collect::<Vec<_>>(),
        onnx_dtype(t.dtype)?,
    );
    let expected = desc.byte_size();
    let data = if let Some(raw) = &t.raw_data {
        if raw.len() != expected {
            return Err(Error::TensorSizeMismatch {
                name: t.name.clone(),
                expected,
                actual: raw.len(),
            });
        }
        raw.clone()
    } else if desc.dtype == DType::F32 && !t.float_data.is_empty() {
        let mut b = Vec::with_capacity(expected);
        for v in &t.float_data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    } else if desc.dtype == DType::I64 && !t.int64_data.is_empty() {
        let mut b = Vec::with_capacity(expected);
        for v in &t.int64_data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    } else {
        return Err(Error::ParseFormat {
            path: "<onnx>".into(),
            format: "onnx".into(),
            reason: format!("initializer `{}` carries no supported inline data", t.name),
        });
    };
    Ok(Tensor {
        desc,
        data: TensorData::from_vec(data),
    })
}

/// Read integer values out of a constant node (used by `Reshape`).
fn const_i64s(g: &Graph, id: NodeId) -> Result<Vec<i64>> {
    let node = g.get_node(id).ok_or_else(|| malformed("missing node"))?;
    let tensor = match &node.op {
        Op::Constant { tensor } => tensor,
        _ => {
            return Err(Error::ParseFormat {
                path: "<onnx>".into(),
                format: "onnx".into(),
                reason: "expected a constant initializer for this input".into(),
            });
        }
    };
    let bytes = tensor.data.as_slice();
    match tensor.desc.dtype {
        DType::I64 => Ok(bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()),
        DType::I32 => Ok(bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
            .collect()),
        other => Err(Error::UnsupportedOperation(format!(
            "integer tensor required, found `{other}`"
        ))),
    }
}

/// Incremental graph under construction.
struct Builder {
    g: Graph,
    /// Value name → producing IR node.
    vals: HashMap<String, NodeId>,
    used_names: std::collections::HashSet<String>,
}

impl Builder {
    fn push(&mut self, name: &str, op: Op, inputs: Vec<NodeId>) -> NodeId {
        // ONNX node names may repeat; only honor them while unique.
        let unique = !name.is_empty() && self.used_names.insert(name.to_string());
        self.g.push(if unique { name } else { "" }, op, inputs)
    }
}

/// Build the IR for one resolved ONNX node, registering its outputs.
///
/// Returns `false` for `Identity` (pure alias — no IR node emitted).
fn build_node(b: &mut Builder, node: &OnnxNode) -> Result<bool> {
    if node.outputs.len() > 1 {
        return Err(Error::UnsupportedOperation(format!(
            "multi-output onnx op `{}` ({})",
            node.op_type, node.name
        )));
    }

    // Resolve inputs; empty strings are absent optional inputs.
    let mut ins: Vec<NodeId> = Vec::with_capacity(node.inputs.len());
    for name in &node.inputs {
        if name.is_empty() {
            continue;
        }
        ins.push(*b.vals.get(name).ok_or_else(|| malformed("unresolved"))?);
    }

    let attrs = &node.attrs;
    let produced: NodeId = match node.op_type.as_str() {
        "Identity" => {
            b.vals.insert(node.outputs[0].clone(), ins[0]);
            return Ok(false);
        }
        "Constant" => {
            let raw = OnnxAttr::get(attrs, "value")
                .and_then(|a| a.tensor.as_ref())
                .ok_or_else(|| malformed("Constant without `value` attribute"))?;
            let tensor = tensor_from_onnx(&parse_tensor(raw)?)?;
            b.push(&node.name, Op::Constant { tensor }, Vec::new())
        }
        "MatMul" => b.push(&node.name, Op::MatMul, ins),
        op @ ("Add" | "Sub" | "Mul" | "Div") => {
            let op = match op {
                "Add" => Op::Add,
                "Sub" => Op::Sub,
                "Mul" => Op::Mul,
                _ => Op::Div,
            };
            b.push(&node.name, op, ins)
        }
        op @ ("Neg" | "Exp" | "Log" | "Sqrt" | "Sin" | "Cos" | "Tanh" | "Erf" | "Relu"
        | "Sigmoid" | "Gelu") => {
            let op = match op {
                "Neg" => Op::Neg,
                "Exp" => Op::Exp,
                "Log" => Op::Log,
                "Sqrt" => Op::Sqrt,
                "Sin" => Op::Sin,
                "Cos" => Op::Cos,
                "Tanh" => Op::Tanh,
                "Erf" => Op::Erf,
                "Relu" => Op::Relu,
                "Sigmoid" => Op::Sigmoid,
                _ => Op::Gelu,
            };
            b.push(&node.name, op, ins)
        }
        "Softmax" => {
            let attrs = SoftmaxAttrs {
                axis: OnnxAttr::i_or(attrs, "axis", -1)? as i32,
            };
            b.push(&node.name, Op::Softmax { attrs }, ins)
        }
        "LayerNormalization" => {
            let attrs = LayerNormAttrs {
                eps: OnnxAttr::f_or(attrs, "epsilon", 1e-5)?,
            };
            b.push(&node.name, Op::LayerNorm { attrs }, ins)
        }
        "Reshape" => {
            let shape = const_i64s(&b.g, ins[1])?;
            b.push(
                &node.name,
                Op::Reshape {
                    attrs: ReshapeAttrs { shape },
                },
                vec![ins[0]],
            )
        }
        "Transpose" => {
            let perm = match OnnxAttr::get(attrs, "perm") {
                Some(a) => a.ints.iter().map(|&p| p as usize).collect(),
                None => {
                    return Err(Error::UnsupportedOperation(
                        "transpose without explicit perm (rank inference pending)".into(),
                    ));
                }
            };
            b.push(
                &node.name,
                Op::Transpose {
                    attrs: TransposeAttrs { perm },
                },
                ins,
            )
        }
        "Concat" => {
            let axis = OnnxAttr::i_or(attrs, "axis", i64::MIN)?;
            if axis < 0 {
                return Err(Error::UnsupportedOperation(
                    "negative concat axis requires rank inference (pending)".into(),
                ));
            }
            b.push(
                &node.name,
                Op::Concat {
                    axis: axis as usize,
                },
                ins,
            )
        }
        "Cast" => {
            let to = onnx_dtype(OnnxAttr::i_or(attrs, "to", i64::MIN)?)?;
            b.push(&node.name, Op::Cast { to }, ins)
        }
        // Standard transformer token embedding; documented approximation.
        "Gather" => b.push(&node.name, Op::Embedding, ins),
        "Gemm" => build_gemm(b, &node.name, attrs, ins)?,
        other => {
            return Err(Error::UnsupportedOperation(format!("onnx op `{other}`")));
        }
    };

    b.vals.insert(node.outputs[0].clone(), produced);
    Ok(true)
}

/// Lower `Gemm(A, B[, C])` into Transpose/MatMul/const-scale/Add nodes.
fn build_gemm(b: &mut Builder, name: &str, attrs: &[OnnxAttr], ins: Vec<NodeId>) -> Result<NodeId> {
    let alpha = OnnxAttr::f_or(attrs, "alpha", 1.0)?;
    let beta = OnnxAttr::f_or(attrs, "beta", 1.0)?;
    let trans_a = OnnxAttr::i_or(attrs, "transA", 0)? != 0;
    let trans_b = OnnxAttr::i_or(attrs, "transB", 0)? != 0;

    let mut a = ins[0];
    if trans_a {
        a = b.push(
            "",
            Op::Transpose {
                attrs: TransposeAttrs { perm: vec![1, 0] },
            },
            vec![a],
        );
    }
    let mut bt = ins[1];
    if trans_b {
        bt = b.push(
            "",
            Op::Transpose {
                attrs: TransposeAttrs { perm: vec![1, 0] },
            },
            vec![bt],
        );
    }
    let mut acc = b.push("", Op::MatMul, vec![a, bt]);
    if alpha != 1.0 {
        let k = b.push(
            "",
            Op::Constant {
                tensor: Tensor::from_f32(vec![], &[alpha]),
            },
            Vec::new(),
        );
        acc = b.push("", Op::Mul, vec![acc, k]);
    }
    if ins.len() > 2 && beta != 0.0 {
        let c = if beta == 1.0 {
            ins[2]
        } else {
            let k = b.push(
                "",
                Op::Constant {
                    tensor: Tensor::from_f32(vec![], &[beta]),
                },
                Vec::new(),
            );
            b.push("", Op::Mul, vec![ins[2], k])
        };
        acc = b.push(name, Op::Add, vec![acc, c]);
    }
    Ok(acc)
}

/// Parse an ONNX `ModelProto` into TPT-IR.
pub fn parse_bytes(bytes: &[u8], source_name: &str) -> Result<Graph> {
    let mut ir_version = 0u64;
    let mut producer = String::new();
    let mut graph_buf: Option<Vec<u8>> = None;
    for_fields(bytes, |field, wt, r| {
        match field {
            1 => ir_version = r.varint()?,
            3 => producer = str_field(r)?,
            7 => graph_buf = Some(r.bytes()?.to_vec()),
            _ => r.skip(wt)?,
        }
        Ok(())
    })?;

    let og = parse_graph(
        graph_buf
            .as_deref()
            .ok_or_else(|| malformed("model has no graph field"))?,
    )?;

    let mut b = Builder {
        g: Graph::new(source_name),
        vals: HashMap::new(),
        used_names: std::collections::HashSet::new(),
    };
    b.g.set_metadata("source_format", "onnx");
    b.g.set_metadata("onnx_ir_version", ir_version.to_string());
    if !producer.is_empty() {
        b.g.set_metadata("onnx_producer", producer);
    }

    // Initializers become constants (ONNX may also list them as graph inputs;
    // that overlap is treated as constants per the spec).
    for t in &og.initializers {
        let tensor = tensor_from_onnx(t)?;
        let id = b.push(&t.name.clone(), Op::Constant { tensor }, Vec::new());
        if b.vals.insert(t.name.clone(), id).is_some() {
            return Err(malformed(&format!("duplicate initializer `{}`", t.name)));
        }
    }

    // True model inputs.
    for vi in &og.inputs {
        if b.vals.contains_key(&vi.name) {
            continue;
        }
        if vi.dtype == 0 || vi.shape.is_empty() {
            return Err(Error::ParseFormat {
                path: "<onnx>".into(),
                format: "onnx".into(),
                reason: format!("input `{}` lacks a static shape", vi.name),
            });
        }
        let desc = TensorDesc::new(
            vi.shape
                .iter()
                .map(|&d| d.max(0) as usize)
                .collect::<Vec<_>>(),
            onnx_dtype(vi.dtype)?,
        );
        let id = b.push(vi.name.as_str(), Op::Input(desc), Vec::new());
        b.g.mark_input(id);
        b.vals.insert(vi.name.clone(), id);
    }

    // Resolve nodes; well-formed ONNX is topologically ordered but be lenient
    // with a deferred multi-pass sweep.
    let mut queue: std::collections::VecDeque<OnnxNode> = og.nodes.into_iter().collect();
    while !queue.is_empty() {
        let mut deferred = std::collections::VecDeque::new();
        let mut progressed = false;
        while let Some(n) = queue.pop_front() {
            let ready = n
                .inputs
                .iter()
                .filter(|i| !i.is_empty())
                .all(|i| b.vals.contains_key(i));
            if ready {
                build_node(&mut b, &n)?;
                progressed = true;
            } else {
                deferred.push_back(n);
            }
        }
        queue = deferred;
        if !queue.is_empty() && !progressed {
            let unresolved: Vec<String> = queue
                .iter()
                .flat_map(|n| n.inputs.iter().cloned())
                .filter(|i| !i.is_empty() && !b.vals.contains_key(i))
                .collect();
            return Err(malformed(&format!(
                "unresolved value names: {}",
                unresolved.join(", ")
            )));
        }
    }

    // Graph outputs wrap their values in `output` nodes.
    for o in &og.outputs {
        let inner = *b
            .vals
            .get(&o.name)
            .ok_or_else(|| malformed(&format!("graph output `{}` never produced", o.name)))?;
        let out = b.push(
            o.name.as_str(),
            Op::Output {
                name: o.name.clone(),
            },
            vec![inner],
        );
        b.g.mark_output(out);
    }

    b.g.validate()?;
    Ok(b.g)
}

/// Ingest an `.onnx` file.
pub fn ingest(path: &std::path::Path) -> Result<Graph> {
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

    // ---- protobuf encoding helpers (test fixtures) ----

    fn varint(mut v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                break;
            }
            out.push(b | 0x80);
        }
        out
    }

    fn tag(field: u32, wt: u8) -> Vec<u8> {
        varint(((field as u64) << 3) | wt as u64)
    }

    fn ld(field: u32, payload: &[u8]) -> Vec<u8> {
        let mut out = tag(field, 2);
        out.extend(varint(payload.len() as u64));
        out.extend_from_slice(payload);
        out
    }

    fn s(field: u32, v: &str) -> Vec<u8> {
        ld(field, v.as_bytes())
    }

    fn vi(field: u32, v: u64) -> Vec<u8> {
        let mut out = tag(field, 0);
        out.extend(varint(v));
        out
    }

    fn packed_i64s(field: u32, vals: &[i64]) -> Vec<u8> {
        let mut body = Vec::new();
        for &v in vals {
            body.extend(varint(v as u64));
        }
        ld(field, &body)
    }

    fn tensor_msg(name: &str, dtype: i64, dims: &[i64], raw: &[f32]) -> Vec<u8> {
        let mut m = s(8, name);
        m.extend(vi(2, dtype as u64));
        m.extend(packed_i64s(1, dims));
        // Our reader prefers raw_data.
        let mut raw_bytes = Vec::new();
        for v in raw {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }
        m.extend(ld(9, &raw_bytes));
        m
    }

    fn attr_i(name: &str, v: i64) -> Vec<u8> {
        let mut m = s(1, name);
        m.extend(vi(3, v as u64));
        m
    }

    fn attr_f(name: &str, v: f32) -> Vec<u8> {
        let mut m = s(1, name);
        m.extend(tag(2, 5));
        m.extend_from_slice(&v.to_le_bytes());
        m
    }

    fn node(op: &str, name: &str, ins: &[&str], outs: &[&str], attrs: Vec<Vec<u8>>) -> Vec<u8> {
        let mut m = Vec::new();
        for i in ins {
            m.extend(s(1, i));
        }
        for o in outs {
            m.extend(s(2, o));
        }
        if !name.is_empty() {
            m.extend(s(3, name));
        }
        m.extend(s(4, op));
        for a in attrs {
            m.extend(ld(5, &a));
        }
        m
    }

    fn value_info(name: &str, elem_type: i64, shape: &[i64]) -> Vec<u8> {
        // ValueInfo { name(1), type(2): TypeProto { tensor_type(1):
        //   Tensor { elem_type(1), shape(2): TensorShapeProto { dim(1) } } } }
        let mut shape_msg = Vec::new();
        for &d in shape {
            let dim = vi(1, d as u64);
            shape_msg.extend(ld(1, &dim));
        }
        let mut tensor = vi(1, elem_type as u64);
        tensor.extend(ld(2, &shape_msg));
        let type_proto = ld(1, &tensor);
        let mut m = s(1, name);
        m.extend(ld(2, &type_proto));
        m
    }

    /// X[1,4] x W[4,4] + B[4] -> Softmax(axis=-1) -> out
    pub(crate) fn fixture_matmul_add_softmax() -> Vec<u8> {
        let mut g = Vec::new();
        g.extend(ld(1, &node("MatMul", "mm", &["X", "W"], &["Y"], vec![])));
        g.extend(ld(1, &node("Add", "add", &["Y", "B"], &["Z"], vec![])));
        g.extend(ld(
            1,
            &node("Softmax", "sm", &["Z"], &["out"], vec![attr_i("axis", -1)]),
        ));
        g.extend(ld(2, "fixture".as_bytes()));
        g.extend(ld(5, &tensor_msg("W", 1, &[4, 4], &[0.25; 16])));
        g.extend(ld(5, &tensor_msg("B", 1, &[4], &[0.5; 4])));
        g.extend(ld(11, &value_info("X", 1, &[1, 4])));
        g.extend(ld(12, &value_info("out", 1, &[1, 4])));
        let mut model = ld(7, &g);
        model.extend(vi(1, 8)); // ir_version
        model
    }

    #[test]
    fn end_to_end_matmul_add_softmax() {
        let bytes = fixture_matmul_add_softmax();
        let g = parse_bytes(&bytes, "fixture").unwrap();
        g.validate().unwrap();

        assert_eq!(g.metadata("source_format"), Some("onnx"));
        assert_eq!(g.metadata("onnx_ir_version"), Some("8"));
        let hist = g.op_histogram();
        assert_eq!(hist.get("input"), Some(&1));
        assert_eq!(hist.get("constant"), Some(&2));
        assert_eq!(hist.get("matmul"), Some(&1));
        assert_eq!(hist.get("add"), Some(&1));
        assert_eq!(hist.get("softmax"), Some(&1));
        assert_eq!(hist.get("output"), Some(&1));

        // Weight values survived the trip.
        let w = g.nodes.iter().find(|n| n.name == "W").unwrap();
        match &w.op {
            Op::Constant { tensor } => {
                let vals = tensor.data.to_f32(&tensor.desc).unwrap();
                assert_eq!(vals[0], 0.25);
                assert_eq!(tensor.desc.shape, vec![4, 4]);
            }
            other => panic!("expected constant, got {other:?}"),
        }

        // JSON roundtrip.
        let json = g.to_json_string().unwrap();
        assert_eq!(Graph::from_json_str(&json).unwrap(), g);
    }

    #[test]
    fn gemm_transb_lowers_to_transpose_matmul_add() {
        let mut g = ld(
            1,
            &node(
                "Gemm",
                "fc",
                &["X", "W", "B"],
                &["Y"],
                vec![
                    attr_i("transB", 1),
                    attr_f("alpha", 2.0),
                    attr_f("beta", 0.5),
                ],
            ),
        );
        g.extend(ld(5, &tensor_msg("W", 1, &[2, 4], &[0.5; 8])));
        g.extend(ld(5, &tensor_msg("B", 1, &[4], &[1.0; 4])));
        g.extend(ld(11, &value_info("X", 1, &[1, 2])));
        g.extend(ld(12, &value_info("Y", 1, &[1, 4])));

        let graph = parse_bytes(&ld(7, &g), "gemm").unwrap();
        graph.validate().unwrap();
        let hist = graph.op_histogram();
        assert_eq!(
            hist.get("transpose"),
            Some(&1),
            "transB becomes a transpose"
        );
        assert_eq!(hist.get("matmul"), Some(&1));
        assert_eq!(hist.get("mul"), Some(&2), "alpha and beta scales");
        assert_eq!(hist.get("add"), Some(&1));
    }

    #[test]
    fn unsupported_op_names_itself() {
        let mut g = ld(1, &node("Loop", "loop", &["X"], &["Y"], vec![]));
        g.extend(ld(11, &value_info("X", 1, &[1, 2])));
        g.extend(ld(12, &value_info("Y", 1, &[1, 2])));

        let err = parse_bytes(&ld(7, &g), "m").unwrap_err();
        assert!(err.to_string().contains("`Loop`"));
    }

    #[test]
    fn dangling_reference_reports_value_name() {
        let mut g = ld(1, &node("MatMul", "mm", &["X", "MISSING"], &["Y"], vec![]));
        g.extend(ld(11, &value_info("X", 1, &[1, 2])));
        g.extend(ld(12, &value_info("Y", 1, &[1, 2])));

        let err = parse_bytes(&ld(7, &g), "m").unwrap_err();
        assert!(err.to_string().contains("MISSING"));
    }

    #[test]
    fn f16_initializer_preserved() {
        let mut t = s(8, "W");
        t.extend(vi(2, 10)); // FLOAT16
        t.extend(packed_i64s(1, &[2]));
        t.extend(ld(9, &[0x00, 0x3c, 0x00, 0x40])); // 1.0 and 2.0 as f16 bits

        let mut g = ld(5, &t);
        g.extend(ld(11, &value_info("X", 10, &[2])));
        g.extend(ld(12, &value_info("X", 10, &[2])));

        let graph = parse_bytes(&ld(7, &g), "f16").unwrap();
        let node = graph.nodes.iter().find(|n| n.name == "W").unwrap();
        match &node.op {
            Op::Constant { tensor } => {
                assert_eq!(tensor.desc.dtype, DType::F16);
                assert_eq!(tensor.data.as_slice(), &[0x00, 0x3c, 0x00, 0x40]);
            }
            _ => panic!("expected constant"),
        }
    }

    #[test]
    fn reshape_reads_shape_initializer() {
        let mut g = ld(1, &node("Reshape", "rs", &["X", "shape"], &["Y"], vec![]));
        let mut shape_t = s(8, "shape");
        shape_t.extend(vi(2, 7)); // INT64
        shape_t.extend(packed_i64s(1, &[2])); // dims: rank 2
        shape_t.extend(packed_i64s(7, &[2, -1])); // int64_data
        g.extend(ld(5, &shape_t));
        g.extend(ld(11, &value_info("X", 1, &[2, 2])));
        g.extend(ld(12, &value_info("Y", 1, &[2, 2])));

        let graph = parse_bytes(&ld(7, &g), "rs").unwrap();
        let rs = graph
            .nodes
            .iter()
            .find(|n| matches!(n.op, Op::Reshape { .. }))
            .unwrap();
        match &rs.op {
            Op::Reshape { attrs } => assert_eq!(attrs.shape, vec![2, -1]),
            other => panic!("expected reshape, got {other:?}"),
        }
    }
}
