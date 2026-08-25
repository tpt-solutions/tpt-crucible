//! TensorFlow SavedModel ingestion (`saved_model.pb`).
//!
//! A native, dependency-free reader built on the shared protobuf wire-format
//! decoder ([`crate::pb`]). A SavedModel is a `SavedModel` proto holding one
//! or more tagged `MetaGraphDef`s, each carrying an encoded `GraphDef` of
//! `NodeDef`s plus optional `SignatureDef` maps naming the served inputs and
//! outputs.
//!
//! Supported ops (v1): `Placeholder` → input, `Const` → constant, `Identity`
//! → alias, `MatMul` / `BatchMatMul(V2/V3)` (transpose attrs lowered to
//! Transpose/MatMul), elementwise binaries and unaries (`Add AddV2 BiasAdd
//! Sub Mul Div Neg Exp Log Sqrt Sin Cos Tanh Erf Relu Sigmoid Gelu`),
//! `Softmax`, `Reshape`, `Transpose`, `ConcatV2`/`Concat`, and `Cast`.
//! Anything else surfaces as [`Error::UnsupportedOperation`] naming the op.
//!
//! Weights must be embedded in the graph as `Const` nodes (as in frozen
//! graphs). TF2-style checkpoint variables (`variables/` read through
//! `VarHandleOp`) are detected and rejected with a structured error.
//!
//! Graph outputs come from `signature_def` when present (preferring the
//! `serving_default` signature); otherwise every dangling producer becomes an
//! output.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use tpt_crucible_common::dtype::DType;
use tpt_crucible_common::error::{Error, Result};
use tpt_crucible_common::ops::{Op, ReshapeAttrs, SoftmaxAttrs, TransposeAttrs};
use tpt_crucible_common::{Graph, NodeId, Tensor, TensorData, TensorDesc};

use crate::pb::{
    for_fields as pb_for_fields, repeated_bool, repeated_f32, repeated_f64, repeated_i64,
    str_field, Rd,
};

const PATH: &str = "<tf-savedmodel>";
const FORMAT: &str = "tf-savedmodel";

fn malformed(reason: &str) -> Error {
    Error::ParseFormat {
        path: PATH.into(),
        format: FORMAT.into(),
        reason: reason.into(),
    }
}

/// Run [`pb_for_fields`] bound to this module's diagnostic context.
fn for_fields<F>(buf: &[u8], on: F) -> Result<()>
where
    F: FnMut(u32, u8, &mut Rd) -> Result<()>,
{
    pb_for_fields(buf, PATH, FORMAT, on)
}

// ---- SavedModel message model ----------------------------------------------

/// Parsed `TensorShapeProto`.
#[derive(Debug, Default, Clone)]
struct Shape {
    dims: Vec<i64>,
    unknown_rank: bool,
}

fn parse_shape(buf: &[u8]) -> Result<Shape> {
    let mut s = Shape::default();
    for_fields(buf, |field, wt, r| {
        match field {
            2 => {
                // Dim { size(1): int64, name(2): string }
                let dim = r.bytes()?;
                let mut size = 0i64;
                for_fields(dim, |f2, w2, r2| {
                    match f2 {
                        1 => size = r2.varint()? as i64,
                        _ => r2.skip(w2)?,
                    }
                    Ok(())
                })?;
                s.dims.push(size);
            }
            3 => s.unknown_rank = r.varint()? != 0,
            _ => r.skip(wt)?,
        }
        Ok(())
    })?;
    Ok(s)
}

/// Parsed `TensorProto` (inline payload variants we consume).
#[derive(Debug, Default, Clone)]
struct TfTensor {
    dtype: i64,
    shape: Shape,
    tensor_content: Vec<u8>,
    half_val: Vec<i64>,
    float_val: Vec<f32>,
    double_val: Vec<f64>,
    int_val: Vec<i64>,
    int64_val: Vec<i64>,
    bool_val: Vec<bool>,
}

fn parse_tensor_proto(buf: &[u8]) -> Result<TfTensor> {
    let mut t = TfTensor::default();
    for_fields(buf, |field, wt, r| {
        match field {
            1 => t.dtype = r.varint()? as i64,
            2 => t.shape = parse_shape(r.bytes()?)?,
            5 => repeated_f32(wt, r, &mut t.float_val)?,
            6 => repeated_f64(wt, r, &mut t.double_val)?,
            7 => repeated_i64(wt, r, &mut t.int_val)?,
            8 => t.tensor_content = r.bytes()?.to_vec(),
            12 => repeated_i64(wt, r, &mut t.int64_val)?,
            13 => repeated_i64(wt, r, &mut t.half_val)?,
            14 => repeated_bool(wt, r, &mut t.bool_val)?,
            _ => r.skip(wt)?,
        }
        Ok(())
    })?;
    Ok(t)
}

/// Parsed `AttrValue` (last `oneof` branch wins, matching protobuf merge
/// semantics for the branches we consume).
#[derive(Debug, Default, Clone)]
struct TfAttr {
    /// Raw `s` (bytes/string) payload; compared, never decoded as UTF-8.
    s: Vec<u8>,
    i: i64,
    f: f32,
    b: bool,
    has_b: bool,
    dtype: i64,
    shape: Option<Shape>,
    /// `list.shape` — where `_output_shapes` lives.
    shapes: Vec<Shape>,
    tensor: Option<TfTensor>,
}

impl TfAttr {
    fn get<'a>(attrs: &'a [(String, TfAttr)], name: &str) -> Option<&'a TfAttr> {
        attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v)
    }

    fn i_or(attrs: &[(String, TfAttr)], name: &str, default: i64) -> i64 {
        Self::get(attrs, name).map(|a| a.i).unwrap_or(default)
    }
}

fn parse_attr_value(buf: &[u8]) -> Result<TfAttr> {
    let mut a = TfAttr::default();
    for_fields(buf, |field, wt, r| {
        match field {
            1 => a.s = r.bytes()?.to_vec(), // AttrValue.s
            2 => a.i = r.varint()? as i64,  // AttrValue.i
            3 => a.f = r.fixed32()?,       // AttrValue.f
            4 => {
                a.b = r.varint()? != 0; // AttrValue.b
                a.has_b = true;
            }
            5 => a.dtype = r.varint()? as i64,                     // AttrValue.type
            6 => a.shape = Some(parse_shape(r.bytes()?)?),         // AttrValue.shape
            7 => a.tensor = Some(parse_tensor_proto(r.bytes()?)?), // AttrValue.tensor
            10 => {
                // ListValue { shape(12), ... } — only shapes matter today.
                let list = r.bytes()?;
                for_fields(list, |f2, w2, r2| {
                    match f2 {
                        12 => a.shapes.push(parse_shape(r2.bytes()?)?),
                        _ => r2.skip(w2)?,
                    }
                    Ok(())
                })?;
            }
            _ => r.skip(wt)?,
        }
        Ok(())
    })?;
    Ok(a)
}

/// Decode one protobuf `map<K, V>` entry message body into `(key, value)`.
fn parse_map_entry(buf: &[u8]) -> Result<(String, Vec<u8>)> {
    let mut key = String::new();
    let mut value = Vec::new();
    for_fields(buf, |field, wt, r| {
        match field {
            1 => key = str_field(r)?,
            2 => value = r.bytes()?.to_vec(),
            _ => r.skip(wt)?,
        }
        Ok(())
    })?;
    Ok((key, value))
}

/// Walk a repeated protobuf `map` field (entries appear back-to-back at the
/// field's number) into `(key, value)` pairs.
fn parse_map_entries(buf: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
    let mut out = Vec::new();
    let mut r = Rd::new(buf, PATH, FORMAT);
    while !r.eof() {
        let key = r.varint()?;
        let field = (key >> 3) as u32;
        let wt = (key & 7) as u8;
        if wt != 2 || (field != 1 && field != 2) {
            r.skip(wt)?;
            continue;
        }
        out.push(parse_map_entry(r.bytes()?)?);
    }
    Ok(out)
}

/// Parsed `NodeDef`. TF `NodeDef`s do **not** list their outputs; every op we
/// support produces exactly one value named after the node.
#[derive(Debug, Default, Clone)]
struct TfNode {
    name: String,
    op: String,
    /// Data inputs, control deps stripped and `:port` suffixes removed.
    inputs: Vec<String>,
    attrs: Vec<(String, TfAttr)>,
}

/// Normalize a `NodeDef.input` string: `^ctrl` deps drop out (`None`), and a
/// trailing numeric `:port` is removed (TF names cannot contain `:`).
fn clean_input(raw: &str) -> Option<String> {
    let s = raw.strip_prefix('^')?;
    let head = match s.rsplit_once(':') {
        Some((h, tail)) if !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) => h,
        _ => s,
    };
    Some(head.to_string())
}

fn parse_node(buf: &[u8]) -> Result<TfNode> {
    let mut n = TfNode::default();
    for_fields(buf, |field, wt, r| {
        match field {
            1 => n.name = str_field(r)?,
            2 => n.op = str_field(r)?,
            3 => {
                if let Some(v) = clean_input(&str_field(r)?) {
                    n.inputs.push(v);
                }
            }
            5 => {
                let (k, v) = parse_map_entry(r.bytes()?)?;
                let attr = parse_attr_value(&v)?;
                n.attrs.push((k, attr));
            }
            _ => r.skip(wt)?,
        }
        Ok(())
    })?;
    Ok(n)
}

// ---- signatures -------------------------------------------------------------

/// One side of a `SignatureDef` map.
#[derive(Debug)]
struct SignatureIo {
    /// The signature-facing name (e.g. `dense` in `outputs`).
    key: String,
    /// The graph tensor feeding it (e.g. `sm` / `sm:0`).
    tensor_name: String,
}

/// Parsed `SignatureDef` with its map key (e.g. `serving_default`).
#[derive(Debug)]
struct Signature {
    name: String,
    #[allow(dead_code)]
    inputs: Vec<SignatureIo>,
    outputs: Vec<SignatureIo>,
}

fn parse_tensor_info_map(buf: &[u8]) -> Result<Vec<SignatureIo>> {
    let mut out = Vec::new();
    for (key, val) in parse_map_entries(buf)? {
        // TensorInfo { name(1): string, dtype(2), tensor_shape(3), ... }
        let mut tensor_name = String::new();
        for_fields(&val, |f2, w2, r2| {
            match f2 {
                1 => tensor_name = str_field(r2)?,
                _ => r2.skip(w2)?,
            }
            Ok(())
        })?;
        out.push(SignatureIo { key, tensor_name });
    }
    Ok(out)
}

fn parse_signature(name: String, buf: &[u8]) -> Result<Signature> {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for_fields(buf, |field, wt, r| {
        match field {
            1 => inputs.extend(parse_tensor_info_map(r.bytes()?)?), // SignatureDef.inputs
            2 => outputs.extend(parse_tensor_info_map(r.bytes()?)?), // SignatureDef.outputs
            _ => r.skip(wt)?,
        }
        Ok(())
    })?;
    Ok(Signature {
        name,
        inputs,
        outputs,
    })
}

// ---- meta-graph / saved-model -----------------------------------------------

/// Parsed `MetaGraphDef`.
#[derive(Debug, Default)]
struct MetaGraph {
    tags: Vec<String>,
    tf_version: String,
    nodes: Vec<TfNode>,
    signatures: Vec<Signature>,
}

fn parse_meta_graph(buf: &[u8]) -> Result<MetaGraph> {
    let mut mg = MetaGraph::default();
    for_fields(buf, |field, wt, r| {
        match field {
            1 => {
                // MetaInfoDef { tags(4), tensorflow_version(5) }
                for_fields(r.bytes()?, |f2, w2, r2| {
                    match f2 {
                        4 => mg.tags.push(str_field(r2)?),
                        5 => mg.tf_version = str_field(r2)?,
                        _ => r2.skip(w2)?,
                    }
                    Ok(())
                })?;
            }
            2 => {
                // GraphDef { node(1): NodeDef }
                for_fields(r.bytes()?, |f2, w2, r2| {
                    match f2 {
                        1 => mg.nodes.push(parse_node(r2.bytes()?)?),
                        _ => r2.skip(w2)?,
                    }
                    Ok(())
                })?;
            }
            3 => {
                // signature_def: map<string, SignatureDef>
                for (sig_name, sig_buf) in parse_map_entries(r.bytes()?)? {
                    mg.signatures
                        .push(parse_signature(sig_name, &sig_buf)?);
                }
            }
            _ => r.skip(wt)?,
        }
        Ok(())
    })?;
    Ok(mg)
}

fn parse_saved_model(buf: &[u8]) -> Result<Vec<MetaGraph>> {
    let mut metas = Vec::new();
    for_fields(buf, |field, wt, r| {
        match field {
            1 => metas.push(parse_meta_graph(r.bytes()?)?), // meta_graphs
            _ => r.skip(wt)?,
        }
        Ok(())
    })?;
    Ok(metas)
}

// ---- conversion to TPT-IR ---------------------------------------------------

/// Map a TF `DataType` enum code onto a TPT-IR dtype.
fn tf_dtype(code: i64) -> Result<DType> {
    Ok(match code {
        1 => DType::F32,
        2 => DType::F64,
        3 => DType::I32,
        4 => DType::U8,
        5 => DType::I16,
        6 => DType::I8,
        9 => DType::I64,
        10 => DType::Bool,
        14 => DType::Bf16,
        17 => DType::U16,
        19 => DType::F16,
        22 => DType::U32,
        23 => DType::U64,
        other => return Err(Error::UnsupportedDType(format!("tf dtype {other}"))),
    })
}

/// Materialize a `Const` `TensorProto` into an IR [`Tensor`].
///
/// `tensor_content` (raw LE bytes) is preferred; typed `*_val` arrays are
/// assembled per element otherwise.
fn tensor_from_tf(t: &TfTensor, name: &str) -> Result<Tensor> {
    let desc = TensorDesc::new(
        t.shape
            .dims
            .iter()
            .map(|&d| d.max(0) as usize)
            .collect::<Vec<_>>(),
        tf_dtype(t.dtype)?,
    );
    let expected = desc.byte_size();

    let data = if !t.tensor_content.is_empty() {
        if t.tensor_content.len() != expected {
            return Err(Error::TensorSizeMismatch {
                name: name.to_string(),
                expected,
                actual: t.tensor_content.len(),
            });
        }
        t.tensor_content.clone()
    } else if expected == 0 {
        Vec::new()
    } else {
        let mut b = Vec::with_capacity(expected);
        match desc.dtype {
            DType::F32 if !t.float_val.is_empty() => {
                for v in &t.float_val {
                    b.extend_from_slice(&v.to_le_bytes());
                }
            }
            DType::F64 if !t.double_val.is_empty() => {
                for v in &t.double_val {
                    b.extend_from_slice(&v.to_le_bytes());
                }
            }
            DType::I64 if !t.int64_val.is_empty() => {
                for v in &t.int64_val {
                    b.extend_from_slice(&v.to_le_bytes());
                }
            }
            DType::I32 if !t.int_val.is_empty() => {
                for v in &t.int_val {
                    b.extend_from_slice(&(*v as i32).to_le_bytes());
                }
            }
            DType::F16 if !t.half_val.is_empty() => {
                // half_val holds the raw IEEE-754 binary16 bit patterns.
                for v in &t.half_val {
                    b.extend_from_slice(&(*v as u16).to_le_bytes());
                }
            }
            DType::Bool if !t.bool_val.is_empty() => {
                for v in &t.bool_val {
                    b.push(u8::from(*v));
                }
            }
            _ => {
                return Err(Error::ParseFormat {
                    path: PATH.into(),
                    format: FORMAT.into(),
                    reason: format!("constant `{name}` carries no supported inline data"),
                });
            }
        }
        if b.len() != expected {
            return Err(Error::TensorSizeMismatch {
                name: name.to_string(),
                expected,
                actual: b.len(),
            });
        }
        b
    };

    Ok(Tensor {
        desc,
        data: TensorData::from_vec(data),
    })
}

/// Read integer values out of a constant node (used by `Reshape`/`Transpose`
/// shape and permutation inputs).
fn const_i64s(g: &Graph, id: NodeId) -> Result<Vec<i64>> {
    let node = g.get_node(id).ok_or_else(|| malformed("missing node"))?;
    let tensor = match &node.op {
        Op::Constant { tensor } => tensor,
        _ => {
            return Err(malformed(
                "expected a constant initializer for this input",
            ));
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

/// Build the static [`TensorDesc`] a `Placeholder` promises.
fn placeholder_desc(node: &TfNode) -> Result<TensorDesc> {
    let dtype = TfAttr::get(&node.attrs, "dtype")
        .map(|a| a.dtype)
        .ok_or_else(|| malformed(&format!("placeholder `{}` lacks a dtype attribute", node.name)))?;
    let shape = TfAttr::get(&node.attrs, "shape")
        .and_then(|a| a.shape.clone())
        .or_else(|| {
            TfAttr::get(&node.attrs, "_output_shapes")
                .and_then(|a| a.shapes.first().cloned())
        });
    let Some(shape) = shape else {
        return Err(malformed(&format!(
            "input `{}` lacks a static shape",
            node.name
        )));
    };
    if shape.unknown_rank || shape.dims.is_empty() {
        return Err(malformed(&format!(
            "input `{}` lacks a static shape",
            node.name
        )));
    }
    Ok(TensorDesc::new(
        shape
            .dims
            .iter()
            .map(|&d| d.max(0) as usize)
            .collect::<Vec<_>>(),
        tf_dtype(dtype)?,
    ))
}

/// Incremental graph under construction.
struct Builder {
    g: Graph,
    /// Value (tensor) name → producing IR node.
    vals: HashMap<String, NodeId>,
    /// Value names in first-producer order (drives the dangling-output
    /// fallback).
    order: Vec<String>,
    used_names: HashSet<String>,
}

impl Builder {
    fn push(&mut self, name: &str, op: Op, inputs: Vec<NodeId>) -> NodeId {
        // TF node names may repeat across subgraphs; only honor them while
        // unique.
        let unique = !name.is_empty() && self.used_names.insert(name.to_string());
        self.g.push(if unique { name } else { "" }, op, inputs)
    }

    fn record(&mut self, value_name: &str, id: NodeId) -> Result<()> {
        if value_name.is_empty() {
            return Ok(());
        }
        if self.vals.contains_key(value_name) {
            return Err(malformed(&format!("duplicate value name `{value_name}`")));
        }
        self.vals.insert(value_name.to_string(), id);
        self.order.push(value_name.to_string());
        Ok(())
    }
}

/// Lower one TF node. Returns the produced IR node when it creates one;
/// `None` for pure aliases (`Identity` et al.), which only re-register their
/// input value under a new name.
fn build_node(b: &mut Builder, node: &TfNode) -> Result<Option<NodeId>> {
    match node.op.as_str() {
        "Const" => {
            let raw = TfAttr::get(&node.attrs, "value")
                .and_then(|a| a.tensor.as_ref())
                .ok_or_else(|| malformed("Const without a `value` attribute"))?;
            let tensor = tensor_from_tf(raw, &node.name)?;
            let id = b.push(&node.name, Op::Constant { tensor }, Vec::new());
            b.record(&node.name, id)?;
            Ok(Some(id))
        }
        "Placeholder" | "PlaceholderV2" | "PlaceholderWithDefault" => {
            let desc = placeholder_desc(node)?;
            let id = b.push(&node.name, Op::Input(desc), Vec::new());
            b.g.mark_input(id);
            b.record(&node.name, id)?;
            Ok(Some(id))
        }
        "Identity" | "StopGradient" | "Snapshot" => {
            let ins = resolve(b, node)?;
            b.record(&node.name, ins[0])?;
            Ok(None)
        }
        op @ ("VariableV2" | "VarHandleOp" | "ReadVariableOp" | "ResourceGather"
        | "ResourceGatherNd" | "MutableHashTableOfTensors" | "MutableHashTableV2") => {
            Err(Error::UnsupportedOperation(format!(
                "tf op `{op}` reads weights from a `variables/` checkpoint; \
                 embed weights as Const nodes (frozen graph) — checkpoint \
                 loading is not supported yet"
            )))
        }
        "MatMul" => {
            let ins = resolve(b, node)?;
            let mut a = ins[0];
            let mut bt = ins[1];
            if TfAttr::i_or(&node.attrs, "transpose_a", 0) != 0 {
                a = b.push(
                    "",
                    Op::Transpose {
                        attrs: TransposeAttrs { perm: vec![1, 0] },
                    },
                    vec![a],
                );
            }
            if TfAttr::i_or(&node.attrs, "transpose_b", 0) != 0 {
                bt = b.push(
                    "",
                    Op::Transpose {
                        attrs: TransposeAttrs { perm: vec![1, 0] },
                    },
                    vec![bt],
                );
            }
            let id = b.push(&node.name, Op::MatMul, vec![a, bt]);
            b.record(&node.name, id)?;
            Ok(Some(id))
        }
        "BatchMatMul" | "BatchMatMulV2" | "BatchMatMulV3" => {
            if TfAttr::i_or(&node.attrs, "adj_x", 0) != 0
                || TfAttr::i_or(&node.attrs, "adj_y", 0) != 0
            {
                return Err(Error::UnsupportedOperation(
                    "batched adjoint (adj_x/adj_y) requires static rank inference".into(),
                ));
            }
            let ins = resolve(b, node)?;
            let id = b.push(&node.name, Op::MatMul, ins);
            b.record(&node.name, id)?;
            Ok(Some(id))
        }
        other => {
            let ins = resolve(b, node)?;
            let id = build_simple_op(b, node, other, &node.attrs, ins)?;
            b.record(&node.name, id)?;
            Ok(Some(id))
        }
    }
}

/// Resolve a node's data inputs against values recorded so far.
fn resolve(b: &Builder, node: &TfNode) -> Result<Vec<NodeId>> {
    node.inputs
        .iter()
        .map(|name| {
            b.vals
                .get(name)
                .copied()
                .ok_or_else(|| malformed("unresolved input"))
        })
        .collect()
}

/// Ops whose TPT-IR mapping is a direct rename or a fixed small expansion.
fn build_simple_op(
    b: &mut Builder,
    node: &TfNode,
    op: &str,
    attrs: &[(String, TfAttr)],
    mut ins: Vec<NodeId>,
) -> Result<NodeId> {
    match op {
        "Add" | "AddV2" | "BiasAdd" | "BiasAddV1" => {
            if op.starts_with("BiasAdd") {
                // NCHW biasing broadcasts along axis 1 rather than the last
                // axis; plain Add would silently change semantics. NHWC (the
                // default) matches Add's broadcasting exactly.
                match TfAttr::get(attrs, "data_format") {
                    None => {}
                    Some(f) if f.s == b"NHWC" => {}
                    Some(_) => {
                        return Err(Error::UnsupportedOperation(
                            "BiasAdd with non-default data_format".into(),
                        ));
                    }
                }
            }
            Ok(b.push(&node.name, Op::Add, ins))
        }
        "Sub" => Ok(b.push(&node.name, Op::Sub, ins)),
        "Mul" | "MulNoNan" => Ok(b.push(&node.name, Op::Mul, ins)),
        "Div" | "RealDiv" | "DivNoNan" => Ok(b.push(&node.name, Op::Div, ins)),
        "Softmax" => {
            let sm_attrs = SoftmaxAttrs {
                axis: TfAttr::i_or(attrs, "axis", -1) as i32,
            };
            Ok(b.push(&node.name, Op::Softmax { attrs: sm_attrs }, ins))
        }
        "Reshape" => {
            let shape = const_i64s(&b.g, ins[1])?;
            ins.truncate(1);
            Ok(b.push(
                &node.name,
                Op::Reshape {
                    attrs: ReshapeAttrs { shape },
                },
                ins,
            ))
        }
        "Transpose" => {
            let perm = const_i64s(&b.g, ins[1])?
                .into_iter()
                .map(|p| p.max(0) as usize)
                .collect();
            ins.truncate(1);
            Ok(b.push(
                &node.name,
                Op::Transpose {
                    attrs: TransposeAttrs { perm },
                },
                ins,
            ))
        }
        "ConcatV2" => {
            // TF puts the concat axis in the LAST input.
            let axis_id = *ins.last().ok_or_else(|| malformed("ConcatV2 with no inputs"))?;
            let axis = const_i64s(&b.g, axis_id)?;
            concat(b, node, axis.first().copied(), {
                ins.truncate(ins.len() - 1);
                ins
            })
        }
        "Concat" => {
            // Legacy v1 op: axis is the FIRST input, values follow.
            let axis = const_i64s(&b.g, ins[0])?;
            ins.remove(0);
            concat(b, node, axis.first().copied(), ins)
        }
        "Cast" => {
            let to = tf_dtype(
                TfAttr::get(attrs, "DstT")
                    .map(|a| a.dtype)
                    .ok_or_else(|| malformed("Cast without a DstT attribute"))?,
            )?;
            Ok(b.push(&node.name, Op::Cast { to }, ins))
        }
        unary @ ("Neg" | "Exp" | "Log" | "Sqrt" | "SqrtV2" | "Sin" | "Cos" | "Tanh" | "Erf"
        | "Relu" | "Sigmoid" | "Gelu") => {
            let op = match unary {
                "Neg" => Op::Neg,
                "Exp" => Op::Exp,
                "Log" => Op::Log,
                "Sqrt" | "SqrtV2" => Op::Sqrt,
                "Sin" => Op::Sin,
                "Cos" => Op::Cos,
                "Tanh" => Op::Tanh,
                "Erf" => Op::Erf,
                "Relu" => Op::Relu,
                "Sigmoid" => Op::Sigmoid,
                _ => Op::Gelu,
            };
            Ok(b.push(&node.name, op, ins))
        }
        other => Err(Error::UnsupportedOperation(format!("tf op `{other}`"))),
    }
}

/// Shared tail of the two Concat variants (negative axes need rank
/// inference, which lowering doesn't do).
fn concat(b: &mut Builder, node: &TfNode, axis: Option<i64>, ins: Vec<NodeId>) -> Result<NodeId> {
    let Some(axis) = axis else {
        return Err(malformed("concat axis constant carries no values"));
    };
    if axis < 0 {
        return Err(Error::UnsupportedOperation(
            "negative concat axis requires rank inference (pending)".into(),
        ));
    }
    Ok(b.push(&node.name, Op::Concat { axis: axis as usize }, ins))
}

/// Parse a binary `saved_model.pb` into TPT-IR.
pub fn parse_bytes(bytes: &[u8], source_name: &str) -> Result<Graph> {
    let metas = parse_saved_model(bytes)?;
    if metas.is_empty() {
        return Err(malformed("SavedModel contains no meta_graphs"));
    }
    // Prefer the serving meta-graph; otherwise take the first one.
    let mg = metas
        .iter()
        .find(|m| m.tags.iter().any(|t| t == "serve"))
        .unwrap_or(&metas[0]);

    let mut b = Builder {
        g: Graph::new(source_name),
        vals: HashMap::new(),
        order: Vec::new(),
        used_names: HashSet::new(),
    };
    b.g.set_metadata("source_format", FORMAT);
    if !mg.tags.is_empty() {
        b.g.set_metadata("tf_meta_graph_tags", mg.tags.join(","));
    }
    if !mg.tf_version.is_empty() {
        b.g.set_metadata("tf_tensorflow_version", &mg.tf_version);
    }

    // Resolve nodes; serialized GraphDefs are usually topologically ordered
    // but sweep leniently when they are not.
    let mut queue: std::collections::VecDeque<TfNode> = mg.nodes.iter().cloned().collect();
    while !queue.is_empty() {
        let mut deferred = std::collections::VecDeque::new();
        let mut progressed = false;
        while let Some(n) = queue.pop_front() {
            let ready = n.inputs.iter().all(|i| b.vals.contains_key(i));
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
                .filter(|i| !b.vals.contains_key(i))
                .collect();
            return Err(malformed(&format!(
                "unresolved value names: {}",
                unresolved.join(", ")
            )));
        }
    }

    // Graph outputs: signature_def entries when present (prefer
    // `serving_default`), else every dangling producer.
    let sig = mg
        .signatures
        .iter()
        .find(|s| s.name == "serving_default")
        .or_else(|| mg.signatures.first());
    let out_names: Vec<String> = match sig {
        Some(s) => s.outputs.iter().map(|o| o.tensor_name.clone()).collect(),
        None => {
            let consumed: HashSet<&str> = mg
                .nodes
                .iter()
                .flat_map(|n| n.inputs.iter())
                .map(String::as_str)
                .collect();
            b.order
                .iter()
                .filter(|name| !consumed.contains(name.as_str()))
                .cloned()
                .collect()
        }
    };
    if out_names.is_empty() {
        return Err(malformed(
            "no graph outputs could be determined (no signature_def and no dangling producers)",
        ));
    }

    for name in out_names {
        let value = clean_input(&name).ok_or_else(|| malformed("control edge named as output"))?;
        let inner = *b
            .vals
            .get(&value)
            .ok_or_else(|| malformed(&format!("graph output `{value}` never produced")))?;
        let out = b.push("", Op::Output { name: value }, vec![inner]);
        b.g.mark_output(out);
    }

    b.g.validate()?;
    Ok(b.g)
}

/// Ingest a SavedModel directory (containing `saved_model.pb`) or a bare
/// `.pb` GraphDef/SavedModel file.
///
/// Text-format graphs (`saved_model.pbtxt`) are rejected with guidance.
pub fn ingest(path: &Path) -> Result<Graph> {
    let pb_path = if path.is_dir() {
        let p = path.join("saved_model.pb");
        if !p.exists() {
            return Err(Error::ParseFormat {
                path: path.display().to_string(),
                format: FORMAT.into(),
                reason: if path.join("saved_model.pbtxt").exists() {
                    "text-format GraphDef (`saved_model.pbtxt`) is not supported; \
                     re-export the model as a binary `saved_model.pb`"
                        .into()
                } else {
                    "`saved_model.pb` not found in directory".into()
                },
            });
        }
        p
    } else {
        path.to_path_buf()
    };
    let bytes = std::fs::read(pb_path)?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    parse_bytes(&bytes, &name)
}