//! TensorFlow SavedModel ingestion (`saved_model.pb`).
//!
//! A native, dependency-free reader built on the shared protobuf wire-format
//! decoder (`crate::pb`). A SavedModel is a `SavedModel` proto holding one
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

/// Parsed `TensorProto` (inline payload variants we consume). Field numbers
/// follow `tensorflow/core/framework/tensor.proto`: `dtype(1)`,
/// `tensor_shape(2)`, `tensor_content(4)`, `float_val(5)`, `double_val(6)`,
/// `int_val(7)`, `int64_val(10)`, `bool_val(11)`, `half_val(13)`.
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
            4 => t.tensor_content = r.bytes()?.to_vec(),
            5 => repeated_f32(wt, r, &mut t.float_val)?,
            6 => repeated_f64(wt, r, &mut t.double_val)?,
            7 => repeated_i64(wt, r, &mut t.int_val)?,
            10 => repeated_i64(wt, r, &mut t.int64_val)?,
            11 => repeated_bool(wt, r, &mut t.bool_val)?,
            13 => repeated_i64(wt, r, &mut t.half_val)?,
            _ => r.skip(wt)?,
        }
        Ok(())
    })?;
    Ok(t)
}

/// Parsed `AttrValue` (last `oneof` branch wins, matching protobuf merge
/// semantics for the branches we consume). Field numbers follow
/// `tensorflow/core/framework/attr_value.proto`: the `oneof value` branches
/// are `list(1)`, `s(2)`, `i(3)`, `f(4)`, `b(5)`, `type(6)`, `shape(7)`,
/// `tensor(8)`; `func(10)`/`placeholder(9)` are skipped.
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

    fn b_or(attrs: &[(String, TfAttr)], name: &str, default: bool) -> bool {
        Self::get(attrs, name).map(|a| a.b).unwrap_or(default)
    }
}

fn parse_attr_value(buf: &[u8]) -> Result<TfAttr> {
    let mut a = TfAttr::default();
    for_fields(buf, |field, wt, r| {
        match field {
            1 => {
                // ListValue { shape(6), ... } — only shapes matter today
                // (`_output_shapes` bookkeeping).
                let list = r.bytes()?;
                for_fields(list, |f2, w2, r2| {
                    match f2 {
                        6 => a.shapes.push(parse_shape(r2.bytes()?)?),
                        _ => r2.skip(w2)?,
                    }
                    Ok(())
                })?;
            }
            2 => a.s = r.bytes()?.to_vec(), // AttrValue.s
            3 => a.i = r.varint()? as i64,  // AttrValue.i
            4 => a.f = r.fixed32()?,        // AttrValue.f
            5 => {
                a.b = r.varint()? != 0; // AttrValue.b
                a.has_b = true;
            }
            6 => a.dtype = r.varint()? as i64, // AttrValue.type
            7 => a.shape = Some(parse_shape(r.bytes()?)?), // AttrValue.shape
            8 => a.tensor = Some(parse_tensor_proto(r.bytes()?)?), // AttrValue.tensor
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
    // Control dependencies carry no data.
    let s = match raw.strip_prefix('^') {
        Some(_) => return None,
        None => raw,
    };
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

/// Parsed one `TensorInfo` map entry: `key -> { name(1) }`.
fn parse_tensor_info_entry(buf: &[u8]) -> Result<SignatureIo> {
    let (key, val) = parse_map_entry(buf)?;
    // TensorInfo { name(1): string, dtype(2), tensor_shape(3), ... }
    let mut tensor_name = String::new();
    for_fields(&val, |f2, w2, r2| {
        match f2 {
            1 => tensor_name = str_field(r2)?,
            _ => r2.skip(w2)?,
        }
        Ok(())
    })?;
    Ok(SignatureIo { key, tensor_name })
}

fn parse_signature(name: String, buf: &[u8]) -> Result<Signature> {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for_fields(buf, |field, wt, r| {
        match field {
            // SignatureDef.inputs/outputs are `map<string, TensorInfo>`; each
            // entry arrives as its own length-delimited field occurrence.
            1 => inputs.push(parse_tensor_info_entry(r.bytes()?)?),
            2 => outputs.push(parse_tensor_info_entry(r.bytes()?)?),
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
                // signature_def: map<string, SignatureDef>; each entry is its
                // own length-delimited occurrence of field 3.
                let (sig_name, sig_buf) = parse_map_entry(r.bytes()?)?;
                mg.signatures.push(parse_signature(sig_name, &sig_buf)?);
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
            return Err(malformed("expected a constant initializer for this input"));
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
        .ok_or_else(|| {
            malformed(&format!(
                "placeholder `{}` lacks a dtype attribute",
                node.name
            ))
        })?;
    let shape = TfAttr::get(&node.attrs, "shape")
        .and_then(|a| a.shape.clone())
        .or_else(|| {
            TfAttr::get(&node.attrs, "_output_shapes").and_then(|a| a.shapes.first().cloned())
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
        op @ ("VariableV2"
        | "VarHandleOp"
        | "ReadVariableOp"
        | "ResourceGather"
        | "ResourceGatherNd"
        | "MutableHashTableOfTensors"
        | "MutableHashTableV2") => Err(Error::UnsupportedOperation(format!(
            "tf op `{op}` reads weights from a `variables/` checkpoint; \
                 embed weights as Const nodes (frozen graph) — checkpoint \
                 loading is not supported yet"
        ))),
        "MatMul" => {
            let ins = resolve(b, node)?;
            let mut a = ins[0];
            let mut bt = ins[1];
            if TfAttr::b_or(&node.attrs, "transpose_a", false) {
                a = b.push(
                    "",
                    Op::Transpose {
                        attrs: TransposeAttrs { perm: vec![1, 0] },
                    },
                    vec![a],
                );
            }
            if TfAttr::b_or(&node.attrs, "transpose_b", false) {
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
            if TfAttr::b_or(&node.attrs, "adj_x", false)
                || TfAttr::b_or(&node.attrs, "adj_y", false)
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
            let axis_id = *ins
                .last()
                .ok_or_else(|| malformed("ConcatV2 with no inputs"))?;
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
    Ok(b.push(
        &node.name,
        Op::Concat {
            axis: axis as usize,
        },
        ins,
    ))
}

/// Every TF op this lowering understands, checked up front so unsupported
/// graphs fail fast naming the op instead of stalling mid-schedule with an
/// opaque "unresolved value names" error. Variable/checkpoint ops are listed
/// too — they are *known* but rejected later with structured guidance.
const KNOWN_OPS: &[&str] = &[
    "Const",
    "Placeholder",
    "PlaceholderV2",
    "PlaceholderWithDefault",
    "Identity",
    "StopGradient",
    "Snapshot",
    "VariableV2",
    "VarHandleOp",
    "ReadVariableOp",
    "ResourceGather",
    "ResourceGatherNd",
    "MutableHashTableOfTensors",
    "MutableHashTableV2",
    "MatMul",
    "BatchMatMul",
    "BatchMatMulV2",
    "BatchMatMulV3",
    "Add",
    "AddV2",
    "BiasAdd",
    "BiasAddV1",
    "Sub",
    "Mul",
    "MulNoNan",
    "Div",
    "RealDiv",
    "DivNoNan",
    "Softmax",
    "Reshape",
    "Transpose",
    "ConcatV2",
    "Concat",
    "Cast",
    "Neg",
    "Exp",
    "Log",
    "Sqrt",
    "SqrtV2",
    "Sin",
    "Cos",
    "Tanh",
    "Erf",
    "Relu",
    "Sigmoid",
    "Gelu",
];

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

    // Fail fast on ops we cannot lower, before topology resolution muddies
    // the diagnostic.
    if let Some(n) = mg
        .nodes
        .iter()
        .find(|n| !KNOWN_OPS.contains(&n.op.as_str()))
    {
        return Err(Error::UnsupportedOperation(format!("tf op `{}`", n.op)));
    }

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
    // `serving_default`), else every dangling producer. Signature outputs are
    // named by their signature key (the externally visible name).
    let sig = mg
        .signatures
        .iter()
        .find(|s| s.name == "serving_default")
        .or_else(|| mg.signatures.first());
    let outs: Vec<(String, String)> = match sig {
        Some(s) => s
            .outputs
            .iter()
            .map(|o| (o.key.clone(), o.tensor_name.clone()))
            .collect(),
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
                .map(|name| (name.clone(), name.clone()))
                .collect()
        }
    };
    if outs.is_empty() {
        return Err(malformed(
            "no graph outputs could be determined (no signature_def and no dangling producers)",
        ));
    }

    for (out_name, tensor_name) in outs {
        let value =
            clean_input(&tensor_name).ok_or_else(|| malformed("control edge named as output"))?;
        let inner = *b
            .vals
            .get(&value)
            .ok_or_else(|| malformed(&format!("graph output `{value}` never produced")))?;
        let out = b.push("", Op::Output { name: out_name }, vec![inner]);
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

    /// `TensorShapeProto` with concrete dims.
    fn shape_msg(dims: &[i64]) -> Vec<u8> {
        let mut m = Vec::new();
        for &d in dims {
            m.extend(ld(2, &vi(1, d as u64)));
        }
        m
    }

    /// `TensorProto` carrying raw little-endian payload (`tensor_content(4)`).
    fn tensor_proto(dtype: i64, dims: &[i64], raw: &[u8]) -> Vec<u8> {
        let mut m = vi(1, dtype as u64);
        m.extend(ld(2, &shape_msg(dims)));
        m.extend(ld(4, raw));
        m
    }

    fn f32_le(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn i64_le(vals: &[i64]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// One `NodeDef.attr` map entry binding `key` to an encoded `AttrValue`.
    fn attr(key: &str, value: &[u8]) -> Vec<u8> {
        let mut e = s(1, key);
        e.extend(ld(2, value));
        e
    }

    /// A `Const.value` attribute: `AttrValue { tensor(8) }`.
    fn value_attr(dtype: i64, dims: &[i64], raw: &[u8]) -> Vec<u8> {
        attr("value", &ld(8, &tensor_proto(dtype, dims, raw)))
    }

    fn node(name: &str, op: &str, ins: &[&str], attrs: Vec<Vec<u8>>) -> Vec<u8> {
        let mut m = s(1, name);
        m.extend(s(2, op));
        for i in ins {
            m.extend(s(3, i));
        }
        for a in attrs {
            m.extend(ld(5, &a));
        }
        m
    }

    fn graph_def(nodes: Vec<Vec<u8>>) -> Vec<u8> {
        let mut m = Vec::new();
        for n in nodes {
            m.extend(ld(1, &n));
        }
        m
    }

    /// `MetaInfoDef` with tags + TensorFlow version.
    fn meta_info(tags: &[&str], version: &str) -> Vec<u8> {
        let mut m = Vec::new();
        for t in tags {
            m.extend(s(4, t));
        }
        m.extend(s(5, version));
        m
    }

    /// One `TensorInfo` map entry (`{key: name}`).
    fn ti_entry(key: &str, tensor: &str) -> Vec<u8> {
        let mut e = s(1, key);
        e.extend(ld(2, &s(1, tensor)));
        e
    }

    /// `SignatureDef` body from its inputs/outputs map entries.
    fn sig_body(inputs: &[Vec<u8>], outputs: &[Vec<u8>]) -> Vec<u8> {
        let mut m = Vec::new();
        for i in inputs {
            m.extend(ld(1, i));
        }
        for o in outputs {
            m.extend(ld(2, o));
        }
        m
    }

    /// `SavedModel { meta_graphs(1) }` over one `MetaGraphDef`.
    fn saved_model(meta: Option<Vec<u8>>, graph: Vec<u8>, sigs: Option<Vec<u8>>) -> Vec<u8> {
        let mut mg = Vec::new();
        if let Some(info) = meta {
            mg.extend(ld(1, &info));
        }
        mg.extend(ld(2, &graph));
        if let Some(sv) = sigs {
            mg.extend(ld(3, &sv));
        }
        ld(1, &mg)
    }

    /// A dense layer served under `serving_default`: x → MatMul(w) → BiasAdd → Softmax.
    ///
    /// Inputs exercise port suffixes (`w:0`) and control deps (`^x`) on edges.
    pub(crate) fn fixture_serving_minimal() -> Vec<u8> {
        let dt_f32 = vi(6, 1); // AttrValue.type = DT_FLOAT
        let x = node(
            "x",
            "Placeholder",
            &[],
            vec![
                attr("dtype", &dt_f32),
                attr("_output_shapes", &ld(1, &ld(6, &shape_msg(&[1, 4])))),
            ],
        );
        let w_bytes = f32_le(&(0..16).map(|i| i as f32).collect::<Vec<_>>());
        let w = node("w", "Const", &[], vec![value_attr(1, &[4, 4], &w_bytes)]);
        let bias = node(
            "b",
            "Const",
            &[],
            vec![value_attr(1, &[4], &f32_le(&[0.5; 4]))],
        );
        let mm = node("mm", "MatMul", &["x", "w:0"], vec![]);
        let bi = node("bi", "BiasAdd", &["mm", "b"], vec![]);
        let sm = node(
            "sm",
            "Softmax",
            &["^x", "bi"],
            vec![attr("axis", &vi(3, (-1i64) as u64))],
        );

        let graph = graph_def(vec![x, w, bias, mm, bi, sm]);
        let info = meta_info(&["serve"], "2.16.1");
        let sigs = {
            let mut e = s(1, "serving_default");
            e.extend(ld(
                2,
                &sig_body(&[ti_entry("x", "x")], &[ti_entry("dense", "sm")]),
            ));
            e
        };
        saved_model(Some(info), graph, Some(sigs))
    }

    fn out_name(g: &Graph) -> String {
        let n = g
            .nodes
            .iter()
            .find(|n| matches!(n.op, Op::Output { .. }))
            .expect("graph has an output node");
        match &n.op {
            Op::Output { name } => name.clone(),
            _ => unreachable!(),
        }
    }

    #[test]
    fn parses_serving_minimal() {
        let g = parse_bytes(&fixture_serving_minimal(), "model").unwrap();
        assert_eq!(g.metadata("source_format"), Some("tf-savedmodel"));
        assert_eq!(g.metadata("tf_meta_graph_tags"), Some("serve"));
        assert_eq!(g.metadata("tf_tensorflow_version"), Some("2.16.1"));

        // x(input) + w(const) + b(const) + matmul + biasadd-as-add + softmax + output
        assert_eq!(g.len(), 7);
        assert_eq!(g.inputs.len(), 1);
        assert_eq!(g.outputs.len(), 1);
        // The signature key names the served output, not the tensor.
        assert_eq!(out_name(&g), "dense");

        let hist = g.op_histogram();
        assert_eq!(hist.get("matmul"), Some(&1));
        assert_eq!(hist.get("add"), Some(&1)); // BiasAdd lowered
        assert_eq!(hist.get("softmax"), Some(&1));
        assert_eq!(hist.get("constant"), Some(&2));

        // Node names survive lowering.
        assert!(
            g.nodes
                .iter()
                .any(|n| n.name == "sm" && matches!(n.op, Op::Softmax { .. })),
            "softmax node should keep its name"
        );
    }

    #[test]
    fn matmul_transpose_b_lowers_to_transpose() {
        let x = node(
            "x",
            "Placeholder",
            &[],
            vec![
                attr("dtype", &vi(6, 1)),
                attr("_output_shapes", &ld(1, &ld(6, &shape_msg(&[1, 4])))),
            ],
        );
        let w = node(
            "w",
            "Const",
            &[],
            vec![value_attr(1, &[4, 4], &f32_le(&[0.0; 16]))],
        );
        let mm = node(
            "mm",
            "MatMul",
            &["x", "w"],
            vec![attr("transpose_b", &vi(5, 1))],
        );
        let bytes = saved_model(None, graph_def(vec![x, w, mm]), None);
        let g = parse_bytes(&bytes, "model").unwrap();
        assert_eq!(g.op_histogram().get("transpose"), Some(&1));
        assert_eq!(g.op_histogram().get("matmul"), Some(&1));
    }

    #[test]
    fn concat_v2_takes_axis_from_last_const() {
        let c = |name: &str, dims: &[i64], raw: Vec<u8>| {
            node(name, "Const", &[], vec![value_attr(1, dims, &raw)])
        };
        let a = c("a", &[2], f32_le(&[1.0, 2.0]));
        let b = c("b", &[2], f32_le(&[3.0, 4.0]));
        let axis = node(
            "axis",
            "Const",
            &[],
            vec![value_attr(9, &[1], &i64_le(&[1]))],
        );
        let cc = node("cc", "ConcatV2", &["a", "b", "axis"], vec![]);
        let bytes = saved_model(None, graph_def(vec![a, b, axis, cc]), None);
        let g = parse_bytes(&bytes, "model").unwrap();
        let n = g
            .nodes
            .iter()
            .find(|n| matches!(n.op, Op::Concat { .. }))
            .expect("concat lowered");
        assert_eq!(n.op, Op::Concat { axis: 1 });
        // The axis constant must NOT become an IR input of the concat.
        assert_eq!(n.inputs.len(), 2);
    }

    #[test]
    fn const_float_val_arrays_materialize() {
        // TensorProto using the typed float_val array instead of raw bytes.
        let mut tp = vi(1, 1u64); // DT_FLOAT
        tp.extend(ld(2, &shape_msg(&[2])));
        tp.extend(ld(5, &f32_le(&[7.5, -2.25]))); // packed float_val
        let v = node("v", "Const", &[], vec![attr("value", &ld(8, &tp))]);
        let bytes = saved_model(None, graph_def(vec![v]), None);
        let g = parse_bytes(&bytes, "model").unwrap();
        let t = match &g.nodes[0].op {
            Op::Constant { tensor } => tensor,
            _ => panic!("expected constant"),
        };
        assert_eq!(t.desc.shape, vec![2]);
        let expected: Vec<u8> = [7.5f32, -2.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(t.data.as_slice(), &expected[..]);
    }

    #[test]
    fn variable_checkpoint_rejected_with_guidance() {
        let var = node("w", "VarHandleOp", &[], vec![attr("dtype", &vi(6, 1))]);
        let bytes = saved_model(None, graph_def(vec![var]), None);
        match parse_bytes(&bytes, "model") {
            Err(Error::UnsupportedOperation(msg)) => {
                assert!(msg.contains("VarHandleOp"), "{msg}");
                assert!(msg.contains("variables/"), "{msg}");
                assert!(msg.contains("frozen graph"), "{msg}");
            }
            other => panic!("wrong result: {other:?}"),
        }
    }

    #[test]
    fn unknown_op_named_in_error() {
        let junk = node("junk", "FrobNorm", &["missing"], vec![]);
        let bytes = saved_model(None, graph_def(vec![junk]), None);
        let err = parse_bytes(&bytes, "model").unwrap_err();
        assert!(
            err.to_string().contains("`FrobNorm`"),
            "error should name the op: {err}"
        );
    }

    #[test]
    fn placeholder_without_shape_rejected() {
        let x = node("x", "Placeholder", &[], vec![attr("dtype", &vi(6, 1))]);
        let y = node("y", "Identity", &["x"], vec![]);
        let bytes = saved_model(None, graph_def(vec![x, y]), None);
        let err = parse_bytes(&bytes, "model").unwrap_err();
        assert!(
            err.to_string().contains("static shape"),
            "error should mention static shapes: {err}"
        );
    }

    #[test]
    fn dangling_producers_become_outputs_without_signatures() {
        let x = node(
            "x",
            "Placeholder",
            &[],
            vec![
                attr("dtype", &vi(6, 1)),
                attr("_output_shapes", &ld(1, &ld(6, &shape_msg(&[1, 4])))),
            ],
        );
        let w = node(
            "w",
            "Const",
            &[],
            vec![value_attr(1, &[4, 4], &f32_le(&[0.0; 16]))],
        );
        let mm = node("mm", "MatMul", &["x", "w"], vec![]);
        let sm = node("sm", "Softmax", &["mm"], vec![]);
        let bytes = saved_model(None, graph_def(vec![x, w, mm, sm]), None);

        let g = parse_bytes(&bytes, "model").unwrap();
        assert_eq!(g.outputs.len(), 1);
        // `sm` is the only producer nothing consumes.
        assert_eq!(out_name(&g), "sm");
    }

    #[test]
    fn pbtxt_directory_gets_export_guidance() {
        let dir = std::env::temp_dir().join("catalyst-tf-tests/pbtxt-only");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("saved_model.pbtxt"), b"# text proto").unwrap();

        let err = super::ingest(&dir).unwrap_err();
        assert!(
            err.to_string().contains("pbtxt"),
            "error should mention pbtxt: {err}"
        );
    }
}
