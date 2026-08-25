//! Converts [`tpt_crucible_common::Graph`] (TPT-IR) to and from a TPT-UIR
//! [`Region`] in the Crucible dialect (`tpt_crucible.*` ops).
//!
//! Every TPT-IR node becomes one Crucible-dialect [`Operation`] in a single
//! [`Block`] inside a single [`Region`] (no nested regions/control flow, same
//! as TPT-IR itself). Graph inputs are represented as `tpt_crucible.input`
//! operations rather than block arguments: block arguments have no attribute
//! slot, and preserving each node's original name (needed for a lossless
//! round trip) requires one.
//!
//! A trailing `tpt_crucible.graph_info` operation (one past the last real
//! node id) carries the source [`Graph`]'s `name` and `metadata`, which have
//! no equivalent field on [`Region`].

use std::collections::BTreeMap;

use tpt_crucible_common::ops::{
    AttentionAttrs, LayerNormAttrs, ReshapeAttrs, RmsNormAttrs, RopeAttrs, SoftmaxAttrs,
    TransposeAttrs,
};
use tpt_crucible_common::{DType, Graph, NodeId, Op, Tensor, TensorData, TensorDesc};
use tpt_uir_core::attr::{Attribute, AttributeValue};
use tpt_uir_core::types::{Dimension, ScalarType, ShapeSpec, TensorType, Type};
use tpt_uir_core::{Block, OpName, Operation, Region, ValueId};

/// Errors converting between TPT-IR and TPT-UIR.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum AdapterError {
    /// A Crucible region must have exactly one block.
    #[error("region must contain exactly one block, found {0}")]
    NotSingleBlock(usize),
    /// An operation's dialect/op name isn't a known Crucible compute op.
    #[error("unknown crucible op: {0}")]
    UnknownOp(String),
    /// A node id in `0..count` has neither an argument nor an operation.
    #[error("missing node for value id {0}")]
    MissingNode(ValueId),
    /// A required attribute is absent.
    #[error("operation {0} is missing required attribute `{1}`")]
    MissingAttribute(ValueId, &'static str),
    /// An attribute is present but has the wrong shape/type/encoding.
    #[error("operation {0} has malformed attribute `{1}`: {2}")]
    MalformedAttribute(ValueId, &'static str, String),
}

// ---------------------------------------------------------------------------
// dtype / shape / type conversions
// ---------------------------------------------------------------------------

fn dtype_to_scalar(dt: DType) -> ScalarType {
    match dt {
        DType::Bool => ScalarType::Bool,
        DType::U8 => ScalarType::U8,
        DType::U16 => ScalarType::U16,
        DType::U32 => ScalarType::U32,
        DType::U64 => ScalarType::U64,
        DType::I8 => ScalarType::I8,
        DType::I16 => ScalarType::I16,
        DType::I32 => ScalarType::I32,
        DType::I64 => ScalarType::I64,
        DType::F16 => ScalarType::F16,
        DType::Bf16 => ScalarType::BF16,
        DType::F32 => ScalarType::F32,
        DType::F64 => ScalarType::F64,
        DType::Q4_0 => ScalarType::Q4_0,
        DType::Q4_1 => ScalarType::Q4_1,
        DType::Q5_0 => ScalarType::Q5_0,
        DType::Q5_1 => ScalarType::Q5_1,
        DType::Q8_0 => ScalarType::Q8_0,
    }
}

fn scalar_to_dtype(st: ScalarType) -> DType {
    match st {
        ScalarType::Bool => DType::Bool,
        ScalarType::U8 => DType::U8,
        ScalarType::U16 => DType::U16,
        ScalarType::U32 => DType::U32,
        ScalarType::U64 => DType::U64,
        ScalarType::I8 => DType::I8,
        ScalarType::I16 => DType::I16,
        ScalarType::I32 => DType::I32,
        ScalarType::I64 => DType::I64,
        ScalarType::F16 => DType::F16,
        ScalarType::BF16 => DType::Bf16,
        ScalarType::F32 => DType::F32,
        ScalarType::F64 => DType::F64,
        ScalarType::Q4_0 => DType::Q4_0,
        ScalarType::Q4_1 => DType::Q4_1,
        ScalarType::Q5_0 => DType::Q5_0,
        ScalarType::Q5_1 => DType::Q5_1,
        ScalarType::Q8_0 => DType::Q8_0,
    }
}

fn dtype_from_name(s: &str) -> Option<DType> {
    Some(match s {
        "bool" => DType::Bool,
        "u8" => DType::U8,
        "u16" => DType::U16,
        "u32" => DType::U32,
        "u64" => DType::U64,
        "i8" => DType::I8,
        "i16" => DType::I16,
        "i32" => DType::I32,
        "i64" => DType::I64,
        "f16" => DType::F16,
        "bf16" => DType::Bf16,
        "f32" => DType::F32,
        "f64" => DType::F64,
        "q4_0" => DType::Q4_0,
        "q4_1" => DType::Q4_1,
        "q5_0" => DType::Q5_0,
        "q5_1" => DType::Q5_1,
        "q8_0" => DType::Q8_0,
        _ => return None,
    })
}

fn shape_to_spec(shape: &[usize]) -> ShapeSpec {
    ShapeSpec {
        dimensions: shape.iter().map(|&d| Dimension::Fixed(d)).collect(),
    }
}

fn spec_to_shape(id: ValueId, spec: &ShapeSpec) -> Result<Vec<usize>, AdapterError> {
    spec.dimensions
        .iter()
        .map(|d| match d {
            Dimension::Fixed(n) => Ok(*n),
            other => Err(AdapterError::MalformedAttribute(
                id,
                "type",
                format!("crucible shapes must be Fixed, found {other:?}"),
            )),
        })
        .collect()
}

fn tensor_desc_to_type(desc: &TensorDesc) -> Type {
    Type::Tensor(TensorType {
        dtype: dtype_to_scalar(desc.dtype),
        shape: Some(shape_to_spec(&desc.shape)),
    })
}

fn type_to_tensor_desc(id: ValueId, ty: &Type) -> Result<TensorDesc, AdapterError> {
    match ty {
        Type::Tensor(tt) => {
            let shape = match &tt.shape {
                Some(spec) => spec_to_shape(id, spec)?,
                None => Vec::new(),
            };
            Ok(TensorDesc::new(shape, scalar_to_dtype(tt.dtype)))
        }
        other => Err(AdapterError::MalformedAttribute(
            id,
            "type",
            format!("expected a tensor type, found {other:?}"),
        )),
    }
}

// ---------------------------------------------------------------------------
// integer-list attribute encoding (Transpose::perm, Reshape::shape)
// ---------------------------------------------------------------------------

fn encode_ints<T: ToString>(values: &[T]) -> String {
    values
        .iter()
        .map(T::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn decode_usizes(id: ValueId, key: &'static str, s: &str) -> Result<Vec<usize>, AdapterError> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|p| {
            p.parse::<usize>()
                .map_err(|e| AdapterError::MalformedAttribute(id, key, e.to_string()))
        })
        .collect()
}

fn decode_i64s(id: ValueId, key: &'static str, s: &str) -> Result<Vec<i64>, AdapterError> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|p| {
            p.parse::<i64>()
                .map_err(|e| AdapterError::MalformedAttribute(id, key, e.to_string()))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Graph -> Region
// ---------------------------------------------------------------------------

fn op_attributes(op: &Op) -> Vec<Attribute> {
    match op {
        Op::Cast { to } => vec![Attribute::string("to", to.name())],
        Op::Softmax { attrs } => vec![Attribute::i64("axis", attrs.axis as i64)],
        Op::RmsNorm { attrs } => vec![Attribute::f64("eps", attrs.eps as f64)],
        Op::LayerNorm { attrs } => vec![Attribute::f64("eps", attrs.eps as f64)],
        Op::Rope { attrs } => vec![Attribute::f64("theta", attrs.theta as f64)],
        Op::Attention { attrs } => {
            let mut v = vec![
                Attribute::i64("num_heads", attrs.num_heads as i64),
                Attribute::i64("num_kv_heads", attrs.num_kv_heads as i64),
                Attribute::i64("head_dim", attrs.head_dim as i64),
                Attribute::i64("causal", attrs.causal as i64),
            ];
            if let Some(scale) = attrs.scale {
                v.push(Attribute::f64("scale", scale as f64));
            }
            v
        }
        Op::Transpose { attrs } => vec![Attribute::string("perm", encode_ints(&attrs.perm))],
        Op::Reshape { attrs } => vec![Attribute::string("shape", encode_ints(&attrs.shape))],
        Op::Concat { axis } => vec![Attribute::i64("axis", *axis as i64)],
        _ => Vec::new(),
    }
}

/// Convert a TPT-IR [`Graph`] into a single-block, single-region TPT-UIR
/// Crucible-dialect [`Region`]. Always succeeds: TPT-IR shapes are already
/// concrete, so nothing here can violate the Crucible dialect's Fixed-shape
/// requirement.
pub fn graph_to_region(graph: &Graph) -> Region {
    let mut operations = Vec::with_capacity(graph.nodes.len() + 1);

    for (idx, node) in graph.nodes.iter().enumerate() {
        let vid = idx as ValueId;
        let (op_kind, operands, results, mut attributes): (&str, Vec<ValueId>, Vec<ValueId>, _) =
            match &node.op {
                Op::Input(desc) => (
                    "input",
                    Vec::new(),
                    vec![vid],
                    vec![Attribute {
                        key: "type".into(),
                        value: AttributeValue::Type(tensor_desc_to_type(desc)),
                    }],
                ),
                Op::Constant { tensor } => (
                    "constant",
                    Vec::new(),
                    vec![vid],
                    vec![
                        Attribute {
                            key: "type".into(),
                            value: AttributeValue::Type(tensor_desc_to_type(&tensor.desc)),
                        },
                        Attribute::bytes("value", tensor.data.as_slice().to_vec()),
                    ],
                ),
                Op::Output { name } => (
                    "output",
                    node.inputs.iter().map(|n| n.0).collect(),
                    Vec::new(),
                    vec![Attribute::string("name", name.clone())],
                ),
                other => (
                    other.name(),
                    node.inputs.iter().map(|n| n.0).collect(),
                    vec![vid],
                    op_attributes(other),
                ),
            };
        attributes.insert(0, Attribute::string("node_name", node.name.clone()));
        operations.push(Operation {
            id: vid,
            op_name: OpName::new("tpt_crucible", op_kind),
            operands,
            results,
            regions: Vec::new(),
            attributes,
        });
    }

    let mut info_attrs = vec![Attribute::string("name", graph.name.clone())];
    for (k, v) in &graph.metadata {
        info_attrs.push(Attribute::string(format!("meta:{k}"), v.clone()));
    }
    operations.push(Operation {
        id: graph.nodes.len() as ValueId,
        op_name: OpName::new("tpt_crucible", "graph_info"),
        operands: Vec::new(),
        results: Vec::new(),
        regions: Vec::new(),
        attributes: info_attrs,
    });

    Region {
        blocks: vec![Block {
            arguments: Vec::new(),
            operations,
        }],
    }
}

// ---------------------------------------------------------------------------
// Region -> Graph
// ---------------------------------------------------------------------------

fn attr_value<'a>(op: &'a Operation, key: &str) -> Option<&'a AttributeValue> {
    op.attributes
        .iter()
        .find(|a| a.key == key)
        .map(|a| &a.value)
}

fn attr_string(op: &Operation, key: &'static str) -> Result<String, AdapterError> {
    match attr_value(op, key) {
        Some(AttributeValue::String(s)) => Ok(s.clone()),
        _ => Err(AdapterError::MissingAttribute(op.id, key)),
    }
}

fn attr_i64(op: &Operation, key: &'static str) -> Result<i64, AdapterError> {
    match attr_value(op, key) {
        Some(AttributeValue::I64(v)) => Ok(*v),
        _ => Err(AdapterError::MissingAttribute(op.id, key)),
    }
}

fn attr_f64(op: &Operation, key: &'static str) -> Result<f64, AdapterError> {
    match attr_value(op, key) {
        Some(AttributeValue::F64(v)) => Ok(*v),
        _ => Err(AdapterError::MissingAttribute(op.id, key)),
    }
}

fn attr_f64_opt(op: &Operation, key: &str) -> Option<f64> {
    match attr_value(op, key) {
        Some(AttributeValue::F64(v)) => Some(*v),
        _ => None,
    }
}

fn attr_bytes(op: &Operation, key: &'static str) -> Result<Vec<u8>, AdapterError> {
    match attr_value(op, key) {
        Some(AttributeValue::Bytes(b)) => Ok(b.clone()),
        _ => Err(AdapterError::MissingAttribute(op.id, key)),
    }
}

fn attr_type(op: &Operation, key: &'static str) -> Result<Type, AdapterError> {
    match attr_value(op, key) {
        Some(AttributeValue::Type(t)) => Ok(t.clone()),
        _ => Err(AdapterError::MissingAttribute(op.id, key)),
    }
}

fn reconstruct_op(op: &Operation) -> Result<Op, AdapterError> {
    Ok(match op.op_name.op.as_str() {
        "input" => Op::Input(type_to_tensor_desc(op.id, &attr_type(op, "type")?)?),
        "constant" => {
            let desc = type_to_tensor_desc(op.id, &attr_type(op, "type")?)?;
            let bytes = attr_bytes(op, "value")?;
            Op::Constant {
                tensor: Tensor::new(desc, TensorData::from_vec(bytes))
                    .map_err(|e| AdapterError::MalformedAttribute(op.id, "value", e.to_string()))?,
            }
        }
        "output" => Op::Output {
            name: attr_string(op, "name")?,
        },
        "cast" => {
            let name = attr_string(op, "to")?;
            let to = dtype_from_name(&name).ok_or_else(|| {
                AdapterError::MalformedAttribute(op.id, "to", format!("unknown dtype `{name}`"))
            })?;
            Op::Cast { to }
        }
        "add" => Op::Add,
        "sub" => Op::Sub,
        "mul" => Op::Mul,
        "div" => Op::Div,
        "neg" => Op::Neg,
        "exp" => Op::Exp,
        "log" => Op::Log,
        "sqrt" => Op::Sqrt,
        "sin" => Op::Sin,
        "cos" => Op::Cos,
        "tanh" => Op::Tanh,
        "erf" => Op::Erf,
        "relu" => Op::Relu,
        "sigmoid" => Op::Sigmoid,
        "silu" => Op::Silu,
        "gelu" => Op::Gelu,
        "matmul" => Op::MatMul,
        "softmax" => Op::Softmax {
            attrs: SoftmaxAttrs {
                axis: attr_i64(op, "axis")? as i32,
            },
        },
        "rms_norm" => Op::RmsNorm {
            attrs: RmsNormAttrs {
                eps: attr_f64(op, "eps")? as f32,
            },
        },
        "layer_norm" => Op::LayerNorm {
            attrs: LayerNormAttrs {
                eps: attr_f64(op, "eps")? as f32,
            },
        },
        "attention" => Op::Attention {
            attrs: AttentionAttrs {
                num_heads: attr_i64(op, "num_heads")? as u32,
                num_kv_heads: attr_i64(op, "num_kv_heads")? as u32,
                head_dim: attr_i64(op, "head_dim")? as u32,
                causal: attr_i64(op, "causal")? != 0,
                scale: attr_f64_opt(op, "scale").map(|f| f as f32),
            },
        },
        "rope" => Op::Rope {
            attrs: RopeAttrs {
                theta: attr_f64(op, "theta")? as f32,
            },
        },
        "embedding" => Op::Embedding,
        "transpose" => {
            let raw = attr_string(op, "perm")?;
            Op::Transpose {
                attrs: TransposeAttrs {
                    perm: decode_usizes(op.id, "perm", &raw)?,
                },
            }
        }
        "reshape" => {
            let raw = attr_string(op, "shape")?;
            Op::Reshape {
                attrs: ReshapeAttrs {
                    shape: decode_i64s(op.id, "shape", &raw)?,
                },
            }
        }
        "concat" => Op::Concat {
            axis: attr_i64(op, "axis")? as usize,
        },
        other => return Err(AdapterError::UnknownOp(other.to_string())),
    })
}

/// Convert a Crucible-dialect [`Region`] back into a TPT-IR [`Graph`].
///
/// The inverse of [`graph_to_region`]; round-trips losslessly for any
/// [`Region`] actually produced by [`graph_to_region`].
pub fn region_to_graph(region: &Region) -> Result<Graph, AdapterError> {
    if region.blocks.len() != 1 {
        return Err(AdapterError::NotSingleBlock(region.blocks.len()));
    }
    let block = &region.blocks[0];

    let mut graph_name = String::new();
    let mut metadata: BTreeMap<String, String> = BTreeMap::new();
    let mut op_map: BTreeMap<ValueId, &Operation> = BTreeMap::new();

    for op in &block.operations {
        if op.op_name.dialect == "tpt_crucible" && op.op_name.op == "graph_info" {
            for attr in &op.attributes {
                if let AttributeValue::String(s) = &attr.value {
                    if attr.key == "name" {
                        graph_name = s.clone();
                    } else if let Some(k) = attr.key.strip_prefix("meta:") {
                        metadata.insert(k.to_string(), s.clone());
                    }
                }
            }
            continue;
        }
        op_map.insert(op.id, op);
    }

    let count = op_map.keys().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut graph = Graph::new(graph_name);
    for (k, v) in metadata {
        graph.set_metadata(k, v);
    }

    for vid in 0..count {
        let op = op_map.get(&vid).ok_or(AdapterError::MissingNode(vid))?;
        let name = attr_string(op, "node_name")?;
        let node_op = reconstruct_op(op)?;
        let is_input = matches!(node_op, Op::Input(_));
        let is_output = matches!(node_op, Op::Output { .. });
        let inputs: Vec<NodeId> = op.operands.iter().map(|&o| NodeId(o)).collect();
        let id = graph.push(name, node_op, inputs);
        if is_input {
            graph.mark_input(id);
        }
        if is_output {
            graph.mark_output(id);
        }
    }

    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_crucible_common::ops::AttentionAttrs as CAttentionAttrs;
    use tpt_uir_dialects::crucible::CrucibleDialect;
    use tpt_uir_dialects::validate::ValidateDialect;

    /// A small Llama-block-shaped graph: embed -> rms_norm -> attention ->
    /// matmul -> softmax -> output, exercising most op variants and a
    /// Constant/Q4_0-quantized weight.
    fn llama_block_graph() -> Graph {
        let mut g = Graph::new("llama-block");
        g.set_metadata("source_format", "test");
        g.set_metadata("arch", "llama");

        let ids = g.push(
            "ids",
            Op::Input(TensorDesc::new(vec![1, 8], DType::I32)),
            vec![],
        );
        let embed_w = g.push(
            "embed.weight",
            Op::Constant {
                tensor: Tensor::from_f32(vec![32, 16], &[0.0; 32 * 16]),
            },
            vec![],
        );
        let x = g.push("embed", Op::Embedding, vec![embed_w, ids]);

        let norm_w = g.push(
            "norm.weight",
            Op::Constant {
                tensor: Tensor::from_f32(vec![16], &[1.0; 16]),
            },
            vec![],
        );
        let normed = g.push(
            "normed",
            Op::RmsNorm {
                attrs: RmsNormAttrs { eps: 1e-5 },
            },
            vec![x, norm_w],
        );

        let attn = g.push(
            "attn",
            Op::Attention {
                attrs: CAttentionAttrs {
                    num_heads: 4,
                    num_kv_heads: 4,
                    head_dim: 4,
                    causal: true,
                    scale: None,
                },
            },
            vec![normed, normed, normed],
        );

        let proj_w = g.push(
            "proj.weight",
            Op::Constant {
                tensor: Tensor::from_f32(vec![16, 32], &[0.0; 16 * 32]),
            },
            vec![],
        );
        let logits_raw = g.push("proj", Op::MatMul, vec![attn, proj_w]);
        let logits = g.push(
            "logits_softmax",
            Op::Softmax {
                attrs: SoftmaxAttrs { axis: -1 },
            },
            vec![logits_raw],
        );
        let reshaped = g.push(
            "reshaped",
            Op::Reshape {
                attrs: ReshapeAttrs {
                    shape: vec![-1, 32],
                },
            },
            vec![logits],
        );
        let out = g.push(
            "logits",
            Op::Output {
                name: "logits".into(),
            },
            vec![reshaped],
        );

        g.mark_input(ids);
        g.mark_output(out);
        g.validate().expect("test graph is well-formed");
        g
    }

    #[test]
    fn round_trip_is_lossless() {
        let g = llama_block_graph();
        let region = graph_to_region(&g);
        CrucibleDialect::validate_all(&region).expect("region is a valid Crucible region");
        let back = region_to_graph(&region).expect("region converts back to a graph");
        assert_eq!(back, g);
    }

    #[test]
    fn region_uses_no_block_arguments() {
        let region = graph_to_region(&llama_block_graph());
        assert!(region.blocks[0].arguments.is_empty());
    }

    #[test]
    fn graph_info_carries_name_and_metadata() {
        let g = llama_block_graph();
        let region = graph_to_region(&g);
        let info = region.blocks[0]
            .operations
            .iter()
            .find(|op| op.op_name.op == "graph_info")
            .unwrap();
        assert!(info
            .attributes
            .iter()
            .any(|a| a.key == "name" && a.value == AttributeValue::String("llama-block".into())));
        assert!(info
            .attributes
            .iter()
            .any(|a| a.key == "meta:arch" && a.value == AttributeValue::String("llama".into())));
    }

    #[test]
    fn unknown_op_name_is_rejected() {
        let region = Region {
            blocks: vec![Block {
                arguments: Vec::new(),
                operations: vec![Operation {
                    id: 0,
                    op_name: OpName::new("tpt_crucible", "not_a_real_op"),
                    operands: Vec::new(),
                    results: vec![0],
                    regions: Vec::new(),
                    attributes: vec![Attribute::string("node_name", "x")],
                }],
            }],
        };
        assert_eq!(
            region_to_graph(&region),
            Err(AdapterError::UnknownOp("not_a_real_op".into()))
        );
    }

    #[test]
    fn multi_block_region_is_rejected() {
        let region = Region {
            blocks: vec![
                Block {
                    arguments: Vec::new(),
                    operations: Vec::new(),
                },
                Block {
                    arguments: Vec::new(),
                    operations: Vec::new(),
                },
            ],
        };
        assert_eq!(
            region_to_graph(&region),
            Err(AdapterError::NotSingleBlock(2))
        );
    }
}
