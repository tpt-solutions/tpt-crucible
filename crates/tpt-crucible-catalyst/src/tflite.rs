//! TensorFlow Lite (`.tflite`) ingestion.
//!
//! A native, dependency-free reader over the hand-rolled FlatBuffers cursor
//! ([`crate::flatbuf`]), following `tensorflow/lite/schema/schema.fbs`
//! ("TFL3"; table field numbers stable since v1).
//!
//! Supported builtin operators:
//!
//! * `ADD`/`SUB`/`MUL`/`DIV` → IR binaries (fused RELU/TANH activations are
//!   expanded; other fused activations are rejected),
//! * `FULLY_CONNECTED` → weight transpose (`[out,in]` → `[in,out]`) + MatMul,
//!   plus an optional bias Add,
//! * `BATCH_MATMUL` → MatMul with optional adjoint transposes,
//! * `CONCATENATION`, `RESHAPE`, `TRANSPOSE`, `CAST`,
//! * `SOFTMAX` (`beta` ignored), `LOGISTIC`→Sigmoid, `TANH`, `RELU`,
//! * `NEG`/`EXP`/`LOG`/`SQRT`/`SIN`.
//!
//! Anything else — including `CUSTOM` ops and the conv/pool family (TPT-IR
//! has no spatial ops yet) — surfaces as [`Error::UnsupportedOperation`]
//! naming it.
//!
//! Limitations: only the first subgraph of a model is converted (control-flow
//! models carry several); quantized tensors are ingested with their stored
//! dtype (no dequantization); unknown-rank tensors are rejected.

use std::path::Path;

use tpt_crucible_common::dtype::DType;
use tpt_crucible_common::error::{Error, Result};
use tpt_crucible_common::ops::{Op, ReshapeAttrs, SoftmaxAttrs, TransposeAttrs};
use tpt_crucible_common::{Graph, NodeId, Tensor, TensorData, TensorDesc};

use crate::flatbuf::{self, Tbl};

const FORMAT: &str = "tflite";

// ---- BuiltinOperator codes (schema.fbs `enum BuiltinOperator : int32`) ------
mod op {
    pub const ADD: i64 = 0;
    pub const CONCATENATION: i64 = 2;
    pub const FULLY_CONNECTED: i64 = 9;
    pub const LOGISTIC: i64 = 14;
    pub const MUL: i64 = 18;
    pub const RELU: i64 = 19;
    pub const RESHAPE: i64 = 22;
    pub const SOFTMAX: i64 = 25;
    pub const TANH: i64 = 28;
    pub const CUSTOM: i64 = 32;
    pub const TRANSPOSE: i64 = 39;
    pub const SUB: i64 = 41;
    pub const DIV: i64 = 42;
    pub const EXP: i64 = 47;
    pub const CAST: i64 = 53;
    pub const NEG: i64 = 59;
    pub const SIN: i64 = 66;
    pub const LOG: i64 = 73;
    pub const SQRT: i64 = 75;
    pub const BATCH_MATMUL: i64 = 126;
}

/// Canonical name for diagnostics (supported ops plus common unsupported
/// ones; anything else falls back to the numeric code).
fn opcode_name(code: i64) -> String {
    let name = match code {
        op::ADD => "ADD",
        op::CONCATENATION => "CONCATENATION",
        op::FULLY_CONNECTED => "FULLY_CONNECTED",
        op::LOGISTIC => "LOGISTIC",
        op::MUL => "MUL",
        op::RELU => "RELU",
        op::RESHAPE => "RESHAPE",
        op::SOFTMAX => "SOFTMAX",
        op::TANH => "TANH",
        op::CUSTOM => "CUSTOM",
        op::TRANSPOSE => "TRANSPOSE",
        op::SUB => "SUB",
        op::DIV => "DIV",
        op::EXP => "EXP",
        op::CAST => "CAST",
        op::NEG => "NEG",
        op::SIN => "SIN",
        op::LOG => "LOG",
        op::SQRT => "SQRT",
        op::BATCH_MATMUL => "BATCH_MATMUL",
        // Frequently encountered unsupported ops get friendly names too.
        1 => "AVERAGE_POOL_2D",
        3 => "CONV_2D",
        4 => "DEPTHWISE_CONV_2D",
        17 => "MAX_POOL_2D",
        _ => return format!("tflite op code {code}"),
    };
    format!("tflite op `{name}`")
}

// ---- BuiltinOptions union indices ------------------------------------------
mod opts {
    pub const ADD: u8 = 11;
    pub const CONCATENATION: u8 = 10;
    pub const FULLY_CONNECTED: u8 = 8;
    pub const MUL: u8 = 21;
    pub const RESHAPE: u8 = 17;
    pub const SOFTMAX: u8 = 9;
    pub const SUB: u8 = 28;
    pub const DIV: u8 = 29;
    pub const TRANSPOSE: u8 = 26;
    pub const CAST: u8 = 37;
    pub const NEG: u8 = 42;
    pub const EXP: u8 = 33;
    pub const BATCH_MATMUL: u8 = 101;
}

// ---- ActivationFunctionType --------------------------------------------------
mod act {
    pub const NONE: u8 = 0;
    pub const RELU: u8 = 1;
    /// Rejected during fused-activation expansion.
    #[allow(dead_code)]
    pub const RELU6: u8 = 3;
    pub const TANH: u8 = 4;
}

// ---- TensorType ---------------------------------------------------------------
mod ty {
    pub const FLOAT32: u8 = 0;
    pub const FLOAT16: u8 = 1;
    pub const INT32: u8 = 2;
    pub const UINT8: u8 = 3;
    pub const INT64: u8 = 4;
    pub const STRING: u8 = 5;
    pub const BOOL: u8 = 6;
    pub const INT16: u8 = 7;
    pub const COMPLEX64: u8 = 8;
    pub const INT8: u8 = 9;
    pub const FLOAT64: u8 = 10;
    pub const COMPLEX128: u8 = 11;
    pub const UINT64: u8 = 12;
    pub const RESOURCE: u8 = 13;
    pub const VARIANT: u8 = 14;
    pub const UINT32: u8 = 15;
    pub const UINT16: u8 = 16;
    pub const INT4: u8 = 17;
}

fn tensor_type(code: u8) -> Result<DType> {
    Ok(match code {
        ty::FLOAT32 => DType::F32,
        ty::FLOAT16 => DType::F16,
        ty::INT32 => DType::I32,
        ty::UINT8 => DType::U8,
        ty::INT64 => DType::I64,
        ty::BOOL => DType::Bool,
        ty::INT16 => DType::I16,
        ty::INT8 => DType::I8,
        ty::FLOAT64 => DType::F64,
        ty::UINT32 => DType::U32,
        ty::UINT64 => DType::U64,
        other => {
            return Err(Error::UnsupportedDType(match other {
                ty::STRING => "string".into(),
                ty::COMPLEX64 => "complex64".into(),
                ty::COMPLEX128 => "complex128".into(),
                ty::RESOURCE => "resource".into(),
                ty::VARIANT => "variant".into(),
                ty::UINT16 => "uint16".into(),
                ty::INT4 => "int4".into(),
                _ => format!("tflite tensor type {other}"),
            }))
        }
    })
}

// ---- parsed tensor ------------------------------------------------------------

struct LiteTensor {
    name: String,
    desc: TensorDesc,
    data: Vec<u8>,
}

/// Parse one `Tensor` table against the model's `buffers` array.
///
/// Field numbers per `table Tensor`: shape(0) type(1) buffer(2) name(3)
/// quantization(4) is_variable(5) sparsity(6) shape_signature(7) has_rank(8).
fn parse_tensor(model: Tbl<'_>, t: Tbl<'_>, index: usize) -> Result<LiteTensor> {
    let name = t.str_field(3)?.unwrap_or_default();
    let type_code = t
        .u8_field(1)?
        .ok_or_else(|| flatbuf::malformed(&format!("tensor {index} lacks a type")))?;
    let dtype = tensor_type(type_code)?;

    // `shape(0)` absent or empty combined with `has_rank(8)=false` marks an
    // unknown-rank tensor.
    let dims = t.i32_vec_field(0)?.unwrap_or_default();
    if t.u8_field(8)?.unwrap_or(0) == 0 && dims.is_empty() {
        return Err(Error::ParseFormat {
            path: flatbuf::PATH.into(),
            format: FORMAT.into(),
            reason: format!("tensor `{name}` has unknown rank"),
        });
    }
    let shape: Vec<usize> = dims
        .into_iter()
        .map(|d| {
            usize::try_from(d.max(0)).map_err(|_| Error::ParseFormat {
                path: flatbuf::PATH.into(),
                format: FORMAT.into(),
                reason: format!("tensor `{name}` has negative dim {d}"),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let desc = TensorDesc::new(shape, dtype);

    // buffer(2): index into model.buffers; 0 is the always-empty sentinel.
    let data: Vec<u8> = match t.i32_field(2)? {
        Some(idx) if idx > 0 => {
            let buf = model.table_vec_get(4, idx as usize)?.ok_or_else(|| {
                flatbuf::malformed(&format!(
                    "tensor `{name}` references buffer {idx} out of range"
                ))
            })?;
            buf.bytes_field(0)?.unwrap_or_default().to_vec()
        }
        _ => Vec::new(),
    };

    Ok(LiteTensor { name, desc, data })
}

// ---- lowering ----------------------------------------------------------------

/// Incremental graph under construction with tensor-index → node mapping.
struct Builder {
    g: Graph,
    vals: Vec<Option<NodeId>>, // per tensor index
}

impl Builder {
    fn get(&self, idx: usize) -> Result<NodeId> {
        self.vals
            .get(idx)
            .copied()
            .flatten()
            .ok_or_else(|| flatbuf::malformed(&format!("tensor {idx} used before it was produced")))
    }

    fn put(&mut self, idx: usize, id: NodeId) -> Result<()> {
        let slot = self
            .vals
            .get_mut(idx)
            .ok_or_else(|| flatbuf::malformed("operator writes out-of-range tensor"))?;
        if slot.is_some() {
            return Err(flatbuf::malformed(&format!(
                "tensor {idx} produced by more than one operator"
            )));
        }
        *slot = Some(id);
        Ok(())
    }

    fn constant(&mut self, idx: usize, t: &LiteTensor) -> Result<NodeId> {
        let expected = t.desc.byte_size();
        if !t.data.is_empty() && t.data.len() != expected {
            return Err(Error::TensorSizeMismatch {
                name: format!("tensor {idx} ({})", t.name),
                expected,
                actual: t.data.len(),
            });
        }
        let tensor = Tensor {
            desc: t.desc.clone(),
            data: TensorData::from_vec(t.data.clone()),
        };
        let name = if t.name.is_empty() { "const" } else { &t.name };
        let id = self
            .g
            .push(name, Op::Constant { tensor }, Vec::<NodeId>::new());
        self.put(idx, id)?;
        Ok(id)
    }

    /// Expand a fused activation (`ActivationFunctionType`) into a trailing IR
    /// op; returns the final value node.
    fn apply_fused(&mut self, act_code: u8, src: NodeId) -> Result<NodeId> {
        match act_code {
            act::NONE => Ok(src),
            act::RELU => Ok(self.g.push("", Op::Relu, vec![src])),
            act::TANH => Ok(self.g.push("", Op::Tanh, vec![src])),
            other => Err(Error::UnsupportedOperation(format!(
                "tflite fused activation code {other} (only NONE/RELU/TANH expand)"
            ))),
        }
    }
}

/// Read an integer vector from a constant input tensor (e.g. TRANSPOSE perm).
fn const_i32s(g: &Graph, id: NodeId) -> Result<Vec<i32>> {
    let node = g
        .get_node(id)
        .ok_or_else(|| flatbuf::malformed("missing node"))?;
    let tensor = match &node.op {
        Op::Constant { tensor } => tensor,
        _ => {
            return Err(flatbuf::malformed(
                "expected a constant initializer for this input",
            ))
        }
    };
    let bytes = tensor.data.as_slice();
    match tensor.desc.dtype {
        DType::I32 => Ok(bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect()),
        DType::I64 => Ok(bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as i32)
            .collect()),
        other => Err(Error::UnsupportedOperation(format!(
            "integer tensor required, found `{other}`"
        ))),
    }
}

/// Reject a mismatched BuiltinOptions union tag for `op`.
fn expect_options(got: u8, want: u8, op_name: &str) -> Result<()> {
    if got != want {
        let reason = if got == 0 {
            format!("{op_name} carries no options table")
        } else {
            format!("{op_name} carries options type {got}, expected {want}")
        };
        return Err(flatbuf::malformed(&reason));
    }
    Ok(())
}

/// Read `fused_activation_function` (field 0 of most option tables), defaulting
/// to NONE when absent.
fn fused_of(opt: Option<Tbl<'_>>) -> Result<u8> {
    match opt {
        Some(t) => Ok(t.u8_field(0)?.unwrap_or(act::NONE)),
        None => Ok(act::NONE),
    }
}

/// Adjoint (transposed-last-two) permutation for a BatchMatMul operand,
/// sized by the tensor's rank: rank 2 -> `[1, 0]`, rank 3 -> `[0, 2, 1]`, ...
fn adjoint_perm(tensors: &[LiteTensor], tensor_idx: usize) -> Result<Vec<usize>> {
    let rank = tensors
        .get(tensor_idx)
        .map(|t| t.desc.shape.len())
        .ok_or_else(|| flatbuf::malformed("BatchMatMul operand tensor out of range"))?;
    if rank < 2 {
        return Err(Error::UnsupportedOperation(format!(
            "BatchMatMul operands must have rank >= 2, got {rank}"
        )));
    }
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.swap(rank - 1, rank - 2);
    Ok(perm)
}

/// Lower one `Operator` table and register its output tensor.
///
/// Operator fields: opcode_index(0) inputs(1) outputs(2)
/// builtin_options_type(3) builtin_options(4); custom_code(5) for CUSTOM ops.
fn build_operator(
    b: &mut Builder,
    tensors: &[LiteTensor],
    o: Tbl<'_>,
    codes: &[(i64, Option<String>)],
) -> Result<()> {
    let opcode_index = o.i32_field(0)?.unwrap_or_default().max(0) as usize;
    let (opcode, custom_code) = codes
        .get(opcode_index)
        .ok_or_else(|| flatbuf::malformed(&format!("opcode_index {opcode_index} out of range")))?;
    let opcode = *opcode;
    let named = opcode_name(opcode);

    let ins: Vec<usize> = match o.i32_vec_field(1)? {
        Some(v) => v
            .into_iter()
            .map(|i| usize::try_from(i.max(0)).unwrap_or(usize::MAX))
            .collect(),
        None => Vec::new(),
    };
    let out_idx = o
        .i32_vec_field(2)?
        .and_then(|v| v.first().copied())
        .map(|i| usize::try_from(i.max(0)).unwrap_or(usize::MAX))
        .ok_or_else(|| flatbuf::malformed("operator has no outputs"))?;
    let opt = o.table_field(4)?;
    let opt_type = o.u8_field(3)?.unwrap_or(0);

    macro_rules! in_id {
        ($i:expr) => {
            b.get(
                *ins.get($i)
                    .ok_or_else(|| flatbuf::malformed(&format!("{named} missing input {}", $i)))?,
            )?
        };
    }

    let value: NodeId = match opcode {
        op::ADD | op::SUB | op::MUL | op::DIV => {
            let (ir, want_opt) = match opcode {
                op::ADD => (Op::Add, opts::ADD),
                op::SUB => (Op::Sub, opts::SUB),
                op::MUL => (Op::Mul, opts::MUL),
                _ => (Op::Div, opts::DIV),
            };
            if opt.is_some() {
                expect_options(opt_type, want_opt, &named)?;
            }
            let fused = fused_of(opt)?;
            let id = b.g.push("", ir, vec![in_id!(0), in_id!(1)]);
            b.apply_fused(fused, id)?
        }
        op::FULLY_CONNECTED => {
            if opt.is_some() {
                expect_options(opt_type, opts::FULLY_CONNECTED, &named)?;
            }
            let fused = fused_of(opt)?;
            let x = in_id!(0);
            let w = in_id!(1);
            // FC stores weights as [out, in]; MatMul needs [in, out].
            let w_t = b.g.push(
                "",
                Op::Transpose {
                    attrs: TransposeAttrs { perm: vec![1, 0] },
                },
                vec![w],
            );
            let mm = b.g.push("", Op::MatMul, vec![x, w_t]);
            let biased = if ins.len() > 2 && ins[2] != usize::MAX {
                b.g.push("", Op::Add, vec![mm, in_id!(2)])
            } else {
                mm
            };
            b.apply_fused(fused, biased)?
        }
        op::BATCH_MATMUL => {
            if opt.is_some() {
                expect_options(opt_type, opts::BATCH_MATMUL, &named)?;
            }
            let mut x = in_id!(0);
            let mut y = in_id!(1);
            if let Some(t) = opt.filter(|_| opt_type == opts::BATCH_MATMUL) {
                // The adjoint swaps the last two dims and keeps batch dims,
                // so the perm depends on the operand's rank.
                if t.u8_field(0)?.unwrap_or(0) != 0 {
                    let perm = adjoint_perm(
                        tensors,
                        *ins.first()
                            .ok_or_else(|| flatbuf::malformed("BATCH_MATMUL missing input 0"))?,
                    )?;
                    x = b.g.push(
                        "",
                        Op::Transpose {
                            attrs: TransposeAttrs { perm },
                        },
                        vec![x],
                    );
                }
                if t.u8_field(1)?.unwrap_or(0) != 0 {
                    let perm = adjoint_perm(
                        tensors,
                        *ins.get(1)
                            .ok_or_else(|| flatbuf::malformed("BATCH_MATMUL missing input 1"))?,
                    )?;
                    y = b.g.push(
                        "",
                        Op::Transpose {
                            attrs: TransposeAttrs { perm },
                        },
                        vec![y],
                    );
                }
            }
            b.g.push("", Op::MatMul, vec![x, y])
        }
        op::CONCATENATION => {
            expect_options(opt_type, opts::CONCATENATION, &named)?;
            let t = opt
                .ok_or_else(|| flatbuf::malformed("CONCATENATION without ConcatenationOptions"))?;
            let axis = t.i32_field(0)?.unwrap_or_default();
            let axis = usize::try_from(axis).map_err(|_| {
                Error::UnsupportedOperation(format!(
                    "negative concat axis {axis} requires rank inference (pending)"
                ))
            })?;
            let ids: Vec<NodeId> = ins.iter().map(|&i| b.get(i)).collect::<Result<_>>()?;
            b.g.push("", Op::Concat { axis }, ids)
        }
        op::RESHAPE => {
            let shape: Vec<i64> = match opt.filter(|_| opt_type == opts::RESHAPE) {
                Some(t) => t
                    .i32_vec_field(0)?
                    .unwrap_or_default()
                    .into_iter()
                    .map(i64::from)
                    .collect(),
                None => const_i32s(&b.g, in_id!(1))?
                    .into_iter()
                    .map(i64::from)
                    .collect(),
            };
            if shape.iter().any(|&d| d < 0) {
                return Err(Error::UnsupportedOperation(
                    "negative reshape dims (-1 inference) are not supported".into(),
                ));
            }
            let x = in_id!(0);
            b.g.push(
                "",
                Op::Reshape {
                    attrs: ReshapeAttrs { shape },
                },
                vec![x],
            )
        }
        op::TRANSPOSE => {
            if opt.is_some() {
                expect_options(opt_type, opts::TRANSPOSE, &named)?;
            }
            let perm = const_i32s(&b.g, in_id!(1))?
                .into_iter()
                .map(|p| usize::try_from(p.max(0)).unwrap_or(0))
                .collect();
            let x = in_id!(0);
            b.g.push(
                "",
                Op::Transpose {
                    attrs: TransposeAttrs { perm },
                },
                vec![x],
            )
        }
        op::SOFTMAX => {
            if opt.is_some() {
                expect_options(opt_type, opts::SOFTMAX, &named)?;
            }
            // LiteRT softmax is last-axis only; beta is ignored.
            b.g.push(
                "",
                Op::Softmax {
                    attrs: SoftmaxAttrs { axis: -1 },
                },
                vec![in_id!(0)],
            )
        }
        op::LOGISTIC => b.g.push("", Op::Sigmoid, vec![in_id!(0)]),
        op::TANH => b.g.push("", Op::Tanh, vec![in_id!(0)]),
        op::RELU => b.g.push("", Op::Relu, vec![in_id!(0)]),
        op::NEG => {
            if opt.is_some() {
                expect_options(opt_type, opts::NEG, &named)?;
            }
            b.g.push("", Op::Neg, vec![in_id!(0)])
        }
        op::EXP => {
            if opt.is_some() {
                expect_options(opt_type, opts::EXP, &named)?;
            }
            b.g.push("", Op::Exp, vec![in_id!(0)])
        }
        op::LOG => b.g.push("", Op::Log, vec![in_id!(0)]),
        op::SQRT => b.g.push("", Op::Sqrt, vec![in_id!(0)]),
        op::SIN => b.g.push("", Op::Sin, vec![in_id!(0)]),
        op::CAST => {
            if opt.is_some() {
                expect_options(opt_type, opts::CAST, &named)?;
            }
            let to = tensors
                .get(out_idx)
                .ok_or_else(|| flatbuf::malformed("CAST output tensor missing"))?
                .desc
                .dtype;
            b.g.push("", Op::Cast { to }, vec![in_id!(0)])
        }
        op::CUSTOM => {
            let custom = match custom_code {
                Some(name) if !name.is_empty() => name.clone(),
                _ => "<unnamed>".to_string(),
            };
            return Err(Error::UnsupportedOperation(format!(
                "tflite custom op `{custom}`"
            )));
        }
        other => return Err(Error::UnsupportedOperation(opcode_name(other))),
    };

    b.put(out_idx, value)?;
    Ok(())
}

/// Parse a `.tflite` model into TPT-IR (first subgraph only).
pub fn parse_bytes(bytes: &[u8], source_name: &str) -> Result<Graph> {
    let model = flatbuf::root(bytes)?;

    // Model fields: version(0) operator_codes(1) subgraphs(2) description(3)
    // buffers(4) metadata_buffer(5) metadata(6) signature_defs(7).
    let description: Option<String> = model.str_field(3)?.filter(|s| !s.is_empty());

    // Resolve operator codes once. Per schema v3a+: `deprecated_builtin_code`
    // (a byte) holds codes < 128; when it equals PLACEHOLDER_FOR_GREATER_OP_CODES
    // (127) the wide `builtin_code` field carries the real value. CUSTOM ops
    // carry their name as `custom_code` (field 1).
    let mut codes: Vec<(i64, Option<String>)> = Vec::new();
    model.table_vec_field(1, |oc| {
        let dep = i64::from(oc.u8_field(0)?.unwrap_or(0));
        let code = if dep == 127 {
            i64::from(oc.i32_field(3)?.unwrap_or(-1))
        } else {
            dep
        };
        codes.push((code, oc.str_field(1)?));
        Ok(())
    })?;

    let subgraph_count = model.table_vec_len(2)?;
    if subgraph_count == 0 {
        return Err(flatbuf::malformed("model contains no subgraphs"));
    }
    let sg = model
        .table_vec_get(2, 0)?
        .ok_or_else(|| flatbuf::malformed("model contains no subgraphs"))?;

    // SubGraph fields: tensors(0) inputs(1) outputs(2) operators(3) name(4).
    let tensor_count = sg.table_vec_len(0)?;
    let mut tensors = Vec::with_capacity(tensor_count);
    for i in 0..tensor_count {
        let t = sg
            .table_vec_get(0, i)?
            .ok_or_else(|| flatbuf::malformed("tensor vanished while re-reading"))?;
        tensors.push(parse_tensor(model, t, i)?);
    }

    let graph_name = sg
        .str_field(4)?
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| source_name.to_string());
    let mut b = Builder {
        g: Graph::new(&graph_name),
        vals: vec![None; tensors.len()],
    };
    b.g.set_metadata("source_format", FORMAT);
    if let Some(d) = &description {
        b.g.set_metadata("tflite_description", d);
    }
    if subgraph_count > 1 {
        b.g.set_metadata("tflite_subgraphs", subgraph_count.to_string());
    }

    // Subgraph inputs become IR inputs.
    let sg_inputs = sg
        .i32_vec_field(1)?
        .ok_or_else(|| flatbuf::malformed("subgraph has no inputs vector"))?;
    for &idx in &sg_inputs {
        let idx = usize::try_from(idx.max(0))
            .map_err(|_| flatbuf::malformed("negative input tensor index"))?;
        let t = tensors
            .get(idx)
            .ok_or_else(|| flatbuf::malformed(&format!("input tensor {idx} out of range")))?;
        let name = if t.name.is_empty() { "input" } else { &t.name };
        let id =
            b.g.push(name, Op::Input(t.desc.clone()), Vec::<NodeId>::new());
        b.g.mark_input(id);
        b.put(idx, id)?;
    }

    // Constant tensors (non-empty buffers not claimed by an input).
    for (idx, t) in tensors.iter().enumerate() {
        if b.vals[idx].is_none() && !t.data.is_empty() {
            b.constant(idx, t)?;
        }
    }

    // Operators, in execution order.
    sg.table_vec_field(3, |o| build_operator(&mut b, &tensors, o, &codes))?;

    // Subgraph outputs become IR outputs, named after their tensors.
    let sg_outputs = sg
        .i32_vec_field(2)?
        .ok_or_else(|| flatbuf::malformed("subgraph has no outputs vector"))?;
    if sg_outputs.is_empty() {
        return Err(flatbuf::malformed("subgraph has no outputs"));
    }
    for (n, &idx) in sg_outputs.iter().enumerate() {
        let idx = usize::try_from(idx.max(0))
            .map_err(|_| flatbuf::malformed("negative output tensor index"))?;
        let inner = b.get(idx)?;
        let name = tensors
            .get(idx)
            .map(|t| t.name.clone())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("out{n}"));
        let out = b.g.push("", Op::Output { name }, vec![inner]);
        b.g.mark_output(out);
    }

    b.g.validate()?;
    Ok(b.g)
}

/// Ingest a `.tflite` file.
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

    // ---- minimal FlatBuffers writer (test fixtures) -------------------------

    /// Field payload for fixture tables: an offset to an already-written
    /// object, or a raw 4-byte scalar.
    enum FieldVal {
        Off(usize),
        Raw([u8; 4]),
    }
    /// A forward-writing FlatBuffers emitter for fixtures. Positions are
    /// body-relative; [`Wip::finish`] prepends `[root:u32]["TFL3"]`. Tables
    /// get one 4-byte inline slot per field id (offsets stored relative to
    /// their own slot), followed by the vtable.
    #[derive(Default)]
    struct Wip {
        b: Vec<u8>,
    }

    impl Wip {
        fn pos(&self) -> usize {
            self.b.len()
        }

        fn push_u16(&mut self, v: u16) {
            self.b.extend_from_slice(&v.to_le_bytes());
        }

        fn push_u32(&mut self, v: u32) {
            self.b.extend_from_slice(&v.to_le_bytes());
        }

        fn push_i32(&mut self, v: i32) {
            self.b.extend_from_slice(&v.to_le_bytes());
        }

        fn create_string(&mut self, s: &str) -> usize {
            let p = self.pos();
            self.push_u32(s.len() as u32);
            self.b.extend_from_slice(s.as_bytes());
            p
        }

        fn create_vec_i32(&mut self, vals: &[i32]) -> usize {
            let p = self.pos();
            self.push_u32(vals.len() as u32);
            for &v in vals {
                self.push_i32(v);
            }
            p
        }

        fn create_vec_bytes(&mut self, bytes: &[u8]) -> usize {
            let p = self.pos();
            self.push_u32(bytes.len() as u32);
            self.b.extend_from_slice(bytes);
            p
        }

        /// Vector-of-tables from already-written table positions. The relative
        /// offset is computed per element *before* pushing it, i.e. against
        /// that element's own slot position.
        fn create_vec_tables(&mut self, tables: &[usize]) -> usize {
            let p = self.pos();
            self.push_u32(tables.len() as u32);
            for &t in tables {
                let slot = self.pos();
                let rel = (t as i64 - slot as i64) as i32;
                self.push_i32(rel);
            }
            p
        }

        fn create_table(&mut self, mut fields: Vec<(u16, FieldVal)>) -> usize {
            fields.sort_by_key(|(id, _)| *id);
            let max_id = fields.last().map(|(id, _)| *id).unwrap_or(0);
            let n_slots = max_id as usize + 1;
            let table_start = self.pos();
            self.push_i32(0); // soffset placeholder, patched below

            let mut slot_rels = vec![0u16; n_slots];
            for (id, val) in &fields {
                let idx = *id as usize;
                let slot_pos = self.pos();
                slot_rels[idx] = (slot_pos - table_start) as u16;
                match val {
                    // Signed relative offset: fixtures reference objects that
                    // precede their slots (canonical files use positive).
                    FieldVal::Off(target) => {
                        let rel = (*target as i64 - slot_pos as i64) as i32;
                        self.push_i32(rel);
                    }
                    FieldVal::Raw(bytes) => self.b.extend_from_slice(bytes),
                }
            }

            // Vtable directly after the table body (soffset is negative).
            let vtable_pos = self.pos();
            self.push_u16((4 + 2 * n_slots) as u16);
            self.push_u16((4 + 4 * n_slots) as u16);
            for rel in &slot_rels {
                self.push_u16(*rel);
            }
            let soffset = (table_start as i64 - vtable_pos as i64) as i32;
            self.b[table_start..table_start + 4].copy_from_slice(&soffset.to_le_bytes());
            table_start
        }

        /// Finalize into a `.tflite` byte stream rooted at `root`.
        fn finish(self, root: usize) -> Vec<u8> {
            let mut out = Vec::with_capacity(self.b.len() + 8);
            out.extend_from_slice(&((root + 8) as u32).to_le_bytes());
            out.extend_from_slice(b"TFL3");
            out.extend_from_slice(&self.b);
            out
        }
    }

    // ---- fixture table builders ---------------------------------------------

    const OPT_NONE: u8 = 0;

    fn fv(id: u16, v: FieldVal) -> (u16, FieldVal) {
        (id, v)
    }

    fn raw_i32(v: i32) -> FieldVal {
        FieldVal::Raw(v.to_le_bytes())
    }

    fn raw_u8(v: u8) -> FieldVal {
        FieldVal::Raw([v, 0, 0, 0])
    }

    fn raw_f32(v: f32) -> FieldVal {
        FieldVal::Raw(v.to_le_bytes())
    }

    /// `OperatorCode { deprecated_builtin_code(0), version(2), builtin_code(3) }`.
    fn opcode_table(w: &mut Wip, code: i64) -> usize {
        let mut fields = vec![fv(0, raw_i32(code as i32)), fv(2, raw_i32(1))];
        if code > 127 {
            fields.push(fv(3, raw_i32(code as i32)));
        }
        w.create_table(fields)
    }

    /// `Tensor { shape(0), type(1), buffer(2), name(3), has_rank(8)=true }`.
    fn tensor_table(w: &mut Wip, name: &str, shape: &[i32], dtype: u8, buffer: i32) -> usize {
        let name_pos = w.create_string(name);
        let shape_pos = w.create_vec_i32(shape);
        w.create_table(vec![
            fv(0, FieldVal::Off(shape_pos)),
            fv(1, raw_u8(dtype)),
            fv(2, raw_i32(buffer)),
            fv(3, FieldVal::Off(name_pos)),
            fv(8, raw_u8(1)),
        ])
    }

    /// `Buffer { data(0) }`.
    fn buffer_table(w: &mut Wip, data: &[u8]) -> usize {
        let d = w.create_vec_bytes(data);
        w.create_table(vec![fv(0, FieldVal::Off(d))])
    }

    /// Empty `BuiltinOptions`-style table with the given (id, value) fields.
    fn options_table(w: &mut Wip, fields: Vec<(u16, FieldVal)>) -> usize {
        w.create_table(fields)
    }

    /// `Operator { opcode_index(0), inputs(1), outputs(2),
    /// builtin_options_type(3), builtin_options(4) }`.
    fn operator_table(
        w: &mut Wip,
        opcode_index: i32,
        inputs: &[i32],
        outputs: &[i32],
        options_type: u8,
        options: Option<usize>,
    ) -> usize {
        let in_pos = w.create_vec_i32(inputs);
        let out_pos = w.create_vec_i32(outputs);
        let mut fields = vec![
            fv(0, raw_i32(opcode_index)),
            fv(1, FieldVal::Off(in_pos)),
            fv(2, FieldVal::Off(out_pos)),
            fv(3, raw_u8(options_type)),
        ];
        if let Some(o) = options {
            fields.push(fv(4, FieldVal::Off(o)));
        }
        w.create_table(fields)
    }

    /// `SubGraph { tensors(0), inputs(1), outputs(2), operators(3), name(4) }`.
    #[allow(clippy::too_many_arguments)]
    fn subgraph_table(
        w: &mut Wip,
        tensors: &[usize],
        inputs: &[i32],
        outputs: &[i32],
        operators: &[usize],
        name: &str,
    ) -> usize {
        let t = w.create_vec_tables(tensors);
        let i = w.create_vec_i32(inputs);
        let o = w.create_vec_i32(outputs);
        let ops = w.create_vec_tables(operators);
        let n = w.create_string(name);
        w.create_table(vec![
            fv(0, FieldVal::Off(t)),
            fv(1, FieldVal::Off(i)),
            fv(2, FieldVal::Off(o)),
            fv(3, FieldVal::Off(ops)),
            fv(4, FieldVal::Off(n)),
        ])
    }

    /// `Model { version(0), operator_codes(1), subgraphs(2), description(3),
    /// buffers(4) }`.
    fn model_table(
        w: &mut Wip,
        opcodes: &[usize],
        subgraphs: &[usize],
        description: &str,
        buffers: &[usize],
    ) -> usize {
        let oc = w.create_vec_tables(opcodes);
        let sg = w.create_vec_tables(subgraphs);
        let d = w.create_string(description);
        let bufs = w.create_vec_tables(buffers);
        w.create_table(vec![
            fv(0, raw_i32(3)),
            fv(1, FieldVal::Off(oc)),
            fv(2, FieldVal::Off(sg)),
            fv(3, FieldVal::Off(d)),
            fv(4, FieldVal::Off(bufs)),
        ])
    }

    fn f32_data(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn i32_data(vals: &[i32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    // ---- tests ---------------------------------------------------------------

    /// Run `f` against one writer and finalize whatever table it returns as
    /// the model root.
    fn assemble(f: impl FnOnce(&mut Wip) -> usize) -> Vec<u8> {
        let mut w = Wip::default();
        let root = f(&mut w);
        w.finish(root)
    }

    /// x[1,4] → FULLY_CONNECTED(w[3,4], b[3]) → SOFTMAX → out.
    ///
    /// Exercises buffer-backed constants, the FC weight-transpose lowering,
    /// bias add, and signature-less outputs named after tensors.
    #[test]
    fn fully_connected_with_bias_and_softmax() {
        let g = parse_bytes(&fixture_fc_softmax(), "model").unwrap();

        g.validate().unwrap();
        assert_eq!(g.metadata("source_format"), Some("tflite"));
        assert_eq!(g.metadata("tflite_description"), Some("fixture"));
        // x + w + b + transpose(w) + matmul + bias-add + softmax + output
        assert_eq!(g.len(), 8);
        assert_eq!(g.inputs.len(), 1);
        assert_eq!(g.outputs.len(), 1);
        let hist = g.op_histogram();
        assert_eq!(hist.get("constant"), Some(&2));
        assert_eq!(hist.get("transpose"), Some(&1)); // FC weight layout fix
        assert_eq!(hist.get("matmul"), Some(&1));
        assert_eq!(hist.get("add"), Some(&1)); // bias
        assert_eq!(hist.get("softmax"), Some(&1));
    }

    /// The dense+softmax model above, shared with cross-module integration
    /// tests (e.g. the `ingest_path` registry test).
    pub(crate) fn fixture_fc_softmax() -> Vec<u8> {
        assemble(|w| {
            let t_x = tensor_table(w, "x", &[1, 4], ty::FLOAT32, 0);
            let t_w = tensor_table(w, "w", &[3, 4], ty::FLOAT32, 1);
            let t_b = tensor_table(w, "b", &[3], ty::FLOAT32, 2);
            let t_y = tensor_table(w, "fc_out", &[1, 3], ty::FLOAT32, 0);
            let t_s = tensor_table(w, "sm", &[1, 3], ty::FLOAT32, 0);
            let fc_opts = options_table(w, vec![fv(0, raw_u8(act::NONE))]);
            let sm_opts = options_table(w, vec![fv(0, raw_f32(1.0))]);
            let op_fc =
                operator_table(w, 0, &[0, 1, 2], &[3], opts::FULLY_CONNECTED, Some(fc_opts));
            let op_sm = operator_table(w, 1, &[3], &[4], opts::SOFTMAX, Some(sm_opts));
            let sg = subgraph_table(
                w,
                &[t_x, t_w, t_b, t_y, t_s],
                &[0],
                &[4],
                &[op_fc, op_sm],
                "main",
            );
            let ocodes = [
                opcode_table(w, op::FULLY_CONNECTED),
                opcode_table(w, op::SOFTMAX),
            ];
            let b0 = w.create_table(vec![]);
            let b1 = buffer_table(w, &f32_data(&(0..12).map(|i| i as f32).collect::<Vec<_>>()));
            let b2 = buffer_table(w, &f32_data(&[0.5; 3]));
            model_table(w, &ocodes, &[sg], "fixture", &[b0, b1, b2])
        })
    }

    /// TRANSPOSE takes its permutation from a constant second input; RESHAPE
    /// takes the new shape from BuiltinOptions when present.
    #[test]
    fn transpose_and_reshape_lowering() {
        let bytes = assemble(|w| {
            let t_x = tensor_table(w, "x", &[2, 3], ty::FLOAT32, 0);
            let t_perm = tensor_table(w, "perm", &[2], ty::INT32, 1);
            let t_y = tensor_table(w, "y", &[3, 2], ty::FLOAT32, 0);
            let t_z = tensor_table(w, "z", &[6], ty::FLOAT32, 0);
            let shape_pos = w.create_vec_i32(&[6]);
            let rs_opts = options_table(w, vec![fv(0, FieldVal::Off(shape_pos))]);
            let op_tr = operator_table(w, 0, &[0, 1], &[2], OPT_NONE, None);
            let op_rs = operator_table(w, 1, &[2], &[3], opts::RESHAPE, Some(rs_opts));
            let sg = subgraph_table(
                w,
                &[t_x, t_perm, t_y, t_z],
                &[0],
                &[3],
                &[op_tr, op_rs],
                "main",
            );
            let ocodes = [opcode_table(w, op::TRANSPOSE), opcode_table(w, op::RESHAPE)];
            let b0 = w.create_table(vec![]);
            let b1 = buffer_table(w, &i32_data(&[1, 0]));
            model_table(w, &ocodes, &[sg], "fixture", &[b0, b1])
        });
        let g = parse_bytes(&bytes, "model").unwrap();
        g.validate().unwrap();

        let tr = g
            .nodes
            .iter()
            .find(|n| matches!(n.op, Op::Transpose { .. }))
            .expect("transpose lowered");
        assert_eq!(
            tr.op,
            Op::Transpose {
                attrs: TransposeAttrs { perm: vec![1, 0] }
            }
        );

        let rs = g
            .nodes
            .iter()
            .find(|n| matches!(n.op, Op::Reshape { .. }))
            .expect("reshape lowered");
        assert_eq!(
            rs.op,
            Op::Reshape {
                attrs: ReshapeAttrs { shape: vec![6] }
            }
        );
    }

    /// CONCATENATION reads its axis from ConcatenationOptions; LOGISTIC
    /// lowers to Sigmoid.
    #[test]
    fn concat_and_logistic() {
        let bytes = assemble(|w| {
            let t_a = tensor_table(w, "a", &[2], ty::FLOAT32, 1);
            let t_b = tensor_table(w, "b", &[2], ty::FLOAT32, 2);
            let t_c = tensor_table(w, "c", &[4], ty::FLOAT32, 0);
            let t_d = tensor_table(w, "d", &[4], ty::FLOAT32, 0);
            let cc_opts = options_table(w, vec![fv(0, raw_i32(0))]);
            let op_cc = operator_table(w, 0, &[0, 1], &[2], opts::CONCATENATION, Some(cc_opts));
            let op_lg = operator_table(w, 1, &[2], &[3], OPT_NONE, None);
            let sg = subgraph_table(w, &[t_a, t_b, t_c, t_d], &[], &[3], &[op_cc, op_lg], "main");
            let ocodes = [
                opcode_table(w, op::CONCATENATION),
                opcode_table(w, op::LOGISTIC),
            ];
            let b0 = w.create_table(vec![]);
            let b1 = buffer_table(w, &f32_data(&[1.0, 2.0]));
            let b2 = buffer_table(w, &f32_data(&[3.0, 4.0]));
            model_table(w, &ocodes, &[sg], "fixture", &[b0, b1, b2])
        });
        let g = parse_bytes(&bytes, "model").unwrap();
        g.validate().unwrap();
        let cc = g
            .nodes
            .iter()
            .find(|n| matches!(n.op, Op::Concat { .. }))
            .expect("concat lowered");
        assert_eq!(cc.op, Op::Concat { axis: 0 });
        assert_eq!(cc.inputs.len(), 2);
        assert!(
            g.nodes.iter().any(|n| matches!(n.op, Op::Sigmoid)),
            "LOGISTIC should lower to sigmoid"
        );
    }

    /// BATCH_MATMUL with adj_x inserts a transpose ahead of the MatMul.
    #[test]
    fn batch_matmul_adjoint() {
        let bytes = assemble(|w| {
            // adj_x = 1: x is stored [1, 3, 2] and adjointed to [1, 2, 3],
            // then batch-matmul with y [1, 3, 2] -> o [1, 2, 2].
            let t_x = tensor_table(w, "x", &[1, 3, 2], ty::FLOAT32, 0);
            let t_y = tensor_table(w, "y", &[1, 3, 2], ty::FLOAT32, 0);
            let t_o = tensor_table(w, "o", &[1, 2, 2], ty::FLOAT32, 0);
            let mm_opts = options_table(w, vec![fv(0, raw_u8(1)), fv(1, raw_u8(0))]);
            let op_mm = operator_table(w, 0, &[0, 1], &[2], opts::BATCH_MATMUL, Some(mm_opts));
            let sg = subgraph_table(w, &[t_x, t_y, t_o], &[0, 1], &[2], &[op_mm], "main");
            let ocodes = [opcode_table(w, op::BATCH_MATMUL)];
            let b0 = w.create_table(vec![]);
            model_table(w, &ocodes, &[sg], "fixture", &[b0])
        });

        let g = parse_bytes(&bytes, "m").unwrap();
        g.validate().unwrap();
        assert_eq!(g.inputs.len(), 2);
        let hist = g.op_histogram();
        assert_eq!(hist.get("transpose"), Some(&1)); // adj_x
        assert_eq!(hist.get("matmul"), Some(&1));

        // The transposed operand feeds the MatMul's first input.
        let mm = g
            .nodes
            .iter()
            .find(|n| matches!(n.op, Op::MatMul))
            .expect("matmul lowered");
        let tr_id = mm.inputs[0];
        assert!(matches!(
            g.get_node(tr_id).unwrap().op,
            Op::Transpose { .. }
        ));
    }

    /// Unsupported builtins name themselves; CUSTOM ops carry their code;
    /// fused RELU6 activations and unknown-rank tensors are rejected.
    #[test]
    fn unsupported_and_malformed_cases() {
        // CONV_2D has no IR equivalent yet.
        let conv_bytes = assemble(|w| {
            let t_x = tensor_table(w, "x", &[1, 8, 8, 3], ty::FLOAT32, 0);
            let t_o = tensor_table(w, "o", &[1, 8, 8, 3], ty::FLOAT32, 0);
            let op_conv = operator_table(w, 0, &[0], &[1], 1, None); // Conv2DOptions
            let sg = subgraph_table(w, &[t_x, t_o], &[0], &[1], &[op_conv], "main");
            let ocodes = [opcode_table(w, 3)]; // CONV_2D
            let b0 = w.create_table(vec![]);
            model_table(w, &ocodes, &[sg], "f", &[b0])
        });
        let err = parse_bytes(&conv_bytes, "m").unwrap_err();
        assert!(err.to_string().contains("`CONV_2D`"), "{err}");

        // A CUSTOM op surfaces with its custom_code name.
        let custom_bytes = assemble(|w| {
            let t_x = tensor_table(w, "x", &[1], ty::FLOAT32, 0);
            let t_o = tensor_table(w, "o", &[1], ty::FLOAT32, 0);
            let op_custom = operator_table(w, 0, &[0], &[1], OPT_NONE, None);
            let sg = subgraph_table(w, &[t_x, t_o], &[0], &[1], &[op_custom], "main");
            let ocodes = [custom_opcode_table(w, "MyCustomKernel")];
            let b0 = w.create_table(vec![]);
            model_table(w, &ocodes, &[sg], "f", &[b0])
        });
        let err = parse_bytes(&custom_bytes, "m").unwrap_err();
        assert!(err.to_string().contains("MyCustomKernel"), "{err}");

        // Fused RELU6 on an ADD is rejected outright.
        let relu6_bytes = assemble(|w| {
            let t_a = tensor_table(w, "a", &[2], ty::FLOAT32, 1);
            let t_b = tensor_table(w, "b", &[2], ty::FLOAT32, 2);
            let t_o = tensor_table(w, "o", &[2], ty::FLOAT32, 0);
            let add_opts = options_table(w, vec![fv(0, raw_u8(act::RELU6))]);
            let op_add = operator_table(w, 0, &[0, 1], &[2], opts::ADD, Some(add_opts));
            let sg = subgraph_table(w, &[t_a, t_b, t_o], &[], &[2], &[op_add], "main");
            let ocodes = [opcode_table(w, op::ADD)];
            let b0 = w.create_table(vec![]);
            let b1 = buffer_table(w, &f32_data(&[1.0, 2.0]));
            let b2 = buffer_table(w, &f32_data(&[3.0, 4.0]));
            model_table(w, &ocodes, &[sg], "f", &[b0, b1, b2])
        });
        let err = parse_bytes(&relu6_bytes, "m").unwrap_err();
        assert!(err.to_string().contains("fused activation"), "{err}");

        // Unknown-rank input tensors are rejected.
        let rank_bytes = assemble(|w| {
            let t_x = unknown_rank_tensor(w, "x", ty::FLOAT32);
            let t_o = tensor_table(w, "o", &[1], ty::FLOAT32, 0);
            let sg = subgraph_table(w, &[t_x, t_o], &[0], &[1], &[], "main");
            let ocodes: [usize; 0] = [];
            let b0 = w.create_table(vec![]);
            model_table(w, &ocodes, &[sg], "f", &[b0])
        });
        let err = parse_bytes(&rank_bytes, "m").unwrap_err();
        assert!(err.to_string().contains("unknown rank"), "{err}");

        // A file without the TFL3 magic is refused up front.
        let err = parse_bytes(b"\x00\x00\x00\x00junkjunk", "m").unwrap_err();
        assert!(err.to_string().contains("TFL3"), "{err}");
    }

    /// An `OperatorCode` for a CUSTOM op carrying `custom_code(1)`.
    fn custom_opcode_table(w: &mut Wip, name: &str) -> usize {
        let name_pos = w.create_string(name);
        w.create_table(vec![
            fv(0, raw_i32(op::CUSTOM as i32)),
            fv(1, FieldVal::Off(name_pos)),
            fv(2, raw_i32(1)),
        ])
    }

    /// A tensor with `has_rank=false` and an empty shape.
    fn unknown_rank_tensor(w: &mut Wip, name: &str, dtype: u8) -> usize {
        let name_pos = w.create_string(name);
        w.create_table(vec![fv(1, raw_u8(dtype)), fv(3, FieldVal::Off(name_pos))])
    }
}
