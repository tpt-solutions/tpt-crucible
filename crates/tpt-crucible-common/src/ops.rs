//! The operator set of TPT-IR.

use serde::{Deserialize, Serialize};

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::{Tensor, TensorDesc};

/// Softmax attributes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SoftmaxAttrs {
    /// Axis to normalize over (negative values index from the end).
    pub axis: i32,
}

/// RMSNorm attributes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RmsNormAttrs {
    /// Epsilon added inside the square root.
    pub eps: f32,
}

/// LayerNorm attributes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LayerNormAttrs {
    /// Epsilon added to variance.
    pub eps: f32,
}

/// Rotary position embedding attributes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RopeAttrs {
    /// RoPE base frequency (typically 10_000).
    pub theta: f32,
}

/// Multi-head attention attributes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AttentionAttrs {
    /// Query heads per layer.
    pub num_heads: u32,
    /// Key/value heads (GQA when fewer than `num_heads`).
    pub num_kv_heads: u32,
    /// Dimension of each head.
    pub head_dim: u32,
    /// Apply a causal mask.
    pub causal: bool,
    /// Softmax scale override; defaults to `1/sqrt(head_dim)`.
    pub scale: Option<f32>,
}

/// Permutation for transpose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransposeAttrs {
    /// Output axis `i` reads input axis `perm[i]`.
    pub perm: Vec<usize>,
}

/// Target shape for reshape; at most one `-1` dimension is inferred.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReshapeAttrs {
    /// New shape (-1 infers one dimension).
    pub shape: Vec<i64>,
}

/// One node's operation in a [`crate::graph::Graph`].
///
/// Ops are pure functions over their input nodes; side effects live only at
/// hardware backends. The enum uses standard external serde tagging so it
/// round-trips through both JSON and bincode (`{"matmul": null}` in JSON).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    /// A named external input with a fixed descriptor.
    Input(TensorDesc),
    /// An immutable weight or literal embedded in the IR.
    Constant {
        /// The value.
        tensor: Tensor,
    },
    /// Marks a graph-level output.
    Output {
        /// Output name (e.g. `"logits"`).
        name: String,
    },
    /// Elementwise numeric cast.
    Cast {
        /// Target dtype.
        to: DType,
    },
    /// `a + b` (broadcasting).
    Add,
    /// `a - b` (broadcasting).
    Sub,
    /// `a * b` (broadcasting).
    Mul,
    /// `a / b` (broadcasting).
    Div,
    Neg,
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Tanh,
    Erf,
    Relu,
    Sigmoid,
    /// SiLU: `x * sigmoid(x)` — the Llama-family activation.
    Silu,
    /// Gaussian-error linear unit (tanh approximation).
    Gelu,
    /// Batched matrix multiply over the last two dimensions.
    MatMul,
    /// Softmax along an axis.
    Softmax {
        /// Attributes.
        attrs: SoftmaxAttrs,
    },
    /// Root-mean-square norm with learned scale; inputs `(x, weight)`.
    RmsNorm {
        /// Attributes.
        attrs: RmsNormAttrs,
    },
    /// Layer normalization; inputs `(x, weight, bias)`.
    LayerNorm {
        /// Attributes.
        attrs: LayerNormAttrs,
    },
    /// Fused scaled-dot-product attention over `(q, k, v[, mask])`.
    Attention {
        /// Attributes.
        attrs: AttentionAttrs,
    },
    /// Rotary position embedding over `(x, position_ids)`.
    Rope {
        /// Attributes.
        attrs: RopeAttrs,
    },
    /// Token embedding lookup over `(weight, ids)`.
    Embedding,
    Transpose {
        /// Attributes.
        attrs: TransposeAttrs,
    },
    Reshape {
        /// Attributes.
        attrs: ReshapeAttrs,
    },
    Concat {
        /// Concatenation axis.
        axis: usize,
    },
}

impl Op {
    /// Canonical snake_case name (matches the serde tag).
    pub fn name(&self) -> &'static str {
        match self {
            Self::Input(_) => "input",
            Self::Constant { .. } => "constant",
            Self::Output { .. } => "output",
            Self::Cast { .. } => "cast",
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Neg => "neg",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sqrt => "sqrt",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tanh => "tanh",
            Self::Erf => "erf",
            Self::Relu => "relu",
            Self::Sigmoid => "sigmoid",
            Self::Silu => "silu",
            Self::Gelu => "gelu",
            Self::Softmax { .. } => "softmax",
            Self::MatMul => "matmul",
            Self::RmsNorm { .. } => "rms_norm",
            Self::LayerNorm { .. } => "layer_norm",
            Self::Attention { .. } => "attention",
            Self::Rope { .. } => "rope",
            Self::Embedding => "embedding",
            Self::Transpose { .. } => "transpose",
            Self::Reshape { .. } => "reshape",
            Self::Concat { .. } => "concat",
        }
    }

    /// Inclusive `(min, max)` input count enforced by graph validation.
    pub fn arity(&self) -> (usize, usize) {
        match self {
            Self::Input(_) | Self::Constant { .. } => (0, 0),
            Self::Output { .. }
            | Self::Cast { .. }
            | Self::Softmax { .. }
            | Self::Transpose { .. }
            | Self::Reshape { .. }
            | Self::Neg
            | Self::Exp
            | Self::Log
            | Self::Sqrt
            | Self::Sin
            | Self::Cos
            | Self::Tanh
            | Self::Erf
            | Self::Relu
            | Self::Sigmoid
            | Self::Silu
            | Self::Gelu => (1, 1),
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::MatMul
            | Self::Embedding
            | Self::RmsNorm { .. }
            | Self::Rope { .. } => (2, 2),
            Self::LayerNorm { .. } => (3, 3),
            Self::Attention { .. } => (3, 4),
            Self::Concat { .. } => (2, usize::MAX),
        }
    }

    /// True for pointwise ops that fuse trivially (fusion-pass hint).
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Sub
                | Self::Mul
                | Self::Div
                | Self::Neg
                | Self::Exp
                | Self::Log
                | Self::Sqrt
                | Self::Sin
                | Self::Cos
                | Self::Tanh
                | Self::Erf
                | Self::Relu
                | Self::Sigmoid
                | Self::Silu
                | Self::Gelu
        )
    }

    /// Infer this op's output descriptor from its inputs' descriptors.
    ///
    /// * `Ok(None)` — inference is impossible (an input descriptor is
    ///   unknown); graph validation simply propagates the unknown.
    /// * `Err` — the wiring is **definitively** inconsistent (e.g. a MatMul
    ///   whose inner dimensions disagree, a Transpose perm that is not a
    ///   permutation of its input's rank).
    /// * `Ok(Some(desc))` — the output shape/dtype, for downstream nodes.
    ///
    /// # Unknown (dynamic) dimensions
    ///
    /// A stored dimension of `0` means "unknown/dynamic" (TensorFlow's `-1`
    /// placeholders map to `0` at ingest). Any comparison that would touch an
    /// unknown dimension is skipped rather than guessed, so dynamic models
    /// validate cleanly while static-shape mistakes still fail loudly.
    ///
    /// # Mixed dtypes
    ///
    /// Dtype rules are deliberately permissive: block-quantized weights are
    /// first-class MatMul operands (dequantized on load), and mixed-precision
    /// float pairs widen instead of erroring. Only semantically wrong dtypes
    /// (non-integer embedding indices) are rejected.
    pub fn infer_output_desc(&self, inputs: &[Option<TensorDesc>]) -> Result<Option<TensorDesc>> {
        // Descriptor of `inputs[i]`, or `None` when either the edge is absent
        // or the producer's descriptor is unknown.
        let get = |i: usize| -> Option<&TensorDesc> { inputs.get(i).and_then(|d| d.as_ref()) };
        let passthrough = || Ok(get(0).cloned());

        match self {
            // Sources carry their own descriptors; nothing to infer.
            Self::Input(desc) => Ok(Some(desc.clone())),
            Self::Constant { tensor } => Ok(Some(tensor.desc.clone())),
            Self::Output { .. } => passthrough(),
            Self::Cast { to } => match get(0) {
                None => Ok(None),
                Some(d) => Ok(Some(TensorDesc::new(d.shape.clone(), *to))),
            },
            // Shape- and dtype-preserving unaries.
            Self::Neg
            | Self::Exp
            | Self::Log
            | Self::Sqrt
            | Self::Sin
            | Self::Cos
            | Self::Tanh
            | Self::Erf
            | Self::Relu
            | Self::Sigmoid
            | Self::Silu
            | Self::Gelu
            | Self::Softmax { .. } => passthrough(),

            // Broadcasting binaries: NumPy-style right-aligned broadcasting.
            Self::Add | Self::Sub | Self::Mul | Self::Div => {
                let (Some(a), Some(b)) = (get(0), get(1)) else {
                    return Ok(None);
                };
                let shape =
                    broadcast(a.shape.clone(), b.shape.clone()).ok_or(Error::ShapeMismatch {
                        expected: format!("{} broadcastable with {}", desc_str(a), desc_str(b)),
                        actual: format!("{} vs {}", fmt_shape(&a.shape), fmt_shape(&b.shape)),
                    })?;
                Ok(Some(TensorDesc::new(shape, widen(a.dtype, b.dtype))))
            }

            // Batched matrix multiply over the last two dimensions.
            Self::MatMul => {
                let (Some(a), Some(b)) = (get(0), get(1)) else {
                    return Ok(None);
                };
                matmul_desc(a, b).map(Some)
            }

            // Norms: (x, weight[, bias]) — affine params span the last axis.
            Self::RmsNorm { .. } => {
                let (Some(x), Some(w)) = (get(0), get(1)) else {
                    return Ok(None);
                };
                check_last_dim_matches(x, w)?;
                Ok(Some(x.clone()))
            }
            Self::LayerNorm { .. } => {
                let (Some(x), Some(w)) = (get(0), get(1)) else {
                    return Ok(None);
                };
                check_last_dim_matches(x, w)?;
                if let Some(bias) = get(2) {
                    check_last_dim_matches(x, bias)?;
                }
                Ok(Some(x.clone()))
            }

            // Attention: q/k/v last dims are dictated by head geometry.
            Self::Attention { attrs } => attention_desc(attrs, get(0), get(1), get(2)).map(Some),

            // RoPE table conventions vary by lowerer; only x is propagated.
            Self::Rope { .. } => passthrough(),

            // Token lookup over (weight [vocab, dim], ids): ids must be
            // integers; the output appends the embedding dim to ids' shape.
            Self::Embedding => {
                let (Some(w), Some(ids)) = (get(0), get(1)) else {
                    return Ok(None);
                };
                if !ids.dtype.is_integer() {
                    return Err(Error::InvalidArgument(format!(
                        "embedding indices must be an integer dtype, got `{}`",
                        ids.dtype
                    )));
                }
                let mut shape = ids.shape.clone();
                match w.shape.last().copied() {
                    Some(dim) => shape.push(dim),
                    None => return Ok(None), // rank < 1 weight: unknown out dim
                }
                Ok(Some(TensorDesc::new(shape, w.dtype)))
            }

            Self::Transpose { attrs } => {
                let Some(d) = get(0) else {
                    return Ok(None);
                };
                transpose_desc(d, attrs).map(Some)
            }

            Self::Reshape { attrs } => {
                let Some(d) = get(0) else {
                    return Ok(None);
                };
                reshape_desc(d, &attrs.shape).map(Some)
            }

            Self::Concat { axis } => concat_desc(*axis, inputs),
        }
    }
}

/// True when `d` is a concrete (non-dynamic) dimension; `0` marks unknown.
fn known(d: usize) -> bool {
    d != 0
}

/// Format a shape for error messages; unknown dims render as `?`.
fn fmt_shape(shape: &[usize]) -> String {
    let parts: Vec<String> = shape
        .iter()
        .map(|d| {
            if known(*d) {
                d.to_string()
            } else {
                "?".to_owned()
            }
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

/// Same, including the dtype.
fn desc_str(d: &TensorDesc) -> String {
    format!("{} {}", fmt_shape(&d.shape), d.dtype)
}

/// NumPy-style right-aligned broadcasting; `None` on definitive mismatch.
/// Unknown (`0`) dims propagate as unknown unless the other side pins them.
fn broadcast(a: Vec<usize>, b: Vec<usize>) -> Option<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // Right-aligned indexing; a rank-deficient side falls back to the
        // implicit leading-1 dimension.
        let da = (a.len() + i)
            .checked_sub(n)
            .and_then(|idx| a.get(idx))
            .copied()
            .unwrap_or(1);
        let db = (b.len() + i)
            .checked_sub(n)
            .and_then(|idx| b.get(idx))
            .copied()
            .unwrap_or(1);
        let dim = match (da, db) {
            (_, 1) => da,
            (1, _) => db,
            (x, y) if x == y => x,
            // One side dynamic, the other static-but-not-1: the result stays
            // dynamic (the dynamic side may resolve to anything).
            (0, _) | (_, 0) => 0,
            _ => return None,
        };
        out.push(dim);
    }
    Some(out)
}

/// Result dtype for a pair of numeric operands (MatMul / elementwise bins).
///
/// Block quants act as compressed storage for the other operand's dtype;
/// floats widen; equal dtypes pass through untouched.
fn widen(a: DType, b: DType) -> DType {
    if a == b {
        return a;
    }
    match (a.is_quantized(), b.is_quantized()) {
        (true, false) => return b,
        (false, true) => return a,
        (true, true) => return DType::F32,
        (false, false) => {}
    }
    match (a.is_float(), b.is_float()) {
        (true, true) => {
            let rank = |d: DType| match d {
                DType::F64 => 3,
                DType::F32 => 2,
                _ => 1, // f16 / bf16
            };
            match (rank(a), rank(b)) {
                (ra, rb) if ra > rb => a,
                (ra, rb) if rb > ra => b,
                _ => DType::F32, // f16 vs bf16 and friends: promote
            }
        }
        (true, false) => a,
        (false, true) => b,
        (false, false) => {
            let width = |d: DType| d.item_size().unwrap_or(1);
            match width(a).cmp(&width(b)) {
                core::cmp::Ordering::Greater => a,
                core::cmp::Ordering::Less => b,
                core::cmp::Ordering::Equal => DType::F32, // same-width sign mix
            }
        }
    }
}

/// MatMul output descriptor: batched over leading dims, inner dims checked.
fn matmul_desc(a: &TensorDesc, b: &TensorDesc) -> Result<TensorDesc> {
    const RANK_TWO: &str = "matmul operands need rank >= 2";
    if a.shape.len() < 2 || b.shape.len() < 2 {
        return Err(Error::ShapeMismatch {
            expected: RANK_TWO.into(),
            actual: format!("{} vs {}", desc_str(a), desc_str(b)),
        });
    }
    let ak = a.shape[a.shape.len() - 1];
    let bk = b.shape[b.shape.len() - 2];
    if known(ak) && known(bk) && ak != bk {
        return Err(Error::ShapeMismatch {
            expected: format!("matmul inner dimensions to agree ({ak} == {bk})"),
            actual: format!("{} · {}", desc_str(a), desc_str(b)),
        });
    }
    let abatch = &a.shape[..a.shape.len() - 2];
    let bbatch = &b.shape[..b.shape.len() - 2];
    let batch = broadcast(abatch.to_vec(), bbatch.to_vec()).ok_or(Error::ShapeMismatch {
        expected: format!("matmul batch dims {} to broadcast", fmt_shape(abatch)),
        actual: format!("against {}", fmt_shape(bbatch)),
    })?;
    let mut shape = batch;
    shape.push(a.shape[a.shape.len() - 2]);
    shape.push(b.shape[b.shape.len() - 1]);
    Ok(TensorDesc::new(shape, widen(a.dtype, b.dtype)))
}

/// Affine-parameter check shared by RmsNorm/LayerNorm.
fn check_last_dim_matches(x: &TensorDesc, param: &TensorDesc) -> Result<()> {
    match (x.shape.last().copied(), param.shape.last().copied()) {
        (Some(lx), Some(lw)) if known(lx) && known(lw) && lx != lw => Err(Error::ShapeMismatch {
            expected: format!("norm parameter spanning the last axis ({lx})"),
            actual: format!("parameter shape {}", fmt_shape(&param.shape)),
        }),
        _ => Ok(()),
    }
}

/// Assert `d`'s last dimension equals `expected` (skipped when unknown).
fn expect_last(d: &TensorDesc, expected: usize, what: &str) -> Result<()> {
    match d.shape.last().copied() {
        Some(last) if known(last) && last != expected => Err(Error::ShapeMismatch {
            expected: format!("{what} = {expected}"),
            actual: format!("shape {}", fmt_shape(&d.shape)),
        }),
        _ => Ok(()),
    }
}

/// Attention output from (q, k, v): last dims follow the head geometry.
///
/// The optional mask input is intentionally unchecked — its broadcast shape
/// conventions differ between lowerers. Degenerate attributes carry no usable
/// geometry and pass `q` through unchanged.
fn attention_desc(
    attrs: &AttentionAttrs,
    q: Option<&TensorDesc>,
    k: Option<&TensorDesc>,
    v: Option<&TensorDesc>,
) -> Result<TensorDesc> {
    let (Some(q), Some(k), Some(v)) = (q, k, v) else {
        return Err(Error::InvalidArgument(
            "attention requires q/k/v descriptors".into(),
        ));
    };
    let q_heads = attrs.num_heads as usize;
    let kv_heads = attrs.num_kv_heads as usize;
    let hd = attrs.head_dim as usize;
    if q_heads == 0 || kv_heads == 0 || hd == 0 {
        return Ok(q.clone());
    }
    // Unknown-rank inputs propagate unchanged; the head arithmetic below
    // would panic slicing an empty shape.
    if q.shape.is_empty() {
        return Ok(q.clone());
    }
    expect_last(q, q_heads * hd, "q heads*head_dim")?;
    expect_last(k, kv_heads * hd, "kv heads*head_dim")?;
    expect_last(v, kv_heads * hd, "kv heads*head_dim")?;
    let mut shape = q.shape[..q.shape.len() - 1].to_vec();
    shape.push(q_heads * hd);
    Ok(TensorDesc::new(shape, q.dtype))
}

/// Transpose output descriptor; `perm` must be a permutation of the rank.
fn transpose_desc(d: &TensorDesc, attrs: &TransposeAttrs) -> Result<TensorDesc> {
    let rank = d.shape.len();
    let mut seen = vec![false; rank];
    let perm_ok = attrs.perm.len() == rank
        && attrs.perm.iter().all(|&p| {
            if p >= rank || seen[p] {
                return false;
            }
            seen[p] = true;
            true
        });
    if !perm_ok {
        return Err(Error::InvalidArgument(format!(
            "transpose perm {:?} is not a permutation of rank {rank}",
            attrs.perm
        )));
    }
    let shape: Vec<usize> = attrs.perm.iter().map(|&p| d.shape[p]).collect();
    Ok(TensorDesc::new(shape, d.dtype))
}

/// Reshape output descriptor.
///
/// Target semantics: `-1` infers one dimension from the element count, `0`
/// copies the corresponding input dimension (ONNX convention), positive
/// values are exact. The element count is preserved either way; checks that
/// would touch unknown (`0`) input dims are skipped.
fn reshape_desc(input: &TensorDesc, target: &[i64]) -> Result<TensorDesc> {
    if target.iter().filter(|&&d| d < 0).count() > 1 {
        return Err(Error::InvalidArgument(
            "reshape allows at most one -1 dimension".into(),
        ));
    }
    let in_elems = input.num_elements();
    let any_unknown_in = input.shape.iter().any(|&d| !known(d));

    let mut resolved: Vec<usize> = target
        .iter()
        .enumerate()
        .map(|(i, &t)| match t {
            0 => input.shape.get(i).copied().unwrap_or(0),
            t if t < 0 => 0, // inferred below / unknown
            t => t as usize,
        })
        .collect();

    if let Some(infer_idx) = target.iter().position(|&t| t < 0) {
        // One inferred slot: with fully-known inputs the positive product
        // must divide the element count exactly, and the slot takes the
        // quotient. With unknown input dims the slot stays unknown.
        if !any_unknown_in {
            // Divisor: every already-resolved dim, including -copied ones.
            let known_prod: usize = resolved
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != infer_idx)
                .map(|(_, d)| *d)
                .product();
            if known_prod == 0 || in_elems % known_prod != 0 {
                return Err(Error::ShapeMismatch {
                    expected: format!("reshape known dims to divide {in_elems} elements"),
                    actual: format!("known dims multiply to {known_prod}"),
                });
            }
            resolved[infer_idx] = in_elems / known_prod;
        }
        return Ok(TensorDesc::new(resolved, input.dtype));
    }

    if !any_unknown_in {
        let out_elems: usize = resolved.iter().product();
        if out_elems != in_elems {
            return Err(Error::ShapeMismatch {
                expected: format!("reshape preserving {in_elems} elements"),
                actual: format!("target {} holds {out_elems}", fmt_shape(&resolved)),
            });
        }
    }
    Ok(TensorDesc::new(resolved, input.dtype))
}

/// Concat output descriptor: same rank everywhere, non-axis dims agree,
/// axis dim sums (unknown if any contributor is dynamic).
fn concat_desc(axis: usize, inputs: &[Option<TensorDesc>]) -> Result<Option<TensorDesc>> {
    let descs: Vec<&TensorDesc> = inputs.iter().flatten().collect();
    let Some(first) = descs.first().copied() else {
        return Ok(None);
    };
    let rank = first.shape.len();
    if axis >= rank {
        return Err(Error::InvalidArgument(format!(
            "concat axis {axis} is out of range for rank {rank}"
        )));
    }
    for d in &descs[1..] {
        if d.shape.len() != rank {
            return Err(Error::ShapeMismatch {
                expected: format!("concat inputs of rank {rank}"),
                actual: format!("rank {}", d.shape.len()),
            });
        }
        for (i, (a, b)) in first.shape.iter().zip(d.shape.iter()).enumerate() {
            if i != axis && known(*a) && known(*b) && a != b {
                return Err(Error::ShapeMismatch {
                    expected: format!("concat non-axis dims to agree ({a})"),
                    actual: format!("dim {i}: {a} vs {b}"),
                });
            }
        }
    }
    let mut shape = first.shape.clone();
    let all_known_axis = descs.iter().all(|d| known(d.shape[axis]));
    shape[axis] = if all_known_axis {
        descs.iter().map(|d| d.shape[axis]).sum()
    } else {
        0
    };
    Ok(Some(TensorDesc::new(shape, first.dtype)))
}

#[cfg(test)]
mod shape_tests {
    use super::*;
    use crate::ops::{AttentionAttrs, ReshapeAttrs, SoftmaxAttrs, TransposeAttrs};

    fn desc(shape: &[usize], dtype: DType) -> Option<TensorDesc> {
        Some(TensorDesc::new(shape.to_vec(), dtype))
    }

    #[test]
    fn matmul_shapes_and_dtypes() {
        let mm = Op::MatMul;
        // Classic 2-D: [1,8] x [8,8] f32 -> [1,8] f32.
        let out = mm
            .infer_output_desc(&[desc(&[1, 8], DType::F32), desc(&[8, 8], DType::F32)])
            .unwrap()
            .unwrap();
        assert_eq!(out.shape, vec![1, 8]);
        assert_eq!(out.dtype, DType::F32);

        // Batched: [2,3,8] x [8,4] -> [2,3,4]; f16 x f32 widens.
        let out = mm
            .infer_output_desc(&[desc(&[2, 3, 8], DType::F16), desc(&[8, 4], DType::F32)])
            .unwrap()
            .unwrap();
        assert_eq!(out.shape, vec![2, 3, 4]);
        assert_eq!(out.dtype, DType::F32);

        // Quantized weights dequantize to the activation dtype.
        let out = mm
            .infer_output_desc(&[desc(&[1, 8], DType::F32), desc(&[8, 8], DType::Q8_0)])
            .unwrap()
            .unwrap();
        assert_eq!(out.dtype, DType::F32);

        // Inner-dim mismatch is definitive.
        assert!(mm
            .infer_output_desc(&[desc(&[1, 8], DType::F32), desc(&[4, 8], DType::F32)])
            .is_err());

        // Rank < 2 is rejected.
        assert!(mm
            .infer_output_desc(&[desc(&[8], DType::F32), desc(&[8], DType::F32)])
            .is_err());
    }

    #[test]
    fn broadcast_rules() {
        let add = Op::Add;
        let out = add
            .infer_output_desc(&[desc(&[2, 1, 8], DType::F32), desc(&[4, 8], DType::F32)])
            .unwrap()
            .unwrap();
        assert_eq!(out.shape, vec![2, 4, 8]);

        // Bias vector against a matrix (the Gemm beta path).
        let out = add
            .infer_output_desc(&[desc(&[16, 8], DType::F32), desc(&[8], DType::F32)])
            .unwrap()
            .unwrap();
        assert_eq!(out.shape, vec![16, 8]);

        // Definitive mismatch.
        assert!(add
            .infer_output_desc(&[desc(&[8], DType::F32), desc(&[3], DType::F32)])
            .is_err());

        // Dynamic batch propagates as unknown (0).
        let out = add
            .infer_output_desc(&[desc(&[0, 8], DType::F32), desc(&[0, 8], DType::F32)])
            .unwrap()
            .unwrap();
        assert_eq!(out.shape, vec![0, 8]);
    }

    #[test]
    fn transpose_validates_perm() {
        let t = Op::Transpose {
            attrs: TransposeAttrs { perm: vec![1, 0] },
        };
        let out = t
            .infer_output_desc(&[desc(&[2, 5], DType::F32)])
            .unwrap()
            .unwrap();
        assert_eq!(out.shape, vec![5, 2]);

        let bad = Op::Transpose {
            attrs: TransposeAttrs { perm: vec![2, 0] },
        };
        assert!(bad.infer_output_desc(&[desc(&[2, 5], DType::F32)]).is_err());

        let dup = Op::Transpose {
            attrs: TransposeAttrs { perm: vec![0, 0] },
        };
        assert!(dup.infer_output_desc(&[desc(&[2, 5], DType::F32)]).is_err());
    }

    #[test]
    fn reshape_element_counts() {
        let mk = |shape: Vec<i64>| Op::Reshape {
            attrs: ReshapeAttrs { shape },
        };
        // Exact.
        let out = mk(vec![4, 4])
            .infer_output_desc(&[desc(&[16], DType::F32)])
            .unwrap()
            .unwrap();
        assert_eq!(out.shape, vec![4, 4]);

        // One -1 infers; divisibility enforced.
        let out = mk(vec![-1, 8])
            .infer_output_desc(&[desc(&[2, 8], DType::F32)])
            .unwrap()
            .unwrap();
        assert_eq!(out.shape, vec![2, 8]);

        assert!(mk(vec![-1, 7])
            .infer_output_desc(&[desc(&[12], DType::F32)])
            .is_err());

        // ONNX zero-copies the input dim at the same index.
        let out = mk(vec![0, -1])
            .infer_output_desc(&[desc(&[2, 8], DType::F32)])
            .unwrap()
            .unwrap();
        assert_eq!(out.shape, vec![2, 8]);

        // Two -1s rejected outright.
        assert!(mk(vec![-1, -1])
            .infer_output_desc(&[desc(&[16], DType::F32)])
            .is_err());
    }

    #[test]
    fn norm_and_embedding_checks() {
        let ln = Op::LayerNorm {
            attrs: crate::ops::LayerNormAttrs { eps: 1e-5 },
        };
        assert!(ln
            .infer_output_desc(&[
                desc(&[1, 64], DType::F32),
                desc(&[64], DType::F32),
                desc(&[64], DType::F32),
            ])
            .unwrap()
            .is_some());
        assert!(ln
            .infer_output_desc(&[
                desc(&[1, 64], DType::F32),
                desc(&[32], DType::F32),
                desc(&[32], DType::F32),
            ])
            .is_err());

        let emb = Op::Embedding;
        let out = emb
            .infer_output_desc(&[desc(&[32000, 64], DType::F16), desc(&[1, 8], DType::I32)])
            .unwrap()
            .unwrap();
        assert_eq!(out.shape, vec![1, 8, 64]);

        // Non-integer ids are semantically wrong.
        assert!(emb
            .infer_output_desc(&[desc(&[32000, 64], DType::F16), desc(&[1, 8], DType::F32),])
            .is_err());
    }

    #[test]
    fn attention_geometry() {
        let attn = Op::Attention {
            attrs: AttentionAttrs {
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 32,
                causal: true,
                scale: None,
            },
        };
        let q = desc(&[1, 512, 128], DType::F16); // 4 heads * 32
        let k = desc(&[1, 512, 64], DType::F16); // 2 kv * 32
        let v = desc(&[1, 512, 64], DType::F16);
        let out = attn
            .infer_output_desc(&[q.clone(), k.clone(), v.clone()])
            .unwrap()
            .unwrap();
        assert_eq!(out.shape, vec![1, 512, 128]);

        let bad_q = desc(&[1, 512, 96], DType::F16);
        assert!(attn.infer_output_desc(&[bad_q, k, v]).is_err());
    }

    #[test]
    fn concat_rules() {
        let cat = Op::Concat { axis: 1 };
        let a = desc(&[2, 3, 8], DType::F32);
        let b = desc(&[2, 5, 8], DType::F32);
        let c = desc(&[2, 4, 8], DType::F32);
        let out = cat.infer_output_desc(&[a, b, c]).unwrap().unwrap();
        assert_eq!(out.shape, vec![2, 12, 8]);

        // Non-axis dim disagreement.
        let d = desc(&[9, 5, 8], DType::F32);
        assert!(cat
            .infer_output_desc(&[desc(&[2, 3, 8], DType::F32), d])
            .is_err());

        // Axis out of range.
        let deep = Op::Concat { axis: 3 };
        assert!(deep
            .infer_output_desc(&[desc(&[2, 3], DType::F32), desc(&[2, 4], DType::F32)])
            .is_err());
    }

    #[test]
    fn passthrough_ops_keep_shape() {
        let sm = Op::Softmax {
            attrs: SoftmaxAttrs { axis: -1 },
        };
        let out = sm
            .infer_output_desc(&[desc(&[1, 8], DType::F32)])
            .unwrap()
            .unwrap();
        assert_eq!(out.shape, vec![1, 8]);
        assert_eq!(out.dtype, DType::F32);

        // Sources are exact even with no inputs.
        let inp = Op::Input(TensorDesc::new(vec![3, 7], DType::Bf16));
        assert_eq!(
            inp.infer_output_desc(&[]).unwrap().unwrap().shape,
            vec![3, 7]
        );
    }
}
