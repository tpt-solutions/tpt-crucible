//! The operator set of TPT-IR.

use serde::{Deserialize, Serialize};

use crate::dtype::DType;
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
}
