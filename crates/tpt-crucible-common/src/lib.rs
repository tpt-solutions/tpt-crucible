//! # tpt-crucible-common
//!
//! Shared **TPT-IR** (TPT Intermediate Representation) types and error handling,
//! used by every other crate in the TPT Crucible workspace.
//!
//! TPT-IR is a pure, hardware-agnostic mathematical graph: models ingested by
//! [`tpt-crucible-catalyst`] are lowered into a [`Graph`] of strongly-typed
//! [`Op`] nodes over [`Tensor`]s with explicit [`DType`]s. Hardware backends
//! (`alloy`, `fusion`, `element`) consume the same IR.
//!
//! ## Serialization
//!
//! Graphs serialize to:
//!
//! * **JSON** — human-readable interchange (`.tptir.json`), via
//!   [`Graph::to_json_string`] / [`Graph::from_json_str`].
//! * **Binary** — compact container with a `"TPTIR"` magic header
//!   (`.tptir`), via [`Graph::to_binary`] / [`Graph::from_binary`].
//!
//! [`Graph::save`] / [`Graph::load`] pick the right format automatically from
//! the file contents.
//!
//! ## Example
//!
//! ```
//! use tpt_crucible_common::{DType, Graph, Op, Tensor, TensorDesc};
//!
//! let mut g = Graph::new("tiny-mlp");
//! let x = g.push("x", Op::Input(TensorDesc::new(vec![1, 8], DType::F32)), vec![]);
//! let w = g.push("w", Op::Constant { tensor: Tensor::from_f32(vec![8, 8], &vec![0.0; 64]) }, vec![]);
//! let y = g.push("y", Op::MatMul, vec![x, w]);
//! let out = g.push("out", Op::Output { name: "logits".into() }, vec![y]);
//! g.mark_input(x);
//! g.mark_output(out);
//! g.validate().expect("graph is well-formed");
//!
//! let json = g.to_json_string().unwrap();
//! assert_eq!(Graph::from_json_str(&json).unwrap(), g);
//! ```
//!
//! [`tpt-crucible-catalyst`]: https://crates.io/crates/tpt-crucible-catalyst

pub mod dtype;
pub mod error;
pub mod graph;
pub mod ops;
pub mod tensor;

pub use dtype::DType;
pub use error::{Error, Result};
pub use graph::{Graph, GraphNode, NodeId, BINARY_MAGIC};
pub use ops::Op;
pub use tensor::{Tensor, TensorData, TensorDesc};

/// Frequently-used imports for downstream crates.
pub mod prelude {
    pub use crate::dtype::DType;
    pub use crate::error::{Error, Result};
    pub use crate::graph::{Graph, NodeId};
    pub use crate::ops::Op;
    pub use crate::tensor::{Tensor, TensorDesc};
}
