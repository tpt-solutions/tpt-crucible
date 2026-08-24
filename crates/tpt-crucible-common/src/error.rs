//! Shared error type for the entire TPT Crucible suite.

use thiserror::Error;

/// Result alias used across all TPT Crucible crates.
pub type Result<T> = core::result::Result<T, Error>;

/// The unified error enum for the TPT Crucible suite.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// Filesystem / stdio failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON (de)serialization failure.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Compact binary (de)serialization failure.
    #[error("binary encoding error: {0}")]
    Binary(#[from] bincode::Error),

    /// Invalid UTF-8 where UTF-8 was required.
    #[error("invalid utf-8: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),

    /// A dtype is syntactically recognized but not supported by this pass.
    #[error("unsupported dtype: {0}")]
    UnsupportedDType(String),

    /// An operation cannot be handled by this pass.
    #[error("unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// Planned but not yet built — see `todo.md` roadmap.
    #[error("not implemented: {feature} (tracked in todo.md)")]
    NotImplemented {
        /// Human-readable feature name.
        feature: String,
    },

    /// A model file matched a known format but violated the format contract.
    #[error("failed to parse `{path}` ({format}): {reason}")]
    ParseFormat {
        /// Source path or `<memory>`.
        path: String,
        /// Format name, e.g. `"gguf"`.
        format: String,
        /// What went wrong.
        reason: String,
    },

    /// The file's format could not be determined.
    #[error("unrecognized model format: `{path}`")]
    UnknownFormat {
        /// Source path.
        path: String,
    },

    /// A node id does not exist in the graph.
    #[error("node id {id} out of range (graph has {len} nodes)")]
    NodeOutOfRange {
        /// The offending id.
        id: u32,
        /// Number of nodes in the graph.
        len: usize,
    },

    /// SSA invariant violation: an edge points to a node defined later.
    #[error(
        "forward reference: node {consumer} references node {producer}, \
         but graphs must be topologically ordered"
    )]
    ForwardReference {
        /// The referencing node index.
        consumer: u32,
        /// The referenced (later) node index.
        producer: u32,
    },

    /// Wrong number of inputs for an op.
    #[error("arity mismatch for op `{op}`: expected {expected:?} inputs, got {actual}")]
    ArityMismatch {
        /// Op name.
        op: &'static str,
        /// Inclusive (min, max) input count.
        expected: (usize, usize),
        /// Actual input count.
        actual: usize,
    },

    /// Two nodes share a name; names must be unique within a graph.
    #[error("duplicate node name: `{0}`")]
    DuplicateNodeName(String),

    /// Tensor shape mismatch.
    #[error("shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch {
        /// Expected shape description.
        expected: String,
        /// Actual shape description.
        actual: String,
    },

    /// Buffer length disagrees with the tensor descriptor.
    #[error(
        "tensor size mismatch for `{name}`: descriptor implies \
         {expected} bytes, buffer holds {actual}"
    )]
    TensorSizeMismatch {
        /// Tensor name.
        name: String,
        /// Expected byte size per the descriptor.
        expected: usize,
        /// Actual buffer byte size.
        actual: usize,
    },

    /// Binary IR container lacks the `TPTIR` magic header.
    #[error("invalid binary ir: missing TPTIR magic header")]
    BadMagic,

    /// A cargo feature was compiled out; rebuild with it enabled.
    #[error("feature `{0}` not enabled: rebuild with `--features {0}`")]
    FeatureNotEnabled(String),

    /// Memory planning failed: a target cannot hold its assigned data.
    #[error("out of memory on node `{node}`: needs {needed} bytes, has {available}")]
    OutOfMemory {
        /// Node that overflowed.
        node: String,
        /// Bytes required.
        needed: u64,
        /// Bytes available.
        available: u64,
    },

    /// Generic invalid argument from CLI or API misuse.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}
