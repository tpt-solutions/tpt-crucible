//! The TPT-IR computation graph and its (de)serialization.

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::ops::Op;

/// Magic prefix of the compact binary container (`"TPTIR"` + format version).
pub const BINARY_MAGIC: &[u8; 6] = b"TPTIR\x01";

/// Handle to a node inside a [`Graph`]; equals the node's index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Index as `usize` for slice access.
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl core::fmt::Display for NodeId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// One node: an op plus named inputs referencing earlier nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique within the graph.
    pub name: String,
    /// The operation.
    pub op: Op,
    /// Input edges, each pointing at an earlier node (SSA order).
    pub inputs: Vec<NodeId>,
}

/// A TPT-IR computation graph.
///
/// Nodes are stored in topological (SSA) order: every edge points from a
/// later node to an earlier one, so plain iteration is a valid execution
/// order and no explicit edge list is stored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    /// Model/graph name (e.g. file stem or architecture).
    pub name: String,
    /// Topologically ordered nodes.
    pub nodes: Vec<GraphNode>,
    /// Graph-level input nodes (subset of `nodes` holding [`Op::Input`]).
    #[serde(default)]
    pub inputs: Vec<NodeId>,
    /// Graph-level output nodes (subset holding [`Op::Output`]).
    #[serde(default)]
    pub outputs: Vec<NodeId>,
    /// Free-form metadata (source format, model hyperparameters, …).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl Graph {
    /// Create an empty graph with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Append a node; empty names get a stable auto-name (`"<op>_<idx>"`).
    pub fn push(
        &mut self,
        name: impl Into<String>,
        op: Op,
        inputs: impl Into<Vec<NodeId>>,
    ) -> NodeId {
        let idx = self.nodes.len() as u32;
        let mut name = name.into();
        if name.is_empty() {
            name = format!("{}_{}", op.name(), idx);
        }
        self.nodes.push(GraphNode {
            name,
            op,
            inputs: inputs.into(),
        });
        NodeId(idx)
    }

    /// Register `id` as a graph-level input (idempotent).
    pub fn mark_input(&mut self, id: NodeId) {
        if !self.inputs.contains(&id) {
            self.inputs.push(id);
        }
    }

    /// Register `id` as a graph-level output (idempotent).
    pub fn mark_output(&mut self, id: NodeId) {
        if !self.outputs.contains(&id) {
            self.outputs.push(id);
        }
    }

    /// Borrow a node or fail with [`Error::NodeOutOfRange`].
    pub fn node(&self, id: NodeId) -> Result<&GraphNode> {
        self.nodes.get(id.as_usize()).ok_or(Error::NodeOutOfRange {
            id: id.0,
            len: self.nodes.len(),
        })
    }

    /// Non-failing node lookup.
    pub fn get_node(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(id.as_usize())
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// All nodes consuming `id`'s output, in graph order.
    pub fn consumers_of(&self, id: NodeId) -> Vec<NodeId> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.inputs.contains(&id))
            .map(|(i, _)| NodeId(i as u32))
            .collect()
    }

    /// Insert a metadata key/value pair.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Look up a metadata value.
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Histogram of op names, useful for summaries and tests.
    pub fn op_histogram(&self) -> BTreeMap<&'static str, usize> {
        let mut hist = BTreeMap::new();
        for n in &self.nodes {
            *hist.entry(n.op.name()).or_default() += 1;
        }
        hist
    }

    /// Validate structural and type-level invariants:
    ///
    /// * every edge targets an existing, earlier node (SSA / topological order),
    /// * op arities are respected,
    /// * `inputs`/`outputs` reference the right kinds of nodes,
    /// * node names are unique,
    /// * connected ops agree on shapes/dtypes where those are statically
    ///   decidable (a `MatMul` with mismatched inner dimensions fails here).
    pub fn validate(&self) -> Result<()> {
        for (idx, n) in self.nodes.iter().enumerate() {
            let id = idx as u32;
            for &inp in &n.inputs {
                if inp.0 >= id {
                    return Err(Error::ForwardReference {
                        consumer: id,
                        producer: inp.0,
                    });
                }
            }
            let (min, max) = n.op.arity();
            if n.inputs.len() < min || n.inputs.len() > max {
                return Err(Error::ArityMismatch {
                    op: n.op.name(),
                    expected: (min, max),
                    actual: n.inputs.len(),
                });
            }
            match &n.op {
                Op::Input(_) | Op::Constant { .. } if !n.inputs.is_empty() => {
                    return Err(Error::InvalidArgument(format!(
                        "node {id} (`{}`): source ops take no inputs",
                        n.name
                    )));
                }
                _ => {}
            }
        }

        for &id in &self.inputs {
            let n = self.node(id)?;
            if !matches!(n.op, Op::Input(_)) {
                return Err(Error::InvalidArgument(format!(
                    "graph input {} is not an `input` op",
                    id
                )));
            }
        }
        for &id in &self.outputs {
            let n = self.node(id)?;
            if !matches!(n.op, Op::Output { .. }) {
                return Err(Error::InvalidArgument(format!(
                    "graph output {} is not an `output` op",
                    id
                )));
            }
        }

        let mut seen = HashSet::new();
        for n in &self.nodes {
            if !seen.insert(&n.name) {
                return Err(Error::DuplicateNodeName(n.name.clone()));
            }
        }

        // Type/shape consistency: infer each node's output descriptor from
        // its op and its producers' descriptors, rejecting definitively
        // inconsistent wiring. Dimensions stored as 0 mean "unknown/dynamic"
        // (see `Op::infer_output_desc`) and are never compared, and ops with
        // unknown inputs simply propagate the unknown.
        let mut descs: Vec<Option<TensorDesc>> = Vec::with_capacity(self.nodes.len());
        for (idx, n) in self.nodes.iter().enumerate() {
            let ins: Vec<Option<TensorDesc>> = n
                .inputs
                .iter()
                .map(|&i| descs[i.as_usize()].clone())
                .collect();
            let out = n.op.infer_output_desc(&ins).map_err(|e| match e {
                Error::InvalidArgument(msg) => Error::InvalidArgument(format!(
                    "node {idx} (`{}`): {msg}",
                    n.name
                )),
                other => other,
            })?;
            descs.push(out);
        }
        Ok(())
    }

    /// Export as a Graphviz DOT digraph (debug/visualization aid).
    pub fn to_dot(&self) -> String {
        let mut s = String::with_capacity(self.nodes.len() * 48);
        s.push_str("digraph \"");
        s.push_str(&self.name.replace('"', "'"));
        s.push_str("\" {\n  rankdir=\"TB\";\n");
        for (i, n) in self.nodes.iter().enumerate() {
            let label = n.name.replace('"', "'");
            s.push_str(&format!(
                "  n{i} [label=\"n{i}: {label}\\n{}\"];\n",
                n.op.name()
            ));
        }
        for (i, n) in self.nodes.iter().enumerate() {
            for &inp in &n.inputs {
                s.push_str(&format!("  n{} -> n{i};\n", inp.0));
            }
        }
        s.push_str("}\n");
        s
    }

    // ---- serialization: JSON ----

    /// Compact JSON encoding.
    pub fn to_json_string(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Pretty JSON encoding.
    pub fn to_json_string_pretty(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Decode from a JSON string.
    pub fn from_json_str(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    // ---- serialization: binary container ----

    /// Compact binary encoding with the [`BINARY_MAGIC`] header.
    ///
    /// The magic makes artifacts self-identifying and lets [`Graph::load`]
    /// sniff the format of any file.
    pub fn to_binary(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(BINARY_MAGIC);
        out.extend_from_slice(&bincode::serialize(self)?);
        Ok(out)
    }

    /// Decode from bytes produced by [`Graph::to_binary`].
    pub fn from_binary(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < BINARY_MAGIC.len() || &bytes[..BINARY_MAGIC.len()] != BINARY_MAGIC {
            return Err(Error::BadMagic);
        }
        Ok(bincode::deserialize(&bytes[BINARY_MAGIC.len()..])?)
    }

    /// Save to `path`: JSON if the extension is `.json`, else binary.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let path = path.as_ref();
        let is_json = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("json"));
        let bytes = if is_json {
            self.to_json_string_pretty()?.into_bytes()
        } else {
            self.to_binary()?
        };
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load from `path`, sniffing binary vs JSON via the magic header.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        if bytes.len() >= BINARY_MAGIC.len() && &bytes[..BINARY_MAGIC.len()] == BINARY_MAGIC {
            Self::from_binary(&bytes)
        } else {
            match String::from_utf8(bytes) {
                Ok(text) => Self::from_json_str(&text),
                Err(_) => Err(Error::ParseFormat {
                    path: path.display().to_string(),
                    format: "tpt-ir".into(),
                    reason: "file is neither a TPTIR binary nor JSON — is this a raw model file? \
                         run `tpt ingest` on it first"
                        .into(),
                }),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::ops::{AttentionAttrs, ReshapeAttrs, SoftmaxAttrs, TransposeAttrs};
    use crate::tensor::{Tensor, TensorDesc};

    fn scalar_desc() -> TensorDesc {
        TensorDesc::new(vec![], DType::F32)
    }

    /// x → matmul(w) → softmax → output
    fn sample_graph() -> Graph {
        let mut g = Graph::new("sample");
        g.set_metadata("source_format", "test");
        let x = g.push(
            "x",
            Op::Input(TensorDesc::new(vec![1, 8], DType::F32)),
            vec![],
        );
        let w = g.push(
            "",
            Op::Constant {
                tensor: Tensor::zeros(TensorDesc::new(vec![8, 8], DType::F32)),
            },
            vec![],
        );
        let mm = g.push("", Op::MatMul, vec![x, w]);
        let sm = g.push(
            "",
            Op::Softmax {
                attrs: SoftmaxAttrs { axis: -1 },
            },
            vec![mm],
        );
        let out = g.push(
            "logits",
            Op::Output {
                name: "logits".into(),
            },
            vec![sm],
        );
        g.mark_input(x);
        g.mark_output(out);
        g
    }

    #[test]
    fn builder_and_validation() {
        let g = sample_graph();
        assert_eq!(g.len(), 5);
        g.validate().unwrap();
        assert_eq!(g.inputs, vec![NodeId(0)]);
        assert_eq!(g.outputs, vec![NodeId(4)]);
        assert_eq!(g.get_node(NodeId(1)).unwrap().name, "constant_1");
        assert_eq!(g.consumers_of(NodeId(0)), vec![NodeId(2)]);
        let hist = g.op_histogram();
        assert_eq!(hist.get("matmul"), Some(&1));
        assert_eq!(hist.get("softmax"), Some(&1));
        assert_eq!(g.metadata("source_format"), Some("test"));
    }

    #[test]
    fn json_roundtrip_is_lossless() {
        let g = sample_graph();
        let json = g.to_json_string_pretty().unwrap();
        let back = Graph::from_json_str(&json).unwrap();
        assert_eq!(back, g);
        assert!(json.contains("matmul"));
    }

    #[test]
    fn binary_roundtrip_and_magic() {
        let g = sample_graph();
        let bin = g.to_binary().unwrap();
        assert!(bin.starts_with(BINARY_MAGIC));
        assert_eq!(Graph::from_binary(&bin).unwrap(), g);
        assert!(matches!(Graph::from_binary(b"nope"), Err(Error::BadMagic)));
    }

    #[test]
    fn save_load_sniffs_format() {
        let dir = std::env::temp_dir().join("tptir-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let g = sample_graph();

        let json_path = dir.join("g.json.tptir.json");
        g.save(&json_path).unwrap();
        assert_eq!(Graph::load(&json_path).unwrap(), g);

        let bin_path = dir.join("g.bin.tptir");
        g.save(&bin_path).unwrap();
        assert_eq!(Graph::load(&bin_path).unwrap(), g);
    }

    #[test]
    fn forward_reference_rejected() {
        let mut g = Graph::new("bad");
        g.push("a", Op::Add, vec![NodeId(1)]);
        g.push("b", Op::Input(scalar_desc()), vec![]);
        assert!(matches!(
            g.validate(),
            Err(Error::ForwardReference {
                consumer: 0,
                producer: 1
            })
        ));
    }

    #[test]
    fn arity_mismatch_rejected() {
        let mut g = Graph::new("bad2");
        g.push("a", Op::Input(scalar_desc()), vec![]);
        g.push("b", Op::Input(scalar_desc()), vec![]);
        g.push("c", Op::Neg, vec![NodeId(0), NodeId(1)]);
        assert!(matches!(
            g.validate(),
            Err(Error::ArityMismatch {
                op: "neg",
                expected: (1, 1),
                actual: 2
            })
        ));
    }

    #[test]
    fn duplicate_names_rejected() {
        let mut g = Graph::new("bad3");
        g.push("dup", Op::Input(scalar_desc()), vec![]);
        g.push("dup", Op::Relu, vec![NodeId(0)]);
        assert!(matches!(
            g.validate(),
            Err(Error::DuplicateNodeName(name)) if name == "dup"
        ));
    }

    #[test]
    fn attention_arity_allows_optional_mask() {
        let mut g = Graph::new("attn");
        let mk = |i: &mut Graph| {
            i.push(
                format!("t{}", i.len()),
                Op::Input(scalar_desc()),
                Vec::<NodeId>::new(),
            )
        };
        let q = mk(&mut g);
        let k = mk(&mut g);
        let v = mk(&mut g);
        let attrs = AttentionAttrs {
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 64,
            causal: true,
            scale: None,
        };
        g.push("", Op::Attention { attrs }, vec![q, k, v]);
        g.validate().unwrap();
    }

    #[test]
    fn dot_export_mentions_nodes() {
        let dot = sample_graph().to_dot();
        assert!(dot.starts_with("digraph"));
        assert!(dot.contains("n4"));
        assert!(dot.contains("matmul"));
    }

    /// x → matmul(w), with caller-chosen operand shapes/dtypes.
    fn matmul_graph(a: (Vec<usize>, DType), b: (Vec<usize>, DType)) -> Graph {
        let mut g = Graph::new("mm");
        let x = g.push("x", Op::Input(TensorDesc::new(a.0, a.1)), vec![]);
        let w = g.push(
            "w",
            Op::Constant {
                tensor: Tensor::zeros(TensorDesc::new(b.0, b.1)),
            },
            vec![],
        );
        g.push("mm", Op::MatMul, vec![x, w]);
        g
    }

    #[test]
    fn matmul_inner_dim_mismatch_rejected() {
        let g = matmul_graph(
            (vec![1, 8], DType::F32),
            (vec![4, 8], DType::F32), // inner dims 8 vs 4
        );
        assert!(matches!(g.validate(), Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn quantized_weights_validate_like_real_gguf_graphs() {
        // GGUF ingestion emits f32 activations against Q8_0 weight constants;
        // mixed dtypes are first-class, so this must stay valid.
        matmul_graph((vec![1, 8], DType::F32), (vec![8, 8], DType::Q8_0))
            .validate()
            .unwrap();
    }

    #[test]
    fn dynamic_zero_dims_are_never_compared() {
        // TensorFlow maps unknown dims to 0 at ingest; such models must keep
        // validating even where static checks would otherwise apply.
        matmul_graph((vec![0, 8], DType::F32), (vec![8, 8], DType::F32))
            .validate()
            .unwrap();
    }

    #[test]
    fn broadcast_bias_ok_but_conflict_rejected() {
        // Gemm's beta*C bias path: [16,8] + [8] broadcasts fine.
        let mut ok = Graph::new("bias");
        let y = ok.push("y", Op::Input(TensorDesc::new(vec![16, 8], DType::F32)), vec![]);
        let c = ok.push(
            "c",
            Op::Constant {
                tensor: Tensor::from_f32(vec![8], &[0.0; 8]),
            },
            vec![],
        );
        ok.push("add", Op::Add, vec![y, c]);
        ok.validate().unwrap();

        // [8] vs [3] is definitively wrong.
        let mut bad = Graph::new("bad-bias");
        let y = bad.push("y", Op::Input(TensorDesc::new(vec![16, 8], DType::F32)), vec![]);
        let c = bad.push(
            "c",
            Op::Constant {
                tensor: Tensor::from_f32(vec![3], &[0.0; 3]),
            },
            vec![],
        );
        bad.push("add", Op::Add, vec![y, c]);
        assert!(matches!(bad.validate(), Err(Error::ShapeMismatch { .. })));
    }

    #[test]
    fn transpose_bad_perm_rejected_with_context() {
        let mut g = Graph::new("tp");
        let x = g.push("x", Op::Input(TensorDesc::new(vec![2, 5], DType::F32)), vec![]);
        g.push(
            "t",
            Op::Transpose {
                attrs: TransposeAttrs { perm: vec![2, 0] },
            },
            vec![x],
        );
        match g.validate() {
            Err(Error::InvalidArgument(msg)) => {
                assert!(msg.contains("node 1 (`t`)"), "context missing: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn reshape_count_mismatch_rejected_but_divisible_ok() {
        let mut bad = Graph::new("rs");
        let x = bad.push("x", Op::Input(TensorDesc::new(vec![12], DType::F32)), vec![]);
        bad.push(
            "r",
            Op::Reshape {
                attrs: ReshapeAttrs {
                    shape: vec![-1, 7],
                },
            },
            vec![x],
        );
        assert!(bad.validate().is_err());

        let mut ok = Graph::new("rs-ok");
        let x = ok.push("x", Op::Input(TensorDesc::new(vec![14], DType::F32)), vec![]);
        ok.push(
            "r",
            Op::Reshape {
                attrs: ReshapeAttrs {
                    shape: vec![-1, 7],
                },
            },
            vec![x],
        );
        ok.validate().unwrap();
    }

    #[test]
    fn embedding_non_integer_ids_rejected() {
        let mut g = Graph::new("emb");
        let w = g.push(
            "w",
            Op::Constant {
                tensor: Tensor::from_f32(vec![64, 8], &vec![0.0; 512]),
            },
            vec![],
        );
        let ids = g.push(
            "ids",
            Op::Input(TensorDesc::new(vec![1, 2], DType::F32)),
            vec![],
        );
        g.push("e", Op::Embedding, vec![w, ids]);
        assert!(g.validate().is_err());
    }

    #[test]
    fn attention_head_geometry_mismatch_rejected() {
        let mk = |q_last: usize| {
            let mut g = Graph::new("attn");
            let attrs = AttentionAttrs {
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 32,
                causal: true,
                scale: None,
            };
            let q = g.push(
                "q",
                Op::Input(TensorDesc::new(vec![1, 512, q_last], DType::F16)),
                vec![],
            );
            let k = g.push(
                "k",
                Op::Input(TensorDesc::new(vec![1, 512, 64], DType::F16)),
                vec![],
            );
            let v = g.push(
                "v",
                Op::Input(TensorDesc::new(vec![1, 512, 64], DType::F16)),
                vec![],
            );
            g.push("", Op::Attention { attrs }, vec![q, k, v]);
            g
        };
        mk(128).validate().unwrap(); // 4 heads * 32
        assert!(mk(96).validate().is_err());
    }
}
