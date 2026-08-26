//! Streaming pre-flight: operator compatibility analysis
//! (`spec2.txt` §3.1 "Key Feature").
//!
//! Walks the TPT-IR graph in SSA order and classifies every op against each
//! target hardware family, emitting one [`PreflightEvent`] *as the node is
//! visited* — callers stream the events wherever they like, most notably to
//! the Observer dashboard over WebSockets (see `tpt preflight --serve`,
//! which bridges into the Observer's telemetry server).
//!
//! The compatibility matrix is a static capability table (v1): it encodes
//! which operator classes each backend can execute natively, emulate, or
//! cannot run at all today. It is deliberately honest about gaps — an
//! unsupported verdict is a planning blocker surfaced *before* firmware is
//! flashed, not a runtime surprise.
//!
//! ```no_run
//! use tpt_crucible_catalyst::ingest_path;
//! use tpt_crucible_catalyst::preflight::{self, Family};
//!
//! let graph = ingest_path("models/tinyllama.gguf")?;
//! let report = preflight::stream(&graph, &[Family::Alloy], &mut |event| {
//!     println!("{}", event.to_json_line().unwrap());
//! });
//! if !report.unsupported_ops.is_empty() {
//!     eprintln!("blockers: {:?}", report.unsupported_ops);
//! }
//! # Ok::<(), tpt_crucible_common::Error>(())
//! ```

use serde::{Deserialize, Serialize};
use tpt_crucible_common::error::Result;
use tpt_crucible_common::{Graph, Op};

/// Target hardware family for compatibility analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    /// Microcontroller swarm (ESP32/RP2040/RISC-V) — the Alloy backend.
    Alloy,
    /// FPGA + HBM overlay — the Fusion backend.
    Fusion,
    /// Analog compute-in-memory — the Element backend.
    Element,
}

impl Family {
    /// Canonical lowercase name (also the serde representation).
    pub fn name(self) -> &'static str {
        match self {
            Self::Alloy => "alloy",
            Self::Fusion => "fusion",
            Self::Element => "element",
        }
    }

    /// Parse a family from its canonical name.
    ///
    /// # Errors
    /// [`tpt_crucible_common::Error::InvalidArgument`] on unknown names.
    pub fn parse(name: &str) -> tpt_crucible_common::Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "alloy" => Ok(Self::Alloy),
            "fusion" => Ok(Self::Fusion),
            "element" => Ok(Self::Element),
            other => Err(tpt_crucible_common::Error::InvalidArgument(format!(
                "unknown hardware family `{other}` (expected alloy|fusion|element)"
            ))),
        }
    }
}

/// How well one op runs on one family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Executes natively at full speed.
    Supported,
    /// Executes with an approximation or software fallback.
    Emulated,
    /// No implementation; a hard blocker for this family.
    Unsupported,
}

/// One streamed classification result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreflightEvent {
    /// Stream position across the whole run (0-based, per analysis).
    pub seq: u64,
    /// Classified IR node.
    pub node_id: u32,
    /// IR node name (auto-names included).
    pub node_name: String,
    /// Canonical op name (`"matmul"`, `"silu"`, …).
    pub op: String,
    /// Hardware family the verdict applies to
    /// (`"alloy"`/`"fusion"`/`"element"`).
    pub family: String,
    /// The verdict.
    pub verdict: Verdict,
    /// Human explanation, present for emulated/unsupported verdicts.
    pub note: Option<String>,
}

impl PreflightEvent {
    /// One JSON line (the wire format streamed to dashboards).
    ///
    /// # Errors
    /// Serialization is infallible for this shape; the error exists for
    /// forward compatibility.
    pub fn to_json_line(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }
}

/// Verdict plus the static note explaining it.
type Detail = (Verdict, Option<&'static str>);

/// Convenience constructor for a [`Detail`].
const fn d(verdict: Verdict, note: Option<&'static str>) -> Detail {
    (verdict, note)
}

use Verdict::{Emulated as E, Supported as S, Unsupported as U};

/// Classify one op against one family using the static v1 capability table.
#[must_use]
pub fn classify(op: &Op, family: Family) -> Verdict {
    classify_detail(op, family).0
}

fn classify_detail(op: &Op, family: Family) -> Detail {
    match family {
        // The swarm executes the whole IR in generated firmware; nothing is
        // blocked, wide integer/float types just cost cycles.
        Family::Alloy => d(S, None),
        Family::Fusion => match op {
            Op::Input(_) | Op::Constant { .. } | Op::Output { .. } => d(S, None),
            Op::MatMul => d(S, Some("overlay MAC array")),
            Op::Attention { .. } => d(S, Some("overlay attention datapath")),
            Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::Neg
            | Op::Relu
            | Op::Sigmoid
            | Op::Cast { .. }
            | Op::Transpose { .. }
            | Op::Reshape { .. }
            | Op::Concat { .. } => d(S, None),
            Op::Exp
            | Op::Log
            | Op::Sqrt
            | Op::Sin
            | Op::Cos
            | Op::Tanh
            | Op::Erf
            | Op::Gelu
            | Op::Silu => d(E, Some("LUT/piecewise approximation in fabric")),
            Op::Softmax { .. } => d(E, Some("approximate exp + online normalization")),
            Op::RmsNorm { .. } | Op::LayerNorm { .. } | Op::Rope { .. } => {
                d(E, Some("reduced-precision accumulation"))
            }
            Op::Embedding => d(U, Some("random-access weight reads not overlay-friendly")),
        },
        Family::Element => match op {
            Op::Input(_) | Op::Constant { .. } | Op::Output { .. } => d(S, None),
            Op::MatMul => d(S, Some("crossbar MAC array")),
            Op::Add | Op::Sub | Op::Mul | Op::Div => d(S, Some("op-amp summing junctions")),
            Op::Relu | Op::Sigmoid | Op::Tanh => d(E, Some("diode/translinear nonlinear circuit")),
            Op::Exp | Op::Log => d(E, Some("translinear loop")),
            _ => d(U, Some("no compact analog primitive; needs digital glue")),
        },
    }
}

/// Aggregate result of a streaming run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Report {
    /// Events emitted (nodes × families).
    pub checked: u64,
    /// Verdicts that were [`Verdict::Supported`].
    pub supported: u64,
    /// Verdicts that were [`Verdict::Emulated`].
    pub emulated: u64,
    /// Verdicts that were [`Verdict::Unsupported`].
    pub unsupported: u64,
    /// Unique op names with an unsupported verdict anywhere in the run,
    /// first-seen order.
    pub unsupported_ops: Vec<String>,
}

impl Report {
    /// True when nothing blocks the run for any analyzed family.
    pub fn has_blockers(&self) -> bool {
        self.unsupported > 0
    }
}

/// Walk `graph` and stream one [`PreflightEvent`] per (node, family) pair to
/// `sink`, in SSA node order. The sink is invoked *as each node is visited* —
/// hand it a closure that forwards to a WebSocket/queue to get real-time
/// pre-flight telemetry.
#[must_use]
pub fn stream(graph: &Graph, families: &[Family], sink: &mut dyn FnMut(PreflightEvent)) -> Report {
    let mut report = Report::default();
    let mut seq = 0u64;
    for (i, node) in graph.nodes.iter().enumerate() {
        for &family in families {
            let (verdict, note) = classify_detail(&node.op, family);
            report.checked += 1;
            match verdict {
                Verdict::Supported => report.supported += 1,
                Verdict::Emulated => report.emulated += 1,
                Verdict::Unsupported => {
                    report.unsupported += 1;
                    let op_name = node.op.name();
                    if !report.unsupported_ops.iter().any(|o| o == op_name) {
                        report.unsupported_ops.push(op_name.to_owned());
                    }
                }
            }
            sink(PreflightEvent {
                seq,
                node_id: i as u32,
                node_name: node.name.clone(),
                op: node.op.name().to_owned(),
                family: family.name().to_owned(),
                verdict,
                note: note.map(str::to_owned),
            });
            seq += 1;
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_crucible_common::NodeId;
    use tpt_crucible_common::{DType, Tensor, TensorDesc};

    fn tiny_graph() -> Graph {
        let mut g = Graph::new("tiny");
        let x = g.push(
            "x",
            Op::Input(TensorDesc::new(vec![1, 4], DType::F32)),
            Vec::<NodeId>::new(),
        );
        let w = g.push(
            "w",
            Op::Constant {
                tensor: Tensor::from_f32(vec![4, 4], &[0.5; 16]),
            },
            Vec::<NodeId>::new(),
        );
        let y = g.push("", Op::MatMul, vec![x, w]);
        let a = g.push("", Op::Silu, vec![y]);
        let o = g.push("", Op::Output { name: "a".into() }, vec![a]);
        g.mark_output(o);
        g
    }

    #[test]
    fn events_stream_in_traversal_order() {
        let g = tiny_graph();
        let mut seen = Vec::new();
        let report = stream(&g, &[Family::Alloy], &mut |e| seen.push(e.seq));
        // One event per node (6 nodes), sequential sequence numbers.
        assert_eq!(seen.len(), g.len());
        assert_eq!(seen, (0..g.len() as u64).collect::<Vec<_>>());
        assert_eq!(report.checked, g.len() as u64);
        assert_eq!(report.supported, g.len() as u64);
        assert!(!report.has_blockers());
    }

    #[test]
    fn element_blocks_digital_glue_ops() {
        let g = tiny_graph();
        let mut events = Vec::new();
        let report = stream(&g, &[Family::Element], &mut |e| events.push(e));
        assert!(report.has_blockers());
        assert!(report.unsupported_ops.contains(&"silu".to_string()));
        // MatMul is the analog sweet spot.
        let mm = events.iter().find(|e| e.op == "matmul").unwrap();
        assert_eq!(mm.verdict, Verdict::Supported);
        assert_eq!(mm.note.as_deref(), Some("crossbar MAC array"));
        let silu = events.iter().find(|e| e.op == "silu").unwrap();
        assert_eq!(silu.verdict, Verdict::Unsupported);
        assert_eq!(
            silu.note.as_deref(),
            Some("no compact analog primitive; needs digital glue")
        );
        assert_eq!(
            report.unsupported + report.emulated + report.supported,
            report.checked
        );
    }

    #[test]
    fn fusion_emulates_transcendentals() {
        let g = tiny_graph();
        let mut events = Vec::new();
        let report = stream(&g, &[Family::Fusion], &mut |e| events.push(e));
        let silu = events.iter().find(|e| e.op == "silu").unwrap();
        assert_eq!(silu.verdict, Verdict::Emulated);
        assert!(!report.has_blockers());
    }

    #[test]
    fn json_line_roundtrip() {
        let ev = PreflightEvent {
            seq: 3,
            node_id: 7,
            node_name: "blk.0.attn_q".into(),
            op: "matmul".into(),
            family: "fusion".into(),
            verdict: Verdict::Supported,
            note: Some("overlay MAC array".into()),
        };
        let line = ev.to_json_line().unwrap();
        let back: PreflightEvent = serde_json::from_str(&line).unwrap();
        assert_eq!(back, ev);
        assert!(line.starts_with("{\"seq\":3"));
    }

    #[test]
    fn family_parse_rejects_unknown() {
        assert!(Family::parse("alloy").is_ok());
        assert!(Family::parse("ESP32").is_err());
    }
}
