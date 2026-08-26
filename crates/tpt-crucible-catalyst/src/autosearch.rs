//! Quantization auto-search (`spec2.txt` §3.1 "Key Feature").
//!
//! Accepts an accuracy budget (the `--accuracy-budget 0.05` flag) and finds
//! the cheapest per-layer quantization that stays inside it: every
//! weight-bearing layer starts at 4-bit ([`DType::Q4_0`]); layers are
//! promoted to 8-bit ([`DType::Q8_0`]) most-fragile-first until the
//! predicted end-to-end error meets the budget.
//!
//! # Where the accuracy numbers come from
//!
//! Two modes:
//!
//! * **Profile mode** — a `.tptprofile` sidecar carries measured per-layer
//!   sensitivities (the output-error degradation each layer causes when kept
//!   at INT4). This is the data the spec's "fast SiL pass" produces; until
//!   the executor lands (todo.md, `alloy` runtime) the profile is authored
//!   offline or carried over from a prior run.
//! * **Heuristic mode** — with no profile, sensitivities are derived from
//!   the graph shape: GEMM-class weight consumers share the error budget
//!   equally, elementwise consumers contribute a tenth of a GEMM share.
//!   Good enough for planning; profile mode is the calibrated path.
//!
//! # Error model
//!
//! Per-layer contributions combine as a root-sum-square (uncorrelated
//! rounding errors add in quadrature):
//!
//! ```text
//! predicted_error = sqrt( Σᵢ sensitivity(i, dtypeᵢ)² )
//! ```
//!
//! where `sensitivity` scales with the quantization step: promoting a layer
//! from Q4_0 to Q8_0 divides its contribution by 4 (the classic ~4× INT8
//! improvement).
//!
//! The search is a planner: it assigns target dtypes but does not requantize
//! tensor payloads (dequant/requant kernels are tracked separately in
//! todo.md).

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tpt_crucible_common::dtype::DType;
use tpt_crucible_common::error::{Error, Result};
use tpt_crucible_common::{Graph, NodeId, Op};

/// Contribution multiplier of a Q8_0 layer relative to the same layer at
/// Q4_0 (see the module-level error model).
const Q8_PROMOTION_FACTOR: f64 = 0.25;

/// Per-layer INT4 sensitivities loaded from a `.tptprofile` sidecar.
///
/// The file format is JSON:
///
/// ```json
/// {
///   "model": "tinyllama",
///   "layers": {
///     "blk.0.attn_q": 0.91,
///     "blk.0.ffn_down": 0.34
///   }
/// }
/// ```
///
/// Values are fractions of the layer's INT4 output-error contribution
/// (0..1]. Names must match the consuming IR node names (the tensors'
/// `blk.N.*` names survive ingestion).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensitivityProfile {
    /// Model the measurements were taken on (informational).
    #[serde(default)]
    pub model: Option<String>,
    /// Node/consumer name → relative INT4 sensitivity (0..1].
    #[serde(default)]
    pub layers: BTreeMap<String, f64>,
}

impl SensitivityProfile {
    /// Parse a `.tptprofile` sidecar from a string.
    ///
    /// # Errors
    /// [`Error::ParseFormat`] on malformed JSON, negative or non-finite
    /// sensitivities.
    pub fn from_json_str(raw: &str) -> Result<Self> {
        let profile: Self = serde_json::from_str(raw).map_err(|e| Error::ParseFormat {
            path: "<inline>".into(),
            format: "tptprofile".into(),
            reason: e.to_string(),
        })?;
        profile.validate()?;
        Ok(profile)
    }

    /// Load a `.tptprofile` sidecar from disk.
    ///
    /// # Errors
    /// [`Error::Io`] when unreadable, [`Error::ParseFormat`] when malformed.
    pub fn from_path(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let profile: Self = serde_json::from_str(&raw).map_err(|e| Error::ParseFormat {
            path: path.display().to_string(),
            format: "tptprofile".into(),
            reason: e.to_string(),
        })?;
        profile.validate()?;
        Ok(profile)
    }

    fn validate(&self) -> Result<()> {
        for (name, v) in &self.layers {
            if !v.is_finite() || *v < 0.0 || *v > 1.0 {
                return Err(Error::ParseFormat {
                    path: "<inline>".into(),
                    format: "tptprofile".into(),
                    reason: format!(
                        "sensitivity for `{name}` must be a finite value in 0..1, got {v}"
                    ),
                });
            }
        }
        Ok(())
    }

    /// Sensitivity of a consumer node; profile misses fall back to a neutral
    /// mid value so unseen layers still participate in promotion ordering.
    pub fn sensitivity(&self, node_name: &str) -> f64 {
        self.layers.get(node_name).copied().unwrap_or(0.5)
    }

    /// Number of profile entries that matched actual IR node names.
    pub fn matched_entries(&self, graph: &Graph) -> usize {
        self.layers
            .keys()
            .filter(|k| graph.nodes.iter().any(|n| &n.name == *k))
            .count()
    }

    /// Heuristic profile derived from the graph when no measurements exist:
    /// GEMM-class weight consumers share the budget equally, everything else
    /// contributes a tenth of a GEMM share.
    pub fn heuristic(graph: &Graph) -> Self {
        let consumers = weight_consumers(graph);
        let gemm_count = consumers
            .iter()
            .filter(|(_, _, op)| matches!(op, Op::MatMul | Op::Attention { .. }))
            .count();
        let share = if gemm_count == 0 {
            0.0
        } else {
            (gemm_count as f64).sqrt().recip()
        };
        let mut layers = BTreeMap::new();
        for (name, op) in consumers.iter().map(|(_, name, op)| (name, op)) {
            let s = if matches!(op, Op::MatMul | Op::Attention { .. }) {
                share
            } else {
                share * 0.1
            };
            layers.insert(name.clone(), s);
        }
        Self {
            model: Some(graph.name.clone()),
            layers,
        }
    }
}

/// Weight-consuming IR nodes: `(node id, node name, op)` for every node that
/// takes at least one floating-point [`Op::Constant`] input.
/// Already-quantized or integer-weight consumers are excluded — there is
/// nothing to promote.
fn weight_consumers(graph: &Graph) -> Vec<(NodeId, String, Op)> {
    let mut out = Vec::new();
    for (i, node) in graph.nodes.iter().enumerate() {
        let consumes_float_weight = node.inputs.iter().any(|&inp| {
            matches!(
                graph.get_node(inp).map(|n| &n.op),
                Some(Op::Constant { tensor })
                    if tensor.desc.dtype.is_float()
            )
        });
        if consumes_float_weight {
            out.push((NodeId(i as u32), node.name.clone(), node.op.clone()));
        }
    }
    out
}

/// Outcome of the auto-search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantPlan {
    /// Consumer node id → target weight dtype. Absent = left as ingested
    /// (non-float weights are never touched).
    pub assignments: BTreeMap<u32, DType>,
    /// Layers promoted from Q4_0 to Q8_0, most-fragile first.
    pub promotions: Vec<NodeId>,
    /// Predicted end-to-end error after the final assignment
    /// (root-sum-square model; 1.0 ≈ the all-INT4 baseline).
    pub predicted_error: f64,
    /// Requested accuracy budget.
    pub budget: f64,
    /// True when [`Self::predicted_error`] fits within [`Self::budget`].
    pub budget_met: bool,
    /// How many profile entries were used (`None` in heuristic mode).
    pub profile_entries_matched: Option<usize>,
}

impl QuantPlan {
    /// Render the plan as pretty JSON (the artifact `tpt quantize --save`
    /// writes).
    ///
    /// # Errors
    /// Serialization is infallible for this shape; the error exists for
    /// forward compatibility.
    pub fn to_json_string_pretty(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// Run the INT4→INT8 auto-search against an accuracy budget.
///
/// `profile` selects profile mode; `None` uses the graph-shape heuristic
/// ([`SensitivityProfile::heuristic`]).
///
/// # Errors
/// * [`Error::InvalidArgument`] when `budget` is not in `(0, 1]`.
pub fn search(
    graph: &Graph,
    profile: Option<&SensitivityProfile>,
    budget: f64,
) -> Result<QuantPlan> {
    if !budget.is_finite() || budget <= 0.0 || budget > 1.0 {
        return Err(Error::InvalidArgument(format!(
            "accuracy budget must be a finite fraction in (0, 1], got {budget}"
        )));
    }
    graph.validate()?;

    let owned;
    let prof = match profile {
        Some(p) => p,
        None => {
            owned = SensitivityProfile::heuristic(graph);
            &owned
        }
    };
    let matched = profile.map(|p| p.matched_entries(graph));

    // Consumers of float weights, in SSA order, each starting at Q4_0.
    let consumers: Vec<(NodeId, f64)> = weight_consumers(graph)
        .into_iter()
        .map(|(id, name, _)| (id, prof.sensitivity(&name)))
        .collect();

    let mut dtypes: Vec<DType> = vec![DType::Q4_0; consumers.len()];
    // Root-sum-square of per-layer contributions (uncorrelated errors add in
    // quadrature); Q8_0 shrinks a layer's contribution by a factor of 4.
    let rss = |dtypes: &[DType]| {
        consumers
            .iter()
            .zip(dtypes)
            .map(|(&(_, s), &d)| {
                let e = if d == DType::Q8_0 {
                    s * Q8_PROMOTION_FACTOR
                } else {
                    s
                };
                e * e
            })
            .sum::<f64>()
            .sqrt()
    };
    let mut error = rss(&dtypes);

    // Promote the most-fragile remaining Q4_0 layer until the budget holds.
    // Ties keep the earliest layer, so runs are deterministic.
    let mut promotions = Vec::new();
    while error > budget {
        let mut best: Option<usize> = None;
        for (i, (_, s)) in consumers.iter().enumerate() {
            if dtypes[i] != DType::Q4_0 {
                continue;
            }
            match best {
                Some(b) if consumers[b].1 >= *s => {}
                _ => best = Some(i),
            }
        }
        let Some(slot) = best else { break };
        dtypes[slot] = DType::Q8_0;
        promotions.push(consumers[slot].0);
        error = rss(&dtypes);
    }

    let assignments: BTreeMap<u32, DType> = consumers
        .iter()
        .zip(&dtypes)
        .map(|((id, _), d)| (id.0, *d))
        .collect();

    Ok(QuantPlan {
        assignments,
        promotions,
        predicted_error: error,
        budget,
        budget_met: error <= budget,
        profile_entries_matched: matched,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_crucible_common::{Tensor, TensorDesc};

    /// Two-MatMul MLP with named weight consumers (`attn_q`, `ffn_down`).
    /// Node ids: 0=x, 1=w0, 2=attn_q, 3=w1, 4=ffn_down, 5=out.
    fn two_layer_graph() -> Graph {
        let mut g = Graph::new("tiny");
        let x = g.push(
            "x",
            Op::Input(TensorDesc::new(vec![1, 8], DType::F32)),
            Vec::<NodeId>::new(),
        );
        let w0 = g.push(
            "w0",
            Op::Constant {
                tensor: Tensor::from_f32(vec![8, 8], &vec![0.5; 64]),
            },
            Vec::<NodeId>::new(),
        );
        let h = g.push("attn_q", Op::MatMul, vec![x, w0]);
        let w1 = g.push(
            "w1",
            Op::Constant {
                tensor: Tensor::from_f32(vec![8, 8], &vec![0.5; 64]),
            },
            Vec::<NodeId>::new(),
        );
        let y = g.push("ffn_down", Op::MatMul, vec![h, w1]);
        let o = g.push("out", Op::Output { name: "y".into() }, vec![y]);
        g.mark_output(o);
        g
    }

    #[test]
    fn fragile_layers_promote_first() {
        let g = two_layer_graph();
        let profile = SensitivityProfile::from_json_str(
            r#"{"model":"tiny","layers":{"attn_q":0.9,"ffn_down":0.2}}"#,
        )
        .unwrap();
        let plan = search(&g, Some(&profile), 0.5).unwrap();
        assert_eq!(
            plan.promotions,
            vec![NodeId(2)],
            "only the fragile layer promotes"
        );
        assert_eq!(plan.assignments[&2], DType::Q8_0);
        assert_eq!(plan.assignments[&4], DType::Q4_0);
        assert!(plan.budget_met);
        // sqrt((0.9*0.25)^2 + 0.2^2) = sqrt(0.050625 + 0.04) ≈ 0.301
        assert!(
            (plan.predicted_error - 0.301).abs() < 0.01,
            "{}",
            plan.predicted_error
        );
        assert_eq!(plan.profile_entries_matched, Some(2));
    }

    #[test]
    fn unreachable_budget_reports_honestly() {
        let g = two_layer_graph();
        let profile =
            SensitivityProfile::from_json_str(r#"{"layers":{"attn_q":1.0,"ffn_down":1.0}}"#)
                .unwrap();
        let plan = search(&g, Some(&profile), 0.01).unwrap();
        assert!(!plan.budget_met);
        assert_eq!(plan.promotions.len(), 2, "everything got promoted");
        assert!(plan.assignments.values().all(|&d| d == DType::Q8_0));
    }

    #[test]
    fn zero_error_when_no_float_weights() {
        let mut g = Graph::new("empty");
        let x = g.push(
            "x",
            Op::Input(TensorDesc::new(vec![1], DType::F32)),
            Vec::<NodeId>::new(),
        );
        let o = g.push("out", Op::Output { name: "o".into() }, vec![x]);
        g.mark_output(o);
        let plan = search(&g, None, 0.05).unwrap();
        assert!(plan.budget_met);
        assert_eq!(plan.predicted_error, 0.0);
        assert!(plan.assignments.is_empty());
    }

    #[test]
    fn heuristic_mode_runs_without_profile() {
        let g = two_layer_graph();
        let plan = search(&g, None, 1.0).unwrap();
        assert_eq!(plan.profile_entries_matched, None);
        // Budget 1.0 always holds at all-INT4 (error == 1.0 by construction:
        // both GEMM layers get share = 1/sqrt(2), RSS = 1.0).
        assert!((plan.predicted_error - 1.0).abs() < 1e-9);
        assert!(plan.budget_met);
        assert!(plan.promotions.is_empty());
        assert_eq!(plan.assignments.len(), 2);
    }

    #[test]
    fn heuristic_promotes_under_tight_budget() {
        let g = two_layer_graph();
        // Heuristic share per GEMM layer is 1/sqrt(2) ≈ 0.7071, so all-INT4
        // error is 1.0. Budget 0.4 forces BOTH layers to Q8_0:
        // sqrt(2) * (0.7071 * 0.25) = 0.25.
        let plan = search(&g, None, 0.4).unwrap();
        assert!(plan.budget_met);
        assert_eq!(plan.promotions, vec![NodeId(2), NodeId(4)]);
        assert!((plan.predicted_error - 0.25).abs() < 1e-9);
    }

    #[test]
    fn invalid_budgets_rejected() {
        let g = two_layer_graph();
        for bad in [0.0, -0.1, 1.5, f64::NAN] {
            let err = search(&g, None, bad).unwrap_err();
            assert!(matches!(err, Error::InvalidArgument(_)));
        }
    }

    #[test]
    fn negative_sensitivity_rejected() {
        let err = SensitivityProfile::from_json_str(r#"{"layers":{"x":-1}}"#).unwrap_err();
        assert!(err.to_string().contains("finite value in 0..1"), "{err}");
    }

    #[test]
    fn plan_serializes_roundtrip() {
        let g = two_layer_graph();
        let plan = search(&g, None, 0.5).unwrap();
        let json = plan.to_json_string_pretty().unwrap();
        let back: QuantPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(back, plan);
    }
}
