//! # tpt-crucible-element
//!
//! **Module 2 (Analog):** physics-to-weight mapping and noise simulation.
//! Lowers TPT-IR weights into physical electrical components, generates SPICE
//! netlists for Xyce/ngspice, runs the "Reality Check" engine (thermal noise,
//! voltage drift, component tolerances), and proposes hardware mitigations
//! with a Monte-Carlo confidence score (`spec2.txt` §3.3).
//!
//! ## Modules
//!
//! * [`netlist`] — extract MatMul layers as crossbar MAC arrays and render a
//!   ngspice/Xyce-ready SPICE netlist (`G`-element transconductance bank +
//!   load resistors).
//! * [`reality`] — seeded, dependency-free Monte-Carlo over component
//!   tolerance / supply drift / Johnson noise; reports deviation statistics,
//!   the dominant physical factor, a **confidence score**, and rule-based
//!   hardware mitigations.
//! * [`simulator`] — ngspice batch-mode orchestration: run a rendered
//!   netlist through the real simulator and parse the operating point back.
//! * [`pcb`] — PCB layout recommendations keyed to array geometry and the
//!   dominant physics factor.
//!
//! ## Example
//!
//! ```
//! use tpt_crucible_common::{DType, Graph, Op, Tensor, TensorDesc};
//!
//! let mut g = Graph::new("tiny");
//! let x = g.push("x", Op::Input(TensorDesc::new(vec![1, 2], DType::F32)), vec![]);
//! let w = g.push("fc0_w", Op::Constant { tensor: Tensor::from_f32(vec![2, 2], &[0.5, -0.25, 0.1, 0.9]) }, vec![]);
//! let y = g.push("fc0", Op::MatMul, vec![x, w]);
//! let o = g.push("out", Op::Output { name: "y".into() }, vec![y]);
//! g.mark_output(o);
//!
//! let bundle = tpt_crucible_element::synthesize(&g, &Default::default(), &Default::default())?;
//! assert!(bundle.netlist.contains(".end"));
//! println!("confidence: {:.1}%", bundle.confidence() * 100.0);
//! # Ok::<(), tpt_crucible_common::Error>(())
//! ```
//!
//! ## Status
//!
//! Phase 3 core is live: netlists, Reality Check, mitigations, PCB guidance,
//! confidence score. Still pending (tracked in todo.md): Xyce/ngspice FFI to
//! run the netlist in a real simulator, and the `ort` ML drift predictor.

pub mod netlist;
pub mod pcb;
pub mod reality;
pub mod simulator;

use tpt_crucible_common::error::Result;
use tpt_crucible_common::Graph;

pub use netlist::{MacLayer, NetlistOptions};
pub use pcb::LayoutRecommendation;
pub use reality::{Factor, RealityConfig, RealityReport};
pub use simulator::{SimResult, SimulatorConfig, SimulatorKind};

/// Human-readable implementation status of this crate.
pub fn status() -> &'static str {
    "phase 3 (The Physics Engine): SPICE netlists + Reality Check live; \
     simulator FFI & ML drift prediction pending"
}

/// Everything one analog synthesis run produces.
#[derive(Debug, Clone)]
pub struct SynthesisBundle {
    /// SPICE netlist text (`netlist.sp`).
    pub netlist: String,
    /// Extracted MAC layers the netlist was built from.
    pub layers: Vec<MacLayer>,
    /// Reality Check statistics.
    pub reality: RealityReport,
    /// Hardware mitigation suggestions.
    pub mitigations: Vec<String>,
    /// PCB layout recommendations.
    pub layout: Vec<LayoutRecommendation>,
}

impl SynthesisBundle {
    /// The confidence score: fraction of simulated outputs within tolerance.
    pub fn confidence(&self) -> f64 {
        self.reality.confidence
    }
}

/// Full analog synthesis pipeline: netlist → reality check → suggestions.
///
/// # Errors
/// Whatever [`netlist::extract_mac_layers`] / [`reality::reality_check`]
/// raise (no float weights, invalid IR, bad config).
pub fn synthesize(
    graph: &Graph,
    cfg: &RealityConfig,
    opts: &NetlistOptions,
) -> Result<SynthesisBundle> {
    let layers = netlist::extract_mac_layers(graph)?;
    let netlist_text = netlist::render(&layers, opts)?;
    let reality_report = reality::reality_check(&layers, cfg)?;
    let mitigations = reality::mitigations(&reality_report);
    let layout = pcb::recommend_layout(&reality_report, &layers);
    Ok(SynthesisBundle {
        netlist: netlist_text,
        layers,
        reality: reality_report,
        mitigations,
        layout,
    })
}
