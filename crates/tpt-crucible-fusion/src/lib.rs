//! # tpt-crucible-fusion
//!
//! **Module 1 (FPGA):** high-bandwidth logic synthesis. Takes TPT-IR and
//! emits an FPGA **overlay bundle** that stages MAC-array weights into block
//! RAM banks and HBM channels behind a pre-synthesized datapath — model
//! compiles stay out of synthesis entirely, targeting the spec's ~10 s
//! per-model budget (`spec2.txt` §3.2).
//!
//! ## Modules
//!
//! * [`overlay`] — resource planner: contiguous BRAM bank staging per GEMM
//!   layer, round-robin HBM channel assignment, DSP requirement accounting,
//!   and `.fusecfg` emission (header comments + versioned JSON body).
//! * [`rtl`] — parameterized Verilog-2001 MAC array + `$readmemh` memory-init
//!   files; vendor FP cores appear as documented instantiation points.
//! * [`tools`] — silent Yosys/Nextpnr wrappers over `std::process::Command`
//!   with structured
//!   [`tpt_crucible_common::Error::ExternalTool`] failures and install hints.
//!
//! ## Example
//!
//! ```
//! use tpt_crucible_common::{DType, Graph, Op, Tensor, TensorDesc};
//! use tpt_crucible_fusion::{compile, CompileOptions, FpgaProfile};
//!
//! # fn build() -> tpt_crucible_common::Result<()> {
//! let mut g = Graph::new("tiny");
//! let x = g.push("x", Op::Input(TensorDesc::new(vec![1, 4], DType::F32)), vec![]);
//! let w = g.push("fc0_w", Op::Constant { tensor: Tensor::from_f32(vec![4, 4], &[0.25; 16]) }, vec![]);
//! let y = g.push("fc0", Op::MatMul, vec![x, w]);
//! let o = g.push("out", Op::Output { name: "y".into() }, vec![y]);
//! g.mark_output(o);
//!
//! let profile = FpgaProfile { luts: 100_000, dsp_slices: 9024, block_ram_bytes: 38 << 20 };
//! let bundle = compile(&g, &profile, &CompileOptions::default())?;
//! assert!(bundle.fusecfg.contains("fusecfg_version"));
//! # Ok(())
//! # }
//! # build().unwrap();
//! ```
//!
//! ## Hybrid boards
//!
//! The profile type converts from `tpt_crucible_alloy::topology::FpgaProfile`,
//! so hybrid swarm boards and standalone overlays share one contract
//! (`todo.md`: "alloy<->fusion bridge").
//!
//! ## Status
//!
//! Overlay planning, `.fusecfg`, RTL/meminit output, and tool wrappers are
//! live. Remaining (todo.md): rust-hdl migration of RTL generation, LiteX/
//! LiteDRAM wrappers, real bitstream patching — plus the Alveo milestone,
//! which needs hardware.

pub mod overlay;
pub mod rtl;
pub mod tools;

use tpt_crucible_common::error::Result;
use tpt_crucible_common::Graph;

pub use overlay::{FpgaProfile, OverlayPlan, PlanOptions, StagedLayer};
pub use rtl::RtlFile;

/// Human-readable implementation status of this crate.
pub fn status() -> &'static str {
    "phase 2 (The Silicon Canvas): overlay planner + .fusecfg + RTL output \
     live; rust-hdl migration & bitstream patching pending"
}

/// Options for [`compile`].
#[derive(Debug, Clone, Default)]
pub struct CompileOptions {
    /// Resource-planning knobs (bank size, channel count).
    pub plan: PlanOptions,
}

/// Everything one overlay compile produces.
#[derive(Debug, Clone)]
pub struct OverlayBundle {
    /// The placement decisions.
    pub plan: OverlayPlan,
    /// `.fusecfg` text (what the pre-synthesized loader consumes).
    pub fusecfg: String,
    /// RTL + meminit artifacts (`rtl/…` paths relative to a bundle root).
    pub files: Vec<RtlFile>,
}

impl OverlayBundle {
    /// Write the bundle into `dir` (`fusecfg.overlay`, `rtl/…`).
    ///
    /// # Errors
    /// [`tpt_crucible_common::Error::Io`] on filesystem failure.
    pub fn write_into(&self, dir: &std::path::Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("overlay.fusecfg"), &self.fusecfg)?;
        for f in &self.files {
            let path = dir.join(&f.path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, &f.contents)?;
        }
        Ok(())
    }
}

/// Compile TPT-IR into an overlay bundle against a fabric profile.
///
/// # Errors
/// Whatever planning ([`overlay::plan_overlay`]) or RTL emission
/// ([`rtl::emit_rtl`]) raise.
pub fn compile(
    graph: &Graph,
    profile: &FpgaProfile,
    options: &CompileOptions,
) -> Result<OverlayBundle> {
    let plan = overlay::plan_overlay(graph, profile, &options.plan)?;
    let fusecfg = plan.to_fusecfg()?;
    let files = rtl::emit_rtl(&plan)?;
    Ok(OverlayBundle {
        plan,
        fusecfg,
        files,
    })
}
