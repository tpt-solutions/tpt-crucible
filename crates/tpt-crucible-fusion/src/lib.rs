//! # tpt-crucible-fusion
//!
//! **Module 1 (FPGA):** high-bandwidth logic synthesis. Takes TPT-IR and emits
//! an FPGA overlay description that wires MAC arrays into pre-verified HBM
//! memory controllers, targeting ~10 s per-model compiles instead of full
//! resynthesis (`spec2.txt` §3.2).
//!
//! ## Status
//!
//! Planned for Phase 2 ("The Silicon Canvas"). The API surface below is the
//! contract the rest of the suite codes against; it currently returns
//! [`Error::NotImplemented`].
//!
//! Roadmap items live in `todo.md`.
//!
//! ## Hybrid boards
//!
//! `tpt_crucible_alloy::topology::FpgaProfile` already describes the fabric
//! (LUTs, DSP slices, block RAM) a swarm node's FPGA offers, for boards that
//! pair normal silicon with reconfigurable logic. Once this crate compiles
//! real overlays, that's the shape its output should target so `alloy` and
//! `fusion` share one contract instead of inventing a second one here.

use tpt_crucible_common::error::{Error, Result};
use tpt_crucible_common::Graph;

/// Human-readable implementation status of this crate.
pub fn status() -> &'static str {
    "planned: Phase 2 (The Silicon Canvas) — see todo.md"
}

/// Compile TPT-IR into a `.fusecfg` overlay configuration plus RTL artifacts.
///
/// Returns the raw artifact bundle once implemented; currently always fails
/// with [`Error::NotImplemented`].
pub fn compile(_graph: &Graph) -> Result<Vec<u8>> {
    Err(Error::NotImplemented {
        feature: "fusion FPGA overlay compilation".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_is_honest_about_not_being_built_yet() {
        let g = Graph::new("x");
        let err = compile(&g).unwrap_err();
        assert!(err.to_string().contains("not implemented"));
        assert!(status().contains("Phase 2"));
    }
}
