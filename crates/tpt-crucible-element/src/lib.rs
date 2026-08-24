//! # tpt-crucible-element
//!
//! **Module 2 (Analog):** physics-to-weight mapping and noise simulation.
//! Lowers TPT-IR weights into physical electrical components, generates SPICE
//! netlists for Xyce/ngspice, runs the "Reality Check" engine (thermal noise,
//! voltage drift, component tolerances), and proposes hardware mitigations
//! with an ML-predicted confidence score (`spec2.txt` §3.3).
//!
//! ## Status
//!
//! Planned for Phase 3 ("The Physics Engine"). The API surface below is the
//! contract the rest of the suite codes against; it currently returns
//! [`Error::NotImplemented`].

use tpt_crucible_common::error::{Error, Result};
use tpt_crucible_common::Graph;

/// Human-readable implementation status of this crate.
pub fn status() -> &'static str {
    "planned: Phase 3 (The Physics Engine) — see todo.md"
}

/// Lower TPT-IR weights into a SPICE netlist of physical components.
///
/// Returns netlist text once implemented; currently always fails with
/// [`Error::NotImplemented`].
pub fn to_spice_netlist(_graph: &Graph) -> Result<String> {
    Err(Error::NotImplemented {
        feature: "element SPICE netlist generation".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn netlist_is_honest_about_not_being_built_yet() {
        let g = Graph::new("x");
        let err = to_spice_netlist(&g).unwrap_err();
        assert!(err.to_string().contains("not implemented"));
        assert!(status().contains("Phase 3"));
    }
}
