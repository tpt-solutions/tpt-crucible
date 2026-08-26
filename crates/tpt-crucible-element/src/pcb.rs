//! PCB layout recommendations for analog arrays (`spec2.txt` §3.3 output).
//!
//! Rule-based, driven by the Reality Check report and the extracted array
//! geometry: the same physics that dominates simulation accuracy also
//! dictates placement, grounding, and thermal design. These are engineering
//! guidance for a human (or a future autorouter), not a full place-and-route.

use serde::{Deserialize, Serialize};

use crate::netlist::MacLayer;
use crate::reality::{Factor, RealityReport};

/// One actionable layout recommendation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayoutRecommendation {
    /// Area of the board this applies to (`"ground"`, `"array"`, `"power"`,
    /// …).
    pub area: String,
    /// The recommendation.
    pub advice: String,
}

/// Produce layout recommendations from a reality report plus the arrays it
/// was measured on.
#[must_use]
pub fn recommend_layout(report: &RealityReport, layers: &[MacLayer]) -> Vec<LayoutRecommendation> {
    let mut out = Vec::new();

    // Universal analog hygiene.
    out.push(LayoutRecommendation {
        area: "ground".into(),
        advice: "Star-ground the array return; keep digital ground returns on a \
                 separate plane that meets the star at exactly one point."
            .into(),
    });
    out.push(LayoutRecommendation {
        area: "array".into(),
        advice: "Ring every crossbar/MAC array with a guard ring tied to the \
                 clean analog ground; guard the sense lines too."
            .into(),
    });
    out.push(LayoutRecommendation {
        area: "supply".into(),
        advice: "One bulk + one 100 nF decoupling capacitor per supply pin, \
                 placed within 2 mm of the pin."
            .into(),
    });

    // Geometry-driven.
    let widest = layers.iter().map(|l| l.cols).max().unwrap_or(0);
    if widest > 64 {
        out.push(LayoutRecommendation {
            area: "input bus".into(),
            advice: format!(
                "{widest} input columns exceed a comfortable single-bus fanout; \
                 segment the array or buffer inputs every 32 columns."
            ),
        });
    }
    let total_rows: usize = layers.iter().map(|l| l.rows).sum();
    if total_rows > 256 {
        out.push(LayoutRecommendation {
            area: "output mux".into(),
            advice: format!(
                "{total_rows} summed rows need multiplexing — plan the output mux \
                 adjacent to the array to keep un-buffered sense lines short."
            ),
        });
    }

    // Physics-driven.
    match report.dominant_factor {
        Factor::ThermalNoise => {
            out.push(LayoutRecommendation {
                area: "thermal".into(),
                advice: "Thermal noise dominates: pour copper under the array as a \
                         heat spreader and keep hot regulators ≥ 10 mm away."
                    .into(),
            });
        }
        Factor::VoltageDrift => {
            out.push(LayoutRecommendation {
                area: "power".into(),
                advice: "Voltage drift dominates: route the array supply as kelvin \
                         pairs (force + sense) back to its dedicated LDO."
                    .into(),
            });
        }
        Factor::ComponentTolerance => {
            out.push(LayoutRecommendation {
                area: "calibration".into(),
                advice: "Tolerance dominates: leave probe pads on reference rows so \
                         post-fab calibration can measure real device spread."
                    .into(),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reality::{FactorDeviations, PerturbationModel};
    use tpt_crucible_common::NodeId;

    fn report(dominant: Factor) -> RealityReport {
        RealityReport {
            trials: 16,
            tolerance: 0.05,
            confidence: 0.9,
            mean_abs_dev_pct: 0.01,
            max_dev_pct: 0.03,
            dominant_factor: dominant,
            factors: FactorDeviations {
                tolerance_only_pct: 0.001,
                drift_only_pct: 0.02,
                thermal_only_pct: 0.005,
            },
            model: PerturbationModel::default(),
            seed: 1,
        }
    }

    fn layer(rows: usize, cols: usize) -> MacLayer {
        MacLayer {
            name: "fc".into(),
            node: NodeId(0),
            rows,
            cols,
            weights: vec![0.5; rows * cols],
        }
    }

    #[test]
    fn always_includes_analog_hygiene() {
        let recs = recommend_layout(&report(Factor::VoltageDrift), &[layer(4, 4)]);
        assert!(recs.iter().any(|r| r.area == "ground"));
        assert!(recs.iter().any(|r| r.area == "array"));
        assert!(recs.iter().any(|r| r.area == "supply"));
    }

    #[test]
    fn geometry_and_physics_add_targeted_advice() {
        // Wide + deep array, drift-dominant.
        let recs = recommend_layout(&report(Factor::VoltageDrift), &[layer(300, 128)]);
        assert!(recs.iter().any(|r| r.area == "input bus"));
        assert!(recs.iter().any(|r| r.area == "output mux"));
        assert!(recs
            .iter()
            .any(|r| r.area == "power" && r.advice.contains("kelvin")));

        // Thermal dominance switches the targeted block.
        let recs = recommend_layout(&report(Factor::ThermalNoise), &[layer(2, 2)]);
        assert!(recs.iter().any(|r| r.area == "thermal"));

        // Small clean array gets only the universal set + calibration note.
        let recs = recommend_layout(&report(Factor::ComponentTolerance), &[layer(2, 8)]);
        assert_eq!(recs.len(), 4);
        assert!(recs.iter().any(|r| r.area == "calibration"));
    }

    #[test]
    fn recommendations_serialize() {
        let recs = recommend_layout(&report(Factor::ThermalNoise), &[layer(1, 1)]);
        let json = serde_json::to_string(&recs).unwrap();
        let back: Vec<LayoutRecommendation> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, recs);
    }
}
