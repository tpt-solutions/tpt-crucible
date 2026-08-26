//! The "Reality Check" engine: simulated physics vs. your analog math
//! (`spec2.txt` §3.3 "Key Feature").
//!
//! Real analog hardware deviates from the netlist's clean algebra in three
//! dominant ways:
//!
//! * **Component tolerance** — every realized weight carries a fabrication
//!   error (resistor/memristor spread, typically ±0.1–5 %);
//! * **Voltage drift** — supply droop and temperature shift the effective
//!   gain of the whole array (a common-mode, correlated error);
//! * **Thermal noise** — Johnson–Nyquist noise on every summing node adds a
//!   small uncorrelated jitter per output.
//!
//! [`reality_check`] runs a seeded Monte-Carlo over those effects on the
//! extracted `MacLayer`s and reports deviation
//! statistics, a **confidence score** (fraction of trials within tolerance),
//! which factor dominates, and rule-based hardware mitigation suggestions.
//!
//! Everything is deliberately deterministic: a fixed-seed xorshift* PRNG
//! stands in for `rand` (no external dependency, identical numbers on every
//! platform — wasm included), so reports are reproducible artifacts.

use serde::{Deserialize, Serialize};

use crate::netlist::MacLayer;

/// A physical perturbation source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Factor {
    /// Fabrication spread on individual weights.
    ComponentTolerance,
    /// Supply/temperature gain drift (correlated across the array).
    VoltageDrift,
    /// Johnson–Nyquist noise on summing nodes.
    ThermalNoise,
}

impl Factor {
    /// Canonical lowercase name.
    pub fn name(self) -> &'static str {
        match self {
            Self::ComponentTolerance => "component_tolerance",
            Self::VoltageDrift => "voltage_drift",
            Self::ThermalNoise => "thermal_noise",
        }
    }
}

/// Physical model knobs (fractions of full scale unless noted).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationModel {
    /// Per-weight fabrication tolerance, ± fraction (Gaussian σ = tol/3,
    /// i.e. ±tol covers ~99.7 %).
    pub component_tolerance: f64,
    /// One-sigma supply-induced gain drift as a fraction.
    pub voltage_drift_sigma: f64,
    /// One-sigma thermal output noise as a fraction of full scale.
    pub thermal_noise_sigma: f64,
}

impl Default for PerturbationModel {
    fn default() -> Self {
        // Conservative hobbyist-fab numbers; tighten for precision fabs.
        Self {
            component_tolerance: 0.01,
            voltage_drift_sigma: 0.02,
            thermal_noise_sigma: 0.005,
        }
    }
}

/// Monte-Carlo configuration.
#[derive(Debug, Clone)]
pub struct RealityConfig {
    /// Trials to run (more → smoother statistics).
    pub trials: u32,
    /// Output tolerance defining "within spec", as a fraction.
    pub tolerance: f64,
    /// Cap on rows sampled per layer (keeps big layers cheap; rows are taken
    /// evenly spaced so coverage stays representative).
    pub max_rows_per_layer: usize,
    /// Seed for the internal xorshift* PRNG.
    pub seed: u64,
    /// The physical model.
    pub model: PerturbationModel,
}

impl Default for RealityConfig {
    fn default() -> Self {
        Self {
            trials: 256,
            tolerance: 0.05,
            max_rows_per_layer: 64,
            seed: 0x5EED_CAFE_F00D_2026,
            model: PerturbationModel::default(),
        }
    }
}

/// Deviation attribution per factor (mean absolute deviation when only that
/// factor is active).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FactorDeviations {
    /// Mean |dev| from tolerance alone.
    pub tolerance_only_pct: f64,
    /// Mean |dev| from drift alone.
    pub drift_only_pct: f64,
    /// Mean |dev| from thermal noise alone.
    pub thermal_only_pct: f64,
}

/// Full report of one Reality Check run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealityReport {
    /// Trials actually run.
    pub trials: u32,
    /// Tolerance the confidence was measured against (fraction).
    pub tolerance: f64,
    /// Fraction of trial outputs within [`Self::tolerance`].
    pub confidence: f64,
    /// Mean absolute output deviation (%).
    pub mean_abs_dev_pct: f64,
    /// Worst observed output deviation (%).
    pub max_dev_pct: f64,
    /// Which single factor contributes the largest mean deviation.
    pub dominant_factor: Factor,
    /// Per-factor isolated contributions.
    pub factors: FactorDeviations,
    /// Model used (echoed so reports are self-describing).
    pub model: PerturbationModel,
    /// Seed used (reproducibility).
    pub seed: u64,
}

/// Deterministic xorshift* PRNG — tiny, dependency-free, portable.
struct XorShift(u64);

impl XorShift {
    fn next_f64(&mut self) -> f64 {
        // xorshift* (Vigna 2014): good statistical quality in 3 lines.
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        // Scale into [0, 1); the multiplier is the xorshift* constant.
        (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximate standard normal via Irwin–Hall (sum of 12 uniforms − 6):
    /// variance exactly 1, plenty accurate for Monte-Carlo smoothing.
    fn next_normal(&mut self) -> f64 {
        let sum: f64 = (0..12).map(|_| self.next_f64()).sum();
        sum - 6.0
    }
}

/// Deterministic probe input for a column: a fixed four-phase pattern that
/// exercises both signs **without** cancelling against constant weights
/// (mean = 0.0625, so uniform-weight rows have a healthy nonzero golden).
fn probe(col: usize) -> f64 {
    match col % 4 {
        0 => -0.5,
        1 => 0.25,
        2 => 0.75,
        _ => -0.25,
    }
}

/// One pass over every sampled row for `trials` trials with the given active
/// factors `(tolerance, drift, thermal)`; returns
/// `(mean |dev|, max |dev|, within-tolerance count, samples)`. Deviation is
/// `|out − golden| / max(|golden|, ε)`.
#[allow(clippy::too_many_arguments)]
fn run_pass(
    layers: &[MacLayer],
    sampled: &[(usize, usize, f64)], // (layer, row, golden)
    probes: &[Vec<f64>],             // per-layer input vector
    rng: &mut XorShift,
    m: &PerturbationModel,
    tol: f64,
    trials: u32,
    active: (bool, bool, bool),
) -> (f64, f64, u64, u64) {
    let mut dev_sum = 0.0;
    let mut dev_max = 0.0f64;
    let mut within = 0u64;
    let mut count = 0u64;
    let eps = 1e-6;
    for _ in 0..trials {
        // Correlated gain error: one draw per trial, shared by all outputs.
        let gain = if active.1 {
            1.0 + rng.next_normal() * m.voltage_drift_sigma
        } else {
            1.0
        };
        for &(li, row, golden) in sampled {
            let mut acc = 0.0;
            for (j, &p) in probes[li].iter().enumerate() {
                let mut w = f64::from(layers[li].at(row, j));
                if active.0 {
                    // ±tolerance covers ~99.7 % → σ = tol/3.
                    w *= 1.0 + rng.next_normal() * m.component_tolerance / 3.0;
                }
                acc += w * p;
            }
            let mut out = gain * acc;
            if active.2 {
                out += rng.next_normal() * m.thermal_noise_sigma;
            }
            let dev = (out - golden).abs() / golden.abs().max(eps);
            dev_sum += dev;
            dev_max = dev_max.max(dev);
            if dev <= tol {
                within += 1;
            }
            count += 1;
        }
    }
    (dev_sum / count as f64, dev_max, within, count)
}

/// Run the Reality Check Monte-Carlo over `layers`.
///
/// # Errors
/// [`tpt_crucible_common::Error::InvalidArgument`] when trials is zero,
/// tolerance/model fractions are not sane, or no layer is non-empty.
pub fn reality_check(
    layers: &[MacLayer],
    cfg: &RealityConfig,
) -> tpt_crucible_common::Result<RealityReport> {
    use tpt_crucible_common::Error;

    if cfg.trials == 0 {
        return Err(Error::InvalidArgument("trials must be at least 1".into()));
    }
    if !cfg.tolerance.is_finite() || cfg.tolerance <= 0.0 || cfg.tolerance >= 1.0 {
        return Err(Error::InvalidArgument(format!(
            "tolerance must be a finite fraction in (0, 1), got {}",
            cfg.tolerance
        )));
    }
    let m = &cfg.model;
    for (label, v) in [
        ("component_tolerance", m.component_tolerance),
        ("voltage_drift_sigma", m.voltage_drift_sigma),
        ("thermal_noise_sigma", m.thermal_noise_sigma),
    ] {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err(Error::InvalidArgument(format!(
                "{label} must be a finite fraction in [0, 1], got {v}"
            )));
        }
    }
    if layers.is_empty() || layers.iter().any(|l| l.rows == 0 || l.cols == 0) {
        return Err(Error::InvalidArgument(
            "reality check needs at least one non-empty MAC layer".into(),
        ));
    }

    // Per-layer probe vectors and evenly spaced sampled rows with golden
    // outputs (row capping keeps big arrays cheap while staying
    // representative).
    let probes: Vec<Vec<f64>> = layers
        .iter()
        .map(|l| (0..l.cols).map(probe).collect())
        .collect();
    let mut sampled: Vec<(usize, usize, f64)> = Vec::new();
    for (li, layer) in layers.iter().enumerate() {
        let step = layer.rows.div_ceil(cfg.max_rows_per_layer).max(1);
        for row in (0..layer.rows).step_by(step) {
            let golden: f64 = (0..layer.cols)
                .map(|j| f64::from(layer.at(row, j)) * probes[li][j])
                .sum();
            sampled.push((li, row, golden));
        }
    }
    if sampled.is_empty() {
        return Err(Error::InvalidArgument(
            "no sampleable rows in the given layers".into(),
        ));
    }

    // All-factors run.
    let mut rng = XorShift(cfg.seed);
    let (mean_dev, max_dev, within, count) = run_pass(
        layers,
        &sampled,
        &probes,
        &mut rng,
        m,
        cfg.tolerance,
        cfg.trials,
        (true, true, true),
    );

    // Single-factor isolation on a fresh stream for each.
    let iso_seed = cfg.seed ^ 0xA5A5_5A5A_DEAD_BEEF;
    let iso = |active: (bool, bool, bool)| {
        let mut r = XorShift(iso_seed);
        run_pass(
            layers,
            &sampled,
            &probes,
            &mut r,
            m,
            f64::INFINITY,
            cfg.trials,
            active,
        )
        .0
    };
    let factors = FactorDeviations {
        tolerance_only_pct: iso((true, false, false)),
        drift_only_pct: iso((false, true, false)),
        thermal_only_pct: iso((false, false, true)),
    };
    let dominant_factor = [
        (Factor::ComponentTolerance, factors.tolerance_only_pct),
        (Factor::VoltageDrift, factors.drift_only_pct),
        (Factor::ThermalNoise, factors.thermal_only_pct),
    ]
    .into_iter()
    .max_by(|a, b| a.1.total_cmp(&b.1))
    .map(|(f, _)| f)
    .unwrap_or(Factor::ComponentTolerance);

    Ok(RealityReport {
        trials: cfg.trials,
        tolerance: cfg.tolerance,
        confidence: within as f64 / count as f64,
        mean_abs_dev_pct: mean_dev,
        max_dev_pct: max_dev,
        dominant_factor,
        factors,
        model: m.clone(),
        seed: cfg.seed,
    })
}

/// Rule-based hardware mitigation suggestions for a report.
#[must_use]
pub fn mitigations(report: &RealityReport) -> Vec<String> {
    let mut out = Vec::new();
    match report.dominant_factor {
        Factor::ComponentTolerance => {
            out.push(
                "Add a post-fab calibration pass: measure reference rows and trim \
                 with a small DAC per column (recovers most tolerance spread)."
                    .into(),
            );
            out.push(
                "Specify precision resistor networks or laser-trimmed devices \
                 instead of generic thin-film parts."
                    .into(),
            );
        }
        Factor::VoltageDrift => {
            out.push(
                "Give the array a dedicated low-noise LDO with kelvin sensing; \
                 share no regulator with digital loads."
                    .into(),
            );
            out.push(
                "Enable chopper-stabilized auto-zero on the sense amplifiers to \
                 cancel slow supply/temperature drift."
                    .into(),
            );
        }
        Factor::ThermalNoise => {
            out.push(
                "Reduce summing-node impedance (halving R halves integrated \
                 Johnson noise) at the cost of power."
                    .into(),
            );
            out.push(
                "Average repeated samples or lengthen integration time — thermal \
                 noise falls with sqrt(N)."
                    .into(),
            );
        }
    }
    if report.confidence < 0.95 && report.dominant_factor != Factor::ThermalNoise {
        out.push(
            "Confidence below 95 %: consider derating the analog array to \
             lower-order bits and keeping sign-critical layers in silicon."
                .into(),
        );
    }
    if report.confidence >= 0.995 {
        out.push(
            "Confidence ≥ 99.5 %: the array is production-viable for this \
             tolerance; lock the model parameters into the datasheet."
                .into(),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(rows: usize, cols: usize, w: f32) -> MacLayer {
        MacLayer {
            name: "fc0".into(),
            node: tpt_crucible_common::NodeId(0),
            rows,
            cols,
            weights: vec![w; rows * cols],
        }
    }

    #[test]
    fn prng_is_deterministic_and_normalish() {
        let mut a = XorShift(42);
        let mut b = XorShift(42);
        for _ in 0..1000 {
            assert_eq!(a.next_f64(), b.next_f64());
        }
        // Same seed, same sequence; different seed, different stream.
        assert_ne!(XorShift(1).next_f64(), XorShift(2).next_f64());
        // Irwin–Hall normals center on zero with unit-ish spread.
        let mut r = XorShift(7);
        let mean: f64 = (0..10_000).map(|_| r.next_normal()).sum::<f64>() / 10_000.0;
        assert!(mean.abs() < 0.05, "{mean}");
    }

    #[test]
    fn ideal_model_keeps_full_confidence() {
        let layers = [layer(4, 8, 0.5)];
        let cfg = RealityConfig {
            trials: 128,
            model: PerturbationModel {
                component_tolerance: 0.0,
                voltage_drift_sigma: 0.0,
                thermal_noise_sigma: 0.0,
            },
            ..RealityConfig::default()
        };
        let report = reality_check(&layers, &cfg).unwrap();
        assert!((report.confidence - 1.0).abs() < 1e-12);
        assert!(report.mean_abs_dev_pct < 1e-9);
    }

    #[test]
    fn noise_lowers_confidence_and_names_a_dominant_factor() {
        let layers = [layer(16, 16, 0.5)];
        let report = reality_check(&layers, &RealityConfig::default()).unwrap();
        assert!(
            report.confidence < 1.0,
            "realistic model must lose some trials"
        );
        assert!((0.0..=1.0).contains(&report.confidence));
        assert!(report.max_dev_pct >= report.mean_abs_dev_pct);
        // Drift (σ=2 %) should dominate tolerance (σ≈tol/3) and thermal
        // (σ=0.5 %) under defaults.
        assert_eq!(report.dominant_factor, Factor::VoltageDrift);
        assert!(report.factors.drift_only_pct > report.factors.thermal_only_pct);
    }

    #[test]
    fn reports_are_reproducible() {
        let layers = [layer(8, 8, -0.75)];
        let a = reality_check(&layers, &RealityConfig::default()).unwrap();
        let b = reality_check(&layers, &RealityConfig::default()).unwrap();
        assert_eq!(a.confidence, b.confidence);
        assert_eq!(a.mean_abs_dev_pct, b.mean_abs_dev_pct);
        assert_eq!(a.seed, b.seed);
    }

    #[test]
    fn mitigation_rules_match_the_dominant_factor() {
        let layers = [layer(4, 4, 0.5)];
        let mut base = reality_check(&layers, &RealityConfig::default()).unwrap();

        base.dominant_factor = Factor::ComponentTolerance;
        base.confidence = 0.9;
        let m = mitigations(&base);
        assert!(m.iter().any(|s| s.contains("calibration")));
        assert!(m.iter().any(|s| s.contains("below 95")));

        base.dominant_factor = Factor::VoltageDrift;
        let m = mitigations(&base);
        assert!(m.iter().any(|s| s.contains("LDO")));

        base.dominant_factor = Factor::ThermalNoise;
        base.confidence = 0.996;
        let m = mitigations(&base);
        assert!(m
            .iter()
            .any(|s| s.contains("Johnson") || s.contains("impedance")));
        assert!(m.iter().any(|s| s.contains("production-viable")));
    }

    #[test]
    fn invalid_configs_rejected() {
        let layers = [layer(1, 1, 0.5)];
        let bad = RealityConfig {
            trials: 0,
            ..RealityConfig::default()
        };
        assert!(reality_check(&layers, &bad).is_err());
        let bad = RealityConfig {
            tolerance: 1.5,
            ..RealityConfig::default()
        };
        assert!(reality_check(&layers, &bad).is_err());
        let mut cfg = RealityConfig::default();
        cfg.model.component_tolerance = -0.1;
        assert!(reality_check(&layers, &cfg).is_err());
        assert!(reality_check(&[], &RealityConfig::default()).is_err());
    }
}
