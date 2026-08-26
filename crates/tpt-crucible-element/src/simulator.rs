//! ngspice orchestration (`spec2.txt` section 3.3, "FFI bindings to Xyce or
//! ngspice").
//!
//! v1 drives **ngspice in batch mode** (`-b`) rather than binding the C API:
//! the netlist renderer can emit a `.control` block that runs the operating
//! point and dumps row voltages to `sim.csv`
//! ([`NetlistOptions::export_node_voltages`](crate::netlist::NetlistOptions::export_node_voltages)),
//! and this module shells out, waits, and parses that file. Same automation
//! posture as the Fusion toolchain wrappers: missing binaries surface as
//! structured [`Error::ExternalTool`] values with install hints.
//!
//! A shared-library FFI (so sweeps run in-process without process spawn cost)
//! stays tracked in todo.md; the result contract below is what that FFI will
//! feed too.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use tpt_crucible_common::error::{Error, Result};

/// Which SPICE engine to drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimulatorKind {
    /// ngspice batch mode (`-b`), the default integration target.
    Ngspice,
    /// Xyce parallel simulator - same orchestration flow; command-line
    /// adapter lands with hardware validation.
    Xyce,
}

impl SimulatorKind {
    /// Default executable name on `PATH`.
    pub fn default_executable(self) -> &'static str {
        match self {
            Self::Ngspice => "ngspice",
            Self::Xyce => "Xyce",
        }
    }

    /// Canonical lowercase name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Ngspice => "ngspice",
            Self::Xyce => "xyce",
        }
    }
}

/// Configuration for one simulation run.
#[derive(Debug, Clone)]
pub struct SimulatorConfig {
    /// Engine to drive.
    pub kind: SimulatorKind,
    /// Explicit binary path; defaults to probing `PATH`.
    pub executable: Option<PathBuf>,
    /// Directory the netlist/CSV live in (defaults to the netlist's parent).
    pub work_dir: Option<PathBuf>,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            kind: SimulatorKind::Ngspice,
            executable: None,
            work_dir: None,
        }
    }
}

/// Parsed operating-point results: one voltage per requested node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimResult {
    /// `(node name, voltage)` in request order.
    pub node_voltages: Vec<(String, f64)>,
    /// Tail of the simulator's stdout (diagnostics).
    pub log_tail: String,
    /// Executable that produced the run.
    pub engine: String,
}

/// Parse ngspice `wrdata` CSV into `(node, voltage)` pairs.
///
/// `wrdata` writes two columns per requested variable (sweep value then
/// value); a DC operating point has a single sweep row per variable. Nodes
/// are matched to `requested` by column order.
///
/// # Errors
/// [`Error::ParseFormat`] when the CSV is empty or holds non-numeric data.
pub fn parse_wrdata(csv: &str, requested: &[String]) -> Result<Vec<(String, f64)>> {
    let mut rows = csv.lines().filter(|l| !l.trim().is_empty());
    let Some(line) = rows.next() else {
        return Err(Error::ParseFormat {
            path: "<sim.csv>".into(),
            format: "wrdata".into(),
            reason: "file is empty".into(),
        });
    };
    // Guard against multi-row exports (transient sweeps): op points are 1 row.
    let values: Vec<f64> = line
        .split(',')
        .map(|c| {
            c.trim().parse::<f64>().map_err(|e| Error::ParseFormat {
                path: "<sim.csv>".into(),
                format: "wrdata".into(),
                reason: format!("non-numeric column `{c}`: {e}"),
            })
        })
        .collect::<Result<_>>()?;

    // Column layout: for each variable i, columns [2i] = x and [2i+1] = y.
    let mut out = Vec::with_capacity(requested.len());
    for (i, name) in requested.iter().enumerate() {
        let idx = 2 * i + 1;
        let Some(&v) = values.get(idx) else {
            return Err(Error::ParseFormat {
                path: "<sim.csv>".into(),
                format: "wrdata".into(),
                reason: format!(
                    "expected {} value columns for {} variables, found {}",
                    requested.len() * 2,
                    requested.len(),
                    values.len()
                ),
            });
        };
        out.push((name.clone(), v));
    }
    Ok(out)
}

/// Node names a rendered netlist exports, in `wrdata` request order
/// (exposed for callers that build the request list themselves).
#[must_use]
pub fn requested_nodes(layers: &[crate::netlist::MacLayer]) -> Vec<String> {
    let mut names = Vec::new();
    for (li, layer) in layers.iter().enumerate() {
        for i in 0..layer.rows {
            names.push(format!("out_{li}_{i}"));
        }
    }
    names
}

/// Run ngspice over a netlist file and collect its operating point.
///
/// The netlist must have been rendered with the export block enabled
/// ([`NetlistOptions::export_node_voltages`](crate::netlist::NetlistOptions::export_node_voltages));
/// the `.control` block it emits is what produces `sim.csv` next to the
/// netlist.
///
/// # Errors
/// * [`Error::ExternalTool`] when the binary is missing or exits non-zero,
/// * [`Error::ParseFormat`] when `sim.csv` never appeared or did not parse,
/// * [`Error::Io`] for other filesystem failures.
pub fn run_ngspice(netlist: &Path, cfg: &SimulatorConfig) -> Result<SimResult> {
    // The node list comes from the netlist's own wrdata line (single source
    // of truth - no drift between renderer and runner).
    let text = std::fs::read_to_string(netlist)?;
    let requested = requested_nodes_from_netlist(&text)?;

    let work = match &cfg.work_dir {
        Some(d) => d.clone(),
        None => netlist.parent().unwrap_or(Path::new(".")).to_owned(),
    };
    let exe = cfg
        .executable
        .clone()
        .unwrap_or_else(|| PathBuf::from(cfg.kind.default_executable()));

    let out = Command::new(&exe)
        .arg("-b")
        .arg(netlist)
        .current_dir(&work)
        .output();

    let out = match out {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(Error::ExternalTool {
                tool: cfg.kind.name().into(),
                message: "not found on PATH - install ngspice (e.g. `apt install \
                          ngspice`) or set SimulatorConfig::executable"
                    .into(),
            });
        }
        Err(e) => {
            return Err(Error::ExternalTool {
                tool: cfg.kind.name().into(),
                message: format!("failed to launch: {e}"),
            });
        }
    };

    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    if !out.status.success() {
        let tail: String = stdout.lines().rev().take(8).collect::<Vec<_>>().join("\n");
        return Err(Error::ExternalTool {
            tool: cfg.kind.name().into(),
            message: format!("simulation failed (exit {:?}):\n{tail}", out.status.code()),
        });
    }

    let csv_path = work.join("sim.csv");
    let csv = std::fs::read_to_string(&csv_path).map_err(|e| Error::ParseFormat {
        path: csv_path.display().to_string(),
        format: "wrdata".into(),
        reason: format!("simulator did not produce results: {e}"),
    })?;

    let log_tail: String = stdout.lines().rev().take(5).collect::<Vec<_>>().join("\n");
    Ok(SimResult {
        node_voltages: parse_wrdata(&csv, &requested)?,
        log_tail,
        engine: exe.to_string_lossy().into_owned(),
    })
}

/// Extract the `v(...)` node list from a netlist's `wrdata sim.csv` line.
fn requested_nodes_from_netlist(text: &str) -> Result<Vec<String>> {
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with("wrdata"))
        .ok_or_else(|| Error::ParseFormat {
            path: "<netlist>".into(),
            format: "spice".into(),
            reason: "no wrdata export found; render with \
                     NetlistOptions::export_node_voltages = true"
                .into(),
        })?;
    Ok(line
        .split_whitespace()
        .skip(2) // "wrdata" + file name, then v(...) tokens
        .map(|tok| {
            tok.trim_start_matches("v(")
                .trim_end_matches(')')
                .to_owned()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn wrdata_parsing_pairs_columns_correctly() {
        // Two columns per variable (x then value); x is the constant sweep
        // index 0 for a DC operating point.
        let csv = "0.000000e+00,1.250000e-01,0.000000e+00,2.000000e+00,0.000000e+00,4.000000e+00\n";
        let got = parse_wrdata(csv, &req(&["out_0_0", "out_0_1", "out_1_0"])).unwrap();
        assert_eq!(
            got,
            vec![
                ("out_0_0".to_string(), 0.125),
                ("out_0_1".to_string(), 2.0),
                ("out_1_0".to_string(), 4.0),
            ]
        );
    }

    #[test]
    fn malformed_wrdata_is_a_parse_error() {
        let names = req(&["a", "b", "c"]);
        assert!(parse_wrdata("", &names).is_err());
        assert!(parse_wrdata("oops\n", &names).is_err());
        // Truncated row: missing value columns.
        assert!(parse_wrdata("0.0,1.0,0.0\n", &names).is_err());
    }

    #[test]
    fn netlist_node_extraction_reads_the_wrdata_line() {
        let netlist = "* header\n.op\n.control\nop\nwrdata sim.csv v(out_0_0) \
                       v(out_1_3)\nquit\n.endc\n.end\n";
        let nodes = requested_nodes_from_netlist(netlist).unwrap();
        assert_eq!(nodes, vec!["out_0_0", "out_1_3"]);
        // Without an export block the error names the option to enable.
        let err = requested_nodes_from_netlist("* x\n.op\n.end\n").unwrap_err();
        assert!(err.to_string().contains("export_node_voltages"), "{err}");
    }

    #[test]
    fn missing_binary_is_an_external_tool_error() {
        let dir = std::env::temp_dir().join(format!("element-sim-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let nl = dir.join("probe.sp");
        std::fs::write(
            &nl,
            ".control\nop\nwrdata sim.csv v(out_0_0)\nquit\n.endc\n.end\n",
        )
        .unwrap();

        let cfg = SimulatorConfig {
            executable: Some(dir.join("definitely-not-a-simulator.exe")),
            ..SimulatorConfig::default()
        };
        let err = run_ngspice(&nl, &cfg).unwrap_err();
        match err {
            Error::ExternalTool { tool, message } => {
                assert_eq!(tool, "ngspice");
                assert!(message.to_lowercase().contains("not found"), "{message}");
            }
            other => panic!("expected ExternalTool, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Full round trip against real ngspice - skipped on machines without
    /// the binary (`cargo test` stays green everywhere).
    #[test]
    fn live_ngspice_run_when_available() {
        if Command::new("ngspice").arg("-v").output().is_err() {
            eprintln!("skipping: ngspice not installed");
            return;
        }
        use tpt_crucible_common::{DType, Op, Tensor, TensorDesc};

        let mut g = tpt_crucible_common::Graph::new("live");
        let x = g.push(
            "x",
            Op::Input(TensorDesc::new(vec![1, 2], DType::F32)),
            Vec::<_>::new(),
        );
        let w = g.push(
            "w",
            Op::Constant {
                tensor: Tensor::from_f32(vec![2, 2], &[1.0, -1.0, 0.5, 2.0]),
            },
            Vec::<_>::new(),
        );
        let y = g.push("fc0", Op::MatMul, vec![x, w]);
        let o = g.push("out", Op::Output { name: "y".into() }, vec![y]);
        g.mark_output(o);

        let opts = crate::netlist::NetlistOptions {
            export_node_voltages: true,
            ..Default::default()
        };
        let netlist = crate::netlist::to_spice_netlist(&g, &opts).unwrap();
        let dir = std::env::temp_dir().join(format!("element-live-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let nl_path = dir.join("live.sp");
        std::fs::write(&nl_path, &netlist).unwrap();

        let result = run_ngspice(&nl_path, &SimulatorConfig::default()).unwrap();
        // All inputs sit at 0 V, so every row sum must be 0 V.
        assert_eq!(result.node_voltages.len(), 2);
        assert!(result.node_voltages.iter().all(|(_, v)| v.abs() < 1e-9));
        std::fs::remove_dir_all(&dir).ok();
    }
}
