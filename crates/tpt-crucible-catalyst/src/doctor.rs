//! `tpt-doctor`: external toolchain discovery and verification.
//!
//! TPT Crucible orchestrates third-party tools (esptool for swarm flashing,
//! Yosys/Nextpnr for FPGA flows, KiCad for PCB export). Doctor scans `PATH`,
//! captures versions, and reports a structured [`DoctorReport`] so failures
//! surface as actionable messages instead of raw process errors
//! (`spec2.txt` §4.6, automation-first).

use std::process::Command;

/// Result of probing one external tool.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCheck {
    /// Canonical tool name.
    pub name: &'static str,
    /// Why it is needed.
    pub purpose: &'static str,
    /// Found + version string, or missing.
    pub status: ToolStatus,
}

/// Whether a tool was found on `PATH`.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolStatus {
    /// Executable found; first line of its version output.
    Found { version: String },
    /// Not found on `PATH`.
    Missing,
}

impl ToolStatus {
    /// True when the tool resolved.
    pub fn is_found(&self) -> bool {
        matches!(self, Self::Found { .. })
    }
}

/// Aggregate doctor output.
#[derive(Debug, Clone, Default)]
pub struct DoctorReport {
    /// One entry per probed tool.
    pub checks: Vec<ToolCheck>,
}

impl DoctorReport {
    /// Count of tools that resolved.
    pub fn found(&self) -> usize {
        self.checks.iter().filter(|c| c.status.is_found()).count()
    }

    /// Total probed tools.
    pub fn total(&self) -> usize {
        self.checks.len()
    }

    /// Render as aligned human-readable text (what `tpt doctor` prints).
    pub fn render(&self) -> String {
        let mut s = String::new();
        for c in &self.checks {
            match &c.status {
                ToolStatus::Found { version } => {
                    s.push_str(&format!("  [ok]      {:<12} {}\n", c.name, c.purpose));
                    if !version.is_empty() {
                        s.push_str(&format!("            `- {}\n", version.trim()));
                    }
                }
                ToolStatus::Missing => {
                    s.push_str(&format!(
                        "  [missing] {:<12} {} - install it to unlock this target\n",
                        c.name, c.purpose
                    ));
                }
            }
        }
        s.push_str(&format!(
            "\n  {}/{} tools available\n",
            self.found(),
            self.total()
        ));
        s
    }
}

const TOOLS: &[(&str, &str)] = &[
    ("python", "drives esptool / LiteX tooling"),
    ("esptool", "flashes ESP32 swarm nodes"),
    ("yosys", "FPGA synthesis (Fusion backend)"),
    ("nextpnr-ice40", "FPGA place & route (Fusion backend)"),
    ("kicad-cli", "PCB export (Element backend)"),
];

fn probe(name: &str) -> ToolStatus {
    // Version flags differ per tool; try the two common conventions.
    for flag in [["--version"], ["-V"]] {
        if let Some(version) = run_version(name, &flag) {
            return ToolStatus::Found { version };
        }
    }
    ToolStatus::Missing
}

fn run_version(bin: &str, flag: &[&str]) -> Option<String> {
    let out = Command::new(bin).args(flag).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let text = if text.trim().is_empty() {
        String::from_utf8_lossy(&out.stderr)
    } else {
        text
    };
    Some(text.lines().next().unwrap_or_default().trim().to_string())
}

/// Scan `PATH` for every known external tool and report versions.
///
/// This *is* the smoke test: a tool counts only when it executes and prints
/// a version banner.
pub fn scan() -> DoctorReport {
    let checks = TOOLS
        .iter()
        .map(|&(name, purpose)| ToolCheck {
            name,
            purpose,
            status: probe(name),
        })
        .collect();
    DoctorReport { checks }
}
