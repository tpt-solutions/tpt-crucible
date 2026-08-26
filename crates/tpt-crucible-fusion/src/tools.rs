//! Yosys / Nextpnr wrappers (`spec2.txt` §3.2 "Implementation").
//!
//! Thin, silent orchestrators around the open-source FPGA toolchain via
//! `std::process::Command` — the automation posture of §4.6: tool failures
//! match a structured catalog ([`Error::ExternalTool`]) instead of leaking
//! raw process errors, and missing tools produce actionable install hints
//! (same philosophy as `tpt-doctor`, which probes these binaries).
//!
//! Commands are built through [`ToolRunner`] so tests can inject fakes; the
//! production runner just spawns and waits.
//!
//! v1 scope: **elaboration** (`yosys read_verilog/hierarchy/proc/check`),
//! which validates that patched overlay RTL still parses and resolves — the
//! check the overlay flow needs after each model load. Place-and-route sits
//! behind the same runner interface for when bitstream backends land.

use std::process::Command;

use tpt_crucible_common::error::{Error, Result};

/// How subprocesses are spawned; swap for fakes in tests.
pub trait ToolRunner {
    /// Run `program` with `args`; returns `(exit_code, stdout, stderr)` or an
    /// io error when the binary could not be launched.
    fn run(&mut self, program: &str, args: &[&str]) -> std::io::Result<(i32, String, String)>;
}

/// Production runner: real subprocesses, output captured.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessRunner;

impl ToolRunner for ProcessRunner {
    fn run(&mut self, program: &str, args: &[&str]) -> std::io::Result<(i32, String, String)> {
        let out = Command::new(program).args(args).output()?;
        Ok((
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        ))
    }
}

fn launch(
    tool: &'static str,
    hint: &str,
    outcome: std::io::Result<(i32, String, String)>,
) -> Result<String> {
    match outcome {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(Error::ExternalTool {
            tool: tool.into(),
            message: format!("not found on PATH - {hint}"),
        }),
        Err(e) => Err(Error::ExternalTool {
            tool: tool.into(),
            message: format!("failed to launch: {e}"),
        }),
        Ok((0, stdout, _)) => Ok(stdout),
        Ok((code, _, stderr)) => {
            let tail: String = stderr.lines().rev().take(8).collect::<Vec<_>>().join("\n");
            Err(Error::ExternalTool {
                tool: tool.into(),
                message: format!("failed (exit {code}):\n{tail}"),
            })
        }
    }
}

/// Elaborate `rtl_path` with Yosys: proves the overlay's patched RTL still
/// parses and resolves its hierarchy.
///
/// # Errors
/// * [`Error::ExternalTool`] naming yosys + an install hint when missing,
/// * [`Error::ExternalTool`] carrying the log tail on non-zero exit.
pub fn yosys_elaborate(rtl_path: &std::path::Path) -> Result<String> {
    yosys_elaborate_with(rtl_path, &mut ProcessRunner)
}

/// Like [`yosys_elaborate`] with an injected runner.
///
/// # Errors
/// Same as [`yosys_elaborate`].
pub fn yosys_elaborate_with(
    rtl_path: &std::path::Path,
    runner: &mut dyn ToolRunner,
) -> Result<String> {
    let script = format!(
        "read_verilog {}; hierarchy -top mac_array; proc; check",
        rtl_path.display()
    );
    let outcome = runner.run("yosys", &["-q", "-p", &script]);
    launch(
        "yosys",
        "install yosys (e.g. `apt install yosys`) or point PATH at your \
         OSS CAD Suite bin/",
        outcome,
    )
}

/// Run Nextpnr place-and-route for `device` on a JSON netlist.
///
/// Present for flow completeness; the overlay path skips it in steady state
/// (only the pre-synthesized shell is ever re-implemented), so this runs on
/// explicit request rather than inside `compile`.
///
/// # Errors
/// [`Error::ExternalTool`] with the same catalog semantics as
/// [`yosys_elaborate`].
pub fn nextpnr_route(device: &str, json_netlist: &std::path::Path) -> Result<String> {
    nextpnr_route_with(device, json_netlist, &mut ProcessRunner)
}

/// Like [`nextpnr_route`] with an injected runner.
///
/// # Errors
/// Same as [`nextpnr_route`].
pub fn nextpnr_route_with(
    device: &str,
    json_netlist: &std::path::Path,
    runner: &mut dyn ToolRunner,
) -> Result<String> {
    let outcome = runner.run(
        "nextpnr-xilinx",
        &[
            "--device",
            device,
            "--json",
            &json_netlist.to_string_lossy(),
        ],
    );
    launch(
        "nextpnr",
        "install nextpnr (OSS CAD Suite bundles the xilinx backend)",
        outcome,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fake runner replaying canned outcomes.
    struct Fake {
        code: i32,
        stdout: String,
        stderr: String,
        missing: bool,
    }

    impl ToolRunner for Fake {
        fn run(&mut self, _p: &str, _a: &[&str]) -> std::io::Result<(i32, String, String)> {
            if self.missing {
                Err(std::io::Error::new(std::io::ErrorKind::NotFound, "gone"))
            } else {
                Ok((self.code, self.stdout.clone(), self.stderr.clone()))
            }
        }
    }

    #[test]
    fn missing_yosys_is_an_actionable_external_tool_error() {
        let mut fake = Fake {
            code: 0,
            stdout: String::new(),
            stderr: String::new(),
            missing: true,
        };
        let err = yosys_elaborate_with(std::path::Path::new("mac_array.v"), &mut fake).unwrap_err();
        match err {
            Error::ExternalTool { tool, message } => {
                assert_eq!(tool, "yosys");
                assert!(message.contains("PATH"), "{message}");
            }
            other => panic!("expected ExternalTool, got {other:?}"),
        }
    }

    #[test]
    fn failing_elaboration_reports_the_log_tail() {
        let mut fake = Fake {
            code: 1,
            stdout: String::new(),
            stderr: "ERROR: syntax error\nline 42: boom\nmore context\n".into(),
            missing: false,
        };
        let err = yosys_elaborate_with(std::path::Path::new("broken.v"), &mut fake).unwrap_err();
        assert!(err.to_string().contains("exit 1"));
        assert!(err.to_string().contains("syntax error"));
    }

    #[test]
    fn successful_run_returns_stdout() {
        let mut fake = Fake {
            code: 0,
            stdout: "elaborated ok\n".into(),
            stderr: String::new(),
            missing: false,
        };
        let out = yosys_elaborate_with(std::path::Path::new("ok.v"), &mut fake).unwrap();
        assert!(out.contains("elaborated ok"));
    }

    #[test]
    fn nextpnr_missing_names_the_install_hint() {
        let mut fake = Fake {
            code: 0,
            stdout: String::new(),
            stderr: String::new(),
            missing: true,
        };
        let err =
            nextpnr_route_with("xcu280", std::path::Path::new("n.json"), &mut fake).unwrap_err();
        assert!(err.to_string().contains("nextpnr"));
        assert!(err.to_string().contains("OSS CAD Suite"));
    }
}
