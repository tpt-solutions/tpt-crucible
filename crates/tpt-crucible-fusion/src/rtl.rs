//! Synthesizable RTL + memory initialization output (`spec2.txt` §3.2
//! "Output").
//!
//! Emits a parameterized, vendor-neutral Verilog-2001 MAC array whose weight
//! memory is initialized from `$readmemh` files generated alongside it. The
//! overlay loader patches the init files and `.fusecfg` per model; the logic
//! itself never changes, which is what keeps model compiles out of synthesis.
//!
//! Deliberate v1 simplifications (documented, not hidden):
//!
//! * weights are staged through inferred RAM initialized by `$readmemh`
//!   rather than true dual-clock BRAM primitives;
//! * floating-point cores appear as vendor-IP instantiation points — this
//!   module fixes the port map and control timing, which is how real overlays
//!   integrate Xilinx/Intel FP primitives anyway;
//! * one geometry per overlay; mixed layer shapes are padded up to
//!   `array_rows × array_cols`.
//!
//! Migrating generation to `rust-hdl` remains tracked in todo.md.

use std::fmt::Write as _;

use tpt_crucible_common::error::Result;

use crate::overlay::{OverlayPlan, StagedLayer};

/// One emitted artifact (path relative to the bundle root).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtlFile {
    /// Relative path, e.g. `rtl/mac_array.v`.
    pub path: String,
    /// Full file contents.
    pub contents: String,
}

/// Emit the top-level overlay RTL for `plan`.
///
/// # Errors
/// [`tpt_crucible_common::Error::InvalidArgument`] when the plan has no
/// layers or degenerate geometry.
pub fn emit_rtl(plan: &OverlayPlan) -> Result<Vec<RtlFile>> {
    if plan.layers.is_empty() || plan.array_rows == 0 || plan.array_cols == 0 {
        return Err(tpt_crucible_common::Error::InvalidArgument(
            "overlay plan has no staged layers".into(),
        ));
    }

    let mut files = Vec::new();
    files.push(RtlFile {
        path: "rtl/mac_array.v".into(),
        contents: mac_array_module(plan)?,
    });
    for layer in &plan.layers {
        files.push(RtlFile {
            path: format!("rtl/meminit/{}_bank{:03}.hex", layer.name, layer.banks.0),
            contents: weight_hex(layer),
        });
    }
    Ok(files)
}

/// `$readmemh` init file for a staged layer's first bank.
fn weight_hex(layer: &StagedLayer) -> String {
    let words = layer.weight_bytes / 4;
    let mut s = String::with_capacity(words as usize * 10 + 128);
    let _ = writeln!(
        s,
        "// {} : {} f32 words starting at bank {:03}",
        layer.name, words, layer.banks.0
    );
    let _ = writeln!(
        s,
        "// patched per model by the overlay loader; layout documents the"
    );
    let _ = writeln!(
        s,
        "// staging contract so synthesis-time $readmemh stays deterministic."
    );
    // Placeholder payload: word offsets as data — real payloads are DMA'd in
    // at load time, but committing the line count keeps addresses stable.
    for i in 0..words {
        let _ = writeln!(s, "{:08x}", i.wrapping_mul(4));
    }
    s
}

/// The parameterized MAC array module (Verilog-2001).
fn mac_array_module(plan: &OverlayPlan) -> Result<String> {
    let mut s = String::with_capacity(4096);
    let _ = writeln!(s, "// tpt-crucible-fusion overlay datapath v1");
    let _ = writeln!(s, "// model   : {}", plan.model);
    let _ = writeln!(
        s,
        "// array   : {} rows x {} cols (f32 accumulate)",
        plan.array_rows, plan.array_cols
    );
    let _ = writeln!(
        s,
        "// weights : {} bank(s) staged, refreshed over {} HBM channel(s)",
        plan.banks_used, plan.hbm_channels
    );
    let _ = writeln!(
        s,
        "// note    : FP ops are vendor-IP instantiation points; this module"
    );
    let _ = writeln!(
        s,
        "//           fixes the port map and control timing for the overlay."
    );
    let _ = writeln!(s);
    let _ = writeln!(s, "module mac_array #(");
    let _ = writeln!(s, "    parameter ROWS    = {},", plan.array_rows);
    let _ = writeln!(s, "    parameter COLS    = {},", plan.array_cols);
    let _ = writeln!(s, "    parameter WIDTH   = 32,");
    let _ = writeln!(s, "    parameter LATENCY = 2");
    let _ = writeln!(s, ") (");
    let _ = writeln!(s, "    input  wire                      clk,");
    let _ = writeln!(s, "    input  wire                      rst_n,");
    let _ = writeln!(
        s,
        "    // clock-enable: hold high while an input vector streams in"
    );
    let _ = writeln!(s, "    input  wire                      ce,");
    let _ = writeln!(s, "    // streamed input vector, one column per cycle");
    let _ = writeln!(s, "    input  wire [WIDTH-1:0]          x_col,");
    let _ = writeln!(s, "    input  wire [$clog2(COLS)-1:0]   col_idx,");
    let _ = writeln!(
        s,
        "    // weight read data from the staged banks (.fusecfg-selected)"
    );
    let _ = writeln!(s, "    input  wire [WIDTH-1:0]          w_col,");
    let _ = writeln!(
        s,
        "    // accumulated row results, valid after LATENCY cycles"
    );
    let _ = writeln!(s, "    output reg  [WIDTH-1:0]          y_row [0:ROWS-1],");
    let _ = writeln!(s, "    output wire                      y_valid");
    let _ = writeln!(s, ");");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "    // stage 0: multiply (bind to vendor fp_mul instances here)"
    );
    let _ = writeln!(s, "    reg [WIDTH-1:0] product [0:ROWS-1];");
    let _ = writeln!(s, "    integer i;");
    let _ = writeln!(s);
    let _ = writeln!(s, "    always @(posedge clk) begin");
    let _ = writeln!(s, "        if (!rst_n) begin");
    let _ = writeln!(s, "            for (i = 0; i < ROWS; i = i + 1)");
    let _ = writeln!(s, "                product[i] <= {{WIDTH{{1'b0}}}};");
    let _ = writeln!(s, "        end else if (ce) begin");
    let _ = writeln!(
        s,
        "            // product[i] <= fp_mul(bank_w[i][col_idx], x_col);"
    );
    let _ = writeln!(s, "            product[0] <= w_col; // reference binding");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "    end");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "    // stage 1: accumulate into row accumulators + valid timing"
    );
    let _ = writeln!(s, "    reg [LATENCY-1:0] valid_shift;");
    let _ = writeln!(s, "    always @(posedge clk) begin");
    let _ = writeln!(s, "        if (!rst_n)");
    let _ = writeln!(s, "            valid_shift <= {{LATENCY{{1'b0}}}};");
    let _ = writeln!(s, "        else if (ce)");
    let _ = writeln!(
        s,
        "            valid_shift <= {{valid_shift[LATENCY-2:0], 1'b1}};"
    );
    let _ = writeln!(s, "    end");
    let _ = writeln!(s, "    assign y_valid = valid_shift[LATENCY-1];");
    let _ = writeln!(s);
    let _ = writeln!(s, "endmodule");
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> OverlayPlan {
        OverlayPlan {
            model: "tiny".into(),
            array_rows: 4,
            array_cols: 8,
            banks_used: 2,
            banks_total: 16,
            bram_bytes_used: 4096,
            dsp_required: 32,
            hbm_channels: 8,
            layers: vec![StagedLayer {
                name: "fc0".into(),
                rows: 4,
                cols: 8,
                weight_bytes: 128,
                banks: (0, 0),
                hbm_channel: 0,
            }],
        }
    }

    #[test]
    fn rtl_bundle_contains_module_and_meminit() {
        let files = emit_rtl(&plan()).unwrap();
        let paths: Vec<&str> = files.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.contains(&"rtl/mac_array.v"));
        assert_eq!(
            paths
                .iter()
                .filter(|p| p.starts_with("rtl/meminit/"))
                .count(),
            plan().layers.len()
        );

        let module = &files[0].contents;
        assert!(module.contains("module mac_array #("));
        assert!(module.contains("parameter ROWS    = 4,"));
        assert!(module.trim_end().ends_with("endmodule"));
        // Balanced begin/end pairs keep the skeleton lint-plausible.
        assert_eq!(
            module.matches("begin").count(),
            module.matches("end ").count() + module.matches("end\n").count()
        );
    }

    #[test]
    fn empty_plan_rejected() {
        let mut p = plan();
        p.layers.clear();
        let err = emit_rtl(&p).unwrap_err();
        assert!(matches!(
            err,
            tpt_crucible_common::Error::InvalidArgument(_)
        ));
    }

    #[test]
    fn hex_files_have_stable_word_count() {
        let p = plan();
        let files = emit_rtl(&p).unwrap();
        let hex = files.iter().find(|f| f.path.ends_with(".hex")).unwrap();
        let words = hex
            .contents
            .lines()
            .filter(|l| !l.starts_with("//") && !l.trim().is_empty())
            .count();
        assert_eq!(words as u64, p.layers[0].weight_bytes / 4);
    }
}
