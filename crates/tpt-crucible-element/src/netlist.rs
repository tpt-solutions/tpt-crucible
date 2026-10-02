//! TPT-IR weights → physical electrical components → SPICE netlist
//! (`spec2.txt` §3.3).
//!
//! A matrix multiply is the natural atom of analog compute: one weight, one
//! multiply-accumulate. The netlist models each [`Op::MatMul`] layer as a
//! bank of **voltage-controlled current sources** (SPICE `G` elements) into
//! per-row summing nodes:
//!
//! ```text
//! V(out_i) = R_load · Σ_j W[i][j] · V(in_j)
//! ```
//!
//! with the transconductance of source `(i, j)` set to `W[i][j] / R_load`, so
//! the load resistor converts the summed row current back to a voltage. This
//! is exactly the math a crossbar/MAC array implements in silicon (or, with
//! memristive devices, in the analog domain), which makes the emitted
//! netlist simulable by ngspice/Xyce as-is: `.op` computes the DC operating
//! point of a forward pass with all inputs at 0 V.
//!
//! Floating activations/nonlinearities are *not* lowered here — activation
//! circuits are emulated by the Reality Check's numeric model and flagged in
//! comments. Weights must be float tensors (`f32`/`f16`/`bf16`); quantized
//! payloads need a dequantization pass first.

use std::fmt::Write as _;

use tpt_crucible_common::error::{Error, Result};
use tpt_crucible_common::{Graph, NodeId, Op};

/// One extracted matrix-multiply layer.
#[derive(Debug, Clone, PartialEq)]
pub struct MacLayer {
    /// Consumer node name (`blk.0.attn_q` style).
    pub name: String,
    /// The consuming MatMul IR node.
    pub node: NodeId,
    /// Rows of the weight matrix (output features).
    pub rows: usize,
    /// Columns of the weight matrix (input features).
    pub cols: usize,
    /// Row-major weights, length `rows * cols`.
    pub weights: Vec<f32>,
}

impl MacLayer {
    /// Weight value at `(row, col)`.
    pub fn at(&self, row: usize, col: usize) -> f32 {
        self.weights[row * self.cols + col]
    }
}

/// Netlist rendering knobs.
#[derive(Debug, Clone)]
pub struct NetlistOptions {
    /// Title baked into the header comment.
    pub title: String,
    /// Analysis temperature (°C) emitted for simulators.
    pub temperature_c: f64,
    /// Summing-node load resistance in kΩ; scales transconductances so tiny
    /// weights stay numerically sane in SPICE units.
    pub load_kohm: f64,
    /// Append an ngspice `.control` block that runs the operating point and
    /// dumps every row-sum voltage to `sim.csv` (consumed by
    /// [`crate::simulator::run_ngspice`]). Replaces the bare `.op` line.
    pub export_node_voltages: bool,
}

impl Default for NetlistOptions {
    fn default() -> Self {
        Self {
            title: "tpt-crucible-element analog synthesis".into(),
            temperature_c: 27.0,
            load_kohm: 10.0,
            export_node_voltages: false,
        }
    }
}

/// Extract every MatMul layer whose weights are float constants.
///
/// Layers come out in SSA order; a MatMul fed multiple constants uses its
/// first float constant as the weight matrix (a second, 1-D constant is
/// treated as bias and noted in a comment).
///
/// # Errors
/// * [`Error::UnsupportedOperation`] when no MatMul has float weights at all,
///   or a candidate weight tensor is not 2-D,
/// * [`Error::InvalidArgument`] when the IR is structurally invalid.
pub fn extract_mac_layers(graph: &Graph) -> Result<Vec<MacLayer>> {
    graph.validate()?;

    let mut layers = Vec::new();
    let mut saw_float_weight = false;
    for (i, node) in graph.nodes.iter().enumerate() {
        if !matches!(node.op, Op::MatMul) {
            continue;
        }
        // First float constant input is the weight matrix.
        let weight_input = node.inputs.iter().copied().find(|&inp| {
            matches!(
                graph.get_node(inp).map(|n| &n.op),
                Some(Op::Constant { tensor }) if tensor.desc.dtype.is_float()
            )
        });
        let Some(wid) = weight_input else {
            continue;
        };
        saw_float_weight = true;
        let Some(Op::Constant { tensor }) = graph.get_node(wid).map(|n| &n.op) else {
            unreachable!("filtered above");
        };
        let shape = &tensor.desc.shape;
        if shape.len() != 2 {
            return Err(Error::UnsupportedOperation(format!(
                "weight `{}` feeding `{}` is {}-dimensional; netlists need 2-D matrices",
                graph.get_node(wid).unwrap().name,
                node.name,
                shape.len()
            )));
        }
        layers.push(MacLayer {
            name: node.name.clone(),
            node: NodeId(i as u32),
            rows: shape[0],
            cols: shape[1],
            weights: tensor.data.to_f32(&tensor.desc)?,
        });
    }

    if !saw_float_weight {
        return Err(Error::UnsupportedOperation(
            "no MatMul with float weights found; ingest a floating-point \
             checkpoint or dequantize first"
                .into(),
        ));
    }
    Ok(layers)
}

/// Render the full SPICE netlist for the extracted layers.
///
/// Node naming: inputs `in_<j>` (shared across layers — a forward pass wires
/// layer outputs into the next layer's inputs), row sums `out_<layer>_<i>`.
///
/// # Errors
/// [`Error::InvalidArgument`] when `load_kohm` is not finite positive.
pub fn render(layers: &[MacLayer], opts: &NetlistOptions) -> Result<String> {
    if !opts.load_kohm.is_finite() || opts.load_kohm <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "load_kohm must be finite positive, got {}",
            opts.load_kohm
        )));
    }

    let mut out = String::with_capacity(4096 + layers.len() * 512);
    out.push_str(&format!("* {}\n", opts.title));
    out.push_str("* generated by tpt-crucible-element (Phase 3 physics engine)\n");
    out.push_str("* model: V(out_i) = R_load * sum_j W[i][j] * V(in_j); g_ij = W[i][j] / R_load\n");
    out.push_str(&format!(".temp {:.1}\n", opts.temperature_c));
    out.push('\n');

    // Drive sources for the widest input width (0 V = neutral input).
    let max_cols = layers.iter().map(|l| l.cols).max().unwrap_or(0);
    if max_cols > 0 {
        out.push_str("* --- external drive (DC sources; 0 V = neutral input) ---\n");
        for j in 0..max_cols {
            out.push_str(&format!("Vin_{j} in_{j} 0 0\n"));
        }
        out.push('\n');
    }

    let load_ohms = opts.load_kohm * 1_000.0;
    for (li, layer) in layers.iter().enumerate() {
        out.push_str(&format!(
            "* --- MAC layer {li} `{}` ({}x{}) ---\n",
            layer.name, layer.rows, layer.cols
        ));
        for i in 0..layer.rows {
            let node = format!("out_{li}_{i}");
            // Load resistor converts the summed row current to a voltage.
            out.push_str(&format!("Rl_{li}_{i} {node} 0 {:.3}k\n", opts.load_kohm));
            for j in 0..layer.cols {
                let w = f64::from(layer.at(i, j));
                let g = w / load_ohms; // siemens
                out.push_str(&format!("G_{li}_{i}_{j} {node} 0 in_{j} 0 {:.6e}\n", g));
            }
        }
        out.push('\n');
    }

    out.push_str("* activations/normalizations are digital glue (see reality.rs)\n");
    if opts.export_node_voltages {
        // ngspice batch scripting: run the op and dump row voltages, then
        // quit so `-b` returns immediately.
        out.push_str(".control\n");
        out.push_str("op\n");
        out.push_str("wrdata sim.csv");
        for (li, layer) in layers.iter().enumerate() {
            for i in 0..layer.rows {
                let _ = write!(out, " v(out_{li}_{i})");
            }
        }
        out.push('\n');
        out.push_str("quit\n");
        out.push_str(".endc\n");
    } else {
        out.push_str(".op\n");
    }
    out.push_str(".end\n");
    Ok(out)
}

/// One-shot helper: extract from `graph` and render with `opts`.
///
/// # Errors
/// See [`extract_mac_layers`] / [`render`].
pub fn to_spice_netlist(graph: &Graph, opts: &NetlistOptions) -> Result<String> {
    let layers = extract_mac_layers(graph)?;
    render(&layers, opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_crucible_common::{DType, Tensor, TensorDesc};

    fn matmul_graph(rows: usize, cols: usize) -> Graph {
        let mut g = Graph::new("analog-mlp");
        let x = g.push(
            "x",
            Op::Input(TensorDesc::new(vec![1, rows], DType::F32)),
            Vec::<_>::new(),
        );
        let w = g.push(
            "blk0_wq",
            Op::Constant {
                tensor: Tensor::from_f32(vec![rows, cols], &vec![0.5; rows * cols]),
            },
            Vec::<_>::new(),
        );
        let y = g.push("blk0_attn_q", Op::MatMul, vec![x, w]);
        let o = g.push("out", Op::Output { name: "y".into() }, vec![y]);
        g.mark_output(o);
        g
    }

    #[test]
    fn extraction_finds_float_matmul_layers() {
        let g = matmul_graph(2, 3);
        let layers = extract_mac_layers(&g).unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].name, "blk0_attn_q");
        assert_eq!((layers[0].rows, layers[0].cols), (2, 3));
        assert_eq!(layers[0].at(1, 2), 0.5);
    }

    #[test]
    fn quantized_weights_are_rejected_with_guidance() {
        let mut g = Graph::new("quant");
        let x = g.push(
            "x",
            Op::Input(TensorDesc::new(vec![1, 1], DType::F32)),
            Vec::<_>::new(),
        );
        let mut t = Tensor::from_f32(vec![1, 1], &[1.0]);
        t.desc.dtype = DType::Q8_0;
        let w = g.push("w", Op::Constant { tensor: t }, Vec::<_>::new());
        let y = g.push("mm", Op::MatMul, vec![x, w]);
        let o = g.push("out", Op::Output { name: "y".into() }, vec![y]);
        g.mark_output(o);
        let err = extract_mac_layers(&g).unwrap_err();
        assert!(err.to_string().contains("dequantize"), "{err}");
    }

    #[test]
    fn netlist_is_structurally_valid_spice() {
        let g = matmul_graph(2, 3);
        let text = to_spice_netlist(&g, &NetlistOptions::default()).unwrap();

        assert!(text.starts_with("* tpt-crucible-element analog synthesis"));
        assert!(text.trim_end().ends_with(".end"));
        assert!(text.contains(".op"));
        assert!(text.contains(".temp 27.0"));

        // Drive sources for every input column.
        for j in 0..3 {
            assert!(text.contains(&format!("Vin_{j} in_{j} 0 0")));
        }
        // Load resistor and transconductance values.
        assert!(text.contains("Rl_0_0 out_0_0 0 10.000k"));
        let g_line = text.lines().find(|l| l.starts_with("G_0_0_0 ")).unwrap();
        let gain: f64 = g_line.split_whitespace().last().unwrap().parse().unwrap();
        assert!((gain - 0.5 / 10_000.0).abs() < 1e-12, "{g_line}");

        // Device/directive lines start in column one (SPICE requirement).
        for line in text.lines() {
            if line.is_empty() || line.starts_with('*') || line.starts_with(['R', 'G', 'V', '.']) {
                continue;
            }
            panic!("unexpected non-device line: {line:?}");
        }
    }

    #[test]
    fn invalid_load_resistance_rejected() {
        let g = matmul_graph(1, 1);
        let opts = NetlistOptions {
            load_kohm: -1.0,
            ..Default::default()
        };
        let err = to_spice_netlist(&g, &opts).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn three_layer_analog_mlp_renders_all_rows() {
        // The spec milestone shape: a small 3-layer analog NN.
        let mut g = Graph::new("three-layer");
        let mut cur = g.push(
            "x",
            Op::Input(TensorDesc::new(vec![1, 4], DType::F32)),
            Vec::<_>::new(),
        );
        for l in 0..3 {
            let w = g.push(
                format!("fc{l}_w"),
                Op::Constant {
                    tensor: Tensor::from_f32(vec![4, 4], &[0.25; 16]),
                },
                Vec::<_>::new(),
            );
            cur = g.push(format!("fc{l}"), Op::MatMul, vec![cur, w]);
        }
        let o = g.push("out", Op::Output { name: "y".into() }, vec![cur]);
        g.mark_output(o);

        let layers = extract_mac_layers(&g).unwrap();
        assert_eq!(layers.len(), 3);
        let text = render(&layers, &NetlistOptions::default()).unwrap();
        assert_eq!(text.matches("Rl_").count(), 12, "4 rows × 3 layers");
        assert_eq!(
            text.lines().filter(|l| l.starts_with('G')).count(),
            48,
            "rows × cols × layers"
        );
    }
}
