//! The overlay resource planner: staging weights into BRAM banks and HBM
//! channels without resynthesizing logic (`spec2.txt` §3.2 "FPGA Overlay
//! Architecture").
//!
//! A pre-synthesized "AI overlay" already contains a fixed datapath — MAC
//! arrays wired to memory controllers. Compiling a *model* therefore means
//! deciding **where each weight lands** and **how the datapath is shaped**,
//! then writing that decision into a small `.fusecfg` configuration consumed
//! by the overlay's loader. No synthesis in the loop: this module runs in
//! microseconds, which is what turns hours of FPGA toolchain into the spec's
//! ~10 s per-model budget (the remainder being bitstream patching).
//!
//! # Resource model (v1)
//!
//! * Block RAM is modeled as uniform `bank_size_bytes` banks (default 2 KiB,
//!   one inferred BRAM block each) up to `profile.block_ram_bytes`.
//! * Each MAC layer's weights are staged contiguously; a layer spanning
//!   multiple banks records its bank list.
//! * Banks are grouped onto `hbm_channels` channels round-robin so streaming
//!   weight refresh spreads across HBM pseudo-channels.
//! * DSP slices: one multiplier per live MAC per cycle; v1 plans for full
//!   unrolled arrays and reports the requirement, leaving array folding to a
//!   later pass when the request exceeds the fabric.
//!
//! The plan is pure computation — deterministic, testable, and independent of
//! any vendor toolchain.

use serde::{Deserialize, Serialize};
use tpt_crucible_common::error::{Error, Result};
use tpt_crucible_common::Graph;

/// Fabric description targeted by an overlay compile.
///
/// This is the same shape [`tpt_crucible_alloy`]'s hybrid boards report via
/// `topology::FpgaProfile`; `From` conversions keep the two crates on one
/// contract (`todo.md`: "alloy<->fusion bridge").
///
/// [`tpt_crucible_alloy`]: https://docs.rs/tpt-crucible-alloy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FpgaProfile {
    /// Available look-up tables.
    pub luts: u32,
    /// Available DSP slices (hard multipliers).
    pub dsp_slices: u32,
    /// Total block RAM capacity in bytes.
    pub block_ram_bytes: u64,
}

impl From<tpt_crucible_alloy::topology::FpgaProfile> for FpgaProfile {
    fn from(p: tpt_crucible_alloy::topology::FpgaProfile) -> Self {
        Self {
            luts: p.luts,
            dsp_slices: p.dsp_slices,
            block_ram_bytes: p.block_ram_bytes,
        }
    }
}

/// One MAC layer staged into the overlay's weight memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StagedLayer {
    /// Source tensor/layer name (`blk.0.attn_q`).
    pub name: String,
    /// Rows (output features) of the weight matrix.
    pub rows: usize,
    /// Columns (input features) of the weight matrix.
    pub cols: usize,
    /// Weight payload bytes as staged (row-major, little-endian f32).
    pub weight_bytes: u64,
    /// Inclusive bank index range `[first_bank, last_bank]` holding the layer.
    pub banks: (u32, u32),
    /// HBM pseudo-channel assigned to stream refreshes for this layer.
    pub hbm_channel: u32,
}

/// The complete overlay placement produced by [`plan_overlay`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverlayPlan {
    /// Model the plan was computed for.
    pub model: String,
    /// Array geometry: rows × cols of the largest staged layer (v1 uses one
    /// geometry per overlay; mixed shapes are padded).
    pub array_rows: usize,
    pub array_cols: usize,
    /// Banks actually used out of the fabric's capacity.
    pub banks_used: u32,
    /// Total banks implied by `block_ram_bytes / bank_size_bytes`.
    pub banks_total: u32,
    /// Bytes of block RAM consumed.
    pub bram_bytes_used: u64,
    /// DSP slices the unrolled arrays require.
    pub dsp_required: u32,
    /// HBM channel count assumed by the channel assignment.
    pub hbm_channels: u32,
    /// Per-layer staging decisions, SSA order.
    pub layers: Vec<StagedLayer>,
}

impl OverlayPlan {
    /// Serialize to `.fusecfg` text (header comments + versioned JSON body).
    ///
    /// # Errors
    /// Serialization is infallible for this shape; the error exists for
    /// forward compatibility.
    pub fn to_fusecfg(&self) -> Result<String> {
        #[derive(Serialize)]
        struct FuseCfg<'a> {
            fusecfg_version: u32,
            #[serde(flatten)]
            plan: &'a OverlayPlan,
        }
        let mut out = String::from("# tpt-crucible-fusion .fusecfg v1\n");
        out.push_str("# Consumed by the pre-synthesized overlay loader.\n");
        out.push_str(&serde_json::to_string_pretty(&FuseCfg {
            fusecfg_version: 1,
            plan: self,
        })?);
        Ok(out)
    }
}

/// Knobs for [`plan_overlay`].
#[derive(Debug, Clone)]
pub struct PlanOptions {
    /// Uniform block size of one inferred BRAM bank (bytes).
    pub bank_size_bytes: u64,
    /// HBM pseudo-channels available for weight streaming.
    pub hbm_channels: u32,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            bank_size_bytes: 2048,
            hbm_channels: 8,
        }
    }
}

/// One float weight layer staged for the overlay (fusion-local extraction;
/// mirrors element's MAC extraction but keeps raw bytes for bank packing).
struct RawLayer {
    name: String,
    rows: usize,
    cols: usize,
    weights: Vec<f32>,
}

fn extract_layers(graph: &Graph) -> Result<Vec<RawLayer>> {
    use tpt_crucible_common::Op;

    graph.validate()?;
    let mut layers = Vec::new();
    for node in &graph.nodes {
        if !matches!(node.op, Op::MatMul) {
            continue;
        }
        let weight_input = node.inputs.iter().copied().find(|&inp| {
            matches!(
                graph.get_node(inp).map(|n| &n.op),
                Some(Op::Constant { tensor }) if tensor.desc.dtype.is_float()
            )
        });
        let Some(wid) = weight_input else { continue };
        let Some(Op::Constant { tensor }) = graph.get_node(wid).map(|n| &n.op) else {
            unreachable!("filtered above");
        };
        if tensor.desc.shape.len() != 2 {
            return Err(Error::UnsupportedOperation(format!(
                "weight `{}` feeding `{}` must be a 2-D matrix for overlay staging",
                graph.get_node(wid).unwrap().name,
                node.name
            )));
        }
        layers.push(RawLayer {
            name: node.name.clone(),
            rows: tensor.desc.shape[0],
            cols: tensor.desc.shape[1],
            weights: tensor.data.to_f32(&tensor.desc)?,
        });
    }
    if layers.is_empty() {
        return Err(Error::UnsupportedOperation(
            "no MatMul with float weights found; the overlay accelerates GEMM \
             workloads only"
                .into(),
        ));
    }
    Ok(layers)
}

/// Plan overlay staging for `graph` against `profile`.
///
/// Layers stage contiguously into BRAM banks; banks map round-robin onto HBM
/// channels. Fails with [`Error::OutOfMemory`] when the weights exceed the
/// fabric's block RAM, naming what did not fit.
///
/// # Errors
/// * [`Error::InvalidArgument`] when `bank_size_bytes` is zero or `profile`
///   capacities are zero,
/// * [`Error::UnsupportedOperation`] when no storable GEMM layer exists,
/// * [`Error::OutOfMemory`] when staging exceeds block RAM.
pub fn plan_overlay(
    graph: &Graph,
    profile: &FpgaProfile,
    options: &PlanOptions,
) -> Result<OverlayPlan> {
    if options.bank_size_bytes == 0 {
        return Err(Error::InvalidArgument(
            "bank_size_bytes must be positive".into(),
        ));
    }
    if profile.block_ram_bytes == 0 || profile.dsp_slices == 0 {
        return Err(Error::InvalidArgument(
            "fabric profile has no block RAM or DSP capacity".into(),
        ));
    }
    let layers = extract_layers(graph)?;

    let banks_total =
        u32::try_from(profile.block_ram_bytes / options.bank_size_bytes).unwrap_or(u32::MAX);
    let mut staged = Vec::with_capacity(layers.len());
    let mut bank_cursor: u64 = 0;
    let mut bram_used: u64 = 0;
    let mut dsp_required: u32 = 0;

    for l in &layers {
        let weight_bytes = (l.weights.len() * 4) as u64;
        let first = bank_cursor;
        let banks_needed = weight_bytes.div_ceil(options.bank_size_bytes);
        let last = first + banks_needed - 1;
        if last >= banks_total as u64 {
            return Err(Error::OutOfMemory {
                node: "fpga block ram".into(),
                needed: bram_used + weight_bytes,
                available: profile.block_ram_bytes,
            });
        }
        bank_cursor = last + 1;
        bram_used += weight_bytes;
        dsp_required = dsp_required.saturating_add(
            (l.rows.saturating_mul(l.cols))
                .try_into()
                .unwrap_or(u32::MAX),
        );
        staged.push(StagedLayer {
            name: l.name.clone(),
            rows: l.rows,
            cols: l.cols,
            weight_bytes,
            banks: (first as u32, last as u32),
            hbm_channel: (first as u32) % options.hbm_channels.max(1),
        });
    }

    let rows = layers.iter().map(|l| l.rows).max().unwrap_or(0);
    let cols = layers.iter().map(|l| l.cols).max().unwrap_or(0);

    Ok(OverlayPlan {
        model: graph.name.clone(),
        array_rows: rows,
        array_cols: cols,
        banks_used: bank_cursor as u32,
        banks_total,
        bram_bytes_used: bram_used,
        dsp_required,
        hbm_channels: options.hbm_channels,
        layers: staged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_crucible_common::{DType, Op, Tensor, TensorDesc};

    fn gemm_graph(layers: &[(usize, usize)]) -> Graph {
        // Parallel matmuls off one input: every layer's weight keeps its
        // declared [rows, cols] shape while staying shape-valid, even when
        // consecutive layers would not chain (rows_b != cols_a).
        let mut g = Graph::new("overlay-test");
        let x = g.push(
            "x",
            Op::Input(TensorDesc::new(vec![1, layers[0].0], DType::F32)),
            Vec::<_>::new(),
        );
        for (i, (rows, cols)) in layers.iter().enumerate() {
            let w = g.push(
                format!("fc{i}_w"),
                Op::Constant {
                    tensor: Tensor::from_f32(vec![*rows, *cols], &vec![0.5; rows * cols]),
                },
                Vec::<_>::new(),
            );
            let y = g.push(format!("fc{i}"), Op::MatMul, vec![x, w]);
            let o = g.push(
                format!("out{i}"),
                Op::Output {
                    name: format!("y{i}"),
                },
                vec![y],
            );
            g.mark_output(o);
        }
        g
    }

    fn fabric(bram_kib: u64) -> FpgaProfile {
        FpgaProfile {
            luts: 100_000,
            dsp_slices: 4096,
            block_ram_bytes: bram_kib * 1024,
        }
    }

    #[test]
    fn banks_stage_contiguously_and_round_robin_channels() {
        // Two 4x8 f32 layers = 128 B each; 2 KiB banks → one bank per layer.
        let g = gemm_graph(&[(4, 8), (4, 8)]);
        let plan = plan_overlay(&g, &fabric(64), &PlanOptions::default()).unwrap();
        assert_eq!(plan.layers.len(), 2);
        assert_eq!(plan.layers[0].banks, (0, 0));
        assert_eq!(plan.layers[1].banks, (1, 1));
        assert_eq!(plan.layers[0].hbm_channel, 0);
        assert_eq!(plan.layers[1].hbm_channel, 1);
        assert_eq!(plan.bram_bytes_used, 256);
        assert_eq!(plan.dsp_required, 64);
        assert_eq!((plan.array_rows, plan.array_cols), (4, 8));
    }

    #[test]
    fn oversized_layer_spans_multiple_banks() {
        // A 64x64 f32 layer is 16 KiB → 8 banks at 2 KiB.
        let g = gemm_graph(&[(64, 64)]);
        let plan = plan_overlay(&g, &fabric(64), &PlanOptions::default()).unwrap();
        assert_eq!(plan.layers[0].banks, (0, 7));
        assert_eq!(plan.banks_used, 8);
    }

    #[test]
    fn exceeding_bram_fails_with_out_of_memory() {
        let g = gemm_graph(&[(64, 64)]); // 16 KiB of weights
        let err = plan_overlay(&g, &fabric(8), &PlanOptions::default()).unwrap_err();
        assert!(
            matches!(
                err,
                Error::OutOfMemory { needed, available, .. } if needed > available
            ),
            "{err}"
        );
    }

    #[test]
    fn fusecfg_roundtrips_through_serde() {
        let g = gemm_graph(&[(4, 8)]);
        let plan = plan_overlay(&g, &fabric(64), &PlanOptions::default()).unwrap();
        let text = plan.to_fusecfg().unwrap();
        assert!(text.starts_with("# tpt-crucible-fusion .fusecfg v1"));
        assert!(text.contains("\"fusecfg_version\": 1"));

        // The JSON body parses back into an equivalent plan (skip headers).
        let json_start = text.find('{').unwrap();
        let back: OverlayPlan = serde_json::from_str(&text[json_start..]).unwrap();
        assert_eq!(back, plan);
    }

    #[test]
    fn alloy_fpga_profile_converts() {
        let alloy_profile = tpt_crucible_alloy::topology::FpgaProfile {
            luts: 42,
            dsp_slices: 7,
            block_ram_bytes: 1 << 20,
        };
        let mine: FpgaProfile = alloy_profile.into();
        assert_eq!(
            mine,
            FpgaProfile {
                luts: 42,
                dsp_slices: 7,
                block_ram_bytes: 1 << 20,
            }
        );
    }

    #[test]
    fn degenerate_inputs_rejected() {
        let g = gemm_graph(&[(4, 4)]);
        let opts = PlanOptions {
            bank_size_bytes: 0,
            ..PlanOptions::default()
        };
        assert!(plan_overlay(&g, &fabric(64), &opts).is_err());
        let dead = FpgaProfile {
            luts: 1,
            dsp_slices: 0,
            block_ram_bytes: 0,
        };
        assert!(plan_overlay(&g, &dead, &PlanOptions::default()).is_err());
    }

    #[test]
    fn graphs_without_gemm_are_rejected() {
        let mut g = Graph::new("no-gemm");
        let x = g.push(
            "x",
            Op::Input(TensorDesc::new(vec![1], DType::F32)),
            Vec::<_>::new(),
        );
        let o = g.push("out", Op::Output { name: "o".into() }, vec![x]);
        g.mark_output(o);
        let err = plan_overlay(&g, &fabric(64), &PlanOptions::default()).unwrap_err();
        assert!(err.to_string().contains("GEMM"), "{err}");
    }
}
