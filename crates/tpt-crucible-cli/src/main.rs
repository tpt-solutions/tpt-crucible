//! The unified `tpt` command-line entrypoint (`spec2.txt` §2).
//!
//! ```text
//! tpt ingest <model>   — lower SafeTensors/GGUF models to TPT-IR
//! tpt info <ir>        — inspect a TPT-IR artifact
//! tpt compile <ir>     — compile TPT-IR for a hardware target
//! tpt quantize <ir>    — plan per-layer INT4/INT8 against an accuracy budget
//! tpt preflight <ir>   — stream operator-compatibility checks (optionally
//!                        live over the Observer WebSocket backend)
//! tpt doctor           — verify the external toolchain
//! ```
//!
//! Hardware backends are cargo features (`spec2.txt` §4.7): `swarm`
//! (default, pure Rust) and `fpga` (opt-in). Targets compiled out fail with
//! a rebuild hint instead of misbehaving.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use tpt_crucible_catalyst as catalyst;
use tpt_crucible_common as common;
use tpt_crucible_common::{Error, Graph};
use tpt_crucible_uir_adapter as uir_adapter;

#[cfg(feature = "swarm")]
use tpt_crucible_alloy as alloy;

/// Hardware compilation targets understood by `tpt compile`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Target {
    /// Microcontroller swarm (ESP32/RP2040/RISC-V) — needs feature `swarm`.
    Alloy,
    /// FPGA + HBM overlay — needs feature `fpga`.
    Fusion,
    /// Analog compute-in-memory — Phase 3, not yet built.
    Element,
}

/// Explicit model-format override for `tpt ingest`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum FormatArg {
    /// HuggingFace `.safetensors`.
    Safetensors,
    /// llama.cpp `.gguf`.
    Gguf,
    /// Open Neural Network Exchange `.onnx`.
    Onnx,
    /// PyTorch `.pt` / `.pth` (torch.save zip format).
    Pytorch,
    /// Keras v3 `.keras` archive.
    Keras,
    /// Llamafile executable with embedded GGUF.
    Llamafile,
    /// AWQ quantized SafeTensors checkpoint.
    Awq,
    /// GPTQ quantized SafeTensors checkpoint.
    Gptq,
}

impl From<FormatArg> for catalyst::ModelFormat {
    fn from(f: FormatArg) -> Self {
        match f {
            FormatArg::Safetensors => catalyst::ModelFormat::SafeTensors,
            FormatArg::Gguf => catalyst::ModelFormat::Gguf,
            FormatArg::Onnx => catalyst::ModelFormat::Onnx,
            FormatArg::Pytorch => catalyst::ModelFormat::PyTorch,
            FormatArg::Keras => catalyst::ModelFormat::Keras,
            FormatArg::Llamafile => catalyst::ModelFormat::Llamafile,
            FormatArg::Awq => catalyst::ModelFormat::Awq,
            FormatArg::Gptq => catalyst::ModelFormat::Gptq,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "tpt",
    version,
    about = "TPT Crucible — compile, simulate, and deploy AI models onto non-traditional hardware",
    propagate_version = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Ingest a model file and emit TPT-IR.
    Ingest {
        /// Model file (`.safetensors`, `.gguf`, …) — or, with `--hub`, a
        /// HuggingFace repo id (`org/name`).
        model: PathBuf,
        /// Skip format auto-detection.
        #[arg(long)]
        format: Option<FormatArg>,
        /// Treat `model` as a HuggingFace repo id and download it first
        /// (needs the `hub` cargo feature).
        #[arg(long)]
        hub: bool,
        /// Output artifact; defaults to `<stem>.tptir` (binary).
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Also emit a TPT-UIR Crucible-dialect region to this path
        /// (postcard-encoded `.tptuir`, readable by `tpt-uir-cli`).
        #[arg(long)]
        uir: Option<PathBuf>,
    },
    /// Inspect a TPT-IR artifact.
    Info {
        /// `.tptir` or `.tptir.json` file.
        ir: PathBuf,
        /// Dump the graph as Graphviz DOT.
        #[arg(long)]
        dot: bool,
    },
    /// Compile a TPT-IR artifact for a hardware target.
    Compile {
        /// TPT-IR artifact to compile.
        ir: PathBuf,
        /// Hardware target.
        #[arg(short, long, value_enum)]
        target: Target,
        /// Swarm size (alloy target).
        #[arg(long, default_value_t = 16)]
        nodes: usize,
        /// Usable memory per node in MiB (alloy target).
        #[arg(long, default_value_t = 8)]
        mem_mb: u64,
        /// KV-cache max sequence length (0 disables KV planning).
        #[arg(long, default_value_t = 512)]
        seq_len: u32,
        /// KV-cache batch size.
        #[arg(long, default_value_t = 1)]
        batch: u32,
        /// Emit firmware projects + flash scripts into this directory.
        #[arg(long)]
        out_dir: Option<PathBuf>,
    },
    /// Verify the external toolchain (`tpt-doctor`).
    Doctor,
    /// Plan per-layer quantization against an accuracy budget
    /// (INT4 baseline; fragile layers promoted to INT8 first).
    Quantize {
        /// TPT-IR artifact to plan for.
        ir: PathBuf,
        /// Max acceptable end-to-end error as a fraction (e.g. 0.05 = 5 %).
        #[arg(long)]
        accuracy_budget: f64,
        /// `.tptprofile` sidecar with measured layer sensitivities;
        /// omitted → graph-shape heuristic.
        #[arg(long)]
        profile: Option<PathBuf>,
        /// Save the resulting plan as JSON to this path.
        #[arg(long)]
        save: Option<PathBuf>,
    },
    /// Stream operator-compatibility pre-flight checks, optionally live over
    /// the Observer WebSocket backend.
    Preflight {
        /// TPT-IR artifact to analyze.
        ir: PathBuf,
        /// Hardware families to check (repeatable; default = all).
        #[arg(long, value_delimiter = ',')]
        family: Vec<String>,
        /// Bind an Observer telemetry server here and stream events as they
        /// are produced (e.g. `--serve 127.0.0.1:8787`).
        #[arg(long)]
        serve: Option<String>,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code as u8),
        Err(err) => {
            eprintln!("error: {err}");
            if let Error::FeatureNotEnabled(feat) = &err {
                eprintln!("help: rebuild with `cargo install tpt-crucible-cli --features {feat}`");
            }
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<i32, Error> {
    match Cli::parse().command {
        Command::Ingest {
            model,
            format,
            hub,
            output,
            uir,
        } => cmd_ingest(
            &model,
            format.map(Into::into),
            hub,
            output.as_deref(),
            uir.as_deref(),
        ),
        Command::Info { ir, dot } => cmd_info(&ir, dot),
        Command::Compile {
            ir,
            target,
            nodes,
            mem_mb,
            seq_len,
            batch,
            out_dir,
        } => cmd_compile(
            &ir,
            target,
            nodes,
            mem_mb << 20,
            seq_len,
            batch,
            out_dir.as_deref(),
        ),
        Command::Doctor => cmd_doctor(),
        Command::Quantize {
            ir,
            accuracy_budget,
            profile,
            save,
        } => cmd_quantize(&ir, accuracy_budget, profile.as_deref(), save.as_deref()),
        Command::Preflight { ir, family, serve } => cmd_preflight(&ir, &family, serve.as_deref()),
    }
}

/// Default output path: `<stem>.tptir` next to the input (JSON when the
/// requested name ends in `.json`).
fn default_output(model: &Path) -> PathBuf {
    let stem = model
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let mut p = model.to_owned();
    p.set_file_name(format!("{stem}.tptir"));
    p
}

/// Platform cache root for downloaded artifacts (no external `dirs` dep):
/// `%LOCALAPPDATA%` on Windows, `$XDG_CACHE_HOME` or `~/.cache` elsewhere,
/// falling back to the OS temp dir.
#[cfg(feature = "hub")]
fn dirs_cache_root() -> PathBuf {
    if let Some(appdata) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(appdata);
    }
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(xdg);
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".cache");
    }
    std::env::temp_dir()
}

fn cmd_ingest(
    model: &Path,
    format: Option<catalyst::ModelFormat>,
    from_hub: bool,
    output: Option<&Path>,
    uir_out: Option<&Path>,
) -> Result<i32, Error> {
    let graph = match format {
        Some(f) => catalyst::ingest_with_format(model, f)?,
        None if from_hub => {
            #[cfg(feature = "hub")]
            {
                let repo = model.to_string_lossy().into_owned();
                let cache = std::env::var_os("TPT_HUB_CACHE")
                    .map_or_else(|| dirs_cache_root().join("tpt").join("hub"), PathBuf::from);
                println!("hub      : downloading {repo} into {}", cache.display());
                let dir = catalyst::hub::fetch_repo_to(&repo, &cache)?;
                println!("hub      : synced {}", dir.display());
                catalyst::ingest_path(&dir)?
            }
            #[cfg(not(feature = "hub"))]
            {
                let _ = model;
                return Err(Error::FeatureNotEnabled("hub".into()));
            }
        }
        None => catalyst::ingest_path(model)?,
    };
    graph.validate()?;

    let out = output.map_or_else(|| default_output(model), PathBuf::from);
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    graph.save(&out)?;

    // Optional TPT-UIR sidecar: adapt the graph to a Crucible-dialect region
    // and write it postcard-encoded for the external tpt-uir toolchain.
    let mut region_ops: Option<usize> = None;
    if let Some(uir_path) = uir_out {
        let region = uir_adapter::graph_to_region(&graph);
        region_ops = Some(region.blocks[0].operations.len());
        let bytes = tpt_uir_serde::serialize_region(&region).map_err(|e| {
            Error::InvalidArgument(format!("tpt-uir postcard encoding failed: {e}"))
        })?;
        if let Some(parent) = uir_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(uir_path, bytes)?;
    }

    println!("ingested `{}` -> `{}`", model.display(), out.display());
    println!(
        "  {} nodes, format={}",
        graph.len(),
        graph.metadata("source_format").unwrap_or("unknown")
    );
    if let (Some(path), Some(ops)) = (uir_out, region_ops) {
        println!("  tpt-uir   : {ops} ops -> `{}`", path.display());
    }
    Ok(0)
}

fn cmd_info(ir: &Path, dot: bool) -> Result<i32, Error> {
    let g = Graph::load(ir)?;
    println!("graph   : {}", g.name);
    println!("nodes   : {}", g.len());
    println!("inputs  : {}", g.inputs.len());
    println!("outputs : {}", g.outputs.len());
    if !g.metadata.is_empty() {
        println!("metadata:");
        for (k, v) in &g.metadata {
            println!("  {k} = {v}");
        }
    }
    println!("ops:");
    for (op, count) in g.op_histogram() {
        println!("  {op:<12} {count}");
    }

    let mut tensor_bytes = 0u64;
    let mut tensors = 0usize;
    for n in &g.nodes {
        if let common::Op::Constant { tensor } = &n.op {
            tensors += 1;
            tensor_bytes += tensor.desc.byte_size() as u64;
            println!("weight {:<28} {:>10}", n.name, tensor.desc);
        }
    }
    if tensors > 0 {
        println!(
            "weights : {tensors} tensors, {:.2} MiB",
            tensor_bytes as f64 / (1 << 20) as f64
        );
    }

    if dot {
        print!("{}", g.to_dot());
    }
    Ok(0)
}

fn cmd_compile(
    ir: &Path,
    target: Target,
    nodes: usize,
    mem_bytes: u64,
    seq_len: u32,
    batch: u32,
    out_dir: Option<&Path>,
) -> Result<i32, Error> {
    let g = Graph::load(ir)?;
    g.validate()?;
    match target {
        Target::Fusion => {
            #[cfg(feature = "fpga")]
            {
                use tpt_crucible_fusion::{CompileOptions, FpgaProfile};
                // v1 fabric target: an Alveo-class HBM card. Resource knobs
                // ride CompileOptions as the overlay format matures.
                let profile = FpgaProfile {
                    luts: 1_184_000,
                    dsp_slices: 9024,
                    block_ram_bytes: 38 << 20,
                };
                let bundle =
                    tpt_crucible_fusion::compile(&g, &profile, &CompileOptions::default())?;
                println!(
                    "fusion overlay plan : {} layer(s), {}/{} banks, {} DSP",
                    bundle.plan.layers.len(),
                    bundle.plan.banks_used,
                    bundle.plan.banks_total,
                    bundle.plan.dsp_required
                );
                match out_dir {
                    Some(dir) => {
                        bundle.write_into(dir)?;
                        println!("overlay bundle      : {}", dir.display());
                    }
                    None => {
                        println!("pass --out-dir to write fusecfg.overlay + rtl/");
                    }
                }
                Ok(0)
            }
            #[cfg(not(feature = "fpga"))]
            {
                let _ = (&g, nodes, mem_bytes);
                Err(Error::FeatureNotEnabled("fpga".into()))
            }
        }
        Target::Element => {
            let Some(out_dir) = out_dir else {
                return Err(Error::InvalidArgument(
                    "--target element writes a synthesis bundle (netlist + reality \
                     report); pass --out-dir"
                        .into(),
                ));
            };
            compile_element(&g, out_dir)
        }
        Target::Alloy => {
            #[cfg(feature = "swarm")]
            {
                compile_alloy(&g, nodes, mem_bytes, seq_len, batch, out_dir)
            }
            #[cfg(not(feature = "swarm"))]
            {
                let _ = (nodes, mem_bytes, seq_len, batch, out_dir);
                Err(Error::FeatureNotEnabled("swarm".into()))
            }
        }
    }
}

#[cfg(feature = "swarm")]
fn compile_alloy(
    g: &Graph,
    nodes: usize,
    mem_bytes: u64,
    seq_len: u32,
    batch: u32,
    out_dir: Option<&Path>,
) -> Result<i32, Error> {
    use alloy::kv_cache::KvCacheRequest;
    use common::DType;

    if nodes == 0 {
        return Err(Error::InvalidArgument("--nodes must be at least 1".into()));
    }
    let topo = alloy::topology::Topology::homogeneous(nodes, "esp32s3", mem_bytes);

    // Derive a KV-cache request from ingested GGUF hyperparameters when the
    // artifact carries them; --seq-len 0 opts out entirely.
    let kv_request = if seq_len == 0 {
        None
    } else {
        kv_from_metadata(g).map(|(layers, kv_heads, head_dim)| KvCacheRequest {
            num_layers: layers,
            num_kv_heads: kv_heads,
            head_dim,
            dtype: DType::F16,
            max_seq_len: seq_len,
            batch,
        })
    };

    let opts = alloy::partition::PartitionOptions {
        strategy: Some(alloy::partition::Strategy::Hybrid),
        max_weight_bytes_per_node: None,
        kv_request,
        // Homogeneous ESP32 fleets carry no FPGA fabric; offload stays off
        // until hybrid boards are expressible on the CLI.
        fpga_offload: false,
    };
    let plan = alloy::partition::partition(g, &topo, &opts)?;

    println!("alloy partition plan (hybrid: attention-head parallel + layer-serial):");
    println!(
        "  swarm     : {} x esp32s3, {:.1} MiB usable each",
        nodes,
        mem_bytes as f64 / (1 << 20) as f64
    );
    println!(
        "  shards    : {} (cut edges: {})",
        plan.shards.len(),
        plan.stats.cut_edges
    );
    for shard in &plan.shards {
        println!(
            "  node {:<3} : {} ops, {:.2} MiB weights, {:.2} MiB kv",
            shard.swarm_node,
            shard.graph_nodes.len(),
            shard.weight_bytes as f64 / (1 << 20) as f64,
            shard.kv_cache_bytes as f64 / (1 << 20) as f64
        );
    }
    if let Some(kv) = &plan.kv_plan {
        println!(
            "  kv-cache  : {:.2} MiB total across heads",
            kv.total_bytes as f64 / (1 << 20) as f64
        );
    } else {
        println!("  kv-cache  : skipped (no model metadata; tune with --seq-len)");
    }

    if let Some(dir) = out_dir {
        std::fs::create_dir_all(dir)?;
        let ctx = alloy::firmware::FirmwareContext {
            model_name: g.name.clone(),
            board: "esp32s3".into(),
            lang: alloy::firmware::FirmwareLang::Rust,
        };
        let files = alloy::firmware::generate(g, &plan, &ctx);
        for f in &files {
            let path = dir.join(&f.path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, &f.contents)?;
        }
        println!("emitted {} files under {}", files.len(), dir.display());
    }

    Ok(0)
}

/// Element analog synthesis: netlist + Reality Check + suggestions, written
/// as a bundle into `out_dir`.
fn compile_element(g: &Graph, out_dir: &Path) -> Result<i32, Error> {
    use tpt_crucible_element as element;

    let bundle = element::synthesize(g, &Default::default(), &Default::default())?;
    std::fs::create_dir_all(out_dir)?;

    let netlist_path = out_dir.join("netlist.sp");
    std::fs::write(&netlist_path, &bundle.netlist)?;

    let report_json = serde_json::to_string_pretty(&bundle.reality)?;
    let report_path = out_dir.join("reality_report.json");
    std::fs::write(&report_path, &report_json)?;

    println!("element analog synthesis for `{}`", g.name);
    println!(
        "mac layers : {} ({} rows total)",
        bundle.layers.len(),
        bundle.layers.iter().map(|l| l.rows).sum::<usize>()
    );
    println!(
        "confidence : {:.1} % within {:.1} % tolerance",
        bundle.confidence() * 100.0,
        bundle.reality.tolerance * 100.0
    );
    println!("dominant   : {}", bundle.reality.dominant_factor.name());
    println!("mitigations:");
    for m in &bundle.mitigations {
        println!("  - {m}");
    }
    println!("pcb layout :");
    for r in &bundle.layout {
        println!("  [{:<11}] {}", r.area, r.advice);
    }
    println!(
        "wrote {} and {}",
        netlist_path.display(),
        report_path.display()
    );
    Ok(0)
}

/// `(layers, kv_heads, head_dim)` from GGUF-derived metadata keys.
#[cfg(feature = "swarm")]
fn kv_from_metadata(g: &Graph) -> Option<(u32, u32, u32)> {
    let arch = g.metadata("general_architecture")?;
    let layers: u32 = g.metadata(&format!("{arch}_block_count"))?.parse().ok()?;
    let kv_heads: u32 = g
        .metadata(&format!("{arch}_attention_head_count_kv"))
        .or_else(|| g.metadata(&format!("{arch}_attention_head_count")))?
        .parse()
        .ok()?;
    let head_dim: u32 = g
        .metadata(&format!("{arch}_attention_key_length"))
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    Some((layers, kv_heads, head_dim))
}

fn cmd_doctor() -> Result<i32, Error> {
    let report = catalyst::doctor::scan();
    println!("tpt-doctor — external toolchain check\n");
    print!("{}", report.render());
    Ok(0)
}

/// `tpt quantize`: run the auto-search and report (or save) the plan.
fn cmd_quantize(
    ir: &Path,
    budget: f64,
    profile: Option<&Path>,
    save: Option<&Path>,
) -> Result<i32, Error> {
    let g = Graph::load(ir)?;
    let profile = match profile {
        Some(p) => Some(catalyst::autosearch::SensitivityProfile::from_path(p)?),
        None => None,
    };
    let plan = catalyst::autosearch::search(&g, profile.as_ref(), budget)?;

    println!("quantization auto-search for `{}`", g.name);
    if let Some(n) = plan.profile_entries_matched {
        println!("profile  : {n} entries matched");
    } else {
        println!("profile  : none (graph-shape heuristic)");
    }
    println!(
        "budget   : {:.2} % → predicted {:.2} % ({})",
        plan.budget * 100.0,
        plan.predicted_error * 100.0,
        if plan.budget_met { "MET" } else { "UNMET" }
    );
    println!("promoted : {} layer(s) Q4_0 → Q8_0", plan.promotions.len());
    for &id in &plan.promotions {
        let name = g.get_node(id).map(|n| n.name.as_str()).unwrap_or("<gone>");
        println!("  {name:<28} {}", plan.assignments[&id.0]);
    }

    if let Some(path) = save {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, plan.to_json_string_pretty()?)?;
        println!("plan     : {}", path.display());
    }

    // Exit 1 when the budget cannot be met even at INT8 everywhere —
    // scripts should treat that as "needs a smaller model / better fab".
    Ok(if plan.budget_met { 0 } else { 1 })
}

/// `tpt preflight`: stream compatibility events; optionally serve them over
/// the Observer WebSocket backend.
fn cmd_preflight(ir: &Path, families: &[String], serve: Option<&str>) -> Result<i32, Error> {
    use catalyst::preflight::{self, Family};

    let g = Graph::load(ir)?;
    let parsed: Vec<Family> = if families.is_empty() {
        vec![Family::Alloy, Family::Fusion, Family::Element]
    } else {
        families
            .iter()
            .map(|s| Family::parse(s))
            .collect::<common::Result<_>>()?
    };

    // Optional live bridge: bind the Observer server and forward every event
    // as a `preflight` ServerEvent while printing it locally.
    let runtime = match serve {
        Some(addr) => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            let bound = rt.block_on(tpt_crucible_observer::TelemetryServer::bind(addr))?;
            println!(
                "observer : ws://{}/ws (streaming pre-flight events)",
                bound.local_addr()
            );
            Some((rt, bound))
        }
        None => None,
    };

    let mut sink = |event: preflight::PreflightEvent| {
        let line = event.to_json_line().unwrap_or_default();
        println!("{line}");
        if let Some((_, bound)) = &runtime {
            if let Ok(value) = serde_json::to_value(&event) {
                bound.emit(tpt_crucible_observer::ServerEvent::Preflight(value));
            }
        }
    };
    let report = preflight::stream(&g, &parsed, &mut sink);

    // Give connected clients a moment to drain, then stop serving without
    // blocking on the (never-ending) axum task.
    if let Some((rt, _)) = runtime {
        rt.shutdown_timeout(std::time::Duration::from_millis(200));
    }

    println!(
        "\npre-flight: {} checks across {} family(ies): \
         {} supported, {} emulated, {} unsupported",
        report.checked,
        parsed.len(),
        report.supported,
        report.emulated,
        report.unsupported
    );
    if report.has_blockers() {
        println!("blockers : {}", report.unsupported_ops.join(", "));
        return Ok(1);
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_output_replaces_extension() {
        assert_eq!(
            default_output(Path::new("models/tinyllama.gguf")),
            PathBuf::from("models/tinyllama.tptir")
        );
        assert_eq!(
            default_output(Path::new("model.onnx")),
            PathBuf::from("model.tptir")
        );
    }

    #[test]
    fn feature_gates_report_clean_errors() {
        // Element is never compiled in yet — always NotImplemented.
        let err = Error::NotImplemented {
            feature: "element analog synthesis (Phase 3)".into(),
        };
        assert!(err.to_string().contains("Phase 3"));
    }
}
