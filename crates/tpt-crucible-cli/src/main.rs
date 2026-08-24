//! The unified `tpt` command-line entrypoint (`spec2.txt` §2).
//!
//! ```text
//! tpt ingest <model>   — lower SafeTensors/GGUF models to TPT-IR
//! tpt info <ir>        — inspect a TPT-IR artifact
//! tpt compile <ir>     — compile TPT-IR for a hardware target
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
}

impl From<FormatArg> for catalyst::ModelFormat {
    fn from(f: FormatArg) -> Self {
        match f {
            FormatArg::Safetensors => catalyst::ModelFormat::SafeTensors,
            FormatArg::Gguf => catalyst::ModelFormat::Gguf,
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
        /// Model file (`.safetensors`, `.gguf`, …).
        model: PathBuf,
        /// Skip format auto-detection.
        #[arg(long)]
        format: Option<FormatArg>,
        /// Output artifact; defaults to `<stem>.tptir` (binary).
        #[arg(short, long)]
        output: Option<PathBuf>,
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
            output,
        } => cmd_ingest(&model, format.map(Into::into), output.as_deref()),
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

fn cmd_ingest(
    model: &Path,
    format: Option<catalyst::ModelFormat>,
    output: Option<&Path>,
) -> Result<i32, Error> {
    let graph = match format {
        Some(f) => catalyst::ingest_with_format(model, f)?,
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

    println!("ingested `{}` -> `{}`", model.display(), out.display());
    println!(
        "  {} nodes, format={}",
        graph.len(),
        graph.metadata("source_format").unwrap_or("unknown")
    );
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
                let bundle = tpt_crucible_fusion::compile(&g)?;
                println!("fusion overlay bundle: {} bytes", bundle.len());
                Ok(0)
            }
            #[cfg(not(feature = "fpga"))]
            {
                let _ = (&g, nodes, mem_bytes);
                Err(Error::FeatureNotEnabled("fpga".into()))
            }
        }
        Target::Element => Err(Error::NotImplemented {
            feature: "element analog synthesis (Phase 3)".into(),
        }),
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
