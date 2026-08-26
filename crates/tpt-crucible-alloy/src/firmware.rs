//! Per-node firmware project and master flashing script generation.
//!
//! Every shard gets a self-contained firmware skeleton (Rust by default â€”
//! memory-safe firmware is the whole point of the platform; C++ available)
//! plus a JSON manifest of exactly which tensors, ops, and KV-slice that
//! node owns. The coordinator gets parallel `esptool` flash scripts (`.sh`
//! + `.ps1`) so a 16-node fleet flashes over USB hubs in one command.
//!
//! Templates are string-built today; migrating them to `askama` templates
//! is tracked in `todo.md`.

use serde::{Deserialize, Serialize};

use crate::partition::PartitionPlan;

/// Firmware language to emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirmwareLang {
    /// Rust (`esp-hal` / `no_std`).
    Rust,
    /// C++ (`ESP-IDF`), for teams already invested there.
    Cxx,
}

impl FirmwareLang {
    /// Source file extension.
    pub fn ext(self) -> &'static str {
        match self {
            Self::Rust => "rs",
            Self::Cxx => "cpp",
        }
    }

    /// Language banner used in generated headers.
    pub fn banner(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Cxx => "c++",
        }
    }
}

/// Inputs describing the compile job.
#[derive(Debug, Clone)]
pub struct FirmwareContext {
    /// Model name baked into manifests.
    pub model_name: String,
    /// Target board string passed through to flash tooling.
    pub board: String,
    /// Source language.
    pub lang: FirmwareLang,
}

/// One emitted file (path relative to an output directory).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedFile {
    /// Relative path, e.g. `firmware/node_00/src/main.rs`.
    pub path: String,
    /// Full file contents.
    pub contents: String,
}

/// Per-node shard manifest embedded as JSON next to the firmware source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardManifest {
    /// Model being deployed.
    pub model: String,
    /// Board identifier.
    pub board: String,
    /// Swarm node index.
    pub swarm_node: usize,
    /// IR node names owned by this node.
    pub nodes: Vec<String>,
    /// Weight bytes owned.
    pub weight_bytes: u64,
    /// Reserved KV-cache bytes.
    pub kv_cache_bytes: u64,
}

/// Generate firmware sources, manifests, and master flash scripts.
///
/// Emits, relative to an output root:
///
/// ```text
/// firmware/node_00/src/main.rs      (or main.cpp)
/// firmware/node_00/manifest.json
/// flash.sh
/// flash.ps1
/// ```
pub fn generate(
    graph: &tpt_crucible_common::Graph,
    plan: &PartitionPlan,
    ctx: &FirmwareContext,
) -> Vec<GeneratedFile> {
    let mut files = Vec::new();

    for shard in &plan.shards {
        let dir = format!("firmware/node_{:02}", shard.swarm_node);
        let node_names: Vec<String> = shard
            .graph_nodes
            .iter()
            .filter_map(|id| graph.get_node(*id).map(|n| n.name.clone()))
            .collect();

        let manifest = ShardManifest {
            model: ctx.model_name.clone(),
            board: ctx.board.clone(),
            swarm_node: shard.swarm_node,
            nodes: node_names.clone(),
            weight_bytes: shard.weight_bytes,
            kv_cache_bytes: shard.kv_cache_bytes,
        };
        files.push(GeneratedFile {
            path: format!("{dir}/manifest.json"),
            contents: serde_json::to_string_pretty(&manifest)
                .expect("manifest serialization is infallible"),
        });

        let src_name = match ctx.lang {
            FirmwareLang::Rust => "src/main.rs",
            FirmwareLang::Cxx => "src/main.cpp",
        };
        files.push(GeneratedFile {
            path: format!("{dir}/{src_name}"),
            contents: render_source(ctx, shard.swarm_node, &node_names),
        });
    }

    let ports: Vec<String> = plan
        .shards
        .iter()
        .map(|s| format!("/dev/ttyUSB{}", s.swarm_node))
        .collect();
    files.push(GeneratedFile {
        path: "flash.sh".into(),
        contents: render_flash_sh(plan, ctx, &ports),
    });
    files.push(GeneratedFile {
        path: "flash.ps1".into(),
        contents: render_flash_ps1(plan, ctx, &ports),
    });

    files
}

// ---- askama templates (spec2.txt section 3.4) ------------------------------
//
// The four generators below are compile-time checked askama templates under
// `templates/`; a broken template fails `cargo build`, not a node's first
// boot. Contexts carry pre-formatted fields so templates stay declarative.

use askama::Template;

/// One per-node flashing target row shared by both master scripts.
#[derive(Debug, Clone)]
pub struct FlashTarget {
    /// Serial port the board was probed on (`/dev/ttyUSB3`).
    pub port: String,
    /// Zero-padded swarm index (`"07"`), matching the firmware dir name.
    pub index: String,
}

#[derive(Template)]
#[template(path = "firmware_rust.txt", ext = "txt")]
struct RustFirmware<'a> {
    model: &'a str,
    board: &'a str,
    node: String,
    ops: &'a [String],
}

#[derive(Template)]
#[template(path = "firmware_cxx.txt", ext = "txt")]
struct CxxFirmware<'a> {
    model: &'a str,
    board: &'a str,
    node: String,
    ops: &'a [String],
}

#[derive(Template)]
#[template(path = "flash_sh.txt", ext = "txt")]
struct FlashScriptSh<'a> {
    model: &'a str,
    board: &'a str,
    targets: &'a [FlashTarget],
}

#[derive(Template)]
#[template(path = "flash_ps1.txt", ext = "txt")]
struct FlashScriptPs1<'a> {
    model: &'a str,
    board: &'a str,
    targets: &'a [FlashTarget],
}

fn render_source(ctx: &FirmwareContext, node: usize, node_names: &[String]) -> String {
    let node = format!("{node:02}");
    match ctx.lang {
        FirmwareLang::Rust => RustFirmware {
            model: &ctx.model_name,
            board: &ctx.board,
            node,
            ops: node_names,
        }
        .render()
        .expect("firmware template renders statically"),
        FirmwareLang::Cxx => CxxFirmware {
            model: &ctx.model_name,
            board: &ctx.board,
            node,
            ops: node_names,
        }
        .render()
        .expect("firmware template renders statically"),
    }
}

fn render_flash_sh(plan: &PartitionPlan, ctx: &FirmwareContext, ports: &[String]) -> String {
    let targets: Vec<FlashTarget> = ports
        .iter()
        .zip(plan.shards.iter())
        .map(|(port, s)| FlashTarget {
            port: port.clone(),
            index: format!("{:02}", s.swarm_node),
        })
        .collect();
    FlashScriptSh {
        model: &ctx.model_name,
        board: &ctx.board,
        targets: &targets,
    }
    .render()
    .expect("flash script template renders statically")
}

fn render_flash_ps1(plan: &PartitionPlan, ctx: &FirmwareContext, ports: &[String]) -> String {
    let targets: Vec<FlashTarget> = ports
        .iter()
        .zip(plan.shards.iter())
        .map(|(port, s)| FlashTarget {
            port: port.clone(),
            index: format!("{:02}", s.swarm_node),
        })
        .collect();
    FlashScriptPs1 {
        model: &ctx.model_name,
        board: &ctx.board,
        targets: &targets,
    }
    .render()
    .expect("flash script template renders statically")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::{self, PartitionOptions};
    use crate::topology::Topology;
    use tpt_crucible_common::{DType, Graph, NodeId, Op, Tensor, TensorDesc};

    fn sample() -> (Graph, PartitionPlan) {
        let mut g = Graph::new("tinyllama-sil");
        let x = g.push(
            "",
            Op::Input(TensorDesc::new(vec![1, 8], DType::F32)),
            Vec::<NodeId>::new(),
        );
        let w = g.push(
            "blk.0.attn_q.weight",
            Op::Constant {
                tensor: Tensor::zeros(TensorDesc::new(vec![8, 8], DType::F16)),
            },
            Vec::<NodeId>::new(),
        );
        let y = g.push("", Op::MatMul, vec![x, w]);
        let out = g.push("", Op::Output { name: "out".into() }, vec![y]);
        g.mark_output(out);
        let topo = Topology::homogeneous(2, "esp32s3", 8 << 20);
        let plan = partition::partition(&g, &topo, &PartitionOptions::default()).unwrap();
        (g, plan)
    }

    fn ctx(lang: FirmwareLang) -> FirmwareContext {
        FirmwareContext {
            model_name: "tinyllama-sil".into(),
            board: "esp32s3".into(),
            lang,
        }
    }

    #[test]
    fn emits_manifest_source_and_both_flash_scripts() {
        let (g, plan) = sample();
        let files = generate(&g, &plan, &ctx(FirmwareLang::Rust));
        let paths: Vec<&str> = files.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.contains(&"flash.sh"));
        assert!(paths.contains(&"flash.ps1"));
        assert_eq!(
            paths
                .iter()
                .filter(|p| p.ends_with("manifest.json"))
                .count(),
            plan.shards.len()
        );

        let src_path = format!("firmware/node_{:02}/src/main.rs", plan.shards[0].swarm_node);
        let src = files.iter().find(|f| f.path == src_path).unwrap();
        assert!(src.contents.contains("no_std"));
        assert!(src.contents.contains("tinyllama-sil"));

        let flash = files.iter().find(|f| f.path == "flash.sh").unwrap();
        assert!(flash.contents.contains("esptool.py"));
        assert!(flash.contents.contains("write_flash"));

        let ps1 = files.iter().find(|f| f.path == "flash.ps1").unwrap();
        assert!(ps1.contents.contains("esptool.py"));
        assert!(ps1.contents.contains("Start-Job"));
    }

    #[test]
    fn cxx_language_changes_extension_and_body() {
        let (g, plan) = sample();
        let files = generate(&g, &plan, &ctx(FirmwareLang::Cxx));
        let cpp = files
            .iter()
            .find(|f| f.path.ends_with("main.cpp"))
            .expect("c++ source emitted");
        assert!(cpp.contents.contains("int main()"));
        assert_eq!(FirmwareLang::Cxx.banner(), "c++");
        assert!(!files.iter().any(|f| f.path.ends_with("main.rs")));
    }

    #[test]
    fn manifests_are_valid_json_with_expected_fields() {
        let (g, plan) = sample();
        let files = generate(&g, &plan, &ctx(FirmwareLang::Rust));
        let mf = files
            .iter()
            .find(|f| f.path.ends_with("manifest.json"))
            .unwrap();
        let parsed: ShardManifest = serde_json::from_str(&mf.contents).unwrap();
        assert_eq!(parsed.model, "tinyllama-sil");
        assert_eq!(parsed.board, "esp32s3");
        assert!(parsed.nodes.contains(&"blk.0.attn_q.weight".to_string()));
    }
}
