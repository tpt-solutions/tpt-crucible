//! Software-only end-to-end walkthrough of the Crucible pipeline — no
//! hardware attached, exactly the README quickstart commands:
//!
//! 1. **ingest** a GGUF model into TPT-IR (`tpt ingest`),
//! 2. **compile --target alloy**: partition across a 4-node swarm with
//!    KV-cache planning and emit firmware projects into an out-dir,
//! 3. **preflight** operator compatibility across every hardware family.
//!
//! The model is the committed fixture `examples/models/tiny-llama-block.gguf`
//! (see that directory's README); if it has not been generated yet this
//! example synthesizes identical bytes into the OS temp dir instead, so it
//! always runs.
//!
//! ```bash
//! cargo run -p tpt-crucible-cli --features swarm --example software_e2e
//! ```

use std::path::PathBuf;

use tpt_crucible_alloy as alloy;
use tpt_crucible_catalyst::preflight::{self, Family};
use tpt_crucible_common::{DType, Graph};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- 1. ingest -------------------------------------------------------
    let committed = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/models/tiny-llama-block.gguf");
    let synthesized;
    let model = if committed.exists() {
        println!("model   : {}", committed.display());
        committed
    } else {
        synthesized = std::env::temp_dir().join(format!("tpt-e2e-{}.gguf", std::process::id()));
        std::fs::write(&synthesized, fixture_gguf_v3())?;
        println!(
            "model   : committed fixture missing - synthesized {}",
            synthesized.display()
        );
        synthesized
    };

    let graph = tpt_crucible_catalyst::ingest_path(&model)?;
    graph.validate()?;
    println!(
        "ingest  : {} nodes, format={}, arch={}",
        graph.len(),
        graph.metadata("source_format").unwrap_or("?"),
        graph.metadata("general_architecture").unwrap_or("?")
    );

    // ---- 2. compile --target alloy --------------------------------------
    let topo = alloy::topology::Topology::homogeneous(4, "esp32s3", 8 << 20);

    // Mirror of the CLI's kv_from_metadata: GGUF hyperparameters drive the
    // KV-cache plan when present; drop it to see "kv-cache: skipped".
    let kv_request = kv_from_metadata(&graph).map(
        |(num_layers, num_kv_heads, head_dim)| alloy::kv_cache::KvCacheRequest {
            num_layers,
            num_kv_heads,
            head_dim,
            dtype: DType::F16,
            max_seq_len: 256,
            batch: 1,
        },
    );

    let opts = alloy::partition::PartitionOptions {
        strategy: Some(alloy::partition::Strategy::Hybrid),
        max_weight_bytes_per_node: None,
        kv_request,
        fpga_offload: false,
    };
    let plan = alloy::partition::partition(&graph, &topo, &opts)?;
    println!(
        "compile : {} x esp32s3, {} shard(s), {} cut edge(s)",
        4,
        plan.shards.len(),
        plan.stats.cut_edges
    );
    for shard in &plan.shards {
        println!(
            "  node {:<2} : {:>2} ops, {:.2} MiB weights, {:.3} MiB kv",
            shard.swarm_node,
            shard.graph_nodes.len(),
            shard.weight_bytes as f64 / (1 << 20) as f64,
            shard.kv_cache_bytes as f64 / (1 << 20) as f64
        );
    }
    match &plan.kv_plan {
        Some(kv) => println!(
            "  kv-plan  : {:.3} MiB total across heads",
            kv.total_bytes as f64 / (1 << 20) as f64
        ),
        None => println!("  kv-plan  : skipped"),
    }

    // Firmware bundle generation (what `--out-dir` writes), into temp.
    let out_dir = std::env::temp_dir().join(format!("tpt-e2e-swarm-{}", std::process::id()));
    let ctx = alloy::firmware::FirmwareContext {
        model_name: graph.name.clone(),
        board: "esp32s3".into(),
        lang: alloy::firmware::FirmwareLang::Rust,
    };
    let files = alloy::firmware::generate(&graph, &plan, &ctx);
    for f in &files {
        let path = out_dir.join(&f.path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, &f.contents)?;
    }
    println!(
        "firmware: {} file(s) under {}",
        files.len(),
        out_dir.display()
    );

    // ---- 3. preflight ----------------------------------------------------
    let families = [Family::Alloy, Family::Fusion, Family::Element];
    let report = preflight::stream(&graph, &families, &mut |event| {
        if let Ok(line) = event.to_json_line() {
            println!("prefly  : {line}");
        }
    });
    println!(
        "pre-flgt: {} checks across {} family(ies): {} supported, {} emulated, {} unsupported",
        report.checked,
        families.len(),
        report.supported,
        report.emulated,
        report.unsupported
    );

    let _ = &synthesized; // kept alive so the temp file outlives ingestion
    if report.has_blockers() {
        println!("blockers: {}", report.unsupported_ops.join(", "));
        return Ok(()); // demo exits cleanly; scripts can treat this as signal
    }
    println!("done    : full pipeline ran in software - zero hardware touched");
    Ok(())
}

/// `(layers, kv_heads, head_dim)` from GGUF-derived metadata keys
/// (kept byte-compatible with the CLI's `kv_from_metadata`).
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

/// Identical bytes to `examples/models/tiny-llama-block.gguf`; regenerate
/// that file via
/// `cargo run -p tpt-crucible-catalyst --example write_fixture_gguf`.
///
/// Keep in sync with the generator (wire types follow catalyst's GGUF parser
/// conventions: STRING = 11, U32 = 2).
fn fixture_gguf_v3() -> Vec<u8> {
    fn kv_string(out: &mut Vec<u8>, key: &str, value: &str) {
        out.extend_from_slice(&(key.len() as u64).to_le_bytes());
        out.extend_from_slice(key.as_bytes());
        out.extend_from_slice(&11u32.to_le_bytes()); // STRING
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }
    fn kv_u32(out: &mut Vec<u8>, key: &str, value: u32) {
        out.extend_from_slice(&(key.len() as u64).to_le_bytes());
        out.extend_from_slice(key.as_bytes());
        out.extend_from_slice(&2u32.to_le_bytes()); // U32
        out.extend_from_slice(&value.to_le_bytes());
    }

    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&2u64.to_le_bytes()); // tensor count
    b.extend_from_slice(&4u64.to_le_bytes()); // kv count

    kv_string(&mut b, "general.architecture", "llama");
    kv_u32(&mut b, "llama.block_count", 1);
    kv_u32(&mut b, "llama.attention.head_count", 2);
    kv_u32(&mut b, "llama.attention.key_length", 8);

    for (name, ne, offset) in [
        ("blk.0.attn_q.weight", [8u64, 16], 0u64),
        ("token_embd.weight", [8, 32], 512),
    ] {
        b.extend_from_slice(&(name.len() as u64).to_le_bytes());
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        for d in ne {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&0u32.to_le_bytes()); // ggml F32
        b.extend_from_slice(&offset.to_le_bytes());
    }

    // GGML mandates 32-byte alignment of the data section.
    while b.len() % 32 != 0 {
        b.push(0);
    }
    for i in 0..128u32 {
        b.extend_from_slice(&(i as f32 * 0.25).to_le_bytes());
    }
    for i in 0..256u32 {
        b.extend_from_slice(&(i as f32 * 0.125).to_le_bytes());
    }
    b
}