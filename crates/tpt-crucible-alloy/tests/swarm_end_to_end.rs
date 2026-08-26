//! Software-in-the-loop dry run of the full swarm deployment pipeline
//! (`todo.md`: "**Milestone:** Load TinyLlama, partition via Alloy, flash to
//! 16 ESP32s" — everything except the physical flashing, which needs USB
//! hubs and real boards).
//!
//! The flow exercised here is exactly what a coordinator runs in the field:
//!
//! 1. ingest a (synthetic-but-real) GGUF v3 model through Catalyst;
//! 2. partition it across a 16-node ESP32-S3 swarm with KV-cache planning;
//! 3. generate per-node firmware + manifests + master flash scripts;
//! 4. hand the plan to the [`ExecutionEngine`], distribute deployments;
//! 5. stream heartbeats, lose a node mid-flight, and watch the engine
//!    re-partition onto survivors;
//! 6. schedule rolling micro-batch pipelines over both plans and confirm
//!    pipelining beats strict layer-serial execution.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use tpt_crucible_alloy::firmware::{self, FirmwareContext, FirmwareLang};
use tpt_crucible_alloy::kv_cache::KvCacheRequest;
use tpt_crucible_alloy::pipeline::{self, PipelineModel};
use tpt_crucible_alloy::runtime::{EngineEvent, ExecutionEngine};
use tpt_crucible_alloy::{heartbeat, partition, topology};
use tpt_crucible_common::DType;

/// Minimal but valid GGUF v3 file: llama metadata + one f32 [2 x 4] tensor
/// (same wire encoding as catalyst's internal fixtures).
fn fixture_gguf_v3() -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
    b.extend_from_slice(&4u64.to_le_bytes()); // kv count

    let kv_string = |buf: &mut Vec<u8>, key: &[u8], value: &[u8]| {
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&11u32.to_le_bytes()); // STRING
        buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
        buf.extend_from_slice(value);
    };
    let kv_u32 = |buf: &mut Vec<u8>, key: &[u8], value: u32| {
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&2u32.to_le_bytes()); // U32
        buf.extend_from_slice(&value.to_le_bytes());
    };

    kv_string(&mut b, b"general.architecture", b"llama");
    kv_u32(&mut b, b"llama.block_count", 2);
    kv_u32(&mut b, b"llama.attention.head_count", 8);
    kv_u32(&mut b, b"llama.attention.key_length", 16);

    // tensor dir: blk.0.attn_q.weight, f32 ne=[2, 4]
    let name = b"blk.0.attn_q.weight";
    b.extend_from_slice(&(name.len() as u64).to_le_bytes());
    b.extend_from_slice(name);
    b.extend_from_slice(&2u32.to_le_bytes()); // n_dims
    b.extend_from_slice(&2u64.to_le_bytes()); // ne[0]
    b.extend_from_slice(&4u64.to_le_bytes()); // ne[1]
    b.extend_from_slice(&0u32.to_le_bytes()); // ggml F32
    b.extend_from_slice(&0u64.to_le_bytes()); // offset

    // GGML mandates 32-byte alignment of the data section.
    while b.len() % 32 != 0 {
        b.push(0);
    }
    for i in 0..8u32 {
        b.extend_from_slice(&(i as f32).to_le_bytes());
    }
    b
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("{name}-{}.tmp", std::process::id()))
}

#[test]
fn swarm_end_to_end_sil_dry_run() {
    // ---- 1. ingest ----------------------------------------------------------
    let model_path = temp_path("tpt-alloy-e2e.gguf");
    std::fs::write(&model_path, fixture_gguf_v3()).unwrap();
    let g = tpt_crucible_catalyst::ingest_path(&model_path)
        .expect("synthetic GGUF ingests through the native parser");
    g.validate().expect("ingested graph is well-formed");
    assert_eq!(g.metadata("general_architecture"), Some("llama"));

    // ---- 2. partition across 16 nodes with KV planning ----------------------
    const FLEET: usize = 16;
    let mem_bytes: u64 = 8 << 20; // ESP32-S3 PSRAM budget
    let topo = topology::Topology::homogeneous(FLEET, "esp32s3", mem_bytes);

    let kv_request = KvCacheRequest {
        num_layers: 2,
        num_kv_heads: 8,
        head_dim: 16,
        dtype: DType::F16,
        max_seq_len: 512,
        batch: 1,
    };
    let options = partition::PartitionOptions {
        strategy: Some(partition::Strategy::Hybrid),
        max_weight_bytes_per_node: None,
        kv_request: Some(kv_request),
        fpga_offload: false,
    };
    let plan = partition::partition(&g, &topo, &options)
        .expect("tiny model partitions across the 16-node fleet");
    assert!(
        !plan.shards.is_empty() && plan.shards.len() <= FLEET,
        "shards fit inside the fleet"
    );
    assert!(plan.kv_plan.is_some(), "kv request was planned");
    for shard in &plan.shards {
        assert!(
            shard.weight_bytes + shard.kv_cache_bytes <= mem_bytes,
            "no shard overflows its node"
        );
    }

    // ---- 3. firmware + flash scripts ----------------------------------------
    let ctx = FirmwareContext {
        model_name: g.name.clone(),
        board: "esp32s3".into(),
        lang: FirmwareLang::Rust,
    };
    let files = firmware::generate(&g, &plan, &ctx);
    assert_eq!(
        files
            .iter()
            .filter(|f| f.path.ends_with("src/main.rs"))
            .count(),
        plan.shards.len(),
        "one firmware source file per busy node"
    );
    assert!(files.iter().any(|f| f.path == "flash.sh"));
    assert!(files.iter().any(|f| f.path == "flash.ps1"));
    assert!(files
        .iter()
        .any(|f| f.path.ends_with("manifest.json") && f.contents.contains("swarm_node")));

    // ---- 4. engine deploy ----------------------------------------------------
    const HEARTBEAT_TIMEOUT: Duration = Duration::from_millis(100);
    let mut engine =
        ExecutionEngine::launch(&g, &topo, options.clone(), HEARTBEAT_TIMEOUT).unwrap();
    let deployments = engine.distribute();
    assert_eq!(deployments.len(), plan.shards.len());
    for d in &deployments {
        d.decode().expect("deployment payloads decode on-device");
    }

    // ---- 5. heartbeats, one death, automatic recovery -------------------------
    let t0 = Instant::now();
    let beat = |id: u32| heartbeat::HeartbeatMsg {
        node_id: id,
        seq: 1,
        timestamp_ms: 0,
        status: heartbeat::NodeStatus::Healthy,
    };
    for id in 0..FLEET as u32 {
        engine.on_heartbeat(&beat(id), t0);
    }
    // Everyone refreshes at +150 ms except one busy node's host; at the
    // +200 ms tick only that host is stale.
    let doomed = plan.shards[plan.shards.len() - 1].swarm_node;
    for id in 0..FLEET as u32 {
        if id as usize != doomed {
            engine.on_heartbeat(&beat(id), t0 + Duration::from_millis(150));
        }
    }

    let events = engine.tick(t0 + Duration::from_millis(200));
    match events.as_slice() {
        [EngineEvent::NodeLost { nodes }, EngineEvent::Recovered { generation, .. }] => {
            assert_eq!(nodes, &vec![doomed]);
            assert_eq!(*generation, 1);
        }
        other => panic!("unexpected engine events: {other:?}"),
    }

    // Post-recovery deployments avoid the dead node entirely.
    for d in engine.distribute() {
        assert_ne!(d.node_id, doomed, "no work left on the failed node");
        assert_eq!(d.generation, 1);
    }

    // ---- 6. rolling pipeline over the recovered plan --------------------------
    let model = PipelineModel::from_plan(
        engine.plan(),
        &topo,
        50_000.0, // 50 KiB/ms compute throughput estimate
        4096,
    )
    .expect("stage costs derive from the recovered plan");
    let sched = pipeline::schedule(&model, 8).unwrap();
    if model.stage_compute_ms.len() > 1 {
        assert!(
            sched.eliminates_stalls(),
            "multi-stage rolling must beat layer-serial"
        );
    } else {
        // Single-shard plans degenerate to serial — still a valid outcome.
        assert!(!sched.eliminates_stalls());
    }

    std::fs::remove_file(&model_path).ok();
}
