//! End-to-end smoke test for the Catalyst -> TPT-UIR pipeline (todo.md:
//! "Wire tpt-crucible-catalyst output through the adapter + tpt-uir-serde").
//!
//! Ingests a synthetic-but-real GGUF v3 fixture through Catalyst's native
//! parser, adapts the resulting TPT-IR graph to a Crucible-dialect Region,
//! round-trips it through postcard bytes and a `.tptuir` file, and converts
//! it back to an identical TPT-IR graph.

use std::path::PathBuf;

use tpt_crucible_common::Graph;
use tpt_crucible_uir_adapter::{graph_to_region, region_to_graph};
use tpt_uir_dialects::{CrucibleDialect, ValidateDialect};
use tpt_uir_serde::{deserialize_region, read_tptuir, serialize_region, write_tptuir};

/// Minimal but valid GGUF v3 file: two metadata kv pairs and one f32 [2 x 4]
/// tensor (same wire encoding as catalyst's internal fixtures).
fn fixture_gguf_v3() -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
    b.extend_from_slice(&2u64.to_le_bytes()); // kv count

    // kv #1: general.architecture = "llama" (STRING)
    let key = b"general.architecture";
    b.extend_from_slice(&(key.len() as u64).to_le_bytes());
    b.extend_from_slice(key);
    b.extend_from_slice(&11u32.to_le_bytes()); // type STRING
    b.extend_from_slice(&5u64.to_le_bytes());
    b.extend_from_slice(b"llama");

    // kv #2: llama.block_count = 22 (U32)
    let key = b"llama.block_count";
    b.extend_from_slice(&(key.len() as u64).to_le_bytes());
    b.extend_from_slice(key);
    b.extend_from_slice(&2u32.to_le_bytes()); // type U32
    b.extend_from_slice(&22u32.to_le_bytes());

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

/// Write the fixture to disk and ingest it via the real Catalyst entrypoint
/// (format auto-detection included).
fn ingest_fixture() -> (PathBuf, Graph) {
    let model = temp_path("tpt-crucible-e2e.gguf");
    std::fs::write(&model, fixture_gguf_v3()).unwrap();
    let g = tpt_crucible_catalyst::ingest_path(&model)
        .expect("synthetic GGUF v3 fixture ingests cleanly");
    (model, g)
}

#[test]
fn gguf_to_postcard_region_and_back_is_lossless() {
    let (model, g) = ingest_fixture();
    assert_eq!(g.metadata("source_format"), Some("gguf"));
    assert_eq!(g.metadata("general_architecture"), Some("llama"));
    assert_eq!(g.metadata("llama_block_count"), Some("22"));

    let region = graph_to_region(&g);
    CrucibleDialect::validate_all(&region).expect("adapted region satisfies the crucible dialect");

    let decoded =
        deserialize_region(&serialize_region(&region).unwrap()).expect("postcard round trip");
    assert_eq!(decoded, region);

    let back = region_to_graph(&decoded).expect("region converts back to TPT-IR");
    assert_eq!(back, g);

    std::fs::remove_file(&model).ok();
}

#[test]
fn adapted_region_roundtrips_through_a_tptuir_file() {
    let (model, g) = ingest_fixture();
    let region = graph_to_region(&g);

    let file = temp_path("tpt-crucible-e2e.tptuir");
    write_tptuir(&file, &region).expect("region serializes to a .tptuir file");
    let back = read_tptuir(&file).expect("written .tptuir parses back");

    assert_eq!(back, region);
    assert_eq!(region_to_graph(&back).unwrap(), g);

    std::fs::remove_file(&model).ok();
    std::fs::remove_file(&file).ok();
}
