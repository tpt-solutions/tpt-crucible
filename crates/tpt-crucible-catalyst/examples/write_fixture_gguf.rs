//! Regenerates the committed software-only fixture model
//! (`examples/models/tiny-llama-block.gguf`).
//!
//! The fixture is a synthetic-but-valid GGUF v3 container: four metadata kv
//! pairs (llama hyperparameters the CLI's KV planner reads) and two f32
//! weight tensors (~1.7 KB total). It exists so examples, docs, and manual
//! smoke tests can run a real ingestion path without shipping model weights.
//!
//! ```bash
//! cargo run -p tpt-crucible-catalyst --example write_fixture_gguf
//! ```
//!
//! Byte-layout note: the wire encoding mirrors the in-crate GGUF test
//! fixtures (metadata types STRING = 11, U32 = 2 as our parser defines
//! them); keep this file in sync with the inline copy inside the CLI's
//! `software_e2e` example.

use std::path::PathBuf;

/// Encode one string-valued metadata kv pair.
fn kv_string(out: &mut Vec<u8>, key: &str, value: &str) {
    out.extend_from_slice(&(key.len() as u64).to_le_bytes());
    out.extend_from_slice(key.as_bytes());
    out.extend_from_slice(&11u32.to_le_bytes()); // STRING
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

/// Encode one u32-valued metadata kv pair.
fn kv_u32(out: &mut Vec<u8>, key: &str, value: u32) {
    out.extend_from_slice(&(key.len() as u64).to_le_bytes());
    out.extend_from_slice(key.as_bytes());
    out.extend_from_slice(&2u32.to_le_bytes()); // U32
    out.extend_from_slice(&value.to_le_bytes());
}

/// One tensor-directory entry.
fn tensor_entry(name: &str, ne: [u64; 2], offset: u64) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&(name.len() as u64).to_le_bytes());
    b.extend_from_slice(name.as_bytes());
    b.extend_from_slice(&2u32.to_le_bytes()); // n_dims
    for d in ne {
        b.extend_from_slice(&d.to_le_bytes());
    }
    b.extend_from_slice(&0u32.to_le_bytes()); // ggml F32
    b.extend_from_slice(&offset.to_le_bytes());
    b
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&2u64.to_le_bytes()); // tensor count
    b.extend_from_slice(&4u64.to_le_bytes()); // kv count

    kv_string(
        &mut b,
        "general.architecture",
        "llama",
    );
    kv_u32(&mut b, "llama.block_count", 1);
    kv_u32(&mut b, "llama.attention.head_count", 2);
    kv_u32(&mut b, "llama.attention.key_length", 8);

    // Tensor directory. Offsets are relative to the (aligned) data region:
    // blk.0.attn_q.weight occupies 16*8*4 = 512 B, token_embd follows.
    b.extend(tensor_entry("blk.0.attn_q.weight", [8, 16], 0));
    b.extend(tensor_entry("token_embd.weight", [8, 32], 512));

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

    let dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/models");
    std::fs::create_dir_all(&dir)?;
    let out = dir.join("tiny-llama-block.gguf");
    std::fs::write(&out, &b)?;
    println!("wrote {} ({} bytes)", out.canonicalize()?, b.len());
    Ok(())
}