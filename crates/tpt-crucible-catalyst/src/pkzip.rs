//! Minimal PK-ZIP reader for `torch.save` archives.
//!
//! PyTorch >= 1.6 writes checkpoints as uncompressed ("STORED") zip archives
//! containing `data.pkl` plus one raw blob per storage. This reader parses
//! the central directory and extracts payloads without decompression;
//! DEFLATED entries surface as a clear [`Error::UnsupportedOperation`]
//! instead of pulling an inflate implementation into the dependency tree.

use tpt_crucible_common::error::{Error, Result};

/// One extracted archive member.
#[derive(Debug)]
pub(crate) struct ZipEntry {
    /// Full path inside the archive, e.g. `model/data/0`.
    pub name: String,
    /// Payload bytes (only valid for STORED entries).
    pub data: Vec<u8>,
}

fn bad(reason: impl Into<String>) -> Error {
    Error::ParseFormat {
        path: "<pytorch>".into(),
        format: "pytorch".into(),
        reason: reason.into(),
    }
}

fn u16_le(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32_le(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Locate the End-of-Central-Directory record (scan backwards over the
/// maximum legal comment size).
fn find_eocd(data: &[u8]) -> Result<usize> {
    const SIG: [u8; 4] = [0x50, 0x4b, 0x05, 0x06]; // "PK\x05\x06"
    let scan_len = data.len().min(66_000);
    if scan_len < 22 {
        return Err(bad("file too small to be a zip archive"));
    }
    let start = data.len() - scan_len;
    let mut i = data.len() - 22;
    loop {
        if data[i..i + 4] == SIG {
            return Ok(i);
        }
        if i == start {
            return Err(bad("end-of-central-directory not found"));
        }
        i -= 1;
    }
}
