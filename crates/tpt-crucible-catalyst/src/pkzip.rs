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

/// Extract every entry from a STORED-only archive.
pub(crate) fn read_entries(data: &[u8]) -> Result<Vec<ZipEntry>> {
    let eocd = find_eocd(data)?;
    let count = u16_le(data, eocd + 10) as usize;
    let mut cd_off = u32_le(data, eocd + 16) as usize;

    let mut out = Vec::new();
    for _ in 0..count {
        if cd_off + 46 > data.len() || data[cd_off..cd_off + 4] != [0x50, 0x4b, 0x01, 0x02] {
            return Err(bad("corrupt central directory"));
        }
        let method = u16_le(data, cd_off + 10);
        let comp_size = u32_le(data, cd_off + 20) as usize;
        let name_len = u16_le(data, cd_off + 28) as usize;
        let extra_len = u16_le(data, cd_off + 30) as usize;
        let comment_len = u16_le(data, cd_off + 32) as usize;
        let local_offset = u32_le(data, cd_off + 42) as usize;
        let name_end = cd_off + 46 + name_len;
        if name_end > data.len() {
            return Err(bad("truncated central directory name"));
        }
        let name = String::from_utf8_lossy(&data[cd_off + 46..name_end]).into_owned();
        cd_off = name_end + extra_len + comment_len;

        // Local header: sig(4), ..., name_len(@26), extra_len(@28).
        if local_offset + 30 > data.len()
            || data[local_offset..local_offset + 4] != [0x50, 0x4b, 0x03, 0x04]
        {
            return Err(bad(format!("bad local header for `{name}`")));
        }
        let lname_len = u16_le(data, local_offset + 26) as usize;
        let lextra_len = u16_le(data, local_offset + 28) as usize;
        let payload_at = local_offset + 30 + lname_len + lextra_len;
        let payload_end = payload_at
            .checked_add(comp_size)
            .ok_or_else(|| bad("payload length overflow"))?;
        if payload_end > data.len() {
            return Err(bad(format!("truncated payload for `{name}`")));
        }

        let payload = match method {
            0 => data[payload_at..payload_end].to_vec(),
            8 => {
                return Err(Error::UnsupportedOperation(
                    "deflated pytorch checkpoints are not supported yet; re-save \
                     with plain `torch.save` (which writes uncompressed zip members)"
                        .into(),
                ));
            }
            m => return Err(bad(format!("unsupported zip method {m} for `{name}`"))),
        };
        out.push(ZipEntry {
            name,
            data: payload,
        });
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) fn build_zip(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    let mut offsets = Vec::new();
    for (name, data) in entries {
        offsets.push(out.len());
        out.extend_from_slice(&[0x50, 0x4b, 0x03, 0x04]);
        out.extend_from_slice(&[0x0a, 0x00]); // version needed
        out.extend_from_slice(&[0x00, 0x00]); // flags
        out.extend_from_slice(&[0x00, 0x00]); // method = STORED
        out.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]); // times + crc
        let n = name.as_bytes();
        out.extend_from_slice(&(data.len() as u32).to_le_bytes()); // comp
        out.extend_from_slice(&(data.len() as u32).to_le_bytes()); // uncomp
        out.extend_from_slice(&(n.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra len
        out.extend_from_slice(n);
        out.extend_from_slice(data);
    }
    for (i, (name, data)) in entries.iter().enumerate() {
        let n = name.as_bytes();
        central.extend_from_slice(&[0x50, 0x4b, 0x01, 0x02]);
        central.extend_from_slice(&[0, 0]); // version made by
        central.extend_from_slice(&[0x0a, 0x00]); // version needed
        central.extend_from_slice(&[0x00, 0x00]); // flags
        central.extend_from_slice(&[0x00, 0x00]); // method
        central.extend_from_slice(&[0; 8]); // mod time + date + crc32
        central.extend_from_slice(&(data.len() as u32).to_le_bytes()); // comp
        central.extend_from_slice(&(data.len() as u32).to_le_bytes()); // uncomp
        central.extend_from_slice(&(n.len() as u16).to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes()); // extra
        central.extend_from_slice(&0u16.to_le_bytes()); // comment
        central.extend_from_slice(&0u16.to_le_bytes()); // disk
        central.extend_from_slice(&0u16.to_le_bytes()); // iattr
        central.extend_from_slice(&0u32.to_le_bytes()); // eattr
        central.extend_from_slice(&(offsets[i] as u32).to_le_bytes());
        central.extend_from_slice(n);
    }
    let cd_offset = out.len() as u32;
    out.extend_from_slice(&central);
    let cd_size = out.len() as u32 - cd_offset;
    out.extend_from_slice(&[0x50, 0x4b, 0x05, 0x06]); // EOCD
    out.extend_from_slice(&0u16.to_le_bytes()); // disk
    out.extend_from_slice(&0u16.to_le_bytes()); // cd disk
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // comment len
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_stored_entries() {
        let z = build_zip(&[
            ("archive/data.pkl", vec![1, 2, 3]),
            ("archive/data/0", vec![9u8; 100]),
        ]);
        let got = read_entries(&z).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].name, "archive/data.pkl");
        assert_eq!(got[0].data, vec![1, 2, 3]);
        assert_eq!(got[1].name, "archive/data/0");
        assert_eq!(got[1].data.len(), 100);
    }

    #[test]
    fn not_a_zip_errors() {
        assert!(read_entries(b"garbage").is_err());
    }
}
