//! Llamafile ingestion — single-file executables with embedded GGUF weights.
//!
//! A Llamafile is an ELF binary (the llama.cpp runtime) with a GGUF model
//! appended behind a zip central directory. We locate the embedded `GGUF`
//! magic and hand the tail of the file to the native [`crate::gguf`] parser.
//! The scan is bounded so hostile executables can't force unbounded work.

use std::path::Path;

use tpt_crucible_common::error::{Error, Result};
use tpt_crucible_common::Graph;

/// Upper bound on how deep into the file we search for the magic. Real
/// Llamafiles expose the model near the front (zip directory), well under 1 MiB.
const MAX_SCAN_BYTES: u64 = 64 << 20;

/// Byte offset of the first `GGUF` magic in `data` (bounded by [`MAX_SCAN_BYTES`]).
fn find_gguf_offset(data: &[u8]) -> Result<usize> {
    let limit = data.len().min(MAX_SCAN_BYTES as usize + 4);
    if limit < 4 {
        return Err(not_found());
    }
    let mut i = 0;
    while i + 4 <= limit {
        if &data[i..i + 4] == b"GGUF" {
            return Ok(i);
        }
        i += 1;
    }
    Err(not_found())
}

fn not_found() -> Error {
    Error::ParseFormat {
        path: "<llamafile>".into(),
        format: "llamafile".into(),
        reason: "no embedded GGUF section found".into(),
    }
}

/// Ingest a Llamafile executable.
pub fn ingest(path: &Path) -> Result<Graph> {
    let data = std::fs::read(path)?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let offset = find_gguf_offset(&data)?;
    crate::gguf::build_graph(&data[offset..], &format!("{name} (llamafile)"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn tmp_llamafile(name: &str, body: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("catalyst-llamafile-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        // ELF-ish prefix + padding, then the embedded model.
        f.write_all(b"\x7fELF\x02\x01\x01\x00").unwrap();
        f.write_all(&vec![0u8; 512]).unwrap();
        f.write_all(body).unwrap();
        p
    }

    #[test]
    fn extracts_embedded_gguf() {
        let p = tmp_llamafile("ok.llamafile", &crate::gguf::fixture_gguf_v3());
        let g = ingest(&p).unwrap();
        assert_eq!(g.metadata("source_format"), Some("gguf"));
        assert!(g.name.ends_with("(llamafile)"));
        assert_eq!(g.len(), 1);
        g.validate().unwrap();
    }

    #[test]
    fn no_embedded_model_errors() {
        let p = tmp_llamafile("empty.llamafile", b"just an elf, nothing else");
        let err = ingest(&p).unwrap_err();
        assert!(err.to_string().contains("no embedded GGUF"));
    }
}
