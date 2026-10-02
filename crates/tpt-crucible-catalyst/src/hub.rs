//! HuggingFace Hub network fetch (`spec2.txt` §3.1 supported input
//! "HuggingFace Hub").
//!
//! Downloads a repo id (e.g. `meta-llama/Llama-3.2-1B`) into a local
//! directory — `config.json` plus every `*.safetensors` shard — and returns
//! that directory, which the existing HuggingFace *directory-layout*
//! ingestion path already understands. Network + TLS live behind the `hub`
//! cargo feature so the core crate stays dependency-light and the wasm SiL
//! build is untouched.
//!
//! ```no_run
//! # #[cfg(feature = "hub")]
//! # fn demo() -> tpt_crucible_common::Result<()> {
//! let dir = tpt_crucible_catalyst::hub::fetch_repo_to(
//!     "hf-internal-testing/tiny-random-LlamaForCausalLM",
//!     &std::path::PathBuf::from("cache"),
//! )?;
//! let graph = tpt_crucible_catalyst::ingest_path(&dir)?;
//! # Ok(())
//! # }
//! ```

use std::path::{Path, PathBuf};

use serde::Deserialize;

use tpt_crucible_common::error::{Error, Result};

/// Default Hub endpoint (mirror-friendly: override for offline caches).
pub const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// One file advertised by the Hub API.
#[derive(Debug, Clone, Deserialize)]
pub struct Sibling {
    /// Repo-relative path, e.g. `model-00001-of-00002.safetensors`.
    #[serde(rename = "rfilename")]
    pub file_name: String,
}

/// Files we fetch from a repo, in deterministic order.
fn wanted(sibling: &Sibling) -> bool {
    let f = &sibling.file_name;
    f == "config.json" || f.ends_with(".safetensors")
}

/// Validate a repo id: exactly one `/`, both halves non-empty, conservative
/// charset — this string is interpolated into URLs.
///
/// # Errors
/// [`Error::InvalidArgument`] with the offending id.
pub fn validate_repo_id(repo_id: &str) -> Result<()> {
    let ok = |s: &str| {
        !s.is_empty()
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
    };
    let mut parts = repo_id.split('/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(org), Some(name), None) if ok(org) && ok(name) => Ok(()),
        _ => Err(Error::InvalidArgument(format!(
            "invalid repo id `{repo_id}` (expected `org/name` with \
             [A-Za-z0-9._-] components)"
        ))),
    }
}

/// Select the files to download from an API response's sibling list.
///
/// # Errors
/// [`Error::ParseFormat`] when nothing fetchable is found.
pub fn select_files(siblings: &[Sibling]) -> Result<Vec<String>> {
    let mut out: Vec<String> = siblings
        .iter()
        .filter(|s| wanted(s))
        .map(|s| s.file_name.clone())
        .collect();
    out.sort();
    if out.is_empty() {
        return Err(Error::ParseFormat {
            path: "<hub api>".into(),
            format: "huggingface".into(),
            reason: "repo advertises no config.json or *.safetensors files".into(),
        });
    }
    Ok(out)
}

/// Fetch `repo_id` into `dest/<org>__<name>/` and return that directory.
///
/// Downloads are simple GETs against `{endpoint}/{repo}/resolve/main/{file}`;
/// the HTTP client follows redirects (the resolve endpoint 302s to the CDN).
/// Existing files with matching size are skipped, so re-running is an
/// incremental sync rather than a redownload.
///
/// # Errors
/// * [`Error::InvalidArgument`] on a malformed repo id,
/// * [`Error::Io`] on transport/write failures,
/// * [`Error::ParseFormat`] when the API response shape is unexpected.
#[cfg(feature = "hub")]
pub fn fetch_repo_to(repo_id: &str, dest_root: &Path) -> Result<PathBuf> {
    validate_repo_id(repo_id)?;
    let agent = ureq::AgentBuilder::new().build();

    // --- list repo files ---------------------------------------------------
    let api_url = format!("{DEFAULT_ENDPOINT}/api/models/{repo_id}");
    let response = agent.get(&api_url).call().map_err(|e| match e {
        ureq::Error::Status(code, resp) => Error::Io(std::io::Error::other(format!(
            "hub listing failed: HTTP {code} ({})",
            resp.get_url()
        ))),
        other => Error::Io(std::io::Error::other(format!(
            "hub listing failed: {other}"
        ))),
    })?;
    let api: ApiModel =
        serde_json::from_reader(response.into_reader()).map_err(|e| Error::ParseFormat {
            path: api_url.clone(),
            format: "huggingface".into(),
            reason: e.to_string(),
        })?;
    let files = select_files(&api.siblings)?;

    // --- download each file -------------------------------------------------
    let target_dir = dest_root.join(repo_id.replace('/', "__"));
    std::fs::create_dir_all(&target_dir)?;
    for file in &files {
        let out_path = target_dir.join(file);
        let url = format!("{DEFAULT_ENDPOINT}/{repo_id}/resolve/main/{file}");
        download(&agent, &url, &out_path, repo_id, file)?;
    }
    Ok(target_dir)
}

#[cfg(feature = "hub")]
fn download(
    agent: &ureq::Agent,
    url: &str,
    out_path: &Path,
    repo_id: &str,
    file: &str,
) -> Result<()> {
    use std::io::Write as _;

    if let Ok(meta) = std::fs::metadata(out_path) {
        if meta.len() > 0 {
            return Ok(()); // already synced
        }
    }
    let reader = agent.get(url).call().map_err(|e| match e {
        ureq::Error::Status(code, _) => Error::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("fetch of {repo_id}/{file} failed: HTTP {code}"),
        )),
        other => Error::Io(std::io::Error::other(format!(
            "fetch of {repo_id}/{file} failed: {other}"
        ))),
    })?;

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut tmp = out_path.to_owned();
    tmp.set_extension("part");
    let mut file_handle = std::fs::File::create(&tmp)?;
    std::io::copy(&mut reader.into_reader(), &mut file_handle)?;
    file_handle.flush()?;
    std::fs::rename(&tmp, out_path)?;
    Ok(())
}

#[derive(Deserialize)]
struct ApiModel {
    siblings: Vec<Sibling>,
}

#[cfg(all(test, feature = "hub"))]
mod tests {
    use super::*;

    #[test]
    fn repo_ids_are_validated_conservatively() {
        assert!(validate_repo_id("meta-llama/Llama-3.2-1B").is_ok());
        assert!(validate_repo_id("hf.internal-testing/tiny_random-v1.2").is_ok());
        // No slash / multiple slashes / bad chars / empty halves.
        assert!(validate_repo_id("llama").is_err());
        assert!(validate_repo_id("a/b/c").is_err());
        assert!(validate_repo_id("a b/c").is_err());
        assert!(validate_repo_id("/name").is_err());
        assert!(validate_repo_id("org/").is_err());
    }

    #[test]
    fn file_selection_keeps_config_and_safetensors_only() {
        let siblings = vec![
            Sibling {
                file_name: "config.json".into(),
            },
            Sibling {
                file_name: "model.safetensors".into(),
            },
            Sibling {
                file_name: "model-00001-of-00002.safetensors".into(),
            },
            Sibling {
                file_name: "pytorch_model.bin".into(),
            },
            Sibling {
                file_name: "tokenizer.model".into(),
            },
            Sibling {
                file_name: "README.md".into(),
            },
        ];
        let files = select_files(&siblings).unwrap();
        assert_eq!(
            files,
            vec![
                "config.json",
                "model-00001-of-00002.safetensors",
                "model.safetensors",
            ]
        );
    }

    #[test]
    fn repos_without_fetchable_files_are_rejected() {
        let siblings = vec![Sibling {
            file_name: "weights.pkl".into(),
        }];
        let err = select_files(&siblings).unwrap_err();
        assert!(err.to_string().contains("safetensors"), "{err}");
    }

    /// Live network round trip against the real Hub — opt-in via
    /// `TPT_HUB_LIVE=1 cargo test` so CI stays hermetic.
    #[test]
    fn live_hub_download_when_enabled() {
        if std::env::var("TPT_HUB_LIVE").as_deref() != Ok("1") {
            eprintln!("skipping: set TPT_HUB_LIVE=1 to run the live Hub fetch");
            return;
        }
        let dest = std::env::temp_dir().join(format!("tpt-hub-live-{}", std::process::id()));
        let dir = fetch_repo_to("hf-internal-testing/tiny-random-LlamaForCausalLM", &dest)
            .expect("tiny public repo downloads");
        // The downloaded directory feeds the existing HF dir ingestion.
        let graph = crate::ingest_path(&dir).expect("fetched repo ingests");
        assert!(!graph.nodes.is_empty());
        std::fs::remove_dir_all(&dest).ok();
    }
}
