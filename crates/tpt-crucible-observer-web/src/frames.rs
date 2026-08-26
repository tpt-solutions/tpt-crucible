//! Mirror of the Observer WebSocket wire schema, parsed on the client.
//!
//! Deliberately duplicated from the backend crate (which cannot target wasm);
//! see the crate docs for why that is safe today and how it stays honest.

use serde::Deserialize;

/// One telemetry sample as rendered in the live table.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TelemetryRow {
    /// Emitting node id (`"esp32s3-07"`).
    pub node_id: String,
    /// Hardware family.
    pub hw_type: String,
    #[serde(default)]
    pub tokens_per_sec: Option<f64>,
    #[serde(default)]
    pub mem_bandwidth_gbps: Option<f64>,
    #[serde(default)]
    pub thermal_drift_c: Option<f64>,
    #[serde(default)]
    pub latency_ms: Option<f64>,
    #[serde(default)]
    pub timestamp_ms: u64,
}

/// One streamed pre-flight compatibility verdict.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PreflightNotice {
    pub seq: u64,
    #[serde(default)]
    pub node_name: String,
    pub op: String,
    pub family: String,
    /// `supported` | `emulated` | `unsupported`.
    pub verdict: String,
    #[serde(default)]
    pub note: Option<String>,
}

/// Anything the `/ws` stream can deliver; unknown kinds collapse to
/// [`Frame::Unknown`] so forward-compatible clients keep working.
#[derive(Debug, Clone, PartialEq)]
pub enum Frame {
    Telemetry(TelemetryRow),
    Preflight(PreflightNotice),
    Unknown,
}

/// Parse one WebSocket text frame into a typed [`Frame`].
///
/// Malformed JSON (never expected from our own backend) degrades to
/// [`Frame::Unknown`] rather than killing the stream.
#[must_use]
pub fn parse_frame(text: &str) -> Frame {
    #[derive(Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum Tagged {
        Telemetry(TelemetryRow),
        Preflight(PreflightNotice),
        #[serde(other)]
        Other,
    }
    match serde_json::from_str::<Tagged>(text) {
        Ok(Tagged::Telemetry(t)) => Frame::Telemetry(t),
        Ok(Tagged::Preflight(p)) => Frame::Preflight(p),
        _ => Frame::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_both_frame_kinds() {
        let telemetry = parse_frame(
            r#"{"kind":"telemetry","node_id":"esp32s3-07","hw_type":"alloy",
                "tokens_per_sec":12.5,"mem_bandwidth_gbps":null,
                "thermal_drift_c":4.2,"latency_ms":0.8,"timestamp_ms":42}"#,
        );
        match telemetry {
            Frame::Telemetry(t) => {
                assert_eq!(t.node_id, "esp32s3-07");
                assert_eq!(t.tokens_per_sec, Some(12.5));
            }
            other => panic!("expected telemetry frame, got {other:?}"),
        }

        let preflight = parse_frame(
            r#"{"kind":"preflight","seq":3,"node_id":7,"node_name":"blk.0.attn_q",
                "op":"matmul","family":"fusion","verdict":"supported",
                "note":"overlay MAC array"}"#,
        );
        match preflight {
            Frame::Preflight(p) => {
                assert_eq!(p.op, "matmul");
                assert_eq!(p.verdict, "supported");
            }
            other => panic!("expected preflight frame, got {other:?}"),
        }
    }

    #[test]
    fn unknown_and_broken_frames_degrade_safely() {
        assert_eq!(parse_frame(r#"{"kind":"future_thing"}"#), Frame::Unknown);
        assert_eq!(parse_frame("not json at all"), Frame::Unknown);
        assert_eq!(parse_frame(""), Frame::Unknown);
    }
}
