//! # tpt-crucible-observer
//!
//! **The Dashboard backend:** an `axum` + WebSocket service that
//! broadcasts live telemetry from Alloy swarms, Fusion overlays, and Element
//! simulations to every connected client in one schema: tokens/sec, memory
//! bandwidth, thermal drift, and node latency (`spec2.txt` §4.5).
//!
//! The frontend lives in `tpt-crucible-observer-web` (Leptos/Wasm).
//!
//! ## Usage
//!
//! ```no_run
//! use tpt_crucible_observer::{TelemetryEvent, TelemetryServer};
//!
//! # async fn demo() -> tpt_crucible_common::Result<()> {
//! let server = TelemetryServer::bind("127.0.0.1:8787").await?;
//! println!("listening on {}", server.local_addr());
//!
//! // Any emitter (Alloy coordinator, Fusion bench, …) fans events out:
//! server.emit(TelemetryEvent {
//!     node_id: "swarm-03".into(),
//!     hw_type: "alloy".into(),
//!     tokens_per_sec: Some(12.5),
//!     mem_bandwidth_gbps: None,
//!     thermal_drift_c: Some(4.2),
//!     latency_ms: Some(0.8),
//!     timestamp_ms: 1_756_000_000_000,
//! });
//! # Ok(())
//! # }
//! ```
//!
//! Every WebSocket client connected to `GET /ws` receives each emitted
//! [`TelemetryEvent`] as one JSON text frame; slow clients lag via the
//! broadcast channel instead of stalling emitters.

pub mod server;

pub use server::TelemetryServer;

/// Human-readable implementation status of this crate.
pub fn status() -> &'static str {
    "phase 4 (The Observer): telemetry schema + websocket backend live; \
     dashboard frontend pending"
}

/// Unified telemetry record schema shared by every hardware type.
///
/// Defined now so Catalyst/Alloy can already emit events against a stable
/// shape before the server lands.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TelemetryEvent {
    /// Emitting node id within its hardware type.
    pub node_id: String,
    /// Hardware family (`"alloy" | "fusion" | "element"`).
    pub hw_type: String,
    /// Generated tokens per second, when applicable.
    pub tokens_per_sec: Option<f64>,
    /// Observed memory bandwidth in GB/s, when applicable.
    pub mem_bandwidth_gbps: Option<f64>,
    /// Thermal drift in °C from nominal, when applicable.
    pub thermal_drift_c: Option<f64>,
    /// Round-trip node latency in ms, when applicable.
    pub latency_ms: Option<f64>,
    /// Millisecond timestamp of the sample.
    pub timestamp_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_event_roundtrips_through_json() {
        let ev = TelemetryEvent {
            node_id: "swarm-03".into(),
            hw_type: "alloy".into(),
            tokens_per_sec: Some(12.5),
            mem_bandwidth_gbps: None,
            thermal_drift_c: Some(4.2),
            latency_ms: Some(0.8),
            timestamp_ms: 1_756_000_000_000,
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: TelemetryEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ev);
        assert!(status().contains("websocket backend live"));
    }
}
