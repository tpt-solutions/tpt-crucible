//! Heartbeat protocol: node liveness messages and a failure detector.
//!
//! Swarm nodes broadcast [`HeartbeatMsg`] frames; the coordinator feeds them

//! into [`FailureDetector`], which tracks the last-seen time per node and
//! flags nodes silent past the timeout. Dead nodes are bypassed instead of
//! stalling inference (`spec2.txt`: "Fault-Tolerant Inference: Swarms
//! degrade gracefully when nodes fail").
//!
//! Frames use the same compact bincode codec as binary TPT-IR, so an ESP32
//! can pack them without pulling in JSON.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Liveness frame broadcast by every swarm node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeartbeatMsg {
    /// Sender's swarm id.
    pub node_id: u32,
    /// Monotonic counter; gaps indicate loss.
    pub seq: u64,
    /// Sender wall-clock in ms (diagnostics only, not used for timeouts).
    pub timestamp_ms: u64,
    /// Reported health.
    pub status: NodeStatus,
}

/// Health reported by a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeStatus {
    /// Serving normally.
    Healthy,
    /// Serving degraded (thermals, ECC retries, ...).
    Degraded,
    /// Draining before shutdown.
    Draining,
}

impl HeartbeatMsg {
    /// Encode to compact bytes (bincode, no framing overhead).
    pub fn encode(&self) -> Vec<u8> {
        bincode::serialize(self).expect("heartbeat serialization is infallible")
    }

    /// Decode from compact bytes.
    pub fn decode(bytes: &[u8]) -> tpt_crucible_common::Result<Self> {
        Ok(bincode::deserialize(bytes)?)
    }
}

/// Tracks last-seen instants and exposes dead-node queries.
///
/// Time is injected ([`FailureDetector::record`]) so behavior is deterministic
/// in tests and portable to wasm (where `Instant` differs).
#[derive(Debug)]
pub struct FailureDetector {
    timeout: Duration,
    last_seen: BTreeMap<u32, (Instant, u64)>,
}

impl FailureDetector {
    /// Detector flagging nodes silent for longer than `timeout`.
    pub fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            last_seen: BTreeMap::new(),
        }
    }

    /// Record an incoming heartbeat observed at `now`.
    pub fn record(&mut self, msg: &HeartbeatMsg, now: Instant) {
        self.last_seen
            .entry(msg.node_id)
            .and_modify(|e| *e = (now, msg.seq))
            .or_insert((now, msg.seq));
    }

    /// True while `node_id` has heartbeated within the timeout window.
    pub fn is_alive(&self, node_id: u32, now: Instant) -> bool {
        match self.last_seen.get(&node_id) {
            Some(&(seen, _)) => now.duration_since(seen) <= self.timeout,
            None => false,
        }
    }

    /// Ids of all tracked nodes that have gone quiet.
    pub fn dead_nodes(&self, now: Instant) -> Vec<u32> {
        self.last_seen
            .iter()
            .filter(|&(_, &(seen, _))| now.duration_since(seen) > self.timeout)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Number of distinct nodes heard from at least once.
    pub fn tracked(&self) -> usize {
        self.last_seen.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hb(node: u32, seq: u64) -> HeartbeatMsg {
        HeartbeatMsg {
            node_id: node,
            seq,
            timestamp_ms: 0,
            status: NodeStatus::Healthy,
        }
    }

    #[test]
    fn roundtrip_through_compact_bytes() {
        let m = hb(7, 42);
        let bytes = m.encode();
        assert_eq!(HeartbeatMsg::decode(&bytes).unwrap(), m);
        // bincode fixed-int: u32 id + u64 seq + u64 ts + u32 variant tag.
        assert_eq!(bytes.len(), 24);
    }

    #[test]
    fn detector_flags_only_silent_nodes() {
        let mut det = FailureDetector::new(Duration::from_millis(100));
        let t0 = Instant::now();
        det.record(&hb(1, 1), t0);
        det.record(&hb(2, 1), t0);
        let later = t0 + Duration::from_millis(150);
        assert!(det.is_alive(1, t0 + Duration::from_millis(99)));
        assert!(!det.is_alive(1, later));
        assert_eq!(det.dead_nodes(later), vec![1, 2]);
        // A fresh beat revives node 1 only.
        det.record(&hb(1, 2), later);
        assert_eq!(det.dead_nodes(later), vec![2]);
        assert_eq!(det.tracked(), 2);
    }

    #[test]
    fn unknown_nodes_are_not_alive() {
        let det = FailureDetector::new(Duration::from_secs(1));
        assert!(!det.is_alive(99, Instant::now()));
    }
}
