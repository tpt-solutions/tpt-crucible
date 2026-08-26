//! Conformance between the ESP32-C3 heartbeat firmware (`hardware/
//! esp32c3-heartbeat`) and the coordinator-side protocol stack.
//!
//! The four records below are **captured off the physical board** over
//! COM5 immediately after flashing (see hardware/README.md): 28-byte
//! records, magic `TPTH`, then the bincode encoding of
//! `HeartbeatMsg { node_id: 0, seq: 1..=4, timestamp_ms: seq*1000,
//! NodeStatus::Healthy }`.
//!
//! The test proves the shipped silicon's bytes decode through the
//! production codec and drive the production failure detector exactly as
//! the field pipeline expects.

use std::time::{Duration, Instant};

use tpt_crucible_alloy::heartbeat::{FailureDetector, HeartbeatMsg};

/// Live captures from the ESP32-C3 (node id 0), one second apart.
const CAPTURED_RECORDS: [&[u8; 28]; 4] = [
    b"TPTH\0\0\0\0\x01\0\0\0\0\0\0\0\xe8\x03\0\0\0\0\0\0\0\0\0\0",
    b"TPTH\0\0\0\0\x02\0\0\0\0\0\0\0\xd0\x07\0\0\0\0\0\0\0\0\0\0",
    b"TPTH\0\0\0\0\x03\0\0\0\0\0\0\0\xb8\x0b\0\0\0\0\0\0\0\0\0\0",
    b"TPTH\0\0\0\0\x04\0\0\0\0\0\0\0\xa0\x0f\0\0\0\0\0\0\0\0\0\0",
];

#[test]
fn firmware_records_decode_through_the_production_codec() {
    for (i, record) in CAPTURED_RECORDS.iter().enumerate() {
        let msg = HeartbeatMsg::decode(&record[4..]).expect("firmware bytes are valid heartbeats");
        assert_eq!(msg.node_id, 0);
        assert_eq!(msg.seq, i as u64 + 1);
        assert_eq!(msg.timestamp_ms, (i as u64 + 1) * 1000);
        assert_eq!(
            msg,
            HeartbeatMsg {
                node_id: 0,
                seq: i as u64 + 1,
                timestamp_ms: (i as u64 + 1) * 1000,
                status: tpt_crucible_alloy::heartbeat::NodeStatus::Healthy,
            }
        );
    }
}

#[test]
fn detector_sees_the_real_node_alive_then_dead_after_timeout() {
    let mut detector = FailureDetector::new(Duration::from_millis(2500));

    // Feed each captured record at its wall-clock arrival (1 s cadence).
    let t0 = Instant::now();
    let arrivals: Vec<(Instant, HeartbeatMsg)> = CAPTURED_RECORDS
        .iter()
        .map(|r| HeartbeatMsg::decode(&r[4..]).unwrap())
        .enumerate()
        .map(|(i, msg)| (t0 + Duration::from_millis(i as u64 * 1000), msg))
        .collect();

    let last_beat = arrivals.last().unwrap().0;
    for (at, msg) in &arrivals {
        detector.record(msg, *at);
    }

    // Alive right up to just-before the timeout window closes...
    let before_expiry = last_beat + Duration::from_millis(2499);
    assert!(detector.is_alive(0, before_expiry));
    assert!(detector.dead_nodes(before_expiry).is_empty());

    // ...and flagged dead once the node goes quiet past the timeout.
    let after_expiry = last_beat + Duration::from_millis(2501);
    assert!(!detector.is_alive(0, after_expiry));
    assert_eq!(detector.dead_nodes(after_expiry), vec![0]);
}
