//! ESP32-C3 swarm node firmware: broadcasts the Alloy heartbeat protocol
//! over the native USB-Serial/JTAG port.
//!
//! Every second the node emits one 28-byte record:
//!
//! ```text
//! "TPTH" (4) | node_id u32le | seq u64le | timestamp_ms u64le | status u32le
//! ```
//!
//! `TPTH` is a resync magic for byte-oriented serial links; the 24 bytes
//! after it are byte-for-byte the wire format of
//! `tpt_crucible_alloy::heartbeat::HeartbeatMsg` encoded with bincode
//! (`NodeStatus::Healthy` = tag 0), so a host-side
//! [`FailureDetector`](tpt_crucible_alloy) consumes records from this node
//! unchanged. The struct is duplicated here on purpose: the alloy crate is
//! std-bound, this firmware is `no_std`.
//!
//! Flash with: cargo install espflash && espflash flash --port COM5 <elf>

#![no_std]
#![no_main]

use esp_hal::delay::Delay;
use esp_hal::usb_serial_jtag::UsbSerialJtag;

/// Swarm id of this physical node (single-node fleet for now).
const NODE_ID: u32 = 0;

/// Wire tag of `heartbeat::NodeStatus::Healthy` in the shared codec.
const STATUS_HEALTHY: u32 = 0;

/// Record magic so a host can resynchronize mid-stream.
const MAGIC: [u8; 4] = *b"TPTH";

/// Halt on panic: heartbeats stop, and the coordinator's failure detector
/// flags this node dead - exactly the degradation the protocol expects.
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}

#[esp_hal::entry]
fn main() -> ! {
    let peripherals = esp_hal::init(esp_hal::Config::default());

    let mut usb = UsbSerialJtag::new(peripherals.USB_DEVICE);
    let delay = Delay::new();

    let mut seq: u64 = 0;
    loop {
        let mut frame: [u8; 28] = [0; 28];
        frame[0..4].copy_from_slice(&MAGIC);
        frame[4..8].copy_from_slice(&NODE_ID.to_le_bytes());
        frame[8..16].copy_from_slice(&seq.to_le_bytes());
        // Timestamp is cadence-based: one record per second of uptime.
        let timestamp_ms: u64 = seq.saturating_mul(1_000);
        frame[16..24].copy_from_slice(&timestamp_ms.to_le_bytes());
        frame[24..28].copy_from_slice(&STATUS_HEALTHY.to_le_bytes());

        let _ = usb.write_bytes(&frame);

        seq = seq.wrapping_add(1);
        delay.delay_millis(1_000);
    }
}
