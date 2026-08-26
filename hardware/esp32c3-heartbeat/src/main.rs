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
//! ## Node identity
//!
//! The swarm id is derived from the chip's factory eFuse MAC address (low 32
//! bits), so every board gets a unique, stable id with zero configuration —
//! flashing a second board can never collide with the first. To pin a
//! specific id instead, build with `TPT_NODE_ID=<n> cargo build --release`.
//!
//! Flash with: cargo install espflash && espflash flash --port COM5 <elf>

#![no_std]
#![no_main]

use esp_hal::delay::Delay;
use esp_hal::efuse::Efuse;
use esp_hal::usb_serial_jtag::UsbSerialJtag;

/// Build-time override for the swarm id (`TPT_NODE_ID=7 cargo build`).
///
/// Unset or unparseable values fall back to MAC derivation below.
const NODE_ID_OVERRIDE: Option<u32> =
    option_env!("TPT_NODE_ID").and_then(|v| v.parse::<u32>().ok());

/// Derive a stable swarm id from the factory MAC (low 32 bits).
///
/// The two high bytes of the ESP32-C3 MAC are an Espressif OUI prefix and
/// identical across boards; the low four bytes are what distinguishes them.
fn mac_node_id() -> u32 {
    let mac = Efuse::mac_address();
    u32::from_be_bytes([mac[2], mac[3], mac[4], mac[5]])
}

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

    let node_id = NODE_ID_OVERRIDE.unwrap_or_else(mac_node_id);

    // Boot banner. Hosts resynchronize on the TPTH magic, so these leading
    // non-record bytes are skipped harmlessly by the serial bridge. Written
    // in pieces: no_std here means no `format!` (no alloc).
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut hex: [u8; 8] = [0; 8];
    for (i, byte) in node_id.to_be_bytes().iter().enumerate() {
        hex[i * 2] = HEX[(byte >> 4) as usize];
        hex[i * 2 + 1] = HEX[(byte & 0x0f) as usize];
    }
    let _ = usb.write_bytes(b"tpt heartbeat firmware: node_id=0x");
    let _ = usb.write_bytes(&hex);
    let _ = usb.write_bytes(if NODE_ID_OVERRIDE.is_some() {
        &b" (source: TPT_NODE_ID)\r\n"[..]
    } else {
        &b" (source: efuse mac)\r\n"[..]
    });

    let mut seq: u64 = 0;
    loop {
        let mut frame: [u8; 28] = [0; 28];
        frame[0..4].copy_from_slice(&MAGIC);
        frame[4..8].copy_from_slice(&node_id.to_le_bytes());
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
