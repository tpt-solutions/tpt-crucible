//! Live coordinator bridge: serial heartbeats from a real node into the
//! production [`FailureDetector`].
//!
//! Pairs with the `hardware/esp32c3-heartbeat` firmware (28-byte `TPTH`
//! records over USB-Serial/JTAG). Run it, watch ALIVE heartbeats land, then
//! unplug the board - the detector flags the node dead exactly per the
//! timeout, which is the fault-tolerance contract on real wires.
//!
//! ```bash
//! cargo run -p tpt-crucible-alloy --features serial \
//!     --example heartbeat_live -- COM5 2500
//! ```
//!
//! Arguments: `<port> [timeout_ms] [baud]`.

use std::time::{Duration, Instant};

use tpt_crucible_alloy::heartbeat::{FailureDetector, HeartbeatMsg};

/// Record layout emitted by the firmware: magic + bincode HeartbeatMsg.
const RECORD_LEN: usize = 28;
const MAGIC: [u8; 4] = *b"TPTH";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let port_name = args.next().unwrap_or_else(|| "COM5".into());
    let timeout_ms: u64 = args
        .next()
        .map(|v| v.parse().map_err(|_| "timeout_ms must be a number"))
        .transpose()?
        .unwrap_or(2500);
    let baud: u32 = args
        .next()
        .map(|v| v.parse().map_err(|_| "baud must be a number"))
        .transpose()?
        .unwrap_or(115_200);

    let started = Instant::now();
    println!("bridge : {port_name} @ {baud} baud, failure timeout {timeout_ms} ms");
    println!("action : unplug the board once heartbeats flow to watch liveness fail");

    let mut port = match serialport::new(&port_name, baud)
        .timeout(Duration::from_millis(100))
        .open()
    {
        Ok(p) => p,
        Err(e) => {
            eprintln!("bridge : could not open {port_name} ({e})");
            eprintln!("hint   : close other readers, or replug the board to clear a wedged USB-CDC handle");
            return Err(e.into());
        }
    };
    println!("link   : opened");

    let mut detector = FailureDetector::new(Duration::from_millis(timeout_ms));
    let mut sync = [0u8; 4]; // rolling window matching the TPTH magic
    let mut beats = 0u64;

    let first_beat_deadline = started + Duration::from_secs(10);
    let hard_stop = started + Duration::from_secs(30);

    loop {
        let now = Instant::now();

        // Failsafes so the bridge always terminates.
        if beats == 0 && now > first_beat_deadline {
            println!("bridge : no heartbeat within 10 s - is the node flashed and enumerated?");
            return Ok(());
        }
        if now > hard_stop {
            println!("bridge : {timeout_ms} ms window never breached within 30 s - node stayed alive");
            return Ok(());
        }

        // Liveness verdict first: silence is as meaningful as traffic.
        if beats > 0 && !detector.is_alive(0, now) {
            println!(
                "[{:>7.3}s] DEAD     node=0 (silent past {timeout_ms} ms) - coordinator would bypass",
                now.duration_since(started).as_secs_f64()
            );
            break;
        }

        let mut byte = [0u8; 1];
        match port.read(&mut byte) {
            Ok(0) => {}
            Ok(_) => {
                sync.copy_within(1.., 0);
                sync[3] = byte[0];
                if sync != MAGIC {
                    continue;
                }

                // Magic matched: read the remaining record bytes.
                let mut rest = vec![0u8; RECORD_LEN - 4];
                let mut got = 0;
                while got < rest.len() {
                    match port.read(&mut rest[got..]) {
                        Ok(0) => {}
                        Ok(n) => got += n,
                        Err(e) if e.kind() == std::io::ErrorKind::TimedOut => continue,
                        Err(e) => return link_lost(started, &e),
                    }
                }
                let msg = HeartbeatMsg::decode(&rest)?;
                detector.record(&msg, Instant::now());
                beats += 1;
                println!(
                    "[{:>7.3}s] HEARTBEAT node={} seq={} uptime={}ms -> ALIVE",
                    now.duration_since(started).as_secs_f64(),
                    msg.node_id,
                    msg.seq,
                    msg.timestamp_ms
                );
            }
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {}
            Err(e) => return link_lost(started, &e),
        }
    }

    println!(
        "summary: {} heartbeat(s) accepted, liveness failed after {} ms of silence",
        beats, timeout_ms
    );
    Ok(())
}

fn link_lost(started: Instant, e: &std::io::Error) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "[{:>7.3}s] LINK LOST ({e}) - heartbeats stopped; the detector's timeout \
         window decides liveness from here",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
