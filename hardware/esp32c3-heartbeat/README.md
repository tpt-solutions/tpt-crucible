# ESP32-C3 Heartbeat Node

Real-hardware reference node for the Alloy swarm: a `no_std` esp-hal
firmware that broadcasts the **Alloy heartbeat protocol** over the board's
native USB-Serial/JTAG port, one 28-byte record per second:

```text
"TPTH" | node_id u32le | seq u64le | timestamp_ms u64le | status u32le
```

Bytes 4..28 are byte-for-byte `HeartbeatMsg` encoded with the same bincode
layout the coordinator stack consumes (`crates/tpt-crucible-alloy/src/
heartbeat.rs`). Wire compatibility is regression-tested against a live
capture of this exact binary in
`crates/tpt-crucible-alloy/tests/heartbeat_firmware_conformance.rs`.

## Provisioned hardware profile

| Field   | Value                          |
| ------- | ------------------------------ |
| Chip    | ESP32-C3, QFN32, rev v0.4      |
| Radio   | Wi-Fi + BT 5 (LE), single core |
| Clock   | 160 MHz                        |
| Flash   | 4 MB embedded (XMC)            |
| USB     | USB-Serial/JTAG (`COM5`)       |
| MAC     | `e8:3d:c1:83:72:d8`            |

`fleet.json` carries the same profile as an alloy `topology::SwarmNode`
entry so planning can start from discovered reality instead of guesses.

## Build / flash / verify

```bash
# one-time tools
rustup target add riscv32imc-unknown-none-elf
cargo install espflash          # v3.x line pairs with esp-hal 0.22
python -m pip install --user esptool   # optional, chip probing

# build + flash + observe
cargo build --release
espflash flash --port COM5 target/riscv32imc-unknown-none-elf/release/esp32c3-heartbeat
# open COM5 @115200 - records arrive every second
```

## Panic semantics

Panics halt the node (no heartbeat handler): heartbeats stop, and the
coordinator's failure detector flags the node dead - precisely the graceful
degradation the protocol defines. A richer panic path (last-gasp
`Degraded` frame) is tracked in workspace todo.md.
