# tpt-crucible-alloy

Swarm backend for TPT Crucible: partition TPT-IR across microcontroller fleets
and generate per-node firmware.

* **Topology auto-discovery** from node-reported latency/bandwidth matrices
* **Partitioning** - topology-aware greedy planner with layer-serial,
  attention-head-parallel, and hybrid (transformer-native) strategies
* **KV-cache planning** - distributes heads across nodes up-front; refuses
  plans that would OOM instead of failing in the field
* **Heartbeat protocol** - compact bincode frames plus a failure detector for
  dead-node bypass
* **Firmware generation** - per-node Rust/C++ skeletons, JSON shard manifests,
  and parallel esptool flash scripts (.sh + .ps1)

Pure Rust and wasm-compatible: this crate powers the zero-install browser SiL
demo.

See the [workspace README](https://github.com/tpt-solutions/tpt-crucible).