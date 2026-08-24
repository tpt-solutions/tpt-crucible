# tpt-crucible-cli

The unified `tpt` binary for the TPT Crucible suite.

```bash
cargo install tpt-crucible-cli
# optional hardware targets:
cargo install tpt-crucible-cli --features fpga,swarm
```

Subcommands:

* `tpt ingest <model> [-o out] [--format gguf|safetensors]` - lower a model to
  TPT-IR
* `tpt info <ir> [--dot]` - inspect an IR artifact
* `tpt compile <ir> --target alloy|fusion|element` - compile for hardware
  (`--nodes`, `--mem-mb`, `--seq-len`, `--out-dir` control the alloy swarm
  plan; firmware + flash scripts land in `--out-dir`)
* `tpt doctor` - verify external toolchain availability

Features: `swarm` (default) enables the ESP32/RP2040/RISC-V path; `fpga`
enables the FPGA overlay path. Targets compiled out fail with a rebuild hint.