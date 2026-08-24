# tpt-crucible-fusion

FPGA backend for TPT Crucible (Phase 2, "The Silicon Canvas"). Will compile
TPT-IR into `.fusecfg` overlay configurations that wire MAC arrays into
pre-verified HBM memory controllers - targeting ~10 s per-model compiles
instead of full resynthesis.

Status: API contract only; returns `NotImplemented`. Track progress in the
[workspace todo.md](https://github.com/tpt-solutions/tpt-crucible/blob/master/todo.md).