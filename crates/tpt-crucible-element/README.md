# tpt-crucible-element

Analog backend for TPT Crucible (Phase 3, "The Physics Engine"). Will map
TPT-IR weights onto physical electrical components, generate SPICE netlists for
Xyce/ngspice, run the "Reality Check" engine (thermal noise, voltage drift,
component tolerances), and emit PCB recommendations with an ML confidence
score.

Status: API contract only; returns `NotImplemented`. Track progress in the
[workspace todo.md](https://github.com/tpt-solutions/tpt-crucible/blob/master/todo.md).