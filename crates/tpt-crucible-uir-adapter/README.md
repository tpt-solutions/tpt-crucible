# tpt-crucible-uir-adapter

Converts between the two intermediate representations used by the TPT
toolchain:

* **TPT-IR** — [`tpt_crucible_common::Graph`], the hardware-agnostic compute
  graph produced by Catalyst.
* **TPT-UIR** — a Crucible-dialect [`Region`] (single block of
  `tpt_crucible.*` operations), consumed by the external
  [TPT-UIR](https://github.com/tpt-solutions/tpt-uir) tools
  (`tpt-uir-cli`, `tpt-uir-text`, `tpt-uir-flatbuffers`).

## Usage

```rust
use tpt_crucible_uir_adapter::{graph_to_region, region_to_graph};

let region = graph_to_region(&graph);          // TPT-IR  -> TPT-UIR
let back = region_to_graph(&region).unwrap(); // TPT-UIR -> TPT-IR
assert_eq!(back, graph);                       // lossless round trip
```

Design notes:

* Every TPT-IR node becomes one Crucible operation; node names survive via a
  `node_name` attribute because value ids alone would not carry them.
* Graph inputs become `tpt_crucible.input` ops instead of block arguments so
  their names (and types) stay attached to an operation.
* Constant tensors embed their payload as an `AttributeValue::Bytes`.
* Graph-level `name`/`metadata` ride on a trailing `tpt_crucible.graph_info`
  op, which has no equivalent field on a bare `Region`.

Serialization uses `tpt-uir-serde` (postcard):

```bash
tpt ingest model.gguf --uir model.tptuir      # emit a region from the CLI
tpt-uir-cli validate model.tptuir --dialect crucible
```

## Status

Functional and round-trip tested against a Llama-block-shaped graph.
`tpt-uir` is currently a sibling-checkout path dependency
(`../../../tpt-uir/crates/...`) — revisit once/if `tpt-uir` publishes to
crates.io (tracked in the workspace `todo.md`).