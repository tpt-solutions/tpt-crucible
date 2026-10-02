# fix(common,catalyst): compile on stable rustc and repair shape inference

`tpt-crucible-common` currently fails to build on stable rustc, which breaks
any consumer using it as a git dependency:

    error[E0425]: cannot find type `TensorDesc` in this scope (graph.rs:230,232)
    error[E0283]: type annotations needed (ops.rs:569)

The test suite additionally could not compile, which had hidden several bugs.
This PR makes the lib, tests, and examples compile, and fixes the real bugs
the compiling suite then surfaced:

- **attention**: rank-0 (scalar) inputs panicked slicing `q.shape`; unknown
  rank now propagates unchanged.
- **reshape**: the `-1` slot was divisibility-checked but never *written* with
  the inferred dimension; the divisor also ignored `0`-copied dims
  (`[0, -1]` targets). Inference now writes the quotient over all resolved
  dims.
- **broadcast**: right-aligned indexing underflowed when the other operand
  had higher rank; now falls back to the implicit leading-1 dimension.
- ops tests: the attention test moved `k`/`v` out of the descriptor array.
- catalyst example: `PathBuf` formatted with `{}` instead of `.display()`.

## Results

- `cargo check -p tpt-crucible-common -p tpt-crucible-catalyst`: clean
- `cargo test`: common 36/36 green; catalyst 79 pass / 3 fail

## Known pre-existing failures (surfaced, not fixed here)

The three remaining catalyst failures are distinct parser/lowering issues,
each with its own repro, best handled in follow-ups:

- `onnx::tests::gemm_transb_lowers_to_transpose_matmul_add` - lowered graph
  reports `matmul inner dimensions to agree (2 == 4)`
- `tensorflow::tests::concat_v2_takes_axis_from_last_const` - `concat axis 1
  is out of range for rank 1`
- `tflite::tests::batch_matmul_adjoint`

(Note for reviewers: `crates/tpt-crucible-cli` / `tpt-crucible-uir-adapter`
require the `../tpt-uir` sibling checkout per their documented Cargo.toml
comments; this PR does not touch them.)
