# Versioning & Release Policy

Every crate in this workspace versions independently under
[Semantic Versioning 2.0.0](https://semver.org/). This document defines what
breaking means today (the `0.x` era), what graduating a crate to `1.0.0`
requires, and how deprecations and the MSRV are handled.

## The 0.x era

While a crate is `0.y.z`:

* **Breaking change** → bump `y` (minor).
* **Additive / bug-fix** → bump `z` (patch).
* The public surface is whatever `cargo doc` renders plus the Serde data
  formats called out below; anything reachable only through `pub(crate)` is
  fair game.

Rationale: pre-1.0 crates iterate quickly, so downstreams pin `0.y` ranges
and minor bumps carry the breakage signal.

## Data-format stability (independent of crate versions)

Two formats are load-bearing across processes and *must* round-trip:

* TPT-IR **JSON** (`serde_json`) and TPT-IR **binary** (`TPTIR` magic +
  bincode) as produced/consumed by `tpt-crucible-common`.
* Heartbeat frames (`tpt-crucible-alloy::heartbeat`, bincode).

Any change that alters bytes-on-the-wire or rejects previously accepted
input is a breaking change even when the Rust signature is unchanged.
GGUF/SafeTensors/TFLite/… *readers* aim to be liberal: widening accepted
inputs is never breaking, narrowing is.

## Graduating a crate to 1.0.0

A crate may publish `1.0.0` when all of the following hold:

1. Two consecutive `0.y` releases shipped without a breaking `y` bump.
2. Its serde contract (if any, see above) is frozen: additive optional
   fields only.
3. Enumerations modeling external realities (`Op`, `ModelFormat`,
   `NodeArch`, …) commit to *additive-only* growth; new variants are minor
   releases, removals or renames require `2.0.0`.
4. Real-world usage exists beyond this workspace's own CLI/tests.

Expected first wave: `tpt-crucible-common`, then `tpt-crucible-catalyst`.

## MSRV

The workspace MSRV lives in the root `Cargo.toml` (`rust-version`). Bumping
it is a breaking change: `y` bump pre-1.0, `z→major` post-1.0. CI builds on
the pinned toolchain; MSRV regressions block merge.

## Deprecations

Public items are removed only in major releases, and only after carrying a
`#[deprecated]` attribute for at least one full minor release cycle with a
migration note pointing at the replacement.

## Internal dependencies & publishing

Path-only internal dependencies gain an explicit `version` field at first
crates.io publication of the dependency (tracked in todo.md). Crates publish
in dependency order: `common` → `catalyst` → {`fusion`, `element`, `alloy`}
→ `cli`; `observer` and `observer-web` are optional tail ends. `--dry-run`
for the whole workspace must pass before any publish.

## Yanking

Yanking is reserved for security fixes or catastrophic packaging mistakes;
otherwise prefer forward-fix releases so pinned builds keep resolving.
