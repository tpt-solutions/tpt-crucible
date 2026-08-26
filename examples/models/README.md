# Fixture models

Tiny synthetic-but-valid models checked in so examples, docs walkthroughs,
and manual smoke tests exercise the real ingestion paths without shipping
anyone's weights.

| File | Format | Contents |
|---|---|---|
| `tiny-llama-block.gguf` | GGUF v3 | ~1.7 KB: four llama-hyperparameter kv pairs (`block_count=1`, `head_count=2`, `key_length=8`) + two f32 weight tensors |

## Usage

```bash
# Ingest it exactly like a real download:
tpt ingest examples/models/tiny-llama-block.gguf --output /tmp/tiny.tptir

# Full software-only pipeline (ingest -> compile --target alloy -> preflight):
cargo run -p tpt-crucible-cli --features swarm --example software_e2e
```

## Regenerating

The file is produced by a checked-in generator so its provenance stays
auditable and the bytes stay reproducible:

```bash
cargo run -p tpt-crucible-catalyst --example write_fixture_gguf
```
