# TPT Crucible developer recipes — `just <recipe>` (https://github.com/casey/just)
#
# Wraps the exact pre-PR triad documented in CONTRIBUTING.md so contributors
# don't hand-run three commands; `just` with no arguments runs that triad.

default: pre-pr

# Format every crate (run before committing).
fmt:
    cargo fmt --all

# Lint the whole workspace at CI strictness (-D warnings).
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Run the full host test suite.
test:
    cargo test --workspace

# The pre-PR triad from CONTRIBUTING.md — exactly what CI enforces.
pre-pr: fmt clippy test

# Compile-check the browser-demo crates for wasm32 (no default features,
# matching the wasm job in .github/workflows/ci.yml).
check-wasm:
    cargo check -p tpt-crucible-common -p tpt-crucible-catalyst -p tpt-crucible-alloy --target wasm32-unknown-unknown --no-default-features

# Lint + test with every hardware cargo feature enabled (fpga + swarm + hub),
# mirroring the `features` CI job.
features:
    cargo clippy --workspace --all-targets --features fpga,swarm,hub -- -D warnings
    cargo test --workspace --features fpga,swarm,hub

# Cross-check the ESP32-C3 heartbeat firmware (needs espup: cargo install espup && espup install).
firmware:
    cd hardware/esp32c3-heartbeat && cargo +esp check --release

# Build the workspace docs and warn about anything missing.
doc:
    cargo doc --workspace --no-deps

# Serve the Observer dashboard frontend locally (trunk; see
# crates/tpt-crucible-observer-web/README.md).
dashboard:
    cd crates/tpt-crucible-observer-web && trunk serve

# Remove build artifacts and local debug logs.
clean:
    cargo clean
    rm -f dbg.log hw_bridge.log hw_bridge.err