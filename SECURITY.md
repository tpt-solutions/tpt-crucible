# Security Policy

## Supported versions

TPT Crucible is pre-1.0; only the latest `0.1.x` release on `master` receives
security fixes. Pin exact versions if you embed the crates.

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Use GitHub's private vulnerability reporting on this repository, or email
`security@tpt.solutions` with:

* affected crate(s) and version / commit
* a minimal reproduction or PoC
* impact assessment and any known workarounds

You will receive an acknowledgement within 72 hours and status updates at
least weekly until resolution. We will credit reporters in the release notes
unless you prefer to remain anonymous.

## Scope notes

Areas worth extra scrutiny in this codebase:

* **Untrusted model files** - SafeTensors/GGUF parsing (`tpt-crucible-catalyst`)
  processes attacker-controlled headers; malformed inputs must fail as
  structured errors, never panic.
* **Generated firmware & flash scripts** - `tpt-crucible-alloy` emits shell/
  PowerShell that invokes external tools; template injection or path traversal
  there would run on developer machines.
* **External tool invocation** - `tpt-doctor` and flashing execute programs
  found on `PATH`.
* The workspace forbids `unsafe` code; report any escape hatch you find.

## Disclosure policy

We follow coordinated disclosure: fixes land in a patch release before public
disclosure, with a CVE requested for remotely exploitable issues.