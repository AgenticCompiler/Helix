# Isolated Triton Workflow Cache Design

## Summary

Triton-language optimize and convert workflows run code agents concurrently. A
shared default Triton cache lets one agent delete files that another agent is
compiling or loading. Each workflow session therefore owns one cache directory
at `<workspace>/.helix-triton-cache/<run-id>`.

## Behavior

- Helix creates a run-specific cache child before launching a Triton agent and
  injects `TRITON_CACHE_DIR`, `TRITON_ALWAYS_COMPILE=1`, and an internal remote
  isolation marker into its environment.
- An optimize cache lease covers the entire multi-invocation session. A convert
  lease covers the agent, verification, and any repair retry. Batch workspaces
  each receive an independent lease.
- Local convert verification receives the lease environment explicitly rather
  than changing the parent environment, so parallel batch workers remain safe.
- Remote execution maps the internal marker to
  `<remote-workspace>/.helix-triton-cache`; the local cache path is never sent
  to the remote host.
- Cleanup removes only the owned run child. It removes the parent only when
  Helix created it and it is empty. Cleanup failures are warnings.

## User-Facing Guidance

Agents receive the exact assigned cache path and are told that recompilation is
already forced. They must not access `~/.triton` and may clear only their own
cache while no evaluation subprocess is active. TileLang workflows are
unchanged.
