# Optimize Supervisor Alias Design

## Summary

Add `--supervisor` as a CLI alias for the existing `--supervise on|off` option.

## User-Visible Behavior

- `helix optimize ... --supervisor on` behaves the same as `--supervise on`.
- `helix optimize-batch ... --supervisor off` behaves the same as `--supervise off`.
- The parsed value still lands in `args.supervise`, so downstream runtime behavior and validation stay unchanged.
- The default remains `off`.

## Scope

- Update optimize command parsing only.
- Cover both `optimize` and `optimize-batch`.
- Add parser tests for the alias.

## Non-Goals

- No runtime orchestration changes.
- No new request fields or separate `args.supervisor` handling.
- No change to the meaning of supervised optimize mode.
