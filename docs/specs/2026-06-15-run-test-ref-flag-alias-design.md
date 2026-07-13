# Run-Test Ref Flag Alias Design

## Summary

- Rename the run-test differential comparison inputs to use `ref-*` naming for consistency with `compare-result`.
- Keep the legacy `baseline-*` flags as CLI aliases so existing scripts continue to work.
- Update parser help text, validation messages, MCP forwarding, and user-facing docs to present `ref-*` as the canonical spelling.

## User-Visible Behavior

- `run-test` and `run-test-optimize` should accept:
  - `--ref-result`
  - `--ref-operator-file`
- They should also continue accepting:
  - `--baseline-result`
  - `--baseline-operator-file`
- Help text, examples, and error messages should prefer `ref-*`.
- Automatic differential comparison semantics remain unchanged.

## Implementation Notes

- Update both CLI entrypoints:
  - `skills/triton-npu-run-eval/scripts/run-command.py`
  - `src/helix/cli.py`
- Update downstream argument consumers to read `args.ref_result` and `args.ref_operator_file`.
- Preserve archive derivation and baseline execution behavior; this is a naming cleanup, not a workflow redesign.
- Update MCP server argument emission to forward canonical `--ref-*` flags.

## Verification

- Add parser tests for canonical `ref-*` flags and legacy aliases.
- Run focused command, CLI, and skill-script regression tests.
- Run strict pyright for modified skill scripts plus repo-level `ruff` and `pyright`.
