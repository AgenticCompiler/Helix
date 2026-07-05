# PT and PROF Cleanup Policy

## User-Visible Semantics

`TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES` is an enum-style environment variable with these values:

- `never`: do not delete ordinary `*_result.pt` files.
- `round`: delete ordinary `*_result.pt` files after an optimize round passes the round contract.
- `run-test`: delete ordinary optimized `*_result.pt` files immediately after each `run-test-optimize` helper command has finished using the result for optional comparison. `run-test-baseline` preserves result archives so later optimize validation can reuse them.

The default is `round`.

For compatibility with the previous boolean form, `1`, `true`, `yes`, and `on` are aliases for `round`; `0`, `false`, `no`, and `off` are aliases for `never`.

After an optimize round passes the round contract, delete `PROF_*` artifacts in that round directory. `profile-bench` must not delete `PROF_*` artifacts immediately because the agent may still need to inspect the raw profiler output before submitting the round. Cleanup failures are best-effort and must not change the parent command result.

## Implementation Notes

The optimize-state round checker remains the round-end authority. It owns round-local cleanup after `check_round()` has accepted the round. The run-eval command helper owns immediate `run-test-optimize` cleanup because that event occurs inside the helper script after optional result comparison.

Cleanup continues to target only ordinary tensor result archives named `test_result.pt` or ending in `_result.pt`, case-insensitively. `PROF_*` cleanup removes files or directories whose names start with `PROF_` only in the selected local directory.
