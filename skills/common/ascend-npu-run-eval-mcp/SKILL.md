---
name: ascend-npu-run-eval
description: Execute and evaluate generated operator artifacts. Use when you need to run generated test cases, run generated benchmark cases, profile benchmark harnesses, summarize profiling data, or compare result and performance artifacts, including during optimization workflows.
---

# Run-Eval Router

Use the corresponding MCP tool for run-eval actions in this staged skill.

Primary MCP tools:

- `run-test-baseline`
- `run-test-convert`
- `run-test-optimize`
- `run-bench`
- `profile-bench`
- `profile-report`
- `compare-perf`

Fast-screening note:

- `probe-bench` is not currently exposed as an MCP tool in this surface.
- When you need a fast baseline-vs-candidate screen, use the non-MCP `ascend-npu-run-eval` skill or the public `triton-agent probe-bench` command if that workspace exposes it.
- If neither surface is available, fall back to canonical `run-bench` plus `compare-perf`.

Read only the focused guide for the MCP tool you are about to call:

- `run-test-baseline` / `run-test-convert` / `run-test-optimize`: [references/run-test.md](references/run-test.md)
- `run-bench`: [references/run-bench.md](references/run-bench.md)
- `profile-bench`: [references/profile-bench.md](references/profile-bench.md)
- `profile-report`: [references/profile-report.md](references/profile-report.md)
- `compare-perf`: [references/compare-perf.md](references/compare-perf.md)

During normal agent use:

- use the corresponding MCP tool
- keep the same arguments and artifact expectations documented in the focused reference
