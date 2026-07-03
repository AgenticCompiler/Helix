---
name: ascend-npu-run-eval
description: Execute and evaluate generated operator artifacts. Use when you need to run generated test cases, run generated benchmark cases, profile benchmark harnesses, summarize profiling data, or compare result and performance artifacts, including during optimization workflows.
---

# Run-Eval Router

Use the bundled helper script in this skill:

```bash
python3 ./scripts/run-command.py <subcommand> ...
```

Read only the focused guide for the subcommand you are about to run:

- `run-test-baseline` / `run-test-optimize`: [references/run-test.md](references/run-test.md)
- `run-bench`: [references/run-bench.md](references/run-bench.md)
- `profile-bench`: [references/profile-bench.md](references/profile-bench.md)
- `profile-report`: [references/profile-report.md](references/profile-report.md)
- `compare-result`: [references/compare-result.md](references/compare-result.md)
- `compare-perf`: [references/compare-perf.md](references/compare-perf.md)

During normal use:

- call `python3 ./scripts/run-command.py <subcommand> ...` directly
- do not read unrelated command guides
- do not reread Python files under `./scripts/` unless you need to debug, patch, or verify helper behavior
