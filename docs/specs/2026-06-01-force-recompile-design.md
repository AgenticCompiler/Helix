# Force Recompile Design

## Summary

- Add `--force-recompile` flag to `run-test`, `run-bench`, and `profile-bench` subcommands.
- When set, inject `TRITON_ALWAYS_COMPILE=1` into the Triton kernel compilation environment.
- Support local, remote, standalone, and msprof execution modes.
- Do not change default behavior — kernels reuse cache unless the flag is explicitly passed.

## Problem

Triton kernels are cached by default. When iterating on kernel code during optimize rounds, cached kernels may mask changes, leading to misleading benchmark results. Operators need a way to force kernel recompilation to ensure measurements reflect the current code.

## Design

### Environment Variable

The canonical Triton flag for forcing recompilation is `TRITON_ALWAYS_COMPILE=1`. This is already used in the IR capture workflow (`skills/triton-npu-analyze-ir/scripts/capture_ir.py`). This design reuses the same mechanism.

### Injection Strategy

| Execution Mode      | Injection Method |
|---------------------|-----------------|
| Local subprocess    | `extra_env={"TRITON_ALWAYS_COMPILE": "1"}` via `run_streaming_process` / `run_buffered_process` |
| Remote subprocess   | `extra_env={"TRITON_ALWAYS_COMPILE": "1"}` via `run_remote_command_streaming` / `run_remote_command_buffered` |
| In-process (standalone bench / standalone profile) | `os.environ["TRITON_ALWAYS_COMPILE"] = "1"` with save/restore in finally block |

### CLI Surface

```
triton-agent run-test --test-file test.py --operator-file op.py --force-recompile
triton-agent run-bench --bench-file bench.py --operator-file op.py --force-recompile
```

At the skill level (used by agents):
```
python3 run-command.py run-test --test-file test.py --operator-file op.py --force-recompile
python3 run-command.py run-bench --bench-file bench.py --operator-file op.py --force-recompile
python3 run-command.py profile-bench --bench-file bench.py --operator-file op.py --force-recompile
```

### Flag Registration

- `cli.py`: Add `has_force_recompile: bool = False` to `_CommandSpec` dataclass.
- `cli.py`: Set `has_force_recompile=True` on `RUN_TEST` and `RUN_BENCH` specs.
- `cli.py`: Register `--force-recompile` store_true argument in `build_parser`.
- `run-command.py`: Add `--force-recompile` to `run-test`, `run-bench`, and `profile-bench` subparsers.

### Threading Chain

```
CLI args.force_recompile
  → handle_run_test / handle_run_bench (commands/execution.py)
    → run_local_test / run_remote_test (src/execution.py)
      → test_runner.run_local_test / run_remote_test (skill)
        → _run_legacy_local_test / _run_legacy_remote_test
          → run_streaming_process(..., extra_env=...)
```

Same pattern for bench and profile-bench through their respective runners.

### Files Modified

| File | Change |
|------|--------|
| `src/triton_agent/cli.py` | Add `has_force_recompile` field; register flag; set on RUN_TEST/RUN_BENCH |
| `src/triton_agent/commands/execution.py` | Thread `args.force_recompile` through handlers |
| `src/triton_agent/execution.py` | Add `force_recompile` to bridge function signatures |
| `skills/triton-npu-run-eval/scripts/run-command.py` | Add flag to subparsers; pass through dispatch |
| `skills/triton-npu-run-eval/scripts/test_runner.py` | Accept and inject `force_recompile` for local/remote |
| `skills/triton-npu-run-eval/scripts/bench_runner.py` | Accept and thread `force_recompile` |
| `skills/triton-npu-run-eval/scripts/bench_runner_standalone.py` | Merge `TRITON_ALWAYS_COMPILE` into `extra_env` |
| `skills/triton-npu-run-eval/scripts/bench_runner_msprof.py` | Merge `TRITON_ALWAYS_COMPILE` into `extra_env` |
| `skills/triton-npu-run-eval/scripts/profile_runner.py` | Inject for local subprocess / in-process / remote |
| `skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py` | In-process `os.environ` for standalone bench |

### Non-Goals

- This does not add a persistent config or global default for force-recompile.
- This does not modify the optimize workflow to auto-enable force-recompile.
- This does not add `--force-recompile` to `gen-test` or `gen-bench` (only execution commands).
