# Force Recompile Implementation Plan

## Goal

Always force Triton kernel recompilation via `TRITON_ALWAYS_COMPILE=1` for `run-test`, `run-bench`, `profile-bench`, `verify`, and `verify-batch`. No CLI flag; always-on behavior.

## Steps

- [x] Add `has_force_recompile` field to `_CommandSpec` and register on `RUN_TEST` / `RUN_BENCH`
- [x] Register `--force-recompile` argument in `build_parser`
- [x] Thread `args.force_recompile` through `handle_run_test` / `handle_run_bench`
- [x] Add `force_recompile` to bridge function signatures in `src/helix/execution.py`
- [x] Update `TestRunnerModule` and `BenchRunnerModule` Protocol signatures
- [x] Add `force_recompile` arg to `run-command.py` subparsers and dispatch
- [x] Inject `extra_env` / `os.environ` for local test runner
- [x] Inject `extra_env` for remote test runner
- [x] Thread through bench runner (all local/remote/standalone/msprof paths)
- [x] Thread through profile runner (local subprocess, in-process, remote paths)
- [x] Handle in-process standalone bench (`standalone_bench_runtime.py`)
- [x] Handle in-process standalone profile (`profile_runner.py`)
- [x] Fix existing test mock assertions
- [x] Change to always-on (force_recompile=True), remove all CLI flags
- [x] Extend to verify / verify-batch commands
- [x] Create PR
