# Force Recompile Implementation Plan

## Goal

Add `--force-recompile` flag to `run-test`, `run-bench`, and `profile-bench` subcommands to force Triton kernel recompilation via `TRITON_ALWAYS_COMPILE=1`.

## Steps

- [x] Add `has_force_recompile` field to `_CommandSpec` and register on `RUN_TEST` / `RUN_BENCH`
- [x] Register `--force-recompile` argument in `build_parser`
- [x] Thread `args.force_recompile` through `handle_run_test` / `handle_run_bench`
- [x] Add `force_recompile` to bridge function signatures in `src/triton_agent/execution.py`
- [x] Update `TestRunnerModule` and `BenchRunnerModule` Protocol signatures
- [x] Add `force_recompile` arg to `run-command.py` subparsers and dispatch
- [x] Inject `extra_env` / `os.environ` for local test runner
- [x] Inject `extra_env` for remote test runner
- [x] Thread through bench runner (all local/remote/standalone/msprof paths)
- [x] Thread through profile runner (local subprocess, in-process, remote paths)
- [x] Handle in-process standalone bench (`standalone_bench_runtime.py`)
- [x] Handle in-process standalone profile (`profile_runner.py`)
- [ ] Fix existing test mock assertions to include `force_recompile=False`
- [ ] Create PR
