# Run-Eval Review Follow-Ups Plan

1. Add regression tests that lock in the desired end state:
   - shared `result_payload.py` exists and key run-eval scripts stop defining local payload helpers;
   - `run-command.py` restores `sys.path` after dynamic imports;
   - `bench_runner.py` no longer uses the globals-backed service locator.
2. Introduce the shared payload helper and update run-eval scripts to import it.
3. Update standalone-runtime support-file staging so copied/runtime-only flows receive the new helper module.
4. Replace the bench-runner dependency locator with an explicit typed dependency object and typed submodule contracts.
5. Run focused unittests plus file-scoped strict pyright for touched skill scripts, then fix any regressions.
