# Run-Eval Torch/NPU Bootstrap Design

## Summary

`triton-npu-run-eval` currently loads user benchmark/test/operator modules with
`importlib.util.module_from_spec(...)` plus `spec.loader.exec_module(...)`.
When a fresh process lets the first `import torch` happen inside those user
modules, later Triton lazy driver resolution can intermittently hang on Ascend
environments. The application-layer mitigation is to let the runtime own the
first Torch/NPU bootstrap before any user module top-level code executes.

## User-Visible Behavior

- `run-test` and benchmark runtime loading should initialize `torch`/`torch_npu`
  before executing user test/bench/operator modules when those dependencies are
  installed.
- Missing `torch` remains a fatal error because these run-eval entrypoints
  cannot execute benchmark/test workloads without PyTorch at all.
- Missing `torch_npu` is best-effort: bootstrap should still restore the
  environment and allow non-NPU local unit tests to import these helpers.
- Dynamic user-module loading should register the temporary module in
  `sys.modules` while executing it, matching the more standard import recipe and
  reducing re-entrant initialization surprises.

## Design

Add a small bootstrap helper function directly inside both runtime scripts
(`test_runner.py` and `bench_runtime.py`) that:

1. Checks whether `torch` is already loaded with an `npu` attribute.
2. Temporarily sets `TORCH_DEVICE_BACKEND_AUTOLOAD=0`.
3. Imports `torch` and lets `ImportError` propagate as a fatal runtime
   dependency failure.
4. Best-effort imports `torch_npu`.
5. Restores the prior environment value.

Call the script-local helper from:

- `test_runner.load_differential_test_cases()`
- `test_runner._run_import_only_standalone_test()`
- `bench_runtime.load_bench_cases()`

Keep the logic duplicated on purpose and document that decision inline. These
scripts are staged and executed in isolation in several local/remote flows, so
keeping the bootstrap code self-contained avoids having benchmark/test support
file lists and dynamic script loaders depend on one more sibling helper module.

`test_runner._load_module()` should also mirror the existing `bench_runtime`
pattern by inserting the temporary module into `sys.modules` for the duration of
`exec_module()`.

## Testing

- Add unit tests proving the bootstrap helper imports `torch` before user module
  execution in both test and benchmark runtimes.
- Add a unit test proving `test_runner._load_module()` temporarily registers the
  executing module in `sys.modules`.
