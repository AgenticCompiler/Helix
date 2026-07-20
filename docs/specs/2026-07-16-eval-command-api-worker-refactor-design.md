# Eval Command API/Worker Refactor Design

## Summary

Refactor the run-eval skill's benchmark, profiling, and probe workflows into
clear three-layer execution paths:

```text
Helix facade API -> local/remote API -> fixed worker
```

The refactor preserves the public `run-bench`, `profile-bench`, and
`probe-bench` contracts while removing the misleading aggregate runner module
names and remote `python -c` source construction.

## Current Problems

- The former runner modules combined public interfaces, local worker protocol,
  remote staging, execution modes, and generated remote programs.
- `run_probe_execution.py` owns useful cache and classification logic but imports
  concrete benchmark implementation functions instead of a stable benchmark
  API.
- The skill CLI contains run-bench baseline generation, comparison, timing,
  and result rendering alongside unrelated command dispatch.

## Target Boundaries

| Layer | Benchmark | Profile | Probe |
| --- | --- | --- | --- |
| Helix facade | `run_bench_api.py` | `run_profile_api.py` | `run_probe_api.py` |
| Local/remote APIs | `run_bench_local_api.py`, `run_bench_remote_api.py` | `run_profile_local_api.py`, `run_profile_remote_api.py` | `run_probe_local_api.py`, `run_probe_remote_api.py` |
| Fixed workers | `run_bench_local_worker.py`, `run_bench_remote_worker.py` | `run_profile_local_worker.py`, `run_profile_remote_worker.py` | uses the benchmark API; no separate probe worker |
| Implementations | `run_bench_execution.py` owns benchmark case execution; `run_bench_modes.py` owns mode-specific orchestration | `run_profile_execution.py` owns profile validation/execution | `run_probe_execution.py` owns cache, comparison, and classification |

`run_bench_api.py` exposes only local and remote benchmark signatures and
metadata parsing. `run_profile_api.py`
exposes the profile signatures to the skill CLI and `helix.eval.runners`.
`run_probe_api.py` exposes the existing probe signatures to Helix.

Local APIs own process launch, result-file recovery, timeout propagation, and
output filtering. Remote APIs own workspace lifecycle, explicit runtime
transfer, worker invocation, artifact recovery, and cleanup. Workers own
argument parsing and user/operator execution. Workers must be versioned files,
not generated source strings.

`remote_python_bundle.py` is a feature-neutral staging helper. It resolves the
static import closure of a fixed worker with AST parsing, then stages it through
a caller-provided copy callback. Remote APIs retain ownership of their
workspace, user input, sidecar, artifact recovery, and cleanup. Execution
modules do not maintain remote support-file lists.

Probe APIs bind probe's warmup/repeats policy to the internal neutral
`run_*_bench_with_limits(...)` transports. `run_probe_execution.py` receives a
measurement callback and does not import benchmark APIs or choose local versus
remote transport. Probe has no dedicated worker and reuses the benchmark worker.

## Compatibility

- Preserve all command names, flags, defaults, output ordering, return codes,
  perf JSONL locations, probe cache layout, and retained-workspace behavior.
- Preserve all three benchmark modes, serial and `--npu-devices` parallel
  execution, probe caps, sidecar staging, `--output`, baseline auto-comparison,
  and optimize timing events.
- Preserve profile `--case-id` validation, hidden `--kernel-name` handling,
  `PROF_*` validation/copying, profile report text, and profile timeout.
- `profile-report`, `compare-perf`, and `simulator_runner.py` stay outside this
  worker refactor.
- Delete only the legacy public runner modules (`bench_runner.py`,
  `profile_runner.py`, and `probe_runner.py`) after all loaders, Helix wrappers,
  and tests use the explicit APIs. The `*_execution.py` modules are internal
  implementations, not compatibility facades.

## Migration And Verification

1. Move benchmark execution behind the benchmark facade and fixed workers;
   validate every mode and remote staging before moving profile or probe.
2. Move profile execution behind its facade and fixed workers; add Helix
   bridge wrappers without adding a public top-level profile command.
3. Move probe cache/comparison implementation behind its facade and bind its
   measurement callback in the actual local/remote probe APIs.
4. After each stage, run focused unit/direct-worker tests and strict pyright
   for every changed skill script. Finish with ruff, pyright, full pytest, and
   diff checks.
5. Validate on `R154_cdj` with a unique `/tmp` root. For each workflow run the
   real skill CLI locally on the server and from the controller with `--remote`.
   Record CLI status, artifacts, and retained worker files; only status-file
   evidence counts if an SSH stream disconnects.

## Rollback

Each migration is reversible by restoring the immediately previous API-to-
implementation delegation. No public protocol or artifact format changes are
introduced, so the implementation structure can be rolled back independently
from user-visible behavior.
