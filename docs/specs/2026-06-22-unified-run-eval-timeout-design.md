# Unified Run-Eval Timeout

## Summary

Use `TRITON_AGENT_EVAL_TIMEOUT_SECONDS` as the single stall-timeout environment variable for both `run-test` and `run-bench`, and change its default from `900` seconds to `300` seconds. Make local `run-test` execute through an internal subprocess path so the timeout can actually interrupt stalled local test runs instead of applying only to remote test runs.

## Problem

- Local `run-test` currently executes test code inline inside the parent Python process.
- Inline local execution has no child-process stdout or stderr stream to watch, so stall detection is structurally impossible in that path today.
- `run-bench` and remote `run-test` already enforce stall detection through subprocess-based helpers, but they use duplicated timeout helpers and separate environment variables.
- `skills/triton-npu-run-eval/scripts/run_runtime.py` currently treats `0` as an immediate timeout on the first idle poll because its buffered and streaming runners do not guard timeout checks with `> 0`.
- The current user-facing timeout story is harder to explain than it needs to be: `run-test` and `run-bench` look like one family of eval commands but expose separate timeout knobs.

## Goals

- Give `run-test` and `run-bench` one shared timeout knob.
- Make the default eval stall timeout `300` seconds.
- Ensure local and remote `run-test` both enforce the configured timeout.
- Keep timeout behavior consistent when the shared variable is set to `0` by treating that as “stall timeout disabled” across eval runners.
- Preserve existing non-negative validation, so negative timeout values remain invalid.

## Non-Goals

- Do not change SSH or SCP timeout controls.
- Do not change profile-specific timeout controls in this change.
- Do not add new CLI flags for test or benchmark timeout configuration.
- Do not redesign benchmark or test execution semantics beyond the timeout-enforcement path.
- Do not consolidate `src/triton_agent/process_runner.py` and `skills/triton-npu-run-eval/scripts/run_runtime.py` into one shared implementation; skill scripts must remain self-contained and must not import `triton_agent`.

## Design

- `TRITON_AGENT_EVAL_TIMEOUT_SECONDS` becomes the only documented timeout variable for `run-test` and `run-bench`.
- The default for `TRITON_AGENT_EVAL_TIMEOUT_SECONDS` becomes `300`.
- `TRITON_AGENT_TEST_TIMEOUT_SECONDS` and `TRITON_AGENT_BENCH_TIMEOUT_SECONDS` stop affecting runtime behavior and are removed from CLI help and documentation.
- The timeout is an idle timeout, not a wall-clock cap. A `300`-second default means “terminate the eval subprocess after 300 seconds without observable output,” not “terminate after 300 seconds total runtime.”
- The shorter `300`-second default is deliberate. The request for this work is to make ordinary eval hangs surface sooner with one shared knob for `run-test` and `run-bench`; users whose remote environments legitimately need longer silent compile or execution gaps can raise `TRITON_AGENT_EVAL_TIMEOUT_SECONDS` explicitly.
- Profile execution remains on `TRITON_AGENT_PROFILE_TIMEOUT_SECONDS` with its current default. That exclusion is deliberate to keep this change scoped to ordinary `run-test` and `run-bench` flows, while avoiding an undocumented behavior change for profiling workloads that can have different setup and artifact-collection silence windows.
- Eval runtime helpers in `run_runtime.py` should mirror the timeout-disable semantics already used in [process_runner.py](/Users/cdj/Projects/triton-agent/src/triton_agent/process_runner.py:188): `0` disables stall termination, while negative values remain invalid because `env_int()` already rejects them.

### Worker Contract

- Local `run-test` should launch a dedicated worker subprocess by executing the same `test_runner.py` file with an internal subcommand, for example:
  - `local_python_executable() test_runner.py local-test-worker --test-file <abs> --operator-file <abs> --test-mode <mode> --result-file <abs> [--verbose]`
- The parent process remains responsible for calling `maybe_print_visible_devices()` before launching the worker so the existing debug-device message still appears in the user-visible parent output path.
- The worker process is responsible for all existing test bootstrap behavior before running user code, including:
  - `_bootstrap_torch_npu()`
  - `_temporary_sys_path_entries(...)`
  - the existing standalone and differential execution helpers
- The parent must not rely on stdout or stderr as the structured control channel, because:
  - worker stderr may contain verbose progress messages
  - worker stdout and stderr may include user test output captured inside the worker-side `ResultPayload`
  - verbose local runs should keep using a streaming subprocess path instead of regressing to fully buffered output
- The parent therefore passes a `--result-file` path to the worker. After execution, the worker writes one structured JSON payload to that file with:
  - `result`: serialized `ResultPayload`
  - `archived_result`: absolute path string or `null`
- For differential mode, the worker computes `_differential_archive_path(operator_file)` locally, runs the existing differential execution helper, and writes `archived_result` only when the returned result succeeded and the archive file exists. This gives the parent a local equivalent of the current remote archive-success contract.
- The parent launches the worker through:
  - `run_streaming_process(...)` when `verbose=True`
  - `run_buffered_process(...)` when `verbose=False`
- After the worker exits successfully, the parent reads `--result-file`, reconstructs the `ResultPayload`, and returns the decoded archive path (or `None`) through the existing `tuple[ResultPayload, Path | None]` API.

### Runtime And Documentation Changes

- In `skills/triton-npu-run-eval/scripts/run_runtime.py`:
  - Keep one shared `eval_stall_timeout_seconds()` helper that reads `TRITON_AGENT_EVAL_TIMEOUT_SECONDS`.
  - Change its default from `900` to `300`.
  - Update buffered and streaming process runners so they only trigger stall termination when `stall_timeout_seconds > 0`.
- In `skills/triton-npu-run-eval/scripts/bench_runner.py`:
  - Remove the bench-specific timeout helper.
  - Reuse the shared eval timeout helper for all local and remote bench subprocess launches.
- In `skills/triton-npu-run-eval/scripts/test_runner.py`:
  - Remove the test-specific timeout helper.
  - Add a private subprocess entrypoint for local test execution that accepts the resolved test mode, test path, operator path, result-file path, and verbosity setting.
  - Keep the existing in-process standalone and differential test implementations as the worker-side execution logic so test semantics stay the same.
  - Have the parent-side `run_local_test()` launch that worker through the existing runtime subprocess helper and decode the structured result payload from `--result-file`.
- In `src/triton_agent/cli.py` and `README.md`:
  - Keep only the shared eval timeout variable in user-facing timeout documentation for `run-test` and `run-bench`.
  - Update the documented default from `900` to `300`.

## Verification

- Update `tests/test_test_runner.py` to cover:
  - local worker invocation shape
  - parent-side decoding of the worker result file
  - archived-result propagation for successful differential runs
  - parent-side `maybe_print_visible_devices()` behavior
- Update `tests/test_bench_runner.py` assertions that currently expect `stall_timeout_seconds == 900` so they instead validate the unified helper and the new `300` default.
- Update `tests/test_cli.py` help-text assertions so they no longer expect `TRITON_AGENT_TEST_TIMEOUT_SECONDS` or `TRITON_AGENT_BENCH_TIMEOUT_SECONDS`.
- Add or update runtime-helper tests so `TRITON_AGENT_EVAL_TIMEOUT_SECONDS=0` disables stall termination, while negative values remain invalid.
- Run:
  - `uv run python -m unittest tests.test_test_runner tests.test_bench_runner tests.test_cli -v`
