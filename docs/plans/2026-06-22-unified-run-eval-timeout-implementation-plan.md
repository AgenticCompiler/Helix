# Unified Run-Eval Timeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify `run-test` and `run-bench` on `HELIX_EVAL_TIMEOUT_SECONDS`, make the default idle timeout `300` seconds, and add real timeout enforcement for local `run-test`.

**Architecture:** Keep the skill-side runtime self-contained. Route local `run-test` through a worker subprocess launched by `test_runner.py` itself, persist the worker result into a result file, and reuse the existing eval subprocess helpers for timeout enforcement. Update the shared eval timeout helper, then remove the duplicate test/bench timeout knobs from runtime code, help text, and docs.

**Tech Stack:** Python 3.11, `argparse`, `json`, `tempfile`, existing triton-npu-run-eval skill scripts, `unittest`, `uv`

---

### Task 1: Lock the new timeout contract with failing tests

**Files:**
- Modify: `tests/test_test_runner.py`
- Modify: `tests/test_bench_runner.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_remote_execution.py`

- [ ] **Step 1: Add a failing local test-runner test that proves `run_local_test()` launches a worker subprocess and reads a result file**

```python
def test_run_local_test_launches_worker_subprocess_and_reads_result_file(self) -> None:
    module = load_test_runner_module()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test_file = root / "test_abs.py"
        operator = root / "abs.py"
        test_file.write_text("# test-mode: standalone\n", encoding="utf-8")
        operator.write_text("def abs_entry():\n    return 1\n", encoding="utf-8")

        result_file_holder: dict[str, Path] = {}

        def fake_buffered(command, workdir, stall_timeout_seconds, extra_env=None):
            self.assertEqual(stall_timeout_seconds, 300)
            result_file = Path(command[command.index("--result-file") + 1])
            result_file_holder["path"] = result_file
            result_file.write_text(
                json.dumps(
                    {
                        "result": {
                            "return_code": 0,
                            "stdout": "worker stdout\n",
                            "stderr": "",
                            "stalled": False,
                            "session_id": None,
                        },
                        "archived_result": None,
                    }
                ),
                encoding="utf-8",
            )
            return make_skill_result(0, "", "")

        with patch.object(module, "run_buffered_process", side_effect=fake_buffered):
            result, archived = module.run_local_test(test_file, operator, "standalone")

    self.assertEqual(result["stdout"], "worker stdout\n")
    self.assertIsNone(archived)
    self.assertTrue(result_file_holder["path"].exists())
```

- [ ] **Step 2: Add a failing local differential test that proves the worker can report the archived result path back to the parent**

```python
def test_run_local_test_reads_archived_result_path_from_worker_payload(self) -> None:
    module = load_test_runner_module()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        archived_path = root / "abs_result.pt"
        test_file = root / "differential_test_abs.py"
        operator = root / "abs.py"
        test_file.write_text("# test-mode: differential\n", encoding="utf-8")
        operator.write_text("def build_api():\n    return lambda value: value\n", encoding="utf-8")

        def fake_buffered(command, workdir, stall_timeout_seconds, extra_env=None):
            result_file = Path(command[command.index("--result-file") + 1])
            archived_path.write_bytes(b"pt")
            result_file.write_text(
                json.dumps(
                    {
                        "result": {
                            "return_code": 0,
                            "stdout": "",
                            "stderr": "",
                            "stalled": False,
                            "session_id": None,
                        },
                        "archived_result": str(archived_path),
                    }
                ),
                encoding="utf-8",
            )
            return make_skill_result(0, "", "")

        with patch.object(module, "run_buffered_process", side_effect=fake_buffered):
            result, archived = module.run_local_test(test_file, operator, "differential")

    self.assertEqual(result["return_code"], 0)
    self.assertEqual(archived, archived_path)
```

- [ ] **Step 3: Add a failing runtime-helper test that proves `HELIX_EVAL_TIMEOUT_SECONDS=0` disables stall termination**

```python
def test_run_runtime_buffered_zero_timeout_disables_stall_termination(self) -> None:
    module = load_operator_eval_script_module("run_runtime")

    class _FakeStdout:
        def __init__(self) -> None:
            self.calls = 0
        def readline(self) -> str:
            self.calls += 1
            return ""
        def close(self) -> None:
            pass

    class _FakeStderr:
        def read(self) -> str:
            return ""
        def close(self) -> None:
            pass

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdout = _FakeStdout()
            self.stderr = _FakeStderr()
            self.returncode = 0
        def poll(self):
            return 0 if self.stdout.calls > 1 else None

    with patch.object(module.subprocess, "Popen", return_value=_FakeProcess()):
        result = module.run_buffered_process(["python3", "bench.py"], ".", 0)

    self.assertFalse(result["stalled"])
    self.assertEqual(result["return_code"], 0)
```

- [ ] **Step 4: Add failing streaming-runner regression tests that prove `0` disables stall termination in both streaming code paths**

```python
def test_run_runtime_streaming_windows_zero_timeout_disables_stall_termination(self) -> None:
    module = load_operator_eval_script_module("run_runtime")
    with patch.object(module, "_IS_WINDOWS", True):
        ...
        result = module._run_streaming_windows(["python3", "bench.py"], ".", 0)
    self.assertFalse(result["stalled"])

def test_run_runtime_streaming_pty_zero_timeout_disables_stall_termination(self) -> None:
    module = load_operator_eval_script_module("run_runtime")
    ...
    result = module._run_streaming_pty(["python3", "bench.py"], ".", 0)
    self.assertFalse(result["stalled"])
```

- [ ] **Step 5: Add a failing negative-value regression test for the shared eval timeout variable**

```python
def test_eval_timeout_env_rejects_negative_values(self) -> None:
    module = load_operator_eval_script_module("run_runtime")
    with patch.dict(module.os.environ, {"HELIX_EVAL_TIMEOUT_SECONDS": "-1"}, clear=False):
        with self.assertRaises(ValueError):
            module.eval_stall_timeout_seconds()
```

- [ ] **Step 6: Update existing bench and CLI expectations to the new unified variable/default**

```python
self.assertEqual(stall_timeout_seconds, 300)
self.assertIn("HELIX_EVAL_TIMEOUT_SECONDS", help_text)
self.assertNotIn("HELIX_TEST_TIMEOUT_SECONDS", help_text)
self.assertNotIn("HELIX_BENCH_TIMEOUT_SECONDS", help_text)
```

- [ ] **Step 7: Run focused tests to confirm the new expectations fail before implementation**

Run: `uv run python -m unittest tests.test_test_runner tests.test_bench_runner tests.test_cli tests.test_remote_execution -v`
Expected: FAIL on missing local worker protocol, old `900` assertions, missing streaming `0` guards, and old timeout-help expectations.

### Task 2: Implement the shared eval timeout behavior in the skill runtime

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/run_runtime.py`
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`

- [ ] **Step 1: Change the shared eval timeout default to `300` and make `0` disable stall detection at all three stall-check sites**

```python
def eval_stall_timeout_seconds() -> int:
    return env_int("HELIX_EVAL_TIMEOUT_SECONDS", 300)

elif stall_timeout_seconds > 0 and time.monotonic() - start > stall_timeout_seconds:
    process.terminate()

if stall_timeout_seconds > 0 and elapsed > stall_timeout_seconds:
    process.terminate()
```

- [ ] **Step 2: Remove the bench-specific timeout helper and route all 12 bench subprocess launches through `eval_stall_timeout_seconds()`**

```python
from run_runtime import (
    ...,
    eval_stall_timeout_seconds,
    ...,
)

stall_timeout_seconds=eval_stall_timeout_seconds(),
```

- [ ] **Step 3: Run the focused bench/runtime tests**

Run: `uv run python -m unittest tests.test_bench_runner tests.test_remote_execution -v`
Expected: PASS for the new `300` default and `0` disables-timeout coverage.

### Task 3: Move local `run-test` onto a worker subprocess

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/test_runner.py`
- Modify: `tests/test_test_runner.py`

- [ ] **Step 1: Add a structured worker payload and helper functions for encoding/decoding it**

```python
def _write_local_test_worker_payload(
    result_file: Path,
    result: ResultPayload,
    archived_result: Path | None,
) -> None:
    result_file.write_text(
        json.dumps(
            {
                "result": {
                    "return_code": int(result["return_code"]),
                    "stdout": str(result["stdout"]),
                    "stderr": str(result["stderr"]),
                    "stalled": bool(result["stalled"]),
                    "session_id": result["session_id"],
                },
                "archived_result": None if archived_result is None else str(archived_result),
            }
        ),
        encoding="utf-8",
    )

def _read_local_test_worker_payload(result_file: Path) -> tuple[ResultPayload, Path | None]:
    ...

def _merge_failed_worker_result(result: ResultPayload) -> tuple[ResultPayload, None]:
    return result, None
```

- [ ] **Step 2: Import `eval_stall_timeout_seconds()`, remove `_test_timeout()`, and unify the remote `run-test` path on the shared helper**

```python
from run_runtime import (
    ...,
    eval_stall_timeout_seconds,
    ...,
)

stall_timeout_seconds=eval_stall_timeout_seconds(),
```

- [ ] **Step 3: Add the internal worker entrypoint and CLI parser inside `test_runner.py`**

```python
def _build_local_test_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    parser.add_argument("command", choices=["local-test-worker"])
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--operator-file", required=True)
    parser.add_argument("--test-mode", choices=["standalone", "differential"], required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--verbose", action="store_true")
    return parser
```

- [ ] **Step 4: Rework `run_local_test()` so the parent prints device debug info, launches the worker, and decodes `--result-file`**

```python
def run_local_test(...):
    maybe_print_visible_devices()
    with tempfile.TemporaryDirectory() as tmp:
        result_file = Path(tmp) / "local-test-result.json"
        command = [
            local_python_executable(),
            str(Path(__file__).resolve()),
            "local-test-worker",
            "--test-file",
            str(test_file),
            "--operator-file",
            str(operator_file),
            "--test-mode",
            test_mode,
            "--result-file",
            str(result_file),
        ]
        if verbose:
            result = run_streaming_process(command, str(test_file.parent), stall_timeout_seconds=eval_stall_timeout_seconds())
        else:
            result = run_buffered_process(command, str(test_file.parent), stall_timeout_seconds=eval_stall_timeout_seconds())
        if result_succeeded(result):
            return _read_local_test_worker_payload(result_file)
        return _merge_failed_worker_result(result)
```

- [ ] **Step 5: Implement the worker-side dispatch using the existing standalone and differential helpers**

```python
def _run_local_test_worker(...) -> int:
    if test_mode == "standalone":
        result = _run_import_only_standalone_test(test_file, operator_file, verbose=verbose)
        archived_result = None
    else:
        archive_path = _differential_archive_path(operator_file)
        result = _run_declarative_differential_test(test_file, operator_file, archive_path, verbose=verbose)
        archived_result = archive_path if result_succeeded(result) and archive_path.exists() else None
    _write_local_test_worker_payload(result_file, result, archived_result)
    return int(result["return_code"])
```

- [ ] **Step 6: Run focused local test-runner coverage**

Run: `uv run python -m unittest tests.test_test_runner -v`
Expected: PASS for worker invocation, payload decoding, archived-result propagation, and existing standalone/differential behavior.

### Task 4: Remove old timeout knobs from help and docs

**Files:**
- Modify: `src/helix/cli.py`
- Modify: `README.md`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Remove the old test/bench timeout help entries and keep the unified eval timeout text at `300`**

```python
(
    "HELIX_EVAL_TIMEOUT_SECONDS",
    "Stall timeout in seconds for run-test and run-bench execution (default: 300).",
),
```

- [ ] **Step 2: Add the shared timeout row to the existing README environment-variable table under the runtime variables section**

```md
| `HELIX_EVAL_TIMEOUT_SECONDS` | No | `run-test`, `run-bench` | Idle stall timeout in seconds for local and remote eval subprocess execution. Default: `300`. Set `0` to disable stall termination. |
```

- [ ] **Step 3: Run the CLI help and focused regression tests**

Run: `uv run python -m unittest tests.test_cli tests.test_skill_command_script -v`
Expected: PASS, with help text only mentioning the shared eval timeout knob.

### Task 5: Final verification

**Files:**
- Test: `tests/test_test_runner.py`
- Test: `tests/test_bench_runner.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_remote_execution.py`

- [ ] **Step 1: Run the full focused suite for the touched timeout paths**

Run: `uv run python -m unittest tests.test_test_runner tests.test_bench_runner tests.test_cli tests.test_remote_execution -v`
Expected: PASS

- [ ] **Step 2: Run the repository-standard targeted verification for the modified Python code if time permits**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_test_runner.py tests/test_bench_runner.py tests/test_cli.py tests/test_remote_execution.py`
Expected: PASS
