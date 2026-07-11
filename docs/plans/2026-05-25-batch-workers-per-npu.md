# Batch Workers Per NPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `HELIX_BATCH_WORKERS_PER_NPU` so every batch command that already supports `HELIX_BATCH_NPU_DEVICES` can allow more than one concurrent workspace per configured NPU while ignoring the new variable when device affinity is disabled.

**Architecture:** Keep the behavior in `src/helix/npu_affinity.py` and treat the new variable as a slot-expansion layer on top of the existing device lease pool. Batch callers should switch from “device tuple” semantics to “expanded slot tuple” semantics without changing the downstream `ASCEND_RT_VISIBLE_DEVICES=<device>` process contract.

**Tech Stack:** Python 3.11, `unittest`, `argparse`, `concurrent.futures`, existing batch orchestration modules, Markdown docs

---

## File Map

- Modify: `src/helix/npu_affinity.py`
- Modify: `src/helix/optimize/batch.py`
- Modify: `src/helix/generation/batch.py`
- Modify: `src/helix/convert/batch.py`
- Modify: `src/helix/cli.py`
- Modify: `README.md`
- Modify: `tests/test_npu_affinity.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_generation_batch.py`
- Modify: `tests/test_convert_commands.py`
- Modify: `tests/test_cli.py`

### Task 1: Add Shared Workers-Per-NPU Parsing And Slot Expansion

**Files:**
- Modify: `src/helix/npu_affinity.py`
- Modify: `tests/test_npu_affinity.py`

- [ ] **Step 1: Write failing unit tests for the new affinity helpers**

```python
import os
import unittest
from unittest.mock import patch

from helix.npu_affinity import (
    BatchNpuAffinityPool,
    affinity_env_for_device,
    configured_batch_npu_slots,
    parse_batch_npu_devices,
    parse_batch_workers_per_npu,
)


class BatchNpuAffinityTests(unittest.TestCase):
    def test_parse_batch_workers_per_npu_defaults_to_one_when_unset(self) -> None:
        self.assertEqual(parse_batch_workers_per_npu(None), 1)

    def test_parse_batch_workers_per_npu_rejects_zero(self) -> None:
        with self.assertRaisesRegex(ValueError, "HELIX_BATCH_WORKERS_PER_NPU"):
            parse_batch_workers_per_npu("0")

    def test_parse_batch_workers_per_npu_rejects_non_integer(self) -> None:
        with self.assertRaisesRegex(ValueError, "HELIX_BATCH_WORKERS_PER_NPU"):
            parse_batch_workers_per_npu("two")

    def test_configured_batch_npu_slots_ignores_workers_when_devices_unset(self) -> None:
        with patch.dict(os.environ, {"HELIX_BATCH_WORKERS_PER_NPU": "3"}, clear=True):
            self.assertIsNone(configured_batch_npu_slots())

    def test_configured_batch_npu_slots_repeats_each_device_by_worker_count(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HELIX_BATCH_NPU_DEVICES": "0,1",
                "HELIX_BATCH_WORKERS_PER_NPU": "2",
            },
            clear=True,
        ):
            self.assertEqual(configured_batch_npu_slots(), ("0", "0", "1", "1"))
```

- [ ] **Step 2: Run the focused affinity tests and confirm they fail first**

Run: `uv run python -m unittest tests.test_npu_affinity -v`

Expected:
- `ImportError` or `AttributeError` for `parse_batch_workers_per_npu` / `configured_batch_npu_slots`
- existing tests still discover `helix.npu_affinity`

- [ ] **Step 3: Implement the new shared helpers in `src/helix/npu_affinity.py`**

```python
_BATCH_NPU_DEVICES_ENV = "HELIX_BATCH_NPU_DEVICES"
_BATCH_WORKERS_PER_NPU_ENV = "HELIX_BATCH_WORKERS_PER_NPU"


def parse_batch_workers_per_npu(raw: str | None) -> int:
    if raw is None:
        return 1
    text = raw.strip()
    if not text:
        raise ValueError(f"{_BATCH_WORKERS_PER_NPU_ENV} must be a positive integer.")
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"{_BATCH_WORKERS_PER_NPU_ENV} must be a positive integer.") from exc
    if value < 1:
        raise ValueError(f"{_BATCH_WORKERS_PER_NPU_ENV} must be at least 1.")
    return value


def configured_batch_workers_per_npu() -> int:
    return parse_batch_workers_per_npu(os.environ.get(_BATCH_WORKERS_PER_NPU_ENV))


def configured_batch_npu_slots() -> tuple[str, ...] | None:
    devices = configured_batch_npu_devices()
    if devices is None:
        return None
    workers_per_npu = configured_batch_workers_per_npu()
    return tuple(device for device in devices for _ in range(workers_per_npu))


def validate_batch_affinity_capacity(
    slots: tuple[str, ...] | None,
    *,
    max_concurrency: int,
) -> None:
    if slots is None:
        return
    if max_concurrency > len(slots):
        raise ValueError(
            "--max-concurrency must not exceed the batch affinity capacity configured by "
            "HELIX_BATCH_NPU_DEVICES and HELIX_BATCH_WORKERS_PER_NPU."
        )
```

- [ ] **Step 4: Re-run the focused affinity tests until they pass**

Run: `uv run python -m unittest tests.test_npu_affinity -v`

Expected:
- `OK`
- new tests pass alongside the existing range/duplicate/pool tests

- [ ] **Step 5: Commit the shared affinity helper task**

```bash
git add src/helix/npu_affinity.py tests/test_npu_affinity.py
git commit -m "feat: add batch workers per npu affinity helpers"
```

### Task 2: Teach Batch Commands To Use Expanded Slot Pools

**Files:**
- Modify: `src/helix/optimize/batch.py`
- Modify: `src/helix/generation/batch.py`
- Modify: `src/helix/convert/batch.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_generation_batch.py`
- Modify: `tests/test_convert_commands.py`

- [ ] **Step 1: Add failing batch tests that require shared assignments when workers-per-NPU is greater than one**

```python
def test_run_optimize_batch_allows_shared_affinity_slots_per_device(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("alpha", "beta", "gamma", "delta"):
            workspace = root / name
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

        seen_devices: list[str] = []

        def fake_run_request(request, stdout=None, stderr=None):
            del stdout, stderr
            seen_devices.append((request.extra_env or {})["ASCEND_RT_VISIBLE_DEVICES"])
            return AgentResult(return_code=0, stdout="ok", stderr="")

        with patch.dict(
            os.environ,
            {
                "HELIX_BATCH_NPU_DEVICES": "0,1",
                "HELIX_BATCH_WORKERS_PER_NPU": "2",
            },
            clear=False,
        ):
            with patch("helix.optimize.batch.render_batch_optimize_results", return_value=0):
                exit_code = run_optimize_batch(root, options, max_concurrency=4, stdout=StringIO(), run_request=fake_run_request)

        self.assertEqual(exit_code, 0)
        self.assertEqual(Counter(seen_devices), Counter({"0": 2, "1": 2}))
```

```python
def test_run_gen_eval_batch_allows_shared_affinity_slots_per_device(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("alpha", "beta", "gamma", "delta"):
            workspace = root / name
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

        seen_devices: list[str] = []

        def fake_run(request, stdout=None, stderr=None):
            del stdout, stderr
            seen_devices.append((request.extra_env or {})["ASCEND_RT_VISIBLE_DEVICES"])
            return AgentResult(return_code=0, stdout="ok", stderr="")

        with patch.dict(
            environ,
            {
                "HELIX_BATCH_NPU_DEVICES": "0,1",
                "HELIX_BATCH_WORKERS_PER_NPU": "2",
            },
            clear=False,
        ):
            exit_code = run_gen_eval_batch(root, options, max_concurrency=4, run_request=fake_run)

        self.assertEqual(exit_code, 0)
    self.assertEqual(Counter(seen_devices), Counter({"0": 2, "1": 2}))


def test_run_convert_batch_allows_shared_affinity_slots_per_device(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("alpha", "beta", "gamma", "delta"):
            workspace = root / name
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

        seen_devices: list[str] = []

        def fake_run(request, stdout=None, stderr=None):
            del stdout, stderr
            seen_devices.append((request.extra_env or {})["ASCEND_RT_VISIBLE_DEVICES"])
            return AgentResult(return_code=0, stdout="ok", stderr="")

        with patch.dict(
            environ,
            {
                "HELIX_BATCH_NPU_DEVICES": "0,1",
                "HELIX_BATCH_WORKERS_PER_NPU": "2",
            },
            clear=False,
        ):
            exit_code = run_convert_batch(root, options, max_concurrency=4, run_request=fake_run)

        self.assertEqual(exit_code, 0)
    self.assertEqual(Counter(seen_devices), Counter({"0": 2, "1": 2}))
```

- [ ] **Step 2: Add a failing oversubscription test that uses effective slot capacity instead of raw device count**

```python
def test_run_optimize_batch_rejects_concurrency_larger_than_effective_affinity_capacity(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        workspace = root / "alpha"
        workspace.mkdir()
        (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

        with patch.dict(
            os.environ,
            {
                "HELIX_BATCH_NPU_DEVICES": "0,1",
                "HELIX_BATCH_WORKERS_PER_NPU": "2",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "HELIX_BATCH_WORKERS_PER_NPU"):
                run_optimize_batch(root, options, max_concurrency=5, stdout=StringIO())
```

- [ ] **Step 3: Run the targeted batch tests and verify the new cases fail before implementation**

Run: `uv run python -m unittest tests.test_optimize_runtime tests.test_generation_batch tests.test_convert_commands -v`

Expected:
- new shared-slot tests fail because batch modules still cap capacity at raw device count
- existing one-device-per-workspace tests continue to pass

- [ ] **Step 4: Update all three batch modules to use `configured_batch_npu_slots()`**

```python
from helix.npu_affinity import (
    BatchNpuAffinityPool,
    affinity_env_for_device,
    configured_batch_npu_slots,
    validate_batch_affinity_capacity,
)

slots = configured_batch_npu_slots()
validate_batch_affinity_capacity(slots, max_concurrency=max_concurrency)
pool = BatchNpuAffinityPool(slots) if slots is not None else None
```

```python
with pool.acquire() as device:
    request.extra_env = affinity_env_for_device(device)
    return optimize_request_runner(request, forwarded_stdout, forwarded_stderr)
```

- [ ] **Step 5: Re-run the targeted batch tests until they pass**

Run: `uv run python -m unittest tests.test_optimize_runtime tests.test_generation_batch tests.test_convert_commands -v`

Expected:
- `OK`
- shared-slot tests pass with repeated device assignments
- legacy tests with only `HELIX_BATCH_NPU_DEVICES` still pass unchanged

- [ ] **Step 6: Commit the batch runtime task**

```bash
git add \
  src/helix/optimize/batch.py \
  src/helix/generation/batch.py \
  src/helix/convert/batch.py \
  tests/test_optimize_runtime.py \
  tests/test_generation_batch.py \
  tests/test_convert_commands.py
git commit -m "feat: support shared batch workers per npu"
```

### Task 3: Surface The New Environment Variable In CLI Help And README

**Files:**
- Modify: `src/helix/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Add a failing CLI help test for the new environment variable**

```python
def test_top_level_help_lists_supported_environment_variables(self) -> None:
    parser = build_parser()
    help_text = parser.format_help()
    self.assertIn("HELIX_BATCH_NPU_DEVICES", help_text)
    self.assertIn("HELIX_BATCH_WORKERS_PER_NPU", help_text)
```

- [ ] **Step 2: Run the focused CLI help test and confirm it fails first**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_top_level_help_lists_supported_environment_variables -v`

Expected:
- FAIL because `HELIX_BATCH_WORKERS_PER_NPU` is not yet listed

- [ ] **Step 3: Update `src/helix/cli.py` and `README.md` with the new capacity semantics**

```python
_TOP_LEVEL_ENVIRONMENT_VARIABLE_GROUPS = (
    (
        "Batch and runtime",
        (
            (
                "HELIX_BATCH_NPU_DEVICES",
                "Comma-separated Ascend NPU device pool for batch workspaces.",
            ),
            (
                "HELIX_BATCH_WORKERS_PER_NPU",
                "Concurrent workspace slots contributed by each configured batch NPU (default: 1).",
            ),
            (
                "HELIX_CODE_AGENT_MAX_RETRIES",
                "Retry limit for transient code-agent failures.",
            ),
        ),
    ),
)
```

```md
| `HELIX_BATCH_WORKERS_PER_NPU` | No | `gen-eval-batch`, `convert-batch`, `optimize-batch` | Positive integer capacity multiplier per configured NPU. Ignored unless `HELIX_BATCH_NPU_DEVICES` is set. |
```

```md
export HELIX_BATCH_NPU_DEVICES=0,1
export HELIX_BATCH_WORKERS_PER_NPU=2
uv run helix optimize-batch --input operators_root --max-concurrency 4
```

- [ ] **Step 4: Re-run the focused CLI test**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_top_level_help_lists_supported_environment_variables -v`

Expected:
- `OK`

- [ ] **Step 5: Review README wording for consistency**

Check:
- the environment-variable table uses the exact variable name
- the batch affinity section explains `device_count * workers_per_npu`
- the docs explicitly say the new variable is ignored when `HELIX_BATCH_NPU_DEVICES` is unset

- [ ] **Step 6: Commit the docs/help task**

```bash
git add src/helix/cli.py tests/test_cli.py README.md
git commit -m "docs: describe batch workers per npu"
```

### Task 4: Final Verification

**Files:**
- Modify: none

- [ ] **Step 1: Run the focused tests for the touched modules**

Run: `uv run python -m unittest tests.test_npu_affinity tests.test_optimize_runtime tests.test_generation_batch tests.test_convert_commands tests.test_cli -v`

Expected:
- `OK`

- [ ] **Step 2: Run repository lint**

Run: `uv run --group dev ruff check`

Expected:
- exit code `0`

- [ ] **Step 3: Run repository type checking**

Run: `uv run pyright`

Expected:
- exit code `0`

- [ ] **Step 4: Run the full unittest suite**

Run: `uv run python -m unittest discover -s tests -v`

Expected:
- all tests pass

- [ ] **Step 5: Summarize the verification evidence in the final change note**

Include:
- the exact test commands that passed
- confirmation that one-device-per-workspace behavior remains unchanged when `HELIX_BATCH_WORKERS_PER_NPU` is unset
- confirmation that `HELIX_BATCH_WORKERS_PER_NPU` is ignored when `HELIX_BATCH_NPU_DEVICES` is unset
