# Batch NPU Affinity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in batch-only NPU affinity feature that assigns one Ascend device per concurrent workspace and propagates that assignment through agent launches, local run-eval subprocesses, and remote SSH-launched runtime commands.

**Architecture:** Keep device selection in one shared CLI/runtime module instead of pushing it into prompts or individual skills. Batch commands acquire per-workspace device leases, attach `extra_env` to the request, and rely on shared subprocess helpers to merge those env overrides into local and remote execution paths. The feature remains completely dormant unless `HELIX_BATCH_NPU_DEVICES` is set.

**Tech Stack:** Python 3.11, `argparse`, `unittest`, `concurrent.futures`, repository batch helpers, skill runtime helper scripts, Markdown docs

---

## File Map

- Create: `src/helix/npu_affinity.py`
- Create: `tests/test_npu_affinity.py`
- Modify: `src/helix/models.py`
- Modify: `src/helix/backends/base.py`
- Modify: `src/helix/process_runner.py`
- Modify: `src/helix/optimize/batch.py`
- Modify: `src/helix/generation/batch.py`
- Modify: `src/helix/convert/batch.py`
- Modify: `skills/triton-npu-run-eval/scripts/run_runtime.py`
- Modify: `tests/test_backends_base.py`
- Modify: `tests/test_process_runner.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_generation_batch.py`
- Modify: `tests/test_convert_commands.py`
- Modify: `tests/test_remote_execution.py`
- Modify: `tests/test_bench_runner.py`
- Modify: `tests/test_profile_runner.py`
- Modify: `README.md`

### Task 1: Lock The Affinity Contract With Failing Unit Tests

**Files:**
- Create: `tests/test_npu_affinity.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_generation_batch.py`
- Modify: `tests/test_convert_commands.py`

- [ ] **Step 1: Add focused parser and lease-pool tests in `tests/test_npu_affinity.py`**

```python
import unittest

from helix.npu_affinity import (
    BatchNpuAffinityPool,
    affinity_env_for_device,
    parse_batch_npu_devices,
)


class BatchNpuAffinityTests(unittest.TestCase):
    def test_parse_batch_npu_devices_returns_none_when_unset(self) -> None:
        self.assertIsNone(parse_batch_npu_devices(None))

    def test_parse_batch_npu_devices_trims_whitespace(self) -> None:
        self.assertEqual(parse_batch_npu_devices(" 0, 1 ,2 "), ("0", "1", "2"))

    def test_parse_batch_npu_devices_rejects_empty_entries(self) -> None:
        with self.assertRaisesRegex(ValueError, "HELIX_BATCH_NPU_DEVICES"):
            parse_batch_npu_devices("0,,1")

    def test_parse_batch_npu_devices_rejects_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_batch_npu_devices("0,1,0")

    def test_affinity_env_for_device_uses_visible_devices_and_diagnostic_env(self) -> None:
        self.assertEqual(
            affinity_env_for_device("3"),
            {
                "ASCEND_RT_VISIBLE_DEVICES": "3",
            },
        )

    def test_pool_reuses_released_devices(self) -> None:
        pool = BatchNpuAffinityPool(("0", "1"))
        with pool.acquire() as first:
            self.assertEqual(first, "0")
        with pool.acquire() as second:
            self.assertEqual(second, "1")
        with pool.acquire() as third:
            self.assertEqual(third, "0")
```

- [ ] **Step 2: Add failing optimize-batch coverage that requires per-workspace `extra_env`**

```python
def test_run_optimize_batch_assigns_distinct_affinity_env_per_workspace(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("alpha", "beta"):
            workspace = root / name
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

        options = OptimizeRunOptions(
            agent_name="codex",
            interact=False,
            verbose=False,
            show_output=False,
            remote=None,
            remote_workdir=None,
            min_rounds=None,
            resume_mode="auto",
            reset_optimize=False,
            no_agent_session=False,
            supervise="off",
            output=None,
            test_mode=None,
            bench_mode=None,
            prompt=None,
        )
        seen_envs: list[dict[str, str]] = []

        def fake_run_request(request, stdout=None, stderr=None):
            del stdout, stderr
            seen_envs.append(request.extra_env or {})
            return AgentResult(return_code=0, stdout="ok", stderr="")

        with patch.dict(os.environ, {"HELIX_BATCH_NPU_DEVICES": "0,1"}, clear=False):
            with patch("helix.optimize.batch.render_batch_optimize_results", return_value=0):
                exit_code = run_optimize_batch(root, options, max_concurrency=2, stdout=StringIO(), run_request=fake_run_request)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            {env["ASCEND_RT_VISIBLE_DEVICES"] for env in seen_envs},
            {"0", "1"},
        )
```

- [ ] **Step 3: Add failing gen-eval-batch and convert-batch coverage that expects the same `extra_env` propagation**

```python
def test_run_gen_eval_batch_assigns_affinity_env_per_workspace(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("alpha", "beta"):
            workspace = root / name
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

        options = GenerationOptions(
            interact=False,
            verbose=False,
            show_output=False,
            force_overwrite=False,
            agent_name="codex",
            remote=None,
            remote_workdir=None,
            min_rounds=None,
            continue_optimize=False,
            output=None,
            test_mode="differential",
            bench_mode="standalone",
            prompt=None,
        )
        seen_envs: list[dict[str, str]] = []

        def fake_run(request, stdout=None, stderr=None):
            del stdout, stderr
            seen_envs.append(request.extra_env or {})
            return AgentResult(return_code=0, stdout="ok", stderr="")

        with patch.dict(os.environ, {"HELIX_BATCH_NPU_DEVICES": "0,1"}, clear=False):
            exit_code = run_gen_eval_batch(root, options, max_concurrency=2, run_request=fake_run)
    self.assertEqual({env["ASCEND_RT_VISIBLE_DEVICES"] for env in seen_envs}, {"0", "1"})


def test_run_convert_batch_assigns_affinity_env_per_workspace(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("alpha", "beta"):
            workspace = root / name
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

        options = ConvertOptions(
            interact=False,
            verbose=False,
            show_output=False,
            force_overwrite=False,
            agent_name="codex",
            remote=None,
            remote_workdir=None,
            min_rounds=None,
            continue_optimize=False,
            output=None,
            test_mode="differential",
            bench_mode=None,
            prompt=None,
        )
        seen_envs: list[dict[str, str]] = []

        def fake_run(request, stdout=None, stderr=None):
            del stdout, stderr
            seen_envs.append(request.extra_env or {})
            return AgentResult(return_code=0, stdout="ok", stderr="")

        with patch.dict(os.environ, {"HELIX_BATCH_NPU_DEVICES": "0,1"}, clear=False):
            exit_code = run_convert_batch(root, options, max_concurrency=2, run_request=fake_run)
    self.assertEqual({env["ASCEND_RT_VISIBLE_DEVICES"] for env in seen_envs}, {"0", "1"})
```

- [ ] **Step 4: Add failing validation tests that reject affinity-enabled oversubscription**

```python
def test_run_optimize_batch_rejects_concurrency_larger_than_affinity_pool(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        workspace = root / "alpha"
        workspace.mkdir()
        (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")
        options = OptimizeRunOptions(
            agent_name="codex",
            interact=False,
            verbose=False,
            show_output=False,
            remote=None,
            remote_workdir=None,
            min_rounds=None,
            resume_mode="auto",
            reset_optimize=False,
            no_agent_session=False,
            supervise="off",
            output=None,
            test_mode=None,
            bench_mode=None,
            prompt=None,
        )

        with patch.dict(os.environ, {"HELIX_BATCH_NPU_DEVICES": "0"}, clear=False):
            with self.assertRaisesRegex(ValueError, "--max-concurrency"):
                run_optimize_batch(root, options, max_concurrency=2, stdout=StringIO())
```

- [ ] **Step 5: Run the new focused tests to verify they fail before implementation**

Run: `uv run python -m unittest tests.test_npu_affinity tests.test_generation_batch tests.test_convert_commands tests.test_optimize_runtime`

Expected:
- `ModuleNotFoundError` or import failure for `helix.npu_affinity`
- batch tests fail because `AgentRequest` has no `extra_env`
- concurrency validation test fails because affinity-aware validation does not exist yet

### Task 2: Add The Shared Affinity Module And Request Contract

**Files:**
- Create: `src/helix/npu_affinity.py`
- Modify: `src/helix/models.py`
- Test: `tests/test_npu_affinity.py`

- [ ] **Step 1: Extend `AgentRequest` with a per-request env override field**

```python
@dataclass
class AgentRequest:
    command_kind: CommandKind
    input_path: Path
    operator_path: Optional[Path]
    output_path: Optional[Path]
    test_mode: Optional[str]
    bench_mode: Optional[str]
    interact: bool
    verbose: bool
    show_output: bool
    force_overwrite: bool
    agent_name: str
    skill_name: str
    prompt: str
    workdir: Path
    extra_env: dict[str, str] | None = None
    min_rounds: Optional[int] = None
```

- [ ] **Step 2: Create `src/helix/npu_affinity.py` with parsing and lease helpers**

```python
from __future__ import annotations

import os
import queue
from collections.abc import Iterator
from contextlib import contextmanager


_BATCH_NPU_DEVICES_ENV = "HELIX_BATCH_NPU_DEVICES"


def parse_batch_npu_devices(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    devices = tuple(part.strip() for part in raw.split(","))
    if not devices or any(not part for part in devices):
        raise ValueError(f"{_BATCH_NPU_DEVICES_ENV} must be a comma-separated non-empty device list.")
    if len(set(devices)) != len(devices):
        raise ValueError(f"{_BATCH_NPU_DEVICES_ENV} must not contain duplicate devices: {raw!r}")
    return devices


def configured_batch_npu_devices() -> tuple[str, ...] | None:
    return parse_batch_npu_devices(os.environ.get(_BATCH_NPU_DEVICES_ENV))


def affinity_env_for_device(device: str) -> dict[str, str]:
    return {"ASCEND_RT_VISIBLE_DEVICES": device}


class BatchNpuAffinityPool:
    def __init__(self, devices: tuple[str, ...]) -> None:
        self._queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        for device in devices:
            self._queue.put(device)

    @contextmanager
    def acquire(self) -> Iterator[str]:
        device = self._queue.get()
        try:
            yield device
        finally:
            self._queue.put(device)
```

- [ ] **Step 3: Add a small validation helper for affinity-enabled concurrency**

```python
def validate_batch_affinity_capacity(
    devices: tuple[str, ...] | None,
    *,
    max_concurrency: int,
) -> None:
    if devices is None:
        return
    if max_concurrency > len(devices):
        raise ValueError(
            "--max-concurrency must not exceed the number of devices configured by "
            "HELIX_BATCH_NPU_DEVICES."
        )
```

- [ ] **Step 4: Run the new unit tests until the affinity module contract turns GREEN**

Run: `uv run python -m unittest tests.test_npu_affinity`

Expected: PASS

- [ ] **Step 5: Commit the affinity scaffolding**

```bash
git add src/helix/models.py src/helix/npu_affinity.py tests/test_npu_affinity.py
git commit -m "feat: add batch npu affinity primitives"
```

### Task 3: Teach Backend Launches And Generic Process Runners To Accept Env Overrides

**Files:**
- Modify: `src/helix/backends/base.py`
- Modify: `src/helix/process_runner.py`
- Modify: `tests/test_backends_base.py`
- Modify: `tests/test_process_runner.py`

- [ ] **Step 1: Add failing backend coverage that expects `extra_env` to reach `run_process`**

```python
def test_base_runner_passes_request_extra_env_to_process_runner(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        runner = _DummyRunner()
        request = AgentRequest(
            command_kind=CommandKind.GEN_TEST,
            input_path=workspace / "op.py",
            operator_path=workspace / "op.py",
            output_path=workspace / "test_op.py",
            test_mode=None,
            bench_mode=None,
            interact=False,
            verbose=False,
            show_output=False,
            force_overwrite=False,
            agent_name="dummy",
            skill_name="triton-npu-gen-test",
            prompt="Prompt body",
            workdir=workspace,
            extra_env={"ASCEND_RT_VISIBLE_DEVICES": "2"},
        )

        with patch("helix.backends.base.run_process", return_value=_ok_result()) as mocked:
            runner.run(request)

    self.assertEqual(mocked.call_args.kwargs["extra_env"], {"ASCEND_RT_VISIBLE_DEVICES": "2"})
```

- [ ] **Step 2: Add failing process-runner coverage that expects merged env to reach `subprocess.Popen`**

```python
def test_buffered_process_runner_merges_extra_env(self) -> None:
    process = _BufferedFakeProcess(stdout_lines=[], stderr_text="", returncode=0)
    with patch("helix.process_runner.subprocess.Popen", return_value=process) as mocked:
        run_buffered_process(
            ["codex", "exec"],
            "/tmp",
            stall_timeout_seconds=10,
            session_id_extractor=lambda _line: None,
            extra_env={"ASCEND_RT_VISIBLE_DEVICES": "7"},
        )
    self.assertEqual(mocked.call_args.kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"], "7")
```

- [ ] **Step 3: Implement env merging in `src/helix/process_runner.py`**

```python
import os


def _merged_env(extra_env: dict[str, str] | None) -> dict[str, str] | None:
    if extra_env is None:
        return None
    merged = dict(os.environ)
    merged.update(extra_env)
    return merged
```

```python
def run_process(
    command: list[str],
    workdir: str,
    mode: str,
    stall_timeout_seconds: int = 0,
    session_id_extractor: Optional[Callable[[str], Optional[str]]] = None,
    stdout: Optional[TextIO] = None,
    output_filter: Optional[OutputFilter] = None,
    interrupt_policy: Optional[InterruptPolicy] = None,
    extra_env: Optional[dict[str, str]] = None,
) -> AgentResult:
    if mode == "interactive":
        return run_interactive_process(command, workdir, extra_env=extra_env)
    if mode == "streaming":
        return run_streaming_process(
            command,
            workdir,
            stall_timeout_seconds=stall_timeout_seconds,
            stdout=stdout,
            output_filter=output_filter,
            session_id_extractor=session_id_extractor or (lambda _text: None),
            interrupt_policy=interrupt_policy,
            extra_env=extra_env,
        )
    if mode == "buffered":
        return run_buffered_process(
            command,
            workdir,
            stall_timeout_seconds=stall_timeout_seconds,
            session_id_extractor=session_id_extractor or (lambda _line: None),
            output_filter=output_filter,
            interrupt_policy=interrupt_policy,
            extra_env=extra_env,
        )
    raise ValueError(f"Unsupported process runner mode: {mode}")
```

```python
process = subprocess.Popen(
    _resolve_command(command),
    cwd=workdir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=_merged_env(extra_env),
    start_new_session=interrupt_policy is not None and not _IS_WINDOWS,
)
```

- [ ] **Step 4: Pass `request.extra_env` from `AgentRunner` into `run_process`**

```python
return run_process(
    command,
    str(request.workdir),
    mode=self._select_mode(request),
    stall_timeout_seconds=self.stall_timeout_seconds,
    session_id_extractor=self.session_id_extractor(),
    stdout=stdout,
    output_filter=self.output_filter(request),
    interrupt_policy=self.interrupt_policy(request),
    extra_env=request.extra_env,
)
```

- [ ] **Step 5: Run focused tests for backend and process propagation**

Run: `uv run python -m unittest tests.test_backends_base tests.test_process_runner`

Expected: PASS

- [ ] **Step 6: Commit the subprocess env propagation layer**

```bash
git add src/helix/backends/base.py src/helix/process_runner.py tests/test_backends_base.py tests/test_process_runner.py
git commit -m "feat: propagate per-request env overrides"
```

### Task 4: Integrate Device Leases Into Batch Commands

**Files:**
- Modify: `src/helix/optimize/batch.py`
- Modify: `src/helix/generation/batch.py`
- Modify: `src/helix/convert/batch.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_generation_batch.py`
- Modify: `tests/test_convert_commands.py`

- [ ] **Step 1: Add a small batch-local helper pattern in `src/helix/optimize/batch.py`**

```python
from helix.npu_affinity import (
    BatchNpuAffinityPool,
    affinity_env_for_device,
    configured_batch_npu_devices,
    validate_batch_affinity_capacity,
)
```

```python
devices = configured_batch_npu_devices()
validate_batch_affinity_capacity(devices, max_concurrency=max_concurrency)
pool = BatchNpuAffinityPool(devices) if devices is not None else None
```

- [ ] **Step 2: Wrap each workspace execution in a lease and attach `extra_env`**

```python
def _run_workspace(item: BatchOptimizeWorkspace) -> AgentResult:
    request = build_optimize_request(item.operator_file, item.workspace, options)
    if pool is None:
        return optimize_request_runner(request)
    with pool.acquire() as device:
        request.extra_env = affinity_env_for_device(device)
        return optimize_request_runner(request)
```

Use the same pattern for the `show_output` branch, passing `stdout` and `stderr` through to the runner inside the lease scope instead of submitting the original runner directly.

- [ ] **Step 3: Mirror the same lease attachment in `src/helix/generation/batch.py` and `src/helix/convert/batch.py`**

```python
with pool.acquire() as device:
    request.extra_env = affinity_env_for_device(device)
    return generation_request_runner(request, forwarded_stream, forwarded_stream)
```

```python
with pool.acquire() as device:
    request.extra_env = affinity_env_for_device(device)
    return convert_request_runner(request, forwarded_stream, forwarded_stream)
```

- [ ] **Step 4: Run batch-focused tests including the new affinity assertions**

Run: `uv run python -m unittest tests.test_optimize_runtime tests.test_generation_batch tests.test_convert_commands`

Expected: PASS

- [ ] **Step 5: Commit the batch integration**

```bash
git add src/helix/optimize/batch.py src/helix/generation/batch.py src/helix/convert/batch.py tests/test_optimize_runtime.py tests/test_generation_batch.py tests/test_convert_commands.py
git commit -m "feat: assign npu affinity to batch workspaces"
```

### Task 5: Propagate Env Overrides Through Run-Eval Local And Remote Runtime Helpers

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/run_runtime.py`
- Modify: `tests/test_remote_execution.py`
- Modify: `tests/test_bench_runner.py`
- Modify: `tests/test_profile_runner.py`

- [ ] **Step 1: Add failing tests that expect local helper subprocesses to receive explicit env overrides**

```python
def test_run_runtime_buffered_process_merges_extra_env(self) -> None:
    module = load_operator_eval_script_module("run_runtime")
    fake_process = _BufferedFakeProcess(stdout_lines=[], stderr_text="", returncode=0)
    with patch.object(module.subprocess, "Popen", return_value=fake_process) as mocked:
        module.run_buffered_process(
            ["python3", "bench.py"],
            ".",
            stall_timeout_seconds=10,
            extra_env={"ASCEND_RT_VISIBLE_DEVICES": "4"},
        )
    self.assertEqual(mocked.call_args.kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"], "4")
```

- [ ] **Step 2: Add failing tests that expect remote SSH command strings to be prefixed with affinity env**

```python
def test_run_remote_command_streaming_prefixes_env_assignments(self) -> None:
    module = load_operator_eval_script_module("run_runtime")
    with patch.object(module, "run_streaming_process", return_value=make_skill_result(0, "", "")) as mocked:
        module.run_remote_command_streaming(
            {"user_host": "alice@example.com", "port": None},
            "/tmp/workspace",
            ["python3", "bench.py"],
            extra_env={"ASCEND_RT_VISIBLE_DEVICES": "4"},
        )
    ssh_command = mocked.call_args.args[0]
    self.assertIn("ASCEND_RT_VISIBLE_DEVICES=4", ssh_command[-1])
```

- [ ] **Step 3: Implement `extra_env` support in `skills/triton-npu-run-eval/scripts/run_runtime.py`**

```python
def _merged_env(extra_env: dict[str, str] | None) -> dict[str, str] | None:
    if extra_env is None:
        return None
    merged = dict(os.environ)
    merged.update(extra_env)
    return merged


def _shell_env_prefix(extra_env: dict[str, str] | None) -> str:
    if not extra_env:
        return ""
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(extra_env.items()))
```

```python
process = subprocess.Popen(
    command,
    cwd=workdir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=_merged_env(extra_env),
)
```

```python
prefix = _shell_env_prefix(extra_env)
body = _normalize_remote_command(remote_command)
command = _ssh_command(
    spec,
    f"cd {shlex.quote(remote_workspace)} && {prefix + ' ' if prefix else ''}{body}",
)
```

- [ ] **Step 4: Thread the new optional `extra_env` argument through the public runtime helpers without changing current call sites**

```python
def run_remote_command_streaming(
    spec: RemoteSpec,
    remote_workspace: str,
    remote_command: str | Sequence[str],
    verbose: bool = False,
    stderr: TextIO | None = None,
    extra_env: dict[str, str] | None = None,
) -> ResultPayload:
    prefix = _shell_env_prefix(extra_env)
    body = _normalize_remote_command(remote_command)
    command = _ssh_command(
        spec,
        f"cd {shlex.quote(remote_workspace)} && {prefix + ' ' if prefix else ''}{body}",
    )
    _maybe_emit_remote_command(command, verbose, stderr)
    return run_streaming_process(command, ".", stall_timeout_seconds=900)


def run_remote_command_buffered(
    spec: RemoteSpec,
    remote_workspace: str,
    remote_command: str | Sequence[str],
    verbose: bool = False,
    stderr: TextIO | None = None,
    extra_env: dict[str, str] | None = None,
) -> ResultPayload:
    prefix = _shell_env_prefix(extra_env)
    body = _normalize_remote_command(remote_command)
    command = _ssh_command(
        spec,
        f"cd {shlex.quote(remote_workspace)} && {prefix + ' ' if prefix else ''}{body}",
    )
    _maybe_emit_remote_command(command, verbose, stderr)
    return run_buffered_process(command, ".", stall_timeout_seconds=900)
```

Keep default `None` so existing callers stay source-compatible.

- [ ] **Step 5: Run the runtime-helper tests**

Run: `uv run python -m unittest tests.test_remote_execution tests.test_bench_runner tests.test_profile_runner`

Expected: PASS

- [ ] **Step 6: Run the required file-scoped strict pyright check for the touched skill script**

Run:

```bash
bash -lc 'tmpdir=$(mktemp -d); printf "[tool.pyright]\npythonVersion = \"3.11\"\ninclude = [\"%s\"]\ntypeCheckingMode = \"strict\"\n" "$PWD/skills/triton-npu-run-eval/scripts/run_runtime.py" > "$tmpdir/pyproject.toml"; uv run pyright --project "$tmpdir/pyproject.toml"'
```

Expected: PASS

- [ ] **Step 7: Commit the runtime-helper propagation**

```bash
git add skills/triton-npu-run-eval/scripts/run_runtime.py tests/test_remote_execution.py tests/test_bench_runner.py tests/test_profile_runner.py
git commit -m "feat: propagate affinity env through runtime helpers"
```

### Task 6: Document The Feature And Run Full Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update batch command docs to describe the opt-in env var and device-count constraint**

```md
### Batch NPU Affinity

Set `HELIX_BATCH_NPU_DEVICES` to a comma-separated device list to pin concurrent batch workspaces to distinct Ascend NPUs:

```bash
export HELIX_BATCH_NPU_DEVICES=0,1,2,3
uv run helix optimize-batch --input operators_root --max-concurrency 4
```

When this variable is set, `--max-concurrency` must not exceed the number of configured devices.
```

- [ ] **Step 2: Run the focused CLI and runtime regression suites**

Run: `uv run python -m unittest tests.test_cli tests.test_optimize_runtime tests.test_generation_batch tests.test_convert_commands tests.test_backends_base tests.test_process_runner tests.test_remote_execution tests.test_bench_runner tests.test_profile_runner`

Expected: PASS

- [ ] **Step 3: Run repository lint and type checks**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

- [ ] **Step 4: Run the full unittest suite**

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS

- [ ] **Step 5: Commit the docs and final verification state**

```bash
git add README.md
git commit -m "docs: describe batch npu affinity"
```
