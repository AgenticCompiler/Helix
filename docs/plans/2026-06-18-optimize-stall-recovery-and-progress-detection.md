# Optimize Stall Recovery And Progress Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let multi-invocation optimize recover from stalled or transiently failed worker launches by continuing from the first unresolved round, while improving stall detection with optimize artifact file activity.

**Architecture:** Keep generic subprocess and backend layers small: add a request-level backend-retry opt-out and a generic progress-source hook at the process-runner boundary, then implement optimize-specific recovery semantics, file-activity rules, and per-round recovery budgets inside a new feature-local optimize recovery module plus the existing optimize execution controller. Preserve baseline and supervisor behavior in the first version, and keep accepted progress anchored to `check_round(...)` over the current worker target range.

**Tech Stack:** Python, unittest, existing `process_runner` / backend runner stack, optimize execution controller, optimize round check helpers

---

### Task 1: Lock Shared Retry Opt-Out And Process-Runner Progress Hooks With Tests

**Files:**
- Modify: `/Users/cdj/Projects/helix/tests/test_backends_base.py`
- Modify: `/Users/cdj/Projects/helix/tests/test_process_runner.py`
- Modify: `/Users/cdj/Projects/helix/src/helix/models.py`
- Modify: `/Users/cdj/Projects/helix/src/helix/backends/base.py`
- Modify: `/Users/cdj/Projects/helix/src/helix/process_runner.py`
- Test: `/Users/cdj/Projects/helix/tests/test_backends_base.py`
- Test: `/Users/cdj/Projects/helix/tests/test_process_runner.py`

- [ ] **Step 1: Write a failing backend test for per-request retry opt-out**

Add a test in `tests/test_backends_base.py` that constructs an `AgentRequest` with a new request-level field:

```python
request = AgentRequest(
    command_kind=CommandKind.OPTIMIZE,
    input_path=workspace / "op.py",
    operator_path=workspace / "op.py",
    output_path=workspace / "opt_op.py",
    test_mode=None,
    bench_mode=None,
    interact=False,
    verbose=False,
    stream_output=False,
    force_overwrite=False,
    agent_name="dummy",
    skill_name="triton-npu-optimize",
    prompt="Prompt body",
    workdir=workspace,
    disable_backend_retry=True,
)
```

and assert that one transient failure is returned immediately:

```python
self.assertEqual(result.return_code, 1)
self.assertEqual(mocked_run.call_count, 1)
mocked_sleep.assert_not_called()
```

- [ ] **Step 2: Write failing process-runner tests for external progress activity**

Add tests in `tests/test_process_runner.py` that exercise a new `progress_probe` hook:

```python
probe_values = iter([None, 0.5, 0.5])

def progress_probe() -> float | None:
    return next(probe_values, 0.5)
```

Cover at least:

```python
self.assertFalse(result.stalled)
```

when no output arrives but the probe reports forward progress, and:

```python
self.assertTrue(result.stalled)
```

when the probe never reports progress.

- [ ] **Step 3: Run the targeted tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_backends_base tests.test_process_runner -v
```

Expected: FAIL because `AgentRequest` does not yet accept `disable_backend_retry`, and `run_process()` / `run_buffered_process()` / `run_streaming_process()` do not yet accept an external progress probe.

- [ ] **Step 4: Add the minimal request-model fields**

Update `src/helix/models.py` to add explicit fields that match the approved design:

```python
@dataclass
class ProgressSourceConfig:
    kind: Literal["optimize_artifacts"]


@dataclass
class AgentRequest:
    ...
    disable_backend_retry: bool = False
    progress_source: ProgressSourceConfig | None = None
```

Keep the new config object simple and declarative. Do not store callbacks on the model.

- [ ] **Step 5: Gate shared retry on the request opt-out**

Change `src/helix/backends/base.py` so the shared retry loop checks the request:

```python
while (
    not request.disable_backend_retry
    and _is_transient_agent_failure(result)
    and attempt < max_retries
):
    attempt += 1
    time.sleep(_retry_delay_seconds(attempt))
    result = self._run_once(
        command,
        request,
        stdout=stdout,
        rendered_chunk_sink=rendered_chunk_sink,
        collect_stdout=collect_stdout,
    )
```

Do not change `_is_transient_agent_failure()` semantics in this task.

- [ ] **Step 6: Thread a generic progress probe through the process runner**

Update `src/helix/process_runner.py` function signatures so they accept an optional probe:

```python
ProgressProbe = Callable[[], float | None]

def run_process(..., progress_probe: ProgressProbe | None = None, ...) -> AgentResult:
    ...

def run_buffered_process(..., progress_probe: ProgressProbe | None = None, ...) -> AgentResult:
    ...

def run_streaming_process(..., progress_probe: ProgressProbe | None = None, ...) -> AgentResult:
    ...
```

and refresh the stall timer inline from the polling loop:

```python
if progress_probe is not None:
    probe_time = progress_probe()
    if probe_time is not None and probe_time > start_ref[0]:
        start_ref[0] = probe_time
```

Apply the same inline pattern to buffered, PTY streaming, and Windows streaming loops. Do not add a background thread.

- [ ] **Step 7: Run the targeted tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_backends_base tests.test_process_runner -v
```

Expected: PASS

### Task 2: Add Optimize Recovery Helpers And File-Activity Rules With Focused Tests

**Files:**
- Create: `/Users/cdj/Projects/helix/src/helix/optimize/recovery.py`
- Modify: `/Users/cdj/Projects/helix/tests/test_optimize_runtime.py`
- Modify: `/Users/cdj/Projects/helix/src/helix/optimize/execution.py`
- Test: `/Users/cdj/Projects/helix/tests/test_optimize_runtime.py`

- [ ] **Step 1: Write failing tests for optimize recovery classification and progress rules**

Add new unit coverage in `tests/test_optimize_runtime.py` for helper-level behavior you expect from `optimize/recovery.py`.

Use assertions shaped like:

```python
self.assertEqual(classify_worker_failure(AgentResult(return_code=1, stdout="", stderr="", stalled=True)), "stall")
self.assertEqual(
    classify_worker_failure(
        AgentResult(return_code=1, stdout="", stderr="ERROR: exceeded retry limit, last status: 429 Too Many Requests")
    ),
    "transient",
)
self.assertEqual(classify_worker_failure(AgentResult(return_code=1, stdout="", stderr="boom")), "fatal")
```

and for whitelist behavior:

```python
self.assertTrue(is_optimize_progress_path(workspace / "opt-round-3" / "round-state.json", workspace))
self.assertTrue(is_optimize_progress_path(workspace / "opt-round-3" / "summary.md", workspace))
self.assertFalse(is_optimize_progress_path(workspace / "helix-logs" / "optimize.show-output.log", workspace))
self.assertFalse(is_optimize_progress_path(workspace / ".helix" / "supervisor-report.md", workspace))
```

- [ ] **Step 2: Write a failing test for directory mtime-only changes**

Add a focused helper test that proves only directory metadata updates do not count:

```python
dir_path = workspace / "opt-round-3"
dir_path.mkdir()
snapshot = scan_optimize_progress(workspace)
os.utime(dir_path, None)
self.assertEqual(scan_optimize_progress(workspace), snapshot)
```

This test should be written around whichever snapshot helper you introduce, but it must prove directory `mtime` alone does not advance progress.

- [ ] **Step 3: Run the targeted optimize runtime tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_optimize_runtime -v
```

Expected: FAIL because `src/helix/optimize/recovery.py` does not exist yet and the helper behavior is not implemented.

- [ ] **Step 4: Create the optimize recovery helper module**

Add `src/helix/optimize/recovery.py` with small, focused helpers such as:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from helix.models import AgentResult
from helix.transient_failures import contains_transient_agent_failure_text


WorkerFailureKind = Literal["stall", "transient", "fatal"]


def classify_worker_failure(result: AgentResult) -> WorkerFailureKind:
    if result.stalled:
        return "stall"
    combined = f"{result.stdout}\n{result.stderr}".lower()
    if result.retryable_failure or contains_transient_agent_failure_text(combined):
        return "transient"
    return "fatal"
```

Keep the precedence exactly as approved in the spec.

- [ ] **Step 5: Implement explicit optimize progress path filtering**

In `src/helix/optimize/recovery.py`, add a narrow allow-list rooted at the workspace:

```python
def is_optimize_progress_path(path: Path, workspace: Path) -> bool:
    relative = path.relative_to(workspace)
    parts = relative.parts
    if relative == Path("opt-note.md") or relative == Path("learned_lessons.md"):
        return True
    if parts and parts[0] == "baseline":
        return path.is_file()
    if parts and re.fullmatch(r"opt-round-\d+", parts[0]):
        return path.is_file()
    return False
```

Do not special-case extensions. Use the allow-list roots and `path.is_file()` so directory-only changes do not count as progress.

- [ ] **Step 6: Implement snapshot-based file progress scanning**

Still in `src/helix/optimize/recovery.py`, add a lightweight snapshot helper:

```python
@dataclass(frozen=True)
class ProgressSnapshot:
    latest_mtime: float | None
    file_fingerprints: tuple[tuple[str, int, float], ...]


def scan_optimize_progress(workspace: Path) -> ProgressSnapshot:
    entries: list[tuple[str, int, float]] = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        if not is_optimize_progress_path(path, workspace):
            continue
        stat = path.stat()
        rel = path.relative_to(workspace).as_posix()
        entries.append((rel, stat.st_size, stat.st_mtime))
    latest = max((item[2] for item in entries), default=None)
    return ProgressSnapshot(latest_mtime=latest, file_fingerprints=tuple(entries))
```

This gives later tasks a stable building block for the probe without depending on loose directory metadata.

- [ ] **Step 7: Run the targeted optimize runtime tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_optimize_runtime -v
```

Expected: PASS for the new helper-focused recovery and progress-rule tests.

### Task 3: Teach The Backend Runner To Build Progress Probes From Request Config

**Files:**
- Modify: `/Users/cdj/Projects/helix/src/helix/backends/base.py`
- Modify: `/Users/cdj/Projects/helix/src/helix/optimize/recovery.py`
- Modify: `/Users/cdj/Projects/helix/tests/test_backends_base.py`
- Test: `/Users/cdj/Projects/helix/tests/test_backends_base.py`

- [ ] **Step 1: Write a failing backend test for request-config-driven progress probes**

Add a test in `tests/test_backends_base.py` that patches a new helper used by the backend runner and asserts it is called when `request.progress_source` is configured:

```python
request = AgentRequest(
    ...,
    progress_source=ProgressSourceConfig(kind="optimize_artifacts"),
)
```

and:

```python
self.assertIsNotNone(mocked_run.call_args.kwargs["progress_probe"])
```

while requests without `progress_source` still pass `None`.

- [ ] **Step 2: Run the targeted backend tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_backends_base -v
```

Expected: FAIL because the backend runner does not yet translate request config into a real probe callable.

- [ ] **Step 3: Add a probe-construction helper**

In `src/helix/optimize/recovery.py`, add a probe builder that closes over workspace snapshot state:

```python
def build_optimize_progress_probe(workspace: Path) -> Callable[[], float | None]:
    state = {"snapshot": scan_optimize_progress(workspace)}

    def probe() -> float | None:
        snapshot = scan_optimize_progress(workspace)
        if snapshot != state["snapshot"]:
            state["snapshot"] = snapshot
            return snapshot.latest_mtime
        return None

    return probe
```

Keep it stateless from the caller's perspective and safe for repeated inline polling.

- [ ] **Step 4: Wire request config to a concrete probe in the backend runner**

Update `src/helix/backends/base.py` so `_run_once()` resolves a probe from request config before calling `run_process()`:

```python
progress_probe = None
if request.progress_source is not None and request.progress_source.kind == "optimize_artifacts":
    from helix.optimize.recovery import build_optimize_progress_probe

    progress_probe = build_optimize_progress_probe(request.workdir)

result = run_process(
    command,
    str(request.workdir),
    mode=self._select_mode(request),
    stall_timeout_seconds=self.stall_timeout_seconds,
    session_id_extractor=self.session_id_extractor(),
    stdout=stdout,
    output_filter=self.output_filter(request),
    interrupt_policy=self.interrupt_policy(request),
    extra_env=request.extra_env,
    rendered_chunk_sink=rendered_chunk_sink,
    collect_stdout=collect_stdout,
    progress_probe=progress_probe,
)
```

Keep this translation layer tiny and generic. Do not move optimize file scanning into `backends/base.py`.

- [ ] **Step 5: Run the targeted backend tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_backends_base -v
```

Expected: PASS

### Task 4: Add Longest-Passing-Prefix And Per-Round Recovery Budget Helpers

**Files:**
- Modify: `/Users/cdj/Projects/helix/src/helix/optimize/recovery.py`
- Modify: `/Users/cdj/Projects/helix/tests/test_optimize_runtime.py`
- Test: `/Users/cdj/Projects/helix/tests/test_optimize_runtime.py`

- [ ] **Step 1: Write failing tests for range-local accepted-progress scanning**

Add focused tests in `tests/test_optimize_runtime.py` that create round directories and patch `check_round()` so the helper stops at the first failure in the current target range:

```python
self.assertEqual(
    compute_range_progress(request, batch_start=11, batch_end=15),
    RangeProgress(
        last_accepted_round=13,
        first_unresolved_round=14,
        next_batch_start=14,
        next_batch_end=15,
    ),
)
```

Also cover that the helper starts at `batch_start`, not round `1`.

- [ ] **Step 2: Write failing tests for per-round recovery budget reset**

Add tests shaped like:

```python
budget = RecoveryBudget(max_attempts=3)
budget.consume(14)
budget.consume(14)
self.assertEqual(budget.remaining(14), 1)
budget.consume(15)
self.assertEqual(budget.remaining(15), 2)
```

and:

```python
budget = RecoveryBudget(max_attempts=3)
budget.consume(14)
budget.consume(14)
budget.consume(14)
self.assertTrue(budget.exhausted(14))
```

- [ ] **Step 3: Run the targeted optimize runtime tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_optimize_runtime -v
```

Expected: FAIL because range-local accepted-progress helpers and recovery-budget helpers do not yet exist.

- [ ] **Step 4: Implement the range-progress helper**

In `src/helix/optimize/recovery.py`, add a small result model and helper:

```python
@dataclass(frozen=True)
class RangeProgress:
    last_accepted_round: int
    first_unresolved_round: int
    next_batch_start: int
    next_batch_end: int


def compute_range_progress(
    workdir: Path,
    *,
    batch_start: int,
    batch_end: int,
    optimize_target: str,
) -> RangeProgress:
    last_accepted = batch_start - 1
    for round_number in range(batch_start, batch_end + 1):
        round_dir = workdir / f"opt-round-{round_number}"
        if not round_dir.is_dir():
            break
        result = check_round(
            round_dir,
            current_round=round_number,
            final_round=batch_end,
            optimize_target=optimize_target,
        )
        if result.status != "pass":
            break
        last_accepted = round_number
    first_unresolved = last_accepted + 1
    return RangeProgress(
        last_accepted_round=last_accepted,
        first_unresolved_round=first_unresolved,
        next_batch_start=first_unresolved,
        next_batch_end=batch_end,
    )
```

- [ ] **Step 5: Implement the per-round recovery budget helper**

Still in `src/helix/optimize/recovery.py`, add a tiny in-memory tracker:

```python
class RecoveryBudget:
    def __init__(self, max_attempts: int = 3) -> None:
        self._max_attempts = max_attempts
        self._attempts: dict[int, int] = {}

    def consume(self, round_number: int) -> None:
        self._attempts[round_number] = self._attempts.get(round_number, 0) + 1

    def remaining(self, round_number: int) -> int:
        return self._max_attempts - self._attempts.get(round_number, 0)

    def exhausted(self, round_number: int) -> bool:
        return self.remaining(round_number) <= 0
```

Keep this process-local only. Do not persist it.

- [ ] **Step 6: Run the targeted optimize runtime tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_optimize_runtime -v
```

Expected: PASS

### Task 5: Add Recovery Prompt Notes And Wrap Optimize Worker Launches In A Recovery Loop

**Files:**
- Modify: `/Users/cdj/Projects/helix/src/helix/optimize/recovery.py`
- Modify: `/Users/cdj/Projects/helix/src/helix/optimize/execution.py`
- Modify: `/Users/cdj/Projects/helix/tests/test_optimize_runtime.py`
- Test: `/Users/cdj/Projects/helix/tests/test_optimize_runtime.py`

- [ ] **Step 1: Write failing runtime tests for transient recovery and stall recovery**

Add targeted controller tests that patch the runner so the worker launch first fails and then succeeds.

For transient recovery, assert that the second invocation keeps the same range:

```python
self.assertEqual(requests[0].current_round, 11)
self.assertEqual(requests[1].current_round, 11)
self.assertIn("transient backend failure", requests[1].prompt)
```

For stall recovery, assert that the second invocation resumes from the unresolved round:

```python
self.assertEqual(requests[0].current_round, 11)
self.assertEqual(requests[1].current_round, 14)
self.assertEqual(requests[1].final_round, 15)
self.assertIn("previous invocation stalled", requests[1].prompt)
```

- [ ] **Step 2: Write a failing runtime test for immediate exit on non-recoverable launch failure**

Add a controller test where the runner returns:

```python
AgentResult(return_code=1, stdout="", stderr="plain error", stalled=False, retryable_failure=False)
```

and assert:

```python
self.assertEqual(mocked_run.call_count, 1)
self.assertEqual(result.return_code, 1)
```

- [ ] **Step 3: Write a failing runtime test for post-run batch failure staying outside recovery**

Add a test where the worker launch succeeds but `check_round()` fails, and assert the controller does not relaunch the worker through the recovery loop:

```python
self.assertEqual(mocked_run.call_count, 1)
self.assertIn("optimize batch check failed", result.stderr)
```

- [ ] **Step 4: Run the targeted optimize runtime tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_optimize_runtime -v
```

Expected: FAIL because the optimize controller still returns immediately on any unsuccessful worker launch.

- [ ] **Step 5: Add recovery-note helpers**

In `src/helix/optimize/recovery.py`, add small helpers:

```python
def build_transient_recovery_note(*, batch_start: int, batch_end: int) -> str:
    return (
        "CLI recovery note: the previous invocation ended in a transient backend failure.\n"
        f"Retry the current target range {batch_start} through {batch_end}."
    )


def build_stall_recovery_note(
    *,
    last_accepted_round: int,
    first_unresolved_round: int,
    batch_end: int,
) -> str:
    accepted_text = "none yet" if last_accepted_round < first_unresolved_round - 1 else str(last_accepted_round)
    return (
        "CLI recovery note: the previous invocation stalled.\n"
        f"Accepted progress reaches round {accepted_text}.\n"
        f"Resume from round {first_unresolved_round} through {batch_end}.\n"
        "Inspect existing artifacts for the unresolved round before deciding whether to repair or finish them."
    )
```

- [ ] **Step 6: Wrap worker launches in a recovery loop inside the optimize controller**

Refactor `MultiInvocationOptimizeController.run_round_loop()` and supporting helpers in `src/helix/optimize/execution.py` so worker launching goes through a dedicated method such as:

```python
def _run_worker_with_recovery(
    self,
    request: AgentRequest,
    *,
    batch_start: int,
    batch_end: int,
    previous_batch_issues: str | None,
) -> AgentResult:
    budget = RecoveryBudget(max_attempts=3)
    current_start = batch_start
    while True:
        worker_request = self._request_with_fresh_batch_prompt(
            request,
            issues=previous_batch_issues,
            batch_start=current_start,
            batch_end=batch_end,
        )
        result = self._run_request(worker_request, show_output_label=f"batch-{current_start}-{batch_end}")
        if result.succeeded:
            return result
        failure_kind = classify_worker_failure(result)
        if failure_kind == "fatal":
            return result
        progress = compute_range_progress(
            request.workdir,
            batch_start=current_start,
            batch_end=batch_end,
            optimize_target=request.optimize_target,
        ) if failure_kind == "stall" else None
        unresolved_round = progress.first_unresolved_round if progress is not None else current_start
        budget.consume(unresolved_round)
        if budget.exhausted(unresolved_round):
            return result
        current_start = unresolved_round
        previous_batch_issues = None
        request = replace(
            request,
            prompt=self._append_recovery_note(
                request,
                failure_kind=failure_kind,
                progress=progress,
                batch_start=current_start,
                batch_end=batch_end,
            ),
        )
```

Keep the recovery loop narrowly scoped to worker-launch failures. Do not let ordinary post-run batch validation failures re-enter it.

- [ ] **Step 7: Run the targeted optimize runtime tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_optimize_runtime -v
```

Expected: PASS

### Task 6: Wire Optimize Worker Requests To The New Request Fields

**Files:**
- Modify: `/Users/cdj/Projects/helix/src/helix/optimize/execution.py`
- Modify: `/Users/cdj/Projects/helix/tests/test_optimize_runtime.py`
- Test: `/Users/cdj/Projects/helix/tests/test_optimize_runtime.py`

- [ ] **Step 1: Write a failing runtime test that optimize worker requests disable backend retry**

Add a test that captures worker `AgentRequest` objects and asserts:

```python
self.assertTrue(worker_request.disable_backend_retry)
self.assertEqual(worker_request.progress_source, ProgressSourceConfig(kind="optimize_artifacts"))
```

for worker launches, while baseline repair or supervisor launches do not gain the same recovery-only semantics in this first version.

- [ ] **Step 2: Run the targeted optimize runtime tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_optimize_runtime -v
```

Expected: FAIL because optimize execution does not yet populate `disable_backend_retry` or `progress_source` on worker requests.

- [ ] **Step 3: Mark worker requests with the new recovery-only request fields**

In `src/helix/optimize/execution.py`, update worker-request construction so round worker invocations use:

```python
return replace(
    request,
    prompt=prompt,
    current_round=batch_start,
    final_round=batch_end,
    disable_backend_retry=True,
    progress_source=ProgressSourceConfig(kind="optimize_artifacts"),
)
```

Do not set these fields on baseline repair or supervisor requests in this version.

- [ ] **Step 4: Run the targeted optimize runtime tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_optimize_runtime -v
```

Expected: PASS

### Task 7: Run Focused Verification, Then Full Repository Verification

**Files:**
- Modify: `/Users/cdj/Projects/helix/docs/specs/2026-06-18-optimize-stall-recovery-and-progress-detection-design.md`
- Modify: `/Users/cdj/Projects/helix/docs/plans/2026-06-18-optimize-stall-recovery-and-progress-detection.md`
- Test: `/Users/cdj/Projects/helix/tests/test_backends_base.py`
- Test: `/Users/cdj/Projects/helix/tests/test_process_runner.py`
- Test: `/Users/cdj/Projects/helix/tests/test_optimize_runtime.py`

- [ ] **Step 1: Run focused unit tests for the changed runtime areas**

Run:

```bash
uv run python -m unittest tests.test_backends_base tests.test_process_runner tests.test_optimize_runtime -v
```

Expected: PASS

- [ ] **Step 2: Run lint**

Run:

```bash
uv run --group dev ruff check
```

Expected: PASS

- [ ] **Step 3: Run type checking**

Run:

```bash
uv run pyright
```

Expected: PASS

- [ ] **Step 4: Run the full repository test suite**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
```

Expected: PASS

- [ ] **Step 5: Update the design and plan docs if implementation naming differs**

If implementation uses slightly different finalized helper names than the examples in this plan or the approved spec, update:

```text
/Users/cdj/Projects/helix/docs/specs/2026-06-18-optimize-stall-recovery-and-progress-detection-design.md
/Users/cdj/Projects/helix/docs/plans/2026-06-18-optimize-stall-recovery-and-progress-detection.md
```

so the written docs reflect the shipped code precisely.

- [ ] **Step 6: Commit the implementation**

Run:

```bash
git add /Users/cdj/Projects/helix/src/helix/models.py \
        /Users/cdj/Projects/helix/src/helix/backends/base.py \
        /Users/cdj/Projects/helix/src/helix/process_runner.py \
        /Users/cdj/Projects/helix/src/helix/optimize/recovery.py \
        /Users/cdj/Projects/helix/src/helix/optimize/execution.py \
        /Users/cdj/Projects/helix/tests/test_backends_base.py \
        /Users/cdj/Projects/helix/tests/test_process_runner.py \
        /Users/cdj/Projects/helix/tests/test_optimize_runtime.py \
        /Users/cdj/Projects/helix/docs/specs/2026-06-18-optimize-stall-recovery-and-progress-detection-design.md \
        /Users/cdj/Projects/helix/docs/plans/2026-06-18-optimize-stall-recovery-and-progress-detection.md
git commit -m "feat: recover optimize workers from stalls"
```

Expected: a new commit containing the runtime changes, tests, and any documentation adjustments needed to match the final code.
