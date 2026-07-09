# Status Baseline State Perf Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `triton-agent status` trust `baseline/state.json` `perf_artifact`
before any legacy baseline perf scanning so canonical baseline state wins over
directory ambiguity.

**Architecture:** Keep the change local to `src/triton_agent/status/core.py`.
Write the regression tests first in `tests/test_status.py`, then add a small
baseline-state resolution helper that uses the shared baseline loader and
preserves the existing legacy fallback path only when state is missing or
unusable.

**Tech Stack:** Python 3.11, `pathlib`, `json`, `unittest`, `uv`

---

## File Map

- Modify: `tests/test_status.py`
  Add regression coverage for state-declared baseline perf selection.
- Modify: `src/triton_agent/status/core.py`
  Add the minimal baseline-state-first selection logic.
- Verify: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_status.py -k baseline`
  Prove the red-green cycle on the touched behavior.

### Task 1: Lock the new baseline precedence with failing tests

**Files:**
- Modify: `tests/test_status.py`
- Test: `tests/test_status.py`

- [ ] **Step 1: Add a regression test for multiple baseline perf files with a declared canonical path**

```python
def test_inspect_optimize_status_workspace_prefers_state_declared_baseline_perf_file(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
        baseline_dir = workspace / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "kernel_perf.txt").write_text("latency-a: 10\nlatency-b: 20\n", encoding="utf-8")
        (baseline_dir / "other_perf.txt").write_text("latency-a: 999\nlatency-b: 999\n", encoding="utf-8")
        (baseline_dir / "state.json").write_text(... perf_artifact points to "kernel_perf.txt" ..., encoding="utf-8")
        (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
        round_one = workspace / "opt-round-1"
        round_one.mkdir()
        (round_one / "opt_kernel_perf.txt").write_text("latency-a: 8\nlatency-b: 18\n", encoding="utf-8")
        self._write_round_state(round_one, perf_artifact="opt_kernel_perf.txt")

        status = inspect_optimize_status_workspace(workspace)

    self.assertEqual(status.state, "ok")
    self.assertEqual(status.best_round, "round-1")
    self.assertNotIn("found multiple baseline perf files", status.warnings)
```

- [ ] **Step 2: Add a regression test for a missing state-declared baseline perf path**

```python
def test_inspect_optimize_status_workspace_warns_when_state_declared_baseline_perf_is_missing(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
        baseline_dir = workspace / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "kernel_perf.txt").write_text("latency-a: 10\nlatency-b: 20\n", encoding="utf-8")
        (baseline_dir / "other_perf.txt").write_text("latency-a: 999\nlatency-b: 999\n", encoding="utf-8")
        (baseline_dir / "state.json").write_text(... perf_artifact points to "missing_perf.txt" ..., encoding="utf-8")
        (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
        (workspace / "opt-round-1").mkdir()

        status = inspect_optimize_status_workspace(workspace)

    self.assertEqual(status.state, "warning")
    self.assertIn("perf_artifact points to a missing file: missing_perf.txt", status.warnings)
    self.assertNotIn("found multiple baseline perf files", status.warnings)
```

- [ ] **Step 3: Run the focused status tests and verify they fail before the code change**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_status.py -k baseline
```

Expected: FAIL because `status` still scans `baseline/*_perf.txt` before the
state-declared `perf_artifact`.

### Task 2: Implement the minimal baseline-state-first selector

**Files:**
- Modify: `src/triton_agent/status/core.py`
- Test: `tests/test_status.py`

- [ ] **Step 1: Add a small helper that resolves `baseline/state.json` `perf_artifact`**

```python
def resolve_declared_baseline_perf_file(workspace: Path) -> tuple[Path | None, str | None, bool]:
    try:
        state = load_baseline_state(workspace)
    except ValueError:
        return None, None, False

    declared_perf = state.perf_artifact
    state_dir = baseline_dir(workspace)
    candidates = (state_dir / declared_perf, workspace / declared_perf)
    for candidate in candidates:
        if candidate.is_file():
            return candidate, None, True
    return None, missing_path_issue("perf_artifact", declared_perf), True
```

- [ ] **Step 2: Call the helper before `baseline/*_perf.txt` scanning**

```python
declared_baseline_perf, baseline_issue, had_declared_state = resolve_declared_baseline_perf_file(workspace)
if declared_baseline_perf is not None:
    return declared_baseline_perf, False
if baseline_issue is not None:
    warnings.append(baseline_issue)
    return None, True
if had_declared_state:
    return None, False
```

- [ ] **Step 3: Keep the existing legacy scan for workspaces with no usable baseline state**

```python
baseline_dir_path = baseline_dir(workspace)
if baseline_dir_path.is_dir():
    operator_perf_files = sorted(baseline_dir_path.glob("*_perf.txt"))
    ...
```

- [ ] **Step 4: Re-run the focused status tests and verify they pass**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_status.py -k baseline
```

Expected: PASS

### Task 3: Finish with targeted verification

**Files:**
- Verify only

- [ ] **Step 1: Run the full status test module**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_status.py
```

Expected: PASS

- [ ] **Step 2: Run the repository Python quality gates for the touched files if the status module stays clean**

Run:

```bash
uv run --group dev ruff check src/triton_agent/status/core.py tests/test_status.py
uv run pyright src/triton_agent/status/core.py tests/test_status.py
```

Expected: PASS
