# Baseline Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the canonical baseline artifact contract so future optimize rounds can depend on a strict `baseline/` layout before any orchestration changes.

**Architecture:** Introduce a standalone baseline helper module that knows how to find, validate, and expose the workspace `baseline/` directory contents, then reuse lightweight dataclasses to surface that contract to later consumers.

**Tech Stack:** `dataclasses`, `pathlib`, `json`, Python `unittest` via `uv run python -m unittest`.

---

### Task 1: Baseline contract helpers

**Files:**
- Create: `src/helix/optimize/baseline.py`
- Modify: `src/helix/optimize/models.py`
- Test: `tests/test_optimize_baseline.py`

- [ ] **Step 1: Write the failing test**

```python
def test_load_baseline_state_requires_state_perf_and_operator_snapshot(self) -> None:
    baseline = workspace / "baseline"
    baseline.mkdir()
    (baseline / "state.json").write_text(json.dumps({"baseline_kind": "original"}), encoding="utf-8")
    with self.assertRaises(ValueError):
        load_baseline_state(workspace)

def test_inspect_baseline_artifacts_prefers_baseline_perf(self) -> None:
    baseline = workspace / "baseline"
    baseline.mkdir(parents=True)
    (baseline / "state.json").write_text(state_json, encoding="utf-8")
    (baseline / "perf.txt").write_text("case: 1.0\n", encoding="utf-8")
    operator = baseline / "kernel.py"
    operator.write_text("print(0)\n", encoding="utf-8")
    result = inspect_baseline_artifacts(workspace)
    self.assertEqual(result.perf_path, operator_parent / "perf.txt")
    self.assertTrue(result.operator_path.samefile(operator))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m unittest tests.test_optimize_baseline -v
```
*Expected:* FAIL because `load_baseline_state`, `inspect_baseline_artifacts`, `BaselineState`, and the test file do not exist yet.

- [ ] **Step 3: Implement the minimal code**

```python
@dataclass(frozen=True)
class BaselineState:
    baseline_kind: str
    source_operator: str
    baseline_operator: str
    test_file: str
    test_mode: str
    bench_file: str
    bench_mode: str
    perf_artifact: str
    correctness_status: str
    benchmark_status: str
    baseline_established: bool
    preparation_notes: str | None = None
```

Helper functions:

```python
def baseline_dir(workspace: Path) -> Path:
    return workspace / "baseline"

def baseline_state_path(workspace: Path) -> Path:
    return baseline_dir(workspace) / "state.json"

def load_baseline_state(workspace: Path) -> BaselineState:
    ...

def inspect_baseline_artifacts(workspace: Path) -> BaselineArtifactsInspection:
    ...
```

Implementation notes:
1. Parse `baseline/state.json` using `json.loads`, enforce the required keys, and reject non-object payloads or missing fields with short `ValueError` messages.
2. Confirm `baseline_performance` file (default `baseline/perf.txt`) exists inside `baseline_dir`.
3. Treat `baseline/` contents excluding `state.json` and `perf.txt` as候 the operator snapshot; error on zero or multiple remaining files.
4. Return dataclass with resolved paths so other layers can inspect the canonical baseline.

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run python -m unittest tests.test_optimize_baseline -v
```

- [ ] **Step 5: Commit baseline contract**

```bash
git add src/helix/optimize/baseline.py src/helix/optimize/models.py tests/test_optimize_baseline.py
git commit -m "feat: add optimize baseline artifact contract"
```
