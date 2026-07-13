# Optimize Resume Baseline-State Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let optimize resume reuse harness paths declared in `baseline/state.json` when they belong to the current operator input.

**Architecture:** Keep the change local to `src/helix/optimize/resume.py` by teaching resume-time harness discovery to prefer declared baseline state paths, then fall back to the existing stem-based compatibility logic. Cover the new behavior with direct unit tests against the resume helper module and preserve CLI-level resume behavior.

**Tech Stack:** Python 3.11, `pathlib`, existing optimize baseline loaders, Python `unittest`

---

### Task 1: Add Failing Resume Tests

**Files:**
- Create: `tests/test_optimize_resume.py`
- Reference: `src/helix/optimize/resume.py`

- [ ] **Step 1: Write a failing test for declared test and bench paths**

```python
def test_classify_optimize_workspace_prefers_matching_baseline_state_paths(self) -> None:
    ...
    inspection = classify_optimize_workspace(operator, workspace)
    self.assertEqual(inspection.state, "resumable-session")
    self.assertEqual(inspection.test_mode, "differential")
    self.assertEqual(inspection.bench_mode, "msprof")
```

- [ ] **Step 2: Run the focused test and confirm it fails for the expected reason**

Run: `uv run python -m unittest tests.test_optimize_resume.OptimizeResumeTests.test_classify_optimize_workspace_prefers_matching_baseline_state_paths -v`
Expected: FAIL because resume only checks stem-derived harness names today.

- [ ] **Step 3: Add a failing mismatch-protection test**

```python
def test_classify_optimize_workspace_ignores_baseline_state_for_different_source_operator(self) -> None:
    ...
    inspection = classify_optimize_workspace(operator, workspace)
    self.assertEqual(inspection.state, "partial-session")
```

- [ ] **Step 4: Run the focused resume test module and confirm the new test set fails correctly**

Run: `uv run python -m unittest tests.test_optimize_resume -v`
Expected: FAIL with at least one assertion showing the declared-path resume behavior is missing.

### Task 2: Implement State-Aware Resume Discovery

**Files:**
- Modify: `src/helix/optimize/resume.py`
- Reference: `src/helix/optimize/baseline.py`

- [ ] **Step 1: Add helpers that resolve declared baseline state paths only when the source operator matches**

```python
def _declared_resume_harnesses(input_path: Path, workdir: Path) -> tuple[list[Path], Path | None]:
    ...
```

- [ ] **Step 2: Update workspace classification and continue-path validation to use declared paths first, then stem fallback**

```python
test_harnesses = _existing_test_harnesses(input_path, workdir)
bench_harness = _existing_bench_harness(input_path, workdir)
```

- [ ] **Step 3: Run the focused resume tests and confirm they pass**

Run: `uv run python -m unittest tests.test_optimize_resume -v`
Expected: PASS

### Task 3: Run Resume Regression Coverage

**Files:**
- Reference: `tests/test_cli.py`
- Reference: `tests/test_optimize_resume.py`

- [ ] **Step 1: Run targeted CLI and resume regression tests**

Run: `uv run python -m unittest tests.test_optimize_resume tests.test_cli -v`
Expected: PASS

- [ ] **Step 2: Run repository verification**

Run: `uv run --group dev ruff check`
Run: `uv run pyright`
Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS
