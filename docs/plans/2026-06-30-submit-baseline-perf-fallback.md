# Submit-Baseline Perf Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `submit-baseline` accept the current `run-bench`-style `baseline/<operator>_perf.txt` when `baseline/state.json` is missing or invalid, without weakening baseline validation.

**Architecture:** Keep valid `baseline/state.json` as the authority. Only change the fallback inspection path in `skills/common/ascend-npu-optimize-state/scripts/baseline/check.py`, then update regression tests so the contract is explicit.

**Tech Stack:** Python, unittest, skill-side optimize-state baseline checker.

---

### Task 1: Add a failing regression test for operator-named fallback perf artifacts

**Files:**
- Modify: `tests/test_optimize_baseline.py`
- Test: `tests/test_optimize_baseline.py`

- [ ] **Step 1: Add a regression test for invalid state plus `baseline/<operator>_perf.txt`**

```python
def test_inspect_baseline_artifacts_accepts_operator_named_perf_when_state_is_invalid(self) -> None:
    ...
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_optimize_baseline.py -k operator_named_perf`
Expected: FAIL because fallback inspection only looks for `baseline/perf.txt`

### Task 2: Implement the minimal fallback discovery update

**Files:**
- Modify: `skills/common/ascend-npu-optimize-state/scripts/baseline/check.py`

- [ ] **Step 1: Add a helper that finds fallback baseline perf artifacts**

```python
def _find_fallback_perf_artifact(root: Path) -> Path | None:
    ...
```

- [ ] **Step 2: Use that helper only when `baseline/state.json` is missing or invalid**

```python
if state is None and perf_path is None:
    perf_path = _find_fallback_perf_artifact(root)
```

- [ ] **Step 3: Broaden the default missing-artifact wording**

```python
issues.append(missing_issue(declared_perf, default_path="baseline perf artifact"))
```

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_optimize_baseline.py tests/test_optimize_checks.py -k baseline`
Expected: PASS

### Task 3: Final verification

**Files:**
- Verify: `skills/common/ascend-npu-optimize-state/scripts/baseline/check.py`
- Verify: `tests/test_optimize_baseline.py`
- Verify: `tests/test_optimize_checks.py`

- [ ] **Step 1: Run the strict skill-script pyright check**

Run: `bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/baseline/check.py`
Expected: PASS

- [ ] **Step 2: Run the standard repository verification commands relevant to touched files**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_optimize_baseline.py tests/test_optimize_checks.py`
Expected: PASS
