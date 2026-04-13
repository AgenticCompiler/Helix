# Optimize Compare-Perf Authority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `compare-perf` the required source of optimize performance conclusions and reject benchmark-passing rounds that do not record that source.

**Architecture:** Tighten optimize prompts and role briefs so agents are told to use `compare-perf`, then extend the round contract and gate so the CLI enforces the same rule independently of prompt compliance.

**Tech Stack:** Python `dataclasses`, JSON round metadata, existing optimize guidance/prompt helpers, Python `unittest`

---

## File Structure

**Modify**

- `src/triton_agent/prompts.py`
  Add explicit `compare-perf` authority wording to optimize worker, unsupervised, and resume prompts.
- `src/triton_agent/optimize_guidance.py`
  Add the same authority wording to worker and supervisor role briefs.
- `src/triton_agent/optimize/models.py`
  Extend `RoundState` with `perf_summary_source`.
- `src/triton_agent/optimize/round_contract.py`
  Require the new round-state field.
- `src/triton_agent/optimize/gate.py`
  Reject benchmark-passing rounds whose performance summary source is not `compare-perf`.
- `tests/test_optimize_guidance.py`
  Pin the new guidance wording.
- `tests/test_optimize_gate.py`
  Add failing and passing gate cases for `perf_summary_source`.
- `tests/test_optimize_runtime.py`
  Update fake successful rounds to satisfy the stricter contract.

### Task 1: Lock The New Contract In Tests

**Files:**
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_optimize_gate.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write the failing tests**

Add assertions that the optimize worker/supervisor guidance mentions `compare-perf`, and add a gate test like:

```python
def test_evaluate_round_gate_requires_compare_perf_as_perf_summary_source(self) -> None:
    round_dir = self._create_round(root, perf_summary_source="manual-calculation")
    result = evaluate_round_gate(round_dir)
    self.assertEqual(result.decision, GateDecision.REVISE_REQUIRED)
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run python -m unittest tests.test_optimize_guidance tests.test_optimize_gate tests.test_optimize_runtime -v`
Expected: FAIL because current prompts and round-state parsing do not require the new source.

### Task 2: Implement The Minimal Contract And Guidance Changes

**Files:**
- Modify: `src/triton_agent/prompts.py`
- Modify: `src/triton_agent/optimize_guidance.py`
- Modify: `src/triton_agent/optimize/models.py`
- Modify: `src/triton_agent/optimize/round_contract.py`
- Modify: `src/triton_agent/optimize/gate.py`

- [ ] **Step 1: Add the new round-state field**

Require `perf_summary_source` in `round-state.json` and expose it via `RoundState`.

- [ ] **Step 2: Enforce the gate**

If a round claims `benchmark_status == "passed"` but `perf_summary_source != "compare-perf"`, block the round with `revise-required`.

- [ ] **Step 3: Tighten prompts and briefs**

Tell workers and supervisors to use `compare-perf` as the only source for performance deltas and speedup metrics.

### Task 3: Verify The Focused Surface

**Files:**
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_optimize_gate.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Run the focused tests to verify they pass**

Run: `uv run python -m unittest tests.test_optimize_guidance tests.test_optimize_gate tests.test_optimize_runtime -v`
Expected: PASS

- [ ] **Step 2: Run broader verification**

Run:
- `uv run python -m unittest discover -s tests -v`

Expected: PASS, or identify any unrelated pre-existing failures separately.
