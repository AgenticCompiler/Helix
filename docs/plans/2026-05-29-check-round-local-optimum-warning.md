# Check-Round Local Optimum Warning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add advisory `check-round` warnings when recent baseline-relative round gains have nearly flattened out, with environment-variable controls for the recent-round window and geomean-gain threshold.

**Architecture:** Keep the feature inside the existing `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` skill contract flow. Add one focused helper under `skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/scripts/` that loads baseline/round perf data using the same metric-source semantics as optimize conclusions, computes recent baseline-relative round scores, and returns advisory local-optimum/config warnings that `check_round()` appends only after the current round already passes the existing contract. Preserve those pass-time warnings in checked and supervised continuation summaries even when the session must continue only because `min_rounds` is not yet satisfied.

**Tech Stack:** Python, unittest, skill-side helper modules, existing perf artifact parsers from `skills/triton-npu-run-eval/scripts/perf_artifacts.py`

---

### Task 1: Add failing local-optimum warning tests

**Files:**
- Modify: `tests/test_optimize_checks.py`
- Reference: `skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/scripts/optimize_check_contract.py`

- [ ] **Step 1: Add a focused warning test for flat recent gains**

Add a new test that writes:
- a valid baseline perf artifact
- three valid round directories ending at `opt-round-3`
- round-local perf values whose baseline-relative geomean speedups increase only marginally across the recent window

Expected assertions:
- `optimize_checks.check_round(round_dir)` returns `decision == "pass"`
- at least one issue contains `optimization may be stagnating`
- the warning mentions reviewing earlier rounds and resuming from a round before the flat sequence

- [ ] **Step 2: Add a focused non-warning test for meaningful gains**

Add a new test that writes three valid rounds where at least one adjacent baseline-relative geomean gain clearly exceeds the threshold.

Expected assertions:
- `decision == "pass"`
- no issue contains `optimization may be stagnating`

- [ ] **Step 3: Add env-var fallback coverage**

Add one test that sets:
- `TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_WINDOW=abc`
- `TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_MAX_GEOMEAN_GAIN=-1`

Expected assertions:
- `decision == "pass"`
- warning issues include both invalid-env fallback messages

- [ ] **Step 4: Run the focused test file to verify RED**

Run:

```bash
uv run python -m unittest tests.test_optimize_checks -v
```

Expected:
- FAIL because the new local-optimum warning expectations are not implemented yet.

### Task 2: Implement skill-side local-optimum analysis

**Files:**
- Create: `skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/scripts/local_optimum_check.py`
- Modify: `skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/scripts/optimize_check_contract.py`
- Reference: `skills/triton-npu-run-eval/scripts/perf_artifacts.py`
- Modify: `src/triton_agent/optimize/execution.py`
- Modify: `src/triton_agent/optimize/prompts.py`

- [ ] **Step 1: Add the helper module**

Implement a focused helper that:
- loads `perf_artifacts.py` from the sibling `triton-npu-run-eval` skill
- parses `TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_WINDOW` and `TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_MAX_GEOMEAN_GAIN`
- walks recent `opt-round-*` directories up to the current round
- computes baseline-relative geomean speedup per round under a normalized metric basis
- returns advisory warnings only, never gate failures

- [ ] **Step 2: Integrate the helper into `check_round()`**

In `optimize_check_contract.py`, once the existing pass path is established:
- call the local-optimum helper
- append any returned config warnings or local-optimum warning to the pass issues tuple
- preserve all existing `decision` behavior

- [ ] **Step 3: Preserve warning propagation across round modes**

Ensure:
- checked mode continuation summaries keep advisory warnings alongside the minimum-round reminder
- supervised mode receives the same CLI warning summary in both supervisor and later worker prompts
- continuous mode explicitly tells the worker how to respond when `check-round` warns about a local optimum

- [ ] **Step 4: Keep the warning wording action-oriented**

Ensure the emitted local-optimum warning says:
- recent baseline-relative gains are marginal
- optimization may be stagnating in the current direction
- the next step is to review earlier rounds and consider resuming from a round before the flat sequence to explore a different path

- [ ] **Step 5: Run the focused test file to verify GREEN**

Run:

```bash
uv run python -m unittest tests.test_optimize_checks -v
```

Expected:
- PASS for the new local-optimum coverage and all existing optimize check tests.

### Task 3: Verify skill-script quality gates

**Files:**
- Modify: `skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/scripts/local_optimum_check.py`
- Modify: `skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/scripts/optimize_check_contract.py`

- [ ] **Step 1: Run the required strict pyright check for modified skill scripts**

Run:

```bash
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/scripts/local_optimum_check.py
```

Expected:
- PASS with no strict pyright errors.

- [ ] **Step 2: Run strict pyright for the updated contract module**

Run:

```bash
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/scripts/optimize_check_contract.py
```

Expected:
- PASS with no strict pyright errors.

- [ ] **Step 3: Re-run the focused optimize check tests as final evidence**

Run:

```bash
uv run python -m unittest tests.test_optimize_checks -v
```

Expected:
- PASS with fresh evidence after the pyright checks.
