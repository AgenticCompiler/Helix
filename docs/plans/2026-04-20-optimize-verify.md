# Optimize Verify Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `optimize-verify` to rerun correctness and benchmark validation for the numeric best optimize round in an isolated verification directory.

**Architecture:** Keep CLI parsing in `src/triton_agent/cli.py`, command handling in `src/triton_agent/commands/optimize.py`, and feature-local resolution/execution in a new `src/triton_agent/optimize/verify.py`. Reuse baseline and round contract loaders plus the existing execution and compare helpers instead of duplicating test, benchmark, or perf parsing logic.

**Tech Stack:** Python 3.11, `argparse`, `pathlib`, `shutil`, existing `triton-npu-run-eval` helper modules, Python `unittest`.

---

### Task 1: Document The User Contract

**Files:**
- Create: `docs/specs/2026-04-20-optimize-verify-design.md`
- Create: `docs/plans/2026-04-20-optimize-verify.md`

- [x] **Step 1: Write the design document**

Capture numeric best selection, isolated output directory behavior, copied operator execution, and state file semantics.

- [x] **Step 2: Write this implementation plan**

Keep the plan scoped to a single local command.

### Task 2: Add Target Resolution Tests

**Files:**
- Create: `tests/test_optimize_verify.py`
- Create: `src/triton_agent/optimize/verify.py`

- [ ] **Step 1: Write failing tests**

Cover resolving:

- numeric best round from existing perf artifacts
- baseline test and benchmark paths from `baseline/state.json`
- selected round-local operator from `opt-round-N`
- fresh `opt-verify/verify-*` directory creation
- copied operator path living inside the verification directory

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m unittest tests.test_optimize_verify -v`

- [ ] **Step 3: Implement target resolution**

Add focused dataclasses and helpers in `optimize/verify.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m unittest tests.test_optimize_verify -v`

### Task 3: Add Execution Tests

**Files:**
- Modify: `tests/test_optimize_verify.py`
- Modify: `src/triton_agent/optimize/verify.py`

- [ ] **Step 1: Write failing tests**

Mock the existing execution and comparison helpers to assert:

- `phase=test` calls only test execution
- `phase=bench` calls benchmark and compare-perf
- `phase=all` calls test, benchmark, then compare-perf
- all runner calls use the copied operator
- failed runner return codes are surfaced and still leave `verify-state.json`

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m unittest tests.test_optimize_verify -v`

- [ ] **Step 3: Implement execution flow**

Reuse `run_local_test`, `run_remote_test`, `run_local_bench`, `run_remote_bench`, and `compare_perf_files`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m unittest tests.test_optimize_verify -v`

### Task 4: Wire The CLI

**Files:**
- Modify: `src/triton_agent/models.py`
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Modify: `src/triton_agent/commands/__init__.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing parser and handler tests**

Cover:

- `optimize-verify` command kind
- `optimize_verify` alias
- `--phase all|test|bench`
- no agent/interact/show-output flags
- remote flags are available

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.CliParserTests -v`

- [ ] **Step 3: Add parser and handler wiring**

Route to `handle_optimize_verify`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m unittest tests.test_cli.CliParserTests -v`

### Task 5: Update User Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add `optimize-verify` to the command map**

- [ ] **Step 2: Add a short usage section near optimize status**

Document isolated output directory behavior and phase options.

### Task 6: Final Verification

**Files:**
- No code changes expected

- [ ] **Step 1: Run focused tests**

Run: `uv run python -m unittest tests.test_optimize_verify tests.test_cli tests.test_execution_commands -v`

- [ ] **Step 2: Run standard repository checks**

Run:

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m unittest discover -s tests -v
```
