# Verification Command Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename optimize verification commands to `verify` and `verify-batch`, move their implementation into a dedicated `verification/` package, and remove the old names entirely.

**Architecture:** Keep verification behavior unchanged while relocating its code out of `optimize/`. Split command parsing from verification runtime by introducing `commands/verification.py` and `verification/{core,batch}.py`, then update CLI enums, help grouping, tests, and docs to the new names without compatibility aliases.

**Tech Stack:** Python 3.12, `argparse`, `pathlib`, existing execution helpers, Python `unittest`

---

### Task 1: Lock in the new command surface with failing CLI tests

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_verify_batch.py`

- [ ] **Step 1: Write the failing tests**

Add tests that assert:
- `verify` maps to the single-workspace verification command kind
- `verify-batch` maps to the batch verification command kind
- `verify-batch` accepts remote flags
- top-level help shows a `Verification` group
- old names `optimize-verify` and `optimize-verify-batch` do not appear in help or alias normalization coverage

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_cli.CliParserTests tests.test_verify_batch.VerifyBatchTests -v`
Expected: FAIL because the CLI still uses optimize-prefixed names and optimize-owned handlers.

- [ ] **Step 3: Write minimal implementation**

Update the command enum, parser registration, command grouping, and alias normalization so only `verify` and `verify-batch` are exposed.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_cli.CliParserTests tests.test_verify_batch.VerifyBatchTests -v`
Expected: PASS

### Task 2: Move verification runtime into a dedicated package

**Files:**
- Create: `src/triton_agent/verification/__init__.py`
- Create: `src/triton_agent/verification/core.py`
- Create: `src/triton_agent/verification/batch.py`
- Modify: `tests/test_verify.py`
- Modify: `tests/test_verify_batch.py`

- [ ] **Step 1: Write the failing import and behavior tests**

Update tests to import verification types and functions from the new package paths and to call the renamed APIs:
- `VerifyOptions`
- `VerifyResult`
- `prepare_verify_target()`
- `run_verify()`
- `run_verify_batch()`

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_verify tests.test_verify_batch -v`
Expected: FAIL because the new package and API names do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Move the current verification logic out of `optimize/verify.py` and `optimize/verify_batch.py` into `verification/core.py` and `verification/batch.py`, renaming the exported types and functions as part of the move.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_verify tests.test_verify_batch -v`
Expected: PASS

### Task 3: Split command handlers out of optimize commands

**Files:**
- Create: `src/triton_agent/commands/verification.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Modify: `tests/test_verify_batch.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing dispatch tests**

Add tests that assert the verification commands dispatch through `commands/verification.py` instead of `commands/optimize.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_cli tests.test_verify_batch -v`
Expected: FAIL because verify handlers still live in `commands/optimize.py`.

- [ ] **Step 3: Write minimal implementation**

Create dedicated verification handlers, update CLI imports and command specs, and remove verify-specific handler ownership from the optimize command module.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_cli tests.test_verify_batch -v`
Expected: PASS

### Task 4: Rename docs and run full verification

**Files:**
- Modify: `README.md`
- Modify: `docs/specs/2026-04-20-verify-design.md`
- Modify: `docs/specs/2026-04-21-verify-batch-design.md`
- Modify: `docs/plans/2026-04-20-verify.md`
- Modify: `docs/plans/2026-04-21-verify-batch.md`

- [ ] **Step 1: Update user-facing docs**

Replace user-facing command references with `verify` and `verify-batch`, and update any module-path references that still point to `optimize/verify*.py`.

- [ ] **Step 2: Run focused regression checks**

Run: `uv run python -m unittest tests.test_cli tests.test_verify tests.test_verify_batch -v`
Expected: PASS

- [ ] **Step 3: Run full repository verification**

Run:
- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m unittest discover -s tests -v`

Expected:
- Ruff: `All checks passed!`
- Pyright: `0 errors, 0 warnings, 0 informations`
- Unittest: all tests PASS
