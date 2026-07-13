# Verify Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `verify-batch`, teach `optimize-status` to surface the latest successful verification as `Verified`, and keep status reporting read-only.

**Architecture:** Extend the command surface with a dedicated batch verify entrypoint instead of overloading `optimize-status`. Keep single-workspace verification in `verification/core.py`, add a focused `verification/batch.py` module for root-level orchestration and latest-result reuse, and teach optimize status inspection/rendering to read the newest `verify-state.json` and derive a compact `verified` flag.

**Tech Stack:** Python 3.11, `argparse`, `pathlib`, existing optimize status/verify helpers, Python `unittest`

---

### Task 1: Add CLI coverage for the new batch verify command

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/helix/models.py`
- Modify: `src/helix/cli.py`

- [ ] **Step 1: Write the failing parser tests**

Add parser coverage for:
- `verify-batch -i workspace-root`
- `verify_batch -i workspace-root`
- `verify-batch -i workspace-root --force-verify`

Assert the command kind, `input`, and `force_verify` values.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_verify_batch_maps_to_command_kind tests.test_cli.CliParserTests.test_verify_batch_accepts_force_verify -v`

Expected: FAIL because the command kind and parser arguments do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Update:
- `src/helix/models.py`
  - add `CommandKind.VERIFY_BATCH = "verify-batch"`
  - add an empty `COMMAND_TO_SKILL` mapping entry
- `src/helix/cli.py`
  - register the new command in `_COMMAND_SPECS`
  - add `--force-verify` support for that command only
  - include snake_case alias normalization

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_verify_batch_maps_to_command_kind tests.test_cli.CliParserTests.test_verify_batch_accepts_force_verify -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_cli.py src/helix/models.py src/helix/cli.py
git commit -m "feat: add optimize verify batch cli"
```

### Task 2: Add optimize status verification metadata and latest verify parsing

**Files:**
- Modify: `tests/test_optimize_status.py`
- Modify: `tests/test_optimize_render.py`
- Modify: `src/helix/optimize/models.py`
- Modify: `src/helix/optimize/status.py`
- Modify: `src/helix/optimize/render.py`

- [ ] **Step 1: Write the failing status and render tests**

Add tests that cover:
- latest verify state discovery prefers the newest `verify-*` directory by name
- `verified` is `True` only when the latest verify result has passed `test`, `rerun_baseline_bench`, `rerun_best_bench`, and `compare_perf`
- partial or failed latest verify results keep `verified = False`
- markdown output adds a `Verified` column and shows either `Verified` or `-`

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_optimize_status tests.test_optimize_render -v`

Expected: FAIL because optimize status results do not yet expose latest verify metadata and markdown output has no `Verified` column.

- [ ] **Step 3: Write minimal implementation**

Update:
- `src/helix/optimize/models.py`
  - extend `OptimizeStatusWorkspace` with:
    - `latest_verify_state: Path | None`
    - `verified: bool`
- `src/helix/optimize/status.py`
  - add helpers to find the latest `opt-verify/verify-*/verify-state.json`
  - parse the latest state file defensively
  - compute `verified` from the latest full successful verify result
  - include the new fields in every `OptimizeStatusWorkspace`
- `src/helix/optimize/render.py`
  - add `Verified` to markdown output
  - render `Verified` only when `item.verified` is `True`
  - optionally print the latest verify state path in text output when present

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_optimize_status tests.test_optimize_render -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_optimize_status.py tests/test_optimize_render.py src/helix/optimize/models.py src/helix/optimize/status.py src/helix/optimize/render.py
git commit -m "feat: surface latest optimize verify status"
```

### Task 3: Add batch verify orchestration with reuse and force-rerun behavior

**Files:**
- Create: `src/helix/verification/batch.py`
- Modify: `tests/test_cli.py`
- Modify: `src/helix/commands/optimize.py`
- Modify: `src/helix/optimize/render.py`

- [ ] **Step 1: Write the failing batch verify behavior tests**

Add tests for:
- reusing the latest verify result by default
- rerunning verification when `--force-verify` is present
- skipping non-verifiable workspaces
- continuing after one workspace fails
- returning non-zero when any rerun fails
- `main(["verify-batch", ...])` dispatches the batch command correctly

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_verify_batch -v`

Expected: FAIL because the batch command handler and orchestration module do not exist.

- [ ] **Step 3: Write minimal implementation**

Add `src/helix/verification/batch.py` with focused helpers to:
- scan child workspace directories under a root
- discover the latest verify state
- decide reuse vs rerun
- call `prepare_verify_target()` and `run_verify()` only when rerun is needed
- collect per-workspace outcomes
- produce an exit code for the batch command

Update:
- `src/helix/commands/optimize.py`
  - add `handle_verify_batch`
- `src/helix/optimize/render.py`
  - add a renderer for batch verify outcomes if the new command needs explicit summary output

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_verify_batch -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/helix/verification/batch.py src/helix/commands/verification.py src/helix/optimize/render.py tests/test_cli.py tests/test_verify_batch.py
git commit -m "feat: add verify batch orchestration"
```

### Task 4: Update docs and run full verification

**Files:**
- Modify: `README.md`
- Modify: `docs/specs/2026-04-21-verify-batch-design.md`

- [ ] **Step 1: Write the failing docs-oriented regression expectation**

This task is doc and verification focused, so use the existing code/tests from Tasks 1-3 as the behavioral guardrail. No additional test file is required here.

- [ ] **Step 2: Run focused verification before doc updates**

Run: `uv run python -m unittest tests.test_cli tests.test_optimize_status tests.test_optimize_render tests.test_verify -v`

Expected: PASS before updating docs.

- [ ] **Step 3: Update user-facing documentation**

Update:
- `README.md`
  - document `verify-batch`
  - document `--force-verify`
  - document the `Verified` markdown column semantics
- `docs/specs/2026-04-21-verify-batch-design.md`
  - adjust wording only if implementation details changed during delivery

- [ ] **Step 4: Run full repository verification**

Run:
- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m unittest discover -s tests -v`

Expected:
- Ruff: `All checks passed!`
- Pyright: `0 errors, 0 warnings, 0 informations`
- Unittest: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add README.md docs/specs/2026-04-21-verify-batch-design.md
git commit -m "docs: document optimize verify batch"
```
