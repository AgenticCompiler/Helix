# Optimize Status Subcommand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only `optimize-status` CLI subcommand that scans immediate child workspaces under one root directory and reports baseline mean latency, best mean latency, average per-case improvement, and both numeric-best and logged-best rounds.

**Architecture:** Keep the feature in the CLI orchestration layer. Extend the command enum and parser with one local-only subcommand, add small helpers to discover optimization artifacts and parse comparable perf files, and render a compact batch summary without invoking any agent backend or skill workflow.

**Tech Stack:** Python 3.11, `argparse`, `pathlib`, existing perf parsing helpers from the operator-eval bench runner, Python `unittest`

---

### Task 1: Lock Parser And Entry-Point Expectations

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/triton_agent/models.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add parser tests for:
- `optimize-status -i <dir>` mapping to a new `CommandKind`
- help text showing `optimize-status`
- `optimize_status` alias mapping to the canonical kebab-case command
- no agent, remote, interact, or output flags on the new command

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_cli.CliParserTests -v`
Expected: FAIL because `optimize-status` is not implemented yet

- [ ] **Step 3: Write minimal implementation**

Add the new command kind, parser entry, and alias normalization until the parser tests pass.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_cli.CliParserTests -v`
Expected: PASS

### Task 2: Lock Numeric Status Semantics In Tests

**Files:**
- Modify: `tests/test_cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add CLI behavior tests for:
- invalid optimize-status input path handling
- empty batch root handling
- `no-session` workspace reporting
- selecting the numeric best round from comparable perf files
- computing average improvement as the mean of per-case improvement rates
- surfacing `Logged best` from `opt-note.md`
- warning when numeric best and logged best differ
- warning when perf ids mismatch or perf files are missing

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_cli.PathResolutionTests -v`
Expected: FAIL because optimize-status scan and comparison helpers do not exist yet

- [ ] **Step 3: Write minimal implementation**

Add the smallest possible test fixtures and assertions that pin down the command's numeric contract.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_cli.PathResolutionTests -v`
Expected: PASS after the implementation in Task 3

### Task 3: Implement Local Optimize Status Scanning

**Files:**
- Modify: `src/triton_agent/models.py`
- Modify: `src/triton_agent/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Use the tests from Tasks 1 and 2 as the red step.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_cli -v`
Expected: FAIL before implementation

- [ ] **Step 3: Write minimal implementation**

Implement:
- command dispatch for `optimize-status`
- workspace discovery across immediate child directories
- baseline and round perf discovery
- comparable perf parsing and average-improvement computation
- logged-best parsing from `opt-note.md`
- compact result rendering with `ok`, `warning`, and `no-session` totals

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_cli -v`
Expected: PASS

### Task 4: Update User-Facing Docs

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Create: `docs/2026-04-07-optimize-status-subcommand.md`

- [ ] **Step 1: Document the new command examples and semantics in `README.md`**
- [ ] **Step 2: Update `AGENTS.md` so the new subcommand is part of the durable command contract**
- [ ] **Step 3: Re-read the design doc and ensure implementation wording still matches the approved numeric semantics**

### Task 5: Final Verification

**Files:**
- Modify: none
- Test: repo-wide checks

- [ ] **Step 1: Run focused tests**

Run: `uv run python -m unittest tests.test_cli -v`
Expected: PASS

- [ ] **Step 2: Run full verification**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS
