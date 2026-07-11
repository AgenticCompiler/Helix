# Differential Run-Test Auto-Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let differential `run-test` optionally execute the archived-result comparison step in the same command when an oracle payload path is provided.

**Architecture:** Keep the CLI thin by extending `run-test` argument parsing and handler orchestration only. Reuse the existing skill-side comparison implementation for the actual result diff, and mirror the same flow inside the standalone `run-command.py` helper so both entrypoints stay aligned.

**Tech Stack:** Python, argparse, unittest, repository Markdown docs

---

### Task 1: Lock The New CLI Contract In Tests

**Files:**
- Modify: `tests/test_execution_commands.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_skill_command_script.py`

- [ ] **Step 1: Write the failing tests**

Add focused assertions for:
- `handle_run_test()` auto-comparing a successful differential run when `--oracle-result` is present
- `main()` returning the compare exit code instead of the raw run-test exit code
- `run-command.py` mirroring the same output and exit semantics

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_execution_commands tests.test_cli tests.test_skill_command_script
```

Expected: FAIL because `run-test` does not yet accept `--oracle-result` / `--compare-level` and still prints only the archived-result hint.

### Task 2: Implement Auto-Compare In Both Entrypoints

**Files:**
- Modify: `src/helix/cli.py`
- Modify: `src/helix/commands/execution.py`
- Modify: `skills/triton-npu-run-eval/scripts/run-command.py`

- [ ] **Step 1: Extend `run-test` argument parsing**

Add optional `--oracle-result` and `--compare-level` arguments to the `run-test` parser surfaces, keeping `--compare-level` optional and validation-driven.

- [ ] **Step 2: Implement minimal orchestration**

After a successful differential test run with an archived result and `--oracle-result`, call the existing compare helper, print its output inline, and return its exit code. Keep the old hint path when no oracle payload was supplied.

- [ ] **Step 3: Run targeted tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_execution_commands tests.test_cli tests.test_skill_command_script
```

Expected: PASS.

### Task 3: Update User-Facing Docs

**Files:**
- Modify: `README.md`
- Modify: `skills/triton-npu-run-eval/references/run-test.md`
- Modify: `skills/triton-npu-run-eval/references/compare-result.md`
- Modify: `skills/triton-npu-gen-test/SKILL.md`
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Update the docs and contract tests**

Document the one-command differential flow with `run-test --oracle-result ...`, while keeping `compare-result` as the manual two-artifact fallback.

- [ ] **Step 2: Run doc-contract tests**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts
```

Expected: PASS.
