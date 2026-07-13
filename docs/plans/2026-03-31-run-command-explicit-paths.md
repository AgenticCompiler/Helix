# Run Command Explicit Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change `run-test` and `run-bench` to require explicit harness-file and operator-file inputs instead of deriving the harness from `--input`.

**Architecture:** Keep generation and optimize commands on the existing operator-input flow, but split the run-command CLI contract into explicit artifact and operator paths. Normalize those values in the CLI layer, update prompt generation to describe both files, and remove derived run-artifact resolution so the skills receive direct execution context.

**Tech Stack:** Python 3.11, `argparse`, `unittest`, `uv`, existing Helix CLI modules

---

### Task 1: Update CLI parsing contract for run commands

**Files:**
- Modify: `src/helix/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing parser tests**

Add parser tests covering:
- `run-test --test-file ... --operator-file ...`
- `run-bench --bench-file ... --operator-file ...`
- rejection of `--input` on both run commands

- [ ] **Step 2: Run parser-focused tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.CliParserTests -v`
Expected: FAIL on missing new flags / accepted old flag contract

- [ ] **Step 3: Implement the parser changes**

In `src/helix/cli.py`:
- keep `--input/-i` only on `gen-test`, `gen-bench`, and `optimize`
- add required `--test-file` + `--operator-file` to `run-test`
- add required `--bench-file` + `--operator-file` to `run-bench`
- keep the canonical subcommand names and alias normalization behavior unchanged

- [ ] **Step 4: Re-run parser-focused tests**

Run: `uv run python -m unittest tests.test_cli.CliParserTests -v`
Expected: PASS

### Task 2: Normalize explicit paths into request construction

**Files:**
- Modify: `src/helix/models.py`
- Modify: `src/helix/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing main/path tests**

Add tests covering:
- run commands validate the explicit harness and operator paths
- run commands no longer derive `test_<op>.py` / `bench_<op>.py`
- workdir comes from the harness-file directory for run commands

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.PathResolutionTests -v`
Expected: FAIL on old derived-artifact behavior

- [ ] **Step 3: Implement minimal request/path model changes**

Make the smallest change that preserves existing generation behavior:
- add explicit `operator_path` to `AgentRequest`
- continue using `input_path` as the primary file path for non-run commands
- for run commands, use `input_path` as the explicit harness file path and `operator_path` as the operator under test
- compute run-command workdir from the harness file directory
- remove run-command reliance on `resolve_execution_target(...)`

- [ ] **Step 4: Re-run targeted path tests**

Run: `uv run python -m unittest tests.test_cli.PathResolutionTests -v`
Expected: PASS

### Task 3: Update prompt and backend request propagation

**Files:**
- Modify: `src/helix/prompts.py`
- Modify: `src/helix/backends/codex.py`
- Modify: `src/helix/backends/opencode.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_codex_runner.py`
- Modify: `tests/test_opencode_runner.py`

- [ ] **Step 1: Write the failing prompt/request tests**

Add tests asserting run-command prompts mention both:
- operator file
- test file or benchmark file

Also update backend request-construction tests if the new request field requires it.

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.PromptTests -v`
Expected: FAIL because run prompts only describe one operator input today

- [ ] **Step 3: Implement minimal prompt and propagation updates**

Update prompt wording so:
- generation commands keep current wording
- `run-test` says `Operator file` and `Test file`
- `run-bench` says `Operator file` and `Benchmark file`

Thread the new request field through resume/copy helpers without changing backend launch semantics.

- [ ] **Step 4: Re-run targeted tests**

Run: `uv run python -m unittest tests.test_cli.PromptTests tests.test_codex_runner tests.test_opencode_runner -v`
Expected: PASS

### Task 4: Update docs and run full verification

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/notes/2026-03-31-helix-cli.md`
- Modify: `docs/notes/2026-03-31-run-command-explicit-operator-and-artifact-paths.md`

- [ ] **Step 1: Update user-facing docs**

Document the new run-command syntax and remove references to deriving run artifacts from the operator path.

- [ ] **Step 2: Run full verification**

Run:
- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m unittest discover -s tests -v`

Expected: all PASS
