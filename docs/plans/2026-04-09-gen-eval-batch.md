# Gen Eval Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `gen-eval-batch` subcommand that scans one root directory of operator workspaces and runs one `gen-eval` workflow per workspace with bounded concurrency and compact batch summaries.

**Architecture:** Reuse the existing single-workspace `gen-eval` request builder and runtime for each workspace. Add one new batch wrapper module for workspace discovery, concurrent execution, prefixed streaming output, and summary rendering, mirroring the existing `optimize-batch` user experience without introducing a second workflow skill.

**Tech Stack:** Python `argparse`, `concurrent.futures`, `pathlib`, existing generation runtime helpers, Python `unittest`

---

### Task 1: Add The Batch CLI Surface

**Files:**
- Modify: `src/helix/models.py`
- Modify: `src/helix/cli.py`
- Modify: `src/helix/commands/generation.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing parser tests**

Add parser coverage for the new command:

```python
def test_gen_eval_batch_maps_to_command_kind(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["gen-eval-batch", "-i", "kernels"])
    self.assertEqual(args.command_kind, CommandKind.GEN_EVAL_BATCH)
    self.assertEqual(args.max_concurrency, 2)
    self.assertEqual(args.test_mode, "differential")
    self.assertEqual(args.bench_mode, "standalone")
    self.assertFalse(hasattr(args, "interact"))
    self.assertFalse(hasattr(args, "output"))
```

- [ ] **Step 2: Run parser tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.CliParserTests -v`
Expected: FAIL because `gen-eval-batch` is not registered yet.

- [ ] **Step 3: Implement the minimal parser and dispatch changes**

Add `CommandKind.GEN_EVAL_BATCH`, register the parser flags, add alias normalization, and dispatch to a new generation-batch handler.

- [ ] **Step 4: Run parser tests to verify they pass**

Run: `uv run python -m unittest tests.test_cli.CliParserTests -v`
Expected: PASS

### Task 2: Add Batch Discovery And Summary Helpers

**Files:**
- Create: `src/helix/generation_batch.py`
- Create: `tests/test_generation_batch.py`

- [ ] **Step 1: Write the failing helper tests**

Add tests for candidate detection and failure summaries:

```python
def test_resolve_batch_gen_eval_operator_file_excludes_generated_artifacts(self) -> None:
    ...

def test_is_batch_gen_eval_operator_candidate_filters_non_operator_names(self) -> None:
    ...

def test_summarize_batch_gen_eval_failure_falls_back_to_return_code(self) -> None:
    result = AgentResult(return_code=7, stdout="   \n", stderr="")
    self.assertEqual(summarize_batch_gen_eval_failure(result), "gen-eval exited with return code 7")
```

- [ ] **Step 2: Run helper tests to verify they fail**

Run: `uv run python -m unittest tests.test_generation_batch -v`
Expected: FAIL because the batch helper module does not exist yet.

- [ ] **Step 3: Implement the minimal helper module**

Add a generation-batch runtime module with:
- candidate-file filtering
- prefixed streaming helper
- failure summarization
- batch summary rendering

- [ ] **Step 4: Run helper tests to verify they pass**

Run: `uv run python -m unittest tests.test_generation_batch -v`
Expected: PASS

### Task 3: Wire Batch Runtime To Single-Workspace `gen-eval`

**Files:**
- Modify: `src/helix/generation.py`
- Modify: `src/helix/commands/generation.py`
- Modify: `src/helix/generation_batch.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing batch CLI tests**

Add CLI coverage mirroring `optimize-batch` behavior:

```python
def test_main_gen_eval_batch_auto_detects_operator_files(self) -> None:
    ...

def test_main_gen_eval_batch_reports_workspace_selection_failures(self) -> None:
    ...

def test_main_gen_eval_batch_honors_max_concurrency(self) -> None:
    ...

def test_main_gen_eval_batch_show_output_prefixes_workspace_streams(self) -> None:
    ...
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli tests.test_generation_batch -v`
Expected: FAIL because the handler and batch runtime are not wired.

- [ ] **Step 3: Implement the minimal runtime wiring**

Reuse `GenerationOptions`, `build_generation_request(CommandKind.GEN_EVAL, ...)`, and `run_generation_request(...)` for each workspace. Extend `run_generation_request` only as needed to support prefixed `--show-output` streaming in batch mode.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `uv run python -m unittest tests.test_cli tests.test_generation_batch -v`
Expected: PASS

### Task 4: Update User-Facing Docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update usage and behavior docs**

Add `gen-eval-batch` examples and document its batch scanning, concurrency, remote propagation, and unsupported flags.

- [ ] **Step 2: Sanity-check the docs against the implemented CLI**

Re-read the parser and the README examples together so the documented flags and defaults match the code.

### Task 5: Verify The Whole Change

**Files:**
- Modify: `README.md`
- Modify: `src/helix/*`
- Modify: `tests/*`
- Create: `src/helix/generation_batch.py`
- Create: `tests/test_generation_batch.py`

- [ ] **Step 1: Run targeted suites**

Run:
- `uv run python -m unittest tests.test_cli tests.test_generation_batch tests.test_generation_commands -v`

Expected: PASS

- [ ] **Step 2: Run repository verification**

Run:
- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m unittest discover -s tests -v`

Expected: PASS

- [ ] **Step 3: If verification fails, fix only `gen-eval-batch`-related regressions and re-run verification**

Keep the scope tight and avoid refactoring unrelated optimize or execution behavior.
