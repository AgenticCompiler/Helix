# Generation Package Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat generation modules with a focused `triton_agent/generation/` package without changing CLI behavior.

**Architecture:** Create a small `generation/` package that separates domain options, output-path handling, single-workspace runtime, and batch orchestration. Update repository imports to use the new package layout, then remove the old top-level generation modules after the tests prove behavior is unchanged.

**Tech Stack:** Python, `argparse`, `unittest`, existing runner/prompt infrastructure

---

### Task 1: Add failing import and package-layout tests

**Files:**
- Modify: `tests/test_generation_commands.py`
- Modify: `tests/test_generation_batch.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests that import generation helpers from the planned package layout**

Add focused coverage for:
- `triton_agent.generation.models.GenerationOptions`
- `triton_agent.generation.outputs.resolve_generation_output_path`
- `triton_agent.generation.runtime.build_generation_request`
- `triton_agent.generation.batch.run_gen_eval_batch`

- [ ] **Step 2: Run the focused tests to verify the package imports fail before implementation**

Run:

```bash
uv run python -m unittest tests.test_generation_commands tests.test_generation_batch -v
```

Expected: FAIL with import errors for the new `triton_agent.generation.*` modules.

### Task 2: Create the generation package and move single-workspace helpers

**Files:**
- Create: `src/triton_agent/generation/__init__.py`
- Create: `src/triton_agent/generation/models.py`
- Create: `src/triton_agent/generation/outputs.py`
- Create: `src/triton_agent/generation/runtime.py`
- Modify: `tests/test_generation_commands.py`

- [ ] **Step 1: Add `GenerationOptions` to `src/triton_agent/generation/models.py`**

- [ ] **Step 2: Move output-path and overwrite helpers into `src/triton_agent/generation/outputs.py`**

Move:
- `resolve_generation_output_path`
- `prepare_generation_target`
- `prepare_generation_targets`

- [ ] **Step 3: Move request-building and runner invocation into `src/triton_agent/generation/runtime.py`**

Move:
- `GEN_EVAL_STAGED_SKILLS`
- `build_generation_request`
- `run_generation_request`

- [ ] **Step 4: Re-export the repository-facing generation symbols from `src/triton_agent/generation/__init__.py`**

- [ ] **Step 5: Run focused generation command tests**

Run:

```bash
uv run python -m unittest tests.test_generation_commands -v
```

Expected: PASS

### Task 3: Move batch orchestration under the generation package

**Files:**
- Create: `src/triton_agent/generation/batch.py`
- Modify: `tests/test_generation_batch.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the imports in tests against `triton_agent.generation.batch`**

- [ ] **Step 2: Move `gen-eval-batch` orchestration from the top-level module into `src/triton_agent/generation/batch.py`**

Keep:
- workspace discovery behavior
- `--input .` single-workspace behavior
- prefixed streaming output
- failure summarization and rendering semantics

- [ ] **Step 3: Run focused batch tests**

Run:

```bash
uv run python -m unittest tests.test_generation_batch tests.test_cli.PathResolutionTests -v
```

Expected: PASS

### Task 4: Update command handlers and delete the old top-level generation modules

**Files:**
- Modify: `src/triton_agent/commands/generation.py`
- Modify: `src/triton_agent/cli.py`
- Delete: `src/triton_agent/generation.py`
- Delete: `src/triton_agent/generation_batch.py`

- [ ] **Step 1: Update generation command handlers to import from the new package modules**

- [ ] **Step 2: Update any remaining repository imports from `triton_agent.generation` or `triton_agent.generation_batch`**

- [ ] **Step 3: Remove the old top-level generation files once all imports are updated**

- [ ] **Step 4: Run CLI regression tests**

Run:

```bash
uv run python -m unittest tests.test_cli -v
```

Expected: PASS

### Task 5: Verify the full refactor and documentation alignment

**Files:**
- Modify: `README.md` only if import or path wording needs adjustment
- Modify: `docs/specs/2026-04-09-generation-package-refactor-design.md` only if implementation proves a design change is needed

- [ ] **Step 1: Re-read the design and confirm the implementation stayed inside the planned package boundaries**

Check:

```bash
sed -n '1,220p' docs/specs/2026-04-09-generation-package-refactor-design.md
```

- [ ] **Step 2: Run targeted verification**

Run:

```bash
uv run python -m unittest tests.test_generation_commands tests.test_generation_batch tests.test_cli -v
```

Expected: PASS

- [ ] **Step 3: Run repository verification**

Run:

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m unittest discover -s tests -v
```

Expected: PASS

- [ ] **Step 4: If verification fails, fix only refactor regressions and re-run verification**
