# Inspect IR Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a unified `inspect_ir.py` helper that lets agents list archived Bisheng stages, summarize one stage, and diff two stages through a stable `--ir-dir` interface.

**Architecture:** Keep IR inspection in a single bundled script under `skills/triton-npu-analyze-ir/scripts/`. Implement shared archive and stage-resolution helpers, then expose three text-oriented subcommands on top: `list-stages`, `stage-summary`, and `diff-stages`. Update the IR analyzer skill and repository docs so captured IR workflows naturally continue into the new inspection helper rather than direct ad hoc file browsing.

**Tech Stack:** Python 3.11, `argparse`, `difflib`, `pathlib`, `re`, `unittest`, Markdown docs

---

### Task 1: Add failing tests for archive inspection behavior

**Files:**
- Create: `tests/test_inspect_ir.py`

- [ ] **Step 1: Add a synthetic archive fixture builder inside the tests so each test can create a minimal `ir-archive/bishengir_stages/...` tree**
- [ ] **Step 2: Add failing tests for `list-stages` output, keyword filtering, and limit handling**
- [ ] **Step 3: Add failing tests for stage selector resolution by relative path, filename stem, and unique substring**
- [ ] **Step 4: Add failing tests for `stage-summary` output sections and keyword counts**
- [ ] **Step 5: Add failing tests for `diff-stages` output headers, delta summary, and unified diff rendering**
- [ ] **Step 6: Run the targeted test module and confirm the script does not exist yet**

### Task 2: Implement the unified inspection script

**Files:**
- Create: `skills/triton-npu-analyze-ir/scripts/inspect_ir.py`

- [ ] **Step 1: Add parser wiring for `list-stages`, `stage-summary`, and `diff-stages`, all sharing `--ir-dir`**
- [ ] **Step 2: Implement archive validation and internal `bishengir_stages/` resolution**
- [ ] **Step 3: Implement stage discovery with stable ordering and human-readable labels**
- [ ] **Step 4: Implement selector resolution that fails explicitly on zero or multiple matches**
- [ ] **Step 5: Implement `list-stages` with optional grep and limit support**
- [ ] **Step 6: Implement `stage-summary` with line count, file size, keyword counts, and a short highlights section**
- [ ] **Step 7: Implement `diff-stages` with delta summary and unified diff output**
- [ ] **Step 8: Run the targeted inspection tests and make them pass**

### Task 3: Update the skill and repository docs

**Files:**
- Modify: `skills/triton-npu-analyze-ir/SKILL.md`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/notes/2026-04-08-inspect-ir-script.md`

- [ ] **Step 1: Update the IR analyzer skill so it tells agents to use `inspect_ir.py` after capture for stage navigation, summaries, and diffs**
- [ ] **Step 2: Add concise README guidance for the new inspection workflow and example commands**
- [ ] **Step 3: Update `AGENTS.md` so the durable repository rules mention the inspection helper as part of the skill-owned IR workflow**
- [ ] **Step 4: Refine the design doc if implementation details change any visible behavior**

### Task 4: Verify the full change

**Files:**
- Modify: `skills/triton-npu-analyze-ir/scripts/inspect_ir.py`
- Modify: `tests/test_inspect_ir.py`

- [ ] **Step 1: Run `uv run python -m unittest tests.test_inspect_ir -v`**
- [ ] **Step 2: Run `uv run python -m unittest discover -s tests -v`**
- [ ] **Step 3: Run `uv run pyright`**
- [ ] **Step 4: Run `uv run --group dev ruff check skills/triton-npu-analyze-ir/scripts/inspect_ir.py tests/test_inspect_ir.py`**
- [ ] **Step 5: Run the skill validation workflow available in your environment against `skills/triton-npu-analyze-ir`**
- [ ] **Step 6: Fix regressions and re-run verification until clean**
