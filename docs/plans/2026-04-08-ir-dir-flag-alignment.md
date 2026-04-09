# IR Directory Flag Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the IR analyzer scripts' user-facing directory flag from `--archive-dir` to `--ir-dir` so it matches the optimize workflow's `opt-round-N/ir/` artifact convention.

**Architecture:** Keep script behavior and output layout unchanged. Update only the public CLI flag, help text, user-facing error wording, tests, and docs that reference the old flag.

**Tech Stack:** Python 3.11, `argparse`, `unittest`, Markdown docs

---

### Task 1: Update parser coverage first

**Files:**
- Modify: `tests/test_ascend_operator_ir_analyzer.py`
- Modify: `tests/test_inspect_ir.py`

- [ ] **Step 1: Add or update parser tests so `capture_ir.py` requires `--ir-dir`**
- [ ] **Step 2: Add or update parser tests so all `inspect_ir.py` subcommands require `--ir-dir`**

### Task 2: Rename the public flags in both scripts

**Files:**
- Modify: `skills/ascend-operator-ir-analyzer/scripts/capture_ir.py`
- Modify: `skills/ascend-operator-ir-analyzer/scripts/inspect_ir.py`

- [ ] **Step 1: Replace `--archive-dir` with `--ir-dir` in argument parsing**
- [ ] **Step 2: Update user-facing help text and error wording from "archive directory" to "IR directory" where appropriate**
- [ ] **Step 3: Keep artifact layout and analysis behavior unchanged**

### Task 3: Align docs and skill guidance

**Files:**
- Modify: `skills/ascend-operator-ir-analyzer/SKILL.md`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/2026-04-07-ascend-operator-ir-analyzer-skill.md`
- Modify: `docs/2026-04-08-inspect-ir-script.md`
- Modify: `docs/2026-04-08-inspect-ir-ranking-and-change-scan.md`

- [ ] **Step 1: Update command examples and parameter references to use `--ir-dir`**
- [ ] **Step 2: Keep optimize-facing wording aligned with `opt-round-N/ir/`**

### Task 4: Verify the rename

**Files:**
- Modify: `skills/ascend-operator-ir-analyzer/scripts/capture_ir.py`
- Modify: `skills/ascend-operator-ir-analyzer/scripts/inspect_ir.py`
- Modify: `tests/test_ascend_operator_ir_analyzer.py`
- Modify: `tests/test_inspect_ir.py`

- [ ] **Step 1: Run `uv run python -m unittest tests.test_ascend_operator_ir_analyzer -v`**
- [ ] **Step 2: Run `uv run python -m unittest tests.test_inspect_ir -v`**
- [ ] **Step 3: Run `uv run python -m unittest discover -s tests -v`**
- [ ] **Step 4: Run `uv run pyright`**
- [ ] **Step 5: Run `uv run --group dev ruff check skills/ascend-operator-ir-analyzer/scripts/capture_ir.py skills/ascend-operator-ir-analyzer/scripts/inspect_ir.py tests/test_ascend_operator_ir_analyzer.py tests/test_inspect_ir.py`**
- [ ] **Step 6: Run `uv run --with pyyaml python /Users/cdj/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/ascend-operator-ir-analyzer`**
