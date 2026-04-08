# Ascend Operator IR Analyzer Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new repository-owned skill that captures complete Triton Ascend compiler IR into a stable archive layout, supports both local and remote execution, and guides the agent to analyze archived IR directly or pair it with profiler evidence when needed.

**Architecture:** Keep IR capture deterministic in a bundled skill script rather than in the CLI. Add a new `skills/ascend-operator-ir-analyzer/` skill with a Python capture helper that accepts a benchmark harness plus operator file, renders the benchmark command under Triton debug env vars, parses stdout for the dump path and Bisheng compile command, rewrites and replays that compile command against an archived `kernel.ttadapter.mlir`, and stores the resulting artifacts plus a manifest in one archive directory. Reuse the shared `skills/operator-eval/scripts/run_runtime.py` SSH and copy helpers for remote execution so remote behavior matches existing repository semantics. Keep the skill text thin and procedural, and point the agent to the existing profiler skill when timing evidence is needed.

**Tech Stack:** Python 3.11, `argparse`, `subprocess`, `shlex`, `json`, existing operator-eval runtime helpers, `unittest`, Markdown skill docs

---

### Task 1: Add failing tests for IR capture parsing and rewriting

**Files:**
- Create: `tests/test_ascend_operator_ir_analyzer.py`

- [ ] **Step 1: Add failing tests for extracting `Dumping intermediate results to ...` and `[DEBUG] cmd_list: ...` lines from command stdout**
- [ ] **Step 2: Add failing tests for normalizing `--append-bisheng-options=...` so the embedded path stays part of one argument**
- [ ] **Step 3: Add failing tests for compile-command rewriting, including input replacement, one-shot filter removal, `--mlir-print-ir-after-all`, `--mlir-print-ir-tree-dir`, and `all-ir.txt` stderr redirection**
- [ ] **Step 4: Add failing tests for manifest contents and archive layout helpers**
- [ ] **Step 5: Add failing tests for remote archive command construction and keep-or-clean remote workspace behavior**
- [ ] **Step 6: Run the targeted test module and confirm the functionality does not exist yet**

### Task 2: Create the skill skeleton and implement the capture script

**Files:**
- Create: `skills/ascend-operator-ir-analyzer/SKILL.md`
- Create: `skills/ascend-operator-ir-analyzer/agents/openai.yaml`
- Create: `skills/ascend-operator-ir-analyzer/scripts/capture_ir.py`

- [ ] **Step 1: Initialize the new skill under `skills/` with a `scripts/` resource folder and generated agent metadata**
- [ ] **Step 2: Implement local command execution with `TRITON_DEBUG=1` and `TRITON_ALWAYS_COMPILE=1`**
- [ ] **Step 3: Implement stdout parsing that fails explicitly when the dump path or compile command is missing**
- [ ] **Step 4: Implement archive creation, dump copy, manifest writing, and replay-command rendering**
- [ ] **Step 5: Implement compile-command rewriting with structured argument parsing instead of brittle string replacement**
- [ ] **Step 6: Implement local compiler replay so `bishengir_stages/` and `all-ir.txt` are produced in the archive**
- [ ] **Step 7: Run the targeted test module and make the new local behavior pass**

### Task 3: Add remote capture support through shared runtime helpers

**Files:**
- Modify: `skills/ascend-operator-ir-analyzer/scripts/capture_ir.py`
- Reuse: `skills/operator-eval/scripts/run_runtime.py`

- [ ] **Step 1: Load the shared remote runtime helpers from `skills/operator-eval/scripts/run_runtime.py` without coupling the new skill to `triton_agent` package imports**
- [ ] **Step 2: Implement remote workspace creation, file staging, remote command execution, and archive copy-back using the same `--remote`, `--remote-workdir`, and `--keep-remote-workdir` semantics used elsewhere in the repository**
- [ ] **Step 3: Ensure remote mode replays the compiler on the remote machine and copies the completed archive directory back to the requested local archive path**
- [ ] **Step 4: Make remote failures short and actionable for missing artifacts, failed remote commands, or local archive collisions**
- [ ] **Step 5: Re-run targeted tests and make the remote behavior pass**

### Task 4: Write the skill contract and update repository docs

**Files:**
- Modify: `skills/ascend-operator-ir-analyzer/SKILL.md`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/2026-04-07-ascend-operator-ir-analyzer-skill.md`

- [ ] **Step 1: Write concise trigger-focused skill metadata that makes the new skill discoverable for Triton Ascend IR capture and analysis requests**
- [ ] **Step 2: Document the default workflow around `capture_ir.py`, archive inspection, and optional profiler pairing**
- [ ] **Step 3: Add a short README section that introduces the new skill and shows local and remote invocation examples**
- [ ] **Step 4: Update `AGENTS.md` so the durable project rules mention the new IR-analysis skill and its relation to the profiler skill**
- [ ] **Step 5: Refine the design doc if implementation details force any visible behavior changes**

### Task 5: Validate the skill and full repository change

**Files:**
- Modify: `tests/test_ascend_operator_ir_analyzer.py`
- Modify: `skills/ascend-operator-ir-analyzer/scripts/capture_ir.py`

- [ ] **Step 1: Run `python3 /Users/cdj/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/ascend-operator-ir-analyzer`**
- [ ] **Step 2: Run the targeted IR-analyzer unittest module**
- [ ] **Step 3: Run `uv run python -m unittest discover -s tests -v`**
- [ ] **Step 4: Run `uv run --group dev ruff check`**
- [ ] **Step 5: Run `uv run pyright`**
- [ ] **Step 6: Fix any regressions and re-run verification until clean**
