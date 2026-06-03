# Optimize Round Performance Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a round-level performance analysis skill for `optimize`, extend existing profile and IR helpers to extract reusable signals, and let optimize rounds record an optional `perf-analysis.md` artifact without making it a mandatory gate.

**Architecture:** Keep the new capability skill-first. Extend `skills/triton-npu-profile-operator/scripts/profile_summary.py` and `skills/triton-npu-analyze-ir/scripts/inspect_ir.py` with structured signal outputs, then add a new `skills/triton-npu-analyze-round-performance/` workflow that combines those signals into `opt-round-N/perf-analysis.md`. Thread the new skill through optimize staging and prompt guidance, and add optional round-state/check support for declared analysis artifacts without tightening the required round contract.

**Tech Stack:** Python 3.11, `argparse`, `csv`, `json`, `pathlib`, existing optimize prompt/staging plumbing, Markdown skill docs, Python `unittest`

---

### Task 1: Add failing tests for profile signal extraction

**Files:**
- Modify: `tests/test_ascend_npu_operator_profiler.py`
- Modify: `skills/triton-npu-profile-operator/scripts/profile_summary.py`
- Reuse fixture: `tests/fixtures/ascend_npu_operator_profiler/realistic_parent_layout/mindstudio_profiler_output/op_statistic_20260402160352.csv`
- Reuse fixture: `tests/fixtures/ascend_npu_operator_profiler/realistic_parent_layout/mindstudio_profiler_output/op_summary_20260402160352.csv`

- [ ] **Step 1: Add a failing test for core-type aggregation**

Add a test that calls a new helper or CLI-facing function and asserts the parsed report can expose aggregated totals by `Core Type`, including scalar/vector/cube-style buckets when those rows exist.

- [ ] **Step 2: Add a failing test for data-movement hotspot heuristics**

Add a test that feeds synthetic `op_statistic` rows with names such as `Copy`, `TransData`, `Memcpy`, or similar transfer-like operators and asserts the summary groups them into a transfer-oriented signal section.

- [ ] **Step 3: Add a failing test for structured JSON output**

Add a test that requests JSON output and asserts the payload contains stable top-level keys for:

- `profile_dir`
- `target_operator`
- `core_type_totals`
- `data_movement_hotspots`
- `top_ops`

- [ ] **Step 4: Run the targeted profiler tests to confirm they fail**

Run: `uv run python -m unittest tests.test_ascend_npu_operator_profiler -v`

Expected: FAIL because `profile_summary.py` does not yet expose the new aggregation and JSON behavior.

### Task 2: Implement profile signal extraction with backward-compatible output

**Files:**
- Modify: `skills/triton-npu-profile-operator/scripts/profile_summary.py`
- Modify: `skills/triton-npu-profile-operator/SKILL.md`
- Modify: `skills/triton-npu-run-eval/scripts/run-command.py`

- [ ] **Step 1: Refactor profile parsing so the script builds reusable structured data before rendering**

Keep the existing Markdown behavior intact, but move parsing into helpers that can support both Markdown and JSON rendering.

- [ ] **Step 2: Implement `Core Type` aggregation helpers**

Add a helper that totals time, count, and ratio by normalized core type. Preserve raw core-type labels when the source data does not map cleanly to scalar/vector/cube.

- [ ] **Step 3: Implement transfer-hotspot heuristics**

Add a small, explicit heuristic matcher for transfer-heavy operator names or types. Keep the matcher conservative and document that it is heuristic rather than authoritative.

- [ ] **Step 4: Add JSON output support**

Expose a `--format markdown|json` option or equivalent API-level parameter. Keep Markdown as the default so existing skill usage and `run-command.py profile-bench` behavior remain unchanged.

- [ ] **Step 5: Keep the current human-readable report stable**

Update Markdown rendering only where needed to include the new signal sections without regressing the current operator timing summary.

- [ ] **Step 6: Update the profiler skill doc to mention the structured signal path**

Document that the script can now produce signal-oriented output that is suitable for round analysis, while keeping the current operator-summary workflow as the default.

- [ ] **Step 7: Run the targeted profiler tests to verify they pass**

Run: `uv run python -m unittest tests.test_ascend_npu_operator_profiler -v`

Expected: PASS

### Task 3: Add failing tests for IR performance-signal inspection

**Files:**
- Modify: `tests/test_inspect_ir.py`
- Modify: `skills/triton-npu-analyze-ir/scripts/inspect_ir.py`

- [ ] **Step 1: Extend the existing synthetic IR fixture builder to cover vector, copy, load/store, and sync-like patterns**

Reuse the current test file’s fixture style instead of adding a second IR fixture system.

- [ ] **Step 2: Add a failing parser test for the new performance-oriented subcommand**

Add a test that asserts `build_parser()` accepts a new subcommand such as `performance-signals` with `--ir-dir` and optional output-format flags.

- [ ] **Step 3: Add a failing behavior test for performance-signal summaries**

Assert that the new command can report structured counts or summaries for:

- vector-like operations
- copy or DMA-like operations
- load/store-heavy stages
- wait/barrier/set-flag patterns

- [ ] **Step 4: Add a failing JSON-output test**

Assert the new command can return machine-readable output with stable keys for stage summaries and suspicious transitions.

- [ ] **Step 5: Run the targeted IR tests to confirm they fail**

Run: `uv run python -m unittest tests.test_inspect_ir -v`

Expected: FAIL because `inspect_ir.py` does not yet expose the new performance-signal workflow.

### Task 4: Implement IR performance-signal summaries without regressing existing inspection flows

**Files:**
- Modify: `skills/triton-npu-analyze-ir/scripts/inspect_ir.py`
- Modify: `skills/triton-npu-analyze-ir/SKILL.md`

- [ ] **Step 1: Add the new performance-signal subcommand and parser options**

Keep `list-stages`, `stage-summary`, `diff-stages`, and `find-changes` unchanged.

- [ ] **Step 2: Build heuristic counters from existing stage text inspection**

Reuse the current keyword-scanning style where possible. Add only the new counters needed for vector, transfer, memory-access, and sync-like signals.

- [ ] **Step 3: Identify suspicious stages and stage transitions**

Use the existing discovered-stage ordering and adjacent-stage comparison approach so the new output stays consistent with the current IR navigation model.

- [ ] **Step 4: Add JSON rendering alongside text rendering**

Return structured output that the new round-analysis skill can consume, while preserving a readable default terminal mode.

- [ ] **Step 5: Update the IR skill doc to mention the new performance-oriented inspection path**

Document when to use the new subcommand and keep the rest of the navigation workflow intact.

- [ ] **Step 6: Run the targeted IR tests to verify they pass**

Run: `uv run python -m unittest tests.test_inspect_ir -v`

Expected: PASS

### Task 5: Add the new round-analysis skill and stage it with optimize

**Files:**
- Create: `skills/triton-npu-analyze-round-performance/SKILL.md`
- Modify: `src/triton_agent/optimize/orchestration.py`
- Modify: `src/triton_agent/models.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/test_skills.py`

- [ ] **Step 1: Add failing tests for staged optimize skills**

Add or update tests so optimize requests can declare staged skill names that include:

- `triton-npu-optimize`
- `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round`
- `triton-npu-analyze-round-performance`

- [ ] **Step 2: Add a failing prompt or contract test that expects the new skill name to appear in optimize-facing guidance**

Use the existing optimize prompt tests instead of creating a new test module.

- [ ] **Step 3: Write the new skill**

Document the required workflow:

- resolve one round
- require or collect `profile`
- collect `ir` only when needed
- strongly recommend spawning a subagent
- compare against parent or baseline when helpful
- write `opt-round-N/perf-analysis.md`
- separate facts, inferences, suggestions, and evidence gaps

- [ ] **Step 4: Teach optimize request building to stage the new skill**

Do this through `AgentRequest.staged_skill_names` so the runtime can copy only the optimize-relevant skill set into the workspace when needed.

- [ ] **Step 5: Run the targeted staging and prompt tests**

Run: `uv run python -m unittest tests.test_models tests.test_generation_contracts tests.test_skills tests.test_cli -v`

Expected: PASS for the touched assertions

### Task 6: Add optional round-state and triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round support for `perf-analysis.md`

**Files:**
- Modify: `src/triton_agent/optimize/models.py`
- Modify: `src/triton_agent/optimize/round_contract.py`
- Modify: `skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/scripts/optimize_check.py`
- Modify: `tests/test_optimize_round_contract.py`
- Modify: `tests/test_optimize_checks.py`

- [ ] **Step 1: Add failing round-contract tests for optional analysis metadata**

Add tests that confirm `RoundState` can parse optional fields:

- `perf_analysis_path`
- `analysis_comparison_sources`

and that artifact inspection can resolve a declared analysis file when present.

- [ ] **Step 2: Add failing triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round tests for declared analysis paths**

Assert that:

- rounds without `perf_analysis_path` still pass the artifact gate
- rounds with `perf_analysis_path` fail if the declared file is missing

- [ ] **Step 3: Extend the round-state model and loader**

Add optional fields only. Do not add them to the required round contract.

- [ ] **Step 4: Extend round artifact inspection and triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round validation**

Check the declared analysis file only when `round-state.json` includes `perf_analysis_path`.

- [ ] **Step 5: Run the targeted round contract tests**

Run: `uv run python -m unittest tests.test_optimize_round_contract tests.test_optimize_checks -v`

Expected: PASS

### Task 7: Integrate the new skill into optimize prompts and optimize docs

**Files:**
- Modify: `src/triton_agent/prompts.py`
- Modify: `skills/triton-npu-optimize/SKILL.md`
- Modify: `skills/triton-npu-optimize/references/workflow.md`
- Modify: `skills/triton-npu-optimize/references/artifacts.md`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add failing prompt tests for round-analysis guidance**

Assert that optimize worker and unsupervised prompts tell the agent to use `triton-npu-analyze-round-performance` when deeper diagnosis is needed, and that the output artifact is `opt-round-N/perf-analysis.md`.

- [ ] **Step 2: Update optimize prompt builders**

Add concise wording that keeps the current optimize contract intact while pointing to the new round-analysis skill when benchmark numbers, profile signals, or IR symptoms need deeper explanation.

- [ ] **Step 3: Update the optimize skill and references**

Document:

- when to call the new round-analysis skill
- that `profile` is the default required evidence
- that `ir` is collected when needed
- that `perf-analysis.md` is optional but preferred when deep diagnosis is performed

- [ ] **Step 4: Run the targeted prompt and doc-contract tests**

Run: `uv run python -m unittest tests.test_cli tests.test_generation_contracts -v`

Expected: PASS for the touched prompt assertions

### Task 8: Verify the whole change set without committing

**Files:**
- Modify only as needed from earlier tasks after verification feedback

- [ ] **Step 1: Run the focused test modules**

Run:

```bash
uv run python -m unittest \
  tests.test_ascend_npu_operator_profiler \
  tests.test_inspect_ir \
  tests.test_optimize_round_contract \
  tests.test_optimize_checks \
  tests.test_models \
  tests.test_generation_contracts \
  tests.test_skills \
  tests.test_cli -v
```

Expected: PASS

- [ ] **Step 2: Run the full unit suite**

Run: `uv run python -m unittest discover -s tests -v`

Expected: PASS

- [ ] **Step 3: Run type checking**

Run: `uv run pyright`

Expected: PASS

- [ ] **Step 4: Run lint for the touched Python files**

Run:

```bash
uv run --group dev ruff check \
  skills/triton-npu-profile-operator/scripts/profile_summary.py \
  skills/triton-npu-analyze-ir/scripts/inspect_ir.py \
  skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/scripts/optimize_check.py \
  src/triton_agent/models.py \
  src/triton_agent/optimize/models.py \
  src/triton_agent/optimize/orchestration.py \
  src/triton_agent/optimize/round_contract.py \
  src/triton_agent/prompts.py \
  tests/test_ascend_npu_operator_profiler.py \
  tests/test_inspect_ir.py \
  tests/test_optimize_round_contract.py \
  tests/test_optimize_checks.py \
  tests/test_models.py \
  tests/test_generation_contracts.py \
  tests/test_skills.py \
  tests/test_cli.py
```

Expected: PASS

- [ ] **Step 5: Run skill validation for the new skill**

Run: the skill validation workflow available in your environment against `skills/triton-npu-analyze-round-performance`

Expected: PASS

- [ ] **Step 6: Review `git diff` and stop for human approval before any commit**

Do not create a commit in this implementation pass unless the user explicitly asks for one after review.
