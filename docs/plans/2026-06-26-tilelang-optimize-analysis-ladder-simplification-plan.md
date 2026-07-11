# TileLang Optimize Analysis Ladder Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove unsupported IR and compiler-source escalation guidance from the TileLang optimize workflow while keeping Triton behavior unchanged.

**Architecture:** Treat this as a TileLang-specific contract alignment change. First add failing tests for the TileLang skill and prompt/guidance builders, then make the shared optimize prompt helpers branch by language and trim TileLang-only skill text and subagent guidance.

**Tech Stack:** Markdown skill contracts, Python `unittest`

---

## File Map

- Add: `docs/specs/2026-06-26-tilelang-optimize-analysis-ladder-simplification-design.md`
- Modify: `skills/tilelang/tilelang-npu-optimize/SKILL.md`
- Modify: `src/helix/optimize/prompts.py`
- Modify: `src/helix/optimize/memory_file.py`
- Modify: `src/helix/optimize/subagents.py`
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_subagents.py`

## Task 1: Lock The TileLang Contract With Failing Tests

**Files:**
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_subagents.py`

- [ ] **Step 1: Add a failing TileLang skill-contract test**

Assert that `skills/tilelang/tilelang-npu-optimize/SKILL.md` keeps `pattern triage -> profiling diagnosis` and no longer contains the IR or compiler-source sections or artifacts.

- [ ] **Step 2: Add a failing TileLang prompt test**

Assert that a TileLang optimize prompt does not contain `IR attribution`, `compiler-source escalation`, or TileLang IR companion guidance, while still containing profiling diagnosis guidance.

- [ ] **Step 3: Add failing TileLang guidance and subagent tests**

Assert that TileLang memory guidance and perf-diagnosis subagents do not advertise IR collection or TileLang IR helper scripts.

- [ ] **Step 4: Run the targeted tests and confirm they fail first**

Run:

```bash
uv run python -m unittest \
  tests.test_generation_contracts.GenerationContractTests.test_tilelang_optimize_skill_stops_at_profiling_diagnosis \
  tests.test_cli.CliParserTests.test_tilelang_optimize_prompt_stops_at_profiling_diagnosis \
  tests.test_optimize_guidance.OptimizeGuidanceTests.test_prepare_shared_guidance_uses_tilelang_specific_analysis_ladder \
  tests.test_subagents.SubagentManagerTests.test_prepare_tilelang_subagent_omits_ir_collection_guidance -v
```

Expected: `FAIL` because the current shared TileLang workflow still advertises IR and compiler-source analysis.

## Task 2: Implement The TileLang-Specific Ladder

**Files:**
- Modify: `skills/tilelang/tilelang-npu-optimize/SKILL.md`
- Modify: `src/helix/optimize/prompts.py`
- Modify: `src/helix/optimize/memory_file.py`
- Modify: `src/helix/optimize/subagents.py`

- [ ] **Step 1: Simplify the TileLang skill contract**

Remove the `IR attribution` and `compiler-source escalation` sections, shorten the default ladder, and stop promising `ir/` and `compiler-analysis.md` artifacts.

- [ ] **Step 2: Branch shared prompt helpers by language**

Make TileLang prompt construction stop at profiling diagnosis and skip TileLang IR companion and compiler-source guidance, while keeping Triton behavior unchanged.

- [ ] **Step 3: Align memory guidance and subagents**

Make TileLang workspace guidance and diagnosis-only subagents avoid IR collection guidance, TileLang IR preloads, and TileLang IR helper-script permissions.

## Task 3: Verify The New Contract

**Files:**
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_subagents.py`

- [ ] **Step 1: Re-run the targeted tests**

Run the same focused `unittest` command from Task 1.

Expected: `PASS`

- [ ] **Step 2: Run the broader optimize-adjacent regression slice**

Run:

```bash
uv run python -m unittest \
  tests.test_generation_contracts \
  tests.test_cli \
  tests.test_optimize_guidance \
  tests.test_subagents -v
```

Expected: all selected suites pass.
