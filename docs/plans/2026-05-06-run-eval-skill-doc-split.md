# Run-Eval Skill Doc Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `skills/triton-npu-run-eval` documentation so the main skill file is a short router and the command usage details live in per-command Markdown guides.

**Architecture:** Keep the existing `triton-npu-run-eval` skill identity and `./scripts/run-command.py` helper entrypoint, but move usage details into one focused document per subcommand. Protect the new structure with a single documentation-contract test in `tests/test_generation_contracts.py`.

**Tech Stack:** Markdown skill docs, Python `unittest`

---

### Task 1: Reshape The Skill Documentation Boundary

**Files:**
- Modify: `skills/triton-npu-run-eval/SKILL.md`
- Create: `skills/triton-npu-run-eval/references/run-test.md`
- Create: `skills/triton-npu-run-eval/references/run-bench.md`
- Create: `skills/triton-npu-run-eval/references/profile-bench.md`
- Create: `skills/triton-npu-run-eval/references/compare-result.md`
- Create: `skills/triton-npu-run-eval/references/compare-perf.md`

- [ ] Replace the long command-by-command content in `SKILL.md` with a short router that names the helper script, maps each subcommand to its focused doc, and says not to read unrelated guides or `scripts/*.py` during normal use.
- [ ] Move the existing `run-test` guidance into `run-test.md`, preserving the required `--test-file` and `--operator-file` contract, metadata override rules, and remote examples.
- [ ] Move the existing `run-bench` guidance into `run-bench.md`, preserving the `standalone` and `msprof` mode notes and representative remote examples.
- [ ] Move the existing `profile-bench` guidance into `profile-bench.md`, preserving the standalone `--case-id` and msprof `--bench` or `--kernel-name` selection guidance plus remote examples.
- [ ] Move the existing compare flows into `compare-result.md` and `compare-perf.md`, preserving the differential-result and performance-summary contracts.

### Task 2: Lock The New Structure With A Contract Test

**Files:**
- Modify: `tests/test_generation_contracts.py`

- [ ] Add one test that asserts `skills/triton-npu-run-eval/SKILL.md` is a router, names the five focused docs, and no longer contains the old long per-command sections.
- [ ] In the same test, assert the new focused docs exist and still contain the key command-specific contracts and examples that downstream skills rely on.

### Task 3: Verify The Documentation Refactor

**Files:**
- No additional file changes

- [ ] Run `uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_run_eval_skill_routes_to_focused_command_docs -v` and confirm it passes.
- [ ] Run `uv run python -m unittest tests.test_generation_contracts -v` and confirm the full documentation-contract suite passes.
