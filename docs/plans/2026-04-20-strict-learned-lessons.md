# Strict Learned Lessons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten `learned_lessons.md` so optimize agents record only reusable, evidence-backed optimization and analysis rules.

**Architecture:** Keep the behavior change in the optimize skill contract and prompt layer. Tests lock the wording in the skill docs, artifact docs, worker prompt, unsupervised prompt, and resume prompt.

**Tech Stack:** Markdown workflow docs, Python prompt builders, Python `unittest`

---

### Task 1: Add Strict Learned-Lessons Contract Tests

**Files:**
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_backends_base.py`

- [ ] **Step 1: Extend the optimize skill contract test**

Require `skills/triton/triton-npu-optimize/SKILL.md` to mention strict admission criteria, evidence support, applicability or limitations, and the ban on round narrative.

- [ ] **Step 2: Add an artifact contract assertion**

Require `skills/triton/triton-npu-optimize/references/artifacts.md` to state that command failures, local operator details, and round narrative belong outside `learned_lessons.md`.

- [ ] **Step 3: Add prompt assertions**

Require worker, unsupervised, and resume prompts to remind agents that `learned_lessons.md` only accepts reusable, evidence-backed rules.

- [ ] **Step 4: Run focused tests and verify RED**

Run: `uv run python -m unittest tests.test_generation_contracts tests.test_cli.PathResolutionTests.test_build_optimize_worker_prompt_mentions_single_round_boundary tests.test_cli.PathResolutionTests.test_build_optimize_unsupervised_prompt_mentions_baseline_state_contract tests.test_backends_base.SharedRunnerBaseTests.test_base_runner_resume_uses_shared_optimize_resume_prompt -v`

Expected: FAIL because the strict wording is not present yet.

### Task 2: Update Skill And Artifact Wording

**Files:**
- Modify: `skills/triton/triton-npu-optimize/SKILL.md`
- Modify: `skills/triton/triton-npu-optimize/references/artifacts.md`

- [ ] **Step 1: Rewrite `## Learned Lessons`**

Define `learned_lessons.md` as a reusable optimization knowledge distillation log with strict admission criteria.

- [ ] **Step 2: Update workflow and quality rules**

Replace broad "whenever you discover reusable knowledge" wording with strict "append only when the lesson passes admission criteria" wording.

- [ ] **Step 3: Update artifact reference**

Add destination guidance for rejected content: use `attempts.md`, `summary.md`, or `opt-note.md`.

- [ ] **Step 4: Run contract tests**

Run: `uv run python -m unittest tests.test_generation_contracts -v`

Expected: PASS.

### Task 3: Update Runtime Prompts

**Files:**
- Modify: `src/helix/prompts.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_backends_base.py`

- [ ] **Step 1: Add a shared learned-lessons prompt block**

Create concise prompt lines that explain the strict learned-lessons boundary.

- [ ] **Step 2: Include it in worker and unsupervised optimize prompts**

Place the block near artifact and evidence expectations.

- [ ] **Step 3: Include it in resume prompts**

Ensure resumed workers keep the same learned-lessons boundary even without rereading the first prompt.

- [ ] **Step 4: Run prompt tests**

Run: `uv run python -m unittest tests.test_cli.PathResolutionTests.test_build_optimize_worker_prompt_mentions_single_round_boundary tests.test_cli.PathResolutionTests.test_build_optimize_unsupervised_prompt_mentions_baseline_state_contract tests.test_backends_base.SharedRunnerBaseTests.test_base_runner_resume_uses_shared_optimize_resume_prompt -v`

Expected: PASS.

### Task 4: Final Verification

**Files:**
- All files changed above

- [ ] **Step 1: Run focused verification**

Run: `uv run python -m unittest tests.test_generation_contracts tests.test_cli.PathResolutionTests.test_build_optimize_worker_prompt_mentions_single_round_boundary tests.test_cli.PathResolutionTests.test_build_optimize_unsupervised_prompt_mentions_baseline_state_contract tests.test_backends_base.SharedRunnerBaseTests.test_base_runner_resume_uses_shared_optimize_resume_prompt -v`

Expected: PASS.

- [ ] **Step 2: Review diff**

Run: `git diff -- docs/specs/2026-04-20-strict-learned-lessons-design.md docs/plans/2026-04-20-strict-learned-lessons.md skills/triton/triton-npu-optimize/SKILL.md skills/triton/triton-npu-optimize/references/artifacts.md src/helix/prompts.py tests/test_generation_contracts.py tests/test_cli.py tests/test_backends_base.py`

Expected: Only strict learned-lessons docs, prompt wording, and matching tests changed.
