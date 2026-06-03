# Optimize Baseline Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `triton-npu-prepare-optimize-baseline`, move baseline preparation out of `triton-npu-optimize`, and align optimize prompts/tests with the new skill boundary.

**Architecture:** Keep optimize orchestration behavior unchanged while moving baseline preparation into a sibling skill contract. `triton-npu-optimize` becomes a round-focused workflow contract, the new baseline skill owns harness reuse/generation plus minimum repair through `check-baseline`, and prompt/test coverage locks the new boundary in place.

**Tech Stack:** Python 3, `unittest`, Markdown skill contracts, existing optimize prompt builders

---

## File Map

- Create: `skills/triton-npu-prepare-optimize-baseline/SKILL.md`
- Modify: `skills/triton-npu-optimize/SKILL.md`
- Modify: `skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/SKILL.md`
- Modify: `README.md`
- Modify: `src/triton_agent/prompts.py`
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_skills.py`

No change is planned for `src/triton_agent/optimize/orchestration.py`: optimize already stages the whole repo skill set with `staged_skill_names=None`, so the new baseline skill will be available without adding a new runtime branch.

### Task 1: Add The Baseline Skill Contract And Rewrite Optimize Baseline Docs

**Files:**
- Create: `skills/triton-npu-prepare-optimize-baseline/SKILL.md`
- Modify: `skills/triton-npu-optimize/SKILL.md`
- Modify: `skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/SKILL.md`
- Modify: `README.md`
- Test: `tests/test_generation_contracts.py`

- [ ] **Step 1: Write the failing doc-contract tests**

```python
    def test_optimize_baseline_preparation_uses_dedicated_skill(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        baseline = _read("skills/triton-npu-prepare-optimize-baseline/SKILL.md")
        optimize_check = _read("skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/SKILL.md")
        readme = _read("README.md")

        self.assertTrue(
            (REPO_ROOT / "skills" / "triton-npu-prepare-optimize-baseline" / "SKILL.md").exists()
        )
        self.assertIn("triton-npu-prepare-optimize-baseline", optimize)
        self.assertIn("triton-npu-gen-test", baseline)
        self.assertIn("triton-npu-gen-bench", baseline)
        self.assertIn("triton-npu-run-eval", baseline)
        self.assertIn("triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round", baseline)
        self.assertNotIn("../triton-npu-run-eval/scripts/run-command.py", optimize)
        self.assertIn("Do not use this skill to generate missing harnesses", optimize_check)
        self.assertIn("triton-npu-prepare-optimize-baseline", readme)
```

- [ ] **Step 2: Run the targeted doc-contract test and confirm it fails first**

Run: `uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_optimize_baseline_preparation_uses_dedicated_skill -v`

Expected: `FAIL` because the new skill file does not exist yet and `skills/triton-npu-optimize/SKILL.md` still contains direct `run-command.py` baseline instructions.

- [ ] **Step 3: Implement the new documentation boundary**

Create `skills/triton-npu-prepare-optimize-baseline/SKILL.md` with a complete baseline-only contract:

```md
---
name: triton-npu-prepare-optimize-baseline
description: Establish a reusable canonical optimize baseline by reusing or generating harnesses, performing minimum repair, and passing `check-baseline`.
---

# Prepare Optimize Baseline

## Goal

Establish a reusable canonical `baseline/` before any optimize round begins.

## Outputs

- reusable correctness and benchmark harnesses
- `baseline/`
- `baseline/state.json`
- `baseline/perf.txt`

## Workflow

### 1. Inspect And Reuse

- Reuse existing correctness and benchmark harnesses when they already validate the current operator workspace.
- If a correctness harness is missing, use the sibling `triton-npu-gen-test` skill.
- If a benchmark harness is missing, use the sibling `triton-npu-gen-bench` skill.

### 2. Reach A Benchmarkable Start

- Use the sibling `triton-npu-run-eval` skill for correctness validation and benchmark validation.
- If the current operator or harnesses need repair before they can validate cleanly, do only the minimum repair needed to reach a correct, benchmarkable starting point.

### 3. Write Canonical Baseline Artifacts

- Read `../triton-npu-optimize/references/artifacts.md` before writing `baseline/state.json`.
- Create `baseline/`.
- Write `baseline/state.json`.
- Write `baseline/perf.txt`.

### 4. Gate The Baseline

- Use the sibling `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` skill to run `check-baseline`.
- Keep repairing baseline state until `check-baseline` passes.
- Stop once the workspace has a reusable canonical baseline.

## Hard Rules

- Do not start `opt-round-N/` from this skill.
- Do not do open-ended optimization work here.
- Do not skip benchmark validation.
```

Replace the baseline section in `skills/triton-npu-optimize/SKILL.md` with a handoff instead of a full procedure:

```md
## Stage 0: Baseline Setup

- Reuse the existing `baseline/` only when it remains canonical for the current operator workspace.
- Otherwise use the sibling `triton-npu-prepare-optimize-baseline` skill to establish or repair the baseline before creating `opt-round-1/`.
- Read [artifacts.md](references/artifacts.md) before choosing authoritative baseline or round artifact paths.
- Read [opt-note-format.md](references/opt-note-format.md) before initializing `opt-note.md`.
```

Tighten `skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/SKILL.md` so it stays gate-only:

```md
- Do not use this skill to generate missing harnesses, repair operator logic, or invent missing baseline evidence.
- Baseline preparation belongs to `triton-npu-prepare-optimize-baseline`.
- Open-ended round analysis belongs to `triton-npu-optimize`.
```

Refresh the optimize section in `README.md` so the user-visible workflow matches the new skill boundary:

```md
- Establish or reuse a canonical `baseline/` directory before treating `opt-round-1` as the first optimization round.
- If `baseline/` is missing or invalid, baseline preparation is handled by `triton-npu-prepare-optimize-baseline` before round work begins.
- Every optimize run follows the default layered analysis ladder: pattern triage -> profiling diagnosis -> IR attribution -> compiler-source escalation.
```

- [ ] **Step 4: Re-run the targeted doc-contract test and then the full doc-contract suite**

Run: `uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_optimize_baseline_preparation_uses_dedicated_skill -v`

Expected: `PASS`

Run: `uv run python -m unittest tests.test_generation_contracts -v`

Expected: all generation-contract tests pass, including the new baseline-skill regression.

- [ ] **Step 5: Commit the documentation boundary change**

```bash
git add README.md \
  skills/triton-npu-prepare-optimize-baseline/SKILL.md \
  skills/triton-npu-optimize/SKILL.md \
  skills/triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round/SKILL.md \
  tests/test_generation_contracts.py
git commit -m "docs: add optimize baseline preparation skill"
```

### Task 2: Align Optimize Prompts With The New Baseline Skill

**Files:**
- Modify: `src/triton_agent/prompts.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing prompt tests for the new baseline handoff**

Add or replace assertions in `tests/test_cli.py` so worker, unsupervised, and supervisor prompts all follow the new boundary:

```python
        self.assertIn("Use the staged `triton-npu-prepare-optimize-baseline` skill", prompt)
        self.assertIn(
            "baseline preparation is needed, use the staged `triton-npu-prepare-optimize-baseline` skill",
            prompt.lower(),
        )
        self.assertNotIn(
            "use the staged `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` skill to run `check-baseline`",
            prompt.lower(),
        )
        self.assertIn(
            "use the staged `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` skill to run `check-round`",
            prompt.lower(),
        )
```

For the supervisor prompt, add:

```python
        self.assertIn("`triton-npu-prepare-optimize-baseline`", prompt)
        self.assertIn("`triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round`", prompt)
```

- [ ] **Step 2: Run the targeted prompt tests and confirm the old wording fails**

Run: `uv run python -m unittest tests.test_cli.PromptTests.test_build_optimize_worker_prompt_mentions_single_round_boundary tests.test_cli.PromptTests.test_build_optimize_unsupervised_prompt_mentions_baseline_state_contract tests.test_cli.PromptTests.test_build_optimize_supervisor_prompt_mentions_audit_role -v`

Expected: `FAIL` because the current prompts still route baseline repair through `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round`.

- [ ] **Step 3: Update prompt strings in `src/triton_agent/prompts.py`**

Replace the worker and unsupervised baseline lines with wording that points baseline work at the new skill and keeps `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` for round gating:

```python
        "Use the staged `triton-npu-prepare-optimize-baseline` skill when baseline artifacts are missing or invalid.",
        "Establish or reuse `baseline/` before creating `opt-round-1`.",
        "If baseline preparation is needed, use the staged `triton-npu-prepare-optimize-baseline` skill and continue only after it has repaired the baseline through `check-baseline`.",
        "Use `baseline/perf.txt` for canonical performance comparisons.",
```

Keep the round gate lines intact:

```python
        "After finishing the round, use the staged `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` skill to run `check-round` and repair the round until it passes.",
        "The current round must pass `check-round` through `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` before the invocation ends.",
```

Update the supervisor contract line so audits read all three relevant skills:

```python
                "Read the staged `triton-npu-optimize`, `triton-npu-prepare-optimize-baseline`, and `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` skills as the workflow contract that the worker round was supposed to follow.",
```

- [ ] **Step 4: Re-run the targeted prompt tests**

Run: `uv run python -m unittest tests.test_cli.PromptTests.test_build_optimize_worker_prompt_mentions_single_round_boundary tests.test_cli.PromptTests.test_build_optimize_unsupervised_prompt_mentions_baseline_state_contract tests.test_cli.PromptTests.test_build_optimize_supervisor_prompt_mentions_audit_role -v`

Expected: `PASS`

Run: `uv run python -m unittest tests.test_cli -v`

Expected: the full CLI prompt suite passes with the new baseline skill wording.

- [ ] **Step 5: Commit the prompt-boundary update**

```bash
git add src/triton_agent/prompts.py tests/test_cli.py
git commit -m "prompts: route optimize baseline through dedicated skill"
```

### Task 3: Lock In Skill Staging Coverage And Run Full Verification

**Files:**
- Modify: `tests/test_skills.py`

- [ ] **Step 1: Add a staging regression for the new skill**

Extend the existing explicit optimize-skill staging test in `tests/test_skills.py`:

```python
            links = manager.prepare_skills(
                "codex",
                workspace,
                skill_names=(
                    "triton-npu-optimize",
                    "triton-npu-prepare-optimize-baseline",
                    "triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round",
                    "triton-npu-analyze-round-performance",
                ),
            )

            target = self._skills_target(workspace, "codex")
            self.assertTrue((target / "triton-npu-optimize" / "SKILL.md").exists())
            self.assertTrue((target / "triton-npu-prepare-optimize-baseline" / "SKILL.md").exists())
            self.assertTrue((target / "triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round" / "SKILL.md").exists())
```

- [ ] **Step 2: Run the targeted staging regression**

Run: `uv run python -m unittest tests.test_skills.SkillLinkManagerTests.test_repo_skills_stage_optimize_and_optimize_check_for_codex -v`

Expected: `PASS`, confirming the new skill can be staged alongside optimize and optimize-check.

- [ ] **Step 3: Run the broader verification suite**

Run: `uv run python -m unittest tests.test_generation_contracts tests.test_cli tests.test_skills -v`

Expected: all targeted documentation, prompt, and staging regressions pass together.

Run: `uv run python -m unittest -v`

Expected: the full repository test suite passes without introducing optimize contract regressions.

- [ ] **Step 4: Commit the staging regression and verified final state**

```bash
git add tests/test_skills.py
git commit -m "test: cover optimize baseline skill staging"
```
