# Convert Prompt Triton Continuity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten the convert skill contract and generated convert prompt so agents are explicitly told to deliver a real Triton kernel path instead of a pure PyTorch rewrite.

**Architecture:** Keep the change prompt-only. Strengthen the convert workflow contract in the skill source of truth and mirror the same policy in the `CommandKind.CONVERT` prompt builder, then pin both with narrow regression tests.

**Tech Stack:** Python, unittest, Markdown skill docs

---

### Task 1: Pin the stronger convert policy in tests

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Write the failing prompt test**

```python
self.assertIn("real Triton kernel path", prompt)
self.assertIn("A pure PyTorch rewrite does not satisfy this convert task", prompt)
self.assertIn("PyTorch-facing wrapper or module API may remain", prompt)
```

- [ ] **Step 2: Write the failing skill-contract test**

```python
self.assertIn("real Triton kernel path", convert_skill)
self.assertIn("pure PyTorch rewrite does not satisfy", convert_skill)
self.assertIn("PyTorch-facing wrapper or module API may remain", convert_skill)
```

- [ ] **Step 3: Run the targeted tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.PromptBuilderTests.test_convert_prompt_mentions_differential_validation_without_baseline tests.test_generation_contracts.GenerationContractTests.test_convert_skill_and_readme_document_differential_only_conversion`

Expected: FAIL because the stronger convert wording is not present yet.

### Task 2: Strengthen the convert workflow contract

**Files:**
- Modify: `skills/triton/triton-npu-convert-pytorch-operator/SKILL.md`
- Modify: `src/triton_agent/prompts.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_generation_contracts.py`

- [ ] **Step 1: Update the convert skill wording**

```markdown
- Convert the operator so the delivered output remains PyTorch-facing but implements the converted computation through a real Triton Ascend NPU kernel path.
- A PyTorch-facing wrapper or `torch.nn.Module` public API may remain when that is the intended interface.
- A pure PyTorch rewrite does not satisfy this convert task, even if differential tests pass.
```

- [ ] **Step 2: Update the convert prompt wording**

```python
"Keep the converted artifact as a PyTorch-facing operator backed by a real Triton Ascend NPU kernel path.",
"A PyTorch-facing wrapper or module API may remain when that is the intended public interface.",
"A pure PyTorch rewrite does not satisfy this convert task, even if differential validation passes.",
```

- [ ] **Step 3: Run the same targeted tests to verify they pass**

Run: `uv run python -m unittest tests.test_cli.PromptBuilderTests.test_convert_prompt_mentions_differential_validation_without_baseline tests.test_generation_contracts.GenerationContractTests.test_convert_skill_and_readme_document_differential_only_conversion`

Expected: PASS

### Task 3: Verify the full touched test files stay green

**Files:**
- Test: `tests/test_cli.py`
- Test: `tests/test_generation_contracts.py`

- [ ] **Step 1: Run the broader targeted regression commands**

Run: `uv run python -m unittest tests.test_cli tests.test_generation_contracts`

Expected: PASS

- [ ] **Step 2: Review the diff for scope**

Run: `git diff -- docs/plans/2026-04-28-convert-prompt-triton-continuity.md docs/specs/2026-04-28-convert-prompt-triton-continuity-design.md skills/triton/triton-npu-convert-pytorch-operator/SKILL.md src/triton_agent/prompts.py tests/test_cli.py tests/test_generation_contracts.py`

Expected: Only plan/spec plus prompt-scope contract/test changes; no runtime enforcement logic.
