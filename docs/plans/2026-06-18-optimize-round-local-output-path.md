# Optimize Round-Local Output Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove misleading root-level optimize output-path hints from optimize prompts so agents only see the round-local artifact contract `opt-round-N/opt_<operator>.py`.

**Architecture:** Keep the internal optimize request `output_path` field unchanged for compatibility, but stop surfacing it in optimize-specific prompts. Implement the behavior with narrow prompt-builder changes so non-optimize commands keep their existing requested-output wording.

**Tech Stack:** Python, unittest, existing shared prompt builders and optimize runtime prompt helpers

---

### Task 1: Lock The Prompt Contract With Failing Tests

**Files:**
- Modify: `/Users/cdj/Projects/helix/tests/test_cli.py`
- Modify: `/Users/cdj/Projects/helix/tests/test_optimize_runtime.py`
- Test: `/Users/cdj/Projects/helix/tests/test_cli.py`
- Test: `/Users/cdj/Projects/helix/tests/test_optimize_runtime.py`

- [ ] **Step 1: Add a failing shared-prompt test for optimize output suppression**

Extend the optimize prompt coverage in `tests/test_cli.py` so optimize prompts explicitly reject root-level requested output lines while still requiring round-local guidance:

```python
self.assertNotIn("Requested output:", prompt)
self.assertIn(
    "For each round, write the optimized operator snapshot as `opt_<original-operator>.py` inside `opt-round-N/`.",
    prompt,
)
```

- [ ] **Step 2: Add a failing baseline-prompt test for optimize output suppression**

Update the baseline-phase runtime assertion in `tests/test_optimize_runtime.py` so the baseline repair prompt no longer accepts:

```python
self.assertIn(f"Requested output: {output_path.as_posix()}", baseline_request.prompt)
```

and instead requires:

```python
self.assertNotIn("Requested output:", baseline_request.prompt)
```

- [ ] **Step 3: Run the targeted tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_cli.CliPromptTests tests.test_optimize_runtime.OptimizeRuntimeTests.test_multi_invocation_controller_baseline_prompt_uses_explicit_context_parameters -v
```

Expected: FAIL because optimize prompts still include a generic requested output path.

### Task 2: Remove Root-Level Output Hints From Optimize Prompts

**Files:**
- Modify: `/Users/cdj/Projects/helix/src/helix/prompts.py`
- Modify: `/Users/cdj/Projects/helix/src/helix/optimize/prompts.py`
- Test: `/Users/cdj/Projects/helix/tests/test_cli.py`
- Test: `/Users/cdj/Projects/helix/tests/test_optimize_runtime.py`

- [ ] **Step 1: Stop the shared prompt builder from printing requested output for optimize**

Narrow the shared prompt output line in `src/helix/prompts.py` from:

```python
if output_path is not None and command_kind != CommandKind.GEN_EVAL:
    lines.append(f"Requested output: {_display_path(output_path)}")
```

to behavior equivalent to:

```python
if output_path is not None and command_kind not in {CommandKind.GEN_EVAL, CommandKind.OPTIMIZE}:
    lines.append(f"Requested output: {_display_path(output_path)}")
```

- [ ] **Step 2: Stop the optimize baseline prompt from printing requested output**

Remove the baseline-prompt-only line in `src/helix/optimize/prompts.py`:

```python
if output_path is not None:
    context_lines.append(f"Requested output: {_display_path(output_path)}")
```

Keep the remaining baseline context unchanged.

- [ ] **Step 3: Run the targeted tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_cli.CliPromptTests tests.test_optimize_runtime.OptimizeRuntimeTests.test_multi_invocation_controller_baseline_prompt_uses_explicit_context_parameters -v
```

Expected: PASS

- [ ] **Step 4: Run a slightly broader optimize prompt regression slice**

Run:

```bash
uv run python -m unittest tests.test_cli.CliPromptTests tests.test_optimize_runtime.OptimizeRuntimeTests tests.test_backends_base.SharedRunnerBaseTests -v
```

Expected: PASS
