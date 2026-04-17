# Optimize Triton Kernel Continuity Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten optimize prompt wording so worker and supervisor roles treat pure PyTorch substitution as an invalid optimize outcome while still allowing a PyTorch-facing wrapper API.

**Architecture:** Keep the first rollout prompt-only. Add failing prompt tests first, then update optimize worker, unsupervised, and supervisor prompt builders in `src/triton_agent/prompts.py` with explicit Triton-kernel continuity language. Do not change optimize contracts, metadata, or runtime gate logic in this iteration.

**Tech Stack:** Python, existing optimize prompt helpers, Python `unittest`

---

### Task 1: Add Prompt Regression Tests For Kernel Continuity Policy

**Files:**
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_optimize_runtime.py`
- Test: `tests/test_optimize_guidance.py`
- Test: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write failing prompt tests for worker and unsupervised optimize policy**

```python
def test_worker_prompt_forbids_pure_pytorch_substitution(self) -> None:
    prompt = build_optimize_worker_prompt(
        Path("kernel.py"),
        Path("opt_kernel.py"),
        test_mode="differential",
        bench_mode="standalone",
    )

    self.assertIn("PyTorch-facing public API may remain as a wrapper", prompt)
    self.assertIn("must continue optimizing the Triton Ascend NPU kernel path", prompt)
    self.assertIn("Do not replace the core computation with a pure PyTorch implementation", prompt)


def test_unsupervised_prompt_forbids_pure_pytorch_substitution(self) -> None:
    prompt = build_optimize_unsupervised_prompt(
        Path("kernel.py"),
        Path("opt_kernel.py"),
        test_mode="differential",
        bench_mode="standalone",
    )

    self.assertIn("PyTorch-facing public API may remain as a wrapper", prompt)
    self.assertIn("Do not replace the core computation with a pure PyTorch implementation", prompt)
```

- [ ] **Step 2: Write a failing supervisor audit prompt test**

```python
def test_supervisor_prompt_rejects_pure_pytorch_substitution(self) -> None:
    prompt = build_optimize_supervisor_prompt(Path("/tmp/workdir"))

    self.assertIn("reject rounds that replace the Triton kernel path with pure PyTorch computation", prompt)
```

- [ ] **Step 3: Run the targeted prompt tests to verify they fail**

Run: `uv run python -m unittest tests.test_optimize_guidance tests.test_optimize_runtime -v`
Expected: FAIL because the current optimize prompts do not mention Triton-kernel continuity or pure PyTorch substitution.

- [ ] **Step 4: Commit the failing-test checkpoint if desired**

```bash
git add tests/test_optimize_guidance.py tests/test_optimize_runtime.py
git commit -m "test: cover optimize kernel continuity prompt policy"
```

### Task 2: Add Kernel Continuity Language To Optimize Prompt Builders

**Files:**
- Modify: `src/triton_agent/prompts.py`
- Test: `tests/test_optimize_guidance.py`
- Test: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Add minimal worker and unsupervised prompt wording**

```python
lines.extend(
    [
        "A PyTorch-facing public API may remain as a wrapper when that is the intended operator entrypoint.",
        "You must continue optimizing the Triton Ascend NPU kernel path itself.",
        "Do not replace the core computation with a pure PyTorch implementation just to improve final outputs or benchmark numbers.",
        "A round that bypasses the Triton kernel with pure PyTorch code does not count as a successful optimize round.",
    ]
)
```

- [ ] **Step 2: Add minimal supervisor audit wording**

```python
lines.append(
    "Reject rounds that preserve only the public API shape but replace the Triton kernel path with pure PyTorch computation."
)
```

- [ ] **Step 3: Re-run the targeted prompt tests**

Run: `uv run python -m unittest tests.test_optimize_guidance tests.test_optimize_runtime -v`
Expected: PASS

- [ ] **Step 4: Review surrounding prompt wording for clarity only**

```python
# Keep existing baseline, artifact, evidence, and compare-perf wording intact.
# Do not add contract or runtime logic in this task.
```

- [ ] **Step 5: Commit the prompt-builder change**

```bash
git add src/triton_agent/prompts.py tests/test_optimize_guidance.py tests/test_optimize_runtime.py
git commit -m "fix: forbid pure pytorch optimize substitutions in prompts"
```

### Task 3: Run Focused Verification And Record Follow-Up Boundary

**Files:**
- Modify: `docs/specs/2026-04-15-optimize-triton-kernel-continuity-prompt-design.md`
- Modify: `docs/plans/2026-04-15-optimize-triton-kernel-continuity-prompt.md`

- [ ] **Step 1: Run the focused verification suite**

Run: `uv run python -m unittest tests.test_optimize_guidance tests.test_optimize_runtime -v`
Expected: PASS

- [ ] **Step 2: Sanity-check that no broader optimize tests regressed**

Run: `uv run python -m unittest tests.test_supervisor tests.test_cli -v`
Expected: PASS

- [ ] **Step 3: Record any follow-up if prompt-only mitigation appears insufficient later**

```markdown
Future escalation path:
- add canonical kernel identity to baseline metadata
- add round continuity metadata
- make triton-npu-optimize-check reject pure PyTorch substitution
```

- [ ] **Step 4: Commit the verification checkpoint**

```bash
git add docs/specs/2026-04-15-optimize-triton-kernel-continuity-prompt-design.md docs/plans/2026-04-15-optimize-triton-kernel-continuity-prompt.md
git commit -m "docs: plan optimize kernel continuity prompt tightening"
```
