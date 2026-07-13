# Optimize Guidance Template Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor optimize-time `AGENTS.md` / `CLAUDE.md` rendering in `guidance.py` from inline string concatenation to code-local multiline templates without changing optimize behavior.

**Architecture:** Keep all rendering logic in `src/helix/optimize/guidance.py`, but separate document structure from block assembly. Introduce small local helpers for bullet and optional line blocks, add two explicit template constants for unsupervised and shared guidance, then rewire the render methods to fill those templates while preserving the current text semantics and test coverage.

**Tech Stack:** Python 3.12, `textwrap.dedent`, existing optimize guidance plumbing, Python `unittest`

---

### Task 1: Add Focused Rendering Helpers With Tests

**Files:**
- Modify: `tests/test_optimize_guidance.py`
- Modify: `src/helix/optimize/guidance.py`

- [ ] **Step 1: Write failing helper tests**

Add these tests near the top of `OptimizeGuidanceManagerTests` in `tests/test_optimize_guidance.py`:

```python
import helix.optimize.guidance as optimize_guidance


def test_render_bullet_block_formats_markdown_list(self) -> None:
    rendered = optimize_guidance._render_bullet_block(
        [
            "Read files cautiously.",
            "Follow the user's instructions strictly.",
        ]
    )

    self.assertEqual(
        rendered,
        "- Read files cautiously.\n"
        "- Follow the user's instructions strictly.\n",
    )


def test_render_line_block_joins_lines_and_omits_empty_block(self) -> None:
    self.assertEqual(
        optimize_guidance._render_line_block(
            [
                "Compiler source analysis is enabled for this optimize run.",
                "Treat the compiler source checkout as read-only.",
            ]
        ),
        "Compiler source analysis is enabled for this optimize run.\n"
        "Treat the compiler source checkout as read-only.\n",
    )
    self.assertEqual(optimize_guidance._render_line_block([]), "")
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `python3 -m unittest tests.test_optimize_guidance.OptimizeGuidanceManagerTests.test_render_bullet_block_formats_markdown_list tests.test_optimize_guidance.OptimizeGuidanceManagerTests.test_render_line_block_joins_lines_and_omits_empty_block -v`

Expected: FAIL with `AttributeError` because `_render_bullet_block` and `_render_line_block` do not exist yet.

- [ ] **Step 3: Add minimal rendering helpers**

In `src/helix/optimize/guidance.py`, add:

```python
def _render_bullet_block(lines: list[str]) -> str:
    return "".join(f"- {line}\n" for line in lines)


def _render_line_block(lines: list[str]) -> str:
    if not lines:
        return ""
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Re-run targeted tests to verify they pass**

Run: `python3 -m unittest tests.test_optimize_guidance.OptimizeGuidanceManagerTests.test_render_bullet_block_formats_markdown_list tests.test_optimize_guidance.OptimizeGuidanceManagerTests.test_render_line_block_joins_lines_and_omits_empty_block -v`

Expected: PASS for both helper tests.

- [ ] **Step 5: Commit helper groundwork**

```bash
git add tests/test_optimize_guidance.py src/helix/optimize/guidance.py
git commit -m "refactor: add optimize guidance render helpers"
```

### Task 2: Introduce Explicit Guidance Templates And Rewire Renderers

**Files:**
- Modify: `tests/test_optimize_guidance.py`
- Modify: `src/helix/optimize/guidance.py`

- [ ] **Step 1: Write failing tests for template constants**

Add these tests in `tests/test_optimize_guidance.py`:

```python
def test_unsupervised_guidance_template_exposes_named_placeholders(self) -> None:
    template = optimize_guidance._UNSUPERVISED_GUIDANCE_TEMPLATE

    self.assertIn("{guidance_filename}", template)
    self.assertIn("{guidance_rules_block}", template)
    self.assertIn("{test_mode}", template)
    self.assertIn("{bench_mode}", template)
    self.assertIn("{operator_name}", template)
    self.assertIn("{analysis_block}", template)
    self.assertIn("{compiler_source_block}", template)


def test_shared_guidance_template_exposes_named_placeholders(self) -> None:
    template = optimize_guidance._SHARED_GUIDANCE_TEMPLATE

    self.assertIn("{guidance_filename}", template)
    self.assertIn("{guidance_rules_block}", template)
    self.assertIn("{analysis_block}", template)
    self.assertIn("{compiler_source_block}", template)
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `python3 -m unittest tests.test_optimize_guidance.OptimizeGuidanceManagerTests.test_unsupervised_guidance_template_exposes_named_placeholders tests.test_optimize_guidance.OptimizeGuidanceManagerTests.test_shared_guidance_template_exposes_named_placeholders -v`

Expected: FAIL with `AttributeError` because the explicit template constants do not exist yet.

- [ ] **Step 3: Add multiline template constants**

In `src/helix/optimize/guidance.py`, import `dedent` and add:

```python
from textwrap import dedent


_UNSUPERVISED_GUIDANCE_TEMPLATE = dedent(
    """\
    # {guidance_filename}

    ## Helix Optimize Session

    This workspace is under an unsupervised optimize run.

    {guidance_rules_block}Own the end-to-end optimize session.
    Use the staged `triton-npu-optimize` skill as the workflow source of truth.
    Use `{test_mode}` correctness validation for this optimize session.
    Use `{bench_mode}` benchmark validation for this optimize session.
    Optimize the operator at `{operator_name}`.
    {analysis_block}{compiler_source_block}"""
)


_SHARED_GUIDANCE_TEMPLATE = dedent(
    """\
    # {guidance_filename}

    ## Helix Optimize Orchestration

    This workspace is under optimize orchestration.

    {guidance_rules_block}Use the staged workspace skills as the workflow source of truth.
    Role-specific behavior comes from the launch prompt.
    Use `.helix/round-brief.md` and `.helix/supervisor-report.md` as live handoff files.
    Treat `baseline/` as the canonical optimize baseline.
    Use `compare-perf` as the authoritative source for round performance summaries.
    {analysis_block}{compiler_source_block}"""
)
```

- [ ] **Step 4: Refactor the render methods to fill templates**

Replace the current string concatenation in `OptimizeGuidanceManager._render_unsupervised_guidance()` and `OptimizeGuidanceManager._render_shared_guidance()` with `.format(...)` over the new constants:

```python
def _render_unsupervised_guidance(... ) -> str:
    return _UNSUPERVISED_GUIDANCE_TEMPLATE.format(
        guidance_filename=guidance_filename,
        guidance_rules_block=_render_bullet_block(optimize_guidance_rule_lines()),
        test_mode=test_mode,
        bench_mode=bench_mode,
        operator_name=operator_path.name,
        analysis_block=_render_bullet_block(
            layered_analysis_lines(round_scope="each round")
        ),
        compiler_source_block=_render_line_block(
            compiler_source_analysis_lines(
                compiler_source_path=compiler_source_path,
                compiler_source_commit=compiler_source_commit,
            )
        ),
    )
```

Use the same pattern for `_render_shared_guidance()`, but omit the operator and mode placeholders.

- [ ] **Step 5: Run the focused guidance regression suite**

Run: `python3 -m unittest tests.test_optimize_guidance -v`

Expected: PASS. Existing guidance-content tests should still pass, proving the template refactor preserved optimize guidance semantics.

- [ ] **Step 6: Commit the template refactor**

```bash
git add tests/test_optimize_guidance.py src/helix/optimize/guidance.py
git commit -m "refactor: template optimize guidance rendering"
```

### Task 3: Verify Runtime Integration Still Sees The Same Guidance Files

**Files:**
- Modify: `tests/test_optimize_runtime.py` only if a runtime assertion needs updating
- Modify: `src/helix/optimize/guidance.py` only if the runtime test reveals spacing or content regressions

- [ ] **Step 1: Run runtime regression tests without changing code first**

Run: `python3 -m unittest tests.test_optimize_runtime -v`

Expected: PASS. The optimize runtime should still create and clean up `AGENTS.md` correctly and should still expose the same guidance semantics during the run.

- [ ] **Step 2: Fix only genuine runtime regressions if the suite fails**

If the runtime suite fails, make the smallest possible adjustment in `src/helix/optimize/guidance.py`. Keep the template structure and preserve the approved guidance text. Do not change optimize execution flow, archive behavior, or prompt semantics.

- [ ] **Step 3: Run the final verification set**

Run:

```bash
python3 -m unittest tests.test_optimize_guidance -v
python3 -m unittest tests.test_optimize_runtime -v
```

Expected: PASS for both suites with no new failures.

- [ ] **Step 4: Commit the verified refactor**

```bash
git add tests/test_optimize_guidance.py src/helix/optimize/guidance.py
git commit -m "test: verify optimize guidance template refactor"
```

## Self-Review

- Spec coverage: Task 1 introduces the local rendering helpers from the spec, Task 2 adds the explicit multiline templates and rewires both optimize guidance renderers, and Task 3 verifies that runtime-facing `AGENTS.md` / `CLAUDE.md` behavior is unchanged.
- Placeholder scan: No `TODO`, `TBD`, or implicit "add tests later" steps remain.
- Type consistency: All new helpers accept `list[str]`, both templates use named `.format(...)` placeholders, and both render methods keep their current public signatures.
