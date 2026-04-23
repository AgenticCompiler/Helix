# Optimize Skill Contract Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `skills/triton-npu-optimize/SKILL.md` the sole optimize workflow contract, delete `references/workflow.md`, and update the optimize doc-contract tests to match the new single-source structure.

**Architecture:** This is a documentation-contract refactor, not a runtime behavior change. Rewrite the optimize doc tests first so they stop reading `references/workflow.md` and instead assert the new stage-based `SKILL.md` structure, then rewrite `SKILL.md` around the approved stage model and delete `workflow.md`, and finally verify there are no remaining live references to the deleted file before finishing.

**Tech Stack:** Markdown skill docs, Python `unittest`, `uv`, `rg`, `git`

---

### Task 1: Rewrite The Optimize Doc-Contract Tests Around A Single Workflow Source

**Files:**
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Replace the workflow-file assertions with single-source `SKILL.md` assertions**

Update the optimize-specific contract tests in `tests/test_generation_contracts.py` so they no longer call `_read("skills/triton-npu-optimize/references/workflow.md")`.

Replace the existing workflow-backed assertions with code shaped like:

```python
    def test_optimize_skills_document_compiler_source_escalation(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        round_analysis = _read("skills/triton-npu-analyze-round-performance/SKILL.md")

        self.assertIn("compiler-source escalation", optimize)
        self.assertIn("triton-npu-analyze-compiler-source", optimize)
        self.assertIn("after profiler and IR evidence", optimize)
        self.assertIn("opt-round-N/compiler-analysis.md", optimize)
        self.assertIn("compiler source analysis is enabled", round_analysis)
        self.assertFalse(
            (REPO_ROOT / "skills/triton-npu-optimize/references/workflow.md").exists()
        )

    def test_optimize_docs_keep_opt_note_round_only_and_put_initial_hypothesis_in_attempts(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        opt_note = _read("skills/triton-npu-optimize/references/opt-note-format.md")
        artifacts = _read("skills/triton-npu-optimize/references/artifacts.md")

        self.assertIn("opt-note.md", optimize)
        self.assertIn("top-level round ledger plus final `## Overall Summary`", optimize)
        self.assertIn(
            "For round 1, record the initial round hypothesis in `opt-round-1/attempts.md`",
            optimize,
        )
        self.assertIn("completed round records and final outcome summary", opt_note)
        self.assertIn(
            "Do not put session-start diagnosis, tentative bottleneck narrative, or other pre-round analysis above the round history",
            opt_note,
        )
        self.assertIn(
            "Do not write session-start diagnosis or tentative bottleneck narrative in `opt-note.md`",
            artifacts,
        )

    def test_optimize_docs_make_layered_analysis_default_and_remove_require_analysis_flag(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        artifacts = _read("skills/triton-npu-optimize/references/artifacts.md")
        readme = _read("README.md")

        self.assertIn("## Core Loop", optimize)
        self.assertIn("## Stage 2: Layered Analysis", optimize)
        self.assertIn("### pattern triage", optimize)
        self.assertIn("### profiling diagnosis", optimize)
        self.assertIn("### IR attribution", optimize)
        self.assertIn("### compiler-source escalation", optimize)
        self.assertIn(
            "Use profiling diagnosis as the default deeper entrypoint when pattern triage is not enough",
            optimize,
        )
        self.assertIn("the current analysis level", artifacts)
        self.assertIn("why the round stayed at that level or why it escalated deeper", artifacts)
        self.assertIn(
            "pattern triage -> profiling diagnosis -> IR attribution -> compiler-source escalation",
            readme,
        )
        self.assertNotIn("--require-analysis", readme)
```

These updated tests should make the deleted workflow file part of the contract by asserting that it does not exist anymore.

- [ ] **Step 2: Run the doc-contract suite to verify RED**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts -v
```

Expected: FAIL because `skills/triton-npu-optimize/SKILL.md` still has the old section layout and `skills/triton-npu-optimize/references/workflow.md` still exists.

### Task 2: Rewrite `SKILL.md` Around The Stage Model And Delete `workflow.md`

**Files:**
- Modify: `skills/triton-npu-optimize/SKILL.md`
- Delete: `skills/triton-npu-optimize/references/workflow.md`

- [ ] **Step 1: Replace the old optimize sections with the new stage-based structure**

Rewrite `skills/triton-npu-optimize/SKILL.md` so it no longer has these top-level sections:

- `## Required References`
- `## Pattern References`
- `## Default Analysis Ladder`
- `## Workflow`

Replace them with this markdown structure and keep the current layered-analysis behavior inside it:

```md
## Goal

Optimize one Triton Ascend NPU operator through validated rounds anchored to a canonical `baseline/`.

## Outputs

- `baseline/`
- `opt-round-N/`
- `opt-note.md`
- `learned_lessons.md`
- round-local `profile/`, `ir/`, `perf-analysis.md`, or `compiler-analysis.md` when needed

## Core Loop

- establish or reuse `baseline/`
- open `opt-round-N/` and start `attempts.md`
- choose the current analysis level
- make one coherent optimization attempt
- validate correctness and benchmark performance
- record the round outcome

## Stage 0: Baseline Setup

- Reuse existing correctness and benchmark harnesses when possible; generate only missing harnesses.
- Read [artifacts.md](references/artifacts.md) before writing `baseline/state.json`.
- Read [opt-note-format.md](references/opt-note-format.md) before initializing `opt-note.md`.
- Use `triton-npu-optimize-check check-baseline` until baseline state passes.

## Stage 1: Round Entry

- Create `opt-round-N/` from a validated parent candidate.
- Start `attempts.md` immediately.
- Record the round hypothesis and the current analysis level before the first code change.
- For round 1, keep the initial hypothesis in `opt-round-1/attempts.md`, not in `opt-note.md`.

## Stage 2: Layered Analysis

### pattern triage

- Inspect current code structure and benchmark behavior.
- Read `references/patterns/index.md`.
- Read a detailed pattern reference only when the index suggests a real match.
- Do not treat pattern triage as permission for blind pattern search.

### profiling diagnosis

- Use profiling diagnosis as the default deeper entrypoint when pattern triage is not enough.
- Use `triton-npu-profile-operator` or `triton-npu-analyze-round-performance` when profiler-backed diagnosis is needed.
- Write `opt-round-N/perf-analysis.md` when the deeper round-analysis flow is used.

### IR attribution

- Use IR only after profiler-backed symptoms still need explanation.
- Use `triton-npu-analyze-ir`.
- Keep IR evidence under `opt-round-N/ir/`.

### compiler-source escalation

- Use compiler source only when analysis is enabled and profiler plus IR evidence have already narrowed a concrete compiler-side question.
- Use `triton-npu-analyze-compiler-source`.
- Write `opt-round-N/compiler-analysis.md`.

## Stage 3: Validate And Record

- Run correctness before trusting performance.
- Run benchmark validation after correctness passes.
- Use `compare-perf` as the sole source of speedup claims.
- Run `triton-npu-optimize-check check-round` before ending or continuing a round.
- Update `summary.md`, `opt-note.md`, and `learned_lessons.md` when eligible.

## Round Records

- `attempts.md`: chronological round log
- `summary.md`: round conclusion and reusable optimization points
- `opt-note.md`: top-level round ledger plus final `## Overall Summary`
- `learned_lessons.md`: strict reusable knowledge only

## Hard Rules

- Optimize the Triton kernel path, not just the wrapper surface.
- Do not claim success without correctness and benchmark evidence.
- Do not hand-calculate speedups.
- Do not begin with blind tiling or launch-parameter search without evidence.
- Do not put round narrative into `learned_lessons.md`.
```

Keep the existing durable constraints that are still valid:

- baseline-first workflow
- round-local attempt logging
- `compare-perf` as authority
- round-local profiler / IR / compiler-analysis artifacts
- strict `learned_lessons.md` boundary
- remote-aware helper-script wording

- [ ] **Step 2: Delete the redundant workflow reference file**

Delete `skills/triton-npu-optimize/references/workflow.md` entirely.

Do not replace it with a stub, redirect file, or deprecation notice.

- [ ] **Step 3: Run the doc-contract suite again to verify GREEN**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts -v
```

Expected: PASS.

### Task 3: Sweep For Stale Live References And Re-Verify The Repo

**Files:**
- Verify: `skills/triton-npu-optimize/SKILL.md`
- Verify: `skills/triton-npu-optimize/references/artifacts.md`
- Verify: `skills/triton-npu-optimize/references/opt-note-format.md`
- Verify: `README.md`
- Verify: `src/`
- Verify: `tests/`

- [ ] **Step 1: Grep for any remaining live references to the deleted workflow file**

Run:

```bash
rg -n "skills/triton-npu-optimize/references/workflow.md|Read \\[workflow.md\\]|references/workflow.md" README.md skills tests src
```

Expected: no output.

If this command prints any live reference under `README.md`, `skills/`, `tests/`, or `src/`, remove that reference before continuing.

- [ ] **Step 2: Run the full test suite**

Run:

```bash
uv run python -m unittest -v
```

Expected: PASS.

- [ ] **Step 3: Commit the doc-contract simplification**

Run:

```bash
git add tests/test_generation_contracts.py skills/triton-npu-optimize/SKILL.md skills/triton-npu-optimize/references/workflow.md
git commit -m "docs: simplify optimize skill contract"
```

Expected: commit succeeds with only the doc-contract simplification changes staged for this task.
