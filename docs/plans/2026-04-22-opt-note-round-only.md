# Opt-Note Round-Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `opt-note.md` a round-only ledger with one final `## Overall Summary`, and move the initial optimize hypothesis into `opt-round-1/attempts.md`.

**Architecture:** Keep the change in the optimize skill contract only. Lock the new artifact boundary with one regression test in `tests/test_generation_contracts.py`, then update the optimize skill, workflow reference, opt-note format reference, and artifact contract so they all describe the same round-local hypothesis model.

**Tech Stack:** Markdown workflow docs, Python `unittest`

---

### Task 1: Add A Regression Test For The New Artifact Boundary

**Files:**
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Add a focused optimize-doc contract test**

Insert a new test near the other optimize-skill contract checks:

```python
    def test_optimize_docs_keep_opt_note_round_only_and_put_initial_hypothesis_in_attempts(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        workflow = _read("skills/triton-npu-optimize/references/workflow.md")
        opt_note = _read("skills/triton-npu-optimize/references/opt-note-format.md")
        artifacts = _read("skills/triton-npu-optimize/references/artifacts.md")

        self.assertIn("completed round records", opt_note)
        self.assertIn("one final `## Overall Summary`", opt_note)
        self.assertIn("Record the initial round hypothesis in `opt-round-1/attempts.md`", optimize)
        self.assertIn("For round 1, record the starting hypothesis in `opt-round-1/attempts.md`", workflow)
        self.assertIn("Do not write session-start diagnosis or tentative bottleneck narrative in `opt-note.md`", artifacts)
        self.assertNotIn("Record a short diagnosis before the first code-changing round", optimize)
        self.assertNotIn("Write a short diagnosis summary before the first code-changing round", workflow)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_optimize_docs_keep_opt_note_round_only_and_put_initial_hypothesis_in_attempts -v
```

Expected: FAIL because the current optimize docs still tell the agent to write a pre-round diagnosis and do not yet describe `opt-note.md` as round-only.

### Task 2: Rewrite The Optimize Skill And Workflow Steps

**Files:**
- Modify: `skills/triton-npu-optimize/SKILL.md`
- Modify: `skills/triton-npu-optimize/references/workflow.md`

- [ ] **Step 1: Tighten the optimize skill output contract**

Change the `Outputs` section in `skills/triton-npu-optimize/SKILL.md` so the `opt-note.md` bullet reads like this:

```md
- Updated `opt-note.md` in the operator workspace with completed round entries and one final `## Overall Summary`
```

- [ ] **Step 2: Replace the pre-round diagnosis instruction in the skill workflow**

Replace the current workflow step that says:

```md
10. Record a short diagnosis before the first code-changing round. The diagnosis should name the suspected bottleneck, the current evidence, and what kind of optimization direction looks justified.
```

with wording like:

```md
10. Create `opt-round-N/`, copy the chosen parent operator into it, and start `attempts.md` immediately so every meaningful attempt and measurement is recorded.
11. For round 1, record the initial round hypothesis in `opt-round-1/attempts.md` before the first code change. State why it may help, what evidence supports starting there, and why profiling or IR capture is being skipped when those tools are not used yet.
12. Before editing code for any round, state the optimization hypothesis for the round, explain why it may help, and cite the supporting evidence.
```

Keep the surrounding workflow logic intact; this is a wording and artifact-boundary change, not a new runtime workflow.

- [ ] **Step 3: Rewrite the pre-round setup section in `workflow.md`**

Replace the current step:

```md
13. Write a short diagnosis summary before the first code-changing round so later readers can see the suspected bottleneck and the initial evidence.
```

with:

```md
13. Treat `opt-note.md` as the top-level round ledger plus one final `## Overall Summary`; do not write session-start diagnosis or tentative bottleneck narrative there.
```

- [ ] **Step 4: Strengthen the round-lifecycle wording for round 1**

Update the `Round Lifecycle` section in `workflow.md` so step 6 explicitly says:

```md
6. For round 1, record the starting hypothesis in `opt-round-1/attempts.md`; for later rounds, record the initial round hypothesis in that round's `attempts.md`, including why it may help and what evidence supports it.
```

Leave the rest of the round-lifecycle flow unchanged.

- [ ] **Step 5: Run the optimize-doc contract test again**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_optimize_docs_keep_opt_note_round_only_and_put_initial_hypothesis_in_attempts -v
```

Expected: still FAIL, because `opt-note-format.md` and `artifacts.md` have not been updated yet.

### Task 3: Make The Reference Docs Match The New Boundary

**Files:**
- Modify: `skills/triton-npu-optimize/references/opt-note-format.md`
- Modify: `skills/triton-npu-optimize/references/artifacts.md`

- [ ] **Step 1: Rewrite the `opt-note.md` purpose section**

Update `skills/triton-npu-optimize/references/opt-note-format.md` so the opening guidance says `opt-note.md` is the top-level running log for completed round records plus one final `## Overall Summary`.

Use wording like:

```md
`opt-note.md` is the top-level running log for the optimization session's completed round records and final outcome summary.
```

- [ ] **Step 2: Add an explicit ban on pre-round diagnosis prose**

Add an `Entry Rules` bullet to `opt-note-format.md`:

```md
- Do not put session-start diagnosis, tentative bottleneck narrative, or other pre-round analysis above the round history; keep that reasoning in round-local artifacts such as `opt-round-N/attempts.md`.
```

- [ ] **Step 3: Update the writing guidance for initial hypotheses**

Add a `Writing Guidance` bullet to `opt-note-format.md`:

```md
- Put initial hypotheses, evolving reasoning, and diagnosis pivots in `opt-round-N/attempts.md`, `summary.md`, or `perf-analysis.md`, not in the top-level note.
```

- [ ] **Step 4: Add a top-level `opt-note.md` artifact boundary to `artifacts.md`**

Insert a short section after `## Workspace Layout` in `skills/triton-npu-optimize/references/artifacts.md`:

```md
## Top-Level Session Note

`opt-note.md` is the top-level ledger for completed round entries and one final `## Overall Summary`.

Do not write session-start diagnosis or tentative bottleneck narrative in `opt-note.md`.

For round 1, record the starting hypothesis in `opt-round-1/attempts.md`. For later rounds, keep the initial hypothesis in that round's `attempts.md`.
```

- [ ] **Step 5: Run the generation-contract test suite**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts -v
```

Expected: PASS.

### Task 4: Final Verification And Review

**Files:**
- Modify: `tests/test_generation_contracts.py`
- Modify: `skills/triton-npu-optimize/SKILL.md`
- Modify: `skills/triton-npu-optimize/references/workflow.md`
- Modify: `skills/triton-npu-optimize/references/opt-note-format.md`
- Modify: `skills/triton-npu-optimize/references/artifacts.md`

- [ ] **Step 1: Re-run the focused verification command**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts -v
```

Expected: PASS with the new opt-note boundary locked in.

- [ ] **Step 2: Review the final diff**

Run:

```bash
git diff -- docs/specs/2026-04-22-opt-note-round-only-design.md docs/plans/2026-04-22-opt-note-round-only.md tests/test_generation_contracts.py skills/triton-npu-optimize/SKILL.md skills/triton-npu-optimize/references/workflow.md skills/triton-npu-optimize/references/opt-note-format.md skills/triton-npu-optimize/references/artifacts.md
```

Expected: only the approved round-only `opt-note.md` contract, matching documentation updates, and the regression test change appear.
