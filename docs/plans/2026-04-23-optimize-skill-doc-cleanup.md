# Optimize Skill Doc Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up `skills/triton-npu-optimize/SKILL.md` so it declares the layered-analysis model before the detailed layers, removes repeated `compare-perf` guidance, and consolidates `learned_lessons.md` rules without changing optimize behavior.

**Architecture:** Treat this as a documentation-contract refactor only. Update the optimize skill text to improve structure and deduplicate repeated guidance, then tighten the single doc-contract test that reads the skill so the new structure and non-redundancy are protected.

**Tech Stack:** Markdown skill contracts, Python `unittest`

---

## File Map

- Modify: `skills/triton-npu-optimize/SKILL.md`
- Modify: `tests/test_generation_contracts.py`

No runtime, prompt, CLI, or artifact-contract files should change for this task.

### Task 1: Rewrite The Optimize Skill Contract For Clarity And Deduplication

**Files:**
- Modify: `skills/triton-npu-optimize/SKILL.md`
- Test: `tests/test_generation_contracts.py`

- [ ] **Step 1: Write the failing doc-contract test**

Add or update one test in `tests/test_generation_contracts.py` to assert the cleaned-up structure directly:

```python
    def test_optimize_skill_declares_layered_analysis_and_deduplicates_compare_perf_and_lessons(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")

        self.assertIn("Optimize analysis is layered.", optimize)
        self.assertIn(
            "Default escalation order: `pattern triage -> profiling diagnosis -> IR attribution -> compiler-source escalation`.",
            optimize,
        )
        self.assertIn(
            "Start each round at the shallowest level that can justify the next move.",
            optimize,
        )
        self.assertEqual(
            optimize.count("use the `triton-npu-run-eval` skill to run `compare-perf`"),
            1,
        )
        self.assertEqual(optimize.count("Maintain `learned_lessons.md`"), 1)
        self.assertIn("Admission criteria:", optimize)
        self.assertIn("Put round-local narrative", optimize)
```

- [ ] **Step 2: Run the targeted test and confirm it fails first**

Run: `uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_optimize_skill_declares_layered_analysis_and_deduplicates_compare_perf_and_lessons -v`

Expected: `FAIL` because the current skill does not yet contain the new layered-analysis summary wording, still repeats the `compare-perf` line, and still spreads `learned_lessons.md` guidance across multiple sections.

- [ ] **Step 3: Rewrite `skills/triton-npu-optimize/SKILL.md`**

Update the layered-analysis section so the general model appears before the per-layer subsections:

```md
## Stage 2: Layered Analysis

Optimize analysis is layered.

- Default escalation order: `pattern triage -> profiling diagnosis -> IR attribution -> compiler-source escalation`.
- Start each round at the shallowest level that can justify the next move.
- Escalate only when the current level is insufficient.
- Record the chosen level and why the round stayed there or escalated deeper.

### pattern triage
...
```

Reduce `compare-perf` guidance in `## Stage 3: Validate And Record` to one action line plus one authority line:

```md
- Once baseline and round perf artifacts both exist, use the `triton-npu-run-eval` skill to run `compare-perf`.
- Treat `compare-perf` as the only authority for claimed benchmark deltas and speedups.
- Do not hand-calculate speedups or percentage improvements from raw perf files.
```

Consolidate `learned_lessons.md` guidance into one contract block and remove overlapping repeats from `Outputs`, `Round Records`, and `Hard Rules`:

```md
## Learned Lessons

Maintain `learned_lessons.md` as a strict reusable optimization-knowledge distillation log.

Admission criteria:

- ...

Good entries include:

- ...

Put round-local narrative, temporary troubleshooting notes, command failures, and shape-specific details in `attempts.md`, `summary.md`, or `opt-note.md` instead.
```

Keep only one lesson-specific hard rule if it adds something not already stated above; otherwise delete the duplicated hard-rule lines.

- [ ] **Step 4: Re-run the targeted doc-contract test**

Run: `uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_optimize_skill_declares_layered_analysis_and_deduplicates_compare_perf_and_lessons -v`

Expected: `PASS`

- [ ] **Step 5: Run the full doc-contract suite**

Run: `uv run python -m unittest tests.test_generation_contracts -v`

Expected: all generation-contract tests pass, confirming the optimize skill cleanup did not regress the existing optimize documentation contract.

- [ ] **Step 6: Commit the documentation cleanup**

```bash
git add skills/triton-npu-optimize/SKILL.md tests/test_generation_contracts.py
git commit -m "docs: tighten optimize skill guidance"
```
