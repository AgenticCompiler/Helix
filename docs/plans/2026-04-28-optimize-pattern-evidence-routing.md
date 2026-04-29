# Optimize Pattern Evidence Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first evidence-routing layer for optimize by making pattern Markdown the authoring source of truth, generating the pattern semantic index from those files, adding a small symptom-routing reference set, and introducing a code-fact extractor that optimize guidance can use during pattern triage.

**Architecture:** Keep the CLI thin and keep the agent as the final decision-maker. Implement one generator script under `skills/triton-npu-optimize/scripts/` that parses predefined Markdown sections and rewrites `references/patterns/index.md`, one code-fact extractor script that emits non-diagnostic structured evidence, a new symptom reference subtree under `triton-npu-analyze-round-performance`, and narrow skill/prompt/test updates that teach the new read order without introducing a new standalone routing skill.

**Tech Stack:** Python 3.11, `unittest`, `ast`, `json`, `argparse`, Markdown skill docs, existing optimize prompt/guidance tests

---

### Task 1: Pin the pattern-card contract and generator behavior in failing tests

**Files:**
- Create: `tests/test_optimize_pattern_tools.py`
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_guidance.py`

- [ ] **Step 1: Add a failing parser test for the required and optional pattern sections**

```python
def test_build_index_requires_summary_and_use_when(self) -> None:
    module = _load_skill_script(
        "skills/triton-npu-optimize/scripts/build_pattern_index.py"
    )
    with tempfile.TemporaryDirectory() as tmp:
        patterns_dir = Path(tmp)
        (patterns_dir / "broken.md").write_text(
            "# Broken Pattern\n\n## Summary\n\nMissing use-when.\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Use When"):
            module.build_index_text(patterns_dir)


def test_build_index_keeps_free_sections_but_ignores_them_for_summary(self) -> None:
    module = _load_skill_script(
        "skills/triton-npu-optimize/scripts/build_pattern_index.py"
    )
    with tempfile.TemporaryDirectory() as tmp:
        patterns_dir = Path(tmp)
        (patterns_dir / "demo.md").write_text(
            """---
id: demo
title: Demo Pattern
---

## Summary

Short summary.

## Use When

- A stable trigger exists.

## Background

Extra prose that should stay in the source file but not become a first-line index field.
""",
            encoding="utf-8",
        )
        rendered = module.build_index_text(patterns_dir)
        self.assertIn("demo", rendered)
        self.assertIn("Short summary.", rendered)
        self.assertNotIn("Extra prose", rendered)
```

- [ ] **Step 2: Add a failing repo-consistency test for the checked-in pattern index**

```python
def test_checked_in_pattern_index_matches_generator(self) -> None:
    module = _load_skill_script(
        "skills/triton-npu-optimize/scripts/build_pattern_index.py"
    )
    patterns_dir = (
        REPO_ROOT / "skills" / "triton-npu-optimize" / "references" / "patterns"
    )
    generated = module.build_index_text(patterns_dir)
    checked_in = (patterns_dir / "index.md").read_text(encoding="utf-8")
    self.assertEqual(generated, checked_in)
```

- [ ] **Step 3: Extend skill-contract tests to require the new pattern-card contract and routing references**

```python
def test_optimize_pattern_cards_use_required_sections_and_generated_index(self) -> None:
    patterns_dir = REPO_ROOT / "skills" / "triton-npu-optimize" / "references" / "patterns"
    for path in sorted(patterns_dir.glob("*.md")):
        if path.name == "index.md":
            continue
        content = path.read_text(encoding="utf-8")
        with self.subTest(path=path.name):
            self.assertIn("## Summary", content)
            self.assertIn("## Use When", content)

    optimize = _read("skills/triton-npu-optimize/SKILL.md")
    self.assertIn("generated `references/patterns/index.md`", optimize)
    self.assertIn("extract_code_facts.py", optimize)
```

- [ ] **Step 4: Extend prompt and guidance tests for the new triage read order**

```python
self.assertIn("Read the generated `references/patterns/index.md` before detailed pattern references.", prompt)
self.assertIn("Use the staged code-fact extractor when code structure is still unclear at pattern triage.", prompt)
self.assertIn("Use symptom cards to narrow pattern candidates after structured profiler or IR evidence exists.", prompt)
```

- [ ] **Step 5: Run the targeted tests to confirm they fail**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_pattern_tools \
  tests.test_generation_contracts \
  tests.test_optimize_guidance \
  tests.test_cli -v
```

Expected: FAIL because the generator script does not exist yet, the pattern files do not yet follow the new section contract, and optimize guidance does not yet mention the new routing order.

### Task 2: Implement the pattern-index generator script

**Files:**
- Create: `skills/triton-npu-optimize/scripts/build_pattern_index.py`
- Modify: `tests/test_optimize_pattern_tools.py`

- [ ] **Step 1: Create the script with a parser and deterministic index renderer**

```python
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


REQUIRED_SECTIONS = ("Summary", "Use When")
OPTIONAL_SECTIONS = (
    "Avoid When",
    "Signals",
    "Related Patterns",
    "What To Verify After Applying",
)


@dataclass
class PatternCard:
    identifier: str
    title: str
    summary: str
    use_when: list[str]
    avoid_when: list[str]
    signals_code: list[str]
    signals_profile: list[str]
    signals_ir: list[str]
    related_patterns: list[str]
    verify_after_applying: list[str]
    source_path: Path


def build_index_text(patterns_dir: Path) -> str:
    cards = [
        parse_pattern_card(path)
        for path in sorted(patterns_dir.glob("*.md"))
        if path.name != "index.md"
    ]
    return render_index(cards)
```

- [ ] **Step 2: Add a CLI that can rewrite the checked-in index or run in check mode**

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    rendered = build_index_text(Path(args.patterns_dir))
    output_path = Path(args.output)
    if args.check:
        current = output_path.read_text(encoding="utf-8")
        if current != rendered:
            print(f"Pattern index is out of date: {output_path}")
            return 1
        return 0

    output_path.write_text(rendered, encoding="utf-8")
    return 0
```

- [ ] **Step 3: Run the focused parser tests and strict pyright check**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_build_index_requires_summary_and_use_when \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_build_index_keeps_free_sections_but_ignores_them_for_summary -v
```

Expected: PASS

Run:

```bash
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize/scripts/build_pattern_index.py
```

Expected: `0 errors, 0 warnings, 0 informations`

### Task 3: Rewrite the optimize pattern library to the new section contract and regenerate the semantic index

**Files:**
- Modify: `skills/triton-npu-optimize/references/patterns/autotune.md`
- Modify: `skills/triton-npu-optimize/references/patterns/cache_use.md`
- Modify: `skills/triton-npu-optimize/references/patterns/classic-matmul.md`
- Modify: `skills/triton-npu-optimize/references/patterns/compile_hint.md`
- Modify: `skills/triton-npu-optimize/references/patterns/diagonal.md`
- Modify: `skills/triton-npu-optimize/references/patterns/discrete_memory_access.md`
- Modify: `skills/triton-npu-optimize/references/patterns/gather-load.md`
- Modify: `skills/triton-npu-optimize/references/patterns/parallel.md`
- Modify: `skills/triton-npu-optimize/references/patterns/program-multiple-rows.md`
- Modify: `skills/triton-npu-optimize/references/patterns/reorder-load.md`
- Modify: `skills/triton-npu-optimize/references/patterns/slice_coalesce.md`
- Modify: `skills/triton-npu-optimize/references/patterns/slice_intermediate.md`
- Modify: `skills/triton-npu-optimize/references/patterns/software-pipeline.md`
- Modify: `skills/triton-npu-optimize/references/patterns/tiling.md`
- Modify: `skills/triton-npu-optimize/references/patterns/vec-cmp.md`
- Modify: `skills/triton-npu-optimize/references/patterns/index.md`
- Test: `tests/test_optimize_pattern_tools.py`
- Test: `tests/test_generation_contracts.py`

- [ ] **Step 1: Normalize the high-traffic pattern cards first**

Use this structure for the first batch:

```markdown
## Summary

One short paragraph describing the pattern.

## Use When

- Concrete trigger 1
- Concrete trigger 2

## Avoid When

- Concrete anti-trigger 1

## Signals

### Code

- Code-shape clue

### Profile

- Structured profiler clue

### IR

- Stage or lowering clue

## What To Verify After Applying

- Correctness or performance risk to re-check

## Related Patterns

- `other-pattern`: short boundary note
```

Apply this first to:

- `classic-matmul.md`
- `software-pipeline.md`
- `tiling.md`
- `reorder-load.md`
- `autotune.md`
- `cache_use.md`

- [ ] **Step 2: Normalize the remaining pattern cards**

Bring the rest of the library to the same minimum contract while preserving free-form explanatory material:

- `compile_hint.md`
- `diagonal.md`
- `discrete_memory_access.md`
- `gather-load.md`
- `parallel.md`
- `program-multiple-rows.md`
- `slice_coalesce.md`
- `slice_intermediate.md`
- `vec-cmp.md`

Use this shorter minimum shape where a pattern does not need all optional sections:

```markdown
## Summary

Short statement of the pattern's role.

## Use When

- Main trigger

## Related Patterns

- `neighbor-pattern`: boundary note
```

- [ ] **Step 3: Regenerate the checked-in semantic index from the rewritten pattern files**

Run:

```bash
python3 skills/triton-npu-optimize/scripts/build_pattern_index.py \
  --patterns-dir skills/triton-npu-optimize/references/patterns \
  --output skills/triton-npu-optimize/references/patterns/index.md
```

Expected: `skills/triton-npu-optimize/references/patterns/index.md` is rewritten deterministically from the pattern cards.

- [ ] **Step 4: Run the repo-consistency and contract tests**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_checked_in_pattern_index_matches_generator \
  tests.test_generation_contracts.GenerationContractTests.test_optimize_pattern_cards_use_required_sections_and_generated_index \
  tests.test_generation_contracts.GenerationContractTests.test_optimize_pattern_library_includes_classic_tiled_matmul -v
```

Expected: PASS

### Task 4: Add the symptom routing references and wire them into the round-analysis skill

**Files:**
- Create: `skills/triton-npu-analyze-round-performance/references/symptoms/index.md`
- Create: `skills/triton-npu-analyze-round-performance/references/symptoms/high-scalar-overhead.md`
- Create: `skills/triton-npu-analyze-round-performance/references/symptoms/high-transfer-pressure.md`
- Create: `skills/triton-npu-analyze-round-performance/references/symptoms/weak-pipeline-overlap.md`
- Create: `skills/triton-npu-analyze-round-performance/references/symptoms/low-cube-utilization.md`
- Create: `skills/triton-npu-analyze-round-performance/references/symptoms/poor-locality.md`
- Create: `skills/triton-npu-analyze-round-performance/references/symptoms/under-parallelized-block-dim.md`
- Modify: `skills/triton-npu-analyze-round-performance/SKILL.md`
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Add a failing contract test for the new symptom-routing references**

```python
def test_round_performance_skill_points_to_symptom_routing_references(self) -> None:
    skill = _read("skills/triton-npu-analyze-round-performance/SKILL.md")
    symptom_index = _read(
        "skills/triton-npu-analyze-round-performance/references/symptoms/index.md"
    )
    self.assertIn("symptom cards", skill)
    self.assertIn("references/symptoms/index.md", skill)
    self.assertIn("weak-pipeline-overlap", symptom_index)
    self.assertIn("high-transfer-pressure", symptom_index)
```

- [ ] **Step 2: Write the symptom index and the first six symptom cards**

Use this shape for each symptom card:

```markdown
# Weak Pipeline Overlap

## What It Usually Means

- Memory movement and compute are not overlapped well enough.

## Confirming Evidence

- High wait or weak overlap in structured profile output.
- IR still looks serial around transfer and sync boundaries.

## Common False Positives

- The loop is not yet structurally tiled, so this is really a `classic-matmul` problem first.

## What To Check Next

- Compare current loop structure against `software-pipeline` and `reorder-load`.

## Candidate Patterns

- `software-pipeline`
- `reorder-load`
```

- [ ] **Step 3: Update the round-analysis skill to use the symptom index after profile and IR extraction**

```markdown
4. Extract profile signals first.
5. Use `references/symptoms/index.md` to choose the most relevant symptom card.
6. Read only the one or two symptom cards that match the current profile or IR evidence.
7. Use those cards to narrow pattern candidates before returning to detailed optimize pattern references.
```

- [ ] **Step 4: Run the targeted contract tests**

Run:

```bash
uv run python -m unittest \
  tests.test_generation_contracts.GenerationContractTests.test_round_performance_skill_points_to_symptom_routing_references \
  tests.test_generation_contracts.GenerationContractTests.test_round_performance_skill_describes_layered_profiler_and_binary_analysis -v
```

Expected: PASS

### Task 5: Implement the code-fact extractor and teach optimize guidance how to use it

**Files:**
- Create: `skills/triton-npu-optimize/scripts/extract_code_facts.py`
- Modify: `skills/triton-npu-optimize/SKILL.md`
- Modify: `src/triton_agent/optimize/prompts.py`
- Modify: `tests/test_optimize_pattern_tools.py`
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_guidance.py`

- [ ] **Step 1: Add a failing extractor test for non-diagnostic code facts**

```python
def test_extract_code_facts_reports_manual_reduction_and_index_load(self) -> None:
    module = _load_skill_script(
        "skills/triton-npu-optimize/scripts/extract_code_facts.py"
    )
    with tempfile.TemporaryDirectory() as tmp:
        operator = Path(tmp) / "kernel.py"
        operator.write_text(
            """
import triton.language as tl

def kernel(x_ptr, idx_ptr):
    acc = 0
    for k in range(0, 128, 32):
        idx = tl.load(idx_ptr + k)
        val = tl.load(x_ptr + idx)
        acc += val
""",
            encoding="utf-8",
        )
        payload = module.extract_code_facts(operator)
        self.assertIn("manual_k_reduction", payload["facts"])
        self.assertIn("index_based_load", payload["facts"])
        self.assertNotIn("weak_pipeline_overlap", payload["facts"])
```

- [ ] **Step 2: Implement the extractor as a small AST-based fact collector**

```python
from __future__ import annotations

import ast
import json
from pathlib import Path


def extract_code_facts(operator_path: Path) -> dict[str, object]:
    tree = ast.parse(operator_path.read_text(encoding="utf-8"))
    facts: set[str] = set()
    evidence: list[dict[str, object]] = []

    # detect tl.dot absence inside manual reduction loops
    # detect tl.load calls that use computed indices
    # detect loop shapes that suggest one-row-per-program or serialized load/compute/store

    return {
        "operator_path": str(operator_path),
        "facts": sorted(facts),
        "evidence": evidence,
    }
```

- [ ] **Step 3: Update optimize skill and prompt guidance to use the generated index and code-fact extractor at pattern triage**

Add guidance like:

```markdown
- Read the generated `references/patterns/index.md` before any detailed pattern reference.
- When code structure is still unclear, run `python3 ./scripts/extract_code_facts.py --operator-file <operator-file>` and use the returned facts as triage evidence.
- Use the code facts plus current benchmark behavior to narrow to one or two candidate pattern cards before deeper reads.
```

And prompt lines like:

```python
"Read the generated `references/patterns/index.md` before detailed pattern references.",
"At pattern triage, prefer small code-fact evidence before opening multiple full pattern documents.",
"Use the staged code-fact extractor when the code structure is still unclear at pattern triage.",
```

- [ ] **Step 4: Run the focused tests and strict pyright checks**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_pattern_tools \
  tests.test_generation_contracts \
  tests.test_optimize_guidance \
  tests.test_cli -v
```

Expected: PASS

Run:

```bash
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize/scripts/extract_code_facts.py
```

Expected: `0 errors, 0 warnings, 0 informations`

### Task 6: Verify the full routing scaffold and review the final scope

**Files:**
- Modify only as needed from earlier tasks after verification feedback

- [ ] **Step 1: Run the full targeted regression suite**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_pattern_tools \
  tests.test_generation_contracts \
  tests.test_optimize_guidance \
  tests.test_cli \
  tests.test_run_skill_loader \
  tests.test_skill_command_script -v
```

Expected: PASS

- [ ] **Step 2: Re-run both strict skill-script pyright checks**

Run:

```bash
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize/scripts/build_pattern_index.py
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize/scripts/extract_code_facts.py
```

Expected: both commands report no errors.

- [ ] **Step 3: Review the final diff for scope**

Run:

```bash
git diff -- \
  docs/specs/2026-04-28-optimize-pattern-evidence-routing-design.md \
  docs/plans/2026-04-28-optimize-pattern-evidence-routing.md \
  skills/triton-npu-optimize/SKILL.md \
  skills/triton-npu-optimize/references/patterns \
  skills/triton-npu-optimize/scripts/build_pattern_index.py \
  skills/triton-npu-optimize/scripts/extract_code_facts.py \
  skills/triton-npu-analyze-round-performance/SKILL.md \
  skills/triton-npu-analyze-round-performance/references/symptoms \
  src/triton_agent/optimize/prompts.py \
  tests/test_optimize_pattern_tools.py \
  tests/test_generation_contracts.py \
  tests/test_optimize_guidance.py \
  tests/test_cli.py
```

Expected: only pattern-routing docs, helper scripts, skill/prompt wording, and their tests are touched; no unrelated CLI behavior or optimize gate logic changed.
