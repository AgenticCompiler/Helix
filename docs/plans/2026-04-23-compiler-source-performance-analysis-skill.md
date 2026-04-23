# Compiler Source Performance Analysis Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the performance-focused redesign of `triton-npu-analyze-compiler-source`, including new navigation references, a light compiler-source navigator script, and updated sibling-skill wording.

**Architecture:** Keep workflow ownership in the compiler-source skill and its repo-owned references. Add one light `inspect_compiler_source.py` helper that only narrows likely source locations under high-value AscendNPU-IR subtrees. Enforce the redesign with `unittest` contract tests so the skill stays performance-oriented and tied to round-local evidence.

**Tech Stack:** Markdown skill docs, Python 3.12, `argparse`, `dataclasses`, `json`, `pathlib`, `re`, `importlib.util`, Python `unittest`

---

## File Map

- Modify: `skills/triton-npu-analyze-compiler-source/SKILL.md`
  The performance-focused skill contract and output requirements.
- Create: `skills/triton-npu-analyze-compiler-source/references/navigation-map.md`
  Source-tree-oriented navigation reference.
- Create: `skills/triton-npu-analyze-compiler-source/references/perf-question-playbook.md`
  Performance-question-oriented navigation reference.
- Create: `skills/triton-npu-analyze-compiler-source/scripts/inspect_compiler_source.py`
  A locate-only navigator for compiler source subtrees.
- Modify: `skills/triton-npu-optimize/SKILL.md`
  Clarify compiler-source escalation as a performance-focused, next-action-oriented step.
- Modify: `skills/triton-npu-analyze-round-performance/SKILL.md`
  Align the compiler-source handoff with the new performance-focused skill identity.
- Modify: `tests/test_generation_contracts.py`
  Skill contract assertions for the redesigned compiler-source workflow and navigation references.
- Create: `tests/test_inspect_compiler_source.py`
  Unit tests for the new navigator script.

### Task 1: Lock In The Compiler-Source Skill Contract

**Files:**
- Modify: `tests/test_generation_contracts.py`
- Modify: `skills/triton-npu-analyze-compiler-source/SKILL.md`
- Create: `skills/triton-npu-analyze-compiler-source/references/navigation-map.md`
- Create: `skills/triton-npu-analyze-compiler-source/references/perf-question-playbook.md`

- [ ] **Step 1: Replace the placeholder contract assertions with failing performance-focused tests**

In `tests/test_generation_contracts.py`, replace the existing compiler-source skill assertion with these tests:

```python
    def test_compiler_source_analysis_skill_focuses_on_performance_navigation_and_next_action(self) -> None:
        content = _read("skills/triton-npu-analyze-compiler-source/SKILL.md")

        self.assertIn("Analyze Compiler Source For Performance", content)
        self.assertIn("Round Performance Question", content)
        self.assertIn("Round Evidence Used", content)
        self.assertIn("Recommended Next Operator Change", content)
        self.assertIn("references/navigation-map.md", content)
        self.assertIn("references/perf-question-playbook.md", content)
        self.assertIn("Inspect `docs/` first", content)
        self.assertIn("`lib/` for implementation evidence", content)
        self.assertIn("`include/` only when declarations", content)
        self.assertIn("`test/` only when a minimal example is genuinely necessary", content)
        self.assertIn("Treat the compiler source checkout as read-only", content)
        self.assertIn("Do not run `git clone`, `git fetch`, or `git pull`", content)
        self.assertIn("CLI-provided compiler source path and commit", content)
        self.assertNotIn("compiler error", content.lower())

    def test_compiler_source_navigation_references_exist_and_capture_expected_sections(self) -> None:
        navigation = _read(
            "skills/triton-npu-analyze-compiler-source/references/navigation-map.md"
        )
        playbook = _read(
            "skills/triton-npu-analyze-compiler-source/references/perf-question-playbook.md"
        )

        self.assertIn("# Compiler Source Navigation Map", navigation)
        self.assertIn("## Default Reading Order", navigation)
        self.assertIn("round evidence -> docs -> lib -> include", navigation)
        self.assertIn("## Symptom To Subtree", navigation)
        self.assertIn("## Search Recipes", navigation)
        self.assertIn("## Anti-Patterns", navigation)

        self.assertIn("# Performance Question Playbook", playbook)
        self.assertIn("## Suspicious Stage Transition", playbook)
        self.assertIn("## Vectorization Loss", playbook)
        self.assertIn("## Copy Or Sync Growth", playbook)
        self.assertIn("## Turning Source Findings Into Operator Actions", playbook)
```

- [ ] **Step 2: Run the contract tests and confirm they fail**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts.GenerationContractTests -v
```

Expected: FAIL because `SKILL.md` still mentions compiler errors and the two reference files do not exist yet.

- [ ] **Step 3: Rewrite `SKILL.md` around a performance-only workflow**

Replace `skills/triton-npu-analyze-compiler-source/SKILL.md` with a document shaped like this:

```markdown
---
name: triton-npu-analyze-compiler-source
description: Use when an optimize round has compiler source analysis enabled and needs source-backed explanation for a performance-related lowering symptom, suspicious pass effect, or compiler-side behavior that profiler and IR evidence have already narrowed but not fully explained.
---

# Analyze Compiler Source For Performance

## Goal

Use the CLI-provided AscendNPU-IR checkout to explain one narrowed compiler-side performance question, then turn that explanation into a concrete next operator change for the current Triton Ascend optimize round.

## Required Inputs

- the current operator workspace and `opt-round-N/`
- at least one round-local performance artifact:
  - `opt-round-N/perf-analysis.md`
  - `opt-round-N/ir/`
- the CLI-provided compiler source path and commit
- one narrowed compiler-side performance question

## When To Use

- Compiler source analysis is enabled by the current optimize launch prompt or guidance.
- Profile and IR evidence have already narrowed the problem to a compiler-side performance behavior.
- The round still needs source-backed explanation before choosing the next operator change.

## When Not To Use

- Do not use compiler source as the first analysis step.
- Do not use this skill when the round has no performance evidence yet.
- Do not use this skill for broad compiler browsing.
- Do not use this skill when profile and IR already justify a concrete next operator change.

## Default Workflow

1. Rewrite the current round symptom into one narrowed compiler-side performance question.
2. Read [`references/navigation-map.md`](references/navigation-map.md).
3. Read [`references/perf-question-playbook.md`](references/perf-question-playbook.md).
4. Inspect `docs/` first to orient on pass, feature, or subsystem meaning.
5. Inspect `lib/` for implementation evidence.
6. Inspect `include/` only when declarations, generated pass interfaces, or API boundaries are needed.
7. Inspect `test/` only when a minimal example is genuinely necessary.
8. Write `opt-round-N/compiler-analysis.md`.

## Navigation Rules

- Prefer `docs/` first for semantic orientation.
- Prefer `lib/` for implementation evidence.
- Treat `include/` as a navigation and contract aid, not the main evidence source.
- Treat `test/` as a rare fallback, not a default source.

## Output Contract

Write `opt-round-N/compiler-analysis.md` with these sections:

1. `# Compiler Source Analysis`
2. `## Executive Summary`
3. `## Round Performance Question`
4. `## Compiler Source Context`
5. `## Round Evidence Used`
6. `## Source Files Inspected`
7. `## Source-Backed Explanation`
8. `## Implications For Current Operator`
9. `## Recommended Next Operator Change`
10. `## Confidence And Evidence Gaps`

## Reasoning Rules

- Use only the CLI-provided compiler source path and commit.
- Treat the compiler source checkout as read-only.
- Do not run `git clone`, `git fetch`, or `git pull`.
- Separate direct facts from inference.
- Cite local source paths and the inspected commit for nontrivial source-backed claims.
- If the analysis still cannot guide the next operator change, keep narrowing instead of stopping at compiler notes.
```

- [ ] **Step 4: Add `navigation-map.md` with source-tree-first navigation guidance**

Create `skills/triton-npu-analyze-compiler-source/references/navigation-map.md` with this structure and the concrete subtree bullets below:

```markdown
# Compiler Source Navigation Map

## Purpose

This reference helps narrow one performance-related compiler question to a small set of AscendNPU-IR source locations.

## Default Reading Order

`round evidence -> docs -> lib -> include(when needed) -> test(rare fallback)`

## Directory Atlas

### `docs/source/*/developer_guide/passes/`
- Read this first when the question is stage-oriented or pass-oriented.
- Use it to understand what a pass family is supposed to do.

### `docs/source/*/developer_guide/features/`
- Read this first when the symptom looks like a feature or subsystem behavior.
- Use it to orient on pipeline concepts before implementation reading.

### `bishengir/lib/Conversion/`
- Default implementation root for lowering-path and pass-effect questions.

### `bishengir/lib/Dialect/`
- Default implementation root for dialect-level behavior and op semantics.

### `bishengir/lib/Transforms/`
- Default implementation root for transform-heavy behavior not anchored to one dialect.

### `bishengir/include/bishengir/`
- Use only when `Passes.td`, `Passes.h`, interface declarations, or registration surfaces are needed.

## Symptom To Subtree

- vectorization loss -> `docs/.../passes/` then `bishengir/lib/Conversion/` and `bishengir/lib/Transforms/`
- copy or sync growth -> `docs/.../features/` then `bishengir/lib/Dialect/` and `bishengir/lib/Conversion/`
- buffer expansion or layout churn -> `docs/.../features/` then `bishengir/lib/Dialect/` and `bishengir/lib/Conversion/`
- suspicious stage transition -> `docs/.../passes/` then the matching `bishengir/lib/` subtree

## Search Recipes

```bash
rg -n "vector|vectorize|auto-vectorize" <source-root>/docs <source-root>/bishengir/lib
rg -n "copy|dma|wait|barrier|sync" <source-root>/docs <source-root>/bishengir/lib
rg -n "Passes\\.td|Passes\\.h" <source-root>/bishengir/include/bishengir
```

## Anti-Patterns

- Do not start with a whole-tree search across every directory.
- Do not read `include/` before you know which subsystem matters.
- Do not treat `test/` as the default path for performance analysis.
```

- [ ] **Step 5: Add `perf-question-playbook.md` with symptom-driven playbooks**

Create `skills/triton-npu-analyze-compiler-source/references/perf-question-playbook.md` with this structure:

```markdown
# Performance Question Playbook

## From Perf Symptom To Compiler Question

Rewrite the current round symptom into one narrow compiler-side question before reading source.

## Suspicious Stage Transition

- Start from the suspicious adjacent stages already identified in `opt-round-N/ir/`.
- Read pass docs first.
- Then inspect the matching `bishengir/lib/` subtree.

## Vectorization Loss

- Confirm the loss in IR first.
- Read pass or feature docs for the likely vectorization subsystem.
- Inspect `bishengir/lib/Conversion/` and `bishengir/lib/Transforms/`.

## Copy Or Sync Growth

- Confirm the growth in IR or `perf-analysis.md`.
- Read feature docs first.
- Inspect dialect or conversion code that can introduce those operations.

## Buffer Expansion Or Memory-Planning Issue

- Confirm the symptom in IR first.
- Read feature docs for memory- or layout-related behavior.
- Inspect the matching implementation subtree under `bishengir/lib/`.

## Fusion Or Lowering Shape Regression

- Confirm where the structure changes across stages.
- Read pass docs first, then inspect the implementation subtree.

## Turning Source Findings Into Operator Actions

Always finish by writing:

- the likely compiler-side explanation
- what that implies for the current Triton operator
- what the next operator change should target
```

- [ ] **Step 6: Run the contract tests again and verify they pass**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts.GenerationContractTests -v
```

Expected: PASS for the new compiler-source skill and reference assertions.

- [ ] **Step 7: Commit the skill-contract rewrite**

Run:

```bash
git add \
  skills/triton-npu-analyze-compiler-source/SKILL.md \
  skills/triton-npu-analyze-compiler-source/references/navigation-map.md \
  skills/triton-npu-analyze-compiler-source/references/perf-question-playbook.md \
  tests/test_generation_contracts.py
git commit -m "docs: redesign compiler source analysis skill"
```

### Task 2: Add The Compiler-Source Navigator Script

**Files:**
- Create: `tests/test_inspect_compiler_source.py`
- Create: `skills/triton-npu-analyze-compiler-source/scripts/inspect_compiler_source.py`

- [ ] **Step 1: Write failing unit tests for a locate-only navigator**

Create `tests/test_inspect_compiler_source.py` with this structure:

```python
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    script = (
        REPO_ROOT
        / "skills"
        / "triton-npu-analyze-compiler-source"
        / "scripts"
        / "inspect_compiler_source.py"
    )
    spec = importlib.util.spec_from_file_location("inspect_compiler_source_test_module", script)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module spec for {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_source_tree(root: Path) -> Path:
    source_root = root / "AscendNPU-IR"
    (source_root / "docs/source/en/developer_guide/passes").mkdir(parents=True)
    (source_root / "bishengir/lib/Conversion/HFusionToHIVM").mkdir(parents=True)
    (source_root / "bishengir/include/bishengir/Conversion").mkdir(parents=True)
    (source_root / "bishengir/test/Conversion/HFusionToHIVM").mkdir(parents=True)

    (source_root / "docs/source/en/developer_guide/passes/HFusionPasses.md").write_text(
        "Auto vectorize and hfusion lowering notes.\n",
        encoding="utf-8",
    )
    (source_root / "bishengir/lib/Conversion/HFusionToHIVM/Vectorize.cpp").write_text(
        "void runVectorizePass();\n",
        encoding="utf-8",
    )
    (source_root / "bishengir/include/bishengir/Conversion/Passes.td").write_text(
        "def HFusionVectorizePass : Pass<\"hfusion-vectorize\">;\n",
        encoding="utf-8",
    )
    (source_root / "bishengir/test/Conversion/HFusionToHIVM/vectorize.mlir").write_text(
        "// not part of default search scope\n",
        encoding="utf-8",
    )
    return source_root


class InspectCompilerSourceTests(unittest.TestCase):
    def test_build_parser_parses_locate_arguments(self) -> None:
        module = _load_module()

        args = module.build_parser().parse_args(
            [
                "locate",
                "--source-root",
                "AscendNPU-IR",
                "--term",
                "hfusion",
                "--term",
                "vectorize",
                "--hint",
                "pass",
                "--format",
                "json",
            ]
        )

        self.assertEqual(args.command, "locate")
        self.assertEqual(args.source_root, "AscendNPU-IR")
        self.assertEqual(args.term, ["hfusion", "vectorize"])
        self.assertEqual(args.hint, "pass")
        self.assertEqual(args.format, "json")

    def test_locate_payload_groups_docs_lib_and_include_matches(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            source_root = _make_source_tree(Path(tmp))
            payload = module.locate_payload(
                source_root,
                terms=["hfusion", "vectorize"],
                hint="pass",
                limit=10,
            )

        self.assertIn("docs", payload)
        self.assertIn("lib", payload)
        self.assertIn("include", payload)
        self.assertEqual(payload["docs"][0]["area"], "docs")
        self.assertTrue(payload["docs"][0]["path"].endswith("HFusionPasses.md"))
        self.assertTrue(payload["lib"][0]["path"].endswith("Vectorize.cpp"))
        self.assertTrue(payload["include"][0]["path"].endswith("Passes.td"))

    def test_locate_payload_omits_test_scope_by_default(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            source_root = _make_source_tree(Path(tmp))
            payload = module.locate_payload(
                source_root,
                terms=["vectorize"],
                hint="conversion",
                limit=10,
            )

        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("bishengir/test", rendered)

    def test_locate_text_renders_grouped_candidates(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            source_root = _make_source_tree(Path(tmp))
            rendered = module.locate_text(
                source_root,
                terms=["vectorize"],
                hint="pass",
                limit=5,
            )

        self.assertIn("docs:", rendered)
        self.assertIn("lib:", rendered)
        self.assertIn("include:", rendered)
        self.assertIn("matched_terms=", rendered)
```

- [ ] **Step 2: Run the navigator tests and verify they fail**

Run:

```bash
uv run python -m unittest tests.test_inspect_compiler_source -v
```

Expected: FAIL because the script file does not exist yet.

- [ ] **Step 3: Implement `inspect_compiler_source.py` with a single `locate` command**

Create `skills/triton-npu-analyze-compiler-source/scripts/inspect_compiler_source.py` around this structure:

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SearchArea:
    name: str
    relative_root: str
    glob: str


SEARCH_AREAS: tuple[SearchArea, ...] = (
    SearchArea("docs", "docs/source/en/developer_guide/passes", "*.md"),
    SearchArea("docs", "docs/source/zh_cn/developer_guide/passes", "*.md"),
    SearchArea("docs", "docs/source/en/developer_guide/features", "*.md"),
    SearchArea("docs", "docs/source/zh_cn/developer_guide/features", "*.md"),
    SearchArea("lib", "bishengir/lib/Conversion", "*"),
    SearchArea("lib", "bishengir/lib/Dialect", "*"),
    SearchArea("lib", "bishengir/lib/Transforms", "*"),
    SearchArea("include", "bishengir/include/bishengir", "*"),
)


def locate_payload(
    source_root: str | Path,
    *,
    terms: list[str],
    hint: str | None = None,
    limit: int = 10,
) -> dict[str, list[dict[str, object]]]:
    root = Path(source_root).expanduser().resolve()
    lowered_terms = [term.lower() for term in terms]
    grouped: dict[str, list[dict[str, object]]] = {"docs": [], "lib": [], "include": []}
    for area in SEARCH_AREAS:
        candidate_root = root / area.relative_root
        if not candidate_root.exists():
            continue
        for path in sorted(candidate_root.rglob(area.glob)):
            if path.is_dir():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            score, matched_terms, why = _score_candidate(
                path=path,
                text=text,
                lowered_terms=lowered_terms,
                hint=hint,
                source_root=root,
                area=area.name,
            )
            if score <= 0:
                continue
            grouped[area.name].append(
                {
                    "area": area.name,
                    "path": str(path),
                    "score": score,
                    "matched_terms": matched_terms,
                    "why": why,
                }
            )
    for key, items in grouped.items():
        grouped[key] = sorted(items, key=lambda item: (-int(item["score"]), str(item["path"])))[:limit]
    return grouped


def locate_text(
    source_root: str | Path,
    *,
    terms: list[str],
    hint: str | None = None,
    limit: int = 10,
) -> str:
    payload = locate_payload(source_root, terms=terms, hint=hint, limit=limit)
    lines: list[str] = ["Compiler source candidates:"]
    for area in ("docs", "lib", "include"):
        lines.append("")
        lines.append(f"{area}:")
        if not payload[area]:
            lines.append("  (no matches)")
            continue
        for item in payload[area]:
            lines.append(
                "  "
                f"{item['path']}  score={item['score']}  "
                f"matched_terms={','.join(item['matched_terms'])}  why={item['why']}"
            )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect compiler source for likely navigation targets.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    locate = subparsers.add_parser("locate")
    locate.add_argument("--source-root", required=True)
    locate.add_argument("--term", action="append", required=True)
    locate.add_argument("--hint", choices=("pass", "feature", "dialect", "conversion", "layout", "memory", "pipeline"))
    locate.add_argument("--limit", type=int, default=10)
    locate.add_argument("--format", choices=("text", "json"), default="text")
    return parser
```

Implement `_score_candidate(...)` so it:

- gives positive weight for term matches in the relative path
- gives positive weight for term matches in the file text
- biases `docs` slightly upward when `hint` is `pass` or `feature`
- biases `lib` slightly upward for `conversion`, `dialect`, `layout`, `memory`, or `pipeline`
- keeps `include` searchable but lower-priority than `lib`
- never searches `bishengir/test/`

Finish `main()` with:

```python
def main() -> int:
    args = build_parser().parse_args()
    if args.command != "locate":
        raise ValueError(f"Unsupported command: {args.command}")
    if args.format == "json":
        print(json.dumps(locate_payload(args.source_root, terms=args.term, hint=args.hint, limit=args.limit), indent=2, sort_keys=True))
    else:
        print(locate_text(args.source_root, terms=args.term, hint=args.hint, limit=args.limit), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the navigator tests again and verify they pass**

Run:

```bash
uv run python -m unittest tests.test_inspect_compiler_source -v
```

Expected: PASS for parser, grouped payload, grouped text output, and omission of `bishengir/test/`.

- [ ] **Step 5: Commit the navigator script**

Run:

```bash
git add \
  skills/triton-npu-analyze-compiler-source/scripts/inspect_compiler_source.py \
  tests/test_inspect_compiler_source.py
git commit -m "feat: add compiler source navigator"
```

### Task 3: Align Sibling Skills With The New Compiler-Source Identity

**Files:**
- Modify: `tests/test_generation_contracts.py`
- Modify: `skills/triton-npu-optimize/SKILL.md`
- Modify: `skills/triton-npu-analyze-round-performance/SKILL.md`

- [ ] **Step 1: Add failing assertions for the sibling-skill wording**

In `tests/test_generation_contracts.py`, update `test_optimize_skills_document_compiler_source_escalation` to assert the new wording:

```python
    def test_optimize_skills_document_compiler_source_escalation(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        round_analysis = _read("skills/triton-npu-analyze-round-performance/SKILL.md")

        self.assertIn("compiler-source escalation", optimize)
        self.assertIn("performance-focused explanation", optimize)
        self.assertIn("next operator change", optimize)
        self.assertIn("after profiler and IR evidence", optimize)
        self.assertIn("opt-round-N/compiler-analysis.md", optimize)

        self.assertIn("compiler source analysis is enabled", round_analysis)
        self.assertIn("performance-related compiler-side question", round_analysis)
        self.assertIn("next operator change", round_analysis)
```

- [ ] **Step 2: Run the contract tests and confirm the new assertions fail**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts.GenerationContractTests -v
```

Expected: FAIL because the sibling skills still describe the compiler-source step with older wording.

- [ ] **Step 3: Update the optimize skill to describe compiler source as a performance-focused escalation**

In `skills/triton-npu-optimize/SKILL.md`, replace the current compiler-source bullets with language shaped like this:

```markdown
### compiler-source escalation

- Use compiler-source escalation only when compiler source analysis is enabled and after profiler and IR evidence have narrowed a performance-related compiler-side question.
- Use the sibling `triton-npu-analyze-compiler-source` skill when the round still needs source-backed performance explanation before choosing the next operator change.
- Treat the compiler source checkout as read-only.
- Write `opt-round-N/compiler-analysis.md`.
```

- [ ] **Step 4: Update the round-performance skill to hand off with the same wording**

In `skills/triton-npu-analyze-round-performance/SKILL.md`, replace the current compiler-source handoff paragraph with:

```markdown
When compiler source analysis is enabled by the launch prompt or workspace guidance, treat it as a later escalation after profile and IR analysis. Use `triton-npu-analyze-compiler-source` only when this skill has narrowed the problem to a concrete performance-related compiler-side question that still needs source-backed explanation before the next operator change is clear.
```

- [ ] **Step 5: Run the contract tests again and verify they pass**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts.GenerationContractTests -v
```

Expected: PASS for the compiler-source and sibling-skill contract assertions.

- [ ] **Step 6: Commit the sibling-skill wording updates**

Run:

```bash
git add \
  skills/triton-npu-optimize/SKILL.md \
  skills/triton-npu-analyze-round-performance/SKILL.md \
  tests/test_generation_contracts.py
git commit -m "docs: align compiler source escalation wording"
```

### Task 4: Run Verification And Final Sanity Checks

**Files:**
- Modify if needed: `skills/triton-npu-analyze-compiler-source/SKILL.md`
- Modify if needed: `skills/triton-npu-analyze-compiler-source/references/navigation-map.md`
- Modify if needed: `skills/triton-npu-analyze-compiler-source/references/perf-question-playbook.md`
- Modify if needed: `skills/triton-npu-analyze-compiler-source/scripts/inspect_compiler_source.py`
- Modify if needed: `skills/triton-npu-optimize/SKILL.md`
- Modify if needed: `skills/triton-npu-analyze-round-performance/SKILL.md`
- Modify if needed: `tests/test_generation_contracts.py`
- Modify if needed: `tests/test_inspect_compiler_source.py`

- [ ] **Step 1: Run the focused verification suite**

Run:

```bash
uv run python -m unittest \
  tests.test_generation_contracts \
  tests.test_inspect_compiler_source \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run repository lint checks**

Run:

```bash
uv run --group dev ruff check
```

Expected: PASS.

- [ ] **Step 3: Run static type checks**

Run:

```bash
uv run pyright
```

Expected: PASS.

- [ ] **Step 4: Run the repository unit test suite**

Run:

```bash
uv run python -m unittest discover -s tests -v
```

Expected: PASS.

- [ ] **Step 5: Commit only if verification required a cleanup fix**

If one of the verification steps forced a follow-up fix, commit it with:

```bash
git add \
  skills/triton-npu-analyze-compiler-source/SKILL.md \
  skills/triton-npu-analyze-compiler-source/references/navigation-map.md \
  skills/triton-npu-analyze-compiler-source/references/perf-question-playbook.md \
  skills/triton-npu-analyze-compiler-source/scripts/inspect_compiler_source.py \
  skills/triton-npu-optimize/SKILL.md \
  skills/triton-npu-analyze-round-performance/SKILL.md \
  tests/test_generation_contracts.py \
  tests/test_inspect_compiler_source.py
git commit -m "chore: finish compiler source skill verification fixes"
```
