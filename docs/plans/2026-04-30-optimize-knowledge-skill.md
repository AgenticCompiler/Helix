# Optimize Knowledge Skill Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move generic optimize pattern and symptom knowledge into a new reference-only `triton-npu-optimize-knowledge` skill while keeping optimize workflow and round-diagnosis ownership unchanged.

**Architecture:** Keep runtime orchestration unchanged and treat this as a knowledge-ownership refactor inside the staged `skills/` tree. Create one new reference-only skill that owns generic pattern cards, symptom cards, and both generated indexes; repoint optimize and round-analysis skills plus prompt/guidance text to that skill; then delete the old duplicate knowledge copies so ownership is unambiguous.

**Tech Stack:** Python 3.11, `unittest`, `argparse`, `pathlib`, Markdown skill docs, existing optimize prompt/guidance plumbing, repo-local skill script pyright wrapper

---

## File Map

- Create: `skills/triton/triton-npu-optimize-knowledge/SKILL.md`
  Reference-only contract for generic optimize knowledge.
- Create: `skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md`
  Generated pattern index, moved from the optimize workflow skill.
- Create: `skills/triton/triton-npu-optimize-knowledge/references/patterns/*.md`
  Authored generic pattern cards, moved from the optimize workflow skill.
- Create: `skills/triton/triton-npu-optimize-knowledge/references/symptom_index.md`
  Generated symptom index, moved from the round-analysis skill.
- Create: `skills/triton/triton-npu-optimize-knowledge/references/symptoms/*.md`
  Authored generic symptom cards, moved from the round-analysis skill.
- Create: `skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py`
  Existing pattern-index generator under the new owner skill.
- Create: `skills/triton/triton-npu-optimize-knowledge/scripts/build_symptom_index.py`
  New symptom-index generator with check mode.
- Modify: `skills/triton/triton-npu-optimize/SKILL.md`
  Repoint generic pattern reading to the sibling knowledge skill.
- Modify: `skills/triton-npu-analyze-round-performance/SKILL.md`
  Repoint symptom routing to the sibling knowledge skill.
- Modify: `src/helix/optimize/prompts.py`
  Mention the staged knowledge skill in optimize prompt and guidance text.
- Modify: `tests/test_optimize_pattern_tools.py`
  Cover the new generator paths plus the symptom-index generator.
- Modify: `tests/test_generation_contracts.py`
  Make the new knowledge skill and AGENTS rules the source-of-truth checks.
- Modify: `tests/test_cli.py`
  Update optimize prompt assertions.
- Modify: `tests/test_optimize_guidance.py`
  Update shared guidance assertions.
- Modify: `AGENTS.md`
  Move durable generic pattern and symptom source-of-truth rules to the knowledge skill.
- Modify: `docs/notes/2026-04-29-optimize-pattern-card-authoring.md`
  Update the pattern authoring path and regeneration commands.
- Create: `docs/notes/2026-04-30-optimize-symptom-card-authoring.md`
  Document the symptom card contract and symptom-index regeneration flow.
- Delete: `skills/triton/triton-npu-optimize/references/pattern_index.md`
- Delete: `skills/triton/triton-npu-optimize/references/patterns/*.md`
- Delete: `skills/triton/triton-npu-optimize/scripts/build_pattern_index.py`
- Delete: `skills/triton-npu-analyze-round-performance/references/symptom_index.md`
- Delete: `skills/triton-npu-analyze-round-performance/references/symptoms/*.md`

### Task 1: Scaffold The Knowledge Skill And Move Pattern Ownership

**Files:**
- Create: `skills/triton/triton-npu-optimize-knowledge/SKILL.md`
- Create: `skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md`
- Create: `skills/triton/triton-npu-optimize-knowledge/references/patterns/*.md`
- Create: `skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py`
- Modify: `tests/test_optimize_pattern_tools.py`
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Write failing tests for the new knowledge-skill pattern paths**

Add these updates in `tests/test_optimize_pattern_tools.py`:

```python
    def test_checked_in_pattern_index_matches_generator(self) -> None:
        module = _load_skill_script(
            "skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py"
        )
        patterns_dir = (
            REPO_ROOT
            / "skills"
            / "triton-npu-optimize-knowledge"
            / "references"
            / "patterns"
        )
        generated = module.build_index_text(patterns_dir)
        checked_in = (
            REPO_ROOT
            / "skills"
            / "triton-npu-optimize-knowledge"
            / "references"
            / "pattern_index.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(generated, checked_in)

    def test_generated_index_links_to_pattern_subdirectory(self) -> None:
        module = _load_skill_script(
            "skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            patterns_dir = Path(tmp)
            (patterns_dir / "demo.md").write_text(
                "# Demo Pattern\n\n## Summary\n\nShort summary.\n\n## Use When\n\n- Stable trigger.\n",
                encoding="utf-8",
            )
            rendered = module.build_index_text(patterns_dir)
            self.assertIn("[demo.md](patterns/demo.md)", rendered)
```

Add these updates in `tests/test_generation_contracts.py`:

```python
    def test_optimize_knowledge_skill_owns_generic_pattern_references(self) -> None:
        knowledge = _read("skills/triton/triton-npu-optimize-knowledge/SKILL.md")
        index = _read("skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md")
        reference = _read(
            "skills/triton/triton-npu-optimize-knowledge/references/patterns/classic-matmul.md"
        )

        self.assertIn("reference-only", knowledge)
        self.assertIn("does not define optimize workflow", knowledge)
        self.assertIn("pattern_index.md", knowledge)
        self.assertIn("classic-matmul", index)
        self.assertIn("matmul-like", reference)
```

- [ ] **Step 2: Run the focused tests to confirm they fail**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_checked_in_pattern_index_matches_generator \
  tests.test_generation_contracts.GenerationContractTests.test_optimize_knowledge_skill_owns_generic_pattern_references -v
```

Expected: FAIL because `skills/triton/triton-npu-optimize-knowledge/` and the moved pattern assets do not exist yet.

- [ ] **Step 3: Create the new knowledge skill scaffold and copy the pattern assets**

Create `skills/triton/triton-npu-optimize-knowledge/SKILL.md` with this content:

```markdown
---
name: triton-npu-optimize-knowledge
description: Generic reference-only optimize knowledge for pattern triage and evidence-backed symptom routing. This skill does not define optimize workflow or own round artifacts.
---

# Optimize Knowledge

## Purpose

This skill is the generic optimize knowledge library for reusable pattern and symptom references.

## Scope

- This skill is reference-only.
- This skill does not define optimize workflow behavior.
- This skill does not own `opt-round-N/perf-analysis.md`, `attempts.md`, `summary.md`, or `opt-note.md`.
- `triton-npu-optimize` owns optimize workflow and validation rules.
- `triton-npu-analyze-round-performance` owns round-level performance diagnosis.

## Reading Order

1. For code-structure-first triage, read `references/pattern_index.md`.
2. For profile- or IR-backed routing, read `references/symptom_index.md`.
3. Read only the one or two most relevant detailed cards after the index narrows the candidate set.

## Reasoning Rules

- Treat pattern cards and symptom cards as routing aids, not a hard rule engine.
- Return to the caller skill for diagnosis, optimization choice, and recordkeeping.
- Keep specialized packs such as `triton-npu-cann-ext-api-patterns` separate unless the caller explicitly needs them.
```

Populate the new tree by copying the existing pattern assets:

```bash
mkdir -p skills/triton/triton-npu-optimize-knowledge/references/patterns
mkdir -p skills/triton/triton-npu-optimize-knowledge/scripts
cp skills/triton/triton-npu-optimize/references/pattern_index.md \
  skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md
cp skills/triton/triton-npu-optimize/references/patterns/*.md \
  skills/triton/triton-npu-optimize-knowledge/references/patterns/
cp skills/triton/triton-npu-optimize/scripts/build_pattern_index.py \
  skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py
```

- [ ] **Step 4: Re-run the focused tests and strict pyright on the moved generator**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_checked_in_pattern_index_matches_generator \
  tests.test_generation_contracts.GenerationContractTests.test_optimize_knowledge_skill_owns_generic_pattern_references -v
```

Expected: PASS

Run:

```bash
bash scripts/run-skill-script-pyright.sh \
  skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py
```

Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 5: Commit the scaffolded pattern-knowledge move**

```bash
git add \
  skills/triton/triton-npu-optimize-knowledge/SKILL.md \
  skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md \
  skills/triton/triton-npu-optimize-knowledge/references/patterns \
  skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py \
  tests/test_optimize_pattern_tools.py \
  tests/test_generation_contracts.py
git commit -m "feat: add optimize knowledge skill pattern scaffold"
```

### Task 2: Add Symptom Generator Support Under The Knowledge Skill

**Files:**
- Create: `skills/triton/triton-npu-optimize-knowledge/references/symptom_index.md`
- Create: `skills/triton/triton-npu-optimize-knowledge/references/symptoms/*.md`
- Create: `skills/triton/triton-npu-optimize-knowledge/scripts/build_symptom_index.py`
- Modify: `tests/test_optimize_pattern_tools.py`
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Add failing tests for symptom generator behavior and new source-of-truth paths**

Append these tests to `tests/test_optimize_pattern_tools.py`:

```python
    def test_build_symptom_index_requires_summary_evidence_and_candidates(self) -> None:
        module = _load_skill_script(
            "skills/triton/triton-npu-optimize-knowledge/scripts/build_symptom_index.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            symptoms_dir = Path(tmp)
            (symptoms_dir / "broken.md").write_text(
                "# broken\n\n## Summary\n\nMissing evidence and pattern directions.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "Evidence To Confirm, Candidate Pattern Directions"
            ):
                module.build_index_text(symptoms_dir)

    def test_checked_in_symptom_index_matches_generator(self) -> None:
        module = _load_skill_script(
            "skills/triton/triton-npu-optimize-knowledge/scripts/build_symptom_index.py"
        )
        symptoms_dir = (
            REPO_ROOT
            / "skills"
            / "triton-npu-optimize-knowledge"
            / "references"
            / "symptoms"
        )
        generated = module.build_index_text(symptoms_dir)
        checked_in = (
            REPO_ROOT
            / "skills"
            / "triton-npu-optimize-knowledge"
            / "references"
            / "symptom_index.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(generated, checked_in)
```

Update `tests/test_generation_contracts.py` with:

```python
    def test_optimize_knowledge_skill_owns_generic_symptom_references(self) -> None:
        knowledge = _read("skills/triton/triton-npu-optimize-knowledge/SKILL.md")
        symptom_index = _read(
            "skills/triton/triton-npu-optimize-knowledge/references/symptom_index.md"
        )
        symptom = _read(
            "skills/triton/triton-npu-optimize-knowledge/references/symptoms/weak-pipeline-overlap.md"
        )

        self.assertIn("symptom_index.md", knowledge)
        self.assertIn("weak-pipeline-overlap", symptom_index)
        self.assertIn("## Evidence To Confirm", symptom)
        self.assertIn("## Candidate Pattern Directions", symptom)
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_build_symptom_index_requires_summary_evidence_and_candidates \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_checked_in_symptom_index_matches_generator \
  tests.test_generation_contracts.GenerationContractTests.test_optimize_knowledge_skill_owns_generic_symptom_references -v
```

Expected: FAIL because the symptom assets and generator do not exist in the knowledge skill yet.

- [ ] **Step 3: Copy symptom assets and implement `build_symptom_index.py`**

Copy the existing symptom cards and index into the new owner skill:

```bash
mkdir -p skills/triton/triton-npu-optimize-knowledge/references/symptoms
cp skills/triton-npu-analyze-round-performance/references/symptom_index.md \
  skills/triton/triton-npu-optimize-knowledge/references/symptom_index.md
cp skills/triton-npu-analyze-round-performance/references/symptoms/*.md \
  skills/triton/triton-npu-optimize-knowledge/references/symptoms/
```

Create `skills/triton/triton-npu-optimize-knowledge/scripts/build_symptom_index.py` with this implementation:

```python
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


REQUIRED_SECTIONS = ("Summary", "Evidence To Confirm", "Candidate Pattern Directions")
_SECTION_HEADING_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$", flags=re.MULTILINE)
_LIST_ITEM_RE = re.compile(r"^(?:-\s+|\d+\.\s+)(?P<item>.+?)\s*$")


@dataclass
class SymptomCard:
    identifier: str
    summary: str
    evidence_to_confirm: list[str]
    candidate_pattern_directions: list[str]
    common_non_matches: list[str]
    source_path: Path


def _top_level_sections(body: str) -> dict[str, str]:
    matches = list(_SECTION_HEADING_RE.finditer(body))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        title = match.group("title").strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[title] = body[start:end].strip()
    return sections


def _extract_bullets(section_text: str) -> list[str]:
    bullets: list[str] = []
    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        match = _LIST_ITEM_RE.match(line)
        if match:
            bullets.append(match.group("item").strip())
    return bullets


def _first_nonempty_paragraph(section_text: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section_text) if part.strip()]
    if not paragraphs:
        return ""
    return " ".join(line.strip() for line in paragraphs[0].splitlines())


def parse_symptom_card(path: Path) -> SymptomCard:
    body = path.read_text(encoding="utf-8")
    sections = _top_level_sections(body)
    missing = [name for name in REQUIRED_SECTIONS if not sections.get(name)]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"{path.name} is missing required section(s): {names}")

    return SymptomCard(
        identifier=path.stem,
        summary=_first_nonempty_paragraph(sections["Summary"]),
        evidence_to_confirm=_extract_bullets(sections["Evidence To Confirm"]),
        candidate_pattern_directions=_extract_bullets(
            sections["Candidate Pattern Directions"]
        ),
        common_non_matches=_extract_bullets(sections.get("Common Non-Matches", "")),
        source_path=path,
    )


def render_index(cards: list[SymptomCard]) -> str:
    lines = [
        "# Symptom Index",
        "",
        "Use this file after structured profile or IR evidence already exists.",
        "",
        "Read this generated index first. Then read only the one or two most relevant detailed symptom cards before returning to detailed pattern references.",
        "",
        "## Generated Symptom Summaries",
        "",
    ]
    for card in cards:
        lines.append(f"### `{card.identifier}`")
        lines.append("")
        lines.append(f"- Summary: {card.summary}")
        lines.append(f"- Source: [{card.source_path.name}](symptoms/{card.source_path.name})")
        if card.evidence_to_confirm:
            lines.append("- Evidence To Confirm:")
            lines.extend(f"  - {item}" for item in card.evidence_to_confirm)
        if card.candidate_pattern_directions:
            lines.append("- Candidate Pattern Directions:")
            lines.extend(f"  - {item}" for item in card.candidate_pattern_directions)
        if card.common_non_matches:
            lines.append("- Common Non-Matches:")
            lines.extend(f"  - {item}" for item in card.common_non_matches)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_index_text(symptoms_dir: Path) -> str:
    cards = [
        parse_symptom_card(path)
        for path in sorted(symptoms_dir.glob("*.md"))
        if path.name not in {"README.md", "index.md"}
    ]
    return render_index(cards)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the optimize symptom index from symptom cards.")
    parser.add_argument("--symptoms-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    rendered = build_index_text(Path(args.symptoms_dir))
    output_path = Path(args.output)
    if args.check:
        current = output_path.read_text(encoding="utf-8")
        if current != rendered:
            print(f"Symptom index is out of date: {output_path}")
            return 1
        return 0

    output_path.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Rebuild the checked-in symptom index and rerun the focused tests**

Run:

```bash
python3 skills/triton/triton-npu-optimize-knowledge/scripts/build_symptom_index.py \
  --symptoms-dir skills/triton/triton-npu-optimize-knowledge/references/symptoms \
  --output skills/triton/triton-npu-optimize-knowledge/references/symptom_index.md
```

Expected: `skills/triton/triton-npu-optimize-knowledge/references/symptom_index.md` is rewritten deterministically.

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_build_symptom_index_requires_summary_evidence_and_candidates \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_checked_in_symptom_index_matches_generator \
  tests.test_generation_contracts.GenerationContractTests.test_optimize_knowledge_skill_owns_generic_symptom_references -v
```

Expected: PASS

Run:

```bash
bash scripts/run-skill-script-pyright.sh \
  skills/triton/triton-npu-optimize-knowledge/scripts/build_symptom_index.py
```

Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 5: Commit the symptom-generator addition**

```bash
git add \
  skills/triton/triton-npu-optimize-knowledge/references/symptom_index.md \
  skills/triton/triton-npu-optimize-knowledge/references/symptoms \
  skills/triton/triton-npu-optimize-knowledge/scripts/build_symptom_index.py \
  tests/test_optimize_pattern_tools.py \
  tests/test_generation_contracts.py
git commit -m "feat: add optimize knowledge symptom generator"
```

### Task 3: Repoint Optimize Workflow, Analysis Docs, And Prompt Text

**Files:**
- Modify: `skills/triton/triton-npu-optimize/SKILL.md`
- Modify: `skills/triton-npu-analyze-round-performance/SKILL.md`
- Modify: `src/helix/optimize/prompts.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Add failing prompt and skill-doc assertions for the knowledge skill wording**

Update `tests/test_cli.py` and `tests/test_optimize_guidance.py` to expect these strings:

```python
self.assertIn(
    "Use the staged `triton-npu-optimize-knowledge` skill for generic pattern and symptom references.",
    prompt,
)
self.assertIn(
    "Read the staged `triton-npu-optimize-knowledge` skill's generated `references/pattern_index.md` before detailed pattern references.",
    prompt,
)
self.assertIn(
    "Use the staged `triton-npu-optimize-knowledge` skill's symptom cards to narrow pattern candidates after structured profiler or IR evidence exists.",
    prompt,
)
```

Update `tests/test_generation_contracts.py` with:

```python
        optimize = _read("skills/triton/triton-npu-optimize/SKILL.md")
        self.assertIn("triton-npu-optimize-knowledge", optimize)
        self.assertIn(
            "../triton-npu-optimize-knowledge/references/pattern_index.md",
            optimize,
        )

    def test_round_performance_skill_points_to_knowledge_symptom_routing_references(self) -> None:
        skill = _read("skills/triton-npu-analyze-round-performance/SKILL.md")
        symptom_index = _read(
            "skills/triton/triton-npu-optimize-knowledge/references/symptom_index.md"
        )
        self.assertIn("triton-npu-optimize-knowledge", skill)
        self.assertIn(
            "../triton-npu-optimize-knowledge/references/symptom_index.md",
            skill,
        )
        self.assertIn("weak-pipeline-overlap", symptom_index)
        self.assertIn("high-transfer-pressure", symptom_index)
```

- [ ] **Step 2: Run the targeted prompt and skill-doc tests to confirm they fail**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.CliPromptTests.test_optimize_prompt_defaults_to_layered_analysis \
  tests.test_optimize_guidance.OptimizeSessionArtifactsManagerTests.test_prepare_shared_guidance_defaults_to_layered_analysis \
  tests.test_generation_contracts.GenerationContractTests.test_optimize_pattern_cards_use_required_sections_and_generated_index \
  tests.test_generation_contracts.GenerationContractTests.test_round_performance_skill_points_to_knowledge_symptom_routing_references -v
```

Expected: FAIL because the prompt text and skill docs still point at the old owners.

- [ ] **Step 3: Update `src/helix/optimize/prompts.py` and both skill docs**

Replace the relevant lines in `src/helix/optimize/prompts.py`:

```python
def layered_analysis_lines(*, round_scope: str) -> list[str]:
    return [
        f"Choose the analysis level for {round_scope} before editing code.",
        "Record the round's primary analysis level separately from its supporting evidence.",
        "Escalate analysis in this order: pattern triage, profiling diagnosis, IR attribution, compiler-source escalation.",
        "Use pattern triage only to decide whether a strong pattern-backed hypothesis already exists.",
        "Use the staged `triton-npu-optimize-knowledge` skill for generic pattern and symptom references.",
        "Read the staged `triton-npu-optimize-knowledge` skill's generated `references/pattern_index.md` before detailed pattern references.",
        "Use the staged code-fact extractor when code structure is still unclear at pattern triage.",
        "Use profiling diagnosis as the default deeper entrypoint when pattern triage is not enough.",
        "Use the staged `triton-npu-optimize-knowledge` skill's symptom cards to narrow pattern candidates after structured profiler or IR evidence exists.",
        "Use IR attribution only after profiler-backed symptoms need explanation.",
        "Use compiler-source escalation only when profiler and IR evidence have already narrowed the issue.",
        "When starting from a deeper level, cite the reused evidence path and explain why the shallower level is already established or insufficient.",
        "Do not begin with blind tiling or launch-parameter search.",
    ]
```

Update the `skills/triton/triton-npu-optimize/SKILL.md` pattern-triage and profiling sections to say:

```markdown
- Use the sibling [`../triton-npu-optimize-knowledge/SKILL.md`](../triton-npu-optimize-knowledge/SKILL.md) as the generic optimize knowledge library.
- Read [`../triton-npu-optimize-knowledge/references/pattern_index.md`](../triton-npu-optimize-knowledge/references/pattern_index.md) before detailed pattern references.
- Read only the one or two most relevant detailed pattern files under [`../triton-npu-optimize-knowledge/references/patterns/`](../triton-npu-optimize-knowledge/references/patterns/) after the generated index has narrowed the candidate set.
- When the kernel is structurally matmul-like, read [`../triton-npu-optimize-knowledge/references/patterns/classic-matmul.md`](../triton-npu-optimize-knowledge/references/patterns/classic-matmul.md) before rewriting the hot loop.
- Use the sibling knowledge skill's symptom cards to narrow pattern candidates after structured profiler or IR evidence exists, rather than rereading the whole pattern library.
```

Update the relevant lines in `skills/triton-npu-analyze-round-performance/SKILL.md` to say:

```markdown
Read [`../triton-npu-optimize-knowledge/references/symptom_index.md`](../triton-npu-optimize-knowledge/references/symptom_index.md) when structured profile or IR evidence is available and you need symptom cards to narrow likely pattern directions before returning to detailed pattern references.

6. Use [`../triton-npu-optimize-knowledge/references/symptom_index.md`](../triton-npu-optimize-knowledge/references/symptom_index.md) and the matching symptom cards to narrow the current hypothesis.
   - Start from the symptom index, then read only the one or two symptom cards under [`../triton-npu-optimize-knowledge/references/symptoms/`](../triton-npu-optimize-knowledge/references/symptoms/) that best match the extracted evidence.
   - Use symptom cards as routing aids, not as a replacement for the underlying profile or IR evidence.
```

- [ ] **Step 4: Re-run the targeted prompt and guidance suites**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.CliPromptTests.test_optimize_prompt_defaults_to_layered_analysis \
  tests.test_optimize_guidance.OptimizeSessionArtifactsManagerTests.test_prepare_shared_guidance_defaults_to_layered_analysis \
  tests.test_generation_contracts.GenerationContractTests.test_optimize_pattern_cards_use_required_sections_and_generated_index \
  tests.test_generation_contracts.GenerationContractTests.test_round_performance_skill_points_to_knowledge_symptom_routing_references -v
```

Expected: PASS

- [ ] **Step 5: Commit the reference rewiring**

```bash
git add \
  skills/triton/triton-npu-optimize/SKILL.md \
  skills/triton-npu-analyze-round-performance/SKILL.md \
  src/helix/optimize/prompts.py \
  tests/test_cli.py \
  tests/test_optimize_guidance.py \
  tests/test_generation_contracts.py
git commit -m "refactor: repoint optimize knowledge references"
```

### Task 4: Update Durable Rules, Remove Old Copies, And Verify The Split

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/notes/2026-04-29-optimize-pattern-card-authoring.md`
- Create: `docs/notes/2026-04-30-optimize-symptom-card-authoring.md`
- Delete: `skills/triton/triton-npu-optimize/references/pattern_index.md`
- Delete: `skills/triton/triton-npu-optimize/references/patterns/*.md`
- Delete: `skills/triton/triton-npu-optimize/scripts/build_pattern_index.py`
- Delete: `skills/triton-npu-analyze-round-performance/references/symptom_index.md`
- Delete: `skills/triton-npu-analyze-round-performance/references/symptoms/*.md`
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Add failing AGENTS and authoring-note assertions**

Extend `tests/test_generation_contracts.py` with:

```python
    def test_agents_declares_knowledge_skill_as_generic_pattern_and_symptom_source(self) -> None:
        agents = _read("AGENTS.md")
        self.assertIn(
            "skills/triton/triton-npu-optimize-knowledge/references/patterns/*.md",
            agents,
        )
        self.assertIn(
            "skills/triton/triton-npu-optimize-knowledge/references/symptoms/*.md",
            agents,
        )
        self.assertIn("## Evidence To Confirm", agents)
        self.assertIn("## Candidate Pattern Directions", agents)

    def test_pattern_and_symptom_authoring_notes_point_to_knowledge_skill(self) -> None:
        pattern_note = _read("docs/notes/2026-04-29-optimize-pattern-card-authoring.md")
        symptom_note = _read("docs/notes/2026-04-30-optimize-symptom-card-authoring.md")

        self.assertIn("skills/triton/triton-npu-optimize-knowledge/references/patterns/", pattern_note)
        self.assertIn("skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py", pattern_note)
        self.assertIn("skills/triton/triton-npu-optimize-knowledge/references/symptoms/", symptom_note)
        self.assertIn("build_symptom_index.py", symptom_note)
```

- [ ] **Step 2: Run the focused doc-contract tests and confirm they fail**

Run:

```bash
uv run python -m unittest \
  tests.test_generation_contracts.GenerationContractTests.test_agents_declares_knowledge_skill_as_generic_pattern_and_symptom_source \
  tests.test_generation_contracts.GenerationContractTests.test_pattern_and_symptom_authoring_notes_point_to_knowledge_skill -v
```

Expected: FAIL because `AGENTS.md` and the symptom authoring note have not been updated yet.

- [ ] **Step 3: Update AGENTS and authoring notes, then remove the old duplicate knowledge copies**

Update the generic source-of-truth rules in `AGENTS.md` to this form:

```markdown
- Treat `skills/triton/triton-npu-optimize-knowledge/references/patterns/*.md` as the authored source of truth for generic optimize patterns; after changing a pattern card, regenerate and commit the checked-in pattern index instead of hand-editing it.
- Treat `skills/triton/triton-npu-optimize-knowledge/references/symptoms/*.md` as the authored source of truth for generic optimize symptoms; after changing a symptom card, regenerate and commit the checked-in symptom index instead of hand-editing it.
...
- Generic optimize pattern cards under `skills/triton/triton-npu-optimize-knowledge/references/patterns/` are authored Markdown sources, while `skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md` is generated and must be regenerated after editing a pattern card instead of hand-edited.
- Generic optimize symptom cards under `skills/triton/triton-npu-optimize-knowledge/references/symptoms/` are authored Markdown sources, while `skills/triton/triton-npu-optimize-knowledge/references/symptom_index.md` is generated and must be regenerated after editing a symptom card instead of hand-edited.
- Every generic optimize pattern card defined in `skills/triton/triton-npu-optimize-knowledge/references/patterns/` must include `## Summary` and `## Use When`; it may additionally use `## Avoid When`, `## Signals`, `## Related Patterns`, and `## What To Verify After Applying`, with optional `### Code`, `### Profile`, and `### IR` under `## Signals`.
- Every generic optimize symptom card defined in `skills/triton/triton-npu-optimize-knowledge/references/symptoms/` must include `## Summary`, `## Evidence To Confirm`, and `## Candidate Pattern Directions`; it may additionally use `## Common Non-Matches`.
```

Update `docs/notes/2026-04-29-optimize-pattern-card-authoring.md` so the path and regeneration commands use the knowledge skill:

```markdown
The Markdown files under `skills/triton/triton-npu-optimize-knowledge/references/patterns/` are the authored source of truth for generic optimize pattern knowledge.

Do not hand-edit `skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md`.

python3 skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py \
  --patterns-dir skills/triton/triton-npu-optimize-knowledge/references/patterns \
  --output skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md
```

Create `docs/notes/2026-04-30-optimize-symptom-card-authoring.md` with:

````markdown
# Optimize Symptom Card Authoring

The Markdown files under `skills/triton/triton-npu-optimize-knowledge/references/symptoms/` are the authored source of truth for generic optimize symptom knowledge.

Do not hand-edit `skills/triton/triton-npu-optimize-knowledge/references/symptom_index.md`. It is generated from the symptom cards in that directory.

## Authoring Contract

Each symptom card must include:

- `## Summary`
- `## Evidence To Confirm`
- `## Candidate Pattern Directions`

Each symptom card may additionally include:

- `## Common Non-Matches`

## Regenerating The Index

```bash
python3 skills/triton/triton-npu-optimize-knowledge/scripts/build_symptom_index.py \
  --symptoms-dir skills/triton/triton-npu-optimize-knowledge/references/symptoms \
  --output skills/triton/triton-npu-optimize-knowledge/references/symptom_index.md
```
````

After docs are updated, remove the old duplicate ownership paths:

```bash
git rm skills/triton/triton-npu-optimize/references/pattern_index.md
git rm -r skills/triton/triton-npu-optimize/references/patterns
git rm skills/triton/triton-npu-optimize/scripts/build_pattern_index.py
git rm skills/triton-npu-analyze-round-performance/references/symptom_index.md
git rm -r skills/triton-npu-analyze-round-performance/references/symptoms
```

- [ ] **Step 4: Regenerate both indexes and run the full verification set**

Run:

```bash
python3 skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py \
  --patterns-dir skills/triton/triton-npu-optimize-knowledge/references/patterns \
  --output skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md \
  --check

python3 skills/triton/triton-npu-optimize-knowledge/scripts/build_symptom_index.py \
  --symptoms-dir skills/triton/triton-npu-optimize-knowledge/references/symptoms \
  --output skills/triton/triton-npu-optimize-knowledge/references/symptom_index.md \
  --check

uv run python -m unittest \
  tests.test_optimize_pattern_tools \
  tests.test_generation_contracts \
  tests.test_optimize_guidance \
  tests.test_cli \
  tests.test_optimize_runtime -v

bash scripts/run-skill-script-pyright.sh \
  skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py

bash scripts/run-skill-script-pyright.sh \
  skills/triton/triton-npu-optimize-knowledge/scripts/build_symptom_index.py

git diff --check
```

Expected:

- both `--check` commands exit `0`
- all listed `unittest` suites PASS
- both pyright wrapper runs report `0 errors, 0 warnings, 0 informations`
- `git diff --check` prints nothing

- [ ] **Step 5: Commit the ownership closure**

```bash
git add \
  AGENTS.md \
  docs/notes/2026-04-29-optimize-pattern-card-authoring.md \
  docs/notes/2026-04-30-optimize-symptom-card-authoring.md \
  tests/test_generation_contracts.py \
  skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md \
  skills/triton/triton-npu-optimize-knowledge/references/symptom_index.md
git commit -m "refactor: split generic optimize knowledge into dedicated skill"
```
