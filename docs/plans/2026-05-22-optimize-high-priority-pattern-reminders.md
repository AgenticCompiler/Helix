# Optimize High-Priority Pattern Reminders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add generated high-priority optimize-pattern reminders to optimize memory files, keep pattern cards as the single source of truth for both pattern indexes and reminder text, support `v1`/`v2`/`v3` generic optimize knowledge selection at runtime, and promote `grid-flatten-and-ub-buffering` with task-kind-aware core-count guidance and explicit cube/vector fallback defaults.

**Architecture:** Refactor each selected generic optimize knowledge tree so a new skill-side `pattern_catalog.py` helper owns pattern-card parsing, priority filtering, index generation, and reminder-line generation. Then add a runtime adapter in `src/helix/optimize/` that resolves the actually selected generic optimize knowledge source from staged-skill metadata, loads the selected helper through the existing skill-loader bridge, and injects a compact generated reminder block into temporary optimize memory files without hardcoding a second pattern list.

**Tech Stack:** Python 3, `unittest`, existing skill-loader/resource helpers, optimize session-artifact rendering, checked-in skill Markdown under `skills/`, and generated pattern indexes

---

## File Map

- Create: `skills/triton/triton-npu-optimize-knowledge/scripts/pattern_catalog.py`
- Create: `skills/triton/triton-npu-optimize-knowledge-v2/scripts/pattern_catalog.py`
- Create: `skills/triton/triton-npu-optimize-knowledge-v3/scripts/pattern_catalog.py`
- Create: `src/helix/optimize/pattern_reminders.py`
- Modify: `skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py`
- Modify: `skills/triton/triton-npu-optimize-knowledge-v2/scripts/build_pattern_index.py`
- Modify: `skills/triton/triton-npu-optimize-knowledge-v3/scripts/build_pattern_index.py`
- Modify: `skills/triton/triton-npu-optimize-knowledge/references/patterns/grid-flatten-and-ub-buffering.md`
- Modify: `skills/triton/triton-npu-optimize-knowledge-v2/references/patterns/grid-flatten-and-ub-buffering.md`
- Modify: `skills/triton/triton-npu-optimize-knowledge-v3/references/patterns/grid-flatten-and-ub-buffering.md`
- Modify: `skills/triton/triton-npu-optimize-knowledge-v2/references/patterns/autotune.md`
- Modify: `skills/triton/triton-npu-optimize-knowledge-v3/references/patterns/autotune.md`
- Modify: `skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md`
- Modify: `skills/triton/triton-npu-optimize-knowledge-v2/references/pattern_index.md`
- Modify: `skills/triton/triton-npu-optimize-knowledge-v3/references/pattern_index.md`
- Modify: `src/helix/optimize/memory_file.py`
- Modify: `src/helix/optimize/session_artifacts.py`
- Modify: `src/helix/optimize/execution.py`
- Modify: `tests/test_optimize_pattern_tools.py`
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_optimize_runtime.py`

## Task 1: Add Failing Pattern-Catalog And Pattern-Content Tests

**Files:**
- Modify: `tests/test_optimize_pattern_tools.py`
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Add failing helper-API coverage for `v1`, `v2`, and `v3` pattern catalogs**

Append these tests to `tests/test_optimize_pattern_tools.py`:

```python
    def test_v1_pattern_catalog_builds_high_priority_reminder_lines(self) -> None:
        module = _load_skill_script(
            "skills/triton/triton-npu-optimize-knowledge/scripts/pattern_catalog.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            patterns_dir = Path(tmp)
            (patterns_dir / "grid.md").write_text(
                """---
id: grid-flatten-and-ub-buffering
priority: high
---

# Grid Flattening And UB Buffering Pattern

## Summary

Flatten logical work items onto physical cores.

## Use When

- Task count is much larger than core count.
""",
                encoding="utf-8",
            )
            (patterns_dir / "tiling.md").write_text(
                """# Tiling Pattern

## Summary

Keep peak footprint bounded.

## Use When

- UB pressure is dominant.
""",
                encoding="utf-8",
            )

            cards = module.list_high_priority_pattern_cards(patterns_dir)
            reminder_lines = module.build_high_priority_reminder_lines(patterns_dir)

            self.assertEqual([card.identifier for card in cards], ["grid-flatten-and-ub-buffering"])
            self.assertEqual(
                reminder_lines,
                ["`grid-flatten-and-ub-buffering`: Flatten logical work items onto physical cores."],
            )

    def test_v2_pattern_catalog_renders_high_priority_section_before_full_summaries(self) -> None:
        module = _load_skill_script(
            "skills/triton/triton-npu-optimize-knowledge-v2/scripts/pattern_catalog.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            patterns_dir = Path(tmp)
            (patterns_dir / "autotune.md").write_text(
                """---
priority: high
---

# Autotune Pattern

## Summary

Use bounded config search when structure is already sound.

## Use When

- Kernel structure is already reasonable.
""",
                encoding="utf-8",
            )
            rendered = module.build_index_text(patterns_dir)
            self.assertIn("## High Priority Patterns", rendered)
            self.assertIn("### `autotune`", rendered)
            self.assertLess(
                rendered.index("## High Priority Patterns"),
                rendered.index("## Generated Pattern Summaries"),
            )

    def test_v3_pattern_catalog_rejects_invalid_priority_value(self) -> None:
        module = _load_skill_script(
            "skills/triton/triton-npu-optimize-knowledge-v3/scripts/pattern_catalog.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            patterns_dir = Path(tmp)
            (patterns_dir / "demo.md").write_text(
                """---
priority: urgent
---

# Demo Pattern

## Summary

Short summary.

## Use When

- Stable trigger.
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "demo.md has invalid priority 'urgent'"
            ):
                module.build_index_text(patterns_dir)
```

- [ ] **Step 2: Add failing checked-in-index parity tests for `v2` and `v3`**

Still in `tests/test_optimize_pattern_tools.py`, add:

```python
    def test_checked_in_v2_pattern_index_matches_generator(self) -> None:
        module = _load_skill_script(
            "skills/triton/triton-npu-optimize-knowledge-v2/scripts/build_pattern_index.py"
        )
        patterns_dir = (
            REPO_ROOT
            / "skills"
            / "triton-npu-optimize-knowledge-v2"
            / "references"
            / "patterns"
        )
        generated = module.build_index_text(patterns_dir)
        checked_in = (
            REPO_ROOT
            / "skills"
            / "triton-npu-optimize-knowledge-v2"
            / "references"
            / "pattern_index.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(generated, checked_in)

    def test_checked_in_v3_pattern_index_matches_generator(self) -> None:
        module = _load_skill_script(
            "skills/triton/triton-npu-optimize-knowledge-v3/scripts/build_pattern_index.py"
        )
        patterns_dir = (
            REPO_ROOT
            / "skills"
            / "triton-npu-optimize-knowledge-v3"
            / "references"
            / "patterns"
        )
        generated = module.build_index_text(patterns_dir)
        checked_in = (
            REPO_ROOT
            / "skills"
            / "triton-npu-optimize-knowledge-v3"
            / "references"
            / "pattern_index.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(generated, checked_in)
```

- [ ] **Step 3: Add failing contract coverage for the updated grid-flattening card guidance**

Extend `tests/test_generation_contracts.py` with assertions like:

```python
    def test_grid_flatten_pattern_documents_runtime_query_and_core_count_fallbacks(self) -> None:
        grid = _read(
            "skills/triton/triton-npu-optimize-knowledge/references/patterns/grid-flatten-and-ub-buffering.md"
        )
        self.assertIn("torch.npu.get_device_properties", grid)
        self.assertIn("cube cores: `24`", grid)
        self.assertIn("vector cores: `48`", grid)
        self.assertIn("`cube`-like operators", grid)
        self.assertIn("`vector`-like operators", grid)
        self.assertIn("`mix` operators", grid)
```

- [ ] **Step 4: Run the new tests and confirm they fail**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_v1_pattern_catalog_builds_high_priority_reminder_lines \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_v2_pattern_catalog_renders_high_priority_section_before_full_summaries \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_v3_pattern_catalog_rejects_invalid_priority_value \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_checked_in_v2_pattern_index_matches_generator \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_checked_in_v3_pattern_index_matches_generator \
  tests.test_generation_contracts.GenerationContractsTests.test_grid_flatten_pattern_documents_runtime_query_and_core_count_fallbacks \
  -v
```

Expected: `FAIL` because `pattern_catalog.py` does not exist yet, `v2`/`v3` builders do not yet render high-priority sections, and the checked-in grid-flattening card does not yet document runtime query plus cube/vector fallback guidance.

## Task 2: Implement Pattern Catalog Helpers, Update High-Priority Cards, And Regenerate Indexes

**Files:**
- Create: `skills/triton/triton-npu-optimize-knowledge/scripts/pattern_catalog.py`
- Create: `skills/triton/triton-npu-optimize-knowledge-v2/scripts/pattern_catalog.py`
- Create: `skills/triton/triton-npu-optimize-knowledge-v3/scripts/pattern_catalog.py`
- Modify: `skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py`
- Modify: `skills/triton/triton-npu-optimize-knowledge-v2/scripts/build_pattern_index.py`
- Modify: `skills/triton/triton-npu-optimize-knowledge-v3/scripts/build_pattern_index.py`
- Modify: `skills/triton/triton-npu-optimize-knowledge/references/patterns/grid-flatten-and-ub-buffering.md`
- Modify: `skills/triton/triton-npu-optimize-knowledge-v2/references/patterns/grid-flatten-and-ub-buffering.md`
- Modify: `skills/triton/triton-npu-optimize-knowledge-v3/references/patterns/grid-flatten-and-ub-buffering.md`
- Modify: `skills/triton/triton-npu-optimize-knowledge-v2/references/patterns/autotune.md`
- Modify: `skills/triton/triton-npu-optimize-knowledge-v3/references/patterns/autotune.md`
- Modify: `skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md`
- Modify: `skills/triton/triton-npu-optimize-knowledge-v2/references/pattern_index.md`
- Modify: `skills/triton/triton-npu-optimize-knowledge-v3/references/pattern_index.md`

- [ ] **Step 1: Create the `v1` pattern-catalog helper**

Create `skills/triton/triton-npu-optimize-knowledge/scripts/pattern_catalog.py` with the parser logic currently embedded in `build_pattern_index.py`, plus the new reminder API:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REQUIRED_SECTIONS = ("Summary", "Use When")
VALID_PRIORITIES = ("high", "normal")
_FRONTMATTER_BOUNDARY = "---"
_SECTION_HEADING_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$", flags=re.MULTILINE)
_SUBSECTION_HEADING_RE = re.compile(r"^###\s+(?P<title>.+?)\s*$", flags=re.MULTILINE)
_LIST_ITEM_RE = re.compile(r"^(?:-\s+|\d+\.\s+)(?P<item>.+?)\s*$")

@dataclass
class PatternCard:
    identifier: str
    title: str
    priority: str
    summary: str
    use_when: list[str]
    avoid_when: list[str]
    signals_code: list[str]
    signals_profile: list[str]
    signals_ir: list[str]
    related_patterns: list[str]
    verify_after_applying: list[str]
    source_path: Path

def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    ...

def parse_pattern_card(path: Path) -> PatternCard:
    ...

def list_high_priority_pattern_cards(patterns_dir: Path) -> list[PatternCard]:
    return [
        card for card in _load_cards(patterns_dir) if card.priority == "high"
    ]

def build_high_priority_reminder_lines(patterns_dir: Path) -> list[str]:
    return [
        f"`{card.identifier}`: {card.summary}"
        for card in list_high_priority_pattern_cards(patterns_dir)
    ]

def build_index_text(patterns_dir: Path) -> str:
    cards = _load_cards(patterns_dir)
    high_priority_cards = [card for card in cards if card.priority == "high"]
    lines = [
        "# Optimization Pattern Index",
        "",
        "Use this file to choose optimization directions before reading any detailed pattern reference.",
        "",
        "Read this generated index first. Then read only the one or two most relevant detailed pattern files for the current bottleneck.",
        "",
        "Before scanning the full list, first analyze whether the operator matches any high-priority patterns below. If it does, try those directions first.",
        "",
        "## High Priority Patterns",
        "",
    ]
    if high_priority_cards:
        for card in high_priority_cards:
            lines.append(f"### `{card.identifier}`")
            lines.append("")
            lines.append(f"- Summary: {card.summary}")
            lines.append(f"- Source: [{card.source_path.name}](patterns/{card.source_path.name})")
            lines.append("")
    else:
        lines.append("- None.")
        lines.append("")
    lines.extend(["## Generated Pattern Summaries", ""])
    ...
    return "\\n".join(lines).rstrip() + "\\n"
```

- [ ] **Step 2: Create matching `v2` and `v3` helpers with tree-specific full-index rendering**

Create `skills/triton/triton-npu-optimize-knowledge-v2/scripts/pattern_catalog.py` and `skills/triton/triton-npu-optimize-knowledge-v3/scripts/pattern_catalog.py` by moving each tree’s current parser/renderer logic out of its `build_pattern_index.py`, then add:

```python
VALID_PRIORITIES = ("high", "normal")

def list_high_priority_pattern_cards(patterns_dir: Path) -> list[PatternCard]:
    return [
        card for card in _load_cards(patterns_dir) if card.priority == "high"
    ]

def build_high_priority_reminder_lines(patterns_dir: Path) -> list[str]:
    return [
        f"`{card.identifier}`: {card.summary}"
        for card in list_high_priority_pattern_cards(patterns_dir)
    ]
```

Keep `v2`/`v3` `build_index_text()` compatible with their current full-summary layout, but prepend the dedicated `## High Priority Patterns` section before `## Generated Pattern Summaries`.

- [ ] **Step 3: Reduce each `build_pattern_index.py` to a thin wrapper around the new helper**

Replace the body of each builder with a thin CLI wrapper. For example, `skills/triton/triton-npu-optimize-knowledge-v2/scripts/build_pattern_index.py` should look like:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from pattern_catalog import build_index_text

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the optimize pattern index from pattern cards.")
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

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Update high-priority pattern metadata and grid-flattening guidance in the selected trees**

Make these content changes:

- `skills/triton/triton-npu-optimize-knowledge/references/patterns/grid-flatten-and-ub-buffering.md`
  - add `priority: high`
  - add the `torch.npu.current_device()` / `torch.npu.get_device_properties()` query snippet
  - add fallback guidance `cube cores: 24`, `vector cores: 48`
  - add task-kind-aware `cube` / `vector` / `mix` guidance
- `skills/triton/triton-npu-optimize-knowledge-v2/references/patterns/grid-flatten-and-ub-buffering.md`
  - add `priority: high`
  - add the same runtime query and fallback guidance in the v2 card style
- `skills/triton/triton-npu-optimize-knowledge-v3/references/patterns/grid-flatten-and-ub-buffering.md`
  - add `priority: high`
  - add the same runtime query and fallback guidance in the v3 card style
- `skills/triton/triton-npu-optimize-knowledge-v2/references/patterns/autotune.md`
  - add `priority: high`
- `skills/triton/triton-npu-optimize-knowledge-v3/references/patterns/autotune.md`
  - add `priority: high`

Use prose like this in each grid-flattening card:

```markdown
### Runtime core-count discovery

If runtime device inspection is available, gather evidence first:

```python
import torch

print(torch.npu.device_count())
device = torch.npu.current_device()
props = torch.npu.get_device_properties(device)
print(props)
```

Use explicit cube/vector core-count facts from `props` when they are available. If the query succeeds but only confirms device identity, treat it as chip-identification evidence and keep launch counts as hypotheses. If the query fails or does not expose explicit core counts, fall back to current-target defaults:

- cube cores: `24`
- vector cores: `48`

### Task-kind-aware launch hypotheses

- `cube`-like operators: start with cube-core-count-aligned launch hypotheses
- `vector`-like operators: start with vector-core-count-aligned launch hypotheses
- `mix` operators: keep both cube-count and vector-count launch sizes as candidates and choose by benchmark/profile evidence
```

- [ ] **Step 5: Regenerate the checked-in pattern indexes**

Run:

```bash
uv run python skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py \
  --patterns-dir skills/triton/triton-npu-optimize-knowledge/references/patterns \
  --output skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md

uv run python skills/triton/triton-npu-optimize-knowledge-v2/scripts/build_pattern_index.py \
  --patterns-dir skills/triton/triton-npu-optimize-knowledge-v2/references/patterns \
  --output skills/triton/triton-npu-optimize-knowledge-v2/references/pattern_index.md

uv run python skills/triton/triton-npu-optimize-knowledge-v3/scripts/build_pattern_index.py \
  --patterns-dir skills/triton/triton-npu-optimize-knowledge-v3/references/patterns \
  --output skills/triton/triton-npu-optimize-knowledge-v3/references/pattern_index.md
```

Expected: all three checked-in `pattern_index.md` files are rewritten deterministically and now include `## High Priority Patterns`.

- [ ] **Step 6: Run focused tests and required file-scoped pyright checks**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_pattern_tools \
  tests.test_generation_contracts \
  -v
```

Then run the required file-scoped skill-script pyright checks:

```bash
bash scripts/run-skill-script-pyright.sh skills/triton/triton-npu-optimize-knowledge/scripts/pattern_catalog.py
bash scripts/run-skill-script-pyright.sh skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py
bash scripts/run-skill-script-pyright.sh skills/triton/triton-npu-optimize-knowledge-v2/scripts/pattern_catalog.py
bash scripts/run-skill-script-pyright.sh skills/triton/triton-npu-optimize-knowledge-v2/scripts/build_pattern_index.py
bash scripts/run-skill-script-pyright.sh skills/triton/triton-npu-optimize-knowledge-v3/scripts/pattern_catalog.py
bash scripts/run-skill-script-pyright.sh skills/triton/triton-npu-optimize-knowledge-v3/scripts/build_pattern_index.py
```

Expected: all focused tests pass and each modified skill-side Python file passes the required strict pyright check.

- [ ] **Step 7: Commit the pattern-catalog and card/index changes**

Run:

```bash
git add \
  skills/triton/triton-npu-optimize-knowledge/scripts/pattern_catalog.py \
  skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py \
  skills/triton/triton-npu-optimize-knowledge/references/patterns/grid-flatten-and-ub-buffering.md \
  skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md \
  skills/triton/triton-npu-optimize-knowledge-v2/scripts/pattern_catalog.py \
  skills/triton/triton-npu-optimize-knowledge-v2/scripts/build_pattern_index.py \
  skills/triton/triton-npu-optimize-knowledge-v2/references/patterns/grid-flatten-and-ub-buffering.md \
  skills/triton/triton-npu-optimize-knowledge-v2/references/patterns/autotune.md \
  skills/triton/triton-npu-optimize-knowledge-v2/references/pattern_index.md \
  skills/triton/triton-npu-optimize-knowledge-v3/scripts/pattern_catalog.py \
  skills/triton/triton-npu-optimize-knowledge-v3/scripts/build_pattern_index.py \
  skills/triton/triton-npu-optimize-knowledge-v3/references/patterns/grid-flatten-and-ub-buffering.md \
  skills/triton/triton-npu-optimize-knowledge-v3/references/patterns/autotune.md \
  skills/triton/triton-npu-optimize-knowledge-v3/references/pattern_index.md \
  tests/test_optimize_pattern_tools.py \
  tests/test_generation_contracts.py
git commit -m "feat: add shared high-priority pattern catalog helpers"
```

## Task 3: Add Failing Runtime Reminder Selection And Memory-File Rendering Tests

**Files:**
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Add failing tests for generic optimize knowledge source resolution**

Add runtime-adapter coverage to `tests/test_optimize_guidance.py`:

```python
from helix.optimize.pattern_reminders import (
    resolve_generic_optimize_knowledge_source,
)

    def test_resolve_generic_optimize_knowledge_source_defaults_to_v1_name(self) -> None:
        self.assertEqual(
            resolve_generic_optimize_knowledge_source(
                staged_skill_names=("triton-npu-optimize", "triton-npu-optimize-knowledge"),
                staged_skill_sources=None,
            ),
            "triton-npu-optimize-knowledge",
        )

    def test_resolve_generic_optimize_knowledge_source_prefers_source_override(self) -> None:
        self.assertEqual(
            resolve_generic_optimize_knowledge_source(
                staged_skill_names=("triton-npu-optimize", "triton-npu-optimize-knowledge"),
                staged_skill_sources={
                    "triton-npu-optimize-knowledge": "triton-npu-optimize-knowledge-v3"
                },
            ),
            "triton-npu-optimize-knowledge-v3",
        )
```

- [ ] **Step 2: Add failing memory-file rendering tests for generated reminders**

Still in `tests/test_optimize_guidance.py`, add:

```python
    def test_prepare_unsupervised_session_renders_high_priority_pattern_reminders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\\n", encoding="utf-8")

            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_unsupervised_session(
                workdir,
                operator_path=operator,
                agent_name="codex",
                test_mode="differential",
                bench_mode="standalone",
                generic_optimize_knowledge_source="triton-npu-optimize-knowledge",
            )

            guidance_content = state.guidance_path.read_text(encoding="utf-8")
            self.assertIn("High-priority pattern reminders:", guidance_content)
            self.assertIn("`autotune`:", guidance_content)
            self.assertIn("`grid-flatten-and-ub-buffering`:", guidance_content)
            self.assertIn("full current high-priority list", guidance_content)

    def test_prepare_shared_guidance_omits_reminder_block_when_no_patterns_are_high_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            manager = OptimizeSessionArtifactsManager()
            with patch(
                "helix.optimize.memory_file.render_high_priority_pattern_reminder_block",
                return_value="",
            ):
                state = manager.prepare_supervised_session(
                    workdir,
                    agent_name="codex",
                    generic_optimize_knowledge_source="triton-npu-optimize-knowledge",
                )
                shared_content = state.guidance_path.read_text(encoding="utf-8")
                self.assertNotIn("High-priority pattern reminders:", shared_content)
```

- [ ] **Step 3: Add failing execution plumbing tests so optimize artifacts receive the selected source tree**

Add a focused test to `tests/test_optimize_runtime.py`:

```python
    def test_execute_unsupervised_optimize_passes_selected_generic_knowledge_source_to_artifacts(self) -> None:
        request = AgentRequest(
            command_kind=CommandKind.OPTIMIZE,
            input_path=workdir / "kernel.py",
            operator_path=workdir / "kernel.py",
            output_path=workdir / "out.py",
            test_mode="differential",
            bench_mode="standalone",
            interact=False,
            verbose=False,
            show_output=False,
            force_overwrite=False,
            agent_name="codex",
            skill_name="triton-npu-optimize",
            prompt="prompt",
            workdir=workdir,
            staged_skill_names=("triton-npu-optimize", "triton-npu-optimize-knowledge"),
            staged_skill_sources={
                "triton-npu-optimize-knowledge": "triton-npu-optimize-knowledge-v3"
            },
        )
        ...
        execute_unsupervised_optimize(...)
        self.assertEqual(
            artifacts_manager.prepare_unsupervised_session.call_args.kwargs[
                "generic_optimize_knowledge_source"
            ],
            "triton-npu-optimize-knowledge-v3",
        )
```

- [ ] **Step 4: Run the new tests and confirm they fail**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_guidance \
  tests.test_optimize_runtime \
  -v
```

Expected: `FAIL` because there is no runtime reminder adapter yet, memory-file preparation does not accept a selected generic knowledge source, and optimize execution does not yet thread that selection into session-artifact rendering.

## Task 4: Implement Runtime Reminder Adapter And Wire It Into Optimize Memory-File Rendering

**Files:**
- Create: `src/helix/optimize/pattern_reminders.py`
- Modify: `src/helix/optimize/memory_file.py`
- Modify: `src/helix/optimize/session_artifacts.py`
- Modify: `src/helix/optimize/execution.py`

- [ ] **Step 1: Create the runtime reminder adapter**

Create `src/helix/optimize/pattern_reminders.py` with:

```python
from __future__ import annotations

from pathlib import Path

from helix.resources import skills_root
from helix.skill_loader import load_skill_script_module

_GENERIC_OPTIMIZE_KNOWLEDGE_SKILL = "triton-npu-optimize-knowledge"

def resolve_generic_optimize_knowledge_source(
    *,
    staged_skill_names: tuple[str, ...] | None,
    staged_skill_sources: dict[str, str] | None,
) -> str | None:
    if staged_skill_names is None:
        return None
    if _GENERIC_OPTIMIZE_KNOWLEDGE_SKILL not in staged_skill_names:
        return None
    if staged_skill_sources and _GENERIC_OPTIMIZE_KNOWLEDGE_SKILL in staged_skill_sources:
        return staged_skill_sources[_GENERIC_OPTIMIZE_KNOWLEDGE_SKILL]
    return _GENERIC_OPTIMIZE_KNOWLEDGE_SKILL

def _patterns_dir_for_skill(skill_name: str) -> Path:
    patterns_dir = skills_root() / skill_name / "references" / "patterns"
    if not patterns_dir.is_dir():
        raise FileNotFoundError(f"Optimize knowledge patterns directory does not exist: {patterns_dir}")
    return patterns_dir

def build_high_priority_pattern_reminder_lines(skill_name: str) -> list[str]:
    module = load_skill_script_module(skill_name, "pattern_catalog")
    patterns_dir = _patterns_dir_for_skill(skill_name)
    reminder_lines = module.build_high_priority_reminder_lines(patterns_dir)
    return [str(line) for line in reminder_lines]

def render_high_priority_pattern_reminder_block(skill_name: str | None) -> str:
    if skill_name is None:
        return ""
    reminder_lines = build_high_priority_pattern_reminder_lines(skill_name)
    if not reminder_lines:
        return ""
    bullets = "\\n".join(f"- {line}" for line in reminder_lines)
    return (
        "High-priority pattern reminders:\\n"
        f"{bullets}\\n"
        "Read the staged optimize knowledge `references/pattern_index.md` for the full current high-priority list and detailed routing.\\n"
    )
```

- [ ] **Step 2: Inject reminder-block rendering into the memory-file templates**

Modify `src/helix/optimize/memory_file.py`:

```python
from helix.optimize.pattern_reminders import (
    render_high_priority_pattern_reminder_block,
)
```

Extend `prepare_unsupervised()` and `prepare_shared()` signatures:

```python
        generic_optimize_knowledge_source: str | None = None,
```

Thread the new parameter through `_render_unsupervised_guidance()` and `_render_shared_guidance()`, then append a new template placeholder:

```python
        Optimize the operator at `{operator_name}`.
        {analysis_block}{compiler_source_block}{cann_ext_api_block}{high_priority_patterns_block}"""
```

Populate it with:

```python
            high_priority_patterns_block=_render_line_block(
                [
                    line
                    for line in render_high_priority_pattern_reminder_block(
                        generic_optimize_knowledge_source
                    ).splitlines()
                ]
            ),
```

- [ ] **Step 3: Thread the selected generic knowledge source through session-artifact preparation**

Modify `src/helix/optimize/session_artifacts.py` so both session-preparation entrypoints accept and forward:

```python
        generic_optimize_knowledge_source: str | None = None,
```

and pass it to `MemoryFileManager.prepare_unsupervised()` / `prepare_shared()`.

- [ ] **Step 4: Resolve and pass the selected source from optimize execution**

Modify `src/helix/optimize/execution.py`:

```python
from helix.optimize.pattern_reminders import (
    resolve_generic_optimize_knowledge_source,
)
```

In both `execute_supervised_optimize()` and `execute_unsupervised_optimize()`:

```python
    generic_optimize_knowledge_source = resolve_generic_optimize_knowledge_source(
        staged_skill_names=request.staged_skill_names,
        staged_skill_sources=request.staged_skill_sources,
    )
```

Then pass:

```python
        generic_optimize_knowledge_source=generic_optimize_knowledge_source,
```

into the matching `prepare_*_session()` call.

- [ ] **Step 5: Run focused runtime tests**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_guidance \
  tests.test_optimize_runtime \
  -v
```

Expected: the new runtime reminder block renders in optimize guidance files for selected knowledge trees, omission stays clean when there are no high-priority lines, and optimize execution passes the selected source tree into the session-artifact layer.

- [ ] **Step 6: Commit the runtime reminder wiring**

Run:

```bash
git add \
  src/helix/optimize/pattern_reminders.py \
  src/helix/optimize/memory_file.py \
  src/helix/optimize/session_artifacts.py \
  src/helix/optimize/execution.py \
  tests/test_optimize_guidance.py \
  tests/test_optimize_runtime.py
git commit -m "feat: render optimize high-priority pattern reminders"
```

## Task 5: Final Verification

**Files:**
- Modify: none expected

- [ ] **Step 1: Re-run the checked-in pattern-index parity tests**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_checked_in_pattern_index_matches_generator \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_checked_in_v2_pattern_index_matches_generator \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_checked_in_v3_pattern_index_matches_generator \
  -v
```

Expected: all three tests pass, confirming the checked-in indexes match the helper-backed generators.

- [ ] **Step 2: Run the standard repository verification commands**

Run:

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m unittest discover -s tests -v
```

Expected: all three commands pass with no new lint, type, or test regressions.

- [ ] **Step 3: If verification fails, make the smallest repair and rerun only the affected checks before rerunning the full verification set**

For example:

```bash
uv run python -m unittest tests.test_optimize_guidance -v
uv run pyright
```

Expected: focused checks pass first, then the full verification set passes cleanly.
