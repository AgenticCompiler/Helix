# Optimize Pattern Priority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional `priority: high|normal` metadata to optimize pattern cards, render a dedicated high-priority section in the generated pattern index, and initially mark `autotune` plus `a5-force-simt-only-discrete-access` as high priority.

**Architecture:** Keep the authored pattern cards as the only source of truth by storing priority in optional frontmatter on each card. Extend the existing pattern-index generator to parse and validate that metadata, default missing values to `normal`, and render a compact `## High Priority Patterns` section ahead of the unchanged full summary list. Update stable authoring rules in `AGENTS.md` and the focused pattern-authoring note so the runtime contract, generated artifact, and human docs stay aligned.

**Tech Stack:** Python 3.11, `unittest`, `pathlib`, Markdown docs, repo-local `uv` commands, file-scoped strict pyright wrapper

---

## File Map

- Modify: `skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py`
  Parse optional priority metadata, validate it, and render the new high-priority section.
- Modify: `tests/test_optimize_pattern_tools.py`
  Add focused generator tests for invalid priority handling and high-priority section rendering.
- Modify: `tests/test_generation_contracts.py`
  Lock in the durable docs contract so `AGENTS.md` and the pattern-authoring note both mention the new priority rule.
- Modify: `AGENTS.md`
  Add the stable pattern-priority authoring rule for this repository.
- Modify: `docs/notes/2026-04-29-optimize-pattern-card-authoring.md`
  Document `priority: high|normal`, the `normal` default, and the generated high-priority index section.
- Modify: `skills/triton/triton-npu-optimize-knowledge/references/patterns/autotune.md`
  Add `priority: high` frontmatter.
- Modify: `skills/triton/triton-npu-optimize-knowledge/references/patterns/a5-force-simt-only-discrete-access.md`
  Add `priority: high` frontmatter.
- Modify: `skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md`
  Regenerated output from the updated generator and card metadata.

### Task 1: Add Failing Generator And Contract Tests

**Files:**
- Modify: `tests/test_optimize_pattern_tools.py`
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Write the failing generator tests for priority parsing and rendering**

Add these tests to `tests/test_optimize_pattern_tools.py`:

```python
    def test_build_index_rejects_invalid_priority_value(self) -> None:
        module = _load_skill_script(
            "skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py"
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

    def test_generated_index_lists_high_priority_patterns_before_full_summaries(self) -> None:
        module = _load_skill_script(
            "skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            patterns_dir = Path(tmp)
            (patterns_dir / "high.md").write_text(
                """---
id: high-pattern
priority: high
---

# High Pattern

## Summary

High summary.

## Use When

- High trigger.
""",
                encoding="utf-8",
            )
            (patterns_dir / "normal.md").write_text(
                """# Normal Pattern

## Summary

Normal summary.

## Use When

- Normal trigger.
""",
                encoding="utf-8",
            )

            rendered = module.build_index_text(patterns_dir)

            self.assertIn("## High Priority Patterns", rendered)
            self.assertIn("### `high-pattern`", rendered)
            self.assertIn("- Summary: High summary.", rendered)
            self.assertIn("[high.md](patterns/high.md)", rendered)
            self.assertLess(
                rendered.index("## High Priority Patterns"),
                rendered.index("## Generated Pattern Summaries"),
            )

    def test_generated_index_renders_none_when_no_patterns_are_high_priority(self) -> None:
        module = _load_skill_script(
            "skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            patterns_dir = Path(tmp)
            (patterns_dir / "normal.md").write_text(
                """# Normal Pattern

## Summary

Normal summary.

## Use When

- Normal trigger.
""",
                encoding="utf-8",
            )

            rendered = module.build_index_text(patterns_dir)

            self.assertIn("## High Priority Patterns", rendered)
            self.assertIn("- None.", rendered)
```

- [ ] **Step 2: Write the failing contract tests for AGENTS and the authoring note**

Update `tests/test_generation_contracts.py`:

```python
    def test_agents_declares_pattern_priority_authoring_rule(self) -> None:
        agents = _read("AGENTS.md")
        self.assertIn("priority: high|normal", agents)
        self.assertIn("default to `normal`", agents)
        self.assertIn("## High Priority Patterns", agents)

    def test_pattern_authoring_note_describes_priority_contract(self) -> None:
        pattern_note = _read("docs/notes/2026-04-29-optimize-pattern-card-authoring.md")
        self.assertIn("priority: high|normal", pattern_note)
        self.assertIn("default to `normal`", pattern_note)
        self.assertIn("## High Priority Patterns", pattern_note)
```

- [ ] **Step 3: Run the focused tests and confirm they fail before implementation**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_build_index_rejects_invalid_priority_value \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_generated_index_lists_high_priority_patterns_before_full_summaries \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_generated_index_renders_none_when_no_patterns_are_high_priority \
  tests.test_generation_contracts.GenerationContractTests.test_agents_declares_pattern_priority_authoring_rule \
  tests.test_generation_contracts.GenerationContractTests.test_pattern_authoring_note_describes_priority_contract -v
```

Expected: `FAIL` because the current generator ignores `priority`, does not render `## High Priority Patterns`, and the stable docs do not yet mention the new metadata contract.

- [ ] **Step 4: Commit the failing tests**

```bash
git add tests/test_optimize_pattern_tools.py tests/test_generation_contracts.py
git commit -m "test: add optimize pattern priority coverage"
```

### Task 2: Implement Priority Parsing And Authoring Rules

**Files:**
- Modify: `skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py`
- Modify: `AGENTS.md`
- Modify: `docs/notes/2026-04-29-optimize-pattern-card-authoring.md`

- [ ] **Step 1: Implement priority parsing and validation in the generator**

Update `skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py` with these changes:

```python
VALID_PRIORITIES = ("high", "normal")
```

Extend `PatternCard`:

```python
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
```

Add a helper and use it from `parse_pattern_card()`:

```python
def _parse_priority(metadata: dict[str, str], source_path: Path) -> str:
    priority = metadata.get("priority", "normal").strip() or "normal"
    if priority not in VALID_PRIORITIES:
        raise ValueError(f"{source_path.name} has invalid priority {priority!r}")
    return priority
```

```python
    return PatternCard(
        identifier=metadata.get("id", path.stem),
        title=_fallback_title(metadata, path, body),
        priority=_parse_priority(metadata, path),
        summary=_first_nonempty_paragraph(sections["Summary"]),
        use_when=_extract_bullets(sections["Use When"]),
        avoid_when=_extract_bullets(sections.get("Avoid When", "")),
        signals_code=_extract_bullets(signals.get("Code", "")),
        signals_profile=_extract_bullets(signals.get("Profile", "")),
        signals_ir=_extract_bullets(signals.get("IR", "")),
        related_patterns=_extract_bullets(sections.get("Related Patterns", "")),
        verify_after_applying=_extract_bullets(
            sections.get("What To Verify After Applying", "")
        ),
        source_path=path,
    )
```

- [ ] **Step 2: Render the new high-priority section before the existing full summary list**

Update `render_index()`:

```python
def render_index(cards: list[PatternCard]) -> str:
    high_priority_cards = [card for card in cards if card.priority == "high"]
    lines = [
        "# Optimization Pattern Index",
        "",
        "Use this file to choose optimization directions before reading any detailed pattern reference.",
        "",
        "Read this generated index first. Then read only the one or two most relevant detailed pattern files for the current bottleneck.",
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

    lines.extend(
        [
            "## Generated Pattern Summaries",
            "",
        ]
    )
    for card in cards:
        lines.append(f"### `{card.identifier}`")
        lines.append("")
        lines.append(f"- Summary: {card.summary}")
        lines.append(f"- Source: [{card.source_path.name}](patterns/{card.source_path.name})")
        if card.use_when:
            lines.append("- Use When:")
            lines.extend(_render_bullets(card.use_when))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 3: Add the durable authoring rule to `AGENTS.md` and the focused note**

Add this rule block to `AGENTS.md` under `## Optimization Patterns`:

```markdown
- Pattern-card frontmatter may include `priority: high|normal`; omit it to default to `normal`.
- `priority` is index-rendering metadata, not a replacement for structured sections in the card body.
- The generated `pattern_index.md` must include a `## High Priority Patterns` section that lists cards marked `high`.
```

Add these bullets to `docs/notes/2026-04-29-optimize-pattern-card-authoring.md`:

```markdown
- pattern-card frontmatter may include `priority: high|normal`
- if omitted, the generator defaults `priority` to `normal`
- `priority` only affects generated index presentation
```

Add this sentence near the generated-index description:

```markdown
The generated `pattern_index.md` also includes a `## High Priority Patterns` section that lists only cards marked `priority: high`.
```

- [ ] **Step 4: Re-run the focused tests and confirm they pass**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_build_index_rejects_invalid_priority_value \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_generated_index_lists_high_priority_patterns_before_full_summaries \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_generated_index_renders_none_when_no_patterns_are_high_priority \
  tests.test_generation_contracts.GenerationContractTests.test_agents_declares_pattern_priority_authoring_rule \
  tests.test_generation_contracts.GenerationContractTests.test_pattern_authoring_note_describes_priority_contract -v
```

Expected: `OK`

- [ ] **Step 5: Run the strict pyright check for the generator**

Run:

```bash
bash scripts/run-skill-script-pyright.sh \
  skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py
```

Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 6: Commit the generator and docs changes**

```bash
git add \
  skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py \
  AGENTS.md \
  docs/notes/2026-04-29-optimize-pattern-card-authoring.md
git commit -m "feat: add optimize pattern priority metadata"
```

### Task 3: Mark High-Priority Cards And Regenerate The Index

**Files:**
- Modify: `skills/triton/triton-npu-optimize-knowledge/references/patterns/autotune.md`
- Modify: `skills/triton/triton-npu-optimize-knowledge/references/patterns/a5-force-simt-only-discrete-access.md`
- Modify: `skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md`
- Modify: `tests/test_optimize_pattern_tools.py`

- [ ] **Step 1: Add one checked-in index regression test for the two explicit high-priority cards**

Append this test to `tests/test_optimize_pattern_tools.py`:

```python
    def test_checked_in_pattern_index_high_priority_section_lists_expected_cards(self) -> None:
        checked_in = (
            REPO_ROOT
            / "skills"
            / "triton-npu-optimize-knowledge"
            / "references"
            / "pattern_index.md"
        ).read_text(encoding="utf-8")

        self.assertIn("## High Priority Patterns", checked_in)
        self.assertIn("### `a5-force-simt-only-discrete-access`", checked_in)
        self.assertIn("### `autotune`", checked_in)
        self.assertNotIn("### `classic-matmul`", checked_in.split("## Generated Pattern Summaries")[0])
```

- [ ] **Step 2: Add `priority: high` frontmatter to the two selected pattern cards**

Update `skills/triton/triton-npu-optimize-knowledge/references/patterns/autotune.md`:

```markdown
---
priority: high
---

# Triton Autotune Pattern
```

Update `skills/triton/triton-npu-optimize-knowledge/references/patterns/a5-force-simt-only-discrete-access.md`:

```markdown
---
priority: high
---

# A5 SIMT-Only Discrete Access Pattern
```

- [ ] **Step 3: Regenerate the checked-in pattern index**

Run:

```bash
python3 skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py \
  --patterns-dir skills/triton/triton-npu-optimize-knowledge/references/patterns \
  --output skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md
```

Expected: command exits `0` and rewrites the checked-in index with the new `## High Priority Patterns` section.

- [ ] **Step 4: Run the checked-in generator validations**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_checked_in_pattern_index_matches_generator \
  tests.test_optimize_pattern_tools.PatternRoutingToolTests.test_checked_in_pattern_index_high_priority_section_lists_expected_cards -v
```

Expected: `OK`

Run:

```bash
python3 skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py \
  --patterns-dir skills/triton/triton-npu-optimize-knowledge/references/patterns \
  --output skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md \
  --check
```

Expected: exit code `0`

- [ ] **Step 5: Commit the card metadata and regenerated index**

```bash
git add \
  skills/triton/triton-npu-optimize-knowledge/references/patterns/autotune.md \
  skills/triton/triton-npu-optimize-knowledge/references/patterns/a5-force-simt-only-discrete-access.md \
  skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md \
  tests/test_optimize_pattern_tools.py
git commit -m "docs: highlight high-priority optimize patterns"
```

## Self-Review

- Spec coverage: Task 1 covers failing tests for generator and docs contract, Task 2 covers priority parsing plus stable authoring rules, and Task 3 covers the initial rollout where only `autotune` and `a5-force-simt-only-discrete-access` are marked high and the index is regenerated.
- Placeholder scan: The plan uses exact file paths, concrete test names, explicit commands, and concrete code snippets; there are no `TODO` or `TBD` placeholders.
- Type consistency: The plan uses one consistent metadata field name, `priority`, with exactly two accepted values, `high` and `normal`, across tests, docs, and generator code.
