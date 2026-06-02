# Optimize High-Priority Pattern Reminder Design

## Summary

- Extend optimize pattern priority from a generated index-only feature into runtime optimize guidance.
- Keep pattern cards as the single source of truth for both `pattern_index.md` and temporary optimize memory-file reminders.
- Generate high-priority reminder text from structured pattern-card data instead of hardcoding a second list in `memory_file.py`.
- Make reminder generation respect the actually selected generic optimize knowledge tree (`v1`, `v2`, or `v3`) rather than assuming one fixed skill path.
- Keep runtime path resolution compatible with both source-tree execution and PyInstaller onefile packaging.
- Promote `grid-flatten-and-ub-buffering` to high priority, but do not create a separate universal "`grid=40`" pattern or rewrite the generic card into a machine-specific rule.
- Extend the `grid-flatten-and-ub-buffering` card so it can recommend best-effort runtime core-count discovery, task-kind-aware core-count selection (`cube`, `vector`, or `mix`), and explicit fallback defaults when runtime discovery is unavailable.

## Problem

The repository already supports per-pattern priority metadata and renders a `## High Priority Patterns` section in the generated optimize pattern index. That helps when an agent reads `pattern_index.md`, but it does not surface the same shortlist in the temporary optimize memory file (`AGENTS.md` / `CLAUDE.md`) that is written into the workspace for each run.

This leaves a gap:

- high-priority patterns exist in the knowledge base, but the runtime guidance file does not call them out explicitly
- the memory file currently contains workflow and analysis-order guidance, but no generated reminder block for the most important patterns
- adding such reminders manually would create a second list that can drift from the pattern cards and generated index
- optimize already supports `--optimize-knowledge v1|v2|v3`, so any reminder generation must follow the selected knowledge tree instead of hardcoding the default tree
- packaged onefile builds cannot rely on repository-relative paths; any runtime reminder implementation must use the same resource-resolution rules as the rest of the bundled `skills/` tree

The immediate motivating case is `grid-flatten-and-ub-buffering`: splitting work by physical core count is important and common enough to surface early, but it should be surfaced as a high-priority generic pattern, not as a blanket "always set `grid = 40`" rule.

## Goals

- Surface high-priority optimize patterns directly in optimize memory files as a short reminder block.
- Keep pattern cards as the single source of truth for priority and one-line summaries.
- Reuse one structured parser for both index generation and runtime reminder generation.
- Respect the selected generic optimize knowledge tree for `v1`, `v2`, and `v3`.
- Keep runtime lookup compatible with source mode and PyInstaller onefile mode.
- Keep the memory-file block compact enough to aid pattern triage without duplicating full pattern documentation.
- Mark `grid-flatten-and-ub-buffering` as high priority while preserving generic `NUM_CORES` / physical-core wording.
- Document how `grid-flatten-and-ub-buffering` should derive or fall back to cube/vector core counts for current-target guidance.

## Non-Goals

- Do not parse checked-in `pattern_index.md` as a runtime data source.
- Do not hardcode a second manual high-priority pattern list in `src/triton_agent/optimize/memory_file.py`.
- Do not create a new standalone "`grid=40`" optimize pattern.
- Do not redesign symptom-index generation or symptom-card behavior.
- Do not add a new ranking field beyond existing `priority: high|normal`.
- Do not add a broad reminder system for `torch-npu-optimize-knowledge` in this change.
- Do not embed full pattern prose into workspace memory files.
- Do not make optimize memory-file rendering execute runtime NPU queries on behalf of the agent.

## Alternatives Considered

### 1. Hardcode high-priority reminders in `memory_file.py`

This would add a small static list such as `autotune`, `a5-force-simt-only-discrete-access`, and `grid-flatten-and-ub-buffering` directly in runtime code.

Pros:

- smallest short-term code change
- no new helper module needed

Cons:

- creates a second source of truth
- easy to forget when new high-priority patterns are added
- breaks alignment between pattern cards, generated index, and runtime reminders
- forces runtime code to know pattern names and summaries that already live in skill content

### 2. Parse generated `pattern_index.md` at runtime

This would reuse the checked-in generated index instead of the pattern cards.

Pros:

- avoids another card parser in runtime code
- reuses an already generated artifact

Cons:

- converts structured authoring data back into Markdown parsing
- couples runtime behavior to generated prose layout
- makes path handling more fragile in bundled execution
- encourages runtime consumers to depend on checked-in generated Markdown instead of the authored source

### 3. Share a structured pattern-card helper between index generation and runtime reminders

This would move the reusable card parsing and high-priority selection logic into a shared helper that both the index builder and runtime reminder adapter can call.

Pros:

- preserves one source of truth
- avoids Markdown reparsing of generated files
- keeps runtime reminder generation deterministic
- supports future high-priority additions without editing runtime lists

Cons:

- requires a small refactor of the current pattern-index builder
- requires runtime plumbing so memory-file rendering knows the selected knowledge tree

## Recommendation

Use alternative 3.

Pattern cards should remain the authored source of truth. The generated index and runtime reminder block should both derive from the same structured pattern-card helper so the system stays maintainable as new high-priority patterns are added.

## Design

### Source Of Truth

High-priority reminder content should come from the same card fields that already drive the generated pattern index:

- frontmatter `priority: high|normal`
- `## Summary`
- the card identifier

The generic optimize knowledge trees selected by `--optimize-knowledge` remain the authoritative owners of this metadata.

This change also updates `grid-flatten-and-ub-buffering` to `priority: high`. That card should continue to describe physical-core-aware mapping in generic `NUM_CORES` terms. If the card adds current-target examples such as a 40-core NPU, they should be framed as a concrete hypothesis or instance, not as a universal "`grid=40` always" rule.

### Grid-Flattening Core-Count Guidance

As part of promoting `grid-flatten-and-ub-buffering`, update that card's detailed guidance so it can recommend best-effort runtime device inspection before falling back to current-target defaults.

Suggested runtime query example:

```python
import torch

print(torch.npu.device_count())
device = torch.npu.current_device()
props = torch.npu.get_device_properties(device)
print(props)
```

This query should be documented as a suggestion for evidence gathering, not as a required runtime step for every optimize session. The card should state explicitly:

- use runtime query output if it clearly exposes the relevant cube/vector core-count facts
- if the query succeeds but does not expose explicit cube/vector counts, treat it as chip-identification evidence only
- if the query fails or is unavailable, fall back to current-target defaults for this guidance path

Current-target fallback defaults for this card:

- cube cores: `24`
- vector cores: `48`

The card should also explain that grid/core-count guidance is task-kind-aware:

- for `cube`-like operators, prefer cube-core-count-aligned hypotheses
- for `vector`-like operators, prefer vector-core-count-aligned hypotheses
- for `mix` operators, keep both cube-count and vector-count launch sizes as candidate hypotheses and choose based on measured bottlenecks rather than forcing one universal count

Task kind should preferably come from existing profiling evidence such as operator-type or pipeline-ratio diagnosis (`cube`, `vector`, `mix`) when available. If that evidence is not available, the card may fall back to code-shape reasoning, but it should present the result as a hypothesis rather than a confirmed classification.

This guidance belongs in the high-priority pattern card itself, not in the generated memory-file reminder block. The reminder block should stay short and only point the reader toward the detailed card.

### High-Priority Authoring Policy

High priority is meant to stay a shortlist, not a second full index category.

The memory-file reminder block will therefore render all current high-priority patterns from the selected generic knowledge tree without introducing an additional rank field or manual cap in this change. This is acceptable because:

- the existing priority design already treats high priority as a small set surfaced first
- generation keeps the list synchronized automatically
- if the shortlist grows large enough to hurt readability later, a separate follow-up can add secondary ranking or truncation rules

### Shared Helper API

Introduce a reusable skill-side helper module in each generic Triton optimize knowledge tree that can be selected by `--optimize-knowledge`:

- `skills/triton-npu-optimize-knowledge/scripts/pattern_catalog.py`
- `skills/triton-npu-optimize-knowledge-v2/scripts/pattern_catalog.py`
- `skills/triton-npu-optimize-knowledge-v3/scripts/pattern_catalog.py`

This helper should own the reusable parsing and selection logic that is currently embedded in each `build_pattern_index.py`.

Expected API shape:

```python
def parse_pattern_cards(patterns_dir: Path) -> list[PatternCard]: ...
def list_high_priority_pattern_cards(patterns_dir: Path) -> list[PatternCard]: ...
def build_index_text(patterns_dir: Path) -> str: ...
def build_high_priority_reminder_lines(patterns_dir: Path) -> list[str]: ...
```

`PatternCard` should include at least:

- identifier
- title
- priority
- summary
- source path

The helper should accept explicit `patterns_dir` paths instead of inferring repository layout from `__file__`. That keeps the API reusable across knowledge trees and avoids tying logic to one specific tree location.

### Pattern Index Builder Refactor

Each `build_pattern_index.py` remains the CLI entrypoint and checked-in-file generator for its tree, but it becomes a thin wrapper around the shared helper.

That means:

- existing CLI behavior stays the same
- existing checked-in `pattern_index.md` contract stays the same
- runtime reminder generation and index generation share one parser and one high-priority selector

This refactor is internal and should not change the generated file shape beyond any intentional card-content updates such as the newly high-priority `grid-flatten-and-ub-buffering` entry.

### Runtime Reminder Adapter

Add a runtime-side adapter, for example:

- `src/triton_agent/optimize/pattern_reminders.py`

This module should not parse pattern cards directly. Instead, it should:

1. receive the actual selected generic optimize knowledge source name
2. resolve the corresponding `patterns/` directory from runtime resources
3. load the selected tree's helper module through the existing skill-loader bridge
4. ask the helper for generated high-priority reminder lines

Expected runtime flow:

```text
execution/request
-> resolve actual selected generic optimize knowledge source
-> session_artifacts / memory_file preparation
-> pattern_reminders adapter
-> load selected skill helper
-> generate reminder lines
-> render memory-file block
```

### Knowledge Tree Selection

The memory-file layer should not duplicate the `v1|v2|v3` mapping logic.

Instead, runtime code should pass the actual selected generic optimize knowledge source name derived from the already resolved staging result. For example:

- `triton-npu-optimize-knowledge`
- `triton-npu-optimize-knowledge-v2`
- `triton-npu-optimize-knowledge-v3`

This avoids a second copy of version-selection logic in the memory-file code path and ensures reminders match the same tree that was staged for the agent.

The operator-target extra skill `torch-npu-optimize-knowledge` remains out of scope for this reminder block. The reminder block in this change should follow the selected generic Triton optimize knowledge tree only.

### Memory-File Rendering

Add an optional compact block to both unsupervised and shared optimize memory-file templates.

Representative shape:

```text
High-priority pattern reminders:
- `autotune`: ...
- `grid-flatten-and-ub-buffering`: ...
- `a5-force-simt-only-discrete-access`: ...
Read the staged optimize knowledge `references/pattern_index.md` for the full current high-priority list and detailed routing.
```

Rendering rules:

- emit the block only when the selected tree currently has at least one high-priority pattern
- include one concise line per high-priority pattern
- use generated one-line summaries rather than full `Use When` bullets
- add one closing line that points the agent back to the staged `pattern_index.md` for the full list and details

This block should stay short and reminder-oriented. It is not a replacement for `pattern_index.md`.

The reminder block should not inline the runtime query snippet, the fallback cube/vector defaults, or the `cube`/`vector`/`mix` branching rules. Those details belong in the detailed pattern card.

### Path Resolution And Packaging

Runtime reminder generation must use the same resource-resolution model as the rest of the packaged application.

Use:

- `triton_agent.resources.skills_root()` to resolve the root of bundled `skills/`
- `triton_agent.skill_loader.load_skill_script_module()` to load the helper module from the selected knowledge tree

Do not:

- assume the repository root exists on disk
- construct paths from `Path(__file__).parents[...]` in runtime code
- read from checked-in `docs/` or generated Markdown as the runtime source

This keeps the design compatible with:

- source-tree execution, where `skills_root()` resolves to the repository `skills/`
- PyInstaller onefile execution, where `application_root()` resolves through `sys._MEIPASS` and the bundled `skills/` tree is extracted there at runtime

### Failure Semantics

Reminder generation should prefer explicit internal failures over silent drift.

For the selected generic optimize knowledge tree:

- if the helper module cannot be loaded, fail optimize session preparation with a short actionable error
- if the `patterns/` directory is missing, fail optimize session preparation explicitly
- if no patterns are marked `high`, treat that as valid and simply omit the reminder block

This keeps reminder generation strict about broken bundled resources while avoiding noisy placeholder text in the runtime memory file.

## Testing Strategy

### Shared Helper Tests

Extend pattern-tool coverage to validate that the new helper:

- parses valid cards
- rejects invalid `priority` values
- returns the same high-priority set used by the generated index
- generates deterministic reminder lines from card summaries

### Index-Builder Regression Tests

Keep existing checked-in-index parity tests and update them so the builder exercises the new helper-backed path without changing the external contract.

### Pattern-Content Contract Tests

Add or update contract coverage so `grid-flatten-and-ub-buffering` explicitly documents:

- runtime query as an evidence-gathering option
- fallback defaults of cube `24` and vector `48`
- task-kind-aware guidance for `cube`, `vector`, and `mix`

### Memory-File Guidance Tests

Add optimize-guidance tests that verify:

- unsupervised memory files include the generated high-priority reminder block
- shared supervised memory files include the same block
- `v1`, `v2`, and `v3` selection produce reminders from the selected generic knowledge tree
- runs with no high-priority patterns omit the block cleanly

### Path-Handling Tests

Add focused tests around the runtime adapter so it:

- resolves skill content from runtime resource helpers rather than repository-relative strings
- uses the selected source skill name instead of re-deriving `v1|v2|v3`

These tests do not need to run a real PyInstaller build. They only need to prove that the runtime code uses the resource helpers and skill-loader bridge that are already designed for bundled execution.

## Scope Boundaries

- Do not redesign optimize memory-file ownership or session-artifact lifecycles.
- Do not add high-priority reminders for Torch NPU operator knowledge in this change.
- Do not add a third content source beyond pattern cards and generated index output.
- Do not introduce a new metadata field just to order high-priority reminders.
- Do not teach runtime code to parse generated Markdown that already came from the pattern cards.
- Do not treat `torch.npu.get_device_properties()` as guaranteed to expose cube/vector counts on every environment; the card should describe it as best-effort evidence with explicit fallbacks.
