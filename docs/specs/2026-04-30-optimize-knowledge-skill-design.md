# Optimize Knowledge Skill Split Design

## Summary

- Add a new reference-only skill, `triton-npu-optimize-knowledge`, to own generic optimize knowledge.
- Move generic pattern cards, symptom cards, and their generated indexes into that new skill.
- Keep `triton-npu-optimize` as the optimize workflow owner.
- Keep `triton-npu-analyze-round-performance` as the round diagnosis owner.
- Keep `triton-npu-cann-ext-api-patterns` separate as a specialized pattern pack in phase 1.
- Treat this change as an ownership and indexing refactor, not an optimize-behavior redesign.

## Problem

Generic optimize knowledge currently lives under multiple skill owners:

- generic pattern references live under `skills/triton-npu-optimize/`
- generic symptom references live under `skills/triton-npu-analyze-round-performance/`

That layout already works, but it mixes three different responsibilities:

- workflow ownership
- evidence extraction and diagnosis ownership
- reusable knowledge ownership

As a result:

- editing shared knowledge requires touching workflow or analysis skill trees
- the source of truth for generic optimize knowledge is split across two skills
- it is harder to explain to future contributors which files define process versus which files define reusable pattern or symptom knowledge
- it is harder to grow pattern and symptom indexing as a coherent knowledge layer

The desired layering is simpler:

- workflow skills explain when to consult knowledge
- analysis skills explain how to collect and interpret evidence
- a knowledge skill owns the reusable cards and indexes

## Goals

- Introduce one reference-only skill for generic optimize knowledge.
- Make generic pattern cards and generic symptom cards owned by one skill tree.
- Keep the existing optimize layered-analysis model unchanged:
  - `pattern triage`
  - `profiling diagnosis`
  - `IR attribution`
  - `compiler-source escalation`
- Keep `triton-npu-optimize` focused on round workflow, contracts, and validation rules.
- Keep `triton-npu-analyze-round-performance` focused on profile, `.bin`, and IR-backed diagnosis.
- Keep `triton-npu-cann-ext-api-patterns` separate in phase 1.
- Keep the CLI thin and avoid moving this behavior into new CLI subcommands.

## Non-Goals

- Do not merge `triton-npu-cann-ext-api-patterns` into the generic knowledge skill in phase 1.
- Do not redesign optimize prompts, artifacts, or round semantics.
- Do not change who owns `opt-round-N/perf-analysis.md`.
- Do not move `extract_code_facts.py` into the knowledge skill in phase 1.
- Do not introduce a rule engine or automatic pattern chooser.
- Do not make symptom cards the authoritative source for per-pattern metadata.

## Alternatives Considered

### 1. Single Generic Knowledge Skill

Add one new skill, `triton-npu-optimize-knowledge`, that owns generic patterns, generic symptoms, and the generated indexes.

Pros:

- gives the clearest ownership model
- keeps the read path simple
- matches the desired split between workflow, analysis, and reusable knowledge

Cons:

- requires path updates across skills, tests, and docs

### 2. Separate Pattern And Symptom Skills

Split generic knowledge into two reference-only skills, one for patterns and one for symptoms.

Pros:

- very pure conceptual separation

Cons:

- makes the read path more fragmented
- adds more indirection for agents and contributors
- does not improve maintainability enough to justify the extra split

### 3. Keep Knowledge In Place And Only Centralize Index Rules

Leave patterns under `triton-npu-optimize` and symptoms under `triton-npu-analyze-round-performance`, but standardize generators and contracts.

Pros:

- lowest migration cost

Cons:

- does not solve the main ownership problem
- keeps reusable knowledge tied to workflow and diagnosis skill trees

### Recommendation

Use alternative 1.

It gives the cleanest long-term model while keeping the phase 1 scope narrow:

- one generic knowledge skill
- one workflow skill
- one round-diagnosis skill
- one separate specialized extension-pattern pack

## Proposed Ownership Model

### `triton-npu-optimize-knowledge`

This new skill owns only reusable generic optimize knowledge:

- generic pattern cards
- generic symptom cards
- generated human-facing indexes
- generator scripts and index-generation contracts

It does not own:

- optimize round workflow
- baseline preparation
- correctness or benchmark validation
- profile collection
- IR capture or navigation
- compiler-source analysis
- round-local artifacts such as `attempts.md`, `summary.md`, or `perf-analysis.md`

### `triton-npu-optimize`

This skill remains the owner of optimize workflow behavior:

- round creation and progression
- analysis-level escalation rules
- recordkeeping requirements
- validation and benchmark gates
- top-level optimize references such as artifact and note contracts

Its knowledge-facing role becomes:

- tell the agent when to consult the generic pattern index
- tell the agent when to consult the generic knowledge skill versus a specialized pattern pack
- avoid owning the generic pattern library directly

### `triton-npu-analyze-round-performance`

This skill remains the owner of round-level performance diagnosis:

- extract and interpret profile evidence
- deepen into `.bin` when needed
- use IR evidence when profiler-only reasoning is insufficient
- write `opt-round-N/perf-analysis.md`

Its knowledge-facing role becomes:

- use the generic symptom index and symptom cards as routing aids
- return to evidence-backed diagnosis after the routing step
- avoid owning the generic symptom library directly

### `triton-npu-analyze-ir`

This skill remains the IR evidence companion:

- capture IR
- inspect IR
- extract stage-level signals

It should not own generic pattern or symptom knowledge.

### `triton-npu-cann-ext-api-patterns`

This skill remains a separate specialized pattern pack in phase 1.

It should continue to:

- own A5-oriented extension-API-specific pattern material
- depend on `triton-npu-optimize` for workflow semantics
- stay outside the new generic knowledge skill until there is a stronger reason to unify pattern-pack packaging

## Proposed File Layout

Add a new skill tree:

```text
skills/
  triton-npu-optimize-knowledge/
    SKILL.md
    references/
      pattern_index.md
      patterns/
        *.md
      symptom_index.md
      symptoms/
        *.md
    scripts/
      build_pattern_index.py
      build_symptom_index.py
```

### Files To Move Into The New Skill

Move these generic pattern assets from `triton-npu-optimize`:

- `references/pattern_index.md`
- `references/patterns/*.md`
- `scripts/build_pattern_index.py`

Move these generic symptom assets from `triton-npu-analyze-round-performance`:

- `references/symptom_index.md`
- `references/symptoms/*.md`

Add a new generator for symptom index consistency:

- `scripts/build_symptom_index.py`

### Files That Stay In Place

Keep these optimize-workflow references under `triton-npu-optimize`:

- `references/artifacts.md`
- `references/opt-note-format.md`
- `references/round-failure-handling.md`

Keep these analysis references under `triton-npu-analyze-round-performance`:

- `references/ascend-npu-profiling-analysis.md`
- `references/ascend-npu-optimization-guidance.md`
- `references/ascend-npu-architecture-notes.md`

Keep this evidence helper outside the new knowledge skill in phase 1:

- `skills/triton-npu-optimize/scripts/extract_code_facts.py`

It is a structured evidence extractor, not reusable optimize knowledge.

## Knowledge Skill Contract

`skills/triton-npu-optimize-knowledge/SKILL.md` should be reference-only.

It should state:

- this skill does not define optimize workflow
- this skill does not own diagnosis artifacts
- start from the relevant generated index
- read only the one or two most relevant detailed cards
- return to the caller skill for the actual diagnosis, optimization choice, and recordkeeping

Suggested read model:

1. start from `pattern_index.md` for code-structure-first pattern triage
2. start from `symptom_index.md` for evidence-driven symptom routing after structured profile or IR summaries exist
3. read only the most relevant detailed cards
4. let the caller skill decide what to do next

## Cross-Skill Read Contract

### `triton-npu-optimize`

At pattern triage, `triton-npu-optimize` should point to:

- `../triton-npu-optimize-knowledge/references/pattern_index.md`
- `../triton-npu-optimize-knowledge/references/patterns/<pattern>.md`

It may still point to specialized pattern packs, such as `triton-npu-cann-ext-api-patterns`, when the round explicitly justifies that path.

### `triton-npu-analyze-round-performance`

During evidence-backed routing, `triton-npu-analyze-round-performance` should point to:

- `../triton-npu-optimize-knowledge/references/symptom_index.md`
- `../triton-npu-optimize-knowledge/references/symptoms/<symptom>.md`

When a symptom card narrows candidate directions, the analysis flow may then return to:

- `../triton-npu-optimize-knowledge/references/pattern_index.md`
- one or two specific pattern cards under the same knowledge skill

This keeps the read order explicit:

- evidence first
- symptom routing second
- detailed pattern drill-down third

## Migration Plan

Use a short migration window and complete the ownership handoff in one focused change set.

### Step 1: Add The New Knowledge Skill

- create `skills/triton-npu-optimize-knowledge/`
- copy generic pattern assets into the new skill
- copy generic symptom assets into the new skill
- add a reference-only `SKILL.md`
- add `build_symptom_index.py` so symptom cards and symptom index follow the same authored-source and generated-index model as patterns

At this step, old paths may still exist temporarily, but the new skill becomes the intended destination.

### Step 2: Repoint Workflow And Analysis Skills

Update `triton-npu-optimize` so its pattern-triage instructions point to the new knowledge skill.

Update `triton-npu-analyze-round-performance` so its symptom-routing instructions point to the new knowledge skill.

This step changes read targets, not optimize semantics.

### Step 3: Repoint Tests And Documentation

Update:

- generation-contract tests
- prompt and guidance tests
- design notes and authoring notes that describe the source of truth
- any helper docs that still say patterns or symptoms are owned by workflow or analysis skills

Add the same kind of consistency check for symptom index generation that already exists for pattern index generation.

### Step 4: Remove Old Generic Knowledge Copies

After references and tests point to the new skill:

- remove generic pattern copies from `triton-npu-optimize`
- remove generic symptom copies from `triton-npu-analyze-round-performance`

Keep only the references that each remaining skill truly owns.

## Compatibility Rules

- Do not maintain long-term dual authorship.
- Do not use path aliases as a permanent solution.
- Do not auto-sync duplicate copies across skills.

During a short transition, an old path may temporarily remain only to support a single refactor patch, but it should not continue as an editable source of truth.

If temporary placeholder files are used, they should clearly say that ownership has moved and should not invite future editing in the old location.

## Test And Documentation Impact

Update tests that currently assume:

- `skills/triton-npu-optimize/references/pattern_index.md` is the generic pattern source of truth
- `skills/triton-npu-analyze-round-performance/references/symptom_index.md` is the generic symptom source of truth
- `skills/triton-npu-optimize/scripts/build_pattern_index.py` is owned by the optimize workflow skill

Add or update checks for:

- the new knowledge skill contract
- the new source-of-truth locations for generic patterns and symptoms
- generated pattern index consistency in the new path
- generated symptom index consistency in the new path

Update authoring and architecture notes so they describe:

- the new knowledge skill as the generic authored source of truth
- the continued separation of the CANN extension pattern pack
- the fact that `extract_code_facts.py` remains an evidence helper, not knowledge ownership

## Risks And Mitigations

### Risk: Path Churn Across Many Tests And Docs

Mitigation:

- make the migration one focused change set
- update tests and docs in the same patch
- avoid leaving long-lived duplicate paths behind

### Risk: Re-blurring Knowledge And Analysis Ownership

Mitigation:

- keep symptom cards in the knowledge skill
- keep profiling and IR interpretation guidance in the analysis skill
- keep explicit statements in both `SKILL.md` files about what they do not own

### Risk: Scope Creep During The Split

Mitigation:

- do not merge the CANN extension pattern pack in phase 1
- do not move `extract_code_facts.py` in phase 1
- do not add rule-engine behavior in phase 1

## Open Follow-Ups

- Decide the exact authoring and rendering contract for `build_symptom_index.py`, including which symptom-card sections are required for index generation.
- Decide later whether generic knowledge should eventually expose an optional machine-readable mirror in addition to the human-facing indexes.
- Revisit whether `extract_code_facts.py` should later move into an analysis-focused utility location rather than remain under `triton-npu-optimize`.
