# Optimize Pattern Evidence Routing Design

## Summary

- Add an evidence-driven pattern-routing layer to the existing `optimize` workflow.
- Keep the current layered analysis order unchanged:
  - `pattern triage`
  - `profiling diagnosis`
  - `IR attribution`
  - `compiler-source escalation`
- Introduce a small set of explicit knowledge objects:
  - pattern Markdown files with a fixed section contract
  - a generated `semantic index`
  - an optional generated machine-readable pattern mirror
  - a new `symptom index` plus `symptom cards`
  - data-only `extractors` for code, profile, IR, and round-state evidence
- Keep agent judgment as the final decision-maker.
- Keep the CLI thin. This design belongs in staged skills, references, and helper scripts rather than new CLI subcommands.

## Problem

The repository already has the main ingredients for good optimize reasoning:

- pattern references under `skills/triton/triton-npu-optimize/references/patterns/`
- profiler-first deep analysis under `triton-npu-analyze-round-performance`
- structured profile JSON from `triton-npu-profile-operator`
- structured IR summaries from `triton-npu-analyze-ir`

What is still missing is a systematic routing layer between these inputs and pattern selection.

Today the pattern library is primarily a Markdown knowledge base. This helps the agent read selectively, but it still leaves several gaps:

- pattern selection is not yet backed by one structured source of truth
- profile and IR outputs are available, but they do not route into pattern choice in a consistent way
- the agent can still jump from one familiar clue to one familiar pattern without a stable intermediate representation
- pattern-selection logic is hard to audit across rounds because the evidence path is mostly prose

The result is that `pattern triage` exists as a workflow concept, but not yet as a well-defined evidence-routing system.

## Goals

- Make pattern selection evidence-driven and layered instead of doc-driven and ad hoc.
- Reuse the current layered analysis contract instead of replacing it.
- Let `pattern triage` start from a small semantic view rather than bulk-loading detailed pattern references.
- Let profiling and IR evidence refine or overturn earlier pattern choices through a stable symptom layer.
- Keep extractor outputs non-diagnostic and data-only.
- Keep one authoring source of truth for pattern-routing knowledge.
- Preserve room for agent insight outside the current catalog when new ideas are clearly recorded and validated.

## Non-Goals

- Do not replace agent reasoning with a hard rule engine.
- Do not move optimize-domain reasoning into the CLI.
- Do not require every round to emit a new standalone machine-readable pattern-routing artifact.
- Do not redesign the current profile or IR analysis skills from scratch.
- Do not require every pattern decision to come from a pre-existing pattern file.
- Do not block novel optimization ideas that are not yet represented in the current pattern set.

## Recommended Approach

Add a thin routing architecture on top of the existing layered workflow:

- `pattern cards`
  - authoritative human-authored pattern references with a fixed section contract
- `semantic index`
  - lightweight generated human-facing entrypoint used during `pattern triage`
- optional generated machine-readable mirror
  - script-produced metadata derived from pattern Markdown when tooling needs it
- `symptom index` and `symptom cards`
  - profiling and IR time routing aids that narrow or rerank candidate patterns
- `extractors`
  - data-only evidence summarizers for code, profile, IR, and current round state
- `agent judgment`
  - final selection, escalation, and optimization choice

The key design choice is that extractors do not diagnose and do not recommend patterns. They only provide structured evidence. Diagnosis and pattern choice stay with the agent.

## Knowledge Model

### Pattern Cards As The Source Of Truth

The pattern Markdown files under:

- `skills/triton/triton-npu-optimize/references/patterns/*.md`

should remain the only human-authored source of truth.

The authoring contract should constrain structure, not vocabulary. In other words:

- each pattern file must answer a fixed set of routing questions
- the answers should be written in short natural-language bullets or short prose
- authors should not be forced to use rigid tokens such as `tiled_loop_exists`
- pattern files may include both predefined sections and arbitrary free-form sections

The predefined section contract should be:

Required:
- `## Summary`
- `## Use When`

Optional:
- `## Avoid When`
- `## Signals`
  - `### Code`
  - `### Profile`
  - `### IR`
- `## What To Verify After Applying`
- `## Related Patterns`

Free-form sections are also allowed anywhere in the document. These remain valuable for human reading and deeper agent context, but they are not required for first-level index generation.

Generator behavior should follow these rules:

- `Summary` and `Use When` are mandatory; a pattern file missing either one is invalid for index generation
- all other predefined sections are optional
- missing optional predefined sections lower the amount of structured routing information, but do not invalidate the pattern
- unknown or free-form sections are preserved in the source file and ignored by the first-level extractor unless a later generator version explicitly chooses to use them
- `Signals` may be absent entirely, or may contain only one of `Code`, `Profile`, or `IR`

Example shape:

```md
---
id: software-pipeline
title: Software Pipeline
---

## Summary

Improve memory and compute overlap in a hot loop that is already structurally tiled.

## Use When

- The kernel already has a real tiled loop, but loads and computation still happen in a mostly serial order.
- Profiling suggests the compute path is waiting on memory or producer stages.

## Avoid When

- The hot loop is still a manual reduction that should first be rewritten into a more regular tiled form.
- Keeping multiple tiles live would likely cause obvious UB pressure.

## Signals

### Code

- The main loop already has explicit tile structure over a reduction axis.
- Load, compute, and store steps are interleaved weakly or not at all.

### Profile

- Wait-heavy timeline or pipe-utilization evidence suggests weak overlap.

### IR

- Lowering still shows transfer and synchronization structure that looks too serial for the tiled loop shape.

## What To Verify After Applying

- Recheck correctness carefully around prefetch order and tile handoff.
- Watch for UB pressure or reduced effective parallelism after the rewrite.

## Related Patterns

- `classic-matmul`: use this first when the loop is not yet a proper tiled matmul.
- `reorder-load`: use this when the issue is simpler load ordering rather than full pipelining.
```

This format keeps pattern authoring natural while still making the content extractable.

### Generated Pattern Artifacts

From the pattern Markdown files, a local helper script may generate:

- `index.md`
- optional `catalog.generated.json`
- optional reverse mappings such as symptom-to-pattern lookup aids

These generated files are build artifacts or helper artifacts. They are not author-edited truth.

If a machine-readable mirror is generated, it should preserve the original natural-language text instead of forcing aggressive early normalization.

### Semantic Index

The semantic index should be generated from the pattern Markdown files and remain the explicit lightweight entrypoint used during `pattern triage`.

Its role is:

- support `pattern triage`
- stay short enough to load first in every relevant round
- route the agent to one or two detailed pattern references only after a small candidate set exists

It should not become a second independently maintained source of pattern metadata.

### Symptom Index And Symptom Cards

Add a symptom-routing layer under the round-analysis skill:

- `skills/triton-npu-analyze-round-performance/references/symptom_index.md`
- `skills/triton-npu-analyze-round-performance/references/symptoms/<symptom-id>.md`

This layer owns the profiling and IR time question:

- “given the current evidence, which bottleneck symptom is most plausible, what should I check next, and which pattern ids are worth narrowing to?”

Each symptom card should contain:

- symptom definition
- common implementation causes
- minimum confirming evidence
- common false positives
- preferred extra checks
- candidate pattern ids or pattern links

Important boundary:

- symptom cards may mention candidate pattern ids
- symptom cards are not the source of truth for full per-pattern metadata
- the pattern Markdown files remain the authoritative source for pattern capabilities and routing hints

This avoids maintaining two independent copies of the same routing rules.

### Existing Detailed Pattern Documents

The current detailed pattern references already play two roles and should continue to do so:

- explain the pattern deeply
- show code-shape expectations
- explain rewrite constraints
- explain validation risks

They are second-level drill-down references, not first-level routing objects.

## Extractor Boundary

### Core Rule

Extractors output structured evidence only. They do not output:

- final bottleneck diagnoses
- pattern recommendations
- round continuation decisions

They may normalize and summarize raw inputs, but their outputs must remain non-diagnostic.

### Evidence Types

Extractor outputs may include:

- directly observable code-shape facts
- numeric metrics
- enum-valued facts
- stage summaries
- hotspot tables
- benchmark summary references
- provenance such as file path, stage name, metric name, or command source

They should not emit labels such as:

- `weak_pipeline_overlap`
- `sync_heavy_stages`
- `choose_software_pipeline`

If a helper wants to highlight suspicious data for navigation, it should use observation-style names such as:

- `stage_sync_op_count`
- `top_sync_stages`
- `task_wait_time_us`
- `l2_hit_ratio`

### Extractor Types

#### Code Extractor

Add a small code-fact extractor under the optimize skill, for example:

- `skills/triton/triton-npu-optimize/scripts/extract_code_facts.py`

Its scope is to extract directly observable implementation facts such as:

- `manual_k_reduction`
- `tiled_loop_exists`
- `serialized_load_compute_store`
- `index_based_load`
- `one_row_per_program`
- `heavy_masking`

This is the only net-new extractor in phase 1.

#### Profile Extractor

Reuse the current structured profile summary:

- `python3 ../triton-npu-profile-operator/scripts/profile_summary.py <profile-dir> --format json`

This already provides a structured profile evidence layer and should remain the preferred profile extractor.

#### IR Extractor

Reuse the current IR performance summary:

- `python3 ../triton-npu-analyze-ir/scripts/inspect_ir.py performance-signals --ir-dir <ir-dir> --format json`

This already provides stage-level IR observations and should remain the preferred IR extractor.

#### Round-State Extractor

Add a light round-state reader later if needed. Its job would be:

- inspect existing round artifacts
- identify reusable evidence paths
- identify the current or reused analysis level
- surface benchmark or compare-perf evidence that already exists

This can remain a later phase if the existing round artifacts are still easy enough for the agent to inspect directly.

## Runtime Flow

### Level 0: Pattern Triage

At `pattern triage`, the agent should:

1. Read the semantic index from `references/pattern_index.md`.
2. Collect small-scope evidence from:
   - current operator code
   - existing benchmark behavior
   - existing compare-perf conclusions when available
3. Use the code facts plus semantic index plus the relevant pattern sections to produce a small candidate set.
4. Read only the one or two most relevant detailed pattern files.
5. Record the chosen hypothesis and why the selected pattern looks plausible.

Expected outcomes:

- one clear pattern-backed hypothesis exists
- or no clear pattern direction exists and the round escalates to profiling

### Level 1: Profiling Diagnosis

At `profiling diagnosis`, the agent should:

1. Reuse the structured profile JSON summary.
2. Use the symptom index to choose one or two relevant symptom cards.
3. Use those symptom cards to:
   - interpret the dominant bottleneck question
   - request missing checks
   - narrow or rerank pattern candidates
4. Escalate to IR only when profile evidence is still not explanatory enough.

This means profiling does not restart pattern search from scratch. It refines or overturns earlier candidate patterns through a symptom layer.

### Level 2: IR Attribution

At `IR attribution`, the agent should:

1. Reuse the structured IR observations.
2. Use symptom-card follow-up checks and the pattern files' `Signals -> IR` guidance to confirm or reject candidate patterns.
3. Explain why the observed lowering or stage structure supports or weakens the current pattern hypothesis.

IR remains explanatory and attributive. It does not become a new first-level pattern index.

### Level 3: Compiler-Source Escalation

`compiler-source escalation` remains outside the normal pattern-routing loop.

Use it only when:

- profiling and IR have already narrowed the question
- a compiler-side explanation is still needed before the next operator change is clear

This layer should not become another general-purpose pattern search mechanism.

### Reuse Across Later Rounds

Later rounds may start from deeper levels when earlier evidence is still valid.

In that case the round should explicitly record:

- the reused evidence path
- the active candidate patterns
- why the shallower level is already established or already exhausted

## Prompt And Skill Contract Changes

### `triton-npu-optimize`

Update optimize guidance so it explicitly teaches:

- `references/pattern_index.md` is the semantic entrypoint for pattern triage
- detailed pattern docs are the source of truth and second-level reads during routing
- pattern selection should use structured code facts and existing benchmark behavior first
- later profile and IR evidence may narrow or overturn an earlier pattern choice

### `triton-npu-analyze-round-performance`

Update round-analysis guidance so it explicitly teaches:

- profile JSON and IR JSON feed a symptom-routing step
- symptom cards are the default diagnostic-routing reference after extraction
- symptom cards narrow pattern candidates instead of replacing the optimize skill

### Prompt Discipline

Prompts should encourage this read order:

1. semantic index first
2. relevant symptom card second when deeper evidence exists
3. detailed pattern doc last

The agent should continue to prefer the smallest source that can unblock the next decision.

## Directory Layout

Phase 1 should keep the current file layout mostly additive:

```text
skills/triton/triton-npu-optimize/
  references/patterns/
    index.md
    autotune.md
    cache_use.md
    classic-matmul.md
    ...
  scripts/
    extract_code_facts.py
    build_pattern_index.py

skills/triton-npu-analyze-round-performance/
  references/symptoms/
    index.md
    high-transfer-pressure.md
    high-scalar-overhead.md
    low-cube-utilization.md
    poor-locality.md
    weak-pipeline-overlap.md
    under-parallelized-block-dim.md
```

Existing profile and IR extractors remain where they already live.

## Round Recording

Phase 1 should avoid adding a mandatory new top-level artifact.

Instead, require the current round records to capture:

- candidate pattern ids considered at the start of the round
- why the selected pattern looked plausible
- which symptom cards were consulted, when deeper evidence was used
- whether profile or IR evidence confirmed, weakened, or overturned the earlier pattern hypothesis

These notes belong in the existing:

- `opt-round-N/attempts.md`
- `opt-round-N/summary.md`

If a later phase needs stronger auditability, a round-local machine-readable routing artifact can be added then.

## Phase 1 Scope

Phase 1 should do only the minimum needed to make the routing model real:

- define a fixed section contract for pattern Markdown
- refactor existing pattern Markdown files to follow that contract
- generate `index.md` from the pattern files
- add a small code extractor for directly observable implementation facts
- add a small symptom index and a handful of symptom cards aligned with the current profiling references
- update optimize and round-analysis guidance to reflect the new routing order
- keep the final decision with the agent rather than hard-gating pattern choice

## Later Phases

Later work may add:

- a thin helper that merges code, profile, and IR evidence into one short routing payload
- optional generated machine-readable mirrors for tooling
- optional consistency checks between generated artifacts and pattern docs
- optional round-local `pattern-routing.json` if supervisor auditing needs more structure
- optional derivation or validation tooling for symptom index entries

These are deliberately deferred so phase 1 can validate the information architecture first.

## Risks And Mitigations

### Metadata Duplication

Risk:

- pattern-to-symptom mappings drift between pattern docs, generated indexes, symptom docs, and prompts

Mitigation:

- keep the pattern Markdown files authoritative
- generate helper artifacts from those files instead of hand-maintaining duplicate truth
- keep symptom cards descriptive and routing-oriented rather than full pattern metadata stores

### Overconfident Extractors

Risk:

- extractor helpers drift from evidence summarization into hidden diagnosis

Mitigation:

- require extractors to emit non-diagnostic observations only
- reserve symptom and pattern conclusions for the agent

### Context Bloat

Risk:

- the routing system recreates the original problem by making the agent read too much

Mitigation:

- keep the semantic index short
- keep symptom cards focused on one bottleneck class each
- keep detailed pattern docs as second-level reads only

### Catalog Ossification

Risk:

- the agent stops trying valid new ideas because they are not in the catalog

Mitigation:

- explicitly allow non-catalog ideas when they are clearly recorded, justified, and validated through the normal optimize workflow

### CLI Creep

Risk:

- routing logic slowly migrates into CLI code

Mitigation:

- keep routing in skills, references, and helper scripts
- keep CLI responsibility limited to orchestration

## Expected Outcome

- `pattern triage` becomes systematic without becoming rigid.
- Existing profile and IR JSON outputs gain a clear consumer in the optimize workflow.
- Later evidence can rerank or overturn earlier pattern choices in a disciplined way.
- The evidence path behind a chosen pattern becomes easier to record and easier to audit.
- The project keeps its current layered-analysis model while adding a more explicit retrieval and routing architecture around patterns.
