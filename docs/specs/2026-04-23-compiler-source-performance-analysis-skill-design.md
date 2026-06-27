# Compiler Source Performance Analysis Skill Design

## Summary

Redesign `triton-npu-analyze-compiler-source` from a placeholder "read-only source escalation" note into a round-level performance analysis skill. The upgraded skill should explain one narrowed compiler-side performance question with evidence from a local AscendNPU-IR checkout, then turn that explanation into a concrete next operator change for the current Triton Ascend optimize round.

The new design keeps the existing optimize analysis ladder intact:

1. pattern triage
2. profiling diagnosis
3. IR attribution
4. compiler-source escalation

Compiler source remains the deepest escalation. The redesign changes what happens after that escalation is triggered: instead of broad source browsing, the agent follows a performance-focused workflow, uses repo-owned navigation references, and may use a very light helper script to narrow source locations.

## Goals

- Keep `triton-npu-analyze-compiler-source` as the single compiler-source skill for optimize rounds.
- Redefine the skill around performance explanation rather than compiler-error triage.
- Require the skill to start from round-local evidence such as `perf-analysis.md` and `ir/`, not from broad source curiosity.
- Make the skill produce operator-relevant conclusions rather than general compiler notes.
- Add repo-owned navigation references so agents can find the right AscendNPU-IR subtrees consistently.
- Allow one light helper script that narrows likely source locations without automating diagnosis.
- Keep the CLI thin: the skill and its references remain the workflow source of truth.

## Non-Goals

- Do not turn compiler source analysis into a first-line optimize workflow.
- Do not make compiler source analysis mandatory for every round.
- Do not make this skill responsible for compiler build failures, parser failures, or generic log triage.
- Do not make `bishengir/test/` part of the default reading path.
- Do not build a full automatic diagnosis pipeline that reads round artifacts and emits optimization advice by itself.
- Do not move compiler-source reasoning into the CLI or prompt construction layer.

## Current Problem

The current `triton-npu-analyze-compiler-source` skill mainly acts as a placeholder contract:

- it says compiler source must be CLI-provided and read-only
- it positions compiler source as a late escalation
- it requires a `compiler-analysis.md` artifact

Those constraints are useful, but they do not yet teach the agent how to analyze compiler source for performance questions. In practice, the current skill has four gaps:

1. It is not explicitly performance-specific.
2. It does not tell the agent how to turn round evidence into a narrow compiler-side question.
3. It does not provide durable navigation knowledge for the AscendNPU-IR tree.
4. It does not provide even a minimal narrowing helper comparable to `inspect_ir.py` or `profile_summary.py`.

As a result, the agent is authorized to inspect compiler source but is not yet given a reliable workflow for doing that inspection well.

## User-Facing Behavior

### Skill Identity

Keep the current skill name:

- `triton-npu-analyze-compiler-source`

But redefine the skill as:

- a performance-question explanation skill
- a round-level optimize escalation
- a source-backed bridge from profile and IR symptoms to the next operator change

The skill should explicitly say what it is not:

- not compiler-error triage
- not a general AscendNPU-IR tour
- not permission to modify compiler source

### Trigger Conditions

The skill should trigger only when all of the following are true:

- compiler source analysis is enabled for the current optimize run
- the current `opt-round-N/` already has round-local performance evidence
- the current question has been narrowed to a compiler-side performance behavior
- profile and IR evidence are not yet enough to choose the next operator change confidently

Typical trigger questions include:

- Why does a suspicious pass transition correspond to worse performance structure?
- Why does vectorization, fusion, or overlap opportunity appear to disappear after a given lowering path?
- Why do copy, sync, or buffer-growth signals in IR likely come from a compiler-side transform or memory-planning behavior?
- Which compiler subsystem should the current Triton operator change target indirectly?

### Non-Trigger Conditions

The skill should not trigger when:

- there is no round-local performance evidence yet
- the agent has not narrowed the question beyond "browse the source tree"
- profile and IR already justify a concrete next operator change
- the task is to understand compiler source in general rather than advance the current round

## Required Inputs

The redesigned skill should treat these as required inputs:

- the current operator workspace and `opt-round-N/`
- at least one round-local performance artifact:
  - `opt-round-N/perf-analysis.md`
  - `opt-round-N/ir/`
- the CLI-provided compiler source path and commit
- one narrowed compiler-side performance question

The skill should explicitly reject starting from:

- raw curiosity about the source tree
- a broad request to summarize the compiler
- a generic instruction to "look for optimization ideas in the compiler"

## Analysis Model

### Core Question Shape

Before reading compiler source, the agent should first rewrite the current round symptom into one narrow question. Good question shapes include:

- "What part of the lowering or pass structure would explain this stage-to-stage degradation?"
- "Which subsystem would plausibly introduce this copy or sync behavior?"
- "Which pass family or dialect behavior best matches this IR symptom?"
- "What compiler-side constraint does this performance symptom imply for the current operator structure?"

The skill should require a narrow question before source inspection begins.

### Default Reading Order

The default source reading order should be:

1. round evidence
2. `docs/`
3. `lib/`
4. `include/` only when needed
5. `test/` only in rare fallback scenarios

This order is intentional:

- round evidence defines the question
- `docs/` establishes pass, feature, and subsystem semantics
- `lib/` provides the main implementation evidence
- `include/` helps confirm declarations, generated pass interfaces, registration, and API boundaries
- `test/` is not the default path for performance analysis

### Source Evidence Priorities

#### `docs/` as semantic orientation

The agent should start in `docs/` to understand what subsystem it is looking for before reading implementation details. The highest-value documentation roots are:

- `docs/source/en/developer_guide/passes/`
- `docs/source/zh_cn/developer_guide/passes/`
- `docs/source/en/developer_guide/features/`
- `docs/source/zh_cn/developer_guide/features/`

These docs help answer:

- what a pass family is supposed to do
- what a feature or subsystem is meant to optimize
- where in the compiler pipeline a behavior likely belongs

#### `lib/` as primary implementation evidence

After semantic orientation, the agent should inspect `bishengir/lib/` for the actual implementation evidence. The most important roots are:

- `bishengir/lib/Conversion/`
- `bishengir/lib/Dialect/`
- `bishengir/lib/Transforms/`

These directories are the default implementation targets for performance explanation.

#### `include/` as secondary evidence

`bishengir/include/` should not be a co-equal main reading path. It should be used only when the agent needs to:

- confirm a pass name or pass family
- inspect `Passes.td` or `Passes.h`
- understand an interface, attribute, or declaration boundary
- find the registration or declaration surface that points back into `lib/`

This makes `include/` a navigation and contract aid, not the main evidence source.

#### `test/` as rare fallback

`bishengir/test/` should not be part of the default workflow. It may be used only when:

- the docs are too abstract and the agent wants a minimal IR example
- the agent needs to confirm the rough before/after shape of a pass on a tiny sample
- pass naming or pipeline usage remains unclear after `docs/`, `lib/`, and `include/`

The skill should explicitly say that `test/` is optional and uncommon for this workflow.

## Navigation Knowledge

The redesign should turn compiler-source navigation into repo-owned reference material rather than leave it implicit in prompts.

### New Reference: `references/navigation-map.md`

Add:

- `skills/triton/triton-npu-analyze-compiler-source/references/navigation-map.md`

This document should organize navigation by source subtree. It should include:

- the default reading order
- a directory atlas for the highest-value roots
- symptom-to-subtree mapping
- a few high-value `rg` recipes
- anti-patterns such as broad source-tree searches or reading `include/` first

The directory atlas should explain:

- when to read a subtree
- what kinds of questions it can answer
- what it should not be expected to answer

### New Reference: `references/perf-question-playbook.md`

Add:

- `skills/triton/triton-npu-analyze-compiler-source/references/perf-question-playbook.md`

This document should organize navigation by performance question instead of by directory layout. It should include playbooks for:

- suspicious stage transition
- vectorization loss
- copy or sync growth
- buffer expansion or memory-planning issue
- fusion or lowering-shape regression

Each playbook should end by forcing the agent to write down:

- the likely compiler-side explanation
- what that implies for the current Triton operator
- what the next operator change should target

### `SKILL.md` vs reference split

The redesigned `SKILL.md` should stay concise. It should contain:

- trigger rules
- workflow rules
- the default reading order
- links to the navigation references
- the output contract

The detailed subtree mapping and question playbooks should live in the reference documents, not inside `SKILL.md`.

## Helper Script Strategy

### Add a light navigator, not an auto-diagnoser

Add one light script:

- `skills/triton/triton-npu-analyze-compiler-source/scripts/inspect_compiler_source.py`

This script should narrow likely source locations. It should not diagnose performance issues automatically and should not write `compiler-analysis.md`.

### First-version interface

The first version only needs one subcommand:

- `locate`

Example:

```bash
python3 ./scripts/inspect_compiler_source.py locate \
  --source-root <compiler-source-path> \
  --term hfusion \
  --term vectorize \
  --hint pass \
  --format json
```

Supported arguments should stay minimal:

- `--source-root`
- repeated `--term`
- `--hint` with a small fixed vocabulary
- `--limit`
- `--format text|json`

### Search scope

The script should search only the high-value roots:

- `docs/source/en/developer_guide/passes/`
- `docs/source/zh_cn/developer_guide/passes/`
- `docs/source/en/developer_guide/features/`
- `docs/source/zh_cn/developer_guide/features/`
- `bishengir/lib/Conversion/`
- `bishengir/lib/Dialect/`
- `bishengir/lib/Transforms/`
- `bishengir/include/bishengir/`

It should not search `bishengir/test/` by default.

### Output contract

The script should return grouped candidates with at least:

- `area`
- `path`
- `score`
- `matched_terms`
- `why`

This keeps the script purely navigational:

- it helps the agent decide where to read next
- it does not replace source reading
- it does not replace the skill's reasoning workflow

## Skill Contract Changes

### Frontmatter description

The skill description should describe only the triggering conditions. It should no longer mention compiler errors. The description should say that the skill applies when:

- compiler source analysis is enabled
- the question is performance-related
- profile and IR evidence have already narrowed the problem
- source-backed explanation is needed before choosing the next operator change

### `SKILL.md` structure

The redesigned `SKILL.md` should contain:

1. `# Analyze Compiler Source For Performance`
2. `## Goal`
3. `## Required Inputs`
4. `## When To Use`
5. `## When Not To Use`
6. `## Default Workflow`
7. `## Navigation Rules`
8. `## Output Contract`
9. `## Reasoning Rules`

### Default workflow

The workflow should be:

1. Read the current round evidence and narrow one compiler-side performance question.
2. Read the skill's navigation references before broad source inspection.
3. Inspect `docs/` first to orient on subsystem meaning.
4. Inspect `lib/` for implementation evidence.
5. Inspect `include/` only when declarations or generated pass interfaces are needed.
6. Inspect `test/` only when a minimal example is genuinely necessary.
7. Write `opt-round-N/compiler-analysis.md`.

### Reasoning rules

The skill should explicitly require:

- facts vs inference separation
- local source path and commit citation for nontrivial source-backed claims
- conclusions tied to the current round's evidence
- implications tied to the current Triton operator
- no generic compiler notebook output

The skill should also explicitly say:

- if the analysis cannot yet guide the next operator change, the analysis is incomplete

## `compiler-analysis.md` Contract

Keep the round-local output path:

- `opt-round-N/compiler-analysis.md`

But tighten the section contract to fit the performance workflow:

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

This structure matters for two reasons:

- it keeps the analysis attached to current round evidence rather than turning into general source notes
- it forces source findings to end in an operator-level next action

## Interaction With Sibling Skills

### `triton-npu-optimize`

`triton-npu-optimize` should continue to describe compiler source as the deepest escalation in the analysis ladder. It should update its wording so the compiler-source step is clearly described as:

- performance-focused
- downstream of profile and IR
- aimed at the next operator change

### `triton-npu-analyze-round-performance`

`triton-npu-analyze-round-performance` should continue to hand off to compiler source only after profiler and IR evidence have already narrowed the problem. Its wording should match the new performance-specific identity of the compiler-source skill.

### `triton-npu-analyze-ir`

No major workflow change is needed. The key contract remains:

- IR narrows and attributes suspicious structure
- compiler source explains the relevant compiler-side behavior when IR alone is not enough

## Testing

The redesign should add or update tests in three groups.

### Skill contract tests

Update generation-contract tests so they assert that the compiler-source skill now documents:

- performance-specific triggering conditions
- no compiler-error focus
- the navigation references
- the `docs -> lib -> include(when needed)` reading order
- the tightened `compiler-analysis.md` sections

### Reference presence tests

Add tests that assert the new reference files exist and contain expected anchors for:

- navigation map
- performance-question playbooks

### Helper script tests

Add lightweight unit tests for the helper script covering:

- grouped candidate output
- scoped subtree search
- `--hint` filtering
- omission of `test/` from the default search scope

## Rollout Shape

The implementation should proceed in this order:

1. rewrite `SKILL.md`
2. add navigation references
3. add the helper script
4. update sibling skill wording if needed
5. add or update tests

This order keeps the workflow contract primary and the helper tooling secondary.
