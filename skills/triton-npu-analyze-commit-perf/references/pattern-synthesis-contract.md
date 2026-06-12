# Pattern Synthesis And Skill Alignment Contract

## Purpose

After incremental file analysis completes in `PERF_KNOWLEDGE_BASE.md`, run one final
synthesis round and write a separate consolidated report. This report groups similar
lessons, compares them against the staged optimize pattern index, and recommends whether
each lesson should update the skills knowledge base.

Do not edit pattern cards or `pattern_index.md` during this workflow. Only recommend.

## Required Input

- Completed `PERF_KNOWLEDGE_BASE.md` from the file-grouped rounds.
- Staged pattern index:
  `triton-npu-optimize-knowledge/references/pattern_index.md`
- Optional detailed pattern cards under:
  `triton-npu-optimize-knowledge/references/patterns/`

Read the generated index first. Open detailed pattern files only when a candidate match
needs confirmation.

## Required Output File

Write the final consolidated report to the path requested in the prompt, default:

`PERF_PATTERN_SYNTHESIS.md`

## Required Structure

```markdown
# Performance Pattern Synthesis

## Run Summary

## Consolidated Pattern Groups

## Pattern Index Alignment

## Per-Item Skill Update Recommendations

## Limitations And Uncertainties
```

## Run Summary

Include:

- source report path (`PERF_KNOWLEDGE_BASE.md`)
- pattern index path used for comparison
- number of consolidated groups
- number of distinct items (lessons) extracted
- counts by skill-update recommendation

## Consolidated Pattern Groups

Cluster similar lessons from file analyses. Merge duplicates that describe the same
underlying mechanism even if they came from different files or commits.

For each group:

```markdown
### <group-title>

- Theme:
- Mechanism summary:
- Applicable scenario:
- Supporting files:
- Supporting commits:
- Evidence strength: strong | moderate | weak

#### Items

##### <item-id> <short-title>
- Source file:
- Source commits:
- What changed:
- Hardware mechanism:
- Reusable rule:
- Counterexample / failure mode:
```

Rules:

- Every performance-relevant lesson from `PERF_KNOWLEDGE_BASE.md` must appear in exactly
  one group item.
- Do not include performance-unrelated commits.
- Prefer fewer, stronger groups over many fragmented duplicates.

## Pattern Index Alignment

For each consolidated group (or each distinct item when alignment differs), compare
against `pattern_index.md`.

Use one subsection per matched or candidate pattern:

```markdown
### <pattern-id or `novel`>

- Existing index entry:
  - pattern id:
  - summary (quoted or paraphrased from index):
- Relationship: `matches` | `partial-overlap` | `extends` | `novel` | `contradicts`
- Supporting items: <item-id list>
- Alignment notes:
- Gap vs existing pattern card:
```

Relationship meanings:

- `matches`: the branch lesson is already covered by an existing pattern.
- `partial-overlap`: related but missing important conditions or signals.
- `extends`: existing pattern applies but this branch adds a new sub-case or stronger rule.
- `novel`: no reasonable existing pattern; candidate for a new card later.
- `contradicts`: branch evidence suggests the existing pattern guidance may be wrong or
  too narrow; explain carefully.

## Per-Item Skill Update Recommendations

Provide a table with one row per item:

| Item ID | Group | Related pattern(s) | Recommendation | Rationale |
| --- | --- | --- | --- | --- |

`Recommendation` must be exactly one of:

- `no-change`: already covered; no skill update needed.
- `extend-existing-card`: add signals, examples, or caveats to an existing pattern card.
- `promote-new-pattern-card`: novel enough to author a new pattern under `references/patterns/`.
- `local-only`: useful for this repo/operator family but not generic enough for skills.
- `reject`: incorrect, too weak, or contradicted by rollback evidence.

Promotion criteria (align with optimize `learned_lessons` and pattern-card authoring):

- generic across a family of Triton Ascend NPU operators
- evidence-backed from commit diffs or explicit perf mechanisms
- written as a reusable rule, not round narrative
- states applicability limits

Do not recommend editing skills when evidence is static-only and weak.

## Limitations And Uncertainties

Record:

- items with weak static evidence
- pattern matches that needed guesswork
- rollback lessons that should remain local warnings only
- pattern index entries that may be outdated but were not changed in this workflow

## Hard Rules

- This synthesis round is mandatory after all file rounds complete.
- Do not modify files under `triton-npu-optimize-knowledge/`.
- Do not hand-edit `pattern_index.md`.
- Performance-unrelated content must not appear in the synthesis report.
