# Skill Update Contract (Agent-Driven)

Integrate lessons from commit analysis into **staged** optimize pattern cards. The
synthesis report may follow `analyze-commit-perf` contract sections, bullet lists, prose,
or mixed formats — **do not depend on a fixed table shape** (for example `G1-I1` rows).

## Read before editing

1. `PERF_PATTERN_SYNTHESIS.md` — full report; extract what should become shared skills
2. `PERF_KNOWLEDGE_BASE.md` — when synthesis lacks file paths, commits, or diffs
3. Staged `triton-npu-optimize-knowledge/references/pattern_index.md`
4. Open individual pattern cards only when extending or confirming overlap

## Decision rules

For each perf lesson you extract from synthesis:

| If the lesson is… | Then… |
|-------------------|--------|
| Already covered by an existing pattern | Skip card edit; ensure validation `expected_patterns` reference the right ID |
| A new sub-case of an existing pattern | **Extend** the named card (`## Use When`, `## Signals`, examples) |
| Generic, reusable, evidence-backed | **Promote** new `references/patterns/<id>.md` if no card fits |
| Useful only for this repo / operator family | **Do not** add to skills; omit from `expected_patterns` |
| Weak, rolled back, or contradicted | **Reject** — do not add to skills |

When synthesis uses recommendation labels (`extend-existing-card`, `local-only`, etc.),
treat them as hints, not hard parsers. Your judgment after reading the full report wins.

## Authoring rules

Pattern cards must include:

- `# Title` heading
- `## Summary` (what, 1–2 sentences)
- `## Use When` (detection conditions; orthogonal to Summary)

Optional: `## Avoid When`, `## Signals`, `## Related Patterns`, `## What To Verify After Applying`

After **any** card edit batch:

```bash
python3 <KNOWLEDGE>/scripts/build_pattern_index.py \
  --patterns-dir <KNOWLEDGE>/references/patterns \
  --output <KNOWLEDGE>/references/pattern_index.md
```

Do **not** hand-edit `pattern_index.md`.

## Target paths

```text
<staged-knowledge>/references/patterns/*.md
```

## Iteration discipline

When audit reports `missing_patterns`:

1. Read failing workspace `opt-round-*/attempts.md` — never triaged vs triaged-and-rejected?
2. Never triaged → strengthen `## Use When` / `## Signals` on missing pattern card(s)
3. Triaged but rejected → add synthesis evidence; cross-link related patterns
4. Pattern ID cited but wrong code change → improve `## What To Verify` and examples
5. Regenerate index; reset rounds; re-run optimize-batch

Record each edit batch in loop state `history`.

## High-priority patterns

Cards with frontmatter `priority: high` appear in `## High Priority Patterns` in the index.
Use for broadly applicable branch lessons (for example Ascend compile options catalog).
