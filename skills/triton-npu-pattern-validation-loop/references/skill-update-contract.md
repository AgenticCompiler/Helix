# Skill Update Contract (Agent-Driven)

Integrate lessons from commit analysis into the **persistent loop skills workdir** under the
target repo (default `pattern-validation-skills/triton-npu-optimize-knowledge/`).

Do **not** edit:

- `$REPO/.codex/skills/` (or other backend staging dirs)
- the triton-agent install `skills/` tree

## Read before editing

1. `PERF_PATTERN_SYNTHESIS.md`
2. `PERF_KNOWLEDGE_BASE.md` when synthesis lacks paths or commits
3. `$KNOWLEDGE/references/pattern_index.md`
4. Individual pattern cards when extending or confirming overlap

## Decision rules

| If the lesson is… | Then… |
|-------------------|--------|
| Already covered by an existing pattern | Skip card edit; tune `expected_patterns` if needed |
| A new sub-case of an existing pattern | Extend the named card |
| Generic, reusable, evidence-backed | Promote new `references/patterns/<id>.md` if no card fits |
| Useful only for this repo / operator family | Do not add to `$KNOWLEDGE`; omit from audit expectations |
| Weak, rolled back, or contradicted | Reject — do not add to `$KNOWLEDGE` |

After **any** card edit batch:

```bash
python3 "$KNOWLEDGE/scripts/build_pattern_index.py" \
  --patterns-dir "$KNOWLEDGE/references/patterns" \
  --output "$KNOWLEDGE/references/pattern_index.md"
```

## Iteration after audit failure

1. Read failing workspace `opt-round-*/attempts.md`
2. Strengthen missing pattern cards under `$KNOWLEDGE`
3. Regenerate index
4. Re-run optimize-batch with `--skills-source-dir "$SKILLS"` (same persistent workdir)
