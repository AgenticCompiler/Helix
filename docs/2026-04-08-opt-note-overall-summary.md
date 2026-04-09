# Opt Note Overall Summary

## Summary

- Add one final `## Overall Summary` section at the end of `opt-note.md` so readers can tell the optimization outcome without reopening every round directory.
- Keep per-round entries unchanged as the running history log.
- Treat the overall summary as a replaceable session-level conclusion, not as another round entry.

## Problem

- `opt-note.md` currently records each round, but it does not provide one clear final answer for the session.
- Readers must infer the final best round and overall improvement by scanning `Best status` markers, `summary.md` files, and benchmark artifacts.
- This creates unnecessary ambiguity at exactly the moment when the optimize flow is supposed to hand off a result.

## User-Visible Behavior

- Completed optimize sessions should leave `opt-note.md` in a state where the final outcome is visible at the bottom of the file.
- The summary should answer:
  - which round is the final best candidate
  - what benchmark improvement it achieved versus baseline
  - what geomean and total speedup it achieved versus baseline
  - which other validated branches remain useful
  - what follow-up direction is recommended, if any
- Continue-optimize flows should refresh the existing overall summary after new rounds are completed instead of appending duplicate final sections.

## Format

Append this block after the round history:

```md
## Overall Summary
Final best round: round-N
Baseline mean: <value or unknown>
Best mean: <value or unknown>
Avg improvement: <value or unknown>
Geomean speedup: <value or unknown>
Total speedup: <value or unknown>
Validated branches: <comma-separated round names or none>
Outcome: <plain-English optimization result>
Next step: <plain-English follow-up or none>
```

## Data Sources

- `Final best round` should come from the round currently marked as the effective winner for the session, using geomean speedup as the headline metric.
- Numeric values should prefer concrete perf artifacts already used by `optimize-status`.
- `Validated branches` should list rounds recorded as `validated branch` in `opt-note.md`.
- `Outcome` and `Next step` remain short natural-language lines written by the optimizing agent.

## Parsing And Status Integration

- Extend optimize-status parsing so it can read the final-best round from `## Overall Summary` when present.
- Keep backward compatibility with existing `Best status: current best` parsing when the summary block is absent.
- If both are present and disagree, surface a warning instead of silently picking one.

## Scope

- Do not introduce a separate `final-summary.md`.
- Do not replace round summaries or `attempts.md`.
- Do not invent numeric calculations that differ from the existing optimize-status metrics.

## Verification

- Unit tests for parsing a final summary block and validated-branch lists.
- Optimize-status tests covering summary-aware best-round resolution and mismatch warnings.
- Repo verification with:
  - `uv run --group dev ruff check`
  - `uv run pyright`
  - `uv run python -m unittest discover -s tests -v`
