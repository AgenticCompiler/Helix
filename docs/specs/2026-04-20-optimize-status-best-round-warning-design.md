# Optimize Status Best-Round Warning Detail Design

## Summary

- Make the `numeric best round differs from logged best round` warning self-contained.
- Include the computed numeric best round and both logged best-round sources directly in the warning text.
- Label the computed value as coming from perf artifacts, and label the logged values as `opt-note.md` overall summary and current-best marker values.

## Goals

- Remove ambiguity when a user sees the warning without comparing nearby `Best round` and `Logged best` lines.
- Preserve the existing warning prefix so existing text searches remain useful.
- Keep the change limited to optimize-status diagnostics.

## Non-Goals

- Do not change how the numeric best round is selected.
- Do not change how `opt-note.md` is parsed.
- Do not add new structured warning types or CLI flags.

## User-Visible Behavior

When numeric perf analysis disagrees with the best round recorded in `opt-note.md`, the warning should read like:

```text
Warning: numeric best round differs from logged best round (computed from perf artifacts by geomean speedup: round-2; opt-note overall summary: round-1; opt-note current-best marker: round-3)
```

If one logged source is absent, print `missing` for that source.

## Implementation Shape

- Update the warning string in `src/triton_agent/optimize/status.py` at the point where computed and logged source values are known.
- Keep `OptimizeStatusWorkspace.warnings` as plain strings.
- Update status and CLI tests to assert the detailed warning text.

## Testing

- Unit test the exact status warning string.
- CLI test the rendered warning includes the source-labeled values.
