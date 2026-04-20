# Optimize Status Best-Round Warning Detail Design

## Summary

- Make the numeric/logged best-round mismatch warning concise.
- Include computed and logged speedup values directly in the warning text.
- Keep best-round source details out of the warning because the text output already prints `Best round` and `Logged best`.

## Goals

- Make the warning short enough to scan in status output.
- Show the current computed speedup and the `opt-note.md` logged speedup without a long explanatory sentence.
- Use a stable short prefix, `numeric best round != logged best`, for future text searches.
- Keep the change limited to optimize-status diagnostics.

## Non-Goals

- Do not change how the numeric best round is selected.
- Do not change how `opt-note.md` is parsed.
- Do not add new structured warning types or CLI flags.

## User-Visible Behavior

When numeric perf analysis disagrees with the best round recorded in `opt-note.md`, the warning should read like:

```text
Warning: numeric best round != logged best. computed speedup: 1.49x, 1.58x; logged speedup: 1.16x, 1.18x
```

The two speedup values are ordered as:

1. `Geomean speedup`
2. `Total speedup`

`computed speedup` comes from perf artifacts for the numeric best round.
`logged speedup` comes from the `## Overall Summary` block in `opt-note.md`.
If one logged speedup value is absent, print `missing` for that value.

## Implementation Shape

- Update the warning string in `src/triton_agent/optimize/status.py` at the point where computed and logged source values are known.
- Keep `OptimizeStatusWorkspace.warnings` as plain strings.
- Update status and CLI tests to assert the detailed warning text.

## Testing

- Unit test the exact status warning string.
- CLI test the rendered warning includes the source-labeled values.
