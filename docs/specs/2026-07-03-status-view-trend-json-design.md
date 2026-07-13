# Status View Trend And JSON Format Design

## Summary

Extend `helix status` with an explicit view selector and JSON output:

- `--view best|trend`, defaulting to `best`
- `--format text|markdown|json`

The existing status behavior remains the default `best` view. The new `trend`
view reports per-operator round speedup trends as a wide table.

## Goals

- Keep `status` as the single read-only optimization status/statistics command.
- Preserve current text and Markdown `best` output by default.
- Let users inspect per-round geomean speedup trends across many operators.
- Provide machine-readable JSON for both `best` and `trend`.

## Non-Goals

- Do not rename `status`.
- Do not add a separate trend command.
- Do not include diagnostic state, notes, or warnings in the trend table.
- Do not change how speedups are calculated.

## User-Visible Behavior

`best` remains the default:

```bash
uv run helix status --input operators_root
uv run helix status --input operators_root --view best
```

`trend` shows a wide table with one operator per row and one column per round:

```bash
uv run helix status --input operators_root --view trend
uv run helix status --input operators_root --view trend --format markdown
uv run helix status --input operators_root --view trend --format json
```

Trend columns use the union of all comparable round names across non-`NO-SESSION`
operators, sorted by round number. Missing values render as `-` in text and
Markdown output.

The trend metric is each round's geomean speedup relative to the selected
baseline, using the same parsing and metric-source rules as the current best
round calculation.

## JSON Contracts

JSON output uses a single top-level `operators` array. Speedup values are raw
JSON numbers, not formatted strings.

For `--view best --format json`:

```json
{
  "operators": [
    {
      "name": "op_a",
      "state": "ok",
      "avg_improvement": 0.3,
      "geomean_speedup": 1.4907119849998598,
      "best_round": "round-2",
      "logged_best": "round-1",
      "verified": false,
      "verified_geomean_speedup": null,
      "warnings": []
    }
  ]
}
```

The best JSON view includes `NO-SESSION` operators so it reflects the complete
status scan.

For `--view trend --format json`:

```json
{
  "operators": [
    {
      "name": "op_a",
      "round_speedups": {
        "round-1": 1.118033988749895,
        "round-2": 1.4907119849998598,
        "round-3": null
      }
    }
  ]
}
```

The trend JSON view filters `NO-SESSION` operators. Missing round values are
`null`.

## Architecture

The status core should expose enough per-round data for renderers instead of
only returning the selected best round. The existing best-round calculation can
continue to choose the maximum geomean speedup, using mean latency only as the
tie-breaker.

Rendering remains separated by view and format:

- `best` + `text` and `best` + `markdown` preserve current output.
- `best` + `json` serializes the current status summary model.
- `trend` renderers consume each operator's comparable round list and build the
  round-name union at render time.

The CLI should validate `--view` with parser choices, mirroring the existing
`--format` handling.

## Testing

Focused tests should cover:

- parser support for `--view best|trend`
- parser support for `--format json`
- default `--view best` behavior
- per-round status inspection returns all comparable round speedups
- trend text output uses a wide table and `-` for missing rounds
- trend Markdown output uses the same wide-table semantics
- trend JSON output uses top-level `operators`, raw floats, and `null` for
  missing rounds
- best JSON output includes `NO-SESSION` operators and current warning fields
- existing best text and Markdown tests remain unchanged apart from any required
  model construction updates
